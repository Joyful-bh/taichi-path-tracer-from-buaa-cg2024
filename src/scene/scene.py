"""
Scene — 场景容器。

持有 SphereSystem、TriangleSystem、BVHSystem、MaterialSystem，
提供 Python 端添加几何/材质的接口，以及 Taichi 端的 intersect() @ti.func。

使用流程：
  scene = Scene()
  mid = scene.materials.add_lambertian([0.8, 0.2, 0.2])
  scene.add_sphere([0,0,0], 1.0, mid)
  scene.bake()   # 上传所有数据，构建 BVH
  # 之后可在 kernel 内调用 scene.intersect()
"""

import taichi as ti
import numpy as np

from src.geometry.primitives    import SphereSystem, TriangleSystem
from src.geometry.bvh           import BVHSystem
from src.geometry.light_sampler import LightSampler
from src.materials.material     import MaterialSystem
from src.textures.texture       import TextureSystem
from src.math_utils             import apply_transform, make_rotation_matrix
from src.constants              import T_MIN, T_MAX


@ti.data_oriented
class Scene:
    def __init__(self,
                 max_spheres:   int = 1024,
                 max_triangles: int = 200_000,
                 max_materials: int = 512,
                 max_bvh_nodes: int = 524_288,
                 bg_color=(0.01, 0.01, 0.01)):
        self.spheres       = SphereSystem(max_spheres)
        self.triangles     = TriangleSystem(max_triangles)
        self.materials     = MaterialSystem(max_materials)
        self.tex_sys       = TextureSystem()
        self.bvh           = BVHSystem(max_bvh_nodes)
        self.light_sampler = LightSampler()

        # 背景色（未命中光线时返回）
        self.bg_color = ti.Vector.field(3, ti.f32, shape=())
        self.bg_color[None] = list(bg_color)
        self.bg_top = ti.Vector.field(3, ti.f32, shape=())
        self.bg_bottom = ti.Vector.field(3, ti.f32, shape=())
        self.bg_top[None] = list(bg_color)
        self.bg_bottom[None] = list(bg_color)
        self.bg_gradient = ti.field(ti.i32, shape=())
        self.bg_gradient[None] = 0

        self._baked = False

    # ------------------------------------------------------------------
    # Python 端：几何添加（代理到子系统）
    # ------------------------------------------------------------------

    def add_sphere(self, center, radius: float, mat_id: int) -> int:
        return self.spheres.add(center, radius, mat_id)

    def add_triangle(self, v0, v1, v2, mat_id: int, **kwargs) -> int:
        return self.triangles.add(v0, v1, v2, mat_id, **kwargs)

    def add_rectangle(self, center, width: float, height: float,
                      rotation_deg=(0.0, 0.0, 0.0), mat_id: int = 0):
        self.triangles.add_rectangle(center, width, height, rotation_deg, mat_id)

    def add_cube(self, center, size: float,
                 rotation_deg=(0.0, 0.0, 0.0), mat_id: int = 0):
        self.triangles.add_cube(center, size, rotation_deg, mat_id)

    def add_cuboid(self, center, dimensions,
                   rotation_deg=(0.0, 0.0, 0.0), mat_id: int = 0):
        self.triangles.add_cuboid(center, dimensions, rotation_deg, mat_id)

    def load_obj(self, obj_path: str, mat_id: int = 0,
                 scale: float = 1.0,
                 translation=(0.0, 0.0, 0.0),
                 rotation_deg=(0.0, 0.0, 0.0)) -> int:
        """加载 OBJ 网格，返回三角形数量。"""
        return self.triangles.load_obj(obj_path, mat_id, scale, translation, rotation_deg)

    def add_mesh(self, vertices: np.ndarray, indices: np.ndarray, mat_id: int = 0,
                 vertex_normals: np.ndarray = None,
                 uvs: np.ndarray = None,
                 tangents: np.ndarray = None) -> tuple:
        """
        从顶点 + 索引数组添加网格（FBX/内存数据）。
        vertices       : (V, 3)  float32
        indices        : (F, 3)  int32
        vertex_normals : (V, 3)  float32，可选
        uvs            : (V, 2)  float32，可选
        tangents       : (V, 3) 或 (V, 4) float32，可选（GLB TANGENT 属性，第4列为手性）
        """
        verts = np.asarray(vertices, np.float32)
        idx   = np.asarray(indices,  np.int32)
        v0s   = verts[idx[:, 0]]
        v1s   = verts[idx[:, 1]]
        v2s   = verts[idx[:, 2]]

        n0s = n1s = n2s = None
        if vertex_normals is not None:
            vn  = np.asarray(vertex_normals, np.float32)
            n0s = vn[idx[:, 0]]
            n1s = vn[idx[:, 1]]
            n2s = vn[idx[:, 2]]

        uv0s = uv1s = uv2s = None
        if uvs is not None:
            uv = np.asarray(uvs, np.float32)
            uv0s = uv[idx[:, 0]]
            uv1s = uv[idx[:, 1]]
            uv2s = uv[idx[:, 2]]

        t0s = t1s = t2s = None
        if tangents is not None:
            tan = np.asarray(tangents, np.float32)
            if tan.ndim == 2 and tan.shape[1] == 3:
                tan = np.hstack([tan, np.ones((len(tan), 1), np.float32)])
            t0s = tan[idx[:, 0]]
            t1s = tan[idx[:, 1]]
            t2s = tan[idx[:, 2]]

        return self.triangles.add_batch(v0s, v1s, v2s, mat_id,
                                        n0s=n0s, n1s=n1s, n2s=n2s,
                                        uv0s=uv0s, uv1s=uv1s, uv2s=uv2s,
                                        t0s=t0s, t1s=t1s, t2s=t2s)

    # ------------------------------------------------------------------
    # 烘焙：上传数据 + 构建 BVH
    # ------------------------------------------------------------------

    def bake(self):
        """
        将所有 Python 端数据上传到 GPU，并构建 BVH。
        必须在开始渲染前调用，且只应调用一次。
        """
        assert not self._baked, "Scene.bake() 只能调用一次"
        self.materials.bake()
        self.tex_sys.bake()
        self.spheres.bake()
        self.triangles.bake()

        sph_bbox = self.spheres.compute_bboxes()
        tri_bbox = self.triangles.compute_bboxes()
        self.bvh.build(sph_bbox, tri_bbox,
                       self.spheres.count,
                       self.triangles.count)

        self.light_sampler.build(self.triangles, self.materials)

        self._baked = True
        n_total = self.spheres.count + self.triangles.count
        print(f"[Scene] 烘焙完成：{self.spheres.count} 球体 + "
              f"{self.triangles.count} 三角形 = {n_total} 图元，"
              f"{self.materials.count} 材质")

    # ------------------------------------------------------------------
    # Taichi 作用域：场景求交
    # ------------------------------------------------------------------

    @ti.func
    def intersect(self, ray_origin, ray_dir, t_min: ti.f32, t_max: ti.f32):
        """
        BVH 遍历求交。
        返回 (hit_t, hit_pos, hit_normal, hit_uv, hit_mat, front_face)。
        hit_mat == -1 表示未命中（背景）。
        """
        return self.bvh.intersect(ray_origin, ray_dir, t_min, t_max,
                                   self.spheres, self.triangles)

    @ti.func
    def background(self, ray_dir):
        color = self.bg_color[None]
        if self.bg_gradient[None] != 0:
            unit_dir = ray_dir.normalized()
            t = ti.max(0.0, ti.min(1.0, (unit_dir[1] + 1.0) * 0.5))
            color = (1.0 - t) * self.bg_bottom[None] + t * self.bg_top[None]
        return color
