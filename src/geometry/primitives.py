"""
几何图元系统 — SphereSystem 和 TriangleSystem。

设计原则：
- SoA（Struct-of-Arrays）布局：每个属性独立存为 ti.field，利于 GPU 内存合并访问。
- Python 端用列表积累数据，调用 bake() 一次性 from_numpy 上传，避免逐三角形的 kernel 调用。
- hit() 是 @ti.func，只在 Taichi 作用域（kernel 内）调用。
- 三角形同时存储面法线（用于正背面判断）和顶点法线（用于 Phong 插值平滑着色）。
"""

import taichi as ti
import numpy as np

from src.math_utils import make_rotation_matrix


@ti.data_oriented
class SphereSystem:
    def __init__(self, max_spheres: int = 1024):
        self._max = max_spheres
        # SoA Taichi 字段
        self.center = ti.Vector.field(3, ti.f32, shape=(max_spheres,))
        self.radius = ti.field(ti.f32, shape=(max_spheres,))
        self.mat_id = ti.field(ti.i32, shape=(max_spheres,))
        # Python 端缓冲
        self._centers  = []
        self._radii    = []
        self._mat_ids  = []
        self._count    = 0

    # ------------------------------------------------------------------
    # Python 端接口
    # ------------------------------------------------------------------

    def add(self, center, radius: float, mat_id: int) -> int:
        """添加一个球体，返回球体本地 ID。"""
        assert self._count < self._max, f"SphereSystem 超出容量 {self._max}"
        self._centers.append(np.array(center, dtype=np.float32))
        self._radii.append(float(radius))
        self._mat_ids.append(int(mat_id))
        idx = self._count
        self._count += 1
        return idx

    def bake(self):
        """将 Python 缓冲一次性上传到 GPU。在 build_bvh 之前调用。"""
        n = self._count
        if n == 0:
            return
        c_arr = np.zeros((self._max, 3), np.float32)
        r_arr = np.zeros(self._max, np.float32)
        m_arr = np.zeros(self._max, np.int32)
        c_arr[:n] = np.stack(self._centers)
        r_arr[:n] = np.array(self._radii, np.float32)
        m_arr[:n] = np.array(self._mat_ids, np.int32)
        self.center.from_numpy(c_arr)
        self.radius.from_numpy(r_arr)
        self.mat_id.from_numpy(m_arr)

    def compute_bboxes(self):
        """
        Python 端计算所有球体 AABB，供 BVH 构建使用。
        返回 (bbox_min, bbox_max)，各为 (n, 3) float32 数组。
        """
        n = self._count
        if n == 0:
            return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32)
        c = np.stack(self._centers)               # (n, 3)
        r = np.array(self._radii, np.float32).reshape(-1, 1)
        return (c - r).astype(np.float32), (c + r).astype(np.float32)

    @property
    def count(self) -> int:
        return self._count

    # ------------------------------------------------------------------
    # Taichi 作用域：单球求交（Möller–Trumbore 二次方程）
    # ------------------------------------------------------------------

    @ti.func
    def hit(self, sphere_id: ti.i32, ray_origin, ray_dir, t_min: ti.f32, t_max: ti.f32):
        """
        返回 (hit_t, hit_pos, hit_normal, hit_tan, hit_uv, hit_mat, front_face)。
        hit_t == t_max 表示未命中。球体不支持 UV 切线，切线返回零向量。
        """
        c = self.center[sphere_id]
        r = self.radius[sphere_id]
        oc = ray_origin - c

        # 使用 half-b 形式减小数值误差
        a      = ray_dir.norm_sqr()
        half_b = oc.dot(ray_dir)
        disc   = half_b * half_b - a * (oc.norm_sqr() - r * r)

        hit_t      = t_max
        hit_pos    = ti.Vector([0.0, 0.0, 0.0])
        hit_normal = ti.Vector([0.0, 0.0, 0.0])
        hit_tan    = ti.Vector([0.0, 0.0, 0.0, 1.0])
        hit_uv     = ti.Vector([0.0, 0.0])
        hit_mat    = -1
        front      = True

        if disc >= 0.0:
            sqrt_d = ti.sqrt(disc)
            root   = (-half_b - sqrt_d) / a
            if not (t_min < root < t_max):
                root = (-half_b + sqrt_d) / a
            if t_min < root < t_max:
                hit_t   = root
                hit_pos = ray_origin + root * ray_dir
                outward = (hit_pos - c) / r
                front   = ray_dir.dot(outward) < 0.0
                hit_normal = outward if front else -outward
                hit_mat = self.mat_id[sphere_id]

        return hit_t, hit_pos, hit_normal, hit_tan, hit_uv, hit_mat, front


# ---------------------------------------------------------------------------

@ti.data_oriented
class TriangleSystem:
    """
    SoA 布局三角形系统，支持顶点法线（平滑着色）和 UV 坐标（贴图）。

    每个三角形存储：
    - v0/v1/v2        顶点位置
    - n0/n1/n2        逐顶点法线（缺省则等于 face_normal）
    - uv0/uv1/uv2     逐顶点 UV（缺省 0）
    - face_normal     面法线（由 v0/v1/v2 叉积预计算，用于正背面检测）
    - e1/e2           预计算边向量（用于 Möller–Trumbore）
    - tan             逐面切线（由 UV 导数计算，用于法线贴图 TBN 矩阵）
    - mat_id          材质 ID
    """

    def __init__(self, max_triangles: int = 200_000):
        self._max = max_triangles
        # SoA Taichi 字段
        self.v0          = ti.Vector.field(3, ti.f32, shape=(max_triangles,))
        self.v1          = ti.Vector.field(3, ti.f32, shape=(max_triangles,))
        self.v2          = ti.Vector.field(3, ti.f32, shape=(max_triangles,))
        self.n0          = ti.Vector.field(3, ti.f32, shape=(max_triangles,))
        self.n1          = ti.Vector.field(3, ti.f32, shape=(max_triangles,))
        self.n2          = ti.Vector.field(3, ti.f32, shape=(max_triangles,))
        self.uv0         = ti.Vector.field(2, ti.f32, shape=(max_triangles,))
        self.uv1         = ti.Vector.field(2, ti.f32, shape=(max_triangles,))
        self.uv2         = ti.Vector.field(2, ti.f32, shape=(max_triangles,))
        self.face_normal = ti.Vector.field(3, ti.f32, shape=(max_triangles,))
        self.e1          = ti.Vector.field(3, ti.f32, shape=(max_triangles,))
        self.e2          = ti.Vector.field(3, ti.f32, shape=(max_triangles,))
        # 逐顶点切线（替代原来的逐面切线），支持 GLB TANGENT 属性插值
        self.t0          = ti.Vector.field(4, ti.f32, shape=(max_triangles,))
        self.t1          = ti.Vector.field(4, ti.f32, shape=(max_triangles,))
        self.t2          = ti.Vector.field(4, ti.f32, shape=(max_triangles,))
        self.mat_id      = ti.field(ti.i32, shape=(max_triangles,))

        self._count = 0
        # Python 端缓冲（列表，bake 时才转 numpy）
        self._v0  = []; self._v1  = []; self._v2  = []
        self._n0  = []; self._n1  = []; self._n2  = []
        self._uv0 = []; self._uv1 = []; self._uv2 = []
        self._mat = []
        # 逐顶点切线缓冲：None 表示使用 bake() 时按面计算的回退值
        self._t0 = []; self._t1 = []; self._t2 = []

    # ------------------------------------------------------------------
    # Python 端接口
    # ------------------------------------------------------------------

    def add(self, v0, v1, v2, mat_id: int,
            n0=None, n1=None, n2=None,
            uv0=None, uv1=None, uv2=None) -> int:
        """添加单个三角形，返回本地 ID。顶点法线和 UV 可选。"""
        assert self._count < self._max, f"TriangleSystem 超出容量 {self._max}"
        v0 = np.array(v0, np.float32)
        v1 = np.array(v1, np.float32)
        v2 = np.array(v2, np.float32)
        fn = _compute_face_normal(v0, v1, v2)

        self._v0.append(v0);  self._v1.append(v1);  self._v2.append(v2)
        self._n0.append(fn if n0 is None else np.array(n0, np.float32))
        self._n1.append(fn if n1 is None else np.array(n1, np.float32))
        self._n2.append(fn if n2 is None else np.array(n2, np.float32))
        self._uv0.append(np.array([0.0, 0.0], np.float32) if uv0 is None else np.array(uv0, np.float32))
        self._uv1.append(np.array([1.0, 0.0], np.float32) if uv1 is None else np.array(uv1, np.float32))
        self._uv2.append(np.array([0.0, 1.0], np.float32) if uv2 is None else np.array(uv2, np.float32))
        self._mat.append(int(mat_id))
        self._t0.append(None); self._t1.append(None); self._t2.append(None)

        idx = self._count
        self._count += 1
        return idx

    def add_batch(self, v0s: np.ndarray, v1s: np.ndarray, v2s: np.ndarray,
                  mat_ids,
                  n0s=None, n1s=None, n2s=None,
                  uv0s=None, uv1s=None, uv2s=None,
                  t0s=None, t1s=None, t2s=None) -> tuple:
        """
        批量添加三角形（numpy 数组），高效用于 FBX/OBJ 导入。
        mat_ids 可以是标量或长度为 n 的数组。
        返回 (start_id, end_id)。
        """
        v0s = np.asarray(v0s, np.float32)
        v1s = np.asarray(v1s, np.float32)
        v2s = np.asarray(v2s, np.float32)
        n   = len(v0s)
        assert self._count + n <= self._max, "TriangleSystem 超出容量"

        fns = _compute_face_normals_batch(v0s, v1s, v2s)

        for i in range(n):
            self._v0.append(v0s[i]);  self._v1.append(v1s[i]);  self._v2.append(v2s[i])
            self._n0.append(fns[i] if n0s is None else np.array(n0s[i], np.float32))
            self._n1.append(fns[i] if n1s is None else np.array(n1s[i], np.float32))
            self._n2.append(fns[i] if n2s is None else np.array(n2s[i], np.float32))
            self._uv0.append(np.zeros(2, np.float32) if uv0s is None else np.array(uv0s[i], np.float32))
            self._uv1.append(np.zeros(2, np.float32) if uv1s is None else np.array(uv1s[i], np.float32))
            self._uv2.append(np.zeros(2, np.float32) if uv2s is None else np.array(uv2s[i], np.float32))
            mid = int(mat_ids) if np.isscalar(mat_ids) else int(mat_ids[i])
            self._mat.append(mid)
            if t0s is not None:
                self._t0.append(np.array(t0s[i], np.float32))
                self._t1.append(np.array(t1s[i], np.float32))
                self._t2.append(np.array(t2s[i], np.float32))
            else:
                self._t0.append(None); self._t1.append(None); self._t2.append(None)

        start = self._count
        self._count += n
        return start, self._count

    def bake(self):
        """将 Python 缓冲一次性上传至 GPU。"""
        n = self._count
        if n == 0:
            return

        def _pad(lst, shape_rest):
            arr = np.zeros((self._max, *shape_rest), np.float32)
            arr[:n] = np.stack(lst)
            return arr

        v0_arr = _pad(self._v0, (3,));  v1_arr = _pad(self._v1, (3,));  v2_arr = _pad(self._v2, (3,))

        self.v0.from_numpy(v0_arr);  self.v1.from_numpy(v1_arr);  self.v2.from_numpy(v2_arr)

        self.n0.from_numpy(_pad(self._n0, (3,)))
        self.n1.from_numpy(_pad(self._n1, (3,)))
        self.n2.from_numpy(_pad(self._n2, (3,)))

        uv0_arr = _pad(self._uv0, (2,))
        uv1_arr = _pad(self._uv1, (2,))
        uv2_arr = _pad(self._uv2, (2,))
        self.uv0.from_numpy(uv0_arr)
        self.uv1.from_numpy(uv1_arr)
        self.uv2.from_numpy(uv2_arr)

        # face normal
        fns = np.zeros((self._max, 3), np.float32)
        fns[:n] = _compute_face_normals_batch(v0_arr[:n], v1_arr[:n], v2_arr[:n])
        self.face_normal.from_numpy(fns)

        # 预计算边向量
        e1_arr = np.zeros((self._max, 3), np.float32);  e1_arr[:n] = v1_arr[:n] - v0_arr[:n]
        e2_arr = np.zeros((self._max, 3), np.float32);  e2_arr[:n] = v2_arr[:n] - v0_arr[:n]
        self.e1.from_numpy(e1_arr)
        self.e2.from_numpy(e2_arr)

        # 逐顶点切线：优先使用 add_batch() 传入的 GLB TANGENT 属性，
        # 否则退回到逐面 UV 导数切线（三个顶点共用同一切线）。
        face_tans = _compute_tangents_batch(v0_arr[:n], v1_arr[:n], v2_arr[:n],
                                            uv0_arr[:n], uv1_arr[:n], uv2_arr[:n])
        t0_arr = np.zeros((self._max, 4), np.float32)
        t1_arr = np.zeros((self._max, 4), np.float32)
        t2_arr = np.zeros((self._max, 4), np.float32)
        t0_arr[:n, :3] = face_tans   # 默认：逐面切线，右手 TBN
        t1_arr[:n, :3] = face_tans
        t2_arr[:n, :3] = face_tans
        t0_arr[:n, 3] = 1.0
        t1_arr[:n, 3] = 1.0
        t2_arr[:n, 3] = 1.0
        # 覆盖有显式顶点切线的三角形
        vert_indices = [i for i in range(n) if self._t0[i] is not None]
        if vert_indices:
            idx = np.array(vert_indices, np.int32)
            t0_arr[idx] = np.stack([self._t0[i] for i in vert_indices])
            t1_arr[idx] = np.stack([self._t1[i] for i in vert_indices])
            t2_arr[idx] = np.stack([self._t2[i] for i in vert_indices])
            print(f"[TriangleSystem] {len(vert_indices)}/{n} 三角形使用逐顶点切线")
        self.t0.from_numpy(t0_arr)
        self.t1.from_numpy(t1_arr)
        self.t2.from_numpy(t2_arr)

        m_arr = np.zeros(self._max, np.int32);  m_arr[:n] = np.array(self._mat, np.int32)
        self.mat_id.from_numpy(m_arr)

    def compute_bboxes(self):
        """
        Python 端计算所有三角形 AABB，供 BVH 构建使用。
        返回 (bbox_min, bbox_max)，各为 (n, 3) float32。
        """
        n = self._count
        if n == 0:
            return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32)
        v0 = np.stack(self._v0)
        v1 = np.stack(self._v1)
        v2 = np.stack(self._v2)
        bbox_min = np.minimum(np.minimum(v0, v1), v2).astype(np.float32)
        bbox_max = np.maximum(np.maximum(v0, v1), v2).astype(np.float32)
        return bbox_min, bbox_max

    @property
    def count(self) -> int:
        return self._count

    # ------------------------------------------------------------------
    # Python 端：构建辅助几何体（矩形、长方体）
    # ------------------------------------------------------------------

    def add_rectangle(self, center, width: float, height: float,
                      rotation_deg=(0.0, 0.0, 0.0), mat_id: int = 0):
        """在世界坐标系中添加一个矩形（两个三角形）。"""
        hw, hh = width * 0.5, height * 0.5
        local = np.array([
            [-hw, -hh, 0.0],
            [ hw, -hh, 0.0],
            [ hw,  hh, 0.0],
            [-hw,  hh, 0.0],
        ], np.float32)
        R = make_rotation_matrix(*rotation_deg)
        world = local @ R.T + np.array(center, np.float32)
        self.add(world[0], world[1], world[2], mat_id)
        self.add(world[0], world[2], world[3], mat_id)

    def add_cuboid(self, center, dimensions, rotation_deg=(0.0, 0.0, 0.0), mat_id: int = 0):
        """添加长方体（六面，每面两个三角形，法线朝外）。"""
        lx, ly, lz = [d * 0.5 for d in dimensions]
        local = np.array([
            [-lx, -ly, -lz], [ lx, -ly, -lz], [ lx,  ly, -lz], [-lx,  ly, -lz],  # 底面 0-3
            [-lx, -ly,  lz], [ lx, -ly,  lz], [ lx,  ly,  lz], [-lx,  ly,  lz],  # 顶面 4-7
        ], np.float32)
        R = make_rotation_matrix(*rotation_deg)
        w = local @ R.T + np.array(center, np.float32)
        # 每面 CCW（法线朝外）
        faces = [
            (0, 2, 1), (0, 3, 2),   # 底面（-Z 方向）
            (4, 5, 6), (4, 6, 7),   # 顶面（+Z 方向）
            (0, 1, 5), (0, 5, 4),   # 前面（-Y 方向）
            (2, 3, 7), (2, 7, 6),   # 后面（+Y 方向）
            (0, 4, 7), (0, 7, 3),   # 左面（-X 方向）
            (1, 2, 6), (1, 6, 5),   # 右面（+X 方向）
        ]
        for a, b, c in faces:
            self.add(w[a], w[b], w[c], mat_id)

    def add_cube(self, center, size: float, rotation_deg=(0.0, 0.0, 0.0), mat_id: int = 0):
        self.add_cuboid(center, [size, size, size], rotation_deg, mat_id)

    # ------------------------------------------------------------------
    # Taichi 作用域：单三角形求交（Möller–Trumbore 算法）
    # ------------------------------------------------------------------

    @ti.func
    def hit(self, tri_id: ti.i32, ray_origin, ray_dir, t_min: ti.f32, t_max: ti.f32):
        """
        返回 (hit_t, hit_pos, hit_normal, hit_tan, hit_uv, hit_mat, front_face)。
        hit_t == t_max 表示未命中。
        法线：使用重心坐标插值顶点法线（Phong 着色），正背面通过面法线判断。
        切线：重心坐标插值逐顶点切线（GLB TANGENT 属性，或退回到逐面 UV 切线），用于法线贴图 TBN。
        """
        e1 = self.e1[tri_id]
        e2 = self.e2[tri_id]
        h  = ray_dir.cross(e2)
        a  = e1.dot(h)

        hit_t      = t_max
        hit_pos    = ti.Vector([0.0, 0.0, 0.0])
        hit_normal = ti.Vector([0.0, 0.0, 0.0])
        hit_tan    = ti.Vector([0.0, 0.0, 0.0, 1.0])
        hit_uv     = ti.Vector([0.0, 0.0])
        hit_mat    = -1
        front      = True

        if ti.abs(a) > 1e-8:
            f = 1.0 / a
            s = ray_origin - self.v0[tri_id]
            u = f * s.dot(h)
            if 0.0 <= u <= 1.0:
                q = s.cross(e1)
                v = f * ray_dir.dot(q)
                if v >= 0.0 and u + v <= 1.0:
                    t = f * e2.dot(q)
                    if t_min < t < t_max:
                        hit_t   = t
                        hit_pos = ray_origin + t * ray_dir
                        hit_mat = self.mat_id[tri_id]

                        # 重心坐标：w = 1-u-v，u，v
                        w          = 1.0 - u - v
                        interp_n   = (w * self.n0[tri_id]
                                    + u * self.n1[tri_id]
                                    + v * self.n2[tri_id]).normalized()
                        face_n     = self.face_normal[tri_id]
                        front      = ray_dir.dot(face_n) < 0.0
                        # 法线朝向光线来源侧
                        hit_normal = interp_n if front else -interp_n

                        hit_uv = (w * self.uv0[tri_id]
                                + u * self.uv1[tri_id]
                                + v * self.uv2[tri_id])

                        # 逐顶点切线插值（GLB TANGENT 属性或逐面回退值）
                        t_interp = (w * self.t0[tri_id]
                                  + u * self.t1[tri_id]
                                  + v * self.t2[tri_id])
                        tangent_xyz = ti.Vector([t_interp[0], t_interp[1], t_interp[2]])
                        t_len = tangent_xyz.norm()
                        if t_len > 1e-6:
                            tangent_xyz = tangent_xyz / t_len
                        else:
                            tangent_xyz = ti.Vector([1.0, 0.0, 0.0])
                        tangent_sign = 1.0 if t_interp[3] >= 0.0 else -1.0
                        hit_tan = ti.Vector([tangent_xyz[0], tangent_xyz[1], tangent_xyz[2], tangent_sign])

        return hit_t, hit_pos, hit_normal, hit_tan, hit_uv, hit_mat, front

    def load_obj(self, obj_path: str, mat_id: int = 0,
                 scale: float = 1.0,
                 translation=(0.0, 0.0, 0.0),
                 rotation_deg=(0.0, 0.0, 0.0)) -> int:
        """从 OBJ 文件加载网格并添加到系统，返回添加的三角形数量。"""
        vertices, faces = [], []
        with open(obj_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if parts[0] == 'v' and len(parts) >= 4:
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elif parts[0] == 'f' and len(parts) >= 4:
                    idxs = [int(p.split('/')[0]) - 1 for p in parts[1:]]
                    for i in range(1, len(idxs) - 1):
                        faces.append([idxs[0], idxs[i], idxs[i + 1]])

        if not vertices or not faces:
            print(f"[OBJ] 警告: {obj_path} 未加载到有效数据")
            return 0

        from src.math_utils import apply_transform
        verts = apply_transform(np.array(vertices, np.float32), scale, rotation_deg, translation)
        faces_np = np.array(faces, np.int32)
        v0s = verts[faces_np[:, 0]]
        v1s = verts[faces_np[:, 1]]
        v2s = verts[faces_np[:, 2]]
        self.add_batch(v0s, v1s, v2s, mat_id)
        return len(faces_np)


# ---------------------------------------------------------------------------
# 模块私有：numpy 法线计算
# ---------------------------------------------------------------------------

def _compute_face_normal(v0, v1, v2) -> np.ndarray:
    e1 = v1 - v0
    e2 = v2 - v0
    n  = np.cross(e1, e2).astype(np.float32)
    norm = np.linalg.norm(n)
    return n / norm if norm > 1e-10 else np.array([0.0, 1.0, 0.0], np.float32)


def _compute_face_normals_batch(v0s, v1s, v2s) -> np.ndarray:
    """向量化批量法线计算。"""
    e1 = v1s - v0s
    e2 = v2s - v0s
    ns = np.cross(e1, e2).astype(np.float32)
    norms = np.linalg.norm(ns, axis=1, keepdims=True)
    norms = np.where(norms > 1e-10, norms, 1.0)
    return ns / norms


def _compute_tangents_batch(v0s, v1s, v2s, uv0s, uv1s, uv2s) -> np.ndarray:
    """
    向量化批量切线计算（MikkTSpace 简化版）。

    利用 UV 导数和位置导数解算切线方向（+U 方向），使切线与 UV 坐标系对齐，
    从而在着色器中正确构建 TBN 矩阵用于法线贴图变换。

    当 UV 退化（全为 0 或行列式极小）时回退到 (+1, 0, 0)。
    """
    dP1 = (v1s - v0s).astype(np.float32)          # (N, 3)
    dP2 = (v2s - v0s).astype(np.float32)          # (N, 3)
    du1 = (uv1s[:, 0] - uv0s[:, 0]).reshape(-1, 1)  # (N, 1)
    dv1 = (uv1s[:, 1] - uv0s[:, 1]).reshape(-1, 1)
    du2 = (uv2s[:, 0] - uv0s[:, 0]).reshape(-1, 1)
    dv2 = (uv2s[:, 1] - uv0s[:, 1]).reshape(-1, 1)

    det = (du1 * dv2 - du2 * dv1).astype(np.float32)  # (N, 1)
    # 行列式极小时使用 1.0 避免除零（切线结果无意义，后续用回退值替换）
    safe_det = np.where(np.abs(det) > 1e-8, det, np.ones_like(det))

    T = (dv2 * dP1 - dv1 * dP2) / safe_det         # (N, 3)

    T_norms = np.linalg.norm(T, axis=1, keepdims=True)
    fallback = np.zeros_like(T)
    fallback[:, 0] = 1.0                             # 退化时回退到 X 轴
    # 用 safe_norms 避免 T_norms=0 时触发 numpy 的 invalid divide warning
    safe_norms = np.where(T_norms > 1e-8, T_norms, np.ones_like(T_norms))
    T = np.where(T_norms > 1e-8, T / safe_norms, fallback)
    return T.astype(np.float32)
