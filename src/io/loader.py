"""
场景加载器。

SceneLoader 使用两阶段加载（满足 Taichi 初始化顺序要求）：
  1. load_config(yaml_path) — 纯 Python，返回原始 YAML 数据和 RenderConfig，可在 ti.init() 前调用。
  2. build(yaml_data, render_cfg) — 在 ti.init() 后调用，创建 Scene 和 Camera。

YAML 新特性：
  variables 段：定义命名常量，可在值中相互引用。
  字符串表达式：任何数值字段可以写成字符串，支持 variables 和 math 函数。
    例：center: [-1.8, "-half + cube_sz/2", 1.0]
        size: "cube_sz"
        width: "room"

预定义材质库：无需在 YAML 中定义，可直接通过名称引用（用户定义的同名材质优先）：
  漫反射 : red, green, blue, white, gray
  金属   : gold, silver, bronze
  介质   : glass, diamond
  光源   : light_white, light_warm
"""

import math as _math
import random
import yaml
import os
import numpy as np

from src.scene.scene import Scene
from src.scene.camera import Camera
from src.math_utils import make_rotation_matrix


# ------------------------------------------------------------------
# 供 YAML 表达式 eval 使用的安全数学环境（无 builtins，无 import）
# ------------------------------------------------------------------
_MATH_ENV = {k: getattr(_math, k) for k in dir(_math) if not k.startswith('_')}
_MATH_ENV.update({'abs': abs, 'round': round, 'min': min, 'max': max})


# ------------------------------------------------------------------
# 预定义材质库（对应原 demo 的 12 种内置材质）
# ------------------------------------------------------------------
_PRESETS = {
    # 漫反射
    'red'         : ('lambertian', {'albedo': [0.8,  0.2,  0.2 ]}),
    'green'       : ('lambertian', {'albedo': [0.2,  0.8,  0.2 ]}),
    'blue'        : ('lambertian', {'albedo': [0.2,  0.2,  0.8 ]}),
    'white'       : ('lambertian', {'albedo': [0.73, 0.73, 0.73]}),
    'gray'        : ('lambertian', {'albedo': [0.5,  0.5,  0.5 ]}),
    # 金属
    'gold'        : ('metal',      {'albedo': [1.0,  0.8,  0.2 ], 'fuzz': 0.10}),
    'silver'      : ('metal',      {'albedo': [0.8,  0.8,  0.8 ], 'fuzz': 0.05}),
    'bronze'      : ('metal',      {'albedo': [0.8,  0.5,  0.2 ], 'fuzz': 0.10}),
    # 介质
    'glass'       : ('dielectric', {'ior': 1.5}),
    'diamond'     : ('dielectric', {'ior': 2.4}),
    # 光源（emit 已乘以 intensity，intensity=1 避免二次缩放）
    'light_white' : ('light',      {'emit': [1.0, 1.0,  1.0 ], 'intensity': 8.0}),
    'light_warm'  : ('light',      {'emit': [6.0, 5.4,  4.2 ], 'intensity': 1.0}),
}


# ------------------------------------------------------------------
# 表达式求值工具
# ------------------------------------------------------------------

def _eval_expr(val, variables: dict):
    """
    将单个 YAML 值解析为 float。
    - int / float → 直接返回
    - str         → 在 variables + 数学环境中 eval
    """
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        env = {**_MATH_ENV, **variables}
        return float(eval(val, {"__builtins__": {}}, env))
    return float(val)


def _resolve(val, variables: dict):
    """
    解析 YAML 值（标量或列表），每个元素均支持字符串表达式。
    返回 float（标量）或 list[float]（向量）。
    """
    if isinstance(val, list):
        return [_eval_expr(v, variables) for v in val]
    return _eval_expr(val, variables)


# ------------------------------------------------------------------
# RenderConfig
# ------------------------------------------------------------------

class RenderConfig:
    __slots__ = [
        'width', 'height', 'spp', 'max_bounce', 'backend', 'output',
        'samples_per_batch', 'adaptive_enabled', 'adaptive_min_spp',
        'adaptive_max_spp', 'adaptive_check_interval',
        'adaptive_relative_error', 'adaptive_absolute_error', 'save_aovs',
    ]

    def __init__(self, d: dict):
        self.width      = int(d.get('width',            800))
        self.height     = int(d.get('height',           600))
        self.spp        = int(d.get('samples_per_pixel', 64))
        self.max_bounce = int(d.get('max_bounce',        16))
        self.backend    = str(d.get('backend',         'cuda'))
        self.output     = str(d.get('output',    'output.png'))
        self.samples_per_batch = max(1, int(d.get('samples_per_batch', 16)))
        self.save_aovs = bool(d.get('save_aovs', False))
        adaptive = d.get('adaptive_sampling', {})
        self.adaptive_enabled = bool(adaptive.get('enabled', True))
        self.adaptive_min_spp = max(2, int(adaptive.get('min_spp', 64)))
        self.adaptive_max_spp = max(self.adaptive_min_spp,
                                    int(adaptive.get('max_spp', self.spp)))
        self.adaptive_check_interval = max(1, int(adaptive.get('check_interval', 32)))
        self.adaptive_relative_error = float(adaptive.get('relative_error', 0.015))
        self.adaptive_absolute_error = float(adaptive.get('absolute_error', 0.001))
        if self.adaptive_enabled:
            self.spp = self.adaptive_max_spp


# ------------------------------------------------------------------
# SceneLoader
# ------------------------------------------------------------------

class SceneLoader:

    def load_config(self, yaml_path: str):
        """
        第一阶段：仅解析 YAML，提取 RenderConfig。
        不创建任何 Taichi 对象，可在 ti.init() 之前调用。
        """
        with open(yaml_path, 'r', encoding='utf-8') as f:
            yaml_data = yaml.safe_load(f)
        render_cfg = RenderConfig(yaml_data.get('render', {}))
        self._yaml_path = yaml_path
        return yaml_data, render_cfg

    def build(self, yaml_data: dict, render_cfg: RenderConfig):
        """
        第二阶段：在 ti.init() 之后调用，创建 Scene 和 Camera。
        顺序：预定义材质 → 用户材质（同名覆盖预定义）→ 变量 → 物体 → 相机。
        """
        yaml_path = getattr(self, '_yaml_path', '.')
        bg_cfg = yaml_data.get('background', [0.01, 0.01, 0.01])
        bg = bg_cfg.get('bottom', [0.01, 0.01, 0.01]) if isinstance(bg_cfg, dict) else bg_cfg

        scene = Scene(bg_color=bg)
        if isinstance(bg_cfg, dict) and bg_cfg.get('type') == 'gradient':
            scene.bg_bottom[None] = list(bg_cfg.get('bottom', bg))
            scene.bg_top[None] = list(bg_cfg.get('top', bg))
            scene.bg_gradient[None] = 1
        scene._mat_map = {}

        # 1. 预定义材质库（始终注册，用户材质可覆盖）
        self._register_presets(scene)

        # 2. 用户自定义材质（同名时覆盖预定义 map 条目）
        self._load_materials(scene, yaml_data.get('materials', []))

        # 3. 变量段：按序 eval，支持前向引用（后定义的变量可引用先定义的）
        variables: dict = {}
        for k, v in yaml_data.get('variables', {}).items():
            variables[k] = _eval_expr(v, variables)

        # 4. 场景物体
        self._load_objects(scene, yaml_data.get('objects', []), yaml_path, variables)

        # 5. 相机
        camera = self._build_camera(yaml_data.get('camera', {}), render_cfg)

        return scene, camera

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    @staticmethod
    def _register_presets(scene: Scene):
        """注册预定义材质库到场景，用户可直接在 YAML 中通过名称引用。"""
        for name, (t, params) in _PRESETS.items():
            if t == 'lambertian':
                mid = scene.materials.add_lambertian(**params)
            elif t == 'metal':
                mid = scene.materials.add_metal(**params)
            elif t == 'dielectric':
                mid = scene.materials.add_dielectric(**params)
            elif t == 'light':
                mid = scene.materials.add_light(**params)
            else:
                continue
            scene._mat_map[name] = mid

    @staticmethod
    def _load_materials(scene: Scene, mat_list: list):
        """注册 YAML 中声明的具名材质，同名时覆盖预定义。"""
        for m in mat_list:
            name     = m['name']
            t        = m.get('type', 'lambertian')
            backcull = int(m.get('backcull', 0))
            if t == 'lambertian':
                mid = scene.materials.add_lambertian(m.get('albedo', [0.8, 0.8, 0.8]),
                                                     backcull=backcull)
            elif t == 'metal':
                mid = scene.materials.add_metal(m.get('albedo', [0.8, 0.8, 0.8]),
                                                fuzz=m.get('fuzz', 0.0),
                                                backcull=backcull)
            elif t == 'dielectric':
                mid = scene.materials.add_dielectric(ior=m.get('ior', 1.5),
                                                     backcull=backcull)
            elif t == 'light':
                mid = scene.materials.add_light(m.get('emit', [1.0, 1.0, 1.0]),
                                                intensity=m.get('intensity', 1.0))
            elif t == 'clearcoat':
                mid = scene.materials.add_clearcoat(
                    albedo=m.get('albedo', [0.8, 0.8, 0.8]),
                    spec_prob=m.get('spec_prob', 0.3),
                    roughness=m.get('roughness', 0.05))
            else:
                print(f"[Loader] 未知材质类型 '{t}'，使用默认漫反射")
                mid = scene.materials.add_lambertian([0.8, 0.8, 0.8])
            scene._mat_map[name] = mid

    @staticmethod
    def _load_objects(scene: Scene, obj_list: list, yaml_path: str, variables: dict):
        base_dir = os.path.dirname(os.path.abspath(yaml_path))
        mat_map  = getattr(scene, '_mat_map', {})

        for obj in obj_list:
            mat_name = obj.get('material', None)
            mat_id   = mat_map.get(mat_name, 0) if mat_name else 0
            if mat_name and mat_name not in mat_map:
                print(f"[Loader] 未知材质 '{mat_name}'，回退到 id=0")

            t = obj.get('type', '')

            if t == 'sphere':
                scene.add_sphere(
                    _resolve(obj['center'], variables),
                    _resolve(obj['radius'], variables),
                    mat_id)

            elif t == 'rectangle':
                scene.add_rectangle(
                    _resolve(obj['center'],               variables),
                    _resolve(obj.get('width',  1.0),      variables),
                    _resolve(obj.get('height', 1.0),      variables),
                    rotation_deg=_resolve(obj.get('rotation', [0, 0, 0]), variables),
                    mat_id=mat_id)

            elif t == 'cube':
                scene.add_cube(
                    _resolve(obj['center'],               variables),
                    _resolve(obj['size'],                 variables),
                    rotation_deg=_resolve(obj.get('rotation', [0, 0, 0]), variables),
                    mat_id=mat_id)

            elif t == 'cuboid':
                scene.add_cuboid(
                    _resolve(obj['center'],               variables),
                    _resolve(obj['dimensions'],           variables),
                    rotation_deg=_resolve(obj.get('rotation', [0, 0, 0]), variables),
                    mat_id=mat_id)

            elif t == 'checkerboard':
                center = _resolve(obj.get('center', [0, 0, 0]), variables)
                tile_size = float(_resolve(obj.get('tile_size', 1.0), variables))
                tiles_x = int(obj.get('tiles_x', 8))
                tiles_z = int(obj.get('tiles_z', 8))
                names = obj.get('materials', ['white', 'gray'])
                mids = [mat_map.get(name, 0) for name in names]
                rotation = _resolve(obj.get('rotation', [90, 0, 0]), variables)
                for ix in range(tiles_x):
                    for iz in range(tiles_z):
                        x = center[0] + (ix - (tiles_x - 1) * 0.5) * tile_size
                        z = center[2] + (iz - (tiles_z - 1) * 0.5) * tile_size
                        scene.add_rectangle([x, center[1], z], tile_size, tile_size,
                                            rotation_deg=rotation,
                                            mat_id=mids[(ix * tiles_z + iz) % 2])

            elif t == 'mesh':
                model_path = os.path.join(base_dir, obj['model'])
                tf  = obj.get('transform', {})
                ext = os.path.splitext(model_path)[1].lower()
                if ext == '.obj':
                    count = scene.load_obj(
                        model_path, mat_id,
                        scale=_resolve(tf.get('scale', 1.0), variables),
                        translation=_resolve(tf.get('translate', [0, 0, 0]), variables),
                        rotation_deg=_resolve(tf.get('rotate', [0, 0, 0]), variables))
                    print(f"[Loader] OBJ {os.path.basename(model_path)} → {count} 三角形")
                elif ext in ('.glb', '.gltf'):
                    count = SceneLoader._load_glb(scene, model_path, mat_id, mat_map, obj, base_dir, variables)
                    print(f"[Loader] GLB/GLTF {os.path.basename(model_path)} → {count} 三角形")
                elif ext in ('.fbx', '.ply'):
                    count = SceneLoader._load_fbx(scene, model_path, mat_id, mat_map, obj, base_dir)
                    print(f"[Loader] FBX/PLY {os.path.basename(model_path)} → {count} 三角形")
                else:
                    print(f"[Loader] 不支持的格式: {ext}")

            elif t == 'random_spheres':
                SceneLoader._load_random_spheres(scene, obj, variables)

            else:
                print(f"[Loader] 未知对象类型 '{t}'")

    @staticmethod
    def _load_random_spheres(scene: Scene, cfg: dict, variables: dict):
        """按固定种子生成不重叠的随机球群，用于可复现旧版程序化场景。"""
        rng = random.Random(int(cfg.get('seed', 0)))
        count = int(cfg.get('count', 0))
        inner = float(_resolve(cfg.get('inner_radius', 0.0), variables))
        outer = float(_resolve(cfg.get('outer_radius', 10.0), variables))
        min_radius = float(_resolve(cfg.get('min_radius', 0.2), variables))
        max_radius = float(_resolve(cfg.get('max_radius', 1.0), variables))
        max_attempts = int(cfg.get('max_attempts', 1000))
        weights = cfg.get('material_weights', {
            'lambertian': 0.7, 'metal': 0.05, 'dielectric': 0.1, 'clearcoat': 0.15,
        })
        kinds = list(weights)
        kind_weights = [float(weights[k]) for k in kinds]
        placed = []

        for _ in range(count):
            for _attempt in range(max_attempts):
                radius = min_radius + rng.random() ** 5 * (max_radius - min_radius)
                angle = rng.uniform(0.0, 2.0 * _math.pi)
                distance = rng.uniform(inner + radius, outer - radius)
                center = [distance * _math.cos(angle), radius, -distance * _math.sin(angle)]
                if any(_math.hypot(center[0] - c[0], center[2] - c[2]) < radius + r
                       for c, r in placed):
                    continue

                kind = rng.choices(kinds, weights=kind_weights, k=1)[0]
                rng.choice(['warm', 'cool', 'neutral', 'vibrant'])  # 保留旧脚本的 RNG 消耗顺序
                color = [rng.uniform(0.0, 1.0) for _ in range(3)]
                if kind == 'metal':
                    mat_id = scene.materials.add_metal(color, fuzz=rng.uniform(0.0, 0.6))
                elif kind == 'dielectric':
                    mat_id = scene.materials.add_dielectric(ior=rng.uniform(1.45, 2.4))
                elif kind == 'clearcoat':
                    mat_id = scene.materials.add_clearcoat(
                        color, spec_prob=rng.uniform(0.1, 0.7), roughness=rng.uniform(0.05, 0.4))
                else:
                    mat_id = scene.materials.add_lambertian(color)
                scene.add_sphere(center, radius, mat_id)
                placed.append((center, radius))
                break
        print(f"[Loader] 程序化球群：请求 {count}，成功生成 {len(placed)}")

    @staticmethod
    def _build_camera(cam_cfg: dict, render_cfg: RenderConfig) -> Camera:
        return Camera(
            lookfrom     = cam_cfg.get('position',   [0, 0, 5]),
            lookat       = cam_cfg.get('look_at',    [0, 0, 0]),
            up           = cam_cfg.get('up',          [0, 1, 0]),
            fov_deg      = cam_cfg.get('fov',         45.0),
            aspect_ratio = render_cfg.width / render_cfg.height,
            aperture     = cam_cfg.get('aperture',    0.0),
            focus_dist   = cam_cfg.get('focus_dist',  None),
        )

    # ------------------------------------------------------------------
    # FBX / GLTF 加载（通过 trimesh）
    # ------------------------------------------------------------------

    @staticmethod
    def _load_fbx(scene: Scene, model_path: str, default_mat_id: int,
                  mat_map: dict, obj_cfg: dict, base_dir: str) -> int:
        """
        用 trimesh 加载 FBX/GLTF/PLY 等格式。
        每个子 mesh 独立处理：尝试匹配 mat_map 中的材质，否则使用 default_mat_id。
        坐标系处理：FBX 通常是 Z-up，转换为 Y-up（交换 Y/Z，翻转 Z）。
        """
        try:
            import trimesh
        except ImportError:
            print("[Loader] 未安装 trimesh，无法加载 FBX。请运行：pip install trimesh[easy]")
            return 0

        tf_cfg     = obj_cfg.get('transform', {})
        scale      = tf_cfg.get('scale',     1.0)
        translate  = np.array(tf_cfg.get('translate', [0, 0, 0]), np.float32)
        rotate_deg = tf_cfg.get('rotate',    [0, 0, 0])

        try:
            loaded = trimesh.load(model_path, process=False, force='scene')
        except Exception as e:
            print(f"[Loader] trimesh 加载失败: {e}")
            return 0

        total = 0

        if hasattr(loaded, 'geometry'):
            meshes = list(loaded.geometry.values())
        else:
            meshes = [loaded]

        from src.math_utils import apply_transform

        for mesh in meshes:
            if not hasattr(mesh, 'faces') or len(mesh.faces) == 0:
                continue

            verts = np.array(mesh.vertices, np.float32)
            verts = apply_transform(verts, scale, rotate_deg, translate)
            faces = np.array(mesh.faces, np.int32)

            vn = None
            if hasattr(mesh, 'vertex_normals') and mesh.vertex_normals is not None:
                vn = np.array(mesh.vertex_normals, np.float32)

            uv = None
            if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None:
                try:
                    uv = np.array(mesh.visual.uv, np.float32)
                except Exception:
                    uv = None

            mat_id = default_mat_id
            if hasattr(mesh, 'metadata') and 'name' in mesh.metadata:
                mat_id = mat_map.get(mesh.metadata['name'], default_mat_id)

            scene.add_mesh(verts, faces, mat_id, vertex_normals=vn, uvs=uv)
            total += len(faces)

        return total

    # ------------------------------------------------------------------
    # GLB / GLTF 加载（带 PBR 材质）
    # ------------------------------------------------------------------

    @staticmethod
    def _load_glb(scene, model_path: str, default_mat_id: int,
                  mat_map: dict, obj_cfg: dict, base_dir: str,
                  variables: dict = None) -> int:
        """
        用 trimesh 加载 GLB/GLTF 格式，自动提取 PBRMaterial 并注册到场景。
        每个 sub-mesh 的材质按 mat.name 去重，相同名称的材质只注册一次。

        坐标变换（translate/rotate/scale）从 obj_cfg['transform'] 读取，
        支持 YAML 变量表达式（通过 variables 参数解析），与 FBX 加载器行为一致。
        """
        try:
            import trimesh
        except ImportError:
            print("[Loader] 未安装 trimesh，无法加载 GLB。请运行：pip install trimesh[easy]")
            return 0

        if variables is None:
            variables = {}

        tf_cfg     = obj_cfg.get('transform', {})
        scale      = float(_resolve(tf_cfg.get('scale', 1.0), variables))
        translate  = np.array(_resolve(tf_cfg.get('translate', [0, 0, 0]), variables), np.float32)
        rotate_deg = _resolve(tf_cfg.get('rotate', [0, 0, 0]), variables)

        try:
            loaded = trimesh.load(model_path, process=False, force='scene')
        except Exception as e:
            print(f"[Loader] trimesh 加载 GLB 失败: {e}")
            return 0

        from src.math_utils import apply_transform

        # 获取场景图中所有 mesh 及其全局变换矩阵
        if hasattr(loaded, 'geometry'):
            # 尝试获取每个 mesh 在世界空间的变换
            geom_transforms = {}
            if hasattr(loaded, 'graph'):
                try:
                    for node_name in loaded.graph.nodes_geometry:
                        T, geom_name = loaded.graph[node_name]
                        geom_transforms[geom_name] = T
                except Exception:
                    pass
            mesh_items = list(loaded.geometry.items())
        else:
            mesh_items = [('mesh', loaded)]
            geom_transforms = {}

        # 按材质对象 id 去重，避免同名但不同贴图的材质共用一个 slot
        # （GLB 里可能有两个 material 名字相同但 baseColorTexture 不同，
        #   按 mat_name 缓存会让第二个拿到错误的贴图）
        pbr_mat_cache: dict = {}   # id(pbr_obj) → mat_id

        # material_override 支持两种格式：
        #   全局覆盖：{metallic: 0, roughness: 0.5, ...}          → 所有材质生效
        #   按名覆盖：{Tardis_Metal_Mat: {albedo: [r,g,b]}, ...}  → 值为 dict 则按材质名匹配
        # 两者可混用：全局作为默认，按名覆盖再叠加（更高优先级）
        mat_override_raw = obj_cfg.get('material_override', {})
        _per_mat_ov  = {k: v for k, v in mat_override_raw.items() if isinstance(v, dict)}
        _global_ov   = {k: v for k, v in mat_override_raw.items() if not isinstance(v, dict)}

        total = 0

        # 预计算用户旋转矩阵（仅旋转部分，用于法线变换）
        R_user = make_rotation_matrix(*rotate_deg) if any(r != 0 for r in rotate_deg) else None

        for geom_name, mesh in mesh_items:
            if not hasattr(mesh, 'faces') or len(mesh.faces) == 0:
                continue

            # 应用场景图节点变换（如有）
            verts = np.array(mesh.vertices, np.float32)
            vn = None
            if hasattr(mesh, 'vertex_normals') and mesh.vertex_normals is not None:
                vn = np.array(mesh.vertex_normals, np.float32)

            # 读取 GLB TANGENT 属性（(V,4): xyz=切线, w=手性）
            tangents = None
            try:
                tang_attr = getattr(mesh, 'vertex_attributes', None)
                if tang_attr is not None:
                    if hasattr(tang_attr, 'get'):
                        tang_data = tang_attr.get('TANGENT', None)
                    else:
                        tang_data = None
                    if tang_data is not None:
                        tangents = np.array(tang_data, np.float32)
            except Exception:
                tangents = None

            if geom_name in geom_transforms:
                T = geom_transforms[geom_name]
                ones = np.ones((len(verts), 1), dtype=np.float32)
                v4   = np.hstack([verts, ones])
                verts = (v4 @ T[:3, :].T)   # (N, 3)
                verts = verts.astype(np.float32)
                M3 = T[:3, :3].astype(np.float32)
                # 法线用场景图变换的旋转部分（scale 因子在归一化后消除）
                if vn is not None:
                    vn = (vn @ M3.T).astype(np.float32)
                # 切线 XYZ 同样用旋转矩阵变换
                if tangents is not None:
                    tang_xyz = (tangents[:, :3] @ M3.T).astype(np.float32)
                    tangents = np.hstack([tang_xyz, tangents[:, 3:4]])

            # 再应用用户指定的 transform（位置：scale + rotate + translate；法线/切线：仅 rotate）
            verts = apply_transform(verts, scale, rotate_deg, translate)
            if vn is not None and R_user is not None:
                vn = (vn @ R_user.T).astype(np.float32)
            if tangents is not None and R_user is not None:
                tang_xyz = (tangents[:, :3] @ R_user.T).astype(np.float32)
                tangents = np.hstack([tang_xyz, tangents[:, 3:4]])

            # 法线归一化（变换后可能不再是单位向量）
            if vn is not None:
                norms = np.linalg.norm(vn, axis=1, keepdims=True)
                vn = np.where(norms > 1e-8, vn / norms, vn).astype(np.float32)
            # 切线 XYZ 归一化（切线存 (V,4)，第4列手性保持不变）
            if tangents is not None:
                tn = np.linalg.norm(tangents[:, :3], axis=1, keepdims=True)
                safe_tn = np.where(tn > 1e-8, tn, np.ones_like(tn))
                tang_xyz = tangents[:, :3] / safe_tn
                tangents = np.hstack([tang_xyz, tangents[:, 3:4]])

            faces = np.array(mesh.faces, np.int32)

            # UV
            uv = None
            if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None:
                try:
                    uv = np.array(mesh.visual.uv, np.float32)
                except Exception:
                    uv = None

            # 材质：先尝试 PBRMaterial，否则回退到 default
            mat_id = default_mat_id
            if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'material'):
                pbr = mesh.visual.material
                mat_name = getattr(pbr, 'name', None) or geom_name
                cache_key = id(pbr)   # 按对象身份缓存，同名不同贴图的材质各自获得独立 slot
                if cache_key in pbr_mat_cache:
                    mat_id = pbr_mat_cache[cache_key]
                    print(f"  [GLB] 网格 '{geom_name}' → 复用材质 '{mat_name}' slot={mat_id}")
                else:
                    # 全局覆盖 + 按名覆盖合并（按名优先）
                    effective_ov = {**_global_ov, **_per_mat_ov.get(mat_name, {})}
                    mat_id = SceneLoader._create_pbr_material(scene, pbr, effective_ov)
                    pbr_mat_cache[cache_key] = mat_id
                    print(f"  [GLB] 网格 '{geom_name}' → 新材质 '{mat_name}' slot={mat_id}")

            scene.add_mesh(verts, faces, mat_id, vertex_normals=vn, uvs=uv, tangents=tangents)
            if tangents is not None:
                print(f"  [GLB] 网格 '{geom_name}' 使用逐顶点切线（{len(tangents)} 顶点）")
            total += len(faces)

        return total

    @staticmethod
    def _create_pbr_material(scene, pbr_mat, override: dict = None) -> int:
        """
        从 trimesh PBRMaterial 对象创建并注册材质。

        特殊处理：名称含 'glass' / 'Glass' 的材质改用 MAT_DIELECTRIC（ior=1.5）。
        贴图通过 scene.tex_sys.add(pil_image) 注册。
        返回材质 ID。

        override 字典支持：
          albedo_scale  : float — albedo 颜色乘以该系数（调试增亮用）
          metallic      : float — 覆盖 metallic 值（0=纯漫反射，1=纯镜面）
          roughness     : float — 覆盖 roughness 值
        """
        if override is None:
            override = {}
        mat_name = getattr(pbr_mat, 'name', '') or ''

        # 玻璃材质特判：名字含 'glass' 且 override 未指定 PBR 属性时，改用折射介质
        # 例外：GLB 材质本身带有自发光贴图（如窗户发光边框），此时保留 PBR 处理以支持自发光
        _pbr_keys = {'emissive', 'emissive_intensity', 'albedo', 'albedo_scale',
                     'metallic', 'roughness'}
        _has_glb_emissive = getattr(pbr_mat, 'emissiveTexture', None) is not None
        if 'glass' in mat_name.lower() and not any(k in override for k in _pbr_keys) and not _has_glb_emissive:
            return scene.materials.add_dielectric(ior=1.5)

        # ---- albedo（glTF 2.0 规范：baseColorFactor 缺省=1.0，最终 = factor × texture）----
        albedo       = (1.0, 1.0, 1.0)
        albedo_tex_id = -1

        base_tex = getattr(pbr_mat, 'baseColorTexture', None)
        if base_tex is not None:
            try:
                albedo_tex_id = scene.tex_sys.add(base_tex)
                import numpy as _np
                _arr = _np.array(base_tex.convert('RGB'), dtype=_np.float32) / 255.0
                print(f"  [GLB] '{mat_name}' baseColorTex 均值 RGB = "
                      f"({_arr[...,0].mean():.3f}, {_arr[...,1].mean():.3f}, {_arr[...,2].mean():.3f})")
            except Exception as e:
                print(f"  [GLB] baseColorTexture 加载失败: {e}")

        base_factor = getattr(pbr_mat, 'baseColorFactor', None)
        if base_factor is not None:
            try:
                albedo = (float(base_factor[0]),
                          float(base_factor[1]),
                          float(base_factor[2]))
            except Exception:
                pass
        print(f"  [GLB] '{mat_name}' baseColorFactor={albedo}, albedo_tex_id={albedo_tex_id}")

        # ---- metallic / roughness（glTF 2.0 规范：metallicFactor/roughnessFactor 缺省=1.0）----
        # 缺省 1.0 意味着：
        #   1) 有 mrTexture 时最终值 = texture 通道值（factor=1 不影响）
        #   2) 无 mrTexture 时默认为完全金属 + 完全粗糙，作者应显式声明
        metallic  = 1.0
        roughness = 1.0
        mr_tex_id = -1

        mr_factor = getattr(pbr_mat, 'metallicFactor', None)
        if mr_factor is not None:
            try:
                metallic = float(mr_factor)
            except Exception:
                pass

        r_factor = getattr(pbr_mat, 'roughnessFactor', None)
        if r_factor is not None:
            try:
                roughness = float(r_factor)
            except Exception:
                pass

        mr_tex = getattr(pbr_mat, 'metallicRoughnessTexture', None)
        if mr_tex is not None:
            try:
                mr_tex_id = scene.tex_sys.add(mr_tex)
                import numpy as _np
                _arr = _np.array(mr_tex.convert('RGB'), dtype=_np.float32) / 255.0
                print(f"  [GLB] '{mat_name}' mrTex 均值 G(rough)={_arr[...,1].mean():.3f} "
                      f"B(metal)={_arr[...,2].mean():.3f}")
            except Exception as e:
                print(f"  [GLB] metallicRoughnessTexture 加载失败: {e}")
        print(f"  [GLB] '{mat_name}' metallic={metallic:.2f}, roughness={roughness:.2f}, mr_tex_id={mr_tex_id}")

        # ---- emissive ----
        emissive       = (0.0, 0.0, 0.0)
        emissive_tex_id = -1

        e_factor = getattr(pbr_mat, 'emissiveFactor', None)
        if e_factor is not None:
            try:
                emissive = (float(e_factor[0]),
                            float(e_factor[1]),
                            float(e_factor[2]))
            except Exception:
                pass

        e_tex = getattr(pbr_mat, 'emissiveTexture', None)
        if e_tex is not None:
            try:
                emissive_tex_id = scene.tex_sys.add(e_tex)
            except Exception as e:
                print(f"  [GLB] emissiveTexture 加载失败: {e}")
        print(f"  [GLB] '{mat_name}' emissiveFactor={tuple(f'{v:.3f}' for v in emissive)}, "
              f"emissive_tex_id={emissive_tex_id}")

        # ---- normal map ----
        normal_tex_id = -1
        n_tex = getattr(pbr_mat, 'normalTexture', None)
        if n_tex is not None:
            try:
                normal_tex_id = scene.tex_sys.add(n_tex)
                print(f"  [GLB] '{mat_name}' normalTex slot={normal_tex_id}")
            except Exception as e:
                print(f"  [GLB] normalTexture 加载失败: {e}")

        # ---- 处理 baseColorFactor > 1（违反 glTF 规范，常见于 Blender 导出）----
        # glTF 规范要求 baseColorFactor ∈ [0,1]；超过 1.0 时路径中 att = albedo > 1
        # 导致每次弹射能量放大（而非衰减），最终 float32 累积溢出。
        # 修复：仅将 hue 保留，等比归一化到最大分量 = 1.0。
        # 注意：不自动转移到 emissive —— baseColorFactor 超标可能只是 Blender 材质
        # 节点中的亮度倍增器，并不代表物理自发光意图。
        max_c = max(albedo)
        if max_c > 1.0:
            albedo = tuple(c / max_c for c in albedo)  # 等比缩放，保留色相
            print(f"  [GLB] '{mat_name}' baseColorFactor>1，归一化 → {tuple(f'{v:.3f}' for v in albedo)}")

        # ---- 应用 material_override ----
        if 'albedo' in override:
            a = override['albedo']
            albedo = (float(a[0]), float(a[1]), float(a[2]))
            albedo_tex_id = -1
        elif 'albedo_scale' in override:
            s = float(override['albedo_scale'])
            albedo = tuple(min(1.0, c * s) for c in albedo)
        if 'metallic' in override or 'roughness' in override:
            # 覆盖 metallic/roughness 时必须丢弃 mr 贴图，否则 _scatter_pbr 运行时会用贴图覆盖标量值
            mr_tex_id = -1
            if 'metallic' in override:
                metallic = float(override['metallic'])
            if 'roughness' in override:
                roughness = float(override['roughness'])
        if 'emissive' in override:
            # 手动指定自发光颜色；emissive_intensity 为可选倍率（默认 1.0）
            e = override['emissive']
            intensity = float(override.get('emissive_intensity', 1.0))
            emissive = (float(e[0]) * intensity,
                        float(e[1]) * intensity,
                        float(e[2]) * intensity)
            emissive_tex_id = -1   # 丢弃 GLB 自发光贴图，使用手动值

        return scene.materials.add_pbr(
            albedo=albedo,
            albedo_tex_id=albedo_tex_id,
            metallic=metallic,
            roughness=roughness,
            mr_tex_id=mr_tex_id,
            emissive=emissive,
            emissive_tex_id=emissive_tex_id,
            normal_tex_id=normal_tex_id,
        )
