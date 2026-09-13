"""
材质系统。

MaterialSystem 存储所有材质参数到 Taichi field（AoS 布局），
并提供统一的 scatter() @ti.func 入口，按 mat_type 分发到各 BSDF。

材质结构字段：
  mat_type        : 材质类型（见 constants.MAT_*）
  albedo          : 基础颜色 / 漫反射率
  emit            : 自发光颜色
  fuzz            : 金属模糊度（MAT_METAL）
  ior             : 折射率（MAT_DIELECTRIC）
  roughness       : 粗糙度（MAT_CLEARCOAT / MAT_PBR）
  spec_prob       : 镜面反射概率（MAT_CLEARCOAT）/ metallic（MAT_PBR）
  albedo_tex_id   : albedo 贴图索引，-1 表示无贴图
  normal_tex_id   : 法线贴图索引（预留）
  mr_tex_id       : metallicRoughness 贴图索引，-1=无（MAT_PBR）
  emissive_tex_id : 自发光贴图索引，-1=无（MAT_PBR）
  emissive_factor : 自发光系数（MAT_PBR）

backcull：材质背面不参与散射，光线穿透。用于单面光源等场景。
"""

import taichi as ti
import numpy as np

from src.constants import (
    MAT_LAMBERTIAN, MAT_METAL, MAT_DIELECTRIC, MAT_LIGHT, MAT_CLEARCOAT, MAT_PBR,
)
from src.math_utils import (
    random_in_unit_sphere, random_cosine_hemisphere, reflect, refract, schlick,
    sample_ggx_h, smith_g1_ggx, fresnel_schlick_vec,
)


# Taichi 材质结构类型（模块级定义，复用于 field 声明）
_MatStruct = ti.types.struct(
    mat_type        = ti.i32,
    albedo          = ti.types.vector(3, ti.f32),
    emit            = ti.types.vector(3, ti.f32),
    fuzz            = ti.f32,
    ior             = ti.f32,
    roughness       = ti.f32,
    spec_prob       = ti.f32,
    backcull        = ti.i32,   # 1 = 背面不散射
    albedo_tex_id   = ti.i32,   # -1 = 无贴图
    normal_tex_id   = ti.i32,
    mr_tex_id       = ti.i32,   # metallicRoughness 贴图（G=roughness, B=metallic）
    emissive_tex_id = ti.i32,   # 自发光贴图索引
    emissive_factor = ti.types.vector(3, ti.f32),  # 自发光颜色系数
)


@ti.data_oriented
class MaterialSystem:
    """
    统一管理所有材质。

    Python 端通过 add_*() 方法注册材质并获得 ID，
    调用 bake() 后数据上传至 GPU，之后 scatter() 可在 kernel 内使用。
    """

    def __init__(self, max_materials: int = 512):
        self._max   = max_materials
        self.mats   = None
        self._count = 0
        # Python 端缓冲
        self._buf = {
            'mat_type'       : [],
            'albedo'         : [],
            'emit'           : [],
            'fuzz'           : [],
            'ior'            : [],
            'roughness'      : [],
            'spec_prob'      : [],
            'backcull'       : [],
            'albedo_tex_id'  : [],
            'normal_tex_id'  : [],
            'mr_tex_id'      : [],
            'emissive_tex_id': [],
            'emissive_factor': [],
        }

    # ------------------------------------------------------------------
    # Python 端：材质注册
    # ------------------------------------------------------------------

    def _add(self, mat_type, albedo=(0.8, 0.8, 0.8), emit=(0.0, 0.0, 0.0),
             fuzz=0.0, ior=1.5, roughness=0.5, spec_prob=0.0,
             backcull=0,
             albedo_tex_id=-1, normal_tex_id=-1,
             mr_tex_id=-1, emissive_tex_id=-1,
             emissive_factor=(0.0, 0.0, 0.0)) -> int:
        assert self._count < self._max, f"MaterialSystem 超出容量 {self._max}"
        b = self._buf
        b['mat_type'       ].append(int(mat_type))
        b['albedo'         ].append(np.array(albedo,          np.float32))
        b['emit'           ].append(np.array(emit,            np.float32))
        b['fuzz'           ].append(float(fuzz))
        b['ior'            ].append(float(ior))
        b['roughness'      ].append(float(roughness))
        b['spec_prob'      ].append(float(spec_prob))
        b['backcull'       ].append(int(backcull))
        b['albedo_tex_id'  ].append(int(albedo_tex_id))
        b['normal_tex_id'  ].append(int(normal_tex_id))
        b['mr_tex_id'      ].append(int(mr_tex_id))
        b['emissive_tex_id'].append(int(emissive_tex_id))
        b['emissive_factor'].append(np.array(emissive_factor, np.float32))
        idx = self._count
        self._count += 1
        return idx

    def add_lambertian(self, albedo, backcull=0) -> int:
        return self._add(MAT_LAMBERTIAN, albedo=albedo, backcull=backcull)

    def add_metal(self, albedo, fuzz: float = 0.0, backcull=0) -> int:
        return self._add(MAT_METAL, albedo=albedo, fuzz=fuzz, backcull=backcull)

    def add_dielectric(self, ior: float = 1.5, albedo=(1.0, 1.0, 1.0), backcull=0) -> int:
        return self._add(MAT_DIELECTRIC, albedo=albedo, ior=ior, backcull=backcull)

    def add_light(self, emit, intensity: float = 1.0, backcull=1) -> int:
        e = [emit[i] * intensity for i in range(3)]
        return self._add(MAT_LIGHT, emit=e, backcull=backcull)

    def add_clearcoat(self, albedo, spec_prob: float = 0.3,
                      roughness: float = 0.05, backcull=0) -> int:
        return self._add(MAT_CLEARCOAT, albedo=albedo,
                         spec_prob=spec_prob, roughness=roughness, backcull=backcull)

    def add_pbr(self, albedo=(1, 1, 1), albedo_tex_id=-1,
                metallic=0.0, roughness=0.5, mr_tex_id=-1,
                emissive=(0, 0, 0), emissive_tex_id=-1,
                normal_tex_id=-1) -> int:
        """
        注册 PBR 材质（glTF metallic-roughness 模型）。
        spec_prob 字段存储 metallic，roughness 字段存储 roughness。
        """
        return self._add(MAT_PBR, albedo=albedo, albedo_tex_id=albedo_tex_id,
                         spec_prob=metallic, roughness=roughness, mr_tex_id=mr_tex_id,
                         emissive_tex_id=emissive_tex_id, emissive_factor=emissive,
                         normal_tex_id=normal_tex_id)

    # ------------------------------------------------------------------
    # Python 端：一次性上传
    # ------------------------------------------------------------------

    def bake(self):
        """上传所有材质到 GPU。在渲染前调用一次。"""
        n = self._count
        capacity = max(n, 1)
        self.mats = _MatStruct.field(shape=(capacity,))
        b = self._buf

        def scalar(name, dtype):
            out = np.zeros(capacity, dtype)
            if n:
                out[:n] = np.asarray(b[name], dtype=dtype)
            return out

        def vector(name):
            out = np.zeros((capacity, 3), np.float32)
            if n:
                out[:n] = np.stack(b[name]).astype(np.float32)
            return out

        self.mats.mat_type.from_numpy(scalar('mat_type', np.int32))
        self.mats.albedo.from_numpy(vector('albedo'))
        self.mats.emit.from_numpy(vector('emit'))
        self.mats.fuzz.from_numpy(scalar('fuzz', np.float32))
        self.mats.ior.from_numpy(scalar('ior', np.float32))
        self.mats.roughness.from_numpy(scalar('roughness', np.float32))
        self.mats.spec_prob.from_numpy(scalar('spec_prob', np.float32))
        self.mats.backcull.from_numpy(scalar('backcull', np.int32))
        self.mats.albedo_tex_id.from_numpy(scalar('albedo_tex_id', np.int32))
        self.mats.normal_tex_id.from_numpy(scalar('normal_tex_id', np.int32))
        self.mats.mr_tex_id.from_numpy(scalar('mr_tex_id', np.int32))
        self.mats.emissive_tex_id.from_numpy(scalar('emissive_tex_id', np.int32))
        self.mats.emissive_factor.from_numpy(vector('emissive_factor'))
        print(f"[Material] 已烘焙 {n} 个材质")

    @property
    def count(self) -> int:
        return self._count

    # ------------------------------------------------------------------
    # Taichi 作用域：统一散射入口
    # ------------------------------------------------------------------

    @ti.func
    def scatter(self, ray_dir, normal, hit_pos, mat_id: ti.i32, hit_uv, front_face: ti.i32,
                hit_tangent, tex_sys: ti.template()):
        """
        统一散射函数。
        返回 (scattered_dir, attenuation, emitted, should_scatter, is_specular)。
        should_scatter == False 时光线被吸收（如光源），调用方应终止路径。
        is_specular == True 表示本次弹射为镜面反射，NEE 不能在此处采样直接光
        （镜面材质依赖路径追踪本身连接光源，同时做 NEE 会双重计数）。
        hit_tangent 用于 PBR 材质的法线贴图 TBN 变换。
        """
        mat = self.mats[mat_id]

        scattered    = ti.Vector([0.0, 0.0, 0.0])
        attenuation  = ti.Vector([1.0, 1.0, 1.0])
        emitted      = ti.Vector([0.0, 0.0, 0.0])
        should       = False
        is_specular  = False
        scatter_pdf  = 0.0

        # 背面剔除：光线从背面命中且材质设置了 backcull，则穿透不散射
        if mat.backcull and not front_face:
            scattered   = ray_dir
            should      = True
            is_specular = True   # 穿透视为镜面（不做 NEE）
        else:
            if mat.mat_type == MAT_LAMBERTIAN:
                scattered, attenuation, should = _scatter_lambertian(normal, mat)
                is_specular = False
                scatter_pdf = ti.max(0.0, scattered.dot(normal)) / 3.141592653589793
            elif mat.mat_type == MAT_METAL:
                scattered, attenuation, should = _scatter_metal(ray_dir, normal, mat)
                is_specular = True
            elif mat.mat_type == MAT_DIELECTRIC:
                scattered, attenuation, should = _scatter_dielectric(ray_dir, normal, mat, front_face)
                is_specular = True
            elif mat.mat_type == MAT_LIGHT:
                emitted     = mat.emit
                should      = False
                is_specular = False
            elif mat.mat_type == MAT_CLEARCOAT:
                scattered, attenuation, should, is_specular = _scatter_clearcoat(ray_dir, normal, mat)
                if not is_specular:
                    scatter_pdf = (1.0 - mat.spec_prob) * \
                                  ti.max(0.0, scattered.dot(normal)) / 3.141592653589793
            elif mat.mat_type == MAT_PBR:
                scattered, attenuation, emitted, should, is_specular, scatter_pdf = _scatter_pbr(
                    ray_dir, normal, hit_uv, hit_tangent, mat, tex_sys)

        return scattered, attenuation, emitted, should, is_specular, scatter_pdf


# ------------------------------------------------------------------
# 各 BSDF 实现（模块级 @ti.func，被 MaterialSystem.scatter 调用）
# ------------------------------------------------------------------

@ti.func
def _scatter_lambertian(normal, mat):
    """朗伯漫反射：余弦加权半球采样。"""
    return random_cosine_hemisphere(normal), mat.albedo, True


@ti.func
def _scatter_metal(ray_dir, normal, mat):
    """金属镜面反射，fuzz 控制模糊度。"""
    r    = reflect(ray_dir.normalized(), normal)
    scat = r + mat.fuzz * random_in_unit_sphere()
    # 散射方向与法线同侧才有效
    ok   = scat.dot(normal) > 0.0
    return scat, mat.albedo, ok


@ti.func
def _scatter_dielectric(ray_dir, normal, mat, front_face):
    """Snell 折射 + Schlick Fresnel，全内反射时强制反射。"""
    eta = 1.0 / mat.ior if front_face else mat.ior
    unit_dir  = ray_dir.normalized()
    cos_theta = ti.min(-unit_dir.dot(normal), 1.0)
    sin_theta = ti.sqrt(1.0 - cos_theta * cos_theta)
    cannot_refract = eta * sin_theta > 1.0

    scat = ti.Vector([0.0, 0.0, 0.0])
    if cannot_refract or schlick(cos_theta, eta) > ti.random():
        scat = reflect(unit_dir, normal)
    else:
        scat = refract(unit_dir, normal, eta)
    return scat, ti.Vector([1.0, 1.0, 1.0]), True


@ti.func
def _scatter_clearcoat(ray_dir, normal, mat):
    """
    清漆材质：以 spec_prob 概率选择镜面反射，否则漫反射。
    镜面使用 roughness 控制模糊度。
    返回 (scat, att, should_scatter, is_specular)。
    """
    scat        = ti.Vector([0.0, 0.0, 0.0])
    att         = ti.Vector([1.0, 1.0, 1.0])
    is_specular = False
    scatter_pdf = 0.0
    if ti.random() < mat.spec_prob:
        r    = reflect(ray_dir.normalized(), normal)
        scat = r + mat.roughness * random_in_unit_sphere()
        is_specular = True
    else:
        scat = random_cosine_hemisphere(normal)
        att  = mat.albedo
    return scat, att, True, is_specular


@ti.func
def _scatter_pbr(ray_dir, normal, hit_uv, hit_tangent, mat, tex_sys: ti.template()):
    """
    PBR 材质（glTF 2.0 metallic-roughness 模型）—— Cook-Torrance 微表面 BSDF。

    实现要点（物理正确）：
      1. baseColor / metallic / roughness / emissive 均为「factor × texture」组合
         （glTF 规范：texture 存在时最终值 = factor · texture，不是替换）。
      2. F0 = mix(0.04, albedo, metallic)：介质默认 4% Fresnel，金属反射色 = albedo。
      3. 镜面：GGX-D 重要性采样微表面法线 h，反射得到出射方向 L，
         抛掷时用 Smith G(V,L) 遮蔽项与 F(V·H) 组合。α = roughness²（Disney 约定）。
         throughput = F · G · (V·H) / ((N·V)·(N·H)) / spec_prob    (D 项被 pdf 消掉)
      4. 漫反射：仅介质有效（金属 diffuse=0）；能量守恒地乘以 (1 - F_avg(V·N))。
         throughput = albedo · (1 - metallic) · (1 - F(V·N)) / (1 - spec_prob)
      5. 分支概率 spec_prob = luminance(F(V·N))，用 Fresnel 引导采样。

    返回：(scattered_dir, attenuation, emitted, should_scatter, is_specular)
    """
    # ---- 1. baseColor：factor × texture（glTF 规范），钳制到 [0,1] ----
    albedo = mat.albedo
    if mat.albedo_tex_id >= 0:
        albedo = albedo * tex_sys.sample(mat.albedo_tex_id, hit_uv[0], hit_uv[1])
    albedo = ti.min(albedo, 1.0)

    # ---- 2. metallic / roughness：factor × texture（G=rough, B=metal）----
    metallic  = mat.spec_prob
    roughness = mat.roughness
    if mat.mr_tex_id >= 0:
        mr = tex_sys.sample(mat.mr_tex_id, hit_uv[0], hit_uv[1])
        roughness = roughness * mr[1]
        metallic  = metallic  * mr[2]
    # roughness 下限：防止 α→0 时 GGX 分布退化为狄拉克，采样数值发散
    roughness = ti.max(0.04, ti.min(1.0, roughness))
    metallic  = ti.max(0.0,  ti.min(1.0, metallic))
    alpha     = roughness * roughness

    # ---- 3. 自发光 ----
    emitted = mat.emissive_factor
    if mat.emissive_tex_id >= 0:
        emitted = emitted * tex_sys.sample(mat.emissive_tex_id, hit_uv[0], hit_uv[1])

    # ---- 4. 法线贴图：TBN 变换到世界空间 ----
    shading_normal = normal
    if mat.normal_tex_id >= 0:
        nm_raw = tex_sys.sample(mat.normal_tex_id, hit_uv[0], hit_uv[1])
        nm = nm_raw * 2.0 - 1.0
        tangent_xyz = ti.Vector([hit_tangent[0], hit_tangent[1], hit_tangent[2]])
        tangent_sign = hit_tangent[3]
        T = tangent_xyz - tangent_xyz.dot(normal) * normal
        T_len = T.norm()
        if T_len > 1e-6:
            T = T / T_len
        else:
            aux = ti.Vector([0.0, 1.0, 0.0])
            if ti.abs(normal[1]) > 0.9:
                aux = ti.Vector([1.0, 0.0, 0.0])
            T = (aux - aux.dot(normal) * normal).normalized()
        B = normal.cross(T) * tangent_sign
        shading_normal = (T * nm[0] + B * nm[1] + normal * nm[2]).normalized()
        if shading_normal.dot(normal) < 0.0:
            shading_normal = normal

    # ---- 5. Fresnel-guided 分支：镜面 vs 漫反射 ----
    V = -ray_dir.normalized()   # 视线方向（从表面指向摄像机侧）
    NdotV = ti.max(1e-4, V.dot(shading_normal))

    # F0：介质 = 0.04，金属 = albedo；线性混合
    dielectric_F0 = ti.Vector([0.04, 0.04, 0.04])
    F0 = dielectric_F0 * (1.0 - metallic) + albedo * metallic

    # 视线角 Fresnel（用于分支概率与漫反射能量补偿）
    F_v = fresnel_schlick_vec(NdotV, F0)
    # 平均通道作为标量分支概率，钳到 (0.02, 0.98) 避免除零
    spec_prob = ti.max(0.02, ti.min(0.98,
                (F_v[0] + F_v[1] + F_v[2]) * (1.0 / 3.0)))

    scat        = ray_dir
    att         = ti.Vector([0.0, 0.0, 0.0])
    should      = True
    is_specular = False
    scatter_pdf = 0.0

    if ti.random() < spec_prob:
        # ---- 镜面分支：GGX-D 重要性采样 H → L ----
        u1 = ti.random()
        u2 = ti.random()
        H  = sample_ggx_h(alpha, shading_normal, u1, u2)
        L  = reflect(-V, H)              # L = 2(V·H)H - V

        NdotL = L.dot(shading_normal)
        VdotH = V.dot(H)
        NdotH = shading_normal.dot(H)

        if NdotL > 1e-4 and VdotH > 1e-4 and NdotH > 1e-4:
            # 微表面处的 Fresnel（V·H 更精确）
            F_h = fresnel_schlick_vec(VdotH, F0)
            # Smith 双向遮蔽 G(V,L)
            G   = smith_g1_ggx(NdotV, alpha) * smith_g1_ggx(NdotL, alpha)
            # BRDF-with-D-sampling 结果：D 项与 PDF(L) 消掉后剩下：
            #   throughput = F · G · (V·H) / ((N·V)·(N·H))
            att = F_h * (G * VdotH / (NdotV * NdotH * spec_prob))
            scat = L.normalized()
            is_specular = True
        else:
            # 反射方向落到面下（掠射角 GGX 会偶发）→ 终止路径，避免有偏
            should = False
    else:
        # ---- 漫反射分支：Lambert，能量守恒 ----
        # 金属：(1 - metallic) = 0 → att = 0，本次贡献消失（等价于金属无漫反射）
        # 介质：乘以 (1 - F(V·N)) 保证 Fresnel 反射带走的能量不重复计入 diffuse
        scat = random_cosine_hemisphere(shading_normal)
        one  = ti.Vector([1.0, 1.0, 1.0])
        att  = albedo * (1.0 - metallic) * (one - F_v) * (1.0 / (1.0 - spec_prob))
        is_specular = False
        scatter_pdf = (1.0 - spec_prob) * ti.max(0.0, scat.dot(shading_normal)) / 3.141592653589793

    return scat, att, emitted, should, is_specular, scatter_pdf
