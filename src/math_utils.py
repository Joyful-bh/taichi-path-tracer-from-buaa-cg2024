import taichi as ti
import numpy as np


# ===== Taichi 作用域：采样工具 =====

@ti.func
def random_in_unit_sphere():
    """拒绝采样：单位球内均匀随机向量。"""
    p = ti.Vector([0.0, 0.0, 0.0])
    while True:
        p = 2.0 * ti.Vector([ti.random(), ti.random(), ti.random()]) - 1.0
        if p.norm_sqr() <= 1.0:
            break
    return p


@ti.func
def random_cosine_hemisphere(normal: ti.template()):
    """余弦加权半球采样，朝向 normal 方向。"""
    u1 = ti.random()
    u2 = ti.random()
    r = ti.sqrt(u1)
    phi = 2.0 * 3.141592653589793 * u2
    local = ti.Vector([r * ti.cos(phi), r * ti.sin(phi), ti.sqrt(1.0 - u1)])
    tangent, bitangent = build_onb(normal)
    return (tangent * local[0] + bitangent * local[1] + normal * local[2]).normalized()


@ti.func
def random_in_unit_disk():
    """拒绝采样：单位圆盘内均匀随机向量（用于景深）。"""
    p = ti.Vector([0.0, 0.0])
    while True:
        p = 2.0 * ti.Vector([ti.random(), ti.random()]) - ti.Vector([1.0, 1.0])
        if p.norm_sqr() <= 1.0:
            break
    return p


# ===== Taichi 作用域：光学计算 =====

@ti.func
def reflect(v, n):
    """计算镜面反射方向。v 为入射方向（指向曲面），n 为法线（朝外）。"""
    return v - 2.0 * v.dot(n) * n


@ti.func
def refract(uv, n, eta):
    """Snell 折射。uv 为归一化入射方向，n 为法线，eta = n_in/n_out。"""
    cos_theta = ti.min(-uv.dot(n), 1.0)
    r_perp    = eta * (uv + cos_theta * n)
    r_parallel = -ti.sqrt(ti.abs(1.0 - r_perp.norm_sqr())) * n
    return r_perp + r_parallel


@ti.func
def schlick(cosine, ref_idx):
    """Schlick 近似：计算 Fresnel 反射率。"""
    r0 = (1.0 - ref_idx) / (1.0 + ref_idx)
    r0 = r0 * r0
    return r0 + (1.0 - r0) * ti.pow(1.0 - cosine, 5.0)


# ===== Taichi 作用域：GGX 微表面 BSDF 工具 =====

PI_F = 3.141592653589793


@ti.func
def build_onb(n):
    """由法线 n 构造正交基 (T, B)：n 为局部 z 轴，T/B 为切平面基。"""
    up = ti.Vector([0.0, 1.0, 0.0])
    if ti.abs(n[1]) > 0.999:
        up = ti.Vector([1.0, 0.0, 0.0])
    T = up.cross(n).normalized()
    B = n.cross(T)
    return T, B


@ti.func
def sample_ggx_h(alpha, n, u1, u2):
    """
    从 GGX(Trowbridge-Reitz) 分布采样微表面法线 h，返回世界空间单位向量。
    参数 alpha = roughness²（Disney/glTF 约定）。

    采样公式（isotropic GGX）：
      φ  = 2π · u1
      cosθ = sqrt((1-u2) / (1 + (α²-1)·u2))
    """
    a2   = alpha * alpha
    phi  = 2.0 * PI_F * u1
    cos_th = ti.sqrt((1.0 - u2) / (1.0 + (a2 - 1.0) * u2))
    sin_th = ti.sqrt(ti.max(0.0, 1.0 - cos_th * cos_th))
    T, B = build_onb(n)
    return (T * (sin_th * ti.cos(phi))
          + B * (sin_th * ti.sin(phi))
          + n * cos_th).normalized()


@ti.func
def smith_g1_ggx(cos_theta, alpha):
    """Smith GGX 单向遮蔽项 G1（可见性）。cos_theta 为方向与法线夹角余弦。"""
    a2 = alpha * alpha
    c2 = cos_theta * cos_theta
    return 2.0 * cos_theta / (cos_theta + ti.sqrt(a2 + (1.0 - a2) * c2))


@ti.func
def fresnel_schlick_vec(cos_theta, F0):
    """向量形式 Schlick Fresnel：F0 为 3 通道颜色。cos_theta = V·H 或 V·N。"""
    one = ti.Vector([1.0, 1.0, 1.0])
    m   = ti.max(0.0, 1.0 - cos_theta)
    m2  = m * m
    return F0 + (one - F0) * (m2 * m2 * m)


# ===== Python 作用域：变换矩阵 =====

def make_rotation_matrix(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    """构造 ZYX 顺序旋转矩阵（先绕 Z，再绕 Y，再绕 X）。"""
    rx = np.radians(rx_deg)
    ry = np.radians(ry_deg)
    rz = np.radians(rz_deg)

    Rx = np.array([
        [1, 0,           0          ],
        [0, np.cos(rx), -np.sin(rx)],
        [0, np.sin(rx),  np.cos(rx)],
    ], dtype=np.float32)

    Ry = np.array([
        [ np.cos(ry), 0, np.sin(ry)],
        [ 0,          1, 0          ],
        [-np.sin(ry), 0, np.cos(ry)],
    ], dtype=np.float32)

    Rz = np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz),  np.cos(rz), 0],
        [0,           0,          1],
    ], dtype=np.float32)

    return Rx @ Ry @ Rz


def apply_transform(vertices: np.ndarray,
                    scale: float = 1.0,
                    rotation_deg=(0.0, 0.0, 0.0),
                    translation=(0.0, 0.0, 0.0)) -> np.ndarray:
    """对 (N,3) 顶点数组依次应用缩放、旋转、平移。"""
    v = vertices.copy().astype(np.float32)
    if scale != 1.0:
        v *= scale
    if any(r != 0 for r in rotation_deg):
        R = make_rotation_matrix(*rotation_deg)
        v = v @ R.T
    t = np.array(translation, dtype=np.float32)
    if np.any(t != 0):
        v += t
    return v
