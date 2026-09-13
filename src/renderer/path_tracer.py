"""
路径追踪积分器。

PathTracer 持有：
  - accumulator  : (W, H, 3) float32 Taichi 场，累积每像素的线性辐亮度之和。
  - pixel_sample_count : 每个像素已累积的采样数。

render_batch(camera, scene, mat_sys, max_bounce, samples_per_batch, ...)
  - 每次 kernel 调用为活跃像素追加一批路径样本。
  - 多次调用自动积分（渐进式渲染）。

get_image()
  - 返回已归一化（除以 sample_count）的 numpy (H, W, 3) float32 图像。

_ray_color (ti.func)
  - 核心路径追踪循环：最多 max_bounce 次反弹。
  - 未命中时返回 scene.bg_color。
  - 俄罗斯轮盘赌（Russian Roulette）在第 5 次反弹后启用，减少噪声偏差。
"""

import taichi as ti
import numpy as np

from src.constants import (
    T_MIN, T_MAX, MAT_LIGHT, MAT_CLEARCOAT, MAT_PBR,
)

_PI = 3.14159265358979


@ti.data_oriented
class PathTracer:
    def __init__(self, width: int, height: int,
                 enable_aovs: bool = False, enable_adaptive: bool = False):
        self.width  = width
        self.height = height
        self.enable_aovs = bool(enable_aovs)
        self.enable_adaptive = bool(enable_adaptive)
        # 线性辐亮度累积缓冲（GPU 端，浮点精度）
        self.accumulator  = ti.Vector.field(3, ti.f32, shape=(width, height))
        self.sample_count = ti.field(ti.i32, shape=())
        count_shape = (width, height) if self.enable_adaptive else (1, 1)
        self.pixel_sample_count = ti.field(ti.i32, shape=count_shape)
        adaptive_shape = (width, height) if self.enable_adaptive else (1, 1)
        aux_shape = ((width, height)
                     if self.enable_aovs or self.enable_adaptive else (1, 1))
        self.converged = ti.field(ti.i32, shape=adaptive_shape)
        self.active_count = ti.field(ti.i32, shape=())
        self.lum_mean = ti.field(ti.f32, shape=aux_shape)
        self.lum_m2 = ti.field(ti.f32, shape=aux_shape)
        self.aov_albedo = ti.Vector.field(3, ti.f32, shape=aux_shape)
        self.aov_normal = ti.Vector.field(3, ti.f32, shape=aux_shape)
        self.aov_depth = ti.field(ti.f32, shape=aux_shape)
        self.sample_count[None] = 0
        self.active_count[None] = width * height

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    @ti.kernel
    def _render_batch_fixed(self, camera: ti.template(), scene: ti.template(),
                            mat_sys: ti.template(), tex_sys: ti.template(),
                            light_sampler: ti.template(), max_bounce: ti.i32,
                            samples_per_batch: ti.i32, max_spp: ti.i32):
        """固定采样快速路径：整批在寄存器中累积，最后一次写回。"""
        for px, py in self.accumulator:
            color_sum = ti.Vector([0.0, 0.0, 0.0])
            actual = ti.min(samples_per_batch, max_spp - self.sample_count[None])
            for _ in range(actual):
                s = (px + ti.random()) / self.width
                t = (py + ti.random()) / self.height
                ray_o, ray_d = camera.get_ray(s, t)
                color, sample_albedo, sample_normal, sample_depth = self._ray_color(
                    ray_o, ray_d, scene, mat_sys, tex_sys, light_sampler, max_bounce)
                color_sum += color
            self.accumulator[px, py] += color_sum
        self.sample_count[None] = ti.min(
            self.sample_count[None] + samples_per_batch, max_spp)

    @ti.kernel
    def _render_batch_fixed_aov(self, camera: ti.template(), scene: ti.template(),
                                mat_sys: ti.template(), tex_sys: ti.template(),
                                light_sampler: ti.template(), max_bounce: ti.i32,
                                samples_per_batch: ti.i32, max_spp: ti.i32):
        """固定采样 + AOV；所有缓冲均每批只写回一次。"""
        for px, py in self.accumulator:
            color_sum = ti.Vector([0.0, 0.0, 0.0])
            albedo_sum = ti.Vector([0.0, 0.0, 0.0])
            normal_sum = ti.Vector([0.0, 0.0, 0.0])
            depth_sum = 0.0
            batch_lum_mean = 0.0
            batch_lum_m2 = 0.0
            batch_n = 0
            actual = ti.min(samples_per_batch, max_spp - self.sample_count[None])
            for _ in range(actual):
                s = (px + ti.random()) / self.width
                t = (py + ti.random()) / self.height
                ray_o, ray_d = camera.get_ray(s, t)
                color, albedo, normal, depth = self._ray_color(
                    ray_o, ray_d, scene, mat_sys, tex_sys, light_sampler, max_bounce)
                color_sum += color
                albedo_sum += albedo
                normal_sum += normal
                depth_sum += depth
                batch_n += 1
                luminance = color.dot(ti.Vector([0.2126, 0.7152, 0.0722]))
                delta = luminance - batch_lum_mean
                batch_lum_mean += delta / float(batch_n)
                batch_lum_m2 += delta * (luminance - batch_lum_mean)
            old_n = self.sample_count[None]
            self.accumulator[px, py] += color_sum
            self.aov_albedo[px, py] += albedo_sum
            self.aov_normal[px, py] += normal_sum
            self.aov_depth[px, py] += depth_sum
            if actual > 0:
                combined_n = old_n + actual
                mean_delta = batch_lum_mean - self.lum_mean[px, py]
                self.lum_m2[px, py] += batch_lum_m2 + mean_delta * mean_delta * \
                    float(old_n) * float(actual) / float(combined_n)
                self.lum_mean[px, py] += mean_delta * float(actual) / float(combined_n)
        self.sample_count[None] = ti.min(
            self.sample_count[None] + samples_per_batch, max_spp)

    @ti.kernel
    def _render_batch_adaptive(self, camera: ti.template(), scene: ti.template(),
                     mat_sys: ti.template(), tex_sys: ti.template(),
                     light_sampler: ti.template(), max_bounce: ti.i32,
                     samples_per_batch: ti.i32,
                     max_spp: ti.i32):
        """在一次 kernel 中为每个活跃像素累积多个样本。"""
        for px, py in self.accumulator:
            if self.converged[px, py] == 0:
                for _ in range(samples_per_batch):
                    if self.pixel_sample_count[px, py] < max_spp:
                        s = (px + ti.random()) / self.width
                        t = (py + ti.random()) / self.height
                        ray_o, ray_d = camera.get_ray(s, t)
                        color, albedo, normal, depth = self._ray_color(
                            ray_o, ray_d, scene, mat_sys, tex_sys,
                            light_sampler, max_bounce)
                        self.accumulator[px, py] += color
                        self.aov_albedo[px, py] += albedo
                        self.aov_normal[px, py] += normal
                        self.aov_depth[px, py] += depth

                        n = self.pixel_sample_count[px, py] + 1
                        luminance = color.dot(ti.Vector([0.2126, 0.7152, 0.0722]))
                        delta = luminance - self.lum_mean[px, py]
                        self.lum_mean[px, py] += delta / float(n)
                        self.lum_m2[px, py] += delta * (luminance - self.lum_mean[px, py])
                        self.pixel_sample_count[px, py] = n
        self.sample_count[None] = ti.min(self.sample_count[None] + samples_per_batch, max_spp)

    def render_batch(self, camera, scene, mat_sys, tex_sys, light_sampler,
                     max_bounce, samples_per_batch, adaptive_enabled, max_spp,
                     save_aovs=False):
        """根据模式分派到编译期独立的 kernel，避免固定路径携带无用统计。"""
        if adaptive_enabled:
            if not self.enable_adaptive:
                raise RuntimeError("PathTracer 创建时未启用自适应缓冲")
            self._render_batch_adaptive(camera, scene, mat_sys, tex_sys,
                                        light_sampler, max_bounce,
                                        samples_per_batch, max_spp)
        elif save_aovs:
            if not self.enable_aovs:
                raise RuntimeError("PathTracer 创建时未启用 AOV 缓冲")
            self._render_batch_fixed_aov(camera, scene, mat_sys, tex_sys,
                                         light_sampler, max_bounce,
                                         samples_per_batch, max_spp)
        else:
            self._render_batch_fixed(camera, scene, mat_sys, tex_sys,
                                     light_sampler, max_bounce,
                                     samples_per_batch, max_spp)

    def render_sample(self, camera, scene, mat_sys, tex_sys, light_sampler, max_bounce):
        """向后兼容的单样本接口。"""
        self.render_batch(camera, scene, mat_sys, tex_sys, light_sampler,
                          max_bounce, 1, 0, 2_147_483_647)

    @ti.kernel
    def _update_convergence(self, min_spp: ti.i32, max_spp: ti.i32,
                            relative_error: ti.f32, absolute_error: ti.f32):
        self.active_count[None] = 0
        for x, y in self.pixel_sample_count:
            n = self.pixel_sample_count[x, y]
            done = n >= max_spp
            if n >= min_spp and n > 1:
                variance = self.lum_m2[x, y] / float(n - 1)
                standard_error = ti.sqrt(ti.max(variance, 0.0) / float(n))
                threshold = ti.max(absolute_error,
                                   relative_error * ti.max(ti.abs(self.lum_mean[x, y]), 1e-3))
                if standard_error <= threshold:
                    done = True
            self.converged[x, y] = 1 if done else 0
            if not done:
                ti.atomic_add(self.active_count[None], 1)

    def update_convergence(self, min_spp: int, max_spp: int,
                           relative_error: float, absolute_error: float):
        self._update_convergence(min_spp, max_spp, relative_error, absolute_error)

    def reset(self):
        """清零累积缓冲，用于相机或场景变更后重新渲染。"""
        self.accumulator.fill(0.0)
        self.pixel_sample_count.fill(0)
        self.converged.fill(0)
        self.lum_mean.fill(0.0)
        self.lum_m2.fill(0.0)
        self.aov_albedo.fill(0.0)
        self.aov_normal.fill(0.0)
        self.aov_depth.fill(0.0)
        self.sample_count[None] = 0
        self.active_count[None] = self.width * self.height

    def get_image(self) -> np.ndarray:
        """
        返回归一化后的 (H, W, 3) float32 图像（线性空间 HDR）。
        row 0 对应图像顶部（标准图像坐标）。
        调用方负责色调映射和 gamma 校正（见 io/image_output.py）。
        """
        if self.enable_adaptive:
            counts = np.maximum(self.pixel_sample_count.to_numpy(), 1)[..., None]
            img = self.accumulator.to_numpy() / counts
        else:
            img = self.accumulator.to_numpy() / max(self.spp, 1)
        img = img.transpose(1, 0, 2) # → (H, W, 3)，row 0 = py=0 = 视口底部
        return np.ascontiguousarray(np.flipud(img))  # 翻转，使 row 0 = 视口顶部

    def get_aovs(self) -> dict:
        if not self.enable_aovs and not self.enable_adaptive:
            raise RuntimeError("PathTracer 创建时未启用 AOV 缓冲")
        if self.enable_adaptive:
            counts_2d = np.maximum(self.pixel_sample_count.to_numpy(), 1)
            n = self.pixel_sample_count.to_numpy()
        else:
            counts_2d = np.full((self.width, self.height), max(self.spp, 1), np.int32)
            n = np.full((self.width, self.height), self.spp, np.int32)
        counts = counts_2d[..., None]
        albedo = self.aov_albedo.to_numpy() / counts
        normal = self.aov_normal.to_numpy() / counts
        normal /= np.maximum(np.linalg.norm(normal, axis=2, keepdims=True), 1e-8)
        depth = self.aov_depth.to_numpy() / counts_2d
        variance = np.zeros_like(depth, np.float32)
        valid = n > 1
        variance[valid] = self.lum_m2.to_numpy()[valid] / (n[valid] - 1)

        def orient(a):
            return np.ascontiguousarray(np.flipud(a.transpose(1, 0, 2)))
        def orient_scalar(a):
            return np.ascontiguousarray(np.flipud(a.T))
        return {
            'albedo': orient(albedo),
            'normal': orient(normal),
            'depth': orient_scalar(depth),
            'variance': orient_scalar(variance),
            'sample_count': orient_scalar(n.astype(np.float32)),
        }

    @ti.kernel
    def blit_to_display(self, dst: ti.template()):
        """
        将累积缓冲 ACES 色调映射 + gamma 2.2 后写入 Taichi display field (W×H)。
        供 GGUI 预览直接调用，无需 CPU 中转，且坐标系与 GGUI 一致：
        dst[x, 0] 显示在屏幕底部，与 accumulator[x, 0]（视口底部像素）对应。
        """
        for x, y in self.accumulator:
            n = float(ti.max(self.sample_count[None], 1))
            if ti.static(self.enable_adaptive):
                n = float(ti.max(self.pixel_sample_count[x, y], 1))
            c = self.accumulator[x, y] / n
            # 钳制超亮 firefly（float32 ACES 乘法在 c > ~3.7e18 时溢出）
            c = ti.Vector([ti.min(c[0], 1e6), ti.min(c[1], 1e6), ti.min(c[2], 1e6)])
            # ACES Hill 近似（逐通道）
            c = (c * (2.51 * c + 0.03)) / (c * (2.43 * c + 0.59) + 0.14)
            dst[x, y] = ti.Vector([
                ti.min(ti.max(c[0], 0.0), 1.0) ** 0.4545,
                ti.min(ti.max(c[1], 0.0), 1.0) ** 0.4545,
                ti.min(ti.max(c[2], 0.0), 1.0) ** 0.4545,
            ])

    @property
    def spp(self) -> int:
        return self.sample_count[None]

    @property
    def active_pixels(self) -> int:
        return self.active_count[None]

    def is_complete(self, max_spp: int, adaptive_enabled: bool) -> bool:
        return self.active_count[None] == 0 if adaptive_enabled else self.spp >= max_spp

    def progress(self, max_spp: int, adaptive_enabled: bool) -> float:
        if adaptive_enabled:
            return 1.0 - self.active_count[None] / float(self.width * self.height)
        return min(self.spp / float(max(max_spp, 1)), 1.0)

    # ------------------------------------------------------------------
    # Taichi 核心
    # ------------------------------------------------------------------

    @ti.func
    def _ray_color(self, ray_origin, ray_dir,
                   scene:         ti.template(),
                   mat_sys:       ti.template(),
                   tex_sys:       ti.template(),
                   light_sampler: ti.template(),
                   max_bounce:    ti.i32):
        """
        单条路径追踪 + NEE（Next Event Estimation）直接光采样。

        NEE 策略：
          - 每次落到非镜面表面时，额外向光源直接发射阴影射线评估直接照明。
          - 为避免双重计数，当前弹射非镜面时，后续路径命中 MAT_LIGHT 不再累积自发光；
            镜面弹射后命中光源则正常累积（NEE 无法沿镜面路径连接光源）。
          - PBR/Clearcoat 混合材质按本次随机选择的分支决定是否计为镜面。

        俄罗斯轮盘赌在第 RR_DEPTH 次反弹后启用，减少低能量路径计算量（无偏）。
        """
        RR_DEPTH = 5

        color         = ti.Vector([0.0, 0.0, 0.0])
        first_albedo  = ti.Vector([0.0, 0.0, 0.0])
        first_normal  = ti.Vector([0.0, 0.0, 0.0])
        first_depth   = 0.0
        attenuation   = ti.Vector([1.0, 1.0, 1.0])
        cur_o         = ray_origin
        cur_d         = ray_dir
        last_specular = True   # 摄像机射线视为“镜面”（初次命中光源需计入）
        last_bsdf_pdf = 0.0
        last_vertex = ray_origin

        for bounce in range(max_bounce):
            hit_t, hit_pos, hit_normal, hit_tan, hit_uv, hit_mat, front = scene.intersect(
                cur_o, cur_d, T_MIN, T_MAX
            )

            if hit_mat < 0:
                color += attenuation * scene.background(cur_d)
                break

            if bounce == 0:
                first_normal = hit_normal
                first_depth = hit_t
                first_mat = mat_sys.mats[hit_mat]
                first_albedo = first_mat.albedo
                if first_mat.albedo_tex_id >= 0:
                    first_albedo *= tex_sys.sample(first_mat.albedo_tex_id, hit_uv[0], hit_uv[1])
                first_albedo = ti.max(0.0, ti.min(1.0, first_albedo))

            scattered, scatter_att, emitted, should_scatter, is_specular, scatter_pdf = mat_sys.scatter(
                cur_d, hit_normal, hit_pos, hit_mat, hit_uv, front, hit_tan, tex_sys
            )

            # 自发光累积：
            #   MAT_LIGHT 只在 last_specular=True 时计入（非镜面情况由 NEE 代劳）；
            #   PBR 自发光（非 MAT_LIGHT）始终计入（NEE 不采样 PBR 自发光面）。
            mat_type = mat_sys.mats[hit_mat].mat_type
            if mat_type == MAT_LIGHT:
                mis_weight = 1.0
                if not last_specular and light_sampler.sampled_material[hit_mat] != 0:
                    light_pdf = light_sampler.pdf_for_hit(
                        last_vertex, hit_pos, hit_normal, hit_mat)
                    mis_weight = _power_heuristic(last_bsdf_pdf, light_pdf)
                color += attenuation * emitted * mis_weight
            else:
                color += attenuation * emitted

            if not should_scatter:
                break

            # NEE：非镜面表面上，向光源直接采样
            if not is_specular and light_sampler.n_lights[None] > 0:
                nee = _nee_contrib(cur_d, hit_pos, hit_normal, hit_mat, hit_uv,
                                   mat_sys, tex_sys, light_sampler, scene)
                color += attenuation * nee

            last_specular = is_specular
            last_bsdf_pdf = scatter_pdf
            last_vertex = hit_pos
            attenuation  *= scatter_att

            # 俄罗斯轮盘赌
            if bounce >= RR_DEPTH:
                rr_prob = ti.max(attenuation[0], attenuation[1], attenuation[2])
                rr_prob = ti.min(rr_prob, 0.95)
                rr_prob = ti.max(rr_prob, 0.05)
                if ti.random() >= rr_prob:
                    break
                attenuation /= rr_prob

            # 不让下一条路径从数学意义上的同一个表面点出发。单靠下一次求交的
            # t_min 无法可靠抵御大坐标/大半径几何体上的 float32 舍入误差。
            offset_sign = 1.0 if scattered.dot(hit_normal) >= 0.0 else -1.0
            cur_o = hit_pos + hit_normal * (T_MIN * offset_sign)
            cur_d = scattered

        return color, first_albedo, first_normal, first_depth


# ---------------------------------------------------------------------------
# NEE 直接光贡献（模块级 @ti.func，在 _ray_color 中调用）
# ---------------------------------------------------------------------------

@ti.func
def _nee_contrib(ray_dir, hit_pos, hit_normal, hit_mat: ti.i32, hit_uv,
                 mat_sys:       ti.template(),
                 tex_sys:       ti.template(),
                 light_sampler: ti.template(),
                 scene:         ti.template()):
    """
    在 hit_pos 处（漫反射或 PBR 漫反射分支）向面光源直接采样。

    返回当前表面接收到的直接照明辐亮度贡献（未乘路径衰减）。

    公式：L_direct = (albedo/π) * L_emit * cos_hit * cos_light / (pdf_A * dist²)
    其中：
      pdf_A    = 1 / total_light_area（面积均匀采样）
      cos_hit  = dot(hit_normal, light_dir)
      cos_light= dot(-light_dir, light_normal)
    """
    contrib = ti.Vector([0.0, 0.0, 0.0])

    u1 = ti.random()
    light_dir, dist, lemit, pdf_w, valid = light_sampler.sample(hit_pos, u1)

    if valid:
        if dist > 1e-6:
            cos_hit   = hit_normal.dot(light_dir)
            if cos_hit > 1e-4 and pdf_w > 1e-12:
                # 阴影射线：从 hit_pos 沿法线偏移 T_MIN，向光源方向追踪至 dist 前
                # 注意：不能用 _ 重复接收不同类型（vec3/vec2 冲突），需用不同名称
                blocked = scene.occluded(
                    hit_pos + hit_normal * T_MIN,
                    light_dir, T_MIN, dist * (1.0 - 1e-3)
                )

                if blocked == 0:   # 路径未被遮挡
                    # 评估与当前随机漫反射分支匹配的 BRDF。
                    # Clearcoat/PBR 只在选中漫反射分支时执行 NEE，
                    # 因此需除以该分支概率，否则直接光会系统性偏暗。
                    mat    = mat_sys.mats[hit_mat]
                    albedo = mat.albedo
                    if mat.albedo_tex_id >= 0:
                        albedo = albedo * tex_sys.sample(mat.albedo_tex_id, hit_uv[0], hit_uv[1])
                    albedo = ti.min(albedo, 1.0)

                    diffuse_brdf = albedo / _PI
                    bsdf_pdf = cos_hit / _PI
                    if mat.mat_type == MAT_CLEARCOAT:
                        branch_pdf = ti.max(1.0 - mat.spec_prob, 1e-4)
                        diffuse_brdf = diffuse_brdf / branch_pdf
                        bsdf_pdf *= branch_pdf
                    elif mat.mat_type == MAT_PBR:
                        metallic = mat.spec_prob
                        if mat.mr_tex_id >= 0:
                            mr = tex_sys.sample(mat.mr_tex_id, hit_uv[0], hit_uv[1])
                            metallic = metallic * mr[2]
                        metallic = ti.max(0.0, ti.min(1.0, metallic))
                        V = -ray_dir.normalized()
                        ndotv = ti.max(1e-4, V.dot(hit_normal))
                        f0 = ti.Vector([0.04, 0.04, 0.04]) * (1.0 - metallic) + albedo * metallic
                        fresnel = f0 + (ti.Vector([1.0, 1.0, 1.0]) - f0) * ((1.0 - ndotv) ** 5)
                        branch_pdf = ti.max(1.0 - (fresnel[0] + fresnel[1] + fresnel[2]) / 3.0, 0.02)
                        diffuse_brdf = albedo * (1.0 - metallic) * \
                                       (ti.Vector([1.0, 1.0, 1.0]) - fresnel) / (_PI * branch_pdf)
                        bsdf_pdf *= branch_pdf

                    mis_weight = _power_heuristic(pdf_w, bsdf_pdf)
                    contrib = diffuse_brdf * lemit * (cos_hit / pdf_w) * mis_weight

    return contrib


@ti.func
def _power_heuristic(pdf_a: ti.f32, pdf_b: ti.f32):
    a2 = pdf_a * pdf_a
    b2 = pdf_b * pdf_b
    weight = 0.0
    if a2 + b2 > 0.0:
        weight = a2 / (a2 + b2)
    return weight
