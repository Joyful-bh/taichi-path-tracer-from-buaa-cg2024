"""
相机模型 — 薄透镜（thin-lens）近似，支持景深。

所有相机参数在 Python 端计算完毕后存入 Taichi 标量场，
get_ray() 在 GPU kernel 内每像素每采样调用一次。
"""

import taichi as ti
import numpy as np

from src.math_utils import random_in_unit_disk


@ti.data_oriented
class Camera:
    def __init__(self,
                 lookfrom,
                 lookat,
                 up,
                 fov_deg: float,
                 aspect_ratio: float,
                 aperture: float = 0.0,
                 focus_dist: float = None):
        """
        Parameters
        ----------
        lookfrom     : 相机位置
        lookat       : 观察目标点
        up           : 世界上方向
        fov_deg      : 垂直视场角（度）
        aspect_ratio : 宽 / 高
        aperture     : 光圈直径；0 = 无景深
        focus_dist   : 对焦距离；None = 自动设为 lookfrom→lookat 距离
        """
        # Taichi 标量场（GPU 可访问）
        self._origin = ti.Vector.field(3, ti.f32, shape=())
        self._llc    = ti.Vector.field(3, ti.f32, shape=())   # lower-left-corner
        self._horiz  = ti.Vector.field(3, ti.f32, shape=())
        self._vert   = ti.Vector.field(3, ti.f32, shape=())
        self._u      = ti.Vector.field(3, ti.f32, shape=())   # 相机右向量
        self._v      = ti.Vector.field(3, ti.f32, shape=())   # 相机上向量
        self._lens_r = ti.field(ti.f32, shape=())             # 透镜半径

        self._setup(np.array(lookfrom, np.float32),
                    np.array(lookat,   np.float32),
                    np.array(up,       np.float32),
                    fov_deg, aspect_ratio, aperture, focus_dist)

    def _setup(self, lf, la, up, fov_deg, ar, aperture, focus_dist):
        w = lf - la
        w /= np.linalg.norm(w)
        u = np.cross(up, w);  u /= np.linalg.norm(u)
        v = np.cross(w, u)

        if focus_dist is None:
            focus_dist = float(np.linalg.norm(lf - la))

        half_h = np.tan(np.radians(fov_deg) * 0.5) * focus_dist
        half_w = ar * half_h

        self._origin[None] = lf.tolist()
        self._u[None]      = u.tolist()
        self._v[None]      = v.tolist()
        self._horiz[None]  = (2.0 * half_w * u).tolist()
        self._vert[None]   = (2.0 * half_h * v).tolist()
        self._llc[None]    = (lf - half_w * u - half_h * v - w * focus_dist).tolist()
        self._lens_r[None] = aperture * 0.5

    # ------------------------------------------------------------------
    # Taichi 作用域
    # ------------------------------------------------------------------

    @ti.func
    def get_ray(self, s: ti.f32, t: ti.f32):
        """
        生成归一化光线。s, t 为 [0,1] 像素坐标（含抖动）。
        返回 (ray_origin, ray_direction)，方向已归一化。
        """
        lens_r = self._lens_r[None]
        offset = ti.Vector([0.0, 0.0, 0.0])
        if lens_r > 0.0:
            rd     = lens_r * random_in_unit_disk()
            offset = self._u[None] * rd[0] + self._v[None] * rd[1]
        origin    = self._origin[None] + offset
        direction = (self._llc[None] + s * self._horiz[None] + t * self._vert[None]
                     - origin).normalized()
        return origin, direction
