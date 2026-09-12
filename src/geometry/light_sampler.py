"""
LightSampler — NEE 直接光采样器。

在 Python 端扫描场景中所有 MAT_LIGHT 三角形，构建面积加权 CDF，
供 Taichi kernel 按面积均匀采样光源表面上的点。

sample() 返回 (pos, nrm, emit, pdf_area, valid)：
  pos      — 光源面上的采样点（世界空间）
  nrm      — 光源面法线（指向场景内侧，即面法线方向）
  emit     — 该光源三角形的自发光辐亮度
  pdf_area — 按面积的采样概率密度（= 1 / total_light_area）
  valid    — 1 表示有效采样，0 表示场景中没有光源
"""

import taichi as ti
import numpy as np

from src.constants import MAT_LIGHT


@ti.data_oriented
class LightSampler:

    def __init__(self, max_lights: int = 2048):
        self._max = max_lights
        # 标量信息
        self.n_lights   = ti.field(ti.i32, shape=())
        self.total_area = ti.field(ti.f32, shape=())
        # 光源三角形数据（SoA）
        self.lv0   = ti.Vector.field(3, ti.f32, shape=(max_lights,))
        self.lv1   = ti.Vector.field(3, ti.f32, shape=(max_lights,))
        self.lv2   = ti.Vector.field(3, ti.f32, shape=(max_lights,))
        self.lnrm  = ti.Vector.field(3, ti.f32, shape=(max_lights,))
        self.lemit = ti.Vector.field(3, ti.f32, shape=(max_lights,))
        # 面积归一化 CDF（最后一项 = 1.0）
        self.cdf   = ti.field(ti.f32, shape=(max_lights,))
        self.sampled_material = ti.field(ti.i32, shape=(512,))
        self.sampled_material.fill(0)
        self.n_lights[None]   = 0
        self.total_area[None] = 0.0

    # ------------------------------------------------------------------
    # Python 端：构建光源列表
    # ------------------------------------------------------------------

    def build(self, tri_sys, mat_sys):
        """
        扫描 tri_sys 中所有 MAT_LIGHT 三角形，计算面积权重 CDF 并上传到 GPU。
        必须在 tri_sys.bake() 和 mat_sys.bake() 之后、渲染前调用。
        """
        n_tri = tri_sys.count
        if n_tri == 0:
            return

        v0s     = np.stack(tri_sys._v0)              # (N, 3)
        v1s     = np.stack(tri_sys._v1)
        v2s     = np.stack(tri_sys._v2)
        mat_ids = np.array(tri_sys._mat, np.int32)

        mat_types = np.array(mat_sys._buf['mat_type'], np.int32)
        emits     = np.stack(mat_sys._buf['emit']).astype(np.float32)  # (M, 3)

        lights_v0   = []
        lights_v1   = []
        lights_v2   = []
        lights_nrm  = []
        lights_emit = []
        areas       = []

        for i in range(n_tri):
            mid = mat_ids[i]
            if mat_types[mid] != MAT_LIGHT:
                continue
            v0 = v0s[i]; v1 = v1s[i]; v2 = v2s[i]
            cross = np.cross(v1 - v0, v2 - v0).astype(np.float32)
            area  = 0.5 * float(np.linalg.norm(cross))
            if area < 1e-12:
                continue
            nrm = cross / (2.0 * area)
            lights_v0.append(v0); lights_v1.append(v1); lights_v2.append(v2)
            lights_nrm.append(nrm)
            lights_emit.append(emits[mid])
            areas.append(area)
            self.sampled_material[mid] = 1

        n = len(lights_v0)
        if n == 0:
            print("[LightSampler] 场景中没有 MAT_LIGHT 三角形，NEE 不生效")
            return

        print(f"[LightSampler] 找到 {n} 个光源三角形，总面积 = {sum(areas):.4f}")

        areas_arr = np.array(areas, np.float32)
        total     = float(areas_arr.sum())
        cdf_arr   = (np.cumsum(areas_arr) / total).astype(np.float32)

        cap = min(n, self._max)

        # 零填充数组（保持未使用 slot 为零）
        def _zero3(): return np.zeros((self._max, 3), np.float32)
        def _zero1(): return np.zeros(self._max, np.float32)

        v0_out = _zero3(); v0_out[:cap] = np.stack(lights_v0[:cap])
        v1_out = _zero3(); v1_out[:cap] = np.stack(lights_v1[:cap])
        v2_out = _zero3(); v2_out[:cap] = np.stack(lights_v2[:cap])
        nm_out = _zero3(); nm_out[:cap] = np.stack(lights_nrm[:cap])
        em_out = _zero3(); em_out[:cap] = np.stack(lights_emit[:cap])
        cd_out = _zero1(); cd_out[:cap] = cdf_arr[:cap]

        self.lv0.from_numpy(v0_out)
        self.lv1.from_numpy(v1_out)
        self.lv2.from_numpy(v2_out)
        self.lnrm.from_numpy(nm_out)
        self.lemit.from_numpy(em_out)
        self.cdf.from_numpy(cd_out)

        self.n_lights[None]   = cap
        self.total_area[None] = total

    # ------------------------------------------------------------------
    # Taichi 作用域：按面积权重采样一个光源点
    # ------------------------------------------------------------------

    @ti.func
    def sample(self, u1: ti.f32):
        """
        按面积权重采样光源。
        u1 : [0,1) 均匀随机数，用于 CDF 选择三角形。
        再用两个 ti.random() 采样三角形内的点。

        返回 (pos, nrm, emit, pdf_area, valid)。
        pdf_area = 1 / total_area（面积均匀分布）。
        """
        n     = self.n_lights[None]
        pos   = ti.Vector([0.0, 0.0, 0.0])
        nrm   = ti.Vector([0.0, 1.0, 0.0])
        emit  = ti.Vector([0.0, 0.0, 0.0])
        pdf_A = 1.0
        valid = 0

        if n > 0:
            # 线性扫描 CDF（n 通常很小：≤ 10）
            light_id = n - 1
            for k in range(n):
                if self.cdf[k] >= u1:
                    light_id = k
                    break

            # 三角形内均匀采样（Shirley 1997 折叠映射）
            r1  = ti.random()
            r2  = ti.random()
            sr1 = ti.sqrt(r1)
            b0  = 1.0 - sr1          # 顶点 v0 权重
            b1  = sr1 * (1.0 - r2)  # 顶点 v1 权重
            b2  = sr1 * r2           # 顶点 v2 权重

            pos  = b0 * self.lv0[light_id] + b1 * self.lv1[light_id] + b2 * self.lv2[light_id]
            nrm  = self.lnrm[light_id]
            emit = self.lemit[light_id]

            # 面积均匀采样的 PDF：1 / 总光源面积
            pdf_A = 1.0 / self.total_area[None]
            valid = 1

        return pos, nrm, emit, pdf_A, valid
