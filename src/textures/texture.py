"""
贴图系统（TextureSystem）。

所有贴图在 Python 端通过 PIL.Image 加载，
统一 resize 到 TEX_SIZE × TEX_SIZE，归一化后写入 Taichi field。

Taichi 端通过 sample(@ti.func) 进行双线性插值采样，
使用 wrap（重复）模式处理 UV 超出 [0,1] 的情况。

设计参数：
  MAX_TEXTURES = 32   — 最多支持贴图数量
  TEX_SIZE     = 512  — 所有贴图统一缩放到 512×512（内存与性能的平衡）

使用方式：
  tex_sys = TextureSystem()
  tid = tex_sys.add(pil_image)   # Python 端注册
  tex_sys.bake()                 # 上传至 GPU（在 ti.init() 后、kernel 前调用）
  # Taichi 作用域中：
  color = tex_sys.sample(tid, u, v)
"""

import taichi as ti
import numpy as np

MAX_TEXTURES = 32
TEX_SIZE     = 512


@ti.data_oriented
class TextureSystem:
    """
    管理所有贴图的加载、存储和采样。

    Python 端：add(pil_image) → bake()
    Taichi 端：sample(tex_id, u, v) → ti.Vector([r, g, b])
    """

    def __init__(self, max_textures: int = MAX_TEXTURES, tex_size: int = TEX_SIZE):
        self._max  = max_textures
        self._sz   = tex_size
        self._gpu_sz = tex_size
        # GPU 字段在 bake() 时按实际纹理数分配。
        self.textures = None
        # Python 侧缓存
        self._buf   = []   # list of (tex_size, tex_size, 3) float32 numpy arrays
        self._count = 0

    # ------------------------------------------------------------------
    # Python 端：贴图注册
    # ------------------------------------------------------------------

    def add(self, image) -> int:
        """
        注册一张 PIL.Image，返回 tex_id（从 0 递增）。
        image 会被 resize 到 (tex_size, tex_size)，转换为 RGB float32 [0,1]。
        """
        assert self._count < self._max, \
            f"TextureSystem 超出容量 {self._max}（已有 {self._count} 张贴图）"

        # 确保 RGB 模式
        img = image.convert('RGB')
        # 统一缩放到 tex_size × tex_size
        img = img.resize((self._sz, self._sz))
        # 转为 float32 numpy 数组，归一化到 [0, 1]
        arr = np.array(img, dtype=np.float32) / 255.0   # (sz, sz, 3)
        # Blender 导出 GLB 时 UV 的 V=0 在图像底部（OpenGL 惯例），
        # 而 PIL row 0 在顶部。垂直翻转使 row 0 = V=0 的底部，与 GLB UV 对齐。
        arr = np.flipud(arr)
        self._buf.append(arr)

        idx = self._count
        self._count += 1
        return idx

    def bake(self):
        """
        将所有 Python 端缓存的贴图上传至 GPU。
        必须在 ti.init() 之后、渲染 kernel 调用之前执行。
        """
        capacity = max(self._count, 1)
        self._gpu_sz = self._sz if self._count else 1
        self.textures = ti.Vector.field(
            3, ti.f32, shape=(capacity, self._gpu_sz, self._gpu_sz))
        # 只上传实际纹理容量；空场景保留一个有效占位层。
        full = np.zeros((capacity, self._gpu_sz, self._gpu_sz, 3), dtype=np.float32)
        for i, arr in enumerate(self._buf):
            full[i] = arr   # arr shape: (sz, sz, 3)

        self.textures.from_numpy(full)
        print(f"[Texture] 已烘焙 {self._count} 张贴图（{self._sz}×{self._sz}）")

    @property
    def count(self) -> int:
        return self._count

    # ------------------------------------------------------------------
    # Taichi 作用域：双线性插值采样
    # ------------------------------------------------------------------

    @ti.func
    def sample(self, tex_id: ti.i32, u: ti.f32, v: ti.f32) -> ti.types.vector(3, ti.f32):
        """
        双线性插值采样贴图。
        tex_id < 0 时返回白色 (1, 1, 1)（避免黑色缺失贴图）。
        UV 使用 wrap（平铺）模式。
        """
        result = ti.Vector([1.0, 1.0, 1.0])
        if tex_id >= 0:
            # wrap UV：取小数部分实现平铺
            uw = u - ti.floor(u)
            vw = v - ti.floor(v)

            # 映射到像素坐标（连续坐标，浮点）
            sz_f  = float(self._gpu_sz)
            px    = uw * (sz_f - 1.0)
            py    = vw * (sz_f - 1.0)

            x0 = int(ti.floor(px))
            y0 = int(ti.floor(py))
            x1 = ti.min(x0 + 1, self._gpu_sz - 1)
            y1 = ti.min(y0 + 1, self._gpu_sz - 1)

            fx = px - ti.floor(px)
            fy = py - ti.floor(py)

            # 4 个采样点的颜色
            # 注意：field 第一维对应 numpy arr 的 row(y)，第二维对应 col(x)
            # 因此访问顺序为 [y, x]，即 [v_pixel, u_pixel]
            c00 = self.textures[tex_id, y0, x0]
            c10 = self.textures[tex_id, y0, x1]
            c01 = self.textures[tex_id, y1, x0]
            c11 = self.textures[tex_id, y1, x1]

            # 双线性插值
            result = (c00 * (1.0 - fx) * (1.0 - fy) +
                      c10 * fx         * (1.0 - fy) +
                      c01 * (1.0 - fx) * fy         +
                      c11 * fx         * fy)
        return result
