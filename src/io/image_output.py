"""
图像输出工具。

提供：
  - ACES Filmic 色调映射（避免高亮过曝的简单线性截断）
  - Gamma 校正（线性空间 → sRGB）
  - PNG 保存
"""

import numpy as np
from pathlib import Path


def aces_tonemap(img: np.ndarray) -> np.ndarray:
    """
    ACES Filmic 色调映射近似（Hill / Narkowicz 2015）。
    输入：线性 HDR，任意范围。
    输出：[0, 1] float32，sRGB 感知曲线（仍需 gamma 校正）。
    """
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    # 升到 float64 计算：float32 在 x > ~3.7e18 时乘法直接溢出
    x = np.nan_to_num(img.astype(np.float64), nan=0.0, posinf=1e6, neginf=0.0).clip(0.0)
    result = ((x * (a * x + b)) / (x * (c * x + d) + e)).clip(0.0, 1.0)
    return result.astype(np.float32)


def gamma_correct(img: np.ndarray, gamma: float = 2.2) -> np.ndarray:
    """线性空间 → 显示 gamma（sRGB 近似为 gamma 2.2）。"""
    return np.power(img.clip(0.0, 1.0), 1.0 / gamma)


def save_image(img: np.ndarray, path: str,
               tone_map: bool = True,
               gamma: float = 2.2) -> None:
    """
    将 float32 线性 HDR 图像保存为 PNG。

    Parameters
    ----------
    img      : (H, W, 3) float32，线性空间 RGB。
    path     : 输出路径（自动创建父目录）。
    tone_map : 是否先做 ACES 色调映射。
    gamma    : gamma 校正指数（2.2 对应标准 sRGB）。
    """
    import imageio

    out = img.copy().astype(np.float32)
    if tone_map:
        out = aces_tonemap(out)
    out = gamma_correct(out, gamma)
    out = (out * 255.0).clip(0, 255).astype(np.uint8)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(path, out)
    print(f"[Output] 已保存：{path}  ({img.shape[1]}×{img.shape[0]})")


def linear_to_display(img: np.ndarray,
                       tone_map: bool = True,
                       gamma: float = 2.2) -> np.ndarray:
    """
    转换为可显示的 [0,1] float32 图像（用于 GGUI canvas.set_image）。
    """
    out = img.copy().astype(np.float32)
    if tone_map:
        out = aces_tonemap(out)
    return gamma_correct(out, gamma)


def save_aovs(aovs: dict, beauty_path: str) -> None:
    """将降噪和诊断所需 AOV 保存为与主图同名的 PNG 侧车文件。"""
    import imageio

    base = Path(beauty_path)
    base.parent.mkdir(parents=True, exist_ok=True)

    albedo = gamma_correct(np.nan_to_num(aovs['albedo']).clip(0.0, 1.0))
    normal = (np.nan_to_num(aovs['normal']) * 0.5 + 0.5).clip(0.0, 1.0)

    depth = np.nan_to_num(aovs['depth'], nan=0.0, posinf=0.0, neginf=0.0)
    valid_depth = depth > 0.0
    depth_display = np.zeros_like(depth)
    if valid_depth.any():
        near, far = np.percentile(depth[valid_depth], [1.0, 99.0])
        if far > near:
            depth_display[valid_depth] = 1.0 - np.clip(
                (depth[valid_depth] - near) / (far - near), 0.0, 1.0)

    variance = np.nan_to_num(aovs['variance'], nan=0.0, posinf=0.0, neginf=0.0)
    variance_display = np.log1p(variance)
    vmax = float(np.percentile(variance_display, 99.0))
    if vmax > 0.0:
        variance_display = np.clip(variance_display / vmax, 0.0, 1.0)

    counts = np.nan_to_num(aovs['sample_count'], nan=0.0)
    count_display = counts / max(float(counts.max()), 1.0)

    outputs = {
        'albedo': albedo,
        'normal': normal,
        'depth': depth_display,
        'variance': variance_display,
        'sample_count': count_display,
    }
    for name, data in outputs.items():
        path = base.with_name(f"{base.stem}_{name}.png")
        out = (data * 255.0).clip(0, 255).astype(np.uint8)
        imageio.imwrite(path, out)
        print(f"[AOV] 已保存：{path}")
