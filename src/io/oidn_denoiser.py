"""Intel Open Image Denoise command-line integration for linear HDR images."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np


class OIDNError(RuntimeError):
    """Raised when OIDN is unavailable or rejects an image."""


def _oidn_candidates(executable: str = 'auto') -> list[str | Path]:
    """Return explicit and conventional OIDN executable locations in priority order."""
    candidates: list[str | Path] = []
    if executable and executable.lower() != 'auto':
        candidates.append(executable)

    env_path = os.environ.get('OIDN_DENOISE_EXECUTABLE')
    if env_path:
        candidates.append(env_path)

    # ``shutil.which`` handles PATH entries and executable-bit checks for us.
    for name in ('oidnDenoise', 'oidnDenoise.exe'):
        found = shutil.which(name)
        if found:
            candidates.append(found)

    # Source builds, especially on macOS, commonly leave the tool in
    # <source-or-install-root>/build/oidnDenoise rather than installing it.
    roots: list[Path] = []
    for variable in ('OIDN_ROOT', 'OIDN_DIR', 'OPEN_IMAGE_DENOISE_ROOT'):
        value = os.environ.get(variable)
        if value:
            roots.append(Path(value).expanduser())
    home = Path.home()
    roots.extend(home / name for name in ('oidn', 'OIDN', 'openimagedenoise', 'OpenImageDenoise'))
    roots.extend((Path.cwd(), Path(__file__).resolve().parents[2]))
    for root in roots:
        for name in ('oidnDenoise', 'oidnDenoise.exe'):
            candidates.extend((
                root / 'build' / name,
                root / 'build' / 'apps' / name,
                root / 'bin' / name,
            ))
    return candidates


def find_oidn(executable: str = 'auto') -> str:
    """Resolve oidnDenoise without silently falling back to another algorithm."""
    candidates = _oidn_candidates(executable)
    seen: set[str] = set()
    for candidate in candidates:
        path = Path(candidate).expanduser()
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        if path.is_file() and os.access(path, os.X_OK):
            return str(path.resolve())
    raise OIDNError(
        "未找到可执行的 oidnDenoise。请将 OIDN 的 bin 目录加入 PATH，或设置"
        " OIDN_DENOISE_EXECUTABLE；源码构建也可放在 ~/oidn/build/oidnDenoise，"
        "或在 denoising.executable 中指定绝对路径。"
    )


def _sanitize(image: np.ndarray, *, auxiliary: bool = False) -> np.ndarray:
    array = np.asarray(image, dtype=np.float32)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"OIDN 图像必须是 (H, W, 3)，实际为 {array.shape}")
    array = np.nan_to_num(array, nan=0.0, posinf=1e6, neginf=0.0)
    if auxiliary:
        return np.ascontiguousarray(array)
    return np.ascontiguousarray(np.maximum(array, 0.0))


def write_pfm(path: str | Path, image: np.ndarray) -> None:
    """Write little-endian RGB PFM while preserving linear float32 values."""
    data = _sanitize(image, auxiliary=True)
    height, width, _ = data.shape
    with open(path, 'wb') as stream:
        stream.write(f"PF\n{width} {height}\n-1.0\n".encode('ascii'))
        np.flipud(data).astype('<f4', copy=False).tofile(stream)


def read_pfm(path: str | Path) -> np.ndarray:
    with open(path, 'rb') as stream:
        if stream.readline().strip() != b'PF':
            raise OIDNError("OIDN 输出不是 RGB PFM 文件")
        dimensions = stream.readline().decode('ascii').strip().split()
        width, height = map(int, dimensions)
        scale = float(stream.readline().decode('ascii').strip())
        endian = '<' if scale < 0 else '>'
        data = np.fromfile(stream, dtype=endian + 'f4', count=width * height * 3)
    if data.size != width * height * 3:
        raise OIDNError("OIDN 输出 PFM 数据不完整")
    return np.ascontiguousarray(np.flipud(data.reshape(height, width, 3)).astype(np.float32))


@contextmanager
def _temporary_pfm_paths(count: int):
    """Yield writable files directly in the temp root (some sandboxes reject temp subdirs)."""
    temp_root = os.environ.get('OIDN_TEMP_DIR') or None
    if temp_root:
        Path(temp_root).mkdir(parents=True, exist_ok=True)
    paths = []
    try:
        for _ in range(count):
            handle = tempfile.NamedTemporaryFile(prefix='taichi-oidn-', suffix='.pfm',
                                                 dir=temp_root, delete=False)
            handle.close()
            paths.append(Path(handle.name))
        yield paths
    finally:
        for path in paths:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass


def denoise_oidn(color: np.ndarray, *, albedo: np.ndarray | None = None,
                 normal: np.ndarray | None = None, executable: str = 'auto',
                 device: str = 'default', quality: str = 'high') -> np.ndarray:
    """Denoise a linear HDR beauty image using optional renderer AOV guides."""
    color = _sanitize(color)
    if albedo is not None and np.asarray(albedo).shape != color.shape:
        raise ValueError("albedo AOV 尺寸必须与 beauty 相同")
    if normal is not None and np.asarray(normal).shape != color.shape:
        raise ValueError("normal AOV 尺寸必须与 beauty 相同")

    exe = find_oidn(executable)
    quality_map = {'h': 'high', 'b': 'balanced', 'f': 'fast'}
    quality = quality_map.get(quality.lower(), quality.lower())
    if quality not in {'high', 'balanced', 'fast', 'default'}:
        raise ValueError("denoising.quality 必须是 high、balanced、fast 或 default")
    if device not in {'default', 'cpu', 'sycl', 'cuda', 'hip', 'metal'} and not device.isdigit():
        raise ValueError("denoising.device 必须是 default/cpu/sycl/cuda/hip/metal 或设备编号")

    path_count = 2 + int(albedo is not None) + int(normal is not None)
    with _temporary_pfm_paths(path_count) as paths:
        color_path, output_path = paths[0], paths[1]
        output_path.unlink(missing_ok=True)
        write_pfm(color_path, color)
        command = [exe, '--device', device, '--filter', 'RT', '--hdr', str(color_path),
                   '--quality', quality, '--output', str(output_path)]
        if albedo is not None:
            albedo_path = paths[2]
            write_pfm(albedo_path, np.clip(_sanitize(albedo, auxiliary=True), 0.0, 1.0))
            command.extend(['--alb', str(albedo_path)])
        if normal is not None:
            normal_path = paths[2 + int(albedo is not None)]
            write_pfm(normal_path, np.clip(_sanitize(normal, auxiliary=True), -1.0, 1.0))
            command.extend(['--nrm', str(normal_path)])

        result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8',
                                errors='replace', check=False)
        if result.returncode != 0 or not output_path.is_file():
            detail = (result.stderr or result.stdout).strip()
            raise OIDNError(f"OIDN 降噪失败（退出码 {result.returncode}）：{detail}")
        return np.maximum(read_pfm(output_path), 0.0)


def denoised_output_path(beauty_path: str, configured_path: str | None = None) -> str:
    if configured_path:
        return configured_path
    path = Path(beauty_path)
    return str(path.with_name(f"{path.stem}_denoised{path.suffix or '.png'}"))
