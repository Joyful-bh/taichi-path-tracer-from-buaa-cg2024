"""Compare a low-SPP image and its OIDN result against a higher-SPP reference."""

import argparse
import time
from pathlib import Path
import sys

import numpy as np
import taichi as ti

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.io.image_output import aces_tonemap, gamma_correct, save_image
from src.io.loader import SceneLoader
from src.io.oidn_denoiser import denoise_oidn
from src.renderer.path_tracer import PathTracer


def display_linear(image):
    return gamma_correct(aces_tonemap(image)).astype(np.float64)


def metrics(candidate, reference):
    a, b = display_linear(candidate), display_linear(reference)
    error = a - b
    mse = float(np.mean(error * error))
    psnr = float('inf') if mse == 0 else -10.0 * np.log10(mse)
    luma_a = a @ np.array([0.2126, 0.7152, 0.0722])
    luma_b = b @ np.array([0.2126, 0.7152, 0.0722])
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ma, mb = float(luma_a.mean()), float(luma_b.mean())
    va, vb = float(luma_a.var()), float(luma_b.var())
    covariance = float(np.mean((luma_a - ma) * (luma_b - mb)))
    ssim = ((2 * ma * mb + c1) * (2 * covariance + c2) /
            ((ma * ma + mb * mb + c1) * (va + vb + c2)))
    mae = float(np.mean(np.abs(error)))
    return psnr, ssim, mae


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', default='scene_files/cornell_box.yaml')
    parser.add_argument('--low-spp', type=int, default=32)
    parser.add_argument('--reference-spp', type=int, default=256)
    parser.add_argument('--width', type=int, default=320)
    parser.add_argument('--height', type=int, default=180)
    parser.add_argument('--device', default='default')
    parser.add_argument('--oidn-executable', default='auto')
    parser.add_argument('--output-prefix', default=None)
    args = parser.parse_args()
    if not 0 < args.low_spp < args.reference_spp:
        parser.error('需要满足 0 < low-spp < reference-spp')

    loader = SceneLoader()
    data, cfg = loader.load_config(args.scene)
    cfg.width, cfg.height = args.width, args.height
    ti.init(arch=ti.cuda, default_fp=ti.f32, random_seed=42)
    scene, camera = loader.build(data, cfg)
    scene.bake()
    renderer = PathTracer(cfg.width, cfg.height, enable_aovs=True)

    start = time.perf_counter()
    while renderer.spp < args.low_spp:
        batch = min(cfg.samples_per_batch, args.low_spp - renderer.spp)
        renderer.render_batch(camera, scene, scene.materials, scene.tex_sys,
                              scene.light_sampler, cfg.max_bounce, batch, False,
                              args.reference_spp, True)
    low_time = time.perf_counter() - start
    low = renderer.get_image()
    low_aovs = renderer.get_aovs()

    denoise_start = time.perf_counter()
    denoised = denoise_oidn(low, albedo=low_aovs['albedo'], normal=low_aovs['normal'],
                            executable=args.oidn_executable, device=args.device,
                            quality='high')
    denoise_time = time.perf_counter() - denoise_start

    while renderer.spp < args.reference_spp:
        batch = min(cfg.samples_per_batch, args.reference_spp - renderer.spp)
        renderer.render_batch(camera, scene, scene.materials, scene.tex_sys,
                              scene.light_sampler, cfg.max_bounce, batch, False,
                              args.reference_spp, True)
    reference = renderer.get_image()
    reference_time = time.perf_counter() - start

    raw_metrics = metrics(low, reference)
    oidn_metrics = metrics(denoised, reference)
    print(f"LOW spp={args.low_spp} render={low_time:.3f}s "
          f"PSNR={raw_metrics[0]:.3f}dB SSIM={raw_metrics[1]:.6f} MAE={raw_metrics[2]:.6f}")
    print(f"OIDN quality=high denoise={denoise_time:.3f}s "
          f"PSNR={oidn_metrics[0]:.3f}dB SSIM={oidn_metrics[1]:.6f} MAE={oidn_metrics[2]:.6f}")
    print(f"REF spp={args.reference_spp} total_render={reference_time:.3f}s")

    if args.output_prefix:
        prefix = Path(args.output_prefix)
        save_image(low, str(prefix.with_name(prefix.name + '_low.png')))
        save_image(denoised, str(prefix.with_name(prefix.name + '_denoised.png')))
        save_image(reference, str(prefix.with_name(prefix.name + '_reference.png')))


if __name__ == '__main__':
    main()
