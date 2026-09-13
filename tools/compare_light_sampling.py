"""Compare unbiased BSDF-only sampling with NEE+MIS at equal SPP."""
import argparse
from pathlib import Path
import sys
import time

import numpy as np
import taichi as ti

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.io.loader import SceneLoader
from src.renderer.path_tracer import PathTracer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("scene")
    p.add_argument("--width", type=int, default=320)
    p.add_argument("--height", type=int, default=180)
    p.add_argument("--spp", type=int, default=64)
    p.add_argument("--disable-direct", action="store_true")
    args = p.parse_args()
    loader = SceneLoader()
    data, cfg = loader.load_config(args.scene)
    cfg.width, cfg.height, cfg.backend = args.width, args.height, "cuda"
    ti.init(arch=ti.cuda, default_fp=ti.f32, random_seed=42)
    scene, camera = loader.build(data, cfg)
    scene.bake()
    if args.disable_direct:
        scene.light_sampler.n_lights[None] = 0
        scene.light_sampler.sampled_material.fill(0)
        scene.light_sampler.material_light_type.fill(0)
    renderer = PathTracer(cfg.width, cfg.height, enable_aovs=True)
    renderer.render_batch(camera, scene, scene.materials, scene.tex_sys,
                          scene.light_sampler, cfg.max_bounce, 1, False,
                          args.spp, True)
    ti.sync()  # compile warm-up
    renderer.reset()
    started = time.perf_counter()
    done = 0
    while done < args.spp:
        batch = min(16, args.spp - done)
        renderer.render_batch(camera, scene, scene.materials, scene.tex_sys,
                              scene.light_sampler, cfg.max_bounce, batch,
                              False, args.spp, True)
        done += batch
    ti.sync()
    elapsed = time.perf_counter() - started
    image = renderer.get_image()
    variance = renderer.get_aovs()['variance']
    lum = image @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    mode = "bsdf_only" if args.disable_direct else "nee_mis"
    print(f"LIGHT_BENCH mode={mode} seconds={elapsed:.6f} "
          f"mean_luminance={float(lum.mean()):.9f} "
          f"sample_variance_mean={float(variance.mean()):.9f} "
          f"sample_variance_p50={float(np.percentile(variance, 50)):.9f} "
          f"sample_variance_p90={float(np.percentile(variance, 90)):.9f} "
          f"sample_variance_p99={float(np.percentile(variance, 99)):.9f}")


if __name__ == "__main__":
    main()
