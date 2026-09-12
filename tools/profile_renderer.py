"""Small repeatable performance probe for the renderer (does not validate images)."""

import argparse
from pathlib import Path
import sys
import time

import taichi as ti

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.io.loader import SceneLoader
from src.renderer.path_tracer import PathTracer


@ti.kernel
def render_color_only(renderer: ti.template(), camera: ti.template(),
                      scene: ti.template(), mat_sys: ti.template(),
                      tex_sys: ti.template(), light_sampler: ti.template(),
                      spp: ti.i32, max_bounce: ti.i32):
    """Diagnostic variant: same paths, but no AOV/variance/count traffic."""
    for px, py in renderer.accumulator:
        total = ti.Vector([0.0, 0.0, 0.0])
        for _ in range(spp):
            s = (px + ti.random()) / renderer.width
            t = (py + ti.random()) / renderer.height
            ray_o, ray_d = camera.get_ray(s, t)
            color, sample_albedo, sample_normal, sample_depth = renderer._ray_color(
                ray_o, ray_d, scene, mat_sys, tex_sys, light_sampler,
                max_bounce)
            total += color
        renderer.accumulator[px, py] += total


def timed_batch(renderer, camera, scene, *, spp, batch, bounce):
    renderer.reset()
    ti.sync()
    start = time.perf_counter()
    done = 0
    while done < spp:
        current = min(batch, spp - done)
        renderer.render_batch(
            camera, scene, scene.materials, scene.tex_sys, scene.light_sampler,
            bounce, current, 0, spp,
        )
        done += current
    ti.sync()
    elapsed = time.perf_counter() - start
    paths = renderer.width * renderer.height * spp
    return elapsed, paths / elapsed / 1e6


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scene")
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=208)
    parser.add_argument("--spp", type=int, default=32)
    args = parser.parse_args()

    loader = SceneLoader()
    data, cfg = loader.load_config(args.scene)
    cfg.width, cfg.height, cfg.backend = args.width, args.height, "cuda"
    ti.init(arch=ti.cuda, default_fp=ti.f32, random_seed=42,
            kernel_profiler=True)
    scene, camera = loader.build(data, cfg)
    build_start = time.perf_counter()
    scene.bake()
    build_seconds = time.perf_counter() - build_start
    renderer = PathTracer(cfg.width, cfg.height)

    # Compile before measuring.
    renderer.render_batch(camera, scene, scene.materials, scene.tex_sys,
                          scene.light_sampler, 2, 1, 0, 1)
    ti.sync()

    print(f"PROFILE scene={args.scene} size={args.width}x{args.height} "
          f"triangles={scene.triangles.count} spheres={scene.spheres.count} "
          f"bvh_nodes={scene.bvh._node_count} build_s={build_seconds:.4f}")
    for bounce in (1, 2, 4, 8, 16, 30):
        elapsed, mpaths = timed_batch(
            renderer, camera, scene, spp=args.spp, batch=16, bounce=bounce)
        print(f"bounce={bounce:2d} batch=16 nee=on  seconds={elapsed:.4f} "
              f"Mpaths/s={mpaths:.3f}")

    original_lights = scene.light_sampler.n_lights[None]
    scene.light_sampler.n_lights[None] = 0
    elapsed, mpaths = timed_batch(
        renderer, camera, scene, spp=args.spp, batch=16, bounce=30)
    print(f"bounce=30 batch=16 nee=off seconds={elapsed:.4f} "
          f"Mpaths/s={mpaths:.3f}")
    scene.light_sampler.n_lights[None] = original_lights

    for batch in (1, 4, 16, 32):
        elapsed, mpaths = timed_batch(
            renderer, camera, scene, spp=args.spp, batch=batch, bounce=30)
        print(f"bounce=30 batch={batch:2d} nee=on  seconds={elapsed:.4f} "
              f"Mpaths/s={mpaths:.3f}")

    renderer.reset()
    render_color_only(renderer, camera, scene, scene.materials, scene.tex_sys,
                      scene.light_sampler, 1, 30)
    ti.sync()
    renderer.reset()
    ti.sync()
    start = time.perf_counter()
    render_color_only(renderer, camera, scene, scene.materials, scene.tex_sys,
                      scene.light_sampler, args.spp, 30)
    ti.sync()
    elapsed = time.perf_counter() - start
    paths = renderer.width * renderer.height * args.spp
    print(f"bounce=30 color-only      seconds={elapsed:.4f} "
          f"Mpaths/s={paths / elapsed / 1e6:.3f}")

    ti.profiler.print_kernel_profiler_info("count")


if __name__ == "__main__":
    main()
