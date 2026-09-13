"""Small repeatable performance probe for the renderer (does not validate images)."""

import argparse
import statistics
from pathlib import Path
import sys
import time

import numpy as np
import taichi as ti

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.io.loader import SceneLoader
from experiments.bvh4 import BVH4System
from src.renderer.path_tracer import PathTracer
from experiments.wavefront_path_tracer import WavefrontPathTracer


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
    parser.add_argument("--check-modes", action="store_true")
    parser.add_argument("--focused", action="store_true",
                        help="只重复测量完整固定采样 kernel，适合跨进程比较 BVH 栈")
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument("--bvh-stack", type=int, default=None,
                        help="首次 kernel 编译前覆盖每线程 BVH 栈容量")
    parser.add_argument("--bvh", choices=("bvh2", "bvh4"), default="bvh2")
    parser.add_argument("--signature", action="store_true",
                        help="输出末轮线性图像统计，用于检查实验 BVH 结果")
    parser.add_argument("--integrator", choices=("megakernel", "wavefront"),
                        default="megakernel")
    parser.add_argument("--wavefront-lanes", type=int, default=4)
    parser.add_argument("--wavefront-bounces", type=int, default=1)
    args = parser.parse_args()

    loader = SceneLoader()
    data, cfg = loader.load_config(args.scene)
    cfg.width, cfg.height, cfg.backend = args.width, args.height, "cuda"
    ti.init(arch=ti.cuda, default_fp=ti.f32, random_seed=42,
            kernel_profiler=True)
    scene, camera = loader.build(data, cfg)
    if args.bvh == "bvh4":
        scene.bvh = BVH4System()
    build_start = time.perf_counter()
    scene.bake()
    build_seconds = time.perf_counter() - build_start
    if args.bvh_stack is not None:
        required = scene.bvh.max_depth + 1
        if args.bvh_stack < required:
            raise ValueError(
                f"BVH 栈 {args.bvh_stack} 小于保守安全容量 {required}；"
                "拒绝运行可能漏交的基准")
        scene.bvh.stack_capacity = args.bvh_stack
    if args.integrator == "wavefront":
        if args.check_modes:
            raise ValueError("wavefront 实验路径不支持 --check-modes")
        renderer = WavefrontPathTracer(
            cfg.width, cfg.height,
            samples_per_wave=args.wavefront_lanes,
            bounces_per_kernel=args.wavefront_bounces)
    else:
        renderer = PathTracer(cfg.width, cfg.height,
                              enable_aovs=args.check_modes,
                              enable_adaptive=args.check_modes)

    print(f"BVH_STATS kind={args.bvh} integrator={args.integrator} "
          f"depth={scene.bvh.max_depth} leaves={scene.bvh.leaf_count} "
          f"leaf_avg={scene.bvh.avg_leaf_size:.3f} "
          f"leaf_max={scene.bvh.max_leaf_size} "
          f"stack_capacity={scene.bvh.stack_capacity}")

    if args.check_modes:
        renderer.render_batch(camera, scene, scene.materials, scene.tex_sys,
                              scene.light_sampler, 4, 1, 0, 1, True)
        ti.sync()
        assert renderer.get_aovs()["albedo"].shape == (cfg.height, cfg.width, 3)
        renderer.reset()
        renderer.render_batch(camera, scene, scene.materials, scene.tex_sys,
                              scene.light_sampler, 4, 2, 1, 2, False)
        ti.sync()
        renderer.update_convergence(2, 2, 0.01, 0.001)
        print("MODE_CHECK fixed+AOV and adaptive kernels compiled successfully")
        return

    # Compile before measuring.
    renderer.render_batch(camera, scene, scene.materials, scene.tex_sys,
                          scene.light_sampler, 2, 1, 0, 1)
    ti.sync()

    if args.focused:
        results = []
        # 第一轮仍可能包含驱动层升频扰动，不纳入统计。
        timed_batch(renderer, camera, scene, spp=args.spp, batch=16, bounce=30)
        for i in range(args.repeat):
            elapsed, mpaths = timed_batch(
                renderer, camera, scene, spp=args.spp, batch=16, bounce=30)
            results.append(mpaths)
            print(f"FOCUSED run={i + 1} seconds={elapsed:.6f} Mpaths/s={mpaths:.3f}")
        median = statistics.median(results)
        mean = statistics.fmean(results)
        spread = statistics.pstdev(results) / mean * 100.0 if mean else 0.0
        print(f"FOCUSED_RESULT stack={scene.bvh.stack_capacity} "
              f"median_Mpaths/s={median:.3f} mean_Mpaths/s={mean:.3f} "
              f"cv_percent={spread:.2f}")
        if args.signature:
            image = renderer.get_image()
            print(f"IMAGE_SIGNATURE finite={bool(np.isfinite(image).all())} "
                  f"sum={float(image.sum(dtype=np.float64)):.9f} "
                  f"mean={float(image.mean(dtype=np.float64)):.9f} "
                  f"max={float(np.max(image)):.9f}")
        return

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
