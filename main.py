"""
光线追踪渲染器入口。

用法：
  python main.py --scene scene_files/cornell_box.yaml --mode preview
  python main.py --scene scene_files/cornell_box.yaml --mode render --spp 1024

模式：
  preview  — 实时 GGUI 窗口，渐进式累积，可交互。
  render   — 跑满 spp 后 ACES 色调映射 + gamma 输出 PNG，不打开窗口。
"""

import argparse
import sys
import os

# 防止 Windows 上 UTF-8 输出乱码
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def parse_args():
    parser = argparse.ArgumentParser(description='Taichi Path Tracer')
    parser.add_argument('--scene',   default='scene_files/cornell_box.yaml',
                        help='场景描述文件路径（YAML）')
    parser.add_argument('--mode',    choices=['preview', 'render'], default='preview',
                        help='运行模式：preview（渐进预览）或 render（离线出图）')
    parser.add_argument('--spp',     type=int, default=None,
                        help='采样数（覆盖 YAML 中的设置）')
    parser.add_argument('--output',  default=None,
                        help='输出路径（render 模式，覆盖 YAML 设置）')
    parser.add_argument('--backend', choices=['cuda', 'vulkan', 'cpu'], default=None,
                        help='Taichi 后端（覆盖 YAML 设置）')
    return parser.parse_args()


def main():
    args = parse_args()

    # ----------------------------------------------------------------
    # 1. 解析场景文件（纯 Python，不创建 ti.field）
    # ----------------------------------------------------------------
    from src.io.loader import SceneLoader
    loader = SceneLoader()
    yaml_data, cfg = loader.load_config(args.scene)

    # 命令行参数覆盖 YAML
    if args.spp is not None:
        cfg.spp = args.spp
        if cfg.adaptive_enabled:
            cfg.adaptive_max_spp = max(cfg.adaptive_min_spp, args.spp)
            cfg.spp = cfg.adaptive_max_spp
    if args.output  is not None: cfg.output  = args.output
    if args.backend is not None: cfg.backend = args.backend

    # ----------------------------------------------------------------
    # 2. 初始化 Taichi（全局只调用一次，在任何 ti.field 分配前）
    # ----------------------------------------------------------------
    import taichi as ti
    backend_map = {'cuda': ti.cuda, 'vulkan': ti.vulkan, 'cpu': ti.cpu}
    ti.init(arch=backend_map.get(cfg.backend, ti.cuda),
            default_fp=ti.f32,
            random_seed=42)

    # ----------------------------------------------------------------
    # 3. 构建场景（分配 ti.field，ti.init() 之后才能执行）
    # ----------------------------------------------------------------
    scene, camera = loader.build(yaml_data, cfg)

    # ----------------------------------------------------------------
    # 4. 烘焙场景（上传到 GPU，构建 BVH）
    # ----------------------------------------------------------------
    scene.bake()

    # ----------------------------------------------------------------
    # 4. 创建渲染器
    # ----------------------------------------------------------------
    from src.renderer.path_tracer import PathTracer
    needs_aovs = cfg.save_aovs or cfg.denoise_enabled
    renderer = PathTracer(cfg.width, cfg.height,
                          enable_aovs=needs_aovs,
                          enable_adaptive=cfg.adaptive_enabled)

    adaptive_label = "开启" if cfg.adaptive_enabled else "关闭"
    print(f"[Main] 分辨率 {cfg.width}×{cfg.height}，模式={args.mode}，"
          f"SPP={cfg.spp}，批量={cfg.samples_per_batch}，"
          f"自适应={adaptive_label}，后端={cfg.backend}")

    # ----------------------------------------------------------------
    # 5. 运行
    # ----------------------------------------------------------------
    if args.mode == 'render':
        _run_offline(renderer, camera, scene, cfg)
    else:
        _run_preview(renderer, camera, scene, cfg)


def _run_offline(renderer, camera, scene, cfg):
    """离线渲染：跑满 SPP 后输出 PNG。"""
    import time
    from src.io.image_output import save_image, save_aovs

    t0 = time.time()
    next_report = 0.05
    while not renderer.is_complete(cfg.spp, cfg.adaptive_enabled):
        batch = min(cfg.samples_per_batch, cfg.spp - renderer.spp)
        renderer.render_batch(camera, scene, scene.materials, scene.tex_sys,
                              scene.light_sampler, cfg.max_bounce, batch,
                              cfg.adaptive_enabled, cfg.spp,
                              cfg.save_aovs or cfg.denoise_enabled)
        if cfg.adaptive_enabled and renderer.spp >= cfg.adaptive_min_spp and (
                renderer.spp % cfg.adaptive_check_interval < batch or renderer.spp >= cfg.spp):
            renderer.update_convergence(
                cfg.adaptive_min_spp, cfg.spp,
                cfg.adaptive_relative_error, cfg.adaptive_absolute_error)
        pct = renderer.progress(cfg.spp, cfg.adaptive_enabled)
        if pct >= next_report or renderer.is_complete(cfg.spp, cfg.adaptive_enabled):
            elapsed = time.time() - t0
            eta = elapsed / max(pct, 1e-6) * (1.0 - pct)
            active = renderer.active_pixels
            print(f"\r  渲染进度：{pct * 100:5.1f}%  SPP={renderer.spp}  "
                  f"活跃像素={active}  已用 {elapsed:.1f}s  预计剩余 {eta:.1f}s",
                  end='', flush=True)
            next_report += 0.05

    print()
    img = renderer.get_image()   # (H, W, 3) linear HDR
    aovs = renderer.get_aovs() if (cfg.save_aovs or cfg.denoise_enabled) else None
    if cfg.denoise_enabled:
        from src.io.oidn_denoiser import denoise_oidn, denoised_output_path
        denoised_path = denoised_output_path(cfg.output, cfg.denoise_output)
        if cfg.denoise_save_noisy:
            save_image(img, cfg.output)
        print(f"[OIDN] 开始降噪：device={cfg.denoise_device}，quality={cfg.denoise_quality}")
        denoised = denoise_oidn(
            img,
            albedo=aovs['albedo'] if cfg.denoise_use_albedo else None,
            normal=aovs['normal'] if cfg.denoise_use_normal else None,
            executable=cfg.denoise_executable,
            device=cfg.denoise_device,
            quality=cfg.denoise_quality,
        )
        save_image(denoised, denoised_path)
        print(f"[OIDN] 已完成：{denoised_path}")
    else:
        save_image(img, cfg.output)
    if cfg.save_aovs:
        save_aovs(aovs, cfg.output)
    print(f"[Main] 渲染完成，总耗时 {time.time() - t0:.1f}s")


def _run_preview(renderer, camera, scene, cfg):
    """
    渐进式预览：GGUI 窗口实时显示，每帧追加 1 spp。
    关闭窗口或达到 SPP 上限后停止。

    注意：display 是 (W, H) 的 Taichi field，GGUI 将 field[x, 0] 显示在屏幕底部，
    与 accumulator[x, 0]（视口底部像素）天然对应，无需翻转。
    """
    import taichi as ti

    window = ti.ui.Window(
        "Path Tracer — 渐进式预览", (cfg.width, cfg.height), vsync=False
    )
    canvas  = window.get_canvas()
    # display field 形状 (W, H)，GGUI 直接读取，无 CPU 中转
    display = ti.Vector.field(3, ti.f32, shape=(cfg.width, cfg.height))

    spp_limit = cfg.spp
    print(f"[Main] 预览启动，按关闭按钮退出（最多 {spp_limit} spp）")

    while window.running and not renderer.is_complete(spp_limit, cfg.adaptive_enabled):
        batch = min(cfg.samples_per_batch, spp_limit - renderer.spp)
        renderer.render_batch(camera, scene, scene.materials, scene.tex_sys,
                              scene.light_sampler, cfg.max_bounce, batch,
                              cfg.adaptive_enabled, spp_limit,
                              cfg.save_aovs or cfg.denoise_enabled)
        if cfg.adaptive_enabled and renderer.spp >= cfg.adaptive_min_spp and (
                renderer.spp % cfg.adaptive_check_interval < batch or renderer.spp >= spp_limit):
            renderer.update_convergence(
                cfg.adaptive_min_spp, spp_limit,
                cfg.adaptive_relative_error, cfg.adaptive_absolute_error)

        # 每 4 spp 更新一次色调映射结果（GPU 上直接完成，无 CPU 中转）
        renderer.blit_to_display(display)

        # 每帧都调用 set_image，否则 GGUI 会显示黑帧
        canvas.set_image(display)
        window.show()

        if renderer.spp % 16 == 0:
            print(f"\r  SPP = {renderer.spp}/{spp_limit}", end='', flush=True)

    print(f"\n[Main] 渲染完成，共 {renderer.spp} spp — 关闭窗口退出")

    # 渲染结束后保存图像，然后保持窗口直到手动关闭
    from src.io.image_output import save_image, save_aovs
    img = renderer.get_image()
    aovs = renderer.get_aovs() if (cfg.save_aovs or cfg.denoise_enabled) else None
    if cfg.denoise_enabled:
        from src.io.oidn_denoiser import denoise_oidn, denoised_output_path
        if cfg.denoise_save_noisy:
            save_image(img, cfg.output)
        denoised = denoise_oidn(
            img,
            albedo=aovs['albedo'] if cfg.denoise_use_albedo else None,
            normal=aovs['normal'] if cfg.denoise_use_normal else None,
            executable=cfg.denoise_executable,
            device=cfg.denoise_device,
            quality=cfg.denoise_quality,
        )
        save_image(denoised, denoised_output_path(cfg.output, cfg.denoise_output))
    else:
        save_image(img, cfg.output)
    if cfg.save_aovs:
        save_aovs(aovs, cfg.output)

    while window.running:
        canvas.set_image(display)
        window.show()


if __name__ == '__main__':
    main()
