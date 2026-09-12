"""
快速烟雾测试：初始化 Cornell Box 场景，烘焙，并渲染 4 spp，
验证路径追踪 pipeline 端到端无崩溃，输出图像非全黑。

运行：
  python -m pytest tests/test_cornell.py -v
  或直接：
  python tests/test_cornell.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import taichi as ti
import numpy as np


def build_cornell_box():
    """用代码直接构建 Cornell Box（不依赖 YAML，便于单元测试）。"""
    from src.scene.scene    import Scene
    from src.scene.camera   import Camera

    scene = Scene(bg_color=[0.0, 0.0, 0.0])
    m  = scene.materials

    white  = m.add_lambertian([0.73, 0.73, 0.73])
    red    = m.add_lambertian([0.65, 0.05, 0.05])
    green  = m.add_lambertian([0.12, 0.45, 0.15])
    light  = m.add_light([1.0, 1.0, 1.0], intensity=12.0)
    glass  = m.add_dielectric(ior=1.5)

    tri = scene.triangles
    tri.add_rectangle([0,  -2.5, 0], 5, 5, (90, 0, 0), white)   # 地板
    tri.add_rectangle([0,   2.5, 0], 5, 5, (90, 0, 0), white)   # 天花板
    tri.add_rectangle([0,   0, -2.5], 5, 5, (0, 0, 0), white)   # 后墙
    tri.add_rectangle([-2.5, 0, 0], 5, 5, (0, 90, 0), red)      # 左墙
    tri.add_rectangle([ 2.5, 0, 0], 5, 5, (0, 90, 0), green)    # 右墙
    tri.add_rectangle([0, 2.49, 0], 1.5, 1.5, (90, 0, 0), light) # 面光源
    tri.add_cuboid([0.8, -1.0, -0.5], [1.5, 3.0, 1.5], (0, -15, 0), white)

    scene.add_sphere([-1.0, -1.6, 0.5], 0.9, glass)

    camera = Camera(
        lookfrom=[0, 0, 4.8], lookat=[0, 0, 0], up=[0, 1, 0],
        fov_deg=50.0, aspect_ratio=4/3
    )
    return scene, camera


def test_pipeline_no_crash():
    ti.init(arch=ti.cpu, random_seed=0)

    scene, camera = build_cornell_box()
    scene.bake()

    from src.renderer.path_tracer import PathTracer
    renderer = PathTracer(160, 120)

    for _ in range(4):
        renderer.render_sample(camera, scene, scene.materials, scene.tex_sys,
                               scene.light_sampler, max_bounce=8)

    img = renderer.get_image()   # (H, W, 3)
    assert img.shape == (120, 160, 3), f"意外图像尺寸：{img.shape}"
    assert np.isfinite(img).all(), "图像包含 NaN/Inf"
    mean_lum = img.mean()
    assert mean_lum > 0.001, f"图像过暗（mean={mean_lum:.5f}），可能光线追踪失败"
    print(f"\n[Test] OK — 4spp，平均亮度 {mean_lum:.4f}")


if __name__ == '__main__':
    test_pipeline_no_crash()
