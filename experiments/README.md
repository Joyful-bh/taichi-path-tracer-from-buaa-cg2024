# 实验模块

本目录保存已经实现、可以单独分析，但尚未进入稳定渲染路径的架构实验。
`main.py` 和安装后的 `taichi-path-tracer` 命令不会导入这些模块。

| 模块 | 状态 | 结论 |
| --- | --- | --- |
| `bvh4.py` | 实验性 | 正确性验证通过，但相对收紧栈容量的 BVH2 收益很小，默认继续使用 BVH2。 |
| `wavefront_path_tracer.py` | 实验性 | 正确性路径已实现，但当前 Taichi/场景组合中队列、原子操作和内存流量使其慢于 megakernel。 |

实验实现可以通过 `tools/profile_renderer.py` 的 `--bvh bvh4` 或
`--integrator wavefront` 显式启用。它们不承诺公共 API 稳定性。
