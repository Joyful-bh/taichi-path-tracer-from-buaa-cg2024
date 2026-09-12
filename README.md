# Taichi 路径追踪渲染器

这是一个使用 Python 和 Taichi 实现的蒙特卡洛路径追踪器。项目支持 YAML 场景配置、BVH 加速、多种材质、GLB/GLTF 纹理材质导入、面光源直接采样，以及渐进式预览和离线 PNG 输出。

## 主要功能

- Taichi CPU、CUDA 和 Vulkan 后端
- 球体、三角形、矩形、立方体和长方体
- CPU 构建、Taichi 栈式遍历的 BVH
- Lambertian、金属、介质、Clearcoat、自发光和 metallic-roughness PBR 材质
- 基础色、法线、金属度/粗糙度和自发光贴图
- OBJ、GLB/GLTF、FBX 和 PLY 网格加载
- 面光源 NEE（Next Event Estimation）
- ACES 色调映射和 Gamma 校正

## 环境要求

- Python 3.10 或更高版本
- 使用 CUDA 后端时需要可用的 NVIDIA GPU 和驱动
- Vulkan 后端需要系统提供 Vulkan 支持

建议在虚拟环境中安装依赖：

```bash
python -m venv .venv
```

Windows PowerShell：

```powershell
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## 使用方法

渐进式预览：

```bash
python main.py --scene scene_files/cornell_box.yaml --mode preview
```

离线渲染：

```bash
python main.py --scene scene_files/cornell_box.yaml --mode render
```

批量采样、AOV 和自适应采样在 YAML 的 `render` 段配置：

```yaml
render:
  samples_per_pixel: 2048
  samples_per_batch: 16
  save_aovs: false
  adaptive_sampling:
    enabled: true
    min_spp: 64
    max_spp: 2048
    check_interval: 32
    relative_error: 0.015
    absolute_error: 0.001
```

`samples_per_batch` 默认为 16，用于减少 GPU kernel 启动次数。`save_aovs` 默认为 `false`；
设为 `true` 时才会保存 `albedo`、`normal`、`depth`、`variance` 和 `sample_count` 侧车 PNG。
这个开关不会关闭渲染器内部的 AOV/方差统计，因此不影响自适应采样，也不妨碍以后直接在内存中接入降噪器。
自适应采样默认开启；像素至少采样 `min_spp`，然后根据
亮度标准误差停止已收敛像素，最多采样 `max_spp`。

覆盖场景中的采样数、后端和输出路径：

```bash
python main.py --scene scene_files/tardis.yaml --mode render --spp 512 --backend cuda --output output/tardis.png
```

`--backend` 可选 `cuda`、`vulkan` 或 `cpu`。如果机器没有可用的 GPU 后端，可显式使用 `--backend cpu`。

## 场景文件

`scene_files/` 包含以下示例：

- `cornell_box.yaml`：Cornell Box 和基础图元
- `tardis.yaml`：户外 TARDIS GLB 场景
- `tardis_in_box.yaml`：Cornell Box 中的 TARDIS
- `tardis_and_dalek.yaml`：TARDIS 和 Dalek 组合场景
- `three_chess_pieces.yaml`：镜面棋盘房间中的马、王和后，迁移自旧版场景
- `dragon_and_spheres.yaml`：清漆龙模型、球形太阳和可复现的随机球群，迁移自旧版场景
- `bvh_knight_cornell.yaml`：多镜面 Cornell Box 中的骑士棋子，迁移自旧版 `改进的BVH.py`

场景由 `render`、`camera`、`background`、`variables`、`materials` 和 `objects` 等字段组成。模型路径相对于当前 YAML 文件解析。
背景可以是固定 RGB，也可以使用 `type: gradient` 定义沿光线 Y 方向的渐变。
`checkerboard` 对象可生成规则棋盘；`random_spheres` 对象使用固定 `seed` 生成可复现的非重叠球群。

## 项目结构

```text
main.py                    命令行入口
scene_files/               YAML 场景定义
data/                      模型与纹理资源
src/geometry/              图元、BVH 和光源采样
src/materials/             材质与 BSDF
src/renderer/              路径追踪积分器
src/scene/                 场景与相机
src/textures/              纹理管理与采样
src/io/                    场景加载与图像输出
tests/test_cornell.py      端到端烟雾测试
output/                    默认渲染输出目录
```

## 运行测试

```bash
python -m pytest -q
```

现有测试使用 CPU 后端构建小型 Cornell Box，渲染 4 spp，并检查输出尺寸、有限值和基本亮度。

## 已知限制

- 当前针对静态网格，不处理骨骼动画和运动模糊。
- GLB/GLTF 的 PBR 材质支持比 FBX/PLY 更完整。
- PBR 自发光网格不会被面光源采样器作为 NEE 光源采样。
- 场景容量和 BVH 遍历栈使用预分配上限。
