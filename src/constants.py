# ===== 材质类型 ID =====
MAT_LAMBERTIAN = 0   # 漫反射
MAT_METAL      = 1   # 金属镜面
MAT_DIELECTRIC = 2   # 折射介质（玻璃）
MAT_LIGHT      = 3   # 自发光（面光源）
MAT_CLEARCOAT  = 4   # 清漆（漫反射 + 镜面概率混合）
MAT_PBR        = 5   # 基于物理的 PBR 材质（glTF metallic-roughness 模型）

# ===== 图元类型 ID（BVH 用） =====
PRIM_SPHERE   = 0
PRIM_TRIANGLE = 1

# ===== 渲染常数 =====
EPSILON          = 1e-4   # 阴影 acne 偏移量
T_MIN            = 1e-3   # 光线有效区间下界
T_MAX            = 1e9    # 光线有效区间上界
BVH_STACK_SIZE   = 32     # BVH 遍历栈深度，支持约 2^32 个图元
MAX_PRIMS_LEAF   = 4      # BVH 叶节点最大图元数（中点划分）
