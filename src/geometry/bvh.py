"""
BVH（层次包围体）系统。

构建阶段（Python CPU 端）:
  - 对所有图元按最长轴中点划分，递归构建二叉树。
  - 子节点先于递归调用分配，保证左子在 start_index，右子在 start_index+1。
  - 展平成数组后通过 from_numpy 批量上传到 Taichi field。

遍历阶段（Taichi GPU 端）:
  - 栈式迭代，栈深度 BVH_STACK_SIZE（默认 32，支持 ~2^31 个图元）。
  - 先测试近子节点，后测试远子节点（减少无效遍历）。
  - 叶节点：对 primitive_indices[start:start+count] 中每个图元做精确求交。
  - 图元 ID 编码：[0, n_spheres) 为球体，[n_spheres, n_spheres+n_triangles) 为三角形。

节点编码：
  - count == 0：内部节点，左子 = nodes[start_index]，右子 = nodes[start_index+1]
  - count > 0 ：叶节点，图元在 primitive_indices[start_index : start_index+count]
"""

import taichi as ti
import numpy as np

from src.constants import BVH_STACK_SIZE, MAX_PRIMS_LEAF, T_MAX


@ti.data_oriented
class BVHSystem:

    def __init__(self, max_nodes: int = 524288):  # 2^19 = 512K，足够覆盖数十万三角形
        self._max_nodes = max_nodes
        # 节点结构（AoS，每线程通常访问同一节点的多个字段）
        BVHNode = ti.types.struct(
            bbox_min    = ti.types.vector(3, ti.f32),
            bbox_max    = ti.types.vector(3, ti.f32),
            start_index = ti.i32,   # 内部节点：左子 ID；叶节点：图元索引表偏移
            count       = ti.i32,   # 0 = 内部节点；>0 = 叶节点图元数
        )
        self.nodes = BVHNode.field(shape=(max_nodes,))
        # 图元索引重排表（BVH 叶节点引用）
        self.prim_ids = ti.field(ti.i32, shape=(max_nodes * 2,))
        # 运行时只读标量
        self.root_id   = ti.field(ti.i32, shape=())
        self.n_spheres = ti.field(ti.i32, shape=())
        self.root_id[None]   = -1
        self.n_spheres[None] = 0

    # ------------------------------------------------------------------
    # Python 端：构建 BVH
    # ------------------------------------------------------------------

    def build(self, sphere_bboxes, triangle_bboxes, n_spheres: int, n_triangles: int):
        """
        在 CPU 上构建 BVH，然后上传到 Taichi field。

        sphere_bboxes   : (n_sph, 3) × 2  tuple(bbox_min, bbox_max)
        triangle_bboxes : (n_tri, 3) × 2  tuple(bbox_min, bbox_max)
        """
        # 合并所有图元（球体优先，三角形次之）
        all_mins, all_maxs, all_ids = [], [], []
        sph_min, sph_max = sphere_bboxes
        for i in range(n_spheres):
            all_mins.append(sph_min[i])
            all_maxs.append(sph_max[i])
            all_ids.append(i)                       # sphere ID: [0, n_spheres)

        tri_min, tri_max = triangle_bboxes
        for i in range(n_triangles):
            all_mins.append(tri_min[i])
            all_maxs.append(tri_max[i])
            all_ids.append(n_spheres + i)          # triangle ID: [n_spheres, ...)

        if not all_ids:
            return

        primitives = [
            {
                'bmin'  : np.array(all_mins[i], np.float32),
                'bmax'  : np.array(all_maxs[i], np.float32),
                'center': (np.array(all_mins[i]) + np.array(all_maxs[i])) * 0.5,
                'id'    : all_ids[i],
            }
            for i in range(len(all_ids))
        ]

        # Python 端节点/图元列表（之后批量上传）
        self._py_nodes    = []
        self._py_prim_ids = []
        self._node_count  = 0
        self._prim_count  = 0

        root = self._alloc_node()
        self._build_recursive(root, primitives, 0, len(primitives), depth=0)
        self._upload()

        self.root_id[None]   = root
        self.n_spheres[None] = n_spheres
        print(f"[BVH] 构建完成：{self._node_count} 节点，{len(all_ids)} 图元")

    def _alloc_node(self) -> int:
        assert self._node_count < self._max_nodes, "BVH 节点数超出最大值"
        self._py_nodes.append({'bmin': np.zeros(3, np.float32),
                                'bmax': np.zeros(3, np.float32),
                                'start': 0, 'count': 0})
        idx = self._node_count
        self._node_count += 1
        return idx

    def _build_recursive(self, node_idx, prims, start, end, depth):
        if start >= end:
            return

        # 计算当前子集的 AABB
        bmin = np.minimum.reduce([p['bmin'] for p in prims[start:end]])
        bmax = np.maximum.reduce([p['bmax'] for p in prims[start:end]])

        # 叶节点条件：图元数 ≤ MAX_PRIMS_LEAF 或深度过大（防止退化场景栈溢出）
        if (end - start) <= MAX_PRIMS_LEAF or depth >= BVH_STACK_SIZE - 2:
            prim_offset = self._prim_count
            for i in range(start, end):
                self._py_prim_ids.append(prims[i]['id'])
                self._prim_count += 1
            self._py_nodes[node_idx] = {
                'bmin': bmin, 'bmax': bmax,
                'start': prim_offset, 'count': end - start,
            }
            return

        # 按最长轴中点划分
        extent = bmax - bmin
        axis = int(np.argmax(extent))
        prims[start:end] = sorted(prims[start:end], key=lambda p: p['center'][axis])
        mid = start + (end - start) // 2

        # 先分配两个子节点（保证 right = left + 1）
        left  = self._alloc_node()
        right = self._alloc_node()
        self._build_recursive(left,  prims, start, mid, depth + 1)
        self._build_recursive(right, prims, mid,   end, depth + 1)

        self._py_nodes[node_idx] = {
            'bmin': bmin, 'bmax': bmax,
            'start': left,   # right = left + 1（由分配顺序保证）
            'count': 0,
        }

    def _upload(self):
        """将 Python 端数据批量上传到 Taichi field。"""
        n  = self._node_count
        np_ = self._prim_count

        # 节点字段
        bmin_arr  = np.stack([nd['bmin']  for nd in self._py_nodes]).astype(np.float32)
        bmax_arr  = np.stack([nd['bmax']  for nd in self._py_nodes]).astype(np.float32)
        start_arr = np.array([nd['start'] for nd in self._py_nodes], np.int32)
        count_arr = np.array([nd['count'] for nd in self._py_nodes], np.int32)

        # 用零填充到 max_nodes
        def _pad3(arr):
            out = np.zeros((self._max_nodes, 3), np.float32)
            out[:n] = arr
            return out
        def _pad1i(arr, max_n):
            out = np.zeros(max_n, np.int32)
            out[:len(arr)] = arr
            return out

        self.nodes.bbox_min.from_numpy(_pad3(bmin_arr))
        self.nodes.bbox_max.from_numpy(_pad3(bmax_arr))
        self.nodes.start_index.from_numpy(_pad1i(start_arr, self._max_nodes))
        self.nodes.count.from_numpy(_pad1i(count_arr, self._max_nodes))

        pid_arr = np.array(self._py_prim_ids, np.int32)
        pid_out = np.zeros(self._max_nodes * 2, np.int32)
        pid_out[:np_] = pid_arr
        self.prim_ids.from_numpy(pid_out)

    # ------------------------------------------------------------------
    # Taichi 作用域：BVH 遍历
    # ------------------------------------------------------------------

    @ti.func
    def intersect(self, ray_origin, ray_dir, t_min: ti.f32, t_max: ti.f32,
                  sphere_sys: ti.template(), tri_sys: ti.template()):
        """
        BVH 栈式迭代遍历求交。
        返回 (hit_t, hit_pos, hit_normal, hit_uv, hit_mat, front_face)。
        未命中时 hit_t == t_max，hit_mat == -1。
        """
        closest_t  = t_max
        hit_pos    = ti.Vector([0.0, 0.0, 0.0])
        hit_normal = ti.Vector([0.0, 0.0, 0.0])
        hit_tan    = ti.Vector([0.0, 0.0, 0.0, 1.0])
        hit_uv     = ti.Vector([0.0, 0.0])
        hit_mat    = -1
        front      = True

        # 每线程独立栈（Taichi GPU 上为线程私有局部变量）
        stack      = ti.Vector([0] * BVH_STACK_SIZE, dt=ti.i32)
        stack_top  = 0
        stack[stack_top] = self.root_id[None]
        stack_top += 1

        n_sph = self.n_spheres[None]

        while stack_top > 0:
            stack_top -= 1
            node_idx = stack[stack_top]
            node     = self.nodes[node_idx]

            if node.count > 0:
                # 叶节点：精确求交
                for k in range(node.start_index, node.start_index + node.count):
                    pid = self.prim_ids[k]
                    # 必须先初始化，Taichi 不允许在分支内首次定义变量
                    t   = closest_t
                    pos = ti.Vector([0.0, 0.0, 0.0])
                    nrm = ti.Vector([0.0, 0.0, 0.0])
                    tan = ti.Vector([0.0, 0.0, 0.0, 1.0])
                    uv  = ti.Vector([0.0, 0.0])
                    mat = -1
                    fr  = True
                    if pid < n_sph:
                        t, pos, nrm, tan, uv, mat, fr = sphere_sys.hit(pid, ray_origin, ray_dir, t_min, closest_t)
                    else:
                        t, pos, nrm, tan, uv, mat, fr = tri_sys.hit(pid - n_sph, ray_origin, ray_dir, t_min, closest_t)
                    if t < closest_t:
                        closest_t  = t
                        hit_pos    = pos
                        hit_normal = nrm
                        hit_tan    = tan
                        hit_uv     = uv
                        hit_mat    = mat
                        front      = fr
            else:
                # 内部节点：AABB 测试决定是否压栈
                left  = node.start_index
                right = left + 1
                dl = _ray_aabb(ray_origin, ray_dir, self.nodes[left].bbox_min,  self.nodes[left].bbox_max,  t_min, closest_t)
                dr = _ray_aabb(ray_origin, ray_dir, self.nodes[right].bbox_min, self.nodes[right].bbox_max, t_min, closest_t)

                # 先处理近子节点（后压栈 = 先出栈）
                if dl < dr:
                    if dr < closest_t and stack_top < BVH_STACK_SIZE - 1:
                        stack[stack_top] = right;  stack_top += 1
                    if dl < closest_t and stack_top < BVH_STACK_SIZE - 1:
                        stack[stack_top] = left;   stack_top += 1
                else:
                    if dl < closest_t and stack_top < BVH_STACK_SIZE - 1:
                        stack[stack_top] = left;   stack_top += 1
                    if dr < closest_t and stack_top < BVH_STACK_SIZE - 1:
                        stack[stack_top] = right;  stack_top += 1

        return closest_t, hit_pos, hit_normal, hit_tan, hit_uv, hit_mat, front


@ti.func
def _ray_aabb(ray_origin, ray_dir, bbox_min, bbox_max, t_min: ti.f32, t_max: ti.f32) -> ti.f32:
    """Slab 法光线-AABB 相交测试，返回入交距离，未命中返回 T_MAX。"""
    inv_dir = 1.0 / ray_dir
    t0 = (bbox_min - ray_origin) * inv_dir
    t1 = (bbox_max - ray_origin) * inv_dir
    t_small = ti.min(t0, t1)
    t_large = ti.max(t0, t1)
    t_near  = ti.max(t_small[0], t_small[1], t_small[2], t_min)
    t_far   = ti.min(t_large[0], t_large[1], t_large[2], t_max)
    result  = T_MAX
    if t_near <= t_far:
        result = t_near
    return result
