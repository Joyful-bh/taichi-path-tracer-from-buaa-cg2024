"""CPU binned-SAH BVH builder and Taichi GPU traversal."""

import numpy as np
import taichi as ti

from src.constants import BVH_STACK_SIZE, MAX_PRIMS_LEAF, T_MAX

SAH_BINS = 16
SAH_MAX_LEAF = 8


def _surface_area(bmin, bmax):
    e = np.maximum(bmax - bmin, 0.0)
    return 2.0 * (e[0] * e[1] + e[1] * e[2] + e[2] * e[0])


@ti.data_oriented
class BVHSystem:
    def __init__(self, max_nodes: int = 524288):
        self._max_nodes = max_nodes
        self.nodes = None
        self.prim_ids = None
        self.root_id = ti.field(ti.i32, shape=())
        self.n_spheres = ti.field(ti.i32, shape=())
        self._node_count = 0
        self._prim_count = 0
        # Python 常量，会在 Taichi 特化 kernel 时决定线程局部栈长度。
        # profile_renderer.py 可在首次编译前覆盖它，用于验证寄存器压力。
        self.stack_capacity = BVH_STACK_SIZE
        self.max_depth = 0
        self.leaf_count = 0
        self.max_leaf_size = 0
        self.avg_leaf_size = 0.0

    def build(self, sphere_bboxes, triangle_bboxes,
              n_spheres: int, n_triangles: int):
        sph_min, sph_max = sphere_bboxes
        tri_min, tri_max = triangle_bboxes
        mins = np.concatenate((sph_min, tri_min), axis=0).astype(np.float32)
        maxs = np.concatenate((sph_max, tri_max), axis=0).astype(np.float32)
        total = n_spheres + n_triangles
        if total == 0:
            self._allocate_fields(1, 1)
            self.root_id[None] = -1
            self.n_spheres[None] = 0
            return
        assert total * 2 - 1 <= self._max_nodes, \
            f"BVH 最坏情况节点数超过容量 {self._max_nodes}"

        centers = (mins + maxs) * np.float32(0.5)
        ids = np.arange(total, dtype=np.int32)
        self._py_nodes = []
        self._py_prim_ids = []
        self._node_count = self._prim_count = 0
        root = self._alloc_node()
        self._build_recursive(root, ids, mins, maxs, centers, 0)
        leaves = [n['count'] for n in self._py_nodes if n['count'] > 0]
        self.leaf_count = len(leaves)
        self.max_leaf_size = max(leaves, default=0)
        self.avg_leaf_size = float(np.mean(leaves)) if leaves else 0.0
        self._upload()
        self.root_id[None] = root
        self.n_spheres[None] = n_spheres
        print(f"[BVH] binned-SAH 构建完成：{self._node_count} 节点，{total} 图元，"
              f"最大深度={self.max_depth}，叶节点={self.leaf_count}，"
              f"平均/最大叶大小={self.avg_leaf_size:.2f}/{self.max_leaf_size}")

    def _allocate_fields(self, node_capacity, prim_capacity):
        node_type = ti.types.struct(
            bbox_min=ti.types.vector(3, ti.f32),
            bbox_max=ti.types.vector(3, ti.f32),
            start_index=ti.i32,
            count=ti.i32,
        )
        self.nodes = node_type.field(shape=(max(node_capacity, 1),))
        self.prim_ids = ti.field(ti.i32, shape=(max(prim_capacity, 1),))

    def _alloc_node(self):
        idx = self._node_count
        self._node_count += 1
        self._py_nodes.append({'bmin': np.zeros(3, np.float32),
                               'bmax': np.zeros(3, np.float32),
                               'start': 0, 'count': 0})
        return idx

    def _choose_split(self, ids, mins, maxs, centers, parent_min, parent_max):
        centroid_min = centers[ids].min(axis=0)
        centroid_max = centers[ids].max(axis=0)
        best_cost, best_axis, best_bin = np.inf, -1, -1
        for axis in range(3):
            extent = float(centroid_max[axis] - centroid_min[axis])
            if extent <= 1e-12:
                continue
            buckets = np.minimum(
                ((centers[ids, axis] - centroid_min[axis])
                 * (SAH_BINS / extent)).astype(np.int32), SAH_BINS - 1)
            counts = np.bincount(buckets, minlength=SAH_BINS).astype(np.int32)
            bin_min = np.full((SAH_BINS, 3), np.inf, np.float32)
            bin_max = np.full((SAH_BINS, 3), -np.inf, np.float32)
            for local, bucket in enumerate(buckets):
                pid = ids[local]
                bin_min[bucket] = np.minimum(bin_min[bucket], mins[pid])
                bin_max[bucket] = np.maximum(bin_max[bucket], maxs[pid])
            lc = np.zeros(SAH_BINS - 1, np.int32)
            rc = np.zeros(SAH_BINS - 1, np.int32)
            la = np.zeros(SAH_BINS - 1, np.float64)
            ra = np.zeros(SAH_BINS - 1, np.float64)
            bmin, bmax, running = np.full(3, np.inf), np.full(3, -np.inf), 0
            for i in range(SAH_BINS - 1):
                if counts[i]:
                    bmin = np.minimum(bmin, bin_min[i])
                    bmax = np.maximum(bmax, bin_max[i])
                running += counts[i]
                lc[i] = running
                la[i] = _surface_area(bmin, bmax) if running else 0.0
            bmin, bmax, running = np.full(3, np.inf), np.full(3, -np.inf), 0
            for i in range(SAH_BINS - 1, 0, -1):
                if counts[i]:
                    bmin = np.minimum(bmin, bin_min[i])
                    bmax = np.maximum(bmax, bin_max[i])
                running += counts[i]
                rc[i - 1] = running
                ra[i - 1] = _surface_area(bmin, bmax) if running else 0.0
            costs = lc * la + rc * ra
            split = int(np.argmin(costs))
            if costs[split] < best_cost:
                best_cost, best_axis, best_bin = float(costs[split]), axis, split

        leaf_cost = len(ids) * max(_surface_area(parent_min, parent_max), 1e-20)
        if best_axis < 0 or (best_cost >= leaf_cost and len(ids) <= SAH_MAX_LEAF):
            return None
        lo = float(centroid_min[best_axis])
        extent = float(centroid_max[best_axis] - centroid_min[best_axis])
        buckets = np.minimum(
            ((centers[ids, best_axis] - lo) * (SAH_BINS / extent)).astype(np.int32),
            SAH_BINS - 1)
        left, right = ids[buckets <= best_bin], ids[buckets > best_bin]
        if len(left) == 0 or len(right) == 0:
            # Degenerate centroids: balanced median split keeps stack depth bounded.
            axis = int(np.argmax(centroid_max - centroid_min))
            ordered = ids[np.argsort(centers[ids, axis], kind='stable')]
            mid = len(ordered) // 2
            left, right = ordered[:mid], ordered[mid:]
        return (left, right) if len(left) and len(right) else None

    def _build_recursive(self, node_idx, ids, mins, maxs, centers, depth):
        self.max_depth = max(self.max_depth, depth)
        bmin, bmax = mins[ids].min(axis=0), maxs[ids].max(axis=0)
        leaf = len(ids) <= MAX_PRIMS_LEAF or depth >= BVH_STACK_SIZE - 2
        split = None if leaf else self._choose_split(ids, mins, maxs, centers, bmin, bmax)
        if leaf or split is None:
            offset = self._prim_count
            self._py_prim_ids.extend(ids.tolist())
            self._prim_count += len(ids)
            self._py_nodes[node_idx] = {'bmin': bmin, 'bmax': bmax,
                                        'start': offset, 'count': len(ids)}
            return
        left_ids, right_ids = split
        left = self._alloc_node()
        right = self._alloc_node()
        self._build_recursive(left, left_ids, mins, maxs, centers, depth + 1)
        self._build_recursive(right, right_ids, mins, maxs, centers, depth + 1)
        self._py_nodes[node_idx] = {'bmin': bmin, 'bmax': bmax,
                                    'start': left, 'count': 0}

    def _upload(self):
        self._allocate_fields(self._node_count, self._prim_count)
        self.nodes.bbox_min.from_numpy(np.stack(
            [n['bmin'] for n in self._py_nodes]).astype(np.float32))
        self.nodes.bbox_max.from_numpy(np.stack(
            [n['bmax'] for n in self._py_nodes]).astype(np.float32))
        self.nodes.start_index.from_numpy(np.asarray(
            [n['start'] for n in self._py_nodes], np.int32))
        self.nodes.count.from_numpy(np.asarray(
            [n['count'] for n in self._py_nodes], np.int32))
        self.prim_ids.from_numpy(np.asarray(self._py_prim_ids, np.int32))

    @ti.func
    def intersect(self, ray_origin, ray_dir, t_min: ti.f32, t_max: ti.f32,
                  sphere_sys: ti.template(), tri_sys: ti.template()):
        closest_t, closest_pid = t_max, -1
        closest_u, closest_v = 0.0, 0.0
        inv_dir = 1.0 / ray_dir
        stack = ti.Vector([0] * self.stack_capacity, dt=ti.i32)
        stack_top = 0
        if self.root_id[None] >= 0:
            stack[0], stack_top = self.root_id[None], 1
        n_sph = self.n_spheres[None]
        while stack_top > 0:
            stack_top -= 1
            node = self.nodes[stack[stack_top]]
            if node.count > 0:
                for k in range(node.start_index, node.start_index + node.count):
                    pid = self.prim_ids[k]
                    hit_t, hit_u, hit_v = closest_t, 0.0, 0.0
                    if pid < n_sph:
                        hit_t = sphere_sys.hit_raw(pid, ray_origin, ray_dir, t_min, closest_t)
                    else:
                        hit_t, hit_u, hit_v = tri_sys.hit_raw(
                            pid - n_sph, ray_origin, ray_dir, t_min, closest_t)
                    if hit_t < closest_t:
                        closest_t, closest_pid = hit_t, pid
                        closest_u, closest_v = hit_u, hit_v
            else:
                left, right = node.start_index, node.start_index + 1
                dl = _ray_aabb_inv(ray_origin, inv_dir, self.nodes[left].bbox_min,
                                   self.nodes[left].bbox_max, t_min, closest_t)
                dr = _ray_aabb_inv(ray_origin, inv_dir, self.nodes[right].bbox_min,
                                   self.nodes[right].bbox_max, t_min, closest_t)
                if dl < dr:
                    if dr < closest_t and stack_top < self.stack_capacity:
                        stack[stack_top], stack_top = right, stack_top + 1
                    if dl < closest_t and stack_top < self.stack_capacity:
                        stack[stack_top], stack_top = left, stack_top + 1
                else:
                    if dl < closest_t and stack_top < self.stack_capacity:
                        stack[stack_top], stack_top = left, stack_top + 1
                    if dr < closest_t and stack_top < self.stack_capacity:
                        stack[stack_top], stack_top = right, stack_top + 1

        hit_pos = ti.Vector([0.0, 0.0, 0.0])
        hit_normal = ti.Vector([0.0, 0.0, 0.0])
        hit_tan = ti.Vector([0.0, 0.0, 0.0, 1.0])
        hit_uv = ti.Vector([0.0, 0.0])
        hit_mat, front = -1, True
        if closest_pid >= 0:
            if closest_pid < n_sph:
                hit_pos, hit_normal, hit_tan, hit_uv, hit_mat, front = \
                    sphere_sys.resolve_hit(closest_pid, ray_origin, ray_dir, closest_t)
            else:
                hit_pos, hit_normal, hit_tan, hit_uv, hit_mat, front = \
                    tri_sys.resolve_hit(closest_pid - n_sph, ray_origin, ray_dir,
                                        closest_t, closest_u, closest_v)
        return closest_t, hit_pos, hit_normal, hit_tan, hit_uv, hit_mat, front

    @ti.func
    def occluded(self, ray_origin, ray_dir, t_min: ti.f32, t_max: ti.f32,
                 sphere_sys: ti.template(), tri_sys: ti.template()):
        blocked = 0
        inv_dir = 1.0 / ray_dir
        stack = ti.Vector([0] * self.stack_capacity, dt=ti.i32)
        stack_top = 0
        if self.root_id[None] >= 0:
            stack[0], stack_top = self.root_id[None], 1
        n_sph = self.n_spheres[None]
        while stack_top > 0 and blocked == 0:
            stack_top -= 1
            node = self.nodes[stack[stack_top]]
            if node.count > 0:
                for k in range(node.start_index, node.start_index + node.count):
                    if blocked == 0:
                        pid = self.prim_ids[k]
                        hit_t = t_max
                        if pid < n_sph:
                            hit_t = sphere_sys.hit_raw(pid, ray_origin, ray_dir, t_min, t_max)
                        else:
                            hit_t, hit_u, hit_v = tri_sys.hit_raw(
                                pid - n_sph, ray_origin, ray_dir, t_min, t_max)
                        if hit_t < t_max:
                            blocked = 1
            else:
                left, right = node.start_index, node.start_index + 1
                dl = _ray_aabb_inv(ray_origin, inv_dir, self.nodes[left].bbox_min,
                                   self.nodes[left].bbox_max, t_min, t_max)
                dr = _ray_aabb_inv(ray_origin, inv_dir, self.nodes[right].bbox_min,
                                   self.nodes[right].bbox_max, t_min, t_max)
                if dl < dr:
                    if dr < t_max and stack_top < self.stack_capacity:
                        stack[stack_top], stack_top = right, stack_top + 1
                    if dl < t_max and stack_top < self.stack_capacity:
                        stack[stack_top], stack_top = left, stack_top + 1
                else:
                    if dl < t_max and stack_top < self.stack_capacity:
                        stack[stack_top], stack_top = left, stack_top + 1
                    if dr < t_max and stack_top < self.stack_capacity:
                        stack[stack_top], stack_top = right, stack_top + 1
        return blocked


@ti.func
def _ray_aabb_inv(ray_origin, inv_dir, bbox_min, bbox_max,
                  t_min: ti.f32, t_max: ti.f32):
    t0 = (bbox_min - ray_origin) * inv_dir
    t1 = (bbox_max - ray_origin) * inv_dir
    lo, hi = ti.min(t0, t1), ti.max(t0, t1)
    near = ti.max(lo[0], lo[1], lo[2], t_min)
    far = ti.min(hi[0], hi[1], hi[2], t_max)
    result = T_MAX
    if near <= far:
        result = near
    return result
