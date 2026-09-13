"""Experimental four-way BVH for measuring Taichi/CUDA traversal behavior."""

import numpy as np
import taichi as ti

from src.constants import T_MAX
from src.geometry.bvh import BVHSystem, _ray_aabb_inv

BVH4_MAX_DEPTH = 14
BVH4_MAX_LEAF = 4


@ti.data_oriented
class BVH4System(BVHSystem):
    """Four-way SAH BVH with the same runtime interface as BVHSystem."""

    def __init__(self, max_nodes: int = 524288):
        super().__init__(max_nodes)
        self.stack_capacity = 1 + 3 * BVH4_MAX_DEPTH

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

        centers = (mins + maxs) * np.float32(0.5)
        self._py_nodes = []
        self._py_prim_ids = []
        self._node_count = self._prim_count = 0
        self.max_depth = 0
        root = self._alloc_node4()
        self._build_recursive4(root, np.arange(total, dtype=np.int32),
                               mins, maxs, centers, 0)
        # A four-way traversal can leave three siblings pending per level.
        self.stack_capacity = 1 + 3 * self.max_depth
        leaves = [n['count'] for n in self._py_nodes if n['count'] > 0]
        self.leaf_count = len(leaves)
        self.max_leaf_size = max(leaves, default=0)
        self.avg_leaf_size = float(np.mean(leaves)) if leaves else 0.0
        self._upload4()
        self.root_id[None] = root
        self.n_spheres[None] = n_spheres
        print(f"[BVH4] 构建完成：{self._node_count} 节点，{total} 图元，"
              f"最大深度={self.max_depth}，叶节点={self.leaf_count}，"
              f"平均/最大叶大小={self.avg_leaf_size:.2f}/{self.max_leaf_size}")

    def _alloc_node4(self):
        assert self._node_count < self._max_nodes, "BVH4 节点数超出容量"
        idx = self._node_count
        self._node_count += 1
        self._py_nodes.append({
            'bmin': np.zeros(3, np.float32),
            'bmax': np.zeros(3, np.float32),
            'children': np.full(4, -1, np.int32),
            'child_count': 0,
            'start': 0,
            'count': 0,
        })
        return idx

    def _build_recursive4(self, node_idx, ids, mins, maxs, centers, depth):
        self.max_depth = max(self.max_depth, depth)
        bmin, bmax = mins[ids].min(axis=0), maxs[ids].max(axis=0)
        if len(ids) <= BVH4_MAX_LEAF or depth >= BVH4_MAX_DEPTH:
            offset = self._prim_count
            self._py_prim_ids.extend(ids.tolist())
            self._prim_count += len(ids)
            self._py_nodes[node_idx].update(
                bmin=bmin, bmax=bmax, start=offset, count=len(ids))
            return

        groups = [ids]
        while len(groups) < 4:
            candidates = []
            for i, group in enumerate(groups):
                if len(group) > BVH4_MAX_LEAF:
                    gmin = mins[group].min(axis=0)
                    gmax = maxs[group].max(axis=0)
                    split = self._choose_split(
                        group, mins, maxs, centers, gmin, gmax)
                    if split is not None:
                        # Split the group with the greatest primitive-weighted area first.
                        score = len(group) * float(np.prod(np.maximum(gmax - gmin, 1e-12)))
                        candidates.append((score, i, split))
            if not candidates:
                break
            _, group_idx, (left, right) = max(candidates, key=lambda item: item[0])
            groups[group_idx:group_idx + 1] = [left, right]

        if len(groups) == 1:
            offset = self._prim_count
            self._py_prim_ids.extend(ids.tolist())
            self._prim_count += len(ids)
            self._py_nodes[node_idx].update(
                bmin=bmin, bmax=bmax, start=offset, count=len(ids))
            return

        child_ids = [self._alloc_node4() for _ in groups]
        children = np.full(4, -1, np.int32)
        children[:len(child_ids)] = child_ids
        self._py_nodes[node_idx].update(
            bmin=bmin, bmax=bmax, children=children,
            child_count=len(child_ids), count=0)
        for child, group in zip(child_ids, groups):
            self._build_recursive4(child, group, mins, maxs, centers, depth + 1)

    def _allocate_fields(self, node_capacity, prim_capacity):
        node_type = ti.types.struct(
            bbox_min=ti.types.vector(3, ti.f32),
            bbox_max=ti.types.vector(3, ti.f32),
            children=ti.types.vector(4, ti.i32),
            child_count=ti.i32,
            start_index=ti.i32,
            count=ti.i32,
        )
        self.nodes = node_type.field(shape=(max(node_capacity, 1),))
        self.prim_ids = ti.field(ti.i32, shape=(max(prim_capacity, 1),))

    def _upload4(self):
        self._allocate_fields(self._node_count, self._prim_count)
        self.nodes.bbox_min.from_numpy(np.stack(
            [n['bmin'] for n in self._py_nodes]).astype(np.float32))
        self.nodes.bbox_max.from_numpy(np.stack(
            [n['bmax'] for n in self._py_nodes]).astype(np.float32))
        self.nodes.children.from_numpy(np.stack(
            [n['children'] for n in self._py_nodes]).astype(np.int32))
        self.nodes.child_count.from_numpy(np.asarray(
            [n['child_count'] for n in self._py_nodes], np.int32))
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
                        hit_t = sphere_sys.hit_raw(
                            pid, ray_origin, ray_dir, t_min, closest_t)
                    else:
                        hit_t, hit_u, hit_v = tri_sys.hit_raw(
                            pid - n_sph, ray_origin, ray_dir, t_min, closest_t)
                    if hit_t < closest_t:
                        closest_t, closest_pid = hit_t, pid
                        closest_u, closest_v = hit_u, hit_v
            else:
                distances = ti.Vector([T_MAX, T_MAX, T_MAX, T_MAX])
                child_ids = node.children
                for i in ti.static(range(4)):
                    if i < node.child_count:
                        child = child_ids[i]
                        distances[i] = _ray_aabb_inv(
                            ray_origin, inv_dir, self.nodes[child].bbox_min,
                            self.nodes[child].bbox_max, t_min, closest_t)
                # Four-element sorting network, ascending by entry distance.
                for pass_id in ti.static(range(3)):
                    for i in ti.static(range(3)):
                        if i < 3 - pass_id and distances[i] > distances[i + 1]:
                            distances[i], distances[i + 1] = distances[i + 1], distances[i]
                            child_ids[i], child_ids[i + 1] = child_ids[i + 1], child_ids[i]
                # Push far-to-near so that the nearest child is popped first.
                for rev in ti.static(range(4)):
                    i = 3 - rev
                    if distances[i] < closest_t and stack_top < self.stack_capacity:
                        stack[stack_top], stack_top = child_ids[i], stack_top + 1

        hit_pos = ti.Vector([0.0, 0.0, 0.0])
        hit_normal = ti.Vector([0.0, 0.0, 0.0])
        hit_tan = ti.Vector([0.0, 0.0, 0.0, 1.0])
        hit_uv = ti.Vector([0.0, 0.0])
        hit_mat, front = -1, True
        if closest_pid >= 0:
            if closest_pid < n_sph:
                hit_pos, hit_normal, hit_tan, hit_uv, hit_mat, front = \
                    sphere_sys.resolve_hit(
                        closest_pid, ray_origin, ray_dir, closest_t)
            else:
                hit_pos, hit_normal, hit_tan, hit_uv, hit_mat, front = \
                    tri_sys.resolve_hit(
                        closest_pid - n_sph, ray_origin, ray_dir,
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
                            hit_t = sphere_sys.hit_raw(
                                pid, ray_origin, ray_dir, t_min, t_max)
                        else:
                            hit_t, hit_u, hit_v = tri_sys.hit_raw(
                                pid - n_sph, ray_origin, ray_dir, t_min, t_max)
                        if hit_t < t_max:
                            blocked = 1
            else:
                distances = ti.Vector([T_MAX, T_MAX, T_MAX, T_MAX])
                child_ids = node.children
                for i in ti.static(range(4)):
                    if i < node.child_count:
                        child = child_ids[i]
                        distances[i] = _ray_aabb_inv(
                            ray_origin, inv_dir, self.nodes[child].bbox_min,
                            self.nodes[child].bbox_max, t_min, t_max)
                for pass_id in ti.static(range(3)):
                    for i in ti.static(range(3)):
                        if i < 3 - pass_id and distances[i] > distances[i + 1]:
                            distances[i], distances[i + 1] = distances[i + 1], distances[i]
                            child_ids[i], child_ids[i + 1] = child_ids[i + 1], child_ids[i]
                for rev in ti.static(range(4)):
                    i = 3 - rev
                    if distances[i] < t_max and stack_top < self.stack_capacity:
                        stack[stack_top], stack_top = child_ids[i], stack_top + 1
        return blocked
