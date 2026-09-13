"""Measure actual BVH work per path ray and bounce without changing render kernels."""

import argparse
from pathlib import Path
import sys
import time

import numpy as np
import taichi as ti

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.constants import BVH_STACK_SIZE, MAT_LIGHT, T_MAX, T_MIN
from src.geometry.bvh import _ray_aabb_inv
from src.io.loader import SceneLoader


@ti.func
def intersect_counted(scene: ti.template(), ray_origin, ray_dir,
                      t_min: ti.f32, t_max: ti.f32):
    """Closest-hit traversal plus thread-local work counters."""
    bvh = scene.bvh
    closest_t, closest_pid = t_max, -1
    closest_u, closest_v = 0.0, 0.0
    node_visits, aabb_tests, leaf_visits, prim_tests = 0, 0, 0, 0
    max_stack = 0
    inv_dir = 1.0 / ray_dir
    stack = ti.Vector([0] * BVH_STACK_SIZE, dt=ti.i32)
    stack_top = 0
    if bvh.root_id[None] >= 0:
        stack[0], stack_top = bvh.root_id[None], 1
        max_stack = 1
    n_sph = bvh.n_spheres[None]
    while stack_top > 0:
        stack_top -= 1
        node_visits += 1
        node = bvh.nodes[stack[stack_top]]
        if node.count > 0:
            leaf_visits += 1
            for k in range(node.start_index, node.start_index + node.count):
                prim_tests += 1
                pid = bvh.prim_ids[k]
                hit_t, hit_u, hit_v = closest_t, 0.0, 0.0
                if pid < n_sph:
                    hit_t = scene.spheres.hit_raw(
                        pid, ray_origin, ray_dir, t_min, closest_t)
                else:
                    hit_t, hit_u, hit_v = scene.triangles.hit_raw(
                        pid - n_sph, ray_origin, ray_dir, t_min, closest_t)
                if hit_t < closest_t:
                    closest_t, closest_pid = hit_t, pid
                    closest_u, closest_v = hit_u, hit_v
        else:
            left, right = node.start_index, node.start_index + 1
            dl = _ray_aabb_inv(ray_origin, inv_dir, bvh.nodes[left].bbox_min,
                               bvh.nodes[left].bbox_max, t_min, closest_t)
            dr = _ray_aabb_inv(ray_origin, inv_dir, bvh.nodes[right].bbox_min,
                               bvh.nodes[right].bbox_max, t_min, closest_t)
            aabb_tests += 2
            if dl < dr:
                if dr < closest_t and stack_top < BVH_STACK_SIZE:
                    stack[stack_top], stack_top = right, stack_top + 1
                if dl < closest_t and stack_top < BVH_STACK_SIZE:
                    stack[stack_top], stack_top = left, stack_top + 1
            else:
                if dl < closest_t and stack_top < BVH_STACK_SIZE:
                    stack[stack_top], stack_top = left, stack_top + 1
                if dr < closest_t and stack_top < BVH_STACK_SIZE:
                    stack[stack_top], stack_top = right, stack_top + 1
            max_stack = ti.max(max_stack, stack_top)

    hit_pos = ti.Vector([0.0, 0.0, 0.0])
    hit_normal = ti.Vector([0.0, 0.0, 0.0])
    hit_tan = ti.Vector([0.0, 0.0, 0.0, 1.0])
    hit_uv = ti.Vector([0.0, 0.0])
    hit_mat, front = -1, True
    if closest_pid >= 0:
        if closest_pid < n_sph:
            hit_pos, hit_normal, hit_tan, hit_uv, hit_mat, front = \
                scene.spheres.resolve_hit(
                    closest_pid, ray_origin, ray_dir, closest_t)
        else:
            hit_pos, hit_normal, hit_tan, hit_uv, hit_mat, front = \
                scene.triangles.resolve_hit(
                    closest_pid - n_sph, ray_origin, ray_dir,
                    closest_t, closest_u, closest_v)
    return (closest_t, hit_pos, hit_normal, hit_tan, hit_uv, hit_mat, front,
            node_visits, aabb_tests, leaf_visits, prim_tests, max_stack)


@ti.func
def occluded_counted(scene: ti.template(), ray_origin, ray_dir,
                     t_min: ti.f32, t_max: ti.f32):
    """Any-hit traversal plus thread-local work counters."""
    bvh = scene.bvh
    blocked = 0
    node_visits, aabb_tests, leaf_visits, prim_tests = 0, 0, 0, 0
    max_stack = 0
    inv_dir = 1.0 / ray_dir
    stack = ti.Vector([0] * BVH_STACK_SIZE, dt=ti.i32)
    stack_top = 0
    if bvh.root_id[None] >= 0:
        stack[0], stack_top = bvh.root_id[None], 1
        max_stack = 1
    n_sph = bvh.n_spheres[None]
    while stack_top > 0 and blocked == 0:
        stack_top -= 1
        node_visits += 1
        node = bvh.nodes[stack[stack_top]]
        if node.count > 0:
            leaf_visits += 1
            for k in range(node.start_index, node.start_index + node.count):
                if blocked == 0:
                    prim_tests += 1
                    pid = bvh.prim_ids[k]
                    hit_t = t_max
                    if pid < n_sph:
                        hit_t = scene.spheres.hit_raw(
                            pid, ray_origin, ray_dir, t_min, t_max)
                    else:
                        hit_t, hit_u, hit_v = scene.triangles.hit_raw(
                            pid - n_sph, ray_origin, ray_dir, t_min, t_max)
                    if hit_t < t_max:
                        blocked = 1
        else:
            left, right = node.start_index, node.start_index + 1
            dl = _ray_aabb_inv(ray_origin, inv_dir, bvh.nodes[left].bbox_min,
                               bvh.nodes[left].bbox_max, t_min, t_max)
            dr = _ray_aabb_inv(ray_origin, inv_dir, bvh.nodes[right].bbox_min,
                               bvh.nodes[right].bbox_max, t_min, t_max)
            aabb_tests += 2
            if dl < dr:
                if dr < t_max and stack_top < BVH_STACK_SIZE:
                    stack[stack_top], stack_top = right, stack_top + 1
                if dl < t_max and stack_top < BVH_STACK_SIZE:
                    stack[stack_top], stack_top = left, stack_top + 1
            else:
                if dl < t_max and stack_top < BVH_STACK_SIZE:
                    stack[stack_top], stack_top = left, stack_top + 1
                if dr < t_max and stack_top < BVH_STACK_SIZE:
                    stack[stack_top], stack_top = right, stack_top + 1
            max_stack = ti.max(max_stack, stack_top)
    return blocked, node_visits, aabb_tests, leaf_visits, prim_tests, max_stack


@ti.data_oriented
class TraversalCounters:
    # Columns: rays, hits/blocked, nodes, AABBs, leaves, primitives, stack sum.
    def __init__(self, max_bounce):
        self.max_bounce = max_bounce
        self.path = ti.field(ti.i64, shape=(max_bounce, 7))
        self.shadow = ti.field(ti.i64, shape=(max_bounce, 7))
        self.path_stack_max = ti.field(ti.i32, shape=(max_bounce,))
        self.shadow_stack_max = ti.field(ti.i32, shape=(max_bounce,))

    @ti.func
    def add_path(self, bounce, hit, nodes, aabbs, leaves, prims, stack):
        ti.atomic_add(self.path[bounce, 0], 1)
        ti.atomic_add(self.path[bounce, 1], hit)
        ti.atomic_add(self.path[bounce, 2], nodes)
        ti.atomic_add(self.path[bounce, 3], aabbs)
        ti.atomic_add(self.path[bounce, 4], leaves)
        ti.atomic_add(self.path[bounce, 5], prims)
        ti.atomic_add(self.path[bounce, 6], stack)
        ti.atomic_max(self.path_stack_max[bounce], stack)

    @ti.func
    def add_shadow(self, bounce, blocked, nodes, aabbs, leaves, prims, stack):
        ti.atomic_add(self.shadow[bounce, 0], 1)
        ti.atomic_add(self.shadow[bounce, 1], blocked)
        ti.atomic_add(self.shadow[bounce, 2], nodes)
        ti.atomic_add(self.shadow[bounce, 3], aabbs)
        ti.atomic_add(self.shadow[bounce, 4], leaves)
        ti.atomic_add(self.shadow[bounce, 5], prims)
        ti.atomic_add(self.shadow[bounce, 6], stack)
        ti.atomic_max(self.shadow_stack_max[bounce], stack)

    @ti.kernel
    def trace(self, camera: ti.template(), scene: ti.template(),
              spp: ti.i32, max_bounce: ti.i32):
        for px, py in ti.ndrange(camera_width(camera), camera_height(camera)):
            for _ in range(spp):
                s = (px + ti.random()) / camera_width(camera)
                t = (py + ti.random()) / camera_height(camera)
                cur_o, cur_d = camera.get_ray(s, t)
                attenuation = ti.Vector([1.0, 1.0, 1.0])
                for bounce in range(max_bounce):
                    (hit_t, hit_pos, hit_normal, hit_tan, hit_uv, hit_mat, front,
                     nodes, aabbs, leaves, prims, stack) = intersect_counted(
                        scene, cur_o, cur_d, T_MIN, T_MAX)
                    self.add_path(bounce, ti.cast(hit_mat >= 0, ti.i32),
                                  nodes, aabbs, leaves, prims, stack)
                    if hit_mat < 0:
                        break
                    scattered, scatter_att, emitted, should, specular, scatter_pdf = \
                        scene.materials.scatter(cur_d, hit_normal, hit_pos, hit_mat,
                                                hit_uv, front, hit_tan, scene.tex_sys)
                    if not should:
                        break

                    if not specular and scene.light_sampler.n_lights[None] > 0:
                        light_dir, dist, lemit, pdf, valid = \
                            scene.light_sampler.sample(hit_pos, ti.random())
                        if valid and dist > 1e-6:
                            if hit_normal.dot(light_dir) > 1e-4:
                                blocked, sn, sa, sl, sp, ss = occluded_counted(
                                    scene, hit_pos + hit_normal * T_MIN, light_dir,
                                    T_MIN, dist * (1.0 - 1e-3))
                                self.add_shadow(bounce, blocked, sn, sa, sl, sp, ss)

                    attenuation *= scatter_att
                    if bounce >= 5:
                        rr = ti.max(attenuation[0], attenuation[1], attenuation[2])
                        rr = ti.max(0.05, ti.min(0.95, rr))
                        if ti.random() >= rr:
                            break
                        attenuation /= rr
                    sign = 1.0 if scattered.dot(hit_normal) >= 0.0 else -1.0
                    cur_o = hit_pos + hit_normal * (T_MIN * sign)
                    cur_d = scattered


@ti.func
def camera_width(camera: ti.template()):
    return camera.image_width


@ti.func
def camera_height(camera: ti.template()):
    return camera.image_height


def print_table(label, values, maxima):
    print(f"\n{label}: bounce rays survival/hit nodes/ray aabb/ray leaves/ray prims/ray "
          "avg_stack max_stack")
    initial = max(int(values[0, 0]), 1)
    for b, row in enumerate(values):
        rays = int(row[0])
        if not rays:
            continue
        print(f"{b:2d} {rays:9d} {rays / initial:7.3f} {row[1] / rays:7.3f} "
              f"{row[2] / rays:9.3f} {row[3] / rays:8.3f} "
              f"{row[4] / rays:10.3f} {row[5] / rays:9.3f} "
              f"{row[6] / rays:9.3f} {int(maxima[b]):9d}")


def totals(values):
    rays = int(values[:, 0].sum())
    return {
        "rays": rays,
        "hit_rate": values[:, 1].sum() / max(rays, 1),
        "nodes_per_ray": values[:, 2].sum() / max(rays, 1),
        "aabbs_per_ray": values[:, 3].sum() / max(rays, 1),
        "leaves_per_ray": values[:, 4].sum() / max(rays, 1),
        "prims_per_ray": values[:, 5].sum() / max(rays, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scene")
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=180)
    parser.add_argument("--spp", type=int, default=8)
    parser.add_argument("--bounce", type=int, default=16)
    args = parser.parse_args()

    loader = SceneLoader()
    data, cfg = loader.load_config(args.scene)
    cfg.width, cfg.height, cfg.backend = args.width, args.height, "cuda"
    ti.init(arch=ti.cuda, default_fp=ti.f32, random_seed=42)
    scene, camera = loader.build(data, cfg)
    scene.bake()
    # Camera stores these as Python members; expose stable values expected above.
    camera.image_width, camera.image_height = cfg.width, cfg.height
    counters = TraversalCounters(args.bounce)
    started = time.perf_counter()
    counters.trace(camera, scene, args.spp, args.bounce)
    ti.sync()
    elapsed = time.perf_counter() - started
    path = counters.path.to_numpy()
    shadow = counters.shadow.to_numpy()
    print(f"TRAVERSAL_PROFILE scene={args.scene} size={cfg.width}x{cfg.height} "
          f"spp={args.spp} elapsed={elapsed:.3f}s")
    print_table("PATH", path, counters.path_stack_max.to_numpy())
    print_table("SHADOW", shadow, counters.shadow_stack_max.to_numpy())
    print("PATH_TOTAL", totals(path))
    print("SHADOW_TOTAL", totals(shadow))
    print(f"WORK_TOTAL aabb_tests={int(path[:, 3].sum() + shadow[:, 3].sum())} "
          f"primitive_tests={int(path[:, 5].sum() + shadow[:, 5].sum())}")


if __name__ == "__main__":
    main()
