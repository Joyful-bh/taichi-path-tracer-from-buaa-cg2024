"""Experimental fixed-sampling wavefront path tracer for the Taichi backend."""

import taichi as ti

from src.constants import T_MAX, T_MIN, MAT_LIGHT
from src.renderer.path_tracer import _nee_contrib, _power_heuristic


@ti.data_oriented
class PathQueue:
    """SoA path state. Capacity is one path per pixel for one wavefront sample."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.origin = ti.Vector.field(3, ti.f32, shape=capacity)
        self.direction = ti.Vector.field(3, ti.f32, shape=capacity)
        self.throughput = ti.Vector.field(3, ti.f32, shape=capacity)
        self.radiance = ti.Vector.field(3, ti.f32, shape=capacity)
        self.pixel = ti.field(ti.i32, shape=capacity)
        self.slot = ti.field(ti.i32, shape=capacity)
        self.last_specular = ti.field(ti.i32, shape=capacity)
        self.bsdf_pdf = ti.field(ti.f32, shape=capacity)
        self.last_vertex = ti.Vector.field(3, ti.f32, shape=capacity)
        self.count = ti.field(ti.i32, shape=())


@ti.data_oriented
class WavefrontPathTracer:
    """Breadth-first path tracer; intentionally excludes AOV/adaptive modes."""

    def __init__(self, width: int, height: int, samples_per_wave: int = 4,
                 bounces_per_kernel: int = 1):
        self.width = width
        self.height = height
        self.samples_per_wave = max(1, int(samples_per_wave))
        self.bounces_per_kernel = max(1, int(bounces_per_kernel))
        self.capacity = width * height * self.samples_per_wave
        self.accumulator = ti.Vector.field(3, ti.f32, shape=(width, height))
        self.finished = ti.Vector.field(3, ti.f32, shape=self.capacity)
        self.sample_count = ti.field(ti.i32, shape=())
        self.queue_a = PathQueue(self.capacity)
        self.queue_b = PathQueue(self.capacity)
        self.sample_count[None] = 0

    @ti.kernel
    def reset(self):
        for i, j in self.accumulator:
            self.accumulator[i, j] = ti.Vector([0.0, 0.0, 0.0])
        self.sample_count[None] = 0
        self.queue_a.count[None] = 0
        self.queue_b.count[None] = 0

    @ti.kernel
    def _clear_queue(self, queue: ti.template()):
        queue.count[None] = 0

    @ti.kernel
    def _generate(self, camera: ti.template(), queue: ti.template(), lanes: ti.i32):
        queue.count[None] = self.width * self.height * lanes
        for px, py, lane in ti.ndrange(self.width, self.height, lanes):
            idx = (lane * self.height + py) * self.width + px
            s = (px + ti.random()) / self.width
            t = (py + ti.random()) / self.height
            ray_o, ray_d = camera.get_ray(s, t)
            queue.origin[idx] = ray_o
            queue.direction[idx] = ray_d
            queue.throughput[idx] = ti.Vector([1.0, 1.0, 1.0])
            queue.radiance[idx] = ti.Vector([0.0, 0.0, 0.0])
            queue.pixel[idx] = py * self.width + px
            queue.slot[idx] = idx
            queue.last_specular[idx] = 1
            queue.bsdf_pdf[idx] = 0.0
            queue.last_vertex[idx] = ray_o

    @ti.kernel
    def _bounce_chunk(self, source: ti.template(), destination: ti.template(),
                      scene: ti.template(), start_bounce: ti.i32,
                      max_bounce: ti.i32):
        for idx in range(source.count[None]):
            cur_o = source.origin[idx]
            cur_d = source.direction[idx]
            attenuation = source.throughput[idx]
            radiance = source.radiance[idx]
            pixel = source.pixel[idx]
            slot = source.slot[idx]
            last_specular = source.last_specular[idx]
            last_bsdf_pdf = source.bsdf_pdf[idx]
            last_vertex = source.last_vertex[idx]
            px = pixel % self.width
            py = pixel // self.width
            alive = True
            for local_bounce in ti.static(range(self.bounces_per_kernel)):
                bounce = start_bounce + local_bounce
                if alive and bounce < max_bounce:
                    hit_t, hit_pos, hit_normal, hit_tan, hit_uv, hit_mat, front = \
                        scene.intersect(cur_o, cur_d, T_MIN, T_MAX)
                    if hit_mat < 0:
                        radiance += attenuation * scene.background(cur_d)
                        alive = False
                    else:
                        scattered, scatter_att, emitted, should_scatter, is_specular, scatter_pdf = \
                            scene.materials.scatter(
                                cur_d, hit_normal, hit_pos, hit_mat, hit_uv,
                                front, hit_tan, scene.tex_sys)
                        mat_type = scene.materials.mats[hit_mat].mat_type
                        if mat_type == MAT_LIGHT:
                            mis_weight = 1.0
                            if (not last_specular and
                                    scene.light_sampler.sampled_material[hit_mat] != 0):
                                light_pdf = scene.light_sampler.pdf_for_hit(
                                    last_vertex, hit_pos, hit_normal, hit_mat)
                                mis_weight = _power_heuristic(last_bsdf_pdf, light_pdf)
                            radiance += attenuation * emitted * mis_weight
                        else:
                            radiance += attenuation * emitted

                        if not should_scatter:
                            alive = False
                        else:
                            if (not is_specular and
                                    scene.light_sampler.n_lights[None] > 0):
                                nee = _nee_contrib(
                                    cur_d, hit_pos, hit_normal, hit_mat, hit_uv,
                                    scene.materials, scene.tex_sys,
                                    scene.light_sampler, scene)
                                radiance += attenuation * nee
                            attenuation *= scatter_att
                            if bounce >= 5:
                                rr_prob = ti.max(
                                    attenuation[0], attenuation[1], attenuation[2])
                                rr_prob = ti.max(0.05, ti.min(0.95, rr_prob))
                                if ti.random() >= rr_prob:
                                    alive = False
                                else:
                                    attenuation /= rr_prob
                            if bounce + 1 >= max_bounce:
                                alive = False
                            if alive:
                                sign = 1.0 if scattered.dot(hit_normal) >= 0.0 else -1.0
                                cur_o = hit_pos + hit_normal * (T_MIN * sign)
                                cur_d = scattered
                                last_specular = ti.cast(is_specular, ti.i32)
                                last_bsdf_pdf = scatter_pdf
                                last_vertex = hit_pos

            if alive:
                out_idx = ti.atomic_add(destination.count[None], 1)
                destination.origin[out_idx] = cur_o
                destination.direction[out_idx] = cur_d
                destination.throughput[out_idx] = attenuation
                destination.radiance[out_idx] = radiance
                destination.pixel[out_idx] = pixel
                destination.slot[out_idx] = slot
                destination.last_specular[out_idx] = last_specular
                destination.bsdf_pdf[out_idx] = last_bsdf_pdf
                destination.last_vertex[out_idx] = last_vertex
            else:
                self.finished[slot] = radiance

    @ti.kernel
    def _accumulate_finished(self, lanes: ti.i32):
        """One non-atomic image write per pixel after a complete wave."""
        for px, py in self.accumulator:
            total = ti.Vector([0.0, 0.0, 0.0])
            for lane in range(lanes):
                slot = (lane * self.height + py) * self.width + px
                total += self.finished[slot]
            self.accumulator[px, py] += total

    def render_batch(self, camera, scene, mat_sys, tex_sys, light_sampler,
                     max_bounce, samples_per_batch, adaptive_enabled=False,
                     max_spp=None, save_aovs=False):
        if adaptive_enabled or save_aovs:
            raise ValueError("实验性 wavefront 路径当前只支持固定采样且不保存 AOV")
        if max_spp is None:
            max_spp = self.spp + samples_per_batch
        actual = min(samples_per_batch, max_spp - self.spp)
        remaining = max(actual, 0)
        while remaining > 0:
            lanes = min(remaining, self.samples_per_wave)
            self._generate(camera, self.queue_a, lanes)
            source, destination = self.queue_a, self.queue_b
            for bounce in range(0, max_bounce, self.bounces_per_kernel):
                self._clear_queue(destination)
                self._bounce_chunk(source, destination, scene, bounce, max_bounce)
                source, destination = destination, source
            self._accumulate_finished(lanes)
            remaining -= lanes
        self.sample_count[None] = min(self.spp + max(actual, 0), max_spp)

    def get_image(self):
        n = max(self.spp, 1)
        return self.accumulator.to_numpy().transpose(1, 0, 2) / float(n)

    @property
    def spp(self):
        return int(self.sample_count[None])
