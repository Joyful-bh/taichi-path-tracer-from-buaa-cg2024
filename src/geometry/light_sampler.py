"""Triangle and sphere light sampling with solid-angle PDFs for MIS."""
import numpy as np
import taichi as ti
from src.constants import MAT_LIGHT

LIGHT_TRIANGLE, LIGHT_SPHERE = 1, 2
_PI = 3.141592653589793


@ti.data_oriented
class LightSampler:
    def __init__(self, max_lights: int = 2048):
        self._max = max_lights
        self.n_lights = ti.field(ti.i32, shape=())
        self.total_area = ti.field(ti.f32, shape=())
        self.light_type = self.mat_id = None
        self.lv0 = self.lv1 = self.lv2 = self.lnrm = self.lemit = self.center = None
        self.radius = self.area = self.select_pdf = self.cdf = None
        self.sampled_material = self.material_light_type = None
        self.material_area_density = self.material_sphere_radius = None
        self.material_sphere_center = self.material_sphere_select_pdf = None

    def build(self, sphere_sys, tri_sys, mat_sys):
        mat_types = np.asarray(mat_sys._buf['mat_type'], np.int32)
        emits = (np.stack(mat_sys._buf['emit']).astype(np.float32)
                 if mat_sys.count else np.zeros((1, 3), np.float32))
        entries = []
        for i in range(tri_sys.count):
            mid = int(tri_sys._mat[i])
            if mat_types[mid] != MAT_LIGHT:
                continue
            v0, v1, v2 = (np.asarray(x[i], np.float32)
                          for x in (tri_sys._v0, tri_sys._v1, tri_sys._v2))
            cross = np.cross(v1 - v0, v2 - v0).astype(np.float32)
            area = 0.5 * float(np.linalg.norm(cross))
            if area > 1e-12:
                entries.append(dict(kind=LIGHT_TRIANGLE, mid=mid, v0=v0, v1=v1,
                                    v2=v2, nrm=cross / (2.0 * area),
                                    center=np.zeros(3, np.float32), radius=0.0,
                                    emit=emits[mid], area=area))
        for i in range(sphere_sys.count):
            mid = int(sphere_sys._mat_ids[i])
            if mat_types[mid] != MAT_LIGHT:
                continue
            radius = float(sphere_sys._radii[i])
            area = 4.0 * _PI * radius * radius
            if area > 1e-12:
                entries.append(dict(kind=LIGHT_SPHERE, mid=mid,
                                    v0=np.zeros(3, np.float32),
                                    v1=np.zeros(3, np.float32),
                                    v2=np.zeros(3, np.float32),
                                    nrm=np.zeros(3, np.float32),
                                    center=np.asarray(sphere_sys._centers[i], np.float32),
                                    radius=radius, emit=emits[mid], area=area))
        n = len(entries)
        assert n <= self._max, f"LightSampler 超出容量 {self._max}"
        cap, mcap = max(n, 1), max(mat_sys.count, 1)
        self.light_type = ti.field(ti.i32, shape=cap)
        self.mat_id = ti.field(ti.i32, shape=cap)
        self.lv0 = ti.Vector.field(3, ti.f32, shape=cap)
        self.lv1 = ti.Vector.field(3, ti.f32, shape=cap)
        self.lv2 = ti.Vector.field(3, ti.f32, shape=cap)
        self.lnrm = ti.Vector.field(3, ti.f32, shape=cap)
        self.lemit = ti.Vector.field(3, ti.f32, shape=cap)
        self.center = ti.Vector.field(3, ti.f32, shape=cap)
        self.radius = ti.field(ti.f32, shape=cap)
        self.area = ti.field(ti.f32, shape=cap)
        self.select_pdf = ti.field(ti.f32, shape=cap)
        self.cdf = ti.field(ti.f32, shape=cap)
        self.sampled_material = ti.field(ti.i32, shape=mcap)
        self.material_light_type = ti.field(ti.i32, shape=mcap)
        self.material_area_density = ti.field(ti.f32, shape=mcap)
        self.material_sphere_center = ti.Vector.field(3, ti.f32, shape=mcap)
        self.material_sphere_radius = ti.field(ti.f32, shape=mcap)
        self.material_sphere_select_pdf = ti.field(ti.f32, shape=mcap)

        def vec(name):
            out = np.zeros((cap, 3), np.float32)
            if n: out[:n] = np.stack([e[name] for e in entries])
            return out
        kinds, mids = np.zeros(cap, np.int32), np.zeros(cap, np.int32)
        radii, areas = np.zeros(cap, np.float32), np.zeros(cap, np.float32)
        if n:
            kinds[:n] = [e['kind'] for e in entries]; mids[:n] = [e['mid'] for e in entries]
            radii[:n] = [e['radius'] for e in entries]; areas[:n] = [e['area'] for e in entries]
            light_emit = np.stack([e['emit'] for e in entries])
            luminance = light_emit @ np.array([0.2126, 0.7152, 0.0722], np.float32)
            weights = areas[:n] * np.maximum(luminance, 1e-6)
            select = (weights / float(weights.sum())).astype(np.float32)
            cdf = np.cumsum(select).astype(np.float32); cdf[-1] = 1.0
        else:
            select, cdf = np.zeros(0, np.float32), np.zeros(0, np.float32)
        sp = np.zeros(cap, np.float32); sp[:n] = select
        cp = np.zeros(cap, np.float32); cp[:n] = cdf
        self.light_type.from_numpy(kinds); self.mat_id.from_numpy(mids)
        self.lv0.from_numpy(vec('v0')); self.lv1.from_numpy(vec('v1')); self.lv2.from_numpy(vec('v2'))
        self.lnrm.from_numpy(vec('nrm')); self.lemit.from_numpy(vec('emit')); self.center.from_numpy(vec('center'))
        self.radius.from_numpy(radii); self.area.from_numpy(areas)
        self.select_pdf.from_numpy(sp); self.cdf.from_numpy(cp)
        sampled, mkind = np.zeros(mcap, np.int32), np.zeros(mcap, np.int32)
        density, sr, ss = (np.zeros(mcap, np.float32) for _ in range(3))
        sc = np.zeros((mcap, 3), np.float32)
        for i, e in enumerate(entries):
            mid = e['mid']; sampled[mid] = 1; mkind[mid] = e['kind']
            if e['kind'] == LIGHT_TRIANGLE: density[mid] = select[i] / e['area']
            else:
                sc[mid], sr[mid], ss[mid] = e['center'], e['radius'], select[i]
        self.sampled_material.from_numpy(sampled); self.material_light_type.from_numpy(mkind)
        self.material_area_density.from_numpy(density); self.material_sphere_center.from_numpy(sc)
        self.material_sphere_radius.from_numpy(sr); self.material_sphere_select_pdf.from_numpy(ss)
        self.n_lights[None] = n
        self.total_area[None] = float(areas[:n].sum()) if n else 0.0
        nt = sum(e['kind'] == LIGHT_TRIANGLE for e in entries)
        print(f"[LightSampler] 光源：{nt} 三角形 + {n-nt} 球体，总面积={self.total_area[None]:.4f}")

    @ti.func
    def sample(self, ref_pos, u_select: ti.f32):
        direction = ti.Vector([0.0, 1.0, 0.0]); emit = ti.Vector([0.0, 0.0, 0.0])
        distance, pdf_w, valid = 0.0, 1.0, 0
        n = self.n_lights[None]
        if n > 0:
            lid = n - 1
            for k in range(n):
                if self.cdf[k] >= u_select:
                    lid = k; break
            psel, emit = self.select_pdf[lid], self.lemit[lid]
            if self.light_type[lid] == LIGHT_TRIANGLE:
                r1, r2 = ti.random(), ti.random(); sr1 = ti.sqrt(r1)
                pos = ((1.0-sr1)*self.lv0[lid] + sr1*(1.0-r2)*self.lv1[lid] + sr1*r2*self.lv2[lid])
                delta = pos - ref_pos; dist2 = delta.dot(delta)
                if dist2 > 1e-12:
                    distance = ti.sqrt(dist2); direction = delta / distance
                    cos_light = -direction.dot(self.lnrm[lid])
                    if cos_light > 1e-6:
                        pdf_w = psel * dist2 / (self.area[lid] * cos_light); valid = 1
            else:
                dc = self.center[lid] - ref_pos; dc2 = dc.dot(dc); radius = self.radius[lid]
                if dc2 > radius*radius*(1.0+1e-6):
                    d = ti.sqrt(dc2); w = dc/d
                    cos_max = ti.sqrt(ti.max(0.0, 1.0-radius*radius/dc2))
                    cos_t = 1.0-ti.random()*(1.0-cos_max)
                    sin_t = ti.sqrt(ti.max(0.0, 1.0-cos_t*cos_t)); phi = 2.0*_PI*ti.random()
                    helper = ti.Vector([0.0, 1.0, 0.0])
                    if ti.abs(w[1]) > 0.9: helper = ti.Vector([1.0, 0.0, 0.0])
                    tangent = helper.cross(w).normalized(); bitangent = w.cross(tangent)
                    direction = (tangent*(sin_t*ti.cos(phi)) + bitangent*(sin_t*ti.sin(phi)) + w*cos_t).normalized()
                    oc = ref_pos-self.center[lid]; hb = oc.dot(direction)
                    disc = hb*hb-(oc.dot(oc)-radius*radius)
                    if disc >= 0.0:
                        distance = -hb-ti.sqrt(disc)
                        if distance > 1e-6:
                            pdf_w = psel/(2.0*_PI*(1.0-cos_max)); valid = 1
                else:
                    z = 1.0-2.0*ti.random(); phi = 2.0*_PI*ti.random(); xy = ti.sqrt(ti.max(0.0, 1.0-z*z))
                    direction = ti.Vector([xy*ti.cos(phi), xy*ti.sin(phi), z])
                    oc = ref_pos-self.center[lid]; hb = oc.dot(direction)
                    disc = hb*hb-(oc.dot(oc)-radius*radius)
                    if disc >= 0.0:
                        distance = -hb+ti.sqrt(disc)
                        if distance > 1e-6:
                            pdf_w = psel/(4.0*_PI); valid = 1
        return direction, distance, emit, pdf_w, valid

    @ti.func
    def pdf_for_hit(self, ref_pos, hit_pos, hit_normal, hit_mat: ti.i32):
        pdf_w = 0.0; kind = self.material_light_type[hit_mat]
        delta = hit_pos-ref_pos; dist2 = delta.dot(delta)
        if kind == LIGHT_TRIANGLE and dist2 > 1e-12:
            direction = delta/ti.sqrt(dist2); cos_light = ti.abs(direction.dot(hit_normal))
            if cos_light > 1e-6: pdf_w = self.material_area_density[hit_mat]*dist2/cos_light
        elif kind == LIGHT_SPHERE:
            # Find the actual sphere, rather than assuming one emissive sphere per material.
            for lid in range(self.n_lights[None]):
                if self.light_type[lid] == LIGHT_SPHERE and self.mat_id[lid] == hit_mat:
                    radius = self.radius[lid]
                    surface_error = ti.abs((hit_pos-self.center[lid]).norm()-radius)
                    if surface_error <= ti.max(1e-3, radius*1e-4):
                        dc = self.center[lid]-ref_pos; dc2 = dc.dot(dc)
                        if dc2 > radius*radius*(1.0+1e-6):
                            cos_max = ti.sqrt(ti.max(0.0, 1.0-radius*radius/dc2))
                            pdf_w = self.select_pdf[lid]/(2.0*_PI*(1.0-cos_max))
                        else:
                            pdf_w = self.select_pdf[lid]/(4.0*_PI)
        return pdf_w
