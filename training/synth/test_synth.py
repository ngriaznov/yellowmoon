"""Checks of the synthetic scene generator (``python3 -m unittest training.synth.test_synth``; about 40 s).

They need the fetched spectral sources (``tools/fetch/spectral.sh``) like the other training scripts and Pillow. Rust
reads what this writes in ``tools/golden`` (``cargo test -p golden``) and ``tools/fetch/synthetic.sh`` runs both.
"""

from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from training.common import paths

paths.require_sources()

from training.alexa import ssf_core as s  # noqa: E402
from training.synth import layout as ly  # noqa: E402
from training.synth import make_scenes as ms  # noqa: E402
from training.synth import preview as pv  # noqa: E402
from training.synth import recipes as rc  # noqa: E402
from training.synth import sensor as sn  # noqa: E402
from training.synth import spectra as sp  # noqa: E402
from training.synth import truth as tr  # noqa: E402

A7R4 = sp.BODIES["ILCE-7RM4"]
GROUPS_WITHOUT_OVERLAP = {"chart", "neutral", "ladder", "shadow", "black", "highlight"}


class Constants(unittest.TestCase):
    """The ARRI steps are those of docs/02 (the numbers printed there)."""

    def test_gamut_matrices_match_the_spec(self):
        g = tr.gamut_matrices()
        awg3_to_awg4 = np.array([
            [0.8892557469, 0.1288981523, -0.0181538992],
            [0.0840832627, 1.0093919566, -0.0934752193],
            [0.0025694496, -0.0615525078, 1.0589830582],
        ])
        np.testing.assert_allclose(g["awg3_to_awg4"], awg3_to_awg4, atol=5e-10)
        awg3_to_ap0 = np.array([
            [0.680205505, 0.236136602, 0.083657893],
            [0.085414980, 1.017470879, -0.102885858],
            [0.002056522, -0.062562500, 1.060505979],
        ])
        np.testing.assert_allclose(g["awg3_to_ap0"], awg3_to_ap0, atol=5e-10)
        np.testing.assert_allclose(g["awg3_to_awg4"].sum(axis=1), 1.0, atol=1e-14)

    def test_logc4_anchor_values(self):
        self.assertAlmostEqual(float(tr.logc4_encode(0.18)), 0.2783958365, places=9)
        self.assertAlmostEqual(float(tr.logc4_encode(0.0)), 0.0928641251, places=9)
        # decode(1.0) = 469.8 (docs/02): the encoder at that point
        x = (2.0 ** (14.0 * (1.0 - tr._C) / tr._B + 6.0) - 64.0) / tr._A  # noqa: SLF001
        self.assertAlmostEqual(x, 469.8, delta=0.1)
        self.assertAlmostEqual(float(tr.logc4_encode(x)), 1.0, places=12)
        # the linear segment below the break t = -0.018057
        self.assertAlmostEqual(float(tr.logc4_encode(-0.02)), (-0.02 - tr._T) / tr._S, places=12)  # noqa: SLF001
        self.assertAlmostEqual(tr._T, -0.01805699611991131, places=15)  # noqa: SLF001

    def test_arri_matrices_keep_balanced_white(self):
        for cct in (2000, 2850, 3200, 4000, 5500, 6500, 7500, 11000, 12000):
            np.testing.assert_allclose(tr.native_to_awg3(cct).sum(axis=1), 1.0, atol=1e-12)

    def test_exposure_metadata_is_a_valid_exposure(self):
        for ev100 in (6.0, 8.0, 12.0, 14.0):
            for iso in (100, 400, 3200, 12800):
                shutter, f, ev = ms.exposure_metadata(ev100, iso)
                t = shutter[0] / shutter[1]
                self.assertGreaterEqual(t, 1 / 8000 * 0.99)
                self.assertLessEqual(t, 5.0)
                self.assertAlmostEqual(ev, ev100 + np.log2(iso / 100))
                n = f[0] / f[1]
                self.assertIn(n, (2.8, 4.0, 5.6, 8.0, 11.0, 16.0))
                # the exposure follows the meter equation within a third of a stop (shutters are rounded)
                self.assertLess(abs(np.log2(n * n / t) - ev), 0.5)


class Lights(unittest.TestCase):
    def test_the_eight_lights_and_the_mixture(self):
        lights = sp.all_lights()
        self.assertEqual(set(lights), {"bb2850", "bb3200", "D5500", "D6500", "D7500", "F11", "LEDB3", "LEDV2", "mixed"})
        self.assertEqual(lights["bb2850"].cct_k, 2850.0)
        self.assertEqual(lights["D6500"].cct_k, 6500.0)
        self.assertEqual(lights["F11"].klass, "fluorescent")
        self.assertEqual(lights["LEDB3"].klass, "led")
        self.assertEqual({k: v.klass for k, v in lights.items() if v.kind != "mixed"}["bb3200"], "tungsten")
        mixed = lights["mixed"]
        self.assertEqual(mixed.klass, "mixed")
        self.assertGreater(mixed.cct_k, 2850.0)
        self.assertLess(mixed.cct_k, 6500.0)
        self.assertEqual(sorted(p for _, p in mixed.class_probs), [0.5, 0.5])
        for light in lights.values():
            self.assertEqual(light.spd.shape, (81,))
            if light.kind != "mixed":
                self.assertAlmostEqual(float(light.spd.max()), 1.0)
        # the lights are scaled to the same luminance, and the mixture is exactly the light at the middle of the blend
        y = lambda spd: float(s.CMF[:, 1] @ spd)  # noqa: E731
        left, right = mixed.component_spds
        self.assertAlmostEqual(y(left), y(right))
        np.testing.assert_allclose(mixed.spd, 0.5 * (left + right))
        np.testing.assert_allclose(left / left.max(), lights["bb2850"].spd, atol=1e-12)  # the same shape
        self.assertEqual(mixed.components, ("bb2850", "D6500"))

    def test_body_curves_are_in_the_published_raw_scale(self):
        for body in sp.BODIES.values():
            c = sp.camera_curves(body)
            self.assertAlmostEqual(float(c[:, 1].max()), 1.0, places=9)
            self.assertLess(float(c[:, 0].max()), 0.7)  # not peak-normalised per channel
        a1 = sp.BODIES["ILCE-1"]
        self.assertFalse(a1.own_curves)
        self.assertEqual(a1.model_id, 384)
        np.testing.assert_array_equal(sp.camera_curves(a1), sp.camera_curves(A7R4))


class Layout(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scene = ly.Scene("chart", 1, sp.training_patches())
        cls.light = sp.single_lights()["D6500"]
        spectra = cls.scene.bank.spectra
        cls.cam = tr.responses(spectra, cls.light.spd, sp.camera_curves(A7R4))
        cls.irr = cls.scene.irradiance()
        cls.canvas = ly.apply_psf(cls.scene.render_refl(cls.cam) * cls.irr[..., None])

    def test_size_and_counts(self):
        sc = self.scene
        self.assertEqual((sc.W, sc.H), (1248, 832))
        self.assertEqual(sc.W % 32, 0)
        groups = {}
        for p in sc.probes:
            groups[p.group] = groups.get(p.group, 0) + 1
        self.assertEqual(groups["chart"], 190)
        self.assertEqual(groups["neutral"], 24)
        self.assertEqual(groups["ladder"], 6 * len(ly.LADDER_EV))
        self.assertEqual(groups["shadow"], 16)
        self.assertEqual(groups["highlight"], 2)
        kinds = {f["kind"] for f in sc.features}
        for kind in ("chart", "gradient", "gradient-2d", "zone-plate", "slanted-edge", "grating", "highlight", "shadow-zone", "rolloff-ramp"):
            self.assertIn(kind, kinds)
        roles = {p.role: p.name for p in sc.probes if p.role}
        self.assertEqual(roles, {"grey_reference": "neutral-0.18", "white_reference": "neutral-0.9"})

    def test_the_evaluation_scene_holds_the_cie_samples(self):
        patches = sp.evaluation_patches()
        self.assertEqual(len(patches), 99 + 24 + 15)
        sc = ly.Scene("eval", 1, patches, evaluation=True)
        chart = [p for p in sc.probes if p.group == "chart"]
        self.assertEqual(len(chart), 138)
        self.assertEqual({p.name.split("-")[0] for p in chart}, {"ces", "cc", "vs"})
        self.assertEqual(sum(p.name.startswith("ces-") for p in chart), 99)
        self.assertFalse(any(p.name.startswith("rta-") for p in chart))  # no rawtoaces patch in the CIE scene
        # the rest of the scene is the same
        self.assertEqual(sum(p.group == "neutral" for p in sc.probes), 24)
        self.assertEqual(sc.W, 1248)

    def test_full_size_scene_is_the_same_composition_five_times_larger(self):
        big = ly.Scene("fullsize", 5, sp.training_patches())
        self.assertEqual((big.W, big.H), (6240, 4160))
        self.assertEqual(big.W % 32, 0)
        self.assertEqual(len(big.probes), len(self.scene.probes))
        for a, b in zip(self.scene.probes, big.probes):
            self.assertEqual(a.name, b.name)
            if a.group in GROUPS_WITHOUT_OVERLAP:  # the probes on gradients are a fixed number of pixels wide
                self.assertEqual(tuple(5 * v for v in a.rect), b.rect, a.name)
        zp = next(f for f in big.features if f["kind"] == "zone-plate")
        self.assertEqual(zp["max_frequency_cycles_per_px"], 0.5)  # the frequency content does not scale
        self.assertEqual(zp["radius_px"], 5 * 170)

    def test_probes_are_inside_the_frame_and_flat_patches_do_not_overlap(self):
        sc = self.scene
        for p in sc.probes:
            for r in (p.rect, p.interior):
                x, y, w, h = r
                self.assertTrue(0 <= x and 0 <= y and x + w <= sc.W and y + h <= sc.H, (p.name, r))
            x, y, w, h = p.rect
            ix, iy, iw, ih = p.interior
            self.assertTrue(x <= ix and y <= iy and ix + iw <= x + w and iy + ih <= y + h, p.name)
        flat = [p for p in sc.probes if p.group in GROUPS_WITHOUT_OVERLAP]
        for i, a in enumerate(flat):
            for b in flat[i + 1 :]:
                ax, ay, aw, ah = a.rect
                bx, by, bw, bh = b.rect
                disjoint = ax + aw <= bx or bx + bw <= ax or ay + ah <= by or by + bh <= ay
                self.assertTrue(disjoint, (a.name, b.name))

    def test_features_do_not_overlap_each_other(self):
        rects = [(f["name"] if "name" in f else f["kind"], f["rect"]) for f in self.scene.features if f["kind"] != "chart"]
        rects.append(("chart", next(f["rect"] for f in self.scene.features if f["kind"] == "chart")))
        for i, (na, a) in enumerate(rects):
            for nb, b in rects[i + 1 :]:
                ax, ay, aw, ah = a
                bx, by, bw, bh = b
                disjoint = ax + aw <= bx or bx + bw <= ax or ay + ah <= by or by + bh <= ay
                self.assertTrue(disjoint, (na, nb, a, b))

    def test_flat_patch_interiors_render_the_exact_response(self):
        sc = self.scene
        checked = 0
        for p in sc.probes:
            if p.group not in ("chart", "neutral", "ladder", "shadow", "black"):
                continue
            x, y, w, h = p.interior
            got = self.canvas[y : y + h, x : x + w].reshape(-1, 3).mean(axis=0)
            want = p.m * tr.mixture_response(self.cam, p.mixture)
            np.testing.assert_allclose(got, want, rtol=2e-4, atol=1e-6, err_msg=p.name)
            checked += 1
        self.assertEqual(checked, 190 + 24 + 48 + 16 + 1)

    def test_gradient_and_ramp_probes_render_their_analytic_value(self):
        for p in self.scene.probes:
            if p.group not in ("gradient", "rolloff"):
                continue
            x, y, w, h = p.interior
            got = self.canvas[y : y + h, x : x + w].reshape(-1, 3).mean(axis=0)
            want = p.m * tr.mixture_response(self.cam, p.mixture)
            np.testing.assert_allclose(got, want, rtol=2.5e-2, atol=1e-4, err_msg=p.name)

    def test_the_highlight_is_far_above_clipping_and_the_shadow_far_below(self):
        sc = self.scene
        lamp = next(p for p in sc.probes if p.name == "lamp-white")
        grey = next(p for p in sc.probes if p.role == "grey_reference")
        k = sn.exposure_scale(tr.mixture_response(self.cam, grey.mixture)[1], sn.MODE_14)
        lamp_dn = k * lamp.m * tr.mixture_response(self.cam, lamp.mixture)
        self.assertTrue((lamp_dn > sn.MODE_14.span).all(), lamp_dn)
        self.assertAlmostEqual(k * tr.mixture_response(self.cam, grey.mixture)[1], sn.GREY_FRACTION * sn.MODE_14.span, places=6)
        shadow = next(p for p in sc.probes if p.name == "shadow-neutral-0.18")
        self.assertAlmostEqual(shadow.m, 2.0**ly.SHADOW_EV)
        dn = k * shadow.m * tr.mixture_response(self.cam, shadow.mixture)
        self.assertLess(dn[1], 0.01 * sn.MODE_14.span)
        self.assertGreater(dn[1], 0.0)

    def test_zone_plate_reaches_nyquist_at_its_edge(self):
        zp = next(f for f in self.scene.features if f["kind"] == "zone-plate")
        x, y, w, h = zp["rect"]
        cx, cy, radius = x + w / 2.0, y + h / 2.0, zp["radius_px"]
        # local frequency along the horizontal axis through the centre: d(phase)/dx / 2pi = 0.5 * r / radius
        r = np.array([10.0, 85.0, 170.0])
        freq = 0.5 * r / radius
        np.testing.assert_allclose(freq, [0.5 * 10 / 170, 0.25, 0.5])
        # the rendered plate is not flat and stays inside the two neutral reflectances
        sub = self.canvas[y : y + h, x : x + w, 1]
        lo = self.cam[self.scene.bank["neutral-0.04"], 1]
        hi = self.cam[self.scene.bank["neutral-0.9"], 1]
        self.assertGreater(sub.std(), 0.1 * (hi - lo))
        self.assertGreaterEqual(float(sub.min()), 0.99 * lo)
        self.assertLessEqual(float(sub.max()), 1.01 * hi)

    def test_slanted_edge_has_the_stated_angle_and_edge_spread(self):
        f = next(f for f in self.scene.features if f.get("name") == "edge-neutral")
        x, y, w, h = f["rect"]
        img = self.canvas[y : y + h, x : x + w, 1].astype(np.float64)
        lo, hi = img.min(), img.max()
        norm = (img - lo) / (hi - lo)
        # subpixel edge position per row: sum of the normalised row
        pos = norm.shape[1] - norm.sum(axis=1)  # width of the dark part in pixels
        slope = np.polyfit(np.arange(h)[10:-10], pos[10:-10], 1)[0]
        # the edge moves by -tan(angle) pixels per row (d > 0 is the light side)
        self.assertAlmostEqual(slope, np.tan(np.radians(f["angle_deg"])), delta=0.01)
        # the edge is blurred over a couple of pixels, not a hard step
        row = norm[h // 2]
        transition = int(((row > 0.02) & (row < 0.98)).sum())
        self.assertTrue(2 <= transition <= 5, transition)

    def test_mixed_scene_blends_two_renders(self):
        sc = self.scene
        w = sc.mix_weight()
        self.assertEqual(w[0], 0.0)
        self.assertEqual(w[-1], 1.0)
        self.assertTrue((np.diff(w) >= 0).all())
        self.assertAlmostEqual(float(w[sc.W // 2]), 0.5, delta=2e-3)
        self.assertTrue((w[: int(0.3 * sc.W)] == 0).all() and (w[int(0.7 * sc.W) + 1 :] == 1).all())
        ctx = ms.Context()
        columns = {}
        for token in ("bb2850", "D6500", "mixed"):
            r = ctx.render("chart", token, "ILCE-7RM4")
            columns[token] = (r.canvas[:, 3:30].copy(), r.canvas[:, -30:-3].copy())
        # the left edge of the mixed scene is lit by the tungsten light alone, the right edge by daylight alone, each
        # scaled to the same luminance (a constant factor against the single-light renders)
        left_spd, right_spd = ctx.lights["mixed"].component_spds
        f_left = float((s.CMF[:, 1] @ left_spd) / (s.CMF[:, 1] @ ctx.lights["bb2850"].spd))
        f_right = float((s.CMF[:, 1] @ right_spd) / (s.CMF[:, 1] @ ctx.lights["D6500"].spd))
        np.testing.assert_allclose(columns["mixed"][0], f_left * columns["bb2850"][0], rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(columns["mixed"][1], f_right * columns["D6500"][1], rtol=1e-5, atol=1e-6)
        # a neutral patch in the middle sees the mixture of both
        mid = ctx.render("chart", "mixed", "ILCE-7RM4")
        probe = next(p for p in mid.scene.probes if p.name == "neutral-0.9")
        s_weight = mid.probe_s[mid.scene.probes.index(probe)]
        self.assertTrue(0.0 < s_weight < 1.0 or s_weight in (0.0, 1.0))


class Sensor(unittest.TestCase):
    def test_noise_model_follows_iso_and_regime(self):
        a100 = sn.noise_model(A7R4, 100, sn.MODE_14)
        a200 = sn.noise_model(A7R4, 200, sn.MODE_14)
        self.assertAlmostEqual(a200.a_dn / a100.a_dn, 2.0)
        self.assertEqual(a100.regime, "low gain")
        self.assertEqual(sn.noise_model(A7R4, 320, sn.MODE_14).regime, "high gain")
        # the read noise in electrons drops at the switch ISO
        self.assertGreater(a100.read_noise_e, sn.noise_model(A7R4, 400, sn.MODE_14).read_noise_e)
        # full well at base ISO from the assumed density and the pixel pitch
        self.assertAlmostEqual(a100.full_well_e, sp.FULL_WELL_E_PER_UM2 * A7R4.pitch_um**2, delta=1e-6 * a100.full_well_e)
        single = sp.BODIES["DSC-RX100M4"]
        self.assertEqual(sn.noise_model(single, 3200, sn.MODE_14).regime, "single gain")
        # below the base ISO the gain stays at the ISO 100 value
        self.assertEqual(sn.noise_model(A7R4, 50, sn.MODE_14).a_dn, a100.a_dn)
        nm = sn.noise_model(A7R4, 800, sn.MODE_14)
        self.assertAlmostEqual(nm.a_norm, nm.a_dn / sn.MODE_14.span)
        self.assertAlmostEqual(nm.b_norm, nm.b_dn2 / sn.MODE_14.span**2)

    def test_samples_follow_the_poisson_gaussian_model(self):
        for iso in (100, 800, 6400):
            nm = sn.noise_model(A7R4, iso, sn.MODE_14)
            for level in (0.02, 0.3, 0.7):
                x = level * sn.MODE_14.span
                clean = np.full((500, 500), x, dtype=np.float32)
                out = sn.sample(clean, sn.MODE_14, nm, np.random.default_rng(11)).astype(np.float64)
                self.assertAlmostEqual((out.mean() - 512) / x, 1.0, delta=0.01 + 4 * np.sqrt(nm.a_dn * x + nm.b_dn2) / x / 500)
                model = nm.a_dn * x + nm.b_dn2 + 1 / 12
                clipped = (out >= sn.MODE_14.white).mean()
                if clipped < 1e-4:
                    self.assertAlmostEqual(out.var() / model, 1.0, delta=0.05, msg=f"ISO {iso} level {level}")

    def test_a_pixel_above_white_clips_and_below_black_never_goes_negative(self):
        nm = sn.noise_model(A7R4, 400, sn.MODE_14)
        bright = np.full((64, 64), 40.0 * sn.MODE_14.span, dtype=np.float32)
        out = sn.sample(bright, sn.MODE_14, nm, np.random.default_rng(1))
        self.assertTrue((out == sn.MODE_14.white).all())
        dark = np.zeros((64, 64), dtype=np.float32)
        out = sn.sample(dark, sn.MODE_14, nm, np.random.default_rng(1))
        self.assertGreaterEqual(int(out.min()), 0)
        self.assertLess(abs(float(out.mean()) - 512), 1.0)
        self.assertEqual(out.dtype, np.uint16)

    def test_sampling_is_deterministic_and_seeded(self):
        nm = sn.noise_model(A7R4, 400, sn.MODE_14)
        clean = np.linspace(0, 8000, 4096, dtype=np.float32).reshape(64, 64)
        a = sn.sample(clean, sn.MODE_14, nm, np.random.default_rng(5))
        b = sn.sample(clean, sn.MODE_14, nm, np.random.default_rng(5))
        c = sn.sample(clean, sn.MODE_14, nm, np.random.default_rng(6))
        np.testing.assert_array_equal(a, b)
        self.assertFalse(np.array_equal(a, c))
        self.assertEqual(ms.seed_of("x", "noise"), ms.seed_of("x", "noise"))
        self.assertNotEqual(ms.seed_of("x", "noise"), ms.seed_of("y", "noise"))
        self.assertNotEqual(ms.seed_of("x", "noise"), ms.seed_of("x", "awb"))

    def test_twelve_bit_levels_are_the_shifted_fourteen_bit_levels(self):
        self.assertEqual(sn.MODE_14.black >> sn.MODE_12.shift, sn.MODE_12.black)
        self.assertEqual(sn.MODE_14.white >> sn.MODE_12.shift, sn.MODE_12.white)
        nm14, nm12 = sn.noise_model(A7R4, 800, sn.MODE_14), sn.noise_model(A7R4, 800, sn.MODE_12)
        self.assertAlmostEqual(nm12.a_dn / nm14.a_dn, sn.MODE_12.span / sn.MODE_14.span)

    def test_cfa_mosaic_picks_the_right_channel_for_every_phase(self):
        canvas = np.zeros((4, 4, 3), dtype=np.float32)
        canvas[..., 0], canvas[..., 1], canvas[..., 2] = 1.0, 2.0, 3.0
        expect = {
            "RGGB": [[1, 2], [2, 3]], "GRBG": [[2, 1], [3, 2]], "GBRG": [[2, 3], [1, 2]], "BGGR": [[3, 2], [2, 1]],
        }
        for cfa, cell in expect.items():
            m = sn.cfa_mosaic(canvas, cfa)
            np.testing.assert_array_equal(m[:2, :2], np.array(cell, dtype=np.float32), err_msg=cfa)
            np.testing.assert_array_equal(m, np.tile(np.array(cell, dtype=np.float32), (2, 2)), err_msg=cfa)


class Truth(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scene = ly.Scene("chart", 1, sp.training_patches())
        cls.lights = sp.all_lights()

    def truth(self, token="D6500", body=A7R4):
        light = self.lights[token]
        lr = light.component_spds if light.kind == "mixed" else None
        return tr.LightTruth(light, body, self.scene.bank.spectra, left_right=lr)

    def test_neutral_patches_are_neutral_in_both_references(self):
        for token in ("bb2850", "D6500", "F11", "LEDV2"):
            lt = self.truth(token)
            for p in self.scene.probes:
                if p.group != "neutral":
                    continue
                rho = float(p.name.split("-")[1])
                v = lt.probe_values(p.mixture, p.m)
                for ref in ("alexa", "sony_through_arri"):
                    np.testing.assert_allclose(v[ref]["awg4"], [rho] * 3, atol=1e-12, err_msg=f"{token} {p.name} {ref}")
                    np.testing.assert_allclose(v[ref]["native_balanced"], [rho] * 3, atol=1e-12)
        grey = next(p for p in self.scene.probes if p.role == "grey_reference")
        v = self.truth().probe_values(grey.mixture, 1.0)
        np.testing.assert_allclose(v["alexa"]["logc4"], [0.2783958365] * 3, atol=1e-9)

    def test_light_truth_is_the_camera_white(self):
        lt = self.truth("D5500")
        light = self.lights["D5500"]
        w = light.spd @ sp.camera_curves(A7R4)
        np.testing.assert_allclose(lt.w_true, w / w[1], atol=1e-14)
        self.assertEqual(lt.w_true[1], 1.0)
        # the raw white of a Sony at daylight: red and blue well below green (the published scale)
        self.assertLess(lt.w_true[0], 0.6)
        self.assertLess(lt.w_true[2], 0.9)
        # the sensor white of a flat-white patch is the same thing times the light
        one = self.scene.bank["neutral-1"]
        cam = lt.probe_values([(one, 1.0)], 1.0)["camera_response"]
        np.testing.assert_allclose(np.array(cam) / cam[1], lt.w_true, atol=1e-14)

    def test_the_reference_is_the_spectral_computation_of_the_spec(self):
        """Independent recomputation of one chart patch with ssf_core, following docs/02 C3-C6 step by step."""
        token, patch_index = "bb3200", 17
        lt = self.truth(token)
        light = self.lights[token]
        p = next(p for p in self.scene.probes if p.name == f"rta-{patch_index + 1:03d}")
        got = lt.probe_values(p.mixture, p.m)["alexa"]
        r = s.training_reflectances()[:, patch_index : patch_index + 1]
        s_a = sp.alexa_curves()
        bal = s.camera_rgb_wb(s_a, light.spd, np.clip(r, 0, None))[0]  # white-balanced ALEXA native RGB
        ccts, mats = s.load_arri_matrices()
        m_ap0 = s.interpolate_matrix(ccts, mats, light.cct_k)
        ap0 = m_ap0 @ bal
        g = tr.gamut_matrices()
        awg3 = np.linalg.inv(g["awg3_to_ap0"]) @ ap0
        awg4 = g["awg3_to_awg4"] @ awg3
        np.testing.assert_allclose(got["awg4"], awg4, atol=1e-12)
        np.testing.assert_allclose(got["awg3"], awg3, atol=1e-12)
        np.testing.assert_allclose(got["logc4"], tr.logc4_encode(awg4), atol=1e-12)

    def test_baseline_equals_the_reference_for_a_camera_with_the_alexa_curves(self):
        lt = self.truth("D6500")
        lt.S_p = lt.S_a  # a "Sony" that is an ALEXA
        lt._tab_p = lt._tab_a  # noqa: SLF001
        lt.white_p = lt.white_a
        for p in self.scene.probes[:60]:
            v = lt.probe_values(p.mixture, p.m)
            np.testing.assert_allclose(v["alexa"]["awg4"], v["sony_through_arri"]["awg4"], atol=1e-12)

    def test_a_real_sony_differs_from_the_alexa_on_saturated_patches(self):
        lt = self.truth("D6500")
        diffs = []
        for p in self.scene.probes:
            if p.group != "chart":
                continue
            v = lt.probe_values(p.mixture, p.m)
            diffs.append(np.abs(np.array(v["alexa"]["awg4"]) - np.array(v["sony_through_arri"]["awg4"])).max())
        self.assertGreater(float(np.max(diffs)), 0.02)  # the sensor map has something to do
        self.assertGreater(float(np.median(diffs)), 0.001)

    def test_exposure_scaling_of_highlights_is_linear(self):
        lt = self.truth("D6500")
        p = next(p for p in self.scene.probes if p.name == "lamp-white")
        a = lt.probe_values(p.mixture, 1.0)["alexa"]["awg4"]
        b = lt.probe_values(p.mixture, p.m)["alexa"]["awg4"]
        np.testing.assert_allclose(np.array(b), p.m * np.array(a), rtol=1e-12)
        self.assertGreater(b[1], 30.0)  # the unclipped truth

    def test_mixed_light_probes_know_their_local_white(self):
        lt = self.truth("mixed")
        for p in self.scene.probes:
            if p.group != "neutral":
                continue
            rho = float(p.name.split("-")[1])
            for s_w in (0.0, 0.5, 1.0):
                v = lt.probe_values(p.mixture, p.m, s_w)
                # the local white balance always makes a neutral surface neutral, the global one does not
                np.testing.assert_allclose(v["alexa_local_white"]["native_balanced"], [rho] * 3, atol=1e-12)
            v0 = lt.probe_values(p.mixture, p.m, 0.0)["alexa"]["native_balanced"]
            self.assertGreater(v0[0] - v0[2], 0.01 * rho)  # the tungsten end under the mixture's white balance: reddish
            v1 = lt.probe_values(p.mixture, p.m, 1.0)["alexa"]["native_balanced"]
            self.assertGreater(v1[2] - v1[0], 0.01 * rho)  # the daylight end: bluish
            vh = lt.probe_values(p.mixture, p.m, 0.5)["alexa"]["native_balanced"]
            np.testing.assert_allclose(vh, [rho] * 3, atol=1e-9)  # at the middle the local light is the reference light


class IndependentRecomputation(unittest.TestCase):
    """The ground truth in the sidecars and the mosaics, recomputed by code that shares nothing with ``training.synth``.

    ``test_the_reference_is_the_spectral_computation_of_the_spec`` repeats the computation with ``ssf_core``, the code the
    generator itself uses, so a defect in a shared step (the mired interpolation, the gamut matrices, LogC4, the
    exposure scale) would pass it. Here only data files are shared: the published Sony and ALEXA curves, the rawtoaces
    training reflectances, the CIE D basis functions and the CMFs (all read straight from their files or from
    colour-science's tables), and ARRI's matrices as the *runtime* table ``data/arri/alexa_native_to_awg3_17cct.json``
    (native to AWG3, already inverted), whose interpolation, the AWG3 to AWG4 matrix, LogC4, the Planckian radiator, the
    D-series formula, the CCT and the exposure are written again below.
    """

    WL = np.arange(380, 781, 5).astype(float)

    @classmethod
    def setUpClass(cls):
        import colour

        cls.colour = colour
        root = paths.REPO_ROOT
        cls.S_A = cls._curves(root / "data" / "alexa" / "alexa_alev3_estimate_380_780_5.json", clip=False)
        cls.S_A /= cls.S_A.max(axis=0)
        tm = json.loads(paths.TRAINING_REFLECTANCES_JSON.read_text())["spectral_data"]["data"]["main"]
        cls.train = np.array([tm[k] for k in sorted(tm, key=float)], dtype=float)  # (81, 190)
        arri = json.loads((root / "data" / "arri" / "alexa_native_to_awg3_17cct.json").read_text())
        cls.arri_cct = np.array([m["cct"] for m in arri["matrices"]], dtype=float)
        cls.arri = np.array([m["native_to_awg3"] for m in arri["matrices"]], dtype=float)
        cmfs = colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]
        cls.cmf = np.stack([np.interp(cls.WL, cmfs.wavelengths, cmfs.values[:, k]) for k in range(3)], axis=1)
        prim3 = [[0.6840, 0.3130], [0.2210, 0.8480], [0.0861, -0.1020]]
        prim4 = [[0.7347, 0.2653], [0.1424, 0.8576], [0.0991, -0.0308]]
        cls.awg3_to_awg4 = np.linalg.inv(cls._npm(prim4)) @ cls._npm(prim3)

    # ---------------------------------------------------------------- independent building blocks
    @classmethod
    def _curves(cls, path, clip=True):
        doc = json.loads(Path(path).read_text())["spectral_data"]["data"]["main"]
        arr = np.array([doc[str(int(w))] for w in cls.WL], dtype=float)
        return np.clip(arr, 0.0, None) if clip else arr

    @classmethod
    def _sony(cls, curves_key):
        """Published raw scale: clipped at 0, one common factor, the green peak at exactly 1."""
        model = curves_key.removeprefix("Sony ")
        arr = cls._curves(paths.SONY_CURVES_DIR / f"Sony_{model}_380_780_5.json")
        return arr / arr[:, 1].max()

    @staticmethod
    def _npm(primaries, white=(0.3127, 0.3290)):
        xyz = np.array([[x / y, 1.0, (1.0 - x - y) / y] for x, y in primaries]).T
        w = np.array([white[0] / white[1], 1.0, (1.0 - white[0] - white[1]) / white[1]])
        return xyz * np.linalg.solve(xyz, w)

    def _planck(self, t):
        lam = self.WL * 1e-9
        v = lam**-5 / np.expm1(1.4388e-2 / (lam * t))
        return v / v.max()

    def _daylight(self, cct):
        from colour.colorimetry import SDS_BASIS_FUNCTIONS_CIE_ILLUMINANT_D_SERIES as basis

        t = cct * 1.4388 / 1.4380  # the rawtoaces CCT correction
        if t <= 7000:
            x = -4.6070e9 / t**3 + 2.9678e6 / t**2 + 0.09911e3 / t + 0.244063
        else:
            x = -2.0064e9 / t**3 + 1.9018e6 / t**2 + 0.24748e3 / t + 0.237040
        y = -3.0 * x * x + 2.870 * x - 0.275
        den = 0.0241 + 0.2562 * x - 0.7341 * y
        m1 = round((-1.3515 - 1.7703 * x + 5.9114 * y) / den, 3)  # CIE rounds M1 and M2 to three decimals
        m2 = round((0.0300 - 31.4424 * x + 30.0717 * y) / den, 3)
        s0, s1, s2 = (np.array([basis[k][w] for w in self.WL]) for k in ("S0", "S1", "S2"))
        v = s0 + m1 * s1 + m2 * s2
        return v / v.max()

    def _named(self, name):
        sd = self.colour.SDS_ILLUMINANTS[name] if name in self.colour.SDS_ILLUMINANTS else self.colour.SDS_LIGHT_SOURCES[name]
        v = np.clip(np.interp(self.WL, np.array(sd.wavelengths, dtype=float), np.array(sd.values, dtype=float)), 0.0, None)
        return v / v.max()

    def _spd(self, token):
        if token.startswith("bb"):
            return self._planck(float(token[2:]))
        if token.startswith("D"):
            return self._daylight(float(token[1:]))
        return self._named({"F11": "FL11", "LEDB3": "LED-B3", "LEDV2": "LED-V2"}[token])

    def _cct_by_search(self, spd):
        """CCT as the nearest point of the Planckian locus in CIE 1960 uv (1 K steps): not Ohno's method."""
        def uv(e):
            x, y, z = self.cmf.T @ e
            return 4 * x / (x + 15 * y + 3 * z), 6 * y / (x + 15 * y + 3 * z)

        u, v = uv(spd)
        temps = np.arange(1500.0, 20000.0, 1.0)
        pts = np.array([uv(self._planck(t)) for t in temps])
        return float(temps[np.argmin((pts[:, 0] - u) ** 2 + (pts[:, 1] - v) ** 2)])

    def _native_to_awg3(self, cct):
        """Mired interpolation between the bracketing entries, clamped to the 2000 to 11000 K table."""
        cct = min(max(cct, self.arri_cct[0]), self.arri_cct[-1])
        i = min(max(int(np.searchsorted(self.arri_cct, cct, side="right")) - 1, 0), len(self.arri_cct) - 2)
        m0, m1, m = 1e6 / self.arri_cct[i], 1e6 / self.arri_cct[i + 1], 1e6 / cct
        w = (m - m0) / (m1 - m0)
        return (1 - w) * self.arri[i] + w * self.arri[i + 1]

    @staticmethod
    def _logc4(x):
        a, b, c = (2.0**18 - 16.0) / 117.45, (1023.0 - 95.0) / 1023.0, 95.0 / 1023.0
        slope = 7.0 * np.log(2.0) * 2.0 ** (7.0 - 14.0 * c / b) / (a * b)
        t = (2.0 ** (14.0 * (-c / b) + 6.0) - 64.0) / a
        x = np.asarray(x, dtype=float)
        return np.where(x >= t, (np.log2(np.maximum(a * x + 64.0, 1e-300)) - 6.0) / 14.0 * b + c, (x - t) / slope)

    def _reflectance(self, name):
        if name.startswith("neutral-"):
            return np.full(self.WL.size, float(name.removeprefix("neutral-")))
        if name.startswith("rta-"):
            return self.train[:, int(name[4:]) - 1]
        raise KeyError(name)

    # ---------------------------------------------------------------- the checks
    def _generate(self, light, body, iso=400, bits=14):
        recipe = rc.Recipe("chart", light, body, iso, bits=bits, variants=("packed12",) if bits == 12 else ("lossless-l",), group="test")
        with tempfile.TemporaryDirectory() as d:
            ms.generate(recipe, 0, Path(d), ms.Context(), force=False)
            doc = json.loads((Path(d) / f"{recipe.stem}.json").read_text())
            mosaic = np.fromfile(Path(d) / f"{recipe.stem}.bin", dtype="<u2")
        return doc, mosaic.reshape(doc["mosaic"]["height"], doc["mosaic"]["width"]).astype(float)

    def _check(self, light, body, iso=400, bits=14):
        doc, mosaic = self._generate(light, body, iso, bits)
        truth = doc["truth"]
        e = self._spd(light)
        s_p = self._sony(truth["light"]["camera_profile"])
        white_p, white_a = e @ s_p, e @ self.S_A
        cct = self._cct_by_search(e) if light.startswith(("F", "LED")) else float(light[2:] if light.startswith("bb") else light[1:])
        # the light: the camera white in the raw scale, the ALEXA native white, the CCT the ARRI matrix is looked up at
        np.testing.assert_allclose(truth["light"]["w_true_camera_rgb"], white_p / white_p[1], rtol=2e-8)
        np.testing.assert_allclose(truth["light"]["w_alexa_native"], white_a / white_a[1], rtol=2e-8)
        self.assertAlmostEqual(truth["arri"]["cct_k"], cct, delta=1.0)  # Ohno 2013 against a locus search: within 1 K
        # exposure: the green of the 18 % grey at 0.12 of white - black, whatever the ISO
        span = doc["mosaic"]["white"] - doc["mosaic"]["black"]
        k = 0.12 * span / (e * 0.18 @ s_p)[1]
        self.assertAlmostEqual(truth["exposure"]["dn_per_response_unit"] / k, 1.0, places=8)
        m_arri = self._native_to_awg3(truth["arri"]["cct_k"])
        worst = {"awg4": 0.0, "logc4": 0.0, "sony": 0.0, "dn": 0.0}
        for p in truth["probes"]:
            refl = sum(wt * self._reflectance(n) for n, wt in p["reflectance"])
            m = p["irradiance"]
            ra = m * (e * refl @ self.S_A) / white_a
            rp = m * (e * refl @ s_p)
            awg4 = self.awg3_to_awg4 @ (m_arri @ ra)
            sony = self.awg3_to_awg4 @ (m_arri @ (rp / white_p))
            scale = max(np.abs(awg4).max(), 1e-9)
            worst["awg4"] = max(worst["awg4"], np.abs(awg4 - p["alexa"]["awg4"]).max() / scale)
            worst["logc4"] = max(worst["logc4"], np.abs(self._logc4(awg4) - p["alexa"]["logc4"]).max())
            worst["sony"] = max(worst["sony"], np.abs(sony - p["sony_through_arri"]["awg4"]).max() / max(np.abs(sony).max(), 1e-9))
            worst["dn"] = max(worst["dn"], np.abs(k * rp - p["camera_dn_above_black"]).max() / max(np.abs(k * rp).max(), 1e-9))
        # the sidecar keeps nine significant digits
        self.assertLess(worst["awg4"], 2e-7, worst)
        self.assertLess(worst["logc4"], 2e-8, worst)
        self.assertLess(worst["sony"], 2e-7, worst)
        self.assertLess(worst["dn"], 2e-7, worst)
        # and the mosaic holds the camera DN of every flat patch, with the noise of the stated model (z-scores of the mean)
        nm = truth["noise"]
        chan = np.empty(mosaic.shape, dtype=int)
        for dy in (0, 1):
            for dx in (0, 1):
                chan[dy::2, dx::2] = ((0, 1), (1, 2))[dy][dx]
        zs, ratios = [], []
        for p in truth["probes"]:
            if p["group"] not in ("chart", "neutral", "ladder", "gradient", "rolloff"):
                continue
            x, y, w, h = p["interior"]
            sub, ch = mosaic[y : y + h, x : x + w] - doc["mosaic"]["black"], chan[y : y + h, x : x + w]
            for c in range(3):
                expect = p["camera_dn_above_black"][c]
                v = sub[ch == c]
                if expect > 0.8 * span or expect < 20 or v.max() + doc["mosaic"]["black"] >= doc["mosaic"]["white"]:
                    continue
                sd = np.sqrt(nm["a_dn_per_electron"] * expect + nm["b_dn2"] + 1 / 12)
                zs.append((v.mean() - expect) / (sd / np.sqrt(v.size)))
                if p["group"] in ("chart", "neutral", "ladder"):  # flat: a ramp's slope adds variance of its own
                    ratios.append(v.var(ddof=1) / sd**2)
        zs = np.array(zs)
        self.assertGreater(zs.size, 300)
        self.assertLess(abs(zs.mean()), 0.25, "the mosaic is unbiased")
        self.assertTrue(0.8 < np.sqrt((zs**2).mean()) < 1.25, np.sqrt((zs**2).mean()))
        self.assertLess(np.abs(zs).max(), 5.5)
        self.assertTrue(0.9 < float(np.mean(ratios)) < 1.1, float(np.mean(ratios)))  # the variance follows a * x + b + 1/12

    def test_daylight_on_the_a7r_iv(self):
        self._check("D6500", "ILCE-7RM4")

    def test_a_blackbody_at_a_high_iso_on_another_body(self):
        self._check("bb2850", "ILCE-7M3", iso=3200)

    def test_a_fluorescent_light_whose_cct_is_not_a_generating_temperature(self):
        self._check("F11", "ILCE-6400")

    def test_a_twelve_bit_mosaic_has_its_own_levels_and_gain(self):
        self._check("D5500", "ILCE-7SM3", bits=12)


class Recipes(unittest.TestCase):
    def test_default_set_covers_the_required_axes(self):
        rs = rc.default_recipes()
        self.assertEqual(len(rs), 37)
        self.assertEqual(sum(len(r.variants) for r in rs), 61)
        self.assertEqual(len({r.stem for r in rs}), len(rs))
        self.assertEqual({r.light for r in rs}, {"bb2850", "bb3200", "D5500", "D6500", "D7500", "F11", "LEDB3", "LEDV2", "mixed"})
        self.assertEqual({r.body for r in rs}, {"ILCE-7RM4", "ILCE-7M3", "ILCE-7SM3", "ILCE-6400", "DSC-RX100M4", "ILCE-1"})
        self.assertEqual({r.cfa for r in rs}, {"RGGB", "GRBG", "GBRG", "BGGR"})
        self.assertEqual({v for r in rs for v in r.variants}, set(rc.ALL_VARIANTS))
        self.assertEqual({r.bits for r in rs}, {12, 14})
        self.assertGreaterEqual(len({r.iso for r in rs}), 5)
        for r in rs:
            for v in r.variants:
                self.assertEqual(12 if v == "packed12" else 14, r.bits, r.stem)
            self.assertIn(r.scene, rc.SCENE_SCALE)
            self.assertEqual(r.tier, "synthetic")
        # every body gets all four 14-bit variants and a packed 12-bit file
        for body in sp.BODIES:
            variants = {v for r in rs if r.body == body and r.light == "D6500" and r.cfa == "RGGB" and r.iso == 400 for v in r.variants}
            self.assertEqual(variants, set(rc.ALL_VARIANTS), body)

    def test_full_set_adds_the_big_the_cross_product_and_the_evaluation_scenes(self):
        extra = rc.full_extra_recipes()
        self.assertEqual({r.scene for r in extra}, {"fullsize", "eval", "chart"})  # chart: the rest of the light x body product
        self.assertTrue(all(r.tier == "synthetic-eval" for r in extra if r.scene == "eval"))
        self.assertTrue(all(r.tier == "synthetic" for r in extra if r.scene == "fullsize"))
        stems = [r.stem for r in rc.recipes(True)]
        self.assertEqual(len(stems), len(set(stems)))
        for r in extra:
            if r.scene == "fullsize":
                self.assertEqual(r.body, "ILCE-7M3")

    def test_the_full_set_extends_the_default_set_without_changing_it(self):
        d, f = rc.recipes(False), rc.recipes(True)
        self.assertEqual([r.stem for r in f[: len(d)]], [r.stem for r in d])  # same files, same capture times
        for a, b in zip(d, f):
            self.assertEqual(a.variants, b.variants)
        self.assertEqual(len(f), 88)
        pairs = {(r.light, r.body) for r in f if r.scene == "chart" and r.iso == 400 and r.cfa == "RGGB" and r.bits == 14}
        self.assertEqual(len(pairs), 9 * 6)  # every light under every body
        self.assertEqual(len({r.stem for r in f}), len(f))

    def test_stems_parse_back(self):
        for r in rc.recipes(True):
            scene, light, body, iso, cfa, bits = r.stem.split("__")
            self.assertEqual((scene, light, body), (r.scene, r.light, r.body))
            self.assertEqual(int(iso.removeprefix("iso")), r.iso)
            self.assertEqual(cfa, r.cfa.lower())
            self.assertEqual(int(bits.removesuffix("b")), r.bits)

    def test_capture_times_are_distinct_and_more_than_two_seconds_apart(self):
        from datetime import datetime

        times = [datetime.strptime(rc.capture_time(i), "%Y:%m:%d %H:%M:%S") for i in range(5)]
        for a, b in zip(times, times[1:]):
            self.assertGreater((b - a).total_seconds(), 2.0)

    def test_the_committed_manifest_is_in_step_with_the_recipes(self):
        path = paths.REPO_ROOT / "tools" / "golden" / "manifest.json"
        if not path.exists():
            self.skipTest("tools/golden/manifest.json not written yet")
        self.assertTrue(ms.write_manifest(path, check=True), "run: python3 -m training.synth.make_scenes --write-manifest tools/golden/manifest.json")
        doc = json.loads(path.read_text())
        entries = [e for e in doc["entries"] if e["tier"].startswith("synthetic")]
        self.assertEqual(len(entries), len(rc.manifest(True)["entries"]))
        for e in entries:
            self.assertEqual(e["settings"], {})
            self.assertEqual((e["expected_hash"], e["reference"]), ("", ""))
            self.assertTrue(e["file"].startswith("synthetic/arw/") and e["file"].endswith(".ARW"))


    def test_the_default_set_stays_ci_sized(self):
        rs = rc.default_recipes()
        self.assertTrue(all(r.scene == "chart" for r in rs), "the full-size and evaluation scenes are --full only")
        mosaic_bytes = sum(2 * ly.VIRTUAL_W * ly.VIRTUAL_H * rc.SCENE_SCALE[r.scene] ** 2 for r in rs)
        # 37 mosaics of 2.08 MB; with the JPEGs, the sidecars and the ARW files the set is about 200 MB and builds in minutes
        self.assertLess(mosaic_bytes, 80e6)
        self.assertLessEqual(len(rs), 40)
        self.assertLessEqual(sum(len(r.variants) for r in rs), 65)

    def test_manifest_sets_name_the_files_each_run_writes(self):
        entries = rc.manifest(True)["entries"]
        by_set: dict[str, set[str]] = {}
        for e in entries:
            by_set.setdefault(e["set"], set()).add(e["file"].rsplit("/", 1)[1])
        self.assertEqual(set(by_set), {"default", "full", "eval"})
        # a golden runner that takes `set == "default"` must find exactly what a plain synthetic.sh run writes
        self.assertEqual(by_set["default"], {f for r in rc.default_recipes() for f in r.arw_names()})
        self.assertEqual(len(by_set["default"]), 61)
        self.assertEqual(len(by_set["full"]), 35)  # the full-size scene (5) and the rest of the light x body product (30)
        self.assertEqual(len(by_set["eval"]), 18)
        self.assertTrue(by_set["default"].isdisjoint(by_set["full"] | by_set["eval"]))
        self.assertTrue(by_set["full"].isdisjoint(by_set["eval"]))
        self.assertEqual(by_set["default"] | by_set["full"] | by_set["eval"], {f for r in rc.recipes(True) for f in r.arw_names()})
        self.assertTrue(all((e["tier"] == "synthetic-eval") == (e["set"] == "eval") for e in entries))


class SceneIndex(unittest.TestCase):
    def test_a_limited_run_does_not_forget_the_other_scenes(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            for stem in ("a", "b", "c"):
                (out / f"{stem}.json").write_text("{}")
            old = {"scenes": [{"id": "a", "recipe_hash": "old"}, {"id": "b", "recipe_hash": "old"}, {"id": "gone", "recipe_hash": "old"}]}
            (out / "index.json").write_text(json.dumps(old))
            merged = ms.merged_index(out, [{"id": "b", "recipe_hash": "new"}, {"id": "c", "recipe_hash": "new"}])
            self.assertEqual([(e["id"], e["recipe_hash"]) for e in merged], [("a", "old"), ("b", "new"), ("c", "new")])
            # a damaged or missing index is the same as an empty one
            (out / "index.json").write_text("{not json")
            self.assertEqual([e["id"] for e in ms.merged_index(out, [{"id": "c"}])], ["c"])
            (out / "index.json").unlink()
            self.assertEqual([e["id"] for e in ms.merged_index(out, [{"id": "c"}])], ["c"])

    def test_the_mixed_light_description_has_no_float_noise(self):
        doc = json.loads(self._mixed_sidecar())
        text = doc["truth"]["light"]["mixed"]["weight_of_right_light"]
        self.assertEqual(text, "smoothstep((x / width - 0.3) / 0.4), per column")

    @staticmethod
    def _mixed_sidecar() -> str:
        recipe = rc.Recipe("chart", "mixed", "ILCE-7RM4", 400, variants=("lossless-l",), group="test")
        with tempfile.TemporaryDirectory() as d:
            ms.generate(recipe, 0, Path(d), ms.Context(), force=False)
            return (Path(d) / f"{recipe.stem}.json").read_text()


class Preview(unittest.TestCase):
    def test_jpeg_structure_and_the_synthetic_marker(self):
        from PIL import Image

        img = (np.random.default_rng(2).random((40, 56, 3)) * 255).astype(np.uint8)
        jpeg = pv.encode_jpeg(img)
        self.assertEqual(jpeg[:4], b"\xff\xd8\xff\xdb")  # like Sony previews: SOI then the quantisation table
        self.assertNotIn(b"JFIF", jpeg[:32])
        self.assertIn(b"synthetic", jpeg[:400])
        self.assertEqual(jpeg[-2:], b"\xff\xd9")
        im = Image.open(io.BytesIO(jpeg))
        im.load()
        self.assertEqual(im.size, (56, 40))
        # no APPn segment before the scan
        pos = 2
        while jpeg[pos + 1] != 0xDA:
            self.assertFalse(0xE0 <= jpeg[pos + 1] <= 0xEF)
            pos += 2 + int.from_bytes(jpeg[pos + 2 : pos + 4], "big")

    def test_the_engine_gain_is_calibrated_to_the_camera_gain_of_the_gate(self):
        # Decision 2026-10-14: the synthetic camera's JPEG engine is Sony-like, +1.11 EV (the median of the cameras'
        # own engines, `ym_color::exposure::CAMERA_GAIN_EV`), so the auto-exposure gate passes by construction.
        from scipy.optimize import brentq

        self.assertEqual(pv.TARGET_GAIN_EV, 1.11)
        self.assertAlmostEqual(pv.known_gain_ev(), pv.TARGET_GAIN_EV, delta=1e-3)
        # Calibrated, not guessed: the gain that gives exactly the target is within 2e-4 of the constant.
        exact = brentq(lambda g: pv.known_gain_ev(g) - pv.TARGET_GAIN_EV, 1.5, 3.0, xtol=1e-12)
        self.assertAlmostEqual(pv.PREVIEW_GAIN, exact, delta=2e-4)
        # known_gain_ev does not leave the module's gain changed
        self.assertEqual(pv.PREVIEW_GAIN, 2.2)
        self.assertAlmostEqual(pv.known_gain_ev(1.5), 0.5575, places=4)  # the earlier engine
        # the same constants as ym and the evaluation harness
        repo = Path(__file__).resolve().parents[2]
        rust = (repo / "crates/ym-color/src/exposure.rs").read_text()
        self.assertIn("pub const CAMERA_GAIN_EV: f64 = 1.11;", rust)
        harness = (repo / "tools/eval/autoexp.py").read_text()
        self.assertIn("SYNTH_GAIN_EV = 1.1100", harness)

    def test_the_tone_curve_reads_middle_grey_at_the_known_raw_level(self):
        # the raw level (balanced white = 1) at which the JPEG shows 0.18 in linear light
        x = 0.18 / 2.0 ** pv.known_gain_ev()
        code = float(pv.tone(np.array([x]))[0])
        lin = code / 12.92 if code <= 0.04045 else ((code + 0.055) / 1.055) ** 2.4
        self.assertAlmostEqual(lin, 0.18, places=9)
        self.assertAlmostEqual(x, 0.0834, places=4)

    def test_rendering_is_white_preserving_and_monotone(self):
        light = sp.single_lights()["D6500"]
        m = pv.camera_to_srgb_matrix(light.spd, sp.camera_curves(A7R4))
        np.testing.assert_allclose(m.sum(axis=1), 1.0, atol=1e-12)
        ramp = np.linspace(0.0, 1.5, 200, dtype=np.float32)
        raw = np.stack([ramp, ramp, ramp], axis=-1)[None]
        img = pv.render(raw, np.ones(3), np.eye(3))[0, :, 1].astype(int)
        self.assertTrue((np.diff(img) >= 0).all())
        self.assertEqual(int(img[-1]), 255)
        self.assertEqual(int(img[0]), 0)
        # the raw level at which the engine shows ISO 12232's 18 % grey (0.0834 of the white level, a Sony-like gain of
        # +1.11 EV) is sRGB code 118
        level = 0.18 / 2.0 ** pv.known_gain_ev()
        iso_grey = pv.render(np.full((1, 1, 3), level, dtype=np.float32), np.ones(3), np.eye(3))[0, 0, 0]
        self.assertTrue(117 <= int(iso_grey) <= 119, int(iso_grey))
        # the scene's grey patch (raw 0.12, half a stop above that, as the scenes are exposed) is brighter: code 135-150
        grey = pv.render(np.full((1, 1, 3), sn.GREY_FRACTION, dtype=np.float32), np.ones(3), np.eye(3))[0, 0, 0]
        self.assertTrue(135 <= int(grey) <= 150, int(grey))
        # a raw white at that exposure (0.6) is above the engine's shoulder: clipped, as a Sony JPEG clips its highlights
        white = pv.render(np.full((1, 1, 3), 0.6, dtype=np.float32), np.ones(3), np.eye(3))[0, 0, 0]
        self.assertEqual(int(white), 255)

    def test_block_mean(self):
        img = np.arange(4 * 6 * 3, dtype=np.float32).reshape(4, 6, 3)
        out = pv.block_mean(img, 2)
        self.assertEqual(out.shape, (2, 3, 3))
        self.assertAlmostEqual(float(out[0, 0, 0]), float(img[:2, :2, 0].mean()))


class EndToEnd(unittest.TestCase):
    """Generate one real recipe into a temporary folder and check everything the Rust side relies on."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.out = Path(cls.tmp.name)
        cls.recipe = rc.Recipe("chart", "D6500", "ILCE-7RM4", 800, variants=("lossless-l",), group="test")
        cls.ctx = ms.Context()
        cls.row = ms.generate(cls.recipe, 3, cls.out, cls.ctx, force=False)
        cls.doc = json.loads((cls.out / f"{cls.recipe.stem}.json").read_text())
        cls.mosaic = np.fromfile(cls.out / f"{cls.recipe.stem}.bin", dtype="<u2")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_files_and_sidecar_agree(self):
        stem = self.recipe.stem
        self.assertEqual(self.row["status"], "written")
        self.assertEqual({p.name for p in self.out.iterdir()}, {f"{stem}.bin", f"{stem}.jpg", f"{stem}.json"})
        m = self.doc["mosaic"]
        self.assertEqual(self.mosaic.size, m["width"] * m["height"])
        self.assertEqual(hashlib.sha256(self.mosaic.tobytes()).hexdigest(), m["sha256"])
        self.assertEqual((m["cfa"], m["bits"], m["black"], m["white"]), ("RGGB", 14, 512, 16383))
        self.assertLessEqual(int(self.mosaic.max()), 16383)
        self.assertTrue(self.doc["synthetic"])
        self.assertIn("Not a photograph", self.doc["notice"])
        w = self.doc["writer"]
        self.assertEqual((w["make"], w["model"], w["model_id"], w["iso"]), ("SONY", "ILCE-7RM4", 375, 800))
        self.assertEqual(w["capture_time"], rc.capture_time(3))
        self.assertTrue(w["preview"]["synthetic"])
        self.assertEqual(w["preview"]["gain"], pv.PREVIEW_GAIN)
        self.assertAlmostEqual(w["preview"]["gain_ev"], 1.11, delta=1e-3)
        self.assertEqual(w["variants"], ["lossless-l"])
        self.assertEqual(w["sr2"]["black"], [512] * 4)
        self.assertEqual(w["sr2"]["white"], [16383] * 3)
        self.assertEqual(w["sr2"]["wb_rggb"][1:3], [1024, 1024])
        self.assertTrue(self.doc["tier"] == "synthetic")

    def test_jpeg_is_embedded_ready(self):
        from PIL import Image

        jpeg = (self.out / f"{self.recipe.stem}.jpg").read_bytes()
        self.assertEqual(jpeg[:4], b"\xff\xd8\xff\xdb")
        w = self.doc["writer"]["preview"]
        self.assertEqual(len(jpeg), w["bytes"])
        im = Image.open(io.BytesIO(jpeg))
        self.assertEqual(im.size, (w["width"], w["height"]))
        self.assertGreaterEqual(max(im.size), 1024)  # ym-raw picks the smallest preview whose long edge is >= 1024

    def test_as_shot_white_balance_is_the_true_gains_perturbed_a_little(self):
        a = self.doc["truth"]["as_shot"]
        true = np.array(a["gains_true"])
        got = np.array([a["wb_rggb_1024"][0], a["wb_rggb_1024"][3]]) / 1024.0
        np.testing.assert_allclose(got / true[[0, 2]], np.exp(a["log_gain_error_rb"]), atol=2e-3)
        self.assertLess(max(abs(x) for x in a["log_gain_error_rb"]), 6 * a["sigma"])
        w_true = np.array(self.doc["truth"]["light"]["w_true_camera_rgb"])
        np.testing.assert_allclose(true, 1.0 / w_true, atol=1e-6)

    def test_the_metered_grey_and_the_noise_are_in_the_mosaic(self):
        m = self.doc["mosaic"]
        mosaic = self.mosaic.reshape(m["height"], m["width"]).astype(np.float64)
        probes = {p["name"]: p for p in self.doc["truth"]["probes"]}
        grey = probes["neutral-0.18"]
        x, y, w, h = grey["interior"]
        yy, xx = np.mgrid[y : y + h, x : x + w]
        g = mosaic[y : y + h, x : x + w][((yy % 2 == 0) & (xx % 2 == 1)) | ((yy % 2 == 1) & (xx % 2 == 0))] - 512  # both greens
        expected = grey["camera_dn_above_black"][1]
        self.assertAlmostEqual(expected / (0.12 * 15871), 1.0, places=6)
        self.assertAlmostEqual(g.mean() / expected, 1.0, delta=0.03)
        nm = self.doc["truth"]["noise"]
        model = nm["a_dn_per_electron"] * expected + nm["b_dn2"] + 1 / 12
        self.assertAlmostEqual(g.var() / model, 1.0, delta=0.35)  # 2 x 160 samples: about 8 % scatter, allow 4 sigma
        self.assertEqual(nm["iso"], 800)
        # the blown lamp is clipped and the deep black is near black
        lamp = probes["lamp-white"]
        lx, ly_, lw, lh = lamp["interior"]
        self.assertTrue((mosaic[ly_ : ly_ + lh, lx : lx + lw] == 16383).all())  # at ISO 800 the well holds 8x the ADC range
        self.assertEqual(lamp["clipped_channels"], [True, True, True])
        bx, by, bw, bh = probes["black"]["interior"]
        self.assertLess(mosaic[by : by + bh, bx : bx + bw].mean() - 512, 0.02 * 15871)

    def test_up_to_date_files_are_left_alone_and_a_forced_run_reproduces_them(self):
        before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in self.out.iterdir()}
        row = ms.generate(self.recipe, 3, self.out, self.ctx, force=False)
        self.assertEqual(row["status"], "up to date")
        row = ms.generate(self.recipe, 3, self.out, ms.Context(), force=True)  # a fresh context: nothing cached
        self.assertEqual(row["status"], "written")
        after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in self.out.iterdir()}
        self.assertEqual(before, after, "generation is deterministic")
        # a truncated mosaic is not "up to date": it is rewritten
        binf = self.out / f"{self.recipe.stem}.bin"
        binf.write_bytes(binf.read_bytes()[:1000])
        row = ms.generate(self.recipe, 3, self.out, self.ctx, force=False)
        self.assertEqual(row["status"], "written")
        self.assertEqual(hashlib.sha256(binf.read_bytes()).hexdigest(), before[binf.name])
        # a different recipe index (capture time) is a different recipe
        self.assertNotEqual(ms.recipe_hash(self.recipe, 3), ms.recipe_hash(self.recipe, 4))

    def test_probe_truth_in_the_sidecar(self):
        probes = {p["name"]: p for p in self.doc["truth"]["probes"]}
        self.assertEqual(len(probes), len(self.doc["truth"]["probes"]))
        grey, white = probes["neutral-0.18"], probes["neutral-0.9"]
        np.testing.assert_allclose(grey["alexa"]["awg4"], [0.18] * 3, atol=1e-8)
        np.testing.assert_allclose(white["sony_through_arri"]["awg4"], [0.9] * 3, atol=1e-8)
        self.assertEqual(grey["role"], "grey_reference")
        for p in probes.values():
            for key in ("alexa", "sony_through_arri"):
                self.assertEqual(set(p[key]), {"native_balanced", "awg3", "awg4", "logc4"})
        self.assertEqual(self.doc["truth"]["arri"]["cct_k"], 6500.0)
        self.assertEqual(self.doc["truth"]["light"]["class"], "daylight")
        flat = [p for p in self.doc["truth"]["probes"] if p["group"] in ("chart", "neutral", "ladder", "shadow", "black", "highlight")]
        self.assertEqual(len(flat), 190 + 24 + 48 + 16 + 1 + 2)


if __name__ == "__main__":
    unittest.main()
