"""Tests of the illuminant training code (``python3 -m unittest training.illum.test_illum`` or pytest).

They need the fetched spectral sources (``tools/fetch/spectral.sh``) like the other training scripts, and run in about a
minute. Rust parity is tested in ``crates/ym-color/tests/illum.rs`` against vectors written by ``export_vectors.py``.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from training.common import paths

paths.require_sources()

from training.illum import baselines as bl  # noqa: E402
from training.illum import dataset as dsm  # noqa: E402
from training.illum import eval_cubepp as cpp  # noqa: E402
from training.illum import export_vectors as xv  # noqa: E402
from training.illum import features as ft  # noqa: E402
from training.illum import gate_estimated as ge  # noqa: E402
from training.illum import model as md  # noqa: E402
from training.illum import robust_prior_study as rps  # noqa: E402
from training.illum import scenes as sc  # noqa: E402
from training.illum import spectra as sp  # noqa: E402

PID = "sony-ilce-7rm4"


class Spectra(unittest.TestCase):
    def test_camera_is_in_the_published_raw_scale(self):
        cam = sp.camera(PID)
        peaks = cam.curves.max(axis=0)
        self.assertAlmostEqual(peaks[1], 1.0, places=9)
        self.assertLess(peaks[0], 0.7)  # not peak-normalised per channel (docs/02 C1)
        self.assertLess(peaks[2], 0.9)

    def test_light_splits_are_disjoint(self):
        rng = np.random.default_rng(0)
        for klass, bands in (("daylight", None), ("tungsten_halogen", None)):
            for split in ("train", "val", "test"):
                for _ in range(200):
                    L = sp.draw_light(rng, klass, split)
                    inside_test = sp._in_bands(L.cct_k, sp._bands("test", "blackbody" if L.family == "blackbody" else "daylight"))
                    inside_val = sp._in_bands(L.cct_k, sp._bands("val", "blackbody" if L.family == "blackbody" else "daylight"))
                    self.assertEqual(inside_test, split == "test", (klass, split, L.cct_k))
                    self.assertEqual(inside_val, split == "val", (klass, split, L.cct_k))
        for klass, test, val in (("fluorescent", sp.TEST_FLUORESCENT, sp.VAL_FLUORESCENT), ("led", sp.TEST_LED, sp.VAL_LED)):
            seen = {s: set() for s in ("train", "val", "test")}
            for split in seen:
                for _ in range(300):
                    L = sp.draw_light(rng, klass, split)
                    if L.family.endswith("_cie"):
                        seen[split].add(L.name)
            self.assertTrue(seen["train"].isdisjoint(test) and seen["train"].isdisjoint(val))
            self.assertTrue(seen["test"] <= set(test) and seen["val"] <= set(val))
        gate = set(sp.GATE_LIGHT_NAMES)
        self.assertTrue({"FL3.1", "FL3.8", "FL3.15", "LED-B3", "LED-RGB1", "LED-V2"} <= gate)
        self.assertTrue({"FL3.1", "FL3.8", "FL3.15"} <= set(sp.TEST_FLUORESCENT))
        self.assertTrue({"LED-B3", "LED-RGB1", "LED-V2"} <= set(sp.TEST_LED))

    def test_reflectance_subsets_do_not_overlap(self):
        tr, va, te = sp.apache_patches()
        self.assertEqual((tr.shape[0], va.shape[0], te.shape[0]), (132, 20, 38))
        key = lambda a: {tuple(np.round(r, 9)) for r in a}  # noqa: E731
        self.assertFalse(key(tr) & key(va) or key(tr) & key(te) or key(va) & key(te))

    def test_generated_lights_are_near_white(self):
        rng = np.random.default_rng(3)
        for make in (sp.generated_led, sp.generated_fluorescent):
            for _ in range(10):
                self.assertLessEqual(sp.duv_of(make(rng)), sp.MAX_GENERATED_DUV)


class Scenes(unittest.TestCase):
    def test_a_scene_is_a_pure_function_of_its_seed(self):
        cam = sp.camera(PID)
        a = sc.render_scene(cam, 5, "test", 17)
        b = sc.render_scene(cam, 5, "test", 17)
        self.assertTrue(np.array_equal(a.img, b.img) and np.array_equal(a.mask, b.mask))
        self.assertTrue(np.array_equal(a.truth_uv, b.truth_uv))
        c = sc.render_scene(cam, 5, "test", 18)
        self.assertFalse(np.array_equal(a.img, c.img))

    def test_a_grey_scene_records_the_cameras_white(self):
        cam = sp.camera(PID)
        light = sp.named_light("LED-B3", "led")
        s = sc.render_scene(cam, 1, "test", 0, light=light)
        # the truth is the light's camera white, to the accuracy of the (clipped, noisy) average
        w = cam.white(light.spd)
        self.assertLess(bl.angular_error_deg(s.truth_w, w), 0.01)
        self.assertEqual(s.img.shape, (sc.H, sc.W, 3))
        self.assertEqual(s.klass, sp.CLASS_INDEX["led"])

    def test_gate_light_scenes_use_the_named_light(self):
        cam = sp.camera(PID)
        for L in sp.gate_lights():
            s = sc.render_scene(cam, 1, "test", 1, light=L)
            self.assertEqual(s.light_names, (L.name,))

    def test_mixed_scenes_have_two_lights(self):
        cam = sp.camera(PID)
        n = 0
        for i in range(60):
            s = sc.render_scene(cam, 2, "train", i, klass="mixed")
            self.assertEqual(len(s.light_names), 2)
            n += 1
        self.assertEqual(n, 60)


class Features(unittest.TestCase):
    def setUp(self):
        self.spec = ft.HistSpec(n_bins=16, bin_size=0.25, lo_u=-1.25, lo_v=-1.0)

    def test_histograms_sum_to_one_and_weigh_by_sqrt_y(self):
        img = np.zeros((4, 4, 3))
        img[:2] = [0.2, 0.4, 0.3]
        img[2:] = [0.1, 0.4, 0.2]
        h, st = ft.histograms(img, None, ft.HistSpec(**{**self.spec.__dict__, "channels": ("intensity",)}))
        self.assertAlmostEqual(h.sum(), 1.0, places=12)
        w1 = np.sqrt(0.25 * 0.2 + 0.5 * 0.4 + 0.25 * 0.3)
        w2 = np.sqrt(0.25 * 0.1 + 0.5 * 0.4 + 0.25 * 0.2)
        i = np.arange(16)
        mean_u = (h.sum(axis=2)[0] * (self.spec.lo_u + i * self.spec.bin_size)).sum()
        want = (w1 * 8 * np.log(0.4 / 0.2) + w2 * 8 * np.log(0.4 / 0.1)) / (8 * w1 + 8 * w2)
        self.assertAlmostEqual(mean_u, want, places=12)  # bilinear splatting keeps the mean exactly
        self.assertEqual(st.valid_fraction[0], 1.0)

    def test_masked_dark_and_out_of_range_pixels_are_dropped(self):
        img = np.full((4, 4, 3), [0.2, 0.4, 0.3])
        mask = np.zeros((4, 4), np.uint8)
        mask[0] = 1
        img[1, 0] = [0.0, 0.4, 0.3]      # dark in red
        img[1, 1] = [0.2, 0.4, 1e-6]     # dark in blue
        img[1, 2] = [0.0012, 0.4, 0.3]   # valid, but its chroma is outside the window
        _, st = ft.histograms(img, mask, ft.HistSpec(**{**self.spec.__dict__, "channels": ("intensity",)}))
        self.assertAlmostEqual(st.valid_fraction[0], 10 / 16)  # 16 - 4 masked - 2 dark; the out-of-range pixel is valid
        self.assertLess(st.in_range[0], 1.0)

    def test_edge_image_ignores_masked_neighbours(self):
        img = np.zeros((3, 3, 3))
        img[1, 1] = [1.0, 1.0, 1.0]
        valid = np.ones((3, 3), bool)
        e, has = ft.edge_image(img, valid)
        self.assertAlmostEqual(e[1, 1, 0], 1.0)           # all 8 neighbours differ by 1
        self.assertAlmostEqual(e[0, 0, 0], 1.0 / 3.0)     # corner: 3 neighbours, one of them the bright centre
        valid[0, 0] = False
        e, has = ft.edge_image(img, valid)
        self.assertFalse(has[0, 0])
        self.assertAlmostEqual(e[1, 1, 0], 1.0)


class Model(unittest.TestCase):
    def test_json_round_trip_and_hash(self):
        m = xv.make_test_model().rounded()
        again = md.IllumModel.from_json(json.loads(json.dumps(m.to_json())))
        self.assertEqual(m.payload_sha256(), again.payload_sha256())
        doc = m.to_json()
        doc["sha256"] = "00"
        with self.assertRaises(ValueError):
            md.IllumModel.from_json(doc)
        self.assertLess(m.param_count(), 50_000)

    def test_estimate_has_the_documented_shape_and_green_one(self):
        m = xv.make_test_model().rounded()
        rng = np.random.default_rng(0)
        img = rng.uniform(0.02, 0.3, size=(24, 36, 3)).astype(np.float32)
        out = m.estimate(img, None, np.array([0.9, 0.5]), 6.0)
        self.assertEqual(out["probs"].shape, (6,))
        self.assertAlmostEqual(float(out["probs"].sum()), 1.0, places=12)
        self.assertTrue(0.0 <= out["confidence"] <= 1.0)
        none = m.estimate(img, np.ones((24, 36), np.uint8), None, None)
        self.assertTrue(none["fallback"])
        np.testing.assert_allclose(none["uv"], m.default_uv)

    def test_fusion_is_a_product_of_gaussians(self):
        m = xv.make_test_model().rounded()
        mu = np.array([[1.0, 0.4]])
        cov = np.array([[[0.01, 0.002], [0.002, 0.02]]])
        a = np.array([[1.1, 0.5]])
        mu_p, cov_p = m.fuse(mu, cov, a)
        prec = np.linalg.inv(cov[0]) + np.eye(2) / m.prior_sigma**2
        np.testing.assert_allclose(cov_p[0], np.linalg.inv(prec), rtol=1e-12)
        np.testing.assert_allclose(mu_p[0], np.linalg.inv(prec) @ (np.linalg.inv(cov[0]) @ mu[0] + a[0] / m.prior_sigma**2), rtol=1e-12)
        mu_n, cov_n = m.fuse(mu, cov, np.array([[np.nan, np.nan]]))
        np.testing.assert_allclose(mu_n, mu)

    def test_confidence_features_and_head(self):
        mu = np.array([[1.0, 0.4]])
        cov = np.array([[[0.01, 0.002], [0.002, 0.02]]])
        post = np.array([[[0.002, 0.0], [0.0, 0.0015]]])
        a = np.array([[1.1, 0.5]])
        ex = md.conf_extras(mu, cov, post, a)[0]
        d = mu[0] - a[0]
        self.assertAlmostEqual(ex[0], float(np.sqrt(d @ np.linalg.inv(cov[0]) @ d)), places=12)
        self.assertAlmostEqual(ex[1], float(np.log(0.00175)), places=12)
        self.assertAlmostEqual(ex[2], float(np.log(0.015)), places=12)
        none = md.conf_extras(mu, cov, post, np.array([[np.nan, np.nan]]))[0]
        self.assertEqual(none[0], 0.0)
        m = xv.make_test_model().rounded()
        x = np.random.default_rng(0).normal(size=(4, m.conf.mean.size))
        p = m.conf.probability(x)
        self.assertTrue(np.all((p > 0) & (p < 1)))

    def test_the_study_baseline_is_the_shipped_fusion_and_a_wide_prior_follows_the_histogram(self):
        m = xv.make_test_model().rounded()
        mu = np.array([[1.0, 0.4], [0.8, 0.5]])
        cov = np.array([[[0.01, 0.002], [0.002, 0.02]]] * 2)
        a = np.array([[1.1, 0.5], [1.6, 0.0]])  # the second as-shot value is far from the histogram estimate
        shipped, _ = m.fuse(mu, cov, a)
        np.testing.assert_allclose(rps.fuse_mixture(mu, cov, a, 0.0, 0.3), shipped, rtol=1e-12)
        mixed = rps.fuse_mixture(mu, cov, a, 0.2, 0.3)
        # near the histogram estimate nothing changes much; far from it the mixture trusts the histogram more
        self.assertLess(np.abs(mixed[0] - shipped[0]).max(), 0.02)
        self.assertLess(np.linalg.norm(mixed[1] - mu[1]), np.linalg.norm(shipped[1] - mu[1]))

    def test_the_shipped_vectors_of_all_profiles_belong_to_the_shipped_models(self):
        doc = json.loads((xv.OUT / "all_profiles_vectors.json").read_text())
        self.assertEqual(sorted(doc["profiles"]), sorted(sp.profile_ids()))
        for pid, entry in doc["profiles"].items():
            m = md.IllumModel.from_json(json.loads((paths.REPO_ROOT / "data" / "models" / "illum" / f"{pid}.json").read_text()))
            self.assertEqual(entry["model_sha256"], m.payload_sha256(), f"{pid}: rerun export_vectors")
        np.testing.assert_array_equal(xv.channel_gain(PID), np.ones(3, dtype=np.float32))

    def test_the_jax_training_graph_equals_the_numpy_reference(self):
        try:
            import jax.numpy as jnp
            from training.illum import train as tr
        except ImportError:  # pragma: no cover
            self.skipTest("jax is not installed")
        m = xv.make_test_model().rounded()
        rng = np.random.default_rng(1)
        x = rng.dirichlet(np.ones(256), size=(5, 2)).reshape(5, 2, 16, 16)
        z = m.filter_scores(x)
        mu, c00, c11, c01 = tr._gauss_moments(jnp.asarray(np.exp(z - z.max((-2, -1), keepdims=True)) / np.exp(z - z.max((-2, -1), keepdims=True)).sum((-2, -1), keepdims=True), dtype=jnp.float32),
                                              16, m.spec.bin_size, m.spec.lo_u, m.spec.lo_v, m.pad)
        mu_ref, cov_ref, _ = m.gaussian_fit(z)
        np.testing.assert_allclose(np.asarray(mu), mu_ref, atol=2e-5)
        np.testing.assert_allclose(np.asarray(c00), cov_ref[:, 0, 0], rtol=1e-3, atol=1e-6)
        np.testing.assert_allclose(np.asarray(c01), cov_ref[:, 0, 1], rtol=1e-3, atol=1e-6)


class Evaluation(unittest.TestCase):
    def test_angular_error(self):
        self.assertAlmostEqual(float(bl.angular_error_deg([1, 1, 1], [1, 1, 1])), 0.0)
        self.assertAlmostEqual(float(bl.angular_error_deg([1, 0, 0], [0, 1, 0])), 90.0)
        uv = np.array([0.8, 0.5])
        np.testing.assert_allclose(bl.uv_from_w(bl.w_from_uv(uv)), uv)

    def test_class_folding_matches_the_rust_rule(self):
        p = np.array([0.1, 0.2, 0.3, 0.1, 0.1, 0.2])
        low = ge.fold_classes(p, np.array(3500.0))
        high = ge.fold_classes(p, np.array(5000.0))
        np.testing.assert_allclose(low, [0.3, 0.5, 0.1, 0.1])
        np.testing.assert_allclose(high, [0.5, 0.3, 0.1, 0.1])

    def test_cubepp_helpers(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "gt.csv"
            p.write_text("image,mean_r,mean_g,mean_b,median_r\na.PNG,0.5,1.0,0.7,9\nb.PNG,0.6,1.0,0.6,9\n")
            gt = cpp.read_ground_truth(p)
            np.testing.assert_allclose(gt["a"], [0.5, 1.0, 0.7])
        img = np.arange(4 * 4 * 3, dtype=float).reshape(4, 4, 3)
        np.testing.assert_allclose(cpp.box_downsample(img, 2)[0, 0], img[:2, :2].mean(axis=(0, 1)))
        # identity source matrix: the converted truth is the profile's matrix applied to the truth
        to_cam = cpp.xyz_to_camera(PID)
        out_img, out_truth = cpp.convert(np.ones((2, 2, 3)), np.array([0.5, 1.0, 0.7]), np.eye(3), to_cam)
        np.testing.assert_allclose(out_truth, to_cam @ np.array([0.5, 1.0, 0.7]))
        # D65 white maps to the camera's D65 white response
        cam = sp.camera(PID)
        from training.alexa import ssf_core as s

        e = s.named_illuminant("D65")
        xyz_w = (e @ s.CMF) / (s.CMF[:, 1] @ e)
        white_cam = cam.response(e)
        mapped = to_cam @ xyz_w
        self.assertLess(float(bl.angular_error_deg(mapped, white_cam)), 2.5)  # least squares: not exact, close

    def test_cubepp_simplecube_layout(self):
        """SimpleCube++: Canon curves mapped to the profile, black level 2048, the cropped cube and clipped pixels
        masked, breakdown by the dataset's labels."""
        from unittest import mock

        from training.alexa import ssf_core as s
        from training.sensor_map import fit

        src = cpp.source_curves("Canon_EOS_600D")
        to_prof = cpp.camera_to_camera(src, PID)
        dst = sp.camera(PID).curves
        for cct in (2856, 4000, 6500):
            e = s.illuminant(cct)
            self.assertLess(float(bl.angular_error_deg(to_prof @ (src.T @ e), dst.T @ e)), 0.3)
        # masks: a zeroed rectangle and, in an image that clips, the pixels at its clip
        raw = np.full((4, 4, 3), 6000.0)
        raw[0, 0] = 0.0
        raw[3, 3] = [16000.0, 9000.0, 9000.0]
        bad = cpp.invalid_pixels(raw, 2048.0, 16383.0)
        self.assertTrue(bad[0, 0] and bad[3, 3] and bad.sum() == 2)
        self.assertEqual(int(cpp.invalid_pixels(np.full((2, 2, 3), 6000.0), 2048.0, 16383.0).sum()), 0)
        # end to end on two patch scenes, PNG reading replaced
        refl = fit.training_reflectances()
        rng = np.random.default_rng(1)
        images, rows = {}, []
        for name, cct in (("1_D", 5500), ("2_D", 3000)):
            e = s.illuminant(cct)
            rgb = (refl[:, rng.integers(0, refl.shape[1], 96)] * e[:, None]).T @ src
            img = rgb.reshape(8, 12, 3).repeat(8, 0).repeat(8, 1)
            images[name] = np.round(2048 + img / img.max() * 0.7 * (16383 - 2048)) / 65535.0
            w = src.T @ e
            rows.append(f"{name},{w[0] / w.sum()},{w[1] / w.sum()},{w[2] / w.sum()}")
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "PNG").mkdir()
            for name in images:
                (root / "PNG" / f"{name}.png").write_bytes(b"")
            (root / "gt.csv").write_text("image,r,g,b\n" + "\n".join(rows) + "\n")
            props = {"1_D": {"place": "outdoor", "daytime": "day"}, "2_D": {"place": "indoor", "daytime": "night"}}
            shipped = paths.REPO_ROOT / "data" / "models" / "illum" / f"{PID}.json"
            model = md.IllumModel.from_json(json.loads(shipped.read_text()))
            with mock.patch.object(cpp, "read_png16", lambda path: images[Path(path).stem]):
                res = cpp.evaluate(model, root, to_prof, None, None, 2048.0, 16383.0, 1, props)
        self.assertEqual(res["n"], 2)
        self.assertLess(res["max"], 3.0)
        self.assertEqual(set(res["by_label"]["place"]), {"indoor", "outdoor"})


if __name__ == "__main__":
    unittest.main()
