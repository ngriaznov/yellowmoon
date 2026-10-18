"""The model file (section 4.3), the two forward passes, the combination (section 4.2) and the rule (section 4.4)."""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import math
import unittest

import jax.numpy as jnp
import numpy as np

from training.autoset import model as md


def random_model(seed: int = 1, members: int = 2) -> md.AutosetModel:
    """A model with seeded random weights (float32-exact, as stored)."""
    rng = np.random.default_rng(seed)
    layers_all = []
    for _ in range(members):
        layers = []
        for out_dim, in_dim in md.LAYER_SHAPES:
            w = rng.normal(0.0, 0.3, size=(out_dim, in_dim))
            b = rng.normal(0.0, 0.1, size=out_dim)
            layers.append((w, b))
        layers_all.append(layers)
    return md.AutosetModel(
        model_version="9.9.9-test",
        input_mean=rng.normal(0.0, 1.0, md.N_FEATURES),
        input_std=rng.uniform(0.5, 2.0, md.N_FEATURES),
        members=layers_all,
    ).rounded()


class FileFormat(unittest.TestCase):
    def test_the_member_has_2468_parameters(self) -> None:
        m = random_model(members=5)
        self.assertEqual(m.param_count_per_member(), 39 * 32 + 32 + 32 * 32 + 32 + 32 * 4 + 4)
        self.assertEqual(m.param_count(), 5 * 2468)

    def test_round_trip_is_identical(self) -> None:
        m = random_model()
        doc = m.to_json()
        back = md.AutosetModel.from_json(json.loads(json.dumps(doc)))
        self.assertEqual(back.to_json(), doc)
        self.assertEqual(back.payload_sha256(), m.payload_sha256())
        self.assertEqual(list(doc.keys()), ["format", "format_version", "model_version", "feature_names", "input_mean",
                                           "input_std", "activation", "outputs", "prior_sigma", "contrast_limit",
                                           "members", "sha256"])
        self.assertEqual(doc["members"][0]["layers"][0]["w"]["shape"], [32, 39])
        self.assertEqual(doc["members"][0]["layers"][2]["b"]["shape"], [4])

    def test_the_checksum_covers_the_member_tensors_in_order(self) -> None:
        m = random_model()
        h = hashlib.sha256()
        for layers in m.members:
            for w, b in layers:
                h.update(np.ascontiguousarray(w, dtype="<f4").tobytes())
                h.update(np.ascontiguousarray(b, dtype="<f4").tobytes())
        self.assertEqual(h.hexdigest(), m.payload_sha256())

    def test_a_changed_weight_is_refused(self) -> None:
        doc = random_model().to_json()
        raw = bytearray(base64.b64decode(doc["members"][0]["layers"][0]["w"]["data_b64"]))
        raw[0] ^= 0x01
        doc["members"][0]["layers"][0]["w"]["data_b64"] = base64.b64encode(bytes(raw)).decode()
        with self.assertRaises(md.AutosetError) as ctx:
            md.AutosetModel.from_json(doc)
        self.assertIn("sha256", str(ctx.exception))

    def test_format_features_and_shapes_are_checked(self) -> None:
        doc = random_model().to_json()
        wrong_format = copy.deepcopy(doc)
        wrong_format["format"] = "something-else"
        with self.assertRaises(md.AutosetError):
            md.AutosetModel.from_json(wrong_format)
        wrong_version = copy.deepcopy(doc)
        wrong_version["format_version"] = 2
        with self.assertRaises(md.AutosetError):
            md.AutosetModel.from_json(wrong_version)
        names = copy.deepcopy(doc)
        names["feature_names"] = names["feature_names"][::-1]
        with self.assertRaises(md.AutosetError):
            md.AutosetModel.from_json(names)
        shape = copy.deepcopy(doc)
        shape["members"][0]["layers"][1]["w"]["shape"] = [32, 31]
        with self.assertRaises(md.AutosetError):
            md.AutosetModel.from_json(shape)

    def test_a_value_that_is_not_finite_is_refused(self) -> None:
        doc = random_model().to_json()
        data = np.full(32 * 39, np.nan, dtype="<f4").tobytes()
        doc["members"][0]["layers"][0]["w"]["data_b64"] = base64.b64encode(data).decode()
        with self.assertRaises(md.AutosetError) as ctx:
            md.AutosetModel.from_json(doc)
        self.assertIn("not finite", str(ctx.exception))

    def test_input_statistics_must_be_finite_and_positive(self) -> None:
        doc = random_model().to_json()
        doc["input_std"][3] = 0.0
        with self.assertRaises(md.AutosetError):
            md.AutosetModel.from_json(doc)


class Forward(unittest.TestCase):
    def test_numpy_and_jax_forward_agree(self) -> None:
        m = random_model(seed=4, members=3)
        X = np.random.default_rng(9).normal(0.0, 2.0, size=(17, md.N_FEATURES))
        z = m.standardise(X)
        ref = m.member_outputs(X)
        for i, layers in enumerate(m.members):
            params = [(jnp.asarray(w), jnp.asarray(b)) for w, b in layers]
            got = np.asarray(md.jax_forward(params, jnp.asarray(z)))
            np.testing.assert_allclose(got, ref[i], rtol=0, atol=1e-12)

    def test_standardisation_uses_the_stored_statistics(self) -> None:
        m = random_model()
        X = np.arange(md.N_FEATURES, dtype=np.float64)[None, :]
        np.testing.assert_allclose(m.standardise(X), (X - m.input_mean) / m.input_std)


class Combination(unittest.TestCase):
    def test_mixture_mean_and_variance(self) -> None:
        rng = np.random.default_rng(3)
        outs = rng.normal(0.0, 1.0, size=(5, 6, 4))
        p = md.combine(outs)
        for i in range(6):
            mus = outs[:, i, 0]
            self.assertAlmostEqual(p.key_mean[i], mus.mean(), places=12)
            v_mean = np.mean([math.exp(outs[k, i, 1]) for k in range(5)])
            self.assertAlmostEqual(p.key_var[i], v_mean + np.mean((mus - mus.mean()) ** 2), places=12)
            cms = outs[:, i, 2]
            self.assertAlmostEqual(p.contrast_mean[i], cms.mean(), places=12)
            cv_mean = np.mean([math.exp(outs[k, i, 3]) for k in range(5)])
            self.assertAlmostEqual(p.contrast_var[i], cv_mean + np.mean((cms - cms.mean()) ** 2), places=12)

    def test_logvars_are_clamped_before_the_exponential(self) -> None:
        outs = np.zeros((2, 1, 4))
        outs[:, :, 1] = 50.0
        outs[:, :, 3] = -50.0
        p = md.combine(outs)
        self.assertAlmostEqual(p.key_var[0], math.exp(md.LOGVAR_LIMIT), places=3)
        self.assertAlmostEqual(p.contrast_var[0], math.exp(-md.LOGVAR_LIMIT), places=30)


class Rule(unittest.TestCase):
    def test_the_weight_formula(self) -> None:
        self.assertEqual(md.weight(0.75, 0.0), 1.0)
        self.assertAlmostEqual(md.weight(0.75, 0.5625), 0.5, places=15)
        self.assertAlmostEqual(md.weight(10.0, 100.0), 0.5, places=15)

    def test_a_certain_key_prediction_gives_the_key_target_minus_the_key(self) -> None:
        # key target 0.4 and key -1.0: EV = 0.4 - (-1.0) = 1.4, reached through the camera gain with weight 1
        a = md.apply_rule(key_mean=0.4, key_var=0.0, contrast_mean=0.0, contrast_var=0.0, key=-1.0)
        self.assertAlmostEqual(a.ev_raw, 1.4, places=12)
        self.assertAlmostEqual(a.ev, 1.4, places=12)
        self.assertEqual(a.ev_confidence, 1.0)

    def test_a_doubtful_exposure_still_applies_and_a_doubtful_contrast_fades(self) -> None:
        # Full rescue (owner's decision of 2026-10-18): the exposure is the model's whatever its confidence.
        a = md.apply_rule(key_mean=-2.0, key_var=1e12, contrast_mean=25.0, contrast_var=1e12, key=0.0)
        self.assertAlmostEqual(a.ev, -2.0, places=12)
        self.assertLess(a.ev_confidence, 1e-9)
        self.assertAlmostEqual(a.contrast, 0.0, places=9)

    def test_ev_and_contrast_are_clamped(self) -> None:
        a = md.apply_rule(key_mean=9.0, key_var=0.0, contrast_mean=100.0, contrast_var=0.0, key=-1.0)
        self.assertEqual(a.ev, md.EV_LIMIT)
        self.assertGreater(a.ev_raw, md.EV_LIMIT)
        self.assertEqual(a.contrast_raw, md.CONTRAST_LIMIT)
        self.assertEqual(a.contrast, 30.0)
        b = md.apply_rule(key_mean=-9.0, key_var=0.0, contrast_mean=-100.0, contrast_var=0.0, key=0.0)
        self.assertEqual(b.ev, -md.EV_LIMIT)
        self.assertEqual(b.contrast, -30.0)

    def test_contrast_rounds_half_away_from_zero(self) -> None:
        self.assertEqual(md.round_tenth(2.25), 2.3)
        self.assertEqual(md.round_tenth(-2.25), -2.3)
        self.assertEqual(md.round_tenth(0.05), 0.1)
        self.assertEqual(md.round_tenth(-0.05), -0.1)
        self.assertEqual(md.round_tenth(0.04), 0.0)
        self.assertEqual(md.round_tenth(29.96), 30.0)
        self.assertEqual(md.round_tenth(7.0), 7.0)

    def test_the_model_applies_its_own_priors_and_limit(self) -> None:
        m = random_model()
        m.contrast_limit = 12.0
        a = m.apply(0.0, 0.0, 50.0, 0.0, 0.0)
        self.assertEqual(a.contrast, 12.0)


if __name__ == "__main__":
    unittest.main()
