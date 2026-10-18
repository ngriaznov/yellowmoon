"""Training (section 6.3), the scene splits, the evaluation metrics and the command line, on synthetic rows."""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from training.autoset import dataset as ds
from training.autoset import evaluation as evm
from training.autoset import model as md
from training.autoset import model_card
from training.autoset import rows as rw
from training.autoset import synthetic
from training.autoset import train as tr


def _prepared(n_scenes: int = 30, **kwargs) -> ds.Prepared:
    objs, _ = synthetic.make_dataset(n_scenes, **kwargs)
    return ds.prepare(rw.read_objects(objs))


def _random_model(seed: int = 3, members: int = 1) -> md.AutosetModel:
    rng = np.random.default_rng(seed)
    layers_all = [[(rng.normal(0.0, 0.2, (o, i)), rng.normal(0.0, 0.1, o)) for o, i in md.LAYER_SHAPES] for _ in range(members)]
    return md.AutosetModel("rand", rng.normal(0.0, 1.0, md.N_FEATURES), rng.uniform(0.5, 2.0, md.N_FEATURES), layers_all).rounded()


QUIET = {"log": lambda _: None}


class Training(unittest.TestCase):
    def test_members_are_seeded(self) -> None:
        a = tr.init_member(1000)
        b = tr.init_member(1000)
        c = tr.init_member(1001)
        for (wa, ba), (wb, bb) in zip(a, b, strict=True):
            np.testing.assert_array_equal(wa, wb)
            np.testing.assert_array_equal(ba, bb)
        self.assertFalse(np.array_equal(a[0][0], c[0][0]))
        self.assertEqual(a[-1][0].shape, (4, 32))

    def test_training_beats_the_constant_rule_on_the_labelled_files(self) -> None:
        prep = _prepared(30)
        cfg = tr.TrainConfig(members=2, steps=400, holdout=0.0)
        model, _ = tr.train_ensemble(prep, np.arange(prep.n), None, cfg, model_version="t", **QUIET)
        pred = model.predict(prep.X)
        files = np.nonzero(prep.key_ok & ~prep.is_aug)[0]
        model_err = np.abs(pred.key_mean[files] - prep.key[files] - prep.level[files]).mean()
        constant_err = np.abs(md.CAMERA_GAIN_EV - prep.level[files]).mean()
        self.assertLess(model_err, 0.5 * constant_err)

    def test_the_augmentations_are_learned_as_equivariant(self) -> None:
        prep = _prepared(30)
        cfg = tr.TrainConfig(members=2, steps=400, holdout=0.0)
        model, _ = tr.train_ensemble(prep, np.arange(prep.n), None, cfg, model_version="t", **QUIET)
        aug = np.nonzero(prep.key_ok & prep.is_aug)[0]
        errors = evm.equivariance_errors(model.predict(prep.X).key_mean, prep, aug)
        self.assertLess(float(np.abs(errors).mean()), 0.05)

    def test_two_trainings_give_identical_bytes(self) -> None:
        prep = _prepared(12)
        cfg = tr.TrainConfig(members=2, steps=30, holdout=0.15)
        scenes = ds.labelled_scenes(prep)
        fit, hold = ds.holdout_split(scenes, cfg.holdout, cfg.seed)
        texts = []
        for _ in range(2):
            model, _ = tr.train_ensemble(prep, ds.rows_of(prep, fit), ds.rows_of(prep, hold), cfg, model_version="1.0.0", **QUIET)
            texts.append(json.dumps(model.to_json(), sort_keys=True))
        self.assertEqual(texts[0], texts[1])

    def test_early_stopping_reports_its_checkpoint(self) -> None:
        prep = _prepared(12)
        cfg = tr.TrainConfig(members=1, steps=60, eval_every=10, holdout=0.2)
        scenes = ds.labelled_scenes(prep)
        fit, hold = ds.holdout_split(scenes, cfg.holdout, cfg.seed)
        _, reports = tr.train_ensemble(prep, ds.rows_of(prep, fit), ds.rows_of(prep, hold), cfg, model_version="t", **QUIET)
        self.assertEqual(len(reports), 1)
        self.assertIsNotNone(reports[0].best_holdout_nll)
        self.assertIn(reports[0].best_step, range(10, 61, 10))

    def test_an_empty_training_set_is_an_error(self) -> None:
        prep = _prepared(4)
        with self.assertRaises(ValueError):
            tr.train_ensemble(prep, np.array([], dtype=np.int64), None, tr.TrainConfig(members=1, steps=2), model_version="t", **QUIET)


class Splits(unittest.TestCase):
    def test_the_holdout_takes_whole_scenes(self) -> None:
        prep = _prepared(20)
        scenes = ds.labelled_scenes(prep)
        fit, hold = ds.holdout_split(scenes, 0.15, 7)
        self.assertEqual(len(hold), round(0.15 * len(scenes)))
        self.assertEqual(sorted(np.concatenate([fit, hold]).tolist()), sorted(scenes.tolist()))
        in_hold = np.isin(prep.scene, hold)
        for i in np.nonzero(prep.is_aug)[0]:
            self.assertEqual(in_hold[i], in_hold[prep.base[i]])

    def test_the_folds_partition_the_scenes(self) -> None:
        prep = _prepared(20)
        scenes = ds.labelled_scenes(prep)
        splits = ds.kfold_splits(scenes, 4, 11)
        tests = np.concatenate([te for _, te in splits])
        self.assertEqual(sorted(tests.tolist()), sorted(scenes.tolist()))
        for tr_s, te_s in splits:
            self.assertEqual(set(tr_s.tolist()) & set(te_s.tolist()), set())
        with self.assertRaises(ValueError):
            ds.kfold_splits(scenes, 1, 11)

    def test_the_kfold_run_predicts_every_labelled_scene_once(self) -> None:
        prep = _prepared(12)
        cfg = tr.TrainConfig(members=1, steps=20, folds=3, holdout=0.2)
        result = tr.run_kfold(prep, cfg, log=lambda _: None)
        labelled_rows = ds.rows_of(prep, ds.labelled_scenes(prep))
        self.assertTrue(result["evaluated"][labelled_rows].all())
        summary = evm.summarise(prep, result["pred"], result["evaluated"], mode="kfold", folds=result["folds"])
        self.assertEqual(summary["evaluated_files"], int((prep.key_ok & ~prep.is_aug).sum()))
        self.assertIn("normal_files", summary["ev"])


class Metrics(unittest.TestCase):
    def test_equivariance_is_exactly_zero_for_a_model_that_ignores_the_headroom(self) -> None:
        prep = _prepared(8)
        m = _random_model()
        layers = [(w.copy(), b.copy()) for w, b in m.members[0]]
        layers[0][0][:, 24] = 0.0  # clip_headroom moves by -k under augmentation (section 3)
        layers[0][0][:, 25] = 0.0  # clipped_share moves for brighter scalings
        ignores = md.AutosetModel(m.model_version, m.input_mean, m.input_std, [layers]).rounded()
        aug = np.nonzero(prep.is_aug)[0]
        errors = evm.equivariance_errors(ignores.predict(prep.X).key_mean, prep, aug)
        self.assertLess(float(np.max(np.abs(errors))), 1e-12)

    def test_equivariance_is_visible_for_a_model_that_uses_the_headroom(self) -> None:
        prep = _prepared(8)
        m = _random_model()
        aug = np.nonzero(prep.is_aug)[0]
        errors = evm.equivariance_errors(m.predict(prep.X).key_mean, prep, aug)
        self.assertGreater(float(np.max(np.abs(errors))), 1e-3)

    def test_stats_skip_missing_values_and_count_tolerances(self) -> None:
        s = evm.stats(np.array([0.1, -0.2, np.nan, 0.8]), tolerances=(0.3,))
        self.assertEqual(s["n"], 3)
        self.assertAlmostEqual(s["mae"], (0.1 + 0.2 + 0.8) / 3)
        self.assertAlmostEqual(s["within_0p3"], 2 / 3)
        self.assertEqual(evm.stats(np.array([np.nan]))["n"], 0)

    def test_the_metrics_document_is_plain_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.json"
            evm.write_metrics(path, {"a": np.float64(1.5), "b": np.int64(3), "c": np.array([1.0, np.nan]), "d": [np.bool_(True)]})
            data = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(data, {"a": 1.5, "b": 3, "c": [1.0, None], "d": [True]})


class CommandLine(unittest.TestCase):
    def test_train_evaluate_and_card_run_end_to_end(self) -> None:
        objs, _ = synthetic.make_dataset(12)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            rows_path = tmp_path / "rows.jsonl"
            synthetic.write_jsonl(objs, rows_path)
            model_path = tmp_path / "autoset.json"
            metrics_path = tmp_path / "metrics.json"
            with contextlib.redirect_stdout(io.StringIO()):
                code = tr.main(["--rows", str(rows_path), "--out", str(model_path), "--metrics", str(metrics_path),
                                "--model-version", "0.1.0", "--members", "2", "--steps", "30", "--kfold", "3"])
            self.assertEqual(code, 0)
            model = md.AutosetModel.from_json(json.loads(model_path.read_text(encoding="utf-8")))
            self.assertEqual(model.model_version, "0.1.0")
            self.assertEqual(len(model.members), 2)
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            self.assertEqual(metrics["evaluation"]["mode"], "kfold")
            self.assertIn("heldout_files", metrics["evaluation"]["ev"])
            self.assertEqual(metrics["final_model"]["model"]["sha256"], model.payload_sha256())
            card_path = tmp_path / "MODEL_CARD.md"
            with contextlib.redirect_stdout(io.StringIO()):
                code = model_card.main(["--metrics", str(metrics_path), "--model", str(model_path), "--date", "2026-10-18",
                                        "--git-sha", "0123456789abcdef", "--out", str(card_path)])
            self.assertEqual(code, 0)
            card = card_path.read_text(encoding="utf-8")
            self.assertIn("Version 0.1.0, 2026-10-18", card)
            self.assertIn("## Metrics", card)
            self.assertIn(model.payload_sha256(), card)

    def test_the_command_needs_something_to_do(self) -> None:
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            tr.main(["--rows", "does-not-matter.jsonl"])


if __name__ == "__main__":
    unittest.main()
