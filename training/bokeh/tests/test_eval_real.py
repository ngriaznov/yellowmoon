"""The local evaluation on research-only blur sets: the rules and the counts, on a made-up photo (no data, no network)."""

import numpy as np
import pytest
import torch
from PIL import Image

from training.bokeh import eval_real as E
from training.bokeh.model import BokehNet


def test_photos_with_either_case_of_the_extension_are_loaded_in_name_order(tmp_path):
    (tmp_path / "image").mkdir()
    (tmp_path / "gt").mkdir()
    for name, ext in (("out_of_focus0002", ".JPG"), ("out_of_focus0001", ".jpg"), ("motion0001", ".jpg")):
        Image.fromarray(np.full((20, 30, 3), 90, np.uint8)).save(tmp_path / "image" / f"{name}{ext}")
        Image.fromarray(np.full((20, 30), 255, np.uint8)).save(tmp_path / "gt" / f"{name}.png")
    items = E.load_cuhk(tmp_path, 10)
    assert [n for n, _, _ in items] == ["out_of_focus0001", "out_of_focus0002"]  # the motion photo is not used
    assert items[0][1].shape == (20, 30, 3) and items[0][2].dtype == bool and bool(items[0][2].all())
    assert len(E.load_cuhk(tmp_path, 1)) == 1


def test_the_mirrored_pass_is_combined_with_the_mean_radius_and_a_scale_that_grows_with_their_disagreement():
    mu_a, mu_b = np.array([[1.0, 2.0]]), np.array([[1.0, 3.0]])
    s = np.log(np.full((1, 2), 0.2))
    mu, ls = E.combine_mirrored((mu_a, s), (mu_b, s))
    assert mu.tolist() == [[1.0, 2.5]]
    assert np.exp(ls)[0].tolist() == pytest.approx([0.2, 0.7])  # 0.2 where they agree, 0.2 + half a unit where they differ by one


def test_every_rule_gives_a_map_of_the_photos_size_and_the_probability_rules_nest():
    torch.manual_seed(0)
    model = BokehNet().eval()
    rgb = (np.random.default_rng(0).random((70, 90, 3)) * 255).astype(np.uint8)
    got = E.predict(model, rgb, tta=True)
    maps = E.rules(got)
    assert set(maps) >= {"raw", "gate c>=0.5", "P>=0.5", "P>=0.7", "P>=0.9", "quantile", "tta raw", "tta P>=0.7", "classical raw", "classical gate c>=0.5"}
    assert all(m.shape == (70, 90) and m.dtype == bool for m in maps.values())
    assert bool((maps["P>=0.9"] <= maps["P>=0.7"]).all()) and bool((maps["P>=0.7"] <= maps["P>=0.5"]).all())  # a higher bar calls fewer pixels
    assert bool((maps["gate c>=0.5"] <= maps["raw"]).all())
    # the lower-quantile rule is the probability rule at 0.9, written out
    assert float((maps["quantile"] != maps["P>=0.9"]).mean()) < 0.002


def test_the_counts_give_precision_recall_f1_and_the_spread_over_photos():
    seg = E.Seg()
    gt = np.zeros((10, 10), bool)
    gt[:5] = True
    seg.add(gt.copy(), gt)  # a perfect photo
    wrong = np.zeros((10, 10), bool)
    wrong[:, :5] = True
    seg.add(wrong, gt)  # half right
    r = seg.result()
    assert r["accuracy"] == pytest.approx(0.75) and r["precision"] == pytest.approx(0.75) and r["recall"] == pytest.approx(0.75)
    assert r["f1_photo_mean"] == pytest.approx((1.0 + 0.5) / 2) and r["f1_photo_std"] == pytest.approx(0.25) and r["photos"] == 2
    empty = E.Seg()
    empty.add(np.zeros((4, 4), bool), np.zeros((4, 4), bool))  # a photo with nothing to find has no F1 of its own
    assert "f1_photo_mean" not in empty.result()


def test_the_segmentation_report_reads_the_polarity_from_the_gradients():
    torch.manual_seed(1)
    model = BokehNet().eval()
    rng = np.random.default_rng(1)
    rgb = np.full((64, 96, 3), 128, np.uint8)
    rgb[:, 48:] = (rng.random((64, 48, 3)) * 255).astype(np.uint8)  # the right half is textured: it is the sharp one
    mask = np.zeros((64, 96), bool)
    mask[:, 48:] = True  # white = the textured half
    rep = E.evaluate_segmentation(model, [("a", rgb, mask), ("b", rgb, mask)], 0, tta=False)
    assert rep["mask_polarity"] == "white = in focus" and rep["images"] == 2
    assert 0.45 < rep["out_of_focus_share"] < 0.55 and "P>=0.7" in rep and "classical raw" in rep
