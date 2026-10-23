"""Unit tests of the crop pilot's pieces that run without a GPU, the network or the teacher."""

import numpy as np
import pytest
import torch
from PIL import Image

from training.crop import budget, data, openimages
from training.crop.data import letterbox
from training.crop.model import CropNet, iou, parameters
from training.crop.predict import closest, frames
from training.crop.train import DEFAULTS, split, train


def row(i, licence=openimages.CC_BY_2, rotation="0.0"):
    return {"ImageID": f"{i:016x}", "Subset": "train", "License": licence, "Author": "a", "AuthorProfileURL": "u",
            "OriginalLandingURL": "l", "Title": "t", "Rotation": rotation}


def test_select_keeps_upright_cc_by_rows_and_is_seeded():
    rows = [row(i) for i in range(50)] + [row(100, licence="https://creativecommons.org/licenses/by-nc/2.0/"),
                                          row(101, rotation="90.0"), row(102, rotation="")]
    a = openimages.select(rows, 10, seed=3)
    assert a == openimages.select(list(reversed(rows)), 10, seed=3)
    assert len(a) == 10 and [r["id"] for r in a] == sorted(r["id"] for r in a)
    assert all(int(r["id"], 16) < 50 for r in a)
    assert openimages.select(rows, 10, seed=4) != a


def test_letterbox_centres_the_photo_and_reports_its_box():
    pix, box = letterbox(Image.new("RGB", (400, 200), (255, 255, 255)), 256)
    assert pix.shape == (256, 256, 3)
    assert box.tolist() == [0.0, 0.25, 1.0, 0.75]
    assert pix[0, 0].tolist() == [0, 0, 0] and pix[128, 128].tolist() == [255, 255, 255]


def test_the_student_answers_valid_boxes_and_starts_large():
    model = CropNet().eval()
    assert parameters(model) < 2_000_000
    with torch.no_grad():
        out = model(torch.rand(3, 3, 256, 256))
    assert out.shape == (3, 4)
    assert torch.all(out[:, 0] < out[:, 2]) and torch.all(out[:, 1] < out[:, 3])
    assert torch.all(out >= 0) and torch.all(out <= 1)
    whole = torch.tensor([[0.0, 0.0, 1.0, 1.0]]).expand(3, 4)
    assert torch.all(iou(out, whole) > 0.75)


def test_iou():
    a = torch.tensor([[0.0, 0.0, 0.5, 0.5], [0.0, 0.0, 1.0, 1.0]])
    b = torch.tensor([[0.25, 0.0, 0.75, 0.5], [0.0, 0.0, 1.0, 1.0]])
    assert iou(a, b).tolist() == pytest.approx([1 / 3, 1.0])


@pytest.mark.parametrize("w,h,ratio", [(1200, 800, 1.0), (1200, 800, 16 / 9), (800, 1200, 0.8), (1200, 800, 1.5)])
def test_aspect_frames_have_the_ratio_and_stay_inside(w, h, ratio):
    cands = frames(w, h, ratio)
    assert np.all(cands >= -1e-12) and np.all(cands <= 1 + 1e-12)
    np.testing.assert_allclose((cands[:, 2] - cands[:, 0]) * w / ((cands[:, 3] - cands[:, 1]) * h), ratio)
    pick = closest(np.array([0.5, 0.2, 0.9, 0.8]), cands)
    assert any(np.array_equal(pick, c) for c in cands)


def test_the_validation_split_is_stable_and_about_a_tenth():
    ids = [f"{i:016x}" for i in range(2000)]
    tr, va = split(ids)
    assert len(tr) + len(va) == 2000 and 150 < len(va) < 250
    _, va_small = split(ids[:1000])
    assert set(va_small) == set(int(i) for i in va if i < 1000)


def test_the_ledger_adds_the_margin_and_stops_counting_nothing():
    ledger = budget.Ledger(20.0)
    l4 = budget.rate(True, 2, 8)
    assert l4 == pytest.approx(0.000222 + 2 * 0.0000131 + 8 * 0.00000222)
    cost = ledger.charge("an hour of one L4", 3600, l4)
    assert cost == pytest.approx(3600 * l4 * budget.MARGIN) and ledger.left() == pytest.approx(20.0 - cost)
    assert ledger.report()["items"] == [("an hour of one L4", round(cost, 4))]


def test_labelling_stops_where_the_money_or_the_time_runs_out():
    l4 = budget.rate(True, 2, 8)
    # Plenty of both: everything wanted, even when it is not a whole number of chunks.
    assert budget.photos_to_label(20, 4, 1.0, l4, 19_750, 6, 3.5, 250) == 19_750
    # Money: $5 left after the reserve at 4 s a photo pays for 5 / (4 * l4 * 1.25) photos, in whole chunks.
    n = budget.photos_to_label(9, 4, 4.0, l4, 19_750, 6, 100, 250)
    assert n % 250 == 0 and n <= 5 / (4 * l4 * budget.MARGIN) < n + 250
    # Time: 6 GPUs for one hour at 3 s a photo.
    assert budget.photos_to_label(1000, 0, 3.0, l4, 19_750, 6, 1, 250) == 7_000
    # Nothing left after the reserve.
    assert budget.photos_to_label(3, 4, 1.0, l4, 19_750, 6, 3.5, 250) == 0


def test_the_training_reserve_covers_every_run_at_its_cap():
    t = budget.rate(True, 4, 16)
    assert budget.training_reserve(3, 40, t) == pytest.approx(3 * 50 * 60 * t * budget.MARGIN)


def test_training_stops_at_max_minutes_with_the_best_epoch_saved(tmp_path):
    rng = np.random.default_rng(0)
    images = tmp_path / "images"
    images.mkdir()
    labels = {}
    for i in range(24):
        Image.fromarray(rng.integers(0, 255, (48, 64, 3), dtype=np.uint8)).save(images / f"p{i:02d}.jpg")
        labels[f"p{i:02d}"] = {"crop": [0.1, 0.1, 0.9, 0.8]}
    data.build(images, labels, tmp_path / "arrays")
    config = {**DEFAULTS, "epochs": 1000, "batch": 8, "device": "cpu", "max_minutes": 0.01}
    m = train(tmp_path / "arrays", tmp_path / "run", config)
    assert 1 <= m["epochs_run"] < 1000 and (tmp_path / "run" / "model.pt").exists()
