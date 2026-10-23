"""Unit tests of the crop pilot's pieces that run without a GPU, the network or the teacher."""

import numpy as np
import pytest
import torch
from PIL import Image

from training.crop import openimages
from training.crop.data import letterbox
from training.crop.model import CropNet, iou, parameters
from training.crop.predict import closest, frames
from training.crop.train import split


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
