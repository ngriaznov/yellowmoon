"""Sources: mask index, instance maps, the sharpness filter, tiles, shards, the split, and `prepare` on a fake volume."""

import gzip
import io
import json
import zipfile

import numpy as np
import pytest
from PIL import Image

from training.bokeh import prepare, sources

from .conftest import blocks_photo, jpeg_of, mask_png

SEG = """MaskPath,ImageID,LabelName,BoxID,BoxXMin,BoxXMax,BoxYMin,BoxYMax,PredictedIoU,Clicks
{a}_m01_aa.png,{a},/m/01,aa,0,1,0,1,0.9,
{a}_m02_bb.png,{a},/m/02,bb,0,1,0,1,0.3,
{b}_m01_cc.png,{b},/m/01,cc,0,1,0,1,0.00000,
{c}_m01_dd.png,{c},/m/01,dd,0,1,0,1,0.8,
"""


def test_mask_index_keeps_the_wanted_images_and_drops_poor_predictions():
    a, b, c = "0a" * 8, "1b" * 8, "2c" * 8
    idx = sources.mask_index(io.StringIO(SEG.format(a=a, b=b, c=c)), {a, b})
    assert set(idx) == {a, b}
    assert [m["path"] for m in idx[a]] == [f"{a}_m01_aa.png"]  # the 0.3 predicted IoU is dropped
    assert idx[b][0]["label"] == "/m/01"  # a hand-drawn mask has IoU 0 and stays


def test_instance_map_paints_small_instances_over_big_ones_and_resizes():
    big = np.zeros((40, 60), bool)
    big[5:35, 5:55] = True
    small = np.zeros((20, 30), bool)  # a mask of another size: resized to the photo's
    small[5:12, 8:20] = True
    ids = sources.instance_map((60, 40), [mask_png(big), mask_png(small)])
    assert ids.shape == (40, 60) and ids.dtype == np.uint8
    assert ids[20, 50] == 1 and ids[14, 24] == 2 and ids[0, 0] == 0  # big is id 1, the smaller is painted on top as 2


def test_the_sharpness_filter_keeps_sharp_tiles_and_drops_soft_photos():
    sharp = blocks_photo(200, 150, 3)
    soft = blocks_photo(200, 150, 3, soft=2.5)
    got, rep = sources.photo_tiles(jpeg_of(sharp), "a" * 16, [], tile=64, min_long_edge=100)
    assert rep["reason"] == "ok" and 1 <= len(got) <= sources.TILES_PER_PHOTO
    none, rep = sources.photo_tiles(jpeg_of(soft), "b" * 16, [], tile=64, min_long_edge=100)
    assert none == [] and rep["reason"] == "soft_or_flat"
    small, rep = sources.photo_tiles(jpeg_of(sharp), "c" * 16, [], tile=64)  # the real minimum: 700 px
    assert small == [] and rep["reason"] == "small"
    for t in got:
        assert t["meta"]["sharp"] >= sources.SHARP_MIN and t["meta"]["soft"] <= sources.SOFT_MAX
    if len(got) == 2:  # two tiles of one photo do not overlap by more than half
        a, b = got[0]["meta"], got[1]["meta"]
        assert abs(a["x"] - b["x"]) >= 32 or abs(a["y"] - b["y"]) >= 32


def test_candidates_are_instances_that_are_mostly_inside_the_tile_and_not_tiny_or_huge():
    full = np.zeros((100, 100), np.uint8)
    full[10:50, 10:50] = 1      # inside, 16 % of the tile
    full[10:20, 60:100] = 2     # half in the tile
    full[70:72, 70:72] = 3      # tiny
    full[0:100, 0:60] = np.where(full[0:100, 0:60] == 0, 4, full[0:100, 0:60])  # huge
    tile = full[:, :70].copy()
    got = sources.candidate_ids(tile, full)
    assert got == [1]


def test_shards_read_back_what_was_written_and_the_split_is_by_photo(tile_folder):
    ts = sources.TileSet(tile_folder, val_share=0.3)
    assert ts.tile == 64 and len(ts) >= 12
    rgb, ids, cands, hd = ts.get(0)
    assert rgb.shape == (64, 64, 3) and rgb.dtype == np.uint8 and ids.shape == (64, 64) and cands.shape == (sources.MAX_CANDS,)
    assert abs(hd - np.hypot(100, 100) / 2) < 1e-3  # stored to three decimals
    r = ts.rows[0][1]
    again = Image.open(io.BytesIO(bytes(np.memmap(tile_folder / "shard-00000.bin", dtype=np.uint8, mode="r")[r["jpg"][0]:r["jpg"][0] + r["jpg"][1]])))
    assert np.array_equal(np.asarray(again.convert("RGB")), rgb)
    by_photo = {}
    for i in ts.val:
        by_photo.setdefault(ts.rows[i][1]["photo"], []).append(i)
    assert not set(ts.train) & set(ts.val) and len(ts.train) + len(ts.val) == len(ts)
    for photo, tiles in by_photo.items():  # both tiles of a photo are validation tiles
        assert all(sources.is_val(ts.rows[i][1]["photo"], 0.3) for i in tiles)
    assert all(ts.rows[i][1]["cands"] == [1] for i in range(len(ts)))
    assert ids.max() == 1 and cands[0] == 1 and cands[1] == 0


def test_the_split_is_stable_and_about_the_share():
    ids = [f"{i:016x}" for i in range(4000)]
    v = [sources.is_val(i) for i in ids]
    assert 0.02 < sum(v) / len(v) < 0.07
    assert v[:2000] == [sources.is_val(i) for i in ids[:2000]]


# --- prepare, on a fake pair of volumes ---------------------------------------------------------------------------


def make_volumes(tmp_path, n=5):
    crop = tmp_path / "crop"
    images = crop / "data" / prepare.CROP_SET / "images"
    images.mkdir(parents=True)
    manifest, seg = [], ["MaskPath,ImageID,LabelName,BoxID,BoxXMin,BoxXMax,BoxYMin,BoxYMax,PredictedIoU,Clicks"]
    zipf = tmp_path / "masks-0.zip"
    with zipfile.ZipFile(zipf, "w") as z:
        for i in range(n):
            pid = f"0{i:015x}"
            (images / f"{pid}.jpg").write_bytes(jpeg_of(blocks_photo(200, 150, i)))
            manifest.append({"id": pid, "subset": "train", "licence": sources.CC_BY_2, "author": f"A{i}", "author_url": "u",
                             "source_url": f"https://flickr/{i}", "title": f"t\t{i}"})
            if i % 2 == 0:
                m = np.zeros((150, 200), bool)
                m[30:70, 40:80] = True
                name = f"{pid}_m01_ab.png"
                z.writestr(name, mask_png(m))
                seg.append(f"{name},{pid},/m/01,ab,0,1,0,1,0.9,")
    # an image listed under another licence is not used
    manifest.append({"id": "0" + "f" * 15, "subset": "train", "licence": "https://creativecommons.org/licenses/by-nc/2.0/",
                     "author": "x", "author_url": "u", "source_url": "s", "title": "nc"})
    (images / ("0" + "f" * 15 + ".jpg")).write_bytes(jpeg_of(blocks_photo(200, 150, 99)))
    (crop / "data" / prepare.CROP_SET / "manifest.json").write_text(json.dumps(manifest))
    seg_path = tmp_path / "seg.csv"
    seg_path.write_text("\n".join(seg) + "\n")
    return crop, tmp_path / "vol", seg_path, zipf


def test_prepare_end_to_end_on_a_fake_volume(tmp_path):
    crop, vol, seg_path, zipf = make_volumes(tmp_path)
    assert len(prepare.crop_photos(crop)) == 5  # the by-nc row and nothing else is left out
    summary = prepare.build_index(crop, vol, 0, 0, 1, lambda: sources.stream_lines(str(seg_path)))
    assert summary["photos"] == 5 and summary["with_masks"] == 3 and summary["masks"] == 3 and summary["chunks"] == 1
    assert prepare.build_index(crop, vol, 0, 0, 1, lambda: sources.stream_lines(str(seg_path))) == summary  # idempotent
    assert prepare.known_summary(vol, 0, 0, 1) == summary and prepare.known_summary(vol, 2, 0, 1) is None
    with pytest.raises(RuntimeError, match="other parameters"):
        prepare.build_index(crop, vol, 2, 0, 1, lambda: sources.stream_lines(str(seg_path)))
    assert prepare.mask_archive(vol, "0", local_zip=zipf) == 3
    assert prepare.mask_archive(vol, "0", local_zip=zipf) == 3  # done: skipped
    res = prepare.tile_chunk(vol, crop, 0, workers=1, tile=64, min_long_edge=100)
    assert res["tiles"] >= 5 and not res["skipped"]
    again = prepare.tile_chunk(vol, crop, 0, workers=1, tile=64, min_long_edge=100)
    assert again["skipped"] and again["tiles"] == res["tiles"]
    final = prepare.finalize(vol)
    assert final["photos"] == 5 and final["tiles_with_instances"] >= 1
    ts = sources.TileSet(vol / "tiles")
    assert len(ts) == res["tiles"] and ts.tile == 64
    rows = gzip.decompress((vol / "tiles" / "credits.tsv.gz").read_bytes()).decode().splitlines()
    assert rows[0].split("\t") == ["image_id", "use", "author", "author_url", "source_url", "title"]
    assert len(rows) == 6 and all(r.split("\t")[1] == "bokeh" and len(r.split("\t")) == 6 for r in rows[1:])
    assert "t 0" in rows[1]  # a tab inside a title does not break the file
    again_bytes = (vol / "tiles" / "credits.tsv.gz").read_bytes()
    prepare.finalize(vol)
    assert (vol / "tiles" / "credits.tsv.gz").read_bytes() == again_bytes  # the same bytes


def test_a_photo_that_cannot_be_read_is_reported_not_fatal(tmp_path):
    crop, vol, seg_path, zipf = make_volumes(tmp_path, n=3)
    (crop / "data" / prepare.CROP_SET / "images" / f"{0:016x}.jpg").write_bytes(b"not a jpeg")
    prepare.build_index(crop, vol, 0, 0, 1, lambda: sources.stream_lines(str(seg_path)))
    prepare.mask_archive(vol, "0", local_zip=zipf)
    res = prepare.tile_chunk(vol, crop, 0, workers=1, tile=64, min_long_edge=100)
    assert res["reasons"].get("error: UnidentifiedImageError") == 1 and res["tiles"] >= 2


def test_probe_reads_a_sample_of_the_volume(tmp_path):
    crop, *_ = make_volumes(tmp_path)
    (crop / "pre" / prepare.PRE_SET).mkdir(parents=True)
    (crop / "pre" / prepare.PRE_SET / "manifest.json").write_text("[]")
    out = prepare.probe(crop, n=4, tile=64, min_long_edge=100)
    assert out["photos_on_volume"] == 5 and out["sampled"] == 4 and out["pass_rate"] > 0.5 and out["seconds_per_photo"] > 0
    assert out["pretraining_manifest"] is True
