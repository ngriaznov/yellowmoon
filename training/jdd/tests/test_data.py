"""One test group per step of the docs/07 sample generator (steps 1 to 7) plus the loaders."""

import math

import numpy as np
import pytest
import torch

from training.jdd import data
from training.jdd.data import (
    JddDataset,
    PixelShift200,
    PixelShift200Source,
    SamplerConfig,
    SyntheticSource,
    generate_sample,
    merge_pixel_shift,
    mosaic_rggb,
    pack_planes,
    scene_is_validation,
    simulate_pixel_shift,
    unpack_planes,
)
from training.jdd.noise import RANGE_DN, VirtualCamera

CFG = SamplerConfig(patch_half=16)


def grey_gt(level=0.3, n=None):
    n = n or CFG.gt_size
    return np.full((n, n, 3), level)


def quiet_camera(**kw):
    base = dict(iso=100.0, k_dn_per_e=0.25, sigma_read_dn=2.0, tukey_lambda=0.0, sigma_row_dn=0.0, bias_dn=(0, 0, 0, 0))
    base.update(kw)
    return VirtualCamera(**base)


def gen(seed, gt=None, cfg=CFG, **kw):
    rng = np.random.default_rng(seed)
    if gt is None:
        gt, _ = SyntheticSource(cfg.gt_size).gt(rng)
    return generate_sample(gt, rng, cfg, **kw)


# ---- step 1 / output contract ------------------------------------------------------------------
def test_output_shapes_dtypes_and_finite():
    s = gen(1)
    assert s.x.shape == (7, 16, 16) and s.y.shape == (3, 32, 32)
    assert s.x.dtype == s.y.dtype == np.float32
    assert np.isfinite(s.x).all() and np.isfinite(s.y).all()
    assert s.x[6].min() >= 0 and s.x[6].max() <= 1
    assert s.x[4].min() >= 0 and s.x[5].min() >= 0


def test_rejects_bad_gt_shape():
    with pytest.raises(ValueError):
        generate_sample(np.zeros((33, 34, 3)), np.random.default_rng(0), CFG)


def test_deterministic_given_seed_and_dataset_index():
    a, b = gen(5), gen(5)
    assert np.array_equal(a.x, b.x) and np.array_equal(a.y, b.y) and a.meta == b.meta
    ds = JddDataset(10, 42, CFG)
    x1, y1 = ds[3]
    x2, y2 = JddDataset(10, 42, CFG)[3]
    assert torch.equal(x1, x2) and torch.equal(y1, y2)
    assert not torch.equal(ds[3][0], ds[4][0])


# ---- step 2: random exposure and white balance ---------------------------------------------------
def test_step2_gains_and_exposure_ranges():
    metas = [gen(i).meta for i in range(150)]
    gr = np.array([m["g0"][0] for m in metas])
    gb = np.array([m["g0"][2] for m in metas])
    assert gr.min() >= CFG.gain_r_range[0] - 1e-9 and gr.max() <= CFG.gain_r_range[1] + 1e-9
    assert gb.min() >= CFG.gain_b_range[0] - 1e-9 and gb.max() <= CFG.gain_b_range[1] + 1e-9
    assert gr.std() > 0.2
    peaks = np.array([m["peak"] for m in metas if not m["highlights"]])
    assert peaks.max() <= CFG.peak_range[1] and peaks.min() >= CFG.peak_range[0]
    assert np.log2(peaks).std() > 1.0  # exposure varies over many stops


# ---- step 3: RGGB mosaic with random phase --------------------------------------------------------
def test_mosaic_rggb_and_pack_roundtrip():
    rgb = np.stack([np.full((6, 8), 1.0), np.full((6, 8), 2.0), np.full((6, 8), 3.0)], axis=2)
    m = mosaic_rggb(rgb)
    assert m[0, 0] == 1 and m[0, 1] == 2 and m[1, 0] == 2 and m[1, 1] == 3  # R G / G B
    p = pack_planes(m)
    assert p.shape == (4, 3, 4) and (p[0] == 1).all() and (p[1] == 2).all() and (p[2] == 2).all() and (p[3] == 3).all()
    assert np.array_equal(unpack_planes(p), m)


def test_step3_clean_mosaic_is_ground_truth_sampled_through_rggb():
    rng = np.random.default_rng(0)
    gt = rng.random((CFG.gt_size, CFG.gt_size, 3)) * 0.5 + 0.1
    s = gen(0, gt=gt, debug=True)
    py, px = s.meta["phase"]
    h = 2 * CFG.patch_half
    crop = gt[py : py + h, px : px + h]
    # reconstruct what the sensor saw: gt * cast * scale / g0 at the CFA site's colour
    ratio = s.debug["clean_mosaic"] / mosaic_rggb(crop)
    g0 = np.array(s.meta["g0"])
    col = np.array([[0, 1], [1, 2]])
    # ratio = cast_c * exposure_scale / g0_c: constant per CFA colour channel
    for c in range(3):
        sel = ratio[np.tile(col, (h // 2, h // 2)) == c]
        assert sel.std() / sel.mean() < 1e-9
    assert g0.min() >= 1.0


def test_step3_random_phase_covers_all_four_phases():
    phases = {tuple(gen(i).meta["phase"]) for i in range(60)}
    assert phases == {(0, 0), (0, 1), (1, 0), (1, 1)}


# ---- step 4: virtual camera noise ---------------------------------------------------------------------
def test_step4_noise_variance_matches_the_true_model():
    """Empirical variance of the noisy planes on a flat patch equals the model behind the sigma map."""
    cfg = SamplerConfig(patch_half=64, p_clean=0.0, p_craw=0.0, p_12bit=0.0, p_highlight=0.0, sigma_jitter=(1.0, 1.0))
    cam = quiet_camera(k_dn_per_e=2.0, sigma_read_dn=5.0)
    s = gen(3, gt=grey_gt(0.3, cfg.gt_size), cfg=cfg, camera=cam, debug=True)
    dn = s.debug["dn"]
    clean = s.debug["signal_dn"] + cam.black_dn
    res = dn - clean
    keep = ~s.debug["clipped"]
    var_emp = res[keep].var()
    sig_true = s.debug["sigma_true_planes"]  # normalised units * g0
    g0 = np.array(s.meta["g0"])
    gains = np.array([g0[0], 1, 1, g0[2]])[:, None, None]
    sigma_dn = sig_true / gains * RANGE_DN
    # variance model averaged over the 4 planes
    assert abs(var_emp / np.mean(sigma_dn**2) - 1) < 0.15


# ---- step 5: quantisation, cRAW, 12-bit ----------------------------------------------------------------------
def test_step5_mode_probabilities():
    cfg = SamplerConfig(patch_half=8)
    n = 600
    modes = [gen(i, cfg=cfg).meta for i in range(n)]
    clean = np.mean([m["clean"] for m in modes])
    craw_ = np.mean([m["mode"] == "craw" for m in modes])
    b12 = np.mean([m["mode"] == "12bit" for m in modes])
    assert abs(clean - 0.1) < 0.04
    assert abs(craw_ - 0.3 * 0.9) < 0.06  # p = 0.3 among the non-clean samples
    assert abs(b12 - 0.1 * 0.9) < 0.04


def test_step5_craw_and_12bit_properties():
    cfg = SamplerConfig(patch_half=16, p_clean=0.0, p_craw=1.0, p_12bit=0.0)
    s = gen(2, cfg=cfg, debug=True)
    assert s.meta["mode"] == "craw"
    q = s.debug["q_dn"]
    assert q.max() > 1 and (q >= 1).all()  # real qstep map, coarser than 1 DN somewhere
    assert s.x[5].max() > 0
    assert np.array_equal(s.debug["q_input_dn"], q)  # cRAW: the network sees the real step
    cfg12 = SamplerConfig(patch_half=16, p_clean=0.0, p_craw=0.0, p_12bit=1.0)
    s12 = gen(2, cfg=cfg12, debug=True)
    assert s12.meta["mode"] == "12bit"
    assert (s12.debug["dn"] % 4 == 0).all() and (s12.debug["q_dn"] == 4).all()
    assert s12.debug["dn"].max() <= 4 * 4095  # 12-bit ADC codes
    cfg14 = SamplerConfig(patch_half=16, p_clean=0.0, p_craw=0.0, p_12bit=0.0)
    s14 = gen(2, cfg=cfg14, debug=True)
    assert (s14.debug["q_dn"] == 1).all() and (s14.debug["dn"] == np.rint(s14.debug["dn"])).all()


def test_qstep_plane_is_zero_unless_craw_but_sigma_keeps_the_physical_step():
    """docs/05 R6: 'zeros for lossless files'. 14-bit and 12-bit samples feed a zero qstep plane."""
    for kw, mode in (
        (dict(p_craw=0.0, p_12bit=0.0), "14bit"),
        (dict(p_craw=0.0, p_12bit=1.0), "12bit"),
    ):
        cfg = SamplerConfig(patch_half=16, p_clean=0.0, sigma_jitter=(1.0, 1.0), **kw)
        s = gen(6, cfg=cfg, camera=quiet_camera(k_dn_per_e=0.01, sigma_read_dn=0.2), debug=True)
        assert s.meta["mode"] == mode
        assert (s.x[5] == 0).all()
        assert (s.debug["q_input_dn"] == 0).all() and (s.debug["q_dn"] > 0).all()
        # sigma still includes q^2/12 of the real step: with almost no other noise it is ~ q / sqrt(12)
        g0 = np.array(s.meta["g0"])
        gains = np.array([g0[0], 1, 1, g0[2]])[:, None, None]
        q = {"14bit": 1.0, "12bit": 4.0}[mode]
        floor = q / np.sqrt(12.0) / RANGE_DN
        assert (s.debug["sigma_true_planes"] / gains >= floor * 0.999).all()
    craw_cfg = SamplerConfig(patch_half=16, p_clean=0.0, p_craw=1.0, p_12bit=0.0)
    sc = gen(6, cfg=craw_cfg, debug=True)
    assert (sc.x[5] > 0).any()


# ---- step 6: clipping, clip mask, highlights ---------------------------------------------------------------------
def test_step6_clip_rule_uses_qstep_for_craw_and_the_adc_code_for_12bit():
    cfg_c = SamplerConfig(patch_half=16, p_clean=0.0, p_highlight=1.0, p_craw=1.0, p_12bit=0.0)
    sc = gen(4, cfg=cfg_c, debug=True)
    dn, q = sc.debug["dn"], sc.debug["q_dn"]
    assert np.array_equal(sc.debug["clipped"], dn >= 16383 - 4 * q)  # docs/05 R1 with the real qstep
    assert (sc.debug["clipped"] & (dn < 16383)).any()  # the margin catches values just below white
    cfg_12 = SamplerConfig(patch_half=16, p_clean=0.0, p_highlight=1.0, p_craw=0.0, p_12bit=1.0)
    s12 = gen(4, cfg=cfg_12, debug=True)
    assert s12.debug["dn"].max() == 4 * 4095  # saturated 12-bit pixels sit on the top ADC code
    assert np.array_equal(s12.debug["clipped"], s12.debug["dn"] >= 4 * 4095)
    assert s12.debug["clipped"].any()


def test_step6_clip_mask_and_highlights_over_white():
    cfg = SamplerConfig(patch_half=16, p_clean=0.0, p_highlight=1.0, p_craw=0.0, p_12bit=0.0)
    s = gen(4, cfg=cfg, debug=True)
    assert s.meta["highlights"] and s.x[6].mean() > 0
    dn, q = s.debug["dn"], s.debug["q_input_dn"]
    assert dn.max() <= 16383
    clipped = s.debug["clipped"]
    # 14-bit file: no qstep plane, so clipped only at exactly the white level (docs/05 R1)
    assert (q == 0).all() and np.array_equal(clipped, dn >= 16383)
    assert clipped.any() and not clipped.all()
    # the target never exceeds the per-channel white g0 (clipped at the same level as the input)
    g0 = np.array(s.meta["g0"])
    assert (s.y.max(axis=(1, 2)) <= g0 + 1e-6).all()
    # without highlights almost nothing clips
    cfg0 = SamplerConfig(patch_half=16, p_clean=0.0, p_highlight=0.0)
    frac = np.mean([gen(i, cfg=cfg0).x[6].mean() for i in range(20)])
    assert frac < 0.03


# ---- step 7: sigma map -----------------------------------------------------------------------------------------------
def test_step7_sigma_is_true_model_jittered_log_uniform():
    cfg = SamplerConfig(patch_half=16, p_clean=0.0)
    jit, ratios = [], []
    for i in range(80):
        s = gen(i, cfg=cfg, debug=True)
        j = s.meta["sigma_jitter"]
        jit.append(j)
        true = np.sqrt(np.mean(np.square(s.debug["sigma_true_planes"]), axis=0))
        ratios.append(s.x[4] / np.maximum(true, 1e-12))
        assert np.allclose(s.x[4], j * true, rtol=1e-4, atol=1e-9)
    jit = np.array(jit)
    assert jit.min() >= 0.75 - 1e-9 and jit.max() <= 1.25 + 1e-9
    # log-uniform: the log of the factor is uniform on [ln .75, ln 1.25]
    lj = np.log(jit)
    assert abs(lj.mean() - (math.log(0.75) + math.log(1.25)) / 2) < 0.04


def test_step7_clean_samples_have_zero_sigma_and_no_noise():
    cfg = SamplerConfig(patch_half=16, p_clean=1.0)
    s = gen(1, cfg=cfg, debug=True)
    assert s.meta["clean"] and (s.x[4] == 0).all()
    # the "clean mosaic": only 14-bit rounding separates input planes from the noise-free signal
    err = np.abs(s.debug["dn"] - np.clip(s.debug["signal_dn"] + 512, 0, 16383))
    assert err.max() <= 0.5 + 1e-9


def test_sigma_grows_with_signal_and_iso():
    cfg = SamplerConfig(patch_half=16, p_clean=0.0, p_craw=0.0, p_12bit=0.0, sigma_jitter=(1, 1))
    lo = gen(1, gt=grey_gt(0.05, cfg.gt_size), cfg=cfg, camera=quiet_camera(iso=100, k_dn_per_e=0.25))
    hi = gen(1, gt=grey_gt(0.05, cfg.gt_size), cfg=cfg, camera=quiet_camera(iso=6400, k_dn_per_e=16.0))
    assert hi.x[4].mean() > 3 * lo.x[4].mean()


# ---- pixel shift, PixelShift200, split ---------------------------------------------------------------------------
def test_pixel_shift_merge_recovers_the_scene():
    rng = np.random.default_rng(0)
    rgb = rng.random((24, 24, 3))
    frames = simulate_pixel_shift(rgb, data.DEFAULT_PIXEL_SHIFTS)
    merged = merge_pixel_shift(frames, data.DEFAULT_PIXEL_SHIFTS)
    # green is the mean of two samples, red and blue are single samples; compare R and B exactly
    assert np.abs(merged[2:-2, 2:-2, 0] - rgb[2:-2, 2:-2, 0]).max() < 1e-12
    assert np.abs(merged[2:-2, 2:-2, 2] - rgb[2:-2, 2:-2, 2]).max() < 1e-12
    g_expect = rgb[2:-2, 2:-2, 1]
    assert np.abs(merged[2:-2, 2:-2, 1] - g_expect).max() < 1e-12


def test_pixelshift200_layout_loader_and_split(tmp_path):
    rng = np.random.default_rng(0)
    names = [f"scene_{i:03d}" for i in range(60)]
    for n in names[:20]:
        d = tmp_path / n
        d.mkdir()
        np.save(d / "gt.npy", rng.random((80, 90, 3)).astype(np.float32))
    store = PixelShift200(tmp_path)
    assert store.scenes() == names[:20]
    assert store.load_scene(names[0]).shape == (80, 90, 3)
    train = PixelShift200Source(tmp_path, 34, "train", seed=1)
    val = PixelShift200Source(tmp_path, 34, "val", seed=1)
    assert set(train.scene_names).isdisjoint(val.scene_names)  # whole scenes are held out
    assert len(train) + len(val) == 20
    frac = np.mean([scene_is_validation(n) for n in names + [f"x{i}" for i in range(400)]])
    assert abs(frac - 0.1) < 0.04
    gt, desc = train.gt(np.random.default_rng(3))
    assert gt.shape == (34, 34, 3) and desc.startswith("pixelshift200/")
    ds = JddDataset(5, 0, SamplerConfig(patch_half=16), [(train, 1.0)])
    assert ds.sample(0).meta["source"].startswith("pixelshift200/")
    assert PixelShift200(tmp_path / "missing").scenes() == []


def test_scene_split_does_not_depend_on_the_training_seed(tmp_path):
    """The held-out scenes are fixed: runs with other seeds, and the evaluation CLI, must agree."""
    rng = np.random.default_rng(0)
    for i in range(40):
        d = tmp_path / f"scene_{i:03d}"
        d.mkdir()
        np.save(d / "gt.npy", rng.random((40, 40, 3)).astype(np.float32))
    from training.jdd import evaluate as E
    from training.jdd import train as T

    base = T.load_config(T.CONFIG_DIR / "dry_run.json")
    base["data"].update(pixelshift200_root=str(tmp_path), pixelshift200_weight=1.0, synthetic_weight=0.0, patch_half=8)
    splits = []
    for seed in (1, 2, 12345):
        cfg = {**base, "seed": seed}
        ds = T.build_dataset(cfg, "val")
        src = [s for s, _ in ds.sources if hasattr(s, "scene_names")][0]
        splits.append(tuple(src.scene_names))
    assert splits[0] == splits[1] == splits[2] and len(splits[0]) > 0
    # the evaluation CLI helper draws the same validation scenes by default
    cfg_eval = SamplerConfig(patch_half=8)
    ev = PixelShift200Source(tmp_path, cfg_eval.gt_size, split="val", seed=data.SPLIT_SEED)
    assert tuple(ev.scene_names) == splits[0]
    assert len(list(E.pixelshift_pairs(tmp_path, 2, 5, 8))) == 2
    assert set(splits[0]).isdisjoint(PixelShift200Source(tmp_path, cfg_eval.gt_size, split="train").scene_names)


def test_dataset_mixes_sources_by_weight():
    cfg = SamplerConfig(patch_half=8)

    class Const:
        def gt(self, rng):
            return np.full((cfg.gt_size, cfg.gt_size, 3), 0.2), "const"

    ds = JddDataset(200, 0, cfg, [(Const(), 3.0), (SyntheticSource(cfg.gt_size), 1.0)])
    frac = np.mean([ds.sample(i).meta["source"] == "const" for i in range(200)])
    assert 0.65 < frac < 0.85
