# 18 · Automatic settings (`autoset`)

Owner's direction (2026-10-18): the automatic settings should understand the scene and set each photo the way a professional photographer would, so an everyday photo looks punchy and finished when it opens, clearly better than Lightroom Auto and the camera's JPEG. Research: `research/ml-auto-settings-2026-10-18.md`. Decisions of the same day (`docs/16`): **full rescue** (Auto ignores the camera's exposure and any dialled compensation), **contrast becomes a bounded per-photo Auto setting**, the **G3 gate applies to normal scenes only and acceptance is a blind A/B**, and **scene understanding comes from our own encoder** labelled offline by Apache-2.0 vision-language models.

This spec covers version 1 (`autoset` v1: exposure and contrast from a small learned model, temperature from a preferred-white rule) and lists what follows. Specs win over the research where they differ.

## 1. What changes for the user

| Setting | Before | With `autoset` v1 |
|---|---|---|
| Exposure (Auto) | the camera's exposure: a constant `CAMERA_GAIN_EV` +1.11 for every photo | the model chooses where the photo's key sits, from what is in the frame and how bright the scene was; a photo shot a stop under comes out as if exposed right (full rescue), and no photo comes out darker than the camera's own JPEG (the camera floor, §4.4) |
| Temperature and tint (Auto) | the estimated light, fully neutralised | the **preferred white**: daylight stays neutral; warm light (tungsten, sunsets) and dim scenes keep part of their warmth (§5) |
| Contrast | a plain number, default 0 | **Auto or Manual**, like Exposure: Auto is the model's bounded per-photo contrast; moving the slider makes it Manual; Reset returns to Auto |
| White lift | per photo, sized at the photo's exposure | unchanged; it is sized at the new exposure |
| Highlight guard, burst sharing | as `docs/02` | unchanged; burst sharing also applies to Auto contrast |

When the model file is missing or cannot be read, every value is the behaviour before `autoset`. An unsure contrast fades to 0; an unsure exposure still applies (full rescue, §4.4), and its confidence is reported. `PIPELINE_VERSION` goes to 0.5.0.

## 2. Where it runs

Once per photo, on the CPU, in `AutoEstimates::compute_with_look` (`crates/ym-core/src/auto.rs`), after the illuminant estimate and the exposure meter and **before** the white-lift estimate, so the lift is sized at the new exposure. Everything is f64, single-threaded, in a fixed order: the same file gives the same bytes on every machine (AGENTS rule 8); the app and `ym` share the code (rule 6). Cost budget: features ≤ 40 ms on the 1/8 preview of a 61 MP file (order statistics with `select_nth_unstable_by`, never a full sort), inference ≤ 1 ms.

Code: `ym_color::autoset` (pure functions over plain buffers: `features`, `model`, `apply`, `preferred_white`), wired in `ym-core` (`auto.rs`, `settings.rs`, `pipeline/color.rs`).

## 3. Inputs: the feature vector (v1, 39 values)

Built from the **metering preview** (`Preview::metered(g0)`: the 1/8 preview, `g0`-balanced, vignetting gain applied) through the fused C1–C6 matrix at EV 0 (`matrix_ev0`, as `auto_exposure` uses it). For every pixel `i` with finite channels: `Y_i` = the AWG4 `Y` row applied to the pixel (as `auto_exposure` computes it), floored at 1e-6; `s_i = log2(Y_i / 0.18)` (stops over middle grey at EV 0). `M_i = max(R, G, B)` of the AWG4 pixel, floored at 1e-6; `t_i = log2(M_i / 0.18)`. Clipped pixels are **included** in the quantiles (they are the highlights); non-finite pixels are skipped. The **key** `m = log2(mean_luminance / 0.18)` with `mean_luminance` the centre-weighted geometric mean that `auto_exposure` already reports (`AutoExposure::mean_luminance`); when the meter has no usable pixel, autoset is not run (fallback).

Quantiles are the linearly interpolated quantiles of `docs/02` ("Percentiles"), computed with selection, never a full sort.

| Index | Name | Definition |
|---|---|---|
| 0 | `key_bv` | `Bv_settings + m`, with `Bv_settings = log2(N²/t) − log2(ISO/100)` (`ShotInfo::bv`); 0 when unknown |
| 1 | `bv_known` | 1 when `Bv_settings` is known, else 0 |
| 2–18 | `q01 … q999` | quantiles of `s − m` at p = 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.98, 0.99, 0.995, 0.999 |
| 19–23 | `mq50 … mq999` | quantiles of `t − m` at p = 0.50, 0.90, 0.98, 0.995, 0.999 |
| 24 | `clip_headroom` | `log2(clip_y / 0.18) − m`, `clip_y` the AWG4 `Y` of the clipped neutral at EV 0 (`FusedChain::clip()` at EV 0) |
| 25 | `clipped_share` | share of preview pixels whose clip mask is 1 |
| 26 | `centre_minus_key` | mean of `s` over the central third (both axes) minus `m` |
| 27 | `top_minus_bottom` | mean of `s` over the top third minus over the bottom third (after orientation is **not** applied: sensor orientation) |
| 28 | `bright_share` | share of pixels with `s − m > 2` |
| 29 | `dark_share` | share of pixels with `s − m < −3` |
| 30–33 | `p_daylight … p_led` | the scene light's class probabilities (daylight, tungsten, fluorescent, LED) |
| 34 | `mired` | `1e6 / cct_k / 100` of the scene light |
| 35 | `duv` | the scene light's Duv × 100 |
| 36 | `light_confidence` | the scene light's confidence |
| 37 | `log_iso` | `log2(ISO / 100)`, 0 when unknown |
| 38 | `flash` | 1 when EXIF Flash says the flash fired, else 0 |

**Equivariance (full rescue by construction).** Scaling the raw data by 2^k (with the exposure settings changed by k, so the scene is the same) shifts every `s`, `t` and `m` by k: features 2–23 and 26–29 are unchanged, `key_bv` is unchanged (`Bv_settings` moves by −k, `m` by +k), only `clip_headroom` moves by −k (less headroom when brighter), as it physically does. This holds exactly while no pixel newly clips: the key is the meter's mean over unclipped pixels, so a frame brightened until its highlights clip moves `m` by less than k (the training labels account for it, §6.3). The model predicts where the key belongs, not a gain, so the result is the same however the camera exposed. Exposure compensation is **not** an input (owner's decision: full rescue).

## 4. The model

### 4.1 Outputs

Two heads, each a mean and a log-variance:

- `key_target`: where the key should sit after exposure, in stops over grey. The exposure the model asks for is `EV_model = key_target − m`.
- `contrast`: the C9 contrast (−100…100 units of `DevelopSettings::contrast`, `docs/02` C9), bounded to ±`CONTRAST_LIMIT` = 30 in v1.

### 4.2 Architecture

An ensemble of **5** multilayer perceptrons, each `39 → 32 → 32 → 4` with `tanh` hidden units (outputs: key mean, key log-variance, contrast mean, contrast log-variance); inputs standardised with the stored mean and standard deviation. About 2,500 parameters per member. Combination: the mean of the members' means; the variance is the mean of the members' variances plus the variance of their means (the mixture variance).

### 4.3 File format (`data/models/autoset/autoset.json`)

```json
{
  "format": "yellowmoon-autoset",
  "format_version": 1,
  "model_version": "1.0.0",
  "feature_names": ["key_bv", "bv_known", "..."],
  "input_mean": [39 numbers], "input_std": [39 numbers],
  "activation": "tanh",
  "outputs": ["key_target", "key_logvar", "contrast", "contrast_logvar"],
  "prior_sigma": {"key_target": 0.75, "contrast": 10.0},
  "contrast_limit": 30.0,
  "members": [ {"layers": [ {"w": T, "b": T}, {"w": T, "b": T}, {"w": T, "b": T} ]}, ... 5 ],
  "sha256": "hex of the concatenated little-endian f32 bytes of every tensor, members in order, layers in order, w before b"
}
```

`T` = `{"shape": [rows, cols], "data_b64": base64 of little-endian f32, row-major}` (the `illum` convention, `crates/ym-color/src/illum/format.rs`). `w` has shape `[out, in]`, `b` shape `[out]`; the SHA-256 covers the member tensors only (not the input statistics or the constants); a layer computes `y = W·x + b` in **f64** from the f32 values. The reader refuses a wrong `format`, `format_version`, feature count, shape or SHA-256, and any non-finite value (`AutosetError`). Weights trained on CC BY-SA data are CC BY-SA 4.0 with their own `LICENSE` in the folder (`docs/14` rule 5); v1 is trained on CC0 data only and is Apache-2.0.

### 4.4 Applying it

With the ensemble mean `μ` and variance `σ²` of a head and its prior sigma `σ0`, the weight `w = σ0² / (σ0² + σ²)` (in (0, 1]: 1 for a certain prediction, towards 0 for a doubtful one):

- **Exposure: the model's, never darker than the camera's JPEG.** `EV_model = clamp(μ_key − m, −EV_LIMIT, +EV_LIMIT)`; with the camera's exposure `L` of the photo's embedded JPEG (§6.1, computed when the photo opens, on the CPU, the same code as the dataset dump), `EV_auto_before_guard = max(EV_model, L)`; when the photo has no embedded JPEG or no label, `EV_model`. Then the highlight guard (`docs/02` step 5) and the burst median (step 6) as before. The weight `w_ev` is reported, not applied: full rescue (owner's decision of 2026-10-18, confirmed after the first evaluation, §8) means the exposure never falls back towards the camera's own. (The first draft shrank the exposure towards `CAMERA_GAIN_EV`; the evaluation showed that this keeps half of a misexposed frame's error, 0.5 to 0.9 EV at ±2 EV, while the model alone holds 0.25 EV at any offset. Shrinking towards the typical key instead did not help either.)
- **The camera floor** (owner's decision of 2026-10-18, after the first blind A/B, §8 gate 5): against the camera JPEG the owner chose `autoset` in 35 to 47 % of trials where it was darker than the camera's exposure `L` and in 56 to 66 % where it was brighter, while against the rule neither direction predicted the choice. On the 121 photos of that test the floor raises 36 of them, by a median of 0.3 EV (at most 1.2), the photos where `autoset` won 43 % against the camera. It takes nothing from the rescue of an under-exposed frame (its JPEG is dark too, so `L` is low); an over-exposed frame comes down only as far as the camera's JPEG and the highlight guard take it, which matches the answers (a +1.3 EV high-key sea was preferred as the camera made it). `AutoSummary` reports `camera_ev` (`L`) and `camera_floor` (whether it raised the exposure).
- `contrast_auto = clamp(w_c · μ_contrast, −contrast_limit, +contrast_limit)`, rounded to 0.1; frames of a burst share the median. A doubtful contrast fades to 0.
- Both weights are reported (`AutoSummary`): `ev_confidence = w_ev`, `contrast_confidence = w_c`.

No model, a model that fails to load, a prediction that is not finite, or no usable preview pixel: the rule (`CAMERA_GAIN_EV`, contrast 0, both weights 0). The value written into exports (`ym:Settings` XMP) records `autoset` = the model version or `"none"`.

## 5. Preferred white (rule, v1)

Auto white balance sets the **user white** of the chain (C4–C5) to a preferred white derived from the scene light; the scene light itself (C1–C3, the spectral sensor map) is never changed (`docs/02`, scene light versus user white). Only when white balance is Auto and the light source is Auto.

With the scene light's `mired_s = 1e6 / cct_k`, `duv_s`, the reference `mired_ref = 1e6 / 6000` (a surface neutral, Zhai and Luo 2017) and the scene key brightness `B = key_bv` (§3):

```
warm  = max(0, mired_s − mired_ref)                       mired warmer than the reference
D_w   = 1 − R_warm · smoothstep(warm / W_span)            R_warm = 0.30, W_span = 150 mired
D_low = 1 − R_dim · smoothstep((B_dim − B) / B_span)      R_dim = 0.15, B_dim = 2, B_span = 4 (only when key_bv is known)
D     = clamp(D_w · D_low, 0.5, 1)
mired_u = mired_ref + D · (mired_s − mired_ref)   when mired_s > mired_ref, else mired_s
duv_u   = duv_s                                    (tint casts are neutralised fully)
```

`smoothstep(x) = 0` for x ≤ 0, `3x² − 2x³` on (0, 1), 1 for x ≥ 1. With `D = 1` the photo is fully neutralised (as before); `D < 1` balances for a cooler light than the scene's, so part of the warm cast stays. At 3000 K in a bright scene `D = 0.70`: the photo is balanced for about 3530 K (`mired_u` = 283). The constants are a starting point for the owner's blind test ([HUMAN] AS-H2); the setting `white_preference` (0 … 1, default 1) scales `1 − D` (0 = fully neutral, the old behaviour). The temperature readout shows the user white.

A learned preferred-white map (Cube++ neutral against camera-AWB pairs, about 600 parameters) replaces the constants in a later version (§9).

## 6. Labels and training (`training/autoset/`)

**Data (v1).** CC0 Sony ARWs from raw.pixls.us and RawDB (196 + 211 listed on 2026-10-18, overlapping: 228 distinct files from 99 bodies, 8.5 GiB, deduplicated by SHA-256; manifest with source, licence and checksum in `tools/fetch/autoset/manifest.json`; never committed raw files). Files are streamed: download, dump, delete (disk). The 12 local photos and `testdata/readme-photos` are **evaluation only** (licences unclear or the owner's own; `docs/test-photo-sources.md`).

**Dump** (`cargo run -p ym-core --release --example autoset_dump -- <files> > rows.jsonl`): one JSON line per file with the 39 features, `m`, the camera's embedded JPEG statistics for the label fits, metadata (DRO mode, creative style, flash, metering mode, exposure compensation, body), and the contrast-fit table (§6.2); plus one augmentation line per k ∈ {−2, −1, −0.5, +0.5, +1, +2}: the same file scaled by 2^k with re-clipping at the sensor clip, with its own features (its exposure label is the file's `L − k`, §6.3).

**Row schema** (one JSON object per line; `aug_k = 0` rows carry the label tables, augmentation rows do not):

```json
{"file": "Sony - ILCE-7M4 - 14bit (3:2).ARW", "sha256": "…", "source": "raw.pixls.us", "licence": "CC0-1.0",
 "body": "ILCE-7M4", "capture_time": "2024-05-01T10:12:33", "aug_k": 0.0,
 "features": [39 numbers], "key": -1.234,
 "meta": {"dro": "off", "creative_style": "standard", "hdr": "off", "flash": 0, "metering": "multi",
          "ec": 0.0, "iso": 100, "exposure_time_s": 0.004, "f_number": 5.6},
 "g3": {"grid": "linspace(0.001, 0.999, 999)", "scene_q": [999 numbers], "jpeg_q": [999 numbers]},
 "label_ev": 1.08,
 "contrast_fit": {"grid": [-40, -30, -20, -10, 0, 10, 20, 30, 40],
                  "p": [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95],
                  "ours": [[7 display-luma quantiles] for each grid value], "jpeg": [7 numbers]}}
```

`scene_q` are quantiles of our scene-linear luminance at EV 0 (AWG4 `Y` of the metered preview through the metering chain) and the JPEG's at a common size: when the aspect ratios differ by more than 2 % the JPEG is first cropped centrally to the preview's (letterbox bars), then each dimension is area-averaged down to the smaller of the two images'; `jpeg_q` of the JPEG's linear Rec.709 luminance (sRGB decoded, or Adobe RGB when EXIF ColorSpace (0xA001) is 0xFFFF, uncalibrated, which Sony writes together with the interoperability index R03). `label_ev` is §6.1's `L`, computed in Rust from the two tables (the training code recomputes it and must agree to 1e-9). Display luma for the contrast fit is Rec.709 luma of the sRGB-encoded render, 0–1. Unknown metadata is `null`.

### 6.1 Exposure label

The G3 fit of `docs/02` (`tools/eval/study.py`): the scene level `x` at EV 0 that the camera's JPEG renders as middle grey, by quantile matching our scene-linear luminance (at the JPEG's size) against the JPEG's linear luminance. `L = log2(0.18 / x)` is the exposure that puts our grey where the camera put it; the label is `key_target = L + m`. Precisely (`training/autoset/labels.py`, ported to Rust in the dump; the two agree to 1e-9):

1. Take the table points of the 0.02 … 0.98 quantiles (`INNER` of the 999-point grid).
2. **Mid-tone band:** keep the points whose JPEG linear luminance lies in `[0.18·2^−2, 0.18·2^+1]` = [0.045, 0.36] (−2 … +1 stop around grey); with fewer than 20 such points, keep all of step 1.
3. Make the curve monotone (cumulative maximum of each column, `quantile_curve_from_tables`) and read `x` where it reaches 0.18 (`camera_gain_ev`: log-log interpolation between the bracketing points, linear extrapolation from the two outermost points, no label when 0.18 is more than 2 stops outside the JPEG column). On a dense table only the bracketing pair matters, so the band changes `L` only where the table has a gap reaching past it; it sets which points the variance below sees.
4. **Variance** (EV²): `V_L = 0.05² · f + min(r², 1)`, with `f = 4` (the standard deviation doubled) when the camera's JPEG was made with DRO or HDR on or with a non-Standard creative style (their mid-tones are tone-mapped; an unknown value does not flag), else 1; `r` is the rms residual of `log2(jpeg)` about its least-squares line on `log2(scene)` over the points used, divided by the line's slope (scene stops).

### 6.2 Contrast label

At the exposure label `L`, the dump renders the preview through the real chain (the app's default look, Signature; default white; white lift as estimated) at contrast c ∈ {−40, −30, …, +40} and records the display luma quantiles (p = 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95) for each c, plus the same quantiles of the camera JPEG. The label is the c that minimises the squared difference of the inter-quantile spreads (p90 − p10 and p75 − p25), with the spreads interpolated linearly between grid points (the minimum is exact on each segment), clamped to ±40. The variance is `(ρ/g)²`, `ρ` the distance between the two pairs of spreads at the minimum and `g` the speed of the spreads per contrast unit on that segment, clamped to [4, 1600]. Augmentation rows inherit both labels of their file. (Contrast fitted to a camera's tone curve partly measures that curve; this is why the head is bounded and shrunk.)

### 6.3 Objective

Per member: Gaussian negative log-likelihood of each head against its label, with the label variance added to the predicted variance; L2 weight decay; an augmentation row asks for the exposure `L − k` (`key_target = L − k + m_aug`; this is the file's `key_target` when no pixel newly clips, and corrects for the metered key moving less than k when highlights clip at k > 0); every **scene** has the same total weight in the loss (a body's nine raw modes of one scene count as one scene); seeds fixed (member i uses seed 1000 + i); full-batch Adam, early stopping on a held-out 15 % of **scenes** (all files and augmentations of a scene stay together; scenes = files of the same body within 10 s, or the same SHA-256 prefix group of raw.pixls.us duplicates). JAX on the CPU (`training/requirements.txt`).

### 6.4 Exports

`python3 -m training.autoset.export` writes `data/models/autoset/autoset.json`, `MODEL_CARD.md` (data, licences, metrics, failure cases, git SHA), `SHA256SUMS`, and `crates/ym-color/tests/data/autoset/vectors.json` (inputs, member outputs and combined outputs computed in numpy f64 from the stored f32 weights); `crates/ym-color/tests/autoset_vectors.rs` must reproduce every vector to 1e-9.

## 7. Settings, interface and command line

- `DevelopSettings::contrast` becomes `ContrastSetting = AutoOr<ContrastValue { value }>` (the `ExposureSetting` pattern; JSON `{"kind":"auto"}` / `{"kind":"manual","value":12}`). A plain number in an older settings document reads as Manual with that value, except 0, which reads as Auto (0 was the default). `AutoFlags.contrast`; `ResolvedValues.contrast` is the effective contrast; `AutoEstimates` gains `contrast`, `ev_confidence`, `contrast_confidence`, `autoset_version`.
- `DevelopSettings::white_preference` (0 … 1, default 1; §5).
- Light tab: the Contrast slider shows the effective value with an AUTO state like Exposure; moving it makes it Manual; its Reset returns Auto. Look tab: the Automatic card gains a **Contrast** row (value, AUTO/EDITED, → Light); the Tone card's AUTO speaks for white balance **and** contrast, and its Reset sends both to Auto. The Automatic card's Exposure row keeps its place. No new card.
- `ym render --contrast <auto|N>`; `--white-preference <0..1>`; `ym info --json` reports the new estimates.
- Copy/Paste settings carry both as any setting does.

**As built (AS-07).** The integration follows the text above with these decisions where the spec is silent (`docs/02` and `docs/STATUS.md` say the same).

- *Metering on the neutral white.* The exposure and the features are metered on the all-Auto chain with the neutral Auto white (`ChainSettings::default`, `white_preference` 0). The model was trained on that chain (the dump of 2026-10-18), so the preferred white of section 5 does not move the features; `white_preference` changes the picture, never the exposure's input.
- *Duv of the user white.* Section 5 keeps `duv_u = duv_s`, which is meant to neutralise the tint cast fully. The tint of a light is its Duv over the locus's Duv scale at its CCT (`wb::scene_tint`), so with the Duv kept literally the user light's tint differs from the scene's by about a tenth of its value (2.5 units for a 3450 K scene of tint -25, 1.7 % of the green channel). The chain therefore scales the Duv by `duv_scale(mired_u) / duv_scale(mired_s)` where the CCT moved: the user light has the scene's tint, and the spec's intent holds. The rule's function `preferred_white` keeps `duv_u = duv_s`, as specified.
- *The preferred white needs no model.* Section 5 is a rule of the scene light and its brightness, so it applies with or without the model file. A picked white (Pick white renders the spot neutral, `docs/11`) and a light preset (the user's white is the preset) are unchanged, as is the Manual white balance.
- *The lift.* The white lift is estimated on the neutral Auto white, as the exposure is, and sized at the new exposure; the preferred white does not change it ("unchanged" in section 1).
- *Re-run.* `reset_automatic` returns contrast to Auto with exposure, the white lift's offset, the white balance and noise; `white_preference` is a preference and stays.
- *Exports.* `ym:Settings` is the settings record of `pinned_settings`: contrast is pinned to the value it resolved to while Auto, as exposure is (a Manual value renders like the Auto it came from, and the file says which number was used). `ym:Autoset` is a separate property holding the model's version or `none`: the settings record must stay readable by `ym render --settings`, whose reader refuses unknown fields. The XMP property list is in `docs/09`.
- *Readout.* The Auto Kelvin and tint (`AutoSummary.kelvin` and `tint`, the Automatic card's White row and `ym info`) are the readout of the Auto white at the default `white_preference`, the preferred white of section 5; the White balance row shows the readout of the chain the photo renders with, which is the same at the default preference. The metering chain's readout is the neutral one and is not shown.
- *Plain numbers.* A plain number of contrast is read as Manual, 0 as Auto, in a settings document, a `--settings` file and a settings patch of the core (old documents). The bridge (`types.ts`) takes the `kind` forms only; its generated bindings (`src/api/bindings.ts`) come from the Tauri side.
- *Burst.* The Auto contrast of a burst is the median of the group's own contrasts, rounded to 0.1 (`round_contrast`), as the exposure's median is shared.
- *Pipeline 0.5.0.* The golden **truth** settings gain contrast Manual 0, `white_preference` 0 and exposure Manual at `CAMERA_GAIN_EV` (`tools/golden/src/truth.rs`, `truth_settings`).

## 8. Gates (owner's decision of 2026-10-18)

1. **Determinism and parity:** Rust against Python ≤ 1e-9 (vectors); the same file gives the same estimates whatever the thread count; app and `ym` bit-identical (existing parity tests).
2. **Equivariance:** the dump's augmentation rows with k < 0 (nothing newly clips): `EV_model` moves by −k ± 0.05 EV for every file. Frames brightened by k > 0 lose highlights, so their features and key move by less than k; they are covered by gate 4.
3. **G3 on normal scenes only (no regression):** on held-out scenes that the camera exposed normally (|EC| < 0.3, no DRO/HDR, Standard style), `|EV_auto − L|` median ≤ 0.25 EV and ≤ 0.5 EV on 90 % of files.
4. **Rescue:** files rendered at −2 … +2 EV off their label come back within ±0.3 EV of it on 90 % of held-out files. With the camera floor, an augmentation row's camera exposure is the file's `L − k` (the camera would have rendered the shifted frame shifted too), so the floor is the row's own label.
5. **Blind A/B** ([HUMAN] AS-H1, the acceptance gate): about 150 photos in everyday strata, the owner's or any others (the pictures are never committed). Three versions of each photo: `autoset` (the new automatic settings), the rule before it (the old automatic settings) and the camera JPEG (the camera's default rendering). Pass at ≥ 60 % pooled wins against each of the two (Wilson lower bound > 50 %), no stratum below 45 %, no rise in gross failures (a photo that needs more than 0.5 EV or 10 mired of correction). Lightroom Auto is not compared (owner's decision of 2026-10-18, `docs/16`).
6. Golden images: re-baselined for 0.5.0 with a note; the golden **truth** keeps its settings (Neutral look, highlight range 0, lift off) plus contrast Manual 0 and `white_preference` 0, and exposure Manual at the pre-autoset rule, so its numbers do not move.

## 9. Later versions (not v1)

- **SceneNet** (rank 4 of the research): our own 1–4 M-parameter encoder, trained from scratch on Open Images V7 CC BY images, labelled offline by Apache-2.0 VLM teachers (Qwen3-VL, InternVL3.5, Gemma 4; never Gemma 3, Llama, MobileCLIP, DINOv3), audited on 300 images by the owner ([HUMAN] AS-H3); 16 scene probabilities feed `autoset`, with per-scene professional priors. Labelling needs a GPU machine.
- Learned preferred white (Cube++ pairs), `illum` v2 (Cube++, metadata, cross-body), HDR+ (CC BY-SA, phone DNGs: needs a DNG reader and generic matrices), Play Raw raws (CC BY-SA) as inputs and test photos; the forum's edits are not training targets (the owner, 2026-10-18, `docs/16`).
- Stage 2: the owner as teacher. Pick-the-best sessions (several renders of each photo, the owner picks one) and the owner's A/B choices become preferences, through a differentiable proxy of the chain and a Hu-style critic on owner-approved renders (rewritten from the MIT code and trained on clean inputs; Hu's released weights and MIT-Adobe FiveK stay out, `docs/14` rule 5).

## 10. Tasks

| ID | Task | Acceptance |
|---|---|---|
| AS-01 | This spec; `docs/02`, `07`, `13`, `15`, `16`, `STATUS` | done in the branch |
| AS-02 | `ym-raw`: EXIF BrightnessValue, MeteringMode, Flash, ExposureProgram, SceneCaptureType; Sony maker note: DRO mode, creative style, scene mode | unit tests on synthetic IFDs; values on the local ARWs match exiftool where installed |
| AS-03 | `ym_color::autoset`: features, model reader and ensemble, shrinkage, preferred white | unit tests: feature layout, equivariance, determinism, format refusals, a hand-made test model, the rule's numbers of §5 |
| AS-04 | `training/autoset`: dataset, labels (§6.1–6.2), training (§6.3), exports and vectors (§6.4) | pytest on synthetic rows; vectors reproduced by Rust to 1e-9 |
| AS-05 | `ym-core` example `autoset_dump` | JSONL per §6 for the local files; deterministic |
| AS-06 | `tools/fetch/autoset`: CC0 Sony ARW manifest and the streaming dataset build | manifest with SHA-256 and licence per file; rerunnable |
| AS-07 | `ym-core` integration: estimates, exposure, contrast `AutoOr`, preferred white in the chain's Auto arm, `white_preference`, XMP, CLI, wire types, `PIPELINE_VERSION` 0.5.0 | tests; golden truth unchanged; parity app/CLI |
| AS-08 | Frontend: contrast Auto/Manual, Automatic card Contrast row, Tone card AUTO, mock, tests, references | vitest, e2e |
| AS-09 | Train v1, evaluate (gates 2–4), golden re-baseline, contact sheets for the owner | numbers in STATUS |
| AS-H1 | [HUMAN] blind A/B sessions (the owner answers the blind page) | gate 5 |
| AS-H2 | [HUMAN] preferred-white constants by eye | §5 constants |
| AS-H3 | [HUMAN] SceneNet: teacher labelling on a GPU machine, 300-image audit | §9 |
