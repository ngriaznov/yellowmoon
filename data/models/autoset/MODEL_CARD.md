# Model card: `autoset`, exposure and contrast

Version 1.0.0, 2026-10-18. Trained with `training/autoset/` at git commit `b3894df`. Runs in `ym_color::autoset` (`crates/ym-color/src/autoset.rs`, AS-03), once per photo, on the CPU, in float64.

## Purpose

Decides where the key of a photo sits and how much contrast it gets, from the photo's brightness statistics (`docs/18` section 3). The exposure head predicts the key target (where the key should sit after exposure, in stops over middle grey); the file's exposure is that target minus the key, so a photo shot a stop under or over comes out the same (full rescue). The exposure applies in full; its weight is reported as a confidence only. The contrast head predicts the C9 contrast, shrunk towards 0 when the ensemble is unsure (section 4.4). Settings are chosen per photo; nothing is stored per photo.

## Files

| File | Model version | Members | Parameters per member | SHA-256 of the member tensors |
|---|---|---|---|---|
| `autoset.json` | 1.0.0 | 5 | 2468 | `2145d5795e4cd3c5ad94e9135407ddbe300666579b18e60c0bf56a32e1381deb` |

Format: `docs/18` section 4.3 (JSON, tensors as base64 little-endian float32, checksum over the member tensors in order). Inputs are standardised with the stored mean and standard deviation; the input statistics and the constants are not covered by the checksum.

## Architecture

5 multilayer perceptrons, each 39 to 32 to 32 to 4 with `tanh` hidden units and a linear output (key target, key log-variance, contrast, contrast log-variance). The combination is the mean of the means; the variance is the mean of the member variances plus the population variance of the means. Prior sigmas 0.75 (key target) and 10 (contrast); contrast limit 30. Weights `w = s0^2 / (s0^2 + variance)`, applied as in `docs/18` section 4.4.

## Training data and licences

Dataset rows (`docs/18` section 6): one line per file with its 39 features, the label tables and the augmentations (scaled by 2^k, k in -2, -1, -0.5, +0.5, +1, +2; the file's exposure label moved by -k and its contrast label). Every file and augmentation of a scene is in the same split.

**Licence of the weights:** Apache-2.0, as the rest of Yellowmoon: every training file is CC0 (`docs/14` rule 5).

| Source | Licence | Files | Scenes | Bodies |
|---|---|---|---|---|
| raw.pixls.us | CC0-1.0 | 191 | 154 | 96 |
| rawdb.dnglab.org | CC0-1.0 | 24 | 22 | 10 |

## Labels

* **Exposure (`docs/18` 6.1).** The G3 fit of the auto-exposure study (`tools/eval/metrics.py`): the scene level at which the camera's JPEG reaches mid grey, by quantile matching over the 0.02 to 0.98 quantiles, restricted to the mid-tone band of JPEG luminance 0.045 to 0.36 (-2 to +1 stop around grey; unrestricted when fewer than 20 points remain). Variance 0.05 EV squared, times 4 for DRO, HDR or a non-Standard creative style, plus the fit's residual.
* **Contrast (`docs/18` 6.2).** The contrast whose display-luma p90 - p10 and p75 - p25 spreads best match the camera's, linearly interpolated between the grid of -40 to +40 in steps of 10, clamped to +-40. Variance from the residual, floor 4.
* **Key target** = exposure label + key.

Files with an exposure label: 215 of 215; with a contrast label: 215; normal scenes: 29; flagged (DRO, HDR or style): 183.

## Training

Full-batch Adam, learning rate 0.01, 3000 steps at most, L2 weight decay 0.01, early stopping on the held-out 15 % of scenes (40 evaluations without a new best). Members are seeded 1000 and up; the scene split uses seed 20261018.

## Metrics

EV error is `EV_model - L` in EV, where `EV_model = key_target - key`. Gates are `docs/18` section 8. The constant rule is `EV = 1.11`, the camera's gain.

| Rows | n | MAE | median | p90 | within 0.5 EV |
|---|---|---|---|---|---|
| held-out files, model | 215 | 0.41 | 0.26 | 0.82 | 73.5 % |
| normal files, model (gate 3) | 29 | 0.58 | 0.32 | 1.74 | 58.6 % |
| held-out files, constant 1.11 | 215 | 0.36 | 0.19 | 0.71 | 81.4 % |
| normal files, constant 1.11 | 29 | 0.41 | 0.11 | 1.67 | 75.9 % |

| Augmentations | n | MAE | median | p90 | share |
|---|---|---|---|---|---|
| rescue: |EV - label| within 0.3 EV (gate 4) | 1290 | 0.42 | 0.25 | 0.91 | 54.7 % |
| equivariance: |EV(aug) - (EV(base) - k)| within 0.05 (gate 2) | 1290 | 0.09 | 0.04 | 0.23 | 54.0 % |

| Contrast | n | MAE | median | p90 | within 5 units |
|---|---|---|---|---|---|
| model, clamped to +-30 | 215 | 21.1 | 19.2 | 41.2 | 14.0 % |
| constant 0 (the rule before autoset) | 215 | 22.1 | 21.8 | 40.0 | 15.3 % |

Calibration (share of labels within the predictive spread): contrast within 1 sd 57.2 %, contrast within 2 sd 87.9 %, key within 1 sd 70.7 %, key within 2 sd 91.6 %.

## Failure cases

The ten largest EV errors on held-out files (file names as in the dataset):

| File | Body | Label EV | Model EV | Error EV | Flagged | Normal |
|---|---|---|---|---|---|---|
| DSC-RX100_ISO_640_12bits_Sony ARW Compressed.ARW | DSC-RX100 | -0.77 | 1.84 | 2.61 | no | yes |
| Sony - DSC-RX1RM3 - 4:3.ARW | DSC-RX1RM3 | -1.72 | 0.88 | 2.60 | yes | no |
| Sony - ILCE-7M2 - 12bit 12bit compressed (3:2).arw | ILCE-7M2 | 1.13 | -1.29 | -2.42 | yes | no |
| Sony - DSC-RX1RM3 - 3:2.ARW | DSC-RX1RM3 | -1.06 | 0.91 | 1.96 | yes | no |
| Sony - ILME-FX3 - 14bit (3:2).ARW | ILME-FX3 | 1.36 | 3.24 | 1.88 | yes | no |
| Sony - ILCE-7CR - losseless compressed (4:3).ARW | ILCE-7CR | -0.61 | 1.23 | 1.84 | no | yes |
| Sony - ILCE-7CR - losseless compressed (4:3).ARW | ILCE-7CR | -0.61 | 1.22 | 1.83 | no | yes |
| Sony - DSC-RX1R - 14bit 14bit compressed (3:2).ARW | DSC-RX1R | 1.12 | 2.86 | 1.74 | yes | no |
| Sony - ILCE-7CR - losseless compressed (4:3).ARW | ILCE-7CR | -0.55 | 1.17 | 1.72 | no | yes |
| Sony - ILCE-7C - 14bit 14bit uncompressed (3:2).ARW | ILCE-7C | 1.03 | -0.52 | -1.55 | yes | no |

## Limits

* The exposure label is the camera's own tone placement, so the model learns the camera's choice where its JPEG is tone-mapped (DRO, HDR, creative styles); those labels carry double variance.
* The contrast label partly measures the camera's tone curve, which is why its head is bounded at +-30 and shrunk.
* Nothing here was judged by eye. The blind A/B against Lightroom Auto and the camera JPEG (`docs/18` gate 5, AS-H1) decides whether the model is better than the rule before it.

