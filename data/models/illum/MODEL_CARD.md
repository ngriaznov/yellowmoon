# Model card: `illum`, the illuminant estimator

Version 1.0.0, 2026-10-12. Trained with `training/illum/` at git base `55bc009af21e` (working tree; training sources SHA-256 `204df1367341aabd`). Runs in `ym_color::illum` (`crates/ym-color/src/illum.rs`), once per photo, on the CPU.

## Purpose

Estimates the colour of the scene light from the 1/8-resolution raw preview: `w_scene` (camera RGB of the light, green = 1, in the camera's raw scale), probabilities over six classes (daylight, shade/overcast, tungsten/halogen, fluorescent, LED, mixed) and a confidence (the probability that the angular error is under 2 degrees). The result drives the residual white balance (C1), the class-blended sensor map (C2) and the ALEXA white (C3) of `docs/02`. One model per camera profile.

## Files

| Profile | File | Parameters | SHA-256 of the tensors |
|---|---|---|---|
| `sony-ilce-7m3` | `sony-ilce-7m3.json` | 13199 | `4a7955bfe27ed683bb22f008cc99bec0e198a39ed5c8a69e5d51a7c7b7d53872` |
| `sony-ilce-7m4` | `sony-ilce-7m4.json` | 13199 | `35906e18ef5f6d4cac22f5dc3947d080d90ba2d1744f221af6e9e70c3a08fd4a` |
| `sony-ilce-7cm2` | `sony-ilce-7cm2.json` | 13199 | `69e6b79e7b065b7f8e9c691a49a3633d7473ed215ef65ea35ed36b2b841c5edd` |
| `sony-ilce-7rm2` | `sony-ilce-7rm2.json` | 13199 | `2e46a5548c80621c4ef17b0a19713e6332a6e22dc14eefe7f503369d2f8c624d` |
| `sony-ilce-7rm3` | `sony-ilce-7rm3.json` | 13199 | `15183632953dfaa44abb959e7c1e043a68c83bb4c14a5edc0f2191cf5bd2b917` |
| `sony-ilce-7rm4` | `sony-ilce-7rm4.json` | 13199 | `dd2ca2f8cd6303721d559e026a5184a7ef55a1d49ddca1b1c3c05e2f6fbf63db` |
| `sony-ilce-7sm2` | `sony-ilce-7sm2.json` | 13199 | `67e31beeb08207eb5758f08184617937725cded2f03f60f035170a18586238af` |
| `sony-ilce-7sm3` | `sony-ilce-7sm3.json` | 13199 | `6df289b39f24baed9e90a8c9c9ca2ccb42c45007efcdc77c1396b2967b1c8411` |
| `sony-ilce-9` | `sony-ilce-9.json` | 13199 | `de2ecf8dbd2ee36b31cda805f4beb78bdd5037a02ca89c0b59dc2b1339ad2aae` |
| `sony-ilce-6400` | `sony-ilce-6400.json` | 13199 | `e2cf279e21f4cee5fda79db3f4e3968e84fa64103a3901e7d95a29efe2a6fc5a` |
| `sony-dsc-rx100m4` | `sony-dsc-rx100m4.json` | 13199 | `041eb0e68a6534aaefe3b58e0bd7d492e3bffbda8cc04d8f6ef75c0cb1a3f5a9` |

Format: JSON with the tensors as base64 of little-endian `f32`; `sha256` covers the tensors in `tensor_order`; the Rust loader checks it. Total under 50 k parameters per model (`docs/07`).

## Architecture

1. **Histograms.** Log-chroma `u = ln(G/R)`, `v = ln(G/B)` of every unmasked pixel (clipped pixels masked), 64 x 64 bins of 0.0625 log-chroma units (the window is centred on the camera's light locus, 1500 to 25000 K), bilinear splatting, weight sqrt(Y) with Y = 0.25 R + 0.5 G + 0.25 B. Two channels: the pixels, and the local absolute deviation (the edges, as in FFCC). Previews are subsampled with the stride isqrt(pixels / 16384): one under 65 536 pixels is histogrammed whole, the 1080 x 720 preview of a 50 MP frame uses every 6th pixel in both directions (21 600 pixels); the 96 x 64 training scenes use all of theirs).
2. **Filter** (Barron and Tsai, FFCC, CVPR 2017; own implementation): `Z = sum_c X_c (*) F_c + B`, a circular convolution evaluated with FFTs, 2 x 64 x 64 filter weights and a 64 x 64 bias. Softmax over the bins, then the mean and covariance of the softmax map (plus a learned variance floor) are the histogram-only estimate.
3. **As-shot prior.** The camera's own white balance (tag 0x7313) as a Gaussian `N(a, 0.05^2 I)` on `(u, v)`, multiplied into the estimate (precision-weighted mean). Without the tag the estimate is the histogram's alone.
4. **Class head.** 25 features (histogram moments of both channels relative to the estimate, the estimate, the as-shot value, `Bv = log2(N^2/t) - log2(ISO/100)`, the fit's log-determinant and peak) go through standardisation, one `tanh` layer of 24 units and a softmax (temperature calibrated on validation scenes). Trained also on copies without the as-shot value and without `Bv`.
5. **Confidence.** A logistic regression (28 weights and a bias) on the class-head features and three more (the Mahalanobis distance between the histogram estimate and the as-shot value, the log variances of the fused and of the histogram-only Gaussian), fitted on the validation scenes, which have held-out lights and patches, with and without the as-shot value: the probability that the angular error of `w_scene` is under 2 degrees. (A confidence read from the Gaussian alone was tried first: the as-shot prior caps the variance, so it carried almost no information, AUC 0.57.)

## Training data and licences

Synthetic only (no photographs, no measurements of our own; `AGENTS.md` V1 data rule). Scenes are rendered through the profile's published spectral curves (Weta Digital via rawtoaces-data, Apache-2.0) in their raw channel balance.

| Input | Licence | Use |
|---|---|---|
| rawtoaces-data 190 training reflectances | Apache-2.0 | 132 patches train, 20 validate, 38 test (fixed split); training also uses smooth random spectra and convex mixes of the patches |
| CIE illuminants (D series, FL1-FL12, FL3.1-FL3.15, LED-B1..V2) via colour-science 0.4.7 | CC BY-SA 4.0 (CIE) | training, validation and test lights, split by light; used at training time only, only fitted weights ship. **HUMAN M1-H4 (CIE-derived data) is still open.** |
| Generated near-white LED and fluorescent spectra, blackbody and D-series formulas | ours | training, validation and test (disjoint seed streams) |
| CIE 2017 CES 99, ColorChecker (Ohta), NIST CQS VS | CC BY-SA 4.0 / public data | **evaluation only**, never trained on |
| Cube++ (CC BY 4.0) | CC BY 4.0 | allowed for this model but **not used**: zenodo.org is unreachable from the build environment (`tools/fetch/cubepp.sh`, `training/illum/eval_cubepp.py` are ready) |

Splits are by light **and** by reflectance: `train`, `val` and `test` never share a light (held-out blackbody and D-series temperature bands, CIE fluorescent and LED spectra, generated spectra from disjoint seed streams; the nine sensor-map gate lights are all in `test`) or a reflectance patch (`training/illum/test_illum.py` checks both).

Scene model: 96 x 64 preview with 6 to 40 soft-edged cells, per-cell brightness, shading, texture, optional highlights and grey patches, dominant-colour scenes, mixed lighting (two lights with a smooth spatial weight; the truth is the brightness-weighted mean light); Poisson-Gaussian noise of nominal sensor parameters at an ISO tied to `Bv`, binned 8 x 8; per-channel clipping. **The as-shot white balance is simulated** (50 % good: a Gaussian error of 0.04 in log-chroma; 27 % off: 0.10; 8 % bad: 0.25; 11 % an unrelated preset; 4 % missing; a median error of about 1.6 degrees against the truth); that distribution is an assumption, not a measurement. Training draws reflectances from a broader mix (50 % smooth random spectra, 20 % single patches, 30 % convex mixes, all with a smooth perturbation) than validation and test (35 / 30 / 35, a milder perturbation), so that the held-out scenes measure generalisation to patches the model has not seen; with the narrower training mix the held-out median was 2.0 degrees histogram-only, with unmixed patches 4 degrees: a model memorises its 132 patches.

Recipe: 20000 training scenes per profile, 1500 Adam steps (batch 128), losses = softmax cross-entropy + Gaussian negative log-likelihood + L1-like error for the histogram-only and the fused estimate, total-variation and L2 regularisers on filter and bias (as FFCC).

## Metrics (held-out synthetic scenes)

Angular error in degrees between `w_scene` and the true camera white of the light. Gate (`docs/02`, `docs/13`): median <= 2 degrees on held-out synthetic scenes. `test` = held-out lights and held-out reflectances; the gate figure is the estimator as shipped, with the as-shot prior; histogram-only is reported because it is what a file without the white-balance tag gets.

| Profile | test, with prior: median / p90 / worst | test, histogram only: median / p90 / worst | CES 99 (eval only), with prior | gate (median <= 2) |
|---|---|---|---|---|
| `sony-ilce-7m3` | 1.26 / 5.03 / 21.3 | 1.91 / 6.65 / 43.2 | 1.29 / 5.60 / 26.3 | pass (hist. only: pass) |
| `sony-ilce-7m4` | 1.27 / 5.14 / 21.9 | 1.93 / 6.67 / 43.3 | 1.32 / 5.68 / 27.0 | pass (hist. only: pass) |
| `sony-ilce-7cm2` | 1.27 / 5.08 / 22.0 | 1.94 / 6.62 / 42.5 | 1.31 / 5.65 / 26.7 | pass (hist. only: pass) |
| `sony-ilce-7rm2` | 1.26 / 5.09 / 22.4 | 1.98 / 6.80 / 44.8 | 1.32 / 5.70 / 27.1 | pass (hist. only: pass) |
| `sony-ilce-7rm3` | 1.23 / 5.05 / 21.6 | 1.91 / 6.63 / 43.9 | 1.30 / 5.60 / 25.9 | pass (hist. only: pass) |
| `sony-ilce-7rm4` | 1.23 / 4.93 / 22.1 | 1.93 / 6.53 / 42.9 | 1.28 / 5.62 / 25.9 | pass (hist. only: pass) |
| `sony-ilce-7sm2` | 1.22 / 4.73 / 19.8 | 1.79 / 5.74 / 38.5 | 1.26 / 5.11 / 23.1 | pass (hist. only: pass) |
| `sony-ilce-7sm3` | 1.26 / 5.00 / 21.5 | 1.92 / 6.60 / 43.5 | 1.30 / 5.65 / 26.1 | pass (hist. only: pass) |
| `sony-ilce-9` | 1.30 / 5.20 / 22.2 | 1.96 / 6.57 / 41.7 | 1.33 / 5.72 / 26.4 | pass (hist. only: pass) |
| `sony-ilce-6400` | 1.22 / 4.95 / 20.2 | 1.87 / 6.20 / 42.0 | 1.24 / 5.38 / 24.4 | pass (hist. only: pass) |
| `sony-dsc-rx100m4` | 1.23 / 4.71 / 19.6 | 1.82 / 5.91 / 43.5 | 1.23 / 5.24 / 21.5 | pass (hist. only: pass) |

### A7R IV in detail

**Against simpler estimators** (test set, same scenes; median / p90 / worst):

| Estimator | median / p90 / worst |
|---|---|
| grey world | 3.76 / 8.63 / 40.0 |
| white patch (99th percentile) | 4.75 / 11.01 / 26.8 |
| shades of grey (p = 6) | 4.48 / 10.33 / 27.9 |
| grey edge | 4.29 / 9.67 / 27.3 |
| the simulated as-shot white balance alone | 1.54 / 6.46 / 26.0 |
| grey world + the same as-shot prior | 1.55 / 6.30 / 24.1 |
| **FFCC filter, histogram only** | 1.93 / 6.53 / 42.9 |
| **FFCC filter + as-shot prior (shipped)** | 1.23 / 4.93 / 22.1 |

**By class** (test, with prior | histogram only; median / p90):

| Class | with prior | histogram only |
|---|---|---|
| daylight | 1.16 / 3.76 | 1.30 / 5.72 |
| shade_overcast | 0.96 / 4.45 | 1.23 / 5.09 |
| tungsten_halogen | 1.20 / 8.19 | 1.87 / 6.46 |
| fluorescent | 1.29 / 4.72 | 2.50 / 7.08 |
| led | 1.56 / 5.62 | 2.44 / 7.44 |
| mixed | 1.32 / 5.27 | 2.35 / 6.95 |

**By light family** (test, histogram only; median / p90, n):

| Family | median / p90 | n |
|---|---|---|
| daylight | 1.28 / 5.42 | 1137 |
| blackbody | 1.87 / 6.46 | 485 |
| led_gen | 2.85 / 7.42 | 273 |
| led_cie | 2.19 / 7.41 | 255 |
| fluor_cie | 2.19 / 6.38 | 234 |
| fluor_gen | 2.89 / 7.71 | 180 |
| mixed:blackbody+daylight | 2.72 / 9.10 | 112 |
| mixed:daylight+daylight | 1.64 / 5.71 | 59 |
| mixed:led_gen+blackbody | 3.50 / 7.92 | 37 |
| mixed:led_cie+daylight | 1.94 / 10.42 | 33 |
| mixed:blackbody+fluor_cie | 2.44 / 6.07 | 32 |
| mixed:led_gen+daylight | 1.92 / 4.16 | 27 |

**By ISO** (test, histogram only; median / p90):

| ISO | median / p90 |
|---|---|
| 100 to 800 | 1.52 / 5.99 |
| 800 to 6400 | 2.08 / 6.85 |
| 6400 and up | 2.31 / 6.90 |

**Classes** (test, with prior): accuracy 62.7 % over the six classes, 72.4 % over the four profile classes (daylight and shade merged, mixed excluded), negative log-likelihood 0.971. Rows true, columns predicted (daylight, shade_overcast, tungsten_halogen, fluorescent, led, mixed):

```
  579    40     2    15     7    21
   63   384     1     7     5    13
    6     0   418     6    41    14
   37    20    51   158    80    68
   55    23   155    97   122    76
   60    21    51    51    34   219
```

**Confidence** (test, with prior): mean 0.685, fraction of scenes under 2 degrees 0.688, Brier 0.174 (the constant base rate gives 0.215), AUC 0.749; histogram only: Brier 0.215 (base rate 0.250), AUC 0.723. Quintiles of confidence against the fraction of scenes under 2 degrees (with prior):

| mean confidence | fraction under 2 degrees | n |
|---|---|---|
| 0.369 | 0.348 | 600 |
| 0.631 | 0.637 | 600 |
| 0.735 | 0.763 | 600 |
| 0.807 | 0.790 | 600 |
| 0.884 | 0.900 | 600 |

**Stand-in sensitivity.** The A7R IV model on scenes rendered through the other bodies' curves (the situation of the A1 and every body without a profile, and of a body whose sensor differs from its profile): median / p90 / worst.

| Body | with prior | histogram only |
|---|---|---|
| `sony-ilce-7m3` | 1.30 / 4.84 / 17.5 | 2.15 / 5.95 / 28.8 |
| `sony-ilce-7m4` | 1.30 / 4.85 / 17.9 | 2.17 / 6.07 / 29.0 |
| `sony-ilce-7cm2` | 1.30 / 4.85 / 17.8 | 2.20 / 6.04 / 28.4 |
| `sony-ilce-7rm2` | 1.23 / 4.73 / 18.4 | 1.96 / 5.98 / 30.0 |
| `sony-ilce-7rm3` | 1.25 / 4.73 / 17.9 | 1.92 / 5.65 / 29.3 |
| `sony-ilce-7sm2` | 1.18 / 4.98 / 16.2 | 1.95 / 5.61 / 25.4 |
| `sony-ilce-7sm3` | 1.30 / 5.09 / 17.7 | 2.22 / 6.05 / 28.1 |
| `sony-ilce-9` | 1.60 / 5.86 / 18.7 | 2.89 / 6.86 / 27.3 |
| `sony-ilce-6400` | 1.64 / 4.81 / 18.0 | 2.83 / 6.69 / 27.2 |
| `sony-dsc-rx100m4` | 1.35 / 5.06 / 17.5 | 2.34 / 6.00 / 25.7 |

## The sensor-map gate with estimated lights

`training/illum/gate_estimated.py` re-runs the M1-05 gate (99 CES, nine held-out lights, tables refitted without LED-B3, LED-RGB1 and LED-V2) with the light taken from the estimator on synthetic scenes lit by each gate light (A7R IV: 200 scenes per light): the camera is balanced with the estimated white, the map is `M(probs, mired)` with the six classes folded into the four and the mired of the CCT read from the profile's locus, ARRI's matrix is taken at that CCT; the reference is the ALEXA rendering under the true light. dE00 median / p95 pooled over scenes and patches, against the known-light gate. Gate: median <= 0.5, p95 <= 1.2.

Three modes split the error by what the estimate is allowed to change: **full** (everything from the estimate: the gate as worded), **map only** (the camera is balanced with the true white; the estimated classes, mired and CCT choose the map and ARRI's matrix: what the estimator costs the sensor map) and **white only** (the estimated white balances the camera; the true class and CCT choose the map: what the white-balance error costs).


**Full, with the as-shot prior (as shipped)**, A7R IV (angular error median / p90 of the estimate, dE00 median / p95 estimated, dE00 median / p95 known light):

| Light | class | angular | estimated | known light | |
|---|---|---|---|---|---|
| blackbody-2850K | tungsten_halogen | 1.02 / 6.50 | 1.774 / 18.839 | 0.300 / 1.045 | over |
| daylight-5250K | daylight | 1.08 / 4.08 | 1.700 / 10.150 | 0.372 / 1.108 | over |
| daylight-6750K | daylight | 1.04 / 3.94 | 1.737 / 9.014 | 0.417 / 1.170 | over |
| FL3.1 | fluorescent | 1.27 / 6.57 | 2.621 / 18.382 | 0.312 / 0.997 | over |
| FL3.8 | fluorescent | 1.35 / 6.73 | 2.329 / 13.792 | 0.206 / 0.593 | over |
| FL3.15 | fluorescent | 1.08 / 3.54 | 1.901 / 9.321 | 0.304 / 1.098 | over |
| LED-B3 | led | 1.29 / 6.36 | 2.113 / 13.003 | 0.235 / 1.015 | over |
| LED-RGB1 | led | 1.46 / 8.24 | 3.042 / 19.827 | 0.316 / 0.984 | over |
| LED-V2 | led | 1.43 / 5.62 | 2.447 / 12.436 | 0.759 / 1.515 | over |

Worst median 3.042, worst p95 19.827; lights over the gate: blackbody-2850K, daylight-5250K, daylight-6750K, FL3.1, FL3.8, FL3.15, LED-B3, LED-RGB1, LED-V2.


**Full, histogram only**, A7R IV (angular error median / p90 of the estimate, dE00 median / p95 estimated, dE00 median / p95 known light):

| Light | class | angular | estimated | known light | |
|---|---|---|---|---|---|
| blackbody-2850K | tungsten_halogen | 1.31 / 3.15 | 1.979 / 7.702 | 0.300 / 1.045 | over |
| daylight-5250K | daylight | 1.12 / 5.61 | 1.752 / 12.190 | 0.372 / 1.108 | over |
| daylight-6750K | daylight | 1.39 / 5.16 | 2.002 / 11.115 | 0.417 / 1.170 | over |
| FL3.1 | fluorescent | 1.54 / 4.03 | 3.170 / 10.852 | 0.312 / 0.997 | over |
| FL3.8 | fluorescent | 2.74 / 7.74 | 3.929 / 14.294 | 0.206 / 0.593 | over |
| FL3.15 | fluorescent | 1.23 / 4.78 | 2.295 / 11.127 | 0.304 / 1.098 | over |
| LED-B3 | led | 2.10 / 7.19 | 3.073 / 13.262 | 0.235 / 1.015 | over |
| LED-RGB1 | led | 2.06 / 3.44 | 3.796 / 9.869 | 0.316 / 0.984 | over |
| LED-V2 | led | 2.28 / 6.70 | 4.010 / 13.912 | 0.759 / 1.515 | over |

Worst median 4.010, worst p95 14.294; lights over the gate: blackbody-2850K, daylight-5250K, daylight-6750K, FL3.1, FL3.8, FL3.15, LED-B3, LED-RGB1, LED-V2.


**Map only, with the as-shot prior**, A7R IV (angular error median / p90 of the estimate, dE00 median / p95 estimated, dE00 median / p95 known light):

| Light | class | angular | estimated | known light | |
|---|---|---|---|---|---|
| blackbody-2850K | tungsten_halogen | 1.02 / 6.50 | 0.394 / 1.428 | 0.300 / 1.045 | over |
| daylight-5250K | daylight | 1.08 / 4.08 | 0.424 / 1.302 | 0.372 / 1.108 | over |
| daylight-6750K | daylight | 1.04 / 3.94 | 0.489 / 1.402 | 0.417 / 1.170 | over |
| FL3.1 | fluorescent | 1.27 / 6.57 | 0.442 / 1.412 | 0.312 / 0.997 | over |
| FL3.8 | fluorescent | 1.35 / 6.73 | 0.359 / 1.286 | 0.206 / 0.593 | over |
| FL3.15 | fluorescent | 1.08 / 3.54 | 0.380 / 1.320 | 0.304 / 1.098 | over |
| LED-B3 | led | 1.29 / 6.36 | 0.329 / 1.357 | 0.235 / 1.015 | over |
| LED-RGB1 | led | 1.46 / 8.24 | 0.351 / 1.364 | 0.316 / 0.984 | over |
| LED-V2 | led | 1.43 / 5.62 | 0.740 / 1.828 | 0.759 / 1.515 | over |

Worst median 0.740, worst p95 1.828; lights over the gate: blackbody-2850K, daylight-5250K, daylight-6750K, FL3.1, FL3.8, FL3.15, LED-B3, LED-RGB1, LED-V2.


**White only, with the as-shot prior**, A7R IV (angular error median / p90 of the estimate, dE00 median / p95 estimated, dE00 median / p95 known light):

| Light | class | angular | estimated | known light | |
|---|---|---|---|---|---|
| blackbody-2850K | tungsten_halogen | 1.02 / 6.50 | 1.779 / 18.757 | 0.300 / 1.045 | over |
| daylight-5250K | daylight | 1.08 / 4.08 | 1.702 / 10.285 | 0.372 / 1.108 | over |
| daylight-6750K | daylight | 1.04 / 3.94 | 1.731 / 9.214 | 0.417 / 1.170 | over |
| FL3.1 | fluorescent | 1.27 / 6.57 | 2.641 / 18.573 | 0.312 / 0.997 | over |
| FL3.8 | fluorescent | 1.35 / 6.73 | 2.331 / 13.632 | 0.206 / 0.593 | over |
| FL3.15 | fluorescent | 1.08 / 3.54 | 1.897 / 9.531 | 0.304 / 1.098 | over |
| LED-B3 | led | 1.29 / 6.36 | 2.129 / 12.966 | 0.235 / 1.015 | over |
| LED-RGB1 | led | 1.46 / 8.24 | 3.056 / 19.343 | 0.316 / 0.984 | over |
| LED-V2 | led | 1.43 / 5.62 | 2.473 / 12.561 | 0.759 / 1.515 | over |

Worst median 3.056, worst p95 19.343; lights over the gate: blackbody-2850K, daylight-5250K, daylight-6750K, FL3.1, FL3.8, FL3.15, LED-B3, LED-RGB1, LED-V2.


All profiles (with the as-shot prior), worst median / worst p95 over the nine lights and the lights over the gate. The known-light column is the M1-05 gate with the same tables.

| Profile | full | map only | known light |
|---|---|---|---|
| `sony-ilce-7m3` | 2.739 / 19.546 (9 over) | 0.569 / 1.532 (5 lights over) | 0.551 / 1.202 |
| `sony-ilce-7m4` | 3.024 / 19.920 (9 over) | 0.767 / 1.785 (9 lights over) | 0.767 / 1.484 |
| `sony-ilce-7cm2` | 3.056 / 19.815 (9 over) | 0.868 / 1.980 (9 lights over) | 0.898 / 1.673 |
| `sony-ilce-7rm2` | 2.955 / 19.928 (9 over) | 0.527 / 1.434 (6 lights over) | 0.492 / 1.030 |
| `sony-ilce-7rm3` | 2.799 / 19.850 (9 over) | 0.397 / 1.406 (6 lights over) | 0.337 / 0.981 |
| `sony-ilce-7rm4` | 3.042 / 19.827 (9 over) | 0.740 / 1.828 (9 lights over) | 0.759 / 1.515 |
| `sony-ilce-7sm2` | 3.640 / 19.338 (9 over) | 0.701 / 2.115 (9 lights over) | 0.622 / 1.868 |
| `sony-ilce-7sm3` | 2.812 / 19.707 (9 over) | 0.487 / 1.427 (5 lights over) | 0.443 / 1.199 |
| `sony-ilce-9` | 2.753 / 19.901 (9 over) | 1.094 / 2.467 (8 lights over) | 1.159 / 2.208 |
| `sony-ilce-6400` | 3.267 / 19.693 (9 over) | 0.755 / 2.108 (9 lights over) | 0.677 / 1.685 |
| `sony-dsc-rx100m4` | 3.669 / 18.751 (9 over) | 0.726 / 1.817 (9 lights over) | 1.012 / 2.526 |

**Reading (A7R IV).** Estimated end to end, the gate fails for every light: the white-balance error of the estimate (a median of about 1 to 1.5 degrees per light, 4 to 8 degrees at p90) shows as a colour cast and costs 1.18 to 1.87 dE00 in the median per degree of median angular error, an order of magnitude more than the sensor map's own error; a median of 0.5 dE00 would need the white to be right to about 0.3 degrees, far below what illuminant estimators reach on photographs (typically 1.5 to 3 degrees). The estimator's classes and mired alone (map only, white known) cost -0.02 to 0.15 dE00 in the median and 0.19 to 0.69 in the p95: the median stays under 0.5 for every light that passed with the light known except LED-V2, and the p95 limit of 1.2, which the known-light fit meets with margins of 0.03 to 0.61, is exceeded on 9 of the nine (LED-V2 is the accepted M1-05 failure either way).


## Failure cases and limits

* **Mixed lighting** has no single illuminant; the truth here is the brightness-weighted mean and the error is the largest of all classes.
* **Dominant-colour scenes** (a frame filled with one hue) shift the histogram and the estimate; the as-shot prior limits the damage when the camera's own estimate is sane.
* **A wrong as-shot white balance** (a preset that does not match the light, 11 % of the synthetic scenes) pulls the fused estimate: the prior is a Gaussian with sigma 0.05 as specified, not a robust one (A7R IV, test scenes with such a preset: median 5.92 / p90 15.5 degrees with the prior, 1.85 / 5.9 without it; `training/illum/robust_prior_study.py` shows what a two-component prior would change). Without the tag the histogram alone gives the histogram-only figures.
* **The histogram-only median is close to the 2-degree line** for some bodies (see the table; on freshly drawn test scenes the A7R II gave 1.98 and 2.08, and the validation scenes of the A7R IV 2.18): the figure that passes with margin is the one with the as-shot prior, which is how the estimator ships, and it rests on the assumed as-shot error model.
* **The brightness prior is five stops low for sunlit scenes.** The synthetic `Bv = log2(N^2/t) - log2(ISO/100)` runs from -2 to 13 (daylight mean 9.3); real sunlit photographs read 14 to 15 (four of the nine real files in `testdata/real/files`). The class head extrapolates in the right direction there (more daylight), but it never saw such values.
* **Daylight and shade/overcast** overlap in temperature and in brightness; the class head separates them only statistically. Both fold into the daylight class of the sensor map, so a confusion between them costs no colour accuracy.
* **Fluorescent versus LED** is the weakest class pair (see the confusion matrix); spiky spectra are separated mostly by their distance from the locus.
* **Real photographs were never seen.** The scenes are synthetic, the class and brightness priors are assumptions, and the as-shot error model is assumed. The Cube++ gate (<= 3 degrees) and the NUS-8 Sony report are blocked here (no access to zenodo.org or the NUS-8 host) and have not been run.
* **Stand-in profiles.** A body that shares a stand-in profile with a different sensor sees raw colours the model was not trained on; the cross-camera table above bounds the cost.
* **Off-locus lights** (coloured stage lighting, sodium or mercury lamps) are outside the training distribution (generated lights are within 0.012 of the locus).
* **The 4000 K junction** of the profile locus makes CCT and Duv unreliable between about 3800 and 4300 K (`docs/03`); the estimator's `w_scene` does not depend on it, but the CCT that is read from it does.

## Provenance

* Date 2026-10-12; model version 1.0.0; training code at git base `55bc009af21ebd001bc3c5c358e62997a80e46b0` plus the working tree (`training_code_sha256` 204df1367341aabdabc14e213ac2d6d3cb07b0068dd01d3e59947d765e985166).
* Reproduce: `pip install -r training/requirements.txt`, `tools/fetch/spectral.sh`, `python3 -m training.illum.run_all`, `python3 -m training.illum.model_card`, `python3 -m training.illum.export_vectors`.
* Rust parity: `crates/ym-color/tests/illum.rs` against `tests/data/illum/` (1e-9).
