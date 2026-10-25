# 02 · Colour pipeline

This document specifies the colour pipeline: stages C1 to C13, from white-balanced Sony camera RGB to display code values, with the constants, formulas and test bounds of each stage. It also covers the automatic estimates made on the preview (exposure, illuminant, white lift) and the Before view.

Crates: `ym-color` (CPU reference in f64, constants, fitting helpers) and `ym-gpu` (WGSL kernels in f32). Data: `data/alexa/`, `data/arri/`, `data/cameras/`.

## Idea in one paragraph

A Sony sensor and an ALEXA sensor see the same scene differently, and how differently depends on the light. Yellowmoon estimates the light in each photo, then maps white-balanced Sony RGB to what the ALEXA's sensor would have recorded under that light (the **sensor map**). From there it follows ARRI's published processing: ALEXA white balance at the user's colour temperature, ARRI's colour matrix for that temperature, ARRI Wide Gamut, LogC. Looks and the display transform work on LogC4 exactly as they do for ALEXA footage. Two quantities stay separate on purpose: the **scene light** (estimated) drives the sensor map; the **user's CCT** drives the ALEXA white balance and the ARRI matrix. "3200 K under daylight" therefore looks the way it would on an ALEXA.

## Stage order

Input to this chain is the output of `docs/05-raw-processing.md`: demosaiced, denoised, lens-corrected, scene-linear camera RGB, black-subtracted, white level = 1.0, balanced with the **initial** gains `g0` used for highlight recovery and demosaic.

| # | Stage | Space in → out | Where |
|---|---|---|---|
| C1 | Residual white balance to the scene light | camera RGB (g0-balanced) → camera RGB balanced for the scene light | fused 3×3 |
| C2 | Sensor map for the scene light | balanced Sony RGB → balanced ALEXA native RGB | fused 3×3 |
| C3 | Re-apply ALEXA's own white for the scene light | balanced → ALEXA native (unbalanced) | fused 3×3 |
| C4 | ALEXA white balance at the user's CCT and tint | ALEXA native → ALEXA native, balanced for the user's white | fused 3×3 |
| C5 | ARRI matrix at the user's CCT | ALEXA native → linear AWG3 | fused 3×3 |
| C6 | AWG3 → AWG4, exposure | → linear AWG4, scene-referred | fused 3×3 |
| C7 | Highlight range and low-light toe | linear AWG4 → linear AWG4 | per pixel |
| C8 | LogC4 encode | linear AWG4 → LogC4 | per pixel |
| C9 | Primary grade (contrast, brightness, zones, colour balance, saturation) | LogC4 → LogC4 | per pixel |
| C9b | Colour warp: pins that move chosen colours across the chroma plane | LogC4 → LogC4 | per pixel + pins in the uniform block |
| C10 | Look: CDL, then 3D LUT, blended by intensity | LogC4 → LogC4 (LogC3 looks via exact conversion) | per pixel + f32 LUT storage buffer |
| C11 | Curve | LogC4 → LogC4 | f32 curve tables (storage buffer) |
| C12 | Display transform: ACES 2 (default; the white lift, the red chroma step and the display placement come first) or an imported ARRI display LUT | LogC4 → display linear or display code values | per pixel |
| C13 | Display encode | → Display P3 (viewport) or the export colour space | per pixel |

C1–C6 are linear and fuse into **one 3×3 matrix per photo**, recomputed on the CPU when a colour setting changes:

```
M = 2^EV · M_awg3→awg4 · M_arri(CCT_user) · diag(g_user) · diag(w_A(E_scene)) · M_map(scene) · diag(g_res)
```

where every `w`, `g` is normalised so its green component is 1.

C7–C13 run per pixel. `ym_color::display` is their CPU reference (the CPU fallback runs it) and every GPU kernel is parity-tested against it: the gate is a maximum absolute error of 2e-4 in LogC4 units, or one 10-bit code in the encoded picture.

## Constants

### LogC4 (ARRI LogC4 specification, 2022)

```
a = (2^18 − 16) / 117.45     = 2231.8263090676883
b = (1023 − 95) / 1023       = 0.9071358748778103
c = 95 / 1023                = 0.09286412512218964
s = 7·ln2·2^(7 − 14c/b) / (a·b) = 0.1135972086105891
t = (2^(14·(−c/b) + 6) − 64) / a = −0.01805699611991131

encode(x) = x ≥ t ? (log2(a·x + 64) − 6) / 14 · b + c : (x − t) / s
decode(E) = E ≥ 0 ? (2^(14·(E − c)/b + 6) − 64) / a : E·s + t
```

Checks: `encode(0.18) = 0.2783958365`, `encode(0) = 0.0928641251`, `decode(1.0) = 469.8` (11.35 stops above 0.18).

### LogC3, EI 800 (ARRI LogC3 VFX document)

```
cut 0.010591  a 5.555556  b 0.052272  c 0.247190  d 0.385537  e 5.367655  f 0.092809
encode(x) = x > cut ? c·log10(a·x + b) + d : e·x + f
decode(t) = t > e·cut + f ? (10^((t − d)/c) − b) / a : (t − f) / e
```

Check: `encode(0.18) = 0.3910068` (= 400/1023 to 1e-8). LogC3 is only used to run LogC3 looks and display LUTs; the EI is always 800. The constants are the VFX document's, rounded to 6 digits (OpenColorIO uses them at full precision, which differs by at most 1e-5 relative above code value 0.1 and up to 7e-4 below linear 1e-3).

### Gamuts (D65 white x 0.3127, y 0.3290)

| | R | G | B |
|---|---|---|---|
| AWG3 | 0.6840, 0.3130 | 0.2210, 0.8480 | 0.0861, −0.1020 |
| AWG4 | 0.7347, 0.2653 | 0.1424, 0.8576 | 0.0991, −0.0308 |

```
AWG3 → XYZ                                   AWG4 → XYZ
 0.6380076193  0.2147038563  0.0977444514     0.7048583204  0.1297602952  0.1158373115
 0.2919537790  0.8238410415 −0.1157948205     0.2545241764  0.7814777327 −0.0360019091
 0.0027982790 −0.0670342357  1.1532937074     0.0000000000  0.0000000000  1.0890577508

AWG3 → AWG4 (exact; same white, so no adaptation)
 0.8892557469  0.1288981523 −0.0181538992
 0.0840832627  1.0093919566 −0.0934752193
 0.0025694496 −0.0615525078  1.0589830582

AWG4 → linear Display P3                     AWG4 → linear sRGB / Rec.709
 1.5205023970 −0.4042986627 −0.1162037343     1.8931234427 −0.7808815036 −0.1122419390
−0.1360315837  1.2698479799 −0.1338163962    −0.2057003583  1.3402574894 −0.1345571311
 0.0058785171 −0.0548756605  1.0489971435    −0.0127057429 −0.1521848758  1.1648906187
```

`ym-color` computes every matrix from the primaries in f64 (normalised primary matrix) and tests them against these numbers. AWG3 → ACES AP0 (needed for ACES 2 and the ARRI table) uses CAT02 (von Kries in CAT02 space) from D65 to the ACES white (x 0.32168, y 0.33767). Computed in f64 it is (ARRI's IDTs print the same values rounded to 6 digits):

```
AWG3 → AP0 (CAT02)
 0.680205505  0.236136602  0.083657893
 0.085414980  1.017470879 −0.102885858
 0.002056522 −0.062562500  1.060505979
```

Rows of this matrix sum to 1 within 1e-15. AP0 is derived from the AP0 primaries and never taken from a library that stores it rounded (colour-science's rounded AP0 matrix shifts row sums by about 2e-7).

### ARRI CCT matrices

`data/arri/Alexa_aces_matrix.txt` (ARRI via aces-dev v1.3.1, AMPAS licence) holds camera-native → AP0 matrices at 17 CCTs: 2000, 2100, 2200, 2400, 2600, 2900, 3200, 3500, 3900, 4300, 4700, 5100, 5600, 6500, 7500, 9000, 11000 K. `data/arri/alexa_native_to_awg3_17cct.json` holds the derived native → AWG3 matrices, `inv(AWG3→AP0) · M_AP0(CCT)` with the exact matrix above; every row sums to 1 within 1e-14, as balanced input requires. `ym-color` derives them itself from the `.txt` and tests against the JSON to 1e-12.

**Interpolation:** linear in mired between the **bracketing** pair of table entries; the CCT is clamped to [2000, 11000]. At the 17 table CCTs the interpolation returns the table entry bit for bit. ARRI's `v3_IDT_maker.py` (aces-dev v1.3.1) pairs differently: it takes `i = searchsorted(ccts, cct)` and interpolates the pair `(i, i + 1)` in 1/CCT, which extrapolates backwards from the next pair for CCTs between table entries, returns the last row unchanged for `i` equal to the last index and fails above 11000 K. `tests/arri.rs` reimplements that behaviour: it agrees with the bracketing rule at the 17 CCTs and differs between them by up to 6.2e-3 in a matrix coefficient (at 3700 K).

### Implementation and cross-checks

`ym_color::{consts, gamut, logc, arri, deltae}`.

- **Constants and types.** `consts` holds the primaries and whites, the normalised primary matrix (`Primaries::rgb_to_xyz`), CAT02 von Kries (`cat02_adaptation`) and the LogC constants; `gamut::matrices()` returns every derived matrix, computed once in f64. Typed linear spaces (`LinearAwg3`, `LinearAwg4`, `LinearAp0`, `LinearRec709`, `LinearDisplayP3`, `LinearRec2020`, `Xyz`) and encoded spaces (`LogC4`, `LogC3Ei800`) are newtypes over `[f64; 3]` with the conversions as methods. The LogC4 constants are stored as literals and checked against their defining formulas in the tests.
- **Table loading.** `ArriTable::builtin()` embeds `data/arri/Alexa_aces_matrix.txt` with `include_str!` (3 KB, so the runtime needs neither the data folder nor the network, and the AMPAS notice ships with the binary). `ArriTable::parse(text)` reads any file in that format. `native_to_awg3(cct)`, `native_to_awg4(cct)`, `native_to_ap0(cct)`; `bracket(cct)` returns the pair index and the mired weight.
- **Measured.** Every printed matrix is reproduced to 5e-11 (the 10-decimal rounding of the f64 values; AWG3 → AP0 to 5e-10). AWG3 → AP0 equals the JSON's `awg3_to_ap0_cat02` to 1e-12. Rows of AWG3/AWG4 → AP0 and of every same-white gamut matrix sum to 1 within 1e-15. Derived native → AWG3 matrices agree with the JSON to 5.1e-15; table rows sum to 1 within 5e-15, derived rows within 5.8e-15. LogC4 round trip over [−0.02, 400]: at most 1e-14 + 1e-12·|x|. LogC3 round trip: 5e-13 in the linear domain (the 2.5e-7 step at `cut` only shows in code values).
- **Independent implementations.** colour-science 0.4.7 agrees with LogC4 and LogC3 to 1e-14, the sRGB transfer to 1e-14 and its AWG4 matrices to 1e-9; its AWG3 matrix is stored to 6 digits and its AP0 to 10, so AWG3 → AP0 and AWG3 → AWG4 differ from ours by up to 6e-7. OpenColorIO's built-in `ARRI_LOGC4_to_ACES2065-1` (float32, about 2.5e-4 relative accuracy) agrees to 2.3e-5 relative and `ARRI_ALEXA-LOGC-EI800-AWG_to_ACES2065-1` to 1.9e-4 (the difference of the LogC3 constants above).

## C1 · Scene light and residual white balance

Crates: `ym_color::wb` (`SceneLight`, `EstimatorClasses`, `LightSource`, `picked_scene_white`, `residual_gains`, `median_rgb`) and `ym_color::chain`.

- The **illuminant estimate** (section *Illuminant estimate*) gives the scene light as camera RGB `w_scene` (green = 1) plus class probabilities.
- Raw processing balances the mosaic with `g0 = 1 / w_scene_initial` (`white_scale / w_scene_initial` for a body with a white scale, see *White scale of stand-in bodies* below). The residual gain is `g_res = w_scene_initial / w_scene` (element-wise, both normalised to green = 1). It is exactly `[1, 1, 1]` unless the scene light changed after the photo was opened (Pick white, a light preset, Re-run).

### Scene light

`SceneLight { w_scene, classes, confidence, cct_k, duv }` holds the camera white (green = 1), the four profile-class probabilities, the confidence, and the CCT and Duv read from the profile's locus (`SceneLight::new`; `from_parts` takes a CCT that is already known). The estimator's six classes fold into the profile's four (`EstimatorClasses::to_profile_classes`): daylight and shade/overcast are daylight, tungsten/halogen is tungsten, fluorescent and LED stay, and the `mixed` mass goes to the class whose reference lights cover the scene CCT (tungsten below 4000 K, daylight from 4000 K, the `E_scene` rule of C3).

The scene light is resolved in this order: the estimate, then Pick white (`ChainSettings::picked_white`, the sample), then the light preset.

### Pick white

1. The sample is the per-channel 5×5 median of demosaiced, `g0`-balanced camera RGB at the clicked point (`median_rgb`). A sample taken where a vignetting gain was applied is divided by that gain first.
2. The mosaic clips at 1.0 per channel before the gains, so channel `c` clips at `1 / w_scene_initial[c]` after `g0` (`white_scale[c] / w_scene_initial[c]` for a body with a white scale). A sample with any channel above 0.9 of its clip level is refused: "That spot is too bright to use. Pick a grey or white area that isn't blown out." (`PickWhiteError::TooBright`).
3. A channel that is not a number, or is below 1/1024 of its clip level (ten stops down), is refused: "That spot is too dark to use. Pick a grey or white area that is lit." (`TooDark`). There is no colour in the noise floor. Too bright wins when both apply.
4. The initial gains are undone: `w_scene = sample ⊙ w_scene_initial`, normalised to green = 1. The class probabilities stay as estimated, the confidence is 1, and CCT and Duv are read from the locus.

A pick changes the scene light, not the user's white. A pick sets the light source and the white balance to Auto, so the Kelvin and tint rows show the readout of the picked light and the picked sample renders neutral exactly. Moving Kelvin switches to Manual with the sensor map still at the picked scene light.

### Light presets

Colour › White balance segment:

| Preset | Scene light for the sensor map | User CCT set to |
|---|---|---|
| Auto | estimate | estimated CCT |
| Day | CIE daylight 5500 K, class daylight | 5500 K |
| Shade | CIE daylight 7500 K, class daylight | 7500 K |
| Tung. | blackbody 3200 K, class tungsten | 3200 K |
| LED | estimated chromaticity, class LED = 1 | estimated CCT |
| Fluor. | estimated chromaticity, class fluorescent = 1 | estimated CCT |

Day, Shade and Tungsten are model lights: the camera white is the profile locus at 5500 K, 7500 K (CIE daylight table) or 3200 K (blackbody table), the CCT is that temperature exactly, Duv is 0, the class probability is 1 and the confidence is 1. LED and Fluorescent keep the estimate's chromaticity, CCT and Duv and put all probability on their class. A preset replaces the estimate; Pick white replaces the estimate first, so for LED and Fluorescent the "estimated chromaticity" is the picked one, while Day, Shade and Tungsten override it. The settings do not record which came last: the front end clears `white_pick` when a preset is chosen and sets `light_source` to Auto when a white is picked.

### The scale of the locus

A **camera white** is `w = (S_R^T E, S_G^T E, S_B^T E) / (S_G^T E)` with the three curves as published in `data/cameras/sony/<body>.json`, clipped at 0 and **never normalised per channel**. The published curves keep the relative sensitivity of the channels (A7 III peaks: R 0.595, G 1, B 0.838), and that sets the raw white balance. Equivalently, it is the raw mosaic after black subtraction, scaled by `1 / white_level` per channel (the same for all three) and divided by green.

This is the scale of the as-shot gains (`w = [g_G / g_R, 1, g_G / g_B]`), of Pick white (`sample ⊙ w_scene_initial`), of the profile's `locus` and `duv_scale`, and of the illuminant estimator's input, training lights and output. The profile records the convention of each table in `provenance.light_conventions`. Anything computed from peak-normalised curves (`ssf_core.load_cameras`) is on the wrong scale for this purpose.

`sensor_map` and `alexa_white` are computed from peak-normalised curves, which is exact there because every response is divided by the response to the white. The chain is scale-free (`g_res`, the sensor map and `w_A` act on ratios of whites). `fit.sony_curves_raw()` loads the shipped curves in the published scale (one common factor per camera with the green peak at exactly 1.0; red peaks at 0.57 to 0.62 and blue at 0.78 to 0.87 of it) and `build_profiles` computes `locus` and `duv_scale` from them; `tables.locus_tables` refuses curves whose three channels peak at the same value. Against peak-normalised curves the log-chroma of a light differs by a constant per camera, `ln(R/G) += ln(peak_R / peak_G)` (−0.476 to −0.567) and `ln(B/G) += ln(peak_B / peak_G)` (−0.142 to −0.248), the same vector at every temperature, so `duv_scale` and the chord geometry do not change.

Acceptance test: `tests/wb.rs::the_profile_locus_is_the_camera_white_in_the_cameras_own_raw_scale` (the camera white of the published curves matches the blackbody locus to 2e-3).

**Measured.** A7 III, 6504 K blackbody: camera white `[0.371, 1, 0.767]` from the published curves, `[0.386, 1, 0.762]` from the camera's own ColorMatrix (its published D65 colour matrix). As-shot gains read Duv −0.0065 (A7 III) and −0.0008 (A7 IV) on the locus. The published curves and the ColorMatrix still disagree by a few percent in the white: the D65 camera white is `[0.361, 1, 0.748]` from the A7 III curves against `[0.374, 1, 0.742]` from its ColorMatrix (3.5 %), and `[0.357, 1, 0.755]` against `[0.381, 1, 0.721]` on the A7 IV (6.2 %). The ColorMatrix white of D65 therefore reads 6250 K (A7 III) and 5850 K (A7 IV) instead of 6504 K, with Duv +0.002: a bias along the locus, not a tint. V1 uses the curves as published; a body-specific calibration would remove the bias.

### White scale of stand-in bodies

A body that renders through another body's profile has raw whites on another scale than the profile's locus. Every registry body carries `white_scale = [r, 1, b]` with `w_profile = w_body ⊙ white_scale` (`docs/03-camera-profiles.md`, "White scale of stand-in bodies"). It is `[1, 1, 1]` for a body with its own curves, for a sibling of one and for a body with no data, and every expression below then reduces to the plain one bit for bit.

The scale is applied where the body's raw white enters:

- the as-shot white (`wb::scale_white`) and the estimator's prior and preview (red and blue times the scale);
- the initial gains `g0 = white_scale / w_scene_initial` (`wb::initial_gains_scaled`), so the balanced mosaic is on the profile's scale;
- the clip level of Pick white (channel `c` clips at `g0[c]`: `wb::picked_scene_white_scaled`; the picked white itself is `sample ⊙ w_scene_initial`, already on the profile's scale);
- the C7 clip level (`chain::clip_level_scaled`: the smallest `white_scale[c] / w_scene[c]`).

C1's `g_res`, the sensor map and C3 to C6 work on the profile's scale and are unchanged. `ColorEstimates::white_scale` and `ChainSettings::white_scale` carry it; `AutoEstimates::compute_scaled` takes it from `CameraMatch::white_scale`. **Measured:** three A1 photos through the A7R IV profile read Auto Kelvin 5200, 5250 and 5200 K with the scale (4450, 4400 and 4450 K without).

## C2 · Sensor map

Maps white-balanced Sony RGB to white-balanced emulated ALEV III RGB for the scene light. Exposure-invariant (linear) and white-preserving (`M·[1,1,1]ᵀ = [1,1,1]ᵀ`) by construction.

**Offline fit** (Python, `training/sensor_map/fit.py`):

1. Inputs: the camera profile's spectral curves `S_P`, the ALEXA estimate `S_A` (`data/alexa/alexa_alev3_estimate_380_780_5.json`), the 190 rawtoaces training reflectances (Apache-2.0), and reference light SPDs.
2. Reference lights per class: tungsten, blackbody 2000–4000 K every 250 K; daylight, CIE daylight 4000–11000 K every 500 K; fluorescent, CIE F1–F12; LED, CIE LED-B1…B5, LED-BH1, LED-RGB1, LED-V1, LED-V2. SPDs come from the `colour-science` datasets at fit time; only the fitted coefficients ship (`docs/14-licensing.md`).
3. For each light `E`: Sony responses `c_s[n] = S_P·(E∘R_n)` and ALEXA responses `c_a[n] = S_A·(E∘R_n)` for the 190 training reflectances `R_n`, each divided by its own response to the white (`R = 1`). Fit the white-preserving 3×3 that minimises the **CIE Lab error after ARRI's matrix**. Let `T_E` be the light's CCT for the ARRI lookup (the generating temperature for blackbody and CIE daylight lights, the Ohno 2013 CCT of the SPD for fluorescent and LED lights), `A(T)` ARRI's native → AP0 matrix interpolated linearly in mired between the bracketing table entries (clamped to 2000–11000 K, see *ARRI CCT matrices*), and `Lab(·)` the CIE Lab of an AP0 triple with the ACES white (AP0 → XYZ, white = ACES white). The parameters are the 6 free coefficients `m = (m0 … m5)` of

   `M(m) = [[m0, m1, 1−m0−m1], [m2, m3, 1−m2−m3], [m4, m5, 1−m4−m5]]`

   (rows sum to 1, so `M·[1,1,1]ᵀ = [1,1,1]ᵀ`), and the objective is

   `J(m) = Σₙ ‖ Lab(A(T_E)·M(m)·c_s[n]) − Lab(A(T_E)·c_a[n]) ‖²`   (Euclidean distance in Lab, 190 patches, no weights).

   `J` is minimised by Levenberg–Marquardt with an analytic Jacobian and a Gauss–Newton polish, started from the weighted linear solution (weights `Y_target^(−2/3)`, `fit.fit_map_weighted_rgb`). Start, solver and options are fixed, so the fit is deterministic: the coefficients agree to about 1e-13 whatever the start (`python3 -m training.sensor_map.fit --stability`), and the shipped tables are checked to 5e-9 (`build_profiles --check`).

```python
def residuals(m):                       # 190 × 3 Lab differences, flattened
    M = white_preserving(m)             # rows [m0, m1, 1 - m0 - m1], ...
    return (ap0_to_Lab((src @ M.T) @ A.T) - ap0_to_Lab(dst @ A.T)).ravel()    # A = A(T_E)

w = clip(Y_target, 1e-3, None) ** (-2/3)                       # start: weighted RGB least squares
Aw = stack([src[:,0] - src[:,2], src[:,1] - src[:,2]], axis=1) * w[:,None]
for r in range(3):
    m, *_ = lstsq(Aw, (dst[:, r] - src[:, 2]) * w)
    M0[r] = [m[0], m[1], 1 - m[0] - m[1]]
m = levenberg_marquardt(residuals, start=M0[:, :2].ravel(), jac="analytic")    # then Gauss-Newton polish
```

4. Store per profile, for each class, a list of `(mired, M)` sorted by mired.

**Runtime:** `M_map = Σ_k p_k · M_k(mired_scene)`, where `p_k` are the class probabilities and `M_k(mired)` interpolates linearly in mired within class `k` and clamps at the ends. A convex combination of white-preserving matrices is white-preserving.

**Optional upgrade:** a tiny MLP (at most 5k parameters) from the illuminant estimate's features to the 6 free coefficients, kept only if it lowers held-out-light ΔE00 by at least 0.05 median against the table blend. It is not built.

**Gate:** 99 CIE CES samples under 9 test lights outside the fit grid, compared with rendering the same reflectances through `S_A` directly. Per light, ΔE00 median ≤ 0.5 and p95 ≤ 1.2. The gate lights, held out of the gate fit, are blackbody 2850 K, CIE daylight 5250 K and 6750 K, CIE FL3.1, FL3.8 and FL3.15, and LED-B3, LED-RGB1 and LED-V2. The first three and the FL3 series fall between or outside the fit grid by construction; for the gate run the profile is refit with LED-B3, LED-RGB1 and LED-V2 removed from the LED class (the shipped fit uses all lights). `python3 -m training.sensor_map.gate --expect-failing LED-V2` runs it.

**Measured** (A7R IV): 8 of the 9 lights pass, every other light below 0.42 median and 1.18 p95. LED-V2 (4070 K) misses the gate with median 0.759 and p95 1.515, and is a reported gate failure.

## C3–C4 · ALEXA white

`w_A(E) = ∫ S_A(λ) E(λ) dλ`, normalised to green = 1 (5 nm sums, 380–780 nm).

- C3 multiplies by `w_A(E_scene)`. `E_scene` is the SPD of the estimated light: blackbody below 4000 K, CIE daylight from 4000 K (the rawtoaces convention, also used to fit our ALEXA curve). For the LED and fluorescent classes it is the precomputed `w_A` of the class's reference lights (in `profile.json`), interpolated in mired and blended by class probability.
- C4 multiplies by `g_user = 1 / w_A(E_user)`, where `E_user` is blackbody or daylight at the user's CCT (same 4000 K rule), then applies tint.
- **Auto white balance** sets `E_user = E_scene`, so C3·C4 is the identity and the scene white lands exactly on neutral. The Kelvin shown is the scene CCT rounded to 50 K; the tint shown is the value that reproduces the scene's distance from the locus.
- **Tint** `t ∈ [−50, 50]`, positive is magenta: the green gain is multiplied by `2^(−t/100)` (±0.5 stop at the ends). ARRI's tint mapping is unpublished; this one is ours.
- `w_A` for blackbody and daylight on a 10 K grid (1500–25000 K) is **precomputed offline** by the fit scripts into `profile.json` (`alexa_white`), together with the camera locus. The runtime only interpolates these tables, so no CIE datasets (observers, daylight basis) are compiled into the app. The `colorimetry` crate and colour-science are training and test dependencies only.

Crates: `ym_color::wb::{scene_alexa_white, scene_tint, tinted_white, auto_readout, manual_user_gains}`.

### Scene tint

The class model of C3 (`w_A` of a blackbody or CIE daylight at the scene CCT, or the LED and fluorescent lists) has no off-locus component, yet a real photo's light sits off the locus and the Tint row has to show how far. So `w_A(E_scene)` carries the scene's own tint. The camera white's signed log-chroma distance from the locus is `d = Duv / duv_scale(CCT)` (positive towards magenta). The green gain that would cancel it gives the scene tint

`t_scene = −100 d / (√2 ln 2)`

(about 102 tint units per log-chroma unit; Duv 0.01 reads a tint of 11 at 5500 K, 16 at 3200 K and 23 at 2500 K), weighted by the daylight plus tungsten probability (the LED and fluorescent class whites already contain their reference lights' distance from the locus) and held in −50…50. Then `w_A(E_scene) = [g w_r, 1, g w_b]` with `[w_r, 1, w_b]` the class-blended `w_A` of the scene CCT and `g = 2^(−t_scene/100)`. A greener light gives a positive tint (the correction adds magenta), a magenta light a negative one, and a preset on the locus gives exactly 0. The Duv is the camera's readout, so it is only as good as `docs/03-camera-profiles.md` says: spiky spectra differ from CIE Duv, and between about 3800 K and 4300 K a Duv beyond ±0.002 is not trustworthy, nor is this tint.

### Auto

`g_user = 1 / w_A(E_scene)` with that tinted, class-blended `w_A` (not the model at the CCT), and the C3·C4 product is set to exactly `[1, 1, 1]`. **Measured:** the scene white lands on `2^EV [1, 1, 1]` to 6e-15 (all 11 profiles, every class mix, CCT 1900–12000 K, Duv −0.01…0.007, any initial white; `tests/chain.rs`).

### Readout

Kelvin is the scene CCT rounded to 50 K and held in 2000–11000 K. Tint is the value that reproduces the scene's distance from the locus. The gains of `Manual { kelvin, tint }` act on the scene white `w_s = w_A(E_scene)` as `[w_s,r / w_u,r, 2^(−t/100), w_s,b / w_u,b]` (`w_u = w_A` at that Kelvin). The three agree only when the scene sits on the locus, and the tint that comes closest on the green axis is

`t = −50 log₂((w_s,r / w_u,r)(w_s,b / w_u,b))`

held in −50…50 and computed for the rounded Kelvin that is shown. Consequences, all tested:

1. `Manual` at the shown Kelvin and tint renders the Auto picture times the green gain `2^(−t/100)` to 1e-12, for daylight and tungsten scenes whose CCT is a multiple of 50 K. The tint scales green only, so it changes brightness by at most half a stop at ±50 and by 3.5 % at ±5.
2. With the Kelvin rounded to 50 K, the scene's ALEXA white is off neutral by the colour of up to 25 K: at most 1.6 % per channel at 2000 to 2300 K (a kelvin is a bigger step in mired at the warm end), 1.3 % up to 4000 K (the junction included) and 0.5 % above (measured every 3 K on all 11 profiles; `tests/wb.rs`).
3. For an LED or fluorescent scene a green gain cannot cancel the red/blue offset of the class white, so the shown pair leaves up to 3.4 % (A7R IV, LED, 2750–6500 K).
4. Applied to the camera white as a green gain, the tint cancels the cast to within 4.1 % of its Duv in a round trip through the locus readout (worst of 11 profiles).

### Manual

Kelvin is held in 2000–11000 K and tint in −50…50 (`WhiteBalance::manual` and the chain refuse values outside). `E_user` is the blackbody table below 4000 K and the CIE daylight table from 4000 K, `g_user = [1 / w_r, 2^(−t/100), 1 / w_b]`, and the user's Kelvin picks the ARRI matrix.

The 4000 K rule makes `w_A` step (red −1.2 %, blue −3.4 % going up, `docs/03-camera-profiles.md`). A Kelvin slider dragged across 4000 K moves the scene white by 1.1 % red, −0.4 % green and 4.4 % blue (A7R IV, daylight scene; about 0.06 stop in blue). The rawtoaces convention is kept: cross-fading the tables over 3500–4500 K would make `w_A` a mixture of two lights that are not on the same locus.

**Direction of a Kelvin setting.** The user's Kelvin is the light the ALEXA is balanced for, as on the camera: a daylight scene set to 3200 K renders **bluer** than Auto, and a tungsten scene set to 5500 K renders warmer. For the A7R IV, CIE D 5500 K, 3200 K, the AWG4 white is `[0.6747, 0.9488, 1.6434]` (blue over red 1.28 stops, green over red 0.49 stops; Auto is `[1, 1, 1]`). This equals the independent spectral simulation of ARRI's balance (ALEXA estimate curves, Planck's law or the CIE D lights of the exporter's fixtures, the 17-CCT matrices) to 3e-9 for a white patch (312 cases: 3 bodies, 13 lights of all four classes, Auto and 8 Kelvin settings; `tests/chain_spectral.rs`). The 24 ColorChecker patches through the sensor map differ from the simulation by the map's error: 0.47 ΔE00 median at worst and 1.4 at most (3 bodies, Auto, 2800/3200/5500 K).

### Apple Auto nudge

With Apple Auto's answer (`docs/19-apple-auto.md` section 5), Auto (light source Auto, white balance Auto, no picked white) moves the user's white. A light preset, a picked white and a manual white balance get no nudge.

**Warmth.** The answer's `white` step (−2…+2; a warmer picture is a negative `m`) gives `m = −10 × step` mired:

`CCT_user = 1e6 / (1e6 / CCT_scene + m)`, held in 1500–25000 K (the white tables).

The user's white is the scene's tinted, class-blended white moved along the locus to `CCT_user`, channel by channel:

`w_A(E_user) = w_A(E_scene) · w_locus(CCT_user) / w_locus(CCT_scene)`

with `w_locus` the blackbody/daylight model (4000 K rule), so it keeps the scene's tint and class mix. `g_user = 1 / w_A(E_user)`, C3·C4 is `w_A(E_scene) / w_A(E_user)` (green exactly 1), and the ARRI matrix takes `CCT_user` held in 2000–11000 K, as for a manual Kelvin. For daylight and tungsten light this equals the class-blended white evaluated at `CCT_user`, and the nudged chain equals `Manual { CCT_user, 0 }` on the locus to 1e-9. Moving the scene's own white as the locus moves is used for every class because the fluorescent and LED class whites are lists of real lights, not ordered in colour and constant beyond their ends: evaluating them at `CCT_user` moves the white the wrong way, or not at all, on every shipped profile. The readout is `auto_readout`'s for the nudged white: Kelvin `CCT_user` rounded to 50 K, and the tint that reproduces it, which is the scene's tint up to the rounding for daylight and tungsten light. `m = 0` takes none of this path: the chain is Auto's bit for bit (all 11 profiles, every class).

**Tint.** The answer's `tint` step adds `Δt = 5 × step` (−10…+10, positive magenta) to the scene's own tint, the sum held in −50…50. The user's white (moved along the locus first, when there is a warmth nudge too) takes the tint as a manual tint is given to a white (`tinted_white`): red and blue times `2^(−Δt/100)`, green 1. The readout's tint moves by `Δt` and its Kelvin stays. `Δt = 0` takes none of this path, and a warmth nudge without a tint nudge is the warmth nudge above unchanged.

API: `ym_color::wb::{nudged_cct, nudged_alexa_white, nudged_readout, nudged_tint}`, `ChainSettings::{white_shift_mired, tint_shift}`, `ColorEstimates::{white_shift_mired, tint_shift}`. Tests: `crates/ym-color/tests/white_nudge.rs` (±10 and ±20 mired move the scene white's R/B in AWG4 and the Kelvin readout the right way, monotonically, for every class, on and off the locus, 2200–10000 K; a magenta nudge takes green out of the picture, both signs, every class).

### Camera RGB ↔ CCT and Duv

For a profile, the fit scripts tabulate camera RGB (green = 1) of blackbody 1500–4000 K and daylight 4000–25000 K every 10 K, as log-chroma `(ln R/G, ln B/G)`, plus the matching CIE xy and the Duv scale, into `profile.json` (`locus`). The camera RGB is in the camera's **raw scale** (the published curves, not normalised per channel; see *The scale of the locus*), so it is comparable with the as-shot gains, Pick white and the illuminant estimate.

`w_scene` converts to CCT by the nearest point on that polyline, with linear interpolation between neighbours. Duv is the signed perpendicular distance, positive towards magenta, rescaled to CIE Duv through the profile's XYZ estimate. This gives the Kelvin and tint readouts and the light presets.

## C5–C6 · ARRI matrix, AWG4, exposure

- `M_arri(CCT_user)` from the interpolated table (*ARRI CCT matrices*). The CCT is the scene CCT in Auto and the user's Kelvin in Manual; `ArriTable` clamps it to 2000–11000 K.
- `M_awg3→awg4` from the constants.
- Exposure: `EV = EV_auto` in Auto, or the slider's absolute value in Manual. The slider range is −4…+4 EV absolute; its detent sits at `EV_auto`.

Crates: `ym_color::chain::{FusedChain, ChainSettings, ProfileTables, ChainStages, ClipLevel, ColorEstimates}`.

- `FusedChain::build(tables, &estimate, w_scene_initial, &settings)` returns the 3×3 (`matrix()`), the factors (`stages()`: `g_res`, `sensor_map`, `w_a_scene`, `g_user`, `c3_c4`, `user_cct_k`, the ARRI and AWG3→AWG4 matrices, `2^EV`), the resolved scene light, the Kelvin and tint to show (`readout()`: the Auto readout, or the user's values for Manual) and the clip level. The matrix is the product of the formula in *Stage order*, checked stage by stage on random pixels to 1e-13.
- `ProfileTables` is the three tables of a profile that the chain reads (`profile.sensor_map.classes`, `profile.alexa_white`, `profile.locus`), so `ym-color` does not depend on `ym-profiles`.
- The exposure must give a usable gain: `FusedChain::build` refuses an EV that is not finite or whose `2^EV` overflows or underflows (`ChainError::InvalidExposure`, beyond about ±1000 stops), and `ColorEstimates::chain` refuses a manual exposure outside −4…+4 (`ExposureSetting::manual`/`validate`, `ChainError::Exposure`). The automatic value is not range-checked (the highlight guard can take it to −4).
- `ColorEstimates { scene, w_scene_initial, ev_auto }` is the colour part of `AutoEstimates`. `.chain(...)` resolves `ExposureSetting::{Auto, Manual { ev }}` against `ev_auto`; `.metering_chain(...)` builds the all-Auto, EV 0 chain that auto exposure meters through. Metering happens once at open and on Re-run, so later white-balance edits do not move the exposure.

**Clip level for C7.** A neutral surface brightens until the first camera channel clips. After C1 it is `[s, s, s]` and channel `c` clips at `s = 1 / w_scene[c]`, so the first clip is at `balanced = min_c 1 / w_scene[c]` (1 for daylight on a Sony sensor, below 1 for a light so red that red clips first). Mapped through the matrix this is `M·[balanced; 3]` (`FusedChain::clip().awg4`; `clip_level(&matrix, w_scene)` for any matrix). C7's highlight range reads it as a luma, `ClipLevel::luma()` (the Rec.709 weights on that triple, the norm the stage applies to every pixel); `ClipLevel::n`, its largest channel, is kept as a diagnostic. In Auto the triple is neutral, `2^EV · balanced` on every channel, so both are `2^EV · balanced`; with a manual white balance the clipped neutral is coloured and the luma is the one C7 compares pixels with.

### Auto exposure

Goal: the exposure the camera chose, with a safety net for highlights, stable across a burst. Auto exposure **is the camera's own exposure**: Yellowmoon meters nothing of its own. A photo processor does not need to rescue a shot (the Exposure slider does that), and an ALEXA has no auto exposure either.

1. Run C1–C6 with EV = 0 on the 1/8-resolution preview (balanced with `g0`, the lens vignetting gain of R3 applied, so the guard sees the picture that is rendered). Compute linear luminance `Y` with the AWG4 `Y` row.
2. Ignore pixels in the clip mask and pixels with `Y` below the 1st or above the 99th percentile of the unmasked pixels.
3. `m = exp(Σ wᵢ ln Yᵢ / Σ wᵢ)`, with the centre weight `wᵢ = exp(−r²/(2·0.35²))` (`r` = distance from the centre over the half-diagonal; `w` is 1 at the centre and 0.017 in a corner). `m` is the scene's key. It is reported as a diagnostic and **does not move the exposure**.
4. `EV_auto = clamp(CAMERA_GAIN_EV, −3, +3)` with `CAMERA_GAIN_EV = +1.11 EV`: the same number for every photo. It is the median gain of the cameras' own JPEG engines (raw level to the level the JPEG renders as middle grey, `log2(0.18 / x)`): +1.11 EV over 12 photos from 9 Sony bodies, std 0.13, range +0.87 … +1.33, with no relation to the scene's key. An ideal ISO 12232 camera would be +0.79; Sony places grey higher. There is **no exposure-compensation term**: the raw data already holds the photographer's compensation, so with a constant gain it reaches the picture in full, as it does in the camera's JPEG (a scene shot with +1 EV of compensation renders exactly one stop brighter than the same scene without). A shot the camera exposed badly stays badly exposed, and the slider fixes it.
5. **Highlight guard:** if more than 0.5 % of the non-clipped pixels would exceed linear 8.0 after exposure, lower `EV_auto` until they do not, by no more than 1 EV. The guard runs after step 4 and can only lower the exposure. 8.0 is 5.47 stops over middle grey, about the clip of an ALEXA (ALEV III) at EI 180, not at EI 800 (7.64 stops over grey, linear 35.9; C7). On Sony data the guard has not acted on any local photo.
6. **Bursts:** files of the same camera within 2 s of each other and with identical exposure settings share the median `EV_auto`. With a constant gain the median is that constant unless the guard acted on a frame.

The **white lift** (C12 step 1b) is the other automatic estimate made on this preview, right after the exposure and at the exposure the photo renders with; it is not part of the exposure rule and does not move it. The automatic settings of Apple Auto (`docs/19-apple-auto.md`) add their step to the rule's value after the guard and burst sharing.

Crate: `ym_color::exposure`, pure functions over plain buffers (`PreviewRgb`: interleaved f32 RGB, the 1/8-resolution preview balanced with `g0`, and an optional clip mask) and small metadata structs, with no decoder or GPU dependency. `auto_exposure(&preview, &matrix_ev0)` is steps 1 to 5 (the 1188×792 preview of a 61 MP frame takes about 0.2 s on a loaded machine; the camera's exposure compensation is not an input) and `share_burst_ev(&frames, &ev_auto)` is step 6. The constants `CAMERA_GAIN_EV` (+1.11), `EV_LIMIT` (3), `GUARD_LEVEL` (8.0), `GUARD_FRACTION` (0.005), `GUARD_MAX_REDUCTION_EV` (1) and `BURST_WINDOW_S` (2) are in `ym_color::exposure`; `TARGET_GREY` (0.18) is the reference of the reported key and of the gain. Where the steps are silent:

- **Luminance.** `Y` is the AWG4 `Y` row applied to `M·rgb` (folded into one row vector), raised to a floor of 1e-4 before the logarithm of the reported key, so black and negative pixels (noise after black subtraction) count as very dark rather than missing. A pixel with a non-finite channel counts as masked. A lens-cap frame, a flat frame and an empty preview all get the camera gain.
- **Percentiles** are the linearly interpolated quantiles (numpy's default) of the sorted unmasked luminances, with the mask applied first; a pixel is kept when `q01 ≤ Y ≤ q99`. The interpolation is written `a + (b − a)·f`, which returns exactly `a` for equal neighbours (the form `(1 − f)·a + f·b` is an ulp off for some, and on a flat frame the 1st percentile would land above every pixel and the band would be empty). If the band is empty anyway (two distinct values and nothing else) every unmasked pixel is metered.
- **Highlight guard.** Over every unmasked pixel (not only those kept by the percentiles, which would remove the highlights it looks for), `Y` after exposure against 8.0. Solved in closed form: the largest EV at which at most `floor(0.005 n)` pixels exceed 8 (from the `(floor(0.005 n) + 1)`-th brightest `Y`), and `EV_auto = max(that, EV_before_guard − 1)` when it is below `EV_before_guard` (the camera gain, +1.11). The result lies in +0.11 … +1.11 and never raises the exposure. `AutoExposure` reports `ev_before_guard`, `guard_reduction_ev`, `mean_luminance` (the key statistic) and the pixel counts.
- **No usable pixel** (everything masked, or an empty preview): `EV_auto = clamp(CAMERA_GAIN_EV)` and `metered = false`.
- **Bursts.** A frame joins a group when the previous frame (in time order) with the same camera string and identical settings (shutter, f-number, ISO, exposure compensation; equal to 1e-9 relative) is at most 2 s earlier. Those are consecutive gaps, so a burst of any length is one group, and a time-lapse at fixed settings with gaps under 2 s is one group too. The median of an even group is the mean of the two middle values; a frame without a capture time is alone. Step 6 follows steps 4 and 5, so a frame whose highlight guard lowered its own EV takes the group median, which can undo the guard for that frame.

#### Evaluation

**The gate** (metric G3, `tools/eval/autoexp.py --gate`; `docs/13-testing.md`, `docs/15-milestones.md`): `G3 = EV_auto − camera gain`, where the camera engine's gain is `log2(0.18 / x)` for the scene-linear level `x` that the embedded JPEG renders as mid grey (ISO 12232, 0.18 in linear light), read by quantile matching of the JPEG's luminance against our scene luminance with both pictures at the JPEG's size. The gate is met when at least 80 % of the files have |G3| ≤ 0.5 EV. A file that fails to analyse, or whose gain cannot be read, counts as a miss; the exit code is 3 when the gate is not met; the outliers (with the camera's key and DRO setting) are printed and written to `report.md` and `results.json`. Two other metrics are reported: **G1**, the median luminance of our scene-linear render at `EV_auto` against the JPEG's (`preview_median_luminance`, `jpeg_median_luminance`, `exposure_error_ev(ours, reference) = log2(ours / reference)` and `evaluate_gate(&entries, 0.5, 0.8)` in `ym_color::exposure`; the tests decode the JPEG with `zune-jpeg`, a dev-dependency, MIT OR Apache-2.0 OR Zlib), and **G2**, the same on the display render. G2 cannot gate exposure, because the display transform places mid grey at a different level from the camera's JPEG.

**Measured.** With the constant `EV_auto`, G3 is `1.11 − camera gain` per file: it shows how far each camera is from the median of the cameras. On the 12 local files (nine bodies of `testdata/real/files` and three A1 photos; local evaluation only, licences of nine unclear): **12 of 12 within ±0.5 EV** (10 needed), G3 median −0.00, mean +0.02, std 0.13, range −0.22 … +0.23. G1: 12 of 12 within, median −0.02, std 0.27. G2: 11 of 12 within, median +0.02, std 0.31, range −0.51 (the flash portrait of the DSLR-A300) … +0.44 (A7 II). G3 against log2 of the JPEG median has a slope of −0.04 EV per EV, so the exposure does not depend on the scene. The highlight guard acted on none of the 12 files: after the +1.11 EV gain the brightest 0.5 % of a Sony frame sits 2.3 … 4.5 EV below the level of 8.0 (A7 II 2.3, A1M01739 2.4, A1M01749 3.4, A7 III 4.5).

The **synthetic tier** (37 chart scenes, one raw variant each; the `golden` job of `ci.yml` runs the gate on every push) gives 37 of 37 within ±0.5 EV, G3 median +0.01 EV, std 0.03, range −0.11 … +0.06: by construction, because the synthetic camera's JPEG engine is Sony-like (`training/synth/preview.py`: gain 2.133, whose own gain is +1.0654 EV and which the estimator reads at a median of +1.104 EV, range +1.05 … +1.22). The tier checks the machinery, not Sony. It also checks the guard (`summary.guard_check`: `ym` and the Python replica agree, nothing was lowered because no chart scene would clip, nothing was raised) and the compensation rule (a scene exposed 2^EC brighter with that EC renders exactly 2^EC brighter). The Rust tests are `crates/ym-color/tests/exposure.rs` (the guard lowers only scenes that would clip, by the documented amount, at most one stop), `crates/ym-core/tests/auto.rs` and `crates/ym-core/tests/exposure_compensation.rs`.

**Limits.** `CAMERA_GAIN_EV` is fitted on the same 12 local files that the gate measures, so a pass says little about other cameras (it is in-sample). Re-fitting it on the golden set is task M1-H5; the rule stays. The camera gains read from the JPEGs are good to about 0.1 EV: a registered, per-bin reading of the NEX-5N comes out 0.15 EV higher than the quantile match, and files shot with DRO (a local tone mapping that lifts shadows) include the effect of DRO in their effective gain.

## C7 · Highlight range and low-light toe

Two gains that keep the channel ratios of a linear AWG4 pixel: the **highlight range** acts on its luma above a knee, the **low-light toe** on its norm `n = max(R, G, B)` near black. The range comes first. They never act on the same pixels: the range is exactly 1 below +1 stop over middle grey, the toe acts within a few noise floors of black.

### Highlight range

Setting `rolloff.highlights = r ∈ [0, 1]`, default 1.

An ALEXA (ALEV III) at EI 800 clips 7.64 stops over middle grey; a Sony body at the camera's exposure clips about 3.6 stops over it (3.58 on all 12 local photos at Auto). Below the Sony's clip the chain already holds what an ALEXA exposed for the same grey would record, so the one difference is where the clip lands. The stage maps the Sony's top stop onto the ALEXA's range: nothing changes below a knee one stop under the Sony clip, the Sony clip is placed on a target between where it was recorded (`r = 0`) and the ALEXA's EI 800 clip (`r = 1`), and nothing goes past the ALEXA's clip. `r = 1` is what an ALEXA's own video output does with its clip ("normalized": the sensor clip at 100 % video at every EI); `r = 0` is ARRI's "photometric" conversion (the clip renders where its scene level puts it).

The range works alongside the white lift of C12 step 1b: the range moves the camera's clip, and the lift brings a photo's brightest areas to a real white when they sit well under the clip, where the range does nothing.

```
Y    = 0.2126 R + 0.7152 G + 0.0722 B          luma of the linear AWG4 pixel (Rec.709 weights, REC709_LUMA_WEIGHTS)
s    = log2(Y / 0.18)                          stops over middle grey
h_S  = log2(Y_clip / 0.18)                     the Sony clip: Y_clip = ClipLevel::luma(), the clipped neutral after C1-C6
h_A  = log2((1 - 256/65535) · 800 / 4)         the ALEXA (ALEV III) clip at EI 800: 7.638209540303716 stops (linear 35.86)
s_k  = max(1, h_S - 1)                         the knee: one stop under the Sony clip, never under +1 stop
w    = clamp((h_S - s_k - 0.25) / 0.75, 0, 1)  fade: 1 when the clip is 2 stops or more over grey, 0 at 1.25 or less
E    = w · r · max(h_A - h_S, 0)               how far the Sony clip moves, in stops (it lands on T = h_S + E)
u    = (s - s_k) / (h_S - s_k)                 only where E > 0, so h_S - s_k >= 0.25
phi  = u²  (u <= 1),   2u - 1  (u > 1)         C¹ at the clip; values R5 rebuilt above the clip keep the clip's slope
s'   = s + E · phi   where s > s_k and E > 0,   s elsewhere
s'   = min(s', h_A)                            the ALEXA's hard clip
gain = 2^(s' - s)                              rgb' = gain · rgb, all three channels
```

`h_A` is ARRI's own formula for the ALEV III clip (the LogC3 VFX document's clipping level and ARRI's ACES input transform for ARRIRAW: grey at `0.01 · 400 / EI` over black, linear up to the 16-bit white; it reproduces ARRI's LogC3 clip code 0.9539 at EI 800). ARRI's data sheets round it to 7.4 stops; the formula is used. The target EI is fixed at 800, ARRI's base EI. The knee depth (1 stop), the floor of the knee (+1 stop), the fade and `phi` are taste, not ARRI figures.

**Properties** (all tested):

- The gain is exactly 1 for `s <= s_k`: the stage returns the input unchanged, bit for bit, so mid grey and everything under +1 stop never move.
- `s'` is continuous with a continuous slope at the knee and at `u = 1`, and non-decreasing in `s` (slope at least 1 up to the cap, 0 above it).
- The Sony clip lands on `T = min(h_S + E, h_A)`.
- The three channels are multiplied by the same number, so a pixel's colour ratios are kept exactly. Below the cap every output channel is non-decreasing in each non-negative input channel (a channel times a non-negative, non-decreasing function of a luma with positive weights). At the cap the luma is held at the ALEXA clip with the ratios kept, so raising one channel there lowers the others' output (what a hard clip of the luma does).
- `r = 0` is the identity below the ALEXA clip.
- When `h_S >= h_A` (Exposure about +4 EV over Auto, or more), `E = 0` and the stage is the identity below `h_A` and the ALEXA's hard clip above: the only case where an ALEXA sensor clip acts on recorded data.

**Exposure.** `Y_clip` is measured after C6, so it includes `2^EV` and `h_S` follows the Exposure slider as an ALEXA's clip follows its EI. With `r = 1` the clip stays on the ALEXA clip at every exposure where the fade is 1, so darkening the photo does not darken a clipped sky (ARRI's "normalized" behaviour). The fade keeps the map continuous when the clip comes within two stops of grey (Exposure about 1.6 stops under Auto): the expansion falls off to nothing at 1.25 stops (2.3 stops under Auto), where the knee would otherwise reach middle grey.

**Edge values.** A pixel with a non-finite channel passes through unchanged, as in the rest of C7; `Y <= 0` (black, or all channels at or under zero) gets a gain of 1. `E`, `s_k` and `h_S` are computed once per chain on the CPU in f64; a `Y_clip` that is not a positive finite number is refused when the chain is built (`ToneError::InvalidClipLevel`).

**Where it acts.** C7 is scene-linear, before the LogC4 encode, so everything after it sees the range: the grade, looks, the curve, ACES 2, the DNG's pixels, and **an imported ARRI display LUT and the LogC3-to-Rec.709 looks**, which receive values up to the ALEXA clip (LogC4 0.760, LogC3 EI 800 0.954: the range ARRI made them for) and apply ARRI's own shoulder to them. The Before view (the camera's default rendering) does not have it.

**Display result** (computed for a neutral, ACES 2 with the placement, no look, sRGB codes; `crates/ym-color/tests/tone.rs`). Stops over grey at Auto and the code they render as:

| Stops over grey | +1 | +2 | +2.58 | +3.08 | +3.33 | +3.58 (the clip) |
|---|---|---|---|---|---|---|
| `r = 0` | 158 | 192 | 208 | 220 | 224 | 229 |
| `r = 0.5` | 158 | 192 | 208 | 229 | 240 | 248 |
| `r = 1` | 158 | 192 | 208 | 236 | 249 | 254 |

### Low-light toe

Setting `rolloff.low_light = r`, default 0.5. `τ = 4r · σ_floor`, where `σ_floor` is the noise model's read noise mapped to linear AWG4 at this exposure. The toe scales the pixel by `f(n)/n` with `f(n) = n³ / (n² + τ²)`, i.e. `n² / (n² + τ²)`: `f` is monotonic, about `n` above 3τ, and crushes the noise floor smoothly below τ. `r = 0` disables it (the exact identity).

### Implementation

`ym_color::tone::ToneParams::new(highlights, low_light, clip_luma, noise_floor)`, `apply(LinearAwg4)`, `apply_rgb(Vec3)`. `ToneParams::range()` is the prepared `HighlightRange { knee_stops, clip_stops, expansion_stops, cap_stops }` (`s_k`, `h_S`, `E`, `h_A`) with `stops(s)` and `gain(luma)`. The constants `HIGHLIGHT_CAP_STOPS` (`h_A`, with `alexa_clip_stops(ei)` the formula), `HIGHLIGHT_KNEE_DEPTH_STOPS` (1), `HIGHLIGHT_KNEE_FLOOR_STOPS` (1) and `HIGHLIGHT_MIN_SPAN_STOPS` (0.25) are in `ym_color::tone`. The pixel is `toe(range(rgb))`: the range's gain on the luma, then the toe's scale `n² / (n² + τ²)` of the gained pixel's norm (for `n <= 0` the toe's scale still applies, so a pixel with `n = 0` and `τ > 0` goes to black). Non-finite input passes through.

Tests (`crates/ym-color/tests/tone.rs`): the formulas against an independent f64 composition; exact identity below the knee; slope continuity at the knee and at `u = 1`; `s'` non-decreasing over a grid of amounts × clip levels × noise floors; the clip landing on `T`; the cap; `r = 0` the identity below the cap; the fade at low clip levels; each output channel non-decreasing in its own input below the cap (proptest); the toe.

## C8 · LogC4 encode

Per channel with the formula above. Negative linear values (out-of-gamut LED colours) pass through the linear segment unclipped. Nothing clips before the display transform.

## C9 · Primary grade (LogC4)

Let `v` be LogC4 RGB. `L = encode(Y_lin)` uses the luminance of the linear values before C8. Grey `g = 0.2784`. One stop in LogC4 is `b/14 = 0.06480` code values; call it `S`.

| Control | Range | Operation |
|---|---|---|
| Contrast `c` | −100…100 | `u = v + (T(L) − L)` on all three channels: the tone curve `T` below, with the slope `k = 2^(0.6·c/100)` at grey, fixed at black, grey and the top code value; then `u`'s chroma, keeping its direction, scaled part of the way (`β = 0.5`) toward the chroma length of `T` on each channel (the colour step below) |
| Brightness `b` | −100…100 | `v += (b/100) · S · bump(L; g, 2.5S)`: a midtone lift that leaves black and white |
| Shadows `s`, Highlights `h`, Black `k`, White `w` | −100…100 | `v += (x/100) · 1.5S · bump(L; centre, 1.5S)` with centres at grey −3, +3, −6, +6 stops |
| Colour balance (shadows, midtones, highlights) | `(a, b)` in the unit disc | `v += 0.03 · (a·e₁ + b·e₂) · zone(L)`, `e₁ = (1, −½, −½)/√1.5`, `e₂ = (0, 1, −1)/√2`; zones centred at −3, 0, +3 stops |
| Saturation `s` | −100…100 | in linear AWG4, `rgb′ = Y + λ·(rgb − Y)`, `rgb = decode(v)` of the pixel at this point of C9 (after the rows above) and `Y` its luminance: `λ = 1 + s/100` for `s < 0`; for `s > 0`, `λ = r′/r` with `r = (Y − min(rgb))/Y`, `r′ = 1 − (1 − r)^F`, `F = 1 + α·s/100`, `α = 1.5`; then encoded to LogC4 again |

`bump(L; μ, σ) = exp(−(L − μ)²/(2σ²))`. The rows apply in the order of the table. The constants are taste values (`k`'s exponent 0.6, `β = 0.5`, `α = 1.5`, the zone centres and widths) and are tunable.

**Auto inputs of this stage** (`docs/19-apple-auto.md` section 5). Contrast, Saturation, Highlights, Shadows and Black are Auto or a manual value, and so are the shadows and highlights colour-balance wheels (a manual point); the midtones wheel stays a point. Auto is 0 (the wheels (0, 0)): the grade above without that row. When Apple Auto answered for the photo and set a value, Auto is the answer's step in the table (Highlights −40, −20, 0, +20, +40; the others −30, −15, 0, +15, +30) and, for a wheel, the colour's direction at its strength's radius (0.3, 0.55, 0.8). A value the answer left empty stays 0. The grade takes the values in effect (`ym_core::pipeline::color::grade_params` resolves them against the photo's estimates); nothing in this stage's maths changes.

### Contrast

Contrast is a **tone curve on the luma**. With black `a = encode(0) = 0.0929`, grey `g`, the top code value `b = 1` (linear 469.8, 11.35 stops over grey; here `b` is not LogC4's constant) and `h_k(t) = k·t / (1 + (k − 1)·t)` (`h(0) = 0`, `h(1) = 1`, slope `k` at 0 and `1/k` at 1, increasing for every `k > 0`):

- `g ≤ x ≤ b`: `T(x) = g + (b − g)·h_k((x − g)/(b − g))`;
- `a ≤ x < g`: `T(x) = g − (g − a)·h_k((g − x)/(g − a))`;
- above `b` and below `a`: the straight line of slope `1/k` through the end, so `T` is defined, continuous, has a continuous slope and is increasing for every `x`.

The change `T(L) − L` is added to all three channels: `u = v + (T(L) − L)`. On the log segment a common offset multiplies `x + 64/a` (`x + 0.029`) by a common gain, which is a common gain in linear light wherever `x` is well above 0.029, so the channel ratios, and with them the hue, are kept while the luma follows `T`. In plain words: the slider has slope `2^(0.6·c/100)` at mid grey, and instead of stretching the whole range about grey it bends toward black and the brightest code, which stay where they are (more contrast flattens the tones near them, less steepens them). `c = 0` skips the row (bit-exact identity). The rows after Contrast use the original `L`.

**The colour step.** On its own the offset adds no colour, so Contrast adds part of the colour a curve on each channel would. With the chroma of a LogC4 triple `x` in the colour-balance plane, `c_x = (e₁·x, e₂·x)`:

1. `p = (T(v_r), T(v_g), T(v_b))`, the same curve `T` on each LogC4 channel; `c_u` and `c_p` are the chromas of `u` and `p` (`c_u = c_v`, as `e₁` and `e₂` sum to 0).
2. `f = 1 + β·(|c_p|/|c_u| − 1)` with `β = CONTRAST_COLOR = 0.5`, and `f = 1` where `|c_u| < 10⁻⁹` (`CONTRAST_NEUTRAL_CHROMA`): neutrals stay neutral.
3. `v′ = u + (f − 1)·(c_u1·e₁ + c_u2·e₂)`: `u`'s chroma scaled by `f`, its direction (the hue in the plane) kept.

The mean of the three channels (the position on the neutral axis) is `u`'s, since `e₁` and `e₂` sum to 0. `T`'s slope lies between `min(k, 1/k)` and `max(k, 1/k)`, so by the mean value theorem on each channel difference `|c_p|/|c_u|` does too, and `f` lies between `1 − β·(1 − min(k, 1/k))` and `1 + β·(max(k, 1/k) − 1)`: 0.83 to 1.26 at the ends of the range. The implementation limits the ratio to that interval, which only guards rounding. `β = 0` is the hue-keeping offset alone, `β = 1` gives `u` the chroma length of the per-channel curve. In plain words: more contrast makes the midtones more colourful, half as much as a curve on each channel would, and takes some colour from the deepest shadows and the brightest tones, where the curve flattens; less contrast does the opposite; black stays black and the hue stays.

**Measured** (three local A1 photos, `ym render --cpu`, JPEG at 1200 px, mean CIELAB chroma change): Contrast +60 changes it by +5.2, −0.8 and +1.9 % (+30: +2.6, −0.1, +1.0 %; −30: −2.6, −0.3, −0.7 %; −60: −5.3, −1.0, −1.3 %). The darkest percent stays where the offset alone puts it (L* 0.3, 8.1, 0.4 at +60).

### Saturation

Saturation keeps the luminance. The chroma `rgb − Y` is scaled about the luminance `Y`; as the AWG4 `Y` row sums to 1, `Y` does not move. In plain words: more saturation makes colours stronger without making them darker or lighter, less makes them greyer, and −100 is the grey of the same luminance.

- Below 0 every colour is blended evenly toward that grey.
- Above 0 the scale depends on how close the darkest channel already is to 0. `r` is 0 for a neutral and 1 when the darkest channel is at 0 (scaling the chroma by `1/r` would take it there), so a muted colour has its chroma scaled by nearly `F`, a strong one by nearly 1, and the darkest channel ends at `Y·(1 − r′) ≥ 0`: no channel is pushed below 0.
- A pixel with `Y ≤ 0` or a channel at or below 0 is left as it is for `s > 0`. `s = 0` skips the row. The hue (the direction of `rgb − Y`) is kept at every setting.
- The ASC CDL saturation inside looks (C10, Rec.709 weights) is a standard and is separate from this row.

`α = SATURATION_BOOST = 1.5` is chosen so that Saturation +60 raises the mean CIELAB chroma of the three local A1 photos by +72.7, +75.1 and +73.0 % (mean +73.6 %; `ym render --cpu`, JPEG at 1200 px, CIELAB of the JPEG, sRGB D65). The gain is about even across photos: strong colours gain less, muted ones more.

### Implementation

`ym_color::grade::{GradeParams, Grade}`; `Grade::apply(LogC4, L)` and `Grade::apply_linear(LinearAwg4)` (C8 then C9, `L` from the linear values). The contrast and saturation functions are `contrast_slope`, `contrast_curve`, `contrast_pixel`, `saturation_factor`, `saturation_chroma_scale` and `saturate`, with the constants `CONTRAST_EXPONENT`, `CONTRAST_BLACK`, `CONTRAST_TOP`, `CONTRAST_COLOR` (β, 0.5), `CONTRAST_NEUTRAL_CHROMA` (1e-9), `SATURATION_BOOST` (α, 1.5), `SATURATION_SMALL_R` and `SATURATION_CODE_LIMIT`. `contrast_pixel(v, L, k, β)` takes `β` as an argument so the tests can check `β = 0` and `β = 1`; the grade calls it with `CONTRAST_COLOR`. The chroma `c_u` is taken from `v` (equal to `u`'s, as `e₁` and `e₂` sum to 0), so in f32 it carries no rounding of the offset. Where the table is silent:

- **Stops are scene-referred.** "Grey ±n stops" is the LogC4 code of `0.18·2^±n` (`grade::logc4_at_stops`), not `g ± n·S`: LogC4 is linear in stops only far above the toe, and `g − 6S` would sit below the code value of black. `S` is used for widths and amplitudes only.
- **Zones.** `zone(L) = bump(L; centre, 1.5 S)` at −3, 0, +3 stops (Gaussian bumps, not a partition of unity: colour balance leaves the extremes alone). Shadows, Highlights, Black and White add `x/100 · 1.5 S · bump(L; centre, 1.5 S)` to all three channels; positive is brighter for all four. Contrast and Saturation are skipped when 0 (the grade is then bit-exactly the identity for that row).
- **Ranges.** Controls clamp to −100…100; a colour-balance vector longer than 1 scales back to the unit circle.
- **Monotonic**, per pixel in the control value: Brightness, Shadows, Highlights, Black and White raise every channel (non-decreasing). For a luma between black and the top code value, Contrast moves the mean of the three channels, and every channel of a neutral, away from `g` (up above, down below; `h_k(t)` grows with `k` for `0 < t < 1`), and beyond the ends toward the end (the line of slope `1/k`); the colour step scales a coloured pixel's chroma by `f` without moving that mean, so a single channel of a coloured pixel is not promised to follow it. Saturation scales `|rgb − Y|` by a factor that never falls as `s` rises (`Y` and neutral pixels are unchanged; at `s = −100` every channel equals `Y`). Colour balance displaces along a fixed direction by `0.03·r·zone`, so the length grows with the radius. As a tone map on a neutral ramp, one control over its full range keeps the slope above 0.39 (brightness ≥ 0.757, zone controls ≥ 0.393, contrast between `2^−0.6 = 0.66` and `2^0.6 = 1.52`, analytically).
- **Not guaranteed:** several tonal controls stacked at ±100 can fold the ramp (all four zone controls and Brightness at −100 with Contrast +100: worst slope −0.15 inside the valid code range, −0.68 below black; at +100: −0.10 with Contrast −100).
- **Saturation where the table is silent.** `Y` is taken from the pixel after Contrast, Brightness, the zone controls and Colour balance (decoded to linear AWG4), so Saturation never changes the luminance those controls set; `L` is not used by this row. Below `r = 1e-9` (`SATURATION_SMALL_R`) `λ` is its limit `F`. That covers `r < 0`: the blue weight of the AWG4 `Y` row is negative, so a blue or violet whose red and green are nearly equal can have every channel above `Y` (linear `(0.02, 0.02, 0.57)` has `Y = 0.0002`). No channel can then reach 0 as the chroma grows, and the formula `r′/r` would exceed `F` without bound as `Y` goes to 0, so such pixels get the plain factor `F`. `r′` is computed as `−expm1(F·ln(1 − r))`, the same number without the cancellation of `1 − (1 − r)^F` at small `r`. A pixel with a code above 7.5 (`SATURATION_CODE_LIMIT`, linear about 7e32; real pixels after C9 stay below 3) is passed on unchanged, so the f32 kernel's decode stays finite.
- **Nothing in C9 clamps.** Saturation below 0 can take a pixel whose `Y` is negative to a negative grey, and Colour balance and the zone controls can take a channel below the LogC4 code of black; C8's inverse, C10 and the display transform take it as it is.

**GPU kernel** (`color.wgsl`: `contrast_curve`, `c9_contrast`, `c9_saturation`, called from `c9_grade`). `k` comes from `grade0.x` and the saturation factor from `grade0.z` (below 1 the blend, above 1 the boost `F`); the constants are generated from Rust (`docs/17-gpu.md` K7). Under fast-math (`docs/17-gpu.md` K8), `c9_contrast`:

- returns `u` for an exact neutral (equal channels) before anything else, because with FMA a fused dot product of three equal codes can round to a few 1e-9 instead of 0;
- squares `c_u` for the neutral test only where both components are below 1e-9, so it cannot overflow;
- divides both chromas by `m = max(|c_u1|, |c_u2|)` (at least 7e-10 past the test) before squaring, so a garbage pixel (a sanitised 1e30 gives codes near 1e32) cannot overflow f32;
- limits the squared ratio to `[min(k, 1/k)², max(k, 1/k)²]` before the square root, which then never sees 0, even where f32 rounds the per-channel curve of a chroma of a few ulp to a neutral;
- keeps every divisor of the contrast curve at least `2^−0.6`.

In f32 the saturation kernel takes `λ` from its series `F·(1 − (F−1)/2·r·(1 − (F−2)/3·r·(1 − (F−3)/4·r)))` below `r = 0.05`, where `1 − q^F` would cancel (`q = min/Y`, kept at least 1e-30 for the `pow`; for `F` up to 2.5 the first neglected term, `F(F−1)(F−2)(F−3)(F−4)/120·r⁴`, is below 1e-7 of `λ`).

**Measured (fast-math).** A C port of `c9_contrast` built with clang `-O3 -ffast-math -ffp-contract=fast -mfma -mrecip=all` against `-O2 -ffp-contract=off`, over 200,000 pixels (exact neutrals, chromas of one ulp and of 1e-13, codes to ±20, garbage codes to ±1.4e32 with lumas to match, `k` over the whole range), gives finite outputs everywhere and leaves exact neutrals neutral in both builds. The two builds are at most 3.1e-7 apart among photo codes and 3.1e-6 for codes to ±20, and agree with an f64 reference to 2.4e-7 among photo codes.

**Tests.** `crates/ym-color/tests/grade.rs` checks the curve (fixed points, slopes `k` and `1/k`, increasing and C¹ for random `k` in `[2^−0.6, 2^0.6]` and random `x` below black and above 1), the colour step (keeps the hue and the mean of the channels, scales the chroma by the `f` of the formula within its bounds; `β = 0` is the offset alone and `β = 1` the per-channel chroma length; neutrals and chromas below 1e-9 get the offset alone; midtones gain colour for `c > 0` and lose it for `c < 0`, deep shadows the reverse), Saturation (keeps `Y` and the hue, never takes a channel below 0 for `s > 0`, is monotone in `s`, leaves neutrals alone, gives the grey of `Y` at −100, scales muted colours more than strong ones) and that `c = 0` and `s = 0` are byte-identical to the grade without those rows. `crates/ym-gpu/tests/gpu_color_monotone.rs` checks the same properties on the kernel's output (the colour step's `f` against the formula to 8.4e-7, exact neutrals exactly neutral, near neutrals of one to a few f32 ulps within 1.7e-7 of the CPU) and a parity test on pixels made for every branch: every pair of Contrast and Saturation in {−100, −60, −30, 0, 30, 60, 100}. `crates/ym-gpu/tests/gpu_color.rs` runs the `contrast/` and `saturation/` chains at ±60 and ±100, alone and on the full chain (C9 8.4e-7), and 60 random sets (C9 4.3e-7).

**Measured (GPU parity, Mesa lavapipe, all three storage modes).** Worst `|GPU − CPU|` at C9 is 4.5e-5 (at a code of −72: an out-of-gamut blue of negative luminance at Contrast 100 and Saturation −100) and 9.4e-6 among codes from −0.2 to 1.5, against the 2e-4 gate.

## C9b · Colour warp

A chroma warp in the manner of DaVinci Resolve's Color Warper, on the Colour tab. Pins move chosen colours across a chroma plane; colours near a pin follow it smoothly, locked pins hold their colours, and neutrals never move.

**The chroma plane.** For a LogC4 pixel `v`, its chroma is `c = (e1·v, e2·v)` with the colour-balance axes of C9 (`e1 = (1, −½, −½)/√1.5`, red against cyan; `e2 = (0, 1, −1)/√2`, green against blue), and its tone is `L = encode(Y(decode(v)))`, the LogC4 code of its linear luminance (as C11's luma curve). Moving `c` by `d` adds `d1·e1 + d2·e2` to `v`, which leaves the mean of the three channels, and so the pixel's brightness along the neutral axis, unchanged. Pure red sits at 0°, green near 120° and blue near 240° (counter-clockwise from `+e1`). The UI draws the plane out to `|c| = 0.30` (`WARP_PLANE_EXTENT`), which holds the colours of ordinary photos: a red of linear (0.18, 0.009, 0.009) in AWG4 reads `|c| = 0.13`.

**A pin** has a source `s` and a target `t` (points of the plane), a range `ρ`, two tonal weights (Shadows `w_lo`, Highlights `w_hi`, 0…1), a pivot `p` in stops from grey, an exposure `x` in stops, and a lock flag. A locked pin's target is its source and it has no exposure or tonal weights.

For a pixel of chroma `c₀` and tone `L` the warp is a flow, not one jump: in `N` small steps every moving pin's Gaussian slides from its source toward its target and the pixel follows it, so a pin reaches as far as its range however long its move.

```
δ_i  = t_i − s_i                                   moving pins
G(d², ρ) = exp(−min(d² / (2ρ²), 80))               every Gaussian, its exponent limited to 80
w_i  = w_lo + (w_hi − w_lo) · smoothstep(P_i − 2S, P_i + 2S, L),   P_i = logc4_at_stops(p_i)
λ    = Π_k (1 − G(|c₀ − l_k|², ρ_k))               locked pins l_k hold their colours
g    = smoothstep(0, 0.02, |c₀|)                   neutrals never move
γ    = g · λ                                       the pixel's own weight, at its original chroma
N    = clamp(⌈1.25 · Σ_i |δ_i| / (ρ_i √e)⌉, 1, 64)  steps, one number for the frame, from the pins
c    = c₀, then for k = 0 … N − 1, τ = k / N:
  m_i  = s_i + τ · γ · w_i · δ_i                   each centre slides by the share of its move this pixel takes
  φ_i  = G(|c − m_i|², ρ_i)                        each moving pin's reach, with the range as set
  c    = c + (γ / N) · Σ (w_i φ_i δ_i) / max(1, Σ φ_i)   overlapping pins share, they do not add up
D    = c − c₀                                      chroma displacement
E    = γ · Σ (w_i G(|c₀ − s_i|², ρ_i) x_i) / max(1, Σ G(|c₀ − s_i|², ρ_i))   exposure in stops, in one step
v'   = v + D1·e1 + D2·e2 + E · S                   S = one stop in LogC4 (0.06480)
```

- **Range is the range.** A pin's range is used as set: a colour three ranges beside or behind the source moves by at most `exp(−4.5)` of the move (exactly that for a move of one step).
- **A single pin reaches its target exactly** (where its tonal weight is 1, away from locks and from neutral): the pixel on the source sits on its pin's centre at every step (`φ = 1`, `n = 1`) and moves with it, to rounding. A pixel that takes a share `γ·w < 1` of the move sees the centre slide by that share and lands on `s + γ·w·δ`, as in one step: Shadows at 40 % moves the source's shadows by 40 % of a long move. Pins that overlap share their reach, so two pins on one colour take it to their target rather than twice as far.
- **No folding for one pin.** At one tone, away from neutral and locks (`γ = 1`), every pixel follows the same flow, that of one pin moving `w·δ`. Its field `w·δ·φ(c)` has the Lipschitz constant `w|δ| / (ρ√e)` (a Gaussian is steepest one range from its centre), an Euler step `c ↦ c + V(c)/N` is one to one with a positive Jacobian determinant when `Lip(V)/N < 1`, and a chain of one-to-one steps is one to one. The rule for `N` keeps `Lip(V)/N ≤ 0.8` for each pin. The cap of 64 never breaks that for one pin: sources and targets lie within 0.5 of neutral, so `|δ| ≤ 1`, and `ρ ≥ 0.01` give at most `1/(0.01√e) = 60.7`. It can bind for many long, narrow moves at once (their bounds add up), and then the plane can fold; so can locks and the neutral guard (`γ` changes across the plane) in odd arrangements. The plane shows where every pin's colour lands.
- **What a long, narrow move does to the colours on its way.** A one-to-one map that takes `s` to `t` must move what lies between them: the colours ahead of the source on the line of the move, within about a range of it, are pushed ahead of the moving centre and arrive packed just past the target, in their order; the colours just behind the source, in a sliver about `2ρ²/|δ|` long, are stretched along the path. Colours more than two ranges from the line barely move. The stretch is steep (up to about `|δ|² / (2ρ²)`, 5,000 for a move of 1 at range 0.01), so in that sliver the flow magnifies rounding (see the f32 numbers below).
- **Steps.** Typical moves take few steps: a move of 0.1 at range 0.04 takes 2 (at the default range 0.02, 4), four such pins 8, a move of 0.2 at `ρ = 0.01` (Range 1) takes 16, and the longest move (1) at `ρ = 0.01` takes the cap (64). A move of up to 1.32 ranges takes one step, which is the one-step formula `D = γ·Σ(w φ δ)/n` with `φ` at the sources. The cost is `N` Gaussians a moving pin a pixel (an `exp` and about twenty other operations, the tonal weight recomputed each step) on top of the one-step exposure: 2 for one default pin and a short drag, 64 × 16 = 1024 at the extreme.
- **Exactness.** No pins, or only locked pins, skip the stage (renders are byte-identical to a pipeline without it). A locked pin's colour does not move at all (`λ = 0` there), at every tone.
- **Ranges.** At most 16 pins (`WARP_MAX_PINS`). Sources and targets are scaled back to `|c| ≤ 0.5`; `ρ` clamps to 0.01…0.25 (default 0.02, shown as 2 on the Range slider); the tonal weights to 0…1 (default 1); the pivot to −6…+6 stops (default 0); the exposure to −2…+2 stops (default 0). Defaults leave `w_i = 1`. The constants are `WARP_STEP_SAFETY` (1.25), `WARP_MAX_STEPS` (64) and `WARP_EXP_LIMIT` (80) in `ym_color::warp`, mirrored in `src/components/warpMath.ts` (a test reads the Rust file); the exponent limit applies on the CPU as in the kernel.
- **Where it runs.** After C9 and before C10, so a look and the curve act on the warped colours. It travels into the DNG with the rest of the LogC4 grade. Before leaves it out (Before is the camera default; its "after the grade" reading point is C9, before the warp).

### GPU kernel

`ym_gpu::kernels::color` and `color.wgsl`, function `c9b_warp`, called between `c9_grade` and `c10_look` in `run_pixel`. The debug stop is `Stop::C9b` (code 10 of `mode.x`), read against `StageTrace::c9b_logc4`.

- **Where the pins live.** In the one uniform block `ChainU` (`docs/17-gpu.md` K3 allows one), after the ACES 2 scalars: `warp_n` (`x` the moving pins, `y` the locks, `z` the flow's steps), `warp_mv` (`array<vec4<f32>, 48>`, three rows a moving pin: `[s.x, s.y, δ.x, δ.y]`, `[1/(2ρ²), w_lo, w_hi, P]`, `[x, 0, 0, 0]`) and `warp_lk` (`array<vec4<f32>, 16>`, one row a lock: `[l.x, l.y, 1/(2ρ²), 0]`). The block is 2096 bytes.
- **Preparation.** `ChainTables::prepare` packs what `ym_color::warp::Warp::new` prepares (clamping, the pivot code, the number of steps from the pins in f64; still pins and lock-only sets are left out), so nothing is computed twice and every pixel of a frame takes the same steps. The sizes (`WARP_MAX_PINS`, `WARP_MOVE_ROWS`, `WARP_LOCK_ROWS`, `WARP_MAX_STEPS`) and `WARP_NEUTRAL_GUARD`, `WARP_TONE_HALF_WIDTH`, `WARP_EXP_LIMIT` are generated into the WGSL from Rust (K7). The kernel takes the steps as `clamp(warp_n.z, 1, WARP_MAX_STEPS)`, so a stray uniform can neither divide by zero nor loop long.
- **Why a uniform and not a storage buffer.** A pin moves on every frame of a drag, so a persistent table would be refilled as often as the uniform is written, and it would cost a binding (7 of 8), a cache key and an upload of its own. The extra 1 KB a dispatch is nothing next to a tile.
- **Identity.** No moving pin gives `warp_n.x = 0` and the kernel returns `v` before anything else, so a chain without a warp renders the same bits as one without the stage (tested for no pins, locks only, still pins, and a moving pin past the sixteenth, on every output).
- **Fast-math (K8).** Nothing rests on an infinity or a NaN. Every Gaussian's exponent is limited to 80 (`exp(−80)` is a normal f32). A pixel with `|c| < 1e-12` returns before the square root is taken (fast-math may compute it as `x·rsqrt(x)`, NaN at 0; the CPU moves such a pixel by less than 1e-20). The lock product has factors in [0, 1]. Every division is by `max(1, Σφ) ≥ 1` or by the step count (at least 1). The tone `L` decodes `min(v, 7.5)` so the decode's `exp2` stays finite in f32 (real pixels after C9 stay below 3; above about 0.8 the tonal weight is already the highlights').

**Measured (fast-math).** A C port of `c9b_warp` (`float` built with clang `-O3 -ffast-math -ffp-contract=fast -mfma -mrecip=all` and with `-O2 -ffp-contract=off`, and a `double` build; 200 random pin sets of 0 to 16 pins, 80,000 pixels among them exact neutrals, `|c| = 1e-13`, chroma of 1e30, codes up to 40 and down to −3e31, pixels on a lock, on a source, and a quarter of them in the sliver behind a source) gives every output finite and exact neutrals unchanged in both float builds. With random pins (ranges 0.01 to 0.25, moves up to 0.3, up to 64 steps) the two float builds are at most 3.1e-6 apart (5.3e-6 in the sliver) and float against double 6.3e-6. With every pin long and narrow (ranges 0.01 to 0.02, targets anywhere on the plane) they are 6.6e-5 apart away from the sliver and 2.7e-4 in it, and float against double 3.3e-4 in it: over the 2e-4 gate for the longest moves at the smallest ranges.

**Measured (GPU parity, Mesa lavapipe, `tests/gpu_color.rs`, all three storage modes).**

| Case | C9b | C10 | C11 | C13 |
|---|---|---|---|---|
| Four fixed warp chains (warp alone, after the full grade, before a display LUT, before a display-baked look; 5 moving pins with split tonal weights, pivots, exposure, a move of 0.12 at range 0.01, 12 steps, plus a lock and a still pin) | 4.2e-7 | 4.5e-7 | 4.9e-7 | at most 4.2e-5 (gate one 10-bit code) |
| 60 seeded random parameter sets with 0 to 18 random pins (about a third locked, controls drawn beyond their ranges) | 6.6e-7 | 6.4e-7 | 6.2e-7 | 2.0e-5 |
| The most pins the uniform holds (16 moving; 1 moving and 15 locks; 8 and 8; 16 moving followed by ignored locks) | 9.1e-7 | 8.7e-7 | 8.2e-7 | 4.4e-6 |

The gate is 2e-4 for C9b to C11. In the last case every moving pin moves its own pixel (the least by 6.1e-3: overlapping pins share) and every lock holds its own. The long-move test (`a_long_narrow_move_matches_the_cpu_where_the_flow_stretches`: a cloud of 12,288 pixels about 1e-3 across around one colour, moved 0.48 across neutral at range 0.01, 37 steps, 34 % of the pixels stretched) gives C9b 7.2e-5. The same cloud moved 0.68 gives 9.3e-5 and moved 0.87 gives 2.0e-4, at the gate: **the gate holds for moves up to about 0.85 at range 0.01 and fails beyond** (the difference is about 0.003 of a stop, in colours the warp stretches across the whole path).

The export tail runs the same kernel through `PreparedChain` (`tests/gpu_tail.rs`): a warped, shrunk and sharpened 8-bit export is at most 1 code from the CPU's, 0.9999 exact, and 56 % of its codes differ from the export without the warp.

### Tests (CPU reference, `ym_color::warp`)

The tests check: the identity cases bit for bit; that a single pin's source lands on its target to 1e-12 for moves up to 1.0 at range 0.01 and through neutral; that Range limits the reach (a colour three ranges beside or behind the source moves by at most `exp(−4.5)` of the move, for moves of 0.1 to 0.46 at ranges 0.01, 0.04 and 0.12); that one pin never folds (5 extreme and 40 seeded random pins, `|δ|` up to 1, ranges 0.01 to 0.25, tonal weights 0.2 to 1: on a grid of a quarter range along and across the move, six ranges around it, every mapped cell keeps its orientation and the Jacobian determinant stays positive, the smallest 7.1e-5, while the same move in one step folds and the test sees it); that a partial weight or a lock's hold moves the source by that share of a long move; that a short move is the one-step formula bit for bit; and the step count, locks, neutrals, tonal weights, exposure, overlapping pins and clamping. `src/components/warpMath.test.ts` checks the TypeScript port the same way (the plane and the mock draw what the render does).

## C10 · Looks

A look is `CDL → LUT`, applied in its own encoding, then blended:

```
in  = v (LogC4)
out = look(in)
v'  = in + k · (out − in)        k = intensity / 100, blended in LogC4
```

- **CDL** (ASC v1.2): `x = max(x·slope + offset, 0)^power` per channel, then saturation with Rec.709 weights (0.2126, 0.7152, 0.0722): `L = w·x; x = L + sat·(x − L)`.
- **3D LUT:** `.cube` (Resolve/Adobe format: `LUT_3D_SIZE`, optional `DOMAIN_MIN/MAX`, optional `LUT_1D_SIZE` shaper, red varies fastest). Tetrahedral interpolation. Inputs are clamped to the domain. The LUT is uploaded as an **f32 storage buffer** (one `vec4<f32>` per node, red varying fastest) and interpolated by hand in the shader (tetrahedral), never with hardware trilinear. It is not a texture: a LUT texture stored as rgba16float rounds values in [0.5, 1) by up to 2.4e-4, which alone is more than the 2e-4 parity target between the GPU kernel and the CPU reference. A 65-point cube costs 4.4 MB and a 129-point one 34 MB as f32, which is acceptable for one look at a time. Only image intermediates are f16 (`ym_gpu::kernels::color`, section "Precision decision").
- **LogC4 log-to-log looks** (ARRI LogC4 Look Library, our built-ins): applied directly.
- **LogC3 log-to-log looks:** LogC4 → linear AWG4 → AWG3 → LogC3 EI 800 → LUT → linear AWG3 → AWG4 → LogC4. The blend happens in LogC4 after converting back.
- **LogC3-to-Rec.709 looks** (look and rendering baked together) **replace** C12: the chain is C9 → C9b → C11 (the curve runs before the look in this case) → LogC4 to LogC3 conversion → LUT → display Rec.709 (gamma 2.4). Intensity blends between our C12 output (from the same C11 output) and the LUT output in display-linear Rec.709. The UI marks these looks "includes display rendering".
  - A DNG of such a look is approximate: ARRI's display rendering has no scene-linear form, so the DNG's pixels are the scene values that ACES 2 (with the display placement, C12) turns into the look's display picture at the chosen intensity. They are found with the inverse output transform (`Aces2Transform::inverse_ap0_linear`; display white and out-of-gamut values clamp, scene values are limited to the range of the DNG profile), and Adobe's renderer, running the profile fitted to ACES 2, shows approximately that picture (`docs/09-export.md`, "What a DNG carries"). The same holds for an imported ARRI display LUT chosen as C12.
- **Built-in looks:** Neutral is the identity. Signature (the default look) is parametric (a luma tone curve and CDL saturation in LogC4, optionally hue/saturation curves) and stored as JSON in `data/looks/builtin/`; it is baked into a 65³ LUT at load, so C10 treats it like any LogC4 LUT (`docs/08-looks.md`, Built-in looks).

### Implementation

`ym_color::{cdl, cube, lut, look}`.

- **CDL.** `x = max(in·slope + offset, 0)^power`, then `L = 0.2126R + 0.7152G + 0.0722B; out = L + sat·(x − L)`. No clamp at 1 (the CDL runs on log codes and look output); OpenColorIO's `CDL_ASC` also clamps at 1 before and after saturation and agrees wherever every intermediate stays in [0, 1]. `Cdl::new` checks the ASC ranges (slope ≥ 0, power > 0, saturation ≥ 0).
- **.cube dialects.** `cube::parse` reads the Adobe/IRIDAS dialect (`TITLE`, `LUT_3D_SIZE` or `LUT_1D_SIZE`, `DOMAIN_MIN/MAX`) and the Resolve dialect (`LUT_1D_INPUT_RANGE`, `LUT_3D_INPUT_RANGE`, a 1D table followed by a 3D table in one file, the 1D table being a shaper). Sizes: 3D 2 to 129, 1D 2 to 65536; at most 256 MiB. Comments, blank lines, LF, CRLF, CR, a BOM and case-insensitive keywords are accepted; every number must be finite; every failure is a typed `CubeError` with the line number. Domains: for a lone table, `DOMAIN_*` or the matching `LUT_*_INPUT_RANGE` (they must agree if both are given); with both tables `DOMAIN_*` and `LUT_1D_INPUT_RANGE` belong to the shaper and `LUT_3D_INPUT_RANGE` to the cube. A 1D-only file parses but `parse_lut` refuses it (a look needs a 3D LUT). OpenColorIO reads the same files the same way (`iridas_cube`, `resolve_cube`).
- **Tetrahedral.** `Lut3d::sample` uses the six-tetrahedra split of the fractional position (the same as OpenColorIO's), clamps to the domain (a NaN input counts as the domain minimum), is exact at grid nodes and continuous across cell faces.
- **Looks.** `look::LookTransform::{neutral, new, eval_logc4, apply_logc4(v, intensity_pct), eval_display_rec709}`, `blend_logc4`, `blend_display_linear(own, look, intensity_pct)` (both in the same linear primaries; the look output is converted with `LinearRec709::to_display_p3` / `to_rec2020` first), `rec709_gamma24_decode` (negative LUT output clamps to 0). 0 % returns the input and 100 % the look output exactly. CDL and LUT run in the look's own encoding (a LogC3 look applies its CDL to LogC3 codes). Inputs above the LUT domain clamp (LogC3 code value 1.0 is linear 55, so LogC4 highlights above that hold the LUT's top value in a LogC3 look).
- **Parametric looks.** `look::ParametricLook::from_json` reads the schema documented at the top of `crates/ym-color/src/look.rs` (`yellowmoon.parametric-look/1`: `cdl`, `hue_vs_sat`, `hue_vs_hue`, `luma_vs_sat`, all optional, evaluated in LogC4, chroma and hue measured along the C9 axes `e1`, `e2`, Rec.709 luma preserved). `bake(65)` samples it on a 65³ cube.

**Measured.** Tetrahedral sampling against OpenColorIO 2.6 tetrahedral on our own synthetic cubes (`tools/vectors/lut_vectors.py`: identity 9, warp 17, ripple 33, an asymmetric-domain 9, a 2-point extreme cube, a 64-entry shaper plus 9-point cube; 3714 points each, including grid nodes, cell diagonals and out-of-domain inputs): the largest difference is 2.1e-7, or 0.0002 code values at 10 bits (OpenColorIO's float32). Trilinear would differ by 1.75e-3 (1.8 code values) on the warp cube, so the gate pins the method. The example parametric look in `tests/data/looks/` bakes with a maximum error of 3.2e-3 LogC4 code values at 65 points (3.7e-3 over the colours where all channels are above black).

## C11 · Curve

Applied in LogC4 after the look and before the display transform, so it travels into DNG export and behaves the same under every display transform. Four monotone cubic splines (Fritsch–Carlson) on [0,1]: luma (applied as `v += curve(L) − L`) and R, G, B (per channel). They are baked into four 1024-entry tables, uploaded as **f32 storage buffers** and looked up by hand with linear interpolation in the shader, not as 1D textures (an f16 texture would round the values in [0.5, 1) by up to 2.4e-4, more than the 2e-4 parity target). The histogram in the Curve tab is of LogC4 values. Numbers in the UI show code values × 255, rounded.

`ym_color::curve::{Curve, CurveSet, BakedCurves}`:

- Fritsch–Carlson tangents: mean of the neighbouring secants, zero at extrema and flat segments, one-sided at the ends, then the `α² + β² ≤ 9` limiter. 2 to 16 control points, `x` strictly increasing, all in [0, 1]; invalid input is a typed `CurveError`.
- The spline is C1 and never overshoots the data of a segment. Between 0 and the first point and between the last point and 1 it holds the end value.
- The 1024-entry table (`CurveTable`, sampled at `i/1023`, linear interpolation; worst deviation from the exact spline 5.9e-7) continues with slope 1 outside [0, 1] (`eval_extended`: `x + table[0]` below 0, `x − 1 + table[1023]` above 1), so the map stays monotone for LogC4 values beyond the unit range. The shader's `curve_eval` applies the same extension explicitly.
- The luma curve uses `L = encode(Y(decode(v)))` of the pixel entering C11 and is applied before the R, G, B curves.

## C12 · Display transform

### ACES 2 (default)

The ACES 2.0 output transform (aces-output, Apache-2.0), ported to Rust (CPU reference) and WGSL.

1. LogC4 → linear AWG4 → AP0 (AWG4 → XYZ, CAT02 D65 → ACES white, XYZ → AP0; the matrix is computed in `ym-color`).

1a. **Placement.** The AP0 triple is multiplied by `2^k` with `k = DISPLAY_PLACEMENT_EV = +0.85` (`ym_color::display`; a constant of the look, not a control, and not a measurement). `aces2_input_matrix()` is `2^k · awg4_to_ap0`: the CPU chain uses it and the GPU kernel's `AWG4_TO_AP0` constant is written from it, so both apply the same numbers. The placement sits after the look and the curve, so a look still sees mid grey at LogC4 0.2784, and before the ACES 2 transform. Through the Neutral look it puts scene 0.18 at display-linear 0.1893 (OpenColorIO's ACES 2.0 SDR value for `0.18 · 2^0.85` is 0.189253; sRGB code 120.4, 50.0 % video at gamma 2.4) instead of 0.1000 (code 89, 38.3 %): +0.92 EV of display light for +0.85 EV of scene light, about where a camera JPEG puts mid grey (ISO 12232: 0.18, code 118; ours is 2.4 codes higher).

It applies to the ACES 2 path only. An imported ARRI display LUT and the LogC3-to-Rec.709 looks keep ARRI's own brightness: with such a look the intensity blend (C10) runs between our C12 (placed) and the look's LUT (not placed), so its Intensity slider also changes the brightness, by up to 0.85 EV. A DNG carries the placement in `BaselineExposure` (`docs/09-export.md`).

Why: ACES 2 SDR is a cinema rendering, which puts scene 0.18 at 10 nits, 0.85 EV under a camera JPEG. Yellowmoon is a photo processor, not a video tool, so its pictures are brighter than the reference rendering.

1b. **White lift** (`white_offset_ev`, the Automatic card's White row). A per-photo scene-linear gain `2^(A · smoothstep((s − 1) / span))` on the luma of the decoded C11 output (`s` the stops over middle grey), applied to linear AWG4 **before the placement** and ACES 2: exactly 1 up to +1 stop over mid grey (mid grey and everything below it are untouched), ratio-preserving, per channel monotone, ACES 2 path only (an imported ARRI display LUT and a LogC3-to-Rec.709 look keep ARRI's highlights). `A` and the anchor are sized per photo from the 1/8 preview when it is opened; the DNG carries it in its pixels. The full text is in *White lift (C12 step 1b)* below.

1c. **Red chroma** (a per-look strength, `red_chroma` in the look file; owner decision of 2026-10-24). A smooth, hue-preserving chroma gain on the reds, applied to linear AWG4 **after the white lift and before the placement** and ACES 2: it gives back the chroma ACES 2 takes out of mid-tone and light reds, is capped below the output gamut's edge, and leaves grey, other hues and skin alone. ACES 2 path only. A DNG carries it in its pixels, like the white lift. The full text is in *Red chroma (C12 step 1c)* below.

2. ACES 2 output transform with peak luminance 100 nits, SDR, white D65, and these limiting primaries:

| Output | Limiting primaries | Encoding |
|---|---|---|
| Viewport | P3-D65 | Display P3 (sRGB transfer) |
| sRGB export | Rec.709 | sRGB transfer |
| Display P3 export | P3-D65 | sRGB transfer |
| Rec.2020 export | P3-D65 | Rec.2020 primaries, BT.1886 (gamma 2.4) |

3. Gamut compression is part of ACES 2; out-of-gamut values from C8 are handled there.

**Measured (placement).** On the 12 local files (2000 px sRGB renders, largest channel, at the camera gain with the placement) pixels at or above code 220 are 0.16 % on average (at most 1.16 %, A7 II; A1M01739 0.44 %, A1M01749 0.28 %, all others 0.05 % or less) and pixels at or above code 250 are 0.000 % on all 12. The median luma code (8-bit, 0.2126 R′ + 0.7152 G′ + 0.0722 B′) runs from 91 to 177. On the three A1 photos the median luma code is 143, 177 and 127 against 137, 195 and 112 in the camera JPEGs (A1M01749, A1M01739, A1M01454), and the 99th percentile luma code is 171, 217 and 195 against 182, 242 and 215 in the JPEGs, 49 % and 33 % of whose pixels on two of the three scenes are at or above code 220, so there is room.

**Test vectors.** `tools/vectors/aces2_vectors.py` writes `crates/ym-color/tests/data/aces2/{viewport,srgb,display_p3,rec2020}.json` with OpenColorIO 2.6.0's float32 output (898 AP0 inputs per output: neutrals from −8 to +12 stops, gamut primaries and secondaries at five exposures, a chromaticity wheel, AP0 primaries, edge cases, seeded random in-gamut, out-of-gamut, near-black and very bright values). Each output is the built-in `ACES-OUTPUT - ACES2065-1_to_CIE-XYZ-D65 - SDR-100nit-<REC709 or P3-D65>_2.0` followed by a `DISPLAY - CIE-XYZ-D65_to_<sRGB, DisplayP3 or REC.1886-REC.2020>` built-in; the script checks that the studio config's sRGB and Display P3 display/view pairs equal the built-in chain exactly. The config has no Rec.1886 Rec.2020 display, so the Rec.2020 vectors use the P3-limited built-in plus the Rec.1886 Rec.2020 display built-in, which is how the config builds its own pairs. Gate: at most 1 code value at 10 bits. The vectors test the transform itself and do not include the placement (step 1a): the chain tests build the AP0 triple with `aces2_input_matrix()`, and the chain-level OpenColorIO vectors (`display_ocio_vectors.rs`, `gpu_color_vectors.rs`) feed a fused matrix of `2^-k` so that the placement cancels.

#### CPU reference

`ym_color::aces2` (`Aces2Output::{Viewport, Srgb, DisplayP3, Rec2020}`, `Aces2Transform::new(output)`, `transform_ap0`, `transform_ap0_linear`; attribution and commit hashes at the top of the module).

- **Source.** The transform's maths lives in the ACES project's **aces-core** repository (`lib/Lib.Academy.*.ctl`); aces-output only holds the per-display presets that call it. The port follows the CTL and was cross-checked against OpenColorIO's C++ implementation.
- **Peak luminance.** `Aces2Params::new` takes any peak between `MIN_PEAK_LUMINANCE_NITS` and `MAX_PEAK_LUMINANCE_NITS`; V1 renders 100 nit SDR only. `tests/aces2_hdr_vectors.rs` checks the peak-dependent maths against OpenColorIO (`tests/data/aces2/hdr_p3.json`, P3-D65 limited, five peaks): |ours − expected| / (peak / 100 nit) ≤ 1e-4, worst measured 1.1e-5.
- **Numerics.** f64 with the reference's iteration tolerances unchanged (the CTL and OpenColorIO run in float32). The tables agree with OpenColorIO's float32 tables to 1e-5 relative. **Measured:** at most 0.03 code values at 10 bits over the vectors and 0.05 over random sweeps of 1.9 million samples per output.
- **Reference glitch.** In float32 the reference returns NaN (black) in a thin band of scene luminance near 128 × white with a little chroma, where the tone-scaled lightness reaches the display maximum. The port guards that case and returns white; an f32 WGSL port needs the same guards (see the module notes).
- **Rec.2020 overshoot.** P3 red lies just outside Rec.2020, so saturated cyan converts to a blue up to 1.00121 in linear light (1.0005 encoded, half a 10-bit code value). The reference leaves it unclamped; `transform_ap0` clamps to [0, 1], `transform_ap0_unclamped` keeps the reference value.
- **The inverse** (for the DNG of a look with its own display rendering). `Aces2Transform::inverse_ap0_linear` is the port of the reference's `outputTransform_inv` (`aces2/inverse.rs`: the inverse tone scale, the inverse chroma compression with the inverse toe, and the gamut compression run backwards along the same compression vector, in two passes above the analytical threshold; the geometry is `compress_gamut_with`, shared with the forward transform through the reference's `invert` flag). Display values are cleaned first (not a number is 0, outside `0..=peak` clamps), so the result is always finite; near display white it grows (display 0.99 is scene 58.6, 1.0 is the roof, 128), and the caller limits it. The inverse runs on the CPU only; the device does not take the inverse of a display rendering.
- **Measured (inverse)** (`tests/aces2_inverse.rs`, vectors `tests/data/aces2/srgb_inverse.json` from OpenColorIO 2.6's built-in transform run in the inverse direction, `tools/vectors/aces2_inverse_vectors.py`): in-range vectors (display up to 0.9, 1,016 of the 1,242) agree to a relative 1.3e-4 (the neutral ramp 5.0e-5; display 0.99 gives 58.559 against OpenColorIO's 58.562). Display to scene to display returns the display value to 5.5e-10 over the same 1,016 colours (the CTL's two-pass inverse is exact to f64 rounding here), and scene to display to scene returns 2.2e-14 relative over 193 colours.

### White lift (C12 step 1b)

A per-photo, scene-linear gain on the tones above +1 stop over middle grey, so that the brightest tones of a photo can reach a real white. Mid grey and everything below +1 stop are untouched. It is an *automatic* setting: its amount is sized from the photo's own highlights when the photo is opened, shown in the Automatic card (`White`) and offset with `white_offset_ev` (Light › White › Offset, `ym render --white-offset`).

**Why.** ACES 2 SDR is a cinema rendering with a flat shoulder: 1.4 scene stops above the placed middle grey lift the display by about 0.4 stop, so a photo whose highlights sit two or three stops over grey ends at light grey. On the 12 local photos at +0.85 EV with no lift, the 99.5th percentile of the display luma code is 147 to 218 against 161 to 247 in the cameras' JPEGs; the placement of step 1a closes part of the gap and not the missing white. The C9 Highlights and White sliders are the wrong tool: to reach the same picture they would need 61 to 84 and 31 to 42 on the six photos that need it, and they change 40 to 83 % of the pixels. The clip level is a constant +3.58 stops over grey on all 12 photos (green clips first), so it does not tell one photo from another; what varies from photo to photo is the distribution of the highlights, and that is what sizes the lift.

**The stage** (`ym_color::exposure::white::WhiteLift`; the CPU reference in `display/transform.rs`; the WGSL `white_lift_gain` in `color.wgsl`):

```
Y    = 0.2126 R + 0.7152 G + 0.0722 B           luma of the decoded C11 output, linear AWG4
s    = log2(Y / 0.18)                           stops over middle grey
t    = clamp((s - 1) / span, 0, 1)              span = max(anchor - 1, 0.25)
gain = 2^(A * t^2 (3 - 2t))                     smoothstep
rgb' = gain * rgb                               all three channels
```

`A` is the amount in stops and `anchor` the stops over grey at which the lift is complete: the **two scalars** of a photo (`AutoEstimates::white`). The stage sits after C11 (so a look still sees mid grey at LogC4 0.2784), in front of the placement of step 1a and ACES 2, **on the ACES 2 path only**. Properties, all tested (`crates/ym-color/tests/white_lift.rs`):

- The gain is exactly 1 up to +1 stop, so mid grey lands on the same display value bit for bit, and nothing below +1 stop moves (to far better than 1e-9 in scene-linear).
- It never decreases with the luma and never exceeds `2^A`.
- The three channels are multiplied by the same number, so the colour ratios of a pixel are kept exactly. Every output channel is non-decreasing in each input channel (a product of a channel and a non-negative, non-decreasing function of the luma).
- `A = 0` is the exact identity (`WhiteLift::NONE`, the same bits as no stage).
- The luma has the Rec.709 weights and not the AWG4 `Y` row: the `Y` row has a slightly negative blue weight (−0.036), and a gain driven by it makes a saturated blue's own output fall when its blue channel rises (slope −0.31 at `[0.46, 0.49, 2.67]`); with positive weights the property is exact. A neutral value is its own luma either way (both rows sum to 1), which is all the estimator measures.

**ARRI's own rendering.** An imported ARRI display LUT and a look that includes its own display rendering (LogC3 to Rec.709) keep ARRI's highlights: the chain ignores the lift for a display LUT (`ColorChain::white_lift()` is then the identity), and for a display-baked look the blend of C12 runs between *our* C12 (placed and lifted) and the look's LUT (neither), like the placement, so at 100 % the picture has no lift and at 0 % it is our C12 with it.

**With the highlight range.** C7's range acts first, on the scene side; the lift acts on the C11 output. The estimator meters the preview after the range at its default, so the range's work is counted: a photo whose highlights reach the camera's clip has them taken towards white by the range already and gets a smaller lift, or none; a photo whose brightest areas sit well under the clip (a sky two stops over grey) is untouched by the range and gets the full lift. Where both act on a pixel the gains multiply, and ACES 2 holds the result at white. The estimator's reference neutral axis uses `ToneParams::IDENTITY` (the range's cap at 7.64 stops would bend it).

**The DNG carries it.** The lift is a scene-linear gain, so it is baked into the DNG's AWG4 pixels after the look and curve and before the fitted profile (`ColorChain::apply_dng_pixel` is C11 decoded and lifted): the DNG's pixels through ACES 2 with the placement (`BaselineExposure`) are the app's picture with the lift in it, to 1e-5 of display light (`crates/ym-core/tests/white_lift.rs`; `crates/ym-color/tests/white_lift.rs` to 1e-12). The lift's amount is not written into the DNG's metadata (a DNG holds the picture).

**The two scalars** (`ym_color::exposure::white::estimate_white`, a pure CPU function of the 1/8 preview, called by `AutoEstimates::compute_with_look` after the exposure). The preview is the one the exposure is metered on (balanced with `g0`, vignetting gain applied), taken through the all-Auto chain at the photo's exposure and C7 at its defaults (the highlight range at 1 and the toe at 0.5, whatever the photo's own Range settings):

1. `L` is the stops over grey of the luma of every pixel. `P` is the 99.5th percentile of `L` (the brightest half percent of the photo) and `B` the 99.5th percentile of a box-blurred copy of `L` (radius `0.007 · width` preview pixels, 8 for a 1188-pixel preview; edges average the part of the window inside the frame). `B` is the level of the photo's highlight *areas*; a few small speculars do not move it. Clipped blocks count like any other pixel (a clipped sky is a white to lift); a non-finite pixel counts as the darkest.
2. `P` and `B` are carried through the **reference look**, the Signature look (`data/looks/builtin/signature.json`, read by `SessionEnv::load`; no look is the identity), on the neutral axis. The lift acts on the C11 output, so what it has to add is what the look has not already given. `anchor` is `B` after the look; the amount is sized for `P` after the look.
3. `d` is the display luma code (0 to 255, ACES 2 with the placement, sRGB) that a neutral tone at `P` has through the reference rendering, and `high_key = clamp((B_before_look − 1) / 1.5, 0, 1)` says how high-key the photo is. The target code is `d + high_key · max(235 − d, 0)`. The ramp reads `B` **before** the look: the look lifts the shoulder by itself (the Signature look by about half a stop above +1 stop), and counting that in the ramp as well would make the lift grow twice for the same photo.
4. `A` is the smallest amount that takes the display code of `P` to the target (a bisection on the neutral axis through the real chain, `ColorChain::apply_pixel`), held to 0 … 2 stops. A photo whose highlights are not above +1 stop gets 0.

The constants are in `ym_color::exposure::white`: `KNEE_EV` (1), `MIN_SPAN_EV` (0.25), `MAX_AUTO_AMOUNT_EV` (2), `MAX_AMOUNT_EV` (3), `OFFSET_RANGE_EV` (3), `TARGET_CODE` (235), `HIGH_KEY_FROM_EV` (1), `HIGH_KEY_TO_EV` (2.5), `PERCENTILE` (0.995) and `BLUR_RADIUS_FRACTION` (0.007). They are **taste, not an ARRI figure and not fitted to any LUT**: an ALEXA at EI 800 holds 7.4 stops above grey and a Sony file at the camera gain 3.58, and the emulation cannot add data, only choose where the shoulder starts.

The estimator never sees the user's settings (light, exposure, grade, look, curve): it runs once per open with the photo's automatic exposure, like the other automatic estimates, so no slider moves it, and "Re-run" recomputes it. The user's offset is added afterwards: `amount = clamp(A + white_offset_ev, 0, 3)`, so offset 0 is the automatic amount, a negative offset of `A` (or more) gives no lift at all (`white_offset_ev = −3` is the switch), and a low-key photo can be given one by hand. There is **no burst rule**: the estimate is a function of the photo's own preview, so the app and `ym` give the same two numbers for any set of photos; it is made at the photo's own exposure and is not re-made when a burst median replaces the exposure (with the constant camera gain that happens only to a frame the highlight guard lowered, by at most one stop). The estimate is **deterministic**: single-threaded, f64 sums in a fixed order and order statistics (`select_nth`), so the same preview gives the same bits whatever the thread count of the decoder or the machine's core count (`tests/white_lift.rs::the_estimate_is_the_same_whatever_the_decode_threads`).

**Measured.** Cost: 100 ms (median of five, dev profile) for the 1188 × 792 preview of a 61 MP frame: the per-pixel pass, the blur and two order statistics (`cargo test -p ym-color --test white_lift -- --ignored --nocapture timing`); the whole open of a photo, decode included, takes 0.2 to 0.5 s on the 12 local photos. On the 12 local photos at Auto (`ym render --cpu`, default look), `A` runs from 0 to 0.98 stops and 4 photos get none: two because the range already takes their highlights to display codes 240 and 247 (A1M01739 and the A7 II) and two because they are not high-key (ILCE-7M3, NEX-5N). A1M01749's 90th percentile of the display luma code is 224.9 with the lift and 203.7 without it. The neutral-axis model of the estimator predicts the measured percentile of the render to within a few codes (`WhiteEstimate::code_after`): the render's percentile is of a colour picture, the model's of a neutral tone.

**GPU.** The amount and `1 / span` ride in the uniform block of the colour kernel (`fl.w` and `grade0.w`; zero for no lift and for a display LUT). The knee, middle grey and the luma weights are written into the shader from the Rust constants (`WHITE_LIFT_KNEE`, `WHITE_LIFT_GREY`, `LUMA_709`, `docs/17-gpu.md` K7). `white_lift_gain` is `log2`, a clamp, a smoothstep and `exp2`: no decision depends on a rounding, so it is correct under Metal's fast-math (K8), and every intermediate is bounded (the exponent is at most the amount, 3). It acts in `display_ours` (ACES 2 path) and in the DNG stop (`Stop::DngLinearAwg4`, unless the chain would take the inverse of a display rendering, which the device does not do) and nowhere else. The `lift/…` cases are also in `chains::cases()`, so every parity, tile-equals-whole and determinism test of the colour kernel runs them. A lift of 0 gives the same bits as no lift, and a lift changes no bit below +1 stop. The CI job `mac-gpu.yml` runs the same tests on Metal.

**Measured (GPU parity, Mesa lavapipe, `crates/ym-gpu/tests/gpu_color_white_lift.rs`, three storage modes).** The gain is within a relative 2.2e-6 (tolerance 5e-6; the gain of a narrow strong ramp is steep, so it is measured on the device's own luma); C11 within 9.4e-7 of LogC4; the DNG stop within 7.5e-6 relative; C12 within 6.6e-6 relative to the largest channel (gate 5e-4); C13 within 4.1e-6 of a code value (gate one 10-bit code).

**Not built.** A local variant driven by the photo's base layer (the same gain, driven by a guided-filter base instead of the pixel's luma, which keeps texture contrast). It needs the base layer computed once on the preview and sampled, never per tile, or tiles would disagree.

### Red chroma (C12 step 1c)

A hue-preserving chroma gain on the reds, in front of the placement and ACES 2. It is the **R1** candidate of the red-fix investigation of 2026-10-24; the owner chose it ("R1 fix looks better than what we have currently"). Code: `ym_color::display::red_chroma` (`RedChromaStep`, the CPU reference), `red_chroma` in `color.wgsl` (the kernel), tables in Rust, not in files.

**Why.** At the same display lightness ACES 2 takes chroma out of mid-tone and light reds (10 to 22 %: the reds "go coral") and adds 30 to 40 % to dark reds. The step gives back what ACES 2 removes above about −0.5 stop, caps the result below the output gamut's edge, and never folds a colour over another. Measured on the investigation's photographs (display chroma against ACES 2 alone): the pink shirt +20 %, the hat in P3 +7 %, the red shirt +4 %; the Lab hue moves by less than 0.1 degree; on 289 reflectances (the spectra of the investigation) nothing outside the reds changes.

**The stage.** `rgb` is linear AWG4 (after the white lift). `Lab` is CIELAB with the white `X 0.95047, Y 1, Z 1.08883`; `YD(s)` is the display luminance of the neutral axis through ACES 2 with the placement, and `CMAX(L*, h)` the largest CIELAB chroma inside the output's limiting gamut (both below).

```
xyz  = AWG4_TO_XYZ * rgb ;  Y = xyz.y            unchanged if Y <= 1e-7 or a component is not finite
s    = log2(Y / 0.18)                            stops over middle grey; s is held at 8 below (the end of YD)
fade = smoothstep(-7.5, -5.5, s)                 unchanged where fade = 0
yd   = clamp(YD(s), 1e-7, 1)
lab  = Lab(xyz / Y * yd) ;  Ld = L* ;  C = |ab| ;  h = atan2(b, a) in [0, 360)
W    = hue window: 1 on [3, 31] degrees, raised cosine to 0 over 30 degrees below and 12 above   unchanged where W = 0
kap  = clamp((20 / max(C, 8))^0.65, 0.35, 1.25)
P(s) = 0.34 (s + 0.5)                      for s >= -0.5          (the up side)
       -0.43 tanh((-0.5 - s) / 1.65)       for s <  -0.5          (the dark side)
lnf  = kap * P
if P > 0:
    k0   = 1 + 0.45 tanh((exp(lnf) - 1) / 0.45)                   the gain, softly held under 1.45
    Cin  = C * k0 ;  Cmax = max(CMAX(Ld, h), 1)
    Lim  = Cmax * exp(kap(0.9 * Cmax) * P) ;  kn = 0.8 * Lim
    Csoft = Cin if Cin <= kn else kn + (Lim - kn) tanh((Cin - kn) / (Lim - kn))     the cap
    x    = Csoft - C ;  Cb = C + (x > 0 ? x^2 / (x + 0.015 C + 1e-9) : 0)           never below C
    lnk  = f_up * W * ln(Cb / C)                                  (ln k0 if C <= 1e-4)
else:
    lnk  = f_dn * W * lnf
lnk *= fade ;  lab.ab *= exp(lnk) ;  Y kept ;  back to XYZ and AWG4
```

**The strengths are per look.** `f_up` multiplies the up side (`P > 0`: from −0.5 stop up) and `f_dn`, the look's **dark relief**, the dark side (`P < 0`). The look file's `red_chroma` block sets them (`docs/08`): `up` (default 1) and `dark_relief` (default 0), each 0 to 1. So Neutral, every imported look and every look that sets nothing have `f_up = 1` and `f_dn = 0`: the full up side, and ACES 2's dark boost left alone. **Signature** sets both to 0 for now (`"red_chroma": { "up": 0, "dark_relief": 0 }`): its colour saturation of 1.45 already gives back more than ACES 2 takes from light reds (the pink shirt reaches C* 51.1 with the up side on, 43.3 without it, against 37.3 in the camera JPEG), and its dark reds are a separate decision of the owner. With both strengths 0 the step is skipped (`ColorChain::red_chroma()` is `None`): no cost, and the picture is bit for bit the one of before the step existed (`look_signature.rs`, `gpu_color_red_chroma.rs`). At look intensity `k` the chain uses `neutral + k (look − neutral)`, per strength (Signature at 50 %: `f_up` 0.5), so 0 % is Neutral's picture exactly; a look with its own display rendering (LogC3 to Rec.709) is blended with our C12, which has Neutral's strengths; an imported ARRI display LUT has none of the step.

**Tables.** Both are computed in Rust from `ym-color`'s own ACES 2 and gamut code, once per limiting gamut (Rec.709 for the sRGB output; P3-D65 for the viewport, Display P3 and Rec.2020 outputs, which ACES 2 limits to P3), and read by the CPU reference and the kernel. `YD(s)`: the Lab white at `0.18 · 2^s` through the placement and ACES 2, as the luminance of the limited display light, 73 entries from −10 to +8 stops in steps of 0.25, linear between them and held at the ends. `CMAX(L*, h)`: 40 bisection steps on the chroma at each of 41 (L* 0 to 100, step 2.5) by 72 (hue 0 to 355, step 5) nodes for the in-gamut test of linear display light, bilinear between them and periodic in hue. They match the investigation's tables (`crates/ym-color/tests/data/red_chroma/{yd_of_s,cmax_srgb,cmax_p3}.csv`): `YD` to a relative 4.1e-9, `CMAX` to 5.0e-4 (the CSVs have three decimals).

**Properties, all tested** (`display::red_chroma::tests`, `tests/chain_cpu.rs`, `tests/look_signature.rs`):

- Grey is bit-exact: the Lab hue of the neutral axis (the 2e-4 offset of AWG4's white from the Lab white) is about 260 degrees, far outside the window.
- The step returns its input bit for bit where `W = 0`, where `fade = 0`, for `Y <= 1e-7` and negative `Y`, for a non-finite component, and on the side of `P` whose strength is 0 (`f_dn = 0` leaves everything under −0.5 stop untouched, `f_up = 0` everything over it). A finite input never gives a NaN.
- Hue (Lab at the display lightness) moves by less than 1e-9 degrees and `Y` by less than 1e-12 relative.
- No fold: the output chroma grows with the input chroma over stops −6 to +6, hues −40 to +50 degrees and chromas up to 1.6 times the gamut edge. Smallest slope of output against input chroma: Rec.709 0.501 up to 1.0 times `CMAX`, 0.158 up to 1.25, 0.095 up to 1.4, 0.0905 up to 1.6; P3 0.493, 0.171, 0.108, 0.0700 (the investigation's 0.16 and 0.13 are sweeps to about 1.25 times the edge). The slope is smallest well outside the gamut, where the cap squeezes the boost out.
- `s` is held at +8 stops (the end of `YD`, display white) so the step stays finite for the brightest scene values; `tanh` arguments in the kernel are held to 20 (`tanh(20)` is 1 in f32 and f64).

**Parity.** Python reference to Rust: `tools/vectors/red_chroma_vectors.py` is an independent numpy implementation of the maths above (the matrices from the primaries, `CMAX` by bisection, `YD` from the committed CSV) and writes 416 AWG4 samples (both gamuts, four pairs of strengths, stops −8.5 to 9.5, hues across the window and elsewhere, chromas to 1.6 times the edge) to `tests/data/red_chroma/vectors.json`; the Rust step agrees to 3.8e-10 absolute (2.9e-9 relative to `max(|v|, 1e-3)`), gate 1e-8. The generator agrees with the investigation's own script (`cand.py`, 408 samples) to 4.8e-9. GPU to CPU (`gpu_color_red_chroma.rs`, Mesa lavapipe, three storage modes, Viewport and sRGB, strengths (1, 0), (1, 0.5), (0.5, 1), (0, 1) on 624 reds and other colours): C12 within 1.4e-5 of the CPU relative to the largest channel (gate 5e-4), C13 within 8.3e-6 in encoded value, 0.009 of a 10-bit code (gate 1 code), the step's own part (the change it makes) within 9.5e-6 of display white (gate 1e-3). The DNG stop (C11 decoded, lifted, through the step with the P3-D65 set) is within 1.4e-5 relative of `apply_dng_pixel`, for both outputs. Tiles equal the whole image bit for bit; grey and hues outside the window are bit-identical to a chain without the step.

**The GPU kernel.** `red_chroma` in `color.wgsl` runs inside `display_ours`, between `lifted_scene` and the matrix `AWG4_TO_AP0` (the placement), and after `lifted_scene` in the DNG stop (6). The two strengths ride in the uniform block (`rc.x`, `rc.y`; both 0 skips the stage and is what the host writes for a display LUT); the tables sit in the persistent `tables` buffer of the output after the ACES 2 tables (`T_RC`): one set of 3,025 floats for Rec.709 and one for P3-D65 (24 KB), because the DNG stop always reads the P3-D65 set while the picture reads the set of the output's limiting gamut (`rc.z`). The constants are written from `ym_color::display::red_chroma` into the shader, like the other stages'. Every branch point is continuous, so nothing rests on a rounding under Metal's fast-math (K8, `docs/17`).

**The DNG carries it.** Like the white lift, the step is a scene-linear AWG4 to AWG4 map and is baked into the DNG's pixels after the look and curve and before the fitted profile (`ColorChain::apply_dng_pixel` is C11 decoded, lifted, and through the step; the device's DNG stop does the same). A DNG has no output gamut of its own, so the step uses the **P3-D65** tables whatever the chain's output (the viewport's gamut; the Mac display path of the app that opens the DNG); the DNG's pixels through ACES 2 with the placement are then the app's Display P3 picture (6.3e-8 of display light for Neutral, `crates/ym-core/tests/export_dng_carries.rs`; to 1e-12 in `tests/white_lift.rs` and `tests/chain_cpu.rs`), and the sRGB picture except for colours near the Rec.709 edge, where the sRGB cap acts. The look's strengths apply (Signature: none, so its DNG is as before). A look with its own display rendering above 0 % and an imported display LUT are inverted from their picture through plain ACES 2 (`docs/09`) and have no step in front of that inverse.

**Switching it off.** `ChainParams::red_chroma` (default `true`) is for tests and tools that want plain ACES 2, such as the OpenColorIO vectors.

**Pipeline version.** The step changes the picture of Neutral, of every imported look and of every look that does not set `red_chroma`, so `PIPELINE_VERSION` is 0.6.0. Signature's pictures are unchanged.

### Imported ARRI display LUT

If the user chose "ARRI's display rendering" in the wizard or preferences, C12 is the imported display `.cube` (for example LogC4 → Rec.709 gamma 2.4, or → P3). LogC4 display LUTs apply to the LogC4 output of C11 directly; LogC3 display LUTs (the older package, including the classic Rec.709) apply after converting LogC4 → linear AWG4 → AWG3 → LogC3 EI 800. For the viewport, the LUT's output transfer (gamma 2.4 for Rec.709) is decoded to display-linear, the primaries are converted to P3, and the result is encoded with the sRGB transfer. For export, the conversion goes to the chosen colour space the same way.

`ArriDisplayLut { lut: Arc<Lut>, input: LogC4 | LogC3Ei800, output: DisplayLutOutput }`: LogC4 input directly, LogC3 after the exact conversion; the output codes are decoded with the LUT's transfer, converted by a 3×3 to the output's primaries (a colour outside the output gamut is clipped by C13, not gamut-mapped) and encoded by C13. `DisplayLutOutput::from_target("rec709-g24")` reads the importer's target slug: primaries from `709`/`srgb`, `p3`, `2020`; transfer from `gNN`/`gammaNN`/`srgb`. With no gamma in the slug, Rec.709 and Rec.2020 are gamma 2.4 and P3-D65 is gamma 2.6; HLG, PQ and DCI-white targets are refused (`UnsupportedTarget`). The slugs of the real ARRI packages are verified in task M3-H2.

## C13 · Display encode

sRGB transfer for sRGB and Display P3, BT.1886 inverse (gamma 2.4) for Rec.2020. 8-bit outputs are dithered with a fixed blue-noise texture (seeded, deterministic). The matching ICC profile is embedded on export (`docs/09-export.md`).

`ym_color::encode`. sRGB (IEC 61966-2-1, thresholds 0.0031308 and 0.04045) for sRGB and Display P3, the pure `x^(1/2.4)` for Rec.2020; both odd-symmetric below zero, clamped to [0, 1] before quantising (`OutputSpace::encode_clamped`). `quantize(v, bits)` rounds to nearest (8, 10 or 16 bits; a NaN is 0).

**Dither (8-bit only).** A 64 × 64 void-and-cluster rank texture (Gaussian σ = 1.5 on a torus, seed `0x59454c4c4f574d4f`, integer weights and lowest-index tie-breaks; generated by `encode::blue_noise::generate`, embedded as a table so no platform's `exp` can change it, fingerprint `0x993fccd5c6db3aa9` checked in the tests). The threshold at absolute output pixel `(x, y)` is `(rank[y mod 64][x mod 64] + 0.5)/4096`; the code is `floor(v·255 + threshold)`; all three channels share the threshold (achromatic noise). Over one tile the thresholds are the 4096 evenly spaced values in (0, 1), so a constant input averages to its exact value (measured error 0 over 1021 levels) and the result is never more than one code from `v·255`. Spectrum: power at radial frequencies 1 to 5 is 1e-4 of the power at 16 to 33 (white noise would be 1). A change to the table changes every dithered export (bump `pipeline_version`).

**C13 transfer.** The chain encodes display-linear with `encode` (IEC 61966-2-1, break at 0.0031308). The ACES 2 port's own `transform_ap0` ends with the reference's continuous "moncurve" (slope 12.9232, break at 0.0030399); the two differ by at most 9.8e-6 near linear 0.00304 (0.0025 of an 8-bit code, 0.64 of a 16-bit code). The Rec.2020 output (a pure power law) is unaffected. The WGSL kernel follows C13 (IEC).

### The C7–C13 CPU chain

`ym_color::display` (`ColorChain`, `ChainParams`, `DisplayTransform`, `OutputTarget`, `Frame`): the CPU reference of everything per pixel after the fused C1–C6 matrix, which the CPU fallback runs and every GPU kernel is parity-tested against.

- **Order.** `M · x` (supplied as a 3×3) → C7 → C8 → C9 (with `L` from the linear values entering C8) → C9b → C10 → C11 → C12 → C13. A LogC3-to-Rec.709 look is classified when the chain is built: C10 is skipped, the curve runs first, and C12 evaluates both our display transform and the look's LUT from the same C11 output and blends them in display-linear (`blend_display_linear`, in the output's primaries, so the look's Rec.709 is converted first). At intensity 0 the result is our C12 bit for bit, at 100 it is the look's. Neutral, an identity look and an intensity of 0 skip C10 and are exact identities. `ColorChain::trace(rgb)` returns every stage's value for debugging kernel parity stage by stage.
- **Entry points.** `apply_pixel` (code values in [0, 1], f64); `apply_to_display_linear` (C1–C12, the picture an exporter resizes in linear light before `encode`, `encode_u8` or `encode_u16`); `apply_logc4_pixel` and `apply_dng_pixel` (stop after C11, decoded to linear AWG4; with a look that includes display rendering at an intensity above 0, or an imported display LUT, it is the scene-linear AWG4 triple that ACES 2 turns into the chain's display-linear sRGB picture, `dng_inverts_display()`, limited to `DNG_SCENE_LIMIT / 2^DISPLAY_PLACEMENT_EV`); `apply_before_pixel`; and band-parallel buffer renders over planar or interleaved f32 (`render_encoded_f32`, `render_display_linear_f32`, `render_dng_linear_awg4_f32`, `render_u8`, `render_u16`, `encode_display_linear_u8/u16`). The planar layout is `ym_gpu::cpu::RgbImage`'s. Threads are `std::thread::scope` workers over row bands (no `rayon` in `ym-color`'s manifest); output is bit-identical for any thread count, and 8-bit dither uses absolute pixel coordinates (`origin`), so tiles agree with the whole image. NaN camera samples count as 0, infinities are limited to ±1e30.
- **Before view.** The photo as the camera would render it by default, outside the stages above (section *The Before view*). `BeforePoint::CameraDefault` is the app's; `BeforePoint::CameraLog` (the output of C8) and `BeforePoint::AfterGrade` (C9) remain in `ym_color::display` for tests and diagnostics.
- **Not on the device.** Every stage of C7–C13 has a WGSL kernel except the inverse display rendering that the DNG of a display-baked look or an imported display LUT needs (`dng_inverts_display()`, the ACES 2 inverse above), which runs on the CPU only. The CI job `mac-gpu.yml` runs the GPU parity tests on Metal; Metal timings are not measured.
- **Measured speed** (`tests/display_perf.rs`, CPU time per pixel; one 4-vCPU machine shared with other builds, so wall times are inflated). Full chain with ACES 2, camera RGB to dithered 8-bit: 1.05 µs of CPU per pixel with zero grade, Neutral look and identity curves, 1.27 µs with grade, a 33³ look and curves; the DNG stop (C1–C11) is 0.48 µs. Projected on four free cores: about 3.6 and 2.9 Mpx/s, a 24 MP frame in 7 to 8 s. Preparing a chain takes 17 to 40 ms the first time (ACES 2 tables), then under 1 ms.

## The Before view

The Before view shows the photo as the camera would render it by default, so the user can see what Yellowmoon changes. It uses the raw stages R1 to R9 of the final picture except R3, the lens vignetting correction, which the camera's own picture leaves out: demosaic, denoise, the distortion and chromatic-aberration corrections, the decode switches, crop and orientation are as for the final picture. Then only constants and facts of the file:

```text
camera RGB (the mosaic x g0, as it enters the chain)
  --M_before--> linear Rec.709          M_before = M_xyz_to_rec709 . inv(CM_n) . diag(gs / g0) . 2^CAMERA_GAIN_EV
  --camera tone curve, per channel--> display-linear Rec.709
  --from_linear_rec709--> the output primaries --C13--> code values
```

- **As-shot white balance** `gs = [R/G, 1, B/G]`: the gains stored in the file, on the body's own raw scale (no white scale, `docs/03-camera-profiles.md`), not our estimate. A file without usable gains keeps `g0` itself (the identity balance). `g0` is divided out because the raw stages already multiplied the mosaic by it, so the result does not depend on it.
- **Body colour matrix** `CM`: the D65 colour matrix of the body (XYZ to camera RGB), numbers from the bodies' published D65 colour matrices (`data/cameras/color-matrices.json`, `docs/14-licensing.md` rule 8). An entry is found by the file's Sony model ID when the file has one, otherwise by its Model string (an ID that is not listed gives the A7R IV entry even when the Model string names a listed body: the ID wins, as in `docs/03-camera-profiles.md`). A body that is not listed, a Make other than Sony and the two bodies with no published matrix (ILCE-7M5 and ILCE-7RM6) use the A7R IV entry, as the profile registry falls back to the A7R IV profile. Each entry's ID is the one Sony publishes for that body. Two entries borrow their numbers and say so in a `note`: the A1 II (ILCE-1M2, ID 400) takes the A1's numbers, because it has the A1's sensor (the published numbers for the two are the same), and the A3000 (ILCE-3000, ID 302, which Sony writes for the A3500 too) takes the generic E-mount matrix, the numbers the published tables give any E-mount body without an entry of its own (the NEX-5N's; the A6000, A5000 and A5100 carry them too). `CM_n` has its rows scaled so that D65 white maps to camera `[1, 1, 1]`, so the as-shot white comes out exactly neutral (R = G = B).
- **Camera gain** `CAMERA_GAIN_EV` (+1.11 EV, *Auto exposure*): the same constant as the final picture's exposure, so middle grey 0.18 sits where the camera's own JPEG puts it.
- **Camera tone curve**, per channel in linear Rec.709, the same on every body: `y = min(1, k t / (t + s))`, `t = max(x, 0)^n`, with `n = 1.6558`, `sigma = 0.4569` (`s = sigma^n = 0.273358`), white 2.5 (`y = 1` from `x = 2.5`) and `k = (white^n + s) / white^n = 1.059955`. Middle grey 0.18 lands on display-linear 0.18675, sRGB code 119.7 of 255 (the median of the 12 embedded JPEGs at that level is 119.8); the slope there is 1.36 in log-log, about 52 codes a stop; the neutral raw clip (2.16) lands on code 253. A negative linear value is clamped to 0 per channel, as in the JPEGs, and the curve is applied in Rec.709 primaries (what the JPEG engine does) before the conversion to the output's primaries, which keeps the picture inside the sRGB gamut. The constants are in `ym_color::camera_default` (`TONE_N`, `TONE_SIGMA`, `TONE_WHITE`, `TONE_S`, `TONE_K`); `data/cameras/camera-default-tone.json` carries them, a 28-point golden table and the fit summary, and a test pins the file to the code.
- **What is not in it:** no sensor map (the ALEXA emulation), look, grade, curves, display transform (ACES 2 or an ARRI LUT), highlight range (C7), display placement or output sharpening, and **no lens vignetting correction (R3)**. No user setting reaches it: exposure, white balance, look, tone, colour, curves and the RAW tab's Range do not (the fused matrix is not used at all), and neither do the Detail controls that sit in the raw group, noise reduction and capture sharpening, which the Before view reads at Auto whatever they say (`PhotoSource::for_before_view`). The decode switches and the lens corrections do change it, as they change the raw (a switch that moves pixels, such as the distortion correction, moves them in the Before view too; the lens switch also decides whether the file's vignetting table exists for the final picture, and the Before view never applies it), and so do crop and orientation (geometry is applied as for the final picture).

### Implementation

CPU reference: `ColorChain::apply_before_pixel` (`BeforePoint::CameraDefault`, the default; `ChainParams::camera_default` is the per-photo `CameraDefault`, `CameraDefault::neutral()` when a caller does not know the body). The pixel readout (`sample_pixel`) and every buffer render of `View::Before` follow it. `ym-core` builds the `CameraDefault` per photo in `color_plan` (`camera_default_of`: the table by model ID, the as-shot gains from the raw, `g0` from the estimates) and puts it in the grade key, so a frame of the Before view is never served for another body or white. The colour kernel (`Stop::Before`, `lk.w = 2`) computes the same: the 3×3 and the curve constants are in the uniform block (`cam0` to `cam2`, `camt`; 1056 bytes), the Rec.709 to output matrix is the one the display-baked look uses (`look_m*`), and the result is already C13-encoded. The histogram is of the final picture and does not change.

**The raw front without R3.** `PhotoSource::for_before_view` sets `before_view` (and puts Detail at Auto); `Job::params` turns that into `RawParams::vignetting = false` when the photo has a vignetting table to apply (`has_vignetting_profile`: a mosaic file, a usable 0x7032 table, the lens switch on). `RawPlan::without_vignetting` is then the plan (levels, lens maps, crop and clip answer are the final picture's; only the profile is left out), and the `FullChain`, the device's reduced front (`reduced_params` has no vignette table, so the prep kernel skips R3) and the tiles are built from it. The flag is part of `RawParams::key`, so every cache of the raw group (`groups`, `tiles`, the display and frame caches) keeps the Before view's entries apart from the final picture's and nothing made without R3 is ever served for a picture that has it. The graph keeps one plan and one prepared chain for each of the two, so flipping back and forth prepares neither again; the sigma plane is shared (it is made before the gain). A photo with no table to apply (none in the file, the lens switch off, a camera-processed Lossless M or S file) is the same with and without R3, so its Before view uses the final picture's raw front and costs no raw work. The device and the CPU follow the same flag.

The raw front is re-run without R3, rather than dividing the vignetting gain back out of the final picture, because only the first is exact: the clip mask and highlight reconstruction of R5, the noise-adaptive denoise and the R9 resampling all act on data with the gain in. With the lens switch on, the Before view is bit-identical to the one with it off for a file whose only table is the vignetting one.

**Measured (cost of flipping to Before).** Linux x86_64, 4 cores, 21 to 31 MP files of three bodies, on the CPU and on Mesa lavapipe (a software Vulkan, far slower than Metal; Metal timings are not measured). The cost is one more raw front, once per photo, and only where the photo has a table.

| First Before frame | CPU | lavapipe | Caches grow by |
|---|---|---|---|
| At Fit (the raw group at 1/2 or 1/4) | 0.5 to 1.2 s (first final frame: 1.1 to 2.3 s) | 1.4 to 2.9 s | 23 to 67 MiB (the group at 1/k) |
| At 100 % (12 to 15 tiles) | 1.8 to 3.5 s (first final frame: 2.8 to 4.8 s) | 3.3 to 6.0 s (final: 5.4 to 9.6 s) | 132 to 167 MiB on the CPU path (about 92 MiB of that is the second full-resolution sigma map of a 24 MP chain, made only when a CPU stage runs); 42 to 51 MiB with the tiles on the device, with a transient peak 260 to 370 MiB above the final picture's |

Both are bounded by the caches' own budgets (a sixth of the CPU budget each for groups and tiles, shared by the two views, so Before's entries push out the oldest). Every later flip is a cache hit (1 to 5 ms) and makes no raw work. Four of the 12 local photos (A300, A500, NEX-5N, NEX-7) have no table: for them the first Before frame costs 55 to 200 ms.

### Tests

- `ym_color` (`chain_cpu.rs`, `camera_default.rs`): the chain equals an independent composition for every output; the view is bit-identical whatever the fused matrix, tone, grade, look, curves, display and dither say; the tone curve is monotone, bounded and defined for NaN and infinities; mid grey after the gain is code 119.7 (within 0.5); the as-shot white is neutral whatever `g0` is.
- `ym-profiles` (`camera_default_data.rs`): both data files validate against their schemas (`crates/ym-profiles/schema/`); the embedded table is the file; every registry body has its own entry except ILCE-7M5 and ILCE-7RM6 (the A3000, ILCE-3000, is in the table but is not a registry body); every entry has the model ID Sony publishes for its body (a pinned list of all 44, because the registry test walks only the 28 registry bodies); the A1 II has the A1's numbers and the A3000 the NEX-5N's, and only those two carry a `note`; an as-shot neutral renders R = G = B for every body in the table; the tone file equals the constants and its table equals the formula.
- `ym-core` (`render_before.rs`): the Before frame is bit-identical across edits of every user setting (exposure, white balance, the grade, white offset, look intensity, curves, roll-off, noise reduction, sharpening, output sharpening), on the CPU and on the device; neutral scenes render neutral for bodies with different matrices, including the A1 (its profile has a white scale); a body that is not in the table gets the A7R IV's `CameraDefault`, the A1 II the A1's and the A3000 (by ID 302, by its Model string, and the A3500) the NEX-5N's; the readout follows the view; extreme manual Detail settings, or none, do not move the Before view at Fit or at 100 % while they do move the final picture of a noisy photo.
- No vignetting correction (`render_before.rs`): on a flat field darkened by a known vignetting table the final picture's corner is as bright as its centre (ratio 1.000) and the Before view's is not (0.947 against 0.941 expected from the camera curve; with a distortion table too: 0.948), on the CPU and the device. The Before view is bit-identical with the lens switch on and off when the vignetting table is the file's only one; its size is the final picture's (with the distortion crop); the CPU and the device agree at Fit (the reduced group) and at 100 % (tiles) to one 8-bit code (none of 270 000 bytes more than a code off). Flipping to Before and back makes its own raw group, chain and tiles once, shares the sigma plane, does no raw work afterwards and changes no picture.
- `ym-core` unit tests (`raw.rs`): the Before plan is the final plan without the profile and the key differs; `has_vignetting_profile` agrees with the plan; the device's reduced group without R3 against `reduced_bayer`: max abs 1.19e-6 at 1/2, 7.2e-7 at 1/4, 4.8e-7 at 1/8 (bound 1.9e-6; the prep kernel without a profile is bit-identical to the CPU in `ym-gpu`'s `gpu_front.rs`). `tests/before_cost.rs` (ignored, local files) measures the costs above.
- `ym-gpu` (`gpu_color.rs`): the kernel against the CPU in every storage mode (f32, f16 packed, f16 native) and all four outputs, with the gates of the other stages (one 10-bit code in the encoded picture, 2e-4 in display-linear light): worst `|GPU − CPU|` 1.08e-6 encoded and 1.07e-6 linear on Mesa lavapipe; bit-identical on the device across settings.

### The fit and how well it holds

The curve was fitted to the embedded JPEGs of nine photos from nine Sony bodies in Standard style and checked on 12 photos from 10 bodies (the three photos of the A1 are in Neutral style with contrast and saturation High and were left out of the fit: they agree with it after about −0.14 EV). The fit is a quantile match of scene luminance against the JPEG's and Naka-Rushton least squares in sRGB code space with the white pinned at 2.5 (a free fit chose 2.46). The per-photo gain offsets are small (median −0.013 EV, std 0.10 EV, range −0.14 to +0.20); the spread between bodies is about 5 codes at mid-tones and 1.5 to 1.9 codes once each file's own gain is removed; leave-one-photo-out moves the mid-grey code between 119.3 and 120.9.

**Measured.** Against the same 12 JPEGs the Rust Before view (registered pixels, central 80 %, flat and unclipped; a local check, not a gate) has a mean per-channel median error of **4.3 codes (95th percentile 12.2) and a mean ΔE76 of 2.8**; the fit's own Python chain has 4.3 (12.0) and 2.9. By channel the mean of the medians is 5.5, 4.4 and 4.3 (red, green, blue). In the outer ring (outside the central 80 %, inside the central 98 %) the figures are 4.7 (14.6) and 3.3. The bias of the ring over the centre (Rust minus JPEG, median codes) is 1.3 codes on average: A7 III +0.1, A7 II +2.5, A7 IV +0.6, and the A1 photos −4.9, 0.0 and −0.4. Eight photos have a vignetting table in use; the other four (A300, A500, NEX-5N, NEX-7) have none. The 95th percentile is dominated by the camera's local tone and colour (DRO, hue), not by the curve.

**Limits.** 12 photos, 10 bodies, one scene each, one rendering style; the bodies after the A7 IV (A7 V, A7R V, A1 II) were not sampled. The hue differs from Sony's by 4.5 degrees at the median (1.3 to 9.9), which a per-channel curve cannot fix; the JPEG's chroma over ours is 0.94 at the median (0.78 to 1.17), so no separate saturation factor is used. In the deep toe (−4.5 to −3.5 stops) the JPEGs are 2 to 3 codes brighter than the curve. Several bodies share one published matrix (sensor siblings; fourteen groups of two to five bodies in the table, 37 of the 44 bodies), which is right where the sensor is the same and unverified where it is not; the A1 II and the A3000 are two of those by assignment, not by publication.

## Illuminant estimate

Input: the 1/8-resolution preview of black-subtracted camera RGB **before** any white balance, with clipped pixels masked, plus metadata (as-shot white balance, exposure settings).

Output: `w_scene` (camera RGB of the light, green = 1); class probabilities over {daylight, shade/overcast, tungsten/halogen, fluorescent, LED, mixed}; a confidence value.

The estimator is FFCC-style (Barron & Tsai 2017, Apache-2.0 reference code): a log-chroma histogram, a learned filter that gives the chromaticity, a Gaussian prior from the camera's as-shot white balance, and a small head for classes and confidence. It is chosen over C5 and DMCC because it wins every held-out comparison below, passes the synthetic gate on all profiles, and costs 13 k parameters and about 4 ms per photo; C5 and DMCC need a multi-camera real dataset that V1 cannot get.

`ym_color::illum` (`IlluminantModel`, `RawPreview`, `ShotInfo`, `IlluminantEstimate`) reads `data/models/illum/<profile-id>.json` (`IlluminantModel::load_for_profile(data_dir, profile_id)`). `training/illum/` is the training code and the numerical reference (the Rust agrees with it to 1e-9 on `crates/ym-color/tests/data/illum/`); `data/models/illum/MODEL_CARD.md` has every number below for all 11 profiles. There is one model per profile (not per family), each 13 k parameters.

### Method

1. **Input contract.** The preview is **raw**: black-subtracted camera RGB at about 1/8 of the sensor resolution, white level 1.0 for every channel, **no white balance applied** (not the `g0`-balanced data of the chain), with a byte mask of the blocks that contain a clipped sample. For a stand-in body with a white scale (`docs/03-camera-profiles.md`) the caller multiplies red and blue by it first, so the estimator reads the profile's raw scale it was trained on (the clip level of that preview is then the scale, not 1.0; the mask, not a level, decides what is clipped). `w_scene` is in the **published raw scale** of the profile's curves (`fit.sony_curves_raw`, clipped at zero, not peak-normalised; see C1, *The scale of the locus*), and it is compared with the as-shot gains (tag 0x7313 as multipliers `[gR, gG, gB]`; `ShotInfo::as_shot_uv` takes `(ln gR/gG, ln gB/gG)`). The estimator, the locus and the as-shot gains all live in that scale; `chain.rs` is scale-free.
2. **Histogram.** Log-chroma `u = ln(G/R)`, `v = ln(G/B)`, two channels of 64 × 64 bins of 0.0625 in `(u, v)`, the window centred on the profile's light locus (1500 to 25000 K) so that the saturated surfaces of every light fit: the pixels (weight `sqrt(Y)`, `Y = 0.25 R + 0.5 G + 0.25 B`) and the 3 × 3 masked local absolute deviation (the edges, as in FFCC; weight `sqrt(Y)` of the edge triple). Splatting is bilinear (not FFCC's rounding). A pixel is counted when all channels are finite and above 1/1024 (edges: 1/4096); clipped blocks are masked. Previews are subsampled with the stride `isqrt(pixels / 16384)`, so a preview of fewer than 65 536 pixels is histogrammed whole and the 1080 × 720 preview of a 50 MP frame uses every 6th pixel in both directions (21 600 pixels).
3. **Filter and Gaussian.** `Z = sum_c IFFT(FFT(X_c) FFT(F_c)) + B` (circular convolution, own radix-2 FFT), softmax, and the mean and covariance of the softmax map in log-chroma plus a learned variance floor. FFCC's toroidal (von Mises) mean is not needed: the window leaves a margin on every side and the bias suppresses the rest. Training uses the FFCC losses (softmax cross-entropy and Gaussian negative log-likelihood, for both the histogram-only and the fused estimate, plus an L1-like error) and the FFCC regularisers (total variation and L2 on filter and bias).
4. **As-shot prior.** The camera's as-shot white balance, converted to a chromaticity, enters as a Gaussian with sigma 0.05 on `(u, v)`, multiplied into the estimate (precision-weighted mean), not added to the logits. Sony's own auto white balance is good; the estimator refines it. Without an as-shot value (or with unusable gains) the estimate is the histogram's alone.
5. **Output.** `w_scene = [exp(-u), 1, exp(-v)]`; six class probabilities from a 25 → 24 → 6 `tanh` head on the histogram moments, the estimate, the as-shot value and the scene brightness `Bv = log2(N²/t) − log2(ISO/100)` (`ShotInfo::bv`); and a confidence, a logistic regression on the same features and three more (Mahalanobis distance of the histogram estimate from the as-shot value, the log variances of the two Gaussians), fitted on validation scenes with held-out lights to be the probability of an error under 2 degrees. A preview without a usable pixel (everything clipped, dark or non-finite) returns the as-shot white, or the camera's D55 if there is none, with confidence 0 and the class frequencies (`fallback = true`). `IlluminantEstimate::scene_light(&locus)` is `SceneLight::from_estimate` (the six classes fold into the profile's four there).

**Training data.** Synthetic only, all splits by light **and** by reflectance (details and licences in the model card). Scenes of soft-edged colour cells with shading, texture, highlights, grey patches and mixed lighting are rendered through the profile's raw curves with Poisson-Gaussian noise and clipping. Reflectances are the rawtoaces 190 patches split 132 / 20 / 38 (train / validation / test) plus smooth random spectra and convex mixes. Lights are CIE D series, blackbody, FL1–FL12, FL3.1–FL3.15, LED-B1..V2 (CC BY-SA) and generated near-white LED and fluorescent spectra, with held-out temperature bands, held-out CIE lamps and disjoint seed streams; the nine sensor-map gate lights are held out. **CES 99 is evaluation only.** The simulated as-shot white balance and the class and brightness priors are assumptions (no camera data in V1).

### Measured

Angular error in degrees, median / p90 / worst. `data/models/illum/MODEL_CARD.md` has all of them for all 11 profiles.

- **Gate, held-out synthetic scenes** (3000 scenes per profile: held-out lights and held-out reflectance patches; gate: median ≤ 2). With the as-shot prior the median is 1.22 to 1.30 over the 11 profiles (all pass; this is the gate figure, because the estimator ships with the prior). Histogram only: 1.79 to 1.98 (re-drawn test sets: 1.72 to 1.98 on ten bodies and 1.98 and 2.08 on the A7R II; the median of 3000 scenes has a sampling noise of about 0.03 degrees, so the histogram-only figure sits within noise of the 2-degree line for the bodies at the top of the range). On the 99 CES (evaluation only): 1.23 to 1.33 with the prior and 1.67 to 1.90 without. A7R IV: 1.23 / 4.93 / 22.1 with the prior, 1.93 / 6.53 / 42.9 histogram only.
- **Against simpler estimators** (A7R IV, same scenes): grey world 3.76 / 8.63 / 40.0, white patch 4.75 / 11.01 / 26.8, shades of grey 4.48 / 10.33 / 27.9, grey edge 4.29 / 9.67 / 27.3; the simulated as-shot white balance alone 1.54 / 6.46 / 26.0; grey world fused with the same as-shot prior 1.55 / 6.30 / 24.1. FFCC without the prior halves the error of every histogram baseline; with the prior it beats the as-shot value alone by 20 % in the median and 24 % at p90 (the as-shot error model is an assumption). Dropping the edge channel costs 0.14 degrees in the median and 0.4 at p90; training on the 132 patches without perturbed and mixed reflectances gives 4 degrees on unseen patches instead of 2, so the training mix is broader than the held-out one.
- **Classes** (A7R IV, with prior): accuracy 62.7 % over six classes, 72.4 % over the four profile classes (mixed excluded). Warm LED and tungsten, and LED and fluorescent, have the same chroma and are separated only by small distances from the locus and by `Bv`; the confusion matrix is in the model card. Confidence: AUC 0.749, Brier 0.174 against 0.215 for a constant.
- **Stand-in cost** (the A7R IV model on scenes rendered through the other ten bodies' curves): median 1.18 to 1.64 with the prior, 1.92 to 2.89 histogram only.
- **Independent scenes.** The 37 chart scenes of `training/synth` (another generator: ColorChecker-style chart of the 190 patches, mosaics through the same curves, ISO 100 to 12800, all four CFA phases, 12 and 14 bit; previews binned 8 × 8 from the mosaic like the Rust preview) read 0.22 degrees median histogram only (grey world 1.64, the camera's as-shot white 0.67), mixed lighting 3.5. That confirms the preview contract (black level, CFA, raw scale) end to end. Classes there: daylight 21 of 21 and tungsten 6 of 7 right; the fluorescent scene and all 7 LED scenes are not recognised (called daylight or mixed).
- **Real photographs.** `training/illum/smoke_real.py` runs the models on the nine ARW files in `testdata/real/files` (no ground truth): the histogram-only estimate lies within 0.4 to 6.0 degrees (median 3.0; grey world 0.2 to 9.4, median 4.0) of the camera's own white balance. This is a gross-failure check, not an accuracy number.
- **Speed.** One estimate of the 1080 × 720 preview of a 50 MP frame (21 600 pixels histogrammed, stride 6): 3.6 ms on a shared 4-core machine, of which the histograms take 3.0 ms; the FFTs, the fit and the head take under 1 ms (`cargo test --release -p ym-color --test illum -- --ignored --nocapture timing`).

**Cross-checks and robustness.** A numpy implementation written from this section alone (not from `training/illum/`) agrees with the reference `training/illum/model.py` to 1.6e-13 on all 11 shipped models (histograms, filter, both Gaussians, class head, confidence). The Rust estimator agrees with the reference to 2.8e-13 on all 11 (`every_shipped_model_matches_the_python_reference`); regenerating the vectors gives byte-identical files. Hardening tests are in `tests/illum_robust.rs`: `ShotInfo::bv` returns `None` for settings whose value overflows; `estimate_from_histograms` checks the shape of the histograms it is given and returns `IllumError::InvalidHistograms`, and ignores a non-finite as-shot value or `Bv`; `histogram::compute` refuses a specification with fewer than two bins; hostile previews (NaN, infinities, negative and enormous values, 0 × 0 to 300 × 200 pixels) with absurd gains and exposure settings give finite, normalised output.

### Known limits

- **A wrong preset white balance.** The as-shot prior is the single fixed Gaussian (sigma 0.05), so the estimate follows a preset that has nothing to do with the light: on the A7R IV's test scenes with such a preset (11 %, an assumption) the median error is 5.9 degrees with the prior and 1.9 without it, p90 15.5 against 5.9. `training/illum/robust_prior_study.py` applies a two-component prior `(1 − eps) N(sigma 0.05) + eps N(sigma_b)` (`eps` 0.1, `sigma_b` 0.2) to the shipped filters without retraining: A7R IV median 1.23 to 1.21, p90 4.93 to 4.16, p95 8.45 to 6.33, preset scenes 5.9 to 3.1, good automatic white balance 0.85 to 0.87, worst case 22 to 33 (the histogram's own outliers are no longer capped by the prior). It is not applied: it alters the shipped fusion and the inputs of the class and confidence heads, which are trained on the single Gaussian, so it needs a retrain of the heads.
- **Brightness prior.** `Bv` here is EV at ISO 100 (sunny 16 is 15). Four of the nine real files read 14.3 to 14.6, outside the synthetic range (−2 to 13, daylight mean 9.3): the brightness prior of the training scenes is about five stops low for sunlit scenes. The head extrapolates the right way (daylight 0.89 to 0.98 with the true `Bv`, 0.52 to 0.81 without it); the prior should be redrawn at the next retraining.
- **Model file checksum.** The SHA-256 of a model file covers the tensors only (the scalars `pad_variance`, `prior_sigma`, `temperature`, the window and `default_uv` are not covered). As-shot gains whose log-chroma is far outside the histogram window are still used, with a low confidence.
- **Cube++ and NUS-8 Sony: not run.** The gate on the Cube++ test split (median ≤ 3 degrees, converted to each profile) and the report on NUS-8's Sony subset (evaluation only) need data hosts that are unreachable from the build environment. `tools/fetch/cubepp.sh` (SHA-256 pins empty until someone runs `--accept-new` with access) and `training/illum/eval_cubepp.py` (conversion to each profile through the source body's colour matrix and a least-squares XYZ to camera 3 × 3, tested on a mock layout only) are ready; the layout assumptions are unverified.

### The sensor-map gate with estimated lights

`training/illum/gate_estimated.py` (A7R IV, 200 scenes per gate light, the C2 tables): **not met, and not meetable as worded.** Estimated end to end, every one of the nine lights is over (worst median 3.042 and worst p95 19.827 against 0.5 and 1.2): the white-balance error of any estimate is a colour cast, about 1.2 to 1.9 ΔE00 per degree of median angular error, so a median of 0.5 would need the white right to about 0.3 degrees. With the white known and only the estimator's classes, mired and CCT in play ("map only"), the median changes by −0.02 to +0.15 ΔE00 (worst median 0.740, LED-V2, the C2 failure) and the p95 rises by 0.19 to 0.69, which puts all nine lights over the gate (LED-V2 by median as before, the others by the p95 limit of 1.2, which the known-light fit meets with margins of 0.03 to 0.61). A re-run with 40 scenes per light gives the same picture (end to end worst median 2.9 and worst p95 17.0; map only worst median 0.73 and p95 1.68).

This is reported, not loosened. Open question for the maintainer: keep the C2 gate as the sensor-map gate (known light), gate the estimator by the angular error alone, and state the estimated-light check as the map-only increase over the known-light numbers (for example at most +0.2 ΔE00 in the median and +0.7 in the p95; the worst measured are +0.153 and +0.693, so these thresholds describe this model and are not an independent criterion), because the white-balance cast is the estimator's error and not the map's.

## Data formats

- `data/alexa/alexa_alev3_estimate_380_780_5.json`: rawtoaces-data JSON format (same as the Sony files). Also `.csv` with min/max/std over the 24-variant ensemble, and `variants.csv`. The loader reads the `main` R, G, B arrays at 5 nm.
- `data/cameras/sony/Sony_<model>_380_780_5.json`: Weta's measured curves (Apache-2.0).
- `data/arri/*`: AMPAS licence (*ARRI CCT matrices*); `AMPAS-LICENSE.md` stays beside them and the notice is in `NOTICE`.
- Fitted per-profile tables (sensor map, CCT locus, `w_A` tables) are generated by `training/` scripts into `data/cameras/profiles/<profile-id>/profile.json` (`docs/03-camera-profiles.md`).

## Tests (ym-color)

- LogC4 encode/decode round trip within 1e-12 relative (1e-14 absolute near zero) over [−0.02, 400]. LogC3 round trip within 1e-7 absolute (the published EI 800 constants leave a 2.5e-7 step at `cut`, so values near `cut` don't round-trip more tightly). Spot values above; `LogC3 encode(0.18)` equals 400/1023 within 2e-8.
- Matrices against the constants to 1e-9; derived ARRI rows sum to 1 within 1e-12; interpolation equals table values at table CCTs.
- `w_A` and the CCT locus against the Python reference (`training/alexa/ssf_core.py`) to 1e-6.
- Sensor map: white preservation to 1e-12; the gate numbers in C2.
- CDL against ASC reference examples; the cube parser against malformed inputs; tetrahedral interpolation against OpenColorIO-generated vectors (at most 1 code value at 10 bits).
- Every WGSL kernel against its CPU reference on random inputs: max abs error ≤ 2e-4 in LogC4 units.
- **Fuzzing** (`crates/ym-color/fuzz`, three targets, run nightly by the `fuzz` job of `ci.yml` for 10 minutes each, `docs/13-testing.md`). `cube_parse` takes `.cube` text (committed seeds in `fuzz/corpus/cube_parse`): `parse_bytes` never panics or aborts on allocation, an accepted file is self-consistent, sampling it at NaN, infinite and huge inputs stays finite, and writing it out and parsing it again gives the same cube. `alf_parse` covers the ARRI Look File reader (`docs/08-looks.md`): `alf::parse_file` and `alf::parse` never panic or abort on allocation, an accepted look is a LUT of 2 to 129 points with finite entries and a valid CDL, and its container is written again as a file that reads back and is written as the same bytes (the checksum and the size field are repaired between reads, so the fuzzer reaches the tables behind them). `look_json` covers the parametric look loader and its curve spline: no panic, no NaN in the output, `bake` succeeds or returns a typed error.
- **Measured (fuzzing).** With AddressSanitizer on a 4-vCPU machine (`-rss_limit_mb=2048`): `cube_parse` one hour (3,601 s), 32,511,747 runs (9,028 per second, coverage 617, features 2,089), and `look_json` 241 s, 4,169,092 runs (17,299 per second, coverage 1,800, features 4,964); no crash, timeout or out-of-memory, resident memory 489 to 537 MB at the end.
