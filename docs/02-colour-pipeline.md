# 02 · Colour pipeline

Crates: `ym-color` (CPU reference, constants, fitting helpers), `ym-gpu` (WGSL kernels). Data: `data/alexa/`, `data/arri/`, `data/cameras/`.

## Idea in one paragraph

A Sony sensor and an ALEXA sensor see the same scene differently, and how differently depends on the light. We estimate the light in each photo, then map white-balanced Sony RGB to what the ALEXA's sensor would have recorded under that light (the **sensor map**). From there we follow ARRI's published processing: ALEXA white balance at the user's colour temperature, ARRI's colour matrix for that temperature, ARRI Wide Gamut, LogC. Looks and the display transform work on LogC4 exactly as they do for ALEXA footage. Two quantities stay separate on purpose: the **scene light** (estimated) drives the sensor map; the **user's CCT** drives the ALEXA white balance and the ARRI matrix. "3200 K under daylight" therefore looks the way it would on an ALEXA.

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
| C7 | Highlight roll-off and low-light toe | linear AWG4 → linear AWG4 | per pixel |
| C8 | LogC4 encode | linear AWG4 → LogC4 | per pixel |
| C9 | Primary grade (contrast, brightness, zones, colour balance, saturation) | LogC4 → LogC4 | per pixel |
| C10 | Look: CDL, then 3D LUT, blended by intensity | LogC4 → LogC4 (LogC3 looks via exact conversion) | per pixel + 3D texture |
| C11 | Curve | LogC4 → LogC4 | 1D textures |
| C12 | Display transform: ACES 2 (default) or an imported ARRI display LUT | LogC4 → display linear or display code values | per pixel |
| C13 | Display encode | → Display P3 (viewport) or the export colour space | per pixel |

C1–C6 are linear and fuse into **one 3×3 matrix per photo**, recomputed on the CPU when a colour setting changes:

```
M = 2^EV · M_awg3→awg4 · M_arri(CCT_user) · diag(g_user) · diag(w_A(E_scene)) · M_map(scene) · diag(g_res)
```

where every `w`, `g` is normalised so its green component is 1.

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

Check: `encode(0.18) = 0.3910068` (= 400/1023 to 1e-8). LogC3 is only used to run LogC3 looks; we always use EI 800.

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

Compute all matrices in `ym-color` from primaries in f64 (normalised primary matrix), and test them against these numbers to 1e-9. AWG3 → ACES AP0 (needed for ACES 2 and the ARRI table) uses CAT02 (von Kries in CAT02 space) from D65 to the ACES white (x 0.32168, y 0.33767). Computed in f64 it is (ARRI's IDTs print the same values rounded to 6 digits):

```
AWG3 → AP0 (CAT02)
 0.680205505  0.236136602  0.083657893
 0.085414980  1.017470879 −0.102885858
 0.002056522 −0.062562500  1.060505979
```

Rows of this matrix sum to 1 within 1e-15. Don't take AP0 from a library that stores it rounded (colour-science's rounded AP0 matrix shifts row sums by ~2e-7); derive it from the AP0 primaries.

### ARRI CCT matrices

`data/arri/Alexa_aces_matrix.txt` (ARRI via aces-dev v1.3.1, AMPAS licence) holds camera-native → AP0 matrices at 17 CCTs: 2000, 2100, 2200, 2400, 2600, 2900, 3200, 3500, 3900, 4300, 4700, 5100, 5600, 6500, 7500, 9000, 11000 K. `data/arri/alexa_native_to_awg3_17cct.json` holds the derived native → AWG3 matrices, `inv(AWG3→AP0) · M_AP0(CCT)` with the exact matrix above; every row sums to 1 within 1e-14, as balanced input requires. `ym-color` derives them itself from the `.txt` and tests against the JSON to 1e-12.

**Interpolation:** linear in mired between the **bracketing** pair; clamp CCT to [2000, 11000]. Note: ARRI's `v3_IDT_maker.py` uses `searchsorted` and then pairs index `i` with `i+1`, which extrapolates from the next pair for CCTs between table entries. We use the bracketing pair. Both agree exactly at the 17 table CCTs; test that.

**As built (M1-01).** `ym_color::{consts, gamut, logc, arri, deltae}`.

- *Constants.* `consts` holds the primaries and whites, the normalised primary matrix (`Primaries::rgb_to_xyz`), CAT02 von Kries (`cat02_adaptation`) and the LogC constants; `gamut::matrices()` returns every derived matrix, computed once in f64. Typed linear spaces (`LinearAwg3`, `LinearAwg4`, `LinearAp0`, `LinearRec709`, `LinearDisplayP3`, `LinearRec2020`, `Xyz`) and encoded spaces (`LogC4`, `LogC3Ei800`) are newtypes over `[f64; 3]` with the conversions as methods. The LogC4 constants are stored as literals and checked against their defining formulas in the tests.
- *Checks.* Every printed matrix is reproduced to 5e-11 (the 10-decimal rounding of the f64 values; AWG3 to AP0 to 5e-10); AWG3 to AP0 equals `alexa_native_to_awg3_17cct.json`'s `awg3_to_ap0_cat02` to 1e-12; rows of AWG3/AWG4 to AP0 and of every same-white gamut matrix sum to 1 within 1e-15 (largest 6.7e-16). LogC4 round trip over [−0.02, 400]: ≤ 1e-14 + 1e-12·|x|; LogC3: 5e-13 in the linear domain (the 2.5e-7 step at `cut` only shows in code values; `encode(x)` is above `decode`'s break, so linear values round trip).
- *colour-science 0.4.7.* LogC4 and LogC3 agree to 1e-14; the sRGB transfer to 1e-14; its AWG4 matrices to 1e-9. Its AWG3 matrix is stored to 6 digits and its AP0 to 10, so AWG3 to AP0 and AWG3 to AWG4 differ from ours by up to 6e-7; ours follow the derivation from primaries. OpenColorIO's built-in `ARRI_LOGC4_to_ACES2065-1` (float32, approximate exp/pow, about 2.5e-4 relative) agrees to 2.3e-5 relative and `ARRI_ALEXA-LOGC-EI800-AWG_to_ACES2065-1` to 1.9e-4: OCIO uses ARRI's LogC3 constants at full precision (0.2471896383, …), docs/02 and the VFX document print them rounded to 6 digits, which is what we implement (relative difference ≤ 1e-5 above code value 0.1, up to 7e-4 below linear 1e-3).
- *Table loading.* `ArriTable::builtin()` embeds `data/arri/Alexa_aces_matrix.txt` with `include_str!` (3 KB; the runtime needs neither the data folder nor network, and the AMPAS notice ships with the binary). `ArriTable::parse(text)` reads any file in that format. `native_to_awg3(cct)`, `native_to_awg4(cct)`, `native_to_ap0(cct)`; `bracket(cct)` returns the pair index and the mired weight. Derived matrices agree with the JSON to 5.1e-15; table rows sum to 1 within 5e-15, derived within 5.8e-15; at the 17 table CCTs the interpolation returns the table entry bit for bit.
- *ARRI's pairing, as published.* `getIDTMatrix` in `v3_IDT_maker.py` (aces-dev v1.3.1) does `i = searchsorted(ccts, cct)`; for `i` equal to the last index it returns that row unchanged, otherwise it interpolates the pair `(i, i + 1)` in 1/CCT with a weight that is negative between two entries (an extrapolation backwards from the next pair), and above the last entry it fails. So every CCT from 9000 K to 11000 K gets the 11000 K matrix. `tests/arri.rs` reimplements that behaviour: it agrees with ours at the 17 CCTs and differs between them by up to 6.2e-3 in a matrix coefficient (at 3700 K).

## C1 · Scene light and residual white balance

- The **illuminant estimate** (below) gives the scene light as camera RGB `w_scene` (green = 1) plus class probabilities.
- Raw processing balanced the mosaic with `g0 = 1 / w_scene_initial`. `g_res = w_scene_initial / w_scene` (element-wise). It is identity unless the scene light changed after open (Pick white, a light preset, Re-run).
- **Pick white:** take the 5×5 median of demosaiced camera RGB (before C1, so still balanced with `g0`) at the clicked point; reject if any channel is above 0.9 of clip ("That spot is too bright to use. Pick a grey or white area that isn't blown out."). Undo the initial gains: `w_scene = sample ⊙ w_scene_initial` (element-wise), then normalise to green = 1. Class probabilities stay as estimated.
- **Light presets** (Colour › White balance segment):

| Preset | Scene light for the sensor map | User CCT set to |
|---|---|---|
| Auto | estimate | estimated CCT |
| Day | CIE daylight 5500 K, class daylight | 5500 K |
| Shade | CIE daylight 7500 K, class daylight | 7500 K |
| Tung. | blackbody 3200 K, class tungsten | 3200 K |
| LED | estimated chromaticity, class LED = 1 | estimated CCT |
| Fluor. | estimated chromaticity, class fluorescent = 1 | estimated CCT |

**As built (M1-07), C1.** `ym_color::wb` (`SceneLight`, `EstimatorClasses`, `LightSource`, `picked_scene_white`, `residual_gains`, `median_rgb`) and `ym_color::chain`.

- *Scene light.* `SceneLight { w_scene, classes, confidence, cct_k, duv }`: the camera white (green = 1), the four profile-class probabilities, the confidence, and the CCT and Duv read from the profile's locus (`SceneLight::new`; `from_parts` takes a CCT that is already known). The estimator's six classes fold into the profile's four (`EstimatorClasses::to_profile_classes`): daylight and shade/overcast are daylight, tungsten/halogen is tungsten, fluorescent and LED stay, and the `mixed` mass goes to the class whose reference lights cover the scene CCT (tungsten below 4000 K, daylight from 4000 K, the `E_scene` rule).
- *Presets.* Day, Shade and Tungsten are model lights: the camera white is the profile locus at 5500 K, 7500 K (CIE daylight table) and 3200 K (blackbody table), the CCT is that temperature exactly, Duv 0, class probability 1, confidence 1. LED and Fluorescent keep the estimate's chromaticity, CCT and Duv and put all probability on their class. A preset always replaces the estimate; Pick white (below) replaces the estimate first, so for LED and Fluorescent the "estimated chromaticity" is the picked one while Day, Shade and Tungsten override it. The settings do not record which came last, so the front end should clear `white_pick` when a preset is chosen and set `light_source` to Auto when a white is picked.
- *Pick white.* The sample is the per-channel 5×5 median of g0-balanced camera RGB (`median_rgb`). The mosaic clips at 1.0 per channel before the gains, so channel `c` clips at `1 / w_scene_initial[c]` after `g0`; a sample with any channel above 0.9 of that is refused with the message above (`PickWhiteError::TooBright`; a sample taken where a vignetting gain was applied should be divided by that gain first). Added: a channel that is not a number or below 1/1024 of its clip level (ten stops down) is refused with "That spot is too dark to use. Pick a grey or white area that is lit." (`TooDark`), because there is no colour in the noise floor. Bright wins when both apply. The result keeps the class probabilities, takes confidence 1 and reads CCT and Duv from the locus. `g_res` is `w_scene_initial / w_scene` with both normalised to green = 1, exactly `[1, 1, 1]` when they are equal.
- *Pick white and white balance.* A pick changes the scene light, not the user's white (`docs/11` says the core sets the white balance from the pick; it does, through the scene light). `ym-core` should leave `white_balance` Auto after a pick: the Kelvin and tint rows then show the readout of the picked light and the picked sample renders neutral exactly. Moving Kelvin switches to Manual with the sensor map still at the picked scene light.

- **Open defect (found in review of M1-07): the locus is not in the raw scale.** The scene light is a camera white in the scale of the raw data (balanced with the initial gains), and the presets (`locus.rgb_at`) and the CCT and Duv readout (`Locus::rgb_to_cct_duv`) read the profile's `locus`. That table was tabulated from Weta's curves after `ssf_core.load_cameras` normalised each channel to a peak of 1 (`training/sensor_map/fit.py::sony_curves` → `tables.locus_tables`). The published curves keep the relative sensitivity of the channels (A7 III peaks: R 0.595, G 1, B 0.838), and that is what sets the raw white balance. The normalisation is harmless for the sensor map (every response is divided by the response to the white) but moves the locus by about ×1.6 in red and ×1.2 in blue. Measured on the A7 III: the camera white of a 6504 K blackbody is `[0.371, 1, 0.767]` from the published curves, `[0.386, 1, 0.762]` from the camera's own ColorMatrix (LibRaw's `rgb_xyz_matrix`), and `[0.623, 1, 0.915]` from the peak-normalised curves the locus is built from. On 9 real ARW files (`testdata/real/files`, provenance unrecorded) the as-shot gains read Duv −0.02 to −0.05 on the locus (9 of 9, tint +25 to +50, mostly pinned at the end of the control) against |Duv| ≤ 0.01 from the camera's ColorMatrix; with a locus built from the unnormalised curves they read within ±0.025 (±0.007 on the two bodies that have their own profile, A7 III and A7 IV; the others use the A7R IV stand-in). On an A7 IV, a real D55 white reads 6100 K and Duv −0.047, and the Day preset renders that white as AWG4 `[0.62, 1.04, 0.74]` instead of a neutral. The chain itself is scale-free (`g_res`, the sensor map and `w_A` act on ratios of whites), so the fix is in the data: rebuild `locus` (and `duv_scale`) from the clipped, unnormalised curves, and train and run the illuminant estimator in the same raw scale (its output `w_scene` and the as-shot prior). The acceptance test is `tests/wb.rs::the_profile_locus_is_the_camera_white_in_the_cameras_own_raw_scale` (ignored until the profiles are rebuilt; the camera white of the published curves must match the blackbody locus to 2e-3).

## C2 · Sensor map

Maps white-balanced Sony RGB to white-balanced emulated ALEV III RGB for the scene light. Exposure-invariant (linear) and white-preserving (`M·[1,1,1]ᵀ = [1,1,1]ᵀ`) by construction.

**Offline fit** (Python in `training/sensor_map/fit.py`; the earlier objective is a port of `research/alexa-ssf/scripts/sony_profiles.py::fit_map`, see step 3):

1. Inputs: the camera profile's spectral curves `S_P`, the ALEXA estimate `S_A` (`data/alexa/alexa_alev3_estimate_380_780_5.json`), the 190 rawtoaces training reflectances (Apache-2.0), and reference light SPDs.
2. Reference lights per class:
   - tungsten: blackbody 2000–4000 K every 250 K;
   - daylight: CIE daylight 4000–11000 K every 500 K;
   - fluorescent: CIE F1–F12;
   - LED: CIE LED-B1…B5, LED-BH1, LED-RGB1, LED-V1, LED-V2.

   SPDs come from the `colour-science` datasets at fit time. Only the fitted coefficients ship (`docs/14-licensing.md`).

**Gate lights (held out of the gate fit):** blackbody 2850 K, CIE daylight 5250 K and 6750 K, CIE FL3.1, FL3.8 and FL3.15, LED-B3, LED-RGB1 and LED-V2. The first three and the FL3 series fall between or outside the fit grid by construction; for the gate run, refit with LED-B3, LED-RGB1 and LED-V2 removed from the LED class. The shipped fit uses all lights. The reference numbers in `research/alexa-ssf/README.md` were measured on lights inside the fit grid, so they are a lower bound, not a target.
3. For each light `E`: Sony responses `c_s[n] = S_P·(E∘R_n)`, ALEXA responses `c_a[n] = S_A·(E∘R_n)`, each divided by its own response to the white (`R = 1`), for the 190 training reflectances `R_n`. Fit the white-preserving 3×3 that minimises the **CIE Lab error after ARRI's matrix**. Let `T_E` be the light's CCT for the ARRI lookup (the generating temperature for blackbody and CIE daylight lights, the Ohno 2013 CCT of the SPD for fluorescent and LED lights), `A(T)` ARRI's native → AP0 matrix interpolated linearly in mired between the bracketing table entries (clamped to 2000–11000 K, see C5), and `Lab(·)` the CIE Lab of an AP0 triple with the ACES white (AP0 → XYZ, white = ACES white). The parameters are the 6 free coefficients `m = (m0 … m5)` of

   `M(m) = [[m0, m1, 1−m0−m1], [m2, m3, 1−m2−m3], [m4, m5, 1−m4−m5]]`

   (rows sum to 1, so `M·[1,1,1]ᵀ = [1,1,1]ᵀ`), and the objective is

   `J(m) = Σₙ ‖ Lab(A(T_E)·M(m)·c_s[n]) − Lab(A(T_E)·c_a[n]) ‖²`   (Euclidean distance in Lab, 190 patches, no weights).

   `J` is minimised by Levenberg–Marquardt with an analytic Jacobian and a Gauss–Newton polish, started from the linear solution of the previous objective (weights `Y_target^(−2/3)`, which approximate `J` and are kept in the code as `fit_map_weighted_rgb`, the objective that reproduces `research/alexa-ssf/out/sony_profiles.csv`). Start, solver and options are fixed, so the fit is deterministic; the coefficients agree to ~1e-13 whatever the start (`python3 -m training.sensor_map.fit --stability`), and the shipped tables are checked to 5e-9 (`build_profiles --check`).

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

   Decision 2026-10-12 (Mykyta): Lab-optimal objective adopted after the M1-05 gate; LED-V2 remains a reported gate failure.

4. Store per profile: for each class, a list of `(mired, M)` sorted by mired.

**Runtime:** `M_map = Σ_k p_k · M_k(mired_scene)`, where `M_k(mired)` interpolates linearly in mired within class `k` and clamps at the ends. A convex combination of white-preserving matrices is white-preserving.

**Optional upgrade (M1, only if it wins):** a tiny MLP (≤ 5k parameters) from the illuminant estimate's features to the 6 free coefficients. Keep it only if it lowers held-out-light ΔE00 by ≥ 0.05 median against the table blend.

**Gate (M1):** 99 CIE CES samples, 9 test lights not in the fit grid. Per-light ΔE00 median ≤ 0.5, p95 ≤ 1.2, against rendering the same reflectances through `S_A` directly. Reference numbers for the A7R IV profile are in `research/alexa-ssf/README.md` (0.16–0.45 median per light).

## C3–C4 · ALEXA white

`w_A(E) = ∫ S_A(λ) E(λ) dλ`, normalised to green = 1 (5 nm sums, 380–780 nm).

- C3 multiplies by `w_A(E_scene)`. `E_scene` is the SPD of the estimated light: blackbody below 4000 K, CIE daylight from 4000 K (the rawtoaces convention, also used to fit our ALEXA curve). For the LED and fluorescent classes use the precomputed `w_A` of the class's reference lights (in `profile.json`), interpolated in mired and blended by class probability.
- C4 multiplies by `g_user = 1 / w_A(E_user)`, where `E_user` is blackbody or daylight at the user's CCT (same 4000 K rule), then applies tint.
- **Auto white balance** sets `E_user = E_scene`, so C3·C4 is the identity and the scene white lands exactly on neutral. The Kelvin shown is the scene CCT rounded to 50 K; the tint shown is the value that reproduces the scene's distance from the locus.
- **Tint:** `t ∈ [−50, 50]`. Positive is magenta. Multiply the green gain by `2^(−t/100)` (±0.5 stop at the ends). ARRI's tint mapping is unpublished; this is ours. Revisit in M1 against ARRI sample footage, if any is usable (`docs/16`).
- `w_A` for blackbody and daylight on a 10 K grid (1500–25000 K) is **precomputed offline** by the fit scripts into `profile.json` (`alexa_white`), together with the camera locus. The runtime only interpolates these tables, so no CIE datasets (observers, daylight basis) are compiled into the app. The `colorimetry` crate and colour-science are training/test dependencies only.

**As built (M1-07), C3–C4.** `ym_color::wb::{scene_alexa_white, scene_tint, tinted_white, auto_readout, manual_user_gains}`.

- *The scene's tint.* The class model of C3 (`w_A` of a blackbody or CIE daylight at the scene CCT, or the LED and fluorescent lists) has no off-locus component, yet a real photo's light sits off the locus, and the Tint row has to show how far. So `w_A(E_scene)` carries the scene's own tint: the camera white's signed log-chroma distance from the locus, `d = Duv / duv_scale(CCT)` (positive towards magenta), becomes the green gain that would cancel it, `t_scene = −100 d / (√2 ln 2)` (about 102 tint units per log-chroma unit; Duv 0.01 reads a tint of 11 at 5500 K, 16 at 3200 K, 23 at 2500 K), weighted by the daylight plus tungsten probability (the LED and fluorescent class whites already contain their reference lights' distance from the locus) and held in −50…50. `w_A(E_scene) = [g w_r, 1, g w_b]` with `[w_r, 1, w_b]` the class-blended `w_A` of the scene CCT and `g = 2^(−t_scene/100)`. A greener light gives a positive tint (the correction adds magenta), a magenta light a negative one; a preset on the locus gives exactly 0. The Duv is the camera's readout, so it is only as good as `docs/03` says (spiky spectra differ from CIE Duv; between about 3800 K and 4300 K a Duv beyond ±0.002 is not trustworthy, and so is this tint).
- *Auto.* `g_user = 1 / w_A(E_scene)` with that tinted, class-blended `w_A` (not the model at the CCT, `docs/03`), and the C3·C4 product is set to exactly `[1, 1, 1]`. The scene white lands on `2^EV [1, 1, 1]` to 6e-15 (all 11 profiles, every class mix, CCT 1900–12000 K, Duv −0.01…0.007, any initial white; `tests/chain.rs`).
- *Readout.* Kelvin is the scene CCT rounded to 50 K and held in 2000–11000 K. Tint is the value that reproduces the scene's distance from the locus: the gains of `Manual { kelvin, tint }` act on the scene white `w_s = w_A(E_scene)` as `[w_s,r / w_u,r, 2^(−t/100), w_s,b / w_u,b]` (`w_u = w_A` at that Kelvin), the three agree only when the scene sits on the locus, and the tint that comes closest on the green axis is `t = −50 log₂((w_s,r / w_u,r)(w_s,b / w_u,b))`, held in −50…50, computed for the rounded Kelvin that is shown. Consequences, all tested: (1) `Manual` at the shown Kelvin and tint renders the Auto picture times the green gain `2^(−t/100)` (the tint scales green only, so it changes brightness by at most half a stop at ±50 and by 3.5 % at ±5) to 1e-12 for daylight and tungsten scenes whose CCT is a multiple of 50 K; (2) with the Kelvin rounded, the scene's ALEXA white is off neutral by the colour of up to 25 K: at most 1.6 % per channel at 2000 to 2300 K (a kelvin is a bigger step in mired at the warm end), 1.3 % up to 4000 K (the junction included), 0.5 % above (measured every 3 K on all 11 profiles; `tests/wb.rs`); (3) for an LED or fluorescent scene a green gain cannot cancel the red/blue offset of the class white, so the shown pair leaves up to 3.4 % (A7R IV, LED, 2750–6500 K). Applied to the camera white as a green gain, the tint cancels the cast to within 4.1 % of its Duv in a round trip through the locus readout (worst of 11 profiles).
- *Manual.* Kelvin is held in 2000–11000 K and tint in −50…50 (`WhiteBalance::manual` and the chain refuse values outside); `E_user` is the blackbody table below 4000 K and the CIE daylight table from 4000 K, `g_user = [1 / w_r, 2^(−t/100), 1 / w_b]`, and the user's Kelvin picks the ARRI matrix. **The 4000 K rule makes `w_A` step** (red −1.2 %, blue −3.4 % going up, `docs/03`); a Kelvin slider dragged across 4000 K moves the scene white by 1.1 % red, −0.4 % green and 4.4 % blue (A7R IV, daylight scene, about 0.06 stop in blue). The rawtoaces convention is kept as specified; cross-fading the tables over 3500–4500 K would make `w_A` a mixture of two lights that are not on the same locus and was not adopted.
- *Direction of a Kelvin setting.* The user's Kelvin is the light the ALEXA is balanced for, as on the camera: a daylight scene set to 3200 K renders **bluer** than Auto, not warmer, and a tungsten scene set to 5500 K renders warmer. For the A7R IV, CIE D 5500 K, 3200 K: AWG4 white `[0.6747, 0.9488, 1.6434]` (blue over red 1.28 stops, green over red 0.49 stops; Auto is `[1, 1, 1]`), equal to the independent spectral simulation of ARRI's balance (ALEXA estimate curves, Planck's law or the CIE D lights of the exporter's fixtures, the 17-CCT matrices) to 3e-9 for a white patch (312 cases: 3 bodies, 13 lights of all four classes, Auto and 8 Kelvin settings; `tests/chain_spectral.rs`). The 24 ColorChecker patches through the sensor map differ from the simulation by the map's error, 0.47 ΔE00 median at worst and 1.4 at most (3 bodies, Auto, 2800/3200/5500 K). docs/15 M1-07 says "warmer"; the maths says cooler (see STATUS, questions).

### Camera RGB ↔ CCT and Duv

For a profile, the fit scripts tabulate camera RGB (green = 1) of blackbody 1500–4000 K and daylight 4000–25000 K every 10 K, as log-chroma `(ln R/G, ln B/G)`, plus the matching CIE xy and the Duv scale, into `profile.json` (`locus`). Convert `w_scene` to CCT by nearest point on that polyline, with linear interpolation between neighbours. Duv is the signed perpendicular distance, positive towards magenta, rescaled to CIE Duv through the profile's XYZ estimate. Use this for the Kelvin and tint readouts and the light presets.

## C5–C6 · ARRI matrix, AWG4, exposure

- `M_arri(CCT_user)` from the interpolated table above.
- `M_awg3→awg4` from the constants.
- Exposure: `EV = EV_auto + 0` in Auto, or the slider's absolute value in Manual. The slider range is −4…+4 EV absolute; its detent sits at `EV_auto`.

**As built (M1-07), C5–C6.** `ym_color::chain::{FusedChain, ChainSettings, ProfileTables, ChainStages, ClipLevel, ColorEstimates}`.

- `FusedChain::build(tables, &estimate, w_scene_initial, &settings)` returns the 3×3 (`matrix()`), the factors (`stages()`: `g_res`, `sensor_map`, `w_a_scene`, `g_user`, `c3_c4`, `user_cct_k`, ARRI and AWG3→AWG4 matrices, `2^EV`), the resolved scene light, the Kelvin and tint to show (`readout()`: the Auto readout, or the user's values for Manual) and the clip level. `ProfileTables` is the three tables of a profile the chain reads (`profile.sensor_map.classes`, `profile.alexa_white`, `profile.locus`), so `ym-color` does not depend on `ym-profiles`. Order of the scene light: estimate, then Pick white (`ChainSettings::picked_white`, the sample), then the preset. The CCT for the ARRI matrix is the scene CCT in Auto and the user's Kelvin in Manual; `ArriTable` clamps it to 2000–11000 K. The matrix is the product of the formula above, checked stage by stage on random pixels to 1e-13. The exposure must give a usable gain: `FusedChain::build` refuses an EV that is not finite or whose `2^EV` overflows or underflows (`ChainError::InvalidExposure`, beyond about ±1000 stops), and `ColorEstimates::chain` refuses a manual exposure outside −4…+4 (`ExposureSetting::manual`/`validate`, `ChainError::Exposure`); the automatic value is not range-checked (the highlight guard can take it to −4).
- `ColorEstimates { scene, w_scene_initial, ev_auto }` is the colour part of `AutoEstimates`; `.chain(...)` resolves `ExposureSetting::{Auto, Manual { ev }}` against `ev_auto` and `.metering_chain(...)` builds the all-Auto, EV 0 chain that auto exposure meters through (metering happens once at open and on Re-run, so later white-balance edits do not move the exposure).
- *Clip level for C7.* A neutral surface brightens until the first camera channel clips. After C1 it is `[s, s, s]` and channel `c` clips at `s = 1 / w_scene[c]`, so the first clip is at `balanced = min_c 1 / w_scene[c]` (1 for daylight on a Sony sensor, below 1 for a light so red that red clips first). Mapped through the matrix this is `M·[balanced; 3]`, and because C7 acts on `n = max(R, G, B)`, `c = max` of that triple (`FusedChain::clip().n`; `clip_level(&matrix, w_scene)` for any matrix). In Auto it is `2^EV · balanced`; with a manual white balance the clipped neutral is coloured and `c` is its largest channel.

### Auto exposure

Goal: a pleasing exposure close to what a careful photographer would set, stable across a burst.

1. Run C1–C6 with EV = 0 on the 1/8-resolution preview. Compute linear luminance `Y` with the AWG4 Y row.
2. Ignore pixels in the clip mask and pixels with `Y` below the 1st or above the 99th percentile.
3. `m = exp(Σ wᵢ ln Yᵢ / Σ wᵢ)`, with a centre weight `wᵢ = exp(−r²/(2·0.35²))` (r = distance from centre over the half-diagonal).
4. `EV_auto = log2(0.18 / m) + 0.5 · EC`, where EC is the camera's exposure compensation from EXIF (the photographer's intent, half weight). Clamp to [−3, +3].
5. Highlight guard: if more than 0.5% of non-clipped pixels would exceed linear 8.0 after exposure, lower `EV_auto` until they don't, but by no more than 1 EV.
6. Files of the same camera within 2 s of each other and with identical exposure settings share the median `EV_auto` (bursts stay consistent).

The constants above are starting values; tune them in M1 on the A1 test set. Keep the algorithm, change the numbers.

**As built (M1-08).** `ym_color::exposure`: pure functions over plain buffers (`PreviewRgb`: interleaved f32 RGB, the 1/8-resolution preview balanced with `g0`, and an optional clip mask) and small metadata structs; no decoder or GPU dependency. `auto_exposure(&preview, &matrix_ev0, exposure_compensation_ev)` is steps 1 to 5 (1188×792 pixels, the 1/8 preview of a 61 MP frame: about 0.2 s on a loaded machine), `share_burst_ev(&frames, &ev_auto)` step 6. Where the steps are silent:

- *Luminance.* `Y` is the AWG4 `Y` row applied to `M·rgb` (folded into one row vector), raised to a floor of 1e-4 before the logarithm so that black and negative pixels (noise after black subtraction) count as very dark rather than missing and a lens-cap frame meters to +3 EV. A pixel with a non-finite channel counts as masked.
- *Percentiles* are the linearly interpolated quantiles (numpy's default) of the sorted unmasked luminances; a pixel is kept when `q01 ≤ Y ≤ q99`. The mask is applied first. With a few percent of outliers in one tail the quantile falls inside that tail and nothing is dropped, which is what the spec's "ignore the 1st and 99th percentile" gives. The interpolation is written `a + (b − a)·f`, which returns exactly `a` for equal neighbours: the form `(1 − f)·a + f·b` is an ulp off for some, and on a flat or black frame the 1st percentile then lands above every pixel and the band is empty. If the band is empty anyway (two distinct values and nothing else) every unmasked pixel is metered.
- *Centre weight.* `r` is measured from the pixel centre to the image centre over the half diagonal (`w = 1` at the centre, 0.017 in a corner).
- *Highlight guard.* Over every unmasked pixel (not only those kept by the percentiles, which would remove the highlights it looks for), `Y` after exposure against 8.0. Solved in closed form: the largest EV at which at most `floor(0.005 n)` pixels exceed 8 (from the `(floor(0.005 n) + 1)`-th brightest `Y`), and `EV_auto = max(that, EV_metered − 1)` when it is below `EV_metered`. The result can leave −3…+3 by up to a stop (−4 at the lowest); it never raises the exposure. `AutoExposure` reports `ev_metered`, `guard_reduction_ev`, `mean_luminance` and the pixel counts.
- *No usable pixel* (everything masked, or an empty preview): `EV_auto = clamp(0.5·EC)`, `metered = false`.
- *Bursts.* A frame joins a group when the previous frame (in time order) with the same camera string and identical settings (shutter, f-number, ISO, exposure compensation; equal to 1e-9 relative) is at most 2 s earlier. Those are consecutive gaps, so a burst of any length is one group; a time-lapse at fixed settings with gaps under 2 s would also be one group. The median of an even group is the mean of the two middle values; a frame without a capture time is alone. Step 6 follows steps 4 and 5, so a frame whose highlight guard lowered its own EV takes the group median, which can undo the guard for that frame; the spec orders it this way.
- *The gate* ("within ±0.5 EV of the embedded JPEG, median luminance in linear light, ≥ 80 % of the golden set"): `preview_median_luminance(&preview, &matrix_ev0, ev)` is the median `Y` of our render over every pixel (masked ones included, as the JPEG's median includes them) at an exposure; `jpeg_median_luminance(&rgb8)` the median `Y` of a decoded 8-bit sRGB image (sRGB decoded to linear, Rec.709 luminance; an Adobe RGB preview would be a little off); `exposure_error_ev(ours, reference) = log2(ours / reference)`; `evaluate_gate(&entries, 0.5, 0.8)` gives the pass rate, the outliers worst first (for the M1-H5 review) and the entries it could not use. Decoding the embedded JPEG and reading the files is the caller's job (the tests decode with `zune-jpeg`, a dev-dependency, MIT OR Apache-2.0 OR Zlib).
- **Gate status: blocked by the missing corpus.** The environment has no ARW files with embedded previews. The harness is tested on synthetic scenes with planted camera offsets (it recovers them to 0.03 EV and classifies them), the algorithm against a plain reimplementation, and the JPEG side on two small JPEG files; the 80 % figure is not measured. The constants are the starting values; tuning them on the A1 set is future work (note that the median of a tone-mapped JPEG is not the geometric mean of scene-linear luminance, so a systematic offset may need the target 0.18 changed, not the algorithm).

## C7 · Highlight roll-off and low-light toe

Both act on a norm `n = max(R, G, B)` and scale RGB by `f(n)/n`, which preserves the ratios.

- **Highlights** (`rolloff.highlights = r ∈ [0,1]`, default 0.5). Let `c` be the clip level after C1–C6 (the smallest per-channel clip, mapped to AWG4 and exposure). Knee `k = c · 2^(−(0.5 + 2r))`. For `n ≤ k`: `f(n) = n`. Above: `f(n) = k + (c' − k) · tanh((n − k)/(c' − k))`, with `c' = 1.5c`. At 0 the shoulder starts 0.5 stop below clip and is short; at 1 it starts 2.5 stops below and is long. ALEV III has about 7.4 stops above grey at EI 800; a Sony body clips earlier at matched grey, so this shoulder stands in for the missing latitude.
- **Low light** (`rolloff.low_light = r`, default 0.5). `τ = 4r · σ_floor`, where `σ_floor` is the noise model's read noise mapped to linear AWG4 at this exposure. `f(n) = n³ / (n² + τ²)`, which is monotonic, ~n above 3τ and crushes the noise floor smoothly below τ. `r = 0` disables it.

**As built (M1-09).** `ym_color::tone::ToneParams::new(highlights, low_light, clip_level, noise_floor)`, `apply(LinearAwg4)`. The two curves compose as `f(n) = toe(shoulder(n))`; `scale(n) = f(n) / n` multiplies all three channels. For `n <= 0` the shoulder is skipped and the toe's scale `n² / (n² + τ²)` applies (a pixel with `n = 0` and `τ > 0` goes to black); non-finite input passes through. `r_low_light = 0` is the exact identity; the highlight shoulder cannot be switched off (at 0 it starts 0.5 stop below clip). Tests: the formulas, value and slope continuity at the knee (slope 1 on both sides), `f` non-decreasing in `n` over a grid of 11 × 11 amounts × 5 clip levels × 5 noise floors, non-negative derivative, each output channel non-decreasing in its own input (proptest), and `f` non-increasing in either amount.

## C8 · LogC4 encode

Per channel with the formula above. Negative linear values (out-of-gamut LED colours) pass through the linear segment unclipped. Nothing clips before the DRT.

## C9 · Primary grade (LogC4)

Let `v` be LogC4 RGB. `L = encode(Y_lin)` uses the luminance of the linear values before C8. Grey `g = 0.2784`. One stop in LogC4 is `b/14 = 0.06480` code values; call it `S`.

| Control | Range | Operation |
|---|---|---|
| Contrast `c` | −100…100 | `v = g + (v − g) · 2^(0.6·c/100)` |
| Brightness `b` | −100…100 | `v += (b/100) · S · bump(L; g, 2.5S)`: a midtone lift that leaves black and white |
| Shadows `s`, Highlights `h`, Black `k`, White `w` | −100…100 | `v += (x/100) · 1.5S · bump(L; centre, 1.5S)` with centres at grey −3, +3, −6, +6 stops |
| Colour balance (shadows, midtones, highlights) | `(a, b)` in the unit disc | `v += 0.03 · (a·e₁ + b·e₂) · zone(L)`, `e₁ = (1, −½, −½)/√1.5`, `e₂ = (0, 1, −1)/√2`; zones centred at −3, 0, +3 stops |
| Saturation `s` | −100…100 | `v = L + (v − L) · (1 + s/100)` |

`bump(L; μ, σ) = exp(−(L − μ)²/(2σ²))`. All constants are starting values; tune them visually in M3. Apply in the order of the table.

**As built (M1-09), decisions where the table is silent.** `ym_color::grade::{GradeParams, Grade}`; `Grade::apply(LogC4, L)` and `Grade::apply_linear(LinearAwg4)` (C8 then C9, `L` from the linear values).

- *Stops are scene-referred.* "Grey ±n stops" is the LogC4 code of `0.18·2^±n` (`grade::logc4_at_stops`), not `g ± n·S`: LogC4 is linear in stops only far above the toe, and `g − 6S` would sit below the code value of black. `S` is used for widths and amplitudes only.
- *Zones.* `zone(L) = bump(L; centre, 1.5 S)` at −3, 0, +3 stops (Gaussian bumps, not a partition of unity: colour balance leaves the extremes alone). Shadows, Highlights, Black and White add `x/100 · 1.5 S · bump(L; centre, 1.5 S)` to all three channels; positive is brighter for all four. Contrast is skipped when 0 (the grade is then bit-exactly the identity).
- *Ranges.* Controls clamp to −100…100; a colour-balance vector longer than 1 scales back to the unit circle.
- *"Monotonic".* Per pixel in the control value: Brightness, Shadows, Highlights, Black and White raise every channel (non-decreasing); Contrast moves a channel away from `g` (up above, down below); Saturation scales `|v − L|` up; Colour balance displaces along a fixed direction by `0.03·r·zone`, so the length grows with the radius. As a tone map on a neutral ramp, one control over its full range keeps the slope above 0.39 (brightness ≥ 0.757, zone controls ≥ 0.393, contrast ≥ 0.66, analytically). **Not guaranteed:** several tonal controls stacked at ±100 can fold the ramp (all four zone controls and Brightness at −100 with Contrast −100: worst slope −0.15 inside the valid code range, −0.67 below black; at +100: −0.24 with Contrast −100). The UI should limit the stack or the grade should be re-fitted (reported in STATUS).
- *Saturation pivot.* `v = L + (v − L)(1 + s/100)` pivots on the *pre-grade* `L` as specified, so with Contrast or Brightness active it also scales the tonal change of neutral pixels by `1 + s/100`. Pivot on the graded luma if that is not wanted.

## C10 · Looks

A look is `CDL → LUT`, applied in its own encoding, then blended:

```
in  = v (LogC4)
out = look(in)
v'  = in + k · (out − in)        k = intensity / 100, blended in LogC4
```

- **CDL** (ASC v1.2): `x = max(x·slope + offset, 0)^power` per channel, then saturation with Rec.709 weights (0.2126, 0.7152, 0.0722): `L = w·x; x = L + sat·(x − L)`.
- **3D LUT:** `.cube` (Resolve/Adobe format: `LUT_3D_SIZE`, optional `DOMAIN_MIN/MAX`, optional `LUT_1D_SIZE` shaper, red varies fastest). Tetrahedral interpolation. Inputs are clamped to the domain. Upload as a 3D texture (rgba16float); interpolate manually in the shader (tetrahedral), never with hardware trilinear.
- **LogC4 log-to-log looks** (ARRI LogC4 Look Library, our built-ins): apply directly.
- **LogC3 log-to-log looks:** LogC4 → linear AWG4 → AWG3 → LogC3 EI 800 → LUT → linear AWG3 → AWG4 → LogC4. The blend happens in LogC4 after converting back.
- **LogC3-to-Rec.709 looks** (look and rendering baked together) **replace** C12: the chain is C9 → C11 (the curve runs before the look in this case) → LogC4 to LogC3 conversion → LUT → display Rec.709 (gamma 2.4). Intensity blends between our C12 output (from the same C11 output) and the LUT output in display-linear Rec.709. The UI marks these looks "includes display rendering". DNG export is unavailable with these looks (`docs/09`).
- **Built-in looks:** Neutral is the identity. Signature is parametric (CDL plus hue/saturation curves in LogC4), designed by Mykyta in M3 and stored as JSON in `data/looks/builtin/`; it is baked into a 33³ LUT at load.

**As built (M1-10).** `ym_color::{cdl, cube, lut, look}`.

- *CDL.* `x = max(in·slope + offset, 0)^power`, then `L = 0.2126R + 0.7152G + 0.0722B; out = L + sat·(x − L)`. No clamp at 1 (the CDL runs on log codes and look output); OpenColorIO's `CDL_ASC` also clamps at 1 before and after saturation and agrees wherever every intermediate stays in [0, 1]. `Cdl::new` checks the ASC ranges (slope ≥ 0, power > 0, saturation ≥ 0).
- *.cube dialects.* `cube::parse` reads the Adobe/IRIDAS dialect (`TITLE`, `LUT_3D_SIZE` or `LUT_1D_SIZE`, `DOMAIN_MIN/MAX`) and the Resolve dialect (`LUT_1D_INPUT_RANGE`, `LUT_3D_INPUT_RANGE`, a 1D table followed by a 3D table in one file, the 1D table being a shaper). Sizes: 3D 2 to 129, 1D 2 to 65536; at most 256 MiB; comments, blank lines, LF, CRLF, CR, a BOM and case-insensitive keywords are accepted; every number must be finite; every failure is a typed `CubeError` with the line number. Domains: for a lone table, `DOMAIN_*` or the matching `LUT_*_INPUT_RANGE` (they must agree if both are given); with both tables `DOMAIN_*` and `LUT_1D_INPUT_RANGE` belong to the shaper and `LUT_3D_INPUT_RANGE` to the cube. A 1D-only file parses but `parse_lut` refuses it (a look needs a 3D LUT). OpenColorIO reads the same files the same way (`iridas_cube`, `resolve_cube`).
- *Tetrahedral.* `Lut3d::sample` uses the six-tetrahedra split of the fractional position (the same as OpenColorIO's), clamps to the domain (a NaN input counts as the domain minimum), is exact at grid nodes and continuous across cell faces. Against OpenColorIO 2.6 tetrahedral on our own synthetic cubes (`tools/vectors/lut_vectors.py`: identity 9, warp 17, ripple 33, an asymmetric-domain 9, a 2-point extreme cube, a 64-entry shaper plus 9-point cube; 3714 points each including grid nodes, cell diagonals and out-of-domain inputs) the largest difference is 2.1e-7 = 0.0002 code values at 10 bits (OpenColorIO's float32); trilinear would be 1.75e-3 (1.8 code values) on the warp cube, so the gate pins the method.
- *Looks.* `look::LookTransform::{neutral, new, eval_logc4, apply_logc4(v, intensity_pct), eval_display_rec709}`, `blend_logc4`, `blend_display_linear(own, look, intensity_pct)` (both in the same linear primaries; convert the look output with `LinearRec709::to_display_p3` / `to_rec2020` first), `rec709_gamma24_decode` (negative LUT output clamps to 0). 0 % returns the input and 100 % the look output exactly. CDL and LUT run in the look's own encoding (a LogC3 look applies its CDL to LogC3 codes). Inputs above the LUT domain clamp (LogC3 code value 1.0 is linear 55, so LogC4 highlights above that hold the LUT's top value in a LogC3 look).
- *Parametric looks.* `look::ParametricLook::from_json` reads the schema documented at the top of `crates/ym-color/src/look.rs` (`yellowmoon.parametric-look/1`: `cdl`, `hue_vs_sat`, `hue_vs_hue`, `luma_vs_sat`, all optional, evaluated in LogC4, chroma and hue measured along the C9 axes `e1`, `e2`, Rec.709 luma preserved) and `bake(33)` samples it on a 33³ cube. The example in `tests/data/looks/` bakes with a median error of 1.1e-4 LogC4 code values and a maximum of 7.7e-3 (the most saturated reds next to the blue floor; 3.7e-3 where all channels are above black).

## C11 · Curve

Applied in LogC4 after the look and before the display transform, so it travels into DNG export and behaves the same under every display transform. Four monotone cubic splines (Fritsch–Carlson) on [0,1]: luma (applied as `v += curve(L) − L`) and R, G, B (per channel). Baked into four 1024-entry 1D textures. The histogram in the Curve tab is of LogC4 values. Numbers in the UI show code values × 255, rounded.

**As built (M1-09).** `ym_color::curve::{Curve, CurveSet, BakedCurves}`. Fritsch–Carlson tangents (mean of the neighbouring secants, zero at extrema and flat segments, one-sided at the ends, then the `α² + β² ≤ 9` limiter); 2 to 16 control points, `x` strictly increasing, all in [0, 1]; invalid input is a typed `CurveError`. The spline is C1 and never overshoots the data of a segment. Between 0 and the first point and between the last point and 1 it holds the end value. The 1024-entry table (`CurveTable`, sampled at `i/1023`, linear interpolation; worst deviation from the exact spline 5.9e-7) continues with slope 1 outside [0, 1] (`eval_extended`: `x + table[0]` below 0, `x − 1 + table[1023]` above 1), so the map stays monotone for LogC4 values beyond the unit range; a shader sampling a 1D texture needs this extension explicitly. The luma curve uses `L = encode(Y(decode(v)))` of the pixel entering C11 and is applied before the R, G, B curves.

## C12 · Display transform

### ACES 2 (default)

Port the ACES 2.0 output transform (aces-output, Apache-2.0) to Rust (CPU reference) and WGSL.

1. LogC4 → linear AWG4 → AP0 (AWG4 → XYZ, CAT02 D65 → ACES white, XYZ → AP0; compute the matrix in `ym-color`).
2. ACES 2 output transform with: peak luminance 100 nits, SDR, white D65, and limiting primaries:

| Output | Limiting primaries | Encoding |
|---|---|---|
| Viewport | P3-D65 | Display P3 (sRGB transfer) |
| sRGB export | Rec.709 | sRGB transfer |
| Display P3 export | P3-D65 | sRGB transfer |
| Rec.2020 export | P3-D65 | Rec.2020 primaries, BT.1886 (gamma 2.4) |

3. Gamut compression is part of ACES 2; out-of-gamut values from C8 are handled there.

Reference values: generate test vectors offline with OpenColorIO ≥ 2.4 (BSD-3), which ships ACES 2 built-in transforms. Commit the vectors (a few hundred RGB triplets per output) under `crates/ym-color/tests/data/`. Gate: ≤ 1 code value at 10 bits against the vectors.

**CPU reference (M1-11), as built.** `ym_color::aces2` (`Aces2Output::{Viewport, Srgb, DisplayP3, Rec2020}`, `Aces2Transform::new(output)`, `transform_ap0`, `transform_ap0_linear`; attribution and commit hashes at the top of the module).

- *Source.* The transform's maths lives in the ACES project's **aces-core** repository (`lib/Lib.Academy.*.ctl`); aces-output only holds the per-display presets that call it. The port follows the CTL and was cross-checked against OpenColorIO's C++ implementation.
- *Vectors.* `tools/vectors/aces2_vectors.py` writes `crates/ym-color/tests/data/aces2/{viewport,srgb,display_p3,rec2020}.json` (898 AP0 inputs per output: neutrals from −8 to +12 stops, gamut primaries and secondaries at five exposures, a chromaticity wheel, AP0 primaries, edge cases, seeded random in-gamut, out-of-gamut, near-black and very bright values) with OpenColorIO 2.6.0's float32 output. Every output is the built-in `ACES-OUTPUT - ACES2065-1_to_CIE-XYZ-D65 - SDR-100nit-<REC709 or P3-D65>_2.0` followed by a `DISPLAY - CIE-XYZ-D65_to_<sRGB, DisplayP3 or REC.1886-REC.2020>` built-in. The studio config has display/view pairs for sRGB (`sRGB - Display` / `ACES 2.0 - SDR 100 nits (Rec.709)`) and Display P3 (`Display P3 - Display` / `ACES 2.0 - SDR 100 nits (P3 D65)`); the script checks they equal the built-in chain exactly. It has no Rec.1886 Rec.2020 display, so the Rec.2020 export is the P3-limited built-in plus the Rec.1886 Rec.2020 display built-in, which is how the config builds its own pairs.
- *Numerics.* f64 with the reference's iteration tolerances unchanged (the CTL and OpenColorIO run in float32). The tables agree with OpenColorIO's float32 tables to 1e-5 relative. Gate result: at most 0.03 code values at 10 bits over the vectors and 0.05 over random sweeps of 1.9 million samples per output.
- *Reference glitch.* In float32 the reference returns NaN (black) in a thin band of scene luminance near 128 × white with a little chroma, where the tone-scaled lightness reaches the display maximum. The port guards that case and returns white; an f32 WGSL port needs the same guards (see the module notes).
- *Rec.2020 overshoot.* P3 red lies just outside Rec.2020, so saturated cyan converts to a blue up to 1.00121 in linear light (1.0005 encoded, half a 10-bit code value). The reference leaves it unclamped; `transform_ap0` clamps to [0, 1], `transform_ap0_unclamped` keeps the reference value.

### Imported ARRI display LUT

If the user chose "ARRI's display rendering" in the wizard or preferences, C12 is the imported display `.cube` (for example LogC4 → Rec.709 gamma 2.4, or → P3). LogC4 display LUTs apply to the LogC4 output of C11 directly; LogC3 display LUTs (the older package, including the classic Rec.709) apply after converting LogC4 → linear AWG4 → AWG3 → LogC3 EI 800. For the viewport, decode the LUT's output transfer (gamma 2.4 for Rec.709) to display-linear, convert primaries to P3, and encode with the sRGB transfer. For export, convert to the chosen colour space the same way.

## C13 · Display encode

sRGB transfer for sRGB and Display P3, BT.1886 inverse (gamma 2.4) for Rec.2020. Dither 8-bit outputs with a fixed blue-noise texture (seeded, deterministic). Embed the matching ICC profile on export (`docs/09-export.md`).

**As built (M1-10).** `ym_color::encode`. sRGB (IEC 61966-2-1, thresholds 0.0031308 and 0.04045) for sRGB and Display P3, the pure `x^(1/2.4)` for Rec.2020; both odd-symmetric below zero, clamped to [0, 1] before quantising (`OutputSpace::encode_clamped`). `quantize(v, bits)` rounds to nearest (8, 10 or 16 bits; a NaN is 0). *Dither (8-bit only):* a 64 × 64 void-and-cluster rank texture (Gaussian σ = 1.5 on a torus, seed `0x59454c4c4f574d4f`, integer weights and lowest-index tie-breaks; generated by `encode::blue_noise::generate`, embedded as a table so no platform's `exp` can change it, fingerprint `0x993fccd5c6db3aa9` checked in the tests). The threshold at absolute output pixel `(x, y)` is `(rank[y mod 64][x mod 64] + 0.5)/4096`; the code is `floor(v·255 + threshold)`; all three channels share the threshold (achromatic noise). Over one tile the thresholds are the 4096 evenly spaced values in (0, 1), so a constant input averages to its exact value (measured error 0 over 1021 levels) and the result is never more than one code from `v·255`. Spectrum: power at radial frequencies 1 to 5 is 1e-4 of the power at 16 to 33 (white noise would be 1). A change to the table changes every dithered export (bump `pipeline_version`).

**As built (C7 to C13 chain).** `ym_color::display` (`ColorChain`, `ChainParams`, `DisplayTransform`, `OutputTarget`, `Frame`): the CPU reference of everything per pixel after the fused C1–C6 matrix, which the CPU fallback runs and every GPU kernel is parity-tested against.

- *Order.* `M · x` (supplied as a 3×3) → C7 → C8 → C9 (with `L` from the linear values entering C8) → C10 → C11 → C12 → C13. A LogC3-to-Rec.709 look is classified when the chain is built: C10 is skipped, the curve runs first, and C12 evaluates both our display transform and the look's LUT from the same C11 output and blends them in display-linear (`blend_display_linear`, in the output's primaries, so the look's Rec.709 is converted first). At intensity 0 the result is our C12 bit for bit, at 100 it is the look's. Neutral, an identity look and an intensity of 0 skip C10 and are exact identities. `ColorChain::trace(rgb)` returns every stage's value for debugging kernel parity stage by stage.
- *Entry points.* `apply_pixel` (code values in [0, 1], f64), `apply_to_display_linear` (C1–C12, the picture an exporter resizes in linear light before `encode`, `encode_u8` or `encode_u16`), `apply_logc4_pixel` and `apply_dng_pixel` (stop after C11, decode to linear AWG4; `DngUnavailable` with a LogC3-to-Rec.709 look), `apply_before_pixel`, and band-parallel buffer renders over planar or interleaved f32 (`render_encoded_f32`, `render_display_linear_f32`, `render_dng_linear_awg4_f32`, `render_u8`, `render_u16`, `encode_display_linear_u8/u16`). The planar layout is `ym_gpu::cpu::RgbImage`'s. Threads are `std::thread::scope` workers over row bands (no `rayon` in `ym-color`'s manifest); output is bit-identical for any thread count, and 8-bit dither uses absolute pixel coordinates (`origin`), so tiles agree with the whole image. NaN camera samples count as 0, infinities are limited to ±1e30.
- *Imported ARRI display LUT.* `ArriDisplayLut { lut: Arc<Lut>, input: LogC4 | LogC3Ei800, output: DisplayLutOutput }`, as C12 above: LogC4 input directly, LogC3 after the exact conversion; the output codes are decoded with the LUT's transfer, converted by a 3×3 to the output's primaries (a colour outside the output gamut is clipped by C13, not gamut-mapped) and encoded by C13. `DisplayLutOutput::from_target("rec709-g24")` reads the importer's target slug: primaries from `709`/`srgb`, `p3`, `2020`; transfer from `gNN`/`gammaNN`/`srgb`. Decisions: with no gamma in the slug Rec.709 and Rec.2020 are gamma 2.4 and P3-D65 is gamma 2.6; HLG, PQ and DCI-white targets are refused (`UnsupportedTarget`). Verify the real package names when they are fingerprinted (M3-H2).
- *C13 transfer.* The chain encodes display-linear with `encode` (IEC 61966-2-1, break at 0.0031308) as specified. The ACES 2 port's own `transform_ap0` ends with the reference's continuous "moncurve" (slope 12.9232, break at 0.0030399); the two differ by at most 9.8e-6 near linear 0.00304 (0.0025 of an 8-bit code, 0.64 of a 16-bit code). The Rec.2020 output (a pure power law) is unaffected. A WGSL port follows C13 (IEC).
- *Before view.* `docs/11` says the photo shows "the LogC4 signal before look and display transform". Read as the camera log: the output of C8, before the grade, look, curve and display (`BeforePoint::CameraLog`, the default; `AfterGrade` stops after C9). Each channel is clamped to [0, 1] and used as a display code value (no transfer, no primaries conversion), so it looks flat and desaturated like unmanaged log footage.
- *Speed (`tests/display_perf.rs`, one 4-vCPU machine shared with other builds, load average 22–30, so wall times are inflated; CPU time per pixel is the stable figure).* Full chain with ACES 2, camera RGB to dithered 8-bit: 1.05 µs of CPU per pixel with zero grade, Neutral look and identity curves, 1.27 µs with grade, a 33³ look and curves; the DNG stop (C1–C11) is 0.48 µs. Projected on four free cores: about 3.6 and 2.9 Mpx/s, a 24 MP frame in 7 to 8 s. Preparing a chain takes 17 to 40 ms the first time (ACES 2 tables), then under 1 ms.

## Illuminant estimate

Input: the 1/8-resolution preview of black-subtracted camera RGB **before** any white balance, with clipped pixels masked, plus metadata.

Output: `w_scene` (camera RGB of the light, green = 1); class probabilities over {daylight, shade/overcast, tungsten/halogen, fluorescent, LED, mixed}; a confidence value.

**Baseline (M1):**

1. Log-chroma histogram `u = ln(G/R)`, `v = ln(G/B)`, 64×64 bins, weighted by `√Y`, as in FFCC (Barron & Tsai 2017, Apache-2.0 reference code).
2. FFCC-style filter, trained per camera profile family, gives the chromaticity. Train on (a) synthetic scenes: random rawtoaces and CES reflectances under the reference lights, rendered through `S_P` with noise, and (b) Cube++ (CC BY 4.0) converted to each profile through its own camera matrices.
3. A small classifier (logistic regression or a 2-layer MLP) on histogram moments, the FFCC output, scene brightness `Bv = log2(N²/t) − log2(ISO/100)`, and the camera's as-shot white balance (tag 0x7313) predicts the class.
4. Prior: the camera's as-shot white balance, converted to a chromaticity, enters as a Gaussian prior on `(u, v)` with σ = 0.05. Sony's own auto white balance is good; we refine it.

Which estimator wins (FFCC, C5 or DMCC style) is decided in M1 by the gate (no test shoots in V1): median angular error ≤ 2° on held-out synthetic scenes and ≤ 3° on the Cube++ test split converted to each profile, reported (not gated) on NUS-8's Sony A57 subset (evaluation only), and the sensor-map gate still met with estimated lights.

## Data formats

- `data/alexa/alexa_alev3_estimate_380_780_5.json`: rawtoaces-data JSON format (same as the Sony files). Also `.csv` with min/max/std over the 24-variant ensemble, and `variants.csv`. Load the `main` R, G, B arrays at 5 nm.
- `data/cameras/sony/Sony_<model>_380_780_5.json`: Weta's measured curves (Apache-2.0).
- `data/arri/*`: see above, AMPAS licence; keep `AMPAS-LICENSE.md` beside them and the notice in `NOTICE`.
- Fitted per-profile tables (sensor map, CCT locus, `w_A` tables) are generated by `training/` scripts into `data/cameras/profiles/<profile-id>/profile.json` (`docs/03`).

## Tests (ym-color)

- LogC4 encode/decode round trip within 1e-12 relative (1e-14 absolute near zero) over [−0.02, 400]. LogC3 round trip within 1e-7 absolute (the published EI 800 constants leave a 2.5e-7 step at `cut`, so values near `cut` don't round-trip more tightly). Spot values above; `LogC3 encode(0.18)` equals 400/1023 within 2e-8.
- Matrices against the constants to 1e-9; derived ARRI rows sum to 1 within 1e-12; interpolation equals table values at table CCTs.
- `w_A` and the CCT locus against the Python reference (`research/alexa-ssf/scripts/ssf_core.py`) to 1e-6.
- Sensor map: white preservation to 1e-12; gate numbers above.
- CDL against ASC reference examples; cube parser against malformed inputs (fuzz with `cargo fuzz` for one hour in M1).
- Tetrahedral interpolation against OCIO-generated vectors: ≤ 1 code value at 10 bits.
- Every WGSL kernel against its CPU reference on random inputs: max abs error ≤ 2e-4 in LogC4 units.
