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

## C2 · Sensor map

Maps white-balanced Sony RGB to white-balanced emulated ALEV III RGB for the scene light. Exposure-invariant (linear) and white-preserving (`M·[1,1,1]ᵀ = [1,1,1]ᵀ`) by construction.

**Offline fit** (Python in `training/sensor_map/`, port of `research/alexa-ssf/scripts/sony_profiles.py::fit_map`):

1. Inputs: the camera profile's spectral curves `S_P`, the ALEXA estimate `S_A` (`data/alexa/alexa_alev3_estimate_380_780_5.json`), the 190 rawtoaces training reflectances (Apache-2.0), and reference light SPDs.
2. Reference lights per class:
   - tungsten: blackbody 2000–4000 K every 250 K;
   - daylight: CIE daylight 4000–11000 K every 500 K;
   - fluorescent: CIE F1–F12;
   - LED: CIE LED-B1…B5, LED-BH1, LED-RGB1, LED-V1, LED-V2.

   SPDs come from the `colour-science` datasets at fit time. Only the fitted coefficients ship (`docs/14-licensing.md`).

**Gate lights (held out of the gate fit):** blackbody 2850 K, CIE daylight 5250 K and 6750 K, CIE FL3.1, FL3.8 and FL3.15, LED-B3, LED-RGB1 and LED-V2. The first three and the FL3 series fall between or outside the fit grid by construction; for the gate run, refit with LED-B3, LED-RGB1 and LED-V2 removed from the LED class. The shipped fit uses all lights. The reference numbers in `research/alexa-ssf/README.md` were measured on lights inside the fit grid, so they are a lower bound, not a target.
3. For each light `E`: Sony responses `c_s = S_P·(E∘R)`, ALEXA responses `c_a = S_A·(E∘R)`, each divided by its own response to the white (`R = 1`). Fit the white-preserving 3×3 by weighted least squares with weights `Y^(−2/3)`:

```python
w = clip(Y_target, 1e-3, None) ** (-2/3)
A = stack([src[:,0] - src[:,2], src[:,1] - src[:,2]], axis=1) * w[:,None]
for r in range(3):
    m, *_ = lstsq(A, (dst[:, r] - src[:, 2]) * w)
    M[r] = [m[0], m[1], 1 - m[0] - m[1]]
```

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

### Camera RGB ↔ CCT and Duv

For a profile, the fit scripts tabulate camera RGB (green = 1) of blackbody 1500–4000 K and daylight 4000–25000 K every 10 K, as log-chroma `(ln R/G, ln B/G)`, plus the matching CIE xy and the Duv scale, into `profile.json` (`locus`). Convert `w_scene` to CCT by nearest point on that polyline, with linear interpolation between neighbours. Duv is the signed perpendicular distance, positive towards magenta, rescaled to CIE Duv through the profile's XYZ estimate. Use this for the Kelvin and tint readouts and the light presets.

## C5–C6 · ARRI matrix, AWG4, exposure

- `M_arri(CCT_user)` from the interpolated table above.
- `M_awg3→awg4` from the constants.
- Exposure: `EV = EV_auto + 0` in Auto, or the slider's absolute value in Manual. The slider range is −4…+4 EV absolute; its detent sits at `EV_auto`.

### Auto exposure

Goal: a pleasing exposure close to what a careful photographer would set, stable across a burst.

1. Run C1–C6 with EV = 0 on the 1/8-resolution preview. Compute linear luminance `Y` with the AWG4 Y row.
2. Ignore pixels in the clip mask and pixels with `Y` below the 1st or above the 99th percentile.
3. `m = exp(Σ wᵢ ln Yᵢ / Σ wᵢ)`, with a centre weight `wᵢ = exp(−r²/(2·0.35²))` (r = distance from centre over the half-diagonal).
4. `EV_auto = log2(0.18 / m) + 0.5 · EC`, where EC is the camera's exposure compensation from EXIF (the photographer's intent, half weight). Clamp to [−3, +3].
5. Highlight guard: if more than 0.5% of non-clipped pixels would exceed linear 8.0 after exposure, lower `EV_auto` until they don't, but by no more than 1 EV.
6. Files of the same camera within 2 s of each other and with identical exposure settings share the median `EV_auto` (bursts stay consistent).

The constants above are starting values; tune them in M1 on the A1 test set. Keep the algorithm, change the numbers.

## C7 · Highlight roll-off and low-light toe

Both act on a norm `n = max(R, G, B)` and scale RGB by `f(n)/n`, which preserves the ratios.

- **Highlights** (`rolloff.highlights = r ∈ [0,1]`, default 0.5). Let `c` be the clip level after C1–C6 (the smallest per-channel clip, mapped to AWG4 and exposure). Knee `k = c · 2^(−(0.5 + 2r))`. For `n ≤ k`: `f(n) = n`. Above: `f(n) = k + (c' − k) · tanh((n − k)/(c' − k))`, with `c' = 1.5c`. At 0 the shoulder starts 0.5 stop below clip and is short; at 1 it starts 2.5 stops below and is long. ALEV III has about 7.4 stops above grey at EI 800; a Sony body clips earlier at matched grey, so this shoulder stands in for the missing latitude.
- **Low light** (`rolloff.low_light = r`, default 0.5). `τ = 4r · σ_floor`, where `σ_floor` is the noise model's read noise mapped to linear AWG4 at this exposure. `f(n) = n³ / (n² + τ²)`, which is monotonic, ~n above 3τ and crushes the noise floor smoothly below τ. `r = 0` disables it.

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

## C11 · Curve

Applied in LogC4 after the look and before the display transform, so it travels into DNG export and behaves the same under every display transform. Four monotone cubic splines (Fritsch–Carlson) on [0,1]: luma (applied as `v += curve(L) − L`) and R, G, B (per channel). Baked into four 1024-entry 1D textures. The histogram in the Curve tab is of LogC4 values. Numbers in the UI show code values × 255, rounded.

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

### Imported ARRI display LUT

If the user chose "ARRI's display rendering" in the wizard or preferences, C12 is the imported display `.cube` (for example LogC4 → Rec.709 gamma 2.4, or → P3). LogC4 display LUTs apply to the LogC4 output of C11 directly; LogC3 display LUTs (the older package, including the classic Rec.709) apply after converting LogC4 → linear AWG4 → AWG3 → LogC3 EI 800. For the viewport, decode the LUT's output transfer (gamma 2.4 for Rec.709) to display-linear, convert primaries to P3, and encode with the sRGB transfer. For export, convert to the chosen colour space the same way.

## C13 · Display encode

sRGB transfer for sRGB and Display P3, BT.1886 inverse (gamma 2.4) for Rec.2020. Dither 8-bit outputs with a fixed blue-noise texture (seeded, deterministic). Embed the matching ICC profile on export (`docs/09-export.md`).

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
