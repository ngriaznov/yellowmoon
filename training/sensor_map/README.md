# training/sensor_map/

Fits the sensor map, builds the camera profiles, registry and target, evaluates and gates them, and writes the
reference values the Rust runtime (`ym-color`, `ym-profiles`) is tested against. Tasks M1-02, M1-04 and the Python side
of M1-05. The Rust runtime is a later task; this file is its specification, together with
`crates/ym-profiles/schema/*.schema.json` and the executable reference `runtime_ref.py`.

| File | Role |
|---|---|
| `fit.py` | the shipped Lab-optimal white-preserving 3×3 fit (`fit_map_lab`), the superseded weighted-RGB fit (`fit_map_weighted_rgb` = `sony_profiles.fit_map`), the per-class `(mired, M)` tables, the runtime blend; the two curve loaders: `sony_curves()` (peak-normalised per channel, for the maps) and `sony_curves_raw()` (the published raw scale, for the locus; see "Which scale the curves are in") |
| `lights.py` | reference and gate lights (SPDs from colour-science, `ssf_core` conventions) |
| `tables.py` | `w_A` tables, camera locus (raw-scale curves only: `require_raw_scale`), CIE xy, Duv scale |
| `build_profiles.py` | writes `data/cameras/profiles/<id>/profile.json` ×11 |
| `registry_data.py`, `build_registry.py` | the `docs/03` camera table → `data/cameras/registry.json` |
| `build_target.py` | `data/alexa/target.json` |
| `validate_profiles.py` | schemas + white preservation + mired order + grids + locus scale (raw-scale curves, 1e-6) + registry vs `docs/03` |
| `runtime_ref.py` | reference implementation of every runtime rule below |
| `evaluate.py` | port of `sony_profiles.py` evaluation (reproduces `sony_profiles.csv`) |
| `gate.py`, `gate_diagnostics.py` | the sensor-map gate and the capacity analysis behind its numbers |
| `check_reference.py` | checks that the reference file matches the shipped profile and that `runtime_ref` reproduces it |
| `spectral_reference.py` | writes `crates/ym-color/tests/data/spectral_reference.json`, the interpolation study |
| `export_fixtures.py` | writes `testdata/spectral/` (git-ignored) for a Rust port of the evaluation |

Regenerate everything (from the repository root, sources fetched):

```bash
python3 -m training.sensor_map.build_target        # only if data/alexa/ changed
python3 -m training.sensor_map.build_profiles      # --git-sha <sha> to stamp a commit
python3 -m training.sensor_map.build_registry
python3 -m training.sensor_map.validate_profiles
python3 -m training.sensor_map.spectral_reference  # depends on the maps (sensor_map_blend): rerun after any refit
python3 -m training.sensor_map.check_reference
```

`build_profiles --check` regenerates in memory and compares with the files (float tolerance 5e-9; the commit in `fit_script` and the library versions in `fit_solver` are ignored).

Checks and reports:

```bash
python3 -m training.sensor_map.evaluate            # must PASS: reproduces research/alexa-ssf/out/sony_profiles.csv (research objective)
python3 -m training.sensor_map.gate                # the M1-05 gate, 11 profiles; exit 1 while any light of the A7R IV is over
python3 -m training.sensor_map.gate --expect-failing blackbody-2850K daylight-5250K daylight-6750K FL3.15 LED-B3 LED-V2   # exit 0 iff exactly these six are over (the reported failures, below)
python3 -m training.sensor_map.gate --objective weighted_rgb  # the same run with the superseded objective
python3 -m training.sensor_map.gate_diagnostics    # capacity of the map families per gate light
python3 -m training.sensor_map.fit --stability     # reproducibility of the Lab fit (495 fits)
```

## Which scale the curves are in (docs/02 §C1; fix of 2026-10-12)

Two scales of the Weta curves are in use, on purpose:

| Table | Curves | Why |
|---|---|---|
| `locus` (log-chroma), `duv_scale` | **published raw scale**: `fit.sony_curves_raw()` = `data/cameras/sony/*.json` clipped at 0, **not normalised** (one common factor per camera, green peak exactly 1.0, red peak 0.57–0.62, blue peak 0.78–0.87 of it) | the scene white at runtime is raw data (as-shot gains, Pick white, the illuminant estimator), so a locus point must be the raw white `S^T E / (S_G^T E)` of its light |
| `sensor_map` matrices | `fit.sony_curves()` = `ssf_core.load_cameras`: clipped, each channel divided by its own peak | scale-free: `src` and `dst` are each divided by their response to the white, so any per-channel scale cancels; kept so the maps are byte for byte what the research LOO and gate numbers use |
| `alexa_white` | the ALEXA estimate, peak-normalised by construction | only ratios enter C3·C4 |
| `locus.xy` | none (CIE only) | |

Peak-normalising the locus (the first version) multiplied the red response by 1.6–1.8 and the blue by 1.15–1.3 (A7 III blackbody 6504 K: raw `[0.371, 1, 0.767]`, peak-normalised `[0.623, 1, 0.915]`, its ColorMatrix `[0.386, 1, 0.762]`) and made real as-shot whites read Duv −0.022 to −0.052. The two locus versions differ by a constant vector per camera in log-chroma, `(ln(peak_R/peak_G), ln(peak_B/peak_G))`, the same at every temperature, so **`duv_scale` and the chord geometry are identical** in both (the rebuild reproduced the old `duv_scale` tables and `spectral_reference.json` tolerances exactly); only the position of the locus moved. `tables.locus_tables` and `tables.duv_scale_tables` raise if the three channels of the curves peak at the same value (the signature of the old input). Anything compared with raw data (the illuminant estimator's training lights, a test white) must use `sony_curves_raw()` or an equivalent unnormalised reading of the same files. `provenance.light_conventions` states the scale of every table.

## The fit (docs/02 §C2)

For one light `E` (peak-normalised SPD, 380–780 nm / 5 nm) with ARRI CCT `T_E`: `src = S_P·(E∘R) / S_P·E`, `dst = S_A·(E∘R) / S_A·E`
over the 190 rawtoaces training reflectances. The shipped map is the white-preserving 3×3 (rows `[m0, m1, 1 − m0 − m1]`, six
free coefficients) that minimises the **CIE Lab error after ARRI's matrix**, `Σₙ ‖Lab(A(T_E)·M·src[n]) − Lab(A(T_E)·dst[n])‖²`,
with `A(T)` ARRI's native → AP0 matrix interpolated in mired at `T_E` and Lab taken with the ACES white (the `ssf_core`
conventions; `fit.fit_map_lab`). It is the `lab` column of `gate_diagnostics.py`, whose numbers carry over unchanged.

*Decision 2026-10-12 (Mykyta, product owner), after the M1-05 gate failed for A7R IV with the specified objective:* option 2 of the
gate report. Fit the white-preserving 3×3 by minimising the CIE Lab error after ARRI's matrix instead of the `Y^(−2/3)`-weighted RGB
least squares, and keep LED-V2 as a reported failure. The gate thresholds (median ≤ 0.5, p95 ≤ 1.2) are unchanged.

- **Start and solver (fixed, deterministic).** The start is the previous objective's linear solution `fit.fit_map_weighted_rgb`
  (weights `clip(Y_target, 1e-3)^(−2/3)`). Then MINPACK Levenberg–Marquardt (`scipy.optimize.least_squares(method="lm")`) with the
  analytic Jacobian and `xtol = ftol = gtol = 1e-15`, then up to 8 Gauss–Newton steps (stop below 1e-13). The polish matters:
  the cost is flat to rounding noise for coefficient changes of ~1e-8, so the solver alone ends anywhere in that valley (differences
  up to ~6e-9 between starts). With it, all 495 fits (11 bodies × 45 lights) agree to 1.5e-14 whether started from the fixed start, from a
  1e-3 random perturbation or from the identity, and a repeat run is bit-identical (`python3 -m training.sensor_map.fit --stability`).
  A Lab fit takes ~7 ms (the weighted-RGB fit 0.5 ms); the 11 profiles build in 30 s. The scipy and numpy versions (1.18.1 / 2.5.3) are in `provenance.fit_solver`.
- **Which CCT.** The one used for the mired axis: the generating temperature for blackbody and D-series lights, the Ohno (2013) CCT of
  the SPD for F and LED lights (the table below). It also selects `A(T_E)`.
- **The old objective stays available.** `fit_map_weighted_rgb` is `sony_profiles.fit_map` unchanged (`evaluate.py` reproduces the
  published CSV with difference 0.000 through it) and is what `gate.py --objective weighted_rgb` and the "previous" columns below use.
- **Recorded in the profile.** `sensor_map.fit.objective = "cie-lab-after-arri-matrix"` and `provenance.sensor_map_objective =
  "lab-optimal white-preserving 3x3 (docs/02 §C2, decision 2026-10-12)"`. This replaces the `fit.weights` field of the schema.
  Profile `version` stays `1.0.0` (nothing has shipped).

Reference lights (45), each class sorted by `(mired, name)`:

| Class | Lights | CCT used for mired |
|---|---|---|
| tungsten | blackbody 2000–4000 K every 250 K (9) | generating temperature |
| daylight | CIE D series 4000–11000 K every 500 K (15) | generating temperature |
| fluorescent | CIE F1–F12 (12) | Ohno 2013 CCT of the SPD |
| led | LED-B1…B5, BH1, RGB1, V1, V2 (9) | Ohno 2013 CCT of the SPD |

`mired = 1e6 / CCT`. Matrices are written with the first two entries per row rounded to 12 decimals and the third as
the remainder, so `M·[1,1,1]ᵀ = [1,1,1]ᵀ` holds to ~1e-16 in the file (`validate_profiles` checks 1e-12).

**Runtime blend** (`runtime_ref.sensor_map`, `fit.blend_map`): `M_map = Σ_k p_k · M_k(mired_scene)`; inside a class, linear
interpolation in mired between the two bracketing entries, clamped at the ends. Class probabilities `p_k` sum to 1.
A convex combination of white-preserving matrices is white-preserving.

## Reading `profile.json` (normative)

**Grids.** `{start_k, step_k, values}`: `values[i]` belongs to `T_i = start_k + i·step_k`. A query outside the table is
clamped to its ends. For `T_i ≤ T ≤ T_(i+1)` the weight is in **mired**:

```
a = (1/T − 1/T_i) / (1/T_(i+1) − 1/T_i)
```

| Table | Interpolation |
|---|---|
| `alexa_white.blackbody`, `.daylight` (`w_A = [r, 1, b]`) | geometric: `w = w_i^(1−a) · w_(i+1)^a` per channel (green stays exactly 1) |
| `locus.blackbody`, `.daylight` (log-chroma), `locus.xy.*`, `locus.duv_scale.*` | linear: `(1−a)·v_i + a·v_(i+1)` |
| class lists (`sensor_map.classes.*`, `alexa_white.fluorescent`, `.led`) | linear in mired between bracketing entries, clamped at the ends (`M` or `w`) |

**Which table.** Blackbody tables for `T < 4000 K`, daylight tables for `T ≥ 4000 K` (the rawtoaces rule of `docs/02`). The
blackbody `w_A` table also extends to 25000 K (Planckian whites at any temperature); the daylight tables start at
4000 K because the CIE D series is only defined from there.

**RGB → (CCT, Duv)** (`runtime_ref.rgb_to_cct_duv`):

1. `p = (ln R/G, ln B/G)` of the camera RGB. The overall scale does not matter, **the balance between the channels does**: it must be the camera's raw balance (the camera RGB of a white, as-shot gains inverted, Pick white, the estimator's `w_scene`; see "Which scale the curves are in").
2. For each branch (blackbody 1500–4000 K, daylight 4000–25000 K) take the polyline of its table entries and find the
   nearest point to `p`: project `p` on every chord, clamp to the chord, keep the chord with the smallest distance (first
   on ties). **Skip chords of zero length** (the D-series staircase can repeat a row exactly).
3. Take the branch with the smaller distance (daylight on a tie).
4. The foot point's position along its chord gives mired: `mired = m_i + t·(m_(i+1) − m_i)`, `CCT = 1e6/mired` (so it
   inverts the locus interpolation exactly on the chord).
5. `Duv = (p − foot)·n · duv_scale(CCT)` with `n = (t_y, −t_x)/|t|`, `t` the chord direction (increasing temperature).
   **Positive toward magenta**, which is the opposite of CIE's sign (CIE: positive above the locus, toward green).
   `duv_scale` comes from the table of the foot point's branch (blackbody below 4000 K), interpolated like the locus.

**(CCT, Duv) → RGB** (`cct_duv_to_rgb`): `q = locus(CCT)`, `p = q + (Duv/duv_scale(CCT))·n` with `n` from the chord that
contains `CCT`, `rgb = (e^p0, 1, e^p1)`. It inverts the first for small Duv (round trip below 1e-6 K and 1e-8 Duv for
every case in the reference file except 1600 K, where the chord curvature moves the foot point by 0.3 K).

**`duv_scale`** is the CIE 1960 uv distance per unit of log-chroma distance **across** the locus, measured with spectral
tints (a smooth green excess or deficit of ±5 % and ±10 % added to the locus light) every 100 K. It runs from ~0.012 at
1500 K to ~0.10 at 11000 K and above, so a single number would be wrong by a factor of two somewhere. Accuracy of the
readout against CIE colorimetry on 90 independent tinted lights (other temperatures, three tilt shapes, ±3 %…±15 %):
median error 4e-5 Duv (2.3 % of the value), p90 7 %, worst 2.3e-4 Duv. It is a camera-space readout: lights with spiky
spectra (fluorescent, LED) can differ from CIE Duv by several 1e-3 because the camera and the observer weigh the spikes
differently (CIE FL2: camera 0.0037 vs CIE-relative-to-locus 0.0010).

## Accuracy of the tables against the spectra (M1-02 "to 1e-6")

`spectral_reference.json` carries, for the A7R IV profile and the ALEXA estimate, the **exact** values computed from the
spectra (`ssf_core` conventions) and the values the interpolation above gives from the shipped tables.

- **On grid points** the tables equal `ssf_core` to 5e-10 (rounding to 9 decimals). Tolerance 1e-6 holds.
- **The Rust port** must reproduce the `interpolated` fields to 1e-9 (same numbers, same arithmetic).
- **Off grid, against the exact spectra, 1e-6 is not reachable with a 10 K table and linear-type interpolation.** Measured
  worst cases (1500 random temperatures per branch; the table lists the A7R IV, all 11 bodies are within 10 %):

| Quantity | Kelvin-linear | **Mired (stated rule)** | Stated tolerance (×2, rounded up) |
|---|---|---|---|
| `w_A` blackbody 1500–2000 K | 7.5e-5 | 1.4e-5 | 3e-5 (all of 1500–4000 K) |
| `w_A` blackbody 2000–4000 K | 2.4e-5 | 2.0e-6 | 5e-6 |
| locus blackbody 1500–2000 K | 1.1e-5 | 1.6e-5 | 4e-5 (all of 1500–4000 K) |
| locus blackbody 2000–4000 K | 6.5e-6 | 2.8e-6 | 6e-6 |
| `w_A` daylight 4000–25000 K | 1.6e-4 | 1.6e-4 | 4e-4 |
| locus daylight 4000–25000 K | 1.8e-4 | 1.8e-4 | 4e-4 |

  *Blackbody:* `ln w` is nearly linear in mired (Wien), but the integral over the sensor's bandwidth adds curvature
  (variance of `c₂/λ` across the band); a 10 K step at 1500 K is 4.4 mired. The mired rule is 3–10× better than kelvin-linear
  for `w_A`; a 1 K grid would reach ~1e-7.
  *Daylight:* colour-science builds the CIE D series with `M1` and `M2` **rounded to three decimals** (as the CIE
  recommends for tabulated illuminants). The spectrum is therefore a staircase in T, steps of ~7e-5 (R/G) and ~1.6e-4
  (B/G), and no interpolation of any table can follow it; a finer grid does not help. The reference keeps the staircase
  because the ALEXA estimate was fitted with it (`ssf_core.illuminant`). Above ~18000 K the steps exceed the 10 K spacing, so
  the daylight locus is not strictly monotone there (`validate_profiles` allows 2e-4 of slack); CCT readouts of D-series
  lights come out up to ~1 K off (5003 K → 5003.8 K, 9876 K → 9876.7 K). All of this is far below anything visible: 4e-4
  in `w_A` is ~0.03 ΔE00.

  **Proposal** (needs a decision, not applied): keep the tables and test off-grid values against the stated tolerances above
  (they are in the JSON as `tolerances.off_grid_vs_exact_abs`). If 1e-6 off-grid is a hard requirement, it needs two
  changes together: generate the daylight tables from the unrounded D series (`sd_CIE_illuminant_D_series(...,
  M1_M2_rounding=False)`, which differs from `ssf_core` by up to 2e-4 on grid) and a 1 K blackbody grid (×10 for that
  part of the file). The profile gets ~100 kB larger per body.

## The sensor-map gate (M1-05, Python side)

`gate.py` implements `docs/02` §C2 "Gate lights" for all 11 profiles:

1. 99 CIE 2017 CES reflectances (colour-science CFI 2017 dataset) × 9 held-out lights: blackbody 2850 K, CIE D 5250 K and
   6750 K, CIE FL3.1, FL3.8, FL3.15, LED-B3, LED-RGB1, LED-V2.
2. **Refit** with LED-B3, LED-RGB1 and LED-V2 removed from the LED class (tungsten, daylight and fluorescent keep all their
   lights; none of the nine lights is a fit light), with the shipped Lab objective like the profiles.
3. **Scene light → class and mired at gate time**: the true light's class (tungsten for the blackbody, daylight for the
   D-series lights, fluorescent for FL3.x, led for the LED lights) with probability 1, and `mired = 1e6/CCT` of the true
   light. The CCT is the generating temperature for blackbody and D-series lights (2850, 5250, 6750 K) and the Ohno (2013)
   CCT of the SPD for FL3.x and LED lights (2931.5, 4005.5, 6507.8 K; 4102.5, 2839.8, 4069.5 K). So the map is the table
   blend with the light known exactly; estimator error is the separate M1-06 gate.
4. Both renderings (Sony through `M`, ALEXA estimate directly) are white-balanced to the light and taken through ARRI's
   matrix at the light's CCT to AP0, then Lab with the ACES white; ΔE00 per reflectance; per-light median and p95.

Gate: median ≤ 0.5 and p95 ≤ 1.2 for the A7R IV profile (reported, never loosened). The thresholds were not touched by the decision of 2026-10-12.

**Result for A7R IV (`sony-ilce-7rm4`), shipped Lab objective: three of nine lights pass and six are over, so the gate reports FAIL.**
LED-V2 is the reported failure the product owner accepted on 2026-10-12 (it was the only light over). The other five came with the owner's decision of
2026-10-25, "Move ALEXA estimate": the ALEXA estimate the gate scores against now carries a hue push (`training/alexa/estimate_alexa.py`, `docs/03`
*The ALEXA estimate*), and a white-preserving 3×3 reaches that curve less well. The thresholds are unchanged and CI expects exactly these six lights
(`--expect-failing`). ΔE00, median / p95 over the 99 CES; "before the move" is the estimate of 2026-10-11 (version 1.0.0 of the profiles):

| Light | Class | CCT (K) | **Gate fit (Lab)** | | Before the move | Previous (weighted RGB) | No mapping | Lab 3×3 fitted at the light |
|---|---|---|---|---|---|---|---|---|
| blackbody 2850 K | tungsten | 2850.0 | **0.464 / 1.505** | **FAIL** | 0.300 / 1.045 | 0.519 / 1.363 (over) | 1.214 / 2.473 | 0.464 / 1.505 |
| D 5250 K | daylight | 5250.0 | **0.453 / 1.441** | **FAIL** | 0.372 / 1.108 | 0.598 / 1.296 (over) | 1.348 / 3.267 | 0.453 / 1.441 |
| D 6750 K | daylight | 6750.0 | **0.502 / 1.448** | **FAIL** | 0.417 / 1.170 | 0.637 / 1.334 (over) | 1.734 / 3.665 | 0.502 / 1.448 |
| FL3.1 | fluorescent | 2931.5 | **0.375 / 1.093** | pass | 0.312 / 0.997 | 0.426 / 1.013 | 1.204 / 2.349 | 0.360 / 1.103 |
| FL3.8 | fluorescent | 4005.5 | **0.282 / 0.745** | pass | 0.206 / 0.593 | 0.289 / 0.753 | 1.100 / 2.012 | 0.267 / 0.756 |
| FL3.15 | fluorescent | 6507.8 | **0.520 / 1.367** | **FAIL** | 0.304 / 1.098 | 0.589 / 1.367 (over) | 1.631 / 3.481 | 0.532 / 1.381 |
| LED-B3 | led | 4102.5 | **0.420 / 1.245** | **FAIL** | 0.235 / 1.015 | 0.477 / 1.171 | 1.379 / 2.813 | 0.410 / 1.259 |
| LED-RGB1 | led | 2839.8 | **0.447 / 1.005** | pass | 0.316 / 0.984 | 0.484 / 0.980 | 1.000 / 2.016 | 0.284 / 0.981 |
| LED-V2 | led | 4069.5 | **0.846 / 1.681** | **FAIL** | 0.759 / 1.515 | 0.934 / 1.546 (over) | 1.197 / 2.841 | 0.634 / 1.651 |

Worst median 0.846, worst p95 1.681 (both LED-V2; 0.759 / 1.515 before the move). Four things to know:

- **The new failures are a capacity limit of the white-preserving 3×3, not an interpolation error.** The 3×3 fitted at the gate light itself is over the gate too
  (last column): blackbody 2850 K 0.464 / 1.505, D 5250 K 0.453 / 1.441, D 6750 K 0.502 / 1.448, FL3.15 0.532 / 1.381, LED-B3 0.410 / 1.259, LED-V2 0.634 / 1.651.
  For tungsten and daylight the blend equals that map to 0.001. A map that is not white-preserving does better on the same data (table below).
- **The same six lights fail under the superseded weighted-RGB objective** except LED-B3 (0.477 / 1.171, a pass); its LED-V2 is 0.934 / 1.546.
- **The Lab objective still buys median at the expense of p95.** It minimises the Euclidean Lab error (ΔE76-like) while the gate scores ΔE00: the median falls on all
  nine lights (by 0.006 to 0.144) and the p95 rises on seven (by 0.02 to 0.15), stays on FL3.15 and falls by 0.007 on FL3.8.
- **Table blend vs a fit at the light for LED.** LED-RGB1 (+0.163 in the median) and LED-V2 (+0.213) pay for the gate refit leaving the LED class with six lights, which is
  also why the shipped tables (all nine on grid, `shipped_fit` in `gate.json`) are better than the gate numbers for these two.

`gate_diagnostics.py` shows what other families would do on the same data (median / p95, fitted at the light):

| Light | **Lab-optimal white-preserving 3×3 (shipped)** | weighted RGB white-preserving 3×3 (previous) | free 3×3 (white not preserved) | root-polynomial (white not preserved) |
|---|---|---|---|---|
| blackbody 2850 K | 0.464 / 1.505 | 0.517 / 1.363 | 0.504 / 1.086 | 0.432 / 1.165 |
| D 5250 K | 0.453 / 1.441 | 0.597 / 1.296 | 0.498 / 1.072 | 0.453 / 1.158 |
| D 6750 K | 0.502 / 1.448 | 0.637 / 1.334 | 0.500 / 1.078 | 0.431 / 1.121 |
| FL3.1 | 0.360 / 1.103 | 0.434 / 1.027 | 0.398 / 0.964 | 0.331 / 0.934 |
| FL3.8 | 0.267 / 0.756 | 0.291 / 0.730 | 0.279 / 0.857 | 0.217 / 0.694 |
| FL3.15 | 0.532 / 1.381 | 0.559 / 1.278 | 0.521 / 1.082 | 0.408 / 1.097 |
| LED-B3 | 0.410 / 1.259 | 0.449 / 1.111 | 0.472 / 0.957 | 0.359 / 0.992 |
| LED-RGB1 | 0.284 / 0.981 | 0.321 / 0.845 | 0.354 / 0.735 | 0.273 / 0.815 |
| LED-V2 | 0.634 / 1.651 | 0.790 / 1.487 | 0.512 / 1.188 | 0.513 / 1.512 |

The root-polynomial map (not white-preserving) would bring five of the six lights back under the gate (blackbody 2850 K 0.432 / 1.165, D 5250 K 0.453 / 1.158,
D 6750 K 0.431 / 1.121, FL3.15 0.408 / 1.097, LED-B3 0.359 / 0.992) and leave LED-V2 at 0.513 / 1.512. That is the optional upgrade of `docs/02` §C2 (a tiny model
predicting the coefficients of a non-linear map); it is not applied, and it is the owner's decision whether to build it.

**Other ten profiles** (worst median / worst p95 over the nine gate lights, with the lights over the gate; the gate itself applies
to A7R IV only). Full tables in `testdata/out/sensor_map/gate.json` after running the gate. "Before the move" is the estimate of 2026-10-11.

| Body | Before the move (Lab) | Previous (weighted RGB) | **Shipped (Lab)** | Lights over the gate |
|---|---|---|---|---|
| A7 III | 0.551 / 1.202 (over: LED-V2) | 0.782 / 1.366 | **0.718 / 1.380** | 5 of 9: blackbody 2850 K, D 5250 K, D 6750 K, FL3.15, LED-V2 |
| A7 IV | 0.767 / 1.484 (over: LED-V2) | 0.933 / 1.554 | **0.870 / 1.649** | 5 of 9: blackbody 2850 K, D 5250 K, D 6750 K, FL3.15, LED-V2 |
| A7C II | 0.898 / 1.673 (over: D 6750 K, LED-V2) | 1.063 / 1.751 | **0.987 / 1.835** | 6 of 9: blackbody 2850 K, D 5250 K, D 6750 K, FL3.15, LED-B3, LED-V2 |
| A7R II | 0.492 / 1.030 (over: none) | 0.674 / 1.222 | **0.651 / 1.239** | LED-V2 (0.651 / 1.239) |
| A7R III | 0.337 / 0.981 (over: none) | 0.633 / 1.284 | **0.591 / 1.220** | FL3.15 (0.468 / 1.220), LED-V2 (0.591 / 1.185) |
| A7S II | 0.622 / 1.868 (over: 8 of 9) | 0.787 / 2.022 | **0.779 / 1.905** | all 9 |
| A7S III | 0.443 / 1.199 (over: none) | 0.740 / 1.401 | **0.645 / 1.378** | 5 of 9: blackbody 2850 K, D 5250 K, D 6750 K, FL3.15, LED-V2 |
| A9 | 1.159 / 2.208 (over: D 5250 K, D 6750 K, LED-V2) | 1.288 / 2.390 | **1.245 / 2.414** | 6 of 9: blackbody 2850 K, D 5250 K, D 6750 K, FL3.1, FL3.15, LED-V2 |
| A6400 | 0.677 / 1.685 (over: LED-V2) | 0.902 / 1.718 | **0.797 / 1.758** | 4 of 9: D 6750 K, FL3.15, LED-B3, LED-V2 |
| RX100 IV | 1.012 / 2.526 (over: 6 of 9) | 0.862 / 2.044 | **0.832 / 1.993** | LED-RGB1 (0.531 / 1.218), LED-V2 (0.832 / 1.993) |

The worst median is LED-V2 for every body. No body passes all nine lights (A7R II, A7R III and A7S III did before the move; so did none of the other eight).

## Reproducing the "own" and "A7R IV" numbers

`evaluate.py` is the port of `sony_profiles.py`: per-light 3×3 fitted at each of the 9 research test lights, scored on
138 colours (CES 99 + Ohta ColorChecker 24 + NIST CQS VS 15). All columns of `research/alexa-ssf/out/sony_profiles.csv`
(own median / p95, A7R IV median / p95, worst light) are reproduced with **maximum difference 0.000** (they are the same
arithmetic on the same inputs; the CSV is rounded to 3 decimals).
The evaluation uses the research objective (`fit_map_weighted_rgb`) whatever the shipped objective is: the CSV was produced with
it, and reproducing it is the check that the port of the arithmetic is right. The "own" and "A7R IV" figures of `docs/03` and
`docs/13` therefore do not describe the shipped Lab-optimal maps; the gate table above does.

## Registry notes

- The 28 model IDs are the table of `docs/03`: Sony's `SonyModelID` values and the model strings the cameras write,
  cross-checked against independent tools (numbers only). `validate_profiles` also parses `docs/03` and fails if the registry
  and the table differ.
- `model_strings` maps each Model string to its ID. no further regional or kit spellings are known for
  these bodies beyond the "A" models already in the table (ILCE-7RM3A, 7RM4A, 6100A, 6400A), so none were added. Hasselblad
  Lunar and Stellar (rebadged Sony bodies) have no model ID in the table; they fall through to the default profile.
- Same-sensor pairings are checked against the bodies' published D65 colour matrices because V1 has no Adobe DNG Converter samples
  (numbers only, `docs/14` rule 8; `crates/ym-profiles/tests/py/sibling_matrices.py`; rows
  normalised to sum 1; limit 0.02 per coefficient). The measured value of every pairing is in `registry_data.SIBLING_MAX_ABS_DIFF`
  and in the profile's `provenance.checks` (`pass: max |diff| 0.0000 ...`; `--verify` of the script compares them with a fresh
  run). Ten pairings pass (0.0000: the published matrices of the two bodies are the same). ILCE-7RM5 and ILCE-7CR differ from the
  A7R IV by 0.0934 and do not ship as same-sensor siblings: they have match kind `default` (the A7R IV profile is the default,
  so they render exactly as before) and the failed check is recorded in the A7R IV profile and in their registry notes.
- `sensor_width_mm`, `active_width_px` (and the optional `pixel_pitch_um` override) on a body give the pixel pitch for
  `docs/algorithms/03` section 2.2. They come from Sony's specification pages, with the source of every entry in
  `sensor_source` and `sensor_source_kind` (`maker` or `restated`); the table is `registry_data.SENSORS`. The Sony pages were
  read through web-search excerpts, not fetched (Sony hosts are not reachable from the build environment). ILME-FX3 has no
  entry: Sony's page gives its stills only as "12M", so it uses the EXIF crop-factor fallback.

## Determinism

Output is a pure function of the fetched inputs, the scripts and colour-science 0.4.7: no timestamps, tables rounded to
1e-9, matrices to 1e-12, fixed seeds in the studies. `provenance.fit_script` records the commit the tree was based on and
`provenance.fit_sources_sha256` a hash of the fit sources, so the code is identified even for an uncommitted tree.
BLAS differences between machines can change the last digit of a matrix; `--check` tolerates 5e-9.
The Lab fit adds a solver: `provenance.fit_solver` records scipy and numpy versions and the fixed options (`fit.LAB_SOLVER`,
`LAB_POLISH_*`); the fit is bit-identical on repeat and independent of the start to ~1e-14 (`fit --stability`), so another BLAS or
scipy moves coefficients by far less than the 5e-9 tolerance of `--check`, which ignores the version string.
