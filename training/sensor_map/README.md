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
python3 -m training.sensor_map.gate                # the M1-05 gate, 11 profiles; exit 1 while LED-V2 is over
python3 -m training.sensor_map.gate --expect-failing LED-V2   # exit 0 iff exactly LED-V2 is over (the reported failure)
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

**Result for A7R IV (`sony-ilce-7rm4`), shipped Lab objective: eight of nine lights pass, LED-V2 fails, so the gate reports FAIL
(reported failure, accepted by the product owner on 2026-10-12).** ΔE00, median / p95 over the 99 CES:

| Light | Class | CCT (K) | **Gate fit (Lab)** | | Previous (weighted RGB) | No mapping | Lab 3×3 fitted at the light |
|---|---|---|---|---|---|---|---|
| blackbody 2850 K | tungsten | 2850.0 | **0.300 / 1.045** | pass | 0.411 / 0.876 | 1.390 / 2.890 | 0.301 / 1.045 |
| D 5250 K | daylight | 5250.0 | **0.372 / 1.108** | pass | 0.471 / 0.999 | 1.416 / 2.832 | 0.372 / 1.107 |
| D 6750 K | daylight | 6750.0 | **0.417 / 1.170** | pass | 0.508 / 1.018 (median over) | 1.624 / 3.334 | 0.417 / 1.170 |
| FL3.1 | fluorescent | 2931.5 | **0.312 / 0.997** | pass | 0.399 / 0.857 | 1.294 / 2.609 | 0.298 / 1.000 |
| FL3.8 | fluorescent | 4005.5 | **0.206 / 0.593** | pass | 0.249 / 0.526 | 1.268 / 2.221 | 0.152 / 0.599 |
| FL3.15 | fluorescent | 6507.8 | **0.304 / 1.098** | pass | 0.457 / 0.988 | 1.563 / 3.161 | 0.305 / 1.093 |
| LED-B3 | led | 4102.5 | **0.235 / 1.015** | pass | 0.339 / 0.903 | 1.370 / 2.588 | 0.219 / 1.001 |
| LED-RGB1 | led | 2839.8 | **0.316 / 0.984** | pass | 0.324 / 0.755 | 1.391 / 3.329 | 0.219 / 0.938 |
| LED-V2 | led | 4069.5 | **0.759 / 1.515** | **FAIL** | 0.773 / 1.377 (over) | 1.403 / 2.566 | 0.667 / 1.414 |

Worst median 0.759, worst p95 1.515 (both LED-V2); every other light is below 0.42 / 1.18. Three things to know:

- **LED-V2 is a capacity limit, not an interpolation error:** a narrow violet-pump LED spike; beyond a white-preserving 3×3. The
  3×3 fitted at LED-V2 itself gives 0.667 / 1.414, and even maps that give up white preservation only just get to 0.51 median
  (table below). It stays a reported failure.
- **The Lab objective buys median at the expense of p95.** It minimises the Euclidean Lab error (ΔE76-like) while the gate scores
  ΔE00, and the tail of the 99 CES grows: p95 rises on all nine lights (by 0.07 to 0.23; LED-V2 1.377 → 1.515), while the medians fall
  on all nine. The thinnest p95 margins to the 1.2 limit: D 6750 K 0.03, D 5250 K 0.09, FL3.15 0.10.
- **Table blend vs a fit at the light.** For tungsten and daylight the blend equals the map fitted at the gate light itself to 0.001.
  Fluorescent costs up to 0.054 in median (FL3.8), LED up to 0.097 (LED-RGB1) and 0.092 (LED-V2): the gate refit leaves the LED class
  with six lights, which is also why the shipped tables (all nine on grid, `shipped_fit` in `gate.json`) are better than the gate
  numbers for these three lights.

`gate_diagnostics.py` shows what other families would do on the same data (median / p95, fitted at the light):

| Light | **Lab-optimal white-preserving 3×3 (shipped)** | weighted RGB white-preserving 3×3 (previous) | free 3×3 (white not preserved) | root-polynomial (white not preserved) |
|---|---|---|---|---|
| blackbody 2850 K | 0.301 / 1.045 | 0.410 / 0.876 | 0.382 / 0.766 | 0.391 / 0.862 |
| D 5250 K | 0.372 / 1.107 | 0.471 / 0.999 | 0.395 / 0.827 | 0.358 / 0.898 |
| D 6750 K | 0.417 / 1.170 | 0.508 / 1.018 | 0.395 / 0.879 | 0.375 / 1.065 |
| FL3.1 | 0.298 / 1.000 | 0.395 / 0.870 | 0.292 / 0.737 | 0.312 / 0.831 |
| FL3.8 | 0.152 / 0.599 | 0.214 / 0.510 | 0.171 / 0.447 | 0.173 / 0.525 |
| FL3.15 | 0.305 / 1.093 | 0.445 / 0.948 | 0.351 / 0.800 | 0.334 / 0.873 |
| LED-B3 | 0.219 / 1.001 | 0.343 / 0.906 | 0.339 / 0.785 | 0.302 / 0.794 |
| LED-RGB1 | 0.219 / 0.938 | 0.293 / 0.707 | 0.295 / 0.657 | 0.263 / 0.702 |
| LED-V2 | 0.667 / 1.414 | 0.702 / 1.352 | 0.513 / 1.113 | 0.506 / 1.309 |

Reaching LED-V2 would need the white-preserving constraint relaxed or a non-linear map (the optional MLP / root-polynomial stage of
`docs/02` §C2), and even those only just reach 0.51. Not applied; the decision is to report it.

**Other ten profiles** (worst median / worst p95 over the nine gate lights, with the lights over the gate; the gate itself applies
to A7R IV only). Full tables in `testdata/out/sensor_map/gate.json` after running the gate.

| Body | Previous (weighted RGB) | **Shipped (Lab)** | Lights over the gate |
|---|---|---|---|
| A7 III | 0.594 / 1.059 | **0.551 / 1.202** | LED-V2 |
| A7 IV | 0.785 / 1.369 | **0.767 / 1.484** | LED-V2 |
| A7C II | 0.899 / 1.574 | **0.898 / 1.673** | D 6750 K (0.503 / 1.155), LED-V2 |
| A7R II | 0.524 / 0.949 | **0.492 / 1.030** | none |
| A7R III | 0.399 / 0.813 | **0.337 / 0.981** | none |
| A7S II | 0.631 / 2.078 | **0.622 / 1.868** | 8 of 9 (all but FL3.8) |
| A7S III | 0.535 / 1.011 | **0.443 / 1.199** | none (p95 margin 0.001) |
| A9 | 1.182 / 2.188 | **1.159 / 2.208** | D 5250 K, D 6750 K, LED-V2 |
| A6400 | 0.692 / 1.737 | **0.677 / 1.685** | LED-V2 |
| RX100 IV | 1.018 / 2.524 | **1.012 / 2.526** | blackbody 2850 K, D 6750 K, FL3.8, FL3.15, LED-RGB1, LED-V2 |

The worst median is LED-V2 for every body except A7S II (LED-RGB1). Three more bodies (A7R II, A7R III, A7S III) would pass all nine
lights; A7R IV, the gated profile, is not among them because of LED-V2.

## Reproducing the "own" and "A7R IV" numbers

`evaluate.py` is the port of `sony_profiles.py`: per-light 3×3 fitted at each of the 9 research test lights, scored on
138 colours (CES 99 + Ohta ColorChecker 24 + NIST CQS VS 15). All columns of `research/alexa-ssf/out/sony_profiles.csv`
(own median / p95, A7R IV median / p95, worst light) are reproduced with **maximum difference 0.000** (they are the same
arithmetic on the same inputs; the CSV is rounded to 3 decimals).
The evaluation uses the research objective (`fit_map_weighted_rgb`) whatever the shipped objective is: the CSV was produced with
it, and reproducing it is the check that the port of the arithmetic is right. The "own" and "A7R IV" figures of `docs/03` and
`docs/13` therefore do not describe the shipped Lab-optimal maps; the gate table above does.

## Registry notes

- The 28 model IDs are the table of `docs/03`; every ID and model string was checked against ExifTool's `Sony.pm`
  (`SonyModelID`, exiftool/exiftool @ 2200871d9cef). `validate_profiles` also parses `docs/03` and fails if the registry
  and the table differ.
- `model_strings` maps each Model string to its ID. ExifTool's Sony tables show no further regional or kit spellings for
  these bodies beyond the "A" models already in the table (ILCE-7RM3A, 7RM4A, 6100A, 6400A), so none were added. Hasselblad
  Lunar and Stellar (rebadged Sony bodies) have no model ID in the table; they fall through to the default profile.
- Same-sensor pairings need a ColorMatrix check against Adobe DNG Converter samples that V1 doesn't have. Each pairing is
  recorded in the profile's `provenance.checks` as `pending: no converted samples available` (12 pairings in 6 profiles).

## Determinism

Output is a pure function of the fetched inputs, the scripts and colour-science 0.4.7: no timestamps, tables rounded to
1e-9, matrices to 1e-12, fixed seeds in the studies. `provenance.fit_script` records the commit the tree was based on and
`provenance.fit_sources_sha256` a hash of the fit sources, so the code is identified even for an uncommitted tree.
BLAS differences between machines can change the last digit of a matrix; `--check` tolerates 5e-9.
The Lab fit adds a solver: `provenance.fit_solver` records scipy and numpy versions and the fixed options (`fit.LAB_SOLVER`,
`LAB_POLISH_*`); the fit is bit-identical on repeat and independent of the start to ~1e-14 (`fit --stability`), so another BLAS or
scipy moves coefficients by far less than the 5e-9 tolerance of `--check`, which ignores the version string.
