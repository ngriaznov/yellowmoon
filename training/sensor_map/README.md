# training/sensor_map/

Fits the sensor map, builds the camera profiles, registry and target, evaluates and gates them, and writes the
reference values the Rust runtime (`ym-color`, `ym-profiles`) is tested against. Tasks M1-02, M1-04 and the Python side
of M1-05. The Rust runtime is a later task; this file is its specification, together with
`crates/ym-profiles/schema/*.schema.json` and the executable reference `runtime_ref.py`.

| File | Role |
|---|---|
| `fit.py` | port of `sony_profiles.py::fit_map`; fits the per-class `(mired, M)` tables |
| `lights.py` | reference and gate lights (SPDs from colour-science, `ssf_core` conventions) |
| `tables.py` | `w_A` tables, camera locus, CIE xy, Duv scale |
| `build_profiles.py` | writes `data/cameras/profiles/<id>/profile.json` ×11 |
| `registry_data.py`, `build_registry.py` | the `docs/03` camera table → `data/cameras/registry.json` |
| `build_target.py` | `data/alexa/target.json` |
| `validate_profiles.py` | schemas + white preservation + mired order + grids + registry vs `docs/03` |
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
python3 -m training.sensor_map.spectral_reference
python3 -m training.sensor_map.check_reference
```

`build_profiles --check` regenerates in memory and compares with the files (float tolerance 5e-9, commit ignored).

## The fit (docs/02 §C2)

For one light `E` (peak-normalised SPD, 380–780 nm / 5 nm): `src = S_P·(E∘R) / S_P·E`, `dst = S_A·(E∘R) / S_A·E` over the
190 rawtoaces training reflectances; `w = clip(Y_target, 1e-3)^(−2/3)` with `Y_target` from the CAT02-adapted scene
colour; the white-preserving 3×3 is the weighted least-squares solution with rows `[m0, m1, 1 − m0 − m1]`. This is
`sony_profiles.fit_map` unchanged (bit-identical: `evaluate.py` reproduces the published CSV with difference 0.000).

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

1. `p = (ln R/G, ln B/G)` of the camera RGB (any scale).
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
   lights; none of the nine lights is a fit light).
3. **Scene light → class and mired at gate time**: the true light's class (tungsten for the blackbody, daylight for the
   D-series lights, fluorescent for FL3.x, led for the LED lights) with probability 1, and `mired = 1e6/CCT` of the true
   light. The CCT is the generating temperature for blackbody and D-series lights (2850, 5250, 6750 K) and the Ohno (2013)
   CCT of the SPD for FL3.x and LED lights (2931.5, 4005.5, 6507.8 K; 4102.5, 2839.8, 4069.5 K). So the map is the table
   blend with the light known exactly; estimator error is the separate M1-06 gate.
4. Both renderings (Sony through `M`, ALEXA estimate directly) are white-balanced to the light and taken through ARRI's
   matrix at the light's CCT to AP0, then Lab with the ACES white; ΔE00 per reflectance; per-light median and p95.

Gate: median ≤ 0.5 and p95 ≤ 1.2 for the A7R IV profile (reported, never loosened).

**Result for A7R IV (`sony-ilce-7rm4`): FAIL.**

| Light | Class | CCT (K) | Gate fit median / p95 | No mapping | 3×3 fitted at the light |
|---|---|---|---|---|---|
| blackbody 2850 K | tungsten | 2850.0 | 0.411 / 0.876 | 1.390 / 2.890 | 0.410 / 0.876 |
| D 5250 K | daylight | 5250.0 | 0.471 / 0.999 | 1.416 / 2.832 | 0.471 / 0.999 |
| D 6750 K | daylight | 6750.0 | **0.508** / 1.018 | 1.624 / 3.334 | 0.508 / 1.018 |
| FL3.1 | fluorescent | 2931.5 | 0.399 / 0.857 | 1.294 / 2.609 | 0.395 / 0.870 |
| FL3.8 | fluorescent | 4005.5 | 0.249 / 0.526 | 1.268 / 2.221 | 0.214 / 0.510 |
| FL3.15 | fluorescent | 6507.8 | 0.457 / 0.988 | 1.563 / 3.161 | 0.445 / 0.948 |
| LED-B3 | led | 4102.5 | 0.339 / 0.903 | 1.370 / 2.588 | 0.343 / 0.906 |
| LED-RGB1 | led | 2839.8 | 0.324 / 0.755 | 1.391 / 3.329 | 0.293 / 0.707 |
| LED-V2 | led | 4069.5 | **0.773 / 1.377** | 1.403 / 2.566 | 0.702 / 1.352 |

Two lights are over: LED-V2 (median 0.773, p95 1.377) and D 6750 K (median 0.508 against 0.5). The "3×3 fitted at the
light" column shows this is not interpolation error: even a map fitted at the gate light itself, on the training patches,
gives 0.702 / 1.352 for LED-V2 and 0.508 for D 6750 K. The white-preserving 3×3 family with the `Y^(−2/3)` objective is
at its capacity limit for these two lights. `gate_diagnostics.py` shows what other families would do on the same data
(median / p95, fitted at the light):

| Light | spec (shipped) | Lab-optimal white-preserving 3×3 | free 3×3 (white not preserved) | root-polynomial (white not preserved) |
|---|---|---|---|---|
| D 6750 K | 0.508 / 1.018 | 0.417 / 1.170 | 0.395 / 0.879 | 0.375 / 1.065 |
| LED-V2 | 0.702 / 1.352 | 0.667 / 1.414 | 0.513 / 1.113 | 0.506 / 1.309 |

A Lab-optimal objective brings eight of nine lights under the median limit (p95 stays under 1.2 for eight) but not
LED-V2 (violet-pump LED, narrow blue spike). Reaching LED-V2 needs the white-preserving constraint relaxed or a non-linear
map; the root-polynomial and free fits only just reach 0.51 even without it. Options for Mykyta/M1-05: keep the gate as
specified and ship the 3×3 tables with the failure reported; adopt the Lab-optimal objective (a change to `docs/02` §C2);
or add the optional MLP/root-polynomial stage and re-measure. None was applied here.

Other profiles (worst median / worst p95 over the nine lights; full tables in `testdata/out/sensor_map/gate.json` after running the gate):
A7 III 0.594 / 1.059, A7 IV 0.785 / 1.369, A7C II 0.899 / 1.574, A7R II 0.524 / 0.949, A7R III 0.399 / 0.813 (the only pass),
A7S II 0.631 / 2.078, A7S III 0.535 / 1.011, A9 1.182 / 2.188, A6400 0.692 / 1.737, RX100 IV 1.018 / 2.524.

## Reproducing the "own" and "A7R IV" numbers

`evaluate.py` is the port of `sony_profiles.py`: per-light 3×3 fitted at each of the 9 research test lights, scored on
138 colours (CES 99 + Ohta ColorChecker 24 + NIST CQS VS 15). All columns of `research/alexa-ssf/out/sony_profiles.csv`
(own median / p95, A7R IV median / p95, worst light) are reproduced with **maximum difference 0.000** (they are the same
arithmetic on the same inputs; the CSV is rounded to 3 decimals).

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
