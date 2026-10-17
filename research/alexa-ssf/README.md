# ALEXA (ALEV III) spectral sensitivity, estimated from ARRI's published matrices

An open, shippable stand-in for the ALEXA sensor's spectral sensitivities, for the Sony A1 → ALEXA emulation project. **It is an estimate, not a measurement.** It is fitted so that the colour matrices these curves would receive match ARRI's own ALEXA matrices at 17 colour temperatures, using a spectral prior built from 52 measured camera curves.

## Files

| File | What it is |
|---|---|
| `out/sony_profiles.csv` | The sensor-map result below (each body's own profile and the A7R IV profile against the ALEXA estimate, 138 colours, 9 lights). `crates/ym-profiles/tests/sensor_map_eval.rs` and `training/sensor_map/evaluate.py` reproduce it |

The curve itself ships in `data/alexa/` (`alexa_alev3_estimate_380_780_5.json`, the `.csv` with the ensemble band, `variants.csv`, `target.json`). The scripts live in `training/alexa/` (estimate and leave-one-out validation) and `training/sensor_map/evaluate.py` (the sensor-map result).

Channel scale is not identifiable from white-balanced matrices, so each channel is normalised to peak 1. That doesn't matter for emulation, because both cameras are white-balanced before mapping and only the curve shapes enter.

## Method

1. **Inputs:**
   - ARRI's ALEXA SUP 3.x camera-native → ACES AP0 matrices at 17 CCTs, 2000–11000 K (aces-dev v1.3.1, ARRI-supplied, AMPAS license).
   - 52 measured camera curves and the 190 training reflectances from rawtoaces-data (Apache-2.0).
2. **Forward model:** Academy P-2013-001 as implemented by colour-science. White-balanced camera RGB, scene colour CAT02-adapted to the ACES white, white-preserving 3×3 chosen to minimise CIE Lab error. The illuminant is blackbody below 4000 K and CIE daylight above, the rawtoaces convention.
3. **Prior:** per-channel PCA of the 52 curves; 4 components with a Gaussian penalty.
4. **Fit:** find the curves whose own optimal matrices equal ARRI's at all 17 CCTs, via a Gauss–Newton step from ARRI's matrix with an analytic Jacobian. This objective is minimised exactly at the true curves when the recipe matches.
   - A first attempt asked instead that ARRI's matrices render the candidate colorimetrically. That objective turned out biased: it prefers "more colorimetric" cameras over the true one, so it was dropped.
5. **Ensemble:** 24 variants: prior with all 52 / without Sony / without the D-21, 3 or 4 components, two penalty strengths, two illuminant models. The recommended curve is all 52, 4 components, penalty 1, rawtoaces illuminants, which validated best.

## Validation

**Leave-one-camera-out.** Each of the 52 measured cameras was given its own 17 CCT matrices. Its curves were then estimated from those matrices alone, with a prior built from the other 51:

| | RFSE median | RFSE p90 | Emulation error, ΔE00 median | ΔE00 p95 (median over cameras) |
|---|---|---|---|---|
| Estimate (illuminant model known) | **2.2%** | 3.6% | **0.23** | 0.63 |
| Estimate (wrong illuminant model assumed) | 2.6% | 4.0% | 0.26 | 0.75 |
| Average camera curve (no fitting) | 4.4% | — | 1.27 | — |

- **RFSE** is RMS curve error relative to peak, averaged over R, G and B.
- **Emulation error** is the colour difference between rendering CIE's 99 test samples (CES) through the true and the estimated curves, under 9 lights from tungsten to RGB LED, with the same matrices applied to both.

**Failure mode.** The estimator cannot leave the span of its prior. The ARRI D-21 is unlike the stills cameras in the set; its matrices could only be reproduced with an RMS error of 0.10 per coefficient. Its estimate was no better than the average curve: 8.9% RFSE, emulation ΔE00 p95 6.6.

**Where the ALEXA sits.** ARRI's matrices are reproduced to 0.0067 RMS per coefficient.
- That is the 94th percentile of the held-out cameras (median 0.0021), and 15× better than the D-21 case.
- The held-out cameras with similar residuals were estimated at **3.2–5.3% RFSE**, so that is the realistic accuracy to expect here, not the 2.2% median.
- Part of the extra residual is probably ARRI's own matrix recipe (training set, rendering intent), which isn't public.

**Ensemble spread.** All 24 variants lie within median 1.7% and max 2.6% RFSE of the recommended curve. This includes the prior with all Sony bodies removed, so the curve's closeness to Sony is not an artefact of the Sony-heavy prior.

**D-21 check (confirms the earlier report).** ARRI's matrices were applied to each camera on the 99 CES samples under the 9 lights, and the result compared with true scene colour (mean ΔE00):

| Camera | Mean ΔE00 | Rank of 54 |
|---|---|---|
| Estimate | 1.45 | 1 (partly by construction) |
| Sony A7R IV | 1.77 | 7 |
| Average of 52 curves | 1.95 | 15 |
| ARRI D-21 | 3.95 | 53 |

## What the curve says

- **Peaks:** R 590, G 530, B 460 nm, the same as the A7R IV. The two curves are 4.4% RFSE apart, while each is about 9% from the D-21.
- **Broader red shoulder than the A7R IV** from 610 to 660 nm: 0.44 vs 0.26 of peak at 650 nm. That fits ARRI's statements that the ALEXA keeps some far red for skin.
- **Beyond about 660 nm the curve is mostly the prior.** Human colour matching is nearly blind there, so the matrices barely constrain it. The estimate falls to near zero by 690 nm, but ALEXA's hot mirror is reported near 700 nm. Treat 660–720 nm as unknown (the ensemble spans 0–0.08 at 680 nm).

## First result for the sensor map: A7R IV → ALEXA estimate

Fitted on the 190 training reflectances, scored on the 99 CES. Both images are rendered through ARRI's matrix at each light's CCT. Values are ΔE00 median / p95.

| Light | No mapping | One 3×3 (5500 K) | 3×3 per light | Root-poly per light |
|---|---|---|---|---|
| Tungsten (A) | 1.39 / 2.88 | 1.19 / 2.62 | 0.36 / 0.85 | 0.33 / 0.82 |
| Studio tungsten (ISO 7589) | 1.38 / 2.71 | 1.01 / 2.25 | 0.36 / 0.86 | 0.33 / 0.82 |
| D55 | 1.43 / 2.93 | 0.42 / 1.06 | 0.42 / 1.06 | 0.31 / 0.93 |
| D65 | 1.59 / 3.26 | 0.49 / 1.15 | 0.45 / 1.09 | 0.32 / 0.98 |
| Fluorescent FL2 | 1.35 / 2.63 | 0.65 / 1.05 | 0.28 / 0.91 | 0.21 / 0.88 |
| Tri-band fluorescent FL11 | 1.30 / 2.23 | 0.67 / 1.03 | 0.16 / 0.53 | 0.12 / 0.49 |
| White LED (LED-B3) | 1.37 / 2.59 | 0.64 / 1.16 | 0.24 / 0.93 | 0.23 / 0.82 |
| Hybrid LED (LED-BH1) | 1.43 / 2.72 | 1.16 / 2.77 | 0.20 / 0.68 | 0.20 / 0.64 |
| RGB LED (LED-RGB1) | 1.39 / 3.33 | 1.13 / 2.73 | 0.25 / 0.80 | 0.22 / 0.72 |

**Readings:**
- **The sensor difference itself is moderate:** about 1.4 ΔE00 median and up to about 3.3 at p95 with no mapping.
- **Conditioning on the light is the big win.** One global 3×3 works only near the light it was fitted at; tungsten stays at 1.19 / 2.62. A 3×3 chosen per light brings every case to 0.16–0.45 median and at most 1.09 at p95.
- **Non-linearity adds little.** Root-polynomial per light gains another 0.00–0.13 median. A small model that predicts per-light coefficients from the illuminant estimate is enough; a large per-pixel network isn't justified by this data.
- **Caveat:** the estimate is pulled toward the prior, so real Sony–ALEXA differences are probably somewhat larger than shown. Reflectance sets also miss narrow-band object colours.

## Caveats

- This is an estimate with an expected 3–5% RFSE and no independent ALEXA measurement to check it against yet. Label it as such in the app and docs.
- It represents ALEXA SUP 3.x / ALEV III processing (Classic through Mini LF and 65). It does not cover the ALEXA 35 or REVEAL.
- The far-red tail above about 660 nm is not determined by the data.
- The leave-one-out tests assume ARRI made its matrices the P-2013-001 way. A different ARRI recipe would add error not captured here; wrongly assuming the illuminant model cost +0.4% RFSE in the test.

## Checks not done

1. Download HdM's `camera_responses.zip` in a browser and compare its ALEXA SXT curve against this estimate and against ARRI's 17 matrices (`ssf_core.camera_matrices` does this in one call).
2. Digitise Fig. 1 of Leonhardt & Brendel (CIC 2015) and compare. It is ARRI's own five-camera mean.
3. If ARRI's Reference Tool license allows, recover ALEXA white-balance gains from a CCT sweep. That is the strongest missing constraint.

## Reproduce

```bash
tools/fetch/spectral.sh                       # the inputs, into testdata/ (pinned commits, SHA-256 checked)
python3 -m training.alexa.validate_loo        # leave-one-camera-out, then: python3 -m training.alexa.check_loo
python3 -m training.alexa.estimate_alexa      # the curve and its ensemble, then compare_estimate against data/alexa/
python3 -m training.sensor_map.evaluate       # reproduces out/sony_profiles.csv
```

colour-science 0.4.7 (`training/README.md`).

## License

The scripts and the derived curves are offered under Apache-2.0. The curves are derived from AMPAS-licensed ARRI matrices and Apache-2.0 rawtoaces-data; the repository's `NOTICE` (items 1 to 3) carries the required attributions. No endorsement by ARRI or A.M.P.A.S. is implied.
