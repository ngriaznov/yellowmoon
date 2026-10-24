# Yellowmoon against "raw processor plus ARRI-like LUT": what is different, and what to build next

For Mykyta, 2026-10-14. Research only (written by the research workflow; copied here unchanged apart from one line about the commercial LUT). Scratch paths below refer to the session that ran it. Nothing under `/home/user/yellowmoon` was changed.

This pulls together three studies (the measurement study P1 to P4, the survey, the prototypes) and five checks of my own. `D` below means `/tmp/claude-0/-home-user-yellowmoon/21063226-7ac2-5ae0-bec4-1d31a6166c2b/scratchpad/dyn`. All numbers are against our estimate of the ALEXA, not a real one. Where a number comes from a web-search excerpt, I say so.

## 0. Short answer

1. **You are right about colour.** Against a good raw processor (camera matrix interpolated between illuminants A and D65 by the white-balance temperature, plus a LUT), our colour is better by about 0.1 ΔE00 in the median. Nobody can see that. Against one LUT after one camera matrix, the gap is real but only away from daylight: 1.4 ΔE00 more in tungsten (3.6 more at p90), 0.5 to 1.0 more in fluorescent and LED, 0.27 more in daylight.
2. **Colour has no headroom left.** In every light class we are at 0.24 to 0.35 ΔE00 median. The ALEXA estimate we aim at is itself uncertain by 0.13 to 0.16. A better colour map would not show.
3. **The biggest colour error is not the map.** It is the white estimate. One degree of white error costs about 1.2 to 1.9 ΔE00 (the whole advantage over a LUT, several times over). On real photos we have no ground truth for it. And on bodies without a profile, the light readout is biased: the A1 reads a daylight photo as about 4430 K.
4. **What a LUT cannot do, and what you can see on typical photos, is around the colour.** Highlight rendering (a Sony file has 3.6 stops above grey, the ALEXA 7.4, and ACES 2 gives no real white). Consistency across a series. Shadow colour at high ISO. Plus raw quality, which no LUT test measures.
5. **Ranked recommendation:** (1) calibrate the light readout of stand-in bodies; (2) a per-photo white placement stage ("auto white lift"); (3) a more robust light estimate and a series rule; (4) shadow chroma restraint at high ISO. Do not build: a local (mixed-light) sensor map, a mixed-light balance, a lens/MTF texture match, a learned residual.
6. **The commercial LUT can be tested today.** The cube in `testdata/luts/` and a harness are ready. Section 4 has the commands and the exact Resolve exports. Two cautions there matter: the A1 is a stand-in body for us but not for Resolve, and the harness is set to +0.57 EV where the repository now says +0.85 EV.

## 1. How different is Yellowmoon from "raw processor plus LUT" today?

### 1.1 What was compared

- **P1** is Yellowmoon: the shipped per-light sensor map, ALEXA white, ARRI's CCT matrix.
- **P2** is one camera matrix (D65) and one 33³ LUT, fitted to the reference under D50 to D75. This is the nearest model of "Resolve plus a commercial LUT".
- **P3** is P2 plus a tungsten pack, picked by the white-balance temperature.
- **P4** is a camera matrix interpolated between A and D65 by the white-balance temperature (DNG style), plus a daylight LUT. This is "a good raw processor".
- The static pipelines get exact whites and matrices fitted on the same model they are scored on. They are generous.
- Display is the same for all: +0.57 EV, ACES 2 SDR, Lab D65, CIEDE2000. 11 bodies, 183 held-out measured spectra, 107 lights (tungsten, daylight, fluorescent, LED, real and CIE). The A1 uses the A7R IV profile.

### 1.2 Light known (median / p90 ΔE00; study `results/summary_disp.md`)

| light class | Yellowmoon | one LUT | two LUT packs | dual matrix + LUT |
|---|---|---|---|---|
| tungsten | 0.24 / 0.79 | 1.68 / 4.40 | 0.48 / 1.42 | 0.39 / 1.11 |
| blackbody 4 to 7 kK | 0.28 / 0.83 | 0.44 / 1.17 | 0.44 / 1.17 | 0.35 / 0.79 |
| daylight (CIE D) | 0.27 / 0.82 | 0.54 / 1.34 | 0.54 / 1.34 | 0.40 / 1.03 |
| fluorescent (real) | 0.26 / 0.69 | 0.84 / 2.69 | 0.52 / 1.34 | 0.37 / 0.78 |
| LED (real) | 0.35 / 0.95 | 1.37 / 3.63 | 0.43 / 1.07 | 0.43 / 0.97 |
| class-balanced | 0.28 / 0.85 | 0.83 / 3.12 | 0.49 / 1.28 | 0.38 / 0.90 |

- Share of colours above 1 ΔE00: Yellowmoon 4 to 11 %. One LUT: tungsten 69 %, real LED 62 %, real fluorescent 44 %, daylight 20 %.
- Difference to Yellowmoon (median): dual matrix +0.05 to +0.16 in every class; one LUT +1.44 tungsten, +1.01 real LED, +0.58 real fluorescent, +0.27 daylight.
- By body (class-balanced, P4 minus P1): 0.10 to 0.16 on the good bodies (A7R IV/III/II, A7 III/IV, A7S III, A7C II, A9), 0.04 on the A6400, 0.03 on the RX100 IV, 0.00 on the A7S II.

### 1.3 By colour temperature (A7R IV, median ΔE00)

| light | Yellowmoon | one LUT | dual matrix + LUT |
|---|---|---|---|
| 2125 K | 0.22 | 3.03 | 1.25 |
| 2856 K (A) | 0.22 | 2.00 | 0.30 |
| 3875 K | 0.23 | 1.19 | 0.41 |
| 4250 K | 0.24 | 1.00 | 0.44 |
| 5502 K (D55) | 0.25 | 0.43 | 0.32 |
| 6503 K (D65) | 0.27 | 0.29 | 0.26 |
| 7505 K (D75) | 0.28 | 0.56 | 0.37 |
| 9750 K | 0.30 | 1.07 | 0.90 |

Yellowmoon is flat from 2125 to 9750 K. One LUT passes 1 ΔE00 below about 4250 K and above about 9500 K. At its design light (D65) all of them are equal. The dual matrix clamps at A and D65, so it fails at the two ends.

### 1.4 By subject

Non-daylight lights (tungsten, fluorescent, LED), median (p90), 11 bodies:

| subject | Yellowmoon | one LUT | dual matrix + LUT |
|---|---|---|---|
| ColorChecker 24 | 0.23 (0.82) | 0.98 (3.68) | 0.34 (0.85) |
| skin-like (16) | 0.25 (0.65) | 1.58 (3.17) | 0.41 (0.83) |
| foliage-like (27) | 0.28 (0.69) | 0.77 (1.52) | 0.34 (0.68) |
| saturated, measured (39) | 0.34 (1.03) | 1.47 (3.88) | 0.43 (0.99) |

- Skin under tungsten: Yellowmoon 0.22, one LUT 2.52 (p90 4.05), dual matrix 0.44.
- In daylight (and 4 to 7 kK) the groups differ by 0.1 to 0.4: skin 0.22 against 0.52, saturated measured 0.24 against 0.64.
- The stress set (160 synthetic saturated spectra) is the worst for every pipeline: 0.54 to 0.65 for us, 1.0 to 1.8 for one LUT.

### 1.5 On real photographs (12 local files, survey)

- Against an ideal colorimetric developer plus a nonlinear D55 LUT, Yellowmoon differs by a median of 0.08 to 0.44 ΔE00 per photo (p95 at most 0.57). No photo has a pixel share above ΔE00 2.
- 85 % of the pixels have C*ab below 25, where all pipelines agree (median 0.12 below C* 10, 0.23 from 10 to 25, 0.45 from 25 to 45).
- Against a single D55 map with ARRI at 5600 K, 11 of 12 photos differ by 0.14 to 0.92. One differs visibly: the A7 III tungsten photo (2662 K), median 2.05, 99 % of pixels above 1 ΔE00.
- Ten of the twelve are daylight or mixed light at ISO 100 to 800. Two are high ISO. The set understates artificial light.

### 1.6 Where the difference is invisible

- Every daylight photo, against any decent static pipeline: 0.27 against 0.54 at worst, and 0.08 to 0.44 on the real files.
- Against a raw processor with dual-illuminant matrices, in every class: 0.05 to 0.16, and on the A6400, RX100 IV and A7S II about nothing.
- Whenever the white is estimated (section 1.8): the white error hides the map.
- Whether Resolve's own ARW decode uses CCT-dependent matrices is **unknown** (survey). If it does, you compete with the dual-matrix column. Section 4.7 has a cheap way to find out.

### 1.7 Where the gap is larger: Kelvin different from the scene light

This is the "3200 K under daylight" case of docs/02. Yellowmoon keeps the scene light and the user's Kelvin apart. A raw processor with a LUT has one white balance.

| study (A7R IV, 24 cases, with LUTs, display) | median / p90 |
|---|---|
| Yellowmoon | 0.20 / 0.65 |
| one LUT | 1.37 / 4.30 |
| two LUT packs | 1.08 / 2.43 |
| dual matrix + LUT | 0.98 / 2.16 |

My independent check (own code, linear maps only, scene metric, 138 held-out spectra) gives the same order but a smaller gap to the dual matrix:

| my check, user Kelvin 100 mired or more from the scene (6 cases) | median |
|---|---|
| Yellowmoon (shipped tables) | 0.28 |
| dual matrix A/D65, by the user's Kelvin | 0.57 |
| one fixed matrix | 1.17 |

- D65 set to 3200 K: Yellowmoon 0.30, dual 0.36, one matrix 3.38. D65 set to 2500 K: 0.29, 1.01, 8.47. Tungsten A set to 5500 K: 0.27, 0.64, 0.88.
- So the advantage over a strong static pipeline in this case is **0.3 to 0.9 ΔE00**, at the edge of visibility. The measurement report's "five times in the median" is not reproduced by my linear model (two times). I did not test why. The likely cause is that the study's LUT adds error on strongly cast input, which my model leaves out. Against one fixed map the advantage is large (up to 8 ΔE00 at the extremes).
- This is the clearest thing Yellowmoon does that a LUT workflow cannot, but only when a user moves Kelvin a long way from the light.

### 1.8 The white estimate dominates

- With a perturbed white (model-card errors: median 1.19°, p90 4.74°), class-balanced medians are: Yellowmoon 1.84, dual 1.86, three-pack 1.95, one LUT 2.52. Yellowmoon and the dual matrix are not distinguishable.
- One degree of white error is about 1.4 ΔE00 (1.2 to 1.9 in the repository's own gate study).
- We have no real-photo ground truth for the estimator (Cube++ and NUS-8 are blocked). The gain "estimator beats as-shot white" rests on an assumed as-shot error model. On 9 real files the estimate lies 0.4 to 6.0° from the camera's own white.
- A LUT user sets white balance by hand or takes "As Shot". We have not shown that our automatic white is better than As Shot on real photos.

### 1.9 What I re-checked

1. **The headline table, from raw results.** I re-aggregated all 11 `results/de_*.npz` files and `per_light_disp.csv` without the study's reporting code. Every cell of the class table is reproduced (for example tungsten: Yellowmoon 0.24 / 0.79 / 4.2, one LUT 1.68 / 4.40 / 10.7, dual 0.39 / 1.11 / 4.9). Paired per-colour differences, dual minus Yellowmoon, are smaller than the differences of medians: +0.02 to +0.10. In tungsten and daylight about 11 % of colours are worse than ours by more than 0.5. Against one LUT, 58 % of tungsten colours and 47 % of real-LED colours are worse by more than 1.0. (`D/rec/recheck1.py`)
2. **An independent implementation of the chain** (own code; only data loading shared). Light known, tungsten (A, 2850 K, 2400 K), pooled median: Yellowmoon 0.27, dual A/D65 0.35, one fixed map 2.25, no map 1.51. Daylight (D55, D65, D75): 0.38, 0.38, 0.51. This agrees with the study (dual about 0.1 behind; one map 1.4 to 2.0 behind in tungsten). Section 1.7 is the second half of the same script. My dual matrix has an anchor exactly at A, which flatters it at A. (`D/rec/recheck2.py`)
3. **The stand-in's white.** For the A1, the camera's own ColorMatrix white of D65 is `[0.374, 1, 0.740]` and its WB presets give `[0.389, 1, 0.686]`. The A7R IV curves we use give `[0.336, 1, 0.733]`. Both oracles say the stand-in's red white is 10 to 14 % low. They disagree on blue (the stand-in is 1 % low by the ColorMatrix and 7 % high by the presets). The ColorMatrix and the presets also disagree with each other by 3 to 4 % in red and 4 to 7 % in blue on the A7 III, A7 IV and A1. So the reference for a white calibration has to be chosen, not assumed. (`D/rec/recheck3.py`)
4. **A tempting fix that does not work.** I tried correcting the stand-in's colour response with the body's ColorMatrix (T = CM of stand-in times the inverse CM of the body, at D65; one construction, not tuned), on five bodies whose own curves give the truth. Median ΔE00 over 8 lights, stand-in then corrected: A7 III 0.51 then 0.32 (floor 0.24); A6400 1.21 then 0.59 (floor 0.36); but A7S III 0.52 then 1.22; A7 IV 0.38 then 0.95; RX100 IV 1.42 then 2.06. Two better, three worse. Do not build it. (`D/rec/recheck4.py`)
5. **The tone prototype at the repository's brightness.** Section 2, stage 2. (`D/rec/p_tone_085.py`)

### 1.10 What these numbers cannot tell you

- The target is our ALEXA estimate (24 plausible variants move it by 0.13 to 0.16 median, 0.35 to 0.41 p90, up to 1.9 at the maximum). P1 scores below the dual matrix in 24 of 24 variants in every class except CIE LEDs.
- Spectral scenes only. No noise, no clipping, no content, one display transform.
- The commercial LUT is untested. Resolve's decode is unknown.
- **The A1 is a stand-in body for us.** A stand-in costs up to 2.7 ΔE00 on other bodies (docs/03: the A7S II on the A7R IV profile). For the A1 the size is unknown. Resolve uses the A1's own ColorMatrix. On A1 daylight photos a LUT pipeline may therefore be closer to the truth than we are, and any daylight difference you see in section 4 may be our stand-in.
- The strongest *dynamic* competitor is probably not Resolve. It is Adobe's Adaptive profiles (per-image machine-learned tables, general availability since February 2025; survey, search excerpt). They aim at "good looking", not at ARRI fidelity.

## 2. Dynamic stages worth building, ranked

"Dynamic" here means the processing depends on the photo: its light, content, ISO, exposure, lens or neighbourhood. Today these stages already are:

| stage | depends on |
|---|---|
| C1 to C4 (sensor map, ALEXA white) | the estimated scene light, the user's Kelvin |
| C7 toe | the noise estimate (ISO, per image) |
| R5 highlights, R6 demosaic, R7 denoise | neighbourhoods, σ map |
| R3, R8, R9 | lens tags, f-number, pixel pitch |
| exposure | the camera's gain and EC (fixed, by your decision) |

The learned demosaic and denoise (`jdd`) would add a neighbourhood stage. It is not trained (M2-H4). It is the one learned stage, besides the illuminant estimator, that docs/07 plans.

I ranked by (visibility on typical photos) times (defensibility under the ALEXA target and the V1 data rule), over (cost and risk). Cost: S is 3 days or less, M is 4 to 10 days, L is more than 10 days of agent work.

### Stage 1. Calibrate the light readout of stand-in bodies (S)

- **Evidence.** `D/sheets/proto/D_cct_calibration.jpg`; `D/proto/results/wb_presets.json`, `wb_calibration.json`, `cct_effect.json`. Every ARW carries WB presets at 2500, 3200, 4500, 6000 and 8500 K. The A1's Daylight preset reads 4433 K on the A7R IV locus; its 6000 K preset reads 4983 K. A two-parameter white scale per body cuts the mean white angle from 0.7 to 10.9° to 0.4 to 1.8°. Rendered colour changes by a median 0.75 (A6300), 0.58 (A7 IV), 0.31, 0.30, 0.22 (the three A1 photos), 0.21 (A7 III), 0.02 or less (NEX-7, A500). Against the stand-in LUT (the prototype's own, not the commercial one), the daylight gap on A1M01749, the A7 IV photo and the A6300 photo falls from 0.75, 0.99, 1.15 to 0.42, 0.43, 0.38.
- **What changes per photo.** Nothing per photo. It is a per-body constant. But it is the largest measured colour error on your own camera and it is as big as the whole advantage over a dual-matrix pipeline (0.1 to 0.16). It also makes the Kelvin readout right (an A1 daylight photo shows 4425 K today).
- **ALEXA rationale.** None needed. It protects the ALEXA white and ARRI matrix choice, which both follow the estimated CCT.
- **Data under the V1 rule.** The WB preset tags inside public ARW files, and the ColorMatrix from LibRaw as an independent oracle. No capture of our own. **Open:** the two references disagree (section 1.9, check 3). Pick the ColorMatrix at D65 as the primary reference and use the presets as a consistency check (residual at most 3°), or the other way round. Do not apply a 3×3 correction to the sensor map (check 4).
- **Cost.** S. A `white_scale` per registry body (not per profile: the A1 and the A7R IV share a profile) applied to the raw white before the locus readout, the as-shot prior, Pick white and the estimator input.
- **Risk.** The synthetic A1 scenes are rendered through the stand-in's own curves, so they need the inverse scale or the calibration reads as an error in the golden gates. Real-file outputs for stand-in bodies move. The preset labels may be nominal.
- **Determinism and parity.** A versioned constant in the registry. Same for the app and `ym`. Profile version changes; `pipeline_version` does not need to.
- **Addendum 2026-10-24: more bodies (owner: "Add the scales").** The presets of 85 local ARW files (the three A1 photos, nine local-eval files and 73 Play Raw files, 30 bodies in all) were fitted file by file. Files of one body do not carry the same presets: the A7 III's eight files lie up to 4.7 % in red and 3.5 % in blue from their median, the A6000's ten up to 3.3 % and 3.9 % in eight distinct sets, the A7 II's six in six distinct sets up to 5.9 % and 10.4 %, the A6500's three in three sets up to 6.1 % and 4.1 %. A 1 % change of red or blue moves the Kelvin reading by 1.4 to 1.7 mired, 3 % by 4 to 5 mired. A first survey fitted one scale per body (the median over its files) for all 21 stand-in bodies with files and ran `ym info` on the 51 stand-in files: the mean distance of Auto's tint from the camera's own (as-shot) tint fell from 4.9 to 3.1 over all of them, but the A7 II (3.5 to 4.3), the A6500 (4.8 to 5.6) and the single-file RX100 (5.3 to 12.6) got worse, and the mired distance rose for the A6000 (6.2 to 10.2) and the A6300 (5.9 to 10.8): where the files of a body disagree, one scale cannot be right for all of them. Hence the consistency rule of `docs/03`: a first version asked the files of a body to lie within 1 % of their median and fitted 16 bodies; the same day it was widened to 5 %, the unit-to-unit spread of the measured bodies' files (4.7 % at most), with the scale the median of the files kept, one outlier allowed from three files, and one file enough. That fits 19 bodies (the A6000 on eight files, the A6300 on five of six, the A6500 on two; two A6000 files and one A6500 file are NonCommercial or unlicensed and are not used) and leaves the A7 II (5.9 % and 10.4 %) and the A230 (outside the limits) without a scale; twelve of the 19 rest on one file, which cannot show the unit-to-unit variation. On the 40 local files of the 18 bodies that got a scale, the tint gap of Auto to the camera's own white balance falls from 5.6 to 3.2 and the Kelvin gap in mired does not improve (7.6 to 9.1). The A1's fourth file (Play Raw) lies 6.1 % from the owner's three in blue and is left out.

### Stage 2. White placement per photo ("auto white lift") (S, then M)

- **Evidence.** `D/sheets/proto/B_auto_tone.jpg` (at +0.57 EV) and `D/rec/B_auto_tone_p085.jpg` (my re-run at +0.85 EV, the repository's value); numbers in `D/proto/results/tone_results.json` and `D/rec/tone_results_p085.json`.
  - The clip level is a constant +3.58 stops above grey (2.16 in AWG4) on all 12 photos, because green clips first. So the survey's idea of a shoulder that follows the clip has nothing to follow; what varies is the photo's own highlight distribution, and the stage is sized from that. The C7 shoulder changes under 0.01 % of pixels by more than ΔE00 1. Base dynamic range is only 2.0 to 7.4 stops, so local range compression never starts.
  - ACES 2's shoulder is flat here: 1.4 scene stops lift the display by about 0.4 stop. So the renders have no real white. The 99.5th percentile of our display luma is code 137 to 212 at +0.57 EV (147 to 218 at +0.85 EV), against 161 to 247 in the camera JPEGs.
  - The stage lifts tones above +1 stop by up to A stops, sized so the 99.5th display percentile heads for code 235 in proportion to how high-key the photo is. Mid grey does not move.
  - At +0.57 EV the lift is 0 on two photos, 0.14 to 0.34 on four, 0.85 to 1.43 stops on six. **At +0.85 EV** it is 0 on two, 0.13 to 0.31 on four, and 0.77 to 1.15 on six (A300 0.77, ILCE-5000 0.85, A500 0.88, A1M01749 1.04, A7 II 1.00, A1M01739 1.15).
  - Share of pixels that change by more than ΔE00 2, on those six at +0.85 EV: A300 33 %, A500 4 %, ILCE-5000 21 %, A7 II 13 %, A1M01739 49 %, A1M01749 38 %. On the others 0 to 1 %.
  - Distance of the display percentiles to the camera JPEG's: median over the 12 photos 18.2 code values at +0.57 EV, 14.2 at +0.85 EV, and with the lift 12.3 and 10.2. So the higher placement closes some of the gap, not the missing white.
  - Pixels above 90 % display white on A1M01739: 0 % before, 6.5 % after (camera JPEG 13.6 %).
  - The existing C9 Highlights and White sliders would need 61 to 84 and 31 to 42 on the six lifted photos at +0.85 EV (78 to 94 and 39 to 47 at +0.57 EV, near their limit) and would change 40 to 83 % of the pixels. They are the wrong tool.
  - At +0.57 EV a per-photo-optimal 3D LUT reproduces 81 to 97 % of the change. So it is dynamic across photos, not spatial. A base-layer variant (B2) keeps texture contrast at ×1.08 to ×1.30 instead of ×1.21 to ×1.39.
  - (A hypothesis about the commercial LUT's tone curve is kept out of the repository; the LUT and anything derived from it stay local.)
- **What changes per photo.** The amount A (0 to about 1.2 stops) and the anchor, from the photo's own highlight distribution. Low-key photos get 0.
- **ALEXA rationale.** Partial. An ALEXA at EI 800 holds 7.6 stops above grey (ARRI quotes 7.4), at EI 160 5.3. A Sony file at the fixed +1.11 EV gain holds 3.58, which is 4.1 stops below EI 800. The emulation cannot add data; it can only choose where the shoulder starts. The *amount* is a taste decision, not an ARRI figure. Say so.
- **Data under the V1 rule.** None. Constants (target code, ramp) tuned on public files and judged by blind A/B. I suggest they are **not** fitted to any commercial or ARRI LUT, to keep the licence position simple.
- **Cost.** S for B1: a CPU estimator on the 1/8 preview (70 to 120 ms, two scalars) and about 15 lines of WGSL. M for B2, which reuses the R7 guided-filter kernel.
- **Risk.** Taste. It moves every golden. A shoulder fitted to ACES 2 should apply on the ACES 2 path only, like the +0.85 placement, after the look and curve and before ACES 2, so ARRI's own renderings and the LogC3-to-Rec.709 looks keep ARRI's highlights. Then the DNG does not carry it (state it in docs/09, as for sharpening). Hue shifts and fringes in skies were not seen on the five sheet photos; they were not stressed.
- **Determinism and parity.** The two scalars come from the CPU reference on the 1/8 preview in fixed order, like `AutoEstimates`, and go to the GPU as constants. The app and `ym` run the same code. The kernel gets a CPU reference and a parity test. B2 needs the base layer computed once on the preview and sampled, never per tile, or tiles would disagree.

### Stage 3. A more robust light estimate and a series rule (S to M)

- **Evidence.** `D/sheets/fig_2_estimated_light.jpg` (the white error sets the error; every pipeline is equal beyond 2°). docs/02, review of M1-06: a two-component as-shot prior moves the A7R IV p90 from 4.93 to 4.16°, wrong-preset scenes from 5.9 to 3.1°, and raises the worst case from 22 to 33°. The three A1 photos read 4410 to 4430 K, so the estimator is already stable on a series.
- **What changes per photo.** How far the estimate trusts the camera's as-shot white. And the scene white shared across frames of one scene.
- **ALEXA rationale.** An operator sets one white balance per scene. A LUT is consistent by construction. A per-photo estimator can drift between frames. This protects the one thing a LUT does better.
- **Data under the V1 rule.** Synthetic scenes only. Retrain on CPU (the M1-06 scripts exist) and redraw the brightness prior at the same time (docs/02 found it about five stops low for sunlit scenes).
- **Cost.** S for the series rule (the burst-median pattern exists). M with the retrain.
- **Risk.** The grouping heuristic (time, body, settings) can group unrelated frames. The worst case gets worse. This is your open question 6 in STATUS.
- **Determinism and parity.** A group is "all files given to one call or session". A single-file export in the app after opening a folder then differs from `ym render one.ARW`. There is precedent: exports already wait for shared estimates. Parity tests must use the same file set. Nothing is stored.

### Stage 4. Shadow chroma restraint at high ISO (S, small)

- **Evidence.** `D/sheets/proto/C_shadow_noise.jpg`. Tungsten chart, mean ΔE00 to the noise-free output, no stage then the shipped toe then toe with chroma restraint (k = 6): ISO 3200, −5 EV: 3.32, 3.20, 2.90; −6 EV: 1.33, 1.19, 0.90. ISO 12800, −4 EV: 10.8, 9.9, 8.4; −5 EV: 6.7, 5.4, 3.3; −6 EV: 4.0, 2.6, 1.0. Nothing changes at ISO 800 and below.
- **What changes per photo.** The σ map and ISO. The toe sits 9.6, 7.5, 5.5, 3.5 stops below grey at ISO 100, 800, 3200, 12800, so it is invisible below ISO 3200.
- **ALEXA rationale.** Weak. EI changes the noise and latitude trade, but no ALEXA noise spectrum is public. It is shadow handling, not noise matching.
- **Data.** Public ISO series and the physics priors of docs/06.
- **Cost.** S, about 10 lines in C7.
- **Risk.** Low. Desaturation bias: 1.4 at −4 EV (ISO 3200) and 4.8 at −4 EV (ISO 12800). The experiment had no demosaic and no denoise, and read noise was nominal. A real denoiser may do most of this. Build it after judging real high-ISO photos, on the A1 ISO ladder.
- **Determinism and parity.** Uses the σ map already on both paths. Same kernel rules as C7.

### Not recommended for V1

- **Local (mixed-light) sensor map.** `D/sheets/proto/A_local_light.jpg`. With the true mixture known it gains almost nothing (0.24 to 0.22). With the shipped estimator per window it loses (0.30 at k = 10; blotchy casts, A+D65 goes from 0.20 to 1.47). The window white error is 1.16° median and each degree costs about 1.4 ΔE00. Worse than global by more than 0.05 in 32 % of scenes, better in 3 %. On the 12 real photos 0 of 25 windows pass the shrink test, so it would stay off. M to L, high risk.
- **"Balance mixed light" (a per-pixel white).** Not ALEXA behaviour (an ALEXA has one white balance), and it needs the same local estimator that fails.
- **Lens and f-number texture matching.** No ALEXA MTF is public.
- **Learned residual or look transfer.** Needs paired ALEXA images. None are allowed or available. The remaining colour headroom is small (a per-light nonlinear correction gains fluorescent 0.22 to 0.14 to 0.17 and LED 0.31 to 0.11 at best, nothing in tungsten or daylight).
- **More work on the light class.** A map chosen by CCT alone is as good (0.24 tungsten, 0.27 daylight, 0.29 fluorescent, 0.30 LED against 0.24, 0.27, 0.26, 0.34 with classes). Two anchors (3000 K blackbody and 6500 K daylight, interpolated in mired) give 0.28 to 0.36 against 0.24 to 0.35. Classes are not where the value is.
- **A ColorMatrix correction of the stand-in's sensor map.** Section 1.9, check 4.

## 3. Does the exposure decision interact? Questions, not a decision

Your decision (docs/16, 2026-10-14): fixed camera gain +1.11 EV, no metering of our own, display placement after the look. The reports used +0.57 EV; the repository says +0.85 EV (docs/16 row of 2026-10-14, STATUS). See question 1 in section 6.

1. **Stage 2 does not touch the fixed gain.** It works above +1 stop and leaves mid grey where it is. But it is a per-photo tone decision made by us, which the fixed gain avoided. Do you read your rule as "do not meter mid grey" or as "no scene-adaptive tone at all"? If the second, stage 2 can only be an offset (default 0) in the Light tab.
2. **Should the lift show in the Automatic card** ("White +0.9 EV", like "Exposure +1.11") and be overridable? That would make it the scene-adaptive part of "automatic settings" without touching exposure.
3. **Do you want a scene-adaptive exposure instead of the fixed gain?** The data argues against. The camera's gain varies by 0.13 EV (std) across scenes and bodies, while its AE decides the key. Metering to a fixed grey was wrong on the four key-extreme files out of 12 (research/auto-exposure). A white-anchored exposure ("expose to the right") would lift dark-key scenes, which is the opposite of your decision. It was not tested. I raise it only because you may want a scene-adaptive stage "instead of" the gain.
4. **At +0.85 EV the missing white is smaller but still there** (section 2, stage 2). Is the lift still wanted at that placement? I think yes: 6 of 12 photos still lift by 0.77 to 1.15 stops.
5. **The highlight guard cannot act.** Its level, 8.0, stands for the ALEXA's headroom. At the fixed gain the Sony clip is at 2.16 in AWG4, so no unclipped pixel can exceed 8.0 and the guard cannot fire (it did not on any of the 12 files even under the old rule). Keep it as a safety net or remove it?
6. **Exposure compensation reaches the picture in full and the clip is in the raw.** A photo shot at +2 EC clips earlier, and no stage can restore a clipped raw. Is "the Exposure slider fixes it" still the intended answer? (Your decision says yes.)

## 4. The commercial LUT comparison

### 4.1 Before you start

- The only cube on the machine is `/home/user/yellowmoon/testdata/luts/Neutral_Burano_G1_65x.cube` (65³, titled "Generated by Resolve", no shaper, black at 0.0141, git-ignored). STATUS calls your commercial LUT "S-Gamut3.Cine/S-Log3 in". The probe cannot confirm the input space. **Please confirm that this is the file, and the vendor's stated input and output spaces.**
- Everything stays local. The harness writes only under `D/lutcmp/out/<lut name>/`. Do not commit the LUT, its derived tables, or sheets that show its behaviour. Aggregate numbers are probably fine for STATUS. That is your call (question 3, section 6).

### 4.2 Step A: the model against the LUT (no Resolve needed, about 2 minutes per run)

```
cd /tmp/claude-0/-home-user-yellowmoon/21063226-7ac2-5ae0-bec4-1d31a6166c2b/scratchpad/dyn/lutcmp
python3 -I lutcmp.py --selftest        # about 3 min; must print PASSED
python3 -I lutcmp.py --lut /home/user/yellowmoon/testdata/luts/Neutral_Burano_G1_65x.cube --probe
python3 -I lutcmp.py --lut /home/user/yellowmoon/testdata/luts/Neutral_Burano_G1_65x.cube --lut-input slog3-sgamut3cine --lut-output rec709-g24
# the next two use the same --lut, --lut-input and --lut-output flags as the line above
python3 -I lutcmp.py --lut <cube> --lut-input slog3-sgamut3cine --lut-output rec709-g24 --kelvin 3200 --files A1M01749 A1M01739   # the case a LUT cannot follow
python3 -I lutcmp.py --lut <cube> --lut-input slog3-sgamut3cine --lut-output rec709-g24 --matrix dual                              # against a better raw processor
```

- For a log-to-log LUT (LogC out), use `python3 -I ../eval_commercial_lut.py --cube /path/outside/the/repo/pack.cube --in slog3-sgamut3cine --out logc3-awg3 --matrix single --align neutral` (it also takes `--matrix dual`, `--body` and `--align none`). That script works on synthetic spectra only. `lutcmp.py` is the one for display LUTs and for photos.
- Read three columns, not the raw one: "neutral axis matched", "chroma and hue at equal lightness", and the pattern across lights.
- Expected values for a neutral static LUT (the harness's own stand-in, placement matched, 36-patch charts, median ΔE00): D65 0.23, D55 0.32, A 3.00, LED-RGB1 2.59, FL3.8 1.37, LED-B3 1.32. User Kelvin 3200 on D65: 3.2, against 0.5 at 5500 K. Real photos 0.18 to 1.15, and 1.85 for the 2657 K A7 III.
- The scratchpad is session-local. If the session ends, `lutcmp/` goes with it. Task D-05 below moves it into the repository.

### 4.3 Step B: what to export from Resolve

For each photo, from photos you **already have** (no new shoot; V1 rule):

1. `<stem>_log.tif`: 16-bit, full range, full resolution (at least 1440 px wide, same framing), in the LUT's input space, **no LUT, no grade, no noise reduction, no sharpening, no resize, no tone or gamut mapping**. Raw settings: white balance As Shot (or Custom, and say which Kelvin and tint), every other raw slider neutral. Write down the Exposure value.
2. `<stem>_final.tif`: the same clip with the LUT, as you normally grade it, rendered to Rec.709 Gamma 2.4 (say if you use sRGB or Gamma 2.2 instead). 16-bit, full range, same framing.
3. A short note: Resolve version, Photo page or Color page raw decode, colour management on or off and its settings, where the LUT sits (before or after a CST), the LUT's name and what the vendor says about its input and output.

Which photos: the three A1 files first (`A1M01749`, `A1M01739`, `A1M01454`). Then, if you have them: one tungsten photo, one LED or fluorescent photo, one mixed-light photo, one ISO 3200 or higher, one daylight portrait with skin. File names must be the photo's stem (`A1M01749.ARW` gives `A1M01749_log.tif`).

### 4.4 Step C: run with your exports

```
python3 -I lutcmp.py --lut /home/user/yellowmoon/testdata/luts/Neutral_Burano_G1_65x.cube \
  --lut-input slog3-sgamut3cine --lut-output rec709-g24 \
  --resolve-log-dir DIR_A --resolve-dir DIR_B --resolve-gamma g24 \
  --files A1M01749 A1M01739 A1M01454
```

The harness applies the LUT itself to your log render and prints a CHECK line against your final export. It must read below 0.5 ΔE00 median. A larger value means the input or output space is wrong, or Resolve did more than the LUT. Do not read any other number until the CHECK passes. For extra photos add `--arw /path/to/photo.ARW` (decoded in a separate process).

### 4.5 How to read the result (rules fixed before the numbers exist)

- **Matched ΔE00 at daylight (D55 or D65 charts, daylight photos) at or below 0.5:** our target and the LUT agree. The gap in artificial light and in Kelvin tests is then the measure of the dynamic part.
- **Matched ΔE00 at daylight above 1.0:** either our ALEXA estimate is off or the cube carries a look. A real LUT built from paired captures is an independent check on our estimate (docs/16 risk 1). Do not conclude anything about dynamics from that LUT until the vendor says which it is.
- **Artificial light (A, LED, fluorescent charts) with a gap of 1.0 or more against 0.5 or less in daylight:** the cube behaves like our P2. This is the model's expectation.
- **A gap in artificial light of 0.4 or less:** Resolve's decode (not the LUT) is doing the adapting. Section 4.7 tells you which.
- **A1 photos:** any daylight difference may be our stand-in (section 1.10). Check the three A1 photos against `--wb-calibration presets` (it exists in the harness) before blaming the LUT.
- **Zones.** Report medians, p90, and the share above ΔE00 1 and 2, by luminance zone (below −2 stops, mid, +1 to +3 stops, top two stops) and by chroma bin (below 10, 10 to 25, 25 to 45, above 45). Tone placement differs between the two by design; the zone view shows where.
- **A blind A/B** on the same photos, order shuffled, brightness matched, at least 20 photos with at least 5 in artificial light. Pass if Yellowmoon is preferred or tied at least 50 %. Pictures stay with you.

### 4.6 Two cautions

1. **Placement.** The harness hard-codes the display placement at +0.57 EV (`PLACEMENT_EV` in `lib/dyn.py`; `lutcmp.py` has no flag for it). The repository says +0.85 EV. The raw "YM vs LUT" column therefore carries 0.28 EV of brightness that does not exist in the app. The matched columns are not affected. Someone should add a `--placement-ev` flag.
2. **Licence and scratch.** Keep the cube, outputs and any sheet from it out of the repository.

### 4.7 A cheap test of how Resolve decodes a Sony ARW

Export the same daylight photo's `_log.tif` three times (no LUT): White balance As Shot, then 3200 K, then 8000 K. If the colour change between them follows one fixed camera matrix, Resolve is single-matrix (our P2 and the one-LUT column apply). If it follows `--matrix dual`, Resolve interpolates matrices by temperature (our P4 applies, and the colour advantage is about 0.1 ΔE00). Comparing the three exports against the harness's log render for `--matrix daylight` and `--matrix dual` is about 20 lines. It is not built. It settles the largest unknown in section 1.

## 5. Spec changes and milestones

Proposed tasks, in the format of docs/15. Dependencies in brackets. [HUMAN] tasks are yours (AGENTS rule 9).

**D-01 · Light readout calibration for stand-in bodies** (M1-04, M1-07)
`white_scale: [r, 1, b]` per registry body, applied to the raw white (locus readout, as-shot prior, Pick white, estimator input). Reference and consistency check recorded in `provenance`. Synthetic stand-in scenes get the inverse scale.
*Done when:* on the local bodies the Kelvin readout of a daylight preset is within ±300 K of the reference; the mean preset white angle is at most 2° on every body; the A1 daylight photos read about 5000 to 5600 K; the sensor-map gate is unchanged; goldens updated with a note.

**D-02 · White placement** (M1-09, M1-11, wait for the +0.85 placement step)
The two scalars estimated on the 1/8 preview; a scene-linear shoulder above +1 stop on the ACES 2 path; "White" row in the Automatic card; "White" offset in the Light tab; CLI option. B2 (base layer) later.
*Done when:* scene 0.18 lands on the same display value to 1e-6; nothing changes below +1 stop; per channel monotone and ratio-preserving; GPU against CPU within 2e-4 (LogC4 units); golden hashes updated; the owner's blind A/B on at least 12 photos decides default on or off.

**D-03 · Robust white and series consistency** (M1-06, task for STATUS question 6)
Two-component as-shot prior and brightness prior redrawn, one retrain; a series rule for the scene white.
*Done when:* A7R IV p90 at most 4.2° and the good-as-shot case not worse than +0.05°; a series test gives identical results in the app and `ym` for the same file set.

**D-04 · Shadow chroma restraint** (M1-09, M2-02)
A chroma term in the C7 toe from the σ map.
*Done when:* ISO 12800 at −5 EV improves by at least 1.0 ΔE00 on the noise chart; no change at ISO 800 and below; judged on real high-ISO photos after the denoiser.

**D-05 · "Beats a static LUT" gate and harness in the repository** (M1-15)
Move the P1 to P4 study and `lutcmp` into `tools/eval/dynamic/`. The synthetic part builds its own P2 and P4 LUTs (no commercial file) and runs in CI. The real-LUT part stays local.
*Done when:* CI prints the table of section 1.2; the thresholds below are in `docs/13`.
Proposed thresholds (yours to set; AGENTS says gates change only with your agreement): per class, Yellowmoon median at most 0.35 and p90 at most 1.1 ΔE00 (today 0.24 to 0.35 and 0.69 to 1.03); margin to one LUT at least 0.4 median in tungsten, fluorescent and LED (today +1.44; +0.49 to +0.58; +0.79 to +1.01); the dual-matrix gap reported, not gated.

**[HUMAN] DH-1** confirm the cube and its spaces; make the Resolve exports. **DH-2** blind A/B against Resolve plus the LUT. **DH-3** decide the default of the white lift. **DH-4** set the gate thresholds.

Spec text to change when these land:

| doc | change |
|---|---|
| `docs/00` | "The point of the app": replace "being defined" with the gate. State plainly where the dynamic part is visible (artificial light, Kelvin away from the light, tone, shadows) and where not (daylight). |
| `docs/02` | "Idea in one paragraph": add the measured size (about 0.1 against a dual-matrix pipeline, 1.4 against one LUT in tungsten). C1: `white_scale`. New "C7b White placement". C7 low light: chroma term. Auto exposure: note the guard cannot act at the fixed gain. |
| `docs/03` | Registry `white_scale`; the ColorMatrix test as a recorded negative result. |
| `docs/05` | R7: B2 reuses the guided filter. |
| `docs/06` | σ map drives the C7 chroma term. |
| `docs/07` | One illuminant retrain (robust prior, brightness prior). No new model. `jdd` stays the only learned neighbourhood stage. |
| `docs/09` | DNG does not carry the white lift (like sharpening). |
| `docs/10`, `docs/11` | CLI option; Automatic card row; Light tab offset. |
| `docs/13` | The new gate. |
| `docs/16` | Decision-log rows; the +0.57 against +0.85 question. |

## 6. Questions for you

1. **Which brightness is current, +0.57 or +0.85 EV?** The task text I was given and all three studies used +0.57. docs/16 (2026-10-14) and STATUS say +0.85 and note that +0.57 was "considered and replaced". I re-ran the tone prototype at +0.85 (section 2). The colour tables are not affected: all pipelines share one display.
2. **Is `Neutral_Burano_G1_65x.cube` the commercial LUT?** What does the vendor state as its input and output space, and is it a neutral LUT or a look?
3. **May aggregate ΔE00 numbers from the commercial LUT go into STATUS?** Pictures stay with you.
4. **Which Resolve flow do you use?** Photo page or Color page, As Shot or manual white balance, a CST or not. And does your Resolve change colour when you move Kelvin (section 4.7)?
5. **What do you want to claim as the selling point?** Colour in artificial light and with the Kelvin moved far from the light is real but narrow. Daylight colour is a tie with a good LUT. Per-photo white and shadow handling, an automatic white that beats As Shot (unproven), and raw quality are the other candidates. Colour cannot carry the claim alone.
6. **Stage 2 as default or offset?** Is it acceptable that its amount is a taste decision and not an ARRI figure? Should its constants stay independent of any LUT?
7. **White calibration reference.** ColorMatrix at D65 (LibRaw table, Adobe's numbers) or the camera's WB presets? Is shipping a per-body constant derived from them acceptable under `docs/14`? The two references disagree by 4 to 7 % in blue.
8. **Adopt the robust as-shot prior** (STATUS question 6; worst case 22° becomes 33°), and a series rule that groups "files in one call"?
9. **The A1 stand-in.** Resolve uses the A1's own matrix; we use the A7R IV curves. Do you want the A1 comparison read with that in mind, or a V2 A1 measurement brought forward?
10. **Which ALEXA is the target?** The classic ALEV III matrices (today) desaturate highlights in ARRI's Rec.709 chain; REVEAL on the LF and Mini LF keeps colour into hot highlights (survey, search excerpt). REVEAL is out of scope in docs/00. Confirm.

## Files

- Studies: `D/README_p1p4.txt`, `D/results/summary_disp.md`, `summary_scene.md`, `ablation.md`, `sensitivity.md`, `wb_mismatch.md`, `per_light_disp.csv`; `D/proto/RERUN.txt`, `D/proto/results/*.json`; `D/lutcmp/USAGE.txt`.
- My checks: `D/rec/recheck1.py` (class table from raw results), `recheck2.py` (independent chain and Kelvin test), `recheck3.py` (preset whites against ColorMatrix), `recheck4.py` (ColorMatrix correction of the stand-in), `p_tone_085.py` with `tone_results_p085.json` and `B_auto_tone_p085.jpg`.
- Sheets: `D/sheets/` (`sheet_0` to `sheet_3`, `fig_1`, `fig_2`), `D/sheets/proto/` (`A` to `E`, `contact_baseline`).
