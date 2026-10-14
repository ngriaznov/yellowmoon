# Auto exposure and the dark look: recommendation

For Mykyta, 2026-10-14. Research only. Nothing under `/home/user/yellowmoon` was changed. Local evaluation files (the 9 public test files and your three A1 photos) were used for measurement only and none was copied into the repository.

AE = the study's local working folder (scripts, renders and contact sheets; not committed: the sheets show the local evaluation files). The numbers can be reproduced with `tools/eval/autoexp.py` on the same files.

## Short answer

- **The photo is dark mainly because of the display transform, not because of auto exposure.** A1M01749 reads 1.16 EV darker than its own camera JPEG. 0.85 EV of that is where ACES 2 puts mid grey. 0.30 EV is our exposure sitting 0.27 EV under the camera's. Contrast and measurement error add about nothing.
- **Metering is a small part.** On this photo the scene plays no role in the exposure gap. It is a constant offset: our target of 0.18 sits 0.23 EV under where cameras typically put the same statistic. Scene key matters on bright and dark scenes (four of 12 files miss the gate because of it), but not on yours.
- **Recommended default: two small, separate changes.** (1) A fixed display placement of +0.35 EV, after the look and before ACES 2, ACES 2 path only. (2) An exposure rule that keeps half of the camera's own placement: `EV = 1.11 + 0.5 · (L − 1.11)`. For your photo this is +0.35 EV of placement and +0.21 EV of exposure. The median display code goes from 94 to 115 (camera JPEG: 138). The M1-08 gate goes from 8 of 12 to 12 of 12 files.
- **It will still be darker than the camera JPEG, on purpose.** Matching the JPEG needs about +1.2 EV, and the result is brighter but flatter. A brighter variant (+0.57) is on the sheet for you to judge.
- **The +0.35 is a taste decision with weak sources.** ARRI's own quoted grey is 38 to 40 % video (secondary sources). +0.35 is 43 %, which is inside the range Art Adams measured (42 to 46 %). It is not an ARRI-documented figure. Your imported ARRI Rec.709 LUT can settle it in one command (question 3).

## 1. Why the photos look dark

### 1.1 The three parts, for the screenshot photo (A1M01749)

Camera: Sony A1, ISO 100, 1/1600, f/2.8, aperture priority, EC 0. `ym` chose EV_auto +0.68. Numbers are in EV; negative means we are darker than the camera JPEG. G2 is measured on the display-referred render.

| part | what it is | EV | source |
|---|---|---|---|
| Display placement | ACES 2 SDR puts scene 0.18 at display 0.100 (sRGB code 89). The camera JPEG puts it at 0.18 (code 118). | **−0.85** | design of ACES 2. The same on all 12 files (−0.84 to −0.86). |
| Our exposure | EV_auto +0.68 against the camera's own gain of +0.95 (G3 −0.27), seen through our display curve (local slope 1.10) | **−0.30** | metering |
| Contrast | our curve is flatter at grey (slope 1.17 to 1.20) than this camera's style (1.30) | −0.05 | camera style |
| Measurement residual | model against measured median | +0.05 | |
| **Total (G2)** | | **−1.16** | measured render: −1.16 to −1.19 |

So about three quarters of the darkness is the display placement and one quarter is exposure.

### 1.2 What is in the exposure part

The exposure gap G3 splits exactly into four terms (check error 1e-16):

| term | meaning | A1M01749 | all 12 files |
|---|---|---|---|
| a | our target 0.18 against where the 12 cameras put the metered statistic on average | −0.23 | −0.23 on every file |
| c | this scene's key as its camera exposed it, against that average | −0.05 | std 0.45, range −0.53 to +0.97 |
| d | highlight guard | 0.00 | 0.00 on every file |
| e | half the camera's exposure compensation | 0.00 | 0.00 (all files have EC 0) |

For your photo the whole gap is the constant `a`. The scene's key adds almost nothing.

### 1.3 What the scene does (the part that is not constant)

Our meter sends every scene to the same grey (the geometric mean goes to 0.18). The camera does not. It exposes the scene and then applies a nearly constant gain (+1.11 EV median, std 0.13, range +0.87 to +1.33, no relation to the scene key). We then normalise the key a second time on top of that. The result:

- **Bright scenes come out too dark.** A1M01739 (a bright sky, camera JPEG median code 195): EV_auto +0.38 against the camera's +1.14, G3 −0.76, G2 −1.57. A300 (flash portrait on a bright wall): G3 −0.72, G2 −1.73.
- **Dark scenes come out too bright.** DSLR-A500 (dark brick): EV_auto +1.90, G3 +0.74. NEX-7 (dark tiles): +1.61, G3 +0.52. A1M01454: +1.34, G3 +0.48 (just passes).
- EV_auto against log2(JPEG median) has slope −0.44 EV per EV (r −0.90).

These four files are the four that miss the gate or nearly miss it. The camera JPEG cannot tell a deliberately dark scene from an underexposed one, and two of the dark files were shot with DRO Auto, which the camera-gain estimate reads as part of its curve. So the sign of the true error there is not certain.

### 1.4 What is not the cause

- **Highlight guard.** It lowered EV_auto by 0.00 on all 12 files. Its level (8.0) stands for ALEXA's 7.4 stops of headroom. After exposure the 99.5th percentile pixel of a Sony frame sits 2.3 to 4.7 EV below it.
- **Look and C7 (highlight roll-off).** The Neutral look is an identity. Moving the roll-off from 0,0 to 1,1 moves the display median by at most 0.04 EV (NEX-7), and by at most 0.007 EV on the other six files tested.
- **Centre weighting, percentiles and the burst median.** None matters at this size (docs/02 evaluation).

### 1.5 The display range is barely used

After auto exposure, the brightest 1 % of our pixels reach display code 114 to 198. The camera JPEGs reach 156 to 245. ACES 2 needs scene light +5.7 EV above grey to reach display 0.90 (checked with OpenColorIO). The A1 photo, metered to 0.18, clips about +3.2 EV above grey (survey, derived). On the neutral axis a scene value of 1.44 (+3 EV) reaches only code 197. So there is no real white in our renders. Raising exposure to match the JPEG median clips nothing (0.00 % on all 12 files), so there is room.

### 1.6 What matching the JPEG would take

+1.16 EV over Auto at the median (range −0.25 to +2.11) to bring the display median to the JPEG's. That is option D below. It reads brighter and flatter, breaks the rule that grey sits at LogC4 0.2784 before the look, and the exposure then misses the camera's gain by +0.97 EV. It is the wrong target for this product.

### 1.7 Limits of this evidence

- The camera JPEG is a proxy for Mac Preview. Quick Look shows the embedded JPEG. Preview.app may decode the raw itself. That cannot be measured here.
- 12 files from 9 bodies, 9 of unclear licence. Camera gains are good to about 0.1 EV. Only luminance was measured; JPEG colour differs visibly from ours (sky, white balance).
- All 12 files have EC 0. Nothing here tests exposure compensation on real data.

### 1.8 What I checked myself

- I re-rendered A1M01749 with the stable `ym` at three exposure offsets (`AE/check/check_render.py`). Display median code and top-1 % code: Auto 94 and 151, +0.21 gives 101 and 159, +0.35 gives 107 and 164. These agree with the sheets and with the tables.
- I recomputed the rules from `AE/study/real12/results.json` without the study's candidate scripts. Camera gain median +1.111 (std 0.126). Today: 8 of 12 within ±0.5 EV, median −0.23. Anchor rule with w 0.5: 12 of 12, median −0.07, range −0.47 to +0.36. Fixed +1.11: 12 of 12, std 0.13. Target +0.33 EV: 9 of 12. Anchor to +0.79 (w 0.5): 10 of 12. These match the reports. Only difference: I get a spread of 0.25 for w 0.5 (population std) where the study has 0.27 (sample std, replica EVs).
- I checked ACES 2 against OpenColorIO directly. Scene 0.18 gives display Y 0.1000 (code 89, 38.3 % at gamma 2.4). +0.35 EV gives 0.132 (code 102, 43.0 %). +0.57 gives 0.1555 (46.0 %). Scene 1.44 gives code 197. All match the reports.
- I read the sheets listed at the end. I built one new sheet that was missing, the brightness A/B (section 2.A).

Not checked: the Python replica of the guard (`ym` does not print its own guard reduction), the synthetic-set stress test, and the survey's ARRI and Adobe figures (secondary sources).

## 2. The options

G3 is the M1-08 gate: our exposure minus the camera's gain. Options A and F change the picture after the exposure is decided, so G3 and the gate do not move. B to E change the exposure. Sheets are in `AE/sheets/`.

Per-file view of the recommended combination (R1 = B + A). G3 is exact (EV minus camera gain). G2 is the pixel model (up to 0.05 EV off the measured renders).

| file | gain | EV now | EV R1 | G3 now | G3 R1 | G2 now | G2 R1 |
|---|---|---|---|---|---|---|---|
| A1M01454 | +0.87 | +1.34 | +1.23 | +0.48 | +0.36 | −0.28 | −0.03 |
| A1M01739 | +1.14 | +0.38 | +0.74 | −0.76 | −0.40 | −1.57 | −0.92 |
| A1M01749 | +0.95 | +0.68 | +0.89 | −0.27 | −0.06 | −1.21 | −0.59 |
| DSLR-A300 | +1.33 | +0.61 | +0.86 | −0.72 | −0.47 | −1.73 | −1.13 |
| DSLR-A500 | +1.15 | +1.90 | +1.50 | +0.74 | +0.35 | +0.33 | +0.28 |
| ILCE-5000 | +0.94 | +0.72 | +0.91 | −0.23 | −0.03 | −1.20 | −0.61 |
| ILCE-6300 | +1.13 | +0.90 | +1.01 | −0.23 | −0.12 | −1.25 | −0.80 |
| ILCE-7M2 | +1.05 | +1.08 | +1.10 | +0.04 | +0.05 | −0.57 | −0.12 |
| ILCE-7M3 | +1.06 | +0.86 | +0.98 | −0.20 | −0.08 | −1.13 | −0.60 |
| ILCE-7M4 | +1.22 | +0.91 | +1.01 | −0.31 | −0.21 | −1.27 | −0.75 |
| NEX-5N | +1.22 | +0.98 | +1.04 | −0.24 | −0.18 | −1.15 | −0.69 |
| NEX-7 | +1.09 | +1.61 | +1.36 | +0.52 | +0.27 | −0.12 | +0.01 |
| median / std | | | | −0.23 / 0.47 | −0.07 / 0.27 | −1.18 / 0.63 | −0.60 / 0.43 |

Eight files move by 0.02 to 0.21 EV (A1M01749 is the largest of them). The four files with the largest key error (A1M01739, A300, A500, NEX-7) move by 0.25 to 0.40 EV, towards the camera. A300 stays at −0.47, only 0.03 inside the ±0.5 limit.

### A. Display placement: a fixed +0.35 EV after the look, before ACES 2 (recommended)

- **What.** Multiply scene-linear AP0 by 2^0.35 just before the ACES 2 transform, on the ACES 2 path only. It sits after the look and curves, so a look still sees grey at LogC4 0.2784.
- **Effect on A1M01749.** Display median code 94 to 107 (Y 0.111 to 0.146). G2 −1.16 to −0.80. Display light moves 1.14 times the EV offset, so +0.35 EV is +0.40 EV on screen. Mid grey goes from display 0.100 to 0.132.
- **Gate.** EV_auto, G3 (8 of 12) and the synthetic gate (37 of 37, EV−truth +0.12) are unchanged.
- **Highlights.** Share of pixels at or above code 250 stays 0.00 %. Share at or above 220 stays at most 0.02 %.
- **Variants.** +0.13 (40 % video): median code 98, G2 −1.06. +0.57 (46 %): median code 114, G2 −0.58, but 0.56 % of pixels at or above code 220 in the brightest scene.
- **ALEXA promise.** Grey lands at 43 % video at gamma 2.4. The grey of ARRI's own Rec.709 rendering is quoted at 38 % (ARRI FAQ, through Art Adams) and 40 % (ARRI's Color by Numbers), and measured at 42 to 46 % by Adams. All secondary sources. ACES 2 as built is at the bottom of that range, +0.35 is mid to upper. The LogC4 0.2784 convention is untouched. The ARRI display LUT path keeps ARRI's own placement.
- **Risks.**
  1. The value is a taste decision and not a measured fact.
  2. Every display-referred picture and golden changes.
  3. **DNG (new finding, missed by both reports).** The DNG profile (tone curve and look table) is fitted to the unplaced ACES 2 (docs/09, `tools/dngfit`). Without a matching change, Adobe apps would show the DNG 0.35 EV darker than the app. Fix: add the placement to the profile's `BaselineExposure` (Adobe applies `2^BaselineExposure` to the linear values before the fitted curve, which is the same operation) or refit with the placed target. Check with the `dng_validate` job.
  4. **LogC3-to-Rec.709 looks (new finding).** Those looks replace C12 and the intensity slider blends our C12 with the look's LUT. Our C12 would be 0.35 EV brighter than the LUT at intensity 0. So the slider would also change brightness by up to 0.35 EV. That is arguably right (the LUT is ARRI's own rendering) but is a visible change.
  5. It does not touch the missing white (section 1.5).
- **Evidence.** `AE/sheets/candidates/A1M01749_brightness_ab.jpg` (new, built for this review: camera JPEG, now, R1 at +0.35, R1 at +0.57). `AE/sheets/candidates/A1M01749.jpg` (panel C4).

### B. Exposure anchored to the camera's gain: `EV = A + w·(L − A)`, A +1.11, w 0.5 (recommended)

- **What.** `L = log2(0.18 / m)` as today. Instead of using `L` directly, correct only half of the distance between this scene's `L` and the cameras' typical gain `A`.
- **Effect on A1M01749.** EV +0.68 to +0.89 (+0.21 EV). Display median code 94 to 101 on its own, and 115 with A. G2 −0.95 on its own.
- **Gate.** G3 12 of 12 (from 8), median −0.07, std 0.27. Leave-one-out std 0.27, so no sign of overfitting to the 12 files. Synthetic gate 37 of 37 (EV−truth +0.32, median G3 +0.35; the largest was not printed).
- **Burst.** Per-frame EV spread falls from 0.15 to 0.07 EV. Taking the rule before or after the burst median differs by at most 2e-16 EV.
- **ALEXA promise.** An ALEXA has no auto exposure. The photographer sets exposure and the look is the same for every shot. Keeping half of the photographer's choice fits that better than a meter that sends every scene to one grey. It also means the same scene shot a stop darker comes out darker.
- **Risks.**
  1. **We give up part of the rescue of bad exposures.** Stress test on the charts, with the camera exposing s stops differently (EV error against the moved truth): today +0.12 at every s. With w 0.5: −0.68, −0.18, +0.82, +1.32 at s = −2, −1, +1, +2. A shot overexposed by a stop comes out 0.82 EV too bright, and the user would pull it down with the slider.
  2. **The gate gets easier by construction.** The anchor is the camera's median gain, and the gate measures distance to the camera's gain. The anchor is one constant fitted on the same 12 files that the gate is run on. It is not circular per photo, but a pass here does not prove the pictures are better. The M1-H5 blind review does.
  3. One global anchor from 12 files and 9 bodies. The three A1 photos alone span 0.27 EV in gain, so I would not try per-body anchors. ISO 50 to 80 was not tested.
  4. **Exposure compensation (new finding).** If `L` is used without a correction, the photographer's EC reaches the picture at weight 0.75 instead of today's 0.5. The `(w − 0.5)·EC` term in section 4 keeps it at 0.5. This is algebra, not tested on a real file.
  5. A300 sits at −0.47, 0.03 inside the limit.
- **Variants.**
  - w 0.25 (C3c): 12 of 12, std 0.18, worst −0.35. Not measured on the synthetic gate. I estimate a median G3 of +0.45 there, which leaves almost no margin.
  - w 0 (C3d, fixed +1.11): 12 of 12, std 0.13. Fails the synthetic gate (0 of 37) because the synthetic JPEG gain is +0.5575 against Sony's +1.11. That is a gate input and I only flag it. The app then no longer meters at all.
  - Soft dead band (C3g, correction fades beyond 2 EV): 12 of 12, std 0.14, keeps more rescue for gross errors than C3d. Same synthetic problem.
  - Anchor at the nominal +0.79 (C3h): 10 of 12, median −0.23. Does not fix the constant.
- **Evidence.** `AE/sheets/candidates/A1M01739_combos.jpg` (bright scene, R1 lifts the sky), `AE/sheets/candidates/DSLR-A500_combos.jpg` (dark scene, R1 and R2 stay dark like the camera), `AE/sheets/candidates/A1M01749_combos.jpg`.

### C. Raise the metering target (C1)

- Target 0.18·2^0.33 (C1a): A1M01749 EV +1.01, median code 106, G2 −0.82. G3 9 of 12. Synthetic gate 35 of 37.
- +0.50 (C1b): 8 of 12, synthetic 0 of 37. +0.85 (C1c): 2 of 12, 7.9 % of pixels at or above code 220.
- It moves the whole library, including the dark scenes that are already too bright. It breaks the rule that scene 0.18 becomes LogC4 0.2784 before the look, which every look assumes. Not recommended.
- Evidence: `AE/sheets/candidates/A1M01749_offsets.jpg`.

### D. Meter on the displayed picture (C2)

- The target is a display geometric mean of 0.212. It is C1 in disguise: EV is today's +1.04 to +1.42. A1M01749 EV +1.93, G2 +0.06, but G3 +0.97 (1 of 12) and the charts read +1.5 EV over truth. It couples metering to C12, so the ARRI LUT path would need its own table. Not recommended.

### E. Key estimate (Reinhard style, C3a)

- The paper formula adds +0.55 EV on average and 0.5 EV of scatter unrelated to the camera. G3 4 of 12, std 0.57. Fitting two parameters gives nothing leave-one-out (0.52 against 0.47). The survey could only quote the formula from memory. Not recommended.

### F. Raise the ACES 2 roof (`r_hit` 128 to 3) (later, after a blind A/B)

- It lets bright pixels reach nearer to white. With a fixed gain of +1.11 and A (R2 plus roof, pixel model): G2 within ±0.5 on 10 of 12, A1M01749 median code 129, G2 −0.20. (It was not run on top of R1.) But 18.6 % of pixels in the brightest scene are at or above code 220.
- It departs from the OpenColorIO vectors that the ≤ 1 code value gate checks (needs our own vectors, a WGSL parameter and a documented departure). The study's render is a ratio model of the tone scale, not a port of the full transform. Gamut and hue behaviour were not evaluated. Evidence is the weakest of all options.
- Evidence: `AE/sheets/candidates/A1M01749_roof.jpg`.

### G. Learned or face-aware exposure

- Defer past V1. MIT-Adobe FiveK and MSEC are for research use, SID has no dataset licence, and the CC0 ARW set is tiny and labelled only by the camera JPEG. Your own blind A/B results are the proper labels.

### H. Change nothing

- You can brighten a photo with the Exposure slider. The photo then reads +0.68 EV in the Automatic card. The README screenshots stay dark.

## 3. Recommended default and ranking

**Default: R1 = A (+0.35 EV display placement) + B (anchor +1.11 EV, w 0.5).** The two parts are independent. Ship A first if you want the largest gain at the lowest risk, and B second after the blind review.

| rank | choice | A1M01749 display median code | G3 (12 files) | why |
|---|---|---|---|---|
| 1 | **R1: A (+0.35) and B (w 0.5)** | 115 | 12 of 12 | fixes the main cause and the four key-extreme files, keeps half the rescue |
| 2 | A alone (+0.35) | 107 | 8 of 12 | lowest risk, no gate change, no metering change |
| 3 | R1 with +0.57 (brighter default) | 123 | 12 of 12 | if the sheet says +0.35 is still dark. Top-end of Adams' range. 0.56 % of pixels at or above code 220 in the brightest scene |
| 4 | B with a soft dead band (C3g) | n/a | 12 of 12 | keeps rescue for gross errors. Needs the synthetic JPEG gain changed to Sony-like first |
| 5 | Fixed gain +1.11 (C3d or R2) | 122 with A | 12 of 12 | simplest, steadiest bursts (spread 0). The app no longer meters. Fails the synthetic gate today |
| 6 | Roof (F) | 129 | 12 of 12 | only after a blind A/B |
| not recommended | C1 raise target, C2 display metering, C3a Reinhard key, learned | | | break the LogC4 convention, or fail the gates, or no signal |

Why R1 over A alone: the metering has two measured deficits, the constant (−0.23 EV) and the key error. The four files that miss the gate are the four with the largest key error. B removes about a third of the constant (median −0.23 to −0.07) and about half of the key error, and it moves typical scenes by at most 0.21 EV. The price is the lost rescue in section 2.B, which is why B follows the blind review.

## 4. Exact spec text

### 4.1 `docs/02-colour-pipeline.md`, "Auto exposure" (replace steps 4 and 5 and the paragraph after step 6; steps 1 to 3 and 6 are unchanged)

Goal line, replace with:

```text
Goal: an exposure close to what the photographer's camera chose, with its worst misses softened, and stable across a burst.
```

Step 4, replace with:

```text
4. `L = log2(0.18 / m)`. `EV_auto = clamp(A + w · (L − A) + (w − 0.5) · EC, −3, +3)`, with `A = +1.11 EV`, `w = 0.5`, and EC the camera's exposure compensation from EXIF.
   `A` is the median gain of the cameras' own JPEG engines (raw level to JPEG mid grey): +1.11 EV over 12 photos from 9 Sony bodies, std 0.13, range +0.87 … +1.33. `w` is how much of the scene's key we correct. `w = 1` meters every scene to 0.18 (the previous rule, `EV_auto = L + 0.5 · EC`). `w = 0` is the camera's gain with no metering. At `w = 0.5` a scene one stop darker than usual is lifted half a stop more than a usual one, where the previous rule lifted it a whole stop.
   Why: the camera exposes the scene and then applies a nearly constant gain. Metering to one grey normalises the key a second time, which pulls bright scenes down and pushes dark ones up beyond what the camera did (EV_auto against log2 of the JPEG median had a slope of −0.44 EV per EV).
   The raw data already holds all of EC (the camera exposed that much more or less). `L` takes `w · EC` of it out again, so the `(w − 0.5) · EC` term leaves exactly half of the photographer's compensation in the picture, as before.
```

Step 5, append:

```text
   Level and fraction are unchanged. On Sony data the guard has not acted (0.00 EV on all 12 real files; the 99.5th percentile pixel sits 2.3 … 4.7 EV below the level), because 8.0 stands for ALEXA's headroom. It stays as a safety net and runs after step 4.
```

Constants line (after step 6), replace with:

```text
`A` and `w` are starting values fitted on 12 local files; re-fit them on the golden set (M1-H5) and keep the algorithm. The gate is unchanged (G3 within ±0.5 EV for at least 80 % of photos).
```

"As built (M1-08)" list, add a bullet:

```text
- *Key correction.* `ANCHOR_EV` (+1.11) and `KEY_WEIGHT` (0.5) in `ym_color::exposure`; `TARGET_GREY` stays 0.18 and is used for `L`. `ev_metered` is the clamped value of step 4; the guard and the burst median follow as before. No usable pixel: unchanged, `clamp(0.5 · EC)`.
```

Gate status paragraph: replace the numbers after the change with a fresh `tools/eval/autoexp.py --gate` run on the real files and the synthetic tier. Expected from the study: 12 of 12 within ±0.5 (in-sample; the anchor was fitted on these files) and synthetic 37 of 37. Say "in-sample" in the text.

### 4.2 `docs/02-colour-pipeline.md`, "C12 · Display transform", ACES 2 (add after step 1 of the numbered list)

```text
1a. **Placement.** Multiply the AP0 triple by `2^k` with `k = DISPLAY_PLACEMENT_EV = +0.35` (a constant, not a control). It sits after the look and the curve, so a look still sees mid grey at LogC4 0.2784, and before the ACES 2 transform. Through the Neutral look it puts scene 0.18 at display-linear 0.132 (sRGB code 102, 43.0 % video at gamma 2.4) instead of 0.100 (code 89, 38.3 %). It applies to ACES 2 only: an imported ARRI display LUT, and the LogC3-to-Rec.709 looks, keep ARRI's own placement.
   Why: ACES 2 SDR puts scene 0.18 at 10 nits, the dark end of the 38 … 46 % video that ARRI's Rec.709 rendering is reported to use, and 0.85 EV under a camera JPEG (code 118). +0.35 brings it into the middle of that range. The value is a look decision, not a measurement; replace it with the measured grey of ARRI's imported Rec.709 LUT (`tools/eval/autoexp.py --display arri`) if you can.
```

"Reference values" paragraph, append:

```text
The vectors test the transform itself and do not include the placement; chain tests apply `2^k` first.
```

Also, `docs/02` "Evaluation (2026-10-13)" bullet 1: add after the first sentence "Since 2026-10-14 the chain applies a placement of +0.35 EV (C12), so G2 now reads about 0.4 EV higher than the figures below."

### 4.3 `docs/09-export.md` (Profile, BaselineExposure rows)

Change "0 for nearly every photo" to "the display placement (+0.35) for nearly every photo". The profile is fitted to the placed ACES 2 target. Final wording depends on whether the PR adds the placement to BaselineExposure or refits (section 5).

## 5. Implementation plan

Wait for the GPU wave (G-03, G-04) to land. It is editing `ym-gpu` and `ym-core` right now and PR 1 touches `shader.rs`.

Branch names in the `m1-07-logc4` style: `m1-11-display-placement` (PR 1) and `m1-08-key-anchor` (PR 2).

### PR 1: display placement

- `crates/ym-color/src/display/transform.rs`:
  - add `pub const DISPLAY_PLACEMENT_EV: f64 = 0.35;` with a doc comment that says why and where it applies;
  - add `pub fn aces2_input_matrix() -> Mat3`, which is `matrices().awg4_to_ap0` with every entry multiplied by `2f64.powf(DISPLAY_PLACEMENT_EV)`;
  - in `PreparedDisplay::new`, `Kind::Aces2` uses it. `Kind::Arri` is unchanged. Do not change `gamut::matrices()` itself: the DNG path and other tests use it.
- `crates/ym-gpu/src/kernels/shader.rs:452`: write `AWG4_TO_AP0` from `aces2_input_matrix()`. `color.wgsl:358` is the only use of that constant, so one change keeps CPU and GPU in step.
- DNG (`crates/ym-export/src/dng.rs`, `docs/09`, `tools/dngfit`): add `DISPLAY_PLACEMENT_EV` to the written `BaselineExposure` (`log2(x_white) + fit_offset_ev + k`), or refit the profile against the placed target. First is exact on neutrals and cheap. Confirm with the `dng_validate` job comparison (`crates/ym-export/tests/dng_validate.rs:697` builds its ACES 2 target from `matrices().awg4_to_ap0`: it must use the placed matrix). If the DNG note in the UI mentions `BaselineExposure` being zero, update it.
- Tests:
  - `crates/ym-color/tests/display_transform.rs:87` and `chain_cpu.rs:180, 206, 1090` build the AP0 triple by hand from `matrices().awg4_to_ap0`: switch them to `aces2_input_matrix()`.
  - New: Neutral look and ACES 2 put scene 0.18 at display Y 0.1320 ± 0.0005 (OpenColorIO's value for 0.18·2^0.35, checked above).
  - New: the placement equals an exposure offset of +k on the Neutral look (C7 is homogeneous), to 1e-9 on pixels away from the low-light toe.
  - New: the ARRI display LUT path is bit-identical to before.
  - CPU/GPU parity tests stay (max abs error as already stated per kernel).
  - ACES 2 vectors (898 per output) do not change. G3, `exposure.rs` tests and the synthetic gate do not change.
- Goldens: **yes, every display-referred golden changes.** `cargo run -p golden --release -- update --note "display placement +0.35 EV (docs/02 C12)"` (all 61 synthetic entries in `tools/golden/expected.json`; the note and the largest ΔE00 are written into the log). `YM_UPDATE_GOLDEN=1` for `crates/ym-export/tests/data/golden_hashes.json`. DNG hashes change with BaselineExposure.
- **Pipeline version bump:** `PIPELINE_VERSION` in `crates/ym-core/src/version.rs` from `0.1.2` to `0.2.0`. Update `crates/ym-cli/tests/cli.rs`, `crates/ym-cli/tests/snapshots/info__json_snapshot_of_a_synthetic_a7r_iv.snap`, `CHANGELOG.md`.
- Docs: section 4.2 and 4.3; `docs/STATUS.md`; a row in `docs/16` (see below); regenerate `docs/images/readme/*` from A1M01749 with the README screenshot script; `docs/11` needs no copy change (the Exposure number does not include the placement).

### PR 2: key anchor

- `crates/ym-color/src/exposure.rs`:
  - add `pub const ANCHOR_EV: f64 = 1.11;` and `pub const KEY_WEIGHT: f64 = 0.5;` with doc comments;
  - in `auto_exposure`, replace the `ev_metered` line (now around line 358) with:

```rust
let l = (TARGET_GREY / mean_luminance).log2();
let ev_metered = (ANCHOR_EV
    + KEY_WEIGHT * (l - ANCHOR_EV)
    + (KEY_WEIGHT - 1.0 + EC_WEIGHT) * ec)
    .clamp(-EV_LIMIT, EV_LIMIT);
```

  - `ec` is the sanitised EC already computed at the top. At `KEY_WEIGHT = 1` the expression is the old `l + EC_WEIGHT · ec`; at 0.5 the EC term is zero. The guard (step 5), the empty-preview branch (`clamp(0.5 · EC)`), `AutoExposure` fields and `share_burst_ev` are unchanged. Update the module doc comment (step 4).
- `crates/ym-core/src/auto.rs`: no change expected (it calls `auto_exposure` and the burst median). Check the doc comments that quote the formula.
- Tests (`crates/ym-color/tests/exposure.rs`):
  - lines 139, 239 and 378 (and the check near line 1118) compute the expected value as `(0.18 / y).log2().clamp(-3, 3)` with or without the `0.5 · ec` term: rewrite them with the new formula and constants;
  - new: `L = A` gives `EV = A`; `L = A + 1` gives `A + 0.5`; the clamp at ±3 still binds (a lens-cap frame still gives +3);
  - new property (proptest, EC in −3…+3, away from the clamp): a scene scaled by `2^EC` and metered with that EC changes `raw · 2^EV` by exactly `2^(0.5·EC)`;
  - new: burst median before and after the rule agrees (the existing burst test, with the new rule);
  - `crates/ym-core/tests/real_files.rs`: update the informational numbers.
- Harness: `tools/eval/metrics.py` (`MeterConfig`: add `anchor_ev` and `key_weight`; `shrunk_ev` already exists), `tools/eval/autoexp.py`, `tools/eval/README.md` and `tools/eval/test_*.py`. The replica must match `ym` to 0.03 EV as before.
- Gate: run `python3 -I tools/eval/autoexp.py <real files> --gate` and the synthetic tier. Expected: real 12 of 12 within ±0.5 (in-sample, worst A300 at −0.47) and synthetic 37 of 37 (the study's median G3 +0.35). Report the real numbers in `docs/STATUS.md`. **Gate thresholds do not change.** If the synthetic tier fails, report it and do not loosen the gate.
- Goldens: **yes, every golden changes again** (EV_auto changes for every file). `golden update --note "key anchor A +1.11, w 0.5 (docs/02 step 4)"`. DNG pixels change too. Bump `PIPELINE_VERSION` again, or ship PR 1 and PR 2 in one release and bump once to `0.2.0`.
- Docs: section 4.1; `docs/13` gate row text (no number change); `docs/STATUS.md`; README screenshots (the Automatic card then reads +0.89 EV for A1M01749); `docs/16` row.

### Draft row for `docs/16`

> 2026-10-14 · Photos looked dark. Of A1M01749's −1.16 EV against its camera JPEG, −0.85 is ACES 2's placement of mid grey and −0.30 is our exposure. Display placement +0.35 EV after the look (ACES 2 only), and an exposure rule that keeps half of the camera's placement (`A +1.11`, `w 0.5`). Gate unchanged; G3 8 of 12 to 12 of 12 on the 12 local files (in-sample), synthetic 37 of 37. · Mykyta (to confirm)

## 6. Questions for you

1. **How bright should the default be?** Open `AE/sheets/candidates/A1M01749_brightness_ab.jpg`: camera JPEG, now, R1 at +0.35 (median code 115), R1 at +0.57 (code 123). Which is right, or something in between? Also look at `AE/sheets/candidates/A1M01739_combos.jpg` (bright scene: it is still darker than the camera).
2. **Do you accept half of the camera's exposure as intent (w 0.5)?** The cost is that mis-exposed shots are rescued less. Or keep full metering and ship A alone (rank 2)? A soft dead band (rank 4) keeps rescue for gross errors but needs the synthetic JPEG gain (+0.5575) changed to Sony-like first. That is a gate input, so it is your call.
3. **Can you import your ARRI Rec.709 LUT and run `autoexp.py --display arri`?** It gives the grey level and the top end of ARRI's own rendering from a primary source. The placement constant can then come from that and not from Adams' tests. The LUT stays local and is not committed.
4. **Can you check Mac Preview?** `sips` on a Mac against the embedded JPEG would tell us what "darker than Preview" means. It cannot be measured here. It belongs under "Waiting on Mykyta" with the other Mac-only checks.
5. **DNG.** Should a DNG open as bright as the app's render? My default is yes (the BaselineExposure change in PR 1).
6. **LogC3-to-Rec.709 looks.** Are you happy that the intensity slider then also changes brightness by up to 0.35 EV, since our C12 is brighter than the look's own rendering?
7. **Blind review.** Do you want the roof (F) as a blind A/B after PR 1 and PR 2, with and without it, on the three A1 photos plus CC0 files? It is the largest remaining gain and the largest departure from the reference.
8. **Gate set.** `A` and `w` were set on 12 local files (9 of unclear licence). Are you happy to ship them as starting values until the CC0 golden set exists (M1-H5)?
