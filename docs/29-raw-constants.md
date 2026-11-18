# 29 · Re-fitting the raw-algorithm constants (STATUS question 12)

This document is the audit and the plan for the constants of the raw-algorithms package (`docs/algorithms/01` to `04`). It answers STATUS question 12 and says how every constant gets a derivation that the project can show.

Date of the audit: 2026-11-13. Branch `raw-constants-audit`, made from `main` at 50bc3fad. The audit changed no product code and no test. It reads the code, the specs and the prior-art record, and it measures what a Linux box can measure with scratch copies of the code (see "How the numbers were made"). Each finding has a fix and a test that proves it. The tasks are RC-01 to RC-14.

## 1. Scope

**The question.** `docs/STATUS.md`, question 12, verbatim:

> 12. **Raw-algorithm constants.** Public sources do not explain some constants of the raw-algorithms package (spec 01: 1.6, 2.5, 18 and 102, 121 and 123, 0.03, 0.75, 5, 3; spec 02: the log2 and exp2 cubics; specs 03 and 04: the constants paragraphs of sections 3.3 and 3.4), and spec 01 section 8 reads as if an integer variant exists elsewhere (`research/raw-algorithms-prior-art.md`). Where do they come from (tuned, fitted, taken from another program)? The answer goes into `docs/14-licensing.md`; patent leads go to M4-H4.

**The owner's answer, 2026-11-13:** "from research, but you need to readjust".

**How this plan reads it.** The constants were taken over from research, not fitted for Yellowmoon. They must now be re-derived by the project itself, with our own method and data, so that each value has a derivation we can show. "Readjust" is read as: where a constant can be derived by mathematics, derive it; where it must be fitted against image data, fit it on data we may use; where a fit does not beat today's value, say so and keep today's value with its derivation written down. The lead decides; the owner sees this plan before any render changes.

**Rules this plan keeps** (`AGENTS.md`):

- *V1 data rule.* Public data and independent tools only. Our own measurements are V2 work. Nothing here asks for a camera capture, a test shoot or a measured spectrum.
- *The owner's standing data rule.* A research-only dataset may be used for local evaluation and fitting. No dataset is ever committed. A fitted constant is a number and may be committed.
- *Hard rule 1.* No GPL, LGPL or AGPL code may be read for this work (RawTherapee, librtprocess, darktable, LibRaw, rawspeed, dcraw, RapidRAW, rawler). Papers may be read. Our fits use our own code and `numpy`.
- *Hard rules 6 and 8.* The CPU reference and the app stay bit-identical for the same inputs, and the same input gives the same bytes. Every fit is seeded and deterministic.
- *The failed-gate rule.* A re-fit that is worse than today's value on its objective or on an existing quality gate is reported, never hidden, and does not ship.

**Out of scope.** The look of the output (the presets, "how sharp is sharp") is a design choice of the owner, not a constant to fit; section 6 asks. Patent questions stay with M4-H4. Spec 05 (single-frame denoise) and spec 06 (masks) are not in question 12; they are not audited here.

**What question 12 names, and how this audit reads "the constants paragraphs".** The prior-art record has a *Constants.* paragraph at the end of each algorithm's section: 3.1 (spec 01), 3.2 (spec 02), 3.3 (spec 03) and 3.4 (spec 04). The question's "sections 3.3 and 3.4" are the sections of the record, not of the specs. Section 2 lists every constant of those four paragraphs and, because the record's "Not assessed" tables name more, the constants of those pieces too.

## How the numbers were made

Every number below is cited from a spec or the code, or it was measured for this audit. A number that was not measured is called an estimate and says how it was made.

- **Machine.** Linux x86_64 VM, 4 vCPU, 15 GB RAM, shared with other jobs (load average 11 to 19 during the runs), no GPU. Every run is the CPU reference path. Times are for this machine only.
- **Scratch code.** The sensitivity runs use a scratch copy of the code (a second worktree that is never committed) in which each constant reads an override from an environment variable or a function call, with today's value as the default. The scratch examples are `rc_sens.rs` (spec 01 and spec 04) and `rc_diff.rs` (spec 03, a copy of `capture_sharpen_ab.rs`). The branch of this audit holds no product code and no test; the scripts are listed in the appendix and are not committed.
- **Spec 01 scenes.** The adaptive ratio demosaic (`adaptive_ratio::demosaic`, the default of R6) on mosaics made from ground-truth images, in four groups:
  - **A, clean:** the procedural set of `tests/cpu_demosaic.rs` (7 images), Kodak (24) and McMaster (18) in linear light at signal scale 1, unit gains, the nominal low-noise model of spec 01 §6.3 (`a = 2e-5`, `b = 1e-8`), shadow matching `Auto` (so off). 49 images.
  - **B, dark and noisy:** every third image of the same set (17 images) at 0.25 of white with Poisson-Gaussian noise added (seeded splitmix64, `a = 5e-4`, `b = 4e-5`, row noise 1e-3: the "high" model of spec 01 §2.2, 23.9 dB at 18 % grey), the matching model passed, shadow matching and shadow sharpening on. This is the scene where the shadow constants act.
  - **C, real mosaics:** two crops (1024 × 768) of a real A1 file (`testdata/readme-photos/src/A1M01749.ARW`, the owner's file, local only), as-shot gains (the file's own white balance), RGGB assumed at the readout origin as for every Sony body, with two stand-in noise models (ISO 100-like and ISO 6400-like, the values of `capture_sharpen_ab`; they are stand-ins, not measurements of that body). No ground truth: only the change against today's output can be measured.
- **Spec 04 scenes.** The screen preset (`OutputSharpen::screen()`, method 1) on the 24 Kodak images in linear light, at scale 1 and at scale 0.33 (a typical downsized export): 48 outputs per setting.
- **Spec 03 scenes.** The three scenes of `capture_sharpen_ab` (fine text, a natural scene, a slanted edge at 5 degrees), blurred with the Airy PSF of today's plan for a 3.76 um pitch at f/8 and f/16, with correlated Poisson-Gaussian noise at the ISO 100 and ISO 800 stand-in models, then restored by the diffraction plan with the constant moved. The scene is always blurred by today's PSF (the truth optics) so a moved constant is tested against the same input.
- **The measure of change.** `docs/13` allows the golden metric (the largest ΔE00) or code values; the sensitivity runs use code values. The change of the output against today's output, in 8-bit sRGB codes (the picture is linear; encoded with the sRGB curve and scaled to 255), reported as the mean of `|difference|` over every pixel and channel and the largest single difference. The largest difference is almost always one pixel where a hard decision (a direction test, a threshold) flipped; it says "something can change" and not "much changes". The mean says how much the picture moves. For spec 01 the PSNR against the ground truth is reported too (mean over the images of a group, 10 px border off, linear light, the protocol of spec 01 §6.3).
- **Not measured, and said so.** The full chain on real ARW files through `ym` and the golden runner: it needs a full build of `ym` and its dependencies, and this box has 5 GB of free disk and shares its cores with other jobs. The effect on the golden pictures is therefore not measured here; RC-13 measures it with the real runner. The GPU kernels (no adapter here). Any Mac. Spec 05 and spec 06.

## Findings at a glance

| Package | Rows | What already works | The biggest gaps |
|---|---|---|---|
| Spec 01 · adaptive ratio demosaic (the default of R6) | 15 | The algorithm is proven against 18 oracle vectors on the CPU and the GPU; `demosaic_gate` measures PSNR, zipper and false colour; the chain is deterministic and tiles bit-exactly | No constant has a derivation. Ten of them are literals in both the CPU file and the shader. 121 and 123 are not constants (`1 + 18 + 102` and `+ 2`). The CA constants (A2 to A4) never reach a render. PSNR rises when A6, A7 and A13 are raised (A13 ×1.5: +0.30 dB), so today's values are not at the optimum of the synthetic, Kodak and McMaster scenes; the shadow constants cannot be fitted by PSNR |
| Spec 02 · log-guided demosaic (no render path) | 3 | An exhaustive test of the two fixed-point functions | The cubics are near-minimax, not minimax: log2 error 4.98 → 4.40 codes and exp2 `k` 1.24e-4 → 9.0e-5 are available in minutes, with no picture changed |
| Spec 03 · diffraction deconvolution (not the default of R8) | 14 | `capture_sharpen_ab` reproduces the numbers of the spec table that were checked (f/8 and f/16 at ISO 100); the plan is cached and planned once on the CPU | No row moves a render today. The σ_G factor 0.75 is 4 % under the exact 0.780 for a Gaussian, and raising it by 10 % improves PSNR and MTF50 together. The clip penalty and the floors were not exercised by the scenes |
| Spec 04 · output sharpening (on by default) | 12 | An edge profile and a noise gain are documented per preset (spec 04 §9); one stage for app and `ym` | The radius-to-σ map moves a picture 10 to 20 times more than any demosaic constant (0.3 codes for ±10 %) and has a step at `R = 0.5`. The preset values are a look, not a fit |
| The package | 44 | The vectors pin the structure; `golden check` pins every default picture | No harness takes a constant as input. The 73 vector files cannot be regenerated (their C++ is not in the repository). No real-sensor ground truth is reachable here. `docs/14`, `docs/16` and `docs/validation-data.md` disagree about what the constants are and what the data may be used for |

---

## 2. The audit

### 2.1 The inventory

One row per constant, or per group of constants that act as one. File and line are at the audit's base commit (`50bc3fad`). Paths are under `crates/ym-gpu/src/`; `adaptive_ratio.rs` is `cpu/demosaic/adaptive_ratio.rs`, `diffraction.rs` is `cpu/sharpen/diffraction.rs`, `output_sharpen.rs` is `cpu/output_sharpen.rs`, and the shaders are under `shaders/`. "GPU: CPU" means the kernel takes the number from the CPU's plan as a uniform or a tap table, so there is no second copy.

**How to read "Derivable".** *Maths*: a mathematical fact or an error budget gives the value, and a test can prove it. *Maths + target*: the formula is exact but one input is a choice (an SNR, an error budget) that must be stated. *Fit*: only image data can choose it; the objective is in section 4. *Design*: the owner's look; not fitted (section 6).

#### Spec 01 · adaptive ratio demosaic (the default of R6; 15 rows)

| ID | Spec § | Value | CPU | GPU | What it controls | Kind | Derivable |
|---|---|---|---|---|---|---|---|
| A1 | 3.2 | `1.6` | `adaptive_ratio.rs:777` | `adaptive_ratio.wgsl:345` | Weight of the second difference of the same colour, `1.6·ρ·|P(0,−2) − 2c + P(0,2)|`, in the pass-1 direction scores. | weight | Fit |
| A2 | 3.2, 3.6 | `2.5` | `adaptive_ratio.rs:816` | `adaptive_ratio.wgsl:389` | How fast the direction test relaxes with the colour misregistration `ca`: `f = 1 + 2.5·ca³`. CA path only. | weight | Fit |
| A3 | 3.2, 3.6 | `0.1` px | `adaptive_ratio.rs:815` | `adaptive_ratio.wgsl:388` | Misregistration below this does not relax the test. CA path only. | threshold | Fit; the floor is the table's own precision (its scale is `2^-21` per unit, `INTEGRATION.md` §6; at a radius of 4000 px one unit is about 0.002 px, an estimate from that scale) |
| A4 | 3.6 | `18`, `102` | `adaptive_ratio.rs:994` | `adaptive_ratio.wgsl:556` | Weight of the centre green in the pass-2 three-tap mean, `w(t) = 1 + 18t + 102t³`, `t = (1 − min(1, 2κ))²`. CA path only. | fitted polynomial | Fit |
| A5 | 3.6 | `121`, `123` | none (they are results) | none | `121 = 1 + 18 + 102` is `w` at `κ = 0`; `123 = 121 + 2` is the divisor there. With no misregistration the centre keeps 121/123 = 98.4 % of the three-tap mean. **Not constants**: they follow from A4. | derived | Maths (from A4) |
| A6 | 3.4.2, 3.4.3 | `0.03` | `adaptive_ratio.rs:1102-1103`, `1150` | `adaptive_ratio.wgsl:637-638`, `656` | Floor of the chroma penalty `|X − G|/max(X + G + 1, 1) + 0.03` that multiplies the cost of a high-chroma sample. | floor | Fit |
| A7 | 3.4.2, 3.4.3 | `0.75` (dominance) | `adaptive_ratio.rs:1101`, `1148` | `adaptive_ratio.wgsl:636`, `653` | The penalty applies when `0.75·ΣX < ΣG`, that is, when the red or blue is not dominant over the greens. | threshold | Fit |
| A8 | 3.4.4 | `0.75` (amount) | `adaptive_ratio.rs:1219` | `adaptive_ratio.wgsl:764` | Share of the soft-limited luma Laplacian that the shadow sharpening adds back: `gain = (y0 + 0.75·f·a)/y0`. The same number as A7 by coincidence, a different job. | weight | Fit |
| A9 | 3.4.4 | `5` | `adaptive_ratio.rs:1215` | `adaptive_ratio.wgsl:757` | Scale of the diagonal Laplacian `d = 5·(2·y0 − p)` that the soft limit then bounds. With A8, the small-signal gain is `0.75 · 5 = 3.75`. | weight | Fit |
| A10 | 3.4.4 | `4T`, `3T` | `adaptive_ratio.rs:1203` | `adaptive_ratio.wgsl:744` | The shadow sharpening is full where every native green is below `T` and fades to nothing at `4T`: `f = clamp((4T − gmax)/(3T), 0, 1)`. Not named in question 12; listed in the record's "Not assessed" table. | window | Fit |
| A11 | 3.4.5 | `3` | `adaptive_ratio.rs:1357` | `adaptive_ratio.wgsl:884` | Weight of the excess error in the shadow colour matching, `u = 3D + 1`: how much more than the noise the match tolerates before it lets the demosaic's own colour through. Not named in question 12 beyond "3". | weight | Fit |
| A12 | 2.2 | `50` and `0.04` | `adaptive_ratio.rs:240` | GPU: CPU | The offset `K = 50(g + √(g² + 0.04r²))`, the signal level where `x/σ(x) = 10`. The formula is exact algebra (the record, A3); the SNR target 10 is the choice. | derived from a target | Maths + target (10) |
| A13 | 2.2 | `0.2` (and `100`) | `adaptive_ratio.rs:241` | GPU: CPU | `s = max((100 − (detail + strength))·0.2, 1)`: the pass-3 SNR target, 20 at the defaults; sets `T`. The factor 0.2 maps the 0 to 100 sliders onto 20 to 1. | target | Fit |
| A14 | 2.2 | `25 dB` (`10^2.5`) | `adaptive_ratio.rs:253` | CPU only (a switch) | Shadow colour matching turns on when the SNR at 18 % grey is below 25 dB. | threshold | Fit (the crossover where matching starts to help) |
| A15 | 3.4.5 | the 5×5 integer Gaussian over 571, the 3×3 binomial | `adaptive_ratio.rs:1266` | `adaptive_ratio.wgsl` | Fixed filters. Known kernels (the record, A14: Framewave). Not tunable. | filter | n/a (document) |

#### Spec 02 · log-guided demosaic (no render path uses it; 3 rows)

| ID | Spec § | Value | CPU | GPU | What it controls | Kind | Derivable |
|---|---|---|---|---|---|---|---|
| L1 | 3.2 | `(46597, 19037, 5208)/32768` | `cpu/demosaic/log_guided.rs:194` | none (the GPU and the float form use `log2`) | Cubic for `log2(1 + f)` in the fixed-point reference, `p(0) = 0`, `p(1) = 1`. | fitted polynomial | Maths (minimax) |
| L2 | 3.4 | `(45607, 14741, 5185)/65536` | `cpu/demosaic/log_guided.rs:209` | none | Cubic for `2^f − 1` in the fixed-point reference, constant term 1. | fitted polynomial | Maths (minimax) |
| L3 | 3.7 | exponent `3` | `cpu/demosaic/log_guided.rs:233` (`cube_floor1`) | `log_guided.wgsl` (`t = (D̄/Ē)³`) | The inverse-cube structure weight of the green blend. A tuning constant in a known family (the record, L16). | weight | Fit |

#### Spec 03 · diffraction deconvolution (not the default of R8; 14 rows)

All of spec 03's planning runs on the CPU (`plan_geometry`, `DiffractionPlan::new`) and reaches the GPU as taps and uniforms, so a constant of the PSF or of `σ_G` has one copy. The clip penalty, the envelope, the ridge and the gate are also in `diffraction.wgsl` (lines 96 to 105, 282, 665).

| ID | Spec § | Value | CPU | GPU | What it controls | Kind | Derivable |
|---|---|---|---|---|---|---|---|
| D1 | 2.6, 3.2 | 461 to 619 nm, weight 1 above 540 nm, `(620 − λ)/80` below | `diffraction.rs:633-639`, `652-659` | GPU: CPU | The spectral weight of the polychromatic Airy PSF. | weight | Maths + public data (the camera's spectral response, `docs/29` §4, D1) |
| D2 | 2.6, 3.2 | radial step `min(p_eff, 0.42·N/π)/4`, 500 samples | `diffraction.rs:89`, `645` | GPU: CPU | Sampling of the radial profile. | window | Maths (convergence) |
| D3 | 2.6, 3.2 | `0.8` (energy), `0.01`, `4·r80`, `6` px, `0.99`, `0.002`, 6 angular samples per px (`+ 0.999`), the 8 × 8 grid, the cos² taper | `diffraction.rs:668`, `680`, `684`, `716`, `761` | GPU: CPU | Where the kernel is cut (80 % encircled-energy radius, support, kernel radius). | window, threshold | Maths (error budget) |
| D4 | 2.6, 3.3 | `π/16` × 8 angles, 3 bins per px, `0.85`, `0.8` | `diffraction.rs:785`, `790`, `814`, `827` | GPU: CPU | How the line spread of the kernel is measured (a slanted-edge style estimate). | window | Maths (test on known kernels) |
| D5 | 2.6, 3.3 | `0.75`, `[0.2, 3.0]` | `diffraction.rs:829-833` | GPU: CPU | `σ_G = 0.75 · mean(x80)` and its clamp: the width of the high-pass Gaussian. For a Gaussian, `x80 = 1.2816 σ`, so the exact factor is 0.780. | weight, clamp | Maths for the Gaussian; Fit for the Airy-like kernel |
| D6 | 2.6, 3.3 | `3.5`, `1`, `8` | `diffraction.rs:836` | GPU: CPU | High-pass Gaussian radius `min(floor(3.5σ_G + 1), 8)`. | window | Maths (tail budget) |
| D7 | 2.6, 3.4 | `0.8` (of white), `25`, `5`, `1` | `diffraction.rs:99`, `263`, `470` | `diffraction.wgsl:282`; the threshold is a uniform | Clip penalty: where the highlight starts and the weights of the sorted channel excesses. | threshold, weight | Fit |
| D8 | 2.6, 3.5 | `12.5`, `0.04` | `diffraction.rs:1086` | `diffraction.wgsl:103-104` | Pull-in of the local envelope estimates. | weight | Fit |
| D9 | 2.6, 3.9 | `0.0005` | `diffraction.rs:1139` | `diffraction.wgsl:98` | Ridge of the 3 × 3 local solve. | weight | Fit |
| D10 | 2.6, 3.9 | `0.5`, `2` (= 1/0.5), `1/4`, `1/2` | `diffraction.rs:97`, `1124`, `1212-1217` | `diffraction.wgsl:97`, `588-590` | The noise gate: where a pixel is left alone, and the fade-out band of the residual. | threshold | Fit |
| D11 | 2.6, 3.9 | `0.01`, `−0.5`, `[0, 1)` | `diffraction.rs:1220` | `diffraction.wgsl:100-101` | Smallest confidence weight; lower bound of `cD`; range of `cH`. | clamp | Fit |
| D12 | 2.6, 3.12 | `1.5` | `diffraction.rs:415` | `diffraction.wgsl:105`, `665` | Cap of the luma-to-RGB ratio. | clamp | Fit |
| D13 | 2.6 | `1e-20`, `1e-15`, `1 DN` | `diffraction.rs:1133`, `1081` | `diffraction.wgsl:99`, `102` | Numeric guards. | guard | n/a (document) |
| D14 | 2.6 | windows 5 × 5, 9 × 9 | structure | structure | Envelope, moment and smoothing windows. | window | Fit (structure; see 4.4) |

#### Spec 04 · output sharpening (on by default, every display-referred render; 12 rows)

| ID | Spec § | Value | CPU | GPU | What it controls | Kind | Derivable |
|---|---|---|---|---|---|---|---|
| O1 | 2.2 | `0.7R + 0.3` for `R ≥ 0.5`, `0.97·√(R − 0.09)` for `0.09 < R < 0.5`, cap 2.5 px | `output_sharpen.rs:98`, `654-662` | GPU: CPU | The map from the preset's radius to the Gaussian σ. A step of 0.03 at `R = 0.5` (0.621 to 0.650). The screen preset reads it at `R = 1.0` (σ = 1.0). | fitted map | Design + Maths (continuity) |
| O2 | 2.3 | `5.0`, `2.5`, `\| 1`, minimum 3, caps 13 and 33 | `output_sharpen.rs:104-106`, `712` | GPU: CPU | Tap count `((int)(5σ + 2.5)) \| 1`: how far the Gaussian is cut. | window | Maths (tail budget) |
| O3 | 2.4 | `0.15` | `output_sharpen.rs:112`, `665-669` | GPU: CPU | The σ below which a downscale does not shrink the radius (`s·(σ − 0.15) + 0.15`). Acts in exports at a reduced size. | knee | Fit |
| O4 | 2.5 | 100 %, 1.0 px, 1 level, halo 60 % | `output_sharpen.rs:279-293` | GPU: CPU | The screen preset. | preset | Design |
| O5 | 2.2 | `sqrt(h/100)`, dead zone `0.001` | `output_sharpen.rs:114`, `688-693` | GPU: CPU | How the halo percentage becomes the limited share of the delta. | map | Design |
| O6 | 3.5 | window 5 × 5, `N = 25`, `k_min = 6`, `1/r = 0.5`, `1/N = 0.04`, tie tolerance `2^-12`, pull fade `0.25`, `η = 2τ/N²` | `output_sharpen.rs:116-128`, `1200` | `output_sharpen.wgsl:100`, `440` | The halo limiter's robust envelope. | window, weight | Fit (structure) |
| O7 | 2.3 | `max(2σ, 1)`, `2s` | `output_sharpen.rs:721-724` | GPU: CPU | The widths of the gain-map and halo-map smoothing. | weight | Fit |
| O8 | 3.7 | `16`, `1e-20` | `output_sharpen.rs:132-134`, `1459` | `output_sharpen.wgsl:95-96` | Ratio cap and luma floor of the transfer back to RGB. Numeric safeguards. | clamp | n/a (document) |
| O9 | 3.3 | `1e-10` | `output_sharpen.rs:130` | `output_sharpen.wgsl:94` | Epsilon of the gain-map denominator. | guard | n/a (document) |
| O10 | 3.3 | `0.3125`, `0.09375`, ramp `0.25` (to `5·g0`) | `output_sharpen.rs:1027-1045` | `output_sharpen.wgsl:337-340` | Method 2 only (the gradient gate). `10/32` and `3/32` are the normalised Scharr kernel (`{3, 10, 3}/16 × {−1, 0, 1}/2`). Method 2 is not reachable from the app or `ym`. | derived, ramp | Maths (Scharr); the ramp is Fit |
| O11 | 2.2 | print: `2500` per px, cap 15 px, 150 % amount, distance 5000 | `output_sharpen.rs:100-102`, `297-306` | GPU: CPU | The print preset. Not reachable from the app or `ym`. | preset | Design |
| O12 | 2.7 | `0.65`, `0.1`, `0.004`, `0.1`, `0.025` | `output_sharpen.rs:696-703` | none | The automatic method rule for extra stages. Unused in v1 (`INTEGRATION.md`). | threshold | n/a (dead until a use) |

**Count.** 15 rows for spec 01 (14 of them constants; A5 is not one), 3 for spec 02, 14 for spec 03 and 12 for spec 04: 44 rows. Of question 12's list, every item has a row: 1.6 (A1), 2.5 (A2), 18 and 102 (A4), 121 and 123 (A5), 0.03 (A6), 0.75 (A7, A8), 5 (A9), 3 (A11), the cubics (L1, L2), the Constants paragraph of the record's 3.3 (D1 to D12) and 3.4 (O1 to O10). The record's "other constants without a source" are A3, A10, A12 to A14, L3, D13, D14 and O11, O12.

**On "spec 01 section 8 reads as if an integer variant exists elsewhere".** `docs/algorithms/01` §8 lists, as choices of the reference, exact division, no fixed-point quantisation of ρ, no integer truncation of the averages and no integer rounding of the CA relaxation. That reads as though another implementation does quantise them. Nothing in this repository needs such a variant: the port is floating point (f64 on the CPU, f32 on the GPU) and the vectors were made in floating point. The fits below do not depend on whether an integer variant exists elsewhere, and none is planned; whether one exists is a question for the owner's research source (section 6, question 6).


### 2.2 The effect on renders

Each constant, or each group that acts together, was moved to ×0.9, ×1.1, ×0.5 and ×1.5 of today's value, one at a time, on the scenes of "How the numbers were made". The tables give the change of the output against today's output in 8-bit sRGB codes (mean over every pixel and channel, and the largest single pixel) and, where there is a ground truth, the change of the PSNR in dB. A figure of 0.0000 is exact: the output did not change in a single bit.

**How to read the size of a change.** These are the conventions of this audit, not measurements of what an eye sees: below 0.01 codes in the mean is negligible, 0.1 codes is a change that only a metric shows, and 1 code is a change that may show in a difference image. The largest single pixel is almost always one pixel at which a hard decision (a direction test, a threshold) flipped, so it says "the output can change" and not "the output changes much". For scale, the methods of `docs/algorithms/01` §6.4 differ by 0.5 to 1.2 dB of PSNR on the same kinds of image.

#### Spec 01 · adaptive ratio demosaic

Today's PSNR (mean over the images of each group, linear light, 10 px border off): group A 35.678 dB (49 images), group B 27.604 dB (17 images). C has no ground truth. The table gives, for each constant moved by −10 % and +10 %, the change of the PSNR of A and B, the mean change of the picture in codes for A, B and C (the larger of the two directions), the largest single pixel in A, and the PSNR change at ×0.5 and ×1.5.

| Constant | ΔPSNR A, −10 % / +10 % (dB) | ΔPSNR B, −10 % / +10 % (dB) | mean change, codes: A / B / C | largest single pixel, A (codes) | ΔPSNR A at ×0.5 / ×1.5 | ΔPSNR B at ×0.5 / ×1.5 |
|---|---|---|---|---|---|---|
| A1 `1.6` | +0.0018 / -0.0027 | +0.0009 / -0.0013 | 0.0142 / 0.0772 / 0.0107 | 135 | +0.0059 / -0.0138 | +0.0027 / -0.0064 |
| A2 `2.5` | +0.0000 / +0.0000 | +0.0000 / +0.0000 | 0.0000 / 0.0000 / 0.0000 | 0 | +0.0000 / +0.0000 | +0.0000 / +0.0000 |
| A4 `18` | +0.0000 / +0.0000 | +0.0000 / +0.0000 | 0.0000 / 0.0000 / 0.0000 | 0 | +0.0000 / +0.0000 | +0.0000 / +0.0000 |
| A4 `102` | +0.0000 / +0.0000 | +0.0000 / +0.0000 | 0.0000 / 0.0000 / 0.0000 | 0 | +0.0000 / +0.0000 | +0.0000 / +0.0000 |
| A6 `0.03` | -0.0144 / +0.0132 | -0.0056 / +0.0051 | 0.0061 / 0.0239 / 0.0065 | 22 | -0.0882 / +0.0572 | -0.0354 / +0.0219 |
| A7 `0.75` | -0.0343 / +0.0384 | -0.0135 / +0.0159 | 0.0218 / 0.0934 / 0.0192 | 87 | -0.1544 / +0.1090 | -0.0394 / +0.0411 |
| A8 `0.75` | +0.0000 / -0.0000 | +0.0480 / -0.0493 | 0.0004 / 0.1737 / 0.0077 | 2 | +0.0001 / -0.0001 | +0.2272 / -0.2565 |
| A9 `5` | +0.0000 / -0.0000 | +0.0019 / -0.0016 | 0.0000 / 0.0122 / 0.0005 | 0 | +0.0000 / -0.0000 | +0.0152 / -0.0061 |
| A11 `3` | +0.0000 / +0.0000 | +0.0062 / -0.0052 | 0.0000 / 0.0624 / 0.0051 | 0 | +0.0000 / +0.0000 | +0.0506 / -0.0189 |
| A12 `50` | -0.0060 / +0.0052 | -0.0032 / +0.0025 | 0.0078 / 0.0473 / 0.0060 | 95 | -0.0373 / +0.0241 | -0.0297 / +0.0092 |
| A13 `0.2` | -0.0676 / +0.0653 | -0.0004 / +0.0000 | 0.0285 / 0.0581 / 0.0061 | 22 | -0.3495 / +0.2997 | -0.0148 / -0.0044 |

- **A2, A3, A4: exactly 0.0000 on every scene.** The chain passes no CA curves (`ca = None`) to the demosaic, because R5b corrects CA on the mosaic first (`INTEGRATION.md` §6), so the CA relaxation and the pass-2 weighting never run in a render. The rows of A2 and A4 are included to show that. (A3, the `0.1` px threshold, sits in the same branch as A2 and was not moved separately.) The CA path has its own harness; its measurement follows.
- **The pass-3 target (A13) and the chroma penalty (A7, A6) move the clean pictures most**, by 0.03, 0.02 and 0.006 codes in the mean for ±10 %, with PSNR changes of ±0.065, ±0.04 and ±0.014 dB. A12 (the SNR-10 offset) and A1 (the pass-1 weight) move them by 0.008 and 0.014 codes.
- **In the dark noisy scenes the shadow constants lead**: A8 (the sharpening amount, 0.17 codes for ±10 %), A7 (0.09), A1 (0.08), A11 (0.06), A13 (0.06), A12 (0.05). A9 moves them by 0.012.
- **On the two real A1 crops (C, stand-in models) every constant moves the picture by 0.02 codes or less**: A7 0.019, A1 0.011, A8 0.008, A6 0.007, A13 0.006, A12 0.006, A11 0.005, A9 0.0005. Nothing here shows a constant that matters on a real mosaic and not on the synthetic scenes.
- **Today's values are not at the PSNR optimum of this set**: PSNR rises when A6, A7, A12 or A13 is raised, and when A1 is lowered, in both ±10 % and ±50 % (A13 ×1.5: +0.30 dB; A7 ×1.5: +0.11 dB; A6 ×1.5: +0.06 dB; A1 ×0.5: +0.006 dB). The gains are small (the methods of `docs/algorithms/01` §6.4 differ by 0.5 to 1.2 dB) and a PSNR optimum on re-mosaicked 8-bit pictures is not a certainty about sensors (G5). They say the fits have something to find for A6, A7 and A13, and very little for A1.
- **A8, A9, A11 (shadow sharpening and shadow colour matching) get a higher PSNR when they act less** (A8 ×0.5: +0.23 dB on B; A11 ×0.5: +0.05 dB). For a sharpening that is the expected sign (G6): PSNR is not its objective.
- **Not moved in this run:** A10 (the 4T/3T window of the shadow sharpening) and A14 (the 25 dB switch). A10 acts in the same shadows as A8 and scales the same step; I expect an effect of the same order as A8's, which is an estimate from the form of the formula and not a measurement. A14 is a per-photo on/off switch; it changes a picture only when the SNR at 18 % grey is near 25 dB, and switching shadow colour matching on or off was not measured separately. RC-08 measures both.

#### The CA path (A2 to A4), on its own harness

`ca_green_residual` at 600 × 400, five scenes (lens-table CA strengths 0.5, 1 and 2 without distortion, and 1 and 2 with 1.5 % barrel distortion), the "adaptive ratio, CA path" row: the residual red and blue displacement against green after R9 (RMS, px, red / blue) and the PSNR of all three planes. The first row moves A1 (`1.6` to `10`) as a reference that the harness sees a constant that matters.

| Setting | strength 0.5: R/B all (px) · PSNR RGB (dB) | strength 1: R/B all (px) · PSNR RGB (dB) | strength 2: R/B all (px) · PSNR RGB (dB) | strength 1, barrel: R/B all (px) · PSNR RGB (dB) | strength 2, barrel: R/B all (px) · PSNR RGB (dB) |
|---|---|---|---|---|---|
| A1 `1.6` to `10` (reference) | 0.055 / 0.044 · 43.33 | 0.115 / 0.092 · 40.03 | 0.204 / 0.176 · 36.06 | 0.113 / 0.094 · 40.27 | 0.188 / 0.161 · 36.59 |
| today | 0.055 / 0.043 · 43.38 | 0.114 / 0.092 · 40.09 | 0.204 / 0.175 · 36.10 | 0.112 / 0.094 · 40.31 | 0.187 / 0.161 · 36.62 |
| A2 `2.5` to 0 | 0.055 / 0.043 · 43.40 | 0.116 / 0.092 · 40.08 | 0.209 / 0.180 · 35.99 | 0.113 / 0.095 · 40.32 | 0.192 / 0.164 · 36.53 |
| A2 ×0.5 | 0.055 / 0.043 · 43.40 | 0.114 / 0.092 · 40.10 | 0.204 / 0.175 · 36.10 | 0.112 / 0.094 · 40.32 | 0.187 / 0.161 · 36.63 |
| A2 ×1.5 | 0.054 / 0.043 · 43.38 | 0.114 / 0.092 · 40.09 | 0.204 / 0.175 · 36.10 | 0.112 / 0.094 · 40.31 | 0.187 / 0.161 · 36.62 |
| A2 ×4 | 0.054 / 0.043 · 43.37 | 0.114 / 0.092 · 40.09 | 0.204 / 0.175 · 36.10 | 0.112 / 0.094 · 40.30 | 0.187 / 0.161 · 36.62 |
| A4 `18`, `102` to 0 (weight 1) | 0.054 / 0.042 · 42.97 | 0.114 / 0.091 · 39.92 | 0.204 / 0.175 · 36.05 | 0.112 / 0.094 · 40.18 | 0.187 / 0.161 · 36.58 |
| A4 ×0.5 | 0.054 / 0.043 · 43.60 | 0.114 / 0.092 · 40.16 | 0.204 / 0.175 · 36.12 | 0.112 / 0.094 · 40.37 | 0.187 / 0.161 · 36.64 |
| A4 ×1.5 | 0.055 / 0.043 · 43.27 | 0.114 / 0.092 · 40.06 | 0.204 / 0.175 · 36.09 | 0.112 / 0.094 · 40.29 | 0.187 / 0.161 · 36.61 |
| A4 ×4 | 0.055 / 0.044 · 43.05 | 0.115 / 0.092 · 39.97 | 0.204 / 0.175 · 36.07 | 0.113 / 0.094 · 40.22 | 0.188 / 0.161 · 36.59 |

- **A2 (`2.5`) hardly matters**: ×0.5, ×1.5 and ×4 change the residual by at most 0.001 px and the PSNR by at most 0.02 dB; removing the relaxation altogether (`f = 1`) costs 0.005 px and 0.11 dB at the strongest CA (strength 2: 0.204 to 0.209 px, 36.10 to 35.99 dB).
- **A4 (`18`, `102`) matters more, and today's weight is not the best one on this scene**: halving both (a smaller share for the centre green) gives +0.22 dB at strength 0.5 (43.38 to 43.60), +0.07 dB at strength 1 and +0.02 dB at strength 2; ×1.5 costs 0.11 dB at strength 0.5; a weight of 1 for the centre (`18` and `102` set to 0) costs 0.41 dB there. The residual displacement barely moves (≤ 0.001 px). So `18` and `102` are a real design parameter of the pass-2 smoothing, in a path that no render reaches.

#### Spec 03 · diffraction deconvolution

Scenes: fine text, natural, slanted edge; f/8 and f/16; ISO 100 and 800 stand-in models; the scene is always blurred by today's PSF. Today's values reproduce the table of spec 03 §6.4 (f/8 ISO 100: MTF50 0.340, overshoot 1.7 %, flat-patch noise ×1.005; f/16 ISO 100: 0.245, 3.0 %, ×1.577). Each cell is the mean over the four cases of the change against today's output: −10 % / +10 %.

| Constant | text PSNR (dB), −10 % / +10 % | natural PSNR (dB) | MTF50 (cycles/px) | overshoot (%) | flat-patch noise (×) |
|---|---|---|---|---|---|
| D1 `80` | +0.003 / -0.003 | -0.005 / +0.006 | -0.0004 / +0.0003 | -0.01 / +0.01 | -0.0019 / +0.0017 |
| D2 `0.42` | +0.000 / +0.000 | +0.001 / -0.000 | +0.0001 / -0.0000 | +0.00 / +0.00 | -0.0000 / +0.0007 |
| D3 `0.8` (energy) | +0.014 / -0.124 | -0.048 / +0.140 | -0.0030 / +0.0084 | -0.08 / +0.28 | -0.0014 / +0.0039 |
| D3 `4`·r80 | +0.012 / -0.017 | -0.028 / +0.026 | -0.0017 / +0.0015 | -0.05 / +0.05 | -0.0008 / +0.0008 |
| D3 `6` px | +0.018 / -0.004 | -0.034 / +0.036 | -0.0013 / +0.0019 | -0.14 / +0.15 | -0.0110 / +0.0097 |
| D3 `0.99` | -0.000 / -0.000 | -0.000 / +0.001 | +0.0000 / +0.0000 | -0.00 / +0.00 | +0.0000 / +0.0000 |
| D3 `0.002` | -0.000 / +0.000 | -0.000 / +0.000 | +0.0000 / +0.0000 | -0.00 / +0.00 | +0.0000 / +0.0000 |
| D4 `0.85` | -0.072 / +0.287 | -0.076 / +0.192 | -0.0057 / +0.0146 | -0.06 / +0.16 | -0.0001 / +0.0012 |
| D4 `0.8` | -0.244 / +0.226 | -0.217 / +0.161 | -0.0180 / +0.0120 | -0.12 / +0.13 | -0.0016 / +0.0010 |
| D5 `0.75` | -0.181 / +0.170 | -0.156 / +0.123 | -0.0127 / +0.0088 | -0.14 / +0.10 | -0.0013 / +0.0008 |
| D5 `0.2` | +0.000 / +0.000 | +0.000 / +0.000 | +0.0000 / +0.0000 | +0.00 / +0.00 | +0.0000 / +0.0000 |
| D5 `3.0` | +0.000 / +0.000 | +0.000 / +0.000 | +0.0000 / +0.0000 | +0.00 / +0.00 | +0.0000 / +0.0000 |
| D6 `3.5` | +0.000 / +0.000 | +0.000 / -0.000 | +0.0000 / -0.0000 | +0.00 / +0.00 | +0.0000 / -0.0000 |
| D7 `25` | +0.000 / +0.000 | +0.000 / +0.000 | +0.0000 / +0.0000 | +0.00 / +0.00 | +0.0000 / +0.0000 |
| D7 `5` | +0.000 / +0.000 | +0.000 / +0.000 | +0.0000 / +0.0000 | +0.00 / +0.00 | +0.0000 / +0.0000 |
| D7 `0.8` (clip) | -0.620 / +0.171 | -0.637 / +0.401 | +0.0000 / +0.0000 | +0.00 / +0.00 | +0.0000 / +0.0000 |
| D8 `12.5` | +0.070 / -0.054 | -0.039 / +0.029 | -0.0019 / +0.0017 | +0.01 / +0.00 | +0.0131 / -0.0084 |
| D8 `0.04` | +0.018 / +0.000 | -0.005 / +0.000 | -0.0001 / +0.0000 | +0.01 / +0.00 | +0.0050 / +0.0000 |
| D9 `0.0005` | +0.021 / -0.019 | +0.020 / -0.019 | -0.0000 / -0.0001 | +0.03 / -0.03 | +0.0013 / -0.0012 |
| D10 `0.5` | +0.034 / -0.034 | +0.031 / -0.031 | -0.0002 / +0.0002 | +0.04 / -0.04 | +0.0287 / -0.0282 |
| D10 `1/4` | -0.000 / +0.000 | -0.016 / +0.013 | -0.0001 / -0.0002 | -0.06 / +0.04 | +0.0000 / +0.0000 |
| D10 `1/2` | -0.000 / +0.000 | -0.010 / +0.006 | +0.0002 / -0.0002 | -0.03 / +0.02 | +0.0000 / +0.0000 |
| D11 `0.01` | -0.000 / +0.000 | -0.000 / +0.000 | +0.0000 / +0.0000 | +0.00 / +0.00 | +0.0000 / +0.0000 |
| D12 `1.5` | +0.000 / +0.000 | -0.000 / -0.000 | +0.0000 / +0.0000 | +0.00 / +0.00 | +0.0000 / +0.0000 |

- **They matter** (PSNR moves 0.1 dB or more for ±10 %): D4 `0.8` (the target of the line-spread fit; ±0.22 dB), D5 `0.75` (the σ_G factor; ±0.17 dB), D4 `0.85`, D3 `0.8` (the energy fraction of the kernel radius; −0.12 dB at +10 %) and D7 `0.8` (the clip threshold; −0.6 dB at −10 %, because the brightest pixels of the natural and text scenes cross it).
- **For D5 the direction is clear and agrees with mathematics:** raising 0.75 by 10 % raises both PSNRs (+0.17 and +0.12 dB) and the MTF50 (+0.009 cycles/px) and costs +0.10 % overshoot and +0.0008 in noise gain. The exact factor for a Gaussian is 0.780, 4 % above today's.
- **Moderate** (0.01 to 0.07 dB): D3 `4·r80` and `6` px, D8, D9, D10 `0.5` (which also moves the flat-patch noise gain by ∓0.03) and its bands.
- **Inert on these scenes** (under 0.01 dB, under 0.001 cycles/px): D1 (the spectral ramp), D2, D3 `0.99` and `0.002`, the σ_G clamps `0.2` and `3.0`, D6, D7 `25` and `5`, D11, D12. D7 `25` and `5` act only where two or three channels pass the clip threshold, and D11 and D12 only where a weight reaches its floor or the ratio its cap; these scenes have no such pixels, so "inert here" is not "inert". A highlight scene (the two lamps of the synthetic golden set) is part of RC-11.

#### Spec 04 · output sharpening (screen preset, method 1; 24 Kodak images at scale 1 and at 0.33)

The table gives the mean change of the output in codes (−10 % / +10 %) at full size and at scale 0.33, the largest single pixel at full size, and the mean change at ×0.5 / ×1.5.

| Constant | mean change at full size, codes: −10 % / +10 % | mean change at scale 0.33: −10 % / +10 % | largest single pixel (codes, full size, ±10 %) | mean change at full size at ×0.5 / ×1.5 |
|---|---|---|---|---|
| O1 slope `0.7` | 0.3137 / 0.2931 | 0.2034 / 0.1941 | 48 | 1.7448 / 1.3487 |
| O1 offset `0.3` | 0.1315 / 0.1283 | 0.0863 / 0.0847 | 41 | 0.6895 / 0.6263 |
| O2 `5.0` | 0.0152 / 0.0070 | 0.0000 / 0.0006 | 18 | 0.2795 / 0.0071 |
| O2 `2.5` | 0.0000 / 0.0000 | 0.0000 / 0.0001 | 0 | 0.0152 / 0.0070 |
| O3 `0.15` | 0.0000 / 0.0000 | 0.0846 / 0.0829 | 0 | 0.0000 / 0.0000 |
| O6 pull fade `0.25` | 0.0037 / 0.0038 | 0.0007 / 0.0007 | 7 | 0.0172 / 0.0198 |
| O8 `16` | 0.0000 / 0.0000 | 0.0000 / 0.0000 | 0 | 0.0000 / 0.0000 |
| O4 radius `1.0` | 0.3137 / 0.2931 | 0.2034 / 0.1941 | 48 | 1.7448 / 1.3487 |
| O4 amount `100` | 0.4358 / 0.4270 | 0.1964 / 0.1952 | 65 | 2.2701 / 2.0457 |
| O4 halo `60` | 0.0516 / 0.0517 | 0.0078 / 0.0079 | 46 | 0.2602 / 0.2560 |
| O4 threshold `1` | 0.0570 / 0.0555 | 0.0321 / 0.0309 | 19 | 0.2981 / 0.2609 |

- **The σ map is the biggest lever of the whole package.** O1's slope `0.7` and offset `0.3` move the picture by 0.3 and 0.13 codes in the mean for ±10 % at full size, 10 to 20 times any constant of spec 01 on the same kind of image, because they set the sharpening radius of every export. At the preset's radius `R = 1.0`, `σ' = 0.7 + 0.3 = 1.0`.
- **The tap rule (O2) is nearly inert** (0.015 and 0.007 codes for the `5.0`; 0.000 for the `2.5` at ±10 %): at the preset's σ the cut is at 3.5σ, with a tail mass of 4.7e-4. It is not inert at ×0.5 of the `5.0` (0.28 codes).
- **O3, the σ knee, acts only below full size** (0.000 at scale 1, 0.08 at scale 0.33).
- **O6 (the limiter's pull fade) moves 0.004 codes; O8 (the ratio cap 16) moves nothing**: the cap was never reached on these images.
- **The preset values** (the lower four rows; the radius row equals the slope row of O1 to the digit, as it must at `R = 1`: `σ' = R·0.7 + 0.3`) move pictures by 0.05 to 0.4 codes for ±10 %; they are the owner's look (section 6, question 4).
- The radius `0.97·√(R − 0.09)` branch, the print preset, method 2 and the automatic method are not reached by the screen preset at `R = 1.0`: 0.0000 by construction, not measured.

#### Spec 02 · the cubics, the exponent

No render path uses the log-guided demosaic, and neither its float form nor the GPU kernel evaluates the cubics (they use `log2` and `exp2`), so L1 and L2 cannot change a picture: 0 by construction. Their own objective, the error of the fixed-point function, was measured over every input in section 4 (F-01). L3, the exponent, is unmeasured and unreached.

#### Summary by what moves a default picture

| Moves a default picture | Rows | Largest measured effect (±10 %, mean codes) |
|---|---|---|
| Every export | O1, O2, O6 (O3 at reduced size) | 0.3 (O1) |
| Every photograph, a little | A13, A7, A1, A12, A6 | 0.03 (A13) |
| Dark and noisy photographs | A8, A11, A9, A14; A7, A1, A13, A12, A6 more than in clean scenes | 0.17 (A8) |
| No picture, today | A2 to A5, L1 to L3, D1 to D14, O10 to O12 | 0 |


### 2.2a The baseline at `Tuning::ORACLE` (RC-06)

This section records the floor every later fit is held to. It fits nothing and changes no picture. It is the harness of RC-05 run at `Tuning::ORACLE` on every objective and gate of section 2.3, in one command (`tools/fit/baseline.sh`, Appendix B) that prints the tables below from the raw output of the examples; none of the numbers was typed. It was run on 2026-11-18 at `main` bda02d7b with the Kodak and McMaster sets of `tools/fetch/demosaic-sets.sh` (24 and 18 files, SHA-256 verified), on a 4-core Linux machine without a GPU, in the release profile.

**How to read it.** A fit of spec 01, 03 or 04 is accepted only if it is no worse than the number in its table, on the fit sets and on the held-out sets (section 2.3, rules 3 and 4; the acceptance rule is the failed-gate rule of `AGENTS.md`). "Fit" and "held out" are the split of section 2.3 rule 2: the procedural set and Kodak are the fit sets, McMaster and the second noise seed (`B2`, and the second seed of the spec 03 cases) are held out. The CA scenes, the spec 03 scenes and the edge scenes are generated by our code and have no held-out part. A cell where the adaptive ratio is not the best method (Kodak and McMaster at scale 1, where LMMSE leads) is part of the floor: the R6 rule counts cells won (7 of 9), and a fit must not lose one. The gates are the existing ones: the spec 01 section 6.3 nine cells, zipper, false colour and noisy table; the spec 03 section 6.3 bars and section 6.4 table; the spec 04 section 9 edge profile.

**What is not in the tables.** *Group P (the real sensor, PixelShift200): not run: PixelShift200 absent; evaluation data, never committed.* The sets of `demosaic_gate` that are not fetchable here (Gharbi hard, Gharbi moire) are not run either, for the same reason. Group C (two real A1 crops) has no ground truth and a change of 0 at its own base, so it has no baseline number; a fit that runs it reports its change against `Tuning::ORACLE` (`--photo`). The 8-bit sets are re-mosaicked pictures and the noise models are stand-ins (G5): the numbers are a floor for the algorithm, not a statement about a sensor.

**What was checked.** `tools/fit/baseline.sh` refuses to run when `Tuning::DEFAULT` differs from `Tuning::ORACLE` in any of the three packages (it compares `--list`), because `demosaic_gate` and `capture_sharpen_ab` have no override flag and run what ships; today they are equal (`DEFAULT = ORACLE`), so those two gates are at the oracle too. The base numbers of section 2.2 reproduce in this run (A 35.678 dB, B 27.604 dB, the four spec 03 cases, the CA row). The spec 01 table of section 6.4 reproduces to its digits (nine-cell mean 36.05 dB against Menon 34.43 and LMMSE 34.28, best in 7 of 9; zipper 20.25 against 39.13 % on the procedural set), and so do the spec 03 section 6.4 rows. Two runs of the command gave byte-identical raw output and tables (Appendix B).

**One difference from the spec text.** Spec 04 section 9 gives the screen preset at full size as rise 2.49 to 1.91 px and overshoot 3.2 %; the test that makes the profile prints 2.489 to 1.905 px and 3.45 % / 3.38 % today (the test's own limit is 3.5 %, `edge_profile_screen_preset`). The other rows agree (s = 0.4: 2.27 px, 0.4 %; print: 1.10 px, 21.1 %; noise gain x1.030). The floor is the measured number; the spec text is left as it is until RC-09, which is where the edge profile is the gate. The test now prints three decimals of rise and two of overshoot so that the floor has digits; no assertion changed.

Every number below was printed by `tools/fit/baseline.sh` at `Tuning::ORACLE` (`Tuning::DEFAULT` equal to it). Data: kodak (24), mcmaster (18), synthetic (cpu_demosaic.rs) (7), CA scenes 600x400 (ours, seed 2024) (5). Not run (absent on this machine): Gharbi hard, Gharbi moire, PixelShift200; the real-sensor group P is therefore not run: PixelShift200 absent; evaluation data, never committed. Everything here is evaluation and local fitting data, re-mosaicked 8-bit sRGB pictures and stand-in noise models, no real-sensor ground truth.

#### Spec 01 - adaptive ratio demosaic

**The nine cells** (`demosaic_gate`: mean PSNR in linear light, 10 px border off, nominal model of spec 01 section 6.3). The R6 rule counts the best of four methods per cell.

| set | split | signal scale | adaptive ratio (dB) | Menon (dB) | LMMSE (dB) | best |
|---|---|---|---|---|---|---|
| synthetic | fit | 1 | 34.94 | 31.60 | 32.00 | adaptive_ratio |
| synthetic | fit | 0.1 | 35.22 | 31.60 | 31.37 | adaptive_ratio |
| synthetic | fit | 0.01 | 35.42 | 31.60 | 30.16 | adaptive_ratio |
| Kodak | fit | 1 | 36.38 | 36.14 | 36.41 | lmmse |
| Kodak | fit | 0.1 | 37.34 | 36.14 | 35.98 | adaptive_ratio |
| Kodak | fit | 0.01 | 36.97 | 36.14 | 35.56 | adaptive_ratio |
| McMaster | held out | 1 | 35.03 | 35.54 | 36.33 | lmmse |
| McMaster | held out | 0.1 | 36.41 | 35.54 | 35.82 | adaptive_ratio |
| McMaster | held out | 0.01 | 36.72 | 35.54 | 34.93 | adaptive_ratio |
| **mean of the 9 cells** |  |  | **36.048** | **34.425** | **34.284** | cells won 7 / 0 / 2 |

**Zipper and false colour** (`demosaic_gate`, signal scale 1, linear light).

| set | split | images | zipper AR (% of edge px) | Menon | LMMSE | false colour AR (% of px) | Menon | LMMSE | chroma dE00 AR |
|---|---|---|---|---|---|---|---|---|---|
| synthetic | fit | 7 | 20.25 | 39.13 | 39.59 | 14.434 | 15.251 | 14.998 | 2.472 |
| Kodak | fit | 24 | 24.15 | 34.29 | 33.32 | 4.267 | 7.635 | 6.775 | 1.513 |
| McMaster | held out | 18 | 26.55 | 41.77 | 40.00 | 6.270 | 11.203 | 10.000 | 1.714 |

**The noisy table** (`demosaic_gate`, Gaussian noise `a x + b`, signal scale 1, matching model for the adaptive ratio; each cell is demosaic / after the R6 bilateral denoise).

| set | split | level | adaptive ratio: demosaic / + R6 (dB) | Menon | LMMSE |
|---|---|---|---|---|---|
| synthetic | fit | ISO 100-like | 33.87 / 34.16 | 31.38 / 31.55 | 31.79 / 31.96 |
| synthetic | fit | ISO 800-like | 31.90 / 33.10 | 30.26 / 31.20 | 30.70 / 31.63 |
| synthetic | fit | ISO 1600-like | 30.77 / 32.44 | 29.44 / 30.83 | 29.89 / 31.30 |
| synthetic | fit | ISO 6400-like | 27.17 / 29.85 | 26.31 / 28.87 | 26.81 / 29.41 |
| Kodak | fit | ISO 100-like | 36.38 / 36.39 | 36.03 / 36.05 | 36.31 / 36.32 |
| Kodak | fit | ISO 800-like | 36.37 / 36.50 | 35.36 / 35.57 | 35.69 / 35.82 |
| Kodak | fit | ISO 1600-like | 35.70 / 36.00 | 34.75 / 35.17 | 35.12 / 35.41 |
| Kodak | fit | ISO 6400-like | 32.32 / 33.62 | 31.84 / 33.30 | 32.33 / 33.55 |
| McMaster | held out | ISO 100-like | 35.10 / 35.14 | 35.46 / 35.51 | 36.25 / 36.29 |
| McMaster | held out | ISO 800-like | 35.64 / 35.91 | 34.93 / 35.28 | 35.72 / 36.01 |
| McMaster | held out | ISO 1600-like | 35.31 / 35.80 | 34.43 / 35.03 | 35.21 / 35.72 |
| McMaster | held out | ISO 6400-like | 32.63 / 34.20 | 31.93 / 33.64 | 32.66 / 34.22 |

**The groups of the fit harness** (`fit_demosaic --held --artifacts --edge --ca`).

| group | split | mean PSNR (dB) |
|---|---|---|
| A (procedural 7, Kodak 24, McMaster 18) | all 49 | 35.678 |
| A, fit set (procedural and Kodak) | fit | 36.052 |
| A, held out (McMaster) | held out | 35.032 |
| B (dark and noisy, 17 images, noise seed 1) | all | 27.604 |
| B, fit images | fit | 27.207 |
| B, held-out images (McMaster) | held out | 28.331 |
| B2 (the same 17 images, second noise seed) | held out | 27.601 |
| P (PixelShift200, real sensor) | - | not run: PixelShift200 absent; evaluation data, never committed |

Artifacts, the 49 images of group A: zipper 24.473 % of edge pixels, false colour 6.455 % of pixels, chroma dE00 1.724.

Slanted edge through the demosaic (levels 0.05 and 0.25 of white): MTF50 0.5906 cycles/px, rise 0.8275 px, overshoot 2.531 %, undershoot 1.909 %, flat-patch sd 0.00977.

**The CA path** (`ca_green_residual`'s scenes through `fit_demosaic --ca`; the residual red and blue displacement against green after R9, and the PSNR of all three planes). These scenes are ours and generated, so none is held out.

| scene (600 x 400) | red residual RMS (px) | blue residual RMS (px) | PSNR, all planes (dB) |
|---|---|---|---|
| strength 0.5 | 0.055 | 0.043 | 43.38 |
| strength 1 | 0.114 | 0.092 | 40.09 |
| strength 2 | 0.204 | 0.175 | 36.10 |
| strength 1, barrel | 0.112 | 0.094 | 40.31 |
| strength 2, barrel | 0.187 | 0.161 | 36.62 |

#### Spec 03 - diffraction deconvolution

**Section 6.4** (`capture_sharpen_ab`: synthetic scenes at a 3.76 um pitch; PSNR against the sharp scene, slanted-edge MTF50, 10 to 90 % rise, overshoot, flat-patch luma noise relative to the input). The scenes are ours; no part of it is held out.

| f | ISO | method | text PSNR (dB) | natural PSNR (dB) | MTF50 (c/px) | rise (px) | overshoot (%) | undershoot (%) | flat noise (x) |
|---|---|---|---|---|---|---|---|---|---|
| 5.6 | ISO 100 | input | 32.40 | 37.18 | 0.323 | 1.46 | 0.3 | 0.1 | 1.000 |
| 5.6 | ISO 100 | Richardson-Lucy Auto | 28.65 | 35.97 | 0.419 | 0.91 | 13.9 | 4.3 | 1.216 |
| 5.6 | ISO 100 | diffraction | 33.32 | 38.02 | 0.389 | 1.15 | 1.1 | 1.0 | 1.000 |
| 5.6 | ISO 800 | input | 31.17 | 35.12 | 0.334 | 1.46 | 0.7 | 0.3 | 1.000 |
| 5.6 | ISO 800 | Richardson-Lucy Auto | 28.89 | 34.41 | 0.408 | 0.98 | 10.8 | 4.1 | 1.168 |
| 5.6 | ISO 800 | diffraction | 31.86 | 35.66 | 0.400 | 1.12 | 1.9 | 1.5 | 1.000 |
| 5.6 | ISO 6400 | input | 25.75 | 28.24 | 0.301 | 1.59 | 3.1 | 0.4 | 1.000 |
| 5.6 | ISO 6400 | Richardson-Lucy Auto | 24.99 | 27.51 | 0.387 | 1.11 | 8.9 | 2.9 | 1.118 |
| 5.6 | ISO 6400 | diffraction | 25.68 | 28.25 | 0.375 | 1.24 | 3.1 | 0.8 | 1.000 |
| 8 | ISO 100 | input | 29.30 | 34.56 | 0.273 | 1.94 | 0.3 | 0.1 | 1.000 |
| 8 | ISO 100 | Richardson-Lucy Auto | 33.19 | 38.04 | 0.380 | 1.11 | 7.0 | 2.7 | 1.255 |
| 8 | ISO 100 | diffraction | 30.68 | 35.80 | 0.340 | 1.26 | 1.7 | 1.6 | 1.005 |
| 8 | ISO 800 | input | 28.65 | 33.31 | 0.275 | 1.92 | 0.7 | 0.3 | 1.000 |
| 8 | ISO 800 | Richardson-Lucy Auto | 31.28 | 34.68 | 0.363 | 1.18 | 6.1 | 2.7 | 1.200 |
| 8 | ISO 800 | diffraction | 29.84 | 34.14 | 0.354 | 1.26 | 2.4 | 2.0 | 1.007 |
| 8 | ISO 6400 | input | 24.87 | 27.80 | 0.269 | 2.06 | 3.1 | 0.4 | 1.000 |
| 8 | ISO 6400 | Richardson-Lucy Auto | 24.92 | 27.21 | 0.299 | 1.48 | 5.6 | 1.7 | 1.141 |
| 8 | ISO 6400 | diffraction | 24.58 | 27.71 | 0.314 | 1.27 | 4.7 | 2.3 | 1.011 |
| 11 | ISO 100 | input | 26.50 | 32.30 | 0.227 | 2.29 | 0.3 | 0.1 | 1.000 |
| 11 | ISO 100 | Richardson-Lucy Auto | 31.24 | 35.60 | 0.312 | 1.55 | 4.6 | 1.8 | 1.255 |
| 11 | ISO 100 | diffraction | 28.49 | 34.05 | 0.303 | 1.59 | 2.7 | 2.7 | 1.200 |
| 11 | ISO 100 | diffraction, noise gate | 28.40 | 33.95 | 0.303 | 1.59 | 2.7 | 2.7 | 1.081 |
| 11 | ISO 800 | input | 26.15 | 31.52 | 0.223 | 2.31 | 0.7 | 0.3 | 1.000 |
| 11 | ISO 800 | Richardson-Lucy Auto | 29.42 | 33.11 | 0.306 | 1.66 | 3.7 | 1.7 | 1.200 |
| 11 | ISO 800 | diffraction | 27.88 | 32.56 | 0.319 | 1.55 | 3.3 | 3.1 | 1.242 |
| 11 | ISO 800 | diffraction, noise gate | 27.86 | 32.58 | 0.319 | 1.55 | 3.3 | 3.1 | 1.103 |
| 11 | ISO 6400 | input | 23.65 | 27.21 | 0.211 | 2.36 | 3.1 | 0.4 | 1.000 |
| 11 | ISO 6400 | Richardson-Lucy Auto | 24.16 | 26.81 | 0.264 | 2.03 | 3.7 | 0.7 | 1.141 |
| 11 | ISO 6400 | diffraction | 22.79 | 26.13 | 0.289 | 1.60 | 6.7 | 3.6 | 1.275 |
| 11 | ISO 6400 | diffraction, noise gate | 23.04 | 26.67 | 0.289 | 1.60 | 6.3 | 3.7 | 1.114 |
| 16 | ISO 100 | input | 23.75 | 30.11 | 0.184 | 2.63 | 0.3 | 0.1 | 1.000 |
| 16 | ISO 100 | Richardson-Lucy Auto | 26.96 | 32.25 | 0.238 | 2.17 | 1.0 | 0.3 | 1.255 |
| 16 | ISO 100 | diffraction | 26.08 | 31.89 | 0.245 | 1.97 | 3.0 | 2.1 | 1.577 |
| 16 | ISO 100 | diffraction, noise gate | 25.87 | 31.67 | 0.244 | 1.97 | 3.0 | 2.1 | 1.085 |
| 16 | ISO 800 | input | 23.56 | 29.62 | 0.183 | 2.67 | 0.7 | 0.3 | 1.000 |
| 16 | ISO 800 | Richardson-Lucy Auto | 25.96 | 30.85 | 0.223 | 2.27 | 0.9 | 0.5 | 1.200 |
| 16 | ISO 800 | diffraction | 25.70 | 30.75 | 0.235 | 1.93 | 4.2 | 2.5 | 1.617 |
| 16 | ISO 800 | diffraction, noise gate | 25.66 | 30.83 | 0.236 | 1.93 | 4.2 | 2.5 | 1.101 |
| 16 | ISO 6400 | input | 22.01 | 26.40 | 0.183 | 2.67 | 3.1 | 0.4 | 1.000 |
| 16 | ISO 6400 | Richardson-Lucy Auto | 22.66 | 26.15 | 0.201 | 2.44 | 3.5 | 0.6 | 1.141 |
| 16 | ISO 6400 | diffraction | 21.30 | 24.37 | 0.227 | 1.97 | 7.4 | 3.2 | 1.714 |
| 16 | ISO 6400 | diffraction, noise gate | 21.84 | 25.99 | 0.232 | 1.99 | 5.9 | 3.4 | 1.118 |
| 22 | ISO 100 | input | 21.45 | 28.38 | 0.152 | 3.15 | 0.3 | 0.1 | 1.000 |
| 22 | ISO 100 | Richardson-Lucy Auto | 23.37 | 29.69 | 0.180 | 2.68 | 0.6 | 0.2 | 1.255 |
| 22 | ISO 100 | diffraction | 24.19 | 30.29 | 0.196 | 2.30 | 4.6 | 3.1 | 1.759 |
| 22 | ISO 100 | diffraction, noise gate | 23.90 | 30.02 | 0.196 | 2.30 | 4.6 | 3.1 | 1.080 |
| 22 | ISO 800 | input | 21.34 | 28.05 | 0.150 | 3.14 | 0.7 | 0.3 | 1.000 |
| 22 | ISO 800 | Richardson-Lucy Auto | 22.84 | 28.87 | 0.173 | 2.78 | 0.9 | 0.4 | 1.200 |
| 22 | ISO 800 | diffraction | 23.85 | 29.37 | 0.195 | 2.28 | 5.5 | 3.3 | 1.815 |
| 22 | ISO 800 | diffraction, noise gate | 23.80 | 29.45 | 0.195 | 2.28 | 5.4 | 3.3 | 1.098 |
| 22 | ISO 6400 | input | 20.35 | 25.58 | 0.153 | 3.08 | 3.1 | 0.4 | 1.000 |
| 22 | ISO 6400 | Richardson-Lucy Auto | 20.88 | 25.39 | 0.169 | 2.85 | 3.5 | 0.5 | 1.141 |
| 22 | ISO 6400 | diffraction | 20.09 | 23.52 | 0.202 | 2.25 | 10.1 | 4.3 | 1.900 |
| 22 | ISO 6400 | diffraction, noise gate | 20.65 | 25.33 | 0.203 | 2.28 | 6.8 | 4.6 | 1.116 |

**The cases of the fit harness** (`fit_diffraction --held`: the four cases of section 2.2 and the same four with a second noise seed, held out).

| case | split | text PSNR (dB) | natural PSNR (dB) | MTF50 (c/px) | rise (px) | overshoot (%) | flat noise (x) |
|---|---|---|---|---|---|---|---|
| f/8 ISO 100 | fit | 30.68 | 35.80 | 0.3401 | 1.262 | 1.65 | 1.0050 |
| f/8 ISO 800 | fit | 29.84 | 34.14 | 0.3541 | 1.258 | 2.38 | 1.0070 |
| f/16 ISO 100 | fit | 26.08 | 31.89 | 0.2450 | 1.970 | 3.00 | 1.5772 |
| f/16 ISO 800 | fit | 25.70 | 30.75 | 0.2352 | 1.927 | 4.19 | 1.6170 |
| f/8 ISO 100 seed 2 | held out | 30.70 | 35.77 | 0.3466 | 1.266 | 1.67 | 1.0012 |
| f/8 ISO 800 seed 2 | held out | 29.90 | 34.11 | 0.3440 | 1.284 | 3.15 | 1.0057 |
| f/16 ISO 100 seed 2 | held out | 26.10 | 31.87 | 0.2441 | 1.971 | 2.86 | 1.5297 |
| f/16 ISO 800 seed 2 | held out | 25.73 | 30.69 | 0.2427 | 1.928 | 5.03 | 1.6589 |

#### Spec 04 - output sharpening

**The edge profile of section 9** (the tests `edge_profile_screen_preset` and `edge_profile_noise_gain_on_a_flat_patch` of `cpu_output_sharpen.rs`, a soft slanted edge of Gaussian sigma 0.8 px, 0.1 to 0.5).

| case | rise before (px) | rise after (px) | overshoot / undershoot (%) |
|---|---|---|---|
| screen, full size | 2.489 | 1.905 | 3.45 / 3.38 |
| screen, s = 0.4 | 2.489 | 2.274 | 0.43 / 0.43 |
| screen, halo 0, full size | 2.489 | 1.869 | 4.79 / 4.79 |
| print, full size | 2.489 | 1.101 | 21.12 / 21.12 |

Noise gain on a flat patch (sigma 0.004): x1.0298 with the 1-level threshold, x1.3786 with threshold 0.

**The edge profile of the fit harness** (`fit_sharpen --held --edge`: a seeded logistic edge with noise, screen preset; ours, no part held out). The change measures (the 48 outputs of Kodak at scales 1 and 0.33) are 0 at the base by definition; a later evaluation adds McMaster as the held-out set with `--held`.

| scale | rise 10-90 (px) | overshoot (%) | undershoot (%) | noise gain (x) |
|---|---|---|---|---|
| 1 | 2.2487 | 0.851 | 1.080 | 1.0319 |
| 0.4 | 2.6139 | 0.243 | 0.351 | 1.0213 |

### 2.3 How each could be fitted

**The protocol every fit follows.**

1. *One harness.* The fits run in `ym-gpu` examples that take the package's `Tuning` (RC-03) and print numbers. They are extensions of the harnesses that exist: `demosaic_gate` (spec 01), `capture_sharpen_ab` (spec 03), `ca_green_residual` (the CA path) and the edge profile of `output_sharpen_bench` (spec 04). None needs a GPU, a model or a network. The search driver is a short script, `tools/fit/search.py` (standard library only; the grids and the coordinate descent below, as `grid` and `descent`; RC-05 decided it is Python so that it needs no build and calls the examples), and `numpy` is used only for the closed-form and exhaustive polynomial searches. Every run is deterministic: scenes from `ym_gpu::cpu::synth`, noise from the seeded splitmix64 of `capture_sharpen_ab.rs`, no thread-order dependence.
2. *Split.* Fit on the synthetic set and Kodak; hold out McMaster and a second noise seed. Both are reported. A constant is changed only if it is better on the fit sets and not worse on the held-out set. Groups have one to three free constants, so over-fitting 18 to 42 images is limited; the held-out set is the check.
3. *No change for nothing.* An optimum that beats today's value by less than 0.01 dB (PSNR objectives) or by less than 1 % of the objective (others) is not taken. Today's value stays, and its derivation becomes "fitted on 2026-11-xx: within 0.00x of the optimum", which is a derivation we can show.
4. *The acceptance rule* (the owner's failed-gate rule applied to a fit). A fitted value replaces today's only if, **measured and reported in the PR and in `docs/STATUS.md`**: (a) it is no worse on the group's objective, on the fit sets and on the held-out set; (b) it is no worse on every number of the spec's existing quality gate (§6.3 or §6.4 of the spec, listed per group below); (c) the golden check moves no picture by more than the numbers reported (RC-13). A fit that is better on one number and worse on another is not taken without the owner (section 6, question 1).
5. *Cost* is CPU time on this box (4 shared vCPU), from the measured times of the sensitivity runs: one evaluation of the 66 spec 01 scenes takes 15 to 25 s here; one evaluation of the 24 × 2 spec 04 images takes about a minute; one evaluation of the spec 03 scenes at two f-numbers and two ISO levels takes about 10 s. No Modal run is needed for any fit. None may be spent without the owner.

**Spec 01.**

- **A1 `1.6`, pass-1 direction scores.** *Objective:* the nine-cell mean PSNR of spec 01 §6.3 (procedural, Kodak, McMaster at scales 1, 0.1, 0.01, nominal model), with the zipper and false-colour rates of the same gate held at or below today's. *Data:* the synthetic set (ours); Kodak and McMaster (local, never committed). *Method:* a one-dimensional grid from 0.4 to 3.2 in steps of 0.1 (29 evaluations), then 0.02 steps around the best. *Cost:* about 10 minutes. *Accept:* the rule above, with `demosaic_gate`'s PSNR, zipper, false colour and noisy tables as the gate.
- **A6 `0.03` and A7 `0.75` (dominance), chroma penalty.** *Objective:* the same PSNR, and the false-colour rate on saturated edges (the penalty acts on chroma edges, so false colour is the sharper number). *Data:* as A1. *Method:* a two-dimensional grid, 0.01 to 0.12 by 0.01 for A6 and 0.5 to 1.2 by 0.05 for A7 (12 × 15 = 180 evaluations of the clean scenes only, about 45 minutes), then refine. A6 and A7 act together (A7 turns the penalty on, A6 sets its floor), so they are fitted together. *Cost:* about 1 hour. *Accept:* as A1. The sensitivity runs say the optimum is not at today's values: PSNR rises when either is raised (section 2.2).
- **A8 `0.75` (amount), A9 `5`, A10 `4T/3T`, shadow sharpening.** *Objective:* **not PSNR.** The step is a sharpening, and PSNR falls as its amount rises (section 2.2: B-scene PSNR +0.23 dB when the amount is halved). The honest objective is the one the step exists for: edge sharpness gained per unit of noise added. Proposed: on the slanted edge (ours) and on the dark noisy scenes, the 10 to 90 % rise or the MTF50 of an edge at 0.25 of white, subject to the flat-patch noise gain and the overshoot staying at or below today's numbers. *Data:* synthetic slanted edge at several levels (ours), Kodak darkened with the high noise model. *Method:* a three-dimensional grid (5 × 5 × 4 points), then the constraint check. *Cost:* about 30 minutes. *Accept:* the MTF50 or rise is no worse, the noise gain and overshoot are no higher, and the B-scene PSNR is not lower by more than 0.05 dB. If the objective cannot separate the values (many points tie), today's values stay (rule 3).
- **A11 `3`, shadow colour matching, and A14 `25 dB`.** *Objective for A11:* PSNR of the B scenes at the four ISO-like levels of the noisy gate, after the demosaic and after the R6 denoise (the gate's own two columns). *Objective for A14:* the SNR at 18 % grey where shadow matching starts to help. *Method:* A11 is a one-dimensional grid (0.5 to 8, 16 points). For A14, run matching on and off over a ladder of noise models (8 levels from 40 dB to 15 dB at 18 % grey) and report the crossover SNR; A14 is set to it, rounded to 1 dB. *Cost:* about 20 minutes. *Accept:* as A1; the clean scenes must be unchanged bit for bit when the switch is off.
- **A12 `50`/`0.04` (the target SNR 10) and A13 `0.2` (the target 20).** The algebra is exact (A12 is the root of `x² = s²(g x + r²)` with `s = 10`; the record confirms it). *Objective:* the B-scene and clean-scene PSNR as the two targets move (a blend point between colour-difference and colour-ratio interpolation). *Method:* two one-dimensional grids (target 5 to 20, and 10 to 40), 14 evaluations each. *Cost:* about 15 minutes. *Accept:* as A1.
- **A2 `2.5`, A3 `0.1` px, A4 `18`/`102`, the CA path.** *Objective:* the residual lateral CA of `metrics::lateral_ca_residual` and the PSNR of the green and of all planes, on the five scenes of `ca_green_residual` (three CA strengths without distortion, two with 1.5 % barrel distortion). *Data:* fully ours (the example generates its scenes). *Method:* coordinate descent over the four constants within ±50 %, then check the 121/123 identity. *Cost:* each evaluation of the example at 600 × 400 (five scenes, four variants) takes about 11 to 13 s here (measured), so about 100 evaluations take about 20 minutes. *Accept:* the residual is no larger and the PSNR no lower at every strength. **This path is not reached by any render** (`INTEGRATION.md` §6: the chain corrects CA on the mosaic first and passes `ca = None`), so the fit changes no picture; it is worth doing only to give the numbers a derivation (section 6, question 5).

**Spec 02.**

- **L1, L2 the cubics.** *Objective:* the maximum error over every input of the fixed-point function, as the spec's own bounds state it (log2: `L − 4096·log2 y` in codes; exp2: the smallest `k` with `−(1 + k·t) ≤ C − t ≤ k·t`), with the anchors exact (`L(1) = 0`, `L(2) = 4096`, `L(65535) = 65535`; `C(4096e) = 2^e`) and, as a tie-break, no more step-backs than today (23 for log2; none for exp2). *Data:* none; it is mathematics. *Method:* the continuous minimax by Lawson's iteration for the seed, then an exhaustive integer search in the neighbourhood with the exact integer arithmetic of the code (numpy). *Cost:* measured: 5 s for log2 (6,561 candidates), 25 s per pass for exp2; a few minutes in all. *Accept:* the new maximum error is no larger, the anchors and monotonicity (exp2) hold, the oracle vectors still pass under `Tuning::ORACLE`. **Measured result** (section 2.2): log2 4.978 → 4.404 codes; exp2 `k` 1.236e-4 → 9.02e-5.
- **L3 the exponent 3.** Not shipped (no render path uses the log-guided demosaic). The same harness (`log_guided_eval`) can fit it; do it only if a render path is ever added (RC-12, optional).

**Spec 03** (not the default of R8; no render changes unless the owner promotes it).

- **D1 the spectral weight.** *Derivation from public data we ship:* the weight of the polychromatic PSF is the camera's luma response across 461 to 619 nm. Take the published Weta curves of the eleven bodies (`rawtoaces-data`, Apache-2.0, already in the repository for the colour work, `docs/03`), combine them with the luma weights of the stage (0.299, 0.587, 0.114) and a daylight illuminant, normalise to 1 above 540 nm, and compare the resulting weights with today's ramp. *Measure:* the L1 distance between the two radial PSFs, and the sensitivity (section 2.2: ±10 % of the ramp width moves the PSNR by 0.006 dB or less). *Data:* Apache-2.0, no new data. *Cost:* minutes. *Accept:* the new weights are taken only if the PSF changes by more than the evaluation's resolution *and* the gate is not worse; otherwise today's ramp stays and the derivation says that the public curves agree with it to within the stated distance.
- **D2 to D4, D6, the sampling and truncation constants.** *Objective:* an error budget: the PSF kernel, the line-spread estimate and the Gaussian radius are checked against a converged reference (a finer radial step, a longer support, a Gaussian of known σ). *Method:* convergence runs and closed forms (`x80 = 1.2816σ` for a Gaussian; the tail mass beyond `3.5σ` is 4.65e-4). *Cost:* minutes. *Accept:* each constant is kept if its error against the converged reference is below the budget, which is stated once (proposed: 1e-3 of the kernel's energy). The section 2.2 numbers say these are almost inert (0.00 dB for the radius rule and the radial step).
- **D5 `0.75` and D7 `0.8`, D8 to D12 the fits.** *Objective:* the bars of spec 03 §6.3 and §7.1: MTF50 no lower than Richardson-Lucy Auto at f/8 and above, overshoot no higher, flat-patch noise at most ×1.10, and PSNR against the sharp scene; the multi-objective is taken lexicographically in that order. *Data:* the three synthetic scenes of `capture_sharpen_ab` (ours) and Kodak crops blurred with the stage's own PSF (local). *Method:* coordinate descent over D5, D7 (threshold), D8, D9, D10 in order of sensitivity. *Cost:* about 10 s per evaluation, so about 1 hour for 300 evaluations. *Accept:* as the protocol; the existing table of spec 03 §6.4 is the floor.

**Spec 04** (on by default).

- **O2 the tap rule** (`(int)(5σ + 2.5) | 1`, caps 13 and 33). *Derivation:* a tail-mass budget. The pixel-integrated kernel of `n = 2h + 1` taps drops `erfc((h + ½)/(σ√2))` of the Gaussian's mass before it is renormalised. Measured for today's rule over the screen map: 4.7e-4 at σ = 1.0 (the preset's value, a cut at 3.5σ), 8.6e-4 at σ = 1.35, 1.2e-3 at σ = 1.7 and 1.5e-3 at σ = 2.05; at σ ≤ 0.5 the 3-tap and 5-tap floors cut at 5σ or more. So today's linear rule is a documented approximation of "cut where the lost mass is 1.5e-3 at most". State that budget, derive `h(σ)` from it and compare; where the two rules give the same tap count (probably every σ of the screen map), nothing changes and the derivation is the budget. *Cost:* minutes. *Accept:* the tap count is equal to or above today's for every σ from 0.1 to 2.5, so no render gets a shorter kernel, and the edge profile of spec 04 §9 does not move by more than 0.01 px of rise.
- **O1 the radius-to-σ map.** *Derivation:* the map has a step of 0.03 at `R = 0.5`. A continuous map that keeps `σ(1) = 1.0` and today's values for `R ≥ 0.5` and fixes the part below 0.5 is a mathematical repair (the default preset never reads it). The slope and offset are a look (section 6, question 4); they are not fitted.
- **O3 `0.15` the σ knee, O6 and O7 the limiter and map widths.** *Objective:* the edge-profile numbers of spec 04 §9 (rise, overshoot, noise gain, halo suppression) at the scales 1.0 and 0.4. *Data:* the synthetic soft edge (ours). *Method:* one-dimensional grids. *Cost:* minutes. *Accept:* rise no worse, overshoot and noise gain no higher.
- **O4, O5, O11 the presets and the halo map.** Design. Not fitted.
- **O10 the Scharr values.** Maths, no fit. Method 2 is unreachable.

### 2.4 What a change breaks

**The oracle vectors** (`crates/ym-gpu/tests/cpu_data/`): 18 files for spec 01 (tolerance 1e-5), 16 for spec 02 (1.5e-5 to 6.2e-5), 23 for spec 03 (0, 1e-6 or 2e-5) and 16 for spec 04 (0 or 1e-5), 73 files and 4.5 MiB in all. They pin today's constants bit for bit. The tests that read them: `cpu_adaptive_ratio.rs` (`vectors_match_within_their_tolerance`, `derived_values_match_the_vectors`, which pins K, T and the SNR at 18 % grey, `the_phase_vectors_agree_after_the_shift`), `cpu_adaptive_ratio_ca.rs`, `gpu_adaptive_ratio.rs` (16 stable vectors against the oracle and two tie vectors with a loose bound), `cpu_log_guided.rs` and `gpu_log_guided.rs`, `cpu_diffraction.rs`, `gpu_diffraction.rs` (the `plan_*` vectors pin the PSF kernels, σ_G and the halo), `cpu_output_sharpen.rs` and `gpu_output_sharpen.rs`. The C++ that made them is not in the repository and the vectors are never hand-edited, so a re-fit would make them fail with no way to refresh them. *Proposal* (RC-03, RC-04): keep them, unchanged, as the oracle of the *structure* (borders, phases, ties, decision logic) by running them under `Tuning::ORACLE`; add a second set generated by our own Rust at the shipped `Tuning::DEFAULT`, named `*_self` and never called an oracle, that pins the shipped numbers as a regression; and write each spec's §6.3 quality gate as numeric floors (today's measured numbers, section 2.2) that a new value must meet. No gate is loosened: the floors are today's numbers. The GPU tests then run the same two sets, with the generated WGSL constant block taken from the same `Tuning`.

**The golden images.** The default set is 61 synthetic pictures (`tools/golden/expected.json`, 1248 × 832; its log has 19 entries and the last is pipeline 0.9.0, because 0.10.0 and 0.11.0 changed no default picture, `docs/16`). Every one passes through the adaptive ratio demosaic (the default of R6) and through output sharpening (on by default), so an accepted fit of A1, A6 to A13 or O1 to O3 changes all 61. The sensitivity of section 2.2 says by how much on photographs: a 10 % change of one constant moves the mean by 0.005 to 0.3 codes. `golden update --note` refuses a change without a note and records the largest ΔE00 it moved. The GPU golden jobs compare the device with the CPU pictures of the same checkout, so they follow the refresh with no change of their own tolerances. One refresh for all accepted fits (RC-13), not one per fit.

**`PIPELINE_VERSION`.** `crates/ym-core/src/version.rs:8` ("Bump it whenever the output pixels change for the same input and settings"). It is pinned by `crates/ym-cli/tests/cli.rs` and written in `docs/01` (the versions-written-into-outputs section, line 965), `docs/13`, `docs/15`, `docs/16` (line 150), `docs/25` and `docs/STATUS.md`. It is written into every export's XMP, so a photograph exported before and after differs in `ym:PipelineVersion` as well as in its pixels (`docs/09`). Question 2 asks which step.

**The GPU parity tests.** Each fit that changes a number changes the CPU value and the generated WGSL block together (RC-03), so the parity tests keep comparing like with like. Two cases need care and are named in the spec (§5): the decision near-ties of the adaptive ratio (the saturation penalty's threshold A7 is a comparison, so a new value moves where the two tie vectors tie) and the window counts of the halo limiter. A change of A7 may move the `edge_knight` and `ca_weighting` tie bounds (`gpu_adaptive_ratio.rs`: at most 20 and 100 samples over, today 10 and 58), which the PR must re-measure and report, not widen silently.

**The DNG path.** A DNG picture is demosaiced by the same adaptive ratio (`docs/09` line 199: "the demosaic's shadow step stays on"), so a fit of spec 01 changes DNG pixels but not the output-sharpening ones (a DNG is never sharpened). The `dng` job (`dng_validate`, `dngfit`, `tools/dng/run_gates.sh`) must pass, and its numbers be reported, in RC-13.

**`docs/14`'s row and `docs/16`'s wording.** The row (line 66 of `docs/14`) says "tuned constants" and "known techniques … plus the project's own combinations and tuned constants"; `docs/16` lines 53, 175 and 179 say "known techniques plus our own combinations and constants". The prior-art record (§2.1) says that wording does not cover the novel combinations, the constants or the pieces not assessed. *Now* (this PR): the row gets "constants being re-derived, `docs/29`" and nothing else. *After RC-14:* "constants taken over from research and re-derived by the project (`docs/29`)", the same words in `docs/16`.

**The CHANGELOG.** `[Unreleased]` is empty at the base. The entry of RC-14 goes under `### Changed` of the release that carries RC-13 (the text is in section 5). A release with only RC-01 to RC-06 needs no entry beyond one line under `### Changed` ("The raw-algorithm constants have recorded derivations; no picture changes"), because no picture changes.

### 2.5 Licences of the data and the tools

Against `docs/14` and `docs/validation-data.md` (where the datasets are classed).

| Item | Used for | Licence | Verdict |
|---|---|---|---|
| Procedural scenes (`ym_gpu::cpu::synth`), the slanted edges and scenes of `capture_sharpen_ab.rs`, the CA scenes of `ca_green_residual.rs`, seeded noise | fitting, gates, new vectors | ours, Apache-2.0 | Free to use and to commit as generated vectors. |
| Kodak, 24 images (500 × 500 crops from `cszn/FFDNet`, pinned by commit and SHA-256 in `tools/fetch/demosaic-sets.json`) | local fitting and evaluation | "Eastman Kodak released the images for unrestricted use"; the copy's repository records no licence (the fetch table's own note) | Evaluation and local fitting only, never committed, no picture made from them committed (`docs/validation-data.md` row 13). Fetched on this box: 24 of 24 files verified. |
| McMaster, 18 images (`JingyunLiang/SwinIR` copy, pinned) | local fitting and evaluation | "for academic research; cite the paper" (Zhang, Wu, Buades, Li, 2011); the copy records no licence | As Kodak. Held out of the fit and used for validation. Fetched on this box: 18 of 18 verified. |
| Gharbi hard and moiré test sets | optional second held-out set | research use (`docs/validation-data.md` row 13); `data.csail.mit.edu` is not reachable from the sandbox | Not used by this plan. If reachable later: evaluation only, as above. |
| PixelShift200 | the one real-sensor ground truth | CC BY 4.0 (row 13); `huggingface.co` is not reachable from the sandbox | Would be the best data for A1, A6 to A11. If it is ever fetched, a fit that uses it needs the attribution in `NOTICE` only if weights trained on it ship (`docs/14` line 88); a fitted number is not such a weight, but the derivation names the dataset. |
| Weta spectral curves of 11 Sony bodies (`rawtoaces-data`) | D1: the spectral weight of the PSF | Apache-2.0, already shipped (`docs/14` line 11) | Free to use. |
| The nine local-eval Sony files and the owner's `readme-photos` A1 files | looking at real mosaics ("how much does it change"); the C scenes of section 2.2 | unclear (local-eval) / the owner's own photographs | LOCAL tier. Never committed, never a fit target; no ground truth. |
| `numpy` (exhaustive and minimax polynomial searches, `rc/polys.py`) | RC-01 script | BSD-3-Clause | A development tool of the repository (Python is for training, fixtures and tooling, not for the app or `ym`; `AGENTS.md` rule 6). Not shipped. The same permissive class as the numpy already used by `training/`. |
| Rust code of the fit harness | RC-05 | ours | No new dependency. |
| Papers (Hamilton and Adams, Menon et al., Burns on slanted edges, Lawson's algorithm, Born and Wolf) | the objectives and the maths | published papers | May be read. |
| RawTherapee, librtprocess, darktable, LibRaw, rawspeed, dcraw, RapidRAW, rawler, the `dng` crate | **none** | GPL, LGPL, AGPL (`AGENTS.md` hard rule 1) | **Not read for this work.** The prior-art record already cites AMaZE and RCD (GPL) as "related" for the coincidence of 0.75 and the RCD structure; this plan does not go back to them, and no fit takes a value from them. |


## 3. Gaps

- **G1. No constant has a recorded derivation.** `docs/14` calls them "tuned constants"; the prior-art record found no public source for the constants of question 12; the owner's answer is that they come from research. Nothing in the repository says which research, what objective, or what data. After this plan each row has a status: *Maths* (a test proves it), *Fit* (an objective, a date, the numbers), *Design* (the owner's look) or *Guard* (a numeric safeguard).
- **G2. A constant has no single home.** Ten spec 01 constants (A1 to A4, A6 to A11) each exist twice, as a literal in `adaptive_ratio.rs` and as a literal in `adaptive_ratio.wgsl` (A1 `:777` and `:345`; A6 `:1102-1103`, `:1150` and `:637-638`, `:656`; and so on, section 2.1). Spec 03 repeats the values of its constants block (`diffraction.wgsl:97-105`) and its clip weights (`:282`), and spec 04 repeats its epsilons, its ratio cap, its pull fade and its gradient weights (`output_sharpen.wgsl:94-100`, `:337-340`). The GPU parity tests compare the two copies only on the inputs of the vectors, and the saturation penalty (A6, A7) runs only where `0.75·ΣX < ΣG`, so a copy that drifted in a branch the vectors do not reach would pass. A fit would have to edit up to six places per constant by hand (A6 has six literals).
- **G3. The vectors pin today's values, and they cannot be regenerated.** The 73 vector files (18 + 16 + 23 + 16) were made by C++ references that are not in the repository (`INTEGRATION.md` §9; the record, §2.1) and "are never hand-edited". A re-fit makes them fail by design, with no way to refresh them. They are also the only proof that the Rust port is the specified algorithm, which must be kept.
- **G4. No fit harness takes a constant.** The objectives exist (`demosaic_gate`, `capture_sharpen_ab`, `ca_green_residual`, the edge profile of `output_sharpen_bench`) but read their constants from the code, so the only way to try a value is to edit the code and rebuild. The measurements of this audit needed a scratch copy for that reason.
- **G5. There is no real-sensor ground truth here.** Kodak and McMaster are 8-bit sRGB pictures that were already processed and demosaiced by someone else; re-mosaicking them tests the algorithm, not a sensor, its noise or its lens. The sets with real sensor data are not reachable from the sandbox: PixelShift200 (CC BY 4.0) is on `huggingface.co`, Gharbi on `data.csail.mit.edu` (`docs/validation-data.md`, row 13). The noise models of the noisy scenes are stand-ins (the spec's own "high" model; the ISO ladder of `capture_sharpen_ab`). The fits must say this, and the shadow constants (A8 to A11, A14) in particular are checked on real mosaics only as "how much does the picture change", never as "is it better".
- **G6. PSNR is not the objective of a sharpening.** Raising the shadow sharpening's amount lowers the PSNR of the noisy scenes (section 2.2); the same holds for output sharpening and for deconvolution. A fit by PSNR would drive every sharpening constant to zero. Their objectives are edge sharpness against noise and overshoot (section 2.3).
- **G7. Most rows act on no render.** Of the 44 rows, the CA path (A2 to A5), the log-guided cubics and exponent (L1 to L3), all of spec 03 (D1 to D14), and the print, method-2 and automatic-method constants of spec 04 (O10 to O12) do not change any picture today. Spec 03 is not the default of R8 (`DEFAULT_CAPTURE_METHOD = RichardsonLucy`, `crates/ym-core/src/pipeline/raw.rs:241`); the CA path is skipped because R5b corrects CA on the mosaic and passes `ca = None` (`INTEGRATION.md` §6); no render path uses the log-guided demosaic. The sensitivity run confirms it: A2, A3 and A4 changed 0.000 codes on every scene. A re-fit of these rows has a documentation value and no render risk; the rows that move pictures are the ones worth the golden refresh.
- **G8. The words disagree.** `docs/14` says "tuned constants" and `docs/16` says "known techniques, plus our own combinations and constants"; the owner says the constants come from research; `docs/validation-data.md` row 13 says Kodak and McMaster are for evaluation only, while the owner's standing rule allows local fitting.
- **G9. 121 and 123 are not constants.** They are `1 + 18 + 102` and `121 + 2`, the weight of the centre green at zero misregistration and the divisor there (`adaptive_ratio.rs:994`; no literal 121 or 123 exists in the code). Question 12 lists them as if they were independent.

## 4. Findings

Each finding has the fix and the test that proves it. Numbers are from section 2.2 unless a line says otherwise.

**F-01. The two cubics are near-minimax but not minimax, and their objective was not recorded.** Measured over every input (scratch scripts `polys.py`, `log2b.py` and `exp2.py`, appendix: exact integer emulation of `log2_fixed` and `exp2_fixed`):

| | today | best found | how |
|---|---|---|---|
| log2, largest `abs(L − 4096·log2 y)` | 4.978 codes (range −4.978 to +2.979) | 4.404 codes (range −4.404 to +4.378), `(46653, 19114, 5229)/32768`, 20 step-backs (today 23), largest step-back 1 code | exhaustive integer search, ±120 around the continuous minimax `(1.422865, −0.582085, 0.159220)`, anchors kept (`A − B + C = 32768`) |
| exp2, smallest `k` with `−(1 + k·t) ≤ C − t ≤ k·t` | 1.236e-4 (upper 1.236e-4, lower 1.172e-4) | 9.02e-5, `(45559, 14919, 5047)/65536`, monotone, `C(4096e) = 2^e` exact, maximum 65518 | hill climb in integers from four starts; best of 9.02e-5, 9.08e-5, 9.08e-5, 1.16e-4 |

The objective depends on a choice. If the objective is the width of the error range instead of its largest value, the log2 optimum is a range of 6.67 codes (today 7.96) at `(46517, 18988, 5239)`, with an error that is almost all negative; it would need a rounding offset to centre. That is a structural change, so it is not proposed. *Why it matters:* it is the one place where "from first principles" is exact; the fit is done in minutes and gives each cubic an objective and an optimality test. No render uses the fixed-point form, so no picture changes. *Fix:* RC-01. *Test:* `cpu_log_guided.rs::log2_and_exp2_fixed_meet_their_exhaustive_bounds` with the new bounds, and a new test that recomputes the integer search around the stored triple and fails if a neighbour is better by the stated objective; the 16 oracle vectors stay green under `Tuning::ORACLE`.

**F-02. 121 and 123 are consequences, and the zero-misregistration weight is 98.4 %.** *Fix:* RC-02 documents them, RC-03 writes `w(t) = 1 + A + B` at `t = 1` as an expression in the code. *Test:* a unit test that `centre_weight(κ = 0) == (1 + 18 + 102)` and `/ (1 + 18 + 102 + 2) == 121/123`, and that the pass-2 value at `κ = 0` differs from the CA-off value only by that 1.6 % share.

**F-03. The constants that exist both in a CPU file and in a shader have no single home** (ten of spec 01, and the values named in G2 for specs 03 and 04). *Fix:* RC-03: `Tuning` structs, a generated WGSL constant block, one name per job (the three 0.75 become `SAT_DOMINANCE` and `SHADOW_AMOUNT`; the saturation floor 0.03 keeps one name for steps 1 and 2). *Test:* `constants_registry.rs` fails when a registered value appears as a literal in the CPU file or the shader outside the block, and when the generated block differs from the registry; GPU parity on the oracle vectors with `Tuning::ORACLE` passes unchanged.

**F-04. Output sharpening is the biggest lever on a default picture; the rest of the effect is small, and many rows do not act at all.** By the tables of section 2.2, on photographs and a ±10 % move: the output sharpening σ map (O1: 0.3 and 0.13 codes mean at full size, 0.2 and 0.09 at a reduced size) is ten times the largest demosaic constant (A13: 0.029 codes, A7: 0.022, A1: 0.014); the other spec 04 rows are at or below 0.015 codes (O2, O6), except the σ knee O3, which acts only on exports at a reduced size (0.08 codes at scale 0.33). In the dark noisy scenes the shadow group leads (A8 0.17 codes, A7 0.09, A1 0.08, A11 0.06, A13 0.06). The CA rows, the cubics and every row of spec 03 move no render. *Fix:* the task order puts the render-moving fits behind one golden refresh (RC-13) and the rows that move nothing into maths and documentation tasks that change no picture. *Test:* the "pictures unchanged" gate of RC-01 to RC-06 is `golden check` with the old hashes.

**F-05. PSNR rewards turning a sharpening off.** *Fix:* section 2.3's objectives for A8 to A10, D5 to D12 and O3, O6, O7 are edge sharpness against noise gain and overshoot, never PSNR alone. *Test:* every fit report (RC-05's harness) prints the edge numbers next to the PSNR, and the acceptance check fails a value that raises the noise gain or the overshoot.

**F-06. The oracle vectors need two roles.** *Fix:* RC-04: they stay, unedited, and are run with `Tuning::ORACLE` (proof that the structure is the spec's); a generator writes a second set from our own Rust at the shipped `Tuning` (`<algorithm>_self/`, named so, never called an oracle), which the tests run with `Tuning::DEFAULT`; the gates of each spec's §6.3 become numeric floors in the spec. *Test:* both sets are read by the same test functions, a test regenerates the self set into a temporary folder and compares it with the committed bytes (deterministic generation), and a test fails if `Tuning::DEFAULT` differs from `ORACLE` while the self set is missing.

**F-07. The data limit (G5) must be said where the numbers are.** *Fix:* each fit's report line and the STATUS gate entry carry "re-mosaicked 8-bit sRGB and stand-in noise models; no real-sensor ground truth"; PixelShift200 is added to the harness when it is reachable (the harness already reads `testdata/demosaic/pixelshift200` when present). *Test:* the report tool prints the data note, and `rc_report_has_data_note` (a test of the report formatter) fails when it is missing.

**F-08. Documents.** *Fix:* RC-14. *Test:* the registry test above also checks that every ID of this document's inventory is in the registry and the other way round, and that no registry entry says "pending" once RC-14 has landed.

**F-09. The golden images and the version move together, once.** Any accepted fit of A1, A6 to A14 or O1 to O3 changes the picture of every golden file that passes through the adaptive ratio demosaic or the output sharpening: all 61 synthetic pictures (the default set, 1248 × 832, `docs/13`). *Fix:* RC-13 batches the accepted fits behind one `golden update --note`, bumps `PIPELINE_VERSION` (`crates/ym-core/src/version.rs:8`, 0.11.1 since the NEX lens tables of 2026-11-13) and the tests and documents that pin it. *Test:* `golden check` fails before the refresh and passes after; `crates/ym-cli/tests/cli.rs` and the GPU golden jobs pass; the largest ΔE00 any picture moved by is in the note.

## 5. Tasks

Each task is one branch (`rc-01-cubics` style), one pull request, with the usual gate (`cargo fmt --check`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test --workspace`, `cargo deny check`). RC-01 to RC-06 change no picture: their *Done when* includes `golden check` with the old hashes. RC-07 to RC-12 are the fits; each changes pictures only if its fit is accepted, and an accepted fit lands behind the one refresh of RC-13. Order: RC-01 to RC-06 in the order given (RC-03 before RC-04), then RC-07 to RC-09 (the fits that move pictures), then RC-10 to RC-12, then RC-13, RC-14.

**RC-01 · The two cubics, re-derived** (F-01; spec 02; changes no picture). `tools/fit/poly_fit.py` (numpy only, deterministic, prints the objective, the search box and the best five) re-derives L1 and L2 with the objective of section 2.3. The fixed-point reference keeps the old triples as `const ORACLE_*` for the 16 oracle vectors (test-only use) and ships the new ones. `docs/algorithms/02` §3.2 and §3.4 restate the cubics, the error bounds and the objective; `research/raw-algorithms-prior-art.md` L10 gets a line pointing here. *Done when:* the exhaustive bounds test passes with the new numbers (log2 at most 4.41 codes, exp2 `k` at most 9.1e-5, anchors exact, exp2 monotone, log2 step-backs at most 23); a new test re-runs the integer search around the stored triple and finds nothing better; the 16 oracle vectors pass with the old triples; the float form, the GPU parity tests and `golden check` are unchanged (identical hashes, no version bump).

*Done (branch `rc-01-cubics`, commit "RC-01: the two cubics re-derived").* `tools/fit/poly_fit.py` reproduces the stored triples in 27 s. log2: `(46653, 19114, 5229)/32768`, largest error 4.404 codes (range −4.404 to +4.378; was 4.978), 20 step-backs of one code (was 23), anchors exact; the search box is B in [18954, 19194], C in [5157, 5277] (4952 feasible triples), best five 4.404, 4.417, 4.420, 4.420, 4.420 codes. exp2: `(45559, 14919, 5047)/65536`, k = 9.020e-5 (upper 8.97e-5, lower 9.02e-5; was 1.236e-4), maximum 65518 (was 65522), monotone, exact at 2^e; the search is the climb of section 4 plus every triple within ±4 of its end (729 triples), four tie at 9.020e-5. Both agree with section 4 (4.404 and 9.02e-5; the 4.41, 9.1e-5 and 23 of the done-when are met). The old triples are `ORACLE_LOG2_CUBIC` and `ORACLE_EXP2_CUBIC` (`log_guided.rs`), used by `demosaic_fixed_with(…, ORACLE_CUBICS)` for the 16 vectors only. Tests (`cpu_log_guided.rs`): `log2_and_exp2_fixed_meet_their_exhaustive_bounds`, `the_oracle_cubics_keep_the_bounds_of_the_first_release`, `nothing_beats_the_stored_cubics_by_the_stated_objective_near_them` (1225 log2 triples, 729 exp2 triples), `vectors_fixed_form`. The pipeline version is unchanged and `golden check` gives the stored hashes.

**RC-02 · A derivation for every constant that mathematics gives** (F-02, G1, G9; no picture changes). For each row of section 2.1 marked *Maths*, write the derivation into its spec's constants table (a "Derivation" column) and prove it with a test: K and T satisfy `x/σ(x) = 10` and `= s` for random `(g, r)` (algebra of the record's A3); `1/6` is the exact quadrature weight of the first ring (compare with a fine integral); `0.3125 = 10/32` and `0.09375 = 3/32` are the normalised Scharr kernel; `x80 = 1.2816σ` for a Gaussian, so the exact σ_G factor is 0.780 (the shipped 0.75 stays until RC-11 decides); `121 = 1 + 18 + 102`; the erf kernel sums to 1 after truncation; the tail masses of the O2 rule over the screen map (section 2.3). Rows marked *Fit* get their task ID in the table; *Design* rows say "the owner's look, `docs/29` §6"; *Guard* rows say what they protect. *Done when:* every one of the 44 rows has a status in the spec tables; the maths tests pass; no source file other than tests changes; `golden check` unchanged.

*Done (branch `rc-02-derivations`, commit "RC-02: a derivation for every constant that mathematics gives").* Every row of section 2.1 has a status in a new table of its spec (spec 01 §2.3, spec 02 §2.6, spec 03 §2.6 "Derivation of every constant", spec 04 §3.8 "Derivation of every constant"): **Maths 12** (A5, A12, A15, L1, L2, D2, D3, D4, D5, D6, O2, O10), **Fit 24** (A1 to A4, A6 to A11, A13, A14, L3, D1, D7 to D12, D14, O3, O6, O7, each with its task), **Design 5** (O1, O4, O5, O11, O12), **Guard 3** (D13, O8, O9). Ten new tests, all in the modules of the code they prove (test code only; no source change, no API widened): `k_and_t_are_the_signal_levels_where_the_snr_is_ten_and_s` (400 random `(g, r, detail, strength)`), `the_pass_2_centre_weight_at_zero_misregistration_is_1_plus_18_plus_102` (`w(κ = 0)` read back from the pass-2 value is within 0.5 of 121, and the value is the centre plus 2/123 of the way to the neighbours' mean; the plain path keeps the centre, so the CA path at `κ = 0` differs from CA off only by that 1.6 % share; `κ = 0.25` gives 7.09375), `the_integer_gaussian_weights_sum_to_their_divisor` (571), `one_sixth_is_the_exact_weight_of_sample_zero_in_the_encircled_energy` (against Simpson's rule on 50 random profiles, 1e-12; 1/8 is off by p0/24) and `the_planner_uses_that_weight_to_find_the_80_percent_radius` (the planner's `r80` is reproduced with 1/6 to 1e-9 and not with 1/8), `the_exact_sigma_g_factor_for_a_gaussian_is_0_780_and_the_shipped_one_is_0_75` (x80 = 1.28155σ, factor 0.78031, shipped 3.9 % below), `the_gaussian_radius_cuts_less_than_the_tail_budget_up_to_sigma_two` (4.65e-4), `the_truncated_erf_kernel_sums_to_one_and_drops_exactly_the_tail_mass`, `the_tap_rule_loses_under_two_per_mille_at_the_screen_map_points_and_3_6_at_worst`, `the_gradient_weights_are_the_normalised_scharr_kernel` (through `gradient_gain`, five directions, three levels). **Two derivations did not hold as this plan wrote them.** (1) O2: the tap rule's lost mass is 4.7e-4, 8.6e-4, 1.2e-3 and 1.5e-3 at the four σ the plan names, but it is a sawtooth in σ and the worst over the screen map is **3.6e-3 at σ = 1.89** (11 taps, cut at 2.9σ), not under 1.5e-3; RC-09 sets the budget. (2) D6: the radius `min(floor(3.5σ_G + 1), 8)` meets the 4.65e-4 budget only up to σ_G = 2.0; the cap of 8 governs above (4.6e-3 at σ_G = 3, which the clamp allows and no lens reaches). Also noted: D2, D3 (except the 1/6), D4 are written as error budgets whose convergence runs belong to RC-11 (the spec table says so); O12 has no clean status (listed as Design, unused in v1).

**RC-03 · One home per constant, CPU and GPU** (F-03, F-08; no picture changes). Per package a `Tuning` struct (`adaptive_ratio::Tuning`, `diffraction::Tuning`, `output_sharpen::Tuning`; the log-guided cubics from RC-01) with one named field per constant of section 2.1, a doc comment per field (kind, spec section, derivation status, task), `Tuning::ORACLE` (today's values, bit for bit) and `Tuning::DEFAULT` (equal to `ORACLE` until a fit lands). The per-frame `Derived` value holds a copy; the hot loops read it. The kernel wrappers prepend a WGSL `const` block generated from the same `Tuning` and the shaders lose their literals. `pub const REGISTRY` lists id, name, value, kind and derivation status. The three 0.75 become two names, 121 and 123 an expression. *Done when:* every existing test passes unchanged; every oracle vector passes with `Tuning::ORACLE` and with `Tuning::DEFAULT`; `constants_registry.rs` fails for a literal of a registered value outside the block and for a registry entry missing from section 2.1 (and the reverse); the CPU time of `adaptive_ratio_timing` and `output_sharpen_bench` is within 2 % of the base; the device parity numbers in `tests/gpu_adaptive_ratio.rs` are unchanged; `golden check` unchanged.

*Done (branch `rc-03-tuning`, commit "RC-03: one home per constant, CPU and GPU").* `adaptive_ratio::Tuning` (15 fields), `diffraction::Tuning` (38) and `output_sharpen::Tuning` (37), each with `ORACLE`, `DEFAULT` (equal), a field doc per row (kind, section, status, task), a `pub const REGISTRY` (`log_guided::REGISTRY` for the cubics and L3) and a `wgsl_block()` that the kernel wrappers prepend (12, 14 and 9 device constants); the per-frame `Derived`, `DiffractionPlan` and `SharpenPlan` hold the copy, and the vectors run under `ORACLE` and `DEFAULT`. The three 0.75 are `SAT_DOMINANCE` and `SHADOW_AMOUNT`, 121 and 123 are `Tuning::centre_weight(1.0)` (+ 2), 0.03 is `CHROMA_FLOOR` in both steps. The device runs `Tuning::DEFAULT` only and refuses another until RC-04 keys the pipelines. Tests: `constants_registry.rs` (11: `the_registries_and_the_inventory_list_the_same_rows`, `every_kind_is_the_status_of_the_specs_derivation_table`, `the_block_of_each_package_is_the_registrys_device_constants`, `every_generated_literal_is_the_f32_the_shader_had_before` (bits as naga reads them), `no_registered_value_is_written_again_outside_its_home` with one allow-list entry, and the rest); each failure was shown once by a temporary edit (a `0.75` back in the CPU file and in the shader, a changed block constant, a row renamed in §2.1, a row deleted from §2.1 so that the registry has an entry with no row). Speed, release, base d4748fdd against the branch, interleaved runs on a 4-core cloud machine. Adaptive ratio (CPU seconds, median of 5): plain whole 0.136 to 0.138 s/MP (+1.5 %), banded 1 thread 0.134 to 0.134, shadow matching whole 0.245 to 0.243, banded 0.223 to 0.222; a second run of 9 read every row within 2 % on the minimum. Output sharpening (wall ms, median / minimum of 9, two runs): screen full size 1 thread +1.2 % / +2.0 % and +2.0 % / +1.5 %, 4 threads +1.1 % / +2.5 % and +3.5 % / +1.9 %; 2048 px from 6000, 1 thread +2.8 % / +1.4 % and +1.3 % / +1.4 %; halo 0, 1 thread +6.0 % / +6.9 % and -2.0 % / -5.1 %; print, 4 threads +12.5 % / +12.8 % and -10.9 % / +1.1 %. The same build reads +6 % in one run and -2 % in the next, so this machine's timing cannot settle 2 %; the instruction count does: Callgrind at 0.3 MP, 1 thread, 5.065 G on the base and 5.057 G on the branch (-0.2 %), with the per-band function at 3.4917 G against 3.4896 G and the same instruction mix. A variant that runs the per-pixel code on `Tuning::DEFAULT` as literals (a `const D: bool` instance) was tried and not kept: 5.083 G (+0.3 %). Device tie bounds unchanged: `edge_knight` 10 samples over 1e-4 (max 4.20e-4), `ca_weighting` 58 (max 6.05e-4). Three real files (A300, A7 III, NEX-7) to JPEG and 16-bit TIFF, CPU and lavapipe, base against branch: 12 of 12 outputs byte-identical (debug `ym`).

**RC-04 · The vector regime** (F-06; no picture changes). The oracle vectors stay unedited and run under `Tuning::ORACLE`; a generator (`crates/ym-gpu/examples/gen_self_vectors.rs`) writes `<algorithm>_self/*.json` from our own Rust under `Tuning::DEFAULT`, with the same fields and the tolerance of the oracle set; the tests read both folders; the spec's §6.1 and `INTEGRATION.md` §9 describe the two classes ("oracle: structure; self: regression of the shipped constants"). *Done when:* the generator is deterministic (a test regenerates into a temp folder and compares bytes), the self set at `DEFAULT = ORACLE` agrees with the oracle set within its tolerance (a check that the generator is right), and the GPU tests run both sets.

*Done (branch `rc-04-vectors`, commits "RC-04: the device keys its pipelines by their constants", "RC-04: a generator for the self vectors, and tests that read both sets", "RC-04: compare the self set by bytes only on Linux x86_64").* **Keying.** The pipeline cache of `Kernels` is keyed by the kernel and by the generated WGSL constants block it was built from (`KernelId::variant`: `None` for the shipped constants, so `ORACLE` and `DEFAULT` share a pipeline while they are equal). The three tuned packages pass their `Tuning` through `PassList::dispatch_tuned`; the three refusals of a non-shipped `Tuning` are gone; there is no user-facing switch. Per package a GPU test runs the shipped tuning and two others on one device and back: each result equals the CPU under the same tuning, the three differ, and a second run gives the first run's bits (it fails when the key is dropped). **Generator.** `crates/ym-gpu/examples/gen_self_vectors.rs` (module `tests/self_gen`, shared with the tests) reads each oracle file and writes `<algorithm>_self/<same name>.json` with the oracle's inputs, parameters, field names and tolerances and our own results under `Tuning::DEFAULT` (the shipped cubics for the log-guided demosaic): 73 files, 4.3 MiB (18 + 16 + 23 + 16). The oracle files are untouched. The CPU and GPU tests read both folders, the oracle under `ORACLE`, the self set under `DEFAULT`. `tests/self_vectors.rs`: the committed set equals a regeneration; two generations give the same bytes; the generator run under the oracle constants agrees with all 73 oracle vectors within their tolerances (derived values 1e-9, log-guided 1e-7 relative); the committed set agrees with the oracle set while `DEFAULT = ORACLE`; the F-06 rule (a moved `DEFAULT` with a missing or stale self set fails, shown on a synthetic set). **The log-guided finding:** the shipped RC-01 cubics differ from the oracle cubics, so the fixed form of the self set differs from the oracle's by up to 1.10e-3 (`detail_minus50`), not within the vector's tolerance. That agreement is checked through the generator under `ORACLE` (and `ORACLE_CUBICS`); the committed log-guided set is held to 1.2e-3 on the fixed form and to the vector's tolerance on the float form and the derived values. **The platform rule.** CI runs `cargo test --workspace` on a Mac, whose libm can differ from glibc's in the last digit. The committed set is compared with a regeneration byte for byte only on Linux x86_64, where it was written; elsewhere number by number within 1e-12 relative, or the vector's tolerance if that is tighter; determinism on one machine is a byte comparison everywhere. Every self file's description says so. `serde_json` gets `float_roundtrip` in the dev-dependencies of `ym-gpu` (no new crate). Device tie bounds unchanged: `edge_knight` 10 samples over 1e-4 (max 4.20e-4) and `ca_weighting` 58 (max 6.06e-4), on both sets. No picture changes and `PIPELINE_VERSION` is unchanged; RC-13 takes the next number.

**RC-05 · The fit harness and its data** (F-05, F-07, G4, G5). Examples `fit_demosaic`, `fit_diffraction`, `fit_sharpen` that take `Tuning` overrides on the command line, run the scenes of section 2.3 with the split of the protocol, and print one JSON line per evaluation plus the data note; a small driver (`tools/fit/search.py`, stdlib only) for the grids and coordinate descent; the CA scenes of `ca_green_residual` as a library function. `docs/validation-data.md` row 13 and `tools/fetch/README.md` say "evaluation and local fitting, never committed". *Done when:* a documented command reproduces the base numbers of section 2.2 to the last digit; the harness reads PixelShift200 when `testdata/demosaic/pixelshift200` exists; no dataset byte is in the repository (`git ls-files` check in the PR).

*Done (branch `rc-05-harness`, commits "RC-05: fit_demosaic and fit_diffraction, the CA scenes as a shared module, the search driver", "RC-05: fit_sharpen, the tests of the harness, --ca-only" and the docs).* **Three examples** in `crates/ym-gpu/examples/`: `fit_demosaic` (spec 01: groups A, B, C, the CA scenes, and the PixelShift200 group P), `fit_diffraction` (spec 03) and `fit_sharpen` (spec 04). Each takes `Tuning` overrides by field name (`--set sat_dominance=0.8`, `--rel chroma_floor=0.9,1.1`, `--eval "label@a=1,b=2"`, `--from oracle|default`, `--list`), evaluates them against the starting `Tuning`, and prints a data note and one JSON line per evaluation (the appendix has the keys). No environment variable and no patched literal: an unknown field name or a value the field cannot hold is refused before a scene loads. The overrides, the data note and the measures of change are one module (`tests/common/fit.rs`); the CA scenes of `ca_green_residual` are another (`tests/common/ca_scenes.rs`: `ca_scene_set`, `ca_path_result`, `run_variant` under a `Tuning`), and `ca_green_residual` prints the same tables from it (checked: its output at 300 x 200 is identical before and after, apart from the timing column). **The driver** `tools/fit/search.py` (standard library only; `grid`, `descent` and `table`): deterministic, with an objective path into the JSON line, constraints (`edge.overshoot_pct<=2.0`), a minimum gain (0.01 by default, rule 3 of section 2.3) and bounds; `tools/fit/test_search.py` tests it against a fake example. **The split of the protocol:** the fit set is the procedural set and Kodak, McMaster is held out (`A_fit`, `A_held`, `B_fit`, `B_held` in `fit_demosaic`; `held_change_codes` in `fit_sharpen`); `--held` adds what costs extra time: the second noise seed of the dark scenes (`B2`, `fit_demosaic`), the same four cases with a second seed (`mean_delta_held`, `fit_diffraction`) and McMaster (`fit_sharpen`). **The objectives next to the PSNR (F-05):** `--artifacts` (zipper, false colour, chroma difference of the gate), `--edge` (a slanted edge at 0.25 of white through the demosaic: MTF50, rise, overshoot, flat noise; and the edge profile of spec 04 §9 at scales 1.0 and 0.4: rise, overshoot, undershoot, noise gain) and, in `fit_diffraction`, the MTF50, rise, overshoot and flat-patch noise of every case. **The data (F-07):** the first line of every run is the data note: the sets used with their image counts, the sets skipped and why, "never committed", and the stand-in noise models and no real-sensor ground truth. `fit_demosaic` reads `testdata/demosaic/pixelshift200` as group P (the same protocol as A) when the folder has images and says "skipped" in the data note when it has not; it is not on this machine, so P is `null` in every line here, and the test `fit_demosaic_skips_a_missing_set_and_reads_pixelshift200_when_it_is_there` shows both cases on a one-image folder. No dataset byte is in the repository (`git ls-files` check in the report of this task); `docs/validation-data.md` row 13 and `tools/fetch/README.md` say "evaluation and local fitting, never committed". **Reproduction of section 2.2, at `Tuning::ORACLE`** (the appendix has the command and the table): the base numbers reproduce to the last digit (A 35.678 dB, B 27.604 dB; the four spec 03 cases; the 48 outputs of spec 04; the CA "today" row), and so does every row of the tables that the harness can express: 41 of 45 lines of the spec 01 table (the real A1 crops of group C included), 48 of 52 of spec 03, 45 of 45 of spec 04 and 10 of 10 settings (50 cells) of the CA table. The rows that differ, and why, are in the appendix: the four `K_50` rows (the audit moved the `50` of `K` alone; the field is the SNR, which moves the `0.04` too, so that experiment is not a `Tuning` override) and four cells of spec 03 that read `-0.000` in the audit and `0.000` here (the audit moved the pixel pitch by 1e-9 um to get past the geometry cache; the cells are zero). The audit's text outputs stay in the scratchpad of the audit; `tools/fit/audit_base.txt` holds the base numbers and `tools/fit/reproduce.sh check` compares them in about ten seconds. Tests: `crates/ym-gpu/tests/fit_harness.rs` (the overrides name every field of the three `Tuning` structs and no other, round-trip, and refuse an unknown name, a fraction for a whole-number field and a number that is not finite; `--rel` gives the decimal a person types, 1.6 x 0.9 = 1.44; the data note; the three examples on tiny scenes, deterministic, with a moved constant moving the output; the CA scenes; the missing and the present PixelShift200 folder; the driver's own tests). No picture changes: `Tuning::DEFAULT` and every source file of the package are untouched (no file under `src/` changes), `golden check` has the stored hashes.

**RC-06 · The baseline** (the floor of every fit). Run RC-05 with `Tuning::ORACLE` on every objective and gate of section 2.3 and write the numbers into this document (a table per package) and into `docs/STATUS.md` "Gate results" (so a later fit cannot quietly loosen them): the nine-cell PSNR, zipper and false-colour rates and the noisy table of `demosaic_gate`; the CA residual table; the §6.3/§6.4 tables of spec 03; the edge profile of spec 04 §9. *Done when:* the numbers are in the two places and the command that made them is in the appendix.

*Done (branch `rc-06-baseline`, commits "RC-06: the baseline command and its tables" and "RC-06: the numbers in docs/29 and STATUS").* The tables are section 2.2a, one per package (spec 01: the nine cells, zipper and false colour, the noisy table, the groups A and B with the fit and held-out split and the second noise seed, the edge through the demosaic, the CA residual table; spec 03: the section 6.4 table and the eight cases of the harness; spec 04: the section 9 edge profile and the harness edge), and the headline numbers are in `docs/STATUS.md` *Gate results*. The command is `tools/fit/baseline.sh OUTDIR` (Appendix B; `tools/fit/baseline.py` formats the tables, `tools/fit/test_baseline.py` tests the formatter); the only other change is that the edge-profile test prints more digits. Group P is not run: PixelShift200 absent; evaluation data, never committed. `Tuning::DEFAULT` and every file under `src/` are untouched and `golden check` gives the stored hashes. No dataset byte is in the repository (`git ls-files testdata` lists nothing, and the data of the run sat outside the checkout). Re-running the full command into a second folder gave byte-identical JSON lines, `gate.json`, `results.json`, the spec 04 output and tables.

**RC-07 · Fit spec 01 pass 1 and the chroma penalty** (A1, A6, A7; changes every picture if accepted). As section 2.3. *Done when:* the PR reports, for each constant, today's value, the fitted value, the objective on the fit sets and on the held-out set, and the full `demosaic_gate` tables; either the values are changed in `Tuning::DEFAULT` with the numbers showing the acceptance rule, or the PR keeps today's values and writes "fitted on <date>: within x of the optimum" into the registry.

**RC-08 · Fit spec 01 pass 3 targets and the shadow group** (A8 to A14; changes pictures where the shadow steps act). As section 2.3, with the sharpening objective for A8 to A10. *Done when:* as RC-07, plus the edge, noise-gain and overshoot numbers for A8 to A10, and the crossover table for A14.

**RC-09 · Output sharpening** (O1 to O3, O6, O7; changes every export if accepted). The tap rule from the tail-mass budget, the continuity repair of the σ map (no change at `R ≥ 0.5`, so none for the preset), the σ knee, and a check of O6 and O7. The presets, the halo map and the radius slope are design and are not touched (section 6, question 4). *Done when:* as RC-07, with the edge profile of spec 04 §9 at `s = 1` and `s = 0.4` as the gate, and the tap count at or above today's at every σ of 0.1 to 2.5.

**RC-10 · The CA path** (A2 to A5; no picture changes while R5b corrects CA on the mosaic). Coordinate descent on the synthetic CA scenes (ours). *Done when:* as RC-07 on the residual and PSNR tables of `ca_green_residual`, and the vectors `ca_weighting` and the CA property tests pass under `Tuning::ORACLE`.

**RC-11 · Spec 03** (D1 to D12; no picture changes while R8 is Richardson-Lucy). D1 from the Weta curves; D2 to D4 and D6 as error budgets; D5, D7 to D12 as fits. The σ_G factor is decided here (0.75 or 0.78). *Done when:* as RC-07 with the §6.3 bars; the plan vectors (`plan_*`) pass under `Tuning::ORACLE`; the gate table of §6.4 is regenerated and put in the spec.

**RC-12 · The log-guided exponent** (L3; optional, no picture). Only if the owner wants it (question 5). *Done when:* as RC-07 on the `log_guided_eval` PSNR table, or the PR is closed unmerged with "not shipped".

**RC-13 · Refresh the golden images and the version** (F-09; needs the accepted fits among RC-07 to RC-09). `golden update --note "…"` with the gate numbers of the accepted fits and the largest ΔE00 moved; `PIPELINE_VERSION` bumped (question 2); the tests and documents that pin the old string (`crates/ym-cli/tests/cli.rs`, `docs/01`, `docs/13`, `docs/15`, `docs/16`) updated; the self vectors regenerated at the new `Tuning::DEFAULT`; the GPU golden jobs (lavapipe subset and nightly full set) green; the DNG gates of the `dng` job (`dng_validate`, `dngfit`: the demosaic is part of a DNG picture, `docs/09`) green or their numbers reported. *Done when:* `golden check --require-all` passes on the new hashes, the note carries the numbers, the `golden-gpu` job passes, and the CHANGELOG has the entry (RC-14).

**RC-14 · The words** (F-08, G8). `docs/14` row: "constants taken over from research and re-derived by the project (`docs/29`)"; `docs/16` lines on the package; the "Constants" paragraphs and §7 of the four specs; `INTEGRATION.md` §8 ("a constant that is not in a spec is not added" stays); the prior-art record's §2.1 and §2.2 item 3 get the answer; `docs/STATUS.md` question 12 closed with the outcome; `CHANGELOG.md` for the release (below). *Done when:* the registry test reports no "pending" status and every doc says the same thing.

**[HUMAN] RC-H1 · A blind look at old and new** (the R6 rule of `docs/05`: "checked on artefacts and on a blind look"). Six photographs of the owner's choice rendered with the old and the new constants (`Tuning::ORACLE` against the shipped `DEFAULT` through the internal switch), the existing blind A/B kit (`tools/eval/ab`); the owner picks. Needed only if a fit that changes pictures is accepted. Listed under "Waiting on the owner" when RC-07 lands.

**The CHANGELOG entry** (for the release that carries RC-13; the file's own format, `### Changed`): "**Demosaic and output sharpening constants re-derived.** The constants of the demosaic's chroma step, its shadow steps and the output sharpening kernel were taken over from research. They are now fitted or derived by the project on public test images and our own synthetic scenes, and each has its derivation in `docs/29`. Pictures change slightly: the largest change on the test set is [ΔE00 from the golden note], the average [mean codes], and exports at a reduced size change the most. The pipeline version is [x.y.z] (was 0.11.1)." If no fit is accepted the entry is "**The constants of the raw algorithms have recorded derivations.** No picture changes."

## 6. Questions for the owner

Each has a recommended default, which the plan follows until you say otherwise.

**Answered 2026-11-13** (the owner: "1 - no, 2 - default, 3 - yes, 4 - keep as they are, 5 - default is fine, 6 - record nothing, it doesn't matter where this is coming from"):
1. A fit that is better on one number and worse on another is **not taken**: today's value stays, and the numbers are reported.
2. One release carries every accepted fit, `PIPELINE_VERSION` 0.11.2 (the answer named 0.11.1, the next patch step at the time; 0.11.1 went to the NEX lens tables the same day), one golden refresh.
3. A blind look of six photographs, old against new, before it ships (RC-H1).
4. The output sharpening's look is kept: the σ map, the presets and the halo map stay; only the tap rule is derived and the step below `R = 0.5` repaired.
5. RC-01 and RC-02 now; RC-10 and RC-11 after the render-moving fits; RC-12 skipped.
6. **No source is recorded**: each derivation stands on our own method and names no origin, and `docs/14` says nothing about one.

1. **A fit that is better on one number and worse on another.** For example a chroma-penalty value that raises the PSNR and the false-colour rate together. *Default: not taken.* Today's value stays with its derivation, the numbers are reported, and you decide case by case. This is the failed-gate rule applied to fits (`AGENTS.md`).
2. **The release and the version.** Several fits would move every picture a little. *Default:* one release that carries every accepted fit together; `PIPELINE_VERSION` 0.11.1 (a patch step: no new kind of node, pictures change slightly; the 0.8.1 rebaseline of all 61 pictures is the precedent) or 0.12.0 if you prefer a minor step; one golden refresh with one note.
3. **A blind look before it ships.** `docs/05` says a demosaic change is checked on artefacts and on a blind look. *Default:* yes, six photographs, old against new, the existing A/B kit; if you see no difference or prefer the new, it ships.
4. **The look of the output sharpening.** The radius-to-σ map (`0.7R + 0.3`), the presets (radius 1.0, amount 100, halo 60) and the halo map are a look, and a fit cannot choose a look. *Default:* keep them as they are; the plan derives only the tap rule (from a tail-mass budget) and repairs the step of the map below `R = 0.5`, which the screen preset never reads. At the preset's radius nothing moves, unless the tap budget gives a longer kernel than today's (it can only get longer, never shorter).
5. **Rows that move no picture today** (the CA path, spec 03, the log-guided exponent). *Default:* do the mathematical ones now (RC-01, RC-02), fit the rest after the render-moving fits as documented numbers (RC-10, RC-11), skip RC-12. They cost minutes to about an hour each of CPU and change nothing you can see; the alternative is to leave them as taken from research and say so in `docs/14`.
6. **Which research.** Your answer says the constants come from research. Can you name it (a paper, a report, the project it came from)? *Default:* if not, each derivation says "taken over from the research package of 2026-10, source not recorded" and stands on our own method, and `docs/14` says the same. A named source can be cited in each row and may answer the patent leads of M4-H4.

## Appendix · How to repeat the measurements

The audit's scratch code is not in this branch (it changes product code). To repeat it:

```bash
# 1. a scratch worktree (never committed), and the two evaluation sets
git worktree add --detach /tmp/rc-measure origin/main
YM_TESTDATA=/tmp/rc-data tools/fetch/demosaic-sets.sh        # Kodak 24 + McMaster 18, SHA-256 pinned

# 2. in the scratch worktree: a module `ym_gpu::cpu::rc_tune` with `rc(index, default) -> f64`
#    (an array of atomics read from the environment variable YM_RC="index:value,..."),
#    and each literal of section 2.1 replaced by rc(index, literal). Examples `rc_sens` (spec 01
#    and 04) and `rc_diff` (a copy of capture_sharpen_ab with a loop over constants).
export CARGO_TARGET_DIR=... CARGO_INCREMENTAL=0
cargo build -p ym-gpu --examples --config 'profile.dev.package.ym-gpu.opt-level=3'

# 3. run
YM_TESTDATA=/tmp/rc-data YM_PHOTO=<an ARW> rc_sens demosaic "A1 x1.1@0:1.76" ...
YM_TESTDATA=/tmp/rc-data rc_sens output "O1 x1.1@60:0.77" "radius x1.1@radius:1.1" ...
rc_diff "D5 x1.1@29:0.825" ...
YM_RC="1:3.75" ca_green_residual --size 900x600 --out /tmp/rc-ca
python3 polys.py        # the cubics: exact integer emulation, continuous minimax by Lawson, integer search
python3 exp2.py         # exp2: hill climb in integers on the relative-error objective
```

The scratch scripts of this audit were in the session scratchpad (`rc/polys.py`, `rc/log2b.py`, `rc/exp2.py`, `rc/demosaic_args.txt`, `rc/output_args.txt`, `rc/diff_args.txt` and the output files); the cubics are now `tools/fit/poly_fit.py` (RC-01) and the harness below replaces the scratch build: it is a committed, tested part of the repository (RC-05), takes the constants as arguments and needs no patched literal.

Index of the audit's scratch constants (`YM_RC` index, section 2.1 row, field of the `Tuning` that `fit_*` takes instead): 0 A1 `second_diff_weight`, 1 A2 `ca_relax_slope`, 2 and 3 A4 `centre_weight_linear` and `centre_weight_cubic`, 4 A6 `chroma_floor`, 5 A7 `sat_dominance`, 6 A8 `shadow_amount`, 7 A9 `shadow_laplacian_scale`, 8 A11 `match_excess_weight`, 9 A12 (`50`; see below) `offset_snr`, 10 A13 `snr_per_slider`; spec 03: 20 `weight_ramp_width_nm`, 21 `airy_scale`, 22 `r80_energy`, 23 `support_r80_factor`, 24 `support_pitch_factor`, 25 `kernel_energy`, 26 `kernel_tail`, 27 `x80_cutoff`, 28 `x80_level`, 29 `sigma_factor`, 30 `sigma_min`, 31 `sigma_max`, 32 `gauss_radius_per_sigma`, 33 `clip_weight_largest`, 34 `clip_weight_middle`, 35 `clip_fraction`, 36 `pull_gain`, 37 `pull_band`, 38 `ridge`, 39 `gate`, 40 `gate_fade_lo`, 43 `gate_fade_hi`, 41 `weight_min`, 42 `ratio_cap`; spec 04: 60 `sigma_slope`, 61 `sigma_offset`, 63 `taps_per_sigma`, 64 `taps_offset`, 65 `sigma_knee`, 67 `halo_pull_fade`, 68 `ratio_max`, and the preset fields `radius`, `amount`, `halo`, `thr` are `screen_radius_px`, `screen_amount_pct`, `screen_halo_pct`, `screen_threshold_levels`.

## Appendix B · The fit harness (RC-05)

**The documented command.** From the repository root, with Kodak and McMaster in `testdata/demosaic` (`tools/fetch/demosaic-sets.sh`; `YM_TESTDATA` moves the root):

```bash
tools/fit/reproduce.sh check                  # about 10 s: the base numbers against tools/fit/audit_base.txt
YM_PHOTO=testdata/readme-photos/src/A1M01749.ARW \
  tools/fit/reproduce.sh demosaic out/        # the spec 01 table of section 2.2, groups A, B and C (without YM_PHOTO: A and B)
tools/fit/reproduce.sh diffraction out/       # the spec 03 table
tools/fit/reproduce.sh sharpen out/           # the spec 04 table (Kodak at scales 1 and 0.33)
tools/fit/reproduce.sh ca out/                # the CA table (five scenes at 600 x 400)
tools/fit/reproduce.sh all out/               # the four, each as `out/<part>.jsonl` and `out/<part>.txt` in the audit's text format
```

The script runs `cargo run -p ym-gpu --example fit_<package> --config 'profile.dev.package.ym-gpu.opt-level=3' -- --rel ...` (the dev profile with `ym-gpu` at `opt-level=3`, as `demosaic_gate` does) and renders the lines with `tools/fit/search.py table`. A machine of the audit's size (4 shared cores) takes about 2.5 minutes for the spec 01 table, 3 minutes for spec 04, 10 seconds for spec 03 and 12 seconds for the CA table. The examples themselves (each line is a run; or run the built binary):

```bash
export CARGO_INCREMENTAL=0
fit="cargo run -q -p ym-gpu --example fit_demosaic --config profile.dev.package.ym-gpu.opt-level=3 --"
$fit --list                                        # the fields of adaptive_ratio::Tuning and their values
$fit                                               # only the base line
$fit --set sat_dominance=0.8 --held --artifacts --edge   # one evaluation, with the held-out scenes and the extra objectives
$fit --rel chroma_floor=0.9,1.1 --photo A1M01749.ARW     # two evaluations, at 0.9 x and 1.1 x, with the two real crops
$fit --ca-only --eval "A1 to 10@second_diff_weight=10"   # the CA scenes only
$fit --set no_such_field=1                         # refused, with the list of fields, before any scene loads
```

`fit_diffraction` and `fit_sharpen` take the same `--from`, `--set`, `--rel`, `--eval`, `--held` and `--list`; `fit_sharpen` adds `--edge` and `fit_demosaic` adds `--photo`, `--artifacts`, `--edge`, `--ca`, `--ca-only` and `--ca-size`. With no evaluation argument only the base line is printed. The starting `Tuning` is `--from oracle` (the default: `Tuning::ORACLE`) or `--from default` (what ships); the base of a run is the starting `Tuning`, and every other line carries its change against it.

**The lines.** Every run prints, in this order: `{"kind":"data_note", ...}` (the sets used with their image counts, the sets skipped and why, and the sentence "Evaluation and local fitting only ... never committed ... no real-sensor ground truth"), then `{"kind":"eval","label":"base", ...}` and one `{"kind":"eval", ...}` per evaluation, with `label`, `set` (the fields moved), `from` and:

| Example | Keys of an evaluation line |
|---|---|
| `fit_demosaic` | `psnr_db` (`A`, `A_fit`, `A_held`, `B`, `B_fit`, `B_held`, `B2` with `--held`, `P` when PixelShift200 is there; `null` for a group that has no scene), `d_psnr_db` (the same against the base), `change_codes` (per group `A`, `B`, `B2`, `C`, `P`: `mean_codes` and `max_codes`, the audit's measure), and with the flags `artifacts` (`zipper_pct`, `false_colour_pct`, `chroma_de00_mean`), `edge` (`mtf50_cpp`, `rise_10_90_px`, `overshoot_pct`, `undershoot_pct`, `flat_sd`), `ca` (per scene `strength`, `barrel`, `red_rms_px`, `blue_rms_px`, `psnr_rgb_db`) |
| `fit_diffraction` | `cases` (per case `name`, `held`, `text_psnr_db`, `natural_psnr_db`, `mtf50_cpp`, `rise_10_90_px`, `overshoot_pct`, `flat_noise_x`), `mean_delta` (the mean change against the base over the four cases, and `max_dout`), `max_dout`, `mean_delta_held` with `--held` |
| `fit_sharpen` | `outputs`, `change_codes` (per scale `1` and `0.33`: `mean_codes`, `max_codes`), `held_change_codes` with `--held`, `edge` with `--edge` (per scale `1` and `0.4`: `rise_10_90_px`, `overshoot_pct`, `undershoot_pct`, `noise_gain`) |

**The search driver.** `tools/fit/search.py` calls an example with `--eval` lines and reads the JSON back; it does no image processing. The searches of section 2.3 are one command each; every command below prints the data note, then one `{"kind":"search", ...}` line with the best point, its objective, the start's, the gain, whether the gain beats `--min-gain` (`taken`) and the whole evaluation line:

```bash
# A1, the grid of section 2.3 (0.4 to 3.2 in steps of 0.1), objective the PSNR of A, with the held-out scenes in the line
tools/fit/search.py grid fit_demosaic --objective psnr_db.A --axis second_diff_weight=0.4:3.2:0.1 -- --held
# A6 and A7 together, then a refinement by coordinate descent from today's values
tools/fit/search.py grid fit_demosaic --objective psnr_db.A --axis chroma_floor=0.01:0.12:0.01 --axis sat_dominance=0.5:1.2:0.05
tools/fit/search.py descent fit_demosaic --objective psnr_db.A --axis chroma_floor=0.03:0.005 --axis sat_dominance=0.75:0.05
# a sharpening constant: the edge, not the PSNR, with the noise gain and the overshoot held to today's
tools/fit/search.py descent fit_demosaic --objective edge.mtf50_cpp --axis shadow_amount=0.75:0.05 --constraint 'edge.flat_sd<=0.0098' -- --edge
```

`grid` takes `name=start:stop:step` (inclusive) or `name=v1,v2,...`; `descent` takes `name=start:step`, tries `-2, -1, +1, +2` steps of one field at a time, in the order of the `--axis` arguments, takes the best that beats the current point by `--min-gain` (ties go to the earlier candidate), and halves every step after a sweep that moved nothing, until all are under `--min-step`. A constraint on an evaluation line (`PATH<=VALUE`, `PATH>=VALUE`) rejects a candidate that breaks it (the acceptance rule 4 of section 2.3 on one number at a time; `d_psnr_db.*` and the `edge` and `artifacts` numbers are in the line). The driver passes everything after `--` to the example. A run is deterministic: the same command gives the same lines.

**What reproduces, measured 2026-11-18 at `Tuning::ORACLE` against the audit's output files.**

| Table of section 2.2 | Lines of the audit | Identical to the displayed digits | The rest |
|---|---|---|---|
| Spec 01, the base (`BASE psnr`) | 1 | 1: A 35.678, B 27.604 | |
| Spec 01, 11 constants x 4 factors (A, B, C: PSNR change, mean and largest change in codes) | 44 | 40 | 4: `K_50` (below) |
| Spec 03, the base (four cases) | 4 | 4: f/8 ISO 100 30.68, 35.80, 0.340, 1.7 %, 1.005; f/8 ISO 800 29.84, 34.14, 0.354, 2.4 %, 1.007; f/16 ISO 100 26.08, 31.89, 0.245, 3.0 %, 1.577; f/16 ISO 800 25.70, 30.75, 0.235, 4.2 %, 1.617 | |
| Spec 03, 24 constants x 2 factors | 48 | 44 | 4: signed zeros (below) |
| Spec 04, the base (48 outputs) | 1 | 1 | |
| Spec 04, 11 constants x 4 factors | 44 | 44 | 0 |
| CA table, 10 settings x 5 scenes (residual and PSNR of the CA path) | 10 | 10 | 0 |

**What does not reproduce, and why.** (1) *The four `K_50` rows of spec 01 (A12 at 0.5, 0.9, 1.1 and 1.5).* The audit replaced the literal `50` of `K = 50 (g + sqrt(g^2 + 0.04 r^2))` and left the `0.04`. Since RC-03 the field is `offset_snr` (`K = (snr^2 / 2)(g + sqrt(g^2 + 4 r^2 / snr^2))`, which is the `50` and the `0.04` for `snr = 10`), so moving it moves both numbers; the harness runs that experiment instead (for example `offset_snr` x 0.9: dPSNR A -0.0101, B -0.0045, mean change 0.0128 / 0.0639 / 0.0110 codes, where the audit's `50` x 0.9 gave -0.0060, -0.0032 and 0.0078 / 0.0473 / 0.0060). The audit's figures for A12 stay what they were, a measurement of the `50`; a fit of A12 moves the SNR, and `docs/29` §2.3's target of 10 is that SNR. The `K` of the audit is not an override the `Tuning` can express and none was added: a constant is added only with a spec (`INTEGRATION.md` §8). (2) *Four cells of spec 03* (`kernel_energy` x 0.9, `kernel_tail` x 0.9, `ratio_cap` x 0.9 and x 1.1): the audit printed `-0.000 dB` for a text PSNR change of about -1e-9 and the harness gives exactly 0. The audit planned each evaluation at a pixel pitch moved by `1e-9 x k` um to get past the geometry cache (which holds only the shipped constants); the harness plans a non-shipped `Tuning` afresh and needs no such change. The cells are zero to the displayed precision in both. The `max|dout|` of those rows is 0.00 in both. (3) *What is not in the audit and is now run:* the second noise seed and McMaster (`--held`), the zipper and false-colour rates and the edge numbers (`--artifacts`, `--edge`), the real-sensor group P when PixelShift200 is present. None of them has a base number in section 2.2 to reproduce; RC-06 records them in section 2.2a.

**The baseline command (RC-06).** From the repository root, with the sets of `tools/fetch/demosaic-sets.sh` in `testdata/demosaic` (`YM_TESTDATA` moves the root):

```bash
export CARGO_TARGET_DIR=... CARGO_INCREMENTAL=0
tools/fit/baseline.sh out/            # about 1 to 2.5 minutes in the release profile: runs everything, prints the tables of section 2.2a
tools/fit/baseline.sh out/ cheap      # skips demosaic_gate (about 30 seconds); needs demosaic_gate/gate.json from an earlier full run in out/
tools/fit/baseline.sh out/ tables     # formats the tables again from the raw output already in out/
```

It runs, in the release profile: `fit_demosaic --held --artifacts --edge --ca`, `fit_diffraction --held`, `fit_sharpen --held --edge`, `capture_sharpen_ab`, `demosaic_gate all` and the two edge-profile tests of `cpu_output_sharpen.rs` (`--nocapture`), keeps their raw output in `out/` and formats it with `python3 -I tools/fit/baseline.py out/`. It exits with an error before running anything if `Tuning::DEFAULT` differs from `Tuning::ORACLE`. Group P is read when `testdata/demosaic/pixelshift200` holds images and is printed as "not run" when it does not. Determinism, measured 2026-11-18: two full runs into two folders gave identical `fit_demosaic.jsonl`, `fit_diffraction.jsonl`, `fit_sharpen.jsonl`, `spec04_edge.txt`, `demosaic_gate/gate.json`, `capture_sharpen_ab/results.json` and `baseline.md` (SHA-256 of the tables de84ff2a...).

**The data.** Kodak and McMaster are read from `testdata/demosaic/{kodak,mcmaster}`, PixelShift200 from `testdata/demosaic/pixelshift200` when that folder holds images, and the real crops (group C) from the file given to `--photo`. They are for evaluation and local fitting (the owner's standing data rule) and are never committed: `testdata/` is git-ignored and `git ls-files testdata` lists no dataset file. The harness writes no picture and no file; it prints numbers. The noise models of the dark scenes and of the crops are stand-ins and the test images are re-mosaicked 8-bit sRGB pictures, so a fit on them is not a statement about a sensor (G5, F-07): every run says so in its data note.
