# Why do our reds read "bright and coral"? An audit of an outside analysis

For Mykyta, 2026-10-29. Research only. Written by a research agent; this file and one line of `research/README.md` are the only changes to the repository, and nothing was committed. Every number here was *measured* with the repository's own code at commit `ea6cc73` (the `ym` binary and a pinned copy of `crates/ym-color`; not the later work-in-progress on the `reds-density` branch, which is described in section 8), on the owner's A1 photos and on two Play Raw photos from `testdata/`, or computed from the repository's own spectral data. The scripts are in the session scratchpad, not in the repository.

**ARRI's own display rendering was used as a yardstick, locally.** On 2026-10-29 the owner allowed ARRI's display LUTs to be downloaded for verification, and then decided that tuning may target ARRI's rendering but that **no number derived from ARRI's LUTs belongs in the repository**. This file therefore contains no measurement made on those LUTs and no ratio, angle or lightness gap between our renders and ARRI's. Where the comparison matters it is stated in words ("lighter than ARRI's display rendering of the same scene colour", "turned toward orange relative to it"). The tables behind those words (CSV, at about 100 colours) are kept by the developers outside the repository; the LUTs are in `testdata/arri-luts/` (git-ignored), to be deleted when the tuning is done. A longer version of this report with those tables is outside the repository too.

All other numbers are against the **scene colour**: for a pixel with scene-linear AWG4 value `p`, luminance `Y` and display luminance `Yn` of the neutral of the same scene luminance through the same variant (its own tone scale), the reference colour is CIELAB (D65) of `XYZ(p) / Y * Yn`, the scene's chromaticity at the lightness its own tone scale gives a grey (the definition R1 uses, `docs/02`). From it: `rho = Y_out / Yn`, `dL*` (lightness against that reference), `C*_out / C*_ref` (chroma ratio) and `dh` (hue difference in degrees; positive is toward orange: CIELAB hue 0 is crimson-magenta, 25 to 35 red, 60 orange).

## 0. Short answer

The analysis is right about the *direction* and wrong about most of the *causes* it names.

| Claim | Verdict | Key numbers |
|---|---|---|
| **A.** ACES 2 renders saturated reds lighter than ARRI's display rendering | **True in direction, but only for very saturated reds; the owner's default look differs from ARRI mainly in other ways.** | Compared with ARRI's display rendering of the same scene colour, plain ACES 2 (with our +0.85 EV placement) renders the most saturated reds clearly lighter, and agrees closely on ordinary reds. On the owner's real reds (chroma 40 to 65) plain ACES 2 and Neutral stay close to the scene colour (rho 1.00 to 1.03, +0.0 to +0.4 L*, hue +0.5 to +1.2 degrees) and are somewhat lighter than ARRI's rendering, which renders such reds a little darker than its greys; the red Santa hat (chroma 90) is lighter by a larger margin. **Signature**, the default look, adds chroma x1.47 to 1.62 of the scene's (ARRI's rendering stays near the scene's), hue +3.2 to +5.4 degrees toward orange (ARRI's stays near the scene's) and +4.5 L* above its own greys (rho 1.43 to 1.46). |
| **A (2).** ARRI rotates saturated red about 5 degrees "clockwise" | **Not supported.** | Relative to the scene colour, ARRI's REVEAL rendering turns the most saturated reds *toward orange* and ordinary reds slightly toward crimson. Plain ACES 2 turns the chroma-119 red *toward crimson*: -1.5 degrees at grey, -6.5 degrees at +3 stops. The old K1S1 LUT turns saturated reds toward crimson only in the highlights. |
| **B.** The 17 ARRI CCT matrices were tuned to feed K1S1 and leave extra green cross-talk and red luminance | **Source: none. Effect on reds: real.** | The matrices are published as an ACES input transform to scene colorimetry; no ARRI or Academy text says K1S1. But the emulated ALEXA (the estimate through those matrices) renders the ColorChecker red (#15) 19 % lighter in luminance than the scene's own colorimetry (27 % with the owner's 4.5 degree push) and #9 10 % (14 %) (section 5). |
| **B (2).** REVEAL uses "a completely redesigned matrix set"; AWG3 to AWG4 leaves the pre-compensation uncancelled | **Half sourced.** | ARRI says REVEAL is "a completely new image processing chain" with a new colour engine (ACE4) and applies it to ARRIRAW from every other digital ARRI camera, ALEV III ones included. The matrices of ACE4 are unpublished. AWG3 to AWG4 *as encoded colorimetry* is a plain primaries change; REVEAL on ALEV III raw is not that. |
| **C.** The sensor map leaves M[G,R] too positive, so reds gain green (+0.04, ~20 % luminance) | **False for the sensor map.** | At D65 the folded Sony to AWG4 matrix has G from R = 0.078; ARRI's own matrix alone has 0.083, so the map *lowers* it. The stand-in A7R IV profile turns the ColorChecker red 9 % *darker* (Y x0.91) with G/R +1 %, and monochromatic 620 to 660 nm stimuli darker (x0.94 to x0.66) and more saturated, with *less* green (G/R 0.136 against 0.155 at 620 nm). The extra green and luminance are in the *target* (section 5). |
| **D.** The highlight range stretches saturated reds into the desaturation zone | **Does not apply to the analysis's example; applies mildly to near-clip reds; owner photos: 0 %.** | `[0.92, 0.12, 0.08]` is +0.67 stops over grey: gain exactly 1 (the knee is +2.58). A red whose raw red channel is at its clip is +2.8 to +3.0 stops: gain 1.19 to 1.56 (0.24 to 0.64 stop). In the owner's four street and flower photos 0 of 73,906 + 36,942 + 1,711 + 384 strongly saturated red pixels get any gain (their median is -1.4 to -1.9 stops, the 99th percentile at most +0.9). |

Where the "coral" comes from, stage by stage, for the ColorChecker red #15 at nominal exposure (D65), in our own measures:

| Stage | Measured against | #15 |
|---|---|---|
| Camera emulation (ALEXA estimate through ARRI's matrices, with the owner's 4.5 degree push), scene-linear | the scene's own colorimetry | +4.7 L*, +0.4 chroma, +4.1 degrees (luminance x1.27; without the push +3.4 L*, +0.2 degrees, x1.19) |
| Neutral (ACES 2 + placement + R1), display | scene colour at its own grey | rho 1.01, +0.3 L*, chroma x1.03, +0.2 degrees |
| Signature (the default), display | scene colour at its own grey | rho 1.25, +4.7 L*, chroma x1.41, +1.8 degrees |

Against ARRI's display rendering of the same scene colour, #15 is lighter in the camera emulation alone, lighter again through ACES 2, and lighter and far more saturated through Signature; for Neutral the camera emulation carries the larger share of the lightness gap and ACES 2 a smaller one. R1 changes none of this: it acts on 2 and 10 % of the strongly red pixels of the owner's two red photos (4 and 15 % in the two Play Raw ones) and by less than one ΔE00 on most of them.

Ranked fixes for the default path (section 7): (1) Signature's reds: a red density stage after the look, with a hue turn in Signature only (the owner's decisions of 2026-10-29, section 9); (2) the camera-side red luminance, which stays, because the owner keeps the 4.5 degree push; the rest is small.

## 1. What was measured, and how

**The chain.** The repository's own CPU reference, `ym_color` at `ea6cc73`, driven in two ways:

1. `ym render` (the shipped binary, `--cpu`, `--auto-provider yellowmoon`, a scratch data folder with extra built-in look files, so no repository file changed) writes the picture of the photo before the display transform as a DNG of the plain look, with the white lift off (`--white-offset -3`): the DNG's pixels are the C7 output, scene-linear AWG4. They are averaged to 1500 px (area filter) and kept as 32-bit floats.
2. A small Rust tool (outside the repository, built against a pinned copy of `crates/ym-color/src` at `ea6cc73`) runs C10 to C12 of the real chain on those pixels for a list of variants: Neutral, Signature, plain ACES 2 (R1 off), Signature with R1 on, and candidate fixes. It uses `ColorChain` for the shipped variants and the same pieces (`Aces2Transform`, `RedChromaStep`, `WhiteLift`) for the candidates.

**Validation.** The tool's pictures against `ym render` TIFFs (16 bit, 1500 px, sRGB, output sharpening 0) of the same photos: median difference 0.07 to 0.72 of an 8-bit code over 13 photo-look pairs (0.12 to 0.14 on the two red-flower photos, 0.17 and 0.21 on the red-pipe photo), mean 0.3 to 2.4 codes, 99th percentile 2.8 to 22 codes (edges from resampling; the larger values are on photos without the saturated reds measured here, two of them with a white lift whose anchor the tool estimates itself). The two Play Raw photos were not compared with a TIFF (same code path). The A1 photos go through the A7R IV profile (a stand-in).

**Strongly saturated red pixels** are those whose reference colour has CIELAB hue in [-10, 45] degrees and chroma at least 40 (or 55), between -3 and +3.5 stops over grey.

**The comparison with ARRI.** ARRI's display LUTs were evaluated locally with tetrahedral interpolation on the same inputs (AWG4 to LogC4; to LogC3 EI 800 for the old LUT, via the exact AWG4 to AWG3 matrix). ARRI's LUTs have no placement of +0.85 EV, so the comparisons were made with ARRI's input raised until the two neutral axes agree at mid grey. The LUTs used: `ARRI_LogC4-to-Gamma24_Rec709-D65_v1_65.cube` (ARRI's in-camera "REC 709 (SDR)" display rendering for the ALEXA 35, LogC4 LUT package v1.1, 10 July 2025), `ARRI_LogC3-to-Gamma24_Rec709_D65-Classic_33.cube` (the "Classic" LogC3 LUT, which ARRI's page names "ALEXA Classic 709" and calls "the well-known K1S1 LUT" [S4]; LogC3 LUT package v1.2.1, 10 July 2025) and, only to see it, `ARRI_LogC3-to-Gamma24_Rec709_D65-v1_33.cube` (not usable on AWG4 reds: it clips them).

## 2. Provenance of the downloads (for the owner)

- Page: `https://www.arri.com/en/learn-help/learn-help-camera-system/tools/lut-generator` (ARRI's LUT Generator page) lists "ARRI LogC4 LUT Package, the 3D LUT package in *.cube-format for color space conversion of LogC4/AWG4, Jul. 10, 2025" and the same for LogC3. Direct links go to `arri.canto.de` (names `ARRI - LogC4 LUT Package_data.zip`, 33,031,672 bytes, SHA-256 `1e2eda8a5273f205a76473f024de24b182cbd0c9fde877d27b4cfb2c0ec7547d`; `ARRI - LogC3 LUT Package_data.zip`, 47,733,638 bytes, `bf9a26d138378424964079fc9201474dd5b97ad586cc711457bfa2145ffe2bd3`). The folders inside are `ARRI_LogC4_v1-1_LUT_Package` and `ARRI_LogC3_v1-2-1_LUT_Package`. Fetched 2026-10-29 about 18:08 UTC with curl through the session's proxy: no login, no click-through.
- Licence terms found: **none**. The zips hold only the `.cube` files and a naming guide PDF; the `.cube` files carry no header comment; the page has no licence text; ARRI's terms page (`https://www.arri.com/en/terms-conditions`) lists sales and purchasing terms only. `docs/14` treats the LUTs as all rights reserved with no grant, and that is what applies. The use rests on the owner's permission of 2026-10-29, for local verification and tuning, with no ARRI-derived number in the repository.
- Stored: `testdata/arri-luts/` holds the three `.cube` files used and the naming guide (12 MB), kept for now and to be deleted when the tuning is done. The zips and the other 67 LUTs were deleted to free disk; the SHA-256 values are above.

## 3. (A) Our default path on a synthetic red, and on real reds

Inputs: AWG4 `[1, g, g]` with `g` = 0.05, 0.1, 0.2, 0.35, 0.5, scaled so that the *luminance* spans -2 to +3 stops over grey (0.18). The label is the CIELAB chroma of the reference colour at mid grey (119, 101, 76, 52, 34; ordinary photographic reds are 40 to 65; sRGB's own red primary has a chroma of 104.6 at its lightness of 53). White lift off. All Rec.709 sRGB output. "Plain" is ACES 2 with the placement and without R1; **Neutral equals plain wherever R1 does not act**, which here is every chroma-119 and chroma-101 point and everything below about -0.5 stops (R1's dark side is off by default).

**rho**, our display luminance relative to a grey of the same scene luminance through the same variant:

| C* | | -2 | -1 | 0 | +1 | +2 | +3 |
|---|---|---|---|---|---|---|---|
| 119 | plain | 1.21 | 1.11 | 0.98 | 0.75 | 0.61 | 0.56 |
| | Signature | 3.84 | 2.61 | 1.43 | 0.89 | 0.77 | 0.91 |
| 101 | plain | 1.16 | 1.07 | 0.98 | 0.80 | 0.68 | 0.66 |
| | Signature | 2.92 | 2.09 | 1.30 | 0.82 | 0.69 | 0.78 |
| 76 | plain | 1.09 | 1.04 | 1.01 | 0.92 | 0.84 | 0.86 |
| | Signature | 1.87 | 1.55 | 1.23 | 0.85 | 0.75 | 0.84 |
| 52 | plain | 1.02 | 1.01 | 1.01 | 1.01 | 0.98 | 0.97 |
| | Signature | 1.38 | 1.24 | 1.15 | 0.98 | 0.93 | 0.98 |
| 34 | plain | 1.01 | 1.01 | 1.01 | 1.01 | 1.01 | 1.00 |
| | Signature | 1.15 | 1.10 | 1.07 | 1.05 | 1.00 | 1.00 |

Reading: plain ACES 2 is *lighter than a grey* below grey for the most saturated reds (rho 1.21 at -2 stops for chroma 119) and darker above (0.56 at +3). Locally, ARRI's display rendering renders saturated reds darker than greys of the same scene luminance, progressively with chroma and exposure, so plain ACES 2 is too light against it for the most saturated reds, mostly below and around grey, and agrees closely on ordinary reds. **Signature** puts a saturated red far *above* its greys in shadows and mid-tones (rho 1.2 to 3.8) and below them in the highlights: the log-space CDL saturation (`docs/02` C10: weights on LogC4 *codes*) raises the linear luminance of a colour whose darkest channels sit in the log toe. The old K1S1 LUT behaves differently again: rho rises above 1 in shadows and falls steeply in highlights.

**Hue**, our own shift against the scene colour (degrees; positive is toward orange):

| C* | | -2 | -1 | 0 | +1 | +2 | +3 |
|---|---|---|---|---|---|---|---|
| 119 | plain | -1.7 | -2.0 | -1.5 | -4.3 | -5.7 | -6.5 |
| | Signature | +15.7 | +9.2 | +3.1 | +1.3 | +0.9 | +1.1 |
| 101 | plain | -1.2 | -1.2 | -1.1 | -3.4 | -5.1 | -6.0 |
| | Signature | +13.1 | +6.4 | +2.0 | -0.3 | -1.0 | -1.2 |
| 76 | plain | +0.3 | +0.2 | 0.0 | -1.8 | -3.8 | -4.7 |
| | Signature | +4.1 | +2.9 | +1.9 | -1.3 | -2.5 | -3.0 |
| 52 | plain | +1.0 | +0.5 | 0.0 | -0.6 | -1.7 | -2.9 |
| | Signature | +3.4 | +2.8 | +2.1 | -0.7 | -2.3 | -2.6 |
| 34 | plain | +0.8 | +0.4 | -0.1 | -0.5 | -0.9 | -1.1 |
| | Signature | +3.1 | +2.2 | +1.2 | +0.4 | -1.2 | -1.8 |

Plain ACES 2 moves colours toward crimson in the highlights (up to -6.5 degrees); Signature moves them toward orange in shadows and mid-tones (+3 to +16 degrees) and then back. Locally, ARRI's REVEAL rendering moves the reddest colours toward orange and the ordinary ones slightly toward crimson, and the old K1S1 LUT turns saturated reds toward crimson in the highlights.

**Chroma** (C* of the output over C* of the scene colour at the display lightness): plain ACES 2 keeps 0.71 (chroma 119), 0.82, 0.96, 0.93 and 0.88 (chroma 34) at grey and 0.38, 0.39, 0.41, 0.48 and 0.43 at +2 stops. In the highlights it keeps less chroma than ARRI's rendering does, which is the loss R1 was built to give back (R1 cannot above chroma 80: the cap). Signature keeps 0.65 (chroma 119) to 1.37 (chroma 52) at grey (it *adds* 9 to 37 % at chroma 34 to 76) and 0.21 to 0.44 at +2 stops, the pastel highlights.

**ColorChecker patches** (spectra: Ohta's measured ColorChecker; scene AWG4 from the ALEXA estimate through ARRI's matrix at 6504 K, so the ALEXA target of section 5; nominal exposure, display Y / L* / C* / h; Neutral equals plain except #9 C* 49.1 and #15 C* 61.8):

| Patch | plain | Signature | plain: rho, dL*, C ratio, dh | Signature: rho, dL*, C ratio, dh |
|---|---|---|---|---|
| #2 light skin | 0.358 / 66.4 / 14.6 / 51.6 | 0.438 / 72.1 / 20.2 / 49.8 | 1.00, +0.1, 0.61, +0.2 | 1.02, +0.7, 0.79, -1.7 |
| #7 orange | 0.307 / 62.3 / 60.2 / 64.9 | 0.377 / 67.8 / 78.1 / 60.7 | 1.02, +0.5, 0.89, +0.3 | 1.08, +2.1, 1.10, -3.9 |
| #9 moderate red | 0.231 / 55.1 / 42.0 / 22.0 | 0.272 / 59.2 / 63.5 / 23.1 | 1.01, +0.2, 0.85, -0.3 | 1.12, +2.7, 1.26, +0.9 |
| #15 red | 0.159 / 46.9 / 60.0 / 30.1 | 0.183 / 49.8 / 83.0 / 31.7 | 1.01, +0.3, 1.00, +0.2 | 1.25, +4.7, 1.41, +1.8 |

Locally, ARRI's display rendering of the same four patches is darker than plain ACES 2 on the reds and orange (a little on #9, more on #15 and #7) and very close to it on skin; Signature is lighter than ARRI's on all four, and much more saturated on the reds.

### 3.1 Real red content

Photos: the owner's A1M03165 (a flower stall, red roses and gerbera: 75,000 pixels or 5.0 % of the picture have chroma at least 40), A1M03175 (dahlias, 37,000 pixels), A1M03185 (a street, a red pipe, 1,700 pixels); two Play Raw files from `testdata/` rendered the same way, `28322_DSC09944` (A7 IV; a dog in a red Santa hat, 99,000 pixels, hat chroma about 90) and `33920_DSC08176` (ILCE-6000 through the A7R IV stand-in; a neon sign, 222,000 pixels). The other owner photos have almost no saturated reds (A1M03159 384 pixels, A1M01749 4, the two others none). Median over the red pixels (chroma at least 40, hue -10 to 45 degrees) of each photo, at the median exposure named in the first column; the white lift is on where the photo gets one (A1M03175 0.58 EV).

| Photo (red pixels) | | rho | dL* | C*/C*scene | dh (deg) |
|---|---|---|---|---|---|
| **A1M03165** flowers, median at -1.5 stops | plain / Neutral | 1.03 | +0.4 | 1.09 | +0.5 |
| | **Signature** | **1.43** | **+4.6** | **1.47** | **+3.2** |
| **A1M03175** dahlias, -1.7 | plain / Neutral | 1.03 | +0.4 | 1.14 | +0.6 |
| | **Signature** | **1.46** | **+4.5** | **1.54** | **+3.9** |
| **A1M03185** red pipe, -1.9 | plain / Neutral | 1.03 | +0.4 | 1.19 | +1.2 |
| | **Signature** | **1.46** | **+4.5** | **1.62** | **+5.4** |
| **28322** Santa hat, +0.3 stops (chroma 90) | plain | 1.00 | +0.1 | 0.85 | -0.7 |
| | Neutral (R1) | 1.00 | 0.0 | 0.88 | -0.7 |
| | Signature | 1.02 | +0.5 | 0.80 | -2.4 |
| **33920** neon, -1.9 | plain / Neutral | 1.02 | +0.3 | 1.29 | +0.4 |
| | Signature | 1.32 | +2.7 | 1.77 | +1.9 |

Locally, ARRI's rendering of the same red pixels stays near the scene's chroma and hue on the flowers, dahlias and pipe, and renders them a little darker than its greys; on the hat it renders them clearly darker than its greys.

Named patches of the flower photo (same measures, medians): red roses (stops -2.1): Signature rho 1.59, +4.2 L*, chroma x1.52, +3.2 degrees. Gerbera centre (chroma 60, -0.75 stops): Signature 1.43, +6.3, x1.36, +2.8; plain 1.02, +0.4, x1.05, +0.5.

**How much coral remains after R1.** In Neutral, the owner's reds are almost exactly the scene's colour: rho 1.00 to 1.03, +0.0 to +0.4 L*, hue +0.5 to +1.2 degrees, chroma x0.88 to 1.29. Against ARRI's rendering the remaining gap is in lightness only: small on moderately saturated reds, larger on the very saturated hat. R1 itself acts on 2.0 % (A1M03175), 3.7 % (neon), 9.6 % (A1M03165) and 15 % (hat) of the red pixels by more than 0.5 ΔE00 (1.0 to 7.1 % by more than 1; 0.2 to 4.6 % by more than 2; 99th percentile 1.0 to 3.9), only where the red is at least about -0.2 stops over grey (median of those pixels -0.04 to +0.28 stops), and not at all on dark reds (its dark side is off). **In Signature** (the default) R1 is switched off and, switched on, changes nothing that matters: with R1 (1, 0) the medians of the flower photo's red pixels do not move (chroma ratio 1.43 against 1.41, nothing in L* or hue; the cap acts at Signature's chroma). What remains in Signature is a lightness of +4.5 L* above its own grey, chroma x1.5 to 1.6, and hue +3 to +5 degrees.

### 3.2 What public data says

ARRI states that REVEAL's colour engine ACE4 brings "pleasing and naturalistic rendition of all skin tones" and "greatly improved bright saturated colors", and shows an ALEXA Mini LF ARRIRAW comparison in which "reds are rendered more naturally in the OPEN sign, the scooter brake lights, and the Chinese lanterns" [S2]. No ARRI document states a number for the luminance or hue of saturated reds; the only published description of the display rendering of REVEAL is its name (`ARRI_LogC4-to-Gamma24_Rec709-D65_v1`, in-camera "REC 709 (SDR)") [S3, S9]. The quantitative comparison was made locally on the LUT and is not recorded here.

## 4. (B) The ARRI matrices, K1S1 and REVEAL

**What the matrices are (sourced).** `Alexa_aces_matrix.txt` (13 November 2013) is read by `v3_IDT_maker.py`, whose output is "ARRI ALEXA IDT for ALEXA linear files": the generated transform reads "white-balanced, black-subtracted linear values" and applies "a CCT-dependent matrix" to ACES primaries (`IDT.ARRI.Alexa-v3-raw-EI<EI>-CCT<CCT>`, "ACES 1.0 Input - ARRIRAW") [S6, S7]. It is an *input* transform. The same script's LogC branch converts LogC3 to ACES with only a CAT02 primaries matrix (AWG3 to AP0: the matrix of `docs/02`) [S7]. ARRI's own LogC4 specification does the same for AWG4 [S1, section 4.3.2]. An ACES input transform is by definition a mapping to scene colorimetry (the Academy's procedure P-2013-001 fits it to scene colour; I could read it only through a summary on ACES Central [S10], which does not resolve in this session). The repository's own derivation (`docs/02`) is consistent: native to AWG3 is `inv(AWG3 to AP0) x M`, every row sums to one.

**Is there a source for "pre-compensation for K1S1"?** None found. ARRI's page says "ALEXA Classic 709" is "the well-known K1S1 LUT" [S4]; AbelCine describes the in-camera Rec.709 of every ALEXA before the ALEXA 35 as a LUT-based output transform [S11]. The ACES Central discussion that I could only see in a search summary frames the K1S1 match as an LMT built so that the ARRI IDT, LMT, RRT and Rec.709 ODT reproduce K1S1 [S10], which would be unnecessary if the IDT already carried it. That is a forum report, not ARRI. **Verdict: unsourced; the claim as worded cannot be checked from public documents.**

**What our own data say about the matrices (measured, section 5).** Whatever their intent, the matrices applied to a plausible sensor are *not* scene-colorimetric for reds: the ALEXA estimate through them renders ColorChecker #15 with luminance x1.19 (24 un-pushed variants, 1.18 to 1.21), #9 x1.10, orange #7 x1.01, skin x1.05 of the scene's, and the shipped estimate (the owner's 4.5 degree push) x1.27, x1.14, x1.04, x1.07. The G-from-R term of ARRI's matrix at 6504 K (native to AWG4) is 0.083; a least-squares colorimetric matrix on the same ColorChecker spectra gives -0.020 to 0.053 for the 11 Sony bodies (median 0.046) and 0.029 for the estimated ALEXA. So the matrices do carry more red-to-green leakage than a fitted colorimetric one. That is the analysis's "extra green cross-talk" as a measurement, with no statement about why.

**REVEAL (sourced).** ARRI: "REVEAL Color Science is a completely new image processing chain used by ALEXA 35/35 Xtreme and ALEXA 265 internally. Our latest ARRI Image Software Developer Kit has made REVEAL Color Science available for ARRIRAW from all other digital ARRI cameras (except ARRIFLEX D-20/21)", with ADA-7 (debayer), ACE4 ("transforms camera-native RGB image data into the AWG4 color space"), AWG4 and LogC4; "a perfect color match with ALEXA 35 images can be achieved by processing ARRIRAW from any other digital ARRI camera through REVEAL"; ART is the tool [S2]. ARRI's mixing guide: "legacy camera models can not output LogC4 over SDI, but can capture ARRIRAW, which then can be debayered into REVEAL color science, which means LogC4/AWG4"; for mixed shows ARRI recommends ACES with the official IDTs rather than a LogC3 to LogC4 matrix [S5]. ARRI's FAQ: "if you have a LogC3-based show LUT, we suggest completely redesigning/visually matching these for cameras using REVEAL color science" [S3]. AbelCine (2022): ART and the SDK let LF and Mini LF ARRIRAW "be processed through the new REVEAL pipeline"; the result should bring them "closer to matching" [S11]. **So ARRI does apply REVEAL to ALEV III files, with ACE4. Its matrices are not published** (the matrices we use are the 2013 IDT's).

**Is AWG3 to AWG4 a pure primaries change?** As colorimetry of an already encoded picture, yes: both are defined RGB primaries with a D65 white, and ARRI's own documents convert each to AP0 with CAT02 [S1, S7]. As a statement about what REVEAL does to ALEV III raw, no: REVEAL re-derives the colour from the raw data with ACE4. Yellowmoon's chain is "ARRI 2013 IDT colorimetry, put in an AWG4 container, then a display rendering". **Verdict:** the first half of claim B (K1S1 pre-compensation) is unsourced; the second (REVEAL differs) is sourced, and the consequence the analysis draws (an uncancelled pre-compensation) is inference. The measured part is that the ALEXA target is lighter in reds than the scene, section 5.

**Display LUT names.** LogC4: `ARRI_LogC4-to-Gamma24_Rec709-D65_v1` (33 and 65 point), the in-camera "REC 709 (SDR)"; ARRI's FAQ calls them the "official ARRI Display Render Transform LUT" and keeps the look in a separate log-to-log file [S3, S9]. LogC3: `ARRI_LogC3-to-Gamma24_Rec709_D65-Classic` (K1S1 [S4]) and `-v1`.

## 5. (C) The sensor map on reds, and where the extra red luminance comes from

Method: spectral data of the repository (`data/alexa/` estimate, `data/cameras/sony/` curves, `data/cameras/profiles/sony-ilce-7rm4/profile.json`, `data/arri/Alexa_aces_matrix.txt`), CIE D65 and a 3200 K blackbody, Ohta's measured ColorChecker. (i) target: estimate response / response to the white, through `M_AWG3 to AWG4 x M_ARRI(cct)`. (ii) pipeline: Sony response / response to the white, through the shipped sensor map (daylight class at 6504 K, tungsten at 3200 K, linear in mired) and the same matrix. The A1 has no curves of its own and renders through this profile, so for the A1 the spectral rows are the A7R IV's. As an A1-specific check, the A1's published D65 colour matrix (`data/cameras/color-matrices.json`) was used as the camera instead.

| D65 | target Y | pipeline Y (x) | G/R target | G/R pipeline | xy target | xy pipeline | ΔE00 | A1 colour matrix: Y (x), G/R, ΔE00 |
|---|---|---|---|---|---|---|---|---|
| #2 skin | 0.380 | 0.373 (0.98) | 0.7185 | 0.7214 | .3864, .3589 | .3846, .3576 | 0.62 | x1.00, 0.695, 1.21 |
| #7 orange | 0.304 | 0.303 (0.99) | 0.4990 | 0.4937 | .5093, .4176 | .5115, .4173 | 0.27 | x1.13, 0.466, 3.88 |
| #9 moderate red | 0.220 | 0.209 (0.95) | 0.3931 | 0.3890 | .4653, .3220 | .4653, .3201 | 1.16 | x1.04, 0.348, 2.28 |
| #15 red | 0.150 | 0.136 (0.91) | 0.2874 | 0.2904 | .5336, .3306 | .5319, .3309 | 1.90 | x1.01, 0.248, 1.96 |

At 3200 K the same patches give Y x0.98 / 0.99 / 0.95 / 0.91, G/R +0.4 / -1.4 / -1.4 / +1.0 %, ΔE00 0.57 / 0.47 / 1.15 / 2.11. The folded Sony to AWG4 matrix has G from R = 0.0782 (D65) and 0.0447 (3200 K); ARRI's matrix alone has 0.0826 and 0.0426.

Monochromatic stimuli (unit power in one 5 nm bin; chromaticity and ratios as they are, ΔE00 after setting both to Y = 0.2; D65):

| nm | G/R target | G/R pipeline | xy target | xy pipeline | Y pipeline / target | ΔE00 at equal Y |
|---|---|---|---|---|---|---|
| 600 | 0.260 | 0.212 | .6143, .3798 | .6333, .3631 | 1.01 | 3.3 |
| 620 | 0.155 | 0.136 | .6411, .3295 | .6560, .3263 | 0.94 | 2.1 |
| 640 | 0.130 | 0.120 | .6450, .3148 | .6586, .3162 | 0.85 | 2.8 |
| 660 | 0.123 | 0.121 | .6407, .3073 | .6510, .3124 | 0.66 | 2.9 |

(3200 K: 0.318 / 0.249, 0.157 / 0.137, 0.120 / 0.113, 0.110 / 0.116; Y ratio 1.01 / 0.96 / 0.88 / 0.69; ΔE00 5.7 / 2.5 / 3.7 / 3.3.)

**Verdict C.** There is **no systematic green leak into reds from the sensor map**. The stand-in's curves have more green in the deep red than the estimate (G/R 0.135 to 0.375 against 0.076 to 0.291 at 600 to 660 nm), and the Lab-optimal 3x3 *removes* part of it: the pipeline's reds are slightly *more saturated* (x 0.633 against 0.614 at 600 nm), with *less* green and *less* luminance. What the map does to reds is darken them (#15 by 9 %, 640 to 660 nm by 15 to 34 %) and over-saturate the spectral red end by about 2.1 to 5.7 ΔE00. A weighted fit would address that (it is the fit's reach: `docs/02` C2); it would not remove any green. The A1's own colour matrix, used as the camera, lands 2.0 to 3.9 ΔE00 from the target on #15, #9 and #7 (orange 13 % lighter), where the stand-in curves land 1.9, 1.2 and 0.3 away: an A1-specific difference that the stand-in cannot see.

**Where the extra red luminance and green are** (measured; D65; target = estimate through ARRI's matrix; truth = CIE 1931 colorimetry in AWG4):

| Patch | Y target / truth, un-pushed estimates (24; min to max) | Y target / truth, shipped (pushed) | G/R truth | G/R un-pushed (median) | G/R shipped | L*, C*, h minus truth, shipped |
|---|---|---|---|---|---|---|
| #2 skin | 1.05 (1.04 to 1.05) | 1.07 | 0.728 | 0.708 | 0.719 | +1.8, +1.7, +0.7 deg |
| #7 orange | 1.01 (1.00 to 1.02) | 1.04 | 0.484 | 0.475 | 0.499 | +0.9, +2.9, +3.2 |
| #9 moderate red | 1.10 (1.09 to 1.10) | 1.14 | 0.378 | 0.371 | 0.393 | +3.0, +0.6, +4.1 |
| #15 red | 1.19 (1.18 to 1.21) | 1.27 | 0.251 | 0.266 | 0.287 | +4.7, +0.4, +4.1 |

The un-pushed estimate with ARRI's matrices already renders the red #15 about 3.4 L* lighter than the scene (hue +0.2 degrees; skin -4.6 degrees, the crimson skin of the owner's earlier complaint). The owner's 4.5 degree push (2026-10-26) adds about +1.3 L* and +4 degrees on the reds (Y +6.5 %, G/R +8 % on #15), by design.

## 6. (D) The highlight range on saturated reds

C7 (`docs/02`, `ym_color::tone`, at `ea6cc73`): the gain is computed on the Rec.709-weighted luma of linear AWG4 and multiplies all three channels; it is exactly 1 up to the knee `s_k = max(1, h_S - 1)`. For the A1 at Auto the clip luma is 2.158 (2^1.11 x green's clip), `h_S` = 3.584, knee 2.584, expansion E = 4.055 stops, cap 7.638 (the repository's value, from `ym_color::tone`). Fused matrix of A1M03165 approximated by the sensor map and ARRI matrix at 4850 K (its Auto Kelvin) times 2^1.11, and the as-shot gains of that file (`ym info`: 2193, 1024, 1024, 1784):

| Red | AWG4 | Y | stops | gain |
|---|---|---|---|---|
| the analysis's `[0.92, 0.12, 0.08]` as AWG4 | 0.92, 0.12, 0.08 | 0.287 | +0.67 | **1.000** |
| the same triple as balanced camera RGB | 2.12, 0.39, 0.19 | 0.745 | +2.05 | **1.000** |
| raw red at its clip, raw G and B at 0 | 4.96, 0.31, 0.10 | 1.279 | +2.83 | 1.185 (0.24 stop) |
| raw red at its clip, G and B at 2 % | 4.95, 0.34, 0.19 | 1.308 | +2.86 | 1.241 (0.31 stop) |
| raw red at its clip, G and B at 10 % | 4.93, 0.47, 0.51 | 1.421 | +2.98 | 1.558 (0.64 stop) |
| raw red at 50 % / 70 % of its clip, G and B 5 / 7 % | | 0.71 / 0.995 | +1.98 / +2.47 | 1.000 |
| a neutral at the Sony clip | 2.16 x 3 | 2.158 | +3.58 | 16.6 (4.05 stops) |

The gain multiplies all three channels, so the colour ratios are kept. **Strongly saturated red pixels of the photos with a gain above 1** (inverting the range's stops map on the C7 output; reds = chroma at least 40/55/70, hue -10 to 45 degrees; gains from a fine table of the shipped C7): A1M03165 0 of 73,906 / 48,891 / 22,339 pixels (median -1.5 stops, 99th percentile +0.9); A1M03175 0 of 36,942 / 22,395 / 13,733; A1M03185 0 of 1,711 / 1,184 / 489; A1M03159 0 of 384 / 23; the hat photo (28322) 0.35 % of the chroma-40 pixels at the ALEXA cap (clipped), 0 of the chroma-55 and chroma-70 ones; the neon photo (33920) 0.9 % of chroma 40 and **4.6 % of chroma 55** (99th percentile gain 2.95 = 1.6 stops; 0.25 % at the cap). Pixels of the whole pictures above the knee: A1M03165 0.73 %, A1M03185 7.5 %, A1M01739 11 %.

**Verdict D.** The analysis's example is below the knee, and the owner's saturated reds are 3 to 4 stops below it. The effect exists for near-clip reds (a raw red channel at its clip lifts the pixel 0.24 to 0.64 stop, in a place where ACES 2 is already desaturating it), so a gate by purity is defensible, but the evidence that it matters on photographs is limited to bright neon or clipped reds. (Section 8: a gate is already on the `reds-density` branch.)

## 7. What this means: ranked fixes for the default path

Everything below changes only the default C10/C12 path. Candidates were prototyped in the scratchpad tool (not in the repository) and measured on the same inputs as sections 3 and 5, against the scene colour and our own greys (the numbers below); the comparison with ARRI's rendering was made locally and is given in words. The owner's decisions of 2026-10-29 (section 9) close two of the options: **Signature is fixed by a red density stage only** (no change of its saturation), with a **hue turn of -3 degrees in Signature only**.

**1. A red density stage ("R2") after the look and in front of ACES 2. The chosen fix; large effect on Signature, small cost.** A hue-preserving scale `1 - D k` on the scene colour, `k` = hue window (CIELAB hue -15 to 50 degrees, full from 0 to 38) x smoothstep of the chroma (30 to 90) x a fade between -4 and -2 stops; skin (chroma under 30, hue over 38) and oranges are untouched. Prototype results, own measures (against the scene colour at the variant's own grey):

| | #15 rho, dL*, C ratio, dh | #9 rho, dL*, C ratio, dh | owner flowers (A1M03165) | owner dahlias (A1M03175) | hat (28322) |
|---|---|---|---|---|---|
| Neutral | 1.01, +0.3, 1.03, +0.2 | 1.01, +0.2, 0.99, -0.3 | 1.03, +0.4, 1.09, +0.5 | 1.03, +0.4, 1.14, +0.6 | 1.00, 0.0, 0.88, -0.7 |
| Neutral + density 0.2 | 0.91, -2.0, 0.99, +0.4 | 0.96, -0.9, 0.85, -0.2 | 1.00, 0.0, 1.08, +0.6 | 1.01, +0.2, 1.13, +0.6 | 0.83, -4.5, 0.88, -0.2 |
| Signature (now) | 1.25, +4.7, 1.41, +1.8 | 1.12, +2.7, 1.26, +0.9 | 1.43, +4.6, 1.47, +3.2 | 1.46, +4.5, 1.54, +3.9 | 1.02, +0.5, 0.80, -2.4 |
| Signature + density 0.25 | 0.93, -1.4, 1.30, +1.9 | 0.91, -2.2, 1.26, +1.2 | 1.18, +2.2, 1.40, +3.2 | 1.26, +2.8, 1.49, +3.9 | 0.89, -2.8, 0.90, -1.5 |
| Signature + density 0.3 + hue turn -3 | 0.87, -2.8, 1.25, -1.2 | 0.87, -3.3, 1.26, -1.2 | 1.14, +1.7, 1.38, +1.8 | 1.22, +2.4, 1.47, +2.9 | 0.85, -4.0, 0.92, -4.3 |

Against ARRI's rendering of the same scene colour (local, words only): density 0.2 on Neutral takes the saturated reds from lighter than ARRI's to about level with it, and the hat from clearly lighter to about level; density 0.25 to 0.3 on Signature removes most of Signature's lightness excess on #15 and #9, and overshoots a little for the dark flower reds (Signature's curve already darkens shadows), but leaves its chroma excess and its orange hue shift, which the hue turn halves.

Costs. *Neutral, density 0.2:* of the 24 ColorChecker patches 3 change by more than 1 ΔE00 (#9 2.5, #15 2.3, magenta #17 1.0, the others by less than 0.5; both skin patches and orange exactly 0.0); on the photos skin-like pixels have a 95th percentile of 0.0 to 1.2 ΔE00, orange 0.0, the whole flower photo 11.7 % of pixels above 1 ΔE00 (the flowers), the neon photo 1.8 %. *Signature, density 0.25:* ColorChecker #15 6.1 ΔE00, #9 4.5, #17 0.6, every other patch 0.0 (skin and orange exactly unchanged); on the photos the red pixels change by 3.0 to 5.8 ΔE00 (95th percentile), skin-like pixels 0.0 to 0.5, orange 0.0; all 61 goldens are re-baselined, but only on their red patches. *Signature, density 0.3 with the hue turn:* red pixels 3.7 to 7.6 ΔE00 (95th percentile); #15 7.8 ΔE00. **A caution on the hue turn:** the very saturated hat red is already toward crimson in Signature (-2.4 degrees against the scene) and the turn takes it to -4.3, which in the local comparison is past ARRI's rendering of it toward crimson; the flowers, dahlias and pipe (Signature +3.2 to +5.4 degrees) are brought to +1.8 to +4.0, still toward orange.

**2. The camera-side red luminance (the target itself).** The ALEXA target lifts #15 by +4.7 L* and +4.1 degrees against the scene's own colorimetry (section 5), and the owner's 4.5 degree push (kept by decision 2026-10-29) is +1.3 L* and +4 degrees of that. For Neutral this is the larger part of its remaining lightness gap to ARRI's rendering of the same scene colour, which REVEAL, a colour engine that ARRI says is more accurate, would not have. A weighted fit of the sensor map cannot touch this (the map is not its source, section 5). Removing the push would bring back crimson skin (-4.6 degrees). Density, tuned on the display side, hides part of it.

**Measured and set aside (recorded for the owner's decisions).** *Lowering Signature's CDL saturation, 1.45 to 1.25 / 1.15 (a JSON edit):* owner reds rho 1.43 to 1.24 / 1.16, chroma x1.47 to 1.34 / 1.28, hue +3.2 to +2.4 / +2.0 degrees (A1M03165); #15 rho 1.25 to 1.14 / 1.09, dL* +4.7 to +2.7 / +1.7; skin chroma ratio against the scene 0.79 to 0.66 / 0.60 (the same as Neutral's 0.61 at 1.15). Cost: every colour; 76 % / 87 % of the flower photo's pixels change by more than 1 ΔE00 (95th percentile 2.6 / 4.0; skin-like pixels 2.7 / 4.1); all 61 goldens are re-baselined. The owner chose 1.45 by eye (2026-10-14 and 10-19) and now keeps it. *Replacing the log-space CDL saturation by a luminance-preserving one in linear AWG4* (C9's Saturation maths, F = 1.45): owner reds rho 1.08 to 1.10, hue +3.4 to +3.8 (unchanged: the orange shift follows the chroma into ACES 2's gamut mapping, not the log-space operation), chroma x1.30 to 1.38; 17 of 24 ColorChecker patches change by more than 1 ΔE00 (mean 1.8, largest 5.9); 31 to 48 % of all photo pixels by more than 1.

**What does not help.** *A stronger or reshaped R1:* R1 only adds chroma and never touches lightness; it acts on the up side (at least about -0.5 stop) and under the gamut cap, so it does nothing for Signature (chroma already above the cap) and little for Neutral's dark reds (it is off there by default); more chroma would move Neutral away from ARRI's rendering at grey. Its direction is right only in the highlights, where ARRI's rendering keeps more chroma than plain ACES 2. *R1 on in Signature* (`red_chroma.up` = 1): the medians do not move. *The sensor map weighted for reds:* no green to remove (section 5); it could take back the 9 % luminance the map takes from #15 (a small step toward lighter). *The range gate:* section 6. *A new display rendering:* plain ACES 2 with the +0.85 EV placement is close to ARRI's rendering on greys over most of the range and on ordinary reds; the large gaps are on the most saturated reds, which are rare in photographs (the owner's reds are 40 to 65 in chroma); the density stage reproduces most of the red gap at a much lower cost.

**Order of effect on the owner's photos** (reds of chroma 40 to 65 at -2 to +1 stops): 1. Signature's chroma, hue and lightness of reds (+4.5 L*, chroma x1.4 to 1.6 of the scene's, +3 to +5 degrees toward orange); 2. the camera-side luminance and the push (about +5 L* and +4 degrees on #15 against the scene's colorimetry); 3. ACES 2 against ARRI's rendering (lighter on moderately saturated reds, more so on the very saturated hat); 4. everything else (R1, the range, the sensor map) under 1 L*.

## 8. Tuning against ARRI's rendering, and what else is in the repository

**Tuning to ARRI.** The owner decided on 2026-10-29 that tuning may target ARRI's rendering and that no ARRI-derived number belongs in the repository. The developers tune against the local LUTs and the CSV tables kept outside the repository; what enters the repository are the constants of R2 and the hue turn themselves (strengths, windows), with `docs/14` updated to say how they were obtained (tuned locally against ARRI's display LUT, which is not in the repository, by measurement only; no LUT data, no fitted table and no number measured from it is committed). `docs/14` now says so (rule 9, owner decision of 2026-10-29); `AGENTS.md` hard rule 2 (no ARRI files in the repository) is unchanged. Alternatives that need no ARRI file at all: tune R2 against the scene (rho 1, chroma ratio 1: "faithful") or against the few public statements (REVEAL's colour engine is "more accurate", with "greatly improved bright saturated colors" [S2]; there is no public number).

**Concurrent work.** While this audit ran, `HEAD` of the shared checkout moved from `default-auto-copy` (`ea6cc73`) to a branch `reds-density` (checked out 18:03, a snapshot commit `7275c23` at 18:17 and uncommitted edits since). It adds, per its own file headers, an achromatic gate on the highlight range and on the white lift (purity 0.35 to 0.75, "owner decision 2026-10-29": the analysis's proposal for D) and a "red density and hue alignment" step (`ym_color::display::red_density`, C12 step 1d), with the aim of "a saturated red darker and a little further towards crimson". This audit measured `ea6cc73` (a hard link of the binary and a pinned copy of the crate kept the numbers independent of the moving tree), so nothing here was measured on that branch. Two facts from this audit bear on it: ARRI's REVEAL rendering turns reds toward crimson only slightly and toward *orange* for the most saturated ones, and the old K1S1 LUT only in the highlights (section 3), so a crimson alignment helps Signature (+3 to +5 degrees toward orange of the scene's colour) but would push Neutral (within about one degree of the scene's colour) away from it, which is the owner's decision of "none in Neutral"; and the density needs a strength per look, because Signature's reds are +4.5 L* above its greys and Neutral's +0.4.

## 9. Questions for the owner, with the answers of 2026-10-29

1. **Derived ARRI numbers in the repository.** Answered: **none.** This file was rewritten to keep only measurements of our own pipeline, data and sources, and to say in words where ARRI's rendering differs. The tables made on ARRI's LUTs are kept outside the repository.
2. **Fitting.** Answered: **tune to ARRI's rendering, with no ARRI-derived numbers in the repository** (section 8).
3. **Which target?** (a) "scene colour" (Neutral already is, within 1 L* and 1 degree on the owner's reds); (b) "an ALEXA 35 with REVEAL" (the camera-side luminance of section 7 item 2 would then have to go); (c) "an ALEV III with the 2013 IDT and a REVEAL display" (what the local comparison measured). *Open:* the owner said tune to ARRI's rendering; with the push kept (question 6), the camera side stays as it is and the display side is tuned.
4. **Signature.** Answered: **fixed by the red density stage only**; its saturation of 1.45 stays.
5. **Density and hue.** Answered: **a hue turn of -3 degrees in Signature only, none in Neutral.** *Open:* the strength of the density in Neutral (the prototype used 0.2; Signature 0.25 to 0.3), and whether Neutral gets a density at all.
6. **The 4.5 degree push (2026-10-26).** Answered: **keep it.**
7. **The branch.** *Open:* `reds-density` is in the shared checkout and its working tree is edited by another session. Is that the intended place for this work? This report's numbers are for `ea6cc73` only.
8. **ARRI files left on disk.** Answered for now: **keep `testdata/arri-luts/`** (12 MB, git-ignored; the developers tune against it); delete it when the work is done.

## 10. Reproduction

The session scratchpad `reds-audit/` holds, outside the repository: the Rust tool (`ymrt`: `ramp`, `image`, `white`, `range`, `yd`; candidate hooks), the Python scripts for sections 3 to 7, a data folder with extra looks (`rc0` = R1 off, `sigr1`, `sigcurve`, `sig115/125/135`) used through `YELLOWMOON_DATA`, and the JSON results. The tables made against ARRI's LUTs (two CSV files and the longer version of this report) are there too and are not in the repository. The photos are `testdata/owner-street/`, `testdata/readme-photos/src/` and two files of `testdata/lens-photos/`; a DNG of each (3 minutes at full resolution on the CPU) is the only large intermediate and was deleted after use.

## Sources

- **[S1]** ARRI, *ARRI LogC4 Logarithmic Color Space Specification* (header date 23 January 2025): AWG4 primaries (section 4.2), LogC4 to AP0 by CAT02 (4.3.2). `https://www.arri.com/resource/blob/278790/bea879ac0d041a925bed27a096ab3ec2/2022-05-arri-logc4-specification-data.pdf`
- **[S2]** ARRI, *REVEAL Color Science*, "Building blocks", "Cross-camera compatibility", the Mini LF comparison. `https://www.arri.com/en/learn/camera-systems/image-science/reveal-color-science`
- **[S3]** ARRI, *FAQ on ARRI's Color Management*: DRT names, look files, "redesigning" LogC3 show LUTs, ALF2 for ALEV III cameras. `https://arri.com/en/learn-help/learn-help-camera-system/image-science/color-faq`
- **[S4]** ARRI, *LUT Generator* page: "ALEXA Classic 709 is the well-known K1S1 LUT", the LogC4 and LogC3 LUT packages (10 July 2025). `https://www.arri.com/en/learn-help/learn-help-camera-system/tools/lut-generator`
- **[S5]** ARRI, *Mixing ARRI LogC3 and LogC4 On Set - Guideline*, 25 November 2024. `https://www.Arri.com/resource/blob/359512/56ec83a18dbcb2f147dd1498b90115f1/arri-mixing-arri-logc3-and-logc4-on-set-guideline-data.pdf`
- **[S6]** AMPAS aces-dev v1.3.1, ARRI ALEXA IDT README. `https://raw.githubusercontent.com/ampas/aces-dev/v1.3.1/transforms/ctl/idt/vendorSupplied/arri/alexa/README.md`
- **[S7]** AMPAS aces-dev v1.3.1, `v3_IDT_maker.py` (v0.09). `https://raw.githubusercontent.com/ampas/aces-dev/v1.3.1/transforms/ctl/idt/vendorSupplied/arri/alexa/v3_IDT_maker.py`
- **[S9]** ARRI, *LUT Naming Convention LogC4 and LogC3 - Guideline* (in the LUT packages; July 2025). `https://www.arri.com/resource/blob/294602/99cf69084be7f6638181660483d3e39d/arri-lut-naming-convention-logc4-and-logc3-guideline-en-data.pdf`
- **[S10]** ACES Central threads, seen only as web-search summaries (the site does not resolve here): *ALEXA muddy look in ACES* `https://community.acescentral.com/t/alexa-muddy-look-in-aces-aces-vs-rec709-in-davinci-resolve/3206`, *Adapting a show LUT to ACES* `https://community.acescentral.com/t/adapting-a-show-lut-to-aces/1428`, *Results from an IDT evaluation* (the Academy's P-2013-001 procedure) `https://community.acescentral.com/t/results-from-an-idt-evaluation/2229`. Forum reports, not ARRI.
- **[S11]** Ian McCausland, AbelCine, *ARRI's New REVEAL Color Science Explained*, 27 September 2022. `https://www.abelcine.com/articles/blog-and-knowledge/tutorials-and-guides/arris-new-reveal-color-science-explained`
