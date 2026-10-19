# Canon and Nikon curves are already in our data; Canon's colour is the catch

*Research of 2026-10-19, for the owner's question: can Yellowmoon source measurements for Nikon and Canon cameras, so that it is not limited to Sony? Background only: nothing here changes a spec. The numbers come from `python3 -m training.other_makes.check` (results in `research/canon-nikon-curves/`), run on the pinned rawtoaces-data of `tools/fetch/spectral.sh`.*

Yes, and nothing new has to be sourced. **The data set we already ship Sony curves from, rawtoaces-data (Apache-2.0), holds 22 Canon and 11 Nikon curves.** Our pinned commit is upstream's latest, and `tools/fetch/spectral.sh` already downloads them: the ALEXA estimate's prior has used them since M1. 20 of the Canon and 6 of the Nikon curves are Weta FX measurements from the same "lightsaber" rig and the same release as our 11 Sony curves. The rest are AMPAS files from 2016–17 and one National Physical Laboratory monochromator measurement. Using them keeps the V1 data rule: they are published measurements, not ours. **Coverage of current bodies is uneven.** Canon's measured set covers 7 of its 17 RF-mount bodies (R, RP, R5, R5 II, R6, R6 II, R10), and 3 more (R8, R5 C, R50) have the identical published colour matrix of a measured body. Nikon has only the Z f among its 14 Z-mount stills bodies, plus D850-era DSLRs; the Z6, Z6 II, Z6III, Z7, Z8 and Z9 have neither curves nor a passing sibling. **No other public source can ship.** The web search found nothing else under a compatible licence for any current body, and Weta has not answered a public request for more cameras since 2021. **The files hold up:** the bodies' published colour matrices render the Weta Canon and Nikon curves at least as well as our Sony curves (median 1.58 and 1.95 ΔE00, against 2.32 for Sony). **The catch is Canon colour.** With its own measured curves, the ALEXA rendering of a Canon body keeps about twice the Sony residual. Every Canon of the last fifteen years fails the M1 sensor-map gate (per light median ≤ 0.5, p95 ≤ 1.2) on 8 or 9 of its 9 lights, daylight included: the R5 is at 0.70 / 2.05 under D6750, against 0.42 / 1.17 for the A7R IV. A root-polynomial fitted at each light still leaves 0.57–0.72 / 1.70–2.40. Seven of the eleven Sony bodies pass everything but LED-V2, and so do three Nikons (D7000, D800E, D810). Even so, Canon's own curves are far better than what a Canon would get today from the A7R IV stand-in (median 0.57 against 1.58, p95 up to 7.6). **The curves are the small part of the job.** No Apache-compatible decoder for NEF, CR2 or CR3 exists. Nikon's High Efficiency raw (TicoRAW) has no open decoder at all. CR3's codec is documented only in GPL text and touches a Canon patent application. Nikon's colour metadata is encrypted with tables that exist publicly only in GPL and Artistic-licence code.

## The data we already fetch holds 33 Canon and Nikon curves

rawtoaces-data at `e9b8503` (v1.1.0, 20 July 2026, upstream's latest) has 52 camera files, all `license: Apache-2.0`. A Weta FX maintainer imported Weta's set from physlight on 16 September 2025 ([cf6452c](https://github.com/AcademySoftwareFoundation/rawtoaces-data/commit/cf6452c3ce44112f6cf3f1c2d7bf6381a4c90638)). The same maintainer refined the curves on 25 November 2025 ([9a3320d](https://github.com/AcademySoftwareFoundation/rawtoaces-data/commit/9a3320dae5f2358f75b1ec52ad284aac61323456), "improve curve accuracy", 40 files). The 1Ds Mark III curves were fixed and four bodies added in April 2026 ([5abe5b2](https://github.com/AcademySoftwareFoundation/rawtoaces-data/commit/5abe5b24e54812c32c53676aac6196e0feca373e), [145e771](https://github.com/AcademySoftwareFoundation/rawtoaces-data/commit/145e7713e8c892f76ef069fb6f92275f9d61de4a)). Weta's own repository, [physlight](https://github.com/wetadigital/physlight) (Apache-2.0), carries the same 44 Weta files; they equal the rawtoaces-data files to the last digit. The legacy files' licence was set to Apache-2.0 in July 2025 ([`research/arri-sensor-spectral-data-sources.md`](arri-sensor-spectral-data-sources.md) traces that chain for the ARRI D21, which came in the same 2016 batch).

| Make | Weta FX, "lightsaber" (2025–26) | AMPAS "scitech" (2017 file date) | NPL double monochromator (2017) |
|---|---|---|---|
| Canon | EOS 5D, 5D Mark II, 5D Mark III, 5D Mark IV, 5DS, 1Ds Mark II, 1Ds Mark III, 1D X Mark II, 600D, 100D, 200D, 200D II (250D), EOS M, **R, RP, R5, R5 Mark II, R6, R6 Mark II, R10** | Digital Rebel XTi (400D), PowerShot S90 | |
| Nikon | D3300, D5300, D800E, D810, D850, **Z f** | D70, D200, D700, D7000 | D5100 |
| Sony (for comparison; these ship) | A7 III, A7 IV, A7C II, A7R II, A7R III, A7R IV, A7S II, A7S III, A9, A6400, RX100 IV | | |

What is known about the lightsaber is thin. physlight says only "as measured with our 'lightsaber' system". Weta's SIGGRAPH 2020 talk calls the data "profiled sensor data" and "an ongoing area of research" ([Langlands & Fascione](https://history.siggraph.org/wp-content/uploads/2022/08/2020-Talks-Langlands_PhysLight-An-End-to-End-Pipeline-for-Scene-Referred-Lighting.pdf)). Requests for its design went unanswered. That is the same footing as the Sony curves we ship, so it does not count against the other makes.

The curve files keep the channel balance (green or blue at 1.0, the other channels below), except the NPL D5100, whose three channels are each scaled to peak 1, so its raw white cannot be read from the curves (the "white" check below skips it).

rawtoaces-data also ships `data/aliases.json` (March 2026, [86818c3](https://github.com/AcademySoftwareFoundation/rawtoaces-data/commit/86818c3e1670aea6d0d893f24be654e6e6e8035c)). It maps regional names (Rebel SL3 and Kiss X10 to the 200D II, and so on). It also lists "different model, same sensor" pairs: 5DS R→5DS, 6D Mark II→RP, R5 C→R5, R50→R10, R8→R6 Mark II, D800→D800E, Z 6II→Z f, and also A7R V→A7R IV, which our own check rejected (`docs/03`, 0.0934). These are the maintainer's assertions, not measurements.

## Nothing else public can ship

A search of the obvious places (Zenodo, colour-datasets, GitHub, the NPL data page, Open Film Tools, photographers' sites) found no Apache-compatible measurement of any Canon or Nikon body beyond the list above. Of the sources it did find, only Glenn Butcher's DIY Z6 covers a current body that the list lacks.

| Source | Canon and Nikon bodies | Licence as stated | Use |
|---|---|---|---|
| Zhao et al. 2009 ([Zenodo 4297288](https://zenodo.org/records/4297288)) | 10D, 400D, 5D, 5D Mark II; D1X, D70 (monochromator, 400–700 nm) | not specified | evaluation only |
| NPL / Darrodi et al. 2015 ([data page](https://spectralestimation.wordpress.com/data/)) | D5100 (the same lab as the rawtoaces-data file) | none on the page; the paper is © OSA | evaluation only |
| Glenn Butcher, [ssf-data](https://github.com/butcherg/ssf-data) | Butcher's own DIY measurements of the **Z6**, D7000, D3500; copies of HdM and camspec files | "Copyrighted by Glenn Butcher with all rights reserved" and CC BY-NC-SA 4.0 | not shippable; the Z6 could check the Z6→Z f pairing if the owner allows non-commercial evaluation data here |
| Jiang et al. camspec ([Zenodo 3245883](https://zenodo.org/records/3245883)) | 28 cameras, mostly 2008–11 DSLRs | CC BY-NC-SA 4.0 | `docs/14`: avoid |
| HdM Open Film Tools | D750, D800, 5D Mark III, 6D (copies in ssf-data) | none | validation at most (`docs/14`) |
| Solomatov & Akkaynak 2023 | estimated from Adobe matrices, not measured | none | evaluation only (`docs/14`) |
| Image Engineering camSPECS | product pages, no data | — | — |

Weta has not said whether more cameras are coming. physlight [issue #3](https://github.com/wetadigital/physlight/issues/3), "Any more measured cameras to add?", has been open since March 2021. A physlight contributor, writing personally, said that publishing could strain relations with camera vendors, and that they would raise the question internally. Nothing was recorded after that. rawtoaces-data has no open pull request for a Nikon Z body, an R3 or an R8. Asking is a [HUMAN] email; the bodies that would matter most are the Z6 II, Z8 and Z9, and the R3 and R7.

## The curves agree with their bodies' matrices as well as the Sony curves do

A curve file says nothing about how well it describes the camera. The one independent check without a camera is each body's published D65 colour matrix (numbers only, `docs/14` rule 8; `training/other_makes/colour_matrices.json`). The check renders the 24 ColorChecker and 99 CES colours under D65 through the curves, turns the white-balanced camera RGB into XYZ with the published matrix, and measures ΔE00 to the true colours. Two references sit next to it. One is the same error with the curves' own best white-preserving 3×3, a floor. The other is the D65 white of the curves (R/G, B/G) against the white of the matrix. On the Sony files this reproduces the known differences of `docs/03` (A7 III red 3.5 %, A7 IV 6.2 %).

| Files | n | Published matrix, ΔE00 median (range of medians) | p95 (median) | Own best 3×3, median | Curve white vs matrix white, \|diff\| median (max) |
|---|---|---|---|---|---|
| Sony, Weta (shipped) | 11 | 2.32 (1.33–2.89) | 4.92 | 0.78 | 5.1 % (11.8) |
| Canon, Weta | 20 | **1.58** (0.91–2.44) | 3.82 | 0.80 | 5.3 % (10.1) |
| Nikon, Weta | 6 | **1.95** (1.22–2.16) | 3.94 | 1.17 | 2.4 % (11.1) |
| Canon, AMPAS | 2 | 2.05 (1.90–2.20) | 4.89 | 1.27 | 1.0 % (1.5) |
| Nikon, AMPAS and NPL | 5 | 2.05 (1.10–2.39) | 4.24 | 1.15 | 1.7 % (12.8) |

No file stands out as broken. The largest white differences are a legacy D200 (+12.8 % R/G), the Weta D810 (+11.1 % B/G) and our own RX100 IV (+11.8 %). This shows that the Canon and Nikon files are as consistent with their cameras as the files we ship. It does not prove either kind accurate (`matrix.csv`).

## Own curves beat the Sony stand-in by a wide margin

The measure is the one of `docs/03` "Why measured profiles matter": ΔE00 between the ALEXA rendering and the body's rendering through a per-light 3×3, median / p95 over 138 colours under 9 lights. It is reproduced here for the 11 Sony bodies to the third decimal (`emulation.csv`; every pair in `emulation_pairs.csv`).

| Body | Own curves | A7R IV (today's default) | Best other body of the same make |
|---|---|---|---|
| Canon EOS R | 0.64 / 2.35 | 2.57 / 5.43 | 5D Mark IV 0.65 / 2.25 |
| Canon EOS RP | 0.59 / 2.09 | 1.34 / 3.43 | R6 Mark II 0.64 / 1.93 |
| Canon EOS R5 | 0.55 / 1.92 | 1.01 / 2.85 | R5 Mark II 0.60 / 1.81 |
| Canon EOS R5 Mark II | 0.56 / 1.90 | 0.95 / 2.77 | R5 0.66 / 2.12 |
| Canon EOS R6 | 0.57 / 2.03 | 1.67 / 4.10 | 5DS 0.78 / 2.07 |
| Canon EOS R6 Mark II | 0.55 / 2.05 | 1.21 / 3.33 | RP 0.64 / 2.26 |
| Canon EOS R10 | 0.73 / 2.79 | 2.64 / 6.13 | 200D 0.86 / 3.23 |
| Canon EOS 5D Mark IV | 0.65 / 2.26 | 2.66 / 5.55 | R 0.71 / 2.49 |
| Nikon Z f | 0.27 / 0.95 | 0.81 / 1.59 | D7000 0.54 / 1.28 |
| Nikon D850 | 0.34 / 0.86 | 2.09 / 3.82 | D200 1.47 / 3.90 |
| Nikon D810 | 0.27 / 0.67 | 0.48 / 1.05 | D800E 0.43 / 1.26 |
| Nikon D5300 | 0.38 / 0.98 | 0.55 / 1.38 | D3300 0.36 / 1.24 |
| *Sony A7 III (shipped)* | *0.27 / 0.72* | *0.43 / 1.05* | *A7R III 0.34 / 0.84* |
| *Sony A7R IV (shipped)* | *0.34 / 0.87* | — | *A7R III 0.41 / 0.97* |

Over all 22 Canon curves the median is 0.57 with their own curves and 1.58 with the A7R IV's (p95 up to 7.55); over the 11 Nikon curves it is 0.30 and 0.81. For a body without curves the best single map of the same make serves the other Weta-measured bodies of that make at a median of their medians of **1.13 for Canon** (the RP's) and **0.50 for Nikon** (the D5300's), against 1.54 and 0.64 for the A7R IV's and 0.56 and 0.29 with their own curves (`make_defaults.csv`). A make's own default is worth having, but it does not replace measured curves. Two other readings stand out. Nikon's recent sensors are spectrally close to Sony's (the A7R IV map serves the D800E, D810 and D5300 at 0.43–0.55). The D850 is an outlier among Nikons (5.2 % RFSE from the Z f).

## Every Canon fails the sensor-map gate, even with its own curves

The M1 gate (`docs/13`, `training/sensor_map/gate.py`) refits the shipped Lab-optimal tables without the three gate LEDs. It then scores the 99 CES under the 9 held-out lights, per light median ≤ 0.5 and p95 ≤ 1.2; the A7R IV passes all but LED-V2, the accepted failure. Run on every curve as if it were a profile (`gate.csv`, `headroom.csv`; the A7R IV comes out at worst median 0.759 and p95 1.515, as recorded in `docs/03`):

| Curves | Passes all but LED-V2 | Daylight 6750 K, median / p95 | Worst light, gate | 3×3 fitted at each light itself, worst | Root-polynomial at each light, worst |
|---|---|---|---|---|---|
| Sony A7R IV | yes | 0.42 / 1.17 | 0.76 / 1.52 (LED-V2) | 0.67 / 1.41 | 0.51 / 1.31 |
| Sony A7 III | yes | 0.30 / 0.91 | 0.55 / 1.20 | 0.46 / 1.09 | 0.42 / 1.07 |
| Nikon D810 | yes | 0.41 / 1.13 | 0.71 / 1.67 (LED-V2) | 0.65 / 1.71 | 0.40 / 1.32 |
| Nikon Z f | no: D5250, D6750 | 0.55 / 1.58 | 0.79 / 2.12 | 0.81 / 2.03 | 0.46 / 1.63 |
| Nikon D850 | no: LED-RGB1 | 0.48 / 1.06 | 1.24 / 2.74 | 0.83 / 1.88 | 0.52 / 1.63 |
| Canon EOS R5 | no: all 9 | 0.70 / 2.05 | 1.31 / 4.06 | 0.76 / 2.21 | 0.57 / 1.70 |
| Canon EOS R5 Mark II | no: all 9 | 0.73 / 2.08 | 1.43 / 3.79 | 0.76 / 2.16 | 0.58 / 1.73 |
| Canon EOS R6 Mark II | no: all 9 | 0.70 / 2.33 | 1.34 / 4.19 | 0.78 / 2.33 | 0.58 / 1.84 |
| Canon EOS RP | no: all 9 | 0.79 / 2.26 | 1.39 / 4.25 | 0.79 / 2.33 | 0.61 / 1.87 |
| Canon EOS R10 | no: all 9 | 0.99 / 2.84 | 1.49 / 4.87 | 1.07 / 3.24 | 0.72 / 2.40 |

In all, 7 of the 11 Sony bodies, 3 of the 11 Nikons (D7000, D800E, D810) and none of the 22 Canons pass all but LED-V2. The closest Canons are the 2004 and 2007 1Ds Mark II and Mark III, which fail only the two LED lights. The failure is not the table blend. A 3×3 fitted at the gate light itself, the lower bound of the shipped map family, is still over the gate on almost every light for every Canon. A second-order root-polynomial (`training/sensor_map/gate_diagnostics.py`) gets the R-series medians to about 0.6 but not the p95 to 1.2. Richer maps (a 3D LUT, a small network) were not tried.

The cause is in the spectra. The Canon R-series sits 7.5–9.8 % RFSE from the ALEXA estimate (Sony 4.4–6.4 %, Nikon 3.8–5.8 %). After per-channel scale alignment, the mean Canon R-series red channel starts earlier: +0.17 of peak at 550–600 nm and −0.11 at 600–650 nm. Its green peaks at 515 nm instead of 530 and carries +0.11 more at 450–500 nm. A 3×3 mixes channels but cannot move where a channel picks up light, so yellows, oranges and cyans that Canon and the ALEXA see differently stay apart. The ALEXA side is an estimate (±10 % band, `research/alexa-ssf/README.md`), but its prior already contains the 22 Canon curves, so the estimate errs, if at all, toward Canon.

What it would look like: the Canon median of 0.55–0.75 is a small error, and the p95 of 2 to 3 is a visible shift on the worst twentieth of the colours. That is better than any stand-in, and not what Yellowmoon promises for Sony.

## Which current bodies would get a profile

`docs/03` lets a body borrow a measured profile when its published colour matrix is within 0.02 per coefficient of the measured body's, after row normalisation (`siblings.csv`, every Canon EOS and Nikon D and Z body with published numbers):

| Make | Measured | Passes (identical matrix unless a difference is given) | Fails the 0.02 check (nearest measured, difference) | No numbers |
|---|---|---|---|---|
| Canon, RF mount | R, RP, R5, R5 Mark II, R6, R6 Mark II, R10 | R5 C (R5), R8 (R6 Mark II), R50 (R10) | R7 (R5, 0.0205), R3 (R5, 0.027), R50 V (R10, 0.037), R1 (RP, 0.050), R100 (R10, 0.061), Ra (astro) | R6 Mark III |
| Canon, EF mount and M, still common | 5D Mark IV, 5DS, 250D | 5DS R (5DS), 850D (250D); M50 Mark II and M200 within 0.0186 of the 250D | 6D Mark II (100D 0.024; RP 0.23), 90D (R5, 0.091), M6 Mark II, 2000D, 4000D | |
| Nikon, Z mount | Z f | — | Z6III (Z f, 0.041), Z 5 (D5100, 0.038), Z 30 (Z f, 0.053), Z 6 and Z 6II (D810, 0.087; Z f 0.19), Z 8 (Z f, 0.10), Z 50 and Z fc (Z f, 0.14), Z 7 and Z 7II (D850, 0.25), Z 9 (D850, 0.34) | Z50II, Z5II |
| Nikon, F mount, still common | D850, D810, D800E, D5300, D3300 | D800 (D800E), D3400 (D3300) | D780 (identical to the unmeasured Z 6), D7500, D500, D6, D5600 and D3500 (D810, 0.031) | |

Under the rule as written, Canon gets 10 of its 17 RF bodies and Nikon 1 of its 14 Z bodies. Nikon's main current bodies (Z6 II, Z6III, Z8, Z9, Z5) would render through a stand-in.

The rule is safe but strict for these makes. In the measured set, the three pairs with identical published matrices (EOS M and 100D, D5100 and D7000, D3300 and D5300) have curves 1.0–2.0 % RFSE apart, and borrowing costs +0.08 to +0.26 median ΔE00. Over all 341 same-make measured pairs, matrix distance and curve distance correlate (Spearman 0.58 with the RFSE, 0.59 with the cost of borrowing): pairs under 0.05 cost a median +0.34, pairs over 0.1 cost +1.39 (`sibling_calibration.csv`). Bodies widely reported to share a sensor, though, have very different published matrices: Z8 and Z9 0.28, 6D Mark II and RP 0.23 (rawtoaces-data's aliases pair them), Z6 and Z f 0.19 (aliased too; reviews say the Z f has the Z6 II's sensor, [DPReview](https://www.dpreview.com/reviews/nikon-zf-full-frame-mirrorless-camera-review/)), R7 and 90D 0.11. Adobe's matrices for these makes are refitted per body and per version, so a non-identical matrix says little. Accepting the aliases would give Nikon the Z 6 and Z 6II. Only a measurement can settle those pairings, and Butcher's DIY Z6 curve is the only one there is.

## The curves are the small part: decoding and metadata

Yellowmoon decodes only ARW, and every Sony assumption in detection, white scale and lens data would need a counterpart.

| Piece | Canon | Nikon |
|---|---|---|
| Container and codec | CR2 (to about 2018): TIFF plus lossless JPEG slices, close to the lossless JPEG `ym-raw` already decodes for Sony (`decode/ljpeg`). CR3 (every RF body, most since 2018): ISO base media plus Canon's CRX (Rice–Golomb with run lengths, LeGall 5/3 wavelet; C-RAW is its lossy form). The best public description is [lclevy/canon_cr3](https://github.com/lclevy/canon_cr3), GPL-3.0: read for facts, never copy. It cites Canon patent application US20160323602A1, whose status was not checked | NEF: TIFF; uncompressed, lossless and lossy Huffman with a linearisation table (`NEFCompression` 1–10, [ExifTool](https://exiftool.org/TagNames/Nikon.html)). **High Efficiency and High Efficiency★ (13, 14) are intoPIX's TicoRAW**, offered on the Z8, Z9, Z f and Z6III among others and licensed by Nikon from intoPIX ([DPReview](https://www.dpreview.com/news/9624409613/nikon-is-licensing-intopix-s-ticoraw-technology-for-its-z9-camera-system/)); no open-source decoder exists, and Apple's does not read it either |
| Permissive decoders to reuse | none. LibRaw `crx.cpp`, rawspeed and rawler's CRX are LGPL | none. quickraw, rawloader, rawler, libopenraw and imagepipe are LGPL; zenraw is AGPL |
| Detection | Canon model ID in the maker notes; regional names (Rebel, Kiss) in rawtoaces-data's aliases | Model string only (`NIKON Z 6_2`, `NIKON Z6_3`) |
| Colour metadata | white-balance presets and colour temperatures in the maker notes, not encrypted (from ExifTool's tag tables; not checked on files), so the presets reference of the white scale (`docs/03`) should carry over | ColorBalance (0x0097) and ShotInfo encrypted for most bodies, keyed on serial number and shutter count. The two 256-byte tables of the cipher exist publicly only in GPL or Artistic-licence code (dcraw, ExifTool, LibRaw). `WB_RBLevels` (0x000c) is in the clear, enough for the as-shot white (inferred, not checked on files) |
| Oracles and test files | LibRaw and rawler as CI oracles (`tools/oracle`, as for ARW); CC0 samples on raw.pixls.us, which was unreachable from CI before (`docs/STATUS.md`) | the same |

A user can always convert to DNG with Adobe DNG Converter, but Yellowmoon reads ARW, not DNG, so that is no route today.

## What it would take

The data question has a plain answer: the measurements exist, are licensed, and are already fetched. Whether to use them is a product decision, because `AGENTS.md` and `docs/00` describe a raw processor for Sony ARW files. Four decisions are the owner's:

1. **Scope.** Do other makes come into V1, a V1.x, or V2? The V1 data rule allows them as it stands: they are published measurements, from the source the rule already names.
2. **Canon's colour against the gate.** With the 3×3 map, no Canon profile can pass the M1 gate, whose thresholds change only with the owner's agreement (`AGENTS.md`). The choices are: ship Canon against a reported per-make gate (the R-series sits at 0.65–1.0 median and 1.9–2.9 p95 under daylight, up to 1.5 and 4.9 under the LEDs); first try a richer map for every make (a per-light 3D LUT fitted on spectra, which also changes the Sony path); or start with Nikon.
3. **Sibling evidence.** Keep the 0.02 matrix rule (Nikon gets the Z f only), or also accept rawtoaces-data's "same sensor" aliases (Z6 and Z6 II to the Z f, 6D Mark II to the RP), which no shippable measurement confirms. Allowing Butcher's CC BY-NC-SA Z6 curve as evaluation-only data would test the first of these.
4. **Format rules.** Whether `docs/14` rule 8 ("numeric facts may come from any source") covers Nikon's two cipher tables, and whether CRX needs the patent check before it ships (part of M4-H4).

If the answer to 1 is yes, the order this research points to is:

| Step | Gives | Effort |
|---|---|---|
| Profiles for the Weta Canon and Nikon bodies (`build_profiles` and the registry with a make field; detection by make) | colour data for 26 bodies and their identical-matrix siblings | small: the fit, tables and checks are make-agnostic, as this research shows |
| A CR3 decoder for lossless CRX, written from the format facts with LibRaw as oracle | 10 of 17 RF bodies with their own or an identical-matrix profile | large: a wavelet codec |
| A NEF decoder for uncompressed and lossless files, plus the cipher if decision 4 allows | the Z f, D850, D810, D800E and others; every other Z body on a stand-in; High Efficiency files refused with a plain message | medium |
| CR2 on the existing lossless JPEG code | 5D Mark IV, 5DS, 1D X Mark II, 200D and the older bodies | small to medium |
| A [HUMAN] email to Weta (physlight issue #3) for the Z6 II, Z8, Z9, R3 and R7 | the largest coverage gaps, if they publish | an email |

Canon first, against a reported gate, would reach the most current bodies per unit of work. Nikon first would keep to the Sony standard of colour, but for one current Z body.

## Side finding: three Sony matrices are another body's

Checking `training/white_scale/colour_matrices.json` against current published numbers showed that the entries for the A1 II (400), A7 V (407) and A7R VI (410) are the matrices of the A1, the 2013 A7 and the 2013 A7R. The tool that read them matches model names by prefix and had no entry for the newer bodies. The A1 II now has a matrix of its own, 0.167 from the A1's after row normalisation, so its `sibling` white scale in `docs/03` rests on the artefact. The matrix-implied scales recorded there for the A7 V and A7R VI (not applied) were computed from the wrong matrices. Of the other 25, 18 are the body's own entry and 7 come from an entry the source shares on purpose (RX100 IV, A7S II, A9 II, A7R III A, A7R IV A) or from the body an "A" revision updates (A6400 A, A6100 A). It is recorded in `docs/STATUS.md` (Known issues) and not fixed here.

## Files

| File | What it holds |
|---|---|
| `research/canon-nikon-curves/emulation.csv`, `emulation_pairs.csv` | ALEXA-rendering error per body (own, A7R IV, best of the same make, the make default, RFSE to the A7R IV and to the ALEXA estimate) and the median for every pair of the 44 Canon, Nikon and Sony curves; `make_defaults.csv` the best single map per make |
| `research/canon-nikon-curves/matrix.csv` | curves against the published matrix (above) |
| `research/canon-nikon-curves/gate.csv`, `headroom.csv` | the sensor-map gate per light, and the 3×3 and root-polynomial fitted at each gate light |
| `research/canon-nikon-curves/siblings.csv`, `sibling_calibration.csv` | the 0.02 check for every Canon and Nikon body; matrix against curve distance for every measured pair |
| `training/other_makes/check.py`, `colour_matrices.json` | the script (3 minutes; `--skip-emulation` 15 s) and the published D65 matrices of 85 Canon and 58 Nikon bodies, numbers only |
