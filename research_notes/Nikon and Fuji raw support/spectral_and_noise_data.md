# Spectral sensitivity (SSF), noise and lens data for Nikon and Fujifilm bodies: public availability and licences (checked 2026-11-11)

Conventions used below.
- "Verified (repo)" means I cloned the repository on 2026-11-11 and read the files. "Verified (page)" means I fetched the page. "Search summary" means only a search-engine summary was seen. "Prior project research" means the fact comes from `research/canon-nikon-curves-2026-10-19.md` in this repository and was not re-checked.
- "RFSE*" is my own quick metric, computed locally on the cloned files: per channel, least-squares scale one curve onto the other, RMS difference divided by the channel peak, in percent; the mean of R, G, B is quoted. It is close in spirit to, but not identical with, the "relative full-scale error" of Solomatov & Akkaynak. Script kept only in the session scratchpad; the numbers are reproducible from the public files.

## 1. rawtoaces-data: which Nikon and Fujifilm bodies have a camera JSON, from whom, under what licence

### Takeaway
rawtoaces-data (HEAD `e9b8503`, 2026-07-20, 52 camera files, all Apache-2.0) holds 11 Nikon and 3 Fujifilm curves: Weta "lightsaber" measurements for the Nikon D3300, D5300, D800E, D810, D850, Z f and the Fujifilm X-T3, X-T4, GFX 100, plus older AMPAS/NPL files for the D70, D200, D700, D7000, D5100. That is the only shippable source found, and it covers exactly one Nikon Z body and three Fujifilm bodies (none of the 40 MP or X-Trans 5 generation).

### Cited Findings
- Repository licence is Apache License 2.0 (LICENSE file read), and every camera JSON carries `"license": "Apache-2.0"` in its header. Verified (repo). — [rawtoaces-data](https://github.com/AcademySoftwareFoundation/rawtoaces-data)
- HEAD at time of check: `e9b8503cf8a0641f40e5345d9757bde60c15f423`, dated 2026-07-20; `data/camera/` has 52 files, all 380–780 nm at 5 nm (81 samples), units "relative", channels R, G, B. Verified (repo). — [rawtoaces-data/data/camera](https://github.com/AcademySoftwareFoundation/rawtoaces-data/tree/main/data/camera)
- Nikon files (11), with header attribution. Verified (repo):

| Body | `document_creator` | `measurement_equipment` / lab | File date |
|---|---|---|---|
| Nikon D3300 | wetadigital | lightsaber | 2025-11-10 |
| Nikon D5300 | wetadigital | lightsaber | 2025-11-10 |
| Nikon D800E | wetadigital | lightsaber | 2025-11-09 |
| Nikon D810 | wetadigital | lightsaber | 2025-11-09 |
| Nikon D850 | wetadigital | lightsaber | 2025-11-09 |
| Nikon Z f | wetadigital | lightsaber | 2025-11-09 |
| Nikon D70 | scitech (AMPAS) | none stated | 2017-04-11 |
| Nikon D200 | scitech (AMPAS) | none stated | 2017-04-11 |
| Nikon D700 | scitech (AMPAS) | none stated | 2017-04-11 |
| Nikon D7000 | scitech (AMPAS) | none stated | 2017-04-11 |
| Nikon D5100 | Jonathon Irons | double monochromator, National Physical Laboratory | 2017-06-09 |

  — [rawtoaces-data/data/camera](https://github.com/AcademySoftwareFoundation/rawtoaces-data/tree/main/data/camera)
- Fujifilm files (3), all `document_creator: wetadigital`, `measurement_equipment: lightsaber`, Apache-2.0: **X-T3** (file date 2025-11-10), **X-T4** (2025-11-10), **GFX 100** (2026-03-31). Verified (repo). There is no other Fujifilm file. — [rawtoaces-data/data/camera](https://github.com/AcademySoftwareFoundation/rawtoaces-data/tree/main/data/camera)
- History (git log, verified): Weta's files were imported from physlight on 2025-09-16 (commit `cf6452c`, "add camera files from https://github.com/wetadigital/physlight"), refined on 2025-11-25 (`9a3320d`, "improve curve accuracy"), and the GFX 100 was added on 2026-04-09 (`145e771`, "add curves for Canon R6m2, Canon RP, Fuji GFX100, Hasselblad L2D"). Author of all three: Anton Dukhovnikov. — [commit cf6452c](https://github.com/AcademySoftwareFoundation/rawtoaces-data/commit/cf6452c3ce44112f6cf3f1c2d7bf6381a4c90638), [commit 145e771](https://github.com/AcademySoftwareFoundation/rawtoaces-data/commit/145e7713e8c892f76ef069fb6f92275f9d61de4a)
- `data/aliases.json` (Apache-2.0, dated 2026-03-27) has two Nikon "different model, same sensor" aliases: **D800 → D800E** and **Z 6_2 (Z6 II) → Z f**. It has **no Fujifilm aliases** and no cross-maker aliases. It also aliases Sony ILCE-7RM5 → ILCE-7RM4. Verified (repo). — [aliases.json](https://github.com/AcademySoftwareFoundation/rawtoaces-data/blob/main/data/aliases.json)
- The X-T3 and X-T4 are X-Trans (6×6 CFA) cameras, yet the files give plain R, G, B curves like the Bayer files; the schema has no CFA-layout field. Verified (repo) for the file content; X-Trans being a 6×6 Fujifilm CFA — [Wikipedia: Fujifilm X-Trans sensor](https://en.wikipedia.org/wiki/Fujifilm_X-Trans_sensor)
- Prior project research states Weta's own repository physlight (Apache-2.0) carries the same Weta files, that little is published about the "lightsaber" rig, and that physlight issue #3 ("Any more measured cameras to add?") has been open since March 2021 without a commitment. Not re-checked today. — [physlight](https://github.com/wetadigital/physlight), [physlight issue #3](https://github.com/wetadigital/physlight/issues/3)

### Inferences
- All 14 Nikon/Fujifilm curves can ship inside an Apache-2.0 app on the same terms as the 11 Sony curves Yellowmoon already ships (same repository, same licence, same Weta rig for 9 of the 14).
- Per-channel R/G/B curves are enough for a spectral emulation of an X-Trans body: the CFA layout affects demosaicing, not the per-channel spectral response.
- The three Fujifilm curves give Fujifilm a "make default" candidate even though only three bodies are measured (see section 4: the three are within about 1 % RFSE* of each other).

### Gaps
- I could not query the GitHub API (pull requests, issues) from this session, so whether any open pull request adds Nikon Z or Fujifilm bodies is unverified; prior project research (2026-10-19) found none for Nikon Z.
- No public description of the lightsaber rig's accuracy was found; the Weta files have no uncertainty data.

## 2. Other SSF datasets: Nikon/Fujifilm bodies, method, range, licence

### Takeaway
No other public dataset is both permissively licensed and relevant: everything outside rawtoaces-data is CC BY-NC-SA, unlicensed, or an estimate rather than a measurement, and almost all of it covers pre-2014 Nikon DSLRs. **No measured Fujifilm SSF was found anywhere outside rawtoaces-data.**

### Cited Findings

| Source | Nikon bodies | Fujifilm bodies | Range / step | Method | Licence (as stated) | Shippable in Apache-2.0 app |
|---|---|---|---|---|---|---|
| rawtoaces-data / physlight (Weta, AMPAS, NPL) | D70, D200, D700, D7000, D5100, D3300, D5300, D800E, D810, D850, Z f | X-T3, X-T4, GFX 100 | 380–780 / 5 nm | Weta "lightsaber"; AMPAS unspecified; NPL double monochromator | Apache-2.0 (verified, repo) | Yes |
| Jiang, Liu, Gu, Süsstrunk 2013 "camspec" (28 cameras) | D3, D3X, D40, D50, D80, D90, D200, D300s, D700, D5100 (10) | none | 400–720 / 10 nm | Monochromator + PR655, each channel normalised to 1 | CC BY-NC-SA 4.0 on Zenodo (verified, page); project page states no licence | No (NC + SA) |
| Glenn Butcher ssf-data (own measurements) | **Z 6**, D7000, D3500, D50 | none | 400–715 / 5 nm (Z6) | DIY diffraction-grating spectroscope, single image | "Copyrighted by Glenn Butcher with all rights reserved" + CC BY-NC-SA 4.0 (verified, repo) | No |
| ssf-data copies of other sets | camspec's 10 Nikons; Open Film Tools D750, D800 | none | as source | as source | camspec: CC BY-NC-SA 4.0; Open Film Tools: "license could not be determined" (verified, repo README) | No |
| HdM Stuttgart Open Film Tools | D750, D800 (per ssf-data directory names and a search summary) | none found | 401–784 / 1 nm per one ssf-data README | spectroscope-type set-up per ssf-data README | none determinable | No (validation at most) |
| Darrodi et al. 2015 (UEA/NPL reference set) | D5100 (plus Sigma SD1 Merrill) | none | not checked | NPL monochromator with per-wavelength error | none on data page; paper © OSA (prior project research) | The NPL D5100 is already in rawtoaces-data under Apache-2.0 |
| Solomatov & Akkaynak 2023, ground truths (49 CSVs + test set) | D1x, D3, D3X, D40, D50, D70, D80, D90, D200, D300s, D700, D5100, Coolpix 5700 (compiled from Jiang, Kawakami, Mauer & Wueller, Brady & Legge, Huynh & Robles-Kelly, Stevens) | **none** | 400–700 / 10 nm | compiled from literature | repository has **no LICENSE file** (verified, repo) | No |
| Solomatov & Akkaynak 2023, predictions (1012 files) | 91 Nikon files incl. Z 5, Z 6, Z 6 II, Z 7, Z 7 II, Z 9, Z 30, Z 50, Z fc, D780, D850, D500, D7500, D6 | 83 Fujifilm files incl. X-T2/T3/T4/**T5**, X-H1/**H2/H2S**, X-Pro1/2/3, X-S10, X-E4, X-T30/T30 II, X100S/T/F/**V**, GFX 50S/50R/50S II/**100/100S** | 400–700 / 10 nm (31 samples) | **estimated** from Adobe DNG matrices with an autoencoder; not measured | no licence (verified, repo) | No (all rights reserved by default); also not a measurement |
| Kawakami et al. 2013 | D1x, D70 (as included in Solomatov's ground truth) | none | — | reported as monochromator-based | not checked | unknown |
| Zhao et al. 2009 (Zenodo 4297288) | D1X, D70 | none | 400–700 nm | monochromator | "not specified" (prior project research) | No |
| Image Engineering camSPECS | product pages only, no public data (prior project research) | — | — | — | — | — |
| Abeysekera, Kuang, Ooi 2025, "Comprehensive Modeling of Camera Spectral and Color Behavior" | search summary says a 126-camera basis database built from online datasets and publications | unknown | unknown | compiled | arXiv abstract page names no data release or data licence (verified, page) | unknown / no |

Sources for the table:
- camspec licence and description (28 cameras; 400–720 nm, 10 nm; PR655 + monochromator + integrating sphere; channels normalised to 1): CC BY-NC-SA 4.0. — [Zenodo 3245883](https://zenodo.org/records/3245883); project page states no licence — [gujinwei.org camspec](https://www.gujinwei.org/research/camspec/)
- camspec Nikon list (from ssf-data `Nikon/*/camspec` directories and the Solomatov ground-truth file names; both verified in the clones). — [ssf-data](https://github.com/butcherg/ssf-data), [Spectral-sensitivity-estimation](https://github.com/COLOR-Lab-Eilat/Spectral-sensitivity-estimation)
- ssf-data README: data "measured by myself ... is Copyrighted by Glenn Butcher with all rights reserved and carries ... CC BY-NC-SA 4.0"; the Z 6 README gives lens NIKKOR Z 24-70mm f/4 S, range "400,715,5", date 2020-08-27, method "Transmissive glass-substrate ruled diffraction grating spectroscope, single-image". ssf-data HEAD `e715dd8`, 2026-05-18. Its manufacturer directories are Arri, Baumer, Blackmagic, Canon, Hasselblad, Nikon, Nokia, Olympus, Pentax, Phase One, Point Grey, Sony: **no Fujifilm directory**. Verified (repo). — [ssf-data](https://github.com/butcherg/ssf-data)
- Open Film Tools: the ssf-data README for an OFT-derived entry says "SSF data is not posted; license could not be determined". The OFT page `hdm-stuttgart.de/open-film-tools/en/camera_responsivities` returned HTTP 404 on 2026-11-11. A search summary says the database has Nikon D750 and D800 samples plus Canon, Sony and GoPro. — [ssf-data](https://github.com/butcherg/ssf-data), [Open Film Tools (404 today)](https://www.hdm-stuttgart.de/open-film-tools/en/camera_responsivities)
- Solomatov & Akkaynak: ground truth is "51 spectral sensitivity functions that we compiled from the literature"; sources listed include Jiang et al. (23 cameras used), Kawakami et al. (Canon 10D, 5D II, 5D; Nikon D1x, D70), Mauer & Wueller (Canon 40D, Leica M8, Nikon D200, D700, Panasonic LX3), Tominaga et al. (phones); no Fujifilm camera in the ground truth. Predictions cover "more than 1,000 cameras that the Adobe DNG Converter currently supports". Paper is CC BY 4.0 on arXiv; no licence for code or data. Repo HEAD `e995808`, 2023-07-24, no LICENSE file. — [arXiv 2304.11549](https://arxiv.org/html/2304.11549v2), [project page](https://color-lab-eilat.github.io/Spectral-sensitivity-estimation-web), [repository](https://github.com/COLOR-Lab-Eilat/Spectral-sensitivity-estimation)
- Darrodi et al.: measured SSFs for a Nikon D5100 and a Sigma SD1 Merrill at NPL with per-wavelength error estimates. — [UEA eprint 52181](https://ueaeprints.uea.ac.uk/id/eprint/52181/)
- Abeysekera et al.: arXiv abstract page gives no camera list, data release or metrics. — [arXiv 2507.04617](https://arxiv.org/abs/2507.04617)
- Prior project research table (Zhao 2009, NPL data page, Image Engineering, licences as stated). — [research/canon-nikon-curves-2026-10-19.md in this repo; sources there: Zenodo 4297288](https://zenodo.org/records/4297288), [spectralestimation data page](https://spectralestimation.wordpress.com/data/)

### Inferences
- A repository without a licence file grants no redistribution right, so the Solomatov predictions and ground-truth compilation cannot ship; they can be downloaded by tooling for evaluation only (the project's `docs/14` already classes them "evaluation only").
- CC BY-NC-SA 4.0 (camspec, Butcher) is incompatible with shipping in an Apache-2.0 app that anyone may use commercially; `docs/14` already lists camspec under "avoid".
- For Fujifilm the situation is binary: the three Weta curves, or nothing measured.

### Gaps
- colour-science/colour-datasets, a "Tucker Downs open spectral database", NUS/Tokyo datasets, dcamprof-shared SSFs and discuss.pixls.us collections were not individually inspected in this session (tool budget). Prior project research (2026-10-19) searched "Zenodo, colour-datasets, GitHub, the NPL data page, Open Film Tools, photographers' sites" and found no Apache-compatible Canon/Nikon measurement beyond rawtoaces-data; that search did not target Fujifilm.
- The camera list, data availability and licence of the 126-camera database in arXiv 2507.04617 were not verified (only the abstract page was read).
- No 2024–2026 dataset covering Nikon Z or Fujifilm X-Trans 5/GFX bodies was found, other than the Weta additions (Z f in 2025, GFX 100 in 2026).
- The Open Film Tools site could not be read today (404), so its exact camera list, range and terms are unverified.

## 3. Current bodies with no public measured SSFs

### Takeaway
Only the Nikon Z f and the Fujifilm X-T3, X-T4 and GFX 100 have shippable measured curves among mirrorless bodies; a non-shippable DIY curve exists for the Nikon Z 6. Every other Nikon Z and every other Fujifilm X/GFX body has no public measurement.

### Cited Findings
- Shippable measured, Nikon Z mount: Z f only. Fujifilm: X-T3, X-T4, GFX 100 only. Verified (repo). — [rawtoaces-data](https://github.com/AcademySoftwareFoundation/rawtoaces-data/tree/main/data/camera)
- Measured but not shippable: Nikon Z 6 (Butcher, CC BY-NC-SA 4.0, DIY spectroscope). — [ssf-data](https://github.com/butcherg/ssf-data)
- Bodies for which only an Adobe-matrix-based *estimate* exists (Solomatov predictions, unlicensed): Nikon Z 5, Z 6 II, Z 7, Z 7 II, Z 9, Z 30, Z 50, Z fc; Fujifilm X-T5, X-H2, X-H2S, X-S10, X-E4, X-Pro3, X100V, GFX 100S, GFX 50S II and others. The prediction set dates from July 2023 and has **no** file for the Nikon Z 8, Z f, Z6III, Z5II, Z50II or the Fujifilm X100VI, X-S20, X-T50, X-M5, X-E5, GFX100 II, GFX100S II. Verified (repo file list). — [Spectral-sensitivity-estimation/data/predictions](https://github.com/COLOR-Lab-Eilat/Spectral-sensitivity-estimation)
- Explicit list, no public measured SSF found (any licence): Nikon Z 5, Z5II, Z 6 II, Z6III, Z 7, Z 7 II, Z 8, Z 9, Z 30, Z 50, Z50II, Z fc, ZR, D780, D500, D7500, D6; Fujifilm X-T5, X-T50, X-T30/T30 II, X-H2, X-H2S, X-S10, X-S20, X-E4, X-E5, X-M5, X-Pro2/Pro3, X100F/V/VI, GFX 50S/50R/50S II, GFX 100S, GFX100 II, GFX100S II. This is the complement of the verified lists above. — [rawtoaces-data](https://github.com/AcademySoftwareFoundation/rawtoaces-data/tree/main/data/camera), [ssf-data](https://github.com/butcherg/ssf-data)
- Prior project research applied the project's sibling rule (published colour matrix within 0.02 per coefficient of a measured body) to Nikon: no Z body passes against the Z f (Z6III 0.041, Z 5 0.038 to the D5100, Z 6 / Z 6 II 0.087 to the D810 and 0.19 to the Z f, Z 8 0.10, Z 9 0.34 to the D850). — research/canon-nikon-curves-2026-10-19.md (this repo); matrices from Adobe as distributed, e.g. via the predictions repo's inputs [data/color-matrices](https://github.com/COLOR-Lab-Eilat/Spectral-sensitivity-estimation)

### Inferences
- Under the rule as written today Nikon would get one Z body with its own profile and Fujifilm three bodies; everything else needs a sibling or make-default fallback.
- The equivalent matrix-sibling check has not been run for Fujifilm (X-T3/X-T4/GFX 100 against X-T30, X-S10, X-Pro3, X100V, X-E4, GFX 100S and so on). It is a small job with the existing `training/other_makes/check.py`.

### Gaps
- "No public SSF" is a negative result from the sources inspected; a private or paywalled measurement (Image Engineering, DxO, the makers) may exist.

## 4. Sensor sharing, and how valid a same-silicon sibling is across makers

### Takeaway
Sharing Sony silicon does **not** mean sharing spectral response: on the public Weta curves, Fujifilm bodies sit about 6 % RFSE* from the Sony bodies on related silicon and Nikon bodies about 4–7 %, while siblings *within* a maker sit at about 1 %. A cross-maker "same sensor" fallback is therefore roughly as bad as the generic A7R IV default; a same-maker fallback is far better.

### Cited Findings
- Sensor attributions are unofficial. Nikon Rumors' 2025 list (which says its information "is not official" and may be wrong for some models) gives Sony as maker for the Z5, Z6II (IMX820 per its table, citing TechInsights), Z7II, Z8, Z9 (IMX609AQJ), Zf, Z5II, Z6III/ZR (IMX820AQJ per its update); Tower Semiconductor for the Z50/Z50II. — [Nikon Rumors, 2025-09-28](https://nikonrumors.com/2025/09/28/who-produces-the-sensors-for-nikons-mirrorless-cameras.aspx/)
- Search summaries (not primary) say: the IMX410 is "likely" behind the Sony A7 III and Nikon Z6 with different phase-detect rows; the IMX571 26 MP APS-C "appears to be the base" of the X-T3 with Fujifilm's own X-Trans filter; Sony lists the IMX461 100 MP medium-format sensor as used in the GFX100S; the IMX455 is the 61 MP A7R IV/A7R V sensor. — [ymcinema on IMX410](https://ymcinema.com/2026/03/05/sony-imx410-sensor-powering-half-camera-industry/), [sonyalpharumors sensor announcement](https://www.sonyalpharumors.com/sony-officially-announces-the-new-medium-format-full-frame-aps-c-and-mft-sensors/), [Digital Camera World on Nikon sensors](https://www.digitalcameraworld.com/cameras/who-makes-the-sensor-inside-your-nikon-camera-hint-it-isnt-just-sony)
- The X-T5 and X-H2 share a 40 MP "X-Trans CMOS 5 HR" sensor; Wikipedia lists the maker as Sony (search summary). — [DPReview X-T5 review](https://www.dpreview.com/?p=292307), [Wikipedia: Fujifilm X-T5](https://wikipedia.com/wiki/Fujifilm_X-T5)
- rawtoaces-data's maintainer asserts Z 6 II → Z f as "different model, same sensor"; no cross-maker alias exists. — [aliases.json](https://github.com/AcademySoftwareFoundation/rawtoaces-data/blob/main/data/aliases.json)
- **My computation on the Weta curves (380–780 nm, RFSE* per channel R/G/B, then mean):**

| Pair | R | G | B | Mean | Reading |
|---|---|---|---|---|---|
| Fujifilm X-T3 vs X-T4 | 0.3 | 0.8 | 0.8 | **0.6** | same maker, same sensor |
| Fujifilm GFX 100 (Bayer) vs X-T3 (X-Trans) | 0.4 | 1.4 | 1.2 | **1.0** | same maker, different sensor and CFA layout |
| Fujifilm X-T3 vs Sony A6400 | 7.9 | 2.6 | 7.3 | **5.9** | cross-maker APS-C |
| Fujifilm X-T3 vs Sony A7R IV | 8.3 | 2.3 | 6.6 | **5.7** | cross-maker, same 3.76 µm pixel generation (reported) |
| Fujifilm GFX 100 vs Sony A7R IV | 8.4 | 2.9 | 7.6 | **6.3** | cross-maker, same pixel generation (reported) |
| Nikon Z f vs Sony A7 III | 3.1 | 2.1 | 6.2 | **3.8** | cross-maker 24 MP full frame |
| Nikon Z f vs Sony A7 IV | 3.0 | 2.6 | 6.8 | 4.1 | |
| Nikon D850 vs Sony A7R III | 4.9 | 8.1 | 7.5 | **6.8** | cross-maker 42–46 MP |
| Nikon D810 vs Sony A7R II | 1.8 | 1.1 | 5.7 | 2.9 | |
| Nikon D810 vs D800E | 1.0 | 0.9 | 1.5 | 1.1 | same maker |
| Nikon D5300 vs D3300 | 1.3 | 0.5 | 0.7 | 0.8 | same maker |
| Nikon Z f vs D850 | 3.3 | 7.3 | 3.1 | 4.6 | same maker, the D850 is the outlier |
| Sony A7 III vs A7R III | 0.8 | 0.3 | 1.7 | 0.9 | same maker |
| Sony A7R III vs A7R IV | 1.0 | 0.7 | 2.4 | 1.4 | same maker |
| Canon R5 vs Sony A7R IV | 10.4 | 7.5 | 4.7 | 7.5 | reference: unrelated makers |

  — computed from [rawtoaces-data/data/camera](https://github.com/AcademySoftwareFoundation/rawtoaces-data/tree/main/data/camera)
- Shape differences behind those numbers (same files): the three Fujifilm red channels peak at 600 nm against 590 nm for the Sony and Nikon files, and keep 1.2–1.9 % of peak at 700 nm against 0.2–0.5 % for Sony and Nikon; Fujifilm red peaks at 0.70–0.74 of green against 0.60–0.61 for the Sony files. — computed from [rawtoaces-data/data/camera](https://github.com/AcademySoftwareFoundation/rawtoaces-data/tree/main/data/camera)
- Evaluation-only check with Butcher's DIY Nikon Z 6 curve (400–715 nm): Z 6 vs Weta Z f 3.0 mean (R 3.3, G 2.7, B 3.0); vs Sony A7 III 4.8 (B 8.1); vs D850 6.1; vs Fujifilm X-T3 5.6. For scale, Butcher's DIY D7000 vs the AMPAS D7000 file in rawtoaces-data is 4.1, so about 4 % is this rig's disagreement with another measurement of the *same* model. — computed from [ssf-data](https://github.com/butcherg/ssf-data) and [rawtoaces-data](https://github.com/AcademySoftwareFoundation/rawtoaces-data)
- Prior project research, in rendered-colour terms (ΔE00 of the ALEXA rendering, median / p95): the Z f with its own curves 0.27 / 0.95, with the A7R IV's 0.81 / 1.59; the D850 0.34 / 0.86 own, 2.09 / 3.82 with the A7R IV's; best single Nikon map (D5300) serves the other Weta Nikons at a median of medians 0.50 against 0.64 for the A7R IV map and 0.29 with own curves. The same research found bodies "widely reported to share a sensor" have very different published matrices (Z8/Z9 0.28, Z6/Z f 0.19). — research/canon-nikon-curves-2026-10-19.md (this repo), using [rawtoaces-data](https://github.com/AcademySoftwareFoundation/rawtoaces-data)
- Independent anecdote: the Nikon D3X and Sony A900 "both have the same sensor and yet score differently" on colour; the author presumes the CFA dyes differ (search summary of an On Landscape article). — [On Landscape](https://www.onlandscape.co.uk/?p=7505)
- Measurement-to-measurement spread is itself large: nine cameras with two published ground-truth derivations differ by about 2.4 % to 17 % relative full-scale error; two copies of one Nikon Coolpix 5700 differ by 6.96 %. — [arXiv 2304.11549](https://arxiv.org/html/2304.11549v2)

### Inferences
- Fujifilm applies its own colour filters and IR-cut stack consistently across its line: an X-Trans APS-C body and a Bayer medium-format body are within about 1 % of each other, and both are about 6 % from Sony bodies. So "Fujifilm X-T3/GFX 100 curve as the Fujifilm make default" is supported by three data points; "Sony A7R IV curve for a Fujifilm on IMX455-generation silicon" is not.
- For Nikon the picture is mixed: the Z f, D810, D800E, D5300 are spectrally close to Sony in red and green (2–3 %) but differ in blue (6–9 %); the D850 differs from everything. The Z 6 to Z f pairing (3.0 %) is within the DIY rig's own noise (about 4 %), so the alias Z 6 II → Z f is plausible but not proven.
- Recommended fallback order, as an inference from these numbers: (1) own measured curve; (2) same-maker body with a matching published matrix; (3) same-maker default (Fujifilm: X-T3 or GFX 100; Nikon: Z f or D5300 per prior research); (4) only then the Sony A7R IV default. Cross-maker same-silicon siblings should not rank above the same-maker default.
- X-Trans 5 (40 MP X-T5/X-H2, 26 MP stacked X-H2S) are a newer generation than any measured Fujifilm body, so even the same-maker fallback is an extrapolation there.

### Gaps
- No primary (Sony Semiconductor or teardown) source was read for any body-to-IMX mapping; all are trade-press or rumour-site claims.
- No published study was found that measures SSFs of two cameras from different makers on the same Sony silicon; the comparison above is my own, on Weta's files, with a non-standard metric.
- Whether the remaining Fujifilm generations (X-Trans 1–3, X-Trans 5, GFX 50) match the X-T3/GFX 100 curves is unknown: no measurement exists.

## 5. Fallbacks without measurement: matrices, SSF estimation, chart shots

### Takeaway
The only no-measurement route with published accuracy is estimation from Adobe's dual-illuminant matrices (Solomatov & Akkaynak: median about 8 % relative full-scale error, range 4–17 %), which on my check is no better than borrowing a same-maker sibling (about 1 %) and about equal to borrowing a cross-maker body (about 6 %). The matrices themselves are numbers the project's own licence rule already allows taking from any source; their copyright status has no authoritative source.

### Cited Findings
- Method: "use the colorimetric mapping matrices provided by the Adobe DNG Converter to formulate the camera-specific system of equations, and constrain the solutions using an autoencoder trained on a database of ground-truth curves". Reported accuracy: errors 4 % to 17 %, median about 8 % (leave-one-out per channel: red 9.3 %, green 6.3 %, blue 8.3 %); averaging all known curves gives 21 %; adding a ColorChecker photo improved accuracy by less than 0.5 %; the matrices cover only illuminants A and D65; training set is only 51 curves. — [arXiv 2304.11549](https://arxiv.org/html/2304.11549v2)
- **My check of those predictions against Weta measurements (400–700 nm, 10 nm, RFSE* mean of R/G/B):** Fujifilm X-T3 6.2, X-T4 6.1, GFX 100 6.0; Nikon D850 6.1, D810 6.6, D800E 6.2, D5300 6.1, D3300 6.3; Sony A7 III 7.3, A7R III 7.3, A7R IV 7.1. Red and blue errors are 6–10 %, green 3–5 %. — computed from [Spectral-sensitivity-estimation/data/predictions](https://github.com/COLOR-Lab-Eilat/Spectral-sensitivity-estimation) and [rawtoaces-data](https://github.com/AcademySoftwareFoundation/rawtoaces-data)
- The predictions repository also ships 2026 Adobe colour-matrix CSVs (`data/color-matrices/*-cm1.csv`, `-cm2.csv`), including Nikon Z 9 and many Fujifilm bodies, with no licence. — [Spectral-sensitivity-estimation](https://github.com/COLOR-Lab-Eilat/Spectral-sensitivity-estimation)
- Jiang et al.: camera SSFs lie in a low-dimensional space; their page reports an average colour difference under D65 of 1.59 for the PCA model against 3.54 (Fourier), 2.43 (radial) and 7 (polynomial) bases, and a rendering difference "close to one" for a Canon 60D simulation; the page does not state units or the metric precisely. — [gujinwei.org camspec](https://www.gujinwei.org/research/camspec/)
- Darrodi et al. found estimation algorithms were not accurate enough compared with the NPL ground truth (search summary of the abstract). — [UEA eprint 52181](https://ueaeprints.uea.ac.uk/id/eprint/52181/)
- Matrices in open-source tools: LibRaw describes dcraw's "large library of RGB to XYZ matrix coefficients known as adobe_coeff"; a LibRaw maintainer says that for cameras Adobe supports LibRaw mostly uses Adobe's matrices; the primaries in dcraw's table and RawTherapee's camconst.json are cam→XYZ matrices at D65 (search summaries of forum posts). — [LibRaw forum node 2186](https://libraw.org/node/2186), [discuss.pixls.us thread](https://discuss.pixls.us/t/more-raspberry-camera-fun/28174/17)
- Adobe's DNG SDK source files carry an Adobe licence notice ("Adobe permits you to use, modify, and distribute this file in accordance with the terms of the Adobe license agreement"); LibRaw's licence file refers to DNG SDK 1.4 code as MIT-licensed (search summary). These notices cover code, not the per-camera matrix values. — [DNG SDK source mirror](https://android.googlesource.com/platform/external/dng_sdk/+/e26fa7d/source/dng_camera_profile.h)
- Project position already written down: `docs/14-licensing.md` rule 8, "Numeric facts may come from any source: colour matrices, white-balance coefficients and similar camera constants may be taken as numbers from any source ... Only the numbers are taken: no code, file, table layout, text or comment". Prior research already holds published D65 matrices for 85 Canon and 58 Nikon bodies under that rule (`training/other_makes/colour_matrices.json`). — this repository; underlying data as distributed by [Adobe DNG Converter](https://helpx.adobe.com/camera-raw/using/adobe-dng-converter.html)
- Prior project research, matrix vs curve consistency: published matrices render the Weta Nikon curves at median 1.95 ΔE00 (Sony 2.32); over 341 same-make measured pairs, matrix distance and curve distance correlate at Spearman 0.58; pairs under 0.05 matrix distance cost a median +0.34 ΔE00 when borrowed, pairs over 0.1 cost +1.39. — research/canon-nikon-curves-2026-10-19.md (this repo), using [rawtoaces-data](https://github.com/AcademySoftwareFoundation/rawtoaces-data)

### Inferences
- A 3×3 matrix pair cannot determine a 3×31-sample SSF; the estimate is the prior (PCA basis or autoencoder) plus a weak constraint. An estimated curve at 6–8 % error is worse than a same-maker sibling and about equal to a cross-maker stand-in, so matrix-derived SSFs add little as a V1 fallback. Their better use is the one the project already makes: the published matrix as a *sibling test* and as an independent consistency check of a curve.
- Yellowmoon could re-implement a matrix-constrained estimate itself (rule 8 covers the matrices; the Weta curves are an Apache-2.0 training set of 44+ bodies, larger and more modern than Jiang's 28), which would avoid the unlicensed predictions. Whether this counts as "simulating a measurement" under the V1 data rule is the owner's call.
- A direct colour transform from the published dual-illuminant matrices (camera → XYZ, no SSF) is always available for any body Adobe supports, but it gives matrix-grade colour, not spectral emulation, and cannot express light-source dependence beyond two illuminants.
- ColorChecker shots from public archives would need the scene illuminant spectrum, which those archives do not publish; with an unknown illuminant, chart-based SSF estimation is under-constrained. Solomatov's own finding (under 0.5 % gain from a chart) suggests little benefit.

### Gaps
- No authoritative legal source was found on whether per-camera colour matrices are copyrightable (facts or measurements versus Adobe's fitted creative output), nor Adobe's explicit terms for the matrices embedded in DNG Converter output. This is a [HUMAN]/legal question; the project's rule 8 is its working position.
- Jiang et al.'s per-camera SSF-recovery error from a single ColorChecker image (the paper's own table) was not retrieved; only the project-page summary numbers above.
- Licence terms for raw files on imaging-resource, the DPReview studio scene and raw.pixls.us were not checked in this session. Prior project research states raw.pixls.us samples are CC0 and that it was unreachable from CI earlier. — [raw.pixls.us](https://raw.pixls.us/)
- No accuracy study of matrix-only or chart-only estimation on Fujifilm or Nikon Z bodies exists, apart from my eleven-body check above.

## 6. Noise data: public per-camera characterisations and single-file estimation

### Takeaway
Public noise characterisations exist for nearly every Nikon and Fujifilm body, but none is under a licence that allows shipping the tables (photonstophotos is all-rights-reserved, darktable's profiles are part of a GPL project). Per-image estimation from the raw file itself, which Yellowmoon already plans for Sony, needs no per-camera data; Nikon files usually carry optical-black pixels, Fujifilm files are unconfirmed.

### Cited Findings
- photonstophotos.net: "Web site and all content © (C)opyright 2026 William J. Claff. All rights reserved." No licence or reuse terms are given. "Data is measured from raw files taken to my specifications and contributed by people from around the world"; DxOMark-derived charts are described as "less reliable than those actually measured". Verified (page). — [photonstophotos.net](https://www.photonstophotos.net/)
- Claff's primer: Nikon raw data is zero-corrected (no bias offset), which complicates lens-cap statistics; "most Nikon models still include some OB photosites in the raw data", and their spread gives a read-noise estimate from a single file (measured with his NefUtil tool) (search summary). — [Read Noise - Optical Black](https://photonstophotos.net/GeneralTopics/Sensors_&_Raw/Sensor_Analysis_Primer/Read_Noise_-_Optical_Black.htm)
- darktable `data/noiseprofiles.json` (fetched from master today, 1.8 MB) has Poisson-Gaussian profiles per ISO for 69 Nikon models, including Z 5, Z5_2, Z 6, Z 6_2, Z6_3, Z 7, Z 7_2, Z 8, Z 9, Z f, Z fc, Z 50, Z50_2, D850, D780, D810, D800/D800E, D750, D500, D7500, and 43 Fujifilm models, including X-T2, X-T3, X-T4, X-T5, X-T30, X-T30 II, X-H1, X-H2, X-H2S, X-S10, X-S20, X-Pro2, X-Pro3, X-E3, X-E4, X-E5, X-M5, X100F, X100V, X100VI, GFX 50R, GFX 100, GFX100S, GFX100S II, GFX100 II. Verified (file). darktable as a whole is GPL-3.0-or-later (known from the project; file header not re-read). — [darktable noiseprofiles.json](https://github.com/darktable-org/darktable/blob/master/data/noiseprofiles.json)
- Fujifilm RAF: a blog post on X-T5 raw files maps extra pixels around the active area, including black pixels on the right, top and bottom, without deriving read noise; Jim Kasson wrote of GFX 100S dark frames that he did not know whether Fujifilm has masked black-reference pixels (search summary). — [blog post on reading X-T5 raw files (author not verified)](https://jacksblog-1j5.pages.dev/en/posts/how-to-read-raw), [Kasson blog](https://blog.kasson.com/?p=31523)
- Single-image estimation is established in the literature and already specified for Yellowmoon's Sony path: model var(μ) = a·(μ − δ_c) + b + row noise + quantisation; Foi-style fit on smooth regions with a clipped-data likelihood, PCA fallback for texture, one denoise-and-refit refinement (YOND reports error falling from up to 29 % to about 2 %); a wide log-normal prior on `a` for bodies without calibration; ELD found borrowing another camera's parameters cost only 0.01–0.36 dB. — research/notes/noise-estimation.md (this repo; its reference list carries the primary papers), e.g. [Foi et al. 2008, Practical Poissonian-Gaussian noise modeling](https://doi.org/10.1109/TIP.2008.2001399)
- DNG `NoiseProfile` (tag 51041) defines the per-plane model N(x) = sqrt(S·x + O); DNGs converted by Adobe carry it per ISO. — [Adobe DNG specification](https://helpx.adobe.com/camera-raw/digital-negative.html)

### Inferences
- The per-image estimator is make-agnostic by design: it needs black level, white level and the mosaic, not a per-camera table. Extending it to Nikon and Fujifilm needs no new data, only validation on public raw files (X-Trans needs per-colour planes extracted from the 6×6 pattern instead of the 2×2 one).
- If priors per body are wanted, darktable's a/b values or photonstophotos' read-noise/gain numbers could arguably be taken as "numeric facts" under `docs/14` rule 8, but both are large compiled tables (database-right and GPL-compilation questions), so that is an owner/legal decision; the safer V1 route is the wide prior the Sony path already uses for uncalibrated bodies.
- Things that differ from Sony and need care: Nikon subtracts black in camera and clips at zero on many bodies (the read-noise distribution is truncated, so a clipped-data likelihood is needed at the black end as well); some Nikon and Fujifilm bodies apply raw noise reduction or scaling at high ISO, which breaks the Poisson-Gaussian assumption (the existing "detect pre-filtered raws" step covers this). These statements are from memory and the Claff primer summary, not verified on files.

### Gaps
- DxOMark and sensorgen terms and current availability were not checked.
- Whether Fujifilm RAF files expose usable masked/optical-black pixels on current bodies is unconfirmed.
- No accuracy figures for single-image noise estimation specifically on Nikon NEF or Fujifilm RAF/X-Trans data were found.
- The exact licence header of darktable's noiseprofiles.json was not read; GPL-3.0-or-later is assumed from the project licence.

## 7. Lens correction data

### Takeaway
lensfun's database is CC BY-SA 3.0 and its library LGPL-3; the library cannot be used under the project's hard rule 1, and the database would bring share-alike obligations. The project spec currently says "No Lensfun" and relies on maker-embedded corrections, which is also the natural route for Nikon and Fujifilm, but that route was not verified here.

### Cited Findings
- lensfun README: the libraries are under LGPL version 3; "the lens database is licensed under the Creative Commons Attribution-Share Alike 3.0 license"; bundled applications are GPL-3. Distributions tag the package "LGPL-3 CC-BY-SA-3.0" / "CC-BY-SA-3.0 AND LGPL-3.0-only". Search summary of the README and distro spec files; the upstream README was not fetched directly. — [lensfun README mirror](https://github.com/kmilos/lensfun), [openSUSE spec](https://build.opensuse.org/projects/openSUSE/packages/lensfun/files/lensfun.spec?expand=0), [Gentoo package](https://packages-cdn-origin.gentoo.org/packages/media-libs/lensfun)
- Yellowmoon's overview spec: "Lens | Sony's embedded correction tags (vignetting, distortion, CA). No Lensfun." — docs/00-overview.md (this repo); background on the lensfun project at [lensfun on GitHub](https://github.com/lensfun/lensfun)
- Project precedent for CC BY-SA data: `docs/14` and `NOTICE` already ship the spektrafilm film profiles under CC BY-SA 4.0, unmodified, with attribution, URL and commit. — docs/14-licensing.md (this repo); licence text at [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)

### Inferences
- Shipping lensfun's XML database unmodified as a separate data file, with attribution, is mere aggregation and is compatible in the same way as the spektrafilm profiles; any modified or converted form of the database must itself be CC BY-SA (3.0 or later). The app's code would stay Apache-2.0, but it would need its own Apache-compatible implementation of the lens models (PTLens, poly3/poly5, TCA, vignetting), because the LGPL library is excluded by hard rule 1.
- CC BY-SA 3.0 (not 4.0) has no explicit one-way compatibility with Apache-2.0; the data cannot be relicensed or merged into Apache-licensed source files.

### Gaps
- Embedded maker corrections for Nikon NEF and Fujifilm RAF (which tags carry distortion, vignetting and CA parameters, whether Nikon's are encrypted, and whether they are documented outside GPL/Artistic-licensed code) were not researched in this session. From memory only, unverified: Fujifilm stores geometric distortion, CA and vignetting parameters in RAF maker notes, and Nikon stores distortion and vignetting data in maker-note blocks that are partly encrypted; both are described in ExifTool's tag tables. — to be checked against [ExifTool tag names](https://exiftool.org/TagNames/)
- lensfun's coverage of current Nikon Z and Fujifilm X/GFX lenses was not checked.
