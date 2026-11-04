# Film simulation with spektrafilm's profiles: licences, data and pipeline (2026-11-01)

Background for `docs/24-film.md`. It records what was read, where, and what could not be read. It describes the world; it holds no instructions. The pages and files were read on 2026-11-01. Nothing here is legal advice. Where a source is someone else's summary it says so.

## 1. What was read

- **spektrafilm** (`github.com/andreavolpato/spektrafilm`), the `main` branch, shallow clone at commit `3bb2c2d2801ff68b92019cf1dbcbb133d60832bc` ("fix: vlog midgray exposure", 2026-06-07). `pyproject.toml` says version 0.3.4; the profile files say 0.3.2, created 2026-05-28; `CITATION.cff` says 0.3.2. `git ls-remote` showed the same commit as the head of `main`, and branches `dev`, `experimental` and `couplers-tuning-iterations` that were not read.
- Read: `README.md`, `SPEKTRAFILM_LICENSE.txt` (a byte-identical copy sits in `src/spektrafilm/data/license/`), `CITATION.cff`, the head of `LICENSE`, `src/spektrafilm/README.md`, the profile files, the data folders, and the modules that say what the stages are and in which order (`runtime/pipeline.py`, `runtime/topology.py`, `runtime/stages/*`, `runtime/params_schema.py`, `profiles/io.py`, `model/develop.py`, `model/couplers.py`, `model/grain.py`, `model/color_filters.py`, `model/illuminants.py`, `utils/spectral_upsampling.py`, `utils/gamut_compression.py` (head only)). **They were read to learn the profile format and the order and purpose of the stages. Nothing was copied, and the spec describes the stages in words.** The default numbers of the coupler, grain, halation and glare parameters are in that code; this note does not record them.
- **spektrafilm-rs** (`github.com/turbasvin/spektrafilm-rs`): `README.md`, `LICENSE`, `Cargo.toml`.
- **vkdt** (`github.com/hanatos/vkdt`, moved to `codeberg.org/hanatos/vkdt`): the top readme, the licence section, `src/pipe/modules/filmsim/readme.md`, the head of `mklut-profiles.py`, the head of `src/tools/spec/mkspectra.c`, and the folder listings through Codeberg's API.
- The pixls.us thread "Spectral film simulations from scratch" (first 20 of 1,047 posts, through the Discourse JSON), the OFX plugin's page (`spektrafilm.114c.de`) and the ART how-to page.
- The Eurographics Digital Library page of Mallett and Yuksel 2019, the RGL page of Jakob and Hanika 2019, `mitsuba-renderer/rgb2spec` (`README.md`, `LICENSE`) and the CIE's data-tables listing page.
- The repository's own `docs/02` (the cine rendering and its inverse), `docs/14`, `docs/17`, `docs/23`.

Not reachable from this session: `github.com` pages and its API (the git proxy served the clone; raw files came from `raw.githubusercontent.com`), the PDFs of the papers, Kodak's and Fujifilm's data sheets, and the rest of the pixls.us thread (the first 20 posts only).

## 2. The licences

### 2.1 The code is GPL-3.0

`LICENSE` is the GNU General Public License, Version 3, 29 June 2007 (SHA-256 `3972dc97…`). `pyproject.toml`: `license = { text = "GPLv3" }`. `CITATION.cff`: `license: "GPL-3.0-or-later"`. The two statements differ in the "or later"; the `LICENSE` file is version 3.

### 2.2 What the README says (verbatim; typos as in the source)

> "spektrafilm (all lower caps) is open for research, integration, and production use. The project is in rapid development, some ares are still being build and will change fast."

> "*The project is GPLv3 licensed*, so any derivative work must also be open source under the same license. Derivatve work includes any software, plugin, or tool that incorporates spektrafilm code or is directly inspired by its methods."

> "*JSON profiles and LUTs are CC BY-SA 4.0.*"

> "If *GPLv3 is not compatible with your project*, please reach out to discuss alternative options. I am very open to collaboration and integration, but I want to ensure that spektrafilm remains open source and for the community."

> "LUTs are on a strict "*commercial use, free share, no resale*" custom [license](SPEKTRAFILM_LICENSE.txt)."

> "Acknowledge spektrafilm in plugin descriptions, marketing, or credits (e.g. "film modeling powered by `spektrafilm`" or "film modeling inspired by `spektrafilm`", see `CITATION.cff`)."

`CITATION.cff` repeats the GPL and "directly inspired by its methods" sentences, and asks for the acknowledgement and for a citation of the repository or its Zenodo DOI in academic work. The preferred citation is "spektrafilm: a measurement-grounded, physically-based spectral framework for open film emulation" (Volpato, 2026), with no DOI yet (`doi: "xxxxx"` in the file).

### 2.3 `SPEKTRAFILM_LICENSE.txt` (SHA-256 `76d62950…`)

Title: "License for spektrafilm profiles, LUTs, and direct derivatives by Andrea Volpato. License: CC BY-SA 4.0 | Preamble v1.1 (2026-05-28)".

**Scope**, verbatim:

> "This license applies to the original spektrafilm profiles that can be found at https://github.com/andreavolpato/spektrafilm, in the subfolder src/spektrafilm/data/profiles, and to all direct derivatives of the profiles, such as copies in other projects, LUTs, or any other format that encodes the same content."

**What it permits**, verbatim:

> "Use these profiles and LUTs freely to grade anything - personal, educational, or commercial; small project or feature film; any platform, any deliverable. The works you make are yours. No royalties. No copyleft on your finished film or photograph. No obligation flowing to your distributor, broadcaster, or audience."

> "Modify, remix, regrade, recombine. Modified profiles or LUTs must stay CC BY-SA 4.0 and credit spektrafilm."

**Attribution**: every copy, redistribution, modification or derivative "in any format, in any context" must preserve "Andrea Volpato, named as the original author", "https://github.com/andreavolpato/spektrafilm, as the canonical source", "The CC BY-SA 4.0 license notice (this file, or a link to it)" and "A note that the file has been modified, if it has, and by whom". It names the places ("a .cube header, an ICC profile field, an app's About screen, a paid plugin, a derived .3dl, training-data documentation"). "Removing or hiding this information ends the license grant immediately (CC BY-SA 4.0, Section 6(a))." Suggested text for unmodified files: "spektrafilm by Andrea Volpato / https://github.com/andreavolpato/spektrafilm / Licensed CC BY-SA 4.0"; for modified files: "Derived from spektrafilm by Andrea Volpato / … / Modified by [your name]."

**"What I ask, beyond the legal floor"** ("These are not legal requirements. They are the spirit of the project"):

- keep the licence file unchanged, and describe changes "in a file CHANGELOG.txt shipped together with this license";
- credit spektrafilm "in the human layer too - project docs, about pages, credits, breakdowns";
- "Don't use "spektrafilm" or my name in product branding without asking. The license covers the files; the name is not part of that grant. Factual reference is welcome and even encouraged, for example "graded with spektrafilm", or "this app uses spektrafilm LUTs"";
- "Paid apps and tools that add real value are welcome. Charge for what you built - UI, performance, support, a service. Please don't repackage the LUTs themselves as a paid LUT pack";
- "Please don't train commercial AI models on the LUTs or their outputs. CC BY-SA doesn't forbid this, but it is asked. … Non-commercial research training is fine."

**Contact**: "GitHub Issues or Discussions on the spektrafilm repository. For private inquiries about commercial licensing, name use, or custom work, write to andrea.volpato@outlook.com." (`CITATION.cff` lists another address for the author.)

The preamble is followed by the verbatim CC BY-SA 4.0 legal code. Points of that text that matter here: 2(a)(4) "simply making modifications authorized by this Section 2(a)(4) never produces Adapted Material" (technical modifications for another medium or format); 2(a)(5)(c) and 3(b)(3), no additional terms and no "Effective Technological Measures" that restrict the rights; 2(a)(6), no implied endorsement; 2(b)(2), "Patent and trademark rights are not licensed under this Public License"; 3(b), the ShareAlike condition applies to "Adapted Material You produce"; 6(a)–(b), termination on non-compliance and reinstatement within 30 days of discovery if cured.

### 2.4 Things in the sources that do not agree, or that need a question

1. **The README calls the licence "strict"** ("commercial use, free share, no resale") and calls it a "custom license"; the file itself is CC BY-SA 4.0 with the no-resale and no-training wishes placed under "not legal requirements". The file is the licence text; the README sentence is a summary. We follow the stricter reading in practice (`docs/24` §6.3).
2. **"directly inspired by its methods"** (README and `CITATION.cff`) says a derivative work includes a tool "directly inspired by its methods". Copyright does not protect methods, and the GPL covers code, but the author states the wish plainly. This is the main legal question of the plan (`docs/24` §9, item 1).
3. **The licence covers the `profiles` folder and its direct derivatives.** The other data in the repository (`data/luts`, `data/filters`, `data/icc`) is not named. See 3.2.
4. **Each profile file carries its own notice** in `metadata.license` and `metadata.copyright` ("Copyright (c) 2026 Andrea Volpato. Licensed under CC BY-SA 4.0."), and a `metadata.datasource` that says: "Original data are property of the respective holders. Film/photo-paper: Kodak and Fujifilm data-sheets, scientific publications, and technical material." The curves were taken from the manufacturers' data sheets and processed (unmixed and fitted) by the author.

### 2.5 The other implementations

- **spektrafilm-rs**: `LICENSE` is GPL-3.0, `Cargo.toml` says `license = "GPL-3.0"`. The README calls it "a Rust port" of the Python project, says the profiles and spectral data "come from there", names the aim as "bit-identical output to the Python reference", and ends with "Claude code for being this awesome." It was not used for anything but the facts of this note.
- **vkdt**: the readme's licence section says "our code is licenced under the 2-clause bsd licence (if not clearly marked otherwise in the respective source files, which contain a bit of viral GPLv3, so handle with care if you're afraid of the GPL)". Its `filmsim` module "is an implementation of Andrea Volpato's spektrafilm"; its readme repeats the profile licence and says "the vkdt datafile lut has been generated from the spektrafilm `profile/*json`" and that "the grain model was swapped out". The licence of the filmsim files themselves was not read (the files were not opened past the headers), so **it is not known whether they are BSD-2 or GPL**. Not used.
- **The OFX plugin** (`spektrafilm.114c.de`) says it is "a free OpenFX plugin"; its licence was not read. **ART** integrates spektrafilm through exported LUTs; that page's own text is CC BY-NC-SA 4.0 and is not relevant.

## 3. The data inventory

### 3.1 The profiles: `src/spektrafilm/data/profiles/`

28 JSON files, **5,794,633 bytes (5.53 MiB)**, between 205,422 and 209,348 bytes each; 1,118,939 bytes as `tar.gz`, 805,804 as `tar.xz`. Every file has the same layout (§4). By kind:

| Kind | Count | Stocks (the profile ids) |
|---|---|---|
| Colour negative, still | 11 | `kodak_portra_160`, `kodak_portra_400`, `kodak_portra_800`, `kodak_portra_800_push1`, `kodak_portra_800_push2`, `kodak_ektar_100`, `kodak_gold_200`, `kodak_ultramax_400`, `fujifilm_c200`, `fujifilm_pro_400h`, `fujifilm_xtra_400` |
| Colour negative, cine | 5 | `kodak_vision3_50d`, `kodak_vision3_250d`, `kodak_vision3_200t`, `kodak_vision3_500t`, `kodak_verita_200d` |
| Colour positive (slide) | 4 | `kodak_ektachrome_100`, `kodak_kodachrome_64`, `fujifilm_velvia_100`, `fujifilm_provia_100f` |
| Print paper (still) | 6 | `kodak_portra_endura`, `kodak_endura_premier`, `kodak_ektacolor_edge`, `kodak_supra_endura`, `kodak_ultra_endura`, `fujifilm_crystal_archive_typeii` |
| Print film (cine) | 2 | `kodak_2383`, `kodak_2393` |

Each negative names its usual print in `info.target_print` (Portra, Ektar, Gold and Ultramax on Portra Endura; the Fujifilm negatives on Crystal Archive Type II; the five cine negatives on 2383). Slides have none and are scanned directly. SHA-256 prefixes (first 16 hex digits) of the files as read are in the appendix.

### 3.2 Everything else the simulation needs

| Item | Where in spektrafilm | Size | Source named in the repository | Licence status |
|---|---|---|---|---|
| Spectral upsampling coefficients, "Hanatos 2025" (`hanatos_irradiance_xy_coeffs_250304.lut`) | `data/luts/spectral_upsampling/` | 4,194,741 B; a 16-byte header (`1234`, `17039362`, 512, 512) then 512 × 512 × 4 float32 coefficients; 421 trailing bytes | the code says "Thanks to hanatos for providing luts and sample code to develop this." The pixls.us thread has hanatos offering "code for all possible and impossible spectral upsampling methods" (2025-02-10) | **no licence stated** for the file; it sits in the GPL repository outside `profiles`. The vkdt tool that makes this kind of table (`src/tools/spec/mkspectra.c`: "creates a lut for spectral upsampling, (x,y) -> c0 c1 c2 b") is in vkdt, BSD-2 by the readme. **Not usable until its author says so** |
| Spectra table derived from it (`irradiance_xy_tc.npy`) | same folder | 5,972,096 B; float16, 192 × 192 × 81 (a spectrum 380–780 nm, 5 nm, for each cell of a chromaticity square) | same | same |
| Dichroic filter spectra (Thorlabs, Edmund Optics, Durst "digital light") | `data/filters/dichroics/` | 3 × 3 CSV files, 4 to 7 KB each | the folders name the vendors; one `info.txt` gives a product-page URL and "curv_52549.pdf" | **manufacturers' curves, no licence stated** in the repository. The default filter model is not these files but an analytic one (error-function edges: one each for the yellow and cyan filters and two for magenta; a code comment names the QD Europe product pages as the data), so they are optional |
| Schott heat-absorbing glass KG1, KG3, KG5 | `data/filters/heat_absorbing/schott/` | 3 CSV files | `info.txt`: a Newport product page | manufacturer curves, no licence stated. The default enlarger lamp uses KG3 |
| Canon 24 mm lens transmission | `data/filters/lens_transmission/canon/` | 1 CSV | `info.txt`: a lensrentals.com blog post | a blog author's measurement, no licence stated. Used only by the optional lamp variant `TH-KG3-L` |
| Neutral print filters (`neutral_print_filters.json`) | `data/filters/` | 22,195 B | none (computed by the project) | no licence stated. It holds, per negative and paper, the filter values that make a grey card neutral; an engine can solve for them instead |
| ICC profiles (Elle Stone, Compact-ICC-Profiles) | `data/icc/` | 916 KB | `data/icc/README.md`: Elle Stone's well-behaved profiles, **CC BY-SA 3.0**; saucecontrol's compact profiles, **MIT** | for embedding in saved images; **not needed** by a node that returns a picture |
| CIE 1931 2° observer, CIE illuminants D50 and D55, a blackbody, an "Incandescent" source, "Kinoton 75P" | not in the repository: read at run time from the `colour-science` library (BSD-3 code, which embeds the CIE tables) | – | `colour` | the CIE tables are recorded in `docs/14` as CC BY-SA 4.0; the CIE listing page that was read states no licence in its text, so **re-check on each dataset page**. The source of "Kinoton 75P" in `colour` was not checked. It is the viewing light of the two cine print films only |
| Reflectance sets used to fit the profiles (Otsu, Munsell, NIST skin, forest colours, Japan colours) | named in `metadata.datasource`; not in the repository | – | Zenodo and NIST pages | **not needed**: they were inputs to the author's fit, not to a run |

### 3.3 Why this matters

The profiles are the only data the licence names, and they are everything a film run needs that is specific to a stock. Everything in the second table is either optional, replaceable by a small formula or a table made by us, or standard colour science that the project has to settle anyway (`docs/14`, the CIE rows). The upsampling table is the one piece that is neither: it is the author's and hanatos's work product, outside the licence, with no licence of its own.

## 4. The profile format

All numbers are JSON; `null` stands for NaN. The wavelength grid is **380 to 780 nm in 5 nm steps (81 samples)**. `data` holds, per profile (names as in the files):

| Field | Shape | Meaning, as the code uses it |
|---|---|---|
| `wavelengths` | [81] | the grid, in nm |
| `log_sensitivity` | [81, 3] | log10 of the spectral sensitivity of the red, green and blue sensitive layers of a film (of the paper for a print profile). Used as `10^x`, NaN read as 0 |
| `hanatos2025_adaptation_window_params` | [4] for the 20 camera films, [0] for the 8 print materials | four numbers of a smooth band-pass applied to the sensitivities before they meet the upsampled spectra: the centre and the width of an error-function edge at the blue end and at the red end. The product is renormalised so that the film's reference illuminant gives the same raw values as without it |
| `hanatos2025_adaptation_surface_params` | [3, 15] for the 20 camera films, [0] for the print materials | three fifteen-number sets of a degree-4 polynomial surface over the chromaticity square (with a warp parameter), soft-limited to ±2 stops, that corrects the film's exposure by chromaticity. **Off by default in the code** (`apply_hanatos2025_adaptation_surface: bool = False`); the window is on |
| `channel_density` | [81, 3] | the spectral absorption of the cyan, magenta and yellow dyes per unit of density (the densitometer is `status_M` for negatives and `status_A` for papers and positives). NaN where undefined |
| `base_density` | [81] | the spectral density of the support, fog and unexposed processed film (the orange mask of a negative) |
| `midscale_neutral_density` | [81] | the spectral density of a neutral at mid exposure |
| `log_exposure` | [256] | the common exposure axis, log10 relative exposure from −3 to 4 |
| `density_curves` | [256, 3] | diffuse density of each dye against `log_exposure`, for a neutral exposure ramp. The code treats them as the result **including** the DIR couplers' effect on a grey ramp and works out the curves without couplers when the couplers are switched on |
| `density_curves_layers` | [256, 3, 3] | each channel's curve split into three sublayers (fast and slow emulsions); used by the grain |
| `density_curves_model` | `{ model_type: "cdfs", centers, amplitudes, sigmas }`, each [3, 3] | the same curves as a sum of three Gaussian cumulative distributions per channel; used to reshape a print's curves |

`info` holds `stock`, `name` (the marketing name, for example "Kodak Portra 400"), `type` (`negative` or `positive`), `support` (`film` or `paper`), `stage` (`filming` or `printing`), `use` (`still` or `cine`), `antihalation` (`strong`, `weak`, `no`), `target_print`, `channel_model` (`color`), `densitometer`, `log_sensitivity_density_over_min`, `reference_illuminant` (D55 for daylight films, T for tungsten, TH-KG3 for papers and print films) and `viewing_illuminant` (D50, and K75P for the two cine print films). Measured on the files, the largest density of a channel's curve lies between 1.5 and 2.9 for the 16 negatives, between 2.1 and 3.6 for the 4 slides, between 1.5 and 2.5 for the 6 papers and between 3.0 and 4.6 for the 2 print films.

### What is not in a profile

The profiles hold no DIR-coupler, grain, halation, glare, diffusion or lens-blur parameters. Those are defaults in the Python code (`params_schema.py`), tuned by the author ("the saturation level is arbitrarily guessed", pixls.us, 2025-02-09). The enlarger lamp (a 3400 K blackbody behind KG3 glass), the yellow, magenta and cyan filters and the scanner are also code. So a program that reads only the profiles has the stocks but not the author's rendering choices. Our engine makes its own (`docs/24` §3).

## 5. The pipeline

Order of the stages and the taps where the Python code can enter and leave (`rgb_in`, `rgb_pre`, `log_e_film`, `cmy_film`, `log_e_print`, `cmy_print`, `rgb_out`):

1. **Pre-process**: input RGB (linear, in named primaries), automatic exposure (optional), the film format sets the size of a pixel in micrometres (the long edge of the picture is taken as 35 mm of film).
2. **Film exposure** (`rgb_in` → `log_e_film`): scene RGB is adapted to the film's reference illuminant, turned into a chromaticity and a scale; the chromaticity indexes a table of spectra; the spectrum meets the film's three sensitivities (through the band-pass window) to give three raw exposures proportional to the scale; then the exposure compensation, a highlight boost, an optional diffusion-filter blur, a lens blur and **halation** (in-emulsion scatter plus back-reflection from the support, in micrometres on the film) act on the exposure; log10.
3. **Film development** (`log_e_film` → `cmy_film`): exposure to dye densities through the characteristic curves; **DIR couplers** (a developer-inhibitor release: density formed in one layer reduces the exposure the same and the other layers see, with a diffusion in micrometres) raise saturation and local contrast; **grain** (a particle-count model whose statistics depend on pixel area against particle area) and a small density blur.
4. **Enlarger and print exposure** (`cmy_film` → `log_e_print`): the negative's spectral density is lit by the enlarger lamp through yellow, magenta and cyan dichroic filters (set in Kodak CC units, neutral values per negative and paper), and the light meets the paper's sensitivities; exposure is normalised to mid grey, then a print-exposure factor, optional pre-flash and an optional diffusion filter.
5. **Print development** (`log_e_print` → `cmy_print`): paper dye densities through the paper's curves (optionally reshaped).
6. **Scan or view** (`cmy_print` → `rgb_out`): the print's spectral density under the viewing illuminant, projected on the CIE 1931 observer to XYZ; scanner glare (a random veiling light), conversion to the output primaries, an output gamut and lightness compression into [0, 1], an optional scanner lens blur and an unsharp mask, an optional encoding curve.

A positive (slide) stock skips stages 4 and 5 and is scanned directly. The code's own LUT mode switches off every spatial and random effect, which shows where the per-pixel part ends: stages 2 to 6 minus halation, the coupler diffusion, grain, glare and the blurs.

Internal LUTs: the enlarger and the scanner reductions are optionally replaced by three-dimensional tables (default grid 17 in the settings, 33 in the LUT creator) with cubic interpolation, and the exposure step by a two-dimensional table of the film's raw response over the chromaticity square. So the author's own implementation already bakes the colour part.

## 6. The papers and sources the stages come from

| Source | What it gives | Availability and licence, as checked |
|---|---|---|
| Mallett and Yuksel, "Spectral Primary Decomposition for Rendering with sRGB Reflectance", Eurographics Symposium on Rendering, DL-only and Industry Track, 2019, doi 10.2312/sr.20191216 | any sRGB reflectance as a linear mix of three spectra, one per Rec.709 primary; the sRGB value is reproduced under D65. The method the first spektrafilm used | the DL page lists the paper (18.3 MB) and a supplement (15.37 MB); **the page states no licence**. The three basis spectra are in the supplement and in `colour-science`. Reflectances only and the sRGB gamut |
| Jakob and Hanika, "A Low-Dimensional Function Space for Efficient Spectral Upsampling", Computer Graphics Forum (Eurographics 2019) 38(2) | a spectrum as a sigmoid of a quadratic polynomial of the wavelength, three coefficients per colour found by optimisation and stored in a table; zero error on the sRGB gamut, large gamuts supported | paper and supplement on the RGL page (17.7 MB and 36.8 MB); **no licence stated on the page**. The reference code `mitsuba-renderer/rgb2spec` is **BSD-3-Clause**, "Copyright (c) 2020 Wenzel Jakob" |
| "Hanatos 2025": a chromaticity-to-spectrum table, the quad (triangle-to-square) mapping of the chromaticity plane, and the 2025-03-04 coefficient file | a table indexed by CIE xy that returns a smooth emission spectrum, so scene light of any chromaticity inside the spectral locus can be upsampled; homogeneous in intensity | **no paper was found.** The name is the spektrafilm authors' and the Rust port's ("Johannes Hanatos"; hanatos is the vkdt author, whose site is `jo.dreggn.org`; we believe he is Johannes Hanika, the second author of the paper above). The generator in vkdt (`mkspectra.c`) includes `fit-spectra.h`, `macadam.h` and an inpainting module; the file headers were not read for licences |
| Giorgianni and Madden, *Digital Color Management*, 2nd ed., Wiley 2008; Hunt, *The Reproduction of Color*, 6th ed., Wiley 2004 (chapter 15 on colour couplers); Jacobson, Ray, Attridge and Axford, *The Manual of Photography*, 9th ed., Focal Press 2000 | the printed background of the stages (the README cites the first two; the pixls.us post cites the three) | books; not opened. Ideas and equations in books are free to implement; nothing is copied |
| Kodak data sheets (for example e4050 for Portra 400 and e4021 for Portra Endura, named by the README) and Fujifilm's | the curves and spectra the profiles are made from | Kodak and Fujifilm documents, copyrighted; not fetched (the one URL tried returned 404). The profile files say "Original data are property of the respective holders" |
| Selwyn's law and the published "RMS granularity" figures on the data sheets | the usual physical basis for a grain amplitude against measuring aperture | general knowledge; not looked up. The data sheets of the stocks carry the figures |

## 7. The trademark question

The profiles' `info.name` fields are the marketing names ("Kodak Portra 400", "Fujifilm Velvia 100", "Kodak Vision3 500T", "Kodak Professional Portra Endura"). The licence grants none of the names' rights (CC BY-SA 4.0, 2(b)(2)). The project's own rule for ARRI (`AGENTS.md` rule 3, `docs/14` rule 6) is that a name is used only as a factual reference and never as the name of anything we make, with a line that says we are not affiliated. In general, naming a product to say what is meant, and no more, with no suggestion of endorsement, is the usual footing of such lines ("nominative use" in US law; similar ideas elsewhere); how far that reaches for a list of stocks inside a picture editor is a legal question this note does not answer. spektrafilm itself, the OFX plugin, vkdt, ART and the Rust port all list the stocks by their names. The options are in `docs/24` §9, item 4.

## 8. What the repository already measured that bears on the choice of variant A

From `docs/02` (C12, *Inverse*), measured by the project:

- The cine rendering has no closed-form inverse. `CineRendering::inverse` is numerical: a first-use table of 840 KiB (0.3 to 0.6 s once per process), then a damped chord iteration. **1.65 µs a colour, 0.61 Mpx/s per thread**: a 24 MP picture takes about 40 s on one thread. It therefore cannot run per pixel on a large picture; it can run at the nodes of a table.
- Display channels are held to [1e-6, 1]. For a reachable colour `forward(inverse(d))` equals `d` within 1e-6 in linear light (the test); the reference alone is at 4e-14.
- **The rendering is not onto.** About 5 % of the 8-bit sRGB colours (5.4 % of 20,000 random ones) are unreachable: bright, strongly coloured ones, display luminance about 0.5 to 0.6 with a channel near the edge. For them the inverse returns the nearest reachable colour (same luminance and hue, less chroma): ΔE00 from the request median 4.6, 95th percentile 13.2, maximum 15.7. **None of the 1.7 million pixels of the eight test photos' own renders is unreachable.** A film print's colours are not those of the rendering's own output; their share is not measured.
- Display white (1.0) is reached by the neutral axis from +10.4 stops over grey (0.18 × 2^10.4, about 243 linear); LogC4 goes to 469.8 (`docs/02` C8). The DNG of a look with its own display rendering already uses the inverse (`docs/09`).
- The rendering runs on each output's own primaries (Rec.709, Display P3, Rec.2020), so the inverse is specific to an output space.
- Middle grey 0.18 renders at display 0.1110 (sRGB code 93.6); a photographic print puts 18 % grey at about 0.18 reflectance, so a print scan is about 0.7 stop brighter than the app's rendering unless the print exposure is placed to match.

## 9. What could not be checked

- The licence of the "Hanatos 2025" tables, of the filter CSVs and of vkdt's filmsim files.
- Whether the CIE's data pages carry CC BY-SA 4.0 (the listing page does not say; `docs/14` says so).
- Whether Eurographics licenses the two papers (the pages show none).
- The Python code's output: spektrafilm was not installed or run (Python 3.13 and a heavy dependency set: napari, PySide6, OpenImageIO). Every statement about its numbers rests on the repository's text.
- Anything about patents on the stages; none was searched for.
- The rest of the pixls.us thread beyond the first 20 posts, where the table's licence may have been discussed.
- How the profiles' `datasource` statement ("Original data are property of the respective holders") would be read by the manufacturers.

## Appendix: the files as read (SHA-256, first 16 hex digits; commit `3bb2c2d`)

`SPEKTRAFILM_LICENSE.txt` 76d62950 · `LICENSE` 3972dc97 · `CITATION.cff` 49358f49.

Profiles: `fujifilm_c200` 30db07d88090aadc · `fujifilm_crystal_archive_typeii` 0db98e5b8e2ed524 · `fujifilm_pro_400h` da1be731ab79fb55 · `fujifilm_provia_100f` f061f7aa55e3fefe · `fujifilm_velvia_100` af696fa9662e0483 · `fujifilm_xtra_400` 5f2a3642c7c8db5e · `kodak_2383` eb7fdfa3b75a909b · `kodak_2393` b564aece5e9fe43a · `kodak_ektachrome_100` ef194fd80c60a1ac · `kodak_ektacolor_edge` 5db23694d478c35c · `kodak_ektar_100` 60acbd46c7500577 · `kodak_endura_premier` fd37807d61eb730c · `kodak_gold_200` aa8a0c1c7851ffac · `kodak_kodachrome_64` f88ede487219cf75 · `kodak_portra_160` 1bc9b65a6940f3a4 · `kodak_portra_400` b5cdc73dbec8f462 · `kodak_portra_800` 5b0c6e861c6273a2 · `kodak_portra_800_push1` dac9cbcd2442442d · `kodak_portra_800_push2` 5ee01e4a00c15772 · `kodak_portra_endura` e3bb6069e02d088d · `kodak_supra_endura` 3a0928013effbb33 · `kodak_ultra_endura` 319fa2514e9a6809 · `kodak_ultramax_400` 248b78bdcc4fdbe8 · `kodak_verita_200d` 5d71a3f544721a71 · `kodak_vision3_200t` 4a28f049fd6bad67 · `kodak_vision3_250d` 9ab85c8627c0533b · `kodak_vision3_500t` d7beca68bf16db88 · `kodak_vision3_50d` fd426abc9de91168.

Upsampling files: `hanatos_irradiance_xy_coeffs_250304.lut` 6bfb031761bd5a68 · `irradiance_xy_tc.npy` 52f90c724811f5fa.
