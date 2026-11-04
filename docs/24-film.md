# 24 · Film node

A **Film node** is a node that makes a photo look as if it had been shot on a colour film, printed on paper and scanned. It simulates a negative, an enlarger, a print and a scan, from published film and paper data, with our own engine (`ym-film`, Apache-2.0). The film and paper profiles are Andrea Volpato's spektrafilm profiles (CC BY-SA 4.0), shipped unmodified as separate, credited data files. This file covers what the node is, where it sits in the graph and how it keeps the picture's colour, the engine, the controls, the data and its licences, the tests, the risks and the tasks.

**Status: Scheduled (the owner, 2026-11-04: "let's continue film-node now and actually add film node"), on branch `film` after 0.2.0. The mockup round (F-01) is the first step.**

It is **not in 0.2.0**. Nothing here is built, and the Add node menu of `docs/23` does not list the Film node until it is. **The mockups are in `design/film/`** (F-01, 2026-11-04); the owner chose among their options on the same day (F-H1, §4.4). The owner's process is "first we imagine UI, then we plan": a mockup round (F-01) comes before the node is built, and §4 is a brief for that round, not a design. The facts behind this file, with sources and quotes, are in `research/film-spektrafilm-2026-11-01.md`. The owner's decisions are in §1.3 and `docs/16` (*Licensing and data*).

**In short.**

- The node is a new kind, `film`, in the graph of `docs/23`: `picture` and `key` ports, masks, bypass, like a corrector.
- **Colour (§2): variant A.** The node ends with the *inverse of the cine rendering*, so it returns LogC4 / AWG4 like every node, and the Output renders it as it renders everything. Nodes after it, masks, thumbnails and the DNG keep working. The cost is a numerical inverse (baked into a table) and a few unreachable colours.
- **Engine (§3).** Our own, from the papers and the data sheets, in f64 on the CPU, with the colour part baked into small tables and the spatial parts (halation, coupler diffusion, grain) as passes of the multi-pass plan.
- **Data (§6).** 28 profiles, 5.5 MiB, committed unmodified in `data/film/spektrafilm/` with `LICENSE`, `ATTRIBUTION.md` and `SHA256SUMS`; credited in `NOTICE`, About and the node.
- **Code is not usable.** spektrafilm's code and the Rust port are GPL-3.0. We read them only for the profile format and the order of the stages, and copy nothing (§3.1).
- **The big open questions (§9):** the author's stated view that a tool "directly inspired by its methods" is a derivative; the naming of film stocks; the licence of the spectral upsampling table.

## 1. What it is

### 1.1 The node

A new node kind `film` in the graph of `docs/23`, listed in the **Add node** menu when it is built (not now):

| Item | Line | State |
|---|---|---|
| **Film** | "Models a colour negative, a print and a scan." | enabled, once built |

It has a `picture` input and output and a `key` input and output, like a corrector (`docs/23` §3.4), so the graph's rules, links, thumbnails, copy and paste, Bypass and masks work as they do for the other nodes. It is not in the default graph. A typical graph is Input → Main → **Film** → Output. The node returns LogC4 / AWG4 (§2), so a corrector after it is allowed.

### 1.2 What it simulates

1. A **negative**: the film's three sensitive layers see the scene's light, scatter and halation spread it, the characteristic curves turn exposure into dye density, DIR couplers add saturation and local contrast, grain adds the film's noise.
2. An **enlarger and a print**: a lamp with yellow, magenta and cyan filters lights the negative; the paper's three layers see it; the paper's curves turn it into dye density.
3. A **scan**: the print under its viewing light, measured with the CIE 1931 observer, returned as colour.

A slide film (a positive) skips the print and is scanned directly. Cine negatives print on cine print film (Kodak 2383 and 2393) once the projector lamp's spectrum is sourced (§6.4).

### 1.3 The owner's decisions (2026-11-01)

| Words | What they allow |
|---|---|
| "can we use this as a new node in future? https://github.com/turbasvin/spektrafilm-rs" | the question; the Rust port is GPL-3.0 and not usable as code |
| "Ok, spektrafilm has it's own license, check it out: https://github.com/andreavolpato/spektrafilm" | the licence check: code GPL-3.0, profiles CC BY-SA 4.0 (`research/film-spektrafilm-2026-11-01.md` §2) |
| The recommendation was our own Apache-2.0 engine that reads the film and paper profiles, shipped as CC BY-SA 4.0 data with full credit. The owner: **"ok, do a new branch and describe how we will ship spektrafilm as a new Film node that we will add later"** | this plan: the engine is ours, the data is Andrea Volpato's, the node comes after milestone N |
| **"Credited, of course"** | the credit lines of §6.2 |

**Decided 2026-11-04** (F-H0, F-H2 and items 1, 3, 4 and 9 of §9):

| Question | The owner's answer | What it means |
|---|---|---|
| Schedule (F-H0) | "let's continue film-node now and actually add film node" | milestone F starts now, on branch `film` |
| Write to the author and to hanatos (§7, F-H2) | **"It's fine just to mention Volpato"** | no letter; the profiles are credited to Andrea Volpato as §6.2 says. The engine is written clean-room from the papers (§3.1 stays in force: the shingle check of F-06 and the module headers), and its own spectral table (§3.3). F-H2 is closed and F-03 does not wait |
| Variant A or B (§2) | **A: back to scene colour** | the node returns LogC4 / AWG4 through the inverse of the cine rendering; A+ stays an optimisation for after F-04 |
| Stock names (§6.5) | **Real names with the disclaimer** (option 1) | "Kodak Portra 400" as data in `data/film/params/names.json`, with "Modelled on published data. Not affiliated with Kodak or Fujifilm." The trademark check (F-H4, with M4-H3) still comes before a release |
| CIE tables (§6.4, §9 item 9) | **Ship them as a separate credited file** | `data/film/cie/` with its own `LICENSE` and credit, as the profiles |
| Tune our models toward the oracle (§8, F-06) | **"Tune toward its outputs"** | our enlarger, lamp, paper exposure and spectral reading are fitted to spektrafilm's outputs only (black box, never its code), as the window was; the oracle's gates stay as written |
| The baking error on random colours (§3.8a, §8, F-04) | **"Gate photos, report random"** | the gate holds on every pixel of the 20 test photos × the 8 stocks; random colours are measured and reported, not gated |
| Grain amplitude without data-sheet figures (§3.9a, F-05) | **"Formula for now"** | every stock uses the ISO formula of `data/film/params/grain.json` until per-stock RMS granularity is entered |
| Running spektrafilm as the test oracle (§3.1, §8) | **"You can use spectrafilm, but never commit its code into our repo"** | its Python runs from a scratch virtualenv as a separate process, as a black-box oracle (F-06); none of its files enters the repository; the fixtures are numbers (CC BY-SA 4.0, credited) |

Shipping CC BY-SA data as separate, credited files is new for this project (weights trained on CC BY-SA data are the earlier case, `docs/14` rule 5); these answers allow it. `docs/14` and `docs/16` record it.

### 1.4 What it is not

- Not a LUT pack, not a LUT exporter, not a preset list with a brand on it. The node's name is **Film**.
- Not spektrafilm. The code is not used, the author is not involved, and the name appears only as credit (§6.3).
- Not a measurement. We take no measurements (`AGENTS.md`, V1 data rule). The curves are the author's reading of the manufacturers' data sheets.

## 2. Where it sits in the graph, and its colour

### 2.1 The problem

Our nodes work in LogC4 / AWG4 between a fixed Input and a fixed Output. The Output makes the picture: the look (C10), the cine rendering (C12), output sharpening, the encode (C13). A film simulation ends **display-referred**: the scan of a print is already a picture, with a paper white and a paper black. It must meet the rest of the graph without breaking it.

### 2.2 Variant A: return LogC4 through the inverse of the cine rendering

The node's last step turns the scan's display colour into the scene colour that the cine rendering renders to it, `CineRendering::inverse` (`docs/02` C12 *Inverse*; the DNG of a look with its own display rendering already uses it, `docs/09`). The node's output is LogC4 / AWG4. With a Neutral look and no node after it, the Output renders the print scan back, so the picture is the print.

**What it costs, with the numbers the project measured (`docs/02`):**

- **The inverse is numerical, and it runs per pixel.** The plain `inverse` costs 1.65 µs a colour and is neither smooth nor cheap enough to bake: F-04 to F-04c stored it in the 3D table T3b and met the gate for Display P3 only at 161³ (67 MB, 9 s per paper and output), and never for sRGB. **F-04d (§3.8c) moves it out of the table.** The scan is a table (T3b′, smooth); the return then runs per pixel, in the inverse's own coordinates: the display side and the scene side in closed form, the core inverse from a 730 KiB table (built once per process, 0.3 s on one thread) and at most five Newton steps with the exact Jacobian. It costs about 1.7 µs a pixel on one CPU thread for photographs. The graph, the port types and the gates do not change.
- **The rendering is not onto.** About 5 % of random 8-bit sRGB colours have no pre-image (bright, strongly coloured: display luminance about 0.5 to 0.6, a channel near the edge). The inverse returns the nearest reachable colour: the same luminance and hue, less chroma (ΔE00 from the request: median 4.6, 95th percentile 13.2, maximum 15.7). None of the 1.7 million pixels of the eight test photos' own renders is unreachable, but **a film print's colours are not the rendering's own outputs, and their share is not measured**. F-04 measures it on 20 public photos and the gate of §8 decides (reconsider B or A+ if it fails). Highly saturated stocks (Velvia, Ektar) may be the cases.
- **Round trip**: for a reachable colour `forward(inverse(d))` equals `d` within 1e-6 in linear light, far below one 10-bit code (about 1e-3). The scan table T3b′ adds its own interpolation error (§3.8c), inside the gate's bound of one 10-bit code; the per-pixel return itself round-trips to 3e-11.
- **What clips**: display values are held to [1e-6, 1]; a paper black is above that and the white is exactly 1. Display white maps to the flat top of the rendering, about +10.4 stops over grey (about 243 linear), inside LogC4's range (469.8). Nothing clips in the encode. A node after the Film node with a large gain (say +4 EV) clips at LogC4's top like after any node.
- **T3b′ depends on the output's primaries.** The scan's display colour is in the output's primaries, and the rendering's gamut stage runs on each output's own (Rec.709, Display P3, Rec.2020); step 1 of the inverse is therefore per output. The viewport (Display P3) and an sRGB export have different T3b′ tables (§3.8), and a colour at the edge of sRGB may differ between them as it does for every picture. The core table does not depend on the output (§3.8c).
- **The look still acts** after the node. A Signature look on top of a film print stacks two creative curves. The person chooses Neutral for the print as scanned; the Output tab says so while a Film node is live (copy in §4.6).
- **Brightness.** Middle grey renders at display 0.111 in the app; a print puts 18 % grey near 0.18. The node places the default print exposure so that scene grey lands where the rendering puts it, so adding the node does not make the picture jump in brightness (§3.6). Whether "as printed" is offered is a question for the mockup round (§9, item 10).

**What it keeps:** the graph format and the port types (`docs/23` §3.4) do not change; nodes, masks, key links, node thumbnails, Show mask, copy and paste, the multi-pass plan, output sharpening and the DNG work as they do for any node.

### 2.3 Variant B: a `display` port type

A new port type `display` that may link only to the Output, which then skips C10 and C12 for that link.

- **Gains**: no inverse, no unreachable colours, no table that depends on the output primaries beyond the gamut mapping; the print is exact within the output gamut; less to bake.
- **Costs**: a **format change** (`docs/23` §3.4 and §3.5: a new port type and its rules, the schema, the plan compiler carrying a picture's domain); **no node after the Film node** (a corrector cannot take a display picture), so the person cannot trim the print; **no masks** (scene and display pictures cannot be mixed); the Output's Look tab means nothing for such a link; node thumbnails need a path for a display picture; **the DNG cannot carry it** without the inverse of A (the DNG note would say "film is not in the DNG", or A's inverse comes back through the side door); the Output must still map the scan's primaries to the output's and compress the gamut.

### 2.4 Other ideas

- **A+: A with the pair elided.** When the Film node feeds the Output directly and the look is Neutral (or at intensity 0), the plan drops both the inverse and C12 and renders the scan itself. This is exact even for the unreachable colours, with no round-trip error. The cost is two code paths that differ at unreachable colours (a Neutral corrector inserted after the node changes the result slightly there), both tested. It is an optimisation to decide after F-04 measures the unreachable share.
- **C: film as a property of the Output** (a rendering choice next to the cine rendering). Natural physically, but it is not a node, has no order, no masks and no key links, and the owner asked for a node.
- **D: a scene-referred "film-like" LogC4 look** (the curves and colour mixing, then the usual rendering). Simple, but it is not the print; it is a look. It leaves out what makes the simulation worth having.

### 2.5 Recommendation: A

A changes nothing in the graph's format, keeps every node feature working (masks, order, thumbnails, DNG) and reuses an inverse that is already built, tested (gate 1e-6) and used for the DNG. B is cleaner for the colour and expensive for the graph. **A, with A+ as a later optimisation if the unreachable share is not small.** The owner decides (§9, item 3).

| | A | B | A+ |
|---|---|---|---|
| Graph format | unchanged | new port type | unchanged |
| Nodes after Film | yes | no | yes |
| Masks, key ports | yes | no | yes |
| Thumbnails, Show mask | as other nodes | new path | as other nodes |
| DNG | baked in | not possible without A's inverse | baked in |
| Print colours outside the rendering's reach | desaturated | exact | exact when fused, desaturated otherwise |
| Extra bake | inverse at 117,649 nodes, per paper and output | none | as A |
| Two code paths | no | yes | yes |

### 2.6 The picture at both ends of the node (variant A)

- **In**: the picture as the node before made it, LogC4 / AWG4, balanced by the Input. The node takes the picture to be lit by **the film's own reference light** (daylight for daylight films, tungsten for tungsten films): a neutral in the picture is a neutral on the film. So a tungsten film shows no blue cast on a daylight photo; the enlarger's neutral filtering (§3.5) balances the print.
- **Out**: LogC4 / AWG4 that the cine rendering turns into the print's scan. Paper white is scaled to display 1.0 (the scanner's white correction) and a paper black is the paper's own.
- **Mask blend** at the end, as for a corrector: `out = in + m · (film(in) − in)` in LogC4, with `m` the mask times the node's Strength.

## 3. The engine, `ym-film`

A new crate, **Apache-2.0, ours.** It depends on `ym-color` (LogC4, AWG4, the cine rendering and its inverse), not on `ym-core`. Rule 6 of `AGENTS.md` holds: nothing runs Python in the app or `ym`. Python may generate the one table we make and write test fixtures (`training/film/`, `tools/oracle/spektrafilm/`).

### 3.1 Provenance: the clean-room rule

spektrafilm's code is GPL-3.0; the Rust port is GPL-3.0 (`docs/14`). The author states that a tool "directly inspired by its methods" is a derivative (§9, item 1). So:

1. **The GPL code may be read only to understand the profile format and the order of the stages. Never to copy it, translate it, port it or follow it line by line.** This covers spektrafilm, spektrafilm-rs and, until its licence is read, vkdt's `filmsim` module.
2. **The engine is written from** the published science (§3.3 to §3.6 name it), the manufacturers' data sheets, the CIE standards and this file.
3. **The author's choices are not copied.** The profiles do not hold coupler, grain, halation, glare or lens-blur parameters, nor the enlarger's filters (`research/film-spektrafilm-2026-11-01.md` §4). Ours are chosen from the literature and by eye, or given to us in writing (§7). Their numbers are not copied from the Python.
4. **Every module of `ym-film` starts with a header naming the papers and data sheets it is written from**, and the PR names the references read (`docs/14` rule 3).
5. **A token-shingle and structural check** against the three repositories runs before the engine merges (F-06), as the raw-algorithms precedent did (`docs/14`, the *Raw algorithms* row).
6. **The Python package may run as a separate process** on test patches, as a test oracle only (§8), like rawler (`AGENTS.md` rule 1). Its numbers are test data; no file of it ships.

### 3.2 The chain

Per photo pixel, in order. "Table" means a baked table (§3.8).

| # | Stage | Kind | Takes | Gives | Data |
|---|---|---|---|---|---|
| 1 | Scene light | per pixel | LogC4 / AWG4 | a chromaticity and a scale | CIE 1931 observer; the film's reference illuminant |
| 2 | Film exposure | per pixel, **table T1** (2D) | chromaticity, scale, exposure | raw exposure `E` (3) | the film's sensitivities and window; our spectra table |
| 3 | Scatter and halation | **spatial** | `E` | `E′` | the stock's antihalation class |
| 4 | Characteristic curves | per pixel (1D) | `E′` | density `D₁` (3) | the profile's `density_curves` |
| 5 | DIR couplers | per pixel, plus a **spatial** diffusion | `D₁`, `E′` | density `D` (3) | our coupler model |
| 6 | Grain | **spatial**, random, seeded | `D` | `D_g` | our grain model |
| 7 | Enlarger | per pixel, **table T3a** (3D) | `D_g` | print log-exposure `P` (3) | the film's dyes and base; lamp; filters; the paper's sensitivities |
| 8 | Paper curves | per pixel (1D) | `P` and the print exposure | print density `D_p` (3) | the paper's `density_curves` |
| 9 | Scan | per pixel, **table T3b′** (3D) | `D_p` | display colour, output primaries | the paper's dyes and base; viewing illuminant; CIE observer |
| 10 | Return (A) | per pixel (§3.8c) | display colour | LogC4 / AWG4 | the core table and Newton steps (`CineRendering::return_scene`), the rule of `CineRendering::inverse_continuous` |

Stages 1, 2, 4, 5 (without its diffusion), 7, 8, 9 and 10 are the per-pixel colour part. A slide film skips 7 and 8, and stage 9 reads its own dyes.

### 3.3 Spectral upsampling

A scene colour is three numbers; a film sees a spectrum. The engine turns a colour into a spectrum, projects it on the film's three sensitivities and gets the raw exposure. Three methods exist:

| Method | What it does | Fits our case? |
|---|---|---|
| **Mallett and Yuksel 2019** | any sRGB reflectance as a linear mix of three spectra | no: reflectance, sRGB gamut only; scene light from a wide-gamut camera falls outside it, and the mix goes negative |
| **Jakob and Hanika 2019** | a spectrum as a sigmoid of a quadratic in wavelength; a three-dimensional table of coefficients over RGB; zero error on sRGB, large gamuts supported | partly: a reflectance model in an RGB cube, bounded to [0, 1] |
| **A chromaticity table** (the method the profiles' `hanatos2025_*` fields are fitted for) | a table indexed by the CIE chromaticity returns a smooth emission spectrum; the spectrum is scaled by the colour's intensity | **yes** |

**Choice: the chromaticity table.** Scene light is radiance (unbounded), so an emission model is right; the table covers the whole spectral locus, not an RGB gamut, which matters for AWG4; and the response is **homogeneous of degree one in intensity**, so the film's raw exposure is `scale × T1(chromaticity)` with `T1` a small **two-dimensional** table per film (§3.8). That is why the per-pixel cost is one lookup.

**The table is ours** (F-03a). The author's table ("Hanatos 2025") has no licence (research §3.2). Our generator (`tools/film/upsampling/make_table.py`, Python, reads the CIE 1931 observer only, never a profile; under `tools/film/`, not `training/`, so the policy check of §6.3 stays simple) solves, for each cell of the chromaticity square, for the three coefficients of a sigmoid-of-a-quadratic spectrum whose chromaticity under the CIE 1931 observer is the cell's (Jakob and Hanika's form; Levenberg–Marquardt in CIELAB with an analytic Jacobian; their paper is read, their reference code is BSD-3 and may be read), then fills the cells that no such spectrum reaches by inpainting. It writes `data/film/upsampling/`, **CC BY-SA 4.0** (the table is computed from the CIE observer, so it is Adapted Material, §6.7), with the generation record. If its author grants a licence (§7), his table can be the oracle for ours.

**The luminance.** A chromaticity gives two equations for three unknowns; the paper has three (an RGB triple). The third is the luminance: each cell is fitted at a target `Y_t(x, y)` that falls smoothly from 0.2 at the white point to 0.03 at the locus (`0.2 (1 − r) + 0.03`, `r` the distance from white relative to the locus along the same ray), because a sigmoid spectrum is bounded by 1 and a saturated colour cannot be bright. A cell that does not converge at `Y_t` is retried at fractions of it. The fit fixes only the *shape* of the spectrum; the engine scales it by the colour's real luminance (below), so `Y_t` never reaches the image except through that shape. **The three numbers of `Y_t` were chosen in F-06b against the oracle's outputs** (the first release used `0.5 (1 − r)^1.5 + 0.004`): `Y_t` is the one free choice of the table, and it decides how deep the valley of a magenta or how long the tail of a green is. A scan of 72 triples (tables of 128 cells, the pooled stage-2 RMS over the seven stocks, window on, relative to the grey, with F-06b's readings of §3.7a) found a broad plateau, not a sharp minimum: 0.0444 log10 for the first release (0.0527 with the Bradford balance of F-03b), **0.0350 at 0.2, 1.0, 0.03 (the shipped)**, and every triple with 0.15 to 0.25, 0.7 to 1.4 and 0.02 to 0.05 within 0.001 of it. The gain is on the saturated colours: the 125-point lattice 0.0517 to 0.0399, the ColorChecker patches 0.0255 to 0.0262 (unchanged, so the choice does not trade the natural colours for the saturated ones). Stage 2's largest difference fell from 0.20 to 0.49 to 0.07 to 0.11 on six stocks (Vision3 500T 0.58 to 0.36, §3.7a); the slides' stage-5 maximum from 1.2 and 0.9 to 0.23 and 0.17. `tests/oracle_diag.rs::table_score` measures a table in `$YM_TABLE`.

#### The table's format (what F-03b reads)

Folder `data/film/upsampling/`. All numbers little-endian, row-major, `iy` the outer index.

| File | Content |
|---|---|
| `coeffs.f32` | float32 `[256][256][4]` = `(c0, c1, c2, Y_s)`; 1,048,576 bytes |
| `flags.u8` | uint8 `[256][256]`: **0 solved**, **1 inpainted**; 65,536 bytes |
| `error.f32` | float32 `[256][256]`: the distance in xy between the cell and the chromaticity of its *stored* (float32) spectrum; for diagnostics, the engine does not read it |
| `table.json` | the same format in words, the generator's version and SHA-256, the SHA-256 of the input observer and of the three outputs |

- **Nodes.** Cell `(ix, iy)` is the chromaticity `x = ix / 255`, `y = iy / 255`: node-centred, corners exact, so `x, y ∈ [0, 1]` and `u = 255 x`, `v = 255 y` index the grid with no half-cell shift.
- **The spectrum of a node.** 81 samples at `λ_j = 380 + 5 j` nm, `j = 0..80`; `t_j = (λ_j − 380) / 400 = j / 80`; `z_j = c0 t_j² + c1 t_j + c2`; `s_j = ½ + z_j / (2 √(1 + z_j²))`, in (0, 1). Evaluate in f64 on the CPU; the coefficients reach about 2.4·10⁵ near the locus (nearly-step spectra), so f32 is not safe for `z` there, and f32 storage is why the node error is 1e-4, not 1e-8.
- **The observer.** CIE 1931 2°, `data/film/cie/CIE_xyz_1931_2deg.csv` (1 nm, 360 to 830), decimated at the 81 wavelengths above (exact). `XYZ(s) = Σ_j s_j · cmf(λ_j) / Σ_j ȳ(λ_j)`, so a flat spectrum of 1 has XYZ = (1, 1, 1) (the three CMF sums agree to 1e-4). **`Y_s` is the Y of `s` in these units.** The range 380 to 780 drops the observer's tails (under 0.1 % of Y).
- **The scale.** For a colour with chromaticity `(x, y)` and luminance `Y_colour` (in the same units: a flat spectrum of 1 has `Y_colour` = 1), the emission spectrum is `Y_colour / Y_s · s(λ)`. Its Y is `Y_colour` and its chromaticity the cell's, to the error below. Because the film's response is linear in the spectrum, the raw exposure is `Y_colour · T1(x, y)` with `T1 = response(s / Y_s)`: **homogeneous of degree one in intensity**, exactly (the test scales by 0.01, 3 and 40 and compares to 1e-12).
- **The lookup between nodes: bilinear interpolation of the spectra, each node's `s / Y_s`, not of the coefficients.** The four neighbours' spectra, normalised to Y = 1, are mixed with the bilinear weights of `(u, v)`. Mixing is linear in the spectrum, so Y is exact and the raw exposure is simply the bilinear mix of the four nodes' raw exposures: **T1 (§3.8) is built at the 256 × 256 nodes, from each node's spectrum, and interpolated, or built at its own nodes by this lookup. No spectrum is ever built per pixel.** Mixing coefficients is wrong: the coefficients jump by up to 10⁵ between neighbours near the locus (the edges of the spectra move), and the result misses the chromaticity by up to 2.8·10⁻² (99th percentile 3.7·10⁻³) between nodes; mixing spectra misses by at most 2.7·10⁻⁴ (99th percentile 5.0·10⁻⁵, mean 1·10⁻⁵).
- **Reading the flags.** The engine compresses chromaticities outside the locus softly toward white before the lookup (below), so it should rarely land on a flag-1 node; if it does, the node's spectrum is finite, inside (0, 1), with `Y_s > 0`, and its chromaticity is wrong by the `error.f32` value (median 4.7·10⁻³ and at most 1.3·10⁻² for the 137 inpainted nodes inside the locus; large outside it). **A node with a flag-1 neighbour is not exact**: bilinear lookup across the locus edge mixes in an inpainted node.
- **Out of range.** `x`, `y` outside [0, 1], or `y ≤ 0`: clamp to the square after the compression. The row `y = 0` and the whole square outside the locus are inpainted and defined.

**Measured** (F-03a; numbers in `docs/STATUS.md`): 21,712 of the 65,536 nodes lie inside the locus; **21,613 (99.54 %) are solved and 99 (0.46 %) inside the locus are inpainted (F-06b's luminance target; the first release had 21,575 and 137)** (the thin rim at the spectral locus, the violet end and the green side, where a sigmoid cannot be nearly monochromatic enough); the other 43,824 nodes are outside the locus and inpainted. The **worst solved node misses its chromaticity by 4.0·10⁻⁶** (at x = 0.039, y = 0.329; 9.1·10⁻⁵ at the locus with the first release; a node that float32 would push past 1e-4 is given to the inpainting; the median is 2.4·10⁻⁸). The generator runs in about 75 s on one core and is deterministic: two runs give identical bytes. **Grid size:** the lookup's chromaticity error between nodes falls with the square of the cell size (worst, 99th percentile): 64 cells 2.3·10⁻³, 7.2·10⁻⁴; 128 cells 7.2·10⁻⁴, 2.0·10⁻⁴; **256 cells 2.7·10⁻⁴, 5.0·10⁻⁵ (shipped)**; 512 cells 8.5·10⁻⁵, 1.3·10⁻⁵ (4 MB, 5 minutes to generate). At the ten white points of the test (D50, D55, D65, E, blackbodies of 2700, 2856, 3200, 4000, 5000 and 6500 K) the interpolated spectrum has the right chromaticity within 1.2·10⁻⁵ and is smooth (largest second difference at 5 nm under 0.3 % of the peak; the grid-size study above was made with the first release and not repeated). 256 is enough for the 1e-3 bound on T1 (§8); a larger grid is not needed. **Smoothness of the inpainted nodes (F-04e; the owner chose "Relative check", 2026-11-04).** The inpainted nodes a lookup can reach are the corners of every bilinear cell that holds a chromaticity stage 1 can produce: the locus scaled by `LIMIT` = 0.985 about the film's white (D55, D50 and a 3200 K blackbody). They are 516 nodes with 431 neighbour pairs at the rim, where spectra are steps (an edge moves by a sample), so an absolute limit cannot hold there: the original limits (median 1e-4, 75th percentile 1e-3, 90th 0.05 of the largest change of the spectrum between neighbours) are missed by orders of magnitude (median 0.071, 75th 0.785, 90th 0.983), and by generator 1.0.0 as well (0.013, 0.789, 0.983). The test (`tools/film/upsampling/test_make_table.py`) therefore compares the inpainted nodes with the **solved** nodes in the same rim band, the reachable nodes within 3 cells of the edge of the reachable region (516 inpainted nodes, 1,823 solved), and asserts the inpainted ones are no rougher than 7 times the solved at the median, 75th and 90th percentile. The factor is the smallest whole number that generator 1.0.0 also passes (its ratios in the same band: 0.84, 6.42, 1.02); this table's are 1.02, 2.12 and 1.01, so the factor is loose at the 75th percentile and the real finding is that the inpainted nodes at the rim are about as rough as the solved ones, not smoother. Absolute figures printed by the test: inpainted at the rim 0.071 / 0.785 / 0.983, solved at the rim 0.069 / 0.370 / 0.970, inpainted outside the locus (43,407 nodes, unreachable, not asserted) 3.7e-5 / 4.1e-3 / 0.052 (median / 75th / 90th).

**Other steps:** the chromaticity of colours outside the locus (negative or extreme camera colours) is compressed softly toward the film's white; the **window** of the profile (four numbers: the centre and width of a blue edge and of a red edge, a band-pass on the sensitivities, renormalised so the reference light gives the same raw values) is applied; the **surface** correction (fifteen numbers per layer) is **off**, as it is by default in the author's tool, and stays in the data. The film is balanced for its reference light: a neutral gives equal raw values. The window was fitted by the author against his own table; ours differs a little, and the first oracle stage (§8) is where that shows. The profile stays unmodified.

### 3.4 Film: curves, couplers, grain, halation

- **Curves.** The profile's `density_curves` (256 points over log exposure −3 to 4) become one-dimensional tables, monotone cubic, 4,096 entries per channel. The largest density of a channel lies between 1.5 and 2.9 for the negatives.
- **DIR couplers.** A developer inhibitor is released in proportion to the density formed in each layer and lowers the effective exposure in its own layer and in the others, after a short diffusion in the plane (Hunt, chapter 15). The curves the engine reads are the **curves without couplers**, found from the profile's curves so that **a neutral ramp still gives the profile's curves**; saturation and local contrast are what the couplers add. The coupler matrix and diffusion are ours (§9, item 6). The diffusion is built in F-05 (§3.9a): the per-pixel solve gives the inhibitor, a Gaussian and an exponential tail spread it, and the pixel develops once more with the spread inhibitor.
- **Grain.** The standard particle model: the number of developed particles in a pixel follows the density and the pixel's area against a particle's area, so the noise in density has a variance of the binomial kind, with sublayers (the profile's `density_curves_layers` give the fast and slow parts) and a dye-cloud blur. Amplitude is calibrated to published RMS granularity (Selwyn's law). The generator is **counter-based and has no transcendental calls** (§3.9).
- **Halation.** Light that passes the emulsion reflects from the support back into it (strength set by the profile's `antihalation`: strong, weak or none) and scatters in the emulsion. A blur of a few Gaussians and a tail, in micrometres on the film, applied to the exposure (numbers and sources: §3.9a).

### 3.5 Enlarger and print

- **The lamp** is a 3400 K blackbody behind heat-absorbing glass (a formula and a published glass curve, §6.4). **The filters** are yellow, magenta and cyan dichroics given in Kodak colour-compensation units (100 units = density 1.0): an analytic model of smooth band edges, fitted to published curves (our own shape and numbers, §6.4).
- **Neutral filtering is solved, not looked up.** At bake time the engine finds the yellow and magenta (and cyan, fixed at 0) that make an 18 % grey print neutral for the film and paper pair. The node's lights are shifts from there, so **0 is neutral** and no per-pair table is needed.
- **Exposure** is normalised to mid grey, then scaled by the print exposure (stops).
- **The paper** is the print profile: sensitivities, dyes, base, curves.

### 3.6 Scan, white, brightness, return

- **Scan.** The print's spectral density under its viewing light (D50 for papers; the cine print films use a projector lamp, §6.4) is projected on the CIE 1931 observer to XYZ and adapted to the output's white (a Bradford or CAT16 transform; F-03 picks against the oracle). Glare is a small constant veil (the author's random glare field is not built).
- **White and black.** Paper white maps to display 1.0 (the scanner's white correction); the paper black stays.
- **Brightness match.** A per-paper constant offsets the default print exposure so that scene 0.18 comes out at the rendering's grey (0.111). Print exposure 0 is that placement.
- **Return (A).** The display colour goes through `CineRendering::inverse`, with the nearest-reachable rule, to LogC4 / AWG4 (§2.2).

### 3.7 The CPU reference

f64, in `ym-film`, one function per stage, each testable on its own, with named taps after stages 2, 4, 5, 8 and 9 (the oracle compares at them, §8). The spatial parts have CPU implementations that the kernels must match (§3.9). Threads split rows; the result is the same for any thread count (`docs/17` D5). No `fma`, no libm functions whose results differ between platforms where a gate depends on them (`docs/17` D6).

### 3.7a What F-03b built (the choices of the CPU reference)

`crates/ym-film`, f64, no `fma`, deterministic for any thread count. Parameter files are ours, Apache-2.0, in `data/film/params/`: `couplers.json`, `enlarger.json`, `kg3.json`. Each module's header names its references.

- **Profiles.** Read and validated (grid, shapes, ranges; `null` is NaN). NaN sensitivity is 0, **NaN dye density and NaN base density take the nearest defined sample** (F-06b: dye density was 0, which let a deep shadow leak red and blue through the ends of a paper's dyes, Portra Endura at density 2.5 giving red 1.9 times and blue 4 times the oracle's; scanning the oracle's own print densities, the 95th percentile of dE00 fell from 4.4 to 0.9 on the Kodak paper and the final p95 of the three Kodak stills from 4.9, 6.1 and 5.8 to 2.1 to 2.9). A damaged file gives a plain error that names the file and the problem; `verify_checksums` checks `SHA256SUMS`.
- **Stage 1.** AWG4 (white D65) is balanced to the film's reference light (D55, or a 3200 K black body for tungsten films) **by a diagonal scaling of the AWG4 channels (F-06b; the Bradford transform of F-03b was replaced: a camera's white balance in its own primaries)**. Against the oracle's stage 2 (relative to the grey, window on) the pooled RMS over the seven stocks is 0.0378 log10 against 0.0466 for no adaptation, 0.0646 for a spectral relighting by the ratio of the reference light to D65, and, with the first table, 0.0569 for Bradford; Vision3 500T 0.087 against 0.121. CAT02 and von Kries (HPE) on XYZ fell in between, a diagonal in Rec.709 was worse (0.12). Chromaticities are compressed along the ray from that white: untouched up to 85 % of the distance to the locus, then `k + (L - k) tanh((r - k) / (L - k))` with `L` at 98.5 %. `Y` is kept.
- **Stage 2.** The table's spectrum (four neighbours' spectra, each over `Y_s`, mixed bilinearly) meets `10^log_sensitivity` times the window. **Window reading (corrected in F-06):** `W = 1/2 erfc((c_b - l)/(w_b sqrt 2)) * 1/2 erfc((l - c_r)/(w_r sqrt 2))`, that is, each edge is a normal distribution's cumulative curve with the width as its standard deviation, renormalised per layer so the reference light gives the same raw values. F-03b divided the argument by the width alone; against the oracle's `log_e_film` (white-normalised, relative to the grey, 7 stocks, 148 points each) the mean RMS error of stage 2 falls from 0.070 to 0.048 log10, and against the real spectra of 328 reflectances (no oracle) from 0.061 to 0.052; the greens (foliage, bluish green, yellow green, green) were off by up to 0.07 in the red layer, which made them olive (Portra 400 green: dE00 3.6 from the oracle's print, hue 129 degrees against 136; now 1.5 and 133). The fit used the oracle's outputs only, and no code or number of it. A wider edge (a factor between 1.4 and 1.7) fits equally well; the square root of two is the reading with a meaning. Residual: saturated lattice points (a green layer at 0.04 of a bright red and blue) are off by up to 0.2 to 0.5 log10, the difference between our spectral table and the author's at the edge of the locus, and the tungsten films (Vision3 500T) are off by 0.12 RMS even with no window on either side. A division by the response of the table's spectrum at the reference white makes a neutral give three equal raw values. (The oracle divides by the response *without* the window, so its neutral is off by +0.005, -0.005, -0.037 log10 on Portra 400 with the window on, RMS 0.004 over seven stocks against that reading; we keep the exact balance because `tests/gates.rs` requires a neutral to give equal exposures and the published curves within 0.01, and the print's mid-grey normalisation absorbs the difference.) **Mid-grey placement (F-06b):** scene 0.18 lands at `x_mid = -0.0095` log10 on every profile (`MID_GREY_LOG10`). The oracle puts the grey there on all seven stocks (its `log_e_film` of the grey ramp is -0.0094 to -0.0096 in each layer, window off, whatever the stock); the earlier per-stock fit to `midscale_neutral_density` (0.15 for Portra 400, 0.30 for Ektar 100, -0.25 for Vision3 500T) put our stage 2 and 5 by 0.07 to 0.25 log10 off the oracle in absolute terms, and the Vision3 500T grey 0.25 off. `exposure::mid_log_exposure(profile)` keeps the old fit for diagnostics. `FilmParams.use_window` turns the window off.
- **Stage 4.** Fritsch-Carlson monotone cubic, held flat beyond the axis.
- **Stage 5.** `log E'_i = log E_i - amount * sum_j M_ij D_j` with `M` in `couplers.json` (own-layer 0.06, other layers 0.03 to 0.14, chosen by hand after Hunt ch. 15). The curves without couplers are found from `C0_i(x - sum_j M_ij C_j(x)) = C_i(x)`; the solve is a damped Newton in three unknowns. `amount = 1` reproduces the profile on a neutral ramp (worst error 0.00045 density); `0` is the steeper curves without couplers. The calibration below is done at the chain's own amount, so the grey does not move when it changes.
- **Stage 7.** 3400 K black body times KG3 times three filters in CC units (`enlarger.json`: smooth error-function edges at 500 nm for yellow, 490 and 595 nm for magenta, 595 nm for cyan). **KG3 is an analytic stand-in** (`kg3.json`, edge 700 nm): Schott's transmittance is a chart on their page and was not reachable as numbers on 2026-11-04. The paper's log exposure is `log10(P_k / P_ref) + shift + stops * log10 2`, with `P_ref` the red layer's exposure from the grey at the neutral filtering.
- **Neutral solve and brightness match.** At build time, per film and paper: Newton on yellow and magenta (cyan 0) until the 18 % grey scans with `a* = b* = 0` (CIELAB, D50, from the display colour), and bisection on `shift` until its luminance equals the cine rendering's own grey for 0.18 (0.1110). The two alternate eight times. **Deviation from section 3.6:** the constant is per film and paper pair, not per paper, because the mask and dyes move the brightness by more than the 0.002 gate. Slides: no print; a gain on the exposure does the match.
- **Stage 9.** `R = 10^-(base + sum D cd)` under the CIE D50 table, the 1931 observer, Bradford D50 to D65, the output's primaries (Rec.709 or Display P3), and the paper base (a slide: the bare light) mapped to 1 per channel. No glare. Values are not clipped here.
- **Stage 10.** `CineRendering::inverse` and LogC4.
- **Cine negatives.** Their `target_print` is a cine print film (hidden, viewed under a projector lamp we do not have). `PaperChoice::Matching` takes `kodak_portra_endura` for them; a print film chosen by id is refused (`ViewingLight`).
- **Taps.** `run_tapped` returns the values after stages 2, 4, 5, 7, 8 and 9.
- **F-06b: tuning toward the oracle's outputs (owner, 2026-11-04).** Numbers fitted or read from the oracle's outputs only, never from its code. Each stage was read on its own with the oracle's taps injected by hand (`tests/oracle/probe_*.json`, `tools/oracle/spektrafilm/run_oracle.py --probes`). **Changes:** (1) the table's luminance target (§3.3); (2) stage 1 balances by a diagonal scaling in AWG4 instead of Bradford (above); (3) `x_mid` is -0.0095 for every stock (above); (4) an undefined dye density holds its nearest value (above). Our code, Apache-2.0, except the table (CC BY-SA 4.0, the folder it already has). **Not changed, and why:** *the enlarger and the lamp*: fed the oracle's own `cmy_film`, our hand-chosen enlarger gives its `cmy_print` within 0.0035 to 0.009 density RMS (largest 0.010 to 0.034) on four of the five negatives and 0.027 (0.096) on Ektar 100; a nine-number fit (lamp temperature, glass edge and width, the yellow edge and the magenta band) on the ColorChecker patches and the grey ramp lowered the training RMS from 0.0156 to 0.0137, moved four parameters to their bounds and raised the held-out RMS on the 125-point lattice for four of five stocks (Portra 400 0.0060 to 0.0077, Pro 400H 0.0132 to 0.0196, Vision3 0.0061 to 0.0107), so no fitted file ships (`tests/oracle_fit.rs`). Cyan is the same as neutral density and the exposure shift (a scan from -40 to +80 CC moves the RMS by under 0.001). Ektar 100's red layer is 0.03 to 0.04 log10 too bright at low cyan densities, which a +0.3 density in its undefined base beyond 700 nm would explain; the cause is not found. *Stage 9*: our scan of the oracle's own print densities (a 512-point lattice and a ramp, injected at `cmy_print`) is within dE00 median 0.29 to 0.41, 95th percentile 0.81 to 0.91, largest 1.3 (a paper's blue channel leaves the gamut a little differently: -0.043 against -0.036); the slides' scan alone median 0.30, p95 0.75 to 0.80. *Stage 5 (curves)*: on the oracle's own `log_e_film` our curves give its `cmy_film` within 0.001 (negatives) and 0.002 (slides), so the slides' stage-5 maxima (1.2 and 0.9 density before, 0.23 and 0.17 after) are the steep reversal curves, up to 3 density per log exposure, magnifying stage 2's error, not the curves or the slides' scan. **Residual, stage 2:** the error grows with the colour's distance from the white to the locus (`tests/oracle_diag.rs::error_by_saturation`): within 0.03 below half the distance, 0.02 to 0.08 between 0.5 and 0.65, 0.06 to 0.2 beyond, on every stock but Vision3 500T; these are points between the spectra of our table and the author's (not used), and stage 1's compression is not involved (the points listed are inside its knee). **Vision3 500T** is still 0.08 RMS (0.05 below 0.3 of the distance, up to 0.2 at 0.65 to 0.8; the blue layer of the orange, yellow, yellow-green and purple patches is 0.15 to 0.24 log10 too bright), through every reading tried: the reference light's temperature (best 4,000 K, 0.084 against 0.087 at 3,200 K), the Bradford, CAT02, von Kries and XYZ transforms, a spectral relighting, a reflectance table (0.16), an ultraviolet or infrared cut in front of the film (worse), and the window (the error is the same with no window on either side). It is not the table's luminance target (best 0.079). The cause is not found; the other six stocks, with the same table and readings, are 0.014 to 0.021.

### 3.8 Baking

The per-pixel colour part is baked on the CPU, in f64, into tables stored as f32, whenever its inputs change. The app and `ym` evaluate the same tables, which is why their exports stay byte-identical (`docs/17` D8). The tables are in memory only (hard rule 4); the cache is an in-memory LRU.

| Table | Maps | Size (starting) | Depends on | Re-baked when |
|---|---|---|---|---|
| **T1** exposure | chromaticity square (2D) → raw exposure per unit scale (3) | 192 × 192 × 3 f32 = 432 KiB | the film, its window, the upsampling table | the stock changes |
| **Curves** | log exposure → density, per channel; the film's with and without couplers, and the paper's | 9 × 4,096 f32 = 144 KiB | the film, the paper | the stock or paper changes |
| **T3a** enlarger | negative density (3D) → print log-exposure (3) | 49³ × vec4 f32 = 1.88 MB | the film, the paper, the **lights** | the stock, the paper or the lights change |
| **T3b′** scan | print density (3D; a slide's film density) → display-linear RGB in the output's primaries, **unclamped** (3); the return to LogC4 runs per pixel after it (§3.8c) | 129³ × vec4 f32 = 34.4 MB (§3.8c; F-04 to F-04c stored the inverse in T3b: 161³, 66.8 MB) | the paper (or the slide film), the **output primaries** | the paper or the output space changes |
| **Core table** | the cine rendering's core inverse, per pixel (§3.8c): `(level, hue, fraction of the reachable chroma)` → `(ln Ys, ca, cb)`, and the reachable length of each ray | 40 × 64 × 24 × 3 f32 + 2,560 = 730 KiB | nothing but the rendering's constants (**not** the paper, the film or the output) | never; built once per process (0.3 s on one thread) |

**Why two 3D tables and not one.** Grain acts on the negative's density (between stages 5 and 7), and the paper's curves sit between the enlarger and the scan. Splitting there keeps each table smooth, **lets the print exposure and the film exposure move without a re-bake** (the film exposure multiplies T1's result, the print exposure shifts the log-exposure at stage 8), and re-bakes only T3a when the lights move. A graph with halation, grain and coupler diffusion all off could use one table, but the split costs two lookups and nothing else, so there is one path.

**Interpolation.** Tetrahedral from an f32 storage buffer, as the look LUTs (`docs/02` C10), never hardware filtering. T1 is bilinear. The axes of T3a run from the film's lowest density (0 or a little below: every profile's curves dip to −0.0005…−0.0016) to its largest (plus a small margin) in each channel, those of T3b′ the same for the paper (or the slide film), warped by squares (§3.8c); a value outside is held at the edge.

**Bake time (measured, §3.8a and §3.8c).** T3a 49³: 77 ms on one thread, 29 ms on four. T3b′ 129³: 1.2 s on one thread, 0.41 s on four (1.6 µs a node: the 81 exponentials of the scan and nothing else), built when the paper is chosen and **cached per paper and output space**; a slide stock bakes its own, 0.36 s. The core table: 0.31 s on one thread, 0.12 s on four, once per process. T1: about 20 ms. **While a lights slider is dragged** T3a is baked at 17³ (4,913 nodes, 3 ms) and refined to 49³ about 150 ms after the last change.

**Error bound against direct evaluation** (starting numbers, which F-04 measures and reports, never loosens): the table chain against the direct f64 chain on 100,000 random colours and every pixel of the test photos: at most **one 10-bit code** after C13 (largest), and 99.9 % within a quarter code; T1 within 1e-3 relative. The size is the smallest that meets it: 49 for T3a; **129 for T3b′** (33, 49 and 65 miss it, §3.8c). The nearest-reachable rule no longer touches a table: it is an exact clamp in the per-pixel return.

**Cache.** An in-memory LRU of eight entries per kind. Keys: T1 (film id, profile SHA-256, engine version), curves (stock ids, SHA-256), T3a (film, paper, lights, SHA-256, engine version), T3b′ (the paper or slide film with its SHA-256, output primaries, size, axis warp, engine version). The core table has no entry: it is a function of the cine rendering's constants, so it is one process-wide table (`CoreTable::shared()`), built on first use. Nothing on disk.

**Determinism.** The bake parallelises over nodes, each node independent, so any thread count gives the same bytes. The profile data version (§6.6) and the engine version are part of every key and of the export's metadata (`ym:FilmData`).

### 3.8a As built (F-04)

*F-04d (§3.8c) replaced T3b as described in this section and §3.8b (the inverse's coordinates in a 3D table, its warp, the row-wise continuation bake). T3b′ now holds the scan only and the return runs per pixel. What follows about T3b is the history that led there; T1, the curves, the couplers, T3a, the tetrahedral twin, the gates on the front of the chain and variant A stand.*

`crates/ym-film/src/bake.rs`. `Baker::new(spectra_id, threads)` bakes and caches (F-04e: it takes the coupler matrix from the chain it bakes, `FilmChain::couplers()`, and the amount from `FilmChain::params()`); `Baker::bake(&chain, &BakeSettings)` gives a `BakedFilm` whose `run(logc4) -> logc4` uses the tables only, plus stage 1 and the couplers. Everything below was measured on 2026-11-04 on the Linux build machine (4 cores), in release.

**What was built, and where it differs from §3.8.**

- **T1 is 196 × 220 nodes, not 192 × 192.** Its nodes sit on the upsampling table's own grid (steps of 1/255, x from 0 to 0.765, y from 0 to 0.859, which covers the spectral locus). The upsampling table mixes its nodes' spectra bilinearly and the exposure is linear in the spectrum, so a bilinear T1 on the same grid is exact: worst relative error **5.7e-8** against the direct stage 2 on 100,000 random colours (gate 1e-3). It stores the exposure for luminance 0.18 at the film's mid-grey placement; the film exposure, the luminance and `x_mid` multiply it at run time. The window reading flows in through `ExposureStage::raw`; a probe of stage 2 is part of T1's key, so a change to the reading changes the key.
- **Curves.** 9 × 4,096 f32 over the profile's own axis (the curves without couplers over an axis extended below by the largest effect of the couplers, the paper's over the paper's). Worst error 3.3e-6 density.
- **The couplers stay a 3 × 3 step.** They act on the three layers together (`log E' = log E - amount * M D`), so a table would be a third 3D table, and F-05's diffusion cannot be a table. The baked chain runs the same damped Newton on the baked curves without couplers and the matrix `M` (at most 16 iterations). Worst difference from the direct solve 2.5e-6 density.
- **T3a** is 49³ by default. Its nodes hold `log10(P / P_ref) + shift` with the neutral filtering and the lights inside; the print exposure is added at run time, so it needs no re-bake. A slide has no T3a.
- **T3b is 65³ by default and its axes are warped.** The nodes sit at `hi * (i / (n - 1))²` and the lookup takes a square root of each density (`T3B_POWER = 2`). The inverse of the cine rendering is logarithmic at the paper white (the soft clip), and a uniform axis puts 20 % of a photograph's pixels more than a quarter code off at 65³; the square root brings that to 0.6 %. The axes still run from 0 to the largest density plus a margin of 2 %.
- **Tetrahedral twin.** `Table3::sample` is the formula of `Lut3d::sample`, on f32 storage with f64 arithmetic; it gives the same numbers as `Lut3d` on the same nodes (difference 0). F-08's f32 kernel is tested against it.
- **Keys.** T1: film id and the SHA-256 of the profile's numbers, the upsampling data's id, the window switch, a probe of stage 2, the engine version. Curves: film and paper with their SHA-256, the coupler matrix. T3a: film, paper, the lights, the chain's calibration (neutral filters, shift, `P_ref`), the size. T3b: the paper (or the slide film) with its SHA-256, the output, the size. All carry `ENGINE_VERSION` (1). LRU: T1 24 entries (the picker's tiles need 20), the other kinds 8. A draft (size 17 or less) is built every time and not cached. Nothing is written to disk.
- **Deviation from F-03's text, closed in F-04e.** `Baker` once read `couplers.json` itself, so an edit of the file between building the chain and the baker could make them disagree. `FilmChain` now exposes its matrix (`couplers()`), the baker keys and bakes with it, and `tests/bake_couplers.rs` checks that a baked film and the direct chain use the same matrix and amount.

**Bake times** (Portra 400 on Portra Endura; one thread / four threads; T3b's first call also builds the inverse's own table, 0.4 to 0.7 s):

| Table | 1 thread | 4 threads |
|---|---|---|
| T1 | 22 ms | 12 ms |
| curves | 2.2 ms | 2.3 ms |
| T3a 17³ (draft) | 3.4 ms | 3.1 ms |
| T3a 33³ / 49³ / 65³ / 97³ | 26 / 84 / 193 / 698 ms | 16 / 48 / 74 / 246 ms |
| T3b 33³ / 49³ / 65³ / 97³ | 0.82 / 1.17 / 3.0 / 8.7 s | 0.19 / 0.50 / 1.2 / 4.0 s |

T3b costs about 11 µs a node (the estimate was 5 µs), mostly the inverse; T3a about 0.7 µs.

**Gate results.**

| Gate | Result |
|---|---|
| T1 within 1e-3 relative | **met**, 5.7e-8 |
| Determinism: same bytes on 1 and 8 threads and on a repeat run | **met** (every table of a negative and of a slide) |
| Variant A round trip `forward(inverse(d))` within 1e-6 | **met**: 2.4e-7 (sRGB) and 1.9e-7 (P3) on 50,000 colours the rendering makes; on the photos' print colours about 1.0e-6 (the inverse's own tolerance, after the display is held to `[D_MIN, 1]`). The inverse missed the pre-image of 0.2 % of those synthetic colours (42 of 21,000; 58 of 28,000) |
| Variant A unreachable share ≤ 0.1 % | **met: 0 pixels of 1,487,970, for each of the 8 stocks, with P3 and with sRGB output** (below) |
| Baking error, the front of the chain (T1, curves, couplers, T3a 49³, paper) | **met**: on 373,078 photo pixels the largest difference is 0.13 code and 99.9 % are within 0.10 code (Portra 400, Vision3 500T, Velvia 100) |
| **Baking error, the whole chain** | **not met** at 33, 49 or 65 |

**The baking error of the whole chain** (P3 output; the direct chain against the table chain, after the cine rendering and the 10-bit transfer, on the unrounded code values; "photos" is every third pixel of the 20 photos below, 834,750 pixels at 4× decimation; T3a is the same size as T3b except in the last column):

| Stock, photos: largest / 99.9 % / share over ¼ code | 33³ | 49³ | 65³ | T3a 49³, T3b 97³ |
|---|---|---|---|---|
| Portra 400 | 27 / 2.2 / 26 % | 7.1 / 0.94 / 4.1 % | 9.7 / 0.51 / 0.62 % | 5.8 / 0.26 / 0.11 % |
| Portra 160 | 31 / 2.5 / 24 % | 10 / 1.0 / 4.5 % | 7.9 / 0.55 / 0.65 % | 6.4 / 0.27 / 0.12 % |
| Ektar 100 | 19 / 2.3 / 27 % | 13 / 1.0 / 5.2 % | 10 / 0.59 / 1.3 % | 8.8 / 0.26 / 0.13 % |
| Gold 200 | 14 / 1.3 / 28 % | 6.0 / 0.56 / 3.7 % | 4.7 / 0.34 / 0.35 % | 5.0 / 0.15 / 0.01 % |
| Ultramax 400 | 6.5 / 1.2 / 25 % | 5.9 / 0.55 / 2.7 % | 6.0 / 0.30 / 0.25 % | 1.6 / 0.17 / 0.01 % |
| Pro 400H | 26 / 1.6 / 27 % | 15 / 1.0 / 4.0 % | 22 / 0.43 / 0.87 % | 13 / 0.23 / 0.07 % |
| Vision3 500T | 2.2 / 1.0 / 26 % | 0.91 / 0.43 / 2.6 % | 0.58 / 0.38 / 0.34 % | 0.58 / 0.30 / 0.10 % |
| Velvia 100 | 11 / 1.0 / 51 % | 7.2 / 0.80 / 7.1 % | 6.8 / 0.74 / 0.52 % | 5.8 / 0.69 / 0.36 % |

The gate asks for a largest difference of at most 1 and 99.9 % within 0.25. On 100,000 random colours (LogC4 uniform in the cube `[0, 1]³`, seed 2024) the largest difference is 450 to 650 codes and 8 to 93 % of the colours are over a quarter code, at every size. Most of that cube is not a photograph (colours far outside AWG4, prints with a display channel at 0), but the gate is written for random colours and it is not met.

**Where the error is.** Not in the front (above). T3b stores the inverse of the cine rendering, and three things in it do not interpolate: (1) the logarithm at the paper white (fixed by the warp); (2) strongly coloured prints, where a display channel is near 0 and the soft floor is logarithmic too, so a cell is wide where the density is high; (3) **a discontinuity in the pre-image the inverse picks.** For Portra 400 on P3 at densities (0.262, 0.400, 1.675) the LogC4 red jumps by 1.2e-3 and blue by 4e-3 between neighbouring densities (scene blue near 0) while `forward` gives the same display colour on both sides. Both are valid pre-images, but a value interpolated between them is not one, and the error does not shrink with the table: the largest difference is 3.0 to 3.3 codes at 81³, 97³ and 129³ (Portra 400). A continuous choice of pre-image in `CineRendering::inverse` would remove it.

**Options, for the lead and the owner** (none is applied): (a) T3b at 97³ (14.6 MB, 8.7 s on one thread, 4.0 s on four) gets 99.9 % within 0.15 to 0.30 code on most stocks, not the largest difference; (b) A+ (§2.4) is exact for the case it covers and, with the unreachable share at 0, loses nothing; (c) a continuous pre-image in the inverse, which also helps A+'s two code paths agree; (d) the bound stated for photographs (the 8 stocks × 20 photos) rather than for random colours, which still needs (c) or (a) for the largest difference.

**Variant A, measured.** 20 public photos, each decoded by `ym render --look neutral --output-sharpening 0 --format png --bits 16 --color-space srgb --long-edge 1000` (so the Output's cine rendering is the identity on them and the inverse gives back the LogC4 the Input and Main nodes made), every third pixel (74,000 a photo). Sources: the owner's seven A1 photos (`testdata/readme-photos/src`: A1M01454, A1M01739, A1M01749, A1M03159, A1M03165, A1M03175, A1M03185; own photos, `docs/test-photo-sources.md`); the nine files of `testdata/real/files` (DSLR-A300, DSLR-A500, ILCE-5000, ILCE-6300 cRAW, ILCE-7M2, ILCE-7M3, ILCE-7M4, NEX-5N, NEX-7: licence unclear, development machine only, no gate depends on them, results reported as local); four Play Raw files of `testdata/lens-photos` (16238_DSC00416, 28322_DSC09944, 43248_DSC01025, 55415_DSC07727: CC licences per post, `docs/test-photo-sources.md` tier 2). The PNGs are in `testdata/film-bake/` (git-ignored); the tests skip when the folder is missing. Through the 8 stocks (Portra 400 and 160, Ektar 100, Gold 200, Ultramax 400, Pro 400H, Vision3 500T, Velvia 100), with the scan evaluated exactly: **no print pixel is unreachable** (0 of 1,487,970 for each stock, for the P3 and the sRGB output). The ΔE00 of the unreachable pixels is therefore empty. Limits: ordinary photographs; a saturated subject (a Velvia print of a red flower) could still produce some. A+ is not needed on this evidence.

Tests: `crates/ym-film/tests/bake_tables.rs` (T1, curves, couplers, twin, determinism, cache, exposures, lights, slide, front of the chain), `bake_variant_a.rs`, `bake_report.rs` (the ignored numbers above and the unmet gate: `cargo test -p ym-film --release --test bake_report -- --ignored --nocapture`).

**F-04b: T3b stores coordinates, and the choice of pre-image is continuous.** Re-measured 2026-11-04 (4 cores, release, P3). The finding of cause 3 above is corrected: for the photographs there is no jump between sheets (the plain and the continuous inverse agree on all 209,750 photo pixels tried). What defeats the table is (a) the *scene side*: a saturated cyan's AWG4 red is a difference of large numbers (Z of 8 against a red of 0.01), so LogC4 red moves 0.25 across one cell while the display moves 2 %; (b) the cube-root cusp where the pre-gamut display Z (or X) crosses 0 (a display blue near 0.08 of Y2 times the floor `kl/e`; photo pixels with blue 0.02 to 0.025), where the scene Z is `(a c - b)^3` with `c` the cube root of a function linear in the density; (c) the edge of the reachable set (the nearest-reachable rule has a kink there and the pre-image a square-root fold). T3b therefore stores `(ln Ys, ca, cb)` of `CineRendering::inverse_coordinates` (smooth where the scene colour is not) and `Table3::sample` finishes in closed form (`scene_from_coordinates`, then LogC4); `Table3::sample_stored` is the plain interpolation. The chain's own return stage uses `inverse_continuous`, so table and reference follow one rule (`docs/02` C12, *Continuous pre-image*). The axis warp uses `u * u` and `sqrt`.

*Per-pixel cost of the finish, for F-08:* one `atanh` (only above the limiter's knee), two inversions of the smooth cube root (a Newton iteration on `r^3 - P r^2 - P d^2`, monotone from `P + (P d^2)^(1/3)`; 6 iterations suffice to f64 1e-15, fewer in f32), one Newton iteration for the luminance floor (from `Y = Ys`; about 5 iterations; each has 1 `sqrt` for the floor and 1 for the smooth maximum), 1 `exp`, a 3x3 matrix and LogC4's `log`. About 3 `sqrt`/`exp`/`log` plus 11 to 12 Newton steps with 2 `sqrt` each; branches: the knee (`limited > CHI_K`) and the clamps (`min`/`max`), no `atan2`, no `pow` with a negative base (`cbrt` of a signed value is done by the Newton iteration on `|q|^3` and `copysign`). Nothing depends on a comparison that rounding could flip except the knee, which is continuous there. Bake: the continuation costs about 20 us a node (T3b 97³: 5 to 8 s on 4 threads, 16 s on one).

**Baking error, whole chain, after F-04b** (P3, photos: every 6th pixel of the 20 photos, 373,078; largest / 99.9 % within / share over ¼ code), T3a 49³:

| Stock | T3b 65³ | T3b 81³ | T3b 97³ |
|---|---|---|---|
| Portra 400 | 1.59 / 0.296 / 0.27 % | 1.34 / 0.192 / 0.03 % | **0.60 / 0.147 / 0.008 %** |
| Portra 160 | 2.05 / 0.300 / 0.29 % | 1.61 / 0.202 / 0.04 % | 1.25 / 0.153 / 0.02 % |
| Ektar 100 | 1.42 / 0.425 / 0.92 % | 1.63 / 0.281 / 0.17 % | **0.99 / 0.190 / 0.009 %** |
| Gold 200 | 0.97 / 0.281 / 0.19 % | **0.58 / 0.182 / 0.02 %** | **0.36 / 0.135 / 0.003 %** |
| Ultramax 400 | 0.62 / 0.264 / 0.14 % | **0.43 / 0.179 / 0.01 %** | **0.23 / 0.135 / 0 %** |
| Pro 400H | 3.68 / 0.420 / 0.63 % | 2.66 / 0.330 / 0.24 % | 2.47 / 0.221 / 0.06 % |
| Vision3 500T | 0.58 / 0.401 / 0.25 % | 0.60 / 0.362 / 0.11 % | 0.59 / 0.361 / 0.11 % |
| Velvia 100 | 2.74 / 0.730 / 0.50 % | 3.01 / 0.698 / 0.41 % | 1.51 / 0.673 / 0.36 % |

(Before F-04b, 65³: largest 4.7 to 22, 99.9 % 0.30 to 0.74.) Met on the photos: Gold 200 and Ultramax 400 from 81³; Portra 400 and Ektar 100 at 97³. **Not met:** Portra 160 (largest 1.25), Pro 400H (2.47), Vision3 500T and Velvia 100 (99.9 % within 0.36 and 0.67 at every size: this is the front of the chain, not T3b; Velvia has no T3a, so it is the curves, the couplers or T1's slide path, F-03b/F-06 territory). The default was 97³ (14.6 MB). **Corrected by F-04c (§3.8b):** the floor of Vision3 500T and Velvia 100 was the axes starting at 0, not the front of the chain; the default is now 161³, which meets the gate on the photos for P3.

*Random colours (100,000, LogC4 uniform in `[0, 1]^3`, seed 2024; Portra 400, 65³; 97³ in brackets):* whole chain largest 245 (246), 99.9 % within 103 (101), 16.5 % (12.8 %) over ¼ code. By stage: the front of the chain (T1, curves, couplers, T3a, paper) is within 0.18 code at 99.9 % and 0.37 at most on colours the rendering reaches; it reaches 142 codes only on the 5.8 % of the cube that has no pre-image (the nearest-reachable projection is discontinuous where a ray has gaps, so two colours 1e-4 apart can land far apart). T3b: unreachable colours 5.8 % (99.9 % within 180 codes), colours with a display channel held at 1e-6 or less 1.7 % (39), reachable others 92.5 % (5.5; largest 157, at display channels of 0.2 to 12 % of Y2 or less, the log floor and the fold). A colour a photograph can hold (inside AWG4, every channel at least 0, luminance from 0.18/512 to 0.18 x 32) is 22.4 % of the cube; on it the error is no better (99.9 % within 123 codes) because a uniform cube is far more saturated than a photograph: the part with a reachable print colour, 18.7 % of the cube, has front error 0.14 (99.9 %) and T3b error 27.8 (99.9 %). The gate as written for random colours is not met and a table in the print's density cannot meet it: the inverse's output near a fold, the reachable edge and a display channel at the floor is a function with a square-root or log singularity, so the error shrinks like the square root of the cell. **What would close it:** (1) an exact evaluation (the continuous inverse, 20 us) for the pixels whose print colour has a channel below about 12 % of Y2 or lies within a few percent of the reachable edge, chosen by a flag from a coarse table; F-08 would need that kernel (about 1,000 f32 operations, iterative); (2) or a bound stated for the colours a photograph can hold in a print. Neither is applied.

### 3.8b F-04c: the gate on photos, the floor of Vision3 and Velvia, the bake time

*Superseded in its conclusion by F-04d (§3.8c): the sRGB half of the gate, which no T3b size met here, is met by the per-pixel return. The axes starting at each profile's lowest density (cause 1) stand.*

Measured 2026-11-04 on the Linux build machine (4 vCPUs), release. **Decision of the owner (2026-11-04, "Gate photos, report random"):** the baking-error gate is *largest ≤ 1 code and 99.9 % within ¼ code on the pixels of the 20 test photos × the 8 stocks (Display P3 and sRGB)*; random colours are measured and reported, not gated. The gate is the test `baking_error_on_photos` in `crates/ym-film/tests/bake_report.rs` (every 6th pixel of the photos in each direction, 373,078 pixels a stock; `YM_BAKE_STRIDE=1` and the ignored `baking_error_every_pixel` take all 13 million; it skips when `testdata/film-bake/` is missing, and fails when a stock misses).

**Re-run all the numbers with one command:** `cargo test -p ym-film --release --test bake_report -- --ignored --nocapture --test-threads 1` (`YM_BAKE_STOCKS`, `YM_BAKE_SIZES`, `YM_BAKE_OUTPUTS=p3,srgb`, `YM_BAKE_STRIDE` narrow it; `YM_BAKE_THREADS=1` or `4` for `bake_times`, which needs separate runs). The causes below were found with `crates/ym-film/tests/bake_diagnose.rs` (stage by stage, the worst pixels, the fast inverse against the continuation on every node of the lattice).

**Cause 1, the floor of Vision3 500T and Velvia 100: the axes started at 0.** Every profile's density curves dip a little below 0 (the base subtracted: Velvia −0.0016, the other slides −0.0016, the papers −0.0008 to −0.001, the negatives −0.0001 to −0.0007), and T3a and T3b held those densities, which are the highlights, at 0. A clear-film highlight then read slightly too dark by 0.3 to 1.5 codes, at every table size, which is why the 99.9 % figure did not move with T3b. Stage by stage on Velvia 100 the front of the chain (T1, curves, couplers, exact scan and return) is within 0.0014 code of the direct chain, so the whole error was T3b. The axes now start at the profile's lowest density (`Table3::axis_min`); `ENGINE_VERSION` is 3. 97³ on P3, photos, before/after: Vision3 500T 0.59 / 0.361 → 0.31 / 0.175; Velvia 100 1.51 / 0.673 → 1.62 / 0.105 (largest / 99.9 %).

**Cause 2, the largest differences (Portra 160, Pro 400H, Velvia 100): features the cell cannot hold.** The worst pixels of Velvia and Pro 400H are dark blues whose display blue is 0.5 to 2 % of the luminance (the pre-gamut display Z crosses 0 at about 7 % of Y2 because the gamut stage's floor `kl (1 + ln (s / kl))` is negative below `kl / e`). There the pre-image's `cb` is `1 − G(cbrt(W))` with `W` linear in the density: along a line of densities `cb` rises by 0.13 within 0.0004 of density (cell width at 97³ and a blue density of 2.2: 0.05). Portra 160's worst pixel is a light cyan (display 0.63, 0.76, 0.87) at a limiter-knee chroma of 1.5, where `ca` bends sharply (a feature 0.01 wide in density against a cell of 0.006). The error of such a feature shrinks slowly with the cell (about its square root: Velvia 100 at 97 / 113 / 129 / 145 / 161: 1.62 / 1.61 / 1.16 / 1.13 / 0.92; Pro 400H: 2.49 / 1.79 / 1.54 / 1.22 / 0.91). Tried and dropped: axis warp 1.5 (99.9 % within 0.36 to 0.59 against 0.1 to 0.24 at 2), 1.6 on blue alone (Portra 400's 99.9 % 0.31); storing the offset of `(ca, cb)` from the display-side coordinates together with the cubes `(1 + ca2)³`, `(1 − cb2)³` (smooth at the cusp; larger errors elsewhere, largest 3.6 and 6.0, because the cube of an exponential in the density interpolates worse than its cube root). **The size is the fix: 161³ meets the gate on P3 for all eight stocks** (table below); 129³ misses on Velvia (1.16), Pro 400H (1.54) and Portra 160 (1.01).

| P3, photos; largest / 99.9 % within | 97³ | 113³ | 129³ | 145³ | 161³ |
|---|---|---|---|---|---|
| Portra 400 | 0.68 / 0.160 | 0.49 / 0.119 | 0.36 / 0.106 | 0.31 / 0.101 | **0.22 / 0.100** |
| Portra 160 | 1.51 / 0.152 | 0.72 / 0.129 | 1.01 / 0.111 | 0.80 / 0.099 | **0.42 / 0.094** |
| Ektar 100 | 0.93 / 0.274 | 0.63 / 0.148 | 0.39 / 0.130 | 1.12 / 0.118 | **0.14 / 0.095** |
| Gold 200 | 0.37 / 0.137 | 0.28 / 0.122 | 0.21 / 0.090 | 0.17 / 0.067 | **0.13 / 0.062** |
| Ultramax 400 | 0.26 / 0.135 | 0.27 / 0.109 | 0.21 / 0.088 | 0.15 / 0.086 | **0.17 / 0.087** |
| Pro 400H | 2.49 / 0.242 | 1.79 / 0.165 | 1.54 / 0.138 | 1.22 / 0.108 | **0.91 / 0.097** |
| Vision3 500T | 0.31 / 0.175 | 0.24 / 0.145 | 0.23 / 0.130 | 0.16 / 0.122 | **0.13 / 0.116** |
| Velvia 100 | 1.62 / 0.105 | 1.61 / 0.077 | 1.16 / 0.112 | 1.13 / 0.147 | **0.92 / 0.120** |

The default T3b size is therefore **161³ (66.8 MB; about 9 s to bake on four threads)**. The lead and the owner may prefer 129³ (35 MB, 4.7 s), which holds 99.9 % within ¼ code for all eight but misses the largest-difference bound on three stocks by up to 0.5 code.

**sRGB is not met, at any size.** At 161³, sRGB output, photos, largest / 99.9 % within: Portra 400 11.6 / 0.53, Portra 160 7.0 / 1.13, Ektar 100 22.8 / 4.7, Gold 200 9.5 / 0.24, Ultramax 400 10.1 / 0.36, Pro 400H 13.2 / 2.7, Vision3 500T 3.2 / 0.13, Velvia 100 12.6 / 0.31. The photos' saturated print colours leave the smaller sRGB gamut: a display channel passes through 0 (the worst pixels have blue −0.0003 to +0.0004 against a red of 0.4 to 0.5), where the direct chain holds it at `D_MIN` and the pre-image has a logarithmic singularity at the crossing and a kink at the hold; the error shows in the other channels (a red 11.6 codes off). The cell cannot hold it, and a larger table does not help much (it is the same cusp as Cause 2, at the gamut edge, for 0.1 to 1.3 % of the pixels). **This is reported, not fixed.** Options for the owner: (a) the gate on the output the viewport uses (P3) with sRGB reported; (b) a smooth hold in place of the clamp at `D_MIN` in both the chain's return stage and T3b's nodes (changes the out-of-gamut channel by a few codes, in both); (c) an exact per-pixel evaluation, flagged from a coarse table, for print colours with a channel below about 5 % of the luminance (F-08 would need the iterative kernel).

**Random colours (reported, not gated).** 100,000 LogC4 colours uniform in `[0, 1]³`, seed 2024, Portra 400, P3, 161³: whole chain largest 186 codes, 99.9 % within 23. By class of the print colour (`random_colours_by_class`): unreachable by the continuous inverse 5.8 % (largest 186, 99.9 % within 160); a display channel at the floor (1e-6 or less) 1.7 % (51, 26); reachable others 92.5 % (107, 1.45). The front of the chain (T1, curves, couplers, T3a, paper) with an exact scan and return: 0.374 largest and 0.114 at 99.9 % on the reachable colours, 142 codes largest over all (the unreachable 5.8 %, where the nearest-reachable projection is discontinuous). The size cuts the 92.5 % class from 5.5 (65³) to 1.45 at 99.9 %; the rest is the singularities above.

**Cause 3, the bake time.** T3b at 97³ took 7.6 s on four threads. Three changes, all in `bake.rs` and the continuous inverse: (1) **a work queue**: the contiguous equal runs of nodes left most threads idle, because a node costs 1 µs in the reachable part of the table and 50 µs where the colour is beyond the reachable set (the thread that got the high-density slab ran alone); the threads now take runs of 256 nodes (T1, T3a, curves) or one row of nodes (T3b) from a shared queue, which also scales 1 to 4 threads by about 4. (2) **`inverse_coordinates` has a fast path**: the chord iteration from a first-use table of the *connected sheet alone* (built by continuation along rays, a step refused when the solution jumps; `Table` in `inverse.rs`), accepted when it converges in 6 steps within 0.005 of its start, otherwise the continuation. (3) **the row march**: a T3b node starts from the prediction of the two before it in the row (`inverse_coordinates_from`), accepted when it converges within 0.004 of that start and within 0.03 of the connected-sheet table, otherwise the fast path. Bytes: the result is a function of the display colour (and, in a row, of the nodes before it), never of the thread count. It is **not** the byte-for-byte output of the continuation: against a table made by the continuation at every node (`t3b_against_the_reference_table`, 97³, the eight stocks, P3): at most 1.1e-5 apart on all stored values for seven of them (the papers of the six negatives share one table) and Velvia 100, and for Pro 400H's paper at most 3 stored values (1 node of 912,673) that differ, by up to 0.46 (Pro 400H's paper: the colour is reachable at once on a sheet next to the connected one while the ray from the neutral colour stops at 0.75 of its chroma; the answer is a better pre-image, the table is not made worse). On the colours of photographs the unreachable share stays 0 (below).

| | 1 thread | 4 threads |
|---|---|---|
| T3b 33³ / 49³ / 65³ / 97³ / 113³ / 129³ | 0.39 / 1.13 / 2.5 / 9.0 / 12.2 / 18.8 s | 0.10 / 0.29 / 0.63 / 2.1 / 3.3 / 4.7 s |
| T3b 161³ (extrapolated from 129³) | about 37 s | about 9 s |
| T1 / curves / T3a 49³ / T3a 97³ | 20 / 2.1 / 79 / 589 ms | 5.7 / 3.2 / 25 / 175 ms |

(Separate runs of `bake_times`; the first use of the inverse builds its table of the connected sheet, 0.4 to 0.7 s on four threads, once per process.) The 2 s target is met by 97³ to within a tenth (2.1 s, four threads) and not for the default 161³: the cost is about 9 µs a node and thread on average (1 µs for a reachable node, 15 to 50 µs for the 11 % of nodes beyond the reachable set), and 4.2 million nodes cost 9 s. What would cut it: a coarser first bake at 97³ shown at once while 161³ bakes in the background (the draft mechanism of T3a), or a cheaper nearest-reachable solve for the unreachable nodes.

**Variant A with the continuous inverse** (`unreachable_share_on_photos`, 20 photos, every 3rd pixel, 1,487,970 pixels × 8 stocks × P3 and sRGB): **0 unreachable** for every stock and both outputs; round trip on the rest 4.5e-7 to 8.0e-7 (P3) and 5.7e-7 to 1.0e-6 (sRGB), against 1e-6; on synthetic colours the rendering makes: 2.4e-7 (sRGB) and 1.9e-7 (P3) for both the plain and the continuous inverse. Met.


### 3.8c F-04d: the return per pixel; T3b′ holds the scan only

Measured 2026-11-04 on the Linux build machine (4 vCPUs), release. **The lead's decision:** F-04 to F-04c stored the inverse of the cine rendering in T3b (print density → LogC4). That met the photo gate for Display P3 at 161³ (67 MB, 9 s per paper and output on four threads) and could not meet it for sRGB at any size (§3.8b: saturated prints leave sRGB, the clamped channel's pre-image is logarithmic at the crossing); a slide stock would need a T3b of its own too, so the stock picker's tiles would wait minutes. Variant A, the graph and the gates do not change.

**The split.**
- **T3b′ holds the scan only:** print density (a slide's film density) → display-linear RGB in the output's primaries, **unclamped** (a saturated colour may be negative in a channel). It is a sum of exponentials of the density and a 3 × 3 matrix, so its bake needs no inverse (1.6 µs a node).
- **The return runs per pixel**, in the inverse's own coordinates (`docs/02` C12 *Per-pixel return*): hold the display to `[D_MIN, 1]` as the direct chain does, step 1 in closed form (`CineRendering::display_coordinates`: the ceiling, the shared luminance `Y2` by Newton, `(ln Y2, ca2, cb2)`), the core inverse from the **core table** and five Newton steps, the nearest-reachable clamp, then the scene side in closed form (`scene_from_coordinates`) and LogC4. `BakedFilm::return_to_logc4` is this; `BakedFilm::run` and `colour_out` use it.
- **The core table** is a function of the cine rendering's constants alone. The brief assumed it depends on the output; it does not (the output primaries enter in step 1, which is per pixel), so there is **one table per process**, not one per output, and it needs no cache key (`CoreTable::shared()`, 730 KiB, 0.31 s on one thread, 0.12 s on four).
- A paper change re-bakes T3b′ only; a slide stock bakes its own T3b′ (0.36 s on four threads); a film or exposure change bakes none. The print exposure still needs no re-bake. `ENGINE_VERSION` is 4; T3b′'s cache key is the paper or slide film with its SHA-256, the output, the size, the axis warp and the engine version.

**The core inverse, two ways (measured, `crates/ym-film/tests/bake_core.rs`).** The display colours of the print for every 12th pixel of the 20 photos (94,080 a stock), the exact scan; the error is the code difference after the Output's rendering and C13 between the direct chain's return (`inverse_continuous`) and the per-pixel return (largest / 99.9 % within), Portra 400, Pro 400H, Vision3 500T and Velvia 100, Display P3 and sRGB.

| Core inverse | Table | Largest / 99.9 % within (codes) | Per pixel |
|---|---|---|---|
| (a) table alone, no iteration | 40 × 64 × 24, 730 KiB | 28 to 38 / 22 to 30 | table lookup |
| (a) | 96 × 128 × 48, 7.0 MB | 8 to 16 / 5 to 13 | |
| (a) | 192 × 256 × 64, 37 MB | 5 to 11 / 3 to 8 | |
| (b) coarse table + 1 Newton step | 40 × 64 × 24, 730 KiB | 4 to 13 / 2 to 6 | |
| (b) + 2 steps | same | 0.06 to 2.1 / 0.01 to 0.74 | |
| (b) + 3 steps | same | 0.00 to 0.08 / 0.00 to 0.015 | |
| **(b) + 4 or 5 steps** | **same** | **0.000 / 0.000** | **1.7 to 2.2 µs** |

**(b) is chosen: the 730 KiB table and a fixed five Newton steps.** (a) fails the gate by two orders of magnitude at every size that is sane (the pre-image has a square-root fold at the reachable boundary and, in the highlights, a chroma that grows a hundredfold while the display chroma grows a tenth: no table size keeps a code). Five steps rather than four because the round trip of variant A (1e-6) is then 2.7e-11 on 50,000 colours the rendering makes; four steps give 5.6e-7 and three 1.4e-6.

*How the table was made accurate (what failed first).* (1) Rays are marched from the neutral colour in steps that start at 0.001 and grow; a first step of 0.04 landed on another sheet for dark saturated greens (the jump bound is loose at the first steps). (2) The nodes sit at fractions `u` of each ray's reachable length (the table covers exactly the reachable set; `u = 1` is the nearest-reachable answer), dense towards `u = 1`. (3) The start reads each of the four surrounding rays at the colour's *own* chroma and blends the four, instead of interpolating nodes at equal `u`: the reachable length falls by a factor of 3 between two rows in the mid-tones, and the blend of equal-`u` nodes mixed pre-images of different chromas (a start 0.1 to 0.3 off in the coordinates, a Newton iteration that never recovered: 2 to 9 of 3,891 colours stayed above 1e-6 even after 12 steps; now none after 8). (4) `ln ρ_max` is interpolated, not `ρ_max`. (5) Levenberg–Marquardt candidates after the Newton step.

**Per-pixel cost, and what F-08's kernel runs** (one CPU thread; Portra 400 on Portra Endura, 23,520 photo pixels): the whole baked chain 2.7 µs a pixel (it was 0.9 with the inverse in T3b); the return alone 1.66 µs; stage 1 and 2 0.37 µs. F-05's 24 MP timing (35 s on one thread, 15 s on four) therefore gains about 40 s on one thread (10 s on four). The kernel, in order:
1. T3b′: a tetrahedral lookup in an f32 storage buffer (the warped axes are `sqrt`), as T3a.
2. Step 1: hold to `[1e-6, 1]` (NaN counts as `D_MIN`), the ceiling inverse (`sqrt`), the luminance sum, then **Newton for `t = ln Y2`** (the existing code loops up to 40 times with an exit on 1e-14; a kernel runs a fixed count that F-08 sets against this code, the loop being convex and monotone after its first step, 3 `exp` and 3 `ln` each), `cbrt` for `ca2`, `cb2`.
3. The start: `s = ln Tinv(Y2)` (closed form, `tone_inverse`), the cell: `atan2` for the hue column (only the cell and the weights depend on it; the table is periodic and continuous across ±π, and an `atan2` approximation of 1e-4 moves the start by 1e-4 and nothing else), `sqrt` for the crowded level row, 4 rays × (`exp` of `ln ρ_max`, a division, `sqrt`, 2 node reads of 12 bytes) and one `exp` for the blended length. 4 × 2 reads of the 24-node rays plus 4 reads of `ln ρ_max`.
4. Five Newton steps, each: one dual-number evaluation of the core (about 30 transcendental calls, derivative arithmetic about 4 times that of the plain core), a 3 × 3 inverse (with a determinant floor), up to 2 more evaluations of the plain core for the step halving, and, if none lowers the residual, the least-squares candidates (three more 3 × 3 inverses and core evaluations). A kernel evaluates the candidates it needs in a fixed order and selects with `select`; **no loop count depends on the data**. Worst case about 5 × (30 + 5 × 30) transcendental calls; typically 2 to 3 steps move the point and the rest find the residual below 1e-13.
5. The clamp: `ρ2 > ρ_max` selects the start with the four rays held at their ends instead of the iteration.
6. The scene side: as in §3.8a (one `atanh` above the knee, two cube-root Newton iterations of 6, the luminance floor's Newton of about 5, one `exp`, the matrix), then LogC4's `log`.
Metal's fast maths (`docs/17` K8): `atan2` (above) and `sqrt` of values clamped to `[0, 1]`; every `pow` or `exp` argument is positive or finite; every division is by `exp(·)`, a determinant floored at 1e-300, or a sum of squares plus a floor; the comparisons that select a Newton candidate (`cost < previous`) compare two residuals that both lowered, and when they tie, the points are within the iteration's tolerance of each other; the comparison `ρ2 > ρ_max` is continuous (both sides give the pre-image of the boundary at equality, to the table's accuracy of the boundary); nothing relies on NaN or infinity (a non-finite value is held first).

**T3b′: size, axes and storage (measured).** The brief expected 33³, 49³ or 65³ to keep T3b′'s own error far below a code. They do not hold display-linear values: the scan is a sum of exponentials, steepest where a code is widest in light, and in the saturated cyans of Ektar and Portra the red channel is a difference of large terms (0.003 to 0.006 against green and blue near 0.2 to 0.5), so an absolute error of 3e-5 is a third of a code. T3b′'s own error (the held display colour of the table against the exact scan, in codes after C13; photos, every 12th pixel; the worst over eight stocks and both outputs, largest / 99.9 % within):

| Axis warp | 97³ | 129³ | 145³ |
|---|---|---|---|
| uniform (power 1) | 0.50 / 0.48 | 0.28 / 0.27 | 0.22 / 0.21 |
| **squares (power 2)** | 0.65 / 0.38 | **0.42 / 0.24** | 0.32 / 0.16 |

At 33³, 49³ and 65³ (three stocks) the largest is 1.4 to 6.4, 0.5 to 2.0 and 0.3 to 1.6 codes. Powers between 1 and 2 are worse than either (a warp `d = w^p` has `d'' ∝ w^(p-2)`, which is infinite at the clear-film end for `p < 2`, and the error of the interpolation contains `f'·d''`); 3 and 4 are worse as well. Tried and dropped, because none brought a size below 97³ to the gate: storing the sRGB-encoded or an `asinh` value (much worse: the dark channels, a difference of large terms, become steep), `ln XYZ` with the matrix after the lookup (1.5 times better than linear at 65³, not enough), the ratio to a fitted exponential of the density (the bulk improves, the largest does not), nodes uniform in `10^(-κ d)`. **The whole chain decides** (T3b′ and the front together, every 12th pixel, 99.9 % within, quarter-code gate): 97³ (14.6 MB) misses Portra 160, Ektar 100 and Pro 400H on sRGB (0.258, 0.344, 0.277); 113³ misses Ektar 100 on sRGB (0.288); **129³ meets the gate for all 16 stock × output pairs** (§8). The default is **129³ with squared axes, 34.4 MB**. This is a deviation from "33³, 49³ or 65³" that the lead should know about; the cost is 1.2 s on one thread and 0.41 s on four, not the 9 s of 161³.

**Gates** (decision of the owner, 2026-11-04: "Gate photos, report random").

**Baking error on the photos** (`baking_error_on_photos`: the table chain against the direct f64 chain, after the Output's cine rendering and C13, every 6th pixel of the 20 photos in each direction, 373,078 pixels a stock; T3b′ 129³ with squared axes, T3a 49³, the per-pixel return; largest / 99.9 % within / share over a quarter code, in 10-bit codes; gate: largest ≤ 1 and 99.9 % within ≤ 0.25):

| Stock | Display P3 | sRGB |
|---|---|---|
| Portra 400 | 0.111 / 0.087 / 0 % | 0.238 / 0.131 / 0 % |
| Portra 160 | 0.101 / 0.081 / 0 % | 0.236 / 0.127 / 0 % |
| Ektar 100 | 0.141 / 0.078 / 0 % | 0.342 / 0.198 / 0.025 % |
| Gold 200 | 0.068 / 0.055 / 0 % | 0.221 / 0.086 / 0 % |
| Ultramax 400 | 0.100 / 0.075 / 0 % | 0.238 / 0.100 / 0 % |
| Pro 400H | 0.098 / 0.060 / 0 % | 0.235 / 0.134 / 0 % |
| Vision3 500T | 0.076 / 0.051 / 0 % | 0.196 / 0.073 / 0 % |
| Velvia 100 | 0.087 / 0.075 / 0 % | 0.183 / 0.083 / 0 % |

**Met for all 16 stock × output pairs, sRGB included** (F-04c: 3.2 to 23 codes). The tightest is Ektar 100 on sRGB at 0.198 of the 0.25 allowed for 99.9 %. **Every pixel holds too** (`baking_error_every_pixel`, all 13,329,000 pixels of the 20 photos a stock): Display P3 largest 0.082 to 0.140, 99.9 % within 0.051 to 0.087, none over a quarter code; sRGB largest 0.219 to 0.333, 99.9 % within 0.074 to 0.191, at most 0.019 % over a quarter code (Ektar 100). The per-pixel return alone adds 0.000 code (table above); the error is T3b′'s interpolation and the front of the chain (T1, curves, couplers, T3a, paper: 0.13 code, §3.8a).

**Variant A** (`bake_variant_a.rs`): the round trip of the per-pixel return on the 50,000 colours the rendering makes (P3 and sRGB) is 2.7e-11 and 2.9e-11 (the plain and the continuous inverse: 1.9e-7 and 2.4e-7; gate 1e-6); on the photos' print colours (20 photos, every 3rd pixel, 1,487,970 pixels × 8 stocks × both outputs) **no pixel is unreachable** and the round trip is 1.1e-11 at worst (the continuous inverse alone: 5.5e-7 to 1.1e-6). 72 of 28,024 colours (P3) and 54 of 21,307 (sRGB) are reported as not reached, 14 and 12 of which the continuous inverse reaches (at the edge of the set, within 0.2 % of `ρ_max`, below the 0.1 % bound).

**Random colours (reported, not gated)**: 100,000 LogC4 colours uniform in `[0, 1]³`, seed 2024, T3b′ 129³ (`random_colours_by_class`); the classes are those of the direct chain's print colour: unreachable by the continuous inverse, a display channel at the floor (1e-6 or less), reachable others. Largest / 99.9 % within, in codes:

| Stock, output | all | unreachable | channel held | reachable others | front only (exact scan, direct return) |
|---|---|---|---|---|---|
| Portra 400, P3 | 255 / 126 | 6.7 %: 255 / 167 | 2.4 %: 39 / 39 | 90.9 %: 235 / 0.117 | 31 / 0.185 |
| Ektar 100, P3 | 235 / 152 | 8.9 %: 232 / 214 | 3.2 %: 48 / 38 | 87.9 %: 235 / 0.128 | 172 / 0.128 |
| Velvia 100, P3 | 71 / 10.8 | 1.8 %: 71 / 70 | 4.1 %: 16 / 0.083 | 94.0 %: 48 / 0.186 | 0.004 / 0.001 |
| Portra 400, sRGB | 194 / 128 | 8.5 %: 194 / 146 | 8.4 %: 67 / 45 | 83.1 %: 98 / 0.203 | 10 / 0.164 |
| Ektar 100, sRGB | 219 / 133 | 12.3 %: 219 / 145 | 11.5 %: 65 / 53 | 76.2 %: 80 / 0.205 | 2.2 / 0.133 |
| Velvia 100, sRGB | 116 / 21 | 2.0 %: 116 / 116 | 9.0 %: 33 / 9.4 | 89.0 %: 91 / 0.310 | 0.004 / 0.001 |

F-04c, Portra 400 on P3 at 161³: all 186 / 23, unreachable 5.8 % 186 / 160, held 1.7 % 51 / 26, reachable others 92.5 % 107 / 1.45. **The reachable colours are about ten times closer at 99.9 % (0.117 against 1.45); the whole-chain 99.9 % figure is higher (126 against 23) because the unreachable colours and those with a channel at the floor, 9 to 24 % of this cube, are now farther from the direct chain's answer.** The cause is not the table: the direct chain's nearest-reachable colour comes from `solve_connected`, which marches in eighths of the colour's own chroma with a loose jump bound, and for far colours (display chroma 1.5 to 4) it can step onto another sheet; the per-pixel return takes the end of the connected interval found by a fine march (steps of at most 0.04 in `ρ`, shrinking where the pre-image moves fast). On 20,000 display colours uniform in the unit cube the two agree on reaching or not for 99.5 % (53 and 49 differ, P3); of the 2,353 colours both call unreachable, the share of the requested chroma kept differs by more than 0.05 for 235 (10 %, worst 0.46). Both are valid nearest-reachable colours and nothing in a photograph (0 of 1,487,970 pixels × 16 unreachable) takes this path. If the lead wants the random-colour 99.9 % figure back, the direct chain's `solve_connected` should march as the table does (it is `ym-color`, not changed here: it alters `inverse_continuous`, which F-06b's reference uses); the channel-held class is the same effect plus a display channel near its floor, whose pre-image is logarithmic, so an absolute error of 1e-5 to 1e-4 in T3b′ on that channel moves the answer.

**Determinism.** The core table is the same bytes on 1, 3 and 8 threads; T3b′, T1, T3a and the curves are the same bytes on 1 and 8 threads and on a repeat run, and `run` of 300 pixels is bit-identical from the three bakes (`bake_tables.rs`, `cine_per_pixel.rs`).

**What was removed:** the coordinate T3b and `Nodes::CineCoordinates`, `Table3::bake_cine`, `Table3::sample_stored`, the warp's role in a logarithm, the row-wise continuation bake and `CineRendering::inverse_coordinates_from` (with its tests); `inverse_continuous`, `inverse_coordinates` and `inverse_coordinates_reference` stay (the direct chain's return stage uses them). Tests: `crates/ym-color/tests/cine_per_pixel.rs` (7), `crates/ym-film/tests/bake_core.rs` (the numbers above, ignored), `bake_report.rs`, `bake_variant_a.rs`, `bake_tables.rs`. Re-run the numbers: `cargo test -p ym-film --release --test bake_core -- --ignored --nocapture --test-threads 1` (`YM_CORE_DIMS`, `YM_CORE_STEPS`, `YM_SCAN_SIZES`, `YM_SCAN_POWERS`, `YM_SCAN_WARPS`, `YM_BAKE_POWER`).


### 3.9 The spatial parts

The spatial parts are passes of the multi-pass plan of `docs/23` §5.2 and `docs/17`, with f32 buffers between passes. The Film node always compiles to several passes. The compiler of N-03 and N-04 is written for that and tests the split on a graph that needs nine registers; the Film node is its first real user.

| Pass | Reads → writes | Size, as a fraction of the photo's long edge |
|---|---|---|
| P1 colour in | LogC4 → `E` (T1) | per pixel |
| P2 scatter and halation | `E` → `E′` | tens of micrometres to about a hundred on the film, **0.1 % to 0.5 %** of the long edge, with a tail; halo up to several percent |
| P3 develop | `E′` → `D₁`, then the coupler field | per pixel, then a diffusion of tens of micrometres with an exponential tail of a few hundred (**0.05 % to 0.6 %**) |
| P4 grain | `D` → `D_g` | the pixel's area against the particle's; a dye-cloud blur of about a micrometre |
| P5 colour out | `D_g` → LogC4 (T3a, curves, T3b) | per pixel |

- **Photo-relative sizes.** The photo frame's long edge is 36 mm of film (a full-frame 35 mm exposure). Every size is in micrometres on the film and becomes `size / 36,000` of the long edge, then pixels at the render's scale (`docs/23` §3.6, `u_px`). A preview and an export therefore show the same film area with the same blur.
- **Halos.** A pass reads a tile and its halo (`docs/17`, D5). The coupler tail and halation decide it: `H = ceil(6 × the largest scale × u_px)`, a few percent of the long edge, worst case about 270 px at 8,000 px. Tiles equal the whole frame bit for bit.
- **Grain is deterministic.** A counter-based generator: a 32-bit integer hash of (the node's `grain.seed`, the pixel's integer coordinates **in the photo frame at the render's scale**, the channel, the sublayer). No state, no order, no tile origin. The normal draw is a sum of four uniforms (bounded tails, no `log`, `sqrt` or trigonometry), so the CPU twin and the kernel agree and Metal's fast maths cannot change a result (`docs/17` K8). The same seed gives the same grain for a given scale.
- **The export is the reference.** At 1:1 the viewport tile and the export tile are the same bytes. At another scale the preview shows the grain *of its own pixel size* (the model's particles per pixel change with the pixel's area, so the variance is the right one for the reduced picture), equal in distribution to the reduced export and not pixel for pixel. A zoom above 100 % enlarges the 1:1 grain (`docs/17`, *Magnified views*). The gate is statistical (§8).
- **Each pass has a CPU twin and a parity gate** (§8). The grain hash is exact; the values match within 1e-5.

### 3.9a As built (F-05)

`crates/ym-film/src/spatial.rs` (the passes, the tile API, the blur) and `grain.rs` (the generator and the calibration); our numbers in `data/film/params/halation.json`, `grain.json` and the `diffusion` key of `couplers.json`, each with its source or "chosen". `BakedFilm` gained the stage methods the passes call (`exposure_of`, `coupler_inhibitor`, `density_with_inhibitor`, `colour_out`); `run` is unchanged. Measured on 2026-11-04 on the Linux build machine (4 vCPUs, shared with another job while the timings ran), in release.

**API for F-07 and F-08.** `FilmFrame::new(baked, SpatialModel::load(data, stock), SpatialSettings)`; `required_input(frame, tile)` is the LogC4 / AWG4 f32 region a tile needs (the tile and every halo, cut to the frame); `run_tile(frame, tile, &input, RunOptions { threads, poison }) -> Plane` is LogC4 over the tile; `halos(frame)` gives the three halos. `Frame` is the photo frame at the render's scale; everything else follows from `Frame::px_per_um() = long edge / 36,000`. A pass reads its input at coordinates clamped into the frame, computes on the tile plus the halos of the later passes (cut to the frame), and a pixel depends only on its receptive field. `SpatialSettings` has `halation_amount`, `halation_size`, `grain_amount`, `grain_size`, `grain_seed` (1.0 is 100 %) and `diffusion_size` (0 turns the diffusion off).

**The passes.**
- *P1* `exposure_of` then `10^`: E linear in f32.
- *P2* E' = (1 - w_scatter - w_c) E + w_scatter G(12 µm) E + w_c (0.44 G(60 µm) + 0.56 G(204 µm)) E, per channel c. The PSF sums to 1 (a uniform area is unchanged). The scatter is the emulsion's own (the Halation sliders do not touch it); `w_c` is the stock's: the class weight from `info.antihalation` (strong 0.012, weak 0.04, none 0.15, chosen) times the layer factor (red 1, green 0.5, blue 0.15: the blue layer is behind the yellow filter layer) times Amount; Size multiplies the two widths. The shape is a core and a radial exponential tail (λ = 120 µm) written as two Gaussians (NNLS fit, within 32 % out to 3 λ).
- *P3* the per-pixel solve gives `D⁰` and the inhibitor `Q⁰ = amount M D⁰`; `Q = (0.74 G(28 µm) + 0.26 G(170 µm)) Q⁰` (a short Gaussian and an exponential tail, λ = 100 µm); the pixel develops once: `D = C0(log E' - Q)`. **Choice:** on a uniform area `Q = Q⁰` and `D` is the per-pixel fixed point exactly, so F-03's curves gate holds; at an edge a neighbour's inhibitor acts on the pixel (the adjacency effect), which a per-pixel solve cannot do. One pass of the diffusion, not an iteration to a spatial fixed point.
- *P4* grain, below. *P5* `colour_out`.
- **Blur.** Separable Gaussians in f64 in a fixed tap order, f32 between passes. Up to 4 px a direct sum of radius `ceil(6 σ)`; wider on a coarse grid whose cells are `f × f` pixels at absolute multiples of `f = 2^k` (box mean, a direct blur of the cells with `σ_c² f² = σ² - f²/4`, a bilinear read back), so a cell never depends on the tile. Halo of a Gaussian: `ceil(6 σ) + 3 f` (the 3 f only on the coarse grid). Against the direct sum on an edge and a disc (peak 1.5): 0.0006 at σ = 7 px, 0.0011 at 12, 0.0038 at 25, 0.0070 at 40.

**Grain (P4).** On the film density (a slide's own). Per channel and sublayer `s` (the profile's three `density_curves_layers`): amplitude `sqrt(a_c r_s Dmax_s² p_s (1 - p_s))` over the pixel's side in µm, `p_s = D_s / Dmax_s` read from a 1,024-entry table against the total density; `r = (1, 0.4, 0.16)` relative particle areas (chosen). The draws are independent per (pixel, channel, sublayer): `normal = sqrt(3) · sum (2 b_i - 65535) / 131072` over four 16-bit words of two 32-bit hashes of `(seed, x, y, channel, sublayer, word)` (a murmur3-finalizer chain; exact integer maths, no `log`, `sqrt` or trigonometry in the draw). The noise of a channel (sublayers summed) is blurred by the dye cloud (σ = 1 µm × Size, a direct Gaussian, 0 to 2 px) and divided by the kernel's power so that the blur keeps the variance. **Calibration (Selwyn).** The particle area `a_c` of each layer is solved at build time so that the model gives the RMS granularity at density 1.0 in the 48 µm aperture (1,810 µm²); a pixel of `A` µm² then has the variance `(RMS / 1000)² · 1,810 / A`. Amount multiplies the standard deviation, Size multiplies the particle's side (variance by Size²). **Data sheets: none read.** Kodak prints granularity as a graph and Fujifilm as a class number, and neither was readable as numbers on 2026-11-04, so **every stock uses the fallback** `RMS × 1000 = 2.5 (ISO / 100)^0.34` (slides × 2.5) in `grain.json`; Portra 400 gets 4.0. A per-stock number from a sheet goes into `rms_x1000.stocks` and wins. This is an open item for the owner's check.

**Halos** (settings at 100 %): 2,000 px long edge (frame 2,000 × 1,333): halation 80 px, diffusion 63 px, grain 0, total 143. 8,000 px: halation 320 px (272 for 6 σ and 48 for the coarse grid's cells), diffusion 251, grain 2, total 573. At 8,658 px: 343, 294, 2. Size 200 % doubles the widths.

**Gates.**

| Gate | Result |
|---|---|
| Tiles equal the whole frame, halos and grain included, two tile sizes (97 × 61, 301 × 200) at four origins, 1,280 × 853, Portra 400; Velvia 100 and Vision3 500T with every setting off the default | **met**, bit for bit |
| Repeat run, 1, 3 and 8 threads, poison (every buffer NaN), a tile under poison | **met**, bit for bit |
| Far corner of an 8,658 × 5,772 frame (40 × 33 tile against 211 × 150 at the corner; a tile at x 8,000 against a larger one) | **met**, bit for bit |
| Uniform frame, halation and grain off (diffusion off and on), against the per-pixel chain in LogC4 | **met**: 4.7e-8 worst (Portra 400, Ektar 100, Velvia 100; gate 1e-6); F-03's gates are untouched (the chain code is unchanged) |
| Preview k = 4 against the reduced export, halation and diffusion, display linear (soft-edged test image, preview input the area mean in linear) | **met**: largest 0.0001, mean 0.00001 (gate 0.02, 0.002); the same comparison with both passes off is 0.0002 and 0.00001, so the number is the sampling floor. With hard edges (1/4 px) the floor itself is 0.11: the colour curves are not linear and an area mean does not commute with them, which no pass can fix |
| Grain: variance of the preview against the reduced export (4,096 px against 1,024 px, flat density) | **met**: ratio 1.008, 0.996, 1.003 (gate within 10 %) |
| Grain: the draw's mean and variance on a flat field against the model | **met**: sigma within 0.1 % of the model at 4,096 px, within 1.5 % at 24,000 px with the dye cloud on; the unit draw has mean < 0.004 and variance within 1 % over 10⁶ draws, per channel and sublayer; neighbours, channels, sublayers and seeds are uncorrelated (< 0.01) |
| Calibration | the model's sigma at density 1.0 in the 48 µm aperture is the RMS granularity within 1 % (Portra 400: 0.00401) |

Tests: `crates/ym-film/tests/spatial.rs`.

**Speed** (CPU, whole frame, all passes at 100 % including the baked colour at 33³ tables; the other agent's job shared the four cores): 24 MP (6,000 × 4,000) **34.6 s on one thread, 15.3 s on four**; a 2,000 px preview 4.0 s and 2.4 s. The per-pixel colour part is 0.9 µs a pixel (exposure 0.41, couplers 0.31, colour out 0.13), 22 s of the 34.6 s at 24 MP; the blurs are the rest. Thread scaling is poor (memory-bound planes, and the shared cores). This is the reference, not a speed target; the GPU twin (F-08) is the fast path (§3.10, §9 item 11). Costs to cut if F-07 needs the CPU fast: a tile run per 512 px square (the halos are 2 to 7 % of the long edge), and f32 accumulation in `mix`.

**Open points.** (1) Every grain amplitude is the fallback formula (above). (2) Halation and diffusion numbers are chosen from the literature's orders of magnitude, not measured (V2 work). (3) The grain draws are independent per pixel; at magnifications where a particle spans many pixels (a zoom above 100 %) the cloud blur is the only correlation. (4) At the frame edge the dye-cloud blur sees the edge pixel repeated, so the variance in the outer 1 to 2 px is slightly higher.

### 3.9b As built (F-08a): the Film node's kernels

`crates/ym-gpu/src/kernels/film.rs` and `src/shaders/film.wgsl`, `film_t3.wgsl`, `film_return.wgsl`; the CPU twins are the `ym-film` functions of §3.8c and §3.9a. `ym-gpu` takes plain data (`FilmColourData`, `FilmPlan`; no `ym-film` type, which is a dev-dependency of the tests only); `tests/film_common/mod.rs` converts `BakedFilm::device_parts()` and `FilmFrame::device_plan(frame)` (additive accessors in `ym-film`: `DeviceParts`, `DevicePlan`, `SceneStage::{to_adapted_xyz, locus}`, `GrainModel::{table, d_top}`, `grain::HASH_CONSTANTS`, `scene::{KNEE, LIMIT}` public) and is the template for the glue of F-08b. **Fixed counts:** coupler step 12 iterations with 4 halvings; the display step's Newton for `Y2` 10; the cube-root inverse 8; the luminance floor 10; the core's Newton steps `CoreTable::steps()` (5), each trying its candidates in the CPU's order. **Choices:** the scene side is always the lenient one of `inverse.rs` (the CPU's strict form falls back to it); `1 - (o - z0) / kh` at the display ceiling and in `tone_inverse` is written `(1 - o) / kh` (a rounding of 1e-7 at the fold `o = 1` would otherwise be a square root of 3e-4 and moved photo pixels by 1e-2 in LogC4); `tone_inverse` is evaluated as `ln Ys` in the log domain with an `expm1` series; the locus hit tolerates `s` of 1e-5 (the CPU 1e-12). The grain hash constants are passed to the kernel from `HASH_CONSTANTS` (K7); the refine settings (0.7, the three scales, the three `mu`, `CONVERGED`) and the fixed iteration counts of the display step are public constants of `ym-color` (`cine::kernel_refine_scalars`, read by `refine` itself), written into the shader as `FX_*` and pinned by `the_newton_settings_are_generated_from_ym_color_not_typed`; the locus tolerance, the coupler step's counts and the determinant floors are kernel-only (the CPU's are in f64 and in `ym-film`).

**Gates (lavapipe, f32 storage; `tests/gpu_film.rs`, `tests/gpu_film_spatial.rs`).** The colour-out parity follows the owner's rule for the analogous baking gate (2026-11-04: "Gate photos, report random"): the pixels of the 20 photos (every 40th, 8,500) and the synthetic saturated colours (six hues, four saturations, −6 to +4 stops) that the cine rendering can reach (chroma fraction 1, no display channel at the floor; 218 to 241 of 264 a stock) are gated strictly, **with no allowance, at 2e-4 in LogC4 and one 10-bit code after C13**; the unreachable or floor-held synthetic colours and the random LogC4 cube are printed and not asserted. Four stock × output pairs, P5 alone / the three passes together (largest LogC4 error, largest after C13): photos 1.8e-5 / 1.2e-4 (Ektar 100 on sRGB), 0.004 code; reachable synthetic 1.0e-4 / 7.9e-5, 0.003 code; **both gates met, no reachable colour exceeds 2e-4.** Reported: unreachable synthetic colours (23 to 46 a stock) at most 3.9e-4 (P5) and 4.0e-4 (together) in LogC4, 0.020 code at most (Vision3 500T); the random cube (4,033 seeded pixels) at most 2.4e-2 for P5 and 1.2 for the chain (Ektar 100 on sRGB, 3.5 % of samples over 2e-4), 0.115 code at most, because the pre-image is ill-conditioned near a fold and a rounding of the display colour in f32 moves it. P1 relative error of `E` 4.3e-5 at most, P3 density 2.0e-6 and inhibitor 4.0e-7. Blurs against `spatial::blur` (σ 0.3 to 40 px, direct and coarse grid, interior and frame edges): 4.8e-7 of the peak (gate 1e-4). Grain: the hash words and integer sums exact (two origins including (8600, 5700), three channel and sublayer pairs, 25,000 words), values 2.4e-7 (gate 1e-5; with the dye cloud on a 6,000 px frame within the node test). The node on a tile against `FilmFrame::run_tile` (Portra 400, Velvia 100, Ektar 100 at 1,280 px; Portra 400 on sRGB at 6,000 px with halos 228, 194, 2): 7.8e-7 and 2.8e-5 in LogC4. **Tiles equal the whole frame bit for bit** at 97 × 61 and 301 × 200 (360 × 241 frame, halation and diffusion doubled, grain on), a repeat run and a run without poison are bit-identical, and the far corner of an 8,658 × 5,772 frame (a 40 × 33 tile against 211 × 150; a tile at x = 8,000 against a larger one; halos 343, 294, 3) too. Rough cost on lavapipe (4 vCPUs, 512 × 256, including the table upload): 0.30 µs a pixel for P1, 0.39 for P3, 0.36 for P5; the kernel alone was not isolated. The Metal run (K8's clang model, the physical Mac) has not been done: the fast-math argument is in the module notes of `film.rs`.

### 3.10 Cost

The per-pixel work before the return is about 200 floating-point operations, three table lookups (one 2D, two 3D tetrahedral) and nine one-dimensional lookups. **The return (§3.8c, F-04d) adds** step 1 of the inverse (a Newton iteration of a few `exp` and `ln`), a read of four rays of the core table, up to five Newton steps with the exact Jacobian (each a dual-number evaluation of the cine core, about 30 transcendental calls and about four times the arithmetic of the plain core, a 3 × 3 inverse and one to three plain evaluations) and the closed-form scene side: by counting, 2,000 to 4,000 floating-point operations and 100 to 200 transcendental calls. Measured on the CPU: 1.7 µs a pixel for the return, 2.7 µs for the whole baked colour part (1.2 µs of which is T3b′ and the front). The spatial passes are separable Gaussians and one hash. On the device this is milliseconds at a preview size; on the CPU fallback it is slow, and the node says so (`viewport_busy`, `docs/11`). Speed targets are in §8; nothing is measured yet.

## 4. Controls

The mockups are in **`design/film/`** (F-01, 2026-11-04): 15 pictures, `index.html` and `README.md`. They draw the interface around hand-made film pictures; this section follows them, and **this file wins where they differ**. Four choices are still open and are marked "Option A / Option B, the owner chooses (F-H1)". Names, ranges and defaults are proposals; the owner approves them with the mockups (F-H1).

### 4.1 The node's tabs

The panel of a Film node (`docs/23` §7.5) has the tabs **Film · Print · Halation · Grain · Mask**, with the head line "Film · whole photo" (screens 02, 08c). **Halation and Grain: Option A / Option B, the owner chooses (F-H1).** Option A is the five tabs above (08a, 08c). Option B is four tabs, **Film · Print · Grain · Mask**, where the one Grain tab holds a Halation card and a Grain card (08b); a better name than "Grain" for that tab may be needed.

| Tab | Control | Range | Step | Default | What it does |
|---|---|---|---|---|---|
| Film | **Stock** | the 20 camera films, grouped Colour negative, Cine negative, Slide (the cine negatives wait for their print film, §6.4). **Picker: Option A / Option B, the owner chooses (F-H1):** A is a grouped list with the ISO on each row (03); B is tiles with a small render of the open photo in each stock (04, six columns, no scrolling; 20 small renders when it opens) | – | Portra 400 | the profile read for stages 1 to 6. A quiet line under the field gives the kind and ISO ("Colour negative · ISO 400"); both come from `names.json` |
| Film | **Exposure** | −4…+4 EV | 0.05 | 0 | exposes the negative more or less; the print follows. 0 is the Input's exposure |
| Film | **Strength** | 0–100 % | 1 | 100 | how much of the film look is applied (a node-wide mask value) |
| Print | **Paper** | "Matching paper" and the 6 papers (the 2 cine print films stay hidden until §6.4 is settled, 05b) | – | Matching paper | the print profile; "Matching" is the stock's own `target_print`, whose name a quiet line shows under the field |
| Print | **Print exposure** | −3…+3 stops | 0.05 | 0 | the lamp's exposure of the paper; 0 keeps the app's brightness |
| Print | **Lights** Yellow, Magenta. **Option A / Option B, the owner chooses (F-H1):** A is two sliders (05); B is a colour pad with one handle for both and two number fields (06) | −50…+50 | 0.5 | 0 | the enlarger's filtration, in Kodak CC units, as shifts from neutral. 0 is neutral for this pair; the hint "0 is neutral for this film and paper." sits under them |
| Print | **Cyan** (under More, both options) | −50…+50 | 0.5 | 0 | the same for cyan; mostly equal to a change of exposure |
| Print | **Brightness** (under More; **Option, the owner chooses (F-H1):** offered or left out) | As the app · As printed | – | As the app | As the app (default) places the print so that adding the node does not change the picture's brightness (§3.6). As printed leaves the paper's own (12). §9 item 10 |
| Halation | **Amount** | 0–300 % | 1 | 100 | 100 is the stock's own (set by its antihalation layer) |
| Halation | **Size** | 50–200 % | 1 | 100 | the spread |
| Grain | **Amount** | 0–200 % | 1 | 100 | 100 is the stock's own granularity |
| Grain | **Size** | 50–200 % | 1 | 100 | the particle size |
| Mask | as the correctors' Mask tab (`docs/23` §7.6) | | | | |

A slide stock dims the Print tab but leaves it openable, so the sentence can be read: "Slide film is scanned as it is. There is no print." The page holds that sentence and the controls greyed (07). The node's head, Bypass, Reset and rename are the correctors'. The Film tab ends with the credit line and its link (02); About carries the same line, the CC BY-SA 4.0 link and the trademark line (11).

### 4.2 Defaults and Auto

A new Film node is **Portra 400 on its matching paper** with every number at its default. (The stock is a mockup question, §9, item 10.) **Auto means nothing here.** Auto belongs to Main (`docs/23` §3.2): the Film node has no Auto fields, and Re-run and the providers (`docs/19`, `docs/22`) do not touch it. What is automatic is built in: the neutral filtering is solved for the pair (§3.5), the default print exposure is placed for the app's brightness (§3.6), and the film exposure 0 is the Input's.

### 4.3 Masks

The node takes masks, the same rows and rules as a corrector (`docs/23` §3.3 and §4), because variant A returns LogC4 (§2.6). With variant B it could not.

### 4.4 What the mockup round answered, and what is still open

Answered (the owner approves with F-H1):

1. **Credit line:** at the foot of the Film tab, with its link; also in About (02, 11).
2. **Slide film's Print tab:** dimmed, openable, the sentence and greyed controls (07).
3. **The Output tab:** the note under the tab row (10).
4. **Names:** real names with the line, as data (§1.3); the picker rows and tiles show them (03, 04).
5. **The card's thumbnail and the Add node entry:** the thumbnail is the node's own small picture of the photo; the entry sits under Corrector, enabled, the Later items follow (01, 02).
6. **Masks:** the Mask tab is a corrector's (09).

**Chosen by the owner (F-H1, 2026-11-04):**

1. **Stock picker: Option B, tiles** (04): a small render of the photo in each of the 20 stocks. The tiles are drawn in the background at thumbnail size (the node thumbnails' size, `docs/23` §6), one stock at a time in the list's order, each appearing as it is ready; they are kept per photo and per the node's settings up to the Film node, in memory only, and the picker never waits for them (a tile not drawn yet shows the stock's name on a plain card). The table cache of §3.8 holds at least 20 T1 entries so the picker does not evict the photo's own. The list order and the groups are those of Option A.
2. **Lights: Option A, two sliders** (05): Yellow and Magenta, Cyan under More.
3. **Halation and Grain: Option A, two tabs** (08a, 08c): **Film · Print · Halation · Grain · Mask**.
4. **Brightness: "As the app" only.** The "As printed" control (12) is not built; print exposure covers the rest.

**Found while drawing, settled by §6.4:** a cine negative's own `target_print` is the hidden Kodak 2383, so until a projector-lamp spectrum is sourced its "Matching paper" is a still paper, **Kodak Professional Portra Endura**, and the line under the field names it. Stock names come from `data/film/params/names.json` (F-02), which uses the profiles' own `info.name`: the mockups' "Kodak Portra 800 push 1" is "Kodak Portra 800 (Push 1)" in the app.

### 4.5 Settings and the format

A node of `kind: "film"` has these fields and no others (`docs/23` §3.2 rules: a missing field takes its default, a bad number is clamped and reported, an unknown field is refused). Plain numbers only; no Auto kinds.

```json
{ "id": 2, "kind": "film", "name": "Film", "bypass": false, "pos": [1, 0],
  "film":     { "stock": "kodak_portra_400", "exposure_ev": 0 },
  "print":    { "paper": null, "exposure_ev": 0, "lights": { "c": 0, "m": 0, "y": 0 } },
  "halation": { "amount": 100, "size": 100 },
  "grain":    { "amount": 100, "size": 100, "seed": 0 },
  "strength": 100,
  "masks":    { "invert": false, "items": [] } }
```

- `stock` and `paper` are profile ids (the file names without `.json`). `paper: null` is the matching paper. A stock or paper that is not installed, or of the wrong kind, is refused: `Unknown film stock "x". Run "ym film list" to see the stocks.`
- `seed` is not in the interface; it keeps the grain a function of the settings (hard rule 8).
- A Film node is a live node when the Output needs its picture, like any node. Copy and paste carry it. The DNG bakes it in (halation and grain too, at the DNG's grid) and the DNG note says so.
- **`ym`**: no new flags (a Film node is not in the default graph, so the flags that set a field of Input, Main or the Output do not reach it). `--settings` carries a graph with a Film node. `ym film list [--json]` lists the stocks and papers with their ids and kinds. `ym doctor` gains a line `Film: 28 profiles, data version 0.3.2`. `docs/10` and the help change in the same commit (`crates/ym-cli/tests/docs.rs`).
- `PIPELINE_VERSION`: no existing graph changes by a bit, and the version takes a minor step to **0.10.0** with the node, because an older build refuses a graph with a `film` node (the owner, 2026-11-04: "Minor step to 0.10.0"). Only the written version string changes (`ym --version`, `ym doctor`, `ym info`, `ym:PipelineVersion`).

### 4.6 Copy

Short, plain sentences, British spelling (`AGENTS.md`). Drafts:

- Menu: "Film" / "Models a colour negative, a print and a scan."
- Film tab: "Modelled on published data. Film profiles by Andrea Volpato (spektrafilm), CC BY-SA 4.0. Not affiliated with Kodak or Fujifilm." (a link on "Andrea Volpato (spektrafilm)" opens the project page in the system browser, like ARRI's page).
- About: the same line with a second link on "CC BY-SA 4.0", and "Kodak, Portra, Vision3, Fujifilm and Velvia are trademarks of their owners."
- Print tab: "Matching paper" (the matching paper's name under it); the brightness control reads "As the app" and "As printed".
- Slide: "Slide film is scanned as it is. There is no print."
- Lights: "0 is neutral for this film and paper."
- Output tab, while a Film node is live: "The look acts after the Film node. Choose Neutral to see the print as scanned."
- Missing data: "Yellowmoon's film data is missing. Reinstall the app."

## 5. Where it fits in the spec set

`docs/23` §7.4 gains the Film row when the node is built; §3.2 gains the `film` kind; `docs/17` K3's storage-buffer count is checked (the Film passes use their own pipelines, not the fused kernel's seven buffers). `docs/02` gets a short section on the node's colour (the inverse is used as in C12). `docs/01` gets the render-graph keys of §3.8. `docs/10` and the help get `ym film list` and the doctor line. `docs/11` gets the panel. These are F-07 to F-10.

## 6. Data and shipping

### 6.1 The profiles

- **Where**: `data/film/spektrafilm/profiles/*.json`, 28 files, **5,794,633 bytes (5.53 MiB; 1.1 MB compressed)**. **Committed**, not fetched at build time: the size is small (like `data/models/`), a build needs no network, and the licence file travels with the data in one commit, which matters because removing the attribution ends the grant (CC BY-SA 4.0, section 6(a)). The app finds them through `find_data_dir` (`docs/01`), in `Contents/Resources/data/film/`.
- **Unmodified**, byte for byte, as taken from the pinned commit. Beside them:
  - `LICENSE`: the text of `SPEKTRAFILM_LICENSE.txt`, byte for byte (SHA-256 `76d62950…` at the commit read);
  - `ATTRIBUTION.md`: the author, the URL, the licence, the commit taken and the date, the profile version, "Modified by: nobody", and a pointer to `CHANGELOG.txt` if that ever changes (draft below);
  - `SHA256SUMS`: every file, the licence included, checked by a test and by `tools/licences/check_bundle.py`;
  - `CHANGELOG.txt`: absent while nothing is modified; the author asks for it ("describe [changes] in a file CHANGELOG.txt shipped together with this license").
- Each profile carries its own notice in `metadata.license` and `metadata.copyright`, which travels with the file.

```
spektrafilm by Andrea Volpato
https://github.com/andreavolpato/spektrafilm
Licensed CC BY-SA 4.0 (the text is in LICENSE, in this folder).

Taken from: <commit> of the main branch, <date>. Profile version <0.3.2>.
Files: 28 profiles in profiles/, not modified. SHA-256 in SHA256SUMS.
Modified by: nobody.
The profiles were made by the author from Kodak and Fujifilm data sheets and other
published material. The original data belong to their holders.
Yellowmoon is not affiliated with Andrea Volpato, Kodak or Fujifilm.
```

**State after F-02 (2026-11-04).** Pinned: commit `3bb2c2d2801ff68b92019cf1dbcbb133d60832bc` ("fix: vlog midgray exposure", 2026-06-07). `git ls-remote` showed it still the head of `main` on 2026-11-04, so nothing changed since the research note: 28 profiles, profile version 0.3.2, 5,794,633 bytes, the licence file `76d62950…` (Preamble v1.1, 2026-05-28); the file hashes equal the appendix of the research note. `tools/film/update-profiles.sh` fetches the commit with `git fetch --depth 1`, copies the files with `git show` (byte for byte), parses every profile, prints added, changed and removed files, **stops if the licence text differs** (`--accept-licence-change` goes on) and writes `LICENSE`, `ATTRIBUTION.md` (with the full commit id) and `SHA256SUMS` (the 28 profiles and `LICENSE`; `ATTRIBUTION.md` is ours and not hashed). The folder was made by that script. `data/film/params/names.json` holds ids, display names (the profiles' own `info.name`), kind, maker, ISO and the usual print; `kodak_2383` and `kodak_2393` are `"hidden": true` (§6.4).

### 6.2 Credits

| Where | Text |
|---|---|
| `NOTICE` (a new item) | the line above, the licence name and URL of the text, "used by the Film node (data/film/spektrafilm/)", the commit, "not modified" |
| About (`docs/11`) | "Film profiles by Andrea Volpato (spektrafilm), CC BY-SA 4.0" with a link; the trademark line (§6.5) |
| The Film node (Film tab) | "Modelled on published data. Film profiles by Andrea Volpato (spektrafilm), CC BY-SA 4.0. Not affiliated with Kodak or Fujifilm." |
| `THIRD-PARTY.md` (in the bundle) | an entry through `tools/licences/` (a hand-added text like the others) |
| README | a credits line |
| Exports | the XMP gets `ym:FilmData` with the profile version, so a file says which data made it. The finished photograph needs no credit ("Not required on a finished film or photograph") |
| `check_bundle.py` | fails when `NOTICE` loses the item or the folder loses `LICENSE`, `ATTRIBUTION.md` or a matching hash |

### 6.3 The requests are honoured

The licence file lists wishes "beyond the legal floor". They are "not legal requirements", and **we treat them as binding**:

| The request | What we do |
|---|---|
| Keep the licence file unchanged; describe changes in `CHANGELOG.txt` | `LICENSE` is a byte copy under a hash; profiles ship unmodified; any change goes in `CHANGELOG.txt` and `ATTRIBUTION.md` and the file stays CC BY-SA 4.0 |
| Credit it "in the human layer too" | About, the node, README, `NOTICE`, this spec |
| Do not use "spektrafilm" or the name "in product branding without asking" | The node, the crate (`ym-film`), settings, commands, menus and presets are called Film. The name appears only as factual credit (§6.2), in docs, in the data folder's name and in the oracle tool's folder. A test fails if a UI string, a setting or a command contains it elsewhere |
| Do not sell the LUTs as "a paid LUT pack" | No LUT is shipped, exported or sold. The tables are baked in memory per photo and never written; **there is no "export as LUT" and none will be added without asking the author.** The app is free (Apache-2.0) |
| "Please don't train commercial AI models on the LUTs or their outputs" | The engine has no learned part. **None of our models is trained on the profiles, on a table baked from them or on a Film node's output.** No script under `training/` reads `data/film/` or a Film render; a CI check fails if one does. The upsampling generator reads CIE data only. The Auto providers see the Input's picture, not a film render (`docs/19` §3) |
| Acknowledge spektrafilm in credits | the lines of §6.2 |
| The README also says LUTs are "strict 'commercial use, free share, no resale'" | followed in practice as above; we ask the author to align the wording (§7) |

### 6.4 Other data

| Item | Used for | Licence and decision |
|---|---|---|
| **Spectral upsampling table** | stage 2 | spektrafilm's ("Hanatos 2025") has no licence. **We make our own** (§3.3, F-03a) and ship it as `data/film/upsampling/`, **CC BY-SA 4.0** with the credit "Computed by Yellowmoon from CIE 1931 2° colour-matching functions (CIE, CC BY-SA 4.0)" and "Modified by: Yellowmoon (computed, not copied)". The reason: every number in it is computed from the CIE observer (CC BY-SA 4.0), which makes it Adapted Material that must be shared under the same licence (§6.7), so "Apache-2.0" was wrong. It is a separate folder with its own `LICENSE` and `ATTRIBUTION.md`, checked by `check_bundle.py`; our code stays Apache-2.0. If its author licenses his table, we may use it as the oracle |
| **CIE 1931 2° observer, D50, D55** (and a blackbody by formula) | stages 1, 9 | CIE tables, recorded in `docs/14` as CC BY-SA 4.0 (re-check on the dataset pages; the listing page states no licence). Recommended: ship the 5 nm tables as a separate CC BY-SA 4.0 file in `data/film/cie/` with its own `LICENSE` and credit, the same mechanism as the profiles. This also touches the open question M1-H4 (`docs/14`). **Done in F-02:** the three tables are in `data/film/cie/` exactly as the CIE publishes them (`tools/film/fetch-cie.sh`): the observer at 360 to 830 nm and D50 at 300 to 830 nm, both in 1 nm steps, and D55 at 300 to 780 nm in 5 nm steps (the CIE has no 5 nm file of the first two; the engine resamples). **The text of the three dataset pages names no licence.** The metadata file each page links (`<name>.csv_metadata.json`) says `"rightsList"`: "Creative Commons Attribution-ShareAlike 4.0 International", `https://creativecommons.org/licenses/by-sa/4.0/`; the script stops if it ever says otherwise. The CIE's D55 description says 380 to 780 nm while its file starts at 300 nm. If refused: an analytic fit of the observer and a blackbody-based stand-in for daylight (the daylight basis tables are CIE's too), with a wider oracle bound |
| **Heat-absorbing glass curve** (Schott KG3) | the lamp | a manufacturer's curve with no licence stated in spektrafilm. Replace with our own transcription from Schott's published data (a few dozen numbers, facts) in `data/film/params/`, or a smooth analytic stand-in |
| **Dichroic filters** | the enlarger | analytic edges with our own numbers fitted to published curves (Edmund Optics, Thorlabs, Durst data sheets). The vendors' CSVs in spektrafilm are not used |
| **Projector lamp** (cine print films) | viewing 2383 and 2393 | the source of the "Kinoton 75P" spectrum in `colour-science` is not checked. Until a licensed or published spectrum is found, **the two cine print films are hidden** and the 5 cine negatives are offered with a still paper as their matching paper |
| **Tungsten ("T") illuminant** for tungsten films | stage 1 | a 3200 K blackbody by formula |
| **Coupler, grain, halation, glare, filter parameters** | stages 3, 5, 6, 7 | ours: from the literature and by eye, in `data/film/params/` (Apache-2.0). A file fitted to a profile's content is an adaptation and is CC BY-SA 4.0 with its own `LICENSE` and a `CHANGELOG.txt` entry; one chosen by hand is Apache-2.0. **F-06b: no parameter file is fitted.** The enlarger's nine numbers were fitted to the oracle's outputs and the fit did not generalise (§3.7a), so `enlarger.json` and `kg3.json` stay hand-chosen; the numbers fitted to the oracle in F-06b are the three of the upsampling table's luminance target (§3.3, in `data/film/upsampling/`, which is CC BY-SA 4.0 already) and the code's own readings (mid-grey placement, the balance), Apache-2.0 |
| **ICC profiles, neutral print filters, reflectance sets** | – | not needed |

### 6.5 Names of film stocks

The profiles name the stocks as the manufacturers do ("Kodak Portra 400", "Fujifilm Velvia 100"). These are trademarks. The CC BY-SA licence gives no trademark right (2(b)(2)) and no endorsement (2(a)(6)). **The names are the owner's decision** (§9, item 4). Options:

| Option | What the person sees | Cost |
|---|---|---|
| **1. Factual names with a line** (recommended) | the stock list shows the names as data ("Kodak Portra 400"), and the Film tab and About say "Modelled on published data. Not affiliated with Kodak or Fujifilm. Kodak, Portra, Vision3, Fujifilm and Velvia are trademarks of their owners." | the same footing as our ARRI lines (`docs/14` rule 6); the names are still the labels of choices in our interface, so a legal check is wise (M4-H3) |
| **2. Neutral names** | "Daylight negative 400, warm", "Slide, vivid" with the stock named only in a tooltip's data line | no trademark use as a label; the person must learn what the names mean; a rename table makes it a one-file change |
| **3. Mixed** | neutral names, and "Modelled on Kodak Portra 400" as a second line | in between |

Whichever is chosen, the names live in **one data file** (`data/film/params/names.json`), the code uses ids, and no logo, box colour, typeface or marketing text of a manufacturer is used. Never in a product, feature or preset name.

### 6.6 Keeping the data current

The profiles are pinned to a commit. `tools/film/update-profiles.sh <commit>` fetches (build tooling may download, `AGENTS.md` rule 7), verifies, rewrites `SHA256SUMS` and `ATTRIBUTION.md`, and shows what changed. A new profile version changes the engine's output, so it bumps `ym:FilmData`, re-records the goldens and gets a line in `CHANGELOG.md`. We never edit a profile in place.

### 6.7 How CC BY-SA 4.0 is kept

- The profiles are **aggregated, not merged**: plain files read at run time, never compiled into Rust source, never packed in an opaque archive, never encrypted or obfuscated (no "Effective Technological Measures", 2(a)(5)(c)). The app bundle's ad hoc signature seals the bundle, not the files.
- A profile read, resampled in memory and baked into a table that lives in memory is a technical modification that "never produces Adapted Material" (2(a)(4)). Anything we **write** that is derived from a profile (a fitted parameter file, an oracle fixture computed from one) is Adapted Material: CC BY-SA 4.0, with the credit and a note that it was modified and by whom, as a separate file.
- Our code stays Apache-2.0. The engine does not adapt the profiles; it reads them.
- Nothing about a photo is saved (hard rule 4): the tables and the grain exist in memory only.

## 7. The other route: asking the author

The README invites it: "If GPLv3 is not compatible with your project, please reach out to discuss alternative options." It is a **[HUMAN]** item (F-H2): the owner writes, to the address in `SPEKTRAFILM_LICENSE.txt` ("private inquiries about commercial licensing, name use, or custom work") or in a GitHub Discussion. Send it **before F-03** and wait for the answer, because it can change the plan. Silence is not consent: after a month without one, the owner decides whether to go on clean-room or to stop.

**What we would ask.**

1. A licence compatible with Apache-2.0 for the code, or for the parts we would port (a dual licence, or MIT or Apache-2.0 for the engine), or **a written statement that an independent implementation, written from the profile format and the published literature, is not a derivative of the code** and may be Apache-2.0.
2. Permission to use the code's default parameters (the DIR couplers, grain, halation, glare and the enlarger's filter edges) as data.
3. That shipping the 28 profiles unmodified in a free Apache-2.0 app, as separate credited files, is within the licence, and which of the README's "strict" wording and the licence file's "not legal requirements" is meant.
4. How he wants to be credited, and whether the line "Film profiles by Andrea Volpato (spektrafilm)" is right.
5. The licence of the "Hanatos 2025" tables, and **a second letter to hanatos** (the vkdt author) for the same.
6. Whether he would like to see the node, and how our fixtures and any new profile data could flow back.

**What it would save.** The clean-room discipline and the shingle check (§3.1) would be optional; the upsampling table (F-03) and the coupler, grain and halation models (F-05) could follow his, so the oracle could be a close match and not a bound; and the main legal risk (§9, item 1) would be closed. **What it does not save:** the engine still has to be Rust, deterministic, tile-safe, baked and on the GPU. If he says no, we continue clean-room and the risk stays what it is, or we stop.

## 8. Tests and gates

Every gate must pass in CI (or the named local job) before its task is done. **A failed gate is reported, never loosened** (`AGENTS.md`). The numbers marked *starting* are the first bounds, which the named task measures and writes into `docs/STATUS.md`.

| Area | Test | Gate | Task |
|---|---|---|---|
| Profile reader | all 28 files parse and validate (grid, shapes, ranges); a damaged file gives a plain error; the SHA-256 of every file matches `SHA256SUMS` | 28 of 28; the error | F-03 |
| Engine against published curves | a neutral exposure ramp through stages 2 to 5 gives each camera film's `density_curves`, and through 8 each paper's, couplers on | within 0.01 density (*starting*) | F-03 |
| Neutral | an 18 % grey prints neutral at default lights for every negative on its matching paper | a\*, b\* within 1.0 (*starting*) | F-03 |
| **Oracle** | spektrafilm's Python (a pinned commit, a separate process, **never a dependency**) on 24 chart patches, a 125-point RGB lattice and a 21-step grey ramp, **with couplers, halation, grain and glare off** (its per-pixel mode), compared at its named taps: `log_e_film`, `cmy_film`, `cmy_print`, `rgb_out`. Fixtures are committed (they are Adapted Material: CC BY-SA 4.0 with credit); the generator is `tools/oracle/spektrafilm/` | per stage: 0.03 log10 / 0.03 density (*starting*); final ΔE00 median ≤ 1.5, 95th percentile ≤ 4 (*starting*). **Measured (§3.7a, F-06b), final ΔE00 median / p95 (before F-06b in brackets): Portra 400 0.68 / 2.1 (1.58 / 5.8), Ektar 100 1.30 / 2.9 (1.40 / 4.9), Gold 200 0.80 / 2.4 (1.49 / 6.1), Pro 400H 0.79 / 1.7 (1.12 / 2.5), Velvia 100 0.69 / 1.6 (0.91 / 3.4), Ektachrome 100 0.99 / 1.7 (1.50 / 5.7): met; Vision3 500T 2.5 / 6.7 (3.70 / 9.3): not met. Stage gates, largest difference after the grey is matched (before): stage 2 0.07 to 0.11 (0.20 to 0.49), stage 5 0.04 to 0.05 on the negatives (0.10 to 0.22) and 0.17 to 0.23 on the slides (0.9 to 1.2), stage 8 0.09 to 0.18 (0.28 to 0.42): not met (saturated colours, §3.3 and §3.7a); Vision3 500T 0.36, 0.19, 0.46 (0.58, 0.31, 0.73): not met. `YM_ORACLE_ENFORCE` stays opt-in** | F-06, F-06b |
| Baking error | the table chain against the direct f64 chain on the pixels of the 20 test photos × the 8 stocks, Display P3 and sRGB (`baking_error_on_photos`); random colours are measured and reported (`random_colours_by_class`), not gated | **largest ≤ one 10-bit code after C13 and 99.9 % within a quarter code, on the photos** (owner, 2026-11-04: "Gate photos, report random"); T1 within 1e-3 relative (*starting*). **Measured (§3.8c, F-04d): T1 5.7e-8 and the front of the chain 0.13 code met; met for all 16 stock × output pairs, sRGB included (every 6th pixel: largest 0.07 to 0.34, 99.9 % within 0.05 to 0.20), with T3b′ 129³ (the scan) and the per-pixel return; random colours reported (§3.8c)** | F-04 |
| Variant A | `forward(inverse(d))` against `d` on the reachable colours; **the share of print pixels with no pre-image** on 20 public photos × the 8 most used stocks, and ΔE00 of those | round trip 1e-6; unreachable share ≤ 0.1 % (*starting*); above it, reconsider B or A+. **Measured (§3.8a, §3.8c): the per-pixel return round-trips to 2.7e-11 (the continuous inverse: 2.4e-7); unreachable share 0 of 1.49 million pixels for each of the 8 stocks, P3 and sRGB; met** | F-04 |
| GPU parity | each pass against its CPU twin on lavapipe: colour passes in LogC4; the blurs; the grain | 2e-4 · one 10-bit code after C13; blurs 1e-4 of the pass's peak (*starting*); grain hash exact, value 1e-5 | F-08 |
| Determinism and tiles | a repeat render is identical; tiles equal the whole frame bit for bit **including halos and grain**; a poison test (`docs/17` D7); the far corner of an 8,658 × 5,772 frame (D4) | bit for bit. **CPU: met (F-05, §3.9a)**; the GPU is F-08 | F-05, F-08 |
| App against `ym` | byte-identical exports of a graph with a Film node, masks and grain on | byte-identical | F-07, F-08 |
| Preview against export | at 1:1 the tile equals the export; at k = 4 the halation and coupler diffusion of the preview against the reduced export; the **variance** of the grain against the reduced export's | largest 0.02, mean 0.002 in display linear; variance within 10 % (*starting*). **CPU: met (F-05, §3.9a): halation and diffusion 0.0001 / 0.00001, grain variance ratio 0.996 to 1.008**; the GPU is F-08 | F-05, F-08 |
| Provenance | the shingle and structural check (`tools/provenance/shingles.py`) against spektrafilm, spektrafilm-rs and vkdt's filmsim; headers in every module | no shared sequence beyond trivial ones | F-06 |

**Provenance result (F-06, 2026-11-04; 8-token shingles, identifiers, numbers and strings normalised; `crates/ym-film/src` as of the F-03b commit `4c7bd7b`).** Commits compared: spektrafilm `3bb2c2d2801ff68b92019cf1dbcbb133d60832bc`, spektrafilm-rs `9dd59b0380194b93686aaa230a8bb9680aa270a4`, vkdt `938adb0d1ffb128b2dd76f4dc33859a1fa974246` (`src/pipe/modules/filmsim`). Share of our 8,877 distinct shingles found anywhere in the tree: spektrafilm 3.4 %, vkdt filmsim 2.1 %, spektrafilm-rs 44.0 %; the same count against our own unrelated `ym-color` is 52.2 %, so the Rust port's figure is the chance level for Rust of this style. Shared non-trivial sequences of 24 or more tokens: spektrafilm 0, vkdt 0, spektrafilm-rs 220 (14 of 48 or more, 1 of 96 or more) against 601 (67, 12) for `ym-color`. Skeleton (keywords and brackets, 16-token shingles) share: 0.06 %, 0.06 %, 4.7 % against 9.5 % for `ym-color`. The longest sequences against spektrafilm-rs are serde struct declarations (the profile format fixes the field list), `pub mod` lists, and the grid-index and 2-D vector idioms of `scene.rs`, `spectra.rs` and `develop.rs`; none is beyond trivial. The full report is regenerated by the command in `tools/provenance/README.md`; the clones never enter the repository.

**Provenance rerun (F-04e, 2026-11-04; same method and commits; the 21 files of `crates/ym-film/src/**` (17) and `crates/ym-color/src/display/cine/**` (4) at the working tree after F-04d, F-06b and F-04e; the report is kept outside the repository).** Share of our 19,801 distinct shingles found anywhere in the tree: spektrafilm 2.8 %, vkdt filmsim 1.9 %, spektrafilm-rs 35.0 %; against our unrelated `ym-color` 58.7 % (the chance level for Rust of this style). Shared sequences of 24 or more tokens that the script does not call trivial: spektrafilm 0, vkdt 2, spektrafilm-rs 341 (20 of 48 or more, 2 of 96 or more) against 932 (154, 26) for `ym-color`. Skeleton (16-token shingles) share: 0.06 %, 0.07 %, 3.3 % against 30.3 %. What the sequences are: (a) against spektrafilm-rs, serde struct declarations with `#[serde(..)] pub field: Type` lists (`profile.rs`, `params.rs`: the profile and parameter formats fix the field lists), one `pub mod` list in `lib.rs`, a `bake.rs` struct pair of fixed arrays, and a 2-D vector idiom in `scene.rs` (the vector from the white to a point, its length, a guard on a small length; the same idiom as before); (b) against spektrafilm-rs, `cine/consts.rs` lines 8-123 as one 486-token run: 64 lines of `pub const NAME: f64 = number;` that match only after names and numbers are normalised; the file is generated from our own `training/cine/constants.json` and the values are ours; (c) against spektrafilm-rs, `cine/core_table.rs` lines 417-422 (63 tokens: `let` bindings with `.len()`, `.ln()`, `.cos()`, `.sin()` calls and a closure; ordinary numerical Rust); (d) against vkdt, two 24-token runs of identifiers and commas closing an array or struct literal (`cine/core_table.rs`, `cine/inverse.rs`) with `filmsim/main.c`. None copies logic; (b) and (c) are above the script's "trivial" line by token count only and are listed for the owner's judgement (no clone file was read by the engineer).
| Licences | `check_bundle.py`: `LICENSE`, `ATTRIBUTION.md`, hashes and the `NOTICE` item; no path under `training/` mentions `data/film/`; no UI string, setting or command contains "spektrafilm" outside the credit lines; the profiles are unmodified | pass | F-02, F-10 |
| Interface | e2e for the node's tabs, the pickers, the slide case, the credit line and About; the glass audit gains the states; screenshots, which the owner looks at before they are committed | green | F-09 |
| CLI | `docs/10`, the help and `ym film list` agree (`crates/ym-cli/tests/docs.rs`) | the test fails on any difference | F-07 |
| Speed (on a Mac) | a slider move in the fit view with a Film node and 7 others; the full-resolution export of a 24 MP photo (reported) | ≤ 100 ms (*starting*, the node view's target) | F-10 |

## 9. Risks and open questions for the owner

Each has a recommendation. The owner decides.

1. **The author's view of "inspired by its methods".** The README and `CITATION.cff` say a derivative includes a tool "directly inspired by its methods", and ask for the same open licence. Copyright protects code, not methods, and we write from the papers and the data sheets (§3.1), but this is the author's stated view and the main legal risk of the plan. **Recommend:** the owner writes to him before F-03 (§7); keep the clean-room rule and the shingle check; if he objects, the plan stops or changes. Not legal advice; a lawyer's view can come with the trademark check (M4-H3).
2. **The licence's wording disagrees with itself.** The README says "strict 'commercial use, free share, no resale'"; the file says the no-resale and no-training wishes are "not legal requirements". **Recommend:** honour them as binding (§6.3) and ask the author to align the wording (§7).
3. **Variant A or B (§2).** **Recommend A**, with A+ if the unreachable share of F-04 is not small. A needs the inverse in a table, loses a few saturated colours and ties the table to the output space; B needs a format change, drops masks and nodes after Film and cannot go in the DNG.
4. **Trademark names (§6.5).** Options: factual names with a line, neutral names, or a mix. **Recommend option 1**, with the names in one data file so option 2 is a one-file change, and a lawyer's look with M4-H3. **The owner decides.**
5. **The upsampling table.** The author's has no licence. **Recommend:** make our own (F-03), and ask hanatos in parallel (§7). Risk: the table is a real piece of work and chromaticities near the locus may be reached only by inpainting.
6. **Coupler, grain and halation parameters are not in the profiles.** They are the author's choices in GPL code ("arbitrarily guessed"). Ours will differ, so the saturation and grain of our node will differ from his. **Recommend:** accept it ("modelled on published data"), tune by eye on public photos in the mockup round, compare the oracle only with the effects off, and ask for his numbers (§7, item 2).
7. **ShareAlike reaches what we derive.** A baked table exported as a file, a fitted parameter file, an oracle fixture, a LUT export: all are Adapted Material. **Recommend:** no LUT export ever (§6.3); fixtures and fitted files are separate CC BY-SA 4.0 files with credit (§6.7).
8. **The profiles' data comes from Kodak's and Fujifilm's data sheets** ("Original data are property of the respective holders"). We rely on the author's licence and on curves being facts; this is not legal advice. **Recommend:** put it to the lawyer with the trademark check; the profiles are data files, so one can be dropped without a code change.
9. **The CIE tables' licence** (observer, D50, D55). Recorded as CC BY-SA 4.0 in `docs/14`; not re-checked; open as M1-H4. **Recommend:** ship them as a separate CC BY-SA 4.0 file with credit, as the profiles, and answer M1-H4 for both together.
10. **Defaults.** The default stock (Portra 400), and whether brightness is "as the app" (recommended, so the picture does not jump) or "as printed". **Recommend:** both as proposed; ask in the mockup round.
11. **Speed.** The Film node is several passes with halos of a few percent of the long edge; the CPU fallback will be slow. **Measured in F-05 (§3.9a): 35 s a 24 MP frame on one CPU thread, 15 s on four.** **Recommend:** measure again in F-10; the node says when it runs on the CPU.
12. **Size.** The data adds 5.5 MiB (1.1 MB in the disk image) plus a small table of ours. **Recommend:** commit it.
13. **The cine print films** need a projector-lamp spectrum we have not sourced. **Recommend:** hide 2383 and 2393 until it is, or choose a published stand-in.
14. **Scheduling.** The Film node is after milestone N. **Recommend:** schedule it when 0.2.0 is out, and start with the letter and the mockup round (F-H2, F-01), which are the cheapest steps and the ones whose answers can change the plan.

## 10. Tasks

Task format of `docs/15`. **Milestone F (planned, unscheduled)** follows milestone N and needs its graph (N-02 to N-04), its panel (N-08, N-09), its masks (N-10, N-12) and its tile-halo machinery (N-13). Nothing here starts until the owner schedules it (F-H0).

**[HUMAN] F-H0 · Schedule.** The owner decides when; nothing runs before 0.2.0 is out.
**F-01 · Mockups and the revised spec** (F-H0, N-09): the mockup round of §4.4 in `design/film/` (the Add node entry, the five tabs, the pickers, the slide case, the credit line, About, the Output tab's note); this file revised to match. *Done when:* the owner approves (F-H1).
**[HUMAN] F-H1 · Approve the mockups; decide variant A or B and the stock names (§9, items 3, 4, 10).**
**[HUMAN] F-H2 · Write to the author and to hanatos (§7).** Before F-03, and F-03 waits for the answer (or for the owner's decision after a month of silence). The answer may change F-03 and F-05.
**F-02 · Data and licences** (F-H1): `data/film/spektrafilm/` with `LICENSE`, `ATTRIBUTION.md`, `SHA256SUMS` from a pinned commit; `tools/film/update-profiles.sh`; the `NOTICE` item; About and `THIRD-PARTY.md` entries; `check_bundle.py`; the CIE decision (§9, item 9); `docs/14` rows move from "planned" to "ship"; the names file. *Done when:* the hash and licence checks pass, a missing `LICENSE` or `NOTICE` item fails them, and no path under `training/` reads `data/film/`.
**F-03 · The engine's colour reference** (F-02, N-03): `ym-film`: profile reader; the upsampling table generator (`tools/film/upsampling/`, F-03a) and its data (`data/film/upsampling/`, CC BY-SA 4.0); stages 1 to 10 in f64 with taps; the neutral solve; the brightness match; provenance headers. *Done when:* the profile-reader, published-curves and neutral gates of §8 pass.
**F-04 · Baking and the return** (F-03): T1, the curves, T3a, T3b′ (the scan) and the per-pixel return with its core table (F-04d); tetrahedral CPU twin; keys, the LRU, draft resolution; bake times measured; the unreachable share measured on 20 public photos. *Done when:* the baking and variant-A gates pass and the numbers are in `docs/STATUS.md`.
**F-05 · The spatial passes on the CPU** (F-03, N-04, N-13): halation, the coupler diffusion, grain (counter-based, no transcendentals), photo-relative sizes, halos; tile independence. *Done when:* tiles equal the whole frame, determinism holds and the preview-against-export bounds pass.
**F-06 · The oracle and the provenance check** (F-03): `tools/oracle/spektrafilm/` (pinned commit, a separate process, fixtures committed with their notice); the shingle and structural check. *Done when:* the oracle and provenance gates pass.
**F-07 · The node in the core and `ym`** (F-04, F-05, N-02, N-03): the `film` kind, validation, patches, copy, resolved settings, XMP (`ym:FilmData`), the plan with the Film passes, DNG baking; `ym film list`, the doctor line, `docs/10` and the help. *Done when:* the app-against-`ym` and CLI gates pass.
**F-08 · The GPU passes** (F-04, F-05, N-04): the kernels and their CPU twins; parity, tiles, determinism on lavapipe. *Done when:* the GPU, tiles and determinism gates pass.
**F-09 · The interface** (F-01, F-07, N-08, N-09, N-12): the menu entry, the panel, the pickers, the credit line, About, the Output note, e2e, the glass audit, the screenshots. *Done when:* e2e passes and the owner has seen the screenshots.
**F-10 · Finish** (F-08, F-09): speed on a Mac, docs (`docs/02`, `01`, `11`, `17`, `23` §7.4), `docs/STATUS.md` (the Mac-only checks under *Waiting on the owner*, M0-H4), `CHANGELOG.md`, README credits, the trademark check. *Done when:* the gates are in `docs/STATUS.md`.
**[HUMAN] F-H3 · Test build and review on a Mac.** **F-H4 · Trademark and data-rights check** (with M4-H3).

**Order.** F-H0, F-01, F-H1, F-H2 first; F-02 then F-03; F-04, F-05 and F-06 alongside once F-03 is done; F-07 and F-08 after; F-09 once F-07 is done.

**Gate F:** the gates of §8 pass; the credit lines are in `NOTICE`, About and the node; the owner's approval of the last test build.
