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

- **The inverse is numerical and slow**: 1.65 µs a colour, 0.61 Mpx/s a thread, so about 40 s for 24 MP on one thread. It cannot run per pixel. It runs **at the nodes of the baked table** (§3.8): a 49³ table needs 117,649 inverses, about 0.2 s once per paper and output space.
- **The rendering is not onto.** About 5 % of random 8-bit sRGB colours have no pre-image (bright, strongly coloured: display luminance about 0.5 to 0.6, a channel near the edge). The inverse returns the nearest reachable colour: the same luminance and hue, less chroma (ΔE00 from the request: median 4.6, 95th percentile 13.2, maximum 15.7). None of the 1.7 million pixels of the eight test photos' own renders is unreachable, but **a film print's colours are not the rendering's own outputs, and their share is not measured**. F-04 measures it on 20 public photos and the gate of §8 decides (reconsider B or A+ if it fails). Highly saturated stocks (Velvia, Ektar) may be the cases.
- **Round trip**: for a reachable colour `forward(inverse(d))` equals `d` within 1e-6 in linear light, far below one 10-bit code (about 1e-3). The table adds its own interpolation error (§3.8), which has a starting bound of one 10-bit code.
- **What clips**: display values are held to [1e-6, 1]; a paper black is above that and the white is exactly 1. Display white maps to the flat top of the rendering, about +10.4 stops over grey (about 243 linear), inside LogC4's range (469.8). Nothing clips in the encode. A node after the Film node with a large gain (say +4 EV) clips at LogC4's top like after any node.
- **The table depends on the output's primaries.** The rendering's gamut stage runs on each output's own primaries (Rec.709, Display P3, Rec.2020), so the inverse is specific to one. The viewport (Display P3) and an sRGB export have different print tables (§3.8), and a colour at the edge of sRGB may differ between them as it does for every picture.
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
| 9 | Scan | per pixel, **table T3b** (3D) | `D_p` | display colour, output primaries | the paper's dyes and base; viewing illuminant; CIE observer |
| 10 | Return (A) | in **T3b** | display colour | LogC4 / AWG4 | `CineRendering::inverse` |

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

**The luminance.** A chromaticity gives two equations for three unknowns; the paper has three (an RGB triple). The third is the luminance: each cell is fitted at a target `Y_t(x, y)` that falls smoothly from 0.5 at the white point to near 0 at the locus (`0.5 (1 − r)^1.5 + 0.004`, `r` the distance from white relative to the locus along the same ray), because a sigmoid spectrum is bounded by 1 and a saturated colour cannot be bright. A cell that does not converge at `Y_t` is retried at fractions of it. The fit fixes only the *shape* of the spectrum; the engine scales it by the colour's real luminance (below), so `Y_t` never reaches the image.

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

**Measured** (F-03a; numbers in `docs/STATUS.md`): 21,712 of the 65,536 nodes lie inside the locus; **21,575 (99.37 %) are solved and 137 (0.63 %) inside the locus are inpainted** (the thin rim at the spectral locus, the violet end and the green side, where a sigmoid cannot be nearly monochromatic enough); the other 43,824 nodes are outside the locus and inpainted. The **worst solved node misses its chromaticity by 9.1·10⁻⁵** (at x = 0.031, y = 0.788, at the locus; the float32 coefficients limit it, and a node that float32 would push past 1e-4 is given to the inpainting; the median is 2.4·10⁻⁸). The generator runs in about 75 s on one core and is deterministic: two runs give identical bytes. **Grid size:** the lookup's chromaticity error between nodes falls with the square of the cell size (worst, 99th percentile): 64 cells 2.3·10⁻³, 7.2·10⁻⁴; 128 cells 7.2·10⁻⁴, 2.0·10⁻⁴; **256 cells 2.7·10⁻⁴, 5.0·10⁻⁵ (shipped)**; 512 cells 8.5·10⁻⁵, 1.3·10⁻⁵ (4 MB, 5 minutes to generate). At the ten white points of the test (D50, D55, D65, E, blackbodies of 2700, 2856, 3200, 4000, 5000 and 6500 K) the interpolated spectrum has the right chromaticity within 1.2·10⁻⁵ and is smooth (largest second difference at 5 nm under 0.3 % of the peak). 256 is enough for the 1e-3 bound on T1 (§8); a larger grid is not needed.

**Other steps:** the chromaticity of colours outside the locus (negative or extreme camera colours) is compressed softly toward the film's white; the **window** of the profile (four numbers: the centre and width of a blue edge and of a red edge, a band-pass on the sensitivities, renormalised so the reference light gives the same raw values) is applied; the **surface** correction (fifteen numbers per layer) is **off**, as it is by default in the author's tool, and stays in the data. The film is balanced for its reference light: a neutral gives equal raw values. The window was fitted by the author against his own table; ours differs a little, and the first oracle stage (§8) is where that shows. The profile stays unmodified.

### 3.4 Film: curves, couplers, grain, halation

- **Curves.** The profile's `density_curves` (256 points over log exposure −3 to 4) become one-dimensional tables, monotone cubic, 4,096 entries per channel. The largest density of a channel lies between 1.5 and 2.9 for the negatives.
- **DIR couplers.** A developer inhibitor is released in proportion to the density formed in each layer and lowers the effective exposure in its own layer and in the others, after a short diffusion in the plane (Hunt, chapter 15). The curves the engine reads are the **curves without couplers**, found from the profile's curves so that **a neutral ramp still gives the profile's curves**; saturation and local contrast are what the couplers add. The coupler matrix and diffusion are ours (§9, item 6).
- **Grain.** The standard particle model: the number of developed particles in a pixel follows the density and the pixel's area against a particle's area, so the noise in density has a variance of the binomial kind, with sublayers (the profile's `density_curves_layers` give the fast and slow parts) and a dye-cloud blur. Amplitude is calibrated to published RMS granularity (Selwyn's law). The generator is **counter-based and has no transcendental calls** (§3.9).
- **Halation.** Light that passes the emulsion reflects from the support back into it (strength set by the profile's `antihalation`: strong, weak or none) and scatters in the emulsion. A blur of a few Gaussians and a tail, in micrometres on the film, applied to the exposure.

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

### 3.8 Baking

The per-pixel colour part is baked on the CPU, in f64, into tables stored as f32, whenever its inputs change. The app and `ym` evaluate the same tables, which is why their exports stay byte-identical (`docs/17` D8). The tables are in memory only (hard rule 4); the cache is an in-memory LRU.

| Table | Maps | Size (starting) | Depends on | Re-baked when |
|---|---|---|---|---|
| **T1** exposure | chromaticity square (2D) → raw exposure per unit scale (3) | 192 × 192 × 3 f32 = 432 KiB | the film, its window, the upsampling table | the stock changes |
| **Curves** | log exposure → density, per channel; the film's with and without couplers, and the paper's | 9 × 4,096 f32 = 144 KiB | the film, the paper | the stock or paper changes |
| **T3a** enlarger | negative density (3D) → print log-exposure (3) | 49³ × vec4 f32 = 1.88 MB | the film, the paper, the **lights** | the stock, the paper or the lights change |
| **T3b** scan and return | print density (3D) → LogC4 / AWG4 (3), inverse included | 49³ × vec4 f32 = 1.88 MB | the paper, the **output primaries** | the paper or the output space changes |

**Why two 3D tables and not one.** Grain acts on the negative's density (between stages 5 and 7), and the paper's curves sit between the enlarger and the scan. Splitting there keeps each table smooth, **lets the print exposure and the film exposure move without a re-bake** (the film exposure multiplies T1's result, the print exposure shifts the log-exposure at stage 8), and re-bakes only T3a when the lights move. A graph with halation, grain and coupler diffusion all off could use one table, but the split costs two lookups and nothing else, so there is one path.

**Interpolation.** Tetrahedral from an f32 storage buffer, as the look LUTs (`docs/02` C10), never hardware filtering. T1 is bilinear. The axes of T3a run from 0 to the film's largest density (plus a small margin) in each channel, those of T3b from 0 to the paper's; a value outside is held at the edge.

**Bake time (an estimate by counting operations and from the measured 1.65 µs inverse; F-04 measures).** Per node of T3a about 2 µs (81 exponentials and three small sums); per node of T3b about 5 µs (81 exponentials, the XYZ sum, the inverse). T3a: about 0.25 s on one thread, about 0.07 s on four. T3b: about 0.6 s on one thread, about 0.15 s on four, built in the background when the paper is chosen and **cached per paper and output space**. T1: about 20 ms. **While a lights slider is dragged** T3a is baked at 17³ (4,913 nodes, a few milliseconds) and refined to 49³ about 150 ms after the last change.

**Error bound against direct evaluation** (starting numbers, which F-04 measures and reports, never loosens): the table chain against the direct f64 chain on 100,000 random colours and every pixel of the test photos: at most **one 10-bit code** after C13 (largest), and 99.9 % within a quarter code; T1 within 1e-3 relative. The size (33, 49 or 65 points) is the smallest that meets it. The nearest-reachable rule makes T3b non-smooth at the edge of the reachable set; if that edge fails the bound, T3b is refined there or the colours are pulled into the reachable set smoothly before the table.

**Cache.** An in-memory LRU of eight entries per kind. Keys: T1 (film id, profile SHA-256, engine version), curves (stock ids, SHA-256), T3a (film, paper, lights, SHA-256, engine version), T3b (paper, output primaries, SHA-256, engine version). Nothing on disk.

**Determinism.** The bake parallelises over nodes, each node independent, so any thread count gives the same bytes. The profile data version (§6.6) and the engine version are part of every key and of the export's metadata (`ym:FilmData`).

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

### 3.10 Cost

The per-pixel work is about 200 floating-point operations, three table lookups (one 2D, two 3D tetrahedral) and nine one-dimensional lookups. The spatial passes are separable Gaussians and one hash. On the device this is milliseconds at a preview size; on the CPU fallback it is slow, and the node says so (`viewport_busy`, `docs/11`). Speed targets are in §8; nothing is measured yet.

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
- `PIPELINE_VERSION`: no existing graph changes by a bit. The version still gets a minor step when the node lands, because an older build refuses a graph with a `film` node (F-07 decides with the owner).

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
| **Coupler, grain, halation, glare, filter parameters** | stages 3, 5, 6, 7 | ours: from the literature and by eye, in `data/film/params/` (Apache-2.0). A file fitted to a profile's content is an adaptation and is CC BY-SA 4.0 with its own `LICENSE` and a `CHANGELOG.txt` entry; one chosen by hand is Apache-2.0 |
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
| **Oracle** | spektrafilm's Python (a pinned commit, a separate process, **never a dependency**) on 24 chart patches, a 125-point RGB lattice and a 21-step grey ramp, **with couplers, halation, grain and glare off** (its per-pixel mode), compared at its named taps: `log_e_film`, `cmy_film`, `cmy_print`, `rgb_out`. Fixtures are committed (they are Adapted Material: CC BY-SA 4.0 with credit); the generator is `tools/oracle/spektrafilm/` | per stage: 0.03 log10 / 0.03 density (*starting*); final ΔE00 median ≤ 1.5, 95th percentile ≤ 4 (*starting*) | F-06 |
| Baking error | the table chain against the direct f64 chain on 100,000 random colours and all pixels of the test photos | largest ≤ one 10-bit code after C13; 99.9 % within a quarter code; T1 within 1e-3 relative (*starting*) | F-04 |
| Variant A | `forward(inverse(d))` against `d` on the reachable colours; **the share of print pixels with no pre-image** on 20 public photos × the 8 most used stocks, and ΔE00 of those | round trip 1e-6; unreachable share ≤ 0.1 % (*starting*); above it, reconsider B or A+ | F-04 |
| GPU parity | each pass against its CPU twin on lavapipe: colour passes in LogC4; the blurs; the grain | 2e-4 · one 10-bit code after C13; blurs 1e-4 of the pass's peak (*starting*); grain hash exact, value 1e-5 | F-08 |
| Determinism and tiles | a repeat render is identical; tiles equal the whole frame bit for bit **including halos and grain**; a poison test (`docs/17` D7); the far corner of an 8,658 × 5,772 frame (D4) | bit for bit | F-05, F-08 |
| App against `ym` | byte-identical exports of a graph with a Film node, masks and grain on | byte-identical | F-07, F-08 |
| Preview against export | at 1:1 the tile equals the export; at k = 4 the halation and coupler diffusion of the preview against the reduced export; the **variance** of the grain against the reduced export's | largest 0.02, mean 0.002 in display linear; variance within 10 % (*starting*) | F-05, F-08 |
| Provenance | the shingle and structural check against spektrafilm, spektrafilm-rs and vkdt's filmsim; headers in every module | no shared sequence beyond trivial ones | F-06 |
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
11. **Speed.** The Film node is several passes with halos of a few percent of the long edge; the CPU fallback will be slow. **Recommend:** measure in F-05 and F-10; the node says when it runs on the CPU.
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
**F-04 · Baking and the return** (F-03): T1, the curves, T3a, T3b with the inverse; tetrahedral CPU twin; keys, the LRU, draft resolution; bake times measured; the unreachable share measured on 20 public photos. *Done when:* the baking and variant-A gates pass and the numbers are in `docs/STATUS.md`.
**F-05 · The spatial passes on the CPU** (F-03, N-04, N-13): halation, the coupler diffusion, grain (counter-based, no transcendentals), photo-relative sizes, halos; tile independence. *Done when:* tiles equal the whole frame, determinism holds and the preview-against-export bounds pass.
**F-06 · The oracle and the provenance check** (F-03): `tools/oracle/spektrafilm/` (pinned commit, a separate process, fixtures committed with their notice); the shingle and structural check. *Done when:* the oracle and provenance gates pass.
**F-07 · The node in the core and `ym`** (F-04, F-05, N-02, N-03): the `film` kind, validation, patches, copy, resolved settings, XMP (`ym:FilmData`), the plan with the Film passes, DNG baking; `ym film list`, the doctor line, `docs/10` and the help. *Done when:* the app-against-`ym` and CLI gates pass.
**F-08 · The GPU passes** (F-04, F-05, N-04): the kernels and their CPU twins; parity, tiles, determinism on lavapipe. *Done when:* the GPU, tiles and determinism gates pass.
**F-09 · The interface** (F-01, F-07, N-08, N-09, N-12): the menu entry, the panel, the pickers, the credit line, About, the Output note, e2e, the glass audit, the screenshots. *Done when:* e2e passes and the owner has seen the screenshots.
**F-10 · Finish** (F-08, F-09): speed on a Mac, docs (`docs/02`, `01`, `11`, `17`, `23` §7.4), `docs/STATUS.md` (the Mac-only checks under *Waiting on the owner*, M0-H4), `CHANGELOG.md`, README credits, the trademark check. *Done when:* the gates are in `docs/STATUS.md`.
**[HUMAN] F-H3 · Test build and review on a Mac.** **F-H4 · Trademark and data-rights check** (with M4-H3).

**Order.** F-H0, F-01, F-H1, F-H2 first; F-02 then F-03; F-04, F-05 and F-06 alongside once F-03 is done; F-07 and F-08 after; F-09 once F-07 is done.

**Gate F:** the gates of §8 pass; the credit lines are in `NOTICE`, About and the node; the owner's approval of the last test build.
