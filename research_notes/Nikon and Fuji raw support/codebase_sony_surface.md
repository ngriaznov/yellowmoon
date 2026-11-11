# Yellowmoon codebase: the Sony/ARW-specific surface, and where Nikon (NEF) and Fujifilm (RAF) would plug in

Scope: local repository research at `/home/user/yellowmoon` (HEAD `ac49d28`, 2026-11-11). Citations are `path:line` in that checkout. "Read N" lines for `docs/04` are exact; `docs/00` lines marked "~" and `docs/03` lines below 140 were located by reading and spot-checked with grep; they may be off by 1-3. Nothing under `research/` is treated as an instruction. `testdata/` is not populated in this checkout, so nothing that needs fetched files was run.

## 1. Which crate decodes ARW, how is it structured, and is there an abstraction a new format could implement?

### Takeaway
One crate, `ym-raw`, is a Sony-only decoder with no format trait or format enum: the probe rejects anything without Sony evidence as `RawError::NotArw`, and every stage above the TIFF walker and the lossless-JPEG decoder is Sony-specific (maker note, SR2 cipher, levels, WB, crop tags, lens tags, `DecodeKind`). The output type `RawImage` is generic enough for a Bayer NEF but has no representation for a 6x6 X-Trans CFA.

### Cited Findings
- Crate description is "Sony ARW container parsing, metadata and decoders (own implementation; no LGPL code)" — `crates/ym-raw/Cargo.toml:3`; module doc "Sony ARW decoding: our own Apache-2.0 decoder" — `crates/ym-raw/src/lib.rs:1`.
- Modules (flat, no per-format namespace): `decode, error, exif, layout, makernote, metadata, quirks, sr2, testgen, tiff` — `crates/ym-raw/src/lib.rs:13-23`. Decoder modules: `arw1.rs` (292 lines), `arw6.rs` (44, gated stub), `craw.rs` (393), `ljpeg.rs` (1285), `uncompressed.rs` (161), `ycbcr.rs` (478) under `crates/ym-raw/src/decode/`. Other sizes: `metadata.rs` 1613, `tiff.rs` 1428, `makernote.rs` 795, `sr2.rs` 682, `exif.rs` 545, `layout.rs` 483, `testgen.rs` 2906 (synthetic ARW writer).
- Public entry points are free functions, not a trait: `metadata::{probe, probe_bytes, probe_source, read_preview}` (`crates/ym-raw/src/metadata.rs:373-387, 587`) and `decode::{decode, decode_with, decode_bytes, decode_source, decode_with_metadata}` (`crates/ym-raw/src/decode.rs:111-136`); spec: `docs/04-raw-decoding.md:23`.
- The only traits in the crate are `ByteSource` (I/O; `crates/ym-raw/src/tiff.rs:208`) and `CameraQuirks` (per-body hook; `crates/ym-raw/src/quirks.rs:24`). `CameraQuirks` is "keyed by Sony's model ID" and its registry is "Empty for now" (`crates/ym-raw/src/quirks.rs:6, 51-56`); spec `docs/03-camera-profiles.md:361-372`.
- Probe is hard-wired to Sony: it computes `is_sony_make`, parses the maker note only for the Sony family, returns `NotArw` when `DNGVersion` is present or the maker-note format is SR2 (`crates/ym-raw/src/metadata.rs:384-400`); `is_sony_family` accepts `SONY` and the Hasselblad HV/Stellar/Lusso/Lunar rebadges (`crates/ym-raw/src/metadata.rs:574-582`). Spec of "Which file is an ARW": TIFF with a raw directory "*and* Sony evidence"; "non-TIFF input and non-Sony raw files are `NotArw`" — `docs/04-raw-decoding.md:40`.
- The error enum bakes the scope into user text: `NotArw` = "This isn't an ARW file. Yellowmoon opens Sony ARW files only." (`crates/ym-raw/src/error.rs:20-22`), kind string `not_arw` (`error.rs:112`); the closed list of messages is in `docs/04-raw-decoding.md:152-173`.
- Output type (`docs/04-raw-decoding.md:8-18`; `crates/ym-raw/src/decode.rs:52`): `RawImage { meta, layout, data: Plane<u16>, black: [f32;4], white: [f32;4], wb_as_shot: Option<[f32;4]>, qstep, lens: Option<LensTags>, tone_curve, preview }`. Comments name Sony tags for each field (SR2 0x7313/0x7303, 0x7032/0x7035/0x7037, 0x7010).
- `Layout` has exactly two variants: `Bayer { pattern: CfaPattern, active_area, crop }` and `Ycbcr { subsampling }` (Sony Lossless M/S) — `crates/ym-raw/src/layout.rs:271-289`. `CfaPattern` is `pub struct CfaPattern(pub [CfaColor; 4])`, a 2x2 pattern; doc says "Sony bodies write RGGB" (`crates/ym-raw/src/layout.rs:147-152`); helper `rggb_to_positions` maps "the order Sony stores" (`layout.rs:208-212`).
- `DecodeKind` variants are all Sony compression variants (`Uncompressed12 | Uncompressed14 | Craw | LosslessL | LosslessMS | Arw1 | Arw6 | Unknown`), chosen by `metadata::classify` from TIFF compression codes 1, 7, 32766, 32767 — `crates/ym-raw/src/layout.rs:293-350`, `crates/ym-raw/src/metadata.rs:760`, `docs/04-raw-decoding.md:56-77`.
- Container: little-endian TIFF, IFD0 preview, raw in SubIFD; reader "walks all IFDs and picks the one with Sony raw compression and the largest dimensions" — `docs/04-raw-decoding.md:34`. `Tiff::open` accepts `II*\0` or `MM\0*` (both byte orders) per `docs/04-raw-decoding.md:40`; `ByteOrder` enum at `crates/ym-raw/src/tiff.rs:168`; limits (120 MP, 2048 entries, 64 directories, depth 4, 64 MiB per probe) at `docs/04-raw-decoding.md:42`.
- Metadata/makernotes: ISO from enciphered Sony tables 0x9416/0x9405 with cipher `c = b^3 mod 249` (`docs/04-raw-decoding.md:36`; `crates/ym-raw/src/makernote.rs:98-103, 317`), model ID 0xB001, ARW version 0xB000, focus tags 0x2027/0x204A/0x2037/0x201C (`docs/04-raw-decoding.md:38`; `SonyMakerNote`, `SonyFocus` at `crates/ym-raw/src/makernote.rs:109, 129`).
- Black/white/WB: SR2Private via `DNGPrivateData` 0xC634, Sony pad cipher, tags 0x7310/0x7300 (black), 0x787F (white), 0x7313/0x7303 (WB); fallback raw-IFD copies, then defaults (black 512 cRAW, `128 << (bits-12)`) — `docs/04-raw-decoding.md:44-54`; code `crates/ym-raw/src/sr2.rs:57-290`. Level provenance enum `LevelSource` at `metadata.rs:168`.
- Crop: 0x7038 `SonyRawImageSize` → `active_area`; 0x74C7/0x74C8 → `crop`; DNG default-crop tags as fallback — `docs/04-raw-decoding.md:39`.
- Lens corrections come only from Sony raw-sub-IFD tags 0x7032 (vignetting), 0x7035 (CA), 0x7037 (distortion): `LensTable`/`LensTags` (`crates/ym-raw/src/metadata.rs:127-158`), scales `ca*2^-21, d*2^-14, v*2^-13`, knot positions `(i+0.5)/(n-1)` — `docs/04-raw-decoding.md:138-150`. "If a tag is missing (older bodies, adapted lenses), lens corrections are skipped for that part and the RAW tab says 'No lens data in this file'" — `docs/04-raw-decoding.md:148`.
- Reusable, format-neutral parts: the lossless JPEG decoder is general ("SOF3, DHT … DRI and RSTn, predictors 1–7, point transform, 2 to 16 bits, 1 to 4 components with any sampling factors") — `docs/04-raw-decoding.md:110`; the TIFF/IFD walker and EXIF reader (`tiff.rs`, `exif.rs`).
- Consumers of `ym_raw`: `ym-core` (32 files), `ym-cli` (9), `ym-gpu` (2), `ym-noise` (1), `ym-profiles` (1, doc only), `src-tauri`, `tools/golden`, `tools/oracle` (grep of `ym_raw::` and Cargo manifests).
- Test tooling inside the crate is ARW-only: `testgen::ArwBuilder` writes synthetic ARW incl. enciphered maker note and SR2 (`docs/04-raw-decoding.md:202`); 7 fuzz targets (`tiff_walker, sr2, probe, craw, ljpeg, ycbcr, decode_any`) in `crates/ym-raw/fuzz/fuzz_targets/`; 12 integration tests in `crates/ym-raw/tests/`.

### Inferences
- NEF (TIFF-based, Bayer) fits the existing shape: reuse `tiff.rs`, `exif.rs`, `ByteSource`, `RawImage`, `Layout::Bayer`; add a Nikon maker-note reader, NEF decoders (uncompressed, Nikon Huffman lossless/lossy with linearisation table), level/WB/crop extraction, and new `DecodeKind` variants. The probe's "Sony evidence" gate, `NotArw`, and `classify` have to be restructured into a per-maker dispatch (no existing seam).
- RAF is not TIFF at its outer level (Fuji's own header with offsets to a JPEG, a tag directory and the CFA block); `Tiff::open` cannot be the entry point, so a new container reader is needed in addition to Fuji compression decoders. This statement about RAF is general knowledge, not something the repository documents.
- X-Trans needs a new `Layout` variant (6x6 pattern) and changes to every consumer that matches on `Layout` (see section 3). `black: [f32;4]`/`white: [f32;4]` "per CFA position" indexing (2x2 raster) would also need redefining for a 36-position pattern.
- `qstep` (cRAW quantisation) is Sony-only today; a lossy NEF or Fuji lossy-compressed file would either need its own quantisation map or pass `None`.

### Gaps
- The repository has no description of NEF or RAF file structure beyond the table in `research/canon-nikon-curves-2026-10-19.md:118-124` (NEF only). Nothing on RAF at all.
- Not checked: whether `exif.rs` hard-codes any Sony assumption (only its public items were listed).

## 2. How are camera colour profiles built and stored, and what would a body from another maker need?

### Takeaway
Profiles are data files generated from rawtoaces-data spectral curves by make-agnostic fit code, but the registry, schemas, file layout, detection and display copy are keyed on Sony's maker-note model ID and the literal string "sony". A new body needs: a curve file, a `profile.json` (sensor-map tables, locus), a per-profile illuminant model, a registry entry under a non-Sony key scheme, a colour matrix for the Before view, and optionally a white scale and sensor geometry.

### Cited Findings
- Spec statement: "Adding a camera means adding a profile file and its tests; the pipeline code stays the same." — `docs/03-camera-profiles.md:5`. "Measured curves for the 11 exact bodies are Weta's (rawtoaces-data, Apache-2.0), copied to `data/cameras/sony/`" — `docs/03-camera-profiles.md:81`.
- Layout (`docs/03-camera-profiles.md:263-272`): `data/cameras/registry.json` (model ID → body/profile/match/white scale/geometry; Model-string aliases; default profile), `sony/Sony_<model>_380_780_5.json` (curves + `LICENSE-rawtoaces-data`), `profiles/<profile-id>/profile.json`, `color-matrices.json`, `camera-default-tone.json`, `noise/gain-regimes.json`. On disk: 11 curve files in `data/cameras/sony/`, 11 folders `data/cameras/profiles/sony-*`.
- `registry.json` is version 1.3.0, `default_profile: "sony-ilce-7rm4"`, 11 profiles, 46 bodies, 46 `model_strings` (inspected with Python). Each body entry is keyed by integer `model_id` and has no `make` field (example entry for 363/ILCE-7M3).
- Schemas pin Sony naming: profile `curves` path pattern `^sony/Sony_[A-Za-z0-9-]+_380_780_5\.json$` (`crates/ym-profiles/schema/profile.schema.json:27`); white-scale source file names must match `\.ARW$` (`crates/ym-profiles/schema/registry.schema.json:95`); `model_id` required per body (`registry.schema.json:48-50`).
- Detection (`docs/03-camera-profiles.md:11-17, 257-261`): Make/Model from IFD0, Sony model ID 0xB001 ("the model ID wins"); lookup order: non-Sony Make → default profile "whatever the model ID says"; model ID → registry; Model string → `model_strings`; else default. Code: `is_sony(make)` (`crates/ym-profiles/src/detect.rs:106-108`, "An absent Make is treated as Sony"), `ProfileStore::detect(make, model, model_id: Option<u16>)` (`detect.rs:240-281`). A Canon Make is explicitly tested as not Sony (`detect.rs:426`) and would get the A7R IV default.
- Display copy is Sony-worded in code: "The {short} has no measured profile yet, so it borrows the {profile_name}. {} Sony cameras have their own…" (`crates/ym-profiles/src/detect.rs:362`); `short_name` strips a leading "Sony " (`detect.rs:152-155`).
- Sibling rule: a same-sensor pairing ships only if the bodies' published D65 ColorMatrix agree "within 0.02 per coefficient after normalisation"; a failing pairing becomes `default` (`docs/03-camera-profiles.md:62-77`). Tool: `crates/ym-profiles/tests/py/sibling_matrices.py`.
- White scale of stand-ins is fitted from white-balance presets "every ARW stores … in the SR2 sub-IFD" via `training/white_scale/sr2.py`, or from the published D65 matrix (`docs/03-camera-profiles.md:85-108`); it is `[1,1,1]` for a body the registry does not list.
- Tooling (`training/sensor_map/`): `build_profiles.py` "Write … profile.json for the 11 measured Sony bodies" (`:1`), uses `fit.sony_curves()` / `fit.sony_curves_raw()` and `paths.SONY_CURVES_DIR` (`:198-202`); `registry_data.py` holds the table as data, `DEFAULT_PROFILE = "sony-ilce-7rm4"` (`:16`), `curves_file = f"Sony_{model}_380_780_5.json"` and `f"sony/{…}"` (`:44-49`), bodies keyed by Sony model ID (`:70-95`); `build_registry.py`, `validate_profiles.py`, `gate.py`. `tools/fetch/spectral.sh` pins rawtoaces-data, expects exactly 52 camera curves (`tools/fetch/spectral.sh:77-80`) and byte-compares the 11 shipped Sony files (`:93-100`).
- Per-profile learned model: the illuminant estimator ships "11 per-profile models in `data/models/illum/`" (`docs/07-ml-models.md:9`), "One model per camera profile … Training data are synthetic scenes rendered through each profile's published curves … no photographs and no measurements of our own" (`docs/07-ml-models.md:31`); files `data/models/illum/sony-*.json` (11). Default CLI arg `--profile sony-ilce-7rm4` (`training/illum/train.py:354`).
- Sensor-map gate: 99 CES under 9 held-out lights, per light ΔE00 median ≤ 0.5 and p95 ≤ 1.2 (`docs/02-colour-pipeline.md:221`).
- The ALEXA target is make-independent: PCA prior of "the 52 measured camera curves of rawtoaces-data" (`docs/03-camera-profiles.md:221`).
- Before view depends on per-body published matrices: `data/cameras/color-matrices.json` holds 44 bodies, purpose "The D65 colour matrix of each Sony body", "Read for each body by its Sony model ID", fallback `ILCE-7RM4` (file header fields); `camera-default-tone.json` is "Tone response of the camera's own JPEG engine" fitted for Sony (file header); spec `docs/03-camera-profiles.md:83`, `docs/02-colour-pipeline.md:839-841`.
- "Adding a camera later": "In V1, a camera gets its own profile only if permissively licensed measurements are published (check rawtoaces-data and similar sources at each release)" then run fit scripts, add `profile.json` + registry entry, run gates (`docs/03-camera-profiles.md:378-385`).
- Prior in-repo research already ran the whole profile toolchain on non-Sony curves: rawtoaces-data at pinned commit `e9b8503` has 22 Canon and 11 Nikon curves; Nikon list: Weta D3300, D5300, D800E, D810, D850, Z f; AMPAS D70, D200, D700, D7000; NPL D5100 (`research/canon-nikon-curves-2026-10-19.md:5, 9-15`). It states "the fit, tables and checks are make-agnostic, as this research shows" (`:141`). Script: `training/other_makes/check.py`, "research only, not shipped" (`training/README.md:32, 57`); results in `research/canon-nikon-curves/*.csv` (8 files).
- Nikon numbers from that research: own-curve ALEXA-rendering error Z f 0.27/0.95, D850 0.34/0.86, D810 0.27/0.67, D5300 0.38/0.98 (median/p95 ΔE00) vs A7R IV stand-in 0.81/1.59, 2.09/3.82, 0.48/1.05, 0.55/1.38 (`:67-70`); sensor-map gate: 3 of 11 Nikons pass all but LED-V2 (D7000, D800E, D810); Z f fails D5250 and D6750; D850 fails LED-RGB1 (`:84-86, 93`). Sibling check: among Z-mount bodies only the Z f is measured and no Z body passes the 0.02 rule; Z6III 0.041, Z 5 0.038, Z 6/Z 6II 0.087, Z 8 0.10, Z 9 0.34 (`:107, 110`). Nikon detection would be "Model string only (`NIKON Z 6_2`, `NIKON Z6_3`)" (`:122`).

### Inferences
- The fit maths needs no change for another maker; the work is renaming/generalising: a `make` field (or string key) in the registry and schemas, a `data/cameras/<make>/` curve folder and schema pattern, `fit.sony_curves*` loaders, `spectral.sh` checks, the `u16` model-ID parameter threaded through `ProfileStore::detect`, noise priors and colour-matrix lookup, and the "N Sony cameras" copy.
- Each new measured body also needs one illuminant model trained (synthetic, CPU-scale training; exact time not found) and its gate numbers recorded; a profile that fails the M1 sensor-map gate (e.g. Z f, D850) cannot ship silently because "Gate thresholds change only when the owner agrees" (AGENTS.md "How to work").
- A non-Sony body without curves currently has no sensible stand-in rule: the spec's default is the A7R IV profile and the white-scale mechanism reads Sony SR2 presets. A per-make default profile and a non-SR2 preset reader would be new design decisions.
- For Fujifilm there is no evidence in the repository that any spectral curve is available at all (see Gaps), so every Fuji body would be a stand-in under the V1 data rule unless public curves exist.

### Gaps
- Whether rawtoaces-data's 52 files include any Fujifilm body could not be verified: `testdata/spectral/` is not present in this checkout, and the repository only enumerates the 44 Canon/Nikon/Sony files (`research/canon-nikon-curves-2026-10-19.md:11-15`); the other 8 files are not listed anywhere I found (the ARRI D21 is mentioned at `:9`). No file in `docs/`, `training/` or `research/` mentions a Fuji camera curve.
- The Canon/Nikon research is dated 2026-10-19 and ran on profiles/target before the ALEXA estimate was pushed (profiles 1.2.0, `docs/STATUS.md:46`); its ΔE numbers may have shifted. Not re-run.
- Training time and procedure for one `illum` model were not read in detail.

## 3. Where does the pipeline assume a Bayer CFA, and what breaks on X-Trans?

### Takeaway
Bayer 2x2 is assumed end to end: the decoder's layout type, the GPU/CPU `Cfa` type, R1-R6 raw stages and their WGSL kernels, the noise model's four planes, the demosaic algorithms (spec says "Non-Bayer patterns: Not supported"), the reduced/binned preview path, and the learned demosaic network's 4-plane packed input. X-Trans would need a parallel raw front end (R1-R6) rather than a parameter change; Bayer NEF would pass through unchanged (all four Bayer phases are supported).

### Cited Findings
- GPU-side type: "A 2x2 Bayer pattern … `pattern[y & 1][x & 1]`" `pub struct Cfa { pattern: [[CfaColor; 2]; 2] }` with constants RGGB/GRBG/GBRG/BGGR; RGGB documented as "all current Sony bodies" — `crates/ym-gpu/src/cpu.rs:153-181`.
- Bridge from decoder to pipeline builds a 2x2 `Cfa` from `Layout::Bayer` (and uses `Cfa::RGGB` for `Layout::Ycbcr`) — `crates/ym-core/src/pipeline/raw.rs:128-138`. Other `match raw.layout` sites: `crates/ym-core/src/auto.rs:238-241, 495, 536, 621, 1656-1699`.
- Raw stages table: R1 "Black subtract, white normalise, clip mask | per CFA position, from SR2 levels" (`docs/05-raw-processing.md:11`); "Inputs are plain slices and a 2×2 `Cfa`" (`docs/05-raw-processing.md:5`).
- Highlight reconstruction works on "a half-resolution superpixel image from each 2×2 quad: R, G = mean of the two greens, B" (`docs/05-raw-processing.md:49`); WGSL indexes `pu.cfa[(y & 1u) * 2u + (x & 1u)]` (`crates/ym-gpu/src/shaders/highlights.wgsl:35, 53`); `bin.wgsl:27, 74` likewise. Shaders referencing cfa/phase: `spike_matrix, bin, log_guided, spike_demosaic, mosaic_ca, highlights, spike_normalize` in `crates/ym-gpu/src/shaders/`.
- Demosaic: "All methods handle the four Bayer phases" (`docs/05-raw-processing.md:89`); Menon is a port of `demosaicing_CFA_Bayer_Menon2007` (`docs/05-raw-processing.md:99`); the default adaptive-ratio demosaic lists "Non-Bayer patterns | Not supported." (`docs/algorithms/01-adaptive-ratio-demosaic.md:502`) and pads so "the pattern is always RGGB" (`:173`); the log-guided preview demosaic removes "the Bayer carriers" with K(π)=0 (`docs/algorithms/02-log-guided-demosaic.md:94`); "every algorithm accepts all four Bayer phases" (`docs/algorithms/INTEGRATION.md:45`).
- Lateral CA is corrected on the mosaic per colour lattice (R5b) — `docs/05-raw-processing.md:16, 238`.
- GPU input: `DeviceRaw { words, qstep, width, height, cfa }` (`docs/17-gpu.md:21`); parity tests cover "all four CFA phases" (`docs/17-gpu.md:200`); a raw frame "whose pattern differs from the parameters" is refused (`docs/05-raw-processing.md:320`).
- Noise model: plane index `p = 2 * (y & 1) + (x & 1)` (`crates/ym-noise/src/model.rs:8`), `delta: [_; 4]` (`model.rs:148, 160`); spec: per "CFA plane `c`", `a` "shared by the four CFA planes" (`docs/06-noise.md:11, 17`); `sigma_quad` = "RMS over the four positions of a 2×2 cell" (`docs/06-noise.md:88`).
- Learned joint demosaic+denoise (`jdd`): input "4 packed Bayer planes (R, G1, G2, B, balanced with `g0`), σ map, quantisation step map, clip mask: 7 channels" at half resolution (`docs/07-ml-models.md:58`); tiling relies on packed planes preserving Bayer phase (`docs/07-ml-models.md:67`); training step "Mosaic (RGGB, with random phase)" (`docs/07-ml-models.md:85`; `training/jdd/data.py:63-73`). Status: "runtime, training code, export and parity tests exist; no trained weights (needs GPU time and PixelShift200, M2-H4)" (`docs/07-ml-models.md:14`); `data/models/README.md` says "`jdd/` is empty until M2-06". So the classical demosaic runs everywhere today (`docs/00-overview.md` ~84).
- Training data for `jdd` is Sony-derived: "PixelShift200 (Sony A7R III, 4-shot pixel shift) | CC BY 4.0 | main ground truth" (`docs/07-ml-models.md:73`); quantisation augmentation passes samples through "our test-only cRAW encoder and decoder" with p=0.3 and simulates 12-bit (`docs/07-ml-models.md:87`); noise sampler ranges "set from physics (14-bit Sony sensors, ISO 100–102400)" (`docs/06-noise.md:79`).
- Stages after demosaic are CFA-free: diffraction deconvolution "nothing depends on `Placement` or on the CFA" (`docs/algorithms/03-diffraction-deconvolution.md:65, 576`), output sharpening is "an RGB stage after the demosaic" (`docs/algorithms/04-output-sharpening.md:660`). The existing REDUCED path (`Layout::Ycbcr`, interleaved camera RGB, skips highlight reconstruction and the demosaic network) shows the pipeline can accept already-demosaiced camera RGB (`docs/04-raw-decoding.md:29, 124-127`).
- Other shipped models are not CFA- or maker-specific by construction: `crop` (Open Images), `bokeh` (Open Images synthetic defocus), `sam2`, `depth` (`data/models/README.md`; `docs/14-licensing.md` inputs table). `illum` is per profile (section 2).
- Synthetic test scenes default to `cfa: str = "RGGB"` (`training/synth/recipes.py:49`) and a "14-bit scale of the SR2 levels" (`training/synth/sensor.py:46`).

### Inferences
- Bayer NEF: no pipeline change expected below the decoder; Nikon patterns are covered by the four-phase support. 12/14-bit handling and black/white normalisation are generic once levels are supplied.
- X-Trans (6x6): `Layout`, `CfaPattern`, `Cfa`, `black/white [f32;4]`, R1 clip mask indexing, R4 gains on the mosaic, R5 highlight reconstruction (2x2 superpixels), R5b mosaic CA, R6 (all four demosaic methods), the binned fit-view/thumbnail path (`bin.wgsl`, `reduced_bayer`), the noise estimator's plane model and `sigma_quad`, the `jdd` network input/tiling/training, and every parity/golden test that enumerates four phases would need an X-Trans counterpart. This is the largest single cost of Fuji support.
- A lower-cost first route for X-Trans would be a CPU/GPU X-Trans demosaic that outputs camera RGB and then joins the pipeline where `Layout::Ycbcr` data joins it today (the REDUCED-style entry), at the price of losing mosaic-domain highlight reconstruction, mosaic CA and the joint denoise. This is my inference; the specs do not propose it.
- Fujifilm Bayer bodies (GFX, X-A, X-T100/200 class; general knowledge, not in the repo) would only need the RAF container/decoder, not the X-Trans work.
- The `jdd` network, when trained, will be trained only on Sony A7R III data with Sony cRAW artefacts; applying it to Nikon mosaics is a domain shift the gates do not measure (gates are Sony: "10 A7R cRAW samples", `docs/07-ml-models.md:134`; A1 ISO ladder M2-H5, `docs/15-milestones.md:182`).

### Gaps
- I did not read the WGSL kernels or `ym-noise/src/estimate/*` line by line; the 2x2 assumption is established from types, specs and the grep hits above, not from an exhaustive audit.
- No X-Trans demosaic algorithm, licence survey or prior art exists anywhere in the repo (`grep -i "x-trans|xtrans"` over docs/, research/, crates/ returned nothing).

## 4. Noise model and priors: what is per camera and what is Sony-specific?

### Takeaway
The noise model itself is estimated per photo from its own pixels and is maker-neutral; the Sony-specific parts are the lookup keys (Sony model ID), the body-class guess from Sony model strings, the gain-regime table (Sony bodies only), ISO quirk rules, and the ISO source (Sony maker note). No per-body prior file ships yet.

### Cited Findings
- "V1 has no calibration shoots … every file gets its noise model estimated from its own pixels, with priors built from public files. Nothing is stored between sessions." — `docs/06-noise.md:7`.
- Priors hierarchy: body prior fitted from public files (M2-01) → class prior "derived from physics: full-well capacity 2000 e⁻/µm² times an assumed pixel area per body class" (`docs/06-noise.md:45-48`). `BodyClass::from_model` matches only Sony strings (ILCE-, NEX-, DSLR-A, SLT-, ILCA-, DSC-RX, ZV-, ILME-) and returns `None` otherwise — `crates/ym-noise/src/priors.rs:205-248`.
- Gain regimes: "Sony bodies switch conversion gain at one ISO…" table A1 500, A7 III 640, A7 IV ~400, A7R IV 320, … "Unknown bodies: no break, wide prior. Data overrides the table." — `docs/06-noise.md:50`; file `data/cameras/noise/gain-regimes.json`, every row "docs/06 table, to be confirmed on samples" (`data/cameras/noise/README.md`).
- Per-body prior files are named `<model-id>-<model>.json` with Sony's model ID, and there are "**none yet**: the public corpus … was not reachable when M2-01 was built. A body without a file gets the class prior" — `data/cameras/noise/README.md`; struct fields `model_id: u32 // Sony model ID` (`crates/ym-noise/src/priors.rs:78-79, 286-301`). Generator: `training/noise/public_priors.py` (`docs/06-noise.md:71-77`).
- ISO quirks are Sony lists: extended low ISOs, Hi ISOs above ~16000 (A1, A7R V, A9 III) etc.; "Use the maker-note ISO, since EXIF ISO saturates at 65535" (`docs/06-noise.md:52`). Row noise "usually small on Sony" (`docs/06-noise.md:20`). Filtered-raw check names "Sony's 'star eater'" but is a generic statistic (`docs/06-noise.md:39`).
- Validation uses "ELD's released Sony A7S II parameters (evaluation-only data)" (`docs/06-noise.md:99`).
- `ym-noise` depends on `ym-raw` (`crates/ym-noise/Cargo.toml`, one source file uses `ym_raw::`).

### Inferences
- A Nikon or Fuji Bayer file would work today with the "unknown body" path (no gain break, wide prior) once `BodyClass` can be derived (from sensor size/pixel pitch or a per-make table). Quality cost is a wider prior, not a failure.
- Adding makers means a maker-qualified key for priors and gain regimes (Sony `model_id` is not unique across makers) and gain-switch ISOs for dual-gain Nikon/Fuji sensors from public measurements (numbers only, rule 8).
- X-Trans breaks the four-plane estimator layout (section 3).

### Gaps
- Whether any estimator constant (bin counts, clip thresholds, `qstep` default of 1 DN / 4 DN) silently assumes Sony 12/14-bit scales was not audited.

## 5. Lens corrections, metadata, file-open filters, UI copy, CLI, tests, oracle and fetch tooling: what is hard-coded to Sony/ARW?

### Takeaway
"ARW" is hard-coded as the only file extension at four independent entry points (core session, CLI inputs, CLI preview, the Tauri open dialog) plus the frontend drop filter, and "Sony"/"ARW" appears throughout UI copy, README, cask, schemas, tests and tooling. Lens correction has no source other than Sony's embedded tags ("No Lensfun"), so Nikon and Fuji files would get no vignetting/distortion/CA correction without a new embedded-data reader per maker.

### Cited Findings
**File filters and open paths**
- Core: `fn is_arw(path)` = extension equals "arw" ignoring case (`crates/ym-core/src/session.rs:989-992`), used by `Session::open` (`:1035`); `NO_ARW_MESSAGE = "No ARW files in what you dropped."` (`session.rs:137`). Spec: "Folders are walked for `*.ARW`" (`docs/01-architecture.md:138`).
- CLI: `crates/ym-cli/src/inputs.rs:77` (extension test), problem kind `"not_arw"` (`:178`), `not_arw_message` (`:144-147`); per-file settings keys must end in `.arw` (`crates/ym-cli/src/develop.rs:285-287, 334`); `crates/ym-core/src/cli_preview.rs:193`. Spec `docs/10-cli.md:26, 158, 164, 201`.
- App: open dialog filter `.add_filter("Sony ARW", &["arw", "ARW"])` (`src-tauri/src/commands.rs:105`). No file associations/UTType declarations were found in `src-tauri/tauri.conf.json` or `Info.plist` (grep for arw/sony/fileAssoc/CFBundleDocument returned only the line above).
- Frontend: `src/components/useWindowDrop.ts:144` "True for a Sony raw file name: ends in `.ARW`"; `src/components/DropZone.tsx:13` "ARW only, as many as you like"; `src/screens/Header.tsx:9` "Drop ARW files to start"; `src/state/store.ts:169` `NO_ARW_NOTICE`; `src/api/openMessages.ts:4`.
- Dev/eval helpers filter on "arw": `crates/ym-core/src/pipeline/raw/ca_green_eval.rs:610`, `capture_sharpen_eval.rs:320`.

**UI copy and docs**
- Start screen: "Drop Sony ARW files." and Camera "Any Sony camera · every kind of ARW" (`docs/11-ui.md:45`); non-ARW drop behaviour (`docs/11-ui.md:50`); Before view = "the picture as a Sony camera would produce it by default" (`docs/11-ui.md:67`); camera card copy "Eleven Sony cameras have their own…" (`docs/11-ui.md:101`); About: "Not affiliated with ARRI or Sony…" (`docs/11-ui.md:405`; `src/screens/dialogs/About.tsx:17-24, 88` "A raw processor for Sony ARW files", "Sony camera spectral measurements by Weta Digital"); licensing rule 6 fixes that About sentence (`docs/14-licensing.md:80`).
- README: "Cinema colour for Sony raw files." (`README.md:3`), feature list (`README.md:50-51`), pipeline description "white-balanced Sony RGB" (`README.md:69`); cask `desc "Cinema colour for Sony raw files"` (`Casks/yellowmoon.rb:7`).
- Overview: "A desktop raw processor for photos from Sony cameras" (`docs/00-overview.md` ~7), principle 4 "how the Sony sensor's response is mapped to the ALEXA's" (~25).

**Lens**
- Settled choice: "Lens | Sony's embedded correction tags (vignetting, distortion, CA). No Lensfun." (`docs/00-overview.md` ~86). Stage table: R3 "Sony 0x7032", R5b "Sony 0x7035", R9 "Sony 0x7037" (`docs/05-raw-processing.md:13, 16, 20`); `ym_gpu::cpu::lens::LensTags::from_stored` reads the Sony table layout (`docs/04-raw-decoding.md:148`); oracle is darktable-cli on "synthetic Sony files" (`docs/05-raw-processing.md:37, 287`). Lens corrections are skipped when `!lens_corrections || raw.is_reduced()` (`crates/ym-core/src/pipeline/raw.rs:502`).
- Diffraction deconvolution needs pixel pitch from registry sensor geometry ("Sony's published figures"), with an "EXIF crop-factor fallback" for unlisted bodies (`docs/03-camera-profiles.md:79`).
- Lens *character* (`ym-lens`, `data/lenses/`) is a creative effect on RGB, not maker data (crate listing; `docs/21`).

**Metadata, export, auto**
- Export EXIF test fixtures use "SONY" but the writer copies Make/Model (`crates/ym-export/src/exif.rs:215, 841`; `docs/09-export.md:257`); DNG sets `UniqueCameraModel = "Yellowmoon AWG4"` "so Lightroom doesn't substitute Adobe's Sony profiles" (`docs/09-export.md:186`; `crates/ym-export/src/dng.rs:100`); XMP `ym:Profile` holds the profile id (`crates/ym-export/src/xmp.rs:24`); naming token slug e.g. `sony-a1` from body name (`crates/ym-export/src/naming.rs:39-44`).
- Focus point for auto crop comes from Sony maker-note tags but "is still read … but no longer used" (`docs/20-auto-crop.md:78`).
- Auto exposure reference: "`EV_auto` is the camera's own exposure (a constant +1.11 EV)" measured against the camera's embedded JPEG on Sony files (`docs/13-testing.md:256`); `camera-default-tone.json` uses `CAMERA_GAIN_EV (+1.11 EV)` (file header).
- Embedded preview selection logic is ARW-specific (IFD0 1616x1080 JPEG, maker-note 0x2001) — `docs/04-raw-decoding.md:41`.

**Tests, oracle, fetch**
- `tools/oracle`: "decodes every `*.ARW`" and compares with LibRaw (rawpy), libjpeg and rawler 0.8 in process, "test-only, LGPL, never linked into a shipped crate" (`docs/04-raw-decoding.md:200`); `find_arw_files` (`tools/oracle/src/main.rs:228-239`); comparisons keyed on `DecodeKind::Arw1/Arw6` and Sony stream assumptions (`tools/oracle/src/compare.rs:186-189, 265-277, 473-508`).
- `tools/fetch/samples.sh`: "Download the CC0 Sony ARW corpus into testdata/arw/<model>/" from RawDB and raw.pixls.us (`tools/fetch/samples.sh:2-11`); RawDB API paths `/api/sets/Sony/<model>` (`docs/test-photo-sources.md` Tier 1 table). Other sets: `white-scale-photos.json` (Play Raw ARW), `owner-photos.sh` (seven Sony A1 photos), `real-eval.sh` (nine Sony files, local only) — `docs/test-photo-sources.md`.
- `docs/test-photo-sources.md` is titled "Test photo sources (A1 and siblings)" and its three tiers list only Sony sources; "No test shoots in V1" replacements table at its end.
- `docs/13-testing.md`: corpus "every Sony variant" (`docs/15-milestones.md:50`), gates "Decoder | every corpus file vs LibRaw (and rawler)", "Camera detection | every Sony corpus sample" (`docs/13-testing.md:260-261`); synthetic ARW set of ~61 files generated by `tools/fetch/synthetic.sh` via `training/synth` and `ArwBuilder` (`docs/13-testing.md:222-224`); golden manifests reference ARW inputs (`docs/13-testing.md:296`). Many CI jobs build synthetic ARWs with the `gen_arw` example (`docs/13-testing.md:11`).
- Illuminant real-photo gate already uses non-Sony data: SimpleCube++ "462 Canon 550D/600D photos mapped to each profile" (`docs/STATUS.md:194, 102`).

### Inferences
- Extension handling should be centralised (one list of supported extensions/formats in `ym-core` exposed to CLI, Tauri and frontend) before adding NEF/RAF; today five places would each need editing and the error kind `not_arw` is part of the CLI JSON contract (`docs/10-cli.md:164`).
- All ARW-worded copy is a closed, spec'd list (`docs/04` Errors, `docs/11`), so a multi-maker release is also a spec/copy/mockup change, including the trademark sentence (Nikon, Fujifilm) governed by licensing rule 6.
- Lens corrections for Nikon/Fuji: Fuji RAF and Nikon NEF carry maker-specific correction parameters (general knowledge; not documented in the repo). Under "No Lensfun" the choices are per-maker embedded-data readers written clean-room, or shipping without lens correction and showing the existing "No lens data in this file" state.
- The oracle approach transfers directly: LibRaw/rawpy and rawler already decode NEF and RAF, and may be used "only as test oracles in CI" (AGENTS.md hard rule 1). The synthetic-file strategy (`ArwBuilder` + test encoders) would have to be rebuilt per format, which is roughly as much code as the decoder (testgen is 2906 + ~1370 lines for ARW).
- The Before view needs a per-body D65 matrix (rule 8 allows numbers from any source; `training/other_makes/colour_matrices.json` already holds "85 Canon and 58 Nikon bodies", `research/canon-nikon-curves-2026-10-19.md:161`) and a camera tone curve; the shipped tone curve is fitted to Sony's JPEG engine and would be approximate for other makers.

### Gaps
- macOS document-type registration: none found, but I did not read `tauri.conf.json` in full.
- I did not enumerate e2e/Playwright fixtures and mock API data (`src/api.mock*.ts`, `e2e/`) which contain many `.ARW`/Sony strings.
- CC0 availability of NEF/RAF samples on RawDB/raw.pixls.us is not recorded in the repo (the Canon/Nikon research says only "CC0 samples on raw.pixls.us, which was unreachable from CI before", `research/canon-nikon-curves-2026-10-19.md:124`).

## 6. Do the specs or research already discuss other makers?

### Takeaway
Specs put non-Sony raw files explicitly out of scope and list no other-maker item even in V2. One research report (2026-10-19) covers Canon and Nikon in depth, concluding the curves exist but decoders are the real cost and listing four owner decisions; Fujifilm cameras, RAF and X-Trans are not mentioned anywhere in the repository.

### Cited Findings
- Out of scope list ends with "Non-Sony raw files." — `docs/00-overview.md` ~62 (section "Out of scope"). V2 list (`docs/00-overview.md` ~48-55): A1 controlled captures, "More measured camera profiles, starting with the A1", HDR, Windows/Linux, far-red ALEXA data, one-click ARRI looks. No other makers. V2 backlog in milestones is A1 captures only (`docs/15-milestones.md:362-372`).
- AGENTS.md opening line: "a free, open-source (Apache-2.0) macOS raw processor for Sony ARW files"; V1 data rule: "V1 relies on available measurements for Sony cameras."
- `docs/03-camera-profiles.md:37`: "Files with `Make` other than SONY but a valid ARW structure (Hasselblad Lunar and Stellar are rebadged Sony bodies) are decoded and use the default profile."
- `research/canon-nikon-curves-2026-10-19.md` (listed in `research/README.md:20`; merged with 0.1.16 per `docs/STATUS.md:46`). Header: "for the owner's question: can Yellowmoon source measurements for Nikon and Canon cameras, so that it is not limited to Sony? Background only: nothing here changes a spec." (`:3`). Key quotes:
  - "The data set we already ship Sony curves from, rawtoaces-data (Apache-2.0), holds 22 Canon and 11 Nikon curves." "Nikon has only the Z f among its 14 Z-mount stills bodies, plus D850-era DSLRs; the Z6, Z6 II, Z6III, Z7, Z8 and Z9 have neither curves nor a passing sibling." "No other public source can ship." (`:5`)
  - "The curves are the small part of the job. No Apache-compatible decoder for NEF, CR2 or CR3 exists. Nikon's High Efficiency raw (TicoRAW) has no open decoder at all. … Nikon's colour metadata is encrypted with tables that exist publicly only in GPL and Artistic-licence code." (`:5`)
  - NEF row: "TIFF; uncompressed, lossless and lossy Huffman with a linearisation table (`NEFCompression` 1–10) … High Efficiency and High Efficiency★ (13, 14) are intoPIX's TicoRAW, offered on the Z8, Z9, Z f and Z6III among others … no open-source decoder exists" (`:120`); permissive decoders to reuse: "none. quickraw, rawloader, rawler, libopenraw and imagepipe are LGPL; zenraw is AGPL" (`:121`); colour metadata: "ColorBalance (0x0097) and ShotInfo encrypted for most bodies, keyed on serial number and shutter count. The two 256-byte tables of the cipher exist publicly only in GPL or Artistic-licence code (dcraw, ExifTool, LibRaw). `WB_RBLevels` (0x000c) is in the clear, enough for the as-shot white (inferred, not checked on files)" (`:123`).
  - "A user can always convert to DNG with Adobe DNG Converter, but Yellowmoon reads ARW, not DNG, so that is no route today." (`:126`)
  - Four owner decisions: scope (V1 / V1.x / V2; "The V1 data rule allows them as it stands"), Canon colour vs the gate, sibling evidence (0.02 rule gives Nikon the Z f only; rawtoaces aliases would add Z6/Z6 II), and format rules ("Whether `docs/14` rule 8 … covers Nikon's two cipher tables") (`:130-135`).
  - Effort table: profiles for Weta Canon/Nikon bodies "small"; "A NEF decoder for uncompressed and lossless files, plus the cipher if decision 4 allows | the Z f, D850, D810, D800E and others; every other Z body on a stand-in; High Efficiency files refused with a plain message | medium"; a [HUMAN] email to Weta for Z6 II, Z8, Z9 (`:139-145`). "Nikon first would keep to the Sony standard of colour, but for one current Z body." (`:147`)
  - Other-source table: Butcher's DIY Z6 curve is CC BY-NC-SA / all rights reserved, "not shippable" (`:31`); camspec CC BY-NC-SA "avoid" (`:32`).
  - rawtoaces-data `aliases.json` "same sensor" pairs include D800→D800E and Z 6II→Z f; "These are the maintainer's assertions, not measurements." (`:21`)
- `training/README.md:32`: "`other_makes/` | research only, not shipped: the Canon and Nikon curves of rawtoaces-data measured against the profile checks".
- Fujifilm appears only as a film-stock trademark (Film node): e.g. `docs/24-film.md:54, 698-699`, `docs/14-licensing.md:99`, `README.md:154`, `CHANGELOG.md:78`; and once as a cited manual on diffraction correction (`research/raw-algorithms-prior-art.md:662`). `grep -iE "x-trans|xtrans|\bRAF\b"` over docs/, research/, crates/, training/, tools/ found nothing about Fuji cameras.
- No owner decision on other makers is recorded in `docs/STATUS.md` (grep for canon/nikon/other makes returns only the merge note at `:46` and Cube++ lines) or in `docs/16-risks-and-decisions.md` (no hits).

### Inferences
- Adding Nikon or Fuji contradicts current authoritative specs (`docs/00` out-of-scope, AGENTS.md product definition), so per AGENTS.md "Source of truth" it needs an owner decision and spec changes first; it is not something an agent may start from the milestones list.
- The Canon/Nikon report's four open decisions remain unanswered in the repo as of this checkout; decision 4 (cipher tables under rule 8) directly gates full-quality NEF white balance.

### Gaps
- Whether the owner has answered the four decisions outside the repository is unknown.
- The report's claims about NEF structure and Nikon encryption are marked by its author as "inferred, not checked on files" for the WB tag; treat as unverified.

## 7. What do the hard rules imply for adding makers?

### Takeaway
The rules allow other makers' published curves but force clean-room decoders, forbid own measurements in V1, and make every gate failure reportable; together they mean Nikon coverage is limited to rawtoaces-measured bodies (plus strict siblings), and Fuji has no documented measured-curve source at all.

### Cited Findings
- Licences (AGENTS.md hard rule 1; `docs/14-licensing.md:77`): "No code copied from GPL/AGPL/LGPL projects. Reading for format knowledge is fine; write the code yourself and say in the PR which references you read." Inputs table: "rawler, rawloader, rawspeed, LibRaw, `ljpeg` crate | LGPL-2.1 … | CI oracles only (`tools/oracle`, rawpy); never linked into shipped binaries"; RawTherapee/darktable/librtprocess "read for ideas only; clean-room"; darktable-cli allowed "as a separate process in CI" as lens oracle (`docs/14-licensing.md` inputs table). `cargo deny check` must pass.
- Rule 8 (`docs/14-licensing.md:82`): "Numeric facts may come from any source: colour matrices, white-balance coefficients and similar camera constants may be taken as numbers from any source … Only the numbers are taken: no code, file, table layout, text or comment … and nothing in the repository or the app refers to such a project as a source."
- Rule 5 (`docs/14-licensing.md:79`): shipped weights only from CC0, CC BY, CC BY-SA, Apache or own data; model card per network.
- V1 data rule (AGENTS.md; `docs/00-overview.md` ~31): "We take no measurements of our own in V1: no spectral measurements, no calibration frames, no test shoots. Bodies without published curves use a same-sensor sibling's profile or the A7R IV profile… Don't plan, request or simulate measurements; if a gate seems to need one, use public data or an independent tool … and note the limitation in `docs/STATUS.md`."
- Gates: "A failed gate is reported, never hidden or loosened. Gate thresholds change only when the owner agrees." (AGENTS.md "How to work"). Hard rule 9: [HUMAN] tasks (emails, legal checks) are the owner's; "Never fake their outputs".
- Test data rule: raw files never committed; `tools/fetch/` scripts with SHA-256; licence per file; CC0 only in the manifest tier 1 (`docs/test-photo-sources.md`; AGENTS.md "How to work"). White-scale rule: "Only files whose licence allows any use count for a shipped scale" (`docs/03-camera-profiles.md:113`).
- Precedent for a patent/IP gate on a codec: ARW 6.0 ships only after "[HUMAN] M2-H3: email Sony about IP for RDD 34-based decoding"; decoder behind cargo feature `arw6`, files detected and refused with a plain message meanwhile (`docs/04-raw-decoding.md:131-136`; `docs/14-licensing.md` inputs table row "SMPTE RDD 34").
- Names rule 3/6 and the About sentence naming Sony's trademark (`docs/14-licensing.md:80`).
- Hard rule 6: all image processing in Rust crates, CLI and app bit-identical; rule 8 determinism with golden tests — new decoders need CPU determinism and golden coverage.

### Inferences
- NEF/RAF decoders must be written from format facts (ExifTool tag tables, public descriptions), with LibRaw/rawler as CI oracles only; the repo's own research found no permissive decoder to depend on for NEF (`research/canon-nikon-curves-2026-10-19.md:121`). Whether a permissive RAF decoder exists is not addressed in the repo.
- Nikon's encrypted WB tables: whether the two 256-byte cipher tables count as "numeric facts" under rule 8 is an open owner/legal question (the research flags it; `:135`). Without them, as-shot WB may be limited to the clear `WB_RBLevels` tag.
- Nikon High Efficiency (TicoRAW) and any proprietary/patented Fuji codec would follow the ARW 6.0 precedent: detect, refuse with a plain sentence, gate on a [HUMAN] enquiry.
- Under the V1 data rule, the shippable measured Nikon bodies are those in rawtoaces-data (11, of which Z f is the only Z body); everything else is a stand-in whose quality cannot be validated by own shots. For Fuji, unless public Apache-compatible curves exist, every body would be a stand-in on some other maker's profile, which the Canon/Nikon research shows costs roughly 0.5-2.6 median ΔE00 for cross-make stand-ins (`:57-74`) — an "honest about estimates" (principle 5) labelling issue.
- New-maker profiles that fail the M1 sensor-map gate (Z f, D850 per the research) need either an owner-agreed per-make threshold or a richer map; an agent cannot loosen it.
- Trademark copy would need Nikon/Fujifilm added to the "Not affiliated…" sentence; the Film node already names Fujifilm as a film trademark holder, so a Fuji *camera* feature would sit next to Fuji *film stock* names in the same app (possible confusion; owner's call).

### Gaps
- No legal analysis of Fuji compression (lossless/lossy "compressed RAF") patents or of Nikon's cipher exists in the repo.
- The repo records no licence survey of RAF sample files or of public Fujifilm spectral sensitivity data.

## Summary table: layer by layer (inference, built from the findings above)

| Layer | Sony-specific today (where) | Nikon NEF (Bayer) | Fuji RAF |
|---|---|---|---|
| Container | TIFF walker generic; probe demands Sony evidence (`ym-raw/src/metadata.rs:384-400`) | reuse TIFF; new per-maker dispatch | new non-TIFF container reader |
| Compression | `DecodeKind` = Sony variants (`layout.rs:293`) | new uncompressed/Huffman decoders; HE (TicoRAW) refused | new Fuji decoders (not described in repo) |
| Makernote, ISO, levels, WB, crop | `makernote.rs`, `sr2.rs`, Sony tags (`docs/04:34-54`) | new Nikon reader; WB cipher is an open rule-8 question | new Fuji reader |
| Layout/CFA | `Layout::{Bayer 2x2, Ycbcr}` (`layout.rs:271`) | fits | X-Trans needs a new variant + R1-R6 + noise + jdd work; Bayer Fuji bodies fit |
| Lens corrections | Sony 0x7032/35/37 only, no Lensfun | none unless Nikon embedded data is read | none unless Fuji embedded data is read |
| Profiles | 11 Sony curves; registry keyed by Sony model ID; schema patterns `sony/Sony_…` | 11 curves already fetched (Z f, D850, D810, D800E, D5300, D3300 + 5 legacy); registry/schema/detect generalisation | no curve source documented |
| Illuminant model | one per profile, synthetic training | train per new profile | n/a without curves |
| Noise | Sony model-ID keys, `BodyClass::from_model`, gain table | generalise keys/classes; estimator works | Bayer bodies as Nikon; X-Trans needs estimator changes |
| Before view | Sony matrices and tone curve | matrices available as numbers (58 Nikon bodies); tone curve approximate | matrices needed; tone approximate |
| Open filters/UI/CLI | "arw" in 5 places; copy throughout | edit + spec/copy changes | same |
| Tests/oracle/fetch | ARW-only testgen, fuzz, oracle, corpus scripts | rebuild per format; LibRaw/rawler oracles usable | same |
