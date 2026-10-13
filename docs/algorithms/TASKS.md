# Tasks: raw algorithms (RA)

> **In this repository (2026-10-13, owner's decision):** the C++ references (`reference/*.cpp`, `tools/check.sh`) are not committed. They were used as a development oracle outside the repository; the vectors in `crates/ym-gpu/tests/cpu_data/<algorithm>/` are the fixed oracle here and are never hand-edited. Where this text says `tools/reference/` or `check.sh`, read it as that external oracle.

For the coding agent. Follow the repository's `AGENTS.md` for everything this file doesn't say: its hard rules, conventions, and definition of done. Read `INTEGRATION.md` first, then the spec for the task in hand.

Work on `claude/yellowmoon-v1`, one commit per task, named like the rest of the branch. Record every task, gate number and question in `docs/STATUS.md`. The specs in `spec/` are authoritative for the algorithms. If a spec and Yellowmoon's code disagree on conventions (units, layout, borders), follow `INTEGRATION.md`. If it doesn't settle the conflict, write a question for Mykyta in `STATUS.md` and continue with other tasks.

## RA-01 · Bring the package in

Copy the package into the repository:

| From | To |
|---|---|
| `spec/0N-*.md` | `docs/algorithms/0N-*.md` |
| `vectors/<algorithm>/` | `crates/ym-gpu/tests/cpu_data/<algorithm>/` |
| `reference/*.cpp` and `tools/check.sh` | `tools/reference/` |

Other steps:
- Link `docs/algorithms/` from `docs/05-raw-processing.md` (R6, R8) and `docs/09-export.md`.
- Add a `ref-check` CI job that runs `REF_DIR=tools/reference VEC_DIR=crates/ym-gpu/tests/cpu_data tools/reference/check.sh`. It needs a C++17 compiler and python3; nothing is shipped.
- **[HUMAN] RA-H1:** add a row for the four algorithms to `docs/14-licensing.md`. Propose this wording and list it under "Waiting on Mykyta": "specification supplied by the project owner; implemented in this repository; Apache-2.0".

*Done when:* `check.sh` passes in CI, the docs link resolves, and the STATUS entry is written.

## RA-02 · Shared plumbing (no behaviour change)

All of this lives in `ym-gpu/src/cpu` (see `INTEGRATION.md` §2–§4):

- **`NoiseDn`.** Add `{ g, r }` at the 16-bit working scale, built from `ym_noise::model::NoiseModel` (`INTEGRATION.md` §3). Unit-test the conversion against the formulas.
- **`DemosaicInputs<'a>`.** Add `{ gains: [f32; 3], noise: NoiseDn, ca: Option<&'a CaCurves> }` and `demosaic_with(m, method, &inputs)`. Keep `demosaic(m, method)` working: it calls `demosaic_with` with unit gains and a zero-noise model.
- **`Chain` field.** Add `noise: Option<NoiseDn>` to `Chain`, filled by `FullChain::new` from `RawParams::noise`.

*Done when:* all existing tests and golden hashes are unchanged.

## RA-03 · Adaptive ratio demosaic (CPU)

Implement `docs/algorithms/01-adaptive-ratio-demosaic.md` as `Method::AdaptiveRatio`, in `ym-gpu/src/cpu/demosaic/adaptive_ratio.rs`:
- the three passes;
- the plain (non-CA) path of pass 2 first; the CA path is RA-03b;
- `Method::halo()` returns the spec's maximum halo;
- add it to `Method::ALL`, and update `psnr_row` and the other fixed-size arrays that assume three methods.

Tests:
- **Vectors.** Every vector passes within its tolerance.
- **Tiling.** Tiled vs whole frame within 1e-4, with `cpu_tiling.rs` picking it up through `Method::ALL`.
- **CFA phases.** All four phases are equivariant.
- **Flat field.** A flat field comes back exact.
- **Determinism.** `--jobs 1` vs `--jobs 4` are byte-identical.

*Done when:* the above pass, plus clippy and fmt. `DEFAULT_METHOD` is unchanged.

## RA-03b · CA-aware green refinement (optional, after RA-05)

Pass 2's CA path (spec §CA) is driven by `CaCurves`, derived from the R9 lens CA knots (`INTEGRATION.md` §6).

*Done when:* its vectors pass, and on files with 0x7035 tags the R9 CA residual after correction is no worse. Report the numbers.

## RA-04 · Wire the demosaic into the chain

In `Chain::run_tile` R6, when `method == AdaptiveRatio`:
1. Undo R4 on the tile: divide each photosite by `g0` of its CFA colour.
2. Call `demosaic_with` with `gains = g0` and `noise = chain.noise`.
3. Keep the output, which is already balanced.

Keep the R6 bilateral denoise and R7 as they are; the spec's evaluation notes say what to A/B later. `Detail`'s `Y_classical` uses the same method as R6.

*Done when:* a full render through `ym` with the method forced (test-only switch) matches a whole-frame `demosaic_with` call. `ym` and the app give identical bytes.

## RA-05 · Demosaic gate and default

Measure Bilinear, Menon, LMMSE and AdaptiveRatio:
- PSNR with a 10 px border excluded, on the synthetic set in `cpu_demosaic.rs` and on Kodak, McMaster, Gharbi hard/moiré and PixelShift200 where `tools/fetch` can get them. Record blocked sets as blocked.
- Zipper and false-colour metrics (`cpu/metrics.rs`).
- CPU time per 50 MP frame.

Following the `docs/05` R6 rule, the method with the best PSNR becomes `DEFAULT_METHOD`. Before re-baselining, render the nine local real ARWs and the synthetic golden set with old and new defaults, and list them under "Waiting on Mykyta" (**[HUMAN] RA-H4**). After his OK, run `golden update --note "R6 default: <method>, PSNR <numbers>"`.

*Done when:* numbers are in STATUS and the decision is in the `docs/16` decision log.

## RA-06 · Log-guided fast demosaic

Implement `docs/algorithms/02-log-guided-demosaic.md` as `Method::LogGuided`, in the float form the spec recommends for the GPU, with the fixed-point form available for the vectors. Use it for the fit view and thumbnails if it meets the `docs/05` fit-preview target (< 300 ms at 1/4 resolution on M1; on Linux report relative timings against Menon). Its PSNR must be within 1 dB of Menon on the synthetic set; report the actual number.

*Done when:* vectors, tiling and CFA tests pass, and timings and PSNR are in STATUS.

## RA-07 · Diffraction deconvolution in R8

Implement `docs/algorithms/03-diffraction-deconvolution.md` as `ym-gpu/src/cpu/sharpen/diffraction.rs`. Changes around it:
- Make R8 an enum, `CaptureSharpen::{RichardsonLucy(SharpenParams), Diffraction(DiffractionParams)}`, in `Chain` and `Stage::Sharpen`, with halos from each spec.
- Add `sensor_width_mm` and `active_width_px` (or `pixel_pitch_um`) to `data/cameras/registry.json` for the registered bodies. Values must come from published specifications; record the source per body in the registry's provenance field. Implement the EXIF crop-factor fallback from `INTEGRATION.md` §7.
- Make the selection internal only (no settings-schema change): Richardson–Lucy stays the default.
- A/B on the local real files at f/2.8, f/8 and f/16 (if present): edge profiles (10–90 % rise distance), overshoot, and noise in flat patches. Put crops under "Waiting on Mykyta" (**[HUMAN] RA-H2**).
- Measure flat-area noise growth after R7 at f/11–f/22 (spec/03 §7.1 item 1). If it exceeds ×1.10, also test the noise-gate variant described there and report both.

*Done when:* vectors pass, tiling parity holds, numbers are in STATUS, and defaults are unchanged.

## RA-08 · Output sharpening in export

Implement `docs/algorithms/04-output-sharpening.md` as `ym-gpu/src/cpu/output_sharpen.rs`. Call it from the export path right after the linear-light Lanczos resize (`ym-core/src/export.rs`, `ym-export/src/resize.rs`), and at the same point for full-size exports. Ship it **off**:
- Add `ExportOptions::output_sharpen: Option<OutputSharpen>` with the spec's screen and print presets, default `None`.
- No UI or CLI change yet (**[HUMAN] RA-H3**).
- Unit tests for the presets.
- An edge-profile test.
- A test that `None` leaves export bytes unchanged.

*Done when:* the vectors pass and the golden hashes are unchanged.

## RA-09 · GPU kernels

Port each algorithm to WGSL in `ym-gpu/src/shaders/`, with a parity test against its CPU reference stating the max abs error (`AGENTS.md`). Order:
1. adaptive ratio demosaic (RA-03);
2. log-guided (RA-06);
3. diffraction (RA-07);
4. output sharpening (RA-08).

Run the parity tests on lavapipe; Metal checks go under "Waiting on Mykyta" (M0-H4).

*Done when:* the parity tests pass on lavapipe.

## RA-10 · Docs

Update the docs:
- `docs/05` R6: the methods and the default; R8: both methods and the selection rule.
- `docs/09`: output sharpening, presets, off by default.
- `docs/16`: decision log entries with numbers.
- `docs/STATUS.md`.
- `CHANGELOG.md`.

*Done when:* the docs match the code.

## Rules specific to this work

- The specs describe exact behaviour. Where your port deviates (for example f32 instead of f64 in a reduction), say so in the module doc comment and keep within the vector tolerances.
- Don't add new constants that aren't in a spec. If a spec misses one, ask in STATUS.
- Keep the C++ references in `tools/reference/` as the oracle for vectors; regenerate vectors only with `check.sh`, and never hand-edit them.
