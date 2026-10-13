# Tasks: raw algorithms (RA)

> **In this repository (2026-10-13, owner's decision):** the C++ references (`reference/*.cpp`, `tools/check.sh`) are not committed. They were used as a development oracle outside the repository; the vectors in `crates/ym-gpu/tests/cpu_data/<algorithm>/` are the fixed oracle here and are never hand-edited. Where this text says `tools/reference/` or `check.sh`, read it as that external oracle.

> **Status, 2026-10-13 (as built).** RA-01 to RA-10 are done except where the table says otherwise. The code wins where this list and it differ; the as-built notes are in the specs (`01` sections 9 to 12, `02` sections 9 and 10, `03` sections 9 and 10, `04` sections 9 and 10) and in `INTEGRATION.md` section 10.
>
> | Task | State | In one line |
> |---|---|---|
> | RA-01 | done, with changes | Specs and vectors are in. No C++ and no `ref-check` job (owner's decision); the licensing row (RA-H1) is in `docs/14`. |
> | RA-02 | done | `NoiseDn`, `DemosaicInputs`, `demosaic_with`, `Chain::noise`. |
> | RA-03 | done | `Method::AdaptiveRatio`, plain path: 13 of 17 vectors bit-identical, the four `phase_*` vectors 1.2e-7 off (f32 gains). |
> | RA-03b | done | CA path implemented and vector-exact. Since R5b (CA corrected on the mosaic, `docs/05`) the chain no longer reaches it. |
> | RA-04 | done | The division by `g0` is inside `demosaic_with`; test-only switch `RenderOptions::demosaic`. |
> | RA-05 | done, two sets blocked | Adaptive ratio is the default (owner, after RA-H4). Gharbi hard/moiré and PixelShift200 could not be fetched. |
> | RA-06 | done | `Method::LogGuided`; within 1 dB of Menon on the synthetic set; not used by any render path. |
> | RA-07 | done | `CaptureSharpen::Diffraction`; Richardson–Lucy stays the default; RA-H2 is open. |
> | RA-08 | done, changed by the owner | A per-photo setting `output_sharpening` (0 to 200 %, default 100), not an export option that ships off. |
> | RA-09 | done on lavapipe, not wired | WGSL kernels for all four algorithms with parity tests; Metal checks wait for a Mac (M0-H4). |
> | RA-10 | done | `docs/05`, `docs/09`, `docs/16`, `docs/STATUS.md`, `CHANGELOG.md`, this list and `INTEGRATION.md` were brought in line with the code. |

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
- Add a `ref-check` CI job that runs `REF_DIR=tools/reference VEC_DIR=crates/ym-gpu/tests/cpu_data tools/reference/check.sh`. It needs a C++17 compiler and python3; nothing is shipped. *[Not done: no C++ in the repository, so no such job (see the note at the top).]*
- **[HUMAN] RA-H1:** add a row for the four algorithms to `docs/14-licensing.md`. Propose this wording and list it under "Waiting on Mykyta": "specification supplied by the project owner; implemented in this repository; Apache-2.0".

*Done when:* `check.sh` passes in CI, the docs link resolves, and the STATUS entry is written. *[As built: there is no `check.sh` here; the vectors are the fixed oracle and the Rust tests read them.]*

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

*Done when:* the above pass, plus clippy and fmt. `DEFAULT_METHOD` is unchanged. *[It was unchanged by this task; RA-05 and the owner then made the adaptive ratio demosaic the default.]*

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
- Zipper and false-colour metrics. *[As built they are not in `cpu/metrics.rs`: they are defined in `crates/ym-gpu/tests/common/ra05_metrics.rs`, tested in `tests/cpu_demosaic_metrics.rs` and run by `examples/ra05_gate.rs`.]*
- CPU time per 50 MP frame.

Following the `docs/05` R6 rule, the method with the best PSNR becomes `DEFAULT_METHOD`. *[As built: adaptive ratio, 36.05 dB against Menon 34.43 and LMMSE 34.28 over nine linear-light cells; the owner confirmed it after RA-H4 on 2026-10-13. `docs/05` R6 has the numbers and the caveats.]* Before re-baselining, render the nine local real ARWs and the synthetic golden set with old and new defaults, and list them under "Waiting on Mykyta" (**[HUMAN] RA-H4**). After his OK, run `golden update --note "R6 default: <method>, PSNR <numbers>"`.

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
*[Superseded by the owner's decision of 2026-10-13 (`docs/16`): output sharpening is a per-photo develop setting, `DevelopSettings.output_sharpening`, 0 to 200 %, default 100 %, slider "Output" in Detail › Sharpening and `ym render --output-sharpening`. It is applied to every display-referred render at its own size and to every raster export after the resize, never to a DNG, with no export switch and no print preset in the UI. The list below is the original text.]*
- Add `ExportOptions::output_sharpen: Option<OutputSharpen>` with the spec's screen and print presets, default `None`.
- No UI or CLI change yet (**[HUMAN] RA-H3**).
- Unit tests for the presets.
- An edge-profile test.
- A test that `None` leaves export bytes unchanged.

*Done when:* the vectors pass and the golden hashes are unchanged. *[As built the vectors pass, and the hashes change by design: at the default 100 % all 61 synthetic golden files move (the maximum ΔE00 per file is 6.65 to 13.10, on the hard patch edges of the charts), and with the amount at 0 all 61 are identical. They are re-baselined together with the R6 default.]*

## RA-09 · GPU kernels

Port each algorithm to WGSL in `ym-gpu/src/shaders/`, with a parity test against its CPU reference stating the max abs error (`AGENTS.md`). Order:
1. adaptive ratio demosaic (RA-03);
2. log-guided (RA-06);
3. diffraction (RA-07);
4. output sharpening (RA-08).

Run the parity tests on lavapipe; Metal checks go under "Waiting on Mykyta" (M0-H4).

*Done when:* the parity tests pass on lavapipe. *[Done for all four, in the order above; the numbers are in `docs/05` (GPU stages) and in the as-built sections of the specs. No kernel is wired into a render or export path yet; that is the next wave (`docs/16`, the owner's GPU direction of 2026-10-13).]*

## RA-10 · Docs

Update the docs:
- `docs/05` R6: the methods and the default; R8: both methods and the selection rule.
- `docs/09`: output sharpening, presets, off by default. *[Superseded by the owner's decision of 2026-10-13 (`docs/16`): on by default as a per-photo setting; `docs/09` describes it as built.]*
- `docs/16`: decision log entries with numbers.
- `docs/STATUS.md`.
- `CHANGELOG.md`.

*Done when:* the docs match the code.

## Rules specific to this work

- The specs describe exact behaviour. Where your port deviates (for example f32 instead of f64 in a reduction), say so in the module doc comment and keep within the vector tolerances.
- Don't add new constants that aren't in a spec. If a spec misses one, ask in STATUS.
- Keep the C++ references in `tools/reference/` as the oracle for vectors; regenerate vectors only with `check.sh`, and never hand-edit them.
