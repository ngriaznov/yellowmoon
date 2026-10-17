# tools/eval/demosaic: the R6 demosaic gate

Measurement only. Nothing here changes `DEFAULT_METHOD`; the owner chose the adaptive ratio demosaic after the blind look (2026-10-13) and that is the default now. The numbers are
summarised in `docs/05-raw-processing.md` (R6, "Demosaic gate measurement") and the recommendation is in `docs/STATUS.md`.

| File | What |
|---|---|
| `gate.sh` | Runs the Rust measurements (`all`, `psnr`, `timing`) or builds the blind-look set (`blind`) |
| `blind_look.py` | `render`: renders the real and synthetic files with two R6 methods, makes 1:1 crops, randomises A/B, writes `index.html` and `key.json`; `score`: tallies the answers downloaded from the page |
| `test_blind_look.py` | Unit tests of the randomisation, the crop windows, the page and the scoring (no `ym`, no ARW files) |

The Rust code is in `crates/ym-gpu`: `examples/demosaic_gate.rs` (PSNR, zipper, false colour, noisy inputs),
`examples/demosaic_timing.rs` (CPU time per 50 MP frame), `tests/common/demosaic_metrics.rs` (the protocol and the metrics) and
`tests/cpu_demosaic_metrics.rs` (their unit tests).

## The numbers

```
tools/fetch/demosaic-sets.sh        # Kodak and McMaster into testdata/demosaic (pinned commits, SHA-256 checked)
tools/eval/demosaic/gate.sh psnr    # testdata/demosaic_gate/numbers.md and gate.json
tools/eval/demosaic/gate.sh timing  # testdata/demosaic_gate/timing.md and timing.json
```

* **PSNR** follows `docs/05`: the ground truth is mosaicked as RGGB (no optical low-pass), demosaiced, and compared over
  the interior (10 px border excluded) in linear light, mean of per-image dB. The Kodak and McMaster images are 8-bit sRGB,
  so they are decoded with the sRGB transfer function first; signal scales 1, 0.1 and 0.01 multiply the linear image and
  the peak. The gamma-encoded values at scale 1 are printed too, to compare with the older numbers of `docs/05`
  (they reproduce exactly: Kodak 29.82 / 38.88 / 39.16, McMaster 32.31 / 34.21 / 35.09 for bilinear, Menon, LMMSE).
* **Noise models.** The adaptive ratio demosaic reads one (spec 01 section 6.3: `a = 2e-5`, `b = 1e-8`, `q = 1/65535`, unit
  gains), kept fixed at every signal scale; the log-guided demosaic gets the zero model (spec 02 section 6.4). Three more
  columns say how much that choice matters.
* **Zipper and false colour.** `cpu/metrics.rs` has PSNR, Lab, CIEDE2000 and `chroma_delta_e00`, but no zipper or
  false-colour metric, so they are defined in `tests/common/demosaic_metrics.rs` (the definitions are in `artifacts`' doc):
  the zipper rate after Lu and Tan (2003) and a chroma-error rate on the CIEDE2000 scale of the R5 gate. They rank
  methods; the absolute rates depend on the thresholds, which are stated with the table.
* **Timing.** 50 MP, 8658 x 5772, a textured scene with noise. One thread is a rayon pool of one; parallel is the default
  pool. Best of three; CPU seconds are the number to trust on a shared machine.
* **Gharbi hard/moire and PixelShift200** are not run here: the hosts (`groups.csail.mit.edu`, `data.csail.mit.edu`,
  Google Drive) are refused by this environment's proxy (`CONNECT tunnel failed, response 403`) and neither set has a
  mirror on GitHub. Put the images in `testdata/demosaic/gharbi_hard`, `gharbi_moire` or `pixelshift200` (8-bit PNG or
  TIFF, ground-truth RGB) and the gate reads them.

## The blind look

```
cargo build -p ym-cli --features force-demosaic     # a `ym` that reads YM_FORCE_DEMOSAIC (test-only, never shipped)
python3 -I tools/eval/demosaic/blind_look.py render --ym target/debug/ym --out testdata/demosaic_gate
#   --baseline menon --candidate adaptive_ratio      the defaults
#   --only ILCE-7M3 chart__D65                       a subset, by name
#   --reuse                                          keep what key.json already has (a run can be stopped and resumed)
open testdata/demosaic_gate/index.html               # answer; "Download results" writes ra05-picks.json
python3 -I tools/eval/demosaic/blind_look.py score --out testdata/demosaic_gate --picks ra05-picks.json
```

The page shows each file twice (A and B) as a full frame and four 1:1 crops (edges, fine texture, saturated colour,
shadows), with a flip mode that shows one image at a time at the same place. The labels are a seeded, balanced shuffle over
the files; `key.json` maps them to methods and is not opened until the answers are in. Both renders of a file use the same
settings and differ in the R6 method only. The real ARW files are local evaluation data (no clear licence): the renders
stay in `testdata/demosaic_gate/`, which is git-ignored, and are never committed or redistributed.

Needs numpy and Pillow. Tests: `python3 -I -m unittest discover -s tools/eval/demosaic -p 'test_*.py'`.
