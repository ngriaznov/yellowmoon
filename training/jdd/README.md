# training/jdd: joint demosaic + denoise (M2-04, M2-05)

PyTorch code for the `jdd` network of `docs/07-ml-models.md`: model, receptive-field calculator, the training
sample generator, loss, training loop, evaluation, ONNX/safetensors export and model card. Not shipped.
**No trained weights exist yet** (training needs a GPU and the PixelShift200 dataset, HUMAN M2-H4), so nothing
here writes to `data/models/` and the classical path ships.

Run from the repository root (`python3 -m training.jdd.<module>`) or through the Makefile
(`make -C training/jdd <target>`, `make -C training/jdd help`).

```bash
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu   # or the CUDA wheel on a GPU box
pip install -r training/jdd/requirements.txt
make -C training/jdd test        # pytest, about a minute on 4 cores
make -C training/jdd dry-run     # 50 steps on CPU, synthetic data, then ONNX/safetensors checks (CI)
make -C training/jdd overfit     # a CPU run that overfits 8 samples
make -C training/jdd sample-sheet
```

| Module | What |
|---|---|
| `model.py` | the U-Net (7 input channels, 32/64/96, SiLU, strided conv down, pixel-shuffle up, bilinear residual), FLOP and parameter counters |
| `receptive_field.py` | receptive field and tile halo from the traced module graph (`torch.fx`), checked against a brute-force gradient probe in the tests; asserts RF <= 128 full-res px |
| `data.py` | PixelShift200 loader (code only, expected layout documented in the class), the docs/07 sample generator (steps 1 to 7), the torch dataset |
| `patterns.py`, `noise.py`, `craw.py` | procedural ground truth; ELD-style virtual-camera noise with Tukey-lambda tails; Python port of the cRAW encoder and decoder (docs/04) with the `qstep` map |
| `loss.py` | L1 on log(1 + 16x) + 0.2 (1 - MS-SSIM) + gradient-domain L1 |
| `train.py`, `configs/*.json` | deterministic training loop (seeds in the config), `--dry-run --steps 50 --device cpu`, resume |
| `evaluate.py` | PSNR (linear and log) against a classical baseline hook, gate sets as image folders |
| `export.py`, `weights_io.py` | safetensors (deterministic bytes, architecture header) and static-shape ONNX opset 17 |
| `model_card.py` | `MODEL_CARD.md` from a run directory |
| `golden_tiles.py` | seeded random-weight fixtures for the Rust parity tests (`crates/ym-ml/tests/data/`) |
| `sample_sheet.py` | PNG review sheet of the generator (do not commit the image) |

Where the data lives (never committed): `testdata/pixelshift200/<scene>/` for PixelShift200 (see `data.PixelShift200`),
`testdata/demosaic/<set>/` for Kodak, McMaster and Gharbi (evaluation only), fetched by scripts that are not part of
this folder yet (`docs/validation-data.md`, rows 13).

On a loaded or shared machine set `train.threads` in the config (or `OMP_NUM_THREADS`): PyTorch's OpenMP threads
busy-wait and can make small CPU runs several times slower than a single thread.
