# 07 · Learned models

Crates: `ym-ml` (runtime), `ym-gpu` (shared device). Training code: `training/` (Python, PyTorch, colour-science; not shipped). Weights: `data/models/` (small, committed with model cards).

## Models in V1

| Model | Job | Size | Runs |
|---|---|---|---|
| `jdd` | joint demosaic + denoise (+ cRAW de-banding) | 0.2–1 M params | export, 1:1 tiles |
| `jdd-ms` | denoise only, for Lossless M/S RGB input | ≤ 0.5 M params | M/S files (optional; classical fallback otherwise) |
| `illum` | illuminant estimate: FFCC filter + class head | < 50 k params | once per photo, CPU |
| `smap` | sensor map MLP (only if it beats the table blend, `docs/02`) | ≤ 5 k params | once per photo, CPU |

Everything else in the pipeline is analytic.

## `jdd` architecture (starting point)

- Input at half resolution (H/2 × W/2): 4 packed Bayer planes (R, G1, G2, B, balanced with `g0`), σ map, quantisation step map, clip mask: 7 channels. Normalise the planes by a fixed scale (0.25) so typical values sit near 0–1 and highlights above 1 survive.
- U-Net, 3 levels, channels 32 / 64 / 96, two 3×3 convolutions per level (SiLU), strided-conv downsampling, pixel-shuffle upsampling (no transposed convolutions), skip connections. Optional NAFNet-style simple gates if they win.
- Output: 12 channels at half resolution → pixel-shuffle ×2 → 3 channels (linear RGB) at full resolution, added to a bilinear demosaic of the input (residual learning).
- Receptive field: about 47 half-res px for this layout (2 convs + strided down per level, 3 levels, mirrored decoder), so about 94 full-res px. Compute it from the final architecture in code; the tile halo is half of it, rounded up to a multiple of 16 (64 full-res px for the starting layout). Keep the receptive field ≤ 128 full-res px.
- Must export to static-shape ONNX (opset 17) for a 512 × 512 half-res tile plus halo, and to safetensors. Avoid ops that CoreML or burn-onnx handle poorly (ConvTranspose and Resize are restricted under ORT's CoreML provider, per `research/sony-arw-to-arri-look-research.md`; prefer pixel-shuffle).
- Budget: ≤ 250 kFLOP per output pixel (≈ 12 TFLOP for 50 MP). Measure.

## Training data (licences decide what we may use)

| Source | Licence | Use |
|---|---|---|
| PixelShift200 (Sony A7R III, 4-shot pixel shift) | CC BY 4.0 | main ground truth: full RGB at every pixel; mosaic it ourselves |
| Noise sampler parameters | physics and the ELD paper's published relations (`docs/06`) | noise synthesis |
| Our own A1 pixel-shift captures | ours | V2 only (no captures in V1) |
| Synthetic patterns (zone plates, fine text, fabrics, star fields, rendered procedurally) | ours | moiré and false-colour robustness |
| ELD virtual-camera sampling (method only, code MIT) | MIT | noise sampler design |
| DND, SID, ELD data, Cube++ | non-commercial / evaluation-only / CC BY | evaluation only, never training (Cube++ is CC BY and allowed for `illum`) |

Never train shipped weights on data without a licence that allows it, including "found" images. Record every dataset and its licence in the model card.

**Sample generation (per training patch):**

1. Take a linear RGB ground-truth patch (camera RGB of the source body; for PixelShift200 apply the A7R III → A7R IV/A1 response difference via the spectral profiles only if it measurably helps).
2. Random exposure scale and white balance, so gains `g0` vary like real files.
3. Mosaic (RGGB, with random phase).
4. Noise: sample a virtual camera (log K uniform over the A1's range ±1 stop; read noise from the ELD relation with scatter; Tukey-λ tails; row noise; per-plane colour bias). Add Poisson–Gaussian noise in DN.
5. Quantise: 14-bit; with probability 0.3 pass through our test-only cRAW encoder and decoder to get real cRAW errors and the `qstep` map; with probability 0.1 simulate 12-bit.
6. Clip at white; build the clip mask; with probability 0.2 push highlights over white.
7. σ map from the **true** model, then jitter it by a random factor in [0.75, 1.25] (log-uniform) so the network tolerates estimate error. With probability 0.1 set σ to zero and use the clean mosaic (teaches "demosaic only" for Luminance = 0).

**Loss:** L1 on `log(1 + 16·x)` plus 0.2 × (1 − MS-SSIM) on the same, plus a gradient-domain L1. Evaluate PSNR in linear and in the log domain.

**Splits:** hold out whole scenes (10%). Never tune on the gate sets.

**Compute:** [HUMAN M2-H4] training runs need a CUDA GPU (one RTX 4080-class GPU: a few hours per run) or Apple silicon with MPS (slower). The agent writes the scripts, a `make train-jdd` entry point, a config with seeds, and a dry-run mode that trains for 50 steps on CPU in CI.

## Runtime (`ym-ml`)

```rust
pub trait TileModel: Send + Sync {
    fn input_shape(&self) -> TileShape;           // fixed, from the ONNX graph
    fn run(&self, inputs: &TileInputs, out: &mut TileOutput, q: &GpuQueue) -> Result<()>;
}
```

Two backends behind this trait:

- **burn 0.22 + burn-onnx** on the shared wgpu device (no copies between the image pipeline and the network).
- **ort 2.x** (ONNX Runtime) with the CoreML execution provider, CPU as fallback. Needs a copy to and from host memory per tile.

**Decision (M0-08):** benchmark both on a 512 × 512 half-res tile of the starting architecture on an M1 and an M3: time per tile, total for 50 MP, peak memory, numerical parity. Pick the faster by ≥ 20%; on a tie pick burn (one device, pure Rust). Record the numbers in `docs/16-risks-and-decisions.md`.

**CPU fallback:** ort's CPU provider, or burn's ndarray backend, for machines with no usable GPU and for the headless CLI on CI runners.

## Parity and versioning

- Golden tiles (20 fixed inputs committed under `crates/ym-ml/tests/data/`) run through PyTorch, ONNX Runtime and the chosen runtime. Max abs difference ≤ 2e-3 (f16) in the output's linear units. Runs on every model change.
- Each weight file has a version and SHA-256; the version goes into export XMP.
- `data/models/<name>/MODEL_CARD.md`: purpose, architecture, parameters, training data and licences, metrics on every gate set, known failure cases, date, git SHA of the training code.

## Gates (M2)

- `jdd` beats the classical fallback by ≥ 2 dB PSNR on Kodak, McMaster, Gharbi hard and moiré sets (mosaicked, clean) and on held-out PixelShift200 scenes with synthetic noise.
- No zipper or false colour in blind review of 30 crops.
- cRAW stripes ("blondie" artefacts) visibly reduced against LibRaw + classical on 10 A7R cRAW samples.
- Visual parity with Capture One and DxO on an A1 ISO ladder 100–25600 (studio ISO series from the evaluation tier, local only) in blind A/B (ours preferred or tied ≥ 50%).
