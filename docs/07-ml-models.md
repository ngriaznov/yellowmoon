# 07 · Learned models

This file describes the learned models of Yellowmoon: the illuminant estimator that ships, and the joint demosaic and denoise network (`jdd`) whose runtime, training code, export and parity tests exist while its weights are not trained yet. Crates: `ym-ml` (runtime), `ym-gpu` (shared device). Training code: `training/` (Python, PyTorch, colour-science; not shipped). Weights: `data/models/` (small, committed with model cards).

## Models in V1

| Model | Job | Size | Runs | State |
|---|---|---|---|---|
| `illum` | illuminant estimate: FFCC filter + class head | 13,199 parameters per model (limit 50 k) | once per photo, CPU (`ym_color::illum`) | ships: 11 per-profile models in `data/models/illum/` |
| `jdd` | joint demosaic + denoise (+ cRAW de-banding) | 0.2–1 M params | export, 1:1 tiles | runtime, training code, export and parity tests exist; no trained weights (needs GPU time and PixelShift200, M2-H4) |
| `jdd-ms` | denoise only, for Lossless M/S RGB input | ≤ 0.5 M params | M/S files | not built; those files use the classical path |
| `smap` | sensor map MLP | ≤ 5 k params | once per photo, CPU | not used: the table blend of `docs/02` C2 is the sensor map |

Everything else in the pipeline is analytic.

**The classical path runs everywhere.** With no `jdd` weights, the 100% view, every export and R7's `Y_classical` use the adaptive ratio demosaic and the single-frame denoise of `docs/05` R6. After weights exist the classical path stays the fallback (`decode.neural_demosaic = false`, `ym --no-neural`). `NeuralStatus::probe` looks for weights under `<data_root>/models/jdd/`; the render path does not call the network yet, so weights alone do not make it available. Every render report and `ym doctor` say which demosaic ran and why ("No trained demosaic model is installed yet, so the classical demosaic was used.").

## `illum`: the illuminant estimator

`illum` estimates the colour of the scene light from the 1/8-resolution raw preview: `w_scene` (camera RGB of the light, green = 1, in the camera's raw scale), probabilities over six classes (daylight, shade/overcast, tungsten/halogen, fluorescent, LED, mixed) and a confidence (the probability that the angular error is under 2°). The result drives C1, C2 and C3 of `docs/02`; the algorithm is described in `docs/02` Illuminant estimate and the model card is `data/models/illum/MODEL_CARD.md`.

1. Log-chroma histograms `u = ln(G/R)`, `v = ln(G/B)` of the unmasked pixels, 64 × 64 bins, in two channels (the pixels and their local absolute deviation, the edges).
2. An FFCC-style filter (a circular convolution with FFTs, 2 × 64 × 64 weights and a 64 × 64 bias) and a softmax over the bins give a histogram-only estimate.
3. The camera's as-shot white balance (tag 0x7313) is a Gaussian prior on `(u, v)` multiplied into the estimate.
4. A small class head (25 features, one `tanh` layer of 24 units, softmax) gives the class probabilities; a logistic regression gives the confidence.

One model per camera profile, each under 50 k parameters, stored as JSON with base64 `f32` tensors and a SHA-256 the Rust loader checks. Training data are synthetic scenes rendered through each profile's published curves (`training/illum/`); no photographs and no measurements of our own are used. On held-out synthetic scenes the median angular error is 1.22–1.30° with the as-shot prior and 1.79–1.98° from the histogram alone, on all 11 profiles (gate: median ≤ 2°).

## `jdd` architecture (starting point)

- Input at half resolution (H/2 × W/2): 4 packed Bayer planes (R, G1, G2, B, balanced with `g0`), σ map, quantisation step map, clip mask: 7 channels. The network takes the **physical** units of `docs/05` and applies its fixed input scaling itself (buffer `in_scale`: planes × 0.25, σ × 16, qstep × 16, clip × 1), so typical values sit near 0–1 and highlights above 1 survive; the runtime never scales. σ, qstep and clip are single half-resolution planes: σ and qstep are the RMS over the four photosites of the 2 × 2 cell of the per-photosite value in balanced units (`value · g0` per colour); the clip plane is 1 where any of the four photosites is clipped.
- U-Net, 3 levels, channels 32 / 64 / 96, two 3×3 convolutions per level (SiLU), strided-conv downsampling, pixel-shuffle upsampling (no transposed convolutions), skip connections. Layout: `enc0` (7 → 32) · down (3×3, stride 2) · `enc1` (64) · down · `bott` (96) · 1×1 convolution to 4C + pixel shuffle + skip concat · `dec1` · 1×1 convolution + shuffle + concat · `dec0` · 1×1 head to 12 channels. Zero padding everywhere. Input height and width are multiples of 4.
- Output: 12 channels at half resolution, pixel-shuffled ×2 to 3 channels (linear RGB) at full resolution, **added to a fixed bilinear demosaic** of the input (a 4 → 12 convolution on the packed planes, `bilinear_weights`), so a freshly initialised network (small head) is the bilinear demosaic. The output is in the same balanced units (white = 1.0 before gains), multiplied back by 4 inside the network.
- Receptive field: 47 half-res px for this layout (2 convs + strided down per level, 3 levels, mirrored decoder, 1×1 convolutions in front of the pixel shuffles and in the head), so 94 full-res px. It is **computed from the real module graph** in `training/jdd/receptive_field.py` (traced with `torch.fx`, checked against a brute-force gradient probe in the tests). The tile halo is half of it, rounded up to a multiple of 16 full-res px, and never less than twice the worst one-sided reach of any stride phase (26 half-res px here): 64 full-res px (32 half-res) for the starting layout. The receptive field stays ≤ 128 full-res px (asserted). The options `up_kernel=3` and `head_kernel=3` give RF 63 / 126 full-res px with both and halo 80.
- The network exports to static-shape ONNX (opset 17) for a 512 × 512 half-res tile **core** plus halo (input 576 × 576 half-res px for the starting layout), and to safetensors. Only Conv, Mul, Sigmoid, Add, Concat, Slice and DepthToSpace appear in the graph (`export.py --decompose-pixel-shuffle` swaps DepthToSpace for Reshape/Transpose for runtimes that lack it); the export fails if ConvTranspose or Resize appear, because ORT's CoreML provider and burn-onnx restrict them (`research/sony-arw-to-arri-look-research.md`).
- Budget: ≤ 250 kFLOP per output pixel (≈ 12 TFLOP for 50 MP). The starting layout, measured by `count_flops` (2 FLOP per multiply-add of every convolution, activations ignored), is 53.7 kFLOP per output pixel with 497,100 parameters.

## Tiling

`crates/ym-ml/src/tiles.rs`. Tiles are cut in half-resolution units, cropped (never blended), and the halo outside the frame is filled by symmetric reflection of the packed planes (the Bayer phase is preserved because the planes are channels). Core sizes and halos are multiples of 4, so every tile has the same stride-2 phase as a whole-frame run. Tiled and whole-frame outputs of the CPU runtime agree to 1e-4 in the tests (bit-identical in practice).

## Training data (licences decide what we may use)

| Source | Licence | Use |
|---|---|---|
| PixelShift200 (Sony A7R III, 4-shot pixel shift) | CC BY 4.0 | main ground truth: full RGB at every pixel; mosaic it ourselves |
| Noise sampler parameters | physics and the ELD paper's published relations (`docs/06`) | noise synthesis |
| Synthetic patterns (zone plates, fine text, fabrics, star fields, rendered procedurally) | ours | moiré and false-colour robustness |
| ELD virtual-camera sampling (method only, code MIT) | MIT | noise sampler design |
| DND, SID, ELD data, Cube++ | non-commercial / evaluation-only / CC BY | evaluation only, never training (Cube++ is CC BY and allowed for `illum`) |

Our own A1 pixel-shift captures are V2 work: V1 takes no captures. Shipped weights are never trained on data without a licence that allows it, including "found" images. Every dataset and its licence is recorded in the model card.

**Sample generation (per training patch;** `training/jdd/data.py`**):**

1. Take a linear RGB ground-truth patch (camera RGB of the source body; for PixelShift200, apply the A7R III → A7R IV/A1 response difference via the spectral profiles only if it measurably helps).
2. Random exposure scale and white balance, so gains `g0` vary like real files. Exposure is set by the 99.5th percentile of the raw signal: log-uniform over 2⁻⁷ … 0.98 of white, or 1.05 … 4.0 with probability 0.2 (highlights pushed over white).
3. Mosaic (RGGB, with random phase).
4. Noise: sample a virtual camera (ISO log-uniform over 50 … 64000, the A1's range ± 1 stop; `K = k0 · ISO / 100` with `k0 = 0.25 DN/e⁻` ± half a stop; read noise from the ELD relation with scatter; Tukey-λ tails; row noise; per-plane colour bias). Add Poisson–Gaussian noise in DN.
5. Quantise: 14-bit; with probability 0.3 pass through our test-only cRAW encoder and decoder to get real cRAW errors and the `qstep` map; with probability 0.1 simulate 12-bit (saturating at ADC code 4095, 16380 on the 14-bit scale). Both apply to the non-clean samples.
6. Clip at white; build the clip mask. The target is the ground truth **clipped at the same per-channel white** (`g0_c`) as the input, so the network is never asked to invent clipped content (highlight reconstruction is stage R5, before this network).
7. σ map from the **true** model, then jitter it by one log-uniform factor in [0.75, 1.25] per sample so the network tolerates estimate error. With probability 0.1 of all samples, σ is zero and the clean mosaic is used (teaches "demosaic only" for Luminance = 0).

The **qstep plane is the cRAW quantisation step and zero for every other mode** (`docs/05` R6: "zeros for lossless files"; uncompressed 12-bit files have no qstep either). The physical step of 14-bit (1 DN) and 12-bit (4 DN) samples still enters the σ model (`q²/12`, as `ym-noise` does with its default step), and the clip rule is `raw ≥ white − 4·qstep` with the plane the network sees, so those samples clip at exactly the white level (`docs/05` R1).

**Loss:** L1 on `log(1 + 16·x)` plus 0.2 × (1 − MS-SSIM) on the same, plus 0.5 × a gradient-domain L1 (weights 1 : 0.2 : 0.5). PSNR is evaluated in linear (peak 1.0) and in the log domain (peak log(1 + 16)).

**Splits:** whole scenes are held out (10%), by a fixed split seed (`data.SPLIT_SEED`, config key `data.split_seed`) and never from the training seed, so runs with different seeds and the evaluation script hold out the same scenes. Nothing is tuned on the gate sets. `evaluate` hands the noisy balanced mosaic to the external baseline on every set. The loader's expected PixelShift200 layout (`data.PixelShift200`) is assumed and unverified against the real download.

**Compute:** [HUMAN M2-H4] training runs need a CUDA GPU (one RTX 4080-class GPU: a few hours per run) or Apple silicon with MPS (slower). The repository has the scripts, a `make train-jdd` entry point (`make -C training/jdd train-jdd`), a config with seeds and a dry-run mode that trains for 50 steps on CPU in CI.

## Runtime (`ym-ml`)

```rust
pub trait TileModel: Send + Sync {
    fn input_shape(&self) -> TileShape;           // fixed, from the ONNX graph
    fn halo_half_px(&self) -> usize;              // from the weights header
    fn run(&self, inputs: &TileInputs, out: &mut TileOutput, ctx: &ExecContext<'_>) -> Result<()>;
    fn name(&self) -> &str;
}
```

`ExecContext` carries an optional type-erased handle to the host's GPU objects (`ExecContext::with_gpu`, `ExecContext::gpu::<T>()`) for backends that can use one. The burn backend does not read it: its weights live on the device they were loaded onto, so a host that owns a burn `Device` passes it to `BurnJdd::with_device` (a different device at run time would put the input away from the weights). `ym-ml` depends on `ym-gpu` only with the `burn` feature, to import the application's own wgpu device (`burn_backend::import_device`).

Three backends sit behind this trait:

- **CPU runtime** (`crates/ym-ml/src/cpu_runtime.rs`, pure Rust): the default, and the fallback for machines with no usable GPU and for the headless CLI on CI runners.
- **burn 0.22** on a wgpu device (cargo feature `burn`): the network is written with burn's tensor API, not `burn-onnx`, which is a build-time code generator that needs the model file at compile time. Sharing the application's wgpu device avoids copies between the image pipeline and the network.
- **ort 2.x** (ONNX Runtime, cargo feature `ort`) with the CoreML execution provider, CPU as fallback. It needs a copy to and from host memory per tile. ONNX Runtime is loaded at run time (`ORT_DYLIB_PATH`; no build-time download) **once per process**, and the first attempt decides: a failed load is remembered and returned again, because `ort` 2.0.0-rc.13 marks its library cell initialised even after a failed load. `OrtJdd::new` with no usable `ORT_DYLIB_PATH` is therefore an error.

Both features are off by default.

Weights are safetensors whose header holds the architecture (`config`, `rf_*`, `halo_*`, `model_version`); the file bytes are written deterministically so the SHA-256 is reproducible. A header is validated at load (`JddConfig::validate`: halo a multiple of 4 up to 1024 half-res px, finite positive input scales): a bad file is an error, never a panic or a huge allocation. `forward_planar` checks its shape too. A tiled run stopped by its callback returns `MlError::Cancelled`.

**Runtime choice (M0-08).** The benchmark compares burn and ort on a 512 × 512 half-res tile of the starting architecture on an M1 and an M3: time per tile, total for 50 MP, peak memory, numerical parity. ort wins only if its 50 MP frame time is at most 0.8 of burn's; on a tie burn wins (one device, pure Rust). The harness is `crates/ym-ml/examples/bench.rs` (`make -C training/jdd bench-weights`, then `cargo run --release -p ym-ml --features burn,ort --example bench -- --dir testdata/ml/random-512 --frame-mp 50`). It runs one child process per backend, so peak memory is per backend, and reports load time, time per 576 × 576 half-res tile (core 512 plus halo), a 50 MP frame tile by tile (or `--extrapolate`), peak RSS and the maximum absolute difference to the PyTorch output and to the CPU runtime. When one run has both a burn and an ort result it applies the rule to the 50 MP frame times and prints the outcome. It prints "not decision-grade" unless run on the target machines, which needs the M1 and M3 runs (HUMAN M0-H4). The choice is recorded in `docs/16-risks-and-decisions.md`.

## Parity and versioning

- Golden tiles (20 fixed inputs under `crates/ym-ml/tests/data/`: 12 for a small 8/16/24 instance with 1×1 up and head convolutions and 8 for a 6/8/12 instance with 3×3 ones; real generator output with noise, cRAW qstep, clipping and clean samples, 4 to 48 half-res px, square and not, down to a 1 × 1 bottleneck) run through PyTorch, ONNX Runtime and the chosen runtime, with **seeded random weights** (no trained weights exist). Max abs difference ≤ 2e-3 (f16) in the output's linear units; measured at about 1e-6. The golden ONNX files (`golden_small.onnx`, `golden_k3.onnx`) have dynamic height and width so ONNX Runtime can run every tile; the shipped export stays static-shape. Runs on every model change.
- Each weight file has a version and SHA-256; the version goes into export XMP.
- `data/models/<name>/MODEL_CARD.md`: purpose, architecture, parameters, training data and licences, **the weights' own licence** (Apache-2.0, or CC BY-SA 4.0 when any training data is CC BY-SA, with that folder's own `LICENSE`; `docs/14` rule 5), metrics on every gate set, known failure cases, date, git SHA of the training code.

## Gates (M2)

- `jdd` beats the classical fallback by ≥ 2 dB PSNR on Kodak, McMaster, Gharbi hard and moiré sets (mosaicked, clean) and on held-out PixelShift200 scenes with synthetic noise.
- No zipper or false colour in blind review of 30 crops.
- cRAW stripes ("blondie" artefacts) are visibly reduced against LibRaw + classical on 10 A7R cRAW samples.
- Visual parity with Capture One and DxO on an A1 ISO ladder 100–25600 (studio ISO series from the evaluation tier, local only) in blind A/B (ours preferred or tied ≥ 50%).
