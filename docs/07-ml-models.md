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
- Receptive field: 47 half-res px for this layout (2 convs + strided down per level, 3 levels, mirrored decoder, 1×1 convolutions in front of the pixel shuffles and in the head), so 94 full-res px. This is **computed from the real module graph** in `training/jdd/receptive_field.py` (traced with `torch.fx`, checked against a brute-force gradient probe in the tests); the tile halo is half of it, rounded up to a multiple of 16 full-res px, and never less than twice the worst one-sided reach of any stride phase (26 half-res px here): 64 full-res px (32 half-res) for the starting layout. Keep the receptive field ≤ 128 full-res px (asserted). Options `up_kernel=3` / `head_kernel=3` exist for M2-06 (RF 63 / 126 full-res px with both, halo 80).
- Must export to static-shape ONNX (opset 17) for a 512 × 512 half-res tile **core** plus halo (input 576 × 576 half-res px for the starting layout), and to safetensors. Avoid ops that CoreML or burn-onnx handle poorly (ConvTranspose and Resize are restricted under ORT's CoreML provider, per `research/sony-arw-to-arri-look-research.md`; prefer pixel-shuffle).
- Budget: ≤ 250 kFLOP per output pixel (≈ 12 TFLOP for 50 MP). Measure. Starting layout, measured by `count_flops` (2 FLOP per multiply-add of every convolution, activations ignored): 53.7 kFLOP per output pixel, 497,100 parameters.

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
- `data/models/<name>/MODEL_CARD.md`: purpose, architecture, parameters, training data and licences, **the weights' own licence** (Apache-2.0, or CC BY-SA 4.0 when any training data is CC BY-SA, with that folder's own `LICENSE`; `docs/14` rule 5), metrics on every gate set, known failure cases, date, git SHA of the training code.

## Gates (M2)

- `jdd` beats the classical fallback by ≥ 2 dB PSNR on Kodak, McMaster, Gharbi hard and moiré sets (mosaicked, clean) and on held-out PixelShift200 scenes with synthetic noise.
- No zipper or false colour in blind review of 30 crops.
- cRAW stripes ("blondie" artefacts) visibly reduced against LibRaw + classical on 10 A7R cRAW samples.
- Visual parity with Capture One and DxO on an A1 ISO ladder 100–25600 (studio ISO series from the evaluation tier, local only) in blind A/B (ours preferred or tied ≥ 50%).

## Implementation notes (M2-04, M2-05, M2-07 runtime, M0-08 harness)

Decisions taken while writing `training/jdd/` and `crates/ym-ml/`; the code and its tests are the reference.

**Network contract** (`training/jdd/model.py`, `crates/ym-ml/src/cpu_runtime.rs`)

- The network takes the **physical** units of `docs/05` and applies its fixed input scaling itself (buffer `in_scale`): planes × 0.25, sigma × 16, qstep × 16, clip × 1. The runtime never scales. The output is linear RGB in the same balanced units (white = 1.0 before gains), multiplied back by 4 inside the network.
- Sigma, qstep and clip are **single half-resolution planes**. Sigma and qstep are the RMS over the four photosites of the 2 × 2 cell of the per-photosite value in balanced units (`value · g0` per colour); the clip plane is 1 where any of the four photosites is clipped.
- Layout of the starting net: `enc0` (7 → 32) · down (3×3, stride 2) · `enc1` (64) · down · `bott` (96) · 1×1 conv to 4C + pixel shuffle + skip concat · `dec1` · 1×1 conv + shuffle + concat · `dec0` · 1×1 head to 12 channels. The head output is **added to a fixed bilinear demosaic** expressed as a 4 → 12 convolution on the packed planes (`bilinear_weights`), then pixel-shuffled once. Zero padding everywhere; the head is initialised small so a fresh network starts as the bilinear demosaic. Input height and width must be multiples of 4.
- Only Conv, Mul, Sigmoid, Add, Concat, Slice and DepthToSpace appear in the ONNX graph (`export.py --decompose-pixel-shuffle` swaps DepthToSpace for Reshape/Transpose for runtimes that lack it); the export fails if ConvTranspose or Resize appear.

**Tiling** (`crates/ym-ml/src/tiles.rs`)

- Tiles are cut in half-resolution units, cropped (never blended), and the halo outside the frame is filled by symmetric reflection of the packed planes (the Bayer phase is preserved because the planes are channels). Core sizes and halos are multiples of 4 so that every tile has the same stride-2 phase as a whole-frame run. Tiled and whole-frame outputs of the CPU runtime agree to 1e-4 in the tests (they are bit-identical in practice).

**Sample generator** (`training/jdd/data.py`)

- The target is the ground truth **clipped at the same per-channel white** (`g0_c`) as the input, so the network is never asked to invent clipped content (highlight reconstruction is stage R5, before this network).
- Exposure is set by the 99.5th percentile of the raw signal: log-uniform over 2⁻⁷ … 0.98 of white, or 1.05 … 4.0 with probability 0.2 (highlights pushed over white). ISO is log-uniform over 50 … 64000 (the A1's range ± 1 stop), `K = k0 · ISO / 100` with `k0 = 0.25 DN/e⁻` ± half a stop. `p = 0.3` cRAW and `p = 0.1` 12-bit apply to the non-clean samples; `p = 0.1` of all samples are clean (sigma 0, no noise). The sigma jitter is one log-uniform factor in [0.75, 1.25] per sample.
- The **qstep plane is the cRAW quantisation step and zero for every other mode** (`docs/05` R6: "zeros for lossless files"; uncompressed 12-bit files have no qstep either). The physical step of 14-bit (1 DN) and 12-bit (4 DN) samples still enters the σ model (`q²/12`, as `ym-noise` does with its default step), and the clip rule is `raw ≥ white − 4·qstep` with the plane the network sees, so those samples clip at exactly the white level (`docs/05` R1). A simulated 12-bit sample saturates at ADC code 4095 (16380 on the 14-bit scale).
- The PixelShift200 train/validation split is by **whole scene, from a fixed split seed** (`data.SPLIT_SEED`, config key `data.split_seed`), never from the training seed: runs with different seeds and the evaluation script must hold out the same scenes. `evaluate` hands the noisy balanced mosaic to the external baseline on every set.
- The loss weights are 1 : 0.2 : 0.5 for L1, (1 − MS-SSIM) and gradient L1 (docs/07 gives no weight for the last term). PSNR is reported with peak 1.0 (linear) and peak log(1 + 16) (log domain).
- PixelShift200's on-disk layout is **assumed** (`data.PixelShift200`) and must be checked against the real download.

**Runtime** (`crates/ym-ml`)

- `TileModel::run` takes an `ExecContext` instead of `&GpuQueue`: `ym-gpu` does not exist yet and `ym-ml` must not depend on it. The context carries an optional type-erased handle to the host's GPU objects (`ExecContext::with_gpu`, `ExecContext::gpu::<T>()`) for backends that can use one. The burn backend does not read it: its weights live on the device it was loaded onto, so a host that owns a burn `Device` passes it to `BurnJdd::with_device` (a different device at run time would put the input away from the weights). The trait also has `halo_half_px()` (from the weights header) and `name()`.
- A weights header is validated at load (`JddConfig::validate`, halo a multiple of 4 up to 1024 half-res px, finite positive input scales): a bad file is an error, never a panic or a huge allocation. `forward_planar` checks its shape too. A tiled run stopped by its callback returns `MlError::Cancelled`.
- ONNX Runtime is loaded **once per process** and the first attempt decides: a failed load is remembered and returned again, because `ort` 2.0.0-rc.13 marks its library cell initialised even after a failed load, and any later call would use an uninitialised handle and panic. `OrtJdd::new` with no usable `ORT_DYLIB_PATH` is therefore an error (it panicked inside `ort` before).
- Weights are safetensors whose header holds the architecture (`config`, `rf_*`, `halo_*`, `model_version`); the file bytes are written deterministically so the SHA-256 is reproducible. Golden tiles (`crates/ym-ml/tests/data/`) are **20 fixed inputs** (12 for a small 8/16/24 instance with 1×1 up and head convolutions, 8 for a 6/8/12 instance with 3×3 ones; real generator output with noise, cRAW qstep, clipping and clean samples, 4 to 48 half-res px, square and not, down to a 1 × 1 bottleneck) with **seeded random weights** (no trained weights exist). The CPU runtime, ONNX Runtime and, with its feature, burn must match PyTorch on every tile to ≤ 2e-3; measured at ~1e-6. The golden ONNX files (`golden_small.onnx`, `golden_k3.onnx`) have dynamic height and width so that ONNX Runtime can run every tile; the shipped export stays static-shape.
- The burn backend writes the network with burn's tensor API instead of `burn-onnx`, which is a build-time code generator that needs the model file at compile time. The ort backend loads ONNX Runtime at run time (`ORT_DYLIB_PATH`; no build-time download). Both are cargo features and off by default.
- **M0-08 harness**: `crates/ym-ml/examples/bench.rs` (`make -C training/jdd bench-weights`, then `cargo run --release -p ym-ml --features burn,ort --example bench -- --dir testdata/ml/random-512 --frame-mp 50`). One child process per backend, so peak memory is per backend; reports load time, time per 576 × 576 half-res tile (core 512 plus halo), a 50 MP frame tile by tile (or `--extrapolate`), peak RSS, and the maximum absolute difference to the PyTorch output and to the CPU runtime. It prints "not decision-grade" unless run on the target machines: the decision rule above needs M1 and M3 numbers (HUMAN M0-H4). When one run has both a burn and an ort result, the harness applies the rule to the 50 MP frame times and prints the outcome (ort only if its time is at most 0.8 of burn's, otherwise burn).
