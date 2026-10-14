# training/synth/

A deterministic, physically based **synthetic raw set with ground truth**: raw mosaics of known scenes under known lights, seen
through the published Sony curves, with a known noise model, written out as ARW files by `tools/golden` (`golden synth`). It
exists because no real ARW file is reachable from the build environment (`docs/validation-data.md`). It is a stop-gap for the
golden runner (M1-15), the decoder oracle (every variant on every body) and the end-to-end colour checks, **until the CC0 tier is
fetchable**. Nothing here is a photograph and nothing is a measurement of a real camera: results obtained on these files are
reported as "synthetic only" and do not close a corpus gate.

```bash
tools/fetch/synthetic.sh            # the default set (37 mosaics, 61 ARW files, 202 MB, about 2.5 minutes), into testdata/synthetic/
tools/fetch/synthetic.sh --full     # 88 mosaics, 114 ARW files, 740 MB: + every light under every body, the 6240 x 4160 scene, the evaluation scene
tools/fetch/synthetic.sh --oracle   # + compare every file with LibRaw and ExifTool
python3 -m unittest training.synth.test_synth      # the checks of this folder (53 tests, about 35 s)
```

The script runs the two steps below; both are idempotent (up to date files are kept, `--force` rewrites).

1. `python3 -m training.synth.make_scenes --out testdata/synthetic/scenes [--full]` writes, per recipe, `<stem>.bin` (the mosaic),
   `<stem>.jpg` (the embedded camera JPEG) and `<stem>.json` (how to write the ARW, and the ground truth).
2. `cargo run -p golden -- synth --scenes testdata/synthetic/scenes --out testdata/synthetic/arw` writes
   `<stem>__<variant>.ARW` for every variant of every scene and decodes each file again with `ym_raw::decode` (exact for the
   lossless variants, within `qstep / 2` for cRAW, see `tools/golden/src/synth.rs`).

| File | Role |
|---|---|
| `spectra.py` | the lights, the six camera bodies, the reflectance bank (rawtoaces 190 patches, flat neutrals; the CIE 99 CES for evaluation) |
| `layout.py` | the scene: patch chart, gradients, neutral ramp, exposure ladder, zone plate, slanted edges, gratings, lamps, shadow zone, roll-off ramps |
| `sensor.py` | exposure, CFA sampling, the Poisson-Gaussian noise model, quantisation, clipping, 12- and 14-bit modes |
| `truth.py` | the ALEXA-estimate reference and the "Sony through the ARRI matrix" baseline, LogC4, AWG3 and AWG4 |
| `preview.py` | the embedded camera JPEG (a made-up rendering) |
| `recipes.py` | what is generated: the recipe list (the single source of file names) and the golden manifest entries |
| `make_scenes.py` | the generator and its command line |
| `test_synth.py` | unit and end-to-end checks, and the independent recomputation of the ground truth (`IndependentRecomputation`: own Planck, CIE D, mired interpolation, gamut matrices, LogC4, exposure and CCT; only the data files are shared) |

## What is simulated

**Scene.** A 1248 x 832 layout (3:2, width a multiple of 32 so cRAW can store it; ×5 for the 6240 x 4160 full-size scene, the
same composition with the frequency content of the fine detail unchanged). Every pixel is a mixture of bank reflectances under an
irradiance factor `m`:

| Region | Content |
|---|---|
| chart | the 190 rawtoaces training patches (19 x 10, 36 x 36 px) |
| neutral ramp | 24 flat reflectances from 0.95 to 0.006, including the white (0.90) and the 18 % grey |
| gradients | neutral (log reflectance, 0.005 to 0.95), hue sweep through eight saturated patches, grey to saturated, a 2D plate |
| exposure ladder | six reflectances at eight exposures, -5 to +2 EV |
| zone plate, edges, gratings | a square zone plate (0.5 cycles/px at the edge, the corners alias), two 5° slanted edges (neutral and an isoluminant red/green), line gratings of period 2, 3, 4, 6 and 8 px in both directions |
| blown highlights | a white lamp at +5.3 EV and a coloured one at +3 EV (clipped; the truth is the unclipped value), and three roll-off ramps from 0 to +6 EV (white, colour, grey) that cross the clip point |
| shadow | a feathered zone at -5 EV holding 16 patches, and a deep black block |

The scene is rendered as camera RGB `S_P^T (E o R) m` (linear in the reflectance and in the light, so one geometry serves every
light and body and the ALEXA reference), with the fine detail supersampled (4 to 8 sub-samples per pixel side, a box aperture),
then blurred by a Gaussian PSF of sigma 0.45 px (optical low-pass plus lens), then sampled through the CFA. A mixed-light scene
blends two renders (tungsten left, daylight right, a smoothstep over the middle 40 %).

**Lights** (`spectra.py`, the conventions of `training.sensor_map.lights`): blackbody 2850 and 3200 K, CIE D 5500, 6500 and
7500 K, CIE FL11, LED-B3, LED-V2 and the mixed scene. The CCT for ARRI's matrix is the generating temperature for the first two
families and the Ohno (2013) CCT of the SPD for FL11 and the LEDs.

**Bodies** (`spectra.py`): A7R IV (also the A1's stand-in), A7 III, A7S III, A6400, RX100 IV and the A1 with model ID 384 and the
A7R IV curves. The scenes of a stand-in are rendered through the stand-in's curves, so its raw is written as the render divided by the registry's `white_scale` of the body (red and blue) and its as-shot gains are multiplied by it (`docs/03`, White scale of stand-in bodies): the pipeline's own scaling then lands on the render again. The sidecar records `camera_white_scale` and `as_shot.white_scale`; `w_true_camera_rgb` stays on the profile's scale; the recipe hash carries the scale only for a body that has one. The curves are the published ones **in the raw scale** (green peaks at 1.0, red and blue lower), so the as-shot
white balance of a file is what a real raw file of that light has (R gain about 3, B gain about 1.4 under D65 for the A7R IV).

**Exposure** (`sensor.py`): the camera's meter puts the green response of the 18 % grey at 0.12 of `white - black` (about 3.1 stops
below clipping), at every ISO, so a higher ISO means fewer electrons for the same raw value. The metadata (shutter, f-number) follows
the meter equation from an assumed scene brightness per light (`ev100`).

**Sensor and noise** (`sensor.py`), in DN of the stored sample scale:

```
x      = k * m * S^T (E o R)                   noise-free signal above black (k: the exposure scale)
n_e   ~ Poisson(min(x, x_sat) / a)             photo-electrons
DN     = round(black + a * n_e + N(0, b)),  clipped to [0, white]
var    = a * x + b + 1/12                      (the clip makes the top a clipped Gaussian)
a      = (white - black) / FWC * ISO / 100     conversion gain, DN per electron (at least the ISO 100 value)
b      = (a * read_noise_e)^2
x_sat  = 1.05 * (white - black) * ISO / 100    the pixel saturates in electrons, 5 % above the ISO 100 white level
```

Black 512 and white 16383 at 14 bits, 128 and 4095 for the 12-bit mosaics (their own noise draw, not a shift). `FWC` is an assumed
areal density (1800 e-/µm², deliberately not the 2000 of `ym_noise::priors::ClassAssumptions`) times the pixel area of the body
(pitch from the public sensor format and pixel count); the read noise drops from a low-gain to a high-gain value at the body's
conversion-gain switch ISO. **These parameters define the truth; they are assumptions, not measurements.** Row noise and per-plane
offsets are zero. The noise truth is in the sidecar (`truth.noise`: `a` and `b` in DN and in normalised units, the regime, ISO).

**ARW metadata** (written by `golden synth`): Make SONY, Model and model ID per body, ISO, exposure time, f-number, focal length,
lens (none for the RX100 IV), a capture time seven seconds after the previous recipe (no accidental bursts), SR2 black, white (on the
14-bit scale, as Sony writes them) and the as-shot white balance = the true gains times `exp(error)` with the error drawn like an
automatic white balance (sigma 2 % for daylight and tungsten, 3 % for fluorescent and LED, 4 % for the mixed scene), neutral lens
tables (all zero: the scene has no lens effects), the embedded JPEG, and `Software` saying the file is synthetic.

**The embedded JPEG** (`preview.py`): the noise-free raw image, as-shot white balance, a colour matrix fitted on the 190 patches
under the true light, a fixed gain of 2.133 (`PREVIEW_GAIN`; +1.0654 EV, which `tools/eval/autoexp.py` reads at +1.11 EV, the median gain of the cameras' own JPEG engines, `ym_color::exposure::CAMERA_GAIN_EV`: the auto-exposure gate then passes on this tier by construction, `docs/02` Auto exposure), a soft shoulder, the sRGB curve and a mild S-curve. The gain does not touch the mosaics (their seeds depend on the generator version and the file name only); a sidecar whose `writer.preview.gain` is not the module's is regenerated. **It is a made-up rendering, not
Sony's**: it lets the auto-exposure comparison machinery (preview extraction, median luminance in linear light) run, and says nothing
about Sony's tone. The sidecar (`writer.preview.synthetic`), a JPEG `COM` segment and the ARW `Software` tag all say so.

## Ground truth in the sidecar (`truth`)

| Key | Content |
|---|---|
| `light` | SPD name, kind, class and class probabilities, CCT and its convention, CIE xy, `w_true_camera_rgb` (green = 1, raw scale), the ALEXA native white, for the mixed scene the two lights and the weight map |
| `as_shot` | the true gains, the SR2 levels written, the error drawn |
| `noise` | the Poisson-Gaussian truth above |
| `exposure` | the reference patch, the DN per response unit, the scene brightness |
| `arri` | the CCT, the native-to-AWG3 matrix at that CCT and AWG3 to AWG4, and the conventions |
| `probes` | per patch (chart, neutral ramp, ladder, shadow, lamps, black) and per sample point on the gradients and roll-off ramps: rect and interior (pixels), reflectance mixture, irradiance `m`, noise-free camera DN above black and which channels clip, and two references: `alexa` (the ALEXA-estimate: white-balanced to the light, ARRI matrix at the CCT, AWG3, AWG4 and LogC4) and `sony_through_arri` (the sensor-map-free baseline: Sony camera RGB through the same ARRI matrix). The mixed scene adds `alexa_local_white` and the local mixture weight |
| `features` | the geometry and parameters of the gradients, zone plate, edges, gratings, highlights and the shadow zone, for the image-quality checks |
| `bank`, `scene` | the reflectance names and sources, the scene size and scale |

The reference follows `docs/02` C3 to C6 and `training/alexa/ssf_core.py` (the arithmetic of the sensor-map gate, written to AWG4):
ALEXA estimate responses `S_A^T (E o R)`, each channel divided by its response to a perfect white under the light, ARRI's
native-to-AP0 matrix at the light's CCT interpolated in mired between the bracketing entries, `inv(AWG3 to AP0)` to AWG3, the
exact AWG3 to AWG4 matrix, LogC4. Values are **reflectance-referred** (exposure EV 0: a perfect white under the light is
`[1, 1, 1]`, the 18 % grey `[0.18, 0.18, 0.18]`, LogC4 0.2784); a render adds its own auto exposure, so compare after dividing by the
grey patch. Neutral patches are exactly neutral in both references (the ARRI matrices keep a balanced white). The highlight
patches carry the **unclipped** truth.

## Naming and the sets

File stem: `<scene>__<light>__<model>__iso<ISO>__<cfa>__<bits>b`, for example `chart__D6500__ILCE-7RM4__iso400__rggb__14b`; the
ARW is `<stem>__<variant>.ARW` with variant one of `uncompressed14`, `packed12`, `craw`, `lossless-l`, `lossless-ms`. Light tokens:
`bb2850`, `bb3200`, `D5500`, `D6500`, `D7500`, `F11`, `LEDB3`, `LEDV2`, `mixed`.

| Group | What | Mosaics |
|---|---|---|
| lights | A7R IV, ISO 400, the eight lights and the mixed scene, lossless L | 9 |
| bodies | the six bodies under D 6500 K, ISO 400: all four 14-bit variants, and a 12-bit mosaic as packed 12-bit | 5 + 6 |
| iso | A7R IV, D 6500 K, ISO 100, 800, 3200, 12800 (the noise ladder) | 4 |
| cfa | A7R IV, D 6500 K, GRBG, BGGR, GBRG: uncompressed, lossless L, cRAW | 3 |
| cross | the other five bodies under 2850 K tungsten and LED-B3, the body's usual variant | 10 |
| `--full` | every light under every body (30 more mosaics at ISO 400), the 6240 x 4160 scene (A7 III, D 5500 K: uncompressed, lossless L, cRAW at ISO 100, lossless L at ISO 3200, 12-bit packed) and the evaluation scene (the 99 CIE CES, 24 Ohta and 15 NIST patches in the chart; 18 mosaics, lossless L, ISO 100) | 30 + 3 + 18 |

`recipes.py` is the single source of these names: `tools/golden/manifest.json` lists the same files
(`python3 -m training.synth.make_scenes --write-manifest tools/golden/manifest.json`, and `test_synth` checks it is in step). Each entry's `set` says which
run writes it: `default` (a plain `synthetic.sh`, 61 files), `full` (only with `--full`, 35) or `eval` (only with `--full`, 18, CIE data).
Lossless M/S is written from a bilinear demosaic of the mosaic (it is camera RGB, not a mosaic); LibRaw reads Lossless L only for the
ILCE-1, ILCE-7x and ILCE-9x models (see `tools/golden/oracle_check.py`).

## Determinism and licences

Output is a pure function of the recipe, `GENERATOR_VERSION` (bumped when output changes), colour-science 0.4.7 and numpy: the
noise seed comes from the file stem, and a repeated run is byte-identical (tested). Another numpy version may draw other noise.
Reflectances: rawtoaces training data (Apache-2.0), flat neutrals. Lights and the evaluation reflectances come from colour-science
at generation time (CIE data, CC BY-SA 4.0): **the evaluation scene's outputs contain CES-derived images and live in `testdata/`
only, never committed or uploaded** (their manifest tier is `synthetic-eval`). The sidecars of all scenes carry their own notice.

## What it cannot show

Real cameras: firmware quirks, real lens tags, real tone curves and JPEG engines, real noise (row noise, banding, hot pixels),
optical black margins, real lossless M/S data, ARW 6.0. The scene has flat patches and test targets, not natural images: demosaic
quality on natural texture and the learned models need the corpus.
