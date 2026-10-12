# tools/dngfit

Fits the profile that a Yellowmoon DNG carries, and checks the DNG gates (`docs/09-export.md`, sections DNG with the look baked in and Gates).
Test and build tooling: never shipped. It renders with Adobe's own reference renderer, `dng_validate` from the DNG SDK
(`tools/dng/build_dng_validate.sh` builds it into `testdata/dng_sdk/`; `$DNG_VALIDATE` or `--dng-validate` name another).

| Command | What it does |
|---|---|
| `dngfit fit` | fit the tone curve, the `BaselineExposure` offset and the look table to ACES 2 and write `data/dng/profile.json` (about 2 s) |
| `dngfit eval` | how close Adobe's render with that profile is to ACES 2 on the test sets |
| `dngfit gate` | the DNG gates on the golden set: `dng_validate` passes, and its render is within a median ΔE00 of 1.5 of our JPEG |
| `dngfit validate FILE.dng...` | `dng_validate` on DNG files; prints what it complains about |

## The problem

A DNG pixel is scene-linear AWG4 after the look and curve (C1 to C11, `docs/02`). Yellowmoon shows it through the ACES 2
output transform (C12) and the sRGB encode. Other raw editors know nothing of ACES 2: they apply the tags of the DNG
profile (`docs/09`). So the profile has to make Adobe's pipeline produce what ACES 2 produces:

```text
x * 2^offset --ForwardMatrix1--> linear ProPhoto --> HSV --ProfileLookTable--> --ProfileToneCurve (RGB tone)--> sRGB
                                                                      target:  x --ACES 2--> sRGB
```

## What the fit does

1. **Scale.** The tone curve's input range is `[0, 1]` after the exposure gain, and ACES 2 reaches 94 % of display light
   only at a scene value of 16 (it reaches white asymptotically). The curve therefore covers scene values `0` to `x_min = 16`:
   `fit_offset_ev` is `-log2(x_min)` = -4, and the stored values are divided by `x_white`, with `BaselineExposure` =
   `log2(x_white) + fit_offset_ev`. **`x_white` must be at least `x_min`** (the writer floors it at 1.0 today): for a
   *negative* `BaselineExposure` the DNG SDK's renderer does not scale, it applies a highlight shoulder (linear below 0.25
   of the range, then a quadratic ending at 1), so the same profile would render a photo with `x_white` 2 differently
   from one with `x_white` 8. Measured with `b = -3` and an identity profile: scene 0.40 renders at 0.086 of display
   light instead of 0.050, and 0.84 at 0.654 instead of 0.105. With the floor, `BaselineExposure` is zero for nearly
   every photo and positive for the few whose 99.99th percentile is above 16.
2. **Tone curve** (neutral axis). The target is ACES 2's response to greys, `T(u) = srgb_decode(aces2(u * x_min))`. The
   renderer evaluates a spline through the profile's points into a table of 4096 steps and interpolates linearly, so the
   fit measures the table with `dng_validate` (4097 greys), corrects each knot by the difference to the target, and repeats
   (8 rounds, 0.2 s). 64 knots, geometric through the shadows. The result matches ACES 2 on the neutral axis to ΔE00
   0.015 at worst.
3. **Look table** (chroma). The renderer applies the table in HSV of linear ProPhoto before the tone curve: a colour
   `(h, s, v)` becomes `(h + dh, s * ds, v * dv)` with the three modifiers interpolated trilinearly from the grid (hue wraps;
   the value scale multiplies the *encoded* value, which is then decoded). `src/model.rs` is a model of the whole pipeline
   (checked against `dng_validate` on 4000 random colours with random tables: median 1.4e-4, worst 1.9e-3 in encoded values).
   What the table must produce at a colour does not depend on the table: it is the colour the tone curve maps to ACES 2's
   output, found by inverting the tone stage exactly. Each test colour on a regular HSV grid (3 samples per grid cell and
   direction, 213k colours, only those the scene data can hold: no negative AWG4 component) gives a required modifier,
   and the table is the trilinear field that fits them (`src/lookfit.rs`: weighted linear least squares by conjugate
   gradients, the weight following how far each modifier moves the colour, a pull towards the node-exact solution where
   no colour is near). The value scale at zero saturation is fixed at 1 (the DNG specification). Fitting the nodes alone is
   exact at the nodes but leaves the error of the interpolation between them, large in the first value division where
   ACES 2's toe bends fast; the least-squares fit cuts the 95th percentile of random colours from 1.4 to 0.5.
4. **Report.** Four test sets that the fit does not use, rendered with `dng_validate` and compared with ACES 2 in CIEDE2000
   (printed, and kept in the `fit` object of the profile file).

The profile file (`data/dng/profile.json`, schema in `src/profile_file.rs`) holds `fit_offset_ev`, `tone_curve`,
`look_table` (`dims`, `encoding` 1, `data` as hue shift in degrees, saturation scale and value scale per entry, value-major
then hue then saturation) and the provenance. `ym-export` embeds it (`docs/09`, As built, dngfit).

## Gates (`dngfit gate`, `tools/dng/run_gates.sh`)

For each golden file the real `ym` writes the DNG and the JPEG (quality 92, sRGB) with the same settings. `dng_validate -v
-32 -cs1 -tif` renders the DNG: the file must pass with no warning and no error, and the render (turned upright by the
DNG's `Orientation`) must be within a median CIEDE2000 of 1.5 of our JPEG, **for every file**. `--png` adds the 16-bit PNG
(no JPEG coding or dither), which shows what the profile alone costs.

Until `ym-core` writes the fitted profile itself, `ym` writes the identity profile, so `dngfit gate` writes each DNG again:
`reprofile` reads the pixels back, writes them with the product's DNG writer (`ym-export`), the fitted profile and the
`x_white` floor, and renders that. The last rows of such a file are white pixels that set the writer's white level; they are
left out of the comparison. The DNG `ym` really wrote (EXIF, XMP, orientation, crop) is validated too (`dng_validate -v`,
no render), so the first gate is about the real export. Which DNGs are written again is decided per file by the DNG's
`ProfileLookTableDims`: the identity profile has 2 x 2 x 2, so a `ym` that already writes the fitted profile is checked as
it is without any change to the command line (`--profile FILE` on such a `ym` is an error: re-profiling would scale its
pixels by the wrong exposure). `--as-written` checks the DNGs exactly as `ym` writes them. `--require-all` (set by
`tools/dng/run_gates.sh` unless `--filter` is given) fails when a file of the selection is not on disk.

## Tests

`cargo test -p dngfit`: the model of the renderer (HSV, RGB tone, table interpolation, tone inverse), the least-squares
solver (a linear field recovered from scattered samples, fixed entries), the node-exact solve, the knot layout, the
profile file (round trip, validation, the committed file loads and is in the form the tool writes), the gate's maths
(orientations, comparison, mask). With `dng_validate`: the model against the renderer, the committed profile against
ACES 2 on the test sets (and a negative control without the look table), a DNG with the profile passing `dng_validate`
without a warning, the fit repeating bit for bit. With python3 (numpy, tifffile, Pillow, rawpy) and exiftool: a DNG with
the fitted profile through `tools/dng/check_dng.py`. Tests that need a tool say `SKIPPED` without it and fail instead when
`YM_REQUIRE_EXTERNAL_TOOLS` is set.
