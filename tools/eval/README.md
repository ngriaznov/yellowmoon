# tools/eval: auto-exposure study

Script-only folder (Python and shell, not a workspace member). It measures what the M1-08 gate (`docs/13`: "auto exposure
within 0.5 EV of the camera's embedded JPEG, median luminance in linear light, for at least 80 % of the golden set") actually
measures, and how far our automatic exposure is from the camera's own decision. It changes no algorithm; its findings are
summarised in `docs/02-colour-pipeline.md` (Auto exposure, Evaluation).

**Since 2026-10-14 `EV_auto` is the camera's own exposure**: a constant `+1.11 EV` (`ym_color::exposure::CAMERA_GAIN_EV`, the
median camera gain of the 12 local files) lowered by the highlight guard when a picture would clip, with no metering of our own
and no exposure-compensation term (`docs/02` Auto exposure, interim rule). So G3 is `1.11 - camera gain` per file: on the 12
local files it says how far each camera is from the median of the cameras (**in-sample**: the constant was fitted on the same
files, so a pass says little about other cameras), and on the synthetic tier it is zero **by construction** (the synthetic
JPEG engine is made with the same +1.11 EV). The sweeps and what-ifs below study the *previous* rule (metering to 0.18) and are
kept for the M1-H5 review and for the intelligent automatic settings that will replace the constant.

```
tools/eval/run.sh                  # unit tests, then the real files and the synthetic set
python3 -I tools/eval/autoexp.py testdata/real/files --label real
python3 -I tools/eval/autoexp.py testdata/synthetic/arw --label synthetic --one-per-scene
python3 -I tools/eval/autoexp.py testdata/real/files --label real --reuse     # redo the arithmetic from the cache
python3 -I tools/eval/autoexp.py testdata/synthetic/arw --label synthetic --one-per-scene --gate   # the M1-08 gate: exit code 3 when not met
python3 -I -m unittest discover -s tools/eval -p 'test_*.py'                  # the unit tests (no ARW needed)
```

Needs `numpy`, `Pillow` and `tifffile` (`pip install numpy pillow tifffile`). `PyOpenColorIO` (ACES 2 placement of mid grey
as an independent number) and `exiftool` (DRO and picture style of the camera, descriptive only) are optional. The `ym` used
is the dev build, `cargo build -p ym-cli` (`target/debug/ym`; the dev profile is optimised, a release build is not needed and
would cost a second profile of disk). Each photo costs two renders (40 to 180 s per photo for 10 to 33 MP on four shared cores, dev build).
The real files (`testdata/real/files`, local evaluation only) are never copied: renders go to `testdata/eval/<label>/`
(git-ignored) and the big ones are deleted once reduced (`--keep-renders` keeps them); the cache is a few MB per photo. `--reuse` takes a cache only for the same file (size and modification time), the same cache schema and the same `--display`; a cache of another version of the arithmetic is rendered again. The cache holds the output of the `ym` that made it (its version is in the report), so after rebuilding `ym` run without `--reuse`.

## The M1-08 gate (`--gate`)

The gate of `docs/13` and `docs/15` M1-08, as decided on 2026-10-13 (`docs/16`): the metric is **G3**, a file passes when `|G3| <= 0.5 EV`, and the gate is met when **at least 80 %** of the files pass.

```
python3 -I tools/eval/autoexp.py <folder or files> --label <name> --gate [--tolerance-ev 0.5] [--min-fraction 0.8] [--one-per-scene]
```

* **Exit codes.** 0 gate met (and every file analysed), 1 a file could not be analysed (the gate still ran), 2 usage, **3 gate not met** (wins over 1). The report is written in every case.
* **What counts.** The denominator is every file that was tried. A file that failed to analyse, or whose camera gain could not be read (a black or missing preview, so G3 is not a number), counts as a miss and is listed, so a run cannot reach 80 % by losing the files it does badly on. The bound itself passes (`|G3| = 0.5` is within). `--tolerance-ev` is the older `--tolerance`, which still works.
* **Outliers for M1-H5.** Printed after the table and written to the `## Gate` section of `report.md` and to `results.json` (`summary.gate`), worst first: file, body, ISO, the camera's DRO setting, the JPEG median (the key the camera rendered), `EV_auto`, the camera gain, G3 and whether we are brighter or darker. The JPEG cannot tell a deliberately dark or bright scene from a wrong exposure (the dark brick wall, the flash portrait against a bright wall), so those need a person (M1-H5); the DRO column marks the files whose camera gain includes a local tone mapping that the quantile match reads as part of the camera's curve.
* **In CI.** The `golden` job runs it on the synthetic tier (one raw variant per scene, 37 files) with the `ym` it just built. **That checks the machinery, not Sony, and G3 is zero there by construction**: the synthetic JPEG is our own rendering made with a known gain of +1.11 EV (`training/synth/preview.py`, calibrated to `CAMERA_GAIN_EV`; the estimator reads it back to about 0.02 EV), and `EV_auto` is that same constant. A pass says the estimator and the arithmetic still agree. What still means something on the tier is `summary.guard_check` (the highlight guard, the only part of `EV_auto` that looks at the scene: `ym` lowered the exposure exactly where the replica says the picture would clip, by the same amount, never raised it and by at most one stop) and the Rust tests of the guard and of the exposure compensation (`crates/ym-color/tests/exposure.rs`, `crates/ym-core/tests/auto.rs`). The real files (`testdata/real/files` and the three A1 photos of `testdata/readme-photos/src`, local only) and the CC0 tier run the same command locally (`tools/eval/run.sh`) until a golden corpus exists; the 80 % bar is a statement about real cameras and can only be closed there.

## What is run

One `ym` process per photo (a photo's automatic exposure depends on the other photos of a run through the burst median, so
batches would change the numbers), with an empty `YELLOWMOON_HOME` (default look Neutral, ACES 2, no imported looks):

```
ym info   <file> --json --quiet
ym render <file> --json --quiet --overwrite --out <work>/renders --format dng  --name {name}_eval_dng_ev0 --exposure 0
ym render <file> --json --quiet --overwrite --out <work>/renders --format tiff --name {name}_eval_tiff --bits 16 --compression none --colour-space srgb
```

* The **DNG at EV 0** is the scene-linear picture: linear AWG4 after C1 to C11 (the look is Neutral, the curve identity), before
  the display transform C12. Stored values are `round(65535 x / x_white)`, undone by `BaselineExposure = log2 x_white`
  (`ym-export` `dng.rs`, "Pixel scaling"). A DNG with the fitted Adobe profile (what `ym` writes since the DNG got its profile) has
  `BaselineExposure = log2(x_white) + fit_offset_ev + k`: the tool takes the offset from `data/dng/profile.json`, after checking that the DNG's
  `ProfileToneCurve` is that file's curve, and refuses the DNG otherwise (a `ym` built from another profile); `k` is the display placement
  (`DISPLAY_PLACEMENT_EV`, +0.85), read from `crates/ym-color/src/display/transform.rs` so that the two cannot drift apart. The
  exposure scales it exactly (C7's knee and toe scale with the exposure; checked on a synthetic chart, DNGs at +1.2 and -0.8 EV against
  the one at 0: the per-pixel ratio is the gain to 0.0006), so every exposure is `x 2^EV` of this one render. `ym render` has no scene-linear
  TIFF or PNG (those are C1 to C13, display-referred); the DNG is the scene-linear export.
* The **16-bit sRGB TIFF at the automatic exposure** is what the user sees: C1 to C13 through ACES 2.
* The **JPEG** is the one `ym info` reports (offset and length in the ARW, not the largest one in the file), decoded as sRGB, or
  as Adobe RGB (1998) when the ARW's Exif says `InteroperabilityIndex R03` (A7 II and A7 III here).

## Metrics

All errors are `log2(ours / reference)` in stops; positive means we are brighter. `Y` is linear luminance (AWG4 `Y` row for our
scene picture, Rec.709 for the JPEG and our display render), medians over every pixel.

| name | definition | what it mixes in |
|---|---|---|
| **G1** (the gate as built) | median `Y` of our scene-linear picture at `EV_auto` against the median `Y` of the JPEG | the camera's tone curve (contrast, DRO) and the scene's key; no display transform |
| **G2** | median `Y` of our display render (ACES 2 with the +0.85 EV display placement) against the JPEG's | everything of G1, plus the display's placement of mid grey (0.18 to 0.189 with the placement; it was 0.10 without) |
| **G3** | `EV_auto` minus the camera's *engine gain* | nothing of the display transform or of the camera's curve shape; only the camera's anchor |

The **camera engine gain** is the scene-referred exposure the camera's own rendering applies to the raw data: the raw level `x`
that its JPEG renders as 0.18 in linear light (the ISO 12232 convention, code 118), `gain = log2(0.18 / x)`. `x` is read from
the camera's tone curve, found by **quantile matching** of the JPEG's luminance against our scene-linear luminance: if the
JPEG is a global monotone function of the raw (it is, except where DRO or other local tone mapping acts), the p-th quantile of
one is that function of the p-th quantile of the other. No registration is needed (different crop and orientation do not
matter), but **the two pictures must be at the same spatial scale**: a block mean is narrower than a downsample, which is
narrower than the full-size picture, and quantile matching reads that difference as contrast. Our luminance is therefore
area-resampled to the size of the JPEG before its quantiles are taken (`y0_q`; the 8 x 8 block means that stand in for the
1/8 preview are used for the algorithm's own replica, not for the camera curve). With the 8 x 8 means the camera gain read
0.1 EV low on a real photo (NEX-5N: 1.13 against 1.21 at the JPEG's scale) and 0.07 to 0.22 EV low on the synthetic charts,
whose JPEG gain is known; at the JPEG's scale it is exact to about 0.02. `registration_corr` in the JSON (rank correlation of the two at 64 x 64, best over the eight
turns and flips) flags a preview that does not show the same picture. The curve's log-log slope at the anchor
(`camera_contrast_slope`) is reported: 1 for a pure gain, above 1 for an S-curve. On synthetic scenes the estimator can be
checked against the gain the synthetic JPEG was made with (`camera_gain_minus_synthetic_ev`).

Other outputs, per photo and in `report.md`:

* `display_of_grey`: our display luminance of scene 0.18 in this render (from its own pixels), and OpenColorIO's ACES 2
  neutral axis (0.100 for the plain transform, 0.189 for `0.18 * 2^0.85`, the placement `ym` applies; both are in the report). The *placement sweep* scales the display median to other placements of mid grey. The ARRI-style values
  are parameters, not measurements: no ARRI file is read (`AGENTS.md` rule 2). To measure a real ARRI display LUT, import it into
  a home folder and run with `--display arri --ym-home <that folder>`: `D(0.18)` then comes from the ARRI render.
* The camera's decision from the EXIF triple: `ev100`, program mode, flash, and `nominal_camera_gain_ev`, the figure for a
  camera that meters an average scene exactly (ISO 2720 meter constant 12.5 and ISO 12232 saturation speed 78 with a lens factor of
  0.65 put the grey at 0.104 of the white level, +0.79 EV). A reference line only.
* The *meter emulation*: for the photos the camera metered (a program mode, no flash), `log2(0.18 / S) + EC` for several
  statistics `S` of the EV 0 scene luminance; a statistic the camera holds at a fixed fraction of the white level gives one
  number for every scene, so its spread says how well it predicts the camera's AE.
* Sweeps of a constant exposure offset, the metering target, other meters (mean, median, percentiles, flat weights) and
  `ev = anchor + w (EV_previous - anchor)` (keeping part of the camera's placement), each scored by G1, G2 and G3. They vary the
  **previous rule** (`MeterConfig(rule="metered")`, metering to 0.18 with half the compensation; `ev_previous_rule` in the JSON is what it
  would have chosen), because `ym` no longer meters: its exposure is the constant, which the offset sweep moves like any other value.
* `ev_replica` is a Python replica of the algorithm (`metrics.meter`, ported from `ym_color::exposure` and tested on the cases
  of `crates/ym-color/tests/exposure.rs`: the camera gain, the highlight guard, the reported key statistic) on the DNG's 8 x 8 block
  means; its difference from `ym` per photo is reported (it must stay within 0.03 EV). `summary.guard_check` compares the two on the
  highlight guard. The what-if variants of the current rule add the replica's change to the value `ym` computed, so the replica's small
  difference from `ym` cancels; the variants of the previous rule are the replica alone.

## Limits

Twelve real photos from nine cameras are not a golden set: percentages and medians here are descriptive, and the sweeps (and
the camera gain that is now `EV_auto`) are fitted on the same twelve photos they are scored on. The JPEG is the camera's rendering (tone curve, DRO, colour) and not a
ground truth of exposure. Luminance differences between our colour (ARRI matrices, estimated light) and the camera's colour
move `Y` of saturated colours (blue sky) by a few tenths of a stop. A local tone mapping of the camera (Sony's DRO; three of
the nine files have DRO Auto) is not a global curve: quantile matching reads its average effect as part of the camera's gain,
and the report has a DRO column for that reason. The camera gain of a real photo has no truth to compare with and is good to
about 0.1 EV (a registered, binned reading of NEX-5N differs from the quantile match by 0.15 EV). The synthetic set's JPEG is a
made-up rendering (`training/synth/preview.py`: gain 2.2, which is +1.11 EV, soft shoulder, sRGB, mild S-curve), its scenes are charts, not
photographs; it checks the machinery against known answers (the camera gain is recovered to 0.02 EV, std 0.02, only when
our luminance is taken at the JPEG's scale) and says nothing about Sony's rendering.

## Files

| file | what |
|---|---|
| `autoexp.py` | the command: finds files, runs `ym`, reads DNG, TIFF and JPEG, writes `results.json`, `results.csv`, `report.md`; `--gate` evaluates the M1-08 gate (`evaluate_gate`) |
| `metrics.py` | pure numpy: colour decoding, the algorithm replica, quantile curves, ACES 2 neutral axis |
| `study.py` | the per-photo numbers and the sweeps over photos |
| `test_metrics.py`, `test_study.py`, `test_autoexp.py` | unit tests (31, 17 and 36; the gate logic and its exit codes and the guard check included), run without any ARW; one end-to-end test runs `ym` on a synthetic scene when `target/debug/ym` and `testdata/synthetic` exist, and is skipped otherwise (the `eval-tools` job of `ci.yml` runs them) |
| `run.sh` | builds `ym`, runs the tests and both sets |
