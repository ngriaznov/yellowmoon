# 09 · Export

How a developed photo becomes a JPEG, TIFF, PNG or DNG file: the options, the queue, the pixel pipeline of each format, colour management, metadata, the writers, the size estimate, the DNG writer with its fitted profile and gates, and the export tail on the graphics device. Crates: `ym-export` (writers) and `ym-core` (export queue). UI: `design/mockups/export.html`. CLI: `ym render` (`docs/10-cli.md`). App and CLI share one code path, so the same inputs and settings give byte-identical files. XMP timestamps are the capture time, never "now", so the bytes are reproducible.

## Options

**Output sharpening.** Every raster export (JPEG, TIFF, PNG; never DNG) is sharpened after the resize by a linear-light luma unsharp mask with halo control (`docs/algorithms/04-output-sharpening.md`). It is a per-photo develop setting, not an export option: Detail › Sharpening › Output, 0–200 %, default 100 %, with the spec's screen preset otherwise (radius 1 px, threshold 1 level, halo suppression 60 %). The same setting applies to the viewport, tiles and thumbnails at their own size. There is no export switch for it (the Export panel and `ym render` have no on/off option beyond the amount, and 0 is off), and no print preset in the app or in `ym`: the spec's print preset exists only in the code (`ym_gpu::cpu::output_sharpen::OutputSharpen::print`).

| Option | Values | Default |
|---|---|---|
| Photos | This photo / Picked · N / All · N | Picked if any are picked, else This photo |
| Format | JPEG, TIFF, PNG, DNG | JPEG |
| JPEG quality | 50–100 | 92 |
| TIFF bit depth / compression | 8 or 16 / None, LZW, ZIP | 16 / LZW |
| PNG bit depth | 8 or 16 | 16 (shares the bit-depth preference with TIFF, as in the mockup) |
| Colour space (not DNG) | sRGB, Display P3, Rec.2020 | sRGB ("Best for sharing and the web.") |
| Size (not DNG) | Full, or Long edge 1000–full px in 20 px steps | Full |
| Save to | folder | `<source folder> export` next to the source folder; `~/Pictures/Yellowmoon export` if that isn't writable |
| File name | template with `{name}`, `{look}`, `{date}`, `{seq}`, `{camera}` | `{name}_{look}` |
| If the file exists | Skip, Replace, Keep both (adds " 2", " 3" …) | Keep both |

The export panel shows an estimate ("≈ 120 MB") from `export_estimate`, and the equivalent CLI command with a Copy button. The command carries the export options, the look and, when it is not the default 100 %, the output sharpening (`--output-sharpening N`: the amount most photos of the job have, the earliest on a tie, like the look; a photo with another amount counts among the adjusted ones below). Per-photo adjustments beyond these don't fit on one command line; when any included photo has them, the panel adds "N photos have adjustments this command doesn't include." (the CLI can take them through `--settings`, `docs/10-cli.md`). Defaults are remembered as preferences; per-photo settings are not.

`ExportOptions` serialises exactly as `ExportOptions` in `src/api/types.ts` (flat: `format`, `jpeg_quality`, `bit_depth`, `tiff_compression`, `color_space`, `size`, `destination`, `name_template`, `if_exists`); `ExportOptions::format_spec()` gives the richer `ExportFormat` enum the writers take.

## Queue

- Jobs run one photo at a time, tiled at full resolution (on the graphics device when the policy allows it, `docs/10-cli.md` Graphics device, else on the CPU), with decode of the next file overlapping.
- The panel shows each file as Waiting / n% / Done or an error. "Cancel" stops after the current tile and deletes the partial file.
- Exports keep running while the user browses or edits other photos; edits to a photo already queued don't affect its job (settings are snapshotted at queue time). The same holds for photos taken out of the filmstrip (Remove from filmstrip, `docs/11-ui.md`) or a session that is closed: the job keeps the photos it was started with and finishes them, and the files on disk are never touched.
- Files are written to a temporary name in the destination folder and renamed when complete. Originals are never touched.
- Auto-update waits: "Update ready" stays disabled until the queue is empty (`docs/12-distribution.md`).
- Finished state: "N photos saved to <folder>." with "Show in Finder" and "Done" (back to the tool panel).

## Pixel pipeline per format

- **JPEG, TIFF, PNG:** the full chain through C13 (`docs/02-colour-pipeline.md`) for the chosen colour space, with crop, straighten and orientation applied to the pixels. Then, in order:
  1. resize (if Long edge) with a Lanczos-3 filter in linear light;
  2. **output sharpening** (Detail › Sharpening › Output, `DevelopSettings.output_sharpening`, default 100 %, 0 is off) on the picture at the size it is written, in linear light;
  3. the display encode (C13);
  4. 8-bit dither (8-bit outputs only).

  The sharpening stage is the screen preset of `docs/algorithms/04-output-sharpening.md` at the photo's amount, with the scale `s = long edge written / long edge rendered` (1 for a full-size export); it runs after the resize and before C13 for a full-size export too. JPEG uses 4:4:4 chroma at quality ≥ 90 and 4:2:0 below; baseline (not progressive); optimised Huffman tables. On the graphics card (`GpuStages::EXPORT_TAIL`, part of `Auto` on a hardware adapter) the sampler, the colour chain, the resize, the output sharpening and C13 all run on the device and the writers pull the finished rows band by band, so the display-linear picture is never a host frame. The CPU path described here is the reference and the fallback (see "As built (export tail on the device)").
- **DNG:** **is never sharpened, in any way.** Both renders of a DNG, the raw pixels and the preview, are made with R8 capture sharpening off (neither Richardson–Lucy nor diffraction deconvolution runs, whatever Detail › Sharpening says) and without output sharpening. The adaptive ratio demosaic's own shadow step (`shadow_sharpening`) stays on: it belongs to the demosaic, not to R8. (`--sharpen` and `--output-sharpening` with `--format dng` are warned about and ignored; both settings are still recorded in the XMP's `ym:Settings`, which is the photo's settings, not a record of what the DNG holds.) The render stops after C11 (look and curve applied), decodes LogC4 back to linear AWG4 and writes LinearRaw (see DNG with the look baked in); the display placement is carried by `BaselineExposure`.

  **A look with its own display rendering (LogC3 to Rec.709) is allowed, approximately.** ARRI's display rendering has no scene-linear form, so the DNG's pixels are the scene values that our ACES 2 rendering (with the display placement) turns into the look's picture at the chosen intensity, and Adobe's renderer, which runs the profile fitted to ACES 2, shows approximately that picture (What a DNG carries has the measured numbers). The same holds when an imported ARRI display LUT is chosen in Preferences. The help text says it in one sentence (`ym_export::dng::OWN_DISPLAY_NOTE`, used word for word by `ym render --help` and, when the job has such a look or display rendering, by the export panel): **"With a look that includes its own display rendering, the DNG is close to, not the same as, the picture you see."**

## Colour management

Every JPEG, TIFF and PNG embeds an ICC profile for its colour space: sRGB (IEC 61966-2.1 curve), Display P3, or Rec.2020 (a pure gamma 2.4, the BT.1886 inverse our encode uses). PNG also gets a `cICP` chunk. TIFF uses tag 34675.

**Profiles (`icc.rs`).** Generated by our own code, byte-for-byte deterministic, 612 to 632 bytes each:

- v4.3, class `mntr`, colour space `RGB `, connection space `XYZ `, a fixed creation date (`PROFILE_DATE`, a constant, never "now") and an MD5 profile ID; `desc` and `cprt` are `mluc` text.
- `wtpt` and the header illuminant are the D50 PCS white (0.9642, 1, 0.8249). `chad` is the Bradford adaptation from D65 (x 0.3127, y 0.3290) to D50. The colorants are `chad` times the primaries' D65 matrix from `ym-color`.
- One shared `para` tone curve: type 3 `g 2.4, a 1/1.055, b 0.055/1.055, c 1/12.92, d 0.04045` for sRGB and Display P3, type 0 `g 2.4` for Rec.2020.
- The same Bradford adaptation is what the DNG writer uses for its forward matrix.

**Verification.** LittleCMS 2.14 (the C library through ctypes, and Pillow's ImageCms) reads 15 files (5 format variants × 3 spaces) × 272 patches (a 6×6×6 cube, a grey ramp, four near-blacks): XYZ within 1.7e-5 of `ym-color`'s maths, sRGB codes within 1 (LittleCMS with `NOOPTIMIZE`; its default fixed-point matrix-shaper optimisation is off by up to 11 codes on extreme out-of-gamut dark patches, which is its approximation and not the profile).

**PNG `cICP`** is written by hand (the `png` crate parses it but does not write it). Code points `[primaries, transfer, matrix, full range]`: sRGB `1, 13, 0, 1`; Display P3 `12, 13, 0, 1`; Rec.2020 `9, 1, 0, 1`. H.273 has no code for the BT.1886 curve; transfer 1 is the one it pairs with a BT.1886 display, and the ICC profile carries the exact gamma 2.4. PNG readers that honour `cICP` use it instead of the ICC profile, so they decode the BT.709 transfer, which is not the pure gamma 2.4 of the embedded profile and of our encode. The `iCCP` profile name written by the `png` crate is `_`.

## Metadata

**EXIF (`exif.rs`).** Built from `ExifInfo` (make, model, lens make and model, exposure time, f-number, ISO, focal length and 35 mm equivalent, exposure compensation, capture time with sub-seconds and UTC offset, optional GPS), never copied from the source file: orientation 1 (the pixels are written upright), no thumbnail, no maker note, no serial numbers, text reduced to ASCII.

- `ComponentsConfiguration` is Y, Cb, Cr plus `YCbCrPositioning` for JPEG and R, G, B for TIFF and PNG; `ColorSpace` is 1 for sRGB and 0xFFFF for the others.
- ISO above 65,535 uses `ISOSpeed` with `SensitivityType` 3.
- GPS is written unless the preference "Remove location from exports" is on; with it on, the GPS directory is not written and the output is byte-identical to one made without a position (tested). GPS seconds are computed on whole ten-thousandths of an arc second, so a latitude a hair below a minute carries into the minute instead of writing "60 seconds".
- `DateTime` (IFD0) is the capture time.
- The Exif pixel dimensions of a DNG are the size of the default crop when there is one (the meaningful image), not of the whole raw frame.
- Every TIFF and DNG directory is padded to an even offset before it is written (`exif::finish`): the `tiff` crate writes a directory where the previous data ended, so an odd-length text value (a 12-character lens name gives 13 bytes with the NUL; an app version such as `0.1.10`) would otherwise put the next directory at an odd offset, which the TIFF specification forbids. `tests/structure.rs` walks every directory of TIFF and DNG files built with odd-length text.

**XMP (`xmp.rs`).** `xmp:CreatorTool` (`"Yellowmoon <version>"`), `xmp:CreateDate`, `xmp:ModifyDate` and `xmp:MetadataDate` (all the capture time), `dc:format`, then the twelve `ym:` properties of our namespace `https://yellowmoon.app/ns/1.0/` (prefix `ym`) in this order, with a fixed property order and no padding:

| Property | Example |
|---|---|
| `ym:PipelineVersion` | `0.5.1` |
| `ym:Profile`, `ym:ProfileVersion`, `ym:ProfileMatch` | `sony-ilce-7rm4`, `1.0.0`, `default` |
| `ym:Look`, `ym:LookSHA256`, `ym:LookIntensity` | `arri:3000:…`, `…`, `100` |
| `ym:DisplayTransform` | `aces2-sdr100-p3lim` or `arri-display:…` |
| `ym:Models` | `jdd 1.2.0 (sha256…)`; with a lens (`docs/21`) `lens vintage 1 (sha256:0123456789ab)`, and `, anamorphic 1 (sha256:…)` after it when Anamorphic is above 0: the lens, its data version and the first 12 hex digits of the SHA-256 of its `lens.json` (and of `anamorphic.json`) from the shipped `data/lenses/SHA256SUMS`, so a file says which lens data made it. Entries are joined with `, `. Absent without a lens (all three amounts at 0, or no lens data), and while the classical demosaic is the only model. |
| `ym:Auto` | `rule` or `apple`: which Auto made the photo's automatic values (`docs/19-apple-auto.md` section 9; `apple` when they came from an Apple Auto answer, the photo's or its burst's, even if the user then set all four by hand) |
| `ym:AppleAuto` | only when Apple Auto was on for the photo: compact JSON, keys sorted, `{"instructions":"default","instructions_sha256":"…","model":"AFM 3 Core","os":"27.0.1 (27A404)","plan":"…","prompt":3,"result":"answered","scene":"…","steps":{"blacks":null,"contrast":1,"exposure":1,…},"table":2}`, or `"result":"failed"` with `"reason"` (`timeout`, `rate_limited`, …) and no steps |
| `ym:Settings` | compact JSON of the resolved DevelopSettings (a `serde_json::Value`, keys sorted), in which Auto contrast and saturation are pinned to the numbers they resolved to, as Auto exposure is |

With Apple Auto off an export carries every property except `ym:AppleAuto`, with `ym:Auto` = `rule`.

No sidecars. Nothing is written anywhere except the exported file.

## Writers (`ym-export`)

The crate depends on `ym-color` and on `ym-gpu` (CPU code only, for output sharpening; `ym-gpu` depends on `ym-color` and `ym-noise`, so there is no cycle), not on `ym-raw` or `ym-core`: its inputs are plain structs that `ym-core` fills in.

**Entry points.** `write_export(&ExportRequest, &ExportDestination, &ExportHooks) -> ExportOutcome` writes one photo to one file, atomically. `encode_to_vec` and `encode_to_writer` give the same bytes in memory, for tests and hashes. `estimate_bytes(width, height, format, size)` and `estimate_bytes_with_detail(width, height, format, size, Option<DetailStat>)` serve `export_estimate` (see Size estimate). `request::write_export_bands(&BandRequest, &mut dyn BandSource, &ExportDestination, &ExportHooks)` is `write_export` for a band source (the export tail on the device); it refuses a DNG.

**Pixels in.** JPEG, TIFF and PNG take `f32` RGB, interleaved: `DisplayLinear` (linear light in the export colour space's primaries, the output of C12; the writers do C13 with `ym_color::encode`) or `DisplayEncoded` (already C13-encoded, 0 to 1; a resize decodes it to linear first). DNG takes `SceneLinearAwg4` (C11 output decoded from LogC4) plus a display-linear sRGB preview. The writers convert a band of about one million pixels at a time, in parallel row by row, so the bytes do not depend on the thread count (tested). On the CPU path the input buffer is one full image; the device path streams bands instead (As built (export tail on the device)).

**Output sharpening in the writer (`PictureStages`, `resize::sharpen_output`).** The photo's amount reaches the writer as `PictureStages { output_sharpening_pct }` (`encode_to_writer_with`, `encode_to_vec_with`, `write_export_with`; the plain `encode_to_writer`, `encode_to_vec` and `write_export` take the default, which is nothing). `ExportRequest` has no field for it: the amount is not an export option, and `ExportOptions` (the wire type) has none. `ym-core::export_photo` passes `DevelopSettings::output_sharpening` for JPEG, TIFF and PNG and nothing for a DNG.

1. In `encode_to_writer_with` the stage sits between the resize and the encode: the resized `LinearImage` (or the rendered pixels, for a full-size export, decoded to linear first when they are `DisplayEncoded`) goes through `sharpen_output`.
2. `sharpen_output` runs `ym_gpu::cpu::output_sharpen::OutputSharpen::screen()` at the amount and the scale `scale_for_edges(out long edge, rendered long edge)`, and the writers take the result.
3. **No amount, no stage:** 0, and anything below 0.1 % (spec 04: `A < 0.001`), makes no image and no change, so the bytes are exactly those of the plain call (tested for JPEG, TIFF 8 and 16 bit, PNG 8 and 16 bit, at full size and resized).
4. The stage runs in strips of 1024 core rows with the stage's own receptive field (`SharpenPlan::halo_px`, 17 px at full size for the screen preset, at most 21) above and below. The tiling contract of spec 04 section 5.2 gives the whole-frame result bit for bit (tested against the whole-frame run on a 1100-row image, so two strips). A cancel check and a progress report fall between strips, and the working planes stay a few tens of megabytes at any size.
5. Progress shares: resize 0 to 30 %, the stage 35 % (right after the resize, or from 0 without one), the encode the rest.

A full-size export holds one more copy of the picture while it runs (12 bytes a pixel: the sharpened image next to the rendered one); a resized one holds the resized image and its sharpened copy. The amount is checked (0 to 200, finite: `ExportError::InvalidInput` otherwise). The written file's XMP records it in `ym:Settings`; settings without the field read as 100 when fed back.

**Resize (`resize.rs`).** Lanczos-3, separable, support scaled by the reduction factor, taps outside the image dropped and weights renormalised: the same convolution as Pillow's `LANCZOS`, to which it is compared on float images (worst difference 0, six size pairs). Accumulation is in f64, so a flat field is exact. Step-edge overshoot stays under 10 %; the zone-plate test bounds the output to [-0.10, 1.10] for shrink factors 2, 3 and 5. The long edge never upscales; the short edge is rounded half up. The UI grid (1000 px up in 20 px steps, then the full size) is `Size::ui_long_edge_choices`; the library accepts any long edge from 16 to 65,535 px.

**JPEG.** Baseline, 4:4:4 at quality 90 and above, 4:2:0 with box-averaged chroma below, optimised Huffman tables, JFIF 300 dpi, then `APP1` Exif, `APP1` XMP and the ICC profile in `APP2` chunks. The `jpeg-encoder` crate writes optimised-table output as one scan per component, which `zune-jpeg` (Rust's `image` crate) decodes with wrong colours; `src/jpeg/interleave.rs` rewrites it as a single interleaved scan with tables rebuilt by T.81 Annex K.2 (coefficients and quantisation tables unchanged; the decoded pixels are identical to the standard-table interleaved file, tested over 17 sizes, both samplings and three quality settings). Sizes above 65,535 px are refused with a message.

**TIFF.** RGB strips of about 512 KB, compressed in parallel with the `tiff` crate's LZW and Deflate; the horizontal predictor for both compressed modes; ICC 34675; XMP 700; Exif and GPS sub-IFDs; orientation 1; 300 dpi. All out-of-line values start on even offsets (the `tiff` crate does not align them; the writer pads). Files above 4 GB are refused.

**PNG.** `iCCP`, `cICP`, `pHYs`, `eXIf` and an `iTXt` with `XML:com.adobe.xmp`.

**Deflate backend.** The bytes of the ZIP strips and of the PNG `IDAT` and `iCCP` depend on which deflate implementation `flate2` is built with: zlib-rs and miniz_oxide write different, equally valid streams, zlib-rs about 1 to 2 % smaller. Cargo features are additive and `ym-looks` pulls `zip` with `deflate-flate2-zlib-rs`, so `ym-export` declares `flate2` with the `zlib-rs` feature itself (`Cargo.toml`) and every build writes the same bytes (`golden_hashes_match`). zlib-rs picks its CPU-specific code at run time; its output at the level used here (level 6) is not known to depend on it, and `golden_hashes_match` must pass on both Apple silicon and Intel (a Mac check, M0-H4). LZW on noisy 16-bit samples can produce a file as large as the uncompressed one (measured 0.8 to 1.2 of raw on noisy data; a 24 MP frame: LZW 182.4 MB, ZIP 132.3 MB, raw 145 MB); Deflate is smaller.

**Names and folders.** `{name}` is the source file name without extension, `{look}` the look's slug, `{date}` the **capture date** as `YYYY-MM-DD` (`undated` when unknown; never the export date, for determinism), `{seq}` the 1-based number of the photo in the job with at least three digits, `{camera}` the camera body as a slug (`sony-a1`). `{camera}` and `{look}` are slugs exactly as the export panel previews them (`Sony A1` gives `sony-a1`): `naming::slugify` is the same function as `slugify` in `src/export/template.ts`, so `ym-core` and the CLI pass `slugify(body)`. Unknown `{words}` stay as typed (as in the UI preview).

- The stem is made safe: `/ \ : * ? " < > |` and control characters become `_`, leading dots and trailing dots and spaces are dropped, at most 200 bytes (`sanitize_stem`).
- The default folder is `<source folder> export` next to the source folder, then `~/Pictures/Yellowmoon export`; a folder the user chose is never replaced by a fallback. The default folder is named after the source folder as the photo was opened. Symbolic links in the path are not resolved (on a Mac `/var/…` stays `/var/…` and does not become `/private/var/…`, and a source folder that is itself a link gets its export folder next to the link, which is where the panel and the CLI command say it goes), except for a folder with no name as written (`..`, or a file with no folder part), which takes the name of the folder it resolves to. A name that is not UTF-8 is kept byte for byte. `ym_export::naming::default_destination` is the one implementation; `ym-core` re-exports it.
- Skip and Keep both never overwrite, even if another process creates the file during the export: the final step is a hard link, or a checked rename on file systems without hard links such as exFAT.
- A name that an earlier photo of the same export job has already written is always numbered (the Keep both style, " 2", " 3" …), whatever "If the file exists" says, so Replace never overwrites the job's own earlier photo and Skip never takes it for a file from before the job. A file that was there before is replaced or skipped as asked; when the choice was Replace or Skip, the numbered photo carries a note saying why, and with Keep both it needs none.
- The temporary file is `.yellowmoon-<pid>-<n>.tmp` in the destination folder, fsynced before the rename; an error, a cancel or a panic deletes it. A target that resolves to the photo's original file is refused.
- Cancel is polled between bands of about one million pixels, so "after the current tile" is a fraction of a second of work.

**Write errors.** An I/O error of the JPEG, PNG or TIFF encoder (a full disk, a folder that went away) maps to `ExportError::Write { format, source }` ("Couldn't write the PNG: No space left on device (os error 28). Check that the folder exists, is writable and has free space."), the same advice as `ExportError::Io`; `write_dng` labels the TIFF encoder's errors as DNG. Non-I/O encoder failures say "Encoding the X failed". The temporary file is removed and a failed Replace keeps the old file. Test: `tests/disk_full.rs` (a sink that fails with ENOSPC after N bytes, for every format).

## Size estimate

A file's size depends on the pixels as much as on the pixel count: nine real renders range from 0.7 to 2.7 bits per pixel at JPEG quality 92 at full size. The estimate (`estimate`) therefore takes a cheap measure of the picture's fine detail next to the export's size and options.

- *The statistic.* `ym_export::estimate::DetailStat` is the mean absolute difference between neighbouring pixels (horizontal and vertical) of the luma (Rec.709 weights on the display-encoded values) of the picture box-averaged by a whole factor to at most 240 px on its long edge, normalised to 240 px with `(s/240)^0.25` for a preview a little smaller (the scale exponent is the median of the measured change between 240 and 480 px). `DetailStat::measure_rgb8(w, h, &[u8])` takes an 8-bit sRGB preview such as the thumbnails `ym-core` already makes (at most 240 px); `measure_encoded_f32` takes display-encoded floats. It returns `None` for a mismatched buffer or under 16 px on the short side. It is 0.006 for a smooth, denoised frame and 0.05 for foliage or visible noise, and costs one pass over the preview.
- *The model.* For every format variant `ln(y) = a + b·x + c·y + d·x·y` with `x = ln(g / G0)` and `y = ln(output megapixels / 10)`; the response is bits per pixel for JPEG and the size over the raw size for TIFF and PNG. `g` is the detail statistic and `G0` is `TYPICAL_GRADIENT` (0.0175, the geometric mean of the calibration renders). There is one `(a, b, c, d)` per JPEG quality knot (50, 60, 75, 80, 85, 89, 90, 92, 95, 100, interpolated in the log domain, never decreasing with quality) and per TIFF (LZW, ZIP) and PNG at 8 and 16 bits. The detail is clamped to 0.003 to 0.12 and the ratio of a TIFF or PNG to 1.25. DNG is the 6 bytes per pixel of its LinearRaw plus the q92 preview. `estimate_bytes` (no preview) assumes the typical detail. The coefficients are the constants `JPEG_FIT` and `RATIO_FIT` in `estimate.rs`.
- *Calibration data.* The nine real files of `testdata/real` (A300, A500, ILCE-5000, ILCE-6300 cRAW, A7 II, A7 III, A7 IV, NEX-5N, NEX-7; 10 to 31 MP after the lens crop), rendered by `ym` with its defaults (classical demosaic, automatic noise reduction and sharpening; 16-bit sRGB PNG), and the synthetic chart of the A7R IV at ISO 100, 400, 800, 3200 and 12800 (1248 × 832). Each was re-encoded into the 16 format variants at the full size and at long edges of 3000, 1600 and 1000 px (the writers' own Lanczos resize): 46 sizes of 14 pictures in all. The tests `measure_renders`, `measure_library_detail` and `report_estimate_errors` of `tests/calibrate_estimate.rs` (run with `--ignored`; they take `YM_ESTIMATE_RENDERS`, `YM_ESTIMATE_SIZES`, `YM_ESTIMATE_CSV` and `YM_ESTIMATE_DETAIL_CSV`) produce two CSVs, and `tests/py/fit_estimate.py` fits the model by ordinary least squares in the log domain and prints the error tables and the Rust constants.
- *Accuracy.* Estimate over actual size, over the nine real renders at the four export sizes (36 files per row): the median, and the 90th percentile of the distance from 1. "Now" is the model with the picture's statistic; "No preview" is `estimate_bytes` (typical detail); "Leave-one-out" refits without a picture and predicts it, by picture, the honest figure for a new frame.

  | Format | Now | No preview | Leave-one-out |
  |---|---|---|---|
  | JPEG q80 | 1.06 / 0.30 | 0.96 / 1.13 | 1.04 / 0.33 |
  | JPEG q92 | 1.00 / 0.29 | 0.99 / 0.80 | 0.95 / 0.35 |
  | JPEG q100 | 0.98 / 0.19 | 0.97 / 0.39 | 0.97 / 0.23 |
  | TIFF 8-bit LZW | 0.99 / 0.22 | 0.98 / 0.44 | 0.98 / 0.27 |
  | TIFF 8-bit ZIP | 1.01 / 0.20 | 0.97 / 0.39 | 1.02 / 0.24 |
  | TIFF 16-bit LZW | 1.02 / 0.10 | 1.00 / 0.20 | 1.02 / 0.14 |
  | TIFF 16-bit ZIP | 1.00 / 0.07 | 1.00 / 0.13 | 1.00 / 0.09 |
  | PNG 8-bit | 1.00 / 0.17 | 0.98 / 0.36 | 0.99 / 0.21 |
  | PNG 16-bit | 1.00 / 0.06 | 1.00 / 0.11 | 1.00 / 0.09 |

  For the nine full-size JPEGs at q92 together: 30.2 MB actual, 30.3 MB estimated. Every row meets the target of a median within 15 % and a 90th percentile within 35 %, except the 90th percentile at q50 and q60 (0.38 and 0.38 on the real renders, 0.37 and 0.35 with the synthetic chart; the leave-one-out figures for q50 to q89 are 0.33 to 0.46), and the leave-one-out figure of q92 is at the limit (0.35). Two frames of the same statistic can carry different fine texture (at q92 and full size the NEX-5N file is 39 % bigger than the estimate, the A7 II file 26 % smaller).
- *What it cannot see.* The statistic is measured on a 240 px preview, where sensor noise is averaged away, so it does not see noise. With the default noise reduction that is fine, because the renders are smooth. With the noise reduction **off** the files are 1.1 to 2.7 times bigger at JPEG q92 (PNG 8-bit 1.15 to 1.85×, TIFF 8-bit LZW 1.2 to 2.0×, 16-bit 1.0 to 1.1×) and the estimate is too low: on five such renders the estimate over the actual size has a median of 0.66 at q92 (0.98 at q75, 0.72 for 8-bit PNG, 0.91 for 16-bit PNG). A noise term would need the residual noise of the render, which `ym-core` knows (the noise model of `AutoEstimates` and the noise-reduction strengths) and the export crate does not.
- *Output sharpening and the estimate.* The stage adds high-frequency content, so the JPEG gets bigger. Measured on three real files (ILCE-7M3, NEX-7, DSLR-A300; the Auto settings, quality 92, Display sRGB, `ym render --output-sharpening 0` against 100), the file size at 100 % over 0 is **+1.3 %, +4.3 %, +1.7 %** at the full size and **+0.5 %, +0.4 %, +1.4 %** at a 2048 px long edge. That is below the 10 % at which the estimate would take a factor, and inside the model's own error, so the estimate has none: its detail statistic is read from the thumbnail, which is sharpened at its own small scale like every picture, and `ym-export` does not know the photo's amount. (The calibration renders carry no output sharpening.)
- *Wiring.* `ym-core`'s `export::estimate` calls `estimate_bytes_with_detail` with `DetailStat::measure_rgb8` of each photo's newest thumbnail (`Renderer::thumbnail_detail`), and `None` (the typical picture) before a thumbnail exists. The app's thumbnails come from a reduced raw group (`ym-core` `pipeline/thumbs.rs`), while the calibration used box-averages of full-size renders; on the nine real files the thumbnails' statistic is 0.85 to 1.10 times the full render's (geometric mean 0.965; ignored test `export_real::the_detail_statistic_of_the_apps_thumbnails_matches_the_full_size_render`), which moves a quality 92 estimate by −7 % to +4 % (size grows as the statistic to the power 0.46). The embedded camera JPEG is not a substitute: it is the camera's own rendering, not ours. The seconds in `export_estimate` come from `ym-core` (`ExportSpeed`, see the export tail section).

## DNG with the look baked in

A Yellowmoon DNG opens in other raw editors looking like our render, and keeps room to adjust exposure, highlights and white balance. Research and app-by-app behaviour: `research/notes/dng-export.md`.

### Pixels

- LinearRaw (PhotometricInterpretation 34892), 3 samples, 16-bit unsigned, scene-linear **AWG4** after C1–C11: demosaiced, denoised, lens-corrected, look and curve applied in LogC4 and decoded back to linear (`ColorChain::apply_dng_pixel`). The highlight range of C7 is in them like every stage before C8 (`docs/02-colour-pipeline.md` C7), so the DNG's pixels are exactly the scene values the app feeds ACES 2, and `BaselineExposure` and the fitted profile give the app's picture, highlights included. Their brightest values reach the ALEXA clip of the range, scene-linear 35.9 (`HIGHLIGHT_CAP_STOPS`).
- Scale: `stored = round(65535 · clamp(x / x_white, 0, 1))` with `x_white = max(p99.99 of the photo's max(R,G,B), 16)`. The floor, 16, is `2^-fit_offset_ev` of the fitted profile (`BaselineExposure::white_floor`); the reason is in As built (M3-12 `tools/dngfit`, DNG gates). `BaselineExposure` compensates: `log2(x_white) + fit_offset_ev + k` with `k = ym_color::display::DISPLAY_PLACEMENT_EV` (+0.85, the brightness the app's ACES 2 path adds before the transform, `docs/02-colour-pipeline.md` C12 step 1a). It equals `k` for every photo whose 99.99th percentile is below 16.
- Negative values (out-of-gamut) are clipped to 0; the clipped-pixel fraction is reported (`DngReport`).
- Shadow precision: with the floor at 16 a stored code is the scene value over 16 times 65535, a step of 2.4e-4 in scene light (mid grey 0.18 is code 737; scene 0.011, four stops below it, is code 46), so 16-bit linear holds about 9.5 stops below grey before codes drop under 1. That suits denoised data: the step in ACES 2's output is about a quarter of an 8-bit code at worst (0.26 at scene 0.019, 0.07 at mid grey, measured on greys with OpenColorIO's ACES 2 Rec.709 view). A shadows push of +3 EV in another editor would show it.
- Full resolution. When the crop has no straighten angle, pixels keep the lens auto-crop only and the user's crop goes into `DefaultCropOrigin/Size`, so other apps can undo it. DNG has no rotated-crop tag, so a non-zero straighten angle is baked into the pixels (rotated, then cropped to the straightened rectangle). 90° steps (`orientation`) go into the TIFF Orientation tag without rotating pixels. For Adobe's apps `DefaultCropOrigin/Size` are the bounds of the final image and give no adjustable crop; DNG 1.4's `DefaultUserCrop` (tag 51125) would. The writer uses `DefaultCropOrigin/Size`; trying `DefaultUserCrop` in Lightroom is part of M3-H6.

### Profile (embedded)

- `ColorMatrix1` = `inv(AWG4 → XYZ)` (XYZ under D65 → camera, i.e. → AWG4) with `CalibrationIlluminant1 = 21` (D65). `ForwardMatrix1` maps white-balanced camera RGB to **XYZ D50**, as the DNG spec requires: `Bradford(D65 → D50) · (AWG4 → XYZ)`, scaled per row so camera white (1,1,1) maps to D50 XYZ exactly. `AsShotNeutral = 1, 1, 1`. Both matrices are `SRATIONAL` with a denominator of 10^7.
- `ProfileToneCurve` (65 points) and a `ProfileLookTable` (HSV, 36 × 16 × 16, `ProfileLookTableEncoding = 1`) **fitted to our default display transform (ACES 2)**, never to ARRI's display LUTs, so Adobe's pipeline shows nearly what we show. The file is `data/dng/profile.json`; `fit_offset_ev` is −4.
  - Fit procedure (`tools/dngfit`): render a set of synthetic test images (neutral ramp, hue/saturation sweeps, our golden photos) through ACES 2 (target) and through Adobe's reference pipeline using `dng_validate` from the DNG SDK (candidate). Fit BaselineExposure and the tone curve on the neutral axis first, then the LookTable for chroma, minimising ΔE00. The LookTable can't change neutrals (spec), so all neutral tone lives in the tone curve.
  - The fitted profile is the same for every photo (it depends only on ACES 2); `BaselineExposure` is per photo (`x_white` changes), and `x_white` is floored at 16 so that it is never negative. The profile stays fitted to the plain transform; the placement `k` is added to `BaselineExposure`, where Adobe's renderer applies `2^BaselineExposure` to the linear values before the profile, so `2^k · x` through the plain transform is the placed transform of `x` and the DNG opens as bright as the app.
- `UniqueCameraModel = "Yellowmoon AWG4"` so Lightroom doesn't substitute Adobe's Sony profiles. `ProfileName = "Yellowmoon"`, `ProfileEmbedPolicy = 3`, `ProfileCopyright` with our notice.
- ARRI looks are baked into pixels only, never into profile tables: a table derived from an ARRI LUT would put a copy of ARRI's data into every file.
- The profile is embedded in the binary (`ym_export::dng::fitted`, see As built (M3-12 `tools/dngfit`, DNG gates)).

### What a DNG carries

**A DNG carries every change Yellowmoon makes except sharpening.** The table says where each stage and setting of a render goes: *pixels* are the LinearRaw data, *tag* a TIFF/DNG tag, *profile* the fitted `ProfileToneCurve` and `ProfileLookTable`.

| Stage or setting | Goes in the DNG as |
|---|---|
| R1 black and white levels, R2 noise and light estimates, R3 vignetting, R4 initial gains | pixels (`decode.lens_corrections` switches R3) |
| R5 highlight reconstruction, R5b lateral CA on the mosaic | pixels (`decode.highlight_recovery`, `decode.chromatic_aberration`) |
| R6 demosaic (adaptive ratio, with its own shadow step) and bilateral denoise, R7 colour and detail controls | pixels, `NoiseReductionApplied` 1.0; **the demosaic's shadow step stays on** (it is part of the demosaic) |
| **R8 capture sharpening** (Richardson–Lucy, or diffraction deconvolution) | **dropped**: both renders (pixels and preview) run with R8 off, whatever Detail › Sharpening says; `BaselineSharpness` 1.0 means "an unsharpened demosaic, like any camera's raw" |
| R9 distortion, auto crop | pixels |
| C1 to C6: white balance (Kelvin, tint, light presets, Pick white), sensor map, exposure (Auto, manual, burst median) | pixels (`AsShotNeutral` 1, 1, 1: nothing of the white balance is left to undo in the other app) |
| C7 highlight range and low-light toe, C8 LogC4 | pixels (the highlight range is a scene-linear gain like the rest of C7) |
| C9 grade (contrast, brightness, zones, colour balance, saturation, the Tone card) | pixels |
| C10 look (CDL and LUT) and its intensity, log-to-log | pixels |
| C10 look with its own display rendering (LogC3 to Rec.709) | **pixels, approximate**: the scene values ACES 2 turns into the look's picture (below) |
| C11 curves | pixels |
| **C12 white lift** (`white_offset_ev`, the automatic amount sized per photo, `docs/02-colour-pipeline.md` C12 step 1b; ACES 2 path only) | **pixels**: a scene-linear gain (`2^(A · smoothstep)` on the luma, ratios kept, exactly 1 up to +1 stop), baked into the AWG4 pixels after the look and curve and before the fitted profile, like the look. Nothing of it is left to undo in the other app, and the DNG's pixels through ACES 2 with the placement are the app's picture to 1e-5 of display light (`tests/white_lift.rs`). With an imported ARRI display LUT, or a look with its own display rendering at 100 %, the picture has no lift (ARRI's highlights) and neither has the DNG; at an intermediate intensity the app blends our lifted C12 with the look's, and the DNG's inverse step takes that blend as it is |
| **C12 red chroma step** (`docs/02-colour-pipeline.md` C12 step 1c; a hue-preserving chroma gain on the reds in front of ACES 2, with the look's strengths; ACES 2 path only) | **pixels**: a scene-linear AWG4 to AWG4 map, baked into the pixels after the white lift and before the fitted profile (`ColorChain::apply_dng_pixel`: C11 decoded, lifted, through the step), like the lift and the look. A DNG has no output gamut of its own, so the step uses the **P3-D65** tables (the viewport's gamut and the Mac display path of the app that opens the DNG) whatever the chain's output; the DNG's pixels through ACES 2 with the placement are then the app's Display P3 picture (to 6.3e-8 of display light for Neutral, `tests/export_dng_carries.rs`), and the app's sRGB picture except where the sRGB cap (the Rec.709 edge) acts on a colour near the gamut edge. A look with strengths 0 (Signature) leaves the pixels as they were. A look with its own display rendering above 0 % and an imported display LUT have no step in the DNG: the pixels are the inverse of the look's picture through plain ACES 2 |
| C12 ACES 2 | profile (tone curve and look table fitted to the plain transform) |
| C12 display placement (`DISPLAY_PLACEMENT_EV`, +0.85) | `BaselineExposure` (`+ k`; the DNG opens as bright as the app) |
| C12 imported ARRI display LUT (Preferences › Display rendering) | **pixels, approximate**, like a look with its own rendering |
| C13 display encode and dither, colour space, size, ICC | not in a DNG (16-bit linear pixels; the preview is an sRGB JPEG, `PreviewColorSpace` 2) |
| Output sharpening | never |
| Orientation (camera and user turns) | tag (`Orientation`) |
| Crop without a straighten angle | tags (`DefaultCropOrigin/Size`; pixels keep the lens auto crop) |
| Crop with a straighten angle, the angle itself | pixels (a DNG has no rotated-crop tag) |
| EXIF, GPS (unless removed), look, profile and pipeline version, the settings (`ym:Settings`) | Exif IFD and XMP (the settings are the photo's, so a DNG records an R8 amount that it does not carry) |

**How a look with its own display rendering (or an imported display LUT) goes in.** A DNG holds scene-linear AWG4 that Adobe's renderer turns into a picture with the profile fitted to ACES 2, so the pixels are chosen as the scene values ACES 2 turns into the picture the chain makes. `ColorChain::apply_dng_pixel`:

1. runs C1 to C11;
2. takes the chain's display-linear sRGB picture (C12 with the look's LUT blended in at the intensity, or the display LUT's output);
3. inverts ACES 2 SDR with Rec.709 limiting primaries (`Aces2Transform::inverse_ap0_linear`, the port of the reference's inverse output transform: it round-trips the display value to 5.5e-10 and agrees with OpenColorIO's inverse to 1.3e-4 relative, `docs/02-colour-pipeline.md` C12);
4. takes the placement out (`2^-k`: Adobe's renderer puts it back through `BaselineExposure`);
5. goes to AWG4.

Two clamps apply. **Display values clamp** to what ACES 2 can output (negative, above white, outside Rec.709: a colour outside the display's gamut is clipped, not gamut-mapped). **Scene values stop at the top of the profile's range** (`DNG_SCENE_LIMIT / 2^k` = 16 / 2^0.85 = 8.9 in AWG4): display white is scene 128 for ACES 2 and has no finite value in practice, and the profile's tone curve ends at 16 anyway. A colour whose scene value would be higher is **dimmed in display light** (all three channels by one factor, found by bisection: the largest factor in 0.5 to 1 whose scene value fits) until it fits. A white of the look therefore comes out of Adobe's renderer at the display value of scene 16 (linear 0.9435, code 0.975 against 1.0, CIE76 2.2), a slightly tinted near-white keeps its tint and loses at most a few percent of light, and the white level of the file stays at its floor instead of crushing the shadows. Over the 819 colours of the top of the display cube that were tested the error stays at CIE76 2.3 at most, mean 0.06 (`chain_cpu.rs`). Through Adobe's renderer, on a synthetic LogC3-to-Rec.709 look of ours that reaches display white and a bright patchwork scene, the highlights (215,000 pixels) against the unsharpened picture measure CIEDE2000 p95 0.70, p99 0.76, worst 1.00 (`export_dng_adobe.rs`). A look with its own rendering at 0 % intensity is our C12 exactly and gets the plain DNG. The preview is the look's picture itself.

**Measured** (`tests/export_dng_carries.rs`, `crates/ym-color/tests/chain_cpu.rs`, `tests/export_dng_adobe.rs`; a look of our own, never an ARRI file):

- In the app's own arithmetic the DNG's pixels through ACES 2 with the placement are the app's picture to 9e-8 of display light for Neutral, a log-to-log look, a LogC3-to-Rec.709 look at 100, 60 and 0 % and a display LUT, wherever the picture is inside what ACES 2 reaches (every pixel of the test scene for the LUT looks). For the looks that are our own ACES 2 rendering (Neutral, a log-to-log look, a LogC3-to-Rec.709 look at 0 %) that is the Display P3 picture and the P3-limited transform, because the red chroma step in the pixels uses the P3-D65 tables; for the inverse cases it is the sRGB picture, as before.
- Through Adobe's renderer (`dng_validate` 1.7.1 with the embedded profile; ΔE00 of its sRGB render against the app's picture without any sharpening, a 16-bit TIFF made with `--sharpen 0,1 --output-sharpening 0`, which isolates what the DNG model costs), for a synthetic LogC3-to-Rec.709 look of ours (33 points: a filmic tone curve darkened to put mid grey at display 0.12, then gamma 2.4, with a little warmth and a little blue density) on eight synthetic chart photos (seven bodies, D55 to LED and mixed light, ISO 400 and 12800): **median 0.043, 95th percentile 0.45, 99th percentile 1.04, worst pixel 7.9**. Neutral (plain ACES 2) on the same chart gives 0.093, 0.30, 0.70 and 10.0, so the approximation costs nothing the profile did not already cost. At 60 % intensity (three files): 0.046, 0.34, 0.82, 7.8. The worst pixels are saturated blues at the display's gamut edge (the app's picture has red at 0), where the profile's own fit is also weakest (median 0.43 and 99th percentile 2.9 for chroma above 80).
- On four real photographs (local evaluation, every third pixel, 8.8 million pixels: ILCE-7M3, DSLR-A300, NEX-7 and an A1 photo of the owner's) the same look gives, against the unsharpened 16-bit TIFF, **median 0.045, 95th percentile 0.20, 99th percentile 0.34, worst pixel 1.05** (per-file medians 0.031 to 0.062). Every DNG passes `dng_validate` with no warning.

**What the DNG model cannot carry** (reported, not closed):

1. White balance, exposure, the look and the tone controls are baked into the pixels; nothing can be undone in the other app, and a white balance change made there acts after the look (the help text says so).
2. A look with its own display rendering, and a display LUT, are approximate by construction: colours outside Rec.709 are clipped, display white comes out at code 0.975, and the scene values are not a scene any more (a push of exposure in the other app moves the picture's tone curve, not its light).
3. A rotated crop (a straighten angle) is baked.
4. Apps other than Adobe's ignore the profile, so they show the stored image darker and without ACES 2 or the placement (Compatibility).
5. Adobe's renderer ends the tone curve at an AWG4 value of 16 / 2^k = 8.9 (5.6 EV over mid grey): brighter values keep their stored value but render at the display value of the plain transform at 16 (the placement note under As built (M3-12 `tools/dngfit`, DNG gates)).
6. The one deliberate loss is sharpening: the converter's own raw sharpening is the only sharpening a DNG gets, the app's capture and output sharpening are not in it, so a DNG opened in Lightroom looks softer than the JPEG until its sharpening is on.

### Tags

IFD0:

| Tag | Value |
|---|---|
| DNGVersion / DNGBackwardVersion | 1.4.0.0 / 1.1.0.0 |
| UniqueCameraModel | `Yellowmoon AWG4` |
| ColorMatrix1, CalibrationIlluminant1, ForwardMatrix1 | XYZ(D65)→AWG4, 21 (D65), AWG4→XYZ(D50) via Bradford |
| AsShotNeutral | 1, 1, 1 |
| BaselineExposure | `log2(x_white) + fit_offset_ev + k` (`x_white` at least 16, `fit_offset_ev` −4, `k` the display placement +0.85): +0.85 for nearly every photo, more for the rest |
| BaselineSharpness | 1.0: the pixels are an unsharpened demosaic, as the raw data of any camera is (no R8, no output sharpening: What a DNG carries), so the converter's own raw sharpening applies as it would to a camera file; the tag is not a record of sharpening done |
| ProfileName, ProfileEmbedPolicy, ProfileCopyright | `Yellowmoon`, 3, `Yellowmoon contributors` |
| ProfileToneCurve, ProfileLookTableDims, ProfileLookTableData, ProfileLookTableEncoding | fitted, 1 |
| DefaultBlackRender | 1 (no further black subtraction) |
| ColorimetricReference | 0 |
| Make, Model, EXIF, XMP | copied / ours as in Metadata |
| PreviewColorSpace | 2 (sRGB; 1 would be "Gray Gamma 2.2") |

Raw sub-IFD: NewSubFileType 0, PhotometricInterpretation 34892, SamplesPerPixel 3, BitsPerSample 16, Compression 1 (uncompressed strips) in V1, BlackLevel 0, WhiteLevel 65535, `NoiseReductionApplied = 1.0`, DefaultCropOrigin/Size, ActiveArea.

Preview sub-IFD: a full-size JPEG (quality 92, 4:4:4, sRGB) of the exact render, so Quick Look and preview-based apps show our image.

There is no `NoiseProfile` (the data is denoised; `docs/06-noise.md`).

### Writer

Our own, on the `tiff` crate (MIT): `write_tag(Tag::Unknown(n), …)` for DNG tags, `extra_directory()` and `finish_with_offsets()` for sub-IFDs, strips with Compression = 1 (`dng.rs`). The `dng` crate (AGPL) and rawler's writer (LGPL) are excluded.

- **Layout.** IFD0 is an 8-bit RGB thumbnail (long edge at most 256 px, `NewSubFileType` 1) carrying the DNG profile tags, the Exif IFD, the GPS IFD and the XMP. `SubIFDs` points to the LinearRaw IFD and to the preview IFD. The directories are written before IFD0, so IFD0 is last in the file.
- **Orientation.** The preview and the thumbnail are in the raw pixels' orientation (not rotated) and cropped as the user sees the render, because the `Orientation` tag of IFD0 applies to all three. `compose_orientation(camera exif orientation, user quarter turns)` gives the tag value.
- **`x_white`** is the upper edge of the histogram bin (2048 bins per octave) that holds the 99.99th percentile of `max(R, G, B)`, at least `BaselineExposure::white_floor` (1.0 for `Fixed` or a zero offset, 16 for the fitted profile). `DngReport` has `x_white`, `baseline_exposure_ev`, the fraction of pixels with a negative channel (clipped to 0) and the fraction above `x_white`.
- **`BaselineExposure`** = `log2(x_white)` + a fitted offset + the display placement (`BaselineExposure::Auto { fit_offset_ev, placement_ev }`: −4 and +0.85 for the embedded profile via `FittedProfile::baseline_exposure()`; 0 and 0 for `BaselineExposure::default()`, the identity profile of the structure tests), or `Fixed(ev)`. `white_level(raw, baseline_exposure)` gives the white level before writing. An `Auto` offset that is not a number or is outside −12 to 4 EV, or a placement that is not a number, is refused.
- **`RawDataUniqueID`** is the first 16 bytes of the SHA-256 of the stored pixels.
- **Profile input.** `ProfileToneCurve` and the look table are inputs (`DngProfile`); `DngProfile::identity()` writes the identity curve and a 2 × 2 × 2 identity table. An identity tone curve shows a flat, dark image in Adobe's apps, so it is for structure tests only: the product embeds the fitted profile.
- **Look table rules** (`LookTable::validate`). A look table with one value division (a "2.5D" table) must use linear encoding (`ProfileLookTableEncoding` 0): with encoding 1 (sRGB) Adobe's reference renderer (DNG SDK 1.7.1) darkens even an identity table (mid grey 0.18 comes out as 0.027), so a table with sRGB encoding needs at least two value divisions, as the 36 × 16 × 16 table does. The validator also enforces the DNG specification's "all zero input saturation entries are required to have a value scale factor of 1.0" (the first saturation division of every hue and value division) and non-negative scales: neutral tone belongs in `ProfileToneCurve`.
- **Checks** (`tools/dng/check_dng.py`, run by `tests/dng_validate.rs`; the tests that run `dng_validate` need the binary and are skipped without it): exiftool lists every tag with the expected value and `exiftool -validate` reports no warnings; tifffile parses IFD0 and the two SubIFDs; LibRaw (rawpy) opens the file and returns exactly the stored pixels, which are compared with the formula computed independently from the original floats; the matrices are recomputed from the AWG4 primaries and agree with colour-science.
- **`dng_validate`.** `tools/dng/build_dng_validate.sh` builds Adobe's `dng_validate` from the DNG SDK 1.7.1 (the AOSP copy, `GrapheneOS/platform_external_dng_sdk` at `0f23d021`, pinned by commit; about 10 minutes on 4 cores) with the XMP toolkit and JPEG XL stubbed out. With it, five DNG variants (plain, 5 × 3, 131 × 77, rotated and cropped with a fitted-style tone curve and look table, mirrored) validate with no warning or error. Adobe's own renderer, given flat AWG4 patches and an identity profile, returns the sRGB and Display P3 colours `ym-color` computes to within 3.0e-4 and 9e-5 in encoded values (16 patches, saturated ones included), so `ColorMatrix1`, `ForwardMatrix1`, `AsShotNeutral`, D50 and the Bradford adaptation agree with Adobe's code, not only with ours. A scene with `x_white` 2.6 renders to the colours `ym-color` computes from the original scene-linear values with `BaselineExposure = log2(x_white)` (worst difference 2.9e-4 in encoded values), and a fitted offset of −1 EV halves the light as expected.

### What other apps show (Compatibility)

| App | Expected |
|---|---|
| Lightroom, Camera Raw | close to Yellowmoon's render (minus fit error) |
| RawTherapee, darktable | the look's colours under their own tone curve; the stored image is about 4 EV darker until their exposure is raised (see below) |
| Capture One | the look's colours; exposure is off, as with other linear DNGs: the stored image is about 4 EV darker (see below) |
| DxO PhotoLab | doesn't open third-party DNGs |
| Apple Photos, Preview, Quick Look | embedded preview or Apple's rendering (to be verified, M3-H6) |

White-balance changes made later in another app act after the baked look, so they are approximate; the help text says so. It also says, in one sentence (`ym_export::dng::OTHER_APPS_NOTE`, used word for word by `ym render --help` (the doc comment of `FormatArg::Dng`, which a test compares with the constant) and by the export panel's DNG paragraph, `src/screens/export/ExportOptionsForm.tsx`, guarded by a drift test): **"Apps other than Adobe's show the stored linear image darker until their own exposure handling is applied."** The reason is the white floor: the stored image is the scene divided by at least 16 (4 EV), and only Adobe's apps undo that through `BaselineExposure` and the profile.

### Legal

Adobe's DNG patent licence is royalty-free and requires this notice "in a prominent manner within its source code and documentation": **"This product includes DNG technology under license by Adobe."** The notice is in `NOTICE`, in the README, in the About window and the DNG export help text. The licence ends if we bring a patent action against Adobe.

The JPEG encoder, `jpeg-encoder`, is `(MIT OR Apache-2.0) AND IJG` (its forward DCT is ported from the Independent JPEG Group's software). `deny.toml` allows `IJG`, and the IJG licence asks that documentation accompanying a binary says "This software is based in part on the work of the Independent JPEG Group." (`NOTICE`, the About window, `docs/14-licensing.md`).

### Gates

- Every exported DNG passes `dng_validate` with no warning and no error in CI.
- `dng_validate` render vs our JPEG of the same photo: ΔE00 median ≤ 1.5 per file over the golden set.
- **Result (pipeline 0.5.1, 61 synthetic files):** 61 of 61 pass. The ISO 12800 chart is the worst file, at 1.237 against the JPEG and 0.696 against the 16-bit PNG; the median of the per-file medians is 0.817.
- [HUMAN M3-H6] Opens in Lightroom Classic, RawTherapee, darktable, Capture One and Apple Photos; screenshots attached to the M3 report.

How the gates run: As built (M3-12 `tools/dngfit`, DNG gates).

## Tests

- Golden exports (`tests/data/golden_hashes.json`, 39 files: every format and colour space of a 64 × 48 synthetic scene, long-edge variants, encoded input, DNGs; update with `YM_UPDATE_GOLDEN=1`). A change in hash needs an explanation in the PR (and usually a `pipeline_version` bump).
- App vs CLI: the same job exported through both is byte-identical.
- Thread-count independence of every writer.
- ICC: every output decodes in a colour-managed reader (LittleCMS through Pillow) to the expected values at test patches within 1 code value.
- Resize: Lanczos in linear light against Pillow; a zone plate for ringing limits.
- DNG: against LibRaw, exiftool and tifffile; `dng_validate` (see Gates).
- Names, destinations, atomic write, cancel and the original-file guard (`tests/write_export.rs`, `src/naming.rs`); write errors (`tests/disk_full.rs`).
- What `exiftool -validate` says about the other formats: JPEG and PNG, no warnings. TIFF: three minor remarks (`ComponentsConfiguration`, `ExifImageWidth` and `ExifImageHeight` are "compressed data" tags in the Exif specification and are kept because readers use them), a note that `GPSProcessingMethod` is not written, and, for Deflate (compression 8), a note that it is not among the values it expects (the `tiff` crate and tifffile read these files). The DNG validates cleanly.
- Tests that need Python modules or exiftool print `SKIPPED` when they are missing and fail instead when `YM_REQUIRE_EXTERNAL_TOOLS` is set (CI sets it).

## As built (export tail on the device)

`docs/17-gpu.md` section 10. With `GpuStages::EXPORT_TAIL` (part of `On` and `Require`, and of `SIGNED_OFF`, so `Auto` runs it on a hardware adapter) a JPEG, TIFF or PNG export never makes the host frame of display-linear `f32` (600 MB at 50 MP). The CPU path described above is the reference and the fallback; the golden hashes pin it.

**`BandSource` (`ym-export::pixels`).** The writers pull encoded rows (stage C13 done) from a `BandSource`: `width`, `height`, `rows_u8(y0, rows, &mut Vec<u8>)` for JPEG and 8-bit TIFF and PNG, `rows_u16(..)` for 16-bit. A source is read in order (`y0` is the row after the last one it gave), so it may stream. `ImageBands { ImageRef, OutputSpace }` is the CPU source: it calls `encode_rows_u8` and `encode_rows_u16`. `write_jpeg_bands`, `write_png_bands` and `write_tiff_bands` are the writers; `write_jpeg`, `write_png` and `write_tiff` wrap them over `ImageBands`. `tests/bands.rs` checks that for every format, depth and colour space the file written through a `BandSource` equals the file written from the image; that a source that fails half way leaves no file; that progress never goes back; and that a source that hands out a band of the wrong size ends the export with an error and no file instead of a corrupt file.

The DNG keeps `ImageRef`: its two renders (the raw pixels and the preview) are host frames of `f32`, with the colour stage of each tile on the device when the tail is on, read back tile by tile (`Stop::DngLinearAwg4` and the preview). Agreement with the CPU is 2.3e-6 and 3.5e-6 relative at worst (`tests/gpu_export.rs`); a DNG has no output sharpening and no R8.

**The tail (`ym-gpu::tail`).**

1. The raw chain renders tile by tile (on the device too when its stages are on); each tile's camera RGB goes to the device, through the sampler and the colour chain C1 to C12 (persistent tables), into **slabs** of the rendered picture (four planes, at most 512 rows and 128 MiB, the design binding of `docs/17-gpu.md` section 4, so the cut does not depend on the device).
2. A horizontal Lanczos-3 pass runs on each slab when it is complete.
3. A vertical pass makes chunks of output rows (about a megapixel) from the rows they need.
4. Output sharpening takes a chunk with the stage's halo (17 to 21 rows).
5. C13 (with the blue-noise dither at absolute coordinates, so any cut gives the same codes) makes the bytes of a **band** (about a megapixel, 8 to 1024 rows).

Each stage reads rows from a store of the pieces the stage before it made and releases a piece as soon as no later row needs it, so at 61 MP the device holds a few slabs and a chunk, not the 700 MB picture (`tests/gpu_tail.rs`: the bytes do not depend on the tile size, a finished tail holds nothing, a cancelled one gives everything back). The resize taps are planned on the CPU in `f64` (`ym_gpu::kernels::resize::Taps`; `ym_export::resize` re-exports them, so the CPU resize and the kernel use the same taps) and the kernel sums in difference form, `out = r + sum w (v - r)` with `r` the middle tap: a flat field is exact in any arithmetic.

**Threads and memory.** `Renderer::start_export_tail` queues the render on the export worker; the job sends the bands through a bounded channel (`BAND_DEPTH` = 3) and the queue's thread (or `ym`'s) pulls them through `GpuBands` while the writer compresses and writes, so the render of the rows that follow overlaps the encoding of the rows before. A writer that stops drops the bands and the job ends at its next tile. The host holds a few bands of codes (a few megabytes each) and the working set of a tile; the ledger's peak of an export is the CPU path's less the frame (`tests/gpu_export.rs`: 33,560,064 B against 27,080,064 B for a 900 × 600 photo, the difference exactly 6,480,000 B = 900 × 600 × 12).

**Look, cancel, progress.** The check that the look did not change while the picture rendered runs when the last rows are asked for, before the writer can finish the file (`GpuBands::check_before_last_rows`). A cancel between bands removes the temporary file and hands every device buffer back (the pool returns to its baseline). The progress an export reports is the writer's, by rows written, and never goes back, also when the photo is made again on the CPU. Queue semantics are the same as on the CPU path: settings are snapshotted at queue time, a job waits for the shared estimates, files are written under a temporary name and renamed, cancel deletes the partial file.

**When the device fails.** A device error in the tail or in the raw chain of a tile ends the attempt with a `RenderError::Gpu`. The writer has seen some rows, so the export removes its temporary file and makes the **whole photo again on the CPU reference path** (`Renderer::spawn_full_cpu`: the same render with no device at all), with the note "The graphics card could not render this photo, so the CPU did: <reason>." in `ExportReport.notes` and `device = cpu`. The file is then the CPU export's, bit for bit.

- Tests (`tests/gpu_export.rs`, with `--features fault-injection`): a shader that does not compile, an allocation out of budget and a lost device, in the tail alone and in every stage, for PNG, TIFF and JPEG; and a device that fails **part way**, after the writer has taken and compressed rows (a raw-chain tile that fails at the 20th of 32; a device lost once the writer has taken a fifth of the rows of a picture that comes out in eight bands). Each result equals the CPU export bit for bit, with one file in the folder, the pool back at its baseline and progress never going back.
- The fallback is for that export only: the next photo tries the device again, **except after a shader of the tail that does not compile**. That is a property of the machine, so the export tail is switched off for the renderer (`Renderer::export_tail_disabled`, also for a DNG's device colour stage; the application and `ym` have one renderer, so it is off for the process), and the next exports go straight to the CPU path without trying the same shader first (the second export is the CPU's bytes, carries no note about the card, and the device's submission counter does not move). The raw chain's and the view stages' shaders have their own switches.
- When the photo's raw words cannot be put on the device again after tiles ran on it (an upload that finds no room, `YM_GPU_FAULT=upload@N`), the export starts again on the CPU as for any device error.
- A strict policy (`ym render --require-gpu`, `GpuPolicy::Require`) fails the photo instead, error kind `gpu`, and leaves no file; it also fails the photo instead of turning a switched-off tail into a quiet CPU export.
- A DNG is two renders (the raw plane and the preview), and a device error that fires once is spent by one of them. When the raw render fell back, the preview is made on the CPU reference path; when only the preview fell back, the raw plane is made again on it. The file is wholly the CPU's and the report says `device = cpu` (`FullImage::fell_back_to_cpu`, the typed signal; `gpu_faults.rs::a_dng_after_a_one_shot_device_error_is_the_cpu_export_whole`).
- A file that appears under the name between the check at the start and the writer's own ("If the file exists: skip") is skipped as on the CPU path, not reported as cancelled: the writer pulls no row, the render is stopped, and the other file is left alone (`gpu_export.rs::a_file_that_appears_before_the_first_row_is_skipped_not_cancelled`).

**Speed priors.** `ExportSpeed` keeps one set of numbers per path, the CPU reference and the tail on the device, each replaced by the photos that took that path (the first sample outright, later ones as a running mean that weighs the newest half). A photo that fell back to the CPU teaches the CPU set only. The set is the path's, not the raw chain's: an export whose raw chain ran on the device but whose colour pass and writer ran on the CPU teaches the CPU set, which is the one its estimate reads (`RenderReport::device`; `gpu_export.rs::an_export_teaches_the_numbers_of_the_path_the_estimate_reads`).

- The path is named by the reader: `estimate()` takes it from `Renderer::export_tail_enabled` into a local, and `ExportSpeed::render_s_per_mp(device)`, `encode_s_per_mp(device, format)` and `combined_s_per_mp(device, format)` say which set they read.
- On the device the writer pulls the picture while it renders, so the time of a photo covers both. A sample is recorded for the format it was taken with (`record_overlapped(format, megapixels, seconds)`): a device estimate for a format that has one is that combined time per megapixel and adds no encode on top, and the formats that have none keep the CPU's numbers (a slow PNG export, limited by deflate, must not set the estimate of a fast JPEG one, which is limited by the DCT).
- No speed of the graphics card has been measured on a Mac yet (M0-H4): the device's set starts as the CPU's.

**Parity (Mesa lavapipe, `tests/gpu_export.rs`, 900 × 600 patchwork photo with noise, 256 px tiles; the bounds in the tests are twice the values below).** The tail against the same export with the raw chain on the device and the CPU doing the rest (sampler, colour, resize, sharpening, C13), full size and long edges, sharpened or not, and a straightened crop with paper in its corners:

| Output | Codes | Result |
|---|---|---|
| PNG 8-bit (dithered) | 1.62 M | at most 1 code off, 0.9999 exact |
| PNG 16-bit, not sharpened | 1.62 M | at most 1 code off |
| PNG 16-bit, sharpened | 1.62 M | 354 over 1 code (2.2e-4), 81 over 8 (5.0e-5), at most 89 codes off |
| JPEG q92, decoded | 1.62 M | 280 over 1 code (1.7e-4), at most 5 codes off |

The sharpened 16-bit outliers are the limiter's rank flips (`docs/algorithms/04-output-sharpening.md` section 10): the f32 and the f64 colour passes differ by 4e-6 and the limiter decides on a rank of luma values. The JPEG's few codes are the quantiser amplifying a one-code input difference.

Against the CPU export (the golden metric of `docs/17-gpu.md` section 12: ΔE2000 on a 4× downsample, gates 0.02, 0.05 and 0.5 for the median, the 99th percentile and the maximum; the app's whole device path, the raw chain included): 8-bit PNG median 0, 99th percentile 0, maximum 0.060 to 0.066 for all five variants; 16-bit PNG and 16-bit TIFF (the same codes) median 0.0001, 99th percentile 0.0005, maximum 0.001 to 0.027 for full size, a 500 px long edge and the straightened crop. The decoded JPEG has median 0 and 99th percentile 0 but a maximum of 1.51 at full size, 0.58 at a 500 px long edge and 0.98 for the straightened crop: the JPEG quantiser turns a one-code difference of its input into up to 5 codes in a block, which a ΔE of a dark block then magnifies. The decoded JPEG of every export (CPU and device) is, sample for sample, the JPEG the writer makes of the 8-bit codes of that export's PNG (`gpu_export.rs::the_jpeg_of_an_export_is_the_jpeg_of_its_8_bit_codes`).

**A JPEG export is therefore judged on the 8-bit codes in front of the DCT**, where the device meets all three gates (maximum 0.064), because the golden metric is defined on rendered pictures, not on a lossy file. The decoded JPEG keeps a regression bound of 3.0 (twice the largest measured), so a change in what the device hands the encoder still fails the test. The kernels' own parity is in `tests/gpu_resize.rs` (worst difference to the CPU resize 3.58e-7 of white, a flat field exact, a step overshoots by less than 10 %, rows independent) and `tests/gpu_tail.rs`. The app's queue and `ym` write the same bytes on the device path (`tests/gpu_export.rs`, `ym-cli` `parity_gpu`).

**Metal fast-math.** The only kernel of the tail that is new is `resize.wgsl` (the rest are the view kernels of `docs/05-raw-processing.md` and buffer copies). Its inner loop was compiled as C with clang `-O3 -ffast-math -ffp-contract=fast -march=native` (the same model as the other GPU kernels): the loop is vectorised and fused, the difference `v - r` stays in front of the weighted sum, and over two million random tap sets a flat field is exact in every case and the worst difference to an `f64` sum is 5.0e-7 of the range, the same as without fast-math. There is no division, no square root and no decision on a rounded value, so no barrier is needed.

**Where the time goes (encode split, `tests/encode_split.rs`, ignored).** `cargo test -p ym-export --release --test encode_split -- --ignored --nocapture` times each stage on a 6000 × 4000 (24 MP) display-linear scene of the shared test generator (smooth gradients, hard-edged patches and noise of 0.01 peak to peak, so noisier than a denoised photo), best of three on four shared x86-64 vCPUs (other builds were running: the figures are pessimistic by up to a half for the stages that use all cores). The first four rows are what the device takes over; the others are what stays on the host. "Rows from memory": the writer is fed by a `BandSource` that copies finished codes, so the figure is the writer alone.

| Stage | Time | MP/s | Output |
|---|---|---|---|
| Lanczos-3 resize to half the long edge | 0.21 s | 115 | 3000 × 2000 |
| Output sharpening, 100 % | 3.2 s | 7.5 | |
| C13, 8 bit, dithered, `f64` | 0.62 s | 39 | |
| C13, 16 bit, `f64` | 0.64 s | 38 | |
| JPEG q92, DCT and Huffman, rows from memory | 0.96 s | 25 | 1.3 MB |
| PNG 8-bit, deflate | 4.7 s | 5.1 | 25.8 MB |
| PNG 16-bit, deflate | 5.5 s | 4.3 | 110.8 MB |
| TIFF 8-bit, uncompressed | 0.12 s | 203 | 72.0 MB |
| TIFF 8-bit, LZW | 1.2 s | 20 | 27.8 MB |
| TIFF 8-bit, Zip | 1.9 s | 12 | 26.5 MB |
| TIFF 16-bit, uncompressed | 0.27 s | 91 | 144.0 MB |
| TIFF 16-bit, LZW | 1.7 s | 14 | 134.4 MB |
| TIFF 16-bit, Zip | 1.7 s | 14 | 110.6 MB |

A full-size sharpened 24 MP export on the CPU spends 3.2 s in output sharpening and 0.6 s in C13 before any byte is compressed; those and the resize are what the device takes, and the tail leaves the writer: about 1 s for a JPEG, 1.2 to 1.7 s for an LZW TIFF, **4.7 to 5.5 s for a PNG** (the `png` crate's deflate; at 61 MP that is 12 to 14 s). A PNG export is therefore bound by its encoder, not by the device, and the 6 s target for a 61 MP export holds for JPEG and TIFF only. The numbers on a Mac, and the 6 s, are M0-H4.

**50 MP (Mesa lavapipe; the Metal CI runs repeat it, `mac-gpu.yml`).** Two `#[ignore]`d tests, run with `cargo test -p ym-gpu --test gpu_tail fifty -- --ignored --nocapture` and `cargo test -p ym-core --test gpu_export fifty -- --ignored --nocapture`.

- *The tail alone* (`gpu_tail.rs::fifty_megapixels_leave_a_few_slabs_on_the_device_and_the_same_bytes_twice`: 8688 × 5792 in 1024 px tiles, the device's pool). Full size, sharpened, 16 bit: the device holds at most 80 MiB between two tile rows and 458 MiB at its peak, 59 % of the 767 MiB the four planes of the rendered picture would be (the bound in the test is three slabs, one sharpening window and 64 MiB, 629 MiB). A 4000 px long edge, sharpened, 8 bit: 72 MiB and 264 MiB. Both give the same codes twice.
- *The export* (`gpu_export.rs::fifty_megapixels_leave_no_host_frame_and_a_dng_keeps_its_own`, a 50 MP synthetic ARW, the raw chain on the device). The host ledger's peak of a 16-bit PNG is 215 MiB with the tail and 791 MiB with the host frame (the display-linear frame is 575 MiB; the difference is exactly the frame), and the process's resident set peaks at 1290 MiB against 2606 MiB. A JPEG at a 4000 px long edge peaks at 215 MiB (ledger) and 1537 MiB (resident set). The DNG, which keeps its two host frames, peaks at 791 MiB (ledger; each render reserves its own frame) and 3272 MiB (resident set). On lavapipe the resident set includes the software device's own buffers, which are host memory; wall times on lavapipe say nothing about a Mac.

**Not on the device yet.**

- The encoders stay on the CPU: JPEG DCT and Huffman, PNG deflate, TIFF LZW and Deflate. PNG is the slowest (see above) and is bound by deflate; a faster compressor, or a lower deflate level (a size trade), would be the way to speed it up.
- JPEG does not overlap its DCT with the render: the `jpeg-encoder` crate takes a whole image, so JPEG fills its 8-bit buffer before it encodes. A streaming JPEG encoder would overlap them.
- The DNG has no tail: its two renders keep host frames of `f32` (the raw plane and the preview), with only the colour stage of each tile on the device.
- No speed of the device path has been measured on a Mac (M0-H4).

## As built (M3-12 `tools/dngfit`, DNG gates)

`dngfit fit` writes `data/dng/profile.json`; `dngfit eval` and `dngfit gate` measure it; `tools/dng/run_gates.sh` runs both (it builds `ym` and `dngfit` in two `cargo` calls, runs `dngfit eval` and `dngfit gate --require-all`); the CI job is the `dng` job of `.github/workflows/ci.yml`. Method, model and tests: `tools/dngfit/README.md`.

- **Adobe's renderer, as found.** `dng_validate` (DNG SDK 1.7.1) renders a LinearRaw DNG per pixel: `x * 2^BaselineExposure` (clipped to 0 to 1), `ForwardMatrix1` to XYZ D50 and to linear ProPhoto, the look table in HSV of that RGB, the tone curve, the matrix to sRGB, the sRGB encode. `tools/dngfit/src/model.rs` is a model of that, checked against the renderer on 3000 random colours with a random table and curve (median error 1.4e-4 in encoded values, worst 2e-3). Two facts the specification does not give: the look table's **value scale multiplies the sRGB-encoded value** (with `ProfileLookTableEncoding` 1) and the result is decoded again (a linear scale would be off by up to 0.12 in code values); and the tone curve is applied by the SDK's *RGB tone*: the largest and smallest channels go through the curve and the middle one keeps its position between them, with the curve sampled into a table of 4096 steps and interpolated linearly. The curve's own spline through the points matters only through that table.
- **A negative `BaselineExposure` is not a gain.** For an exposure below zero the renderer applies a highlight shoulder instead (linear with slope `2^b` up to 0.25 of the range, then a quadratic that ends at 1): with `b = −3` and an identity profile scene 0.40 renders at 0.086 of display light instead of 0.050 and 0.84 at 0.654 instead of 0.105 (`tests/model.rs` pins it). A profile cannot be fitted for that, because the shoulder depends on `b`, which depends on the photo's `x_white`. And the tone curve's input range is `[0, 1]` after the gain, while ACES 2 is at 94 % of display light only at a scene value of 16. So the profile covers scene values up to 16 (`fit_offset_ev` −4) and **`x_white` must be at least 16: `x_white = max(p99.99, 16)`, `BaselineExposure = log2(x_white / 16)`** (plus the display placement, below). That is zero for every photo whose 99.99th percentile is below 16 (nearly all) and positive for the rest. The writer floors `x_white` at `2^-fit_offset_ev` (`BaselineExposure::white_floor`). The cost is shadow precision (Pixels).
- **Profile.** 65-point `ProfileToneCurve` (geometric through the shadows), a 36 × 16 × 16 `ProfileLookTable` with `ProfileLookTableEncoding` 1 (a 36 × 32 × 16 table is barely better, median 0.070 against 0.075, and twice the size), `fit_offset_ev` −4. The fit takes about 2 s. Against ACES 2, through `dng_validate`, in CIEDE2000, on sets the fit does not use (`dngfit eval`):

  | Set | n | median | p95 | p99 | max |
  |---|---|---|---|---|---|
  | neutral ramp, 0.002 to 16 | 400 | 0.004 | 0.019 | 0.028 | 0.031 |
  | hue and chroma sweep (36 hues, 10 chromas, L* 25 to 85) | 1314 | 0.077 | 0.50 | 1.02 | 2.06 |
  | random photograph-like colours | 20000 | 0.075 | 0.46 | 0.73 | 8.65 |
  | chart patches of the synthetic golden scenes, −4 to +3 EV | 56240 | 0.103 | 0.60 | 0.92 | 4.68 |
  | the same random colours with the tone curve only (no look table) | 20000 | 0.69 | 3.7 | 7.4 | 12.2 |

  The worst colours are very saturated blues at the edge of the AWG4 gamut (ProPhoto hue 4.1 to 4.2 of 6, saturation 0.9) where ACES 2's gamut mapping bends faster than an HSV grid follows, and the darkest part of the range, where the first value division of the table spans three stops (median 0.27 at −3 to −4 stops from grey). Colours that need a negative ProPhoto channel are clipped by the renderer before the table; the scene data has few.
- **`ym` writes the fitted profile.**
  1. `ym-export`: `ym_export::dng::fitted` holds the schema of `data/dng/profile.json` (`tools/dngfit/src/profile_file.rs` re-exports it and adds `load` and `save`, so the tool and the product have one reader and one writer of the format) and embeds the committed file with `include_str!`: no file is read at run time, and `bundle.resources` needs no entry. `FittedProfile::embedded()` is the `FittedProfile { profile: DngProfile, fit_offset_ev }`, parsed and validated once; the `expect` on it is an invariant checked by the test `the_embedded_profile_parses_and_validates` (which also requires the file to be in the form `dngfit fit` writes). `BaselineExposure::white_floor()` is `max(2^-fit_offset_ev, 1.0)` for `Auto` (16 for the embedded profile) and 1.0 for `Fixed`, so Adobe's renderer never meets a negative `BaselineExposure`.
  2. `ym-core`: `ExportContext::new` holds `FittedProfile::embedded()` (`with_dng_profile` takes a `FittedProfile`) and `export_photo` writes `BaselineExposure::Auto { fit_offset_ev, … }`. The app and the CLI come through the same function, so their DNGs are the same bytes (`ym-cli`'s parity test covers DNG).
  3. Tests: `ym-export/tests/dng_validate.rs` writes a DNG with the embedded profile and checks `BaselineExposure`, `dng_validate` with no warning, the independent readers (`tools/dng/check_dng.py --white-floor 16`), and that Adobe's renderer shows flat patches within CIEDE2000 1.5 of the placed ACES 2 target for a photo below the floor and one above it; `ym-core/tests/export_dng_profile.rs` checks the tags of an exported DNG.
- **Gate method (`dngfit gate`).** For every file `ym` writes the DNG and our JPEG with the same settings. The DNG is checked as `ym` wrote it (`--as-written`, no re-profiling). `dng_validate -v -32 -cs1 -tif` must report no warning and no error on the file it renders, and its render, against the JPEG, must have a median CIEDE2000 of at most 1.5 **per file** (stricter than the median over the set). `dng_validate` turns its render upright by the file's `Orientation` tag, so the as-written path does not turn it a second time (`turn_after_render`, tested). `--png` adds the comparison with the 16-bit PNG of the same render, which separates the profile's cost from the JPEG's own coding and dither. `--require-all` (set by `tools/dng/run_gates.sh` unless `--filter` or `--files` is given) fails when a file of the selection is not on disk, so a partly generated set cannot pass. The default set is the 61 synthetic golden files; `--files` takes real ARW files instead (local evaluation, not part of the gate). The numbers are in Gates.
- **Display placement in `BaselineExposure`.** The app's ACES 2 path brightens the scene by `2^k` (`k` = `DISPLAY_PLACEMENT_EV`, +0.85) before the transform (`docs/02-colour-pipeline.md` C12 step 1a). The fitted profile is of the plain transform, so the writer adds `k` to `BaselineExposure` (`log2(x_white) + fit_offset_ev + k`, 0.85 for a photo below the floor) instead of refitting. Adobe's renderer scales by `2^BaselineExposure` (clipped to 0..1) before the look table and the tone curve, which is the same operation, so the DNG opens as bright as the app. The profile, `dngfit` and the white floor are unchanged: the floor stays at scene 16, because the fit offset alone sets it and the placement only adds to a value that is never negative (lowering the floor to `2^-(fit_offset_ev + k)` = 8.9 would give 0.85 EV finer shadow codes and take away the stored headroom that a user who lowers the exposure in the other editor needs).
  **What this means for bright scene values.** The tone curve's input range ends where `x · 2^k / 16 = 1`, so Adobe's renderer clips at scene 16 / 2^0.85 = 8.9 (5.6 EV over mid grey), at the display value of the plain transform at 16 (Y 0.9435, code 0.975). The app keeps rising slowly above that (ACES 2 of 2^k · x), so Adobe's picture is flatter in the last stops: CIEDE2000 against the app 0.16 at scene 10, 0.38 at 12, 0.65 at 16, 0.92 at 24 and 1.14 at 40 (neutral greys, OpenColorIO). At the default exposure (+1.11 EV) the C7 shoulder tops a frame out near 1.5 times its clip level, about scene 3.3, so a photo reaches Adobe's clip point only after about +1.4 EV more exposure. Measured with `dng_validate` (the product DNG against the placed ACES 2 target, `tests/dng_validate.rs`): a photo below the floor (`BaselineExposure` 0.850) worst CIEDE2000 0.012 for greys and 0.55 for colours, a photo above it (x_white 40, `BaselineExposure` 2.172) 0.058 and 1.14.
- **Not verified.** Adobe's apps (Lightroom, Camera Raw) are not `dng_validate`: they apply the same profile tags, but nothing here proves they match its renderer (the SDK is Adobe's reference, and Camera Raw adds its own defaults). [HUMAN M3-H6] stays open. RawTherapee and darktable ignore `ProfileToneCurve` and the look table (`research/notes/dng-export.md`) and show the stored linear image, about 4 EV darker than the render before their own exposure and tone handling, which the DNG export help text says (Compatibility).
