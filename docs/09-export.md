# 09 · Export

Crate: `ym-export` (writers), `ym-core` (export queue). UI: `design/mockups/export.html`. CLI: `yellowmoon render` (`docs/10-cli.md`). App and CLI share one code path, so the same inputs and settings give byte-identical files (apart from the XMP timestamp, which both write as the capture time, not "now").

## Options

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

The export panel shows an estimate ("≈ 120 MB") from `export_estimate`, and the equivalent CLI command with a Copy button. The command carries the export options and the look. Per-photo adjustments beyond the look don't fit on one command line; when any included photo has them, the panel adds "N photos have adjustments this command doesn't include." (the CLI can take them through `--settings`, `docs/10`). Defaults are remembered as preferences; per-photo settings are not.

## Queue

- Jobs run one photo at a time on the GPU (full-resolution neural path, tiled), with decode of the next file overlapping.
- The panel shows each file as Waiting / n% / Done or an error. "Cancel" stops after the current tile and deletes the partial file.
- Exports keep running while the user browses or edits other photos; edits to a photo already queued don't affect its job (settings are snapshotted at queue time).
- Files are written to a temporary name in the destination folder and renamed when complete. Originals are never touched.
- Auto-update waits: "Restart to update" stays disabled until the queue is empty.
- Finished state: "N photos saved to <folder>." with "Show in Finder" and "Done" (back to the tool panel).

## Pixel pipeline per format

- **JPEG, TIFF, PNG:** full chain through C13 (`docs/02`) for the chosen colour space, crop, straighten and orientation applied to the pixels, then resize (if Long edge) with a Lanczos-3 filter in linear light before the display encode, then 8-bit dither (8-bit outputs only). JPEG uses 4:4:4 chroma at quality ≥ 90 and 4:2:0 below; baseline (not progressive); optimised Huffman tables.
- **DNG:** stops after C11 (look and curve applied), decodes LogC4 back to linear AWG4, writes LinearRaw (below). Not available when the look is a LogC3-to-Rec.709 look (it includes ARRI's display rendering, which can't be represented in scene-linear data): the DNG option is disabled with "This look includes its own display rendering, so it can't go into a DNG. Pick another look or export TIFF."

## Colour management

Embed an ICC profile in every JPEG, TIFF and PNG: sRGB IEC61966-2.1, Display P3, or Rec.2020 (BT.1886 transfer to match our encoding), generated with moxcms or shipped as small v4 profiles with permissive licences. PNG also gets a `cICP` chunk. TIFF uses tag 34675.

## Metadata

- Copy EXIF from the original: camera, lens, exposure, ISO, capture time, GPS (unless the preference "Remove location from exports" is on), orientation reset to normal (we write upright pixels).
- XMP (`xmp:CreatorTool = "Yellowmoon <version>"`) plus our namespace `https://yellowmoon.app/ns/1.0/`, prefix `ym`:

| Property | Example |
|---|---|
| `ym:PipelineVersion` | `1.0.0` |
| `ym:Profile`, `ym:ProfileVersion`, `ym:ProfileMatch` | `sony-ilce-7rm4`, `1.0.0`, `default` |
| `ym:Look`, `ym:LookSHA256`, `ym:LookIntensity` | `arri:3000:…`, `…`, `100` |
| `ym:DisplayTransform` | `aces2-sdr100-p3lim` or `arri-display:…` |
| `ym:Models` | `jdd 1.2.0 (sha256…)` |
| `ym:Settings` | compact JSON of the resolved DevelopSettings |

- No sidecars. Nothing is written anywhere except the exported file.

## DNG with the look baked in

Goal: a Yellowmoon DNG opens in other raw editors looking like our render, and keeps room to adjust exposure, highlights and white balance. Research and app-by-app behaviour: `research/notes/dng-export.md`.

### Pixels

- LinearRaw (PhotometricInterpretation 34892), 3 samples, 16-bit unsigned, scene-linear **AWG4** after C1–C11: demosaiced, denoised, lens-corrected, look and curve applied in LogC4 and decoded back to linear.
- Scale: `stored = round(65535 · clamp(x / x_white, 0, 1))` with `x_white = max(p99.99 of the photo's max(R,G,B), 1.0)`. BaselineExposure compensates (fitted, below).
- Negative values (out-of-gamut) are clipped to 0 at this point; record the clipped-pixel fraction in the export log.
- Shadow precision: 16-bit linear holds about 10 stops below grey before codes drop under 1. Acceptable for denoised data; revisit float or a LinearizationTable only if testing shows banding.
- Full resolution. When the crop has no straighten angle, pixels keep the lens auto-crop only and the user's crop goes into DefaultCropOrigin/Size, so other apps can undo it. DNG has no rotated-crop tag, so a non-zero straighten angle is baked into the pixels (rotated, then cropped to the straightened rectangle). 90° steps (`orientation`) go into the TIFF Orientation tag without rotating pixels.

### Profile (embedded)

- `ColorMatrix1` = `inv(AWG4 → XYZ)` (XYZ under D65 → camera, i.e. → AWG4) with `CalibrationIlluminant1 = 21` (D65). `ForwardMatrix1` maps white-balanced camera RGB to **XYZ D50**, as the DNG spec requires: `Bradford(D65 → D50) · (AWG4 → XYZ)`, scaled so camera white (1,1,1) maps to D50 XYZ. `AsShotNeutral = 1, 1, 1`.
- `ProfileToneCurve` and a `ProfileLookTable` (HSV, ≤ 36×32×16, `ProfileLookTableEncoding = 1`) **fitted to our default display transform (ACES 2)**, never to ARRI's display LUTs, so Adobe's pipeline shows nearly what we show.
  - Fit procedure (`tools/dngfit`): render a set of synthetic test images (neutral ramp, hue/saturation sweeps, our golden photos) through ACES 2 (target) and through Adobe's reference pipeline using `dng_validate` from the DNG SDK (candidate). Fit BaselineExposure and the tone curve on the neutral axis first, then the LookTable for chroma, minimising ΔE00. The LookTable can't change neutrals (spec), so all neutral tone lives in the tone curve.
  - The fitted profile is the same for every photo (it depends only on ACES 2); BaselineExposure is per photo (`x_white` changes).
- `UniqueCameraModel = "Yellowmoon AWG4"` so Lightroom doesn't substitute Adobe's Sony profiles. `ProfileName = "Yellowmoon"`, `ProfileEmbedPolicy = 3`, `ProfileCopyright` with our notice.
- ARRI looks are baked into pixels only, never into profile tables: a table derived from an ARRI LUT would put a copy of ARRI's data into every file.

### Tags

IFD0:

| Tag | Value |
|---|---|
| DNGVersion / DNGBackwardVersion | 1.4.0.0 / 1.1.0.0 |
| UniqueCameraModel | `Yellowmoon AWG4` |
| ColorMatrix1, CalibrationIlluminant1, ForwardMatrix1 | XYZ(D65)→AWG4, 21 (D65), AWG4→XYZ(D50) via Bradford |
| AsShotNeutral | 1, 1, 1 |
| BaselineExposure | fitted per photo |
| BaselineSharpness | 1.0 (already sharpened; adjust if Lightroom over-sharpens) |
| ProfileName, ProfileEmbedPolicy, ProfileCopyright | `Yellowmoon`, 3, `Yellowmoon contributors` |
| ProfileToneCurve, ProfileLookTableDims, ProfileLookTableData, ProfileLookTableEncoding | fitted, 1 |
| DefaultBlackRender | 1 (no further black subtraction) |
| ColorimetricReference | 0 |
| Make, Model, EXIF, XMP | copied / ours as above |
| PreviewColorSpace | sRGB |

Raw sub-IFD: NewSubFileType 0, PhotometricInterpretation 34892, SamplesPerPixel 3, BitsPerSample 16, Compression 1 (uncompressed strips) in V1, BlackLevel 0, WhiteLevel 65535, `NoiseReductionApplied = 1.0`, DefaultCropOrigin/Size, ActiveArea.

Preview sub-IFD: a full-size JPEG (quality 92, sRGB) of the exact render, so Quick Look and preview-based apps show our image.

No `NoiseProfile` in V1 (the data is denoised; `docs/06`).

### Writer

Our own, on the `tiff` crate (MIT): `write_tag(Tag::Unknown(n), …)` for DNG tags, `extra_directory()` and `finish_with_offsets()` for sub-IFDs, strips with Compression = 1. The `dng` crate (AGPL) and rawler's writer (LGPL) are excluded.

### What other apps show

| App | Expected |
|---|---|
| Lightroom, Camera Raw | close to Yellowmoon's render (minus fit error) |
| RawTherapee, darktable | the look's colours under their own tone curve |
| Capture One | the look's colours; exposure may be off, as with other linear DNGs |
| DxO PhotoLab | doesn't open third-party DNGs |
| Apple Photos, Preview, Quick Look | embedded preview or Apple's rendering (verify in M3) |

White-balance changes made later in another app act after the baked look, so they're approximate. Say so in the help text.

### Legal

Adobe's DNG patent licence is royalty-free and requires this notice "in a prominent manner within its source code and documentation": **"This product includes DNG technology under license by Adobe."** Put it in `NOTICE`, the README, the About window and the DNG export help text. The licence ends if we bring a patent action against Adobe.

### Gates

- Every exported DNG passes `dng_validate` with no errors in CI.
- `dng_validate` render vs our JPEG of the same photo: ΔE00 median ≤ 1.5 over the golden set.
- [HUMAN M3-H6] Opens in Lightroom Classic, RawTherapee, darktable, Capture One and Apple Photos; screenshots attached to the M3 report.

## Tests

- Golden exports: fixed A1 and corpus files × each format × each colour space, hashed. A change in hash needs an explanation in the PR (and usually a `pipeline_version` bump).
- App vs CLI: the same job exported through both is byte-identical.
- ICC: every output decodes in a colour-managed reader (ImageMagick or moxcms) to the expected values at test patches within 1 code value.
- Resize: Lanczos in linear light; check a zone plate for ringing limits.

## As built (M1-13 export writers, M3-12 DNG writer)

Crate `ym-export`. It depends on `ym-color` only (not on `ym-raw` or `ym-core`): its inputs are plain structs that `ym-core` fills in. Entry points: `write_export(&ExportRequest, &ExportDestination, &ExportHooks) -> ExportOutcome` (one photo to one file, atomically), `encode_to_vec` / `encode_to_writer` (the same bytes into memory, for tests and hashes), `estimate_bytes(width, height, format, size)` for `export_estimate`. `ExportOptions` serialises exactly as `ExportOptions` in `src/api/types.ts` (flat: `format`, `jpeg_quality`, `bit_depth`, `tiff_compression`, `color_space`, `size`, `destination`, `name_template`, `if_exists`); `ExportOptions::format_spec()` gives the richer `ExportFormat` enum the writers take.

**Pixels in.** JPEG, TIFF and PNG take `f32` RGB, interleaved: `DisplayLinear` (linear light in the export colour space's primaries, the output of C12; the writers do C13 with `ym_color::encode`) or `DisplayEncoded` (already C13-encoded, 0 to 1; a resize decodes it to linear first). DNG takes `SceneLinearAwg4` (C11 output decoded from LogC4) plus a display-linear sRGB preview. The writers convert a band of about one million pixels at a time, in parallel row by row, so the bytes do not depend on the thread count (tested). The input buffer is one full image; streaming tile rows into the writers is a follow-up.

**Names and folders.** `{name}` is the source file name without extension, `{look}` the look's slug, `{date}` the **capture date** as `YYYY-MM-DD` (`undated` when unknown; never the export date, for determinism), `{seq}` the 1-based number of the photo in the job with at least three digits, `{camera}` the camera body as a slug (`sony-a1`; see the review notes below). Unknown `{words}` stay as typed (as in the UI preview). The stem is made safe: `/ \ : * ? " < > |` and control characters become `_`, leading dots and trailing dots and spaces are dropped, at most 200 bytes. The default folder is `<source folder> export` next to the source folder, then `~/Pictures/Yellowmoon export`; a folder the user chose is never replaced by a fallback. Skip and Keep both never overwrite, even if another process creates the file during the export (the final step is a hard link, or a checked rename on file systems without hard links such as exFAT). The temporary file is `.yellowmoon-<pid>-<n>.tmp` in the destination folder, fsynced before the rename; an error, a cancel or a panic deletes it. A target that resolves to the photo's original file is refused. Cancel is polled between bands of about one million pixels, so "after the current tile" is a fraction of a second of work (a batch of TIFF strips or a PNG band compresses in well under a second).

**JPEG.** Baseline, 4:4:4 at quality 90 and above, 4:2:0 with box-averaged chroma below, optimised Huffman tables, JFIF 300 dpi, then `APP1` Exif, `APP1` XMP and the ICC profile in `APP2` chunks. The `jpeg-encoder` crate writes optimised-table output as one scan per component, which `zune-jpeg` (Rust's `image` crate) decodes with wrong colours; `src/jpeg/interleave.rs` rewrites it as a single interleaved scan with tables rebuilt by T.81 Annex K.2 (coefficients and quantisation tables unchanged; the decoded pixels are identical to the standard-table interleaved file, tested over 17 sizes, both samplings and three quality settings). Sizes above 65,535 px are refused with a message.

**TIFF.** RGB strips of about 512 KB, compressed in parallel with the `tiff` crate's LZW and Deflate; the horizontal predictor for both compressed modes; ICC 34675; XMP 700; Exif and GPS sub-IFDs; orientation 1; 300 dpi; all out-of-line values start on even offsets (the `tiff` crate does not align them; the writer pads). Files above 4 GB are refused. LZW on noisy 16-bit samples can produce a file as large as the uncompressed one (measured 0.8 to 1.2 of raw on noisy data); Deflate is smaller. The default "16-bit, LZW" of the table above is therefore a candidate for "16-bit, ZIP".

**PNG.** `iCCP`, `cICP` (written by hand; the `png` crate parses but does not write it), `pHYs`, `eXIf` and an `iTXt` with `XML:com.adobe.xmp`. cICP code points `[primaries, transfer, matrix, full range]`: sRGB `1, 13, 0, 1`; Display P3 `12, 13, 0, 1`; Rec.2020 `9, 1, 0, 1` (H.273 has no code for the BT.1886 curve; transfer 1 is the one it pairs with a BT.1886 display, and the ICC profile carries the exact gamma 2.4). The `iCCP` profile name written by the `png` crate is `_`.

**ICC (`icc.rs`).** Generated, 612 to 632 bytes each: v4.3, class `mntr`, `RGB `/`XYZ `, fixed creation date 2026-10-01, MD5 profile ID, `desc` and `cprt` as `mluc`, `wtpt` and the header illuminant = D50 PCS (0.9642, 1, 0.8249), `chad` = Bradford from D65 (x 0.3127, y 0.3290) to D50, colorants = `chad` times the primaries' D65 matrix from `ym-color`, one shared `para` tone curve: type 3 `g 2.4, a 1/1.055, b 0.055/1.055, c 1/12.92, d 0.04045` for sRGB and Display P3, type 0 `g 2.4` for Rec.2020. Verified with LittleCMS 2.14 (C library through ctypes, and Pillow's ImageCms): 15 files (5 formats x 3 spaces) x 272 patches (a 6x6x6 cube, a grey ramp, four near-blacks): XYZ within 1.7e-5 of `ym-color`'s maths, sRGB codes within 1 (LittleCMS with `NOOPTIMIZE`; its default fixed-point matrix-shaper optimisation is off by up to 11 codes on extreme out-of-gamut dark patches, which is its approximation and not the profile).

**EXIF (`exif.rs`).** Built from `ExifInfo` (make, model, lens make and model, exposure time, f-number, ISO, focal length and 35 mm equivalent, exposure compensation, capture time with sub-seconds and UTC offset, optional GPS), never copied from the source: orientation 1, no thumbnail, no maker note, no serial numbers, text reduced to ASCII. `ComponentsConfiguration` is Y, Cb, Cr plus `YCbCrPositioning` for JPEG and R, G, B for TIFF and PNG; `ColorSpace` is 1 for sRGB and 0xFFFF for the others. ISO above 65,535 uses `ISOSpeed` with `SensitivityType` 3. With "Remove location" on, the GPS directory is not written (the output is byte-identical to one made without a position; tested). `DateTime` (IFD0) is the capture time.

**XMP (`xmp.rs`).** `xmp:CreatorTool`, `xmp:CreateDate`, `xmp:ModifyDate` and `xmp:MetadataDate` (all the capture time), `dc:format`, then the ten `ym:` properties in the order of the table above; `ym:Settings` is the compact JSON of a `serde_json::Value` (keys sorted). Fixed property order, no padding.

**Resize (`resize.rs`).** Lanczos-3, separable, support scaled by the reduction factor, taps outside the image dropped and weights renormalised: the same convolution as Pillow's `LANCZOS`, to which it is compared on float images (worst difference 0, six size pairs). Accumulation in f64, so a flat field is exact. Step-edge overshoot stays under 10 %; the zone-plate test bounds the output to [-0.10, 1.10] for shrink factors 2, 3 and 5. The long edge never upscales; the short edge is rounded half up. The UI grid (1000 px up in 20 px steps, then the full size) is `Size::ui_long_edge_choices`; the library accepts any long edge from 16 to 65,535 px.

**DNG (`dng.rs`).** Layout: IFD0 is an 8-bit RGB thumbnail (long edge at most 256 px, `NewSubFileType` 1) carrying the DNG profile tags, the Exif IFD, the GPS IFD and the XMP; `SubIFDs` points to the LinearRaw IFD (`NewSubFileType` 0, 3 x uint16, `Compression` 1, `BlackLevel` 0, `WhiteLevel` 65535, `NoiseReductionApplied` 1, `DefaultCropOrigin/Size`, `ActiveArea`) and to the preview IFD (full-size JPEG, quality 92, 4:4:4, `PreviewColorSpace` 2). The directories are written before IFD0, so IFD0 is last in the file. **`PreviewColorSpace` 2 is sRGB** (1 is "Gray Gamma 2.2"); exiftool reported the first draft's value 1. The preview and the thumbnail are in the raw pixels' orientation (not rotated) and cropped as the user sees the render, because the `Orientation` tag of IFD0 applies to all three. `compose_orientation(camera exif orientation, user quarter turns)` gives the tag value. `x_white` is the upper edge of the histogram bin (2048 bins per octave) that holds the 99.99th percentile of `max(R, G, B)`, at least 1.0; `DngReport` has `x_white`, `baseline_exposure_ev`, the fraction of pixels with a negative channel (clipped to 0) and the fraction above `x_white`. **`BaselineExposure` = `log2(x_white)` + a fitted offset** (`BaselineExposure::Auto { fit_offset_ev }`, offset 0 until `tools/dngfit` exists), or `Fixed(ev)`. `RawDataUniqueID` is the first 16 bytes of the SHA-256 of the stored pixels. `ProfileToneCurve` and the look table are inputs (`DngProfile`); `DngProfile::identity()` writes the identity curve and a 2 x 2 x 2 identity table until `tools/dngfit` has fitted the real ones (an identity tone curve shows a flat, dark image in Adobe's apps, so a DNG written before the fit is for structure tests only). **A look table with one value division (a "2.5D" table) must use linear encoding (`ProfileLookTableEncoding` 0):** with encoding 1 (sRGB) Adobe's reference renderer (DNG SDK 1.7.1) darkens even an identity table (mid grey 0.18 comes out as 0.027). `LookTable::validate` rejects the combination; `tools/dngfit` should fit a table with at least two value divisions when it uses sRGB encoding, as the 36 x 32 x 16 of this document does. Matrices: `ColorMatrix1 = inv(AWG4 to XYZ)`, `ForwardMatrix1 = Bradford(D65 to D50) * (AWG4 to XYZ)` scaled per row so (1,1,1) maps to D50 exactly, both as `SRATIONAL` with a denominator of 10^7. Checks (`tools/dng/check_dng.py`, run by `tests/dng_validate.rs`, 62 checks per file; the two `dng_validate` checks need the binary and are skipped without it): exiftool lists every tag with the expected value and `exiftool -validate` reports no warnings; tifffile parses IFD0 and the two SubIFDs; LibRaw (rawpy) opens the file and returns exactly the stored pixels, which are compared with the formula computed independently from the original floats; the matrices are recomputed from the AWG4 primaries and agree with colour-science. `tools/dng/build_dng_validate.sh` builds Adobe's `dng_validate` from the DNG SDK 1.7.1 (the AOSP copy, `GrapheneOS/platform_external_dng_sdk` at `0f23d021`, pinned by commit; about 10 minutes on 4 cores) with the XMP toolkit and JPEG XL stubbed out. With it, five DNG variants (plain, 5 x 3, 131 x 77, rotated and cropped with a fitted-style tone curve and look table, mirrored) validate with no warning or error, and Adobe's own renderer, given flat AWG4 patches and an identity profile, returns the sRGB and Display P3 colours `ym-color` computes to within 3.0e-4 and 9e-5 in encoded values (16 patches, saturated ones included): `ColorMatrix1`, `ForwardMatrix1`, `AsShotNeutral`, D50 and the Bradford adaptation agree with Adobe's code, not only with ours.

**Size estimate (`estimate_bytes`).** Measured on the 24 Kodak images (`tests/calibrate_estimate.rs`, run with `--ignored`): JPEG bits per pixel 0.99 (q50), 1.48 (q75), 2.00 (q85), 2.35 (q89), 2.95 (q90), 3.28 (q92), 4.25 (q95), 10.2 (q100); TIFF/PNG over raw size: 8-bit LZW 0.70, ZIP 0.63, PNG 0.60; 16-bit (with the noise of a rendered frame) LZW 0.90, ZIP 0.74, PNG 0.72. Film scans are not camera frames: see the review notes below for a real 24 MP frame, which came out larger than the estimate for JPEG at quality 90 and above and for 16-bit TIFF. The seconds in `export_estimate` come from `ym-core`.

**Tests.** Golden hashes (`tests/data/golden_hashes.json`, 39 files: every format and colour space of a 64 x 48 synthetic scene, long-edge variants, encoded input, DNGs; update with `YM_UPDATE_GOLDEN=1`); thread-count independence; ICC against LittleCMS; resize against Pillow; DNG against LibRaw/exiftool/tifffile; names, destinations, atomic write, cancel and the original-file guard (`tests/write_export.rs`, `src/naming.rs`). Tests that need Python modules or exiftool print `SKIPPED` when they are missing and fail instead when `YM_REQUIRE_EXTERNAL_TOOLS` is set (use it in CI).

**Review notes (independent review of M1-13 and the M3-12 writer).** Clarifications and decisions that came out of re-checking the writers against the spec and against independent readers; the code and tests are in `crates/ym-export`.

- **Directories start on even offsets.** The `tiff` crate writes a directory where the previous data ended, so an odd-length text value (a 12-character lens name gives 13 bytes with the NUL; an app version such as `0.1.10`) put the next directory at an odd offset, which the TIFF specification forbids. Every TIFF and DNG directory is now padded to an even offset before it is written (`exif::finish`), and `tests/structure.rs` walks every directory of TIFF and DNG files built with odd-length text.
- **Look table rule.** `LookTable::validate` also enforces the DNG specification's "all zero input saturation entries are required to have a value scale factor of 1.0" (the first saturation division of every hue and value division) and non-negative scales. `tools/dngfit` must produce tables that satisfy it: neutral tone belongs in `ProfileToneCurve`.
- **`BaselineExposure`.** Checked against Adobe's renderer, not only against our arithmetic: a scene with `x_white` 2.6 renders to the colours `ym-color` computes from the original scene-linear values with `BaselineExposure = log2(x_white)` (worst difference 2.9e-4 in encoded values), and a fitted offset of -1 EV halves the light as expected.
- **Exif pixel dimensions of a DNG** are the size of the default crop when there is one (the meaningful image), not of the whole raw frame.
- **`{camera}` and `{look}`** are slugs, exactly as the export panel previews them (`Sony A1` gives `sony-a1`); `naming::slugify` is the same function as `slugify` in `src/export/template.ts`, so `ym-core` and the CLI should pass `slugify(body)`.
- **GPS seconds** are computed on whole ten-thousandths of an arc second, so a latitude a hair below a minute carries into the minute instead of writing "60 seconds".
- **Open question for M3-H6: undoable crop.** `DefaultCropOrigin` and `DefaultCropSize` are, for Adobe's apps, the bounds of the final image; they do not give the user an adjustable crop. DNG 1.4's `DefaultUserCrop` (tag 51125, four rationals relative to the default crop) is the tag Lightroom and Camera Raw show as an initial, adjustable crop. This document says `DefaultCropOrigin/Size` and the writer does that; try `DefaultUserCrop` when the DNG is opened in Lightroom (M3-H6).
- **Licence acknowledgement for the JPEG encoder.** `jpeg-encoder` is `(MIT OR Apache-2.0) AND IJG` (its forward DCT is ported from the Independent JPEG Group's software). `cargo deny check licenses` fails until `IJG` is added to `deny.toml`, and the IJG licence asks that documentation accompanying a binary says "This software is based in part on the work of the Independent JPEG Group." (`NOTICE`, the About window, `docs/14`).
- **What `exiftool -validate` says about the other formats.** JPEG and PNG: no warnings. TIFF: three minor remarks (`ComponentsConfiguration`, `ExifImageWidth` and `ExifImageHeight` are "compressed data" tags in the Exif specification and are kept because readers use them), a note that `GPSProcessingMethod` is not written, and, for Deflate (compression 8), a note that it is not among the values it expects (the `tiff` crate and tifffile read these files). The DNG validates cleanly.
- **Rec.2020 `cICP`.** PNG readers that honour `cICP` use it instead of the ICC profile, and H.273 has no code for the BT.1886 curve, so they decode the BT.709 code point (transfer 1), which is not the pure gamma 2.4 of the embedded profile and of our encode. If that shows in practice, write no `cICP` for Rec.2020 and leave the colour to the ICC profile.
- **Size estimate on a real 24 MP frame, and the TIFF default.** The ILCE-7M3 sample (6024 x 4024) developed by LibRaw, which does not denoise and is not a Yellowmoon render, exported to (actual / estimate, MB): JPEG q50 2.4 / 3.0, q75 4.0 / 4.5, q89 7.1 / 7.1, q92 15.2 / 9.9, q100 43.9 / 30.9; TIFF 8-bit LZW 61.4 / 50.9; TIFF 16-bit LZW 182.4 / 130.9 and ZIP 132.3 / 107.6 (the raw size is 145 MB); PNG 8-bit 51.2 / 43.6 and 16-bit 132.7 / 104.7. Sensor noise compresses worse than a film scan, so the estimate runs low for the quality 90 and above (4:4:4) JPEGs and the 16-bit files; the quality 50 and 75 JPEGs are over-estimated. A Yellowmoon render is denoised, so re-measure on renders in M3 (the ignored `calibrate_estimate` test takes any set of images). **16-bit LZW wrote a file 26 % larger than the uncompressed one and 38 % larger than ZIP** for similar time (10 s and 12.5 s in a release build on a loaded 4-CPU Linux machine): the default "16-bit, LZW" of the Options table should become "16-bit, ZIP" (a decision for Mykyta; nothing changed here). Release timings of the same run for 24 MP: JPEG q92 5.9 s, TIFF 8-bit LZW 4.8 s, PNG 8-bit 7.3 s, PNG 16-bit 9.9 s (including the display encode, with other builds running).
