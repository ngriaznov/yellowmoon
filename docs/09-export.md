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
