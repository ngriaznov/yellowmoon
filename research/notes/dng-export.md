## Yellowmoon DNG export research (11 Oct 2026)

**Source caveat:** I couldn't download the DNG 1.7.1 PDF. Adobe's server blocks automated fetches, and the proxy here blocks it too. Instead I used Adobe's DNG SDK source code: 1.4 from a GitHub mirror and 1.7.1 from AOSP (Android's source tree). For the spec wording I used the tag text reproduced by AWare Systems, and ExifTool's source for tag IDs and versions.

### 1. Recommendation: design (a), with a full-size rendered preview

- **Pixels:** Use LinearRaw (PhotometricInterpretation 34892), 3×uint16, scene-linear AWG4. Apply the look in LogC4, then decode back to linear before writing. The look's colour changes then survive in every app that opens linear DNGs.
  - Avoid float16. Capture One users report pink previews on Lightroom's float HDR DNGs [12].
- **Profile:** ColorMatrix1/ForwardMatrix1 describe linear AWG4 at D65, and AsShotNeutral is 1,1,1. Fit a ProfileToneCurve plus a 3D ProfileLookTable so that Adobe's pipeline approximates our display transform.
  - Fit the tone curve on the neutral axis first, then fit the LookTable for chroma.
  - The LookTable can't change neutrals (section 3). Display transforms keep neutrals neutral, so that limit matters less here than in (b).
  - Fit to ACES 2 rather than ARRI's display LUT (licensing, section 5).
- **UniqueCameraModel:** set a unique value, e.g. "Yellowmoon AWG4", so Lightroom doesn't swap in Adobe's Sony profiles, which assume Sony-native primaries. This is my reasoning; the behaviour is unverified.
- **Preview:** embed a full-size JPEG of the exact render. Apple ProRAW does the same [8].
- **Latitude:** highlights, shadows, exposure and white balance stay editable. White-balance changes are applied after the baked look, so they're approximate.

**Why not (b), look only as a LookTable:** only Adobe reads it by default. RawTherapee reads it only if the user picks "Use embedded" [10]. Neutrals can't be tinted or re-toned, and the file carries a derivative of ARRI's LUT (section 5).

**Why not (c), display-referred:** ColorimetricReference=1 makes the DNG SDK use an identity tone curve and no shadow ramp [4]. Other apps still apply their own curves on top (unverified), latitude is lost, and you effectively get a 16-bit TIFF. Offer TIFF for that case instead.

### 2. What each app will show with (a)

- **Lightroom / Camera Raw:** the embedded profile is the default for unknown camera models [15]. They honour the profile tags, so the result is close to Yellowmoon's, minus fit error.
- **RawTherapee:** the default "Camera standard" uses the matrix only. "Use embedded" loads the DNG itself as a DCP (camera profile) [10]. It doesn't read the DNG BaselineExposure tag. Expect the look's colours with RawTherapee's own tone.
- **darktable:** rawspeed opens LinearRaw (uint16 or float) [11]. Current master reads ColorMatrix/ForwardMatrix and ProfileHueSatMap, but not LookTable, ToneCurve or BaselineExposure. Expect the baked look under darktable's own tone mapper.
- **Capture One:** no DCP support (user statement [12]). DxO PureRAW linear DNGs open about ⅔ to 2 EV too bright with colour shifts, and an admin said a fix is "not on the roadmap". Expect the look's colours with wrong exposure (unverified for our files).
- **DxO PhotoLab:** officially opens only DNGs from Adobe software or supported cameras. Third-party DNGs fail [13].
- **Apple Photos / Preview:** Core Image applies its own "boost" curve [9]. WWDC21 says ProRAW's BaselineExposure, ProfileToneCurve and ProfileGainTableMap are the "recipe" [8]. Whether Apple reads LookTable is unverified.
- **Affinity Photo:** uses its Serif Labs or Apple engine with a "suggested tone curve" [16]. Support for profile tags is unverified.
- **Previews:** any app that shows embedded previews (Quick Look, asset managers, Lightroom's "Embedded & Sidecar" import) shows the exact render.

**Precedents:**

| Product | Baked into pixels | Left to tags / host app |
|---|---|---|
| Apple ProRAW | demosaic only (LinearRaw, 12-bit companded) | rendering lives in profile tags ("recipe, not the cake") [8] |
| DxO PureRAW / PhotoLab | demosaic, denoise, optics | colour stays in "the camera's linear sensor color space" [13] |
| Adobe Enhance | 16-bit LinearRaw, plus the original CFA mosaic | some apps read the CFA and lose the denoise [14] |
| Lightroom HDR / pano merge | 16-bit float LinearRaw | colour and tone (camera profile, settings) [12] |
| Topaz | unverified | unverified |

None of these bakes a creative look into the pixels.

### 3. How ProfileLookTable is applied

From the SDK [4][5] and the spec text [6]:

- **Where it runs:** in linear ProPhoto RGB (camera → XYZ D50 → ProPhoto). Spec: "after any exposure compensation and/or fill light stages, but before any tone curve stage." ProfileHueSatMap (interpolated by white balance) runs before exposure.
- **Lookup:** RGB is converted to hexcone HSV (V = max channel). Hue wraps; S runs 0–1; V is linear or sRGB-encoded (ProfileLookTableEncoding=1). With 2 dims the table is "2.5D".
- **Entries:** each holds a hue shift in degrees, a saturation scale and a value scale. The SDK interpolates these modifiers linearly across H, S and V.
- **Output:** s′ = min(s·k, 1); in SDR profiles V is clamped to [0,1].
- **Neutral limit:** spec: "All zero input saturation entries are required to have a value scale factor of 1.0." Exact greys can't be tinted or re-toned, so split-toning is impossible. Tone on the neutral axis must go into the single ProfileToneCurve.
- **Accuracy:** smooth hue and saturation changes fit well. An arbitrary 3D LUT does not. Measure ΔE on test grids; I'm not claiming any figure.
- **Size:** the SDK caps look tables at 36×32×16 samples [5].

**Newer DNG features:**
- **RGBTables** (DNG 1.6, tag 0xCD3F): a real RGB LUT, up to 32³, 16-bit. Primaries sRGB, Adobe RGB, ProPhoto, P3 or Rec2020; gamma linear, sRGB, 1.8, 2.2 or Rec2020; gamut clip or extend; optional semantic mask; an amount range [5]. It can tint neutrals. Only Adobe is known to read it, and its position in the pipeline is unverified.
- **ProfileGainTableMap** (1.6) and **ProfileGainTableMap2** (1.7): spatial gain maps for local tone mapping.
- **ProfileDynamicRange** (1.7): Standard/High plus HintMaxOutputValue. It enables overrange values in the 1.7.1 renderer.
- **JPEG XL** (1.7; JXL tuning tags in 1.7.1) [7]: avoid it, because support outside Adobe and Apple is unverified.

### 4. Tag list for (a)

- **IFD0:**
  - DNGVersion 1.4.0.0 (1.6.0.0 if you use RGBTables); DNGBackwardVersion 1.1.0.0
  - UniqueCameraModel (unique value, see above)
  - ColorMatrix1 + CalibrationIlluminant1=21 (D65); ForwardMatrix1; AsShotNeutral
  - BaselineExposure (fitted); BaselineSharpness
  - ProfileName / AsShotProfileName / ProfileCopyright; ProfileEmbedPolicy=3
  - ProfileToneCurve; ProfileLookTableDims/Data; ProfileLookTableEncoding=1
  - DefaultBlackRender=1 ("should not perform any black subtraction")
  - ColorimetricReference=0; PreviewColorSpace
- **Raw SubIFD:** NewSubFileType=0, 34892, 3×16-bit, Compression=1 (or 7), BlackLevel=0, WhiteLevel, NoiseReductionApplied=1.0 ("should not apply additional noise reduction by default"), default crop tags.
- **Preview SubIFD:** full-size JPEG.

Validate files with Adobe's `dng_validate` tool from the DNG SDK, which renders through the reference pipeline.

### 5. Licensing (risk flags, not legal advice)

**Adobe DNG patent licence** [1][3]:
- Grant: "worldwide, royalty-free, nontransferable, nonexclusive right under all Essential Claims to make, have made, use, sell, import, and distribute Compliant Implementations."
- Scope: "solely to permit the reading and writing of image files that comply with the DNG Specification."
- Revocation: Adobe may revoke the rights if the licensee "brings any patent action against Adobe or its affiliates."
- Attribution: the notice must be "displayed in a prominent manner within its source code and documentation." The current text is "This product includes DNG technology under license by Adobe." (the older text says "Adobe Systems Incorporated").
- Rights are provided "as-is".

**ARRI LUT:**
- Baking the look into pixels produces images and ships no reusable copy of the LUT.
- A LookTable or RGBTables derived from the LUT puts a resampled copy of ARRI's all-rights-reserved data into every exported file. It is extractable (anything that can load a DNG as a profile, e.g. RawTherapee [10]), and ProfileEmbedPolicy is only a hint. That is a materially higher redistribution risk.
- The same applies if the profile is fitted to ARRI's display LUT. The ACES 2 output transforms are Apache-2.0 [18].
- ARRI's Look Library FAQ gives no licence terms [19].

### 6. Rust writer options

- **`tiff` 0.11.3 (MIT):** workable.
  - `write_tag(Tag::Unknown(n), …)` covers integers, floats, Rational/SRational, ASCII and IFD offsets.
  - `extra_directory()` plus `finish_with_offsets()` builds SubIFDs.
  - Tags are stored in a BTreeMap, so writing PhotometricInterpretation=34892 again overrides the RGB value the image encoder sets.
  - Compression is none/LZW/Deflate/PackBits only: no tiled writer and no lossless JPEG. Use strips with Compression=1, or write tiles by hand. rawspeed reads both strips and tiles.
- **`rawler` 0.8.0 (LGPL-2.1):** a full DNG writer (LinearRaw, tiles, lossless JPEG, float). LGPL relinking obligations are awkward for a statically linked Rust binary.
- **`ljpeg` 0.1.2:** lossless-JPEG encoder, also LGPL-2.1.
- **`dng` 1.6.0:** AGPL-3.0, excluded.
- **`tiff-encoder` 0.3.2 (MIT):** low-level; maintenance unverified.

Writing DNG on top of `tiff` is straightforward.

### Sources
1. https://helpx.adobe.com/camera-raw/desktop/dng-and-file-formats/digital-negative.html
2. https://helpx.adobe.com/content/dam/help/en/camera-raw/digital-negative/jcr_content/root/content/flex/items/position/position-par/download_section_733958301/download-1/DNG_Spec_1_7_1_0.pdf (not fetchable here)
3. https://android.googlesource.com/platform/external/dng_sdk/+/b76282adc88bc87cc28ce629f45c9051efd95e53/PATENTS
4. https://github.com/yanburman/dng_sdk (source/dng_render.cpp, dng_reference.cpp, dng_shared.cpp, dng_camera_profile.cpp)
5. https://android.googlesource.com/platform/external/dng_sdk/+/6b5cf2a88ebd2b099e56d0d4717e962772ff9067/source/ (dng_tag_codes.h, dng_big_table.h, dng_render.cpp)
6. https://awaresystems.be/imaging/tiff/tifftags/profilelooktabledata.html (also profilehuesatmapdata1, profilelooktableencoding, defaultblackrender, colorimetricreference, noisereductionapplied)
7. https://github.com/exiftool/exiftool/blob/master/lib/Image/ExifTool/Exif.pm
8. https://developer.apple.com/videos/play/wwdc2021/10160/
9. https://developer.apple.com/documentation/coreimage/cirawfilter
10. https://rawpedia.rawtherapee.com/Color_Management ; https://github.com/Beep6581/RawTherapee (rtengine/rawimagesource.cc, dcp.cc)
11. https://github.com/darktable-org/darktable (src/common/exif.cc, src/iop/colorin.c) ; https://github.com/darktable-org/rawspeed (DngDecoder.cpp)
12. https://support.captureone.com/hc/fr/community/posts/13107964895261-Notice-for-users-exporting-DNGs-from-DxO-PureRAW ; https://support.captureone.com/hc/fr/community/posts/10170600184733-Integrated-native-DCP-ICC-profiler ; https://support.captureone.com/hc/de/community/posts/360012417318-Unable-to-open-a-DNG-file-from-Lightroom-HDR-merge
13. https://forum.dxo.com/t/pl6-dng-export-options/28584?page=2 ; https://forum.dxo.com/t/support-any-dng-file/17809
14. https://community.adobe.com/questions-675/can-an-enhance-nr-dng-image-file-be-opened-in-third-party-applications-like-capture-one-dxo-etc-970252
15. https://community.adobe.com/feature-requests-564/p-leica-dng-embedded-camera-profile-issue-666585
16. https://affinity.help/photo/en-US.lproj/pages/Raw/raw.html
17. https://crates.io/crates/tiff , /rawler , /dng , /ljpeg , /tiff-encoder (licences read from each crate's Cargo.toml)
18. https://github.com/ampas/aces-output (LICENSE: Apache-2.0)
19. https://www.arri.com/en/learn-help/learn-help-camera-system/image-science/look-files/look-library-faq