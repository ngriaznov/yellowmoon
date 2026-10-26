# 03 · Camera detection and profiles

This file covers how a photo's camera is detected, which colour profile it gets, what a profile file holds and how the runtime reads it, and the white scale that stand-in bodies carry. Crates: `ym-raw` (detection), `ym-profiles` (registry and data). Data: `data/cameras/`.

The camera is an input to the pipeline from the start. Adding a camera means adding a profile file and its tests; the pipeline code stays the same.

V1 relies on available measurements only: the 11 bodies Weta measured and published in rawtoaces-data. We do not measure cameras ourselves in V1; every other body maps to a same-sensor sibling or the A7R IV.

## Detection

For every file, before decoding pixels:

1. Read `Make` (0x010F) and `Model` (0x0110) from IFD0.
2. Read Sony's model ID (maker note tag `0xB001`, SonyModelID). When the model ID and the Model string disagree, the model ID wins (firmware variants and regional "A" models write odd strings).
3. Record the compression mode and bit depth from the raw IFD (`docs/04-raw-decoding.md`). This step belongs to the decoder: `DecodeKind` (`Uncompressed12 | Uncompressed14 | Craw | LosslessL | LosslessMS | Arw6 { .. }`) lives in `ym_raw::RawMetadata`. `ym-profiles` does not depend on `ym-raw`; `ym-core` composes the two (the UI and `ym info` show both).

Result type (`ym_profiles::detect`):

```rust
pub struct CameraMatch {
    pub make: String, pub model: String, pub model_id: Option<u16>,
    pub body: String,                 // "Sony A1" for display, from the registry or built from Make and Model
    pub profile: ProfileId,           // e.g. "sony-ilce-7rm4"
    pub profile_version: semver::Version,
    pub how: MatchKind,               // Exact | SensorFamily { via: String } | Default | Forced { detected: ProfileId }
}
```

- *`MatchKind::Forced { detected }`* is for the CLI's `--profile <id>`: the body is detected as usual and the forced profile replaces the detected one (`ProfileStore::detect_forced`). Forcing the profile detection chose returns the detected match unchanged. A forced profile is never reported as `Exact`, so the STAND-IN tag shows.
- *`model` and `model_id`.* `model_id` is what the file gave (`None` when it had no usable 0xB001). `model` is the registry's name for the body (`ILCE-1`) when the body is known, because the model ID wins; for an unknown body it is the trimmed Model string from the file.
- *`body`* is the registry's display name (`Sony A1`); for a body the registry does not list it is built from Make and Model (`Sony ILCE-7M6`, `Hasselblad Lunar`).
- *Entry points.* `ProfileStore::load(data_root)` loads the registry, the target and every profile once (the caller passes `ym_core::data_dir()`'s result; `ym-profiles` does not locate `data/` itself). `ProfileStore::detect(make, model, model_id) -> CameraMatch` (also `ym_profiles::detect::detect(&store, ..)`) never fails. `ProfileStore::resolve(&CameraMatch)` gives the `Profile` back.
- *Display copy.* `ProfileStore::camera_copy(&CameraMatch) -> CameraCopy { detected, profile, automatic_row, stand_in, note }` produces the text of `docs/11` for the app and the CLI: `Sony A1 (ILCE-1)` (the model in brackets is left out when the body name already says it: `Sony ZV-E1`, `Hasselblad Lunar`), `A1 · A7R IV profile` (a measured body shows just `A7R IV`), the STAND-IN flag, and the sentences for `Default` ("The A1 has no measured profile yet, so it borrows the A7R IV. Eleven Sony cameras have their own, and cameras that share their sensors use them."; the number is the count of profiles, in words) and `SensorFamily` ("The A7C shares its sensor with the A7 III, so it uses that profile."). A measured body has no note.

The app shows the body and the profile with each photo (Look tab, Automatic card: "A1 · A7R IV profile"; RAW tab, Camera card with Detected / Profile / a STAND-IN tag when `how != Exact`). `ym info` prints both.

Files with `Make` other than SONY but a valid ARW structure (Hasselblad Lunar and Stellar are rebadged Sony bodies) are decoded and use the default profile.

## Which profile each camera gets

| Bodies (model ID) | Profile | Match |
|---|---|---|
| ILCE-7M3 A7 III (363) | sony-ilce-7m3 | exact |
| ILCE-7C A7C (381) | sony-ilce-7m3 | same sensor |
| ILCE-7M4 A7 IV (388) | sony-ilce-7m4 | exact |
| ILCE-7CM2 A7C II (397) | sony-ilce-7cm2 | exact |
| ILCE-7RM2 A7R II (347) | sony-ilce-7rm2 | exact |
| ILCE-7RM3 A7R III (362) | sony-ilce-7rm3 | exact |
| ILCE-7RM3A A7R III A (386) | sony-ilce-7rm3 | same sensor |
| ILCE-7RM4 A7R IV (375) | sony-ilce-7rm4 | exact |
| ILCE-7RM4A (387) | sony-ilce-7rm4 | same sensor |
| ILCE-7SM2 A7S II (350) | sony-ilce-7sm2 | exact |
| ILCE-7SM3 A7S III (383) | sony-ilce-7sm3 | exact |
| ILME-FX3 (385), ZV-E1 (393) | sony-ilce-7sm3 | same sensor |
| ILCE-9 A9 (358) | sony-ilce-9 | exact |
| ILCE-9M2 A9 II (376) | sony-ilce-9 | same sensor |
| ILCE-6400 (371) | sony-ilce-6400 | exact |
| ILCE-6100 (379), ILCE-6600 (378), ILCE-6400A (402), ILCE-6100A (403) | sony-ilce-6400 | same sensor |
| DSC-RX100M4 RX100 IV (341) | sony-dsc-rx100m4 | exact |
| Everything else, including ILCE-1 A1 (384), ILCE-1M2 (400), ILCE-9M3 (392), ILCE-7M5 A7 V (407), ILCE-7RM6 A7R VI (410), the two bodies whose same-sensor pairing failed its check, ILCE-7RM5 A7R V (390) and ILCE-7CR (396), and the eighteen older, APS-C and compact bodies that carry a white scale fitted on their own files (registry 1.3.0), ILCE-7 A7 (306), ILCE-7R A7R (311), ILCE-7S A7S (318), ILCE-5000 A5000 (313), ILCE-6700 A6700 (394), ZV-E10 (382), NEX-5N (288), NEX-5R (296), NEX-7 (289), DSLR-A300 A300 (261), DSLR-A500 A500 (274), DSC-RX100 (297), DSC-RX100M2 RX100 II (308), DSC-RX100M7 RX100 VII (374), DSC-RX10M4 RX10 IV (365), ILCE-6000 A6000 (312), ILCE-6300 A6300 (357) and ILCE-6500 A6500 (360) | sony-ilce-7rm4 | default |

Model IDs are Sony's `SonyModelID` values (maker-note tag 0xB001) and the Model strings the cameras write for them. Before a same-sensor pairing ships, it is checked: the sibling's ColorMatrix must match the profile body's within 0.02 per coefficient after normalisation. The check, with its measured value, the tool and its version, is recorded in the profile's `provenance`. A pairing that fails does not ship: the body gets match kind `default` (the camera card says it borrows the A7R IV, which is also what it renders with) and the failed check stays in the provenance.

**Sibling check.** V1 has no Adobe DNG Converter samples (the V1 data rule), so the check compares the bodies' published D65 colour matrices (`ColorMatrix`, keyed by the Sony model ID; numbers only, `docs/14` rule 8), read by `crates/ym-profiles/tests/py/sibling_matrices.py`, rows normalised to sum 1: the largest per-coefficient difference between the sibling and the profile body.

| Pairing | Max \|diff\| | Result |
|---|---|---|
| ILCE-7C (381) to ILCE-7M3 | 0.0000 | pass |
| ILCE-7RM3A (386) to ILCE-7RM3 | 0.0000 | pass |
| ILCE-7RM4A (387) to ILCE-7RM4 | 0.0000 | pass |
| ILCE-7RM5 (390) to ILCE-7RM4 | 0.0934 | fail: default |
| ILCE-7CR (396) to ILCE-7RM4 | 0.0934 | fail: default |
| ILME-FX3 (385), ZV-E1 (393) to ILCE-7SM3 | 0.0000 | pass |
| ILCE-9M2 (376) to ILCE-9 | 0.0000 | pass |
| ILCE-6100 (379), ILCE-6600 (378), ILCE-6400A (402), ILCE-6100A (403) to ILCE-6400 | 0.0000 | pass |

0.0000 means the published matrices of the two bodies are the same, so the check is strict but says only that the published data does not tell them apart. The A7R V and the A7CR have a matrix of their own (a newer sensor generation, 0.0934 from the A7R IV's): they render with the A7R IV profile and are not described as sharing its sensor. `sibling_matrices.py --verify` compares the recorded values with a fresh run.

**Sensor geometry (registry 1.1.0).** Each registry body can carry `sensor_width_mm` and `active_width_px` (and an optional `pixel_pitch_um` override), the pixel pitch for the diffraction deconvolution of `docs/algorithms/03-diffraction-deconvolution.md` section 2.2: `1000 * sensor_width_mm / active_width_px` micrometres. They are Sony's published figures (sensor size in mm; the width of the largest 3:2 still image from the full sensor), with the source of every entry in `sensor_source` and `sensor_source_kind` (`maker` for an excerpt of a Sony specification page, `restated` for a retailer or reference site restating Sony's figures; the A7 III, A7 IV, A7C II, A7S III and A1 are `restated`). The Sony pages were read through web-search excerpts and the figures have not been compared with the pages themselves. 27 of the 46 bodies have them (pitches 2.41 µm for the RX100 IV to 8.40 µm for the A7S II and A7S III); the ILME-FX3 has none, because Sony's page gives its stills only as "12M", and neither have the eighteen bodies added in registry 1.3.0 (below, for their white scale); they use the EXIF crop-factor fallback. A body the registry does not list has none either. `ProfileStore::detect` puts them on the `CameraMatch` (`sensor: Option<SensorGeometry>`).

Measured curves for the 11 exact bodies are Weta's (rawtoaces-data, Apache-2.0), copied to `data/cameras/sony/`.

**Colour matrices.** `data/cameras/color-matrices.json` holds the published D65 colour matrix (XYZ to camera RGB) of 44 bodies, numbers only (`docs/14` rule 8). The Before view uses them (`docs/02` The Before view); the ILCE-1M2 (model ID 400) carries the ILCE-1's numbers and the ILCE-3000 (model ID 302, which Sony writes for the ILCE-3500 as well) the generic E-mount matrix, each with a note in the file. `data/cameras/camera-default-tone.json` holds the tone response of the camera's own JPEG engine, also for the Before view.

## White scale of stand-in bodies

A stand-in body renders through another body's profile, and that profile reads the stand-in's whites wrongly. A white is camera RGB with green 1 in the camera's **raw** scale (`docs/02` C1): the as-shot white balance, Pick white and the illuminant estimator's output are all whites of that kind, and the profile's `locus` is the white of every colour temperature in the scale of the curves it was measured with. A body with other channel sensitivities has other raw whites for the same light. On the A7R IV profile an A1 reads a daylight photo as about 4430 K (its Daylight preset reads 4433 K), and the Kelvin readout, the light class, the sensor map's mired and the ARRI matrix all follow that reading.

Each registry body therefore has a **white scale** `white_scale = [r, 1, b]` that takes its raw white to the white of its profile's curves: `w_profile = w_body ⊙ white_scale`, green fixed at 1. It belongs to the body and to the profile the registry gave it (so it is `[1, 1, 1]` after `--profile` has forced another profile, and for a body the registry does not list). It is not a 3 × 3 on the sensor map: correcting the stand-in's colour response by the body's colour matrix made three of five measured bodies worse (`research/dynamic-vs-lut-2026-10-14.md` check 4), so only the white is scaled.

**Where it acts** (`ym_color::wb`, `ym_color::chain`, `ym_core::auto`; `CameraMatch::white_scale` carries it from detection to `AutoEstimates::compute_scaled`):

| Place | What the scale does |
|---|---|
| as-shot white (the fallback light, and the estimator's prior) | the camera's white times the scale; the prior's gains are divided by it |
| illuminant estimator input | the 1/8 preview's red and blue are multiplied by the scale, so the estimator reads the body as the profile's raw (it is trained on that) |
| initial gains `g0` of the raw stages | `g0 = white_scale / w_scene_initial` (green 1), with `w_scene_initial` on the profile's scale: the mosaic of the body, balanced with `g0`, is what the profile's curves would have recorded for that light |
| Pick white | the sample is `raw_body * g0`, so `sample ⊙ w_scene_initial` is already the picked white on the profile's scale; the scale moves only the clip level the sample is judged by (channel `c` clips at `g0[c]`) |
| highlight clip level (C7) | a neutral surface clips the channel with the smallest `white_scale[c] / w_scene[c]` |
| everything else (C1's `g_res`, the sensor map, C3 to C6) | nothing: they work on the profile's scale already |

With `[1, 1, 1]` every one of these is the plain expression to the last bit (`1.0 / w`), so a body with its own curves, a body that shares its sensor, a body the registry does not list and a body with no data render exactly as without a scale; `tests/white_scale.rs` of `ym-color` and `ym-core` pin that.

**The two references** (`training/white_scale/`, `python3 -m training.white_scale.fit --report`). The scale needs a second source for the body's raw white, and V1 has two, both data the camera or its maker publishes and not a measurement of ours.

- *(a) The camera's own white-balance presets.* Every ARW stores the white balance of each of its presets in the SR2 sub-IFD (`docs/04`; our own reader, `training/white_scale/sr2.py`): Daylight, Cloudy, Shade, Tungsten, Flash, the Fluorescent ones and five Kelvin presets (2500, 3200, 4500, 6000, 8500 K). They are constants of the body and its firmware, up to the unit: the three A1 files of one camera carry identical ones, files of different cameras of one model differ by 1 to 10 % (*Consistency of a body's files* below). The fit is done file by file and puts the five Kelvin presets at their labels on the profile's locus, least squares in log chroma: `ln s_f = mean_i(ln locus(T_i) − ln preset_f,i)` per channel; the body's scale is the median of the scales of its files.
- *(b) The body's published colour matrix at D65* (XYZ to camera RGB; `training/white_scale/colour_matrices.json`). Its white of D65, `M · XYZ_D65` with green 1, is put on the profile's D65 (daylight 6504 K): `s = locus(6504 K) / that`. Two numbers, two conditions, no residual at D65. Only the D65 matrix is available; the matrix at A (2856 K) is not.

A reference decides when the mean angle between the Kelvin presets' whites and the locus whites at their labels is at most 2 degrees, and the Daylight preset (labelled 5500 K) reads between 5000 and 5600 K (within 300 K of its label). For the A1 the presets give 0.54 degrees and 5212 K; the matrix gives 2.07 degrees and 4720 K and fails both limits. **The presets set the scale**, and the A1's `white_scale` is `[0.8675, 1, 1.0861]` (rounded to four decimals; the registry's `white_scale_source` records the files, both references' residuals and the matrix scale).

**Consistency of a body's files (registry 1.3.0).** The decision rule says whether a reference puts a body's whites where the labels say; it does not say whether the body's files agree with one another, and they do not exactly. Sony writes the presets per camera, so files of one model from different cameras (or firmware versions) differ by a few percent in red or blue. The files of the bodies with curves of their own show how much, as their largest distance from the median in red / blue: the A7 III's eight local files 4.7 % / 3.5 %, the A6400's nine 4.1 % / 3.1 %, the A7 IV's five 0.7 % / 3.7 %, the A7C II's four 2.6 % / 3.6 %. That is the ordinary unit-to-unit spread of the presets, not a sign that they are unusable, and the median over several units is a better estimate of a model's average white than one unit. A second A1, another photographer's camera, lies 1.2 % / 6.1 % from the owner's. So, besides the decision rule, a body's scale is fitted by these rules (`training/white_scale/fit.py`: `AGREE_LIMIT`, `MIN_FILES`, `OUTLIER_MIN_FILES`):

- Every file is fitted on its own, and the files of a body must **agree**: each file's scale lies within 5 % of the median of the files in red and in blue, the unit-to-unit spread above (4.7 % at most). 5 % in red or blue moves a Kelvin reading by 7 to 9 mired (1 % by 1.4 to 1.7) and the white by 0.7 to 1.3 degrees, against the decision rule's 2 degrees and 300 K and the 30 mired by which Sony's Kelvin labels sit off the published curves (below). The limit is the measured bodies' spread rounded up to a whole percent; it was not moved to include or exclude any body.
- The scale is the **median** of the scales of the files kept (rounded to four decimals). With the median over several units a stand-in's scale is a model value, not one camera's.
- Two files agree when they are within 10 % of each other (each within 5 % of their median, which is their mean); that is the widest agreement the rule accepts, and the A6500's two files differ by 8.1 % in red.
- **Only files whose licence allows any use count for a shipped scale.** A Play Raw file whose post states a NonCommercial licence or none is not pinned and not used, and neither are the local-eval files (unclear or non-commercial licences). A body with no such file keeps `[1, 1, 1]`: five bodies that were fitted on one local-eval file each (A300, A500, A5000, NEX-5N, NEX-7) are listed with `none` and the note "no file of a licence that allows any use; scale dropped 2026-10-24" (below).
- With three files or more, if one file is farther than 5 % from the median and leaving out the file farthest from it makes the rest agree, that file is an **outlier**: it is left out, the scale is the median of the others, and the entry's note names it. With two files that disagree, or when more than one file is off, nothing says which is right, and the body is not fitted: it has no registry entry, or `none`.
- A body with a **single file** is fitted if that file passes the decision rule (`MIN_FILES` is 1): one unit that passes is better than none. The entry's note says that one file cannot show the unit-to-unit variation, which is 4 to 6 mired in the Kelvin reading for a 3 to 4 % spread (the decision of 2026-10-24 for the pre-release, to be revisited when more files are pinned).
- The decision rule (2 degrees, 5000 to 5600 K) must hold for every file kept after the median scale is applied; a body for which it does not hold for some file is not fitted, and the note of its `none` entry says how far it misses. No fitted body misses it; the A230's one file does (it has no scale).
- The entry records the largest distance of a kept file from the median (`file_spread_percent`, at most 5).

**Measured.** Mean angle and Daylight reading before and after the preset scale, and the matrix reference for comparison, for every stand-in body with files on disk. **Bold** rows are fitted and shipped; they name the sets their files come from (below). Rows marked local are files that no fetch script pins, reported and not shipped; the five *local-eval file* rows are fits on the one file of the local-eval set (`tools/fetch/real-eval.sh`) that were **dropped** because that set's licences are unclear or non-commercial (below). *Files* is the files kept, of the files found when one was left out as an outlier; *spread* is the largest distance of a file's scale from the median, red / blue, in percent, over the files kept (in brackets, over all files found). The scale in brackets of a body that was not fitted is the median of all its files, to show what it would have been; its angles and readings use it. The A7 III and A7 IV rows are bodies with curves of their own, from their one local file each (`ILCE-7M3.ARW`, `ILCE-7M4.ARW`), for the claim about Sony's labels below.

| Body (sets) | Files | Spread r / b % | Presets: scale `[r, b]` | Mean angle before to after | Daylight K before to after | Matrix: scale `[r, b]` | angle | Daylight K | Result |
|---|---|---|---|---|---|---|---|---|---|
| **A1** (owner-photos) | 3 of 4 | 0.0 / 0.0 (1.2 / 6.1 with all 4) | **0.8675, 1.0861** | 3.54 to 0.54 | 4433 to 5212 | 0.896, 0.991 | 2.07 | 4720 | fitted |
| **A7** (play-raw) | 4 | 3.0 / 0.9 | **0.8769, 0.9553** | 2.87 to 0.76 | 5172 to 5455 | 0.926, 0.909 | 1.51 | 5017 | fitted |
| **A6000** (play-raw) | 8 (10 local, 2 not used) | 4.3 / 3.9 | **1.0019, 1.115** | 2.41 to 0.72 | 4893 to 5375 | 0.973, 0.989 | 2.66 | 4939 | fitted |
| **A6300** (play-raw) | 5 (6 local, 1 not used) | 3.6 / 2.7 (5.2 / 2.2 with all 6) | **1.0065, 1.1698** | 3.27 to 1.04 | 4735 to 5390 | 1.014, 1.102 | 1.34 | 5081 | fitted |
| **A6500** (play-raw) | 2 (3 local, 1 not used) | 4.0 / 0.5 | **0.9714, 1.1687** | 3.36 to 1.20 | 4661 to 5434 | 1.014, 1.102 | 1.64 | 4998 | fitted |
| **ZV-E10** (play-raw) | 3 | 0.8 / 1.5 | **1.0111, 1.2103** | 3.78 to 0.63 | 4517 to 5236 | none | none | none | fitted |
| **NEX-5R** (play-raw) | 2 | 0.0 / 0.0 | **0.9974, 1.0016** | 1.42 to 1.43 | 5523 to 5545 | 0.947, 0.994 | 1.42 | 5724 | fitted |
| **A7R** (play-raw) | 1 | 0.0 / 0.0 | **0.9208, 1.0028** | 1.80 to 0.76 | 5091 to 5417 | 0.931, 0.938 | 1.40 | 5060 | fitted |
| **A7S** (play-raw) | 1 | 0.0 / 0.0 | **0.9457, 1.0407** | 1.47 to 0.87 | 4831 to 5196 | 0.928, 1.047 | 0.95 | 5293 | fitted |
| **A6700** (play-raw) | 1 | 0.0 / 0.0 | **0.9838, 1.0194** | 0.59 to 0.60 | 5089 to 5236 | none | none | none | fitted |
| **RX100** (play-raw) | 1 | 0.0 / 0.0 | **0.778, 1.4495** | 9.89 to 0.57 | 3450 to 5380 | none | none | none | fitted |
| **RX100 II** (play-raw) | 1 | 0.0 / 0.0 | **1.0145, 1.341** | 5.83 to 1.03 | 4350 to 5461 | none | none | none | fitted |
| **RX100 VII** (play-raw) | 1 | 0.0 / 0.0 | **0.8139, 1.0925** | 5.16 to 0.63 | 4000 to 5349 | none | none | none | fitted |
| **RX10 IV** (play-raw) | 1 | 0.0 / 0.0 | **0.8349, 1.2195** | 6.10 to 0.39 | 3990 to 5350 | none | none | none | fitted |
| A7 II (local) | 6 | 5.9 / 10.4 | (0.889, 0.981) | 2.76 to 1.24 | 4992 to 5337 | | | | not fitted: files disagree |
| A230 (local) | 1 | 0.0 / 0.0 | (0.618, 1.169) | 11.55 to 2.58 | 3638 to 5691 | | | | not fitted: misses the limits |
| A5000 (local-eval file) | 1 | 0.0 / 0.0 | (1.0058, 1.112) | 2.20 to 1.30 | 5027 to 5509 | 0.973, 0.989 | 2.35 | 5077 | not shipped: the file's licence, scale dropped |
| NEX-5N (local-eval file) | 1 | 0.0 / 0.0 | (0.9684, 1.008) | 0.73 to 0.43 | 5196 to 5355 | 0.973, 0.989 | 0.46 | 5245 | not shipped: the file's licence, scale dropped |
| NEX-7 (local-eval file) | 1 | 0.0 / 0.0 | (0.9623, 0.9851) | 1.06 to 0.53 | 5290 to 5362 | 1.002, 0.916 | 2.03 | 4890 | not shipped: the file's licence, scale dropped |
| A300 (local-eval file) | 1 | 0.0 / 0.0 | (0.6418, 1.1858) | 10.87 to 1.75 | 3629 to 5573 | 0.683, 1.026 | 3.33 | 4704 | not shipped: the file's licence, scale dropped |
| A500 (local-eval file) | 1 | 0.0 / 0.0 | (0.8365, 0.8628) | 4.73 to 0.69 | 5412 to 5371 | 0.840, 0.875 | 0.81 | 5423 | not shipped: the file's licence, scale dropped |
| A7 III (local, **measured body**, own curves) | 1 | | 0.923, 1.078 | 2.52 to 0.77 | 4783 to 5420 | 0.965, 1.007 | 1.85 | 4932 | reference only |
| A7 IV (local, **measured body**, own curves) | 1 | | 0.930, 1.122 | 3.03 to 0.46 | 4533 to 5239 | 0.938, 1.047 | 1.52 | 4905 | reference only |

Seven of the fourteen fitted bodies rest on one file (the A7R, A7S, A6700, RX100, RX100 II, RX100 VII, RX10 IV); the A6000 rests on eight files, the A6300 on five, the A7 on four, the A1 and the ZV-E10 on three, the NEX-5R and the A6500 on two. The A6700 and the NEX-5R already read on the A7R IV's locus: their scales are 1.6 % and 1.9 % (A6700) and 0.3 % and 0.2 % (NEX-5R) from 1, and move the Daylight reading by 147 K and 22 K. **Under the 1 % limit of the first version** the A6000, A6300, A6500 and A7 II were not fitted; **under the 5 % limit** the A6000 (10 local files in 8 sets of presets, 3.3 % / 3.9 %) agrees; its median is taken over the eight files that may be used, because two are not pinned for their licence (`35354_DSC00912.ARW` is NonCommercial, `12024__DSC1085.ARW` states none; with all ten the scale would be `[1.0113, 1.115]`, 0.9 % further in red). The A6300 has six local files in 4 sets; the five Play Raw files agree within 3.6 % / 2.7 % and give the scale, and the sixth, a cRAW file of the local-eval set (`ILCE-6300_craw.ARW`, -5.2 % in red), is not used for its licence (it would be an outlier). The A6500 (3 local files) has one NonCommercial file (`48272_DSC03902.ARW`, CC BY-NC-SA 4.0) that is not used; its other two files, 8.1 % apart in red and each 4.0 % from their median, agree and give the scale (with all three the scale would be `[1.0018, 1.1981]`, one outlier left out). The A7 II still does not agree: its six files lie 5.9 % / 10.4 % from the median, the one file that is farthest (`43154_DSC04956.ARW`, +10.4 % in blue) leaves the other five up to 6.1 % from their median in red, so it is not fitted. The A230's one file would read the Daylight preset at 5691 K and leave a mean angle of 2.58 degrees, outside both limits. The matrix reference could not be computed for the ZV-E10, the A6700, the three RX100s and the RX10 IV: they have no published matrix (`data/cameras/color-matrices.json`), and their entries leave the three matrix fields out.

**Effect on Auto** (`ym info` of a debug build on the 35 local files of the thirteen new bodies that have a scale, with the registry without their scales and with them, including the four local files that are not used for a scale; Auto's Kelvin and tint against the camera's own as-shot white read on the same locus with the same scale). The tint gap between Auto and the camera's own white balance falls from 5.7 to 3.4 on average (median 4.6 to 1.6), better on 26 files and worse on 7; on the 31 files that count, from 5.7 to 3.1 (median 4.6 to 1.2, better on 24, worse on 5). Auto's Kelvin, averaged over the files, moves from 4941 to 5663 K, and a daylight photo reads where the camera put it. The Kelvin gap in mired does not improve (mean 7.3 to 9.0, median 3.9 to 5.4, better on 8 files and worse on 21): before, Auto and the camera's own white balance shared the offset of the locus, so they agreed with each other although both read the light too warm or too cool; with the scale both move to the right reading, and each is rounded to 50 K (up to 4 mired at 5000 K). By group:

| Files | Tint gap (mean abs) | Kelvin gap in mired (mean abs) | Auto Kelvin (mean) |
|---|---|---|---|
| the 16 files of the A7 (4), ZV-E10 (3), NEX-5R (2), A7R, A7S, A6700, RX100, RX100 II, RX100 VII, RX10 IV | 4.9 to 2.6 (better 11, worse 3) | 8.6 to 8.7 | 4762 to 5366 K |
| A6000 (10), A6300 (6), A6500 (3) | 6.4 to 4.1 (median 4.6 to 1.6; better 15, worse 4) | 6.2 to 9.3 | 5092 to 5913 K |
| of those, the 15 files the scales were fitted on | 6.6 to 3.6 (better 13, worse 2) | 6.7 to 10.2 | 5073 to 5870 K |
| of those, the 4 files not used (the A6300 cRAW and the 3 files left out for their licence) | 5.6 to 5.9 (better 2, worse 2) | 4.4 to 6.0 | 5162 to 6075 K |

Per body, tint gap before to after: A6000 8.4 to 4.8 (all ten files improve), A6300 3.8 to 2.2, A6500 4.8 to 5.5 (the one body that does not improve: of its three files one improves, 5.9 to 1.8, and two get worse, 0.6 to 2.6 and the unused file 8.0 to 12.1; it is fitted on two files 8.1 % apart), A7 4.8 to 1.7, ZV-E10 8.9 to 2.2, and the single-file bodies as in the table; the RX100 gets worse (5.3 to 12.6, its Kelvin now agrees). The A1 is unchanged, and so are the five bodies whose scale was dropped (A300, A500, A5000, NEX-5N, NEX-7; their five files give the same readings with and without). The hue gap to the camera JPEG on the A6000 portraits `p14077` and `p26551` (the earlier investigation, with a scale of `[1.0113, 1.115]` that is 0.9 % from today's in red, before the display change of pipeline 0.6.0; its helper scripts are no longer in the scratch folder, so it was not re-run): 3.2 to 2.1 degrees on `p14077` (lips 1.6 to 1.1, cheek 7.0 to 1.3, chin 0.9 to 4.0) and 3.9 to 2.4 on `p26551` (raspberry shirt 3.6 to 4.3, skin 4.1 to 0.4). With the first version of the rule (1 %), a survey that gave the A7 II and the A6500 one scale from the median of their files made the tint gap of the A7 II worse (3.5 to 4.3) and of the A6500 (4.8 to 5.6), and the mired gap of the A6000 (6.2 to 10.2) and A6300 (5.9 to 10.8) (`research/dynamic-vs-lut-2026-10-14.md`, Stage 1, addendum): the same effect is in the table above, which the 5 % limit accepts; the A7 II stays out.

With the scale, the three A1 photos read 5200, 5250 and 5200 K (4450, 4400 and 4450 K without it), tint −2.4 to 0.2 (−6.5 to −1.6), and the light confidence rises (0.73 to 0.81, 0.82 to 0.90, 0.81 to 0.87).

**Limits of the two references.**

- They differ by 4 to 7% in blue and 3 to 4% in red on the A1, A7 III and A7 IV (`research/dynamic-vs-lut-2026-10-14.md` section 1.9 check 3); on the A1 the presets put the Daylight preset 490 K above where the matrix does.
- The presets are a property of the camera unit, so a scale fitted on one unit is a scale for the model only up to the variation between units (*Consistency of a body's files*). The A1's Play Raw file (`60282_DSC00636.ARW`, another photographer's camera) lies 1.2 % from the owner's three in red and 6.1 % in blue, and was left out as the outlier; the A1's scale is the owner's camera's. The agreement rule makes the scale robust against a file with unusual presets; it cannot make one camera stand for all of them.
- The matrix reference is a usable estimate of a body's raw white, about as good as the curves themselves. Treating each of the ten other measured bodies as a stand-in on the A7R IV profile, its scale from the published curves is known (`locus(6504 K)` of the two profiles), and the matrix reference reproduces it to 0.6 to 3.0 degrees (mean 1.5; A7 III 0.58, A7 IV 1.67, A7C II 1.56, A7R II 1.23, A7R III 1.15, A7S II 1.63, A7S III 1.10, A9 2.00, A6400 1.10, RX100 IV 3.02). Curves and matrix differ by 3.5% in the A7 III's D65 red and 6.2% in the A7 IV's.
- Sony's Kelvin labels sit about 30 mired warmer than the published curves put them, on both measured bodies that have the curves and the presets: on their own curves the A7 III's Daylight preset reads 4783 K and the A7 IV's 4533 K (the 2500, 3200, 6000 and 8500 K presets read +16 to +40 mired on the A7 III and +31 to +34 on the A7 IV; the 4500 K preset reads on the 4000 K junction of the locus and gives no number). Fitting a stand-in's presets to their labels therefore also moves its daylight photos about 30 mired cooler than the same photos read on a measured body: an A1 daylight photo reads 5200 K where an A7 IV photo reads 4650 K. The rendered colour moves little, because the white scale moves the light class, the mired of the sensor map and the ARRI matrix and not the neutral, which Auto balances exactly (0.22 to 0.31 median ΔE00 on the three A1 photos, `research/dynamic-vs-lut-2026-10-14.md` stage 1). A body with its own curves is `[1, 1, 1]` by rule.
- Matrix-implied scales for the stand-ins without data are not applied: A7R V and A7CR `[0.892, 1.099]` (3.2 degrees from the A7R IV), A9 III `[0.780, 1.061]` (3.8), A7 V `[0.925, 0.909]` (2.6), A7R VI `[0.931, 0.938]` (1.8). Apart from the A9 III these are the size of the matrix reference's own error.

**Provenance per body** (registry `white_scale_source`, schema `registry.schema.json`; the table is generated by `python3 -m training.white_scale.fit --write` into `training/white_scale/white_scale.json`, and `training.sensor_map.build_registry` copies it, registry 1.3.0):

| Kind | Bodies | Scale |
|---|---|---|
| `own_curves` | the 11 measured bodies | `[1, 1, 1]` |
| `same_sensor` | the 10 sensor-family bodies (the colour-matrix check above) | `[1, 1, 1]` |
| `presets` | 14 bodies: the A1 (ILCE-1), A7 (ILCE-7), A6000 (ILCE-6000), A6300 (ILCE-6300), A6500 (ILCE-6500), A7R (ILCE-7R), A7S (ILCE-7S), A6700 (ILCE-6700), ZV-E10, NEX-5R, RX100 (DSC-RX100), RX100 II (DSC-RX100M2), RX100 VII (DSC-RX100M7) and RX10 IV (DSC-RX10M4). The entry names every file the presets were read from with its SHA-256 and its set, the residuals of the fit, the largest distance of a file from the median (`file_spread_percent`) and a note that names the file left out as an outlier, if any, and says when only one file was available | the table above; the A1's is `[0.8675, 1, 1.0861]` |
| `sibling` | A1 II (ILCE-1M2): no file of its own; its published colour matrix is the A1's (largest per-coefficient difference after row normalisation 0.0000, the limit of the sibling check is 0.02) | the A1's |
| `none` | A7R V, A7CR, A9 III, A7 V, A7R VI: no file with presets and no sibling with data. A300, A500, A5000, NEX-5N, NEX-7: no file of a licence that allows any use (their one local-eval file is not used; note "scale dropped 2026-10-24") | `[1, 1, 1]` |

The eighteen stand-in bodies of registry 1.3.0 are listed in the table "Which profile each camera gets" (match `default`, the A7R IV profile) to carry their scale (thirteen) or their name (the five with `none`); their model IDs and Model strings were read from the same files. A stand-in that is not listed has `[1, 1, 1]`, like the A7 II, whose files disagree, and the A230, which misses the limits.

**The files, and how to get them.** A shipped scale can be reproduced: every file it was fitted on belongs to a set that a script of the repository fetches and verifies, and a file that no script fetches is never the only source of a scale. `fit.py` finds the pinned files by their content in `--photos` folders (by SHA-256), so a file's name or folder does not matter. Both sets are for evaluation: nothing from them is committed, shipped or redistributed, and the scales take only numbers from them, the presets the camera wrote (`docs/14` rule 8).

| Set | Fetch script (pins) | Files used | Licence |
|---|---|---|---|
| `owner-photos` | `tools/fetch/owner-photos.sh` (SHA-256 in `owner-photos.sha256`) | the seven A1 photos | © Mykyta Griaznov; he permits them as test and evaluation data (`docs/test-photo-sources.md`) |
| `play-raw` | `tools/fetch/white-scale-photos.sh` (SHA-256, size and address in `white-scale-photos.json`) | 32 Play Raw files of discuss.pixls.us: 4 A7, 8 A6000, 5 A6300, 2 A6500, 3 ZV-E10, 2 NEX-5R, one each of A7R, A7S, A6700, RX100, RX100 II, RX100 VII, RX10 IV, and the A1 file left out | the posts state "Creative Commons, By-Attribution, Share-Alike" (the Play Raw template, no version), CC BY-SA 4.0 or CC0, "Public Domain" for the RX100 |

**Licence rule (decision of 2026-10-24, conservative):** a shipped scale rests only on files whose licence allows any use. A Play Raw file whose post states a NonCommercial licence or none is not pinned and not used; the local-eval files (`tools/fetch/real-eval.sh`, unclear or non-commercial licences, `docs/validation-data.md`) are not used either, and the set is not a set a registry entry can name. Three Play Raw files were left out under the rule: `35354_DSC00912.ARW` (an A6000 file, NonCommercial-ShareAlike), `12024__DSC1085.ARW` (an A6000 file whose post states no licence) and `48272_DSC03902.ARW` (an A6500 file, CC BY-NC-SA 4.0); and five scales fitted on one local-eval file each were dropped: the A300 (a CC BY-NC-SA file), A500, A5000, NEX-5N and NEX-7 (unclear licences). The local-eval files stay available for looking at real photographs and for `--report`; with them the five scales would be A300 `[0.6418, 1.1858]`, A500 `[0.8365, 0.8628]`, A5000 `[1.0058, 1.112]`, NEX-5N `[0.9684, 1.008]`, NEX-7 `[0.9623, 0.9851]` (rows above). Replacing them needs a CC0 or CC BY file of the same body.

The address in `white-scale-photos.json` is the one the post's attachment link leads to on the forum's file store; its file name is the SHA-1 of the file, so it cannot change under the pin (the 32 sizes and SHA-1s were checked against the pinned copies without downloading them, and two files were downloaded again and matched their SHA-256).

To fit a body or add files: pin the files in a set (a Play Raw file goes into `white-scale-photos.json`), add a new body to `training/sensor_map/registry_data.py` and to the table above, fetch the sets, run `python3 -m training.white_scale.fit --write` and `python3 -m training.sensor_map.build_registry`. The fit reads `testdata/readme-photos/src` and `testdata/white-scale/play-raw`; `--photos DIR` (repeatable) adds folders. `--write` and `--check` refuse to run while a pinned file is missing; `--check` compares a fresh fit with the committed table. `--report` prints the table above for every ARW it finds, pinned or not, with the consistency verdict of each body (`single`, `agree`, `outlier` or `disagree`). The A7 II and A230 rows come from Play Raw files of `testdata/lens-photos` and `testdata/cropeval/playraw` that no script pins, and the five dropped rows from `testdata/real/files` (`--photos` those folders).

**Synthetic scenes of a stand-in** are rendered through the stand-in's own curves, so the A1's raw would read as a measurement error against the scale. `training/synth` therefore writes a body's raw as the render divided by its registry `white_scale` (red and blue) and its as-shot gains multiplied by it; the sidecar records `camera_white_scale`, the truth stays on the profile's scale (`w_true_camera_rgb`), and the recipe hash carries the scale only for a body that has one, so the files of the other bodies are untouched. With `tools/fetch/synthetic.sh --only ILCE-1` the A1's synthetic scenes read their lights as the plain profile would (6450, 4050 and 2850 K for 6500, 4102 and 2850 K), and the end-to-end colour error of `golden truth` equals the A7R IV's on the same light to within 0.05 ΔE00.

## Why measured profiles matter

From `research/alexa-ssf/out/sony_profiles.csv`: ΔE00 to the ALEXA rendering, median / p95 over 138 colours under 9 lights, using each body's own profile versus the A7R IV profile. The numbers are those of the ALEXA estimate with its hue push of 4.5 degrees (below); the A7R IV itself reads 0.53 / 1.58. Before the pushes (the estimate of 2026-10-11) it read 0.34 / 0.87, with the push of 1.5 degrees (2026-10-25) 0.44 / 1.17. Every own median reads 0.15 to 0.27 higher than before the pushes, except the RX100 IV's, which reads 0.28 / 0.84 against 0.31 / 1.05.

| Camera | Own | A7R IV |
|---|---|---|
| A7 IV | 0.51 / 1.51 | 0.55 / 1.53 |
| A7C II | 0.53 / 1.54 | 0.57 / 1.48 |
| A7R III | 0.52 / 1.49 | 0.54 / 1.60 |
| A7 III | 0.51 / 1.55 | 0.64 / 1.68 |
| A7R II | 0.51 / 1.48 | 0.70 / 1.82 |
| A7S III | 0.54 / 1.66 | 0.87 / 1.97 |
| A9 | 0.52 / 1.49 | 0.79 / 1.81 |
| A6400 | 0.50 / 1.45 | 1.34 / 3.05 |
| RX100 IV | 0.28 / 0.84 | 1.69 / 3.89 |
| A7S II | 0.73 / 2.15 | 2.69 / 4.52 |

Stand-ins come only from shared sensors, and estimated Sony curves are not shipped: curves estimated from two published colour matrices were worse than the A7R IV stand-in for 8 of 10 bodies.

## The ALEXA estimate

The target of every sensor map is an **estimate** of the ALEXA ALEV III's spectral sensitivities (`data/alexa/`, `research/alexa-ssf/`, `training/alexa/`). It is found as the set of curves whose own best Academy P-2013-001-style matrices equal ARRI's published native → AP0 matrices at 17 colour temperatures, inside the span of a 4-component PCA prior of the 52 measured camera curves of rawtoaces-data. Seventeen matrices do not fix such a curve to better than the spread of its 24-variant ensemble (other priors, component counts, penalties and illuminant models): all 24 reproduce the matrices to 0.0064 to 0.0093 RMS per coefficient, and they differ from the recommended curve by 1.7 % RFSE in the median and 2.6 % at most.

**The hue push (owner decisions 2026-10-25, "Move ALEXA estimate", and 2026-10-26, "4.5°"; target 1.2.0, profiles 1.2.0).** Yellowmoon's pictures of ochre walls, oranges and reds came out more crimson than the camera's own picture (Neutral against the camera's Before view, on warm patches: the owner's A1 photos a median of −5.5 degrees of CIELAB hue, a photo of a body with its own curves −8.3; the investigation of 2026-10-25 measured −4.8 and −8.9 on 25 and 15 patches). The ALEXA curve is a place where a part of that gap can be closed and where a choice is open: the family of curves that agree with ARRI's matrices contains curves that render warm colours more than a degree differently. The shipped curve is the one that renders the orange, red and ochre training reflectances (47 of the 190: Lab hue 25 to 95 degrees and chroma above 12 under the un-pushed estimate) **4.5 degrees** of hue toward yellow at 5200 K through ARRI's matrix.

*Two decisions, and the second goes past the matrix limit.* On 2026-10-25 the owner chose the push of 1.5 degrees, the largest 0.5 degree step whose matrix RMS stays within the ensemble's maximum. The owner still saw oranges and ochres too red, was shown sheets of 3 and 4.5 degrees with their costs, and on 2026-10-26 chose **4.5 degrees**. That is past the ensemble's limit: ARRI's published matrices no longer agree with the curve as well as they agree with the 24 variants. It is a documented decision, not a fit that ARRI's matrices ask for, and it is recorded where the script reads it: `training/alexa/push_decisions.py` holds, for the push of 4.5 degrees, a `beyond_matrix_limit` record (the decision, and the matrix RMS it accepted, 0.0241). `estimate_alexa` writes the record into the curve's description (`data/alexa/alexa_alev3_estimate_380_780_5.json`) and into `diagnostics.json`, `build_target` into `data/alexa/target.json` (`provenance.method` and `caveats`, which say "beyond_matrix_limit"), and `training/alexa/test_push.py` checks them. The stopping rule is still enforced: a push whose matrix RMS exceeds the ensemble's maximum is written only when it has such a record and reaches no more than the RMS the record names, so any other push past the limit stops the run with nothing written, and so does 4.5 degrees if it ever came out larger than 0.0241.

| Push (degrees) | 0 | 0.5 | 1.0 | 1.5 | 2.0 | 3.0 | **4.5** |
|---|---|---|---|---|---|---|---|
| Matrix RMS per coefficient | 0.0067 | 0.0070 | 0.0081 | 0.0092 | 0.0112 | 0.0166 | **0.0241** |

(The ensemble spans 0.0064 to 0.0093; the 52 held-out cameras have a median of 0.0021 and a 90th percentile of 0.0051; 0.0067 and 0.0092 were their 94th percentile, 0.0241 is their 96th percentile, with only the ARRI D-21 and the Canon EOS R10 above it, and 2.6 times the largest of the ensemble.) `python3 -m training.alexa.estimate_alexa` makes the shipped files, seeded and offline: the 24 variants, then the recommended variant with one more residual, `10 · (median hue shift of the 47 reflectances − 4.5)`. `--push-deg 1.5` writes the estimate of 2026-10-25 (to `estimate_push_1.5/`; it reproduces the data of that date) and `--no-push` the estimate of 2026-10-11 (`compare_estimate` reproduces both to 0). `variants.csv` and the band of `alexa_alev3_estimate_380_780_5.csv` are of the 24 variants and so without the push; the shipped curve is outside the band at 146 of 243 samples, by 0.14 of peak at most (B at 415 nm; G at 490 nm 0.11, R at 670 nm 0.03). `python3 -m unittest training.alexa.test_push` checks the committed files against the push's definition and its decision record.

| The curve | Without the push | Push of 1.5 degrees (2026-10-25) | **Shipped, 4.5 degrees** |
|---|---|---|---|
| R at 580 nm (of its peak) | 0.597 | 0.494 | **0.457** |
| R at 650 / 660 / 680 nm | 0.439 / 0.314 / 0.025 | 0.400 / 0.291 / 0.068 | **0.390 / 0.292 / 0.108** |
| R half-power, short / long side | 580.1 / 643.5 nm | 580.1 / 639.0 nm | **580.7 / 637.9 nm** |
| Peaks R / G / B | 590 / 530 / 460 nm | 590 / 530 / 460 nm | 590 / **520** / 460 nm |
| Matrix RMS per coefficient | 0.0067 | 0.0092 | **0.0241** |
| RFSE to the curve without the push | | 2.2 % | **3.7 %** |
| Mean ΔE00 to scene colour with ARRI's matrices (99 CES, 9 lights; the 52 cameras and the estimate; the best of the real cameras is 1.662) | 1.455 (1st) | 1.548 (1st) | **2.018 (14th)** |

(The script's printed list, which also has the estimate without the push and the mean camera in it, puts the shipped curve 16th.) It moves R at 560 to 585 nm (by −0.14 at 580 nm), lowers the red shoulder by 0.02 to 0.05 at 615 to 660 nm and raises R at 665 to 690 nm (by 0.09 at 670 to 680 nm, a part the matrices barely constrain), raises G at 475 to 620 nm (by 0.22 at 490 nm, 0.12 at 580 to 590 nm, and moves G's peak from 530 to 520 nm) and B at 405 to 425 nm (by 0.16 at 415 to 420 nm). Against the curve of 1.5 degrees the largest differences are 0.045 in R (670 nm), 0.098 in G (490 nm) and 0.198 in B (415 nm).

**What it does to pictures** (the sheets of 2026-10-26: real `ym render` pictures of seven photos at 1500 px, the CPU, five A1 photos of the owner, a photo of an A7 IV and a portrait of an A6000, with the push of 1.5 degrees against 4.5): on warm patches (Neutral hue 15 to 100 degrees, C* above 10 at 1.5 degrees) the hue moves a further 1.9 degrees toward yellow in Neutral and 1.8 in Signature (Neutral on the 25 patches of the A1 photos: the gap to Before −5.5 degrees before the first push, −4.1 at 1.5 and −2.0 now; the gap to the camera's JPEG −1.2 and +0.6, to its embedded preview −0.2 and +2.0; Signature −7.1, −5.9 and −3.5 against Before, −3.7 and −1.8 against the JPEG; the A7 IV photo, Neutral, 4 patches: −6.7 now −3.5 against Before). On the A1 photos the warm patches therefore land on the camera's picture to within a degree in Neutral and 1.8 degrees redder in Signature, and on the A7 IV photo and the portrait (14077, skin) they go past it (Neutral +0.5 degrees from the camera's JPEG on the A7 IV photo, +4.4 degrees from it on the portrait's face, +2.3 at 1.5 degrees). Pixel by pixel against the push of 1.5 degrees: skin-like pixels change by ΔE00 0.5 to 0.95 on average (hue +1.4 to +2.6 degrees; the push of 1.5 degrees changed them by 0.36 to 0.59 from the estimate without it), reds by 0.3 to 1.1 (+0.9 to +2.1 degrees), greys by 0.05 to 0.08 (95th percentile at most 0.16) and blues by 0.15 to 0.5. On the left sunlit wall of A1M03159 (Neutral) the hue goes from 63.6 to 65.7 degrees; the camera's JPEG has 65.2 (its embedded preview 63.8) and Before 67.7; the chroma (C* 23 against the JPEG's 32) is not touched. It is a push of 4.5 degrees out of a gap of five to nine, and on the A1 photos the warm patches are no longer redder than the camera's picture.

**What it costs** (the owner was told before choosing, with the numbers of the sheets of 2026-10-26):

- **The curve no longer agrees with ARRI's matrices.** Matrix RMS 0.0241 per coefficient, past the ensemble's 0.0093 (the owner's `beyond_matrix_limit` decision, above); mean ΔE00 to scene colour with ARRI's matrices 2.018 against 1.548 at 1.5 degrees and 1.455 without the push, the 14th place (the best real camera has 1.662). The ARRI matrices applied to this curve do not give true colours as well as they do on most of the 52 cameras.
- **The sensor-map gate fails on more lights.** Thresholds unchanged (median 0.5, p95 1.2). A7R IV: 2 of 9 lights pass (FL3.1 0.402 / 1.188, FL3.8 0.401 / 1.082), where 3 passed at 1.5 degrees and 8 before the pushes; over now are blackbody 2850 K 0.662 / 2.103, D 5250 K 0.715 / 2.010, D 6750 K 0.715 / 1.962, FL3.15 0.687 / 1.878, LED-B3 0.521 / 1.499, LED-RGB1 0.555 / 1.369 and LED-V2 0.806 / 1.894 (at 1.5 degrees six, without LED-RGB1: 0.447 / 1.005; before the pushes only LED-V2, 0.759 / 1.515). Across the 11 bodies 81 of the 99 body-light pairs are over (51 at 1.5 degrees); no body passes all nine lights (none did at 1.5 degrees either; before the first push A7R II, A7R III and A7S III did). The 3×3 fitted at the light itself is over for six of the seven lights (blackbody 0.664 / 2.103, D 5250 K 0.714 / 2.009, D 6750 K 0.715 / 1.962, FL3.15 0.697 / 1.889, LED-B3 0.483 / 1.527, LED-V2 0.637 / 1.862), so it is the white-preserving 3×3's reach and not the table blend; LED-RGB1's own 3×3 passes (0.362 / 1.084), so there the gate refit's table blend also counts. The root-polynomial map of `docs/02` §C2 would now clear only two of the seven (LED-B3 0.423 / 1.166 and LED-RGB1 0.299 / 0.911; blackbody 2850 K 0.533 / 1.585, D 5250 K 0.530 / 1.376, D 6750 K 0.540 / 1.368, FL3.15 0.523 / 1.284 and LED-V2 0.527 / 1.340 stay over; five of six at 1.5 degrees, `training/sensor_map/README.md`). CI expects exactly these seven (`--expect-failing`).
- **The research table of `sony_profiles.csv` reads higher:** the A7R IV's own error is 0.53 / 1.58 against 0.44 / 1.17 at 1.5 degrees and 0.34 / 0.87 before the pushes, the median over the 11 bodies' own profiles 0.52 / 1.51 against 0.41 / 1.14 and 0.34 / 0.87 (above, *Why measured profiles matter*).
- **The leave-one-out validation has no counterpart for a pushed estimator.** A held-out camera's curve would be moved away from its measured one by the same recipe, so `validate_loo` and `check_loo` run the estimator the push starts from and their numbers are unchanged (median RFSE 2.17 %, emulation ΔE00 median 0.233); they are not a validation of the shipped curve. What the push costs on the 52 known cameras (`python3 -m training.alexa.loo_push`, not a gate): RFSE median 2.17 % without the push, 2.44 % at 1.5 degrees and 3.31 % at 4.5 degrees (p90 5.07 %; below the RFSE without the push for only 8 of the 52 cameras at 4.5 degrees), emulation ΔE00 median 0.233, 0.519 and 1.293 (the gate's limit is 0.3). The measured cameras' own warm-colour hue is within 0.5 degrees of their un-pushed estimates for 49 of 52 (median −0.04, 10th to 90th percentile −0.17 to +0.25): the push is a choice about the picture, not a correction the validation can see, and 4.5 degrees is far past the half degree within which those cameras' warm hue lies of their estimates (`research/alexa-ssf/README.md`).
- **Every synthetic golden picture moves** (re-baselined with a note in `tools/golden/expected.json`, `docs/13`; the numbers are in `docs/STATUS.md`).
- **The ColorChecker map error of the chain grows** (`crates/ym-color/tests/chain_spectral.rs`): 24 patches through the sensor map against the simulation, worst of 36 cases, median 0.54 (0.42 at 1.5 degrees), p95 1.73 (1.33) and maximum 2.24 (1.80); the test's bounds were raised with the owner's agreement (2026-10-26) to a p95 of 1.8 and a maximum of 2.3 (1.4 and 1.8 before); the Kelvin-rounding test's 2300 to 4000 K band (`wb.rs`, 1.65 % measured) to 1.7 % (1.4 % before).
- The Kelvin settings and the Auto white move a little: for a D 5500 K scene balanced to 3200 K the AWG4 white is `[0.6821, 0.9451, 1.7137]` against `[0.6789, 0.9478, 1.6629]` at 1.5 degrees and `[0.6747, 0.9488, 1.6434]` before the pushes, and `w_A` steps by red −0.75 % and blue −4.0 % at 4000 K (−1.0 % and −3.6 % at 1.5 degrees).

## Lookup

`exact model ID → sensor family → default (A7R IV)`. A Sony file is never refused for lack of a profile. The CLI's `--profile <id>` forces a profile (for comparing a new one against the default); the app has no override.

In order: (1) a Make other than Sony (a non-empty Make that does not start with "sony", ignoring case; the rebadged Hasselblad bodies) gets the default profile whatever the model ID says; (2) with a model ID, the registry body with that ID, and an ID the registry does not list is a body we have not seen, so it gets the default profile even when the Model string looks familiar (the ID wins); (3) without an ID, the body the Model string names through `model_strings`, compared ignoring case, surrounding whitespace and NULs and a leading "Sony "; (4) anything else gets the default profile. A missing Make counts as Sony.

## Profile files

```
data/cameras/
  registry.json                       model ID → { model, body, profile, match, via, notes, white scale, sensor geometry }; Model-string aliases; default profile
  sony/Sony_<model>_380_780_5.json    measured curves (Weta, Apache-2.0) + LICENSE-rawtoaces-data
  profiles/<profile-id>/profile.json  generated by training/ scripts, versioned
  color-matrices.json                 published D65 colour matrices of 44 bodies; camera-default-tone.json: the camera's tone response (Before view)
  noise/gain-regimes.json             gain-switch table of docs/06; per-body noise priors <model-id>-<model>.json
```

`profile.json` (schemas in `crates/ym-profiles/schema/`: `profile.schema.json`, `registry.schema.json`, `target.schema.json`; checked by `python3 -m training.sensor_map.validate_profiles` and by Rust tests). Abbreviated; `…` marks a table filled by the fit script:

```json
{
  "id": "sony-ilce-7rm4",
  "version": "1.2.0",
  "display_name": "A7R IV",
  "spectral_curves": "sony/Sony_ILCE-7RM4_380_780_5.json",
  "sensor_map": {
    "method": "per-light-3x3",
    "target": "alexa/target.json",
    "fit": { "white_preserving": true, "objective": "cie-lab-after-arri-matrix", "training_reflectances": 190 },
    "classes": {
      "daylight":    [{ "light": "daylight-11000K", "mired": 90.909090909, "m": [[…], […], […]] }, …],
      "tungsten":    [{ "light": "blackbody-4000K", "mired": 250.0, "m": … }, …],
      "fluorescent": [{ "light": "FL7", "mired": 153.97…, "m": … }, …],
      "led":         [{ "light": "LED-B5", "mired": 151.57…, "m": … }, …]
    }
  },
  "locus": {
    "axes": ["ln(R/G)", "ln(B/G)"],
    "interpolation": "linear in mired",
    "blackbody": { "start_k": 1500, "step_k": 10, "values": [[ln R/G, ln B/G], …] },
    "daylight":  { "start_k": 4000, "step_k": 10, "values": […] },
    "xy": { "blackbody": { "start_k": 1500, "step_k": 10, "values": [[x, y], …] },
            "daylight":  { "start_k": 4000, "step_k": 10, "values": […] } },
    "duv_scale": { "blackbody": { "start_k": 1500, "step_k": 100, "values": [s, …] },
                   "daylight":  { "start_k": 4000, "step_k": 100, "values": […] } },
    "duv_sign": "positive toward magenta"
  },
  "alexa_white": {
    "interpolation": "bb/daylight: geometric in mired; fluorescent/led: linear in mired",
    "blackbody": { "start_k": 1500, "step_k": 10, "values": [[r, 1, b], …] },
    "daylight":  { "start_k": 4000, "step_k": 10, "values": […] },
    "fluorescent": [{ "light": "FL7", "mired": 153.97…, "w": [r, 1, b] }, …],
    "led":         [{ "light": "LED-B5", "mired": 151.57…, "w": [r, 1, b] }, …]
  },
  "decode": { "quirks": [] },
  "provenance": {
    "curves": "Weta Digital via rawtoaces-data, Apache-2.0",
    "curves_source": { "repository": "…", "commit": "<rawtoaces-data sha>", "file": "…", "sha256": "…" },
    "target": { "file": "…", "sha256": "…", "note": "…" },
    "fit_script": "training/sensor_map/fit.py@<git sha>",
    "fit_sources_sha256": "…", "colour_science": "0.4.7",
    "sensor_map_objective": "…", "fit_solver": "…", "duv_scale_method": "…", "light_conventions": "…",
    "reference_lights": { "daylight": ["daylight-11000K", …], "tungsten": […], "fluorescent": […], "led": […] },
    "cie_data": "CIE SPDs … used at fit time only … pending HUMAN M1-H4",
    "checks": [{ "pairing": "ILCE-7C (381) uses sony-ilce-7m3", "check": "…", "result": "pass: max |diff| 0.0000 per coefficient (limit 0.02) …" }]
  }
}
```

What the tables hold, and how the runtime reads them (full rules in `training/sensor_map/README.md`; reference implementation `training/sensor_map/runtime_ref.py`):

- **Grids** are `{ start_k, step_k, values }`; `values[i]` belongs to `start_k + i·step_k`; queries clamp to the ends. With `T_i ≤ T ≤ T_(i+1)` the interpolation weight is in **mired**: `a = (1/T − 1/T_i) / (1/T_(i+1) − 1/T_i)`. `w_A` interpolates geometrically (`w_i^(1−a) · w_(i+1)^a`, green stays 1); log-chroma, xy and the Duv scale interpolate linearly. Blackbody tables serve below 4000 K, daylight tables from 4000 K.
- `locus` is the camera RGB (green = 1, **in the camera's raw scale**: the published curves, clipped at 0, not normalised per channel) of blackbody 1500–4000 K and CIE daylight 4000–25000 K every 10 K, as log-chroma, with the matching CIE xy. `duv_scale` is the CIE 1960 uv distance per unit of log-chroma distance across the locus, tabulated every 100 K (it varies by a factor of 5 to 8 over the range depending on the body, so it is a table, not a number). Duv is positive toward magenta, the opposite of CIE's own sign.
- `alexa_white` holds `w_A` for blackbody 1500–25000 K and daylight 4000–25000 K every 10 K (the D series is only defined from 4000 K), and for the fluorescent and LED reference lights as `(mired, w)` lists. `w_A` does not depend on the Sony profile but lives here so a second target (for example a different ALEXA estimate) can be versioned with the profiles that were fitted against it.
- `sensor_map.classes.<class>` lists the maps fitted at that class's reference lights, sorted by strictly increasing mired; every matrix has rows summing to 1 (white preserving). `light` names the reference light (for traceability and to tell the gate which entries to drop).
- `locus` and `alexa_white` are precomputed offline so the app never needs CIE datasets at run time (`docs/02`). The off-grid accuracy of these tables against the spectra is stated in `crates/ym-color/tests/data/spectral_reference.json` (`tolerances`).
- **Which curve scale each table uses** (recorded in `provenance.light_conventions`; `docs/02` C1). `locus` and `duv_scale` come from the curves exactly as Weta published them (`data/cameras/sony/*.json`, clipped at 0, one common factor per camera with the green peak at 1.0, red and blue peaks 0.57 to 0.62 and 0.78 to 0.87 of it), because the scene white at run time is raw data (as-shot gains, Pick white, the illuminant estimator), so a locus point must be the raw white of its light. `sensor_map` and `alexa_white` divide every response by the response to the white of the same curves, so their scale cancels; they are computed from the peak-normalised curves of `ssf_core.load_cameras` (the path the published leave-one-out numbers use). Measured on nine real ARWs in `testdata/real/files`: as-shot Duv −0.0065 (A7 III) and −0.0008 (A7 IV), −0.005 to +0.025 on the seven bodies served by the A7R IV stand-in. The published curves and each camera's own ColorMatrix still differ by 3.5% (A7 III) and 6% (A7 IV) in the D65 white, a bias along the locus of 250 to 650 K and not a tint (`docs/02` C1).

**Runtime** (`ym_color::spectral` reads the grids, the locus and `w_A`: `Grid`, `Locus`, `AlexaWhite`; `SpectralCurves` loads rawtoaces-format curve files for tests and tools. `ym_color::sensor_map` reads the maps: `SensorMapTables`. `ym_profiles::profile::Profile` holds them with the ids, versions and provenance: `Profile::sensor_map`, `scene_white`, `user_white`, `rgb_to_cct_duv`, `cct_duv_to_rgb`.) Where the reading rules above leave a choice, the runtime does this:

- *Class blend of `w_A` (`docs/02` C3).* `scene_white(cct, probs) = Σ_k p_k · w_k(mired)` with the class probabilities normalised to sum 1 and green set to exactly 1. The fluorescent and LED classes interpolate their reference-light lists linearly in mired (clamped at the ends); the daylight and tungsten classes are the scene-light model at `1e6 / mired` K (blackbody below 4000 K, CIE daylight from 4000 K, geometric in mired). A fluorescent or LED class without reference lights falls back to the model.
- *Empty classes in the sensor map.* A class without entries is dropped and the others renormalised: its probability moves to the non-empty classes in proportion to their own; if those have none either, to the non-empty classes equally; with no tables at all the map is the identity. The result is always white preserving.
- *Auto white balance.* Stages C3 and C4 cancel only when `E_user` is the same light as `E_scene`, so Auto takes `g_user = 1 / scene_white(cct, probs)` (the class-blended `w_A`), not `1 / user_white(cct)` (the blackbody/daylight model); otherwise an LED or fluorescent scene gets a residual cast of the difference between its reference-light `w_A` and the model's.
- *Duv sign.* The chord normal is `(t_y, −t_x)/|t|`, flipped when it would point away from magenta (`n_x + n_y < 0`; magenta is `(1, 1)` in log-chroma space, perpendicular to the locus). For every forward chord this is the rule above bit for bit. The CIE D series as colour-science builds it (M1 and M2 rounded to three decimals) is a staircase, and above about 16000 K some chords run backwards; the flip keeps the sign of Duv right everywhere, but the staircase also makes the chord normal noisy, so the *size* of Duv drifts down with temperature. Measured as readout / (tint × Duv scale) for tints of ±0.01 and ±0.05 log-chroma units (Duv of about ±0.001 to ±0.004) on all 11 profiles, magenta tints first and green in brackets:

  | Band | median | worst |
  |---|---|---|
  | 4400 to 11000 K | 1.000 | within 2.6% |
  | 11000 to 14000 K | 0.998 | 0.91 (0.955) |
  | 14000 to 18000 K | 0.95 (0.99) | 0.65 (0.80) |
  | 18000 to 20000 K | 0.93 (1.00) | 0.58 (0.86) |
  | 20000 to 25000 K | 0.66 (0.96) | 0.56 (0.85) |

  The same staircase makes a temperature read back up to 10 K off above 20000 K (two temperatures share a camera RGB); below 16000 K a locus point reads back to 1e-6 K. All of this is above the UI's Kelvin range (2000 to 11000 K); a readout used above 14000 K should treat the tint as approximate.
- *The 4000 K junction.* The blackbody and daylight branches meet at 4000 K but are not collinear: the D series sits about 0.04 log-chroma units (0.011 in ln R/G, 0.039 in ln B/G) on the green side of the Planckian locus there. The readout takes the branch with the smaller distance, so a light between the two branches can be read from the other one, and a tinted light whose perpendicular foot would lie beyond the end of the branch it is nearest to is clamped to that end and projected on the end chord alone. Measured on the A7R IV profile (the other ten show the same pattern), as readout Duv / (tint × scale) and the CCT error: a 1.5% green excess on a 3894 K light reads as 4000 K with Duv +0.0009 instead of a CCT near 3890 K and a Duv near −0.0017; a magenta tint of 0.03 log-chroma units (Duv about +0.0025) on a 4100 K light reads 4000 K with a Duv of −0.14 times the tint (wrong sign), and 0.05 on a 4200 K light 0.29 times with the CCT 200 K low; green tints on 3800 to 3950 K lights read −0.1 times (0.03, wrong sign) or 0.33 to 0.37 times (0.05) with the CCT 80 to 200 K high. Tints of 0.01 (Duv about 0.0008) and anything below 3700 K or above 4300 K (checked up to 0.05) read correctly. CCT and Duv are continuous within a branch, not across 4000 K, and in the band of about 3800 to 4300 K a Duv beyond ±0.002 is not trustworthy, so the illuminant estimator and the tint readout do not rely on Duv being continuous or exact there. This is a property of the specified readout (nearest chord of two tables that meet at 4000 K), not of the port: `runtime_ref.py` gives the same numbers, and `tests/spectral.rs::the_4000_k_junction_is_documented_behaviour` pins the 3894 K and 4100 K examples.
- *`w_A` at 4000 K.* The same rule makes `w_A` step at 4000 K: the blackbody table gives `[0.7982, 1, 0.6286]` at 4000 K and the daylight table `[0.7922, 1, 0.6032]` (red 0.8% and blue 4.0% lower; the ALEXA estimate is the same for every profile). `user_white` and the daylight and tungsten classes of `scene_white` therefore jump by that much when the CCT crosses 4000 K, which a Kelvin slider dragged across 4000 K shows as a small step in blue (about 0.05 stop). It is the rawtoaces convention the ALEXA curve was fitted with.
- *Validation.* Loading checks what the schema cannot express (`Profile::validate`, the Rust twin of `validate_profiles.py` without the file hashes): grid ranges and steps, green exactly 1 in every `w_A`, strictly increasing mired in every class list, rows of every matrix summing to 1 within 1e-12, reference-light names equal to the sensor-map entries and mired of formula lights equal to `1e6 / T`, the locus running the right way, `decode.quirks` empty. A profile that fails is refused with its id in the message.

`registry.json` has `default_profile`, `profiles` (id → file), `bodies` (one entry per model ID in the table above: `model_id`, `model`, `body`, `profile`, `match` = `exact` | `sensor_family` | `default`, `via` = the measured body for `sensor_family` and `null` otherwise, `notes`, the sensor geometry fields, and `white_scale` with `white_scale_source`, registry 1.3.0) and `model_strings` (IFD0 Model string → model ID, for files without a usable 0xB001). A model ID that is not in `bodies` gets `default_profile` with `match = default`.

Noise priors are not part of a colour profile: they belong to a body, not a sensor family. They live in `data/cameras/noise/`, keyed by model ID (for example `384-ilce-1.json`), fitted from public files, with `data/cameras/noise/gain-regimes.json` holding the gain-switch table from `docs/06`. No per-body file exists yet; a body without its own file gets the class prior (`docs/06` Priors).

The ALEXA target uses the same structure (`data/alexa/target.json`: `id`, `version`, `kind` = estimate, `spectral_curves`, provenance with method, inputs, licence, published validation numbers, caveats and file hashes). A second target would slot in the same way.

## Per-camera code hook

```rust
pub trait CameraQuirks: Send + Sync {
    fn adjust_metadata(&self, meta: &mut RawMetadata) {}
    fn crop(&self, meta: &RawMetadata) -> Option<Rect> { None }
    fn black_level_override(&self, meta: &RawMetadata) -> Option<[u16; 4]> { None }
    fn star_eater_risk(&self, meta: &RawMetadata) -> bool { false }
}
```

Registered by model ID in `ym-raw`. Code changes only when a body has new decoding quirks. Profiles carry everything else as data.

## Traceability

The profile id and version are written into every export's XMP (`docs/09-export.md`). If a profile changes, `pipeline_version` does not need to change, but the profile version must.

## Adding a camera later

In V1, a camera gets its own profile only if permissively licensed measurements are published (check rawtoaces-data and similar sources at each release). Our own measurements start in V2:

1. [HUMAN, V2] Measure spectral curves (monochromator or narrowband filters), or obtain permissively licensed measurements.
2. [HUMAN, V2] Shoot noise frames (dark and flat, per ISO) and a test set (colour chart under daylight, tungsten and LED).
3. Run the fit scripts; add `profile.json` and a registry entry; run the profile gates (`docs/13-testing.md`).
4. Ship. The A1 is first, in V2.

## Tests

- Every Sony sample in the test corpus (`tools/fetch/samples.sh`) is detected as the right body and gets the profile in the table; the fallback is reported for the rest.
- Registry round trip: every model ID maps to exactly one profile; every profile file validates against the schema and loads.
- The 11 exact profiles reproduce `sony_profiles.csv` "own" numbers within ±0.02 ΔE00 when evaluated by the Rust code (a port of the Python check).

`crates/ym-profiles/tests/`:

- `data_files.rs`: every `profile.json`, the registry and `target.json` validate against `schema/*.schema.json` with the `jsonschema` crate (a dev-dependency only, MIT; the HTTP and file reference resolvers are off), the schemas reject sixteen broken documents, the typed Rust values write back documents the schemas accept, and the registry round-trips (model IDs unique, each maps to one profile that exists, loads and has the id the registry gives it, exactly one exact body per profile, defaults present). The registry equals the table in this file (parsed from the Markdown); the curve and target files match the SHA-256 in `provenance`; each of the 10 same-sensor pairings has its `provenance.checks` entry with a numeric result within 0.02, and the two rejected pairings are recorded as failed; loading reports a missing, truncated, non-white-preserving or mislabelled file (profile, registry or target) with the file or profile named. For the white scale it checks every body's scale and source (the rules above), that each of the 14 fitted bodies meets the rules (files agree within 5 %, mean angle at most 2 degrees, Daylight between 5000 and 5600 K) and that every file it names is pinned by its set's fetch script (`owner-photos.sha256`, `white-scale-photos.json`), the A1's residuals against the limits and its left-out file, the A1 II as the A1's sibling, the five bodies whose scale was dropped for the licence of their file (`none`, `[1, 1, 1]`, the note), that the six bodies without a published matrix leave the matrix fields out, and that the schema and the typed rules refuse twenty-eight broken scales and sources (a file from the local-eval set among them).
- `detect.rs`: all 46 bodies by model ID (with a wrong Model string), by Model-string alias (case, spacing, NULs, "Sony " prefix), the ID-wins rule, unknown Sony bodies, non-Sony makes, `--profile`, odd strings and all 65536 model IDs (never a panic, always a profile that exists), the camera-card copy (the sentence is checked against `docs/11` and the mockup), and that detection carries the white scale and drops it for a forced profile and for an unknown body.
- `sensor_map_eval.rs`: the Rust port of `evaluate.py` and `gate.py` (below).
- `training/white_scale/test_white_scale.py`: the SR2 reader (the pad cipher against the vectors of `crates/ym-raw/src/sr2.rs`, the presets of the A1 photos), the consistency rule (single file, agreement, outlier, two files off, the 5 % limit, that it covers the measured bodies' spread, and what 1 % and 5 % mean in mired), the fit on constructed bodies (the scale recovered, the median, an outlier named, files that disagree, limits missed, the shape of the registry entry with and without a matrix), the pins (every file the registry names is pinned by a fetch script with the same SHA-256, no file of a set with an unclear licence, and the five dropped bodies), the rules every fitted body meets, and that `white_scale.json` is the registry's table (and a fresh fit of the pinned files when they are on disk).
- `tools/fetch/test_white_scale_photos.py`: the Play Raw fetch script against a folder read through `file://` (download, hash check, a second run without network, `--list`, `--force`, an unreachable store, a table that is refused) and the shipped table (32 files, SHA-256, post, poster and licence of each, no NonCommercial or unlicensed file, an address that names the file's SHA-1).

The evaluation needs CMFs, light SPDs and reflectances, which are CIE (CC BY-SA 4.0) and rawtoaces data that must not be committed. It reads the git-ignored fixtures that `python3 -m training.sensor_map.export_fixtures` writes to `testdata/spectral/`, and skips with a message when they are missing (`YM_REQUIRE_FIXTURES=1` turns a missing folder into a failure). CI runs, before `cargo test -p ym-color -p ym-profiles`: `pip install -r training/requirements.txt`, `tools/fetch/spectral.sh`, `python3 -m training.sensor_map.export_fixtures`, with `YM_REQUIRE_FIXTURES=1` and `YM_TESTDATA` unchanged (the Rust reader and the exporter both use `testdata/` or `$YM_TESTDATA`).

Results (Rust against the published CSV and the Python reference; the research objective `fit_map_weighted_rgb`, 138 colours under 9 lights):

| Check | Largest difference |
|---|---|
| `sony_profiles.csv` own median / p95, A7R IV median / p95, worst-light median (11 bodies) | 0.0005 ΔE00 (the CSV is rounded to 3 decimals); gate ±0.02; the worst light is the same for all 11 |
| Python `expected.json`, every per-light median and p95 | 8.2e-10 |
| Sensor-map gate with the shipped tables (11 bodies × 9 lights) against the Python `shipped_fit` columns | 1.2e-9 (asserted to 1e-6) |
| Lab-optimal fit redone in Rust (Levenberg–Marquardt, started from the research objective) against the shipped matrices, 3 bodies × 45 lights | 5.3e-9 per matrix element |
| Gate with the Rust refit (LED-B3, LED-RGB1, LED-V2 held out), 11 bodies × 9 lights, against `gate.json` | 5.3e-9 (asserted to 1e-5; measured before the ALEXA estimate moved and asserted since: the suite passes); A7R IV worst median 0.806 (LED-V2) and p95 2.103 (blackbody 2850 K), and exactly the seven accepted lights over (LED-V2 since 2026-10-12; blackbody 2850 K, D 5250 K, D 6750 K, FL3.15 and LED-B3 since 2026-10-25; LED-RGB1 since 2026-10-26), the test's list `accepted_failures` |

`crates/ym-color/tests/` holds the runtime checks: `spectral.rs` (the reference file: the `interpolated` fields to 1e-9, the exact fields within the recorded tolerances, CCT/Duv readout and inverse, the size and sign of Duv on all 11 profiles by an independent construction, the 4000 K junction as documented, every profile's tables), `spectral_fixtures.rs` (the tables against the spectra they were computed from, 11 profiles × 45 lights, 5e-10), `sensor_map.rs` (the blend against `runtime_ref` to 2e-16, white preservation to 6.7e-16 over 44000 random cases).
