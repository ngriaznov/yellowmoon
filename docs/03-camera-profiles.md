# 03 · Camera detection and profiles

Crates: `ym-raw` (detection), `ym-profiles` (registry and data). Data: `data/cameras/`.

The camera is an input to the pipeline from day one. Adding a camera later means adding a profile file and its tests; the pipeline code stays the same.

V1 relies on available measurements only: the 11 bodies Weta measured and published in rawtoaces-data. We don't measure cameras ourselves in V1; every other body maps to a same-sensor sibling or the A7R IV.

## Detection

For every file, before decoding pixels:

1. Read `Make` (0x010F) and `Model` (0x0110) from IFD0.
2. Read Sony's model ID (maker note tag `0xB001`, SonyModelID). When the model ID and the Model string disagree, the model ID wins (firmware variants and regional "A" models write odd strings).
3. Record the compression mode and bit depth from the raw IFD (`docs/04-raw-decoding.md`).

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

**As built (M1-04), deviations from the first draft of this section.**

- *No `decode` field.* `DecodeKind` (`Uncompressed12 | Uncompressed14 | Craw | LosslessL | LosslessMS | Arw6 { .. }`) belongs to the decoder: it lives in `ym_raw::RawMetadata`. `ym-profiles` does not depend on `ym-raw`, and `ym-core` composes the two (the UI and `ym info` show both). Step 3 above is therefore the decoder's, not detection's.
- *`MatchKind::Forced { detected }`* is a fourth variant, for the CLI's `--profile <id>`: the body is detected as usual and the forced profile replaces the detected one (`ProfileStore::detect_forced`). Forcing the profile detection chose returns the detected match unchanged. A forced profile is never reported as `Exact`, so the STAND-IN tag shows.
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
| Everything else, including ILCE-1 A1 (384), ILCE-1M2 (400), ILCE-9M3 (392), ILCE-7M5 A7 V (407), ILCE-7RM6 A7R VI (410), and the two bodies whose same-sensor pairing failed its check, ILCE-7RM5 A7R V (390) and ILCE-7CR (396) | sony-ilce-7rm4 | default |

Model IDs come from ExifTool's Sony tables. Before a same-sensor pairing ships, check it: the sibling's ColorMatrix must match the profile body's within 0.02 per coefficient after normalisation. Record the check, with its measured value, the tool and its version, in the profile's `provenance`. A pairing that fails does not ship: the body gets match kind `default` (the camera card says it borrows the A7R IV, which is also what it renders with) and the failed check stays in the provenance.

*Sibling check, as run (V1, 2026-10-13).* V1 has no Adobe DNG Converter samples (the V1 data rule), so the independent tool is LibRaw: its per-model camera matrix (`rgb_xyz_matrix`, the D65 ColorMatrix from Adobe's tables, keyed by the Sony model ID) read through rawpy 0.27.1 (LibRaw 0.22.1) by `crates/ym-profiles/tests/py/sibling_matrices.py`, rows normalised to sum 1, the largest per-coefficient difference between the sibling and the profile body. LibRaw is a test oracle here; nothing of it ships.

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

0.0000 means LibRaw holds the same matrix for both bodies, so the check is strict but says only that LibRaw does not tell them apart. The A7R V and the A7CR have a matrix of their own (a newer sensor generation, 0.0934 from the A7R IV's); they render with the A7R IV profile exactly as before, but they are no longer described as sharing its sensor. `sibling_matrices.py --verify` compares the recorded values with a fresh run.

*Sensor geometry (registry 1.1.0).* Each registry body can carry `sensor_width_mm` and `active_width_px` (and an optional `pixel_pitch_um` override), the pixel pitch for the diffraction deconvolution of `docs/algorithms/03-diffraction-deconvolution.md` section 2.2: `1000 * sensor_width_mm / active_width_px` micrometres. They are Sony's published figures (sensor size in mm; the width of the largest 3:2 still image from the full sensor), with the source of every entry in `sensor_source` and `sensor_source_kind` (`maker` for an excerpt of a Sony specification page, `restated` for a retailer or reference site restating Sony's figures). The Sony pages were read through web-search excerpts, not fetched. 27 of the 28 bodies have them (pitches 2.41 um for the RX100 IV to 8.40 um for the A7S II and A7S III); the ILME-FX3 has none, because Sony's page gives its stills only as "12M", and uses the EXIF crop-factor fallback. A body the registry does not list has none either. `ProfileStore::detect` puts them on the `CameraMatch` (`sensor: Option<SensorGeometry>`).

Measured curves for the 11 exact bodies are Weta's (rawtoaces-data, Apache-2.0), copied to `data/cameras/sony/`.

## Why measured profiles matter

From `research/alexa-ssf/out/sony_profiles.csv`: ΔE00 to the ALEXA rendering, median / p95 over 138 colours under 9 lights, using each body's own profile versus the A7R IV profile.

| Camera | Own | A7R IV |
|---|---|---|
| A7 IV | 0.34 / 0.84 | 0.37 / 0.82 |
| A7C II | 0.36 / 0.92 | 0.40 / 0.92 |
| A7R III | 0.25 / 0.65 | 0.33 / 0.74 |
| A7 III | 0.27 / 0.72 | 0.44 / 1.05 |
| A7R II | 0.27 / 0.65 | 0.58 / 1.13 |
| A7S III | 0.28 / 0.76 | 0.59 / 1.45 |
| A9 | 0.37 / 0.99 | 0.63 / 1.48 |
| A6400 | 0.35 / 0.98 | 1.20 / 3.13 |
| RX100 IV | 0.31 / 1.05 | 1.60 / 3.63 |
| A7S II | 0.47 / 1.55 | 2.69 / 4.43 |

Curves estimated from two published colour matrices were worse than the A7R IV stand-in for 8 of 10 bodies, so we don't ship estimated Sony curves. Stand-ins come only from shared sensors.

## Lookup

`exact model ID → sensor family → default (A7R IV)`. A Sony file is never refused for lack of a profile. The CLI's `--profile <id>` forces a profile (for comparing a new one against the default); the app has no override.

**As built (M1-04).** In order: (1) a Make other than Sony (a non-empty Make that does not start with "sony", ignoring case; the rebadged Hasselblad bodies) gets the default profile whatever the model ID says; (2) with a model ID, the registry body with that ID, and an ID the registry does not list is a body we have not seen, so it gets the default profile even when the Model string looks familiar (the ID wins); (3) without an ID, the body the Model string names through `model_strings`, compared ignoring case, surrounding whitespace and NULs and a leading "Sony "; (4) anything else gets the default profile. A missing Make counts as Sony.

## Profile files

```
data/cameras/
  registry.json                       model ID → { model, body, profile, match, via, notes }; Model-string aliases; default profile
  sony/Sony_<model>_380_780_5.json    measured curves (Weta, Apache-2.0) + LICENSE-rawtoaces-data
  profiles/<profile-id>/profile.json  generated by training/ scripts, versioned
  noise/<model-id>-<model>.json       per-body noise priors (A1 from M2); gain-regimes.json
```

`profile.json` (schemas in `crates/ym-profiles/schema/`: `profile.schema.json`, `registry.schema.json`, `target.schema.json`; checked by `python3 -m training.sensor_map.validate_profiles` and, from M1-04, by Rust tests). Abbreviated; `…` marks a table filled by the fit script:

```json
{
  "id": "sony-ilce-7rm4",
  "version": "1.0.0",
  "display_name": "A7R IV",
  "spectral_curves": "sony/Sony_ILCE-7RM4_380_780_5.json",
  "sensor_map": {
    "method": "per-light-3x3",
    "target": "alexa/target.json",
    "fit": { "white_preserving": true, "weights": "Y^(-2/3)", "training_reflectances": 190 },
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
    "reference_lights": { "daylight": ["daylight-11000K", …], "tungsten": […], "fluorescent": […], "led": […] },
    "cie_data": "CIE SPDs … used at fit time only … pending HUMAN M1-H4",
    "checks": [{ "pairing": "ILCE-7C (381) uses sony-ilce-7m3", "check": "…", "result": "pending: no converted samples available" }]
  }
}
```

What the tables hold, and how the runtime reads them (full rules in `training/sensor_map/README.md`; reference implementation `training/sensor_map/runtime_ref.py`):

- **Grids** are `{ start_k, step_k, values }`; `values[i]` belongs to `start_k + i·step_k`; queries clamp to the ends. With `T_i ≤ T ≤ T_(i+1)` the interpolation weight is in **mired**: `a = (1/T − 1/T_i) / (1/T_(i+1) − 1/T_i)`. `w_A` interpolates geometrically (`w_i^(1−a) · w_(i+1)^a`, green stays 1); log-chroma, xy and the Duv scale interpolate linearly. Blackbody tables serve below 4000 K, daylight tables from 4000 K.
- `locus` is the camera RGB (green = 1, **in the camera's raw scale**: the published curves, clipped at 0, not normalised per channel) of blackbody 1500–4000 K and CIE daylight 4000–25000 K every 10 K, as log-chroma, with the matching CIE xy. `duv_scale` is the CIE 1960 uv distance per unit of log-chroma distance across the locus, tabulated every 100 K (it varies by a factor of 5 to 8 over the range depending on the body, so it is a table, not a number). Duv is positive toward magenta, which is the opposite of CIE's own sign.
- `alexa_white` holds `w_A` for blackbody 1500–25000 K and daylight 4000–25000 K every 10 K (the D series is only defined from 4000 K), and for the fluorescent and LED reference lights as `(mired, w)` lists. `w_A` doesn't depend on the Sony profile but lives here so a future second target (for example a different ALEXA estimate) can be versioned with the profiles that were fitted against it.
- `sensor_map.classes.<class>` lists the maps fitted at that class's reference lights, sorted by strictly increasing mired; every matrix has rows summing to 1 (white preserving). `light` names the reference light (for traceability and to tell the gate which entries to drop).
- `locus` and `alexa_white` are precomputed offline so the app never needs CIE datasets at runtime (`docs/02`). The off-grid accuracy of these tables against the spectra is stated in `crates/ym-color/tests/data/spectral_reference.json` (`tolerances`).
- **Which curve scale each table uses (fix of 2026-10-12, `docs/02` C1; recorded in `provenance.light_conventions`).** `locus` and `duv_scale` come from the curves exactly as Weta published them (`data/cameras/sony/*.json`, clipped at 0, one common factor per camera with the green peak at 1.0, red and blue peaks 0.57 to 0.62 and 0.78 to 0.87 of it): the scene white at runtime is raw data (as-shot gains, Pick white, the illuminant estimator), so a locus point must be the raw white of its light. `sensor_map` and `alexa_white` divide every response by the response to the white of the same curves, so their scale cancels; they are still computed from the peak-normalised curves of `ssf_core.load_cameras` (the path the published leave-one-out numbers use) and are byte for byte unchanged by the fix. The first locus was built from the peak-normalised curves too, which put it off by ×1.6 in red and ×1.2 in blue (the A7 III blackbody 6504 K white read `[0.623, 1, 0.915]` instead of `[0.371, 1, 0.767]`) and made real as-shot whites read Duv −0.022 to −0.052. The two differ by a constant per camera in log-chroma (`ln(peak_R/peak_G)`, `ln(peak_B/peak_G)`), a translation, so `duv_scale` and all chord geometry came out the same. Measured after the fix on the 9 real ARWs in `testdata/real/files`: as-shot Duv −0.0065 (A7 III) and −0.0008 (A7 IV), −0.005 to +0.025 on the seven bodies served by the A7R IV stand-in. The published curves and each camera's own ColorMatrix still differ by 3.5 % (A7 III) and 6 % (A7 IV) in the D65 white, a bias along the locus of 250 to 650 K and not a tint (`docs/02` C1).

**As built (M1-02 and M1-05 runtime).** `ym_color::spectral` reads the grids, the locus and `w_A` (`Grid`, `Locus`, `AlexaWhite`; `SpectralCurves` loads rawtoaces-format curve files for tests and tools), `ym_color::sensor_map` the maps (`SensorMapTables`), and `ym_profiles::profile::Profile` holds them with the ids, versions and provenance (`Profile::sensor_map`, `scene_white`, `user_white`, `rgb_to_cct_duv`, `cct_duv_to_rgb`). Where the reading rules above leave a choice, the runtime does this:

- *Class blend of `w_A` (docs/02 C3).* `scene_white(cct, probs) = Σ_k p_k · w_k(mired)` with the class probabilities normalised to sum 1 and green set to exactly 1. The fluorescent and LED classes interpolate their reference-light lists linearly in mired (clamped at the ends); the daylight and tungsten classes are the scene-light model at `1e6 / mired` K (blackbody below 4000 K, CIE daylight from 4000 K, geometric in mired). A fluorescent or LED class without reference lights falls back to the model.
- *Empty classes in the sensor map.* A class without entries is dropped and the others renormalised: its probability moves to the non-empty classes in proportion to their own; if those have none either, to the non-empty classes equally; with no tables at all the map is the identity. The result is always white preserving.
- *Duv sign.* The chord normal is `(t_y, −t_x)/|t|`, flipped when it would point away from magenta (`n_x + n_y < 0`; magenta is `(1, 1)` in log-chroma space, perpendicular to the locus). For every forward chord this is the rule above bit for bit. The CIE D series as colour-science builds it (M1 and M2 rounded to three decimals) is a staircase, and above about 16000 K some chords run backwards; without the flip their normals point towards green and the sign of Duv is wrong for tints of 0.01 to 0.05 log-chroma units from about 17400 K up (measured on 11 profiles). With the flip the sign is right everywhere, but the staircase also makes the chord normal noisy, so the *size* of Duv drifts down with temperature. Measured as readout / (tint × Duv scale) for tints of ±0.01 and ±0.05 log-chroma units (Duv of about ±0.001 to ±0.004) on all 11 profiles, magenta tints first: 4400 to 11000 K within 2.6 % (median 1.000); 11000 to 14000 K median 0.998, worst 0.91 (green 0.955); 14000 to 18000 K median 0.95 (green 0.99), worst 0.65 (green 0.80); 18000 to 20000 K median 0.93 (green 1.00), worst 0.58 (green 0.86); 20000 to 25000 K median 0.66 (green 0.96), worst 0.56 (green 0.85). The same staircase makes a temperature read back up to 10 K off above 20000 K (two temperatures share a camera RGB); below 16000 K a locus point reads back to 1e-6 K. All of this is above the UI's Kelvin range (2000 to 11000 K); a readout that is used above 14000 K should treat the tint as approximate.
- *The 4000 K junction.* The blackbody and daylight branches meet at 4000 K but are not collinear: the D series sits about 0.04 log-chroma units (0.011 in ln R/G, 0.039 in ln B/G) on the green side of the Planckian locus there. The readout takes the branch with the smaller distance, so a light between the two branches can be read from the other one, and a tinted light whose perpendicular foot would lie beyond the end of the branch it is nearest to is clamped to that end and projected on the end chord alone. Measured on the A7R IV profile (the other ten show the same pattern: 0.26 to 0.50 times for 0.05 on a 4200 K light), as readout Duv / (tint × scale) and the CCT error: a 1.5 % green excess on a 3894 K light reads as 4000 K with Duv +0.0009 instead of a CCT near 3890 K and a Duv near −0.0017; a magenta tint of 0.03 log-chroma units (Duv about +0.0025) on a 4100 K light reads 4000 K with a Duv of −0.14 times the tint (wrong sign), and 0.05 on a 4200 K light 0.29 times with the CCT 200 K low; green tints on 3800 to 3950 K lights read −0.1 times (0.03, wrong sign) or 0.33 to 0.37 times (0.05) with the CCT 80 to 200 K high. Tints of 0.01 (Duv about 0.0008) and anything below 3700 K or above 4300 K (checked up to 0.05) read correctly. CCT and Duv are continuous within a branch, not across 4000 K, and in the band of about 3800 to 4300 K a Duv beyond ±0.002 is not trustworthy. This is a property of the specified readout (nearest chord of two tables that meet at 4000 K), not of the port: `runtime_ref.py` gives the same numbers. A fix is a design decision about which locus defines tint near 4000 K, not a patch: a point between two loci 0.04 apart is simply near both, and straight extensions of the end chords were tried and only move the problem. Options are one locus for the readout across the whole range (the Planckian, which is how CIE defines Duv, keeping the D series for the `w_A` model) or a blend of the two branches over about 3500 to 4500 K; either changes `profile.json`, `runtime_ref.py` and the reference file together. Until then the illuminant estimator and the tint readout must not rely on Duv being continuous or exact there (`tests/spectral.rs::the_4000_k_junction_is_documented_behaviour` pins the 3894 K and 4100 K examples, so a change to the rule shows up in a test and in this paragraph).
- *`w_A` at 4000 K.* The same rule makes `w_A` step at 4000 K: the blackbody table gives `[0.9415, 1, 0.7355]` at 4000 K and the daylight table `[0.9306, 1, 0.7103]` (red 1.2 % and blue 3.4 % lower; the ALEXA estimate is the same for every profile). `user_white` and the daylight and tungsten classes of `scene_white` therefore jump by that much when the CCT crosses 4000 K, which a Kelvin slider dragged across 4000 K shows as a small step in blue (about 0.05 stop). It is the rawtoaces convention the ALEXA curve was fitted with; a smoother rule (cross-fading the two tables over about 3500 to 4500 K) is a decision for M1-07, not a runtime fix.
- *Auto white balance.* Stage C3 and C4 cancel only when `E_user` is the same light as `E_scene`: for Auto, M1-07 must take `g_user = 1 / scene_white(cct, probs)` (the class-blended `w_A`), not `1 / user_white(cct)` (the blackbody/daylight model), or an LED or fluorescent scene gets a residual cast of the difference between its reference-light `w_A` and the model's.
- *Validation.* Loading checks what the schema cannot express (`Profile::validate`, the Rust twin of `validate_profiles.py` without the file hashes): grid ranges and steps, green exactly 1 in every `w_A`, strictly increasing mired in every class list, rows of every matrix summing to 1 within 1e-12, reference-light names equal to the sensor-map entries and mired of formula lights equal to `1e6 / T`, the locus running the right way, `decode.quirks` empty. A profile that fails is refused with its id in the message.

`registry.json` has `default_profile`, `profiles` (id → file), `bodies` (one entry per model ID in the table above: `model_id`, `model`, `body`, `profile`, `match` = `exact` | `sensor_family` | `default`, `via` = the measured body for `sensor_family` and `null` otherwise, `notes`) and `model_strings` (IFD0 Model string → model ID, for files without a usable 0xB001). A model ID that isn't in `bodies` gets `default_profile` with `match = default`.

Noise priors are not part of a colour profile: they belong to a body, not a sensor family. They live in `data/cameras/noise/`, keyed by model ID (for example `384-ilce-1.json`), fitted from public files in M2-01, with `data/cameras/noise/gain-regimes.json` holding the gain-switch table from `docs/06`. A body without its own file gets the wide prior.

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

The profile id and version are written into every export's XMP (`docs/09-export.md`). If a profile changes, `pipeline_version` doesn't need to change, but the profile version must.

## Adding a camera later

In V1, a camera gets its own profile only if permissively licensed measurements are published (check rawtoaces-data and similar sources at each release). Our own measurements start in V2:

1. [HUMAN, V2] Measure spectral curves (monochromator or narrowband filters), or obtain permissively licensed measurements.
2. [HUMAN, V2] Shoot noise frames (dark and flat, per ISO) and a test set (colour chart under daylight, tungsten and LED).
3. Run the fit scripts; add `profile.json` and a registry entry; run the profile gates (`docs/13-testing.md`).
4. Ship. The A1 is first, in V2.

## Tests

- Every Sony sample in the test corpus (`tools/fetch/samples.sh`) is detected as the right body and gets the profile in the table; the fallback is reported for the rest.
- Registry round trip: every model ID maps to exactly one profile; every profile file validates against the schema and loads.
- The 11 exact profiles reproduce `sony_profiles.csv` "own" numbers within ±0.02 ΔE00 when evaluated by the Rust code (port of the Python check).

**As built (M1-04).** `crates/ym-profiles/tests/`:

- `data_files.rs`: every `profile.json`, the registry and `target.json` validate against `schema/*.schema.json` with the `jsonschema` crate (a dev-dependency only, MIT; the HTTP and file reference resolvers are off), the schemas reject sixteen broken documents, the typed Rust values write back documents the schemas accept, the registry round-trips (model IDs unique, each maps to one profile that exists, loads and has the id the registry gives it, exactly one exact body per profile, defaults present), equals the table in this file (parsed from the Markdown), the curve and target files match the SHA-256 in `provenance`, each of the 10 same-sensor pairings has its `provenance.checks` entry with a numeric result within 0.02, and the two rejected pairings are recorded as failed, and loading reports a missing, truncated, non-white-preserving or mislabelled file (profile, registry or target) with the file or profile named.
- `detect.rs`: all 28 bodies by model ID (with a wrong Model string), by Model-string alias (case, spacing, NULs, "Sony " prefix), the ID-wins rule, unknown Sony bodies, non-Sony makes, `--profile`, odd strings and all 65536 model IDs (never a panic, always a profile that exists), and the camera-card copy (the sentence is checked against `docs/11` and the mockup).
- `sensor_map_eval.rs`: the Rust port of `evaluate.py` and `gate.py` (below).

The evaluation needs CMFs, light SPDs and reflectances, which are CIE (CC BY-SA 4.0) and rawtoaces data that must not be committed. It reads the git-ignored fixtures that `python3 -m training.sensor_map.export_fixtures` writes to `testdata/spectral/`, and skips with a message when they are missing (`YM_REQUIRE_FIXTURES=1` turns a missing folder into a failure). CI runs, before `cargo test -p ym-color -p ym-profiles`: `pip install -r training/requirements.txt`, `tools/fetch/spectral.sh`, `python3 -m training.sensor_map.export_fixtures`, with `YM_REQUIRE_FIXTURES=1` and `YM_TESTDATA` unchanged (the Rust reader and the exporter both use `testdata/` or `$YM_TESTDATA`).

Results (Rust against the published CSV and the Python reference; the research objective `fit_map_weighted_rgb`, 138 colours under 9 lights):

| Check | Largest difference |
|---|---|
| `sony_profiles.csv` own median / p95, A7R IV median / p95, worst-light median (11 bodies) | 0.0005 ΔE00 (the CSV is rounded to 3 decimals); gate ±0.02; the worst light is the same for all 11 |
| Python `expected.json`, every per-light median and p95 | 8.2e-10 |
| Sensor-map gate with the shipped tables (11 bodies × 9 lights) against the Python `shipped_fit` columns | 1.2e-9 (asserted to 1e-6) |
| Lab-optimal fit redone in Rust (Levenberg–Marquardt, started from the research objective) against the shipped matrices, 3 bodies × 45 lights | 5.3e-9 per matrix element |
| Gate with the Rust refit (LED-B3, LED-RGB1, LED-V2 held out), 11 bodies × 9 lights, against `gate.json` | 5.3e-9 (asserted to 1e-5); A7R IV worst median 0.759 and p95 1.515, both LED-V2, the one accepted failure |

`crates/ym-color/tests/` holds the runtime checks: `spectral.rs` (the reference file: the `interpolated` fields to 1e-9, the exact fields within the recorded tolerances, CCT/Duv readout and inverse, the size and sign of Duv on all 11 profiles by an independent construction, the 4000 K junction as documented, every profile's tables), `spectral_fixtures.rs` (the tables against the spectra they were computed from, 11 profiles × 45 lights, 5e-10), `sensor_map.rs` (the blend against `runtime_ref` to 2e-16, white preservation to 6.7e-16 over 44000 random cases).
