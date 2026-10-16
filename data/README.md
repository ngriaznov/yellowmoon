# data/

Data that ships inside the app bundle (via `bundle.resources` in `tauri.conf.json`). Everything here must be Apache-compatible and carry its provenance. Large or non-shippable data never goes here (`tools/fetch/` downloads test data into `testdata/`).

| Path | What | Licence | Source |
|---|---|---|---|
| `alexa/alexa_alev3_estimate_380_780_5.json` | ALEXA ALEV III spectral sensitivity **estimate**, 380–780 nm at 5 nm, rawtoaces-data JSON format | ours, Apache-2.0; derived from AMPAS and Apache data (`NOTICE`) | `research/alexa-ssf/` |
| `alexa/alexa_alev3_estimate_380_780_5.csv` | same curve with min/max/std over the 24-variant ensemble | as above | as above |
| `alexa/variants.csv` | all 24 ensemble members (sensitivity tests, training augmentation) | as above | as above |
| `arri/Alexa_aces_matrix.txt` | ARRI's camera-native → ACES AP0 matrices at 17 CCTs, 13 Nov 2013 | AMPAS licence (`arri/AMPAS-LICENSE.md`); keep the notice | aces-dev v1.3.1 |
| `arri/alexa_native_to_awg3_17cct.json` | derived native → AWG3 matrices (`inv(AWG3→AP0) · M`), rows sum to 1 | AMPAS licence (derived) | computed in f64 from the published primaries (`docs/02`) |
| `cameras/sony/Sony_<model>_380_780_5.json` | measured spectral sensitivities of 11 Sony bodies: ILCE-7M3, 7M4, 7CM2, 7RM2, 7RM3, 7RM4, 7SM2, 7SM3, ILCE-9, ILCE-6400, DSC-RX100M4 | Apache-2.0 (`cameras/sony/LICENSE-rawtoaces-data`) | Weta Digital via rawtoaces-data |

| `alexa/target.json` | the ALEXA target definition: curve file, version, provenance | as `alexa/` | `training/sensor_map/build_target.py` |
| `cameras/registry.json` | 28 Sony model IDs → body name, profile, match kind (exact / same sensor / default); model-string aliases | ours (Apache-2.0) | `training/sensor_map/build_registry.py` from `docs/03` and `training/white_scale/white_scale.json` (the model IDs are Sony's own values; the white scale of each body, with its provenance) |
| `cameras/profiles/<id>/profile.json` | per-profile sensor maps (Lab-optimal white-preserving 3×3 per light class and mired), camera locus and Duv tables (camera raw scale), ALEXA white tables; 11 profiles | ours (Apache-2.0); fitted coefficients derived at fit time from CIE SPDs (CC BY-SA 4.0) — HUMAN M1-H4 | `training/sensor_map/build_profiles.py` |
| `cameras/color-matrices.json` | the D65 colour matrix (XYZ to camera RGB, numbers only) of 42 Sony bodies, by Sony model ID, for the Before view; the four bodies with no matrix of their own are listed under `excluded` and use the A7R IV entry (`fallback`). Embedded in `ym-profiles` (`include_str!`); schema `crates/ym-profiles/schema/color-matrices.schema.json` | ours (Apache-2.0); numbers from the bodies' published D65 colour matrices (`docs/14` rule 8) | `docs/02`, *The Before view* |
| `cameras/camera-default-tone.json` | the camera tone curve of the Before view: constants (pinned to `ym_color::camera_default` by a test), a 28-point golden table and the fit summary (aggregate numbers and body names only) | ours (Apache-2.0); fitted to the embedded JPEGs of 12 local photos from 10 Sony bodies | `docs/02`, *The Before view* |
| `cameras/noise/gain-regimes.json` | conversion-gain switch ISO per body (docs/06 table, to be confirmed on samples) and schemas | ours | `docs/06` |
| `models/illum/<id>.json` | illuminant estimator per profile (FFCC-style filter, class head, confidence; 13 k parameters each), `MODEL_CARD.md`, `SHA256SUMS` | ours (Apache-2.0); trained on synthetic scenes from rawtoaces reflectances (Apache-2.0) and CIE SPDs at fit time (HUMAN M1-H4) | `training/illum/` |
| `dng/profile.json` | the DNG camera profile (65-point `ProfileToneCurve`, 36×16×16 `ProfileLookTable`) that makes Adobe's DNG renderer reproduce our ACES 2 rendering of scene-linear AWG4; `fit_offset_ev` −4 | ours (Apache-2.0); fitted to our own ACES 2 port on generated CIELAB colours | `tools/dngfit` (`dngfit fit`) |
| `looks/arri-packages.json` | recognition patterns for ARRI's five free look packages (names, counts, LUT sizes; fingerprints empty until HUMAN M3-H2) — never LUT data | ours | `tools/looks/fingerprint.py` |

Still to come: per-body noise priors `cameras/noise/<model-id>-<model>.json` (M2-01, needs the CC0 corpus), `looks/builtin/signature.json` (HUMAN M3-H1), and `models/jdd/` weights with their model card (M2-06, needs GPU training — HUMAN M2-H4; the classical demosaic ships until then).
