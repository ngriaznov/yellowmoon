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

To be added by tasks:

- `cameras/registry.json`, `cameras/profiles/<id>/profile.json` (M1-04), `cameras/noise/384-ilce-1.json` and `cameras/noise/gain-regimes.json` (M2-01).
- `looks/builtin/neutral.json`, `looks/builtin/signature.json` (M1-10, M3-10); `looks/arri-packages.json` exists as a placeholder and gets fingerprints (never LUT data) in M3-09.
- `models/<name>/` weights, ONNX and `MODEL_CARD.md` (M2-06; the folder exists now with a README so the bundle config resolves).
- `alexa/target.json` (curve, version, provenance) (M1-02).
