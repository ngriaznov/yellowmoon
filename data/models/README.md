# data/models/

Model weights shipped in the app bundle, one folder per model, each with its weights, `SHA256SUMS` and `MODEL_CARD.md`. See `docs/07-ml-models.md`. The folder exists from the start because `tauri.conf.json` bundles it.

| Folder | Model | Weights licence |
|---|---|---|
| `illum/` | the illuminant estimator, one file per camera profile (M1-06) | Apache-2.0 |
| `autoset/` | automatic exposure and contrast (`docs/18`), `autoset.json` with its metrics | Apache-2.0 (trained on CC0 files only) |

`jdd/`, `jdd-ms/` and `smap/` come with M2-06. Weights trained on CC BY-SA data would be CC BY-SA 4.0 with their own `LICENSE` in their folder (`docs/14` rule 5).
