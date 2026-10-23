# data/models/

Model weights shipped in the app bundle, one folder per model (`jdd/`, `illum/`, `crop/`, optionally `jdd-ms/`, `smap/`), each with its weights, SHA-256 and `MODEL_CARD.md`. `illum/` ships since M1-06 and `crop/` (Auto crop's crop model, `docs/20-auto-crop.md` §6) since 2026-10-23; `jdd/` is empty until M2-06. See `docs/07-ml-models.md`. The folder exists from the start because `tauri.conf.json` bundles it.
