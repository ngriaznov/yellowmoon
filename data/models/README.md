# data/models/

Model weights shipped in the app bundle, one folder per model (`jdd/`, `illum/`, optionally `jdd-ms/`, `smap/`), each with the ONNX file, safetensors, SHA-256 and `MODEL_CARD.md`. Empty until M2-06 (and M1-06 for `illum`). See `docs/07-ml-models.md`. The folder exists from the start because `tauri.conf.json` bundles it.
