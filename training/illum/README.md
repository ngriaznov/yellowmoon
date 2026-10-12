# training/illum/

The illuminant estimator of `docs/02` (section Illuminant estimate), task M1-06: synthetic scenes, an FFCC-style
filter on log-chroma histograms, an as-shot prior and a class head, trained per camera profile. The Rust inference is
`crates/ym-color/src/illum.rs`; the shipped weights are `data/models/illum/<profile-id>.json` with `MODEL_CARD.md`.

Run from the repository root; needs `pip install -r training/illum/requirements.txt` (adds jax for the training) and
`tools/fetch/spectral.sh`.

| File | Role |
|---|---|
| `spectra.py` | lights (blackbody, CIE D, F, LED, generated near-white LED and fluorescent spectra), train / val / test splits by light, reflectances (rawtoaces 190 split, smooth random spectra, CES 99 for evaluation only), camera curves in the **published raw scale** |
| `scenes.py` | one synthetic preview (Voronoi layout, shading, specular, noise, clipping), the truth, the as-shot white balance, `Bv`, ISO |
| `features.py` | the log-chroma histograms (reference of `ym_color::illum::histogram`) |
| `model.py` | the model: filter, Gaussian fit, as-shot prior, class head, JSON file (reference of `ym_color::illum`) |
| `dataset.py` | datasets of scenes cached under `testdata/out/illum/cache` (git-ignored; delete freely) |
| `train.py` | JAX training of the filter, the class head, calibration |
| `baselines.py` | grey world, white patch, shades of grey, grey edge; the angular error |
| `evaluate.py`, `report.py` | metrics on held-out synthetic scenes (test, CES, gate lights, validation, cross-camera); the model report |
| `gate_estimated.py` | the M1-05 sensor-map gate with the light taken from the estimator |
| `eval_cubepp.py` | Cube++ evaluation (blocked: zenodo.org is unreachable here); `tools/fetch/cubepp.sh` |
| `export_vectors.py` | test vectors for the Rust port (`crates/ym-color/tests/data/illum/`): a random-weight model with every stage, the A7R IV model, and the final outputs of all 11 shipped models |
| `run_all.py` | train, ship and report all 11 profiles (about 15 minutes each on 4 shared cores); `model_card.py` then writes `MODEL_CARD.md` and `SHA256SUMS` from the reports |
| `smoke_real.py` | gross-failure check on real ARW files (local only, no ground truth); passes `Bv` from the EXIF when `exiftool` is installed |
| `robust_prior_study.py` | review study: what a two-component (robust) as-shot prior would change on the shipped filters; not part of the shipped estimator (`docs/02`) |
| `requirements.txt` | adds jax to `training/requirements.txt` |
| `test_illum.py` | unit tests (`python3 -m unittest training.illum.test_illum`) |

## Data rules

* Training data is synthetic: rawtoaces patches (Apache-2.0) and smooth random spectra under CIE lights (blackbody, D
  series, FL1-FL12, FL3.1-FL3.15, LED-B1..V2; CC BY-SA 4.0 via colour-science, used at training time only, HUMAN M1-H4
  pending) and generated spectra. The 99 CES are evaluation only.
* No camera data of our own, no real photographs. The camera's sensitivities are the published Weta curves; noise
  parameters are nominal values for the sensor class.
* Splits by light **and** reflectance: `train`, `val` (calibration) and `test` (the gates) never share a light or a
  reflectance patch (`spectra.py`; `test_illum.py` checks it).
