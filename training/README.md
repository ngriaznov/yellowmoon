# training/

Offline fitting, training and validation scripts. Nothing here ships in the app: it produces the data in
`data/` (profiles, registry, target) and the reference values the Rust tests compare with
(`crates/ym-color/tests/data/spectral_reference.json`), and it reproduces the published ALEXA-curve numbers in CI.

Everything runs **from the repository root** as `python3 -m training.<package>.<module>`, is deterministic
(no timestamps, fixed seeds, rounding before writing) and reads its public inputs from `testdata/sources/`
(git-ignored), filled by `tools/fetch/spectral.sh` at pinned commits.

```bash
pip install -r training/requirements.txt          # colour-science==0.4.7, numpy, scipy, matplotlib, jsonschema
tools/fetch/spectral.sh                            # rawtoaces-data @ e9b8503, aces-dev v1.3.1 (ARRI matrices only)
```

Python 3.12 is what CI uses; 3.13 gives the same numbers. Set `YM_TESTDATA` to move `testdata/` elsewhere.

## Layout

| Path | What |
|---|---|
| `common/paths.py` | repository, `testdata/` and shipped-data locations |
| `common/jsonfmt.py` | deterministic compact JSON writer for the generated data files |
| `alexa/` | the ALEXA curve pipeline (M1-03): `ssf_core.py` (spectral model), `validate_loo.py` (leave-one-camera-out), `check_loo.py` (compare with the published numbers), `estimate_alexa.py` + `compare_estimate.py` (regenerate the shipped curve) |
| `sensor_map/` | sensor-map fit, profiles, registry, gate, reference values (M1-02, M1-04, M1-05): see `sensor_map/README.md` |
| `illum/` | illuminant estimator: synthetic scenes, FFCC-style training, class head, gates, Cube++ evaluation (M1-06): see `illum/README.md` |
| `noise/` | per-body noise priors from public files (M2-01): see `noise/README.md` |
| `jdd/` | joint demosaic + denoise network: data generator, training, export, model card (M2-04, M2-05): see `jdd/README.md` |
| `synth/` | the synthetic golden set: spectral scenes, noise, ground truth (stand-in for the CC0 corpus): see `synth/README.md` |
| `white_scale/` | the white scale of stand-in bodies (`docs/03`): our own reader of the camera's white-balance presets (`sr2.py`), the two references and the fit (`reference.py`, `fit.py`), the table that `build_registry` copies into the registry (`white_scale.json`), the published colour matrices it cross-checks with (`colour_matrices.json`) |
| `other_makes/` | research only, not shipped: the Canon and Nikon curves of rawtoaces-data measured against the profile checks (`check.py`), with the published colour matrices of Canon and Nikon bodies it compares them with (`colour_matrices.json`); results in `research/canon-nikon-curves/` |

`alexa/ssf_core.py`, `validate_loo.py` and `estimate_alexa.py` are the scripts of the ALEXA curve estimate
(`research/alexa-ssf/README.md`: method and validation), with their numerics as they were published.

## Commands

| Command | Does | Time (4 cores) |
|---|---|---|
| `python3 -m training.alexa.validate_loo` | leave-one-camera-out over the 52 cameras → `testdata/out/alexa/loo_results.csv` (the per-camera results behind the published table); `--quick` runs only the shipped variant (K=4, λ=1) | 5–10 min (full), ~2 min (`--quick`) |
| `python3 -m training.alexa.check_loo` | compares the run with the published numbers (median RFSE 2.2 %, emulation ΔE00 median 0.23, plus the other published figures) within ±0.1, and with the `docs/13` gate (RFSE ≤ 3 %, ΔE00 ≤ 0.3); non-zero exit on failure | 1 s |
| `python3 -m training.alexa.estimate_alexa` then `compare_estimate` | regenerates `data/alexa/*` into `testdata/out/alexa/estimate/` and checks it equals the shipped files | ~10 min |
| `python3 -m training.sensor_map.evaluate` | reproduces `research/alexa-ssf/out/sony_profiles.csv` (the published sensor-map result) | 5 s |
| `python3 -m training.sensor_map.build_profiles` | writes the 11 `data/cameras/profiles/*/profile.json` (`--check` compares without writing) | 20 s |
| `python3 -m training.sensor_map.build_registry` / `build_target` | write `data/cameras/registry.json` / `data/alexa/target.json` | 1 s |
| `python3 -m training.white_scale.fit --write` / `--check` / `--report` | fits the white scale of every stand-in body that has files in `testdata/readme-photos/src` and writes `white_scale.json` (run `build_registry` after) / compares a fresh fit with it / prints both references' residuals for those files and for `--local` | 2 s |
| `python3 -m unittest training.white_scale.test_white_scale` | the SR2 reader, the fit, and that the registry carries `white_scale.json` | 1 s |
| `python3 -m training.sensor_map.validate_profiles` | schema + semantic checks of all shipped camera data (needs no fetched sources) | 5 s |
| `python3 -m training.sensor_map.gate` | sensor-map gate (M1-05) for all 11 profiles | 10 s |
| `python3 -m training.other_makes.check` | the Canon and Nikon research checks (emulation, curves against matrices, sibling candidates, sensor-map gate and headroom) → `research/canon-nikon-curves/*.csv`; `--skip-emulation` leaves out the two emulation tables | 3 min (15 s with `--skip-emulation`) |
| `python3 -m training.sensor_map.check_reference` | checks `spectral_reference.json` against the shipped profile (stale content digest of `profile.json` without its provenance block, `runtime_ref` reproduces every field) | 2 s |
| `python3 -m training.sensor_map.spectral_reference` | regenerates `crates/ym-color/tests/data/spectral_reference.json` (`--study` prints the interpolation study) | 10 s (1 min for `--study`) |
| `python3 -m training.sensor_map.export_fixtures` | writes `testdata/spectral/` for a Rust port of the evaluation (never committed) | 10 s |

## Licences and data

- Inputs: rawtoaces-data (Apache-2.0: 52 camera curves, 190 training reflectances), aces-dev v1.3.1 (AMPAS licence: the
  ARRI 17-CCT matrices), colour-science 0.4.7 (BSD-3).
- colour-science also supplies **CIE data (CC BY-SA 4.0)**: the 1931 observer, the D series basis, F1–F12 and LED SPDs,
  the 99 CES. They are used at fit and test time only. Committed files hold fitted coefficients and tables derived from
  them, never the SPDs; `export_fixtures.py` writes the CIE data to the git-ignored `testdata/spectral/` for tests.
  Whether derived tables may ship is the open question **HUMAN M1-H4** (`docs/14`); every profile says so in
  `provenance.cie_data`.
- No ARRI look or LUT data is used anywhere (`AGENTS.md` rule 2).

## CI

`.github/workflows/alexa-curve.yml` (ubuntu-latest, Python 3.12): `pip install -r training/requirements.txt`,
`tools/fetch/spectral.sh`, `validate_loo`, `check_loo`. It runs on changes under `training/alexa/`,
`training/common/`, `data/alexa/`, `data/arri/`, nightly, and on demand.
