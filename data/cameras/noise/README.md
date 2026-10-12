# data/cameras/noise/

Noise priors for the per-image estimator in `crates/ym-noise` (`docs/06-noise.md`, section Priors). They belong
to a **body**, not to a sensor family or a colour profile, and they are statistics only: fitted from public
files by `training/noise/public_priors.py`, never the files themselves.

| File | What | Status |
|---|---|---|
| `gain-regimes.json` | per body: the ISO where the conversion gain switches, the base ISO, the ISO above which the raw looks processed | the table of `docs/06`; every row says `"source": "docs/06 table, to be confirmed on samples"`; data overrides it |
| `<model-id>-<model>.json` | per-body prior: `ln a = alpha ln(ISO) + beta` and `b` per gain regime | **none yet**: the public corpus (`tools/fetch/samples.sh`) was not reachable when M2-01 was built. A body without a file gets the class prior |
| `noise-prior.schema.json`, `gain-regimes.schema.json` | JSON Schemas of the two kinds of file | checked by `python3 -m unittest training.noise.test_public_priors` |

A body file is named `<model-id>-<model>.json` with the model string lower-cased and every run of other characters
turned into `-`: `384-ilce-1.json` (A1), `375-ilce-7rm4.json`. The model ID is Sony's (maker note `0xB001`, the table
in `docs/03`); the loader checks that the file name starts with the `model_id` inside.

## Body file (`noise-prior.schema.json`)

```json
{
  "schema_version": 1,
  "model_id": 384, "model": "ILCE-1", "body": "Sony A1",
  "regimes": [
    { "name": "low-gain", "iso_min": 100, "iso_max": 499,
      "a": { "alpha": 0.98, "beta": -14.9, "sigma_log": 0.25, "n_isos": 7, "n_files": 9, "rms_log_residual": 0.04 },
      "b": { "gamma": 0.5, "beta": -21.6, "sigma_rel": 0.4, "n_isos": 7, "n_files": 9, "rms_log_residual": 0.04 } },
    { "name": "high-gain", "iso_min": 500, "iso_max": null, "a": { "...": "..." }, "b": null }
  ],
  "provenance": { "kind": "...", "method": "...", "script": "...", "files": [ { "file": "A1_800.ARW", "iso": 800, "gain_iso": 800, "regime": 1, "a": 1.2e-3, "b": 3e-8, "coverage": 0.8 } ], "excluded": [], "notes": [] }
}
```

* `gain ISO` is the maker-note ISO with extended low ISOs mapped to the base gain: `max(ISO, base_iso)`, `base_iso` 100 unless
  `gain-regimes.json` says otherwise (A9 III: 250).
* `a(ISO) = exp(alpha ln(gain ISO) + beta)` with a prior sigma of `sigma_log` in `ln a` (0.25). Present only for regimes
  with files at 5 or more ISOs; `alpha` must lie in [0.6, 1.4] (the gain is proportional to ISO to first order).
* `b(ISO) = exp(beta + gamma ln(gain ISO))` with a relative prior sigma `sigma_rel` (0.40). Present only for regimes with files
  at 3 or more ISOs. `gamma` lies in [0, 2]: `gamma = 0` is a constant per regime (the form `docs/06` names); read noise
  referred to the output grows with the gain, so the fit allows up to the square of it.
* A regime lacking `a` or `b` uses the class prior for that parameter (`ym_noise::priors::class_prior`).
* `regimes` are sorted by ISO; the regime of a file is the one whose `iso_min <= gain ISO <= iso_max` (`null` = open).
* Numbers are statistics of the estimator's per-file output; `provenance.files` lists them with the file names (not the files).
  Files are kept only when the estimator's smooth-region bins cover at least 60% of the intensity range and the raw is not
  flagged as filtered.

## `gain-regimes.json` (`gain-regimes.schema.json`)

One row per body: `switch_iso` (the high-gain regime starts here; `null` = no break), `base_iso` (`null` = 100),
`processed_hi_iso_above` (ISOs above this widen the prior; `null` = not known) and `source`. The values are those of the
`docs/06` table (A1 500, A1 II 500, A7 III 640, A7 IV ~400, A7 V 1000, A7R IV 320, A7R V 320, A7S III 1600, A7C 640,
A7C II 400, A9 500, A9 II 640, A9 III 800 with base 250) and the processed-ISO thresholds of the same section (16000 for A1,
A7R V and A9 III; 64000 for A7 III and A7 IV; 128000 for A7S III). They come from public measurements quoted in
`research/notes/noise-estimation.md` and are unconfirmed.

## Loading

`ym_noise::priors::PriorStore::load_dir(<data>/cameras/noise)` reads the table and every body file;
`PriorStore::prior_for(&CameraContext { model_id, model, iso, class })` returns the prior for one file. No body file
exists yet, so every body gets the class prior (derived from physics with the assumptions in
`ym_noise::priors::ClassAssumptions`, one stop wide).

Regenerate: `cargo build --release -p ym-noise --example estimate_arw`, then
`python3 -m training.noise.public_priors --corpus testdata/samples --plots testdata/out/noise`.
