# training/noise/

M2-01: noise priors from public files (`docs/06-noise.md`, section "Priors from public files").

| File | Role |
|---|---|
| `public_priors.py` | runs the prior-free estimator (`crates/ym-noise/examples/estimate_arw.rs`) over a corpus folder, keeps estimates whose smooth-region bins cover at least 60% of the intensity range (and that are not flagged as filtered), fits `ln a = alpha ln ISO + beta` per gain regime where a body has files at 5 or more ISOs and `b = exp(beta) ISO^gamma` where it has 3 or more, writes `data/cameras/noise/<model-id>-<model>.json` with provenance, prints the fits and the bodies without enough data, and draws a residual plot per qualifying body |
| `test_public_priors.py` | tests of the fitting, selection and output logic on **synthetic estimate tables** (the corpus is unreachable in V1 builds); keeps `crates/ym-noise/tests/data/384-ilce-1.json`, a synthetic body file the Rust loader test reads |

```bash
cargo build --release -p ym-noise --example estimate_arw
python3 -m training.noise.public_priors --corpus testdata/samples --plots testdata/out/noise
python3 -m training.noise.public_priors --estimates estimates.jsonl --dry-run     # JSON lines from estimate_arw
python3 -m unittest training.noise.test_public_priors
```

Status: written and tested on synthetic tables. **Not run on the corpus**: `tools/fetch/samples.sh` could not reach
raw.pixls.us and RawDB when this was built. The example and the script were exercised end to end on nine ARW files that
sit in the git-ignored `testdata/real/files/` (provenance unrecorded, so they are a smoke test, not corpus data):
`estimate_arw` decodes and estimates eight of them (the ninth, a 12-bit DSLR-A300 file, is reported as unsupported by
`ym-raw`), and `public_priors.py --estimates` reads its lines. Seven of the eight have smooth bins covering less than 60%
of the intensity range, which is how ordinary photographs behave, so the script keeps them out of the fits; a corpus run
will need many files per body. Until it runs, `data/cameras/noise/` holds only `gain-regimes.json` and every body uses the
class prior.
