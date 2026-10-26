"""Write ``data/alexa/target.json``: the ALEXA target's curves path, version and provenance.

``docs/03`` (last paragraph): the ALEXA target uses the same structure as a camera profile
(curves, version, provenance); a second target would slot in beside it.

    python3 -m training.sensor_map.build_target
"""

from __future__ import annotations

import argparse
import hashlib
import sys

from training.alexa import push_decisions as pd
from training.common import jsonfmt, paths

#: 1.0.0: the estimate without the hue push; 1.1.0: the push of 1.5 degrees (owner decision 2026-10-25, "Move ALEXA
#: estimate"); 1.2.0: the push of 4.5 degrees, past the matrix limit (owner decision 2026-10-26, "4.5°").
TARGET_VERSION = "1.2.0"


def sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build() -> dict:
    d = paths.ALEXA_DIR
    deg = pd.DEFAULT_PUSH_DEG
    record = pd.DECISIONS[deg]
    beyond = record["beyond_matrix_limit"]
    assert beyond is not None, "the shipped push is past the matrix limit: its decision must be recorded"
    lo, hi = beyond["ensemble_matrix_rms_range"]
    return {
        "id": "alexa-alev3-estimate",
        "version": TARGET_VERSION,
        "display_name": "ALEXA ALEV III (estimate)",
        "kind": "estimate",
        "spectral_curves": "alexa/alexa_alev3_estimate_380_780_5.json",
        "ensemble_csv": "alexa/alexa_alev3_estimate_380_780_5.csv",
        "variants_csv": "alexa/variants.csv",
        "provenance": {
            "summary": "Estimated spectral sensitivity of the ALEXA ALEV III sensor (SUP 3.x processing; "
                       "ALEXA Classic through Mini LF). An estimate, not a measurement.",
            "method": "Curves whose own optimal Academy P-2013-001-style matrices equal ARRI's published "
                      "native-to-ACES AP0 matrices at 17 CCTs (2000-11000 K), under a 4-component PCA prior "
                      "built from the 52 measured camera curves in rawtoaces-data (all 52, penalty 1, "
                      "rawtoaces illuminants), moved by a hue push of " + f"{deg:g}" + " degrees (" + record["decision"] + ", " + record["reason"] + "): the "
                      "curve that renders the 47 orange, red and ochre training reflectances (Lab hue 25-95 "
                      "degrees, chroma above 12) " + f"{deg:g}" + " degrees of CIELAB hue toward yellow at 5200 K through "
                      "ARRI's matrix. Its matrix RMS is " + f"{beyond['matrix_rms']:.4f}" + " per coefficient (0.0067 without the "
                      "push), past the maximum of the 24-variant ensemble (" + f"{lo:.4f} to {hi:.4f}" + "): "
                      "beyond_matrix_limit, the owner's recorded decision (" + beyond["decision"] + ", matrix RMS "
                      "accepted " + f"{beyond['matrix_rms']:.4f}" + "; training/alexa/push_decisions.py), not a fit that ARRI's "
                      "matrices ask for. The "
                      "24 variants in variants.csv and the band in the ensemble CSV are without the push; the "
                      "estimate without it is `estimate_alexa --no-push`, the one with the push of 1.5 degrees "
                      "`estimate_alexa --push-deg 1.5`.",
            "inputs": [
                "ARRI ALEXA SUP 3.x native-to-ACES AP0 matrices, 17 CCTs, 13 Nov 2013 "
                "(aces-dev v1.3.1, ARRI-supplied, AMPAS licence, data/arri/AMPAS-LICENSE.md)",
                "52 measured camera spectral sensitivities and 190 training reflectances "
                "(rawtoaces-data, Apache-2.0)",
            ],
            "licence": "Apache-2.0 (ours), derived from AMPAS-licensed and Apache-2.0 data; see NOTICE",
            "fit_script": "training/alexa/estimate_alexa.py",
            "recommended_variant": {"prior": "all", "K": 4, "lam": 1.0, "conv": "rta"},
            "validation": {
                "method": "leave-one-camera-out over the 52 measured cameras (training/alexa/validate_loo.py, "
                          "checked by training/alexa/check_loo.py in .github/workflows/alexa-curve.yml); the "
                          "numbers are those of the estimator without the hue push, which has no held-out "
                          "counterpart (the push leaves the curves that agree best with the matrices on purpose)",
                "published": {
                    "rfse_median_pct": 2.2,
                    "rfse_p90_pct": 3.6,
                    "emulation_dE00_median": 0.23,
                    "emulation_dE00_p95_median_over_cameras": 0.63,
                },
                "tolerance": 0.1,
                "gate": "median RFSE <= 3 %, emulation dE00 median <= 0.3 (docs/13)",
            },
            "caveats": [
                "Expected accuracy 3-5 % RFSE for this camera, not the 2.2 % LOO median "
                "(research/alexa-ssf/README.md).",
                "The shipped curve is the estimate plus a hue push of 4.5 degrees (owner decision 2026-10-26, "
                "beyond_matrix_limit), 3.7 % RFSE from the estimate without it, and past the ensemble's "
                "matrix-consistency limit: it reproduces ARRI's published matrices less well (matrix RMS 0.0241 "
                "per coefficient, the ensemble 0.0064 to 0.0093; mean dE00 to scene colour with ARRI's matrices "
                "2.02, 14th of the 52 cameras and the estimate, against 1.46 and first without the push). The red channel is "
                "lowered at 560-585 nm (0.46 against 0.60 of peak at 580 nm) and by 0.02 to 0.05 at 615-660 nm "
                "and raised at 665-690 nm (0.11 against 0.03 at 680 nm), the green channel raised (by 0.22 at "
                "490 nm and 0.12 at 580-590 nm), so that oranges, reds and ochres render less crimson. The "
                "curve lies outside the 24-variant band at 146 of 243 samples, by at most 0.14 of peak (B at "
                "415 nm). It is a choice, not a measurement.",
                "Beyond about 660 nm the curve is mostly the prior; treat 660-720 nm as unknown.",
                "Represents ALEXA SUP 3.x / ALEV III processing; not the ALEXA 35 or REVEAL.",
                "Assumes ARRI made its matrices the P-2013-001 way; a different recipe adds error not "
                "captured by the leave-one-out test.",
            ],
            "sha256": {
                "alexa_alev3_estimate_380_780_5.json": sha256(paths.ALEXA_CURVES_JSON),
                "alexa_alev3_estimate_380_780_5.csv": sha256(d / "alexa_alev3_estimate_380_780_5.csv"),
                "variants.csv": sha256(d / "variants.csv"),
            },
        },
    }


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    text = jsonfmt.dumps(build())
    paths.ALEXA_TARGET_JSON.write_text(text)
    print("wrote", paths.ALEXA_TARGET_JSON.relative_to(paths.REPO_ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
