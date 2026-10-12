"""Write ``data/alexa/target.json``: the ALEXA target's curves path, version and provenance.

``docs/03`` (last paragraph): the ALEXA target uses the same structure as a camera profile
(curves, version, provenance); a second target would slot in beside it.

    python3 -m training.sensor_map.build_target
"""

from __future__ import annotations

import argparse
import hashlib
import sys

from training.common import jsonfmt, paths

TARGET_VERSION = "1.0.0"


def sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build() -> dict:
    d = paths.ALEXA_DIR
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
                      "rawtoaces illuminants). 24-variant ensemble in variants.csv.",
            "inputs": [
                "ARRI ALEXA SUP 3.x native-to-ACES AP0 matrices, 17 CCTs, 13 Nov 2013 "
                "(aces-dev v1.3.1, ARRI-supplied, AMPAS licence, data/arri/AMPAS-LICENSE.md)",
                "52 measured camera spectral sensitivities and 190 training reflectances "
                "(rawtoaces-data, Apache-2.0)",
            ],
            "licence": "Apache-2.0 (ours), derived from AMPAS-licensed and Apache-2.0 data; see NOTICE",
            "fit_script": "research/alexa-ssf/scripts/estimate_alexa.py "
                          "(copied unchanged to training/alexa/estimate_alexa.py)",
            "recommended_variant": {"prior": "all", "K": 4, "lam": 1.0, "conv": "rta"},
            "validation": {
                "method": "leave-one-camera-out over the 52 measured cameras (training/alexa/validate_loo.py, "
                          "checked by training/alexa/check_loo.py in .github/workflows/alexa-curve.yml)",
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
