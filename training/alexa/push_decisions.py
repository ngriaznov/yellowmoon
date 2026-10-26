"""The owner's decisions about the ALEXA estimate's hue push (docs/03 *The ALEXA estimate*).

Plain data, no dependencies and no fetched sources, so that ``estimate_alexa`` (which writes the curve),
``sensor_map.build_target`` (which writes ``data/alexa/target.json``) and the tests read the same record.

* ``DEFAULT_PUSH_DEG``: the push the shipped data are made with (``estimate_alexa`` with no flag).
* ``DECISIONS``: every push the script is allowed to write, keyed by its degrees, with the date and the words of
  the owner's decision. ``--push-deg 1.5`` reproduces the estimate of 2026-10-25 and ``--no-push`` the one of
  2026-10-11, both byte for byte.
* ``beyond_matrix_limit``: the stopping rule is that a push's matrix RMS (its disagreement with ARRI's published
  matrices, per coefficient) stays within the maximum of the 24-variant ensemble (0.0093). A push past that limit
  is written only when it has a record here, naming the decision and the matrix RMS the owner accepted; the run
  refuses the same push if it reaches more than that RMS, and refuses any other push past the limit.
"""

from __future__ import annotations

#: The push of the shipped data, in degrees of CIELAB hue toward yellow (the owner's decision of 2026-10-26: "4.5°").
DEFAULT_PUSH_DEG = 4.5

#: The estimate without the push: made on 2026-10-11, shipped until 2026-10-25.
DATE_NOPUSH = "2026-10-11"

DECISIONS: dict[float, dict] = {
    1.5: {
        "date": "2026-10-25",
        "decision": '"Move ALEXA estimate" (candidate TM15), owner decision 2026-10-25',
        "reason": "oranges, ochres and reds rendered crimson against the camera's own picture",
        "beyond_matrix_limit": None,
    },
    4.5: {
        "date": "2026-10-26",
        "decision": '"4.5°", owner decision 2026-10-26',
        "reason": "after the push of 1.5 degrees of 2026-10-25 (\"Move ALEXA estimate\") the owner still saw oranges and ochres too red",
        "beyond_matrix_limit": {
            "decision": 'owner decision 2026-10-26: "4.5°"',
            #: The matrix RMS the owner accepted, per coefficient, rounded to 4 decimals (the run measured 0.02407).
            "matrix_rms": 0.0241,
            "ensemble_matrix_rms_range": [0.0064, 0.0093],
            "costs_the_owner_knew": [
                "the curve no longer agrees with ARRI's published matrices as well as the ensemble does "
                "(matrix RMS 0.0241 against 0.0064 to 0.0093)",
                "mean ΔE00 to scene colour with ARRI's matrices on the 99 CIE CES: 2.02, 14th of the 52 cameras and "
                "the estimate (16th in the list the script prints, which also puts the estimate without the push and "
                "the mean camera ahead of it); 1.55 and first at 1.5 degrees",
                "the A7R IV sensor-map gate has 7 of 9 lights over (6 at 1.5 degrees), 81 of 99 lights over "
                "across the 11 bodies, no body passes all nine",
            ],
        },
    },
}


def decision(deg: float) -> dict | None:
    """The record of the push of ``deg`` degrees, or ``None`` when the owner has not decided on that push."""
    for d, rec in DECISIONS.items():
        if abs(d - deg) < 1e-9:
            return rec
    return None


def accepts(deg: float, matrix_rms: float, ensemble_limit: float) -> bool:
    """True when a push of ``deg`` degrees with this matrix RMS may be written.

    Within the ensemble's maximum any push is allowed; past it only a push with a ``beyond_matrix_limit``
    record, and only up to the matrix RMS that record names.
    """
    if matrix_rms <= ensemble_limit:
        return True
    rec = decision(deg)
    beyond = rec and rec["beyond_matrix_limit"]
    return bool(beyond) and round(matrix_rms, 4) <= beyond["matrix_rms"]
