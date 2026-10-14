"""The camera table of docs/03 as data: profiles, and which Sony model ID gets which one.

Single source of truth for ``build_profiles`` (profile files, sibling-check records) and
``build_registry`` (``data/cameras/registry.json``). Change the table here and in
``docs/03-camera-profiles.md`` together.

Model IDs are Sony's maker-note ``SonyModelID`` (tag 0xB001). All IDs and model strings below
were checked against ExifTool's ``Sony.pm`` (exiftool/exiftool @ 2200871d9cef, ``%sonyModelID``
table, read as a data source only): every ID maps to exactly the model string given here.
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_PROFILE = "sony-ilce-7rm4"

#: docs/03: a same-sensor pairing ships only if the sibling's ColorMatrix is within this of the profile body's.
SIBLING_LIMIT = 0.02
#: The independent tool of the check (AGENTS.md V1 data rule: no Adobe DNG Converter samples in V1).
SIBLING_TOOL = (
    "LibRaw 0.22.1 camera matrix (rgb_xyz_matrix, D65 ColorMatrix keyed by the Sony model ID) via rawpy 0.27.1, "
    "rows normalised to sum 1; crates/ym-profiles/tests/py/sibling_matrices.py"
)


@dataclass(frozen=True)
class Profile:
    """One measured body (Weta curves from rawtoaces-data)."""

    id: str
    display_name: str   # short name shown in the app: "A7R IV"
    model: str          # Sony model string of the measured body: "ILCE-7RM4"
    body: str           # long display name: "Sony A7R IV"

    @property
    def curves_name(self) -> str:
        """Key into ``ssf_core.load_cameras()``."""
        return f"Sony {self.model}"

    @property
    def curves_file(self) -> str:
        return f"Sony_{self.model}_380_780_5.json"

    @property
    def spectral_curves(self) -> str:
        """Path of the curves relative to ``data/cameras/``."""
        return f"sony/{self.curves_file}"


PROFILES: tuple[Profile, ...] = (
    Profile("sony-ilce-7m3", "A7 III", "ILCE-7M3", "Sony A7 III"),
    Profile("sony-ilce-7m4", "A7 IV", "ILCE-7M4", "Sony A7 IV"),
    Profile("sony-ilce-7cm2", "A7C II", "ILCE-7CM2", "Sony A7C II"),
    Profile("sony-ilce-7rm2", "A7R II", "ILCE-7RM2", "Sony A7R II"),
    Profile("sony-ilce-7rm3", "A7R III", "ILCE-7RM3", "Sony A7R III"),
    Profile("sony-ilce-7rm4", "A7R IV", "ILCE-7RM4", "Sony A7R IV"),
    Profile("sony-ilce-7sm2", "A7S II", "ILCE-7SM2", "Sony A7S II"),
    Profile("sony-ilce-7sm3", "A7S III", "ILCE-7SM3", "Sony A7S III"),
    Profile("sony-ilce-9", "A9", "ILCE-9", "Sony A9"),
    Profile("sony-ilce-6400", "A6400", "ILCE-6400", "Sony A6400"),
    Profile("sony-dsc-rx100m4", "RX100 IV", "DSC-RX100M4", "Sony RX100 IV"),
)
PROFILE_BY_ID = {p.id: p for p in PROFILES}


@dataclass(frozen=True)
class Body:
    """One Sony model ID and the profile it gets."""

    model_id: int
    model: str          # Model string as the camera writes it (IFD0 0x0110) and ExifTool names it
    body: str           # display name
    profile: str        # profile id
    match: str          # "exact" | "sensor_family" | "default"


def _exact(model_id: int, model: str, body: str, profile: str) -> Body:
    return Body(model_id, model, body, profile, "exact")


def _family(model_id: int, model: str, body: str, profile: str) -> Body:
    return Body(model_id, model, body, profile, "sensor_family")


def _default(model_id: int, model: str, body: str) -> Body:
    return Body(model_id, model, body, DEFAULT_PROFILE, "default")


# Order follows docs/03 "Which profile each camera gets".
BODIES: tuple[Body, ...] = (
    _exact(363, "ILCE-7M3", "Sony A7 III", "sony-ilce-7m3"),
    _family(381, "ILCE-7C", "Sony A7C", "sony-ilce-7m3"),
    _exact(388, "ILCE-7M4", "Sony A7 IV", "sony-ilce-7m4"),
    _exact(397, "ILCE-7CM2", "Sony A7C II", "sony-ilce-7cm2"),
    _exact(347, "ILCE-7RM2", "Sony A7R II", "sony-ilce-7rm2"),
    _exact(362, "ILCE-7RM3", "Sony A7R III", "sony-ilce-7rm3"),
    _family(386, "ILCE-7RM3A", "Sony A7R IIIA", "sony-ilce-7rm3"),
    _exact(375, "ILCE-7RM4", "Sony A7R IV", "sony-ilce-7rm4"),
    _family(387, "ILCE-7RM4A", "Sony A7R IVA", "sony-ilce-7rm4"),
    _default(390, "ILCE-7RM5", "Sony A7R V"),
    _default(396, "ILCE-7CR", "Sony A7CR"),
    _exact(350, "ILCE-7SM2", "Sony A7S II", "sony-ilce-7sm2"),
    _exact(383, "ILCE-7SM3", "Sony A7S III", "sony-ilce-7sm3"),
    _family(385, "ILME-FX3", "Sony FX3", "sony-ilce-7sm3"),
    _family(393, "ZV-E1", "Sony ZV-E1", "sony-ilce-7sm3"),
    _exact(358, "ILCE-9", "Sony A9", "sony-ilce-9"),
    _family(376, "ILCE-9M2", "Sony A9 II", "sony-ilce-9"),
    _exact(371, "ILCE-6400", "Sony A6400", "sony-ilce-6400"),
    _family(379, "ILCE-6100", "Sony A6100", "sony-ilce-6400"),
    _family(378, "ILCE-6600", "Sony A6600", "sony-ilce-6400"),
    _family(402, "ILCE-6400A", "Sony A6400A", "sony-ilce-6400"),
    _family(403, "ILCE-6100A", "Sony A6100A", "sony-ilce-6400"),
    _exact(341, "DSC-RX100M4", "Sony RX100 IV", "sony-dsc-rx100m4"),
    _default(384, "ILCE-1", "Sony A1"),
    _default(400, "ILCE-1M2", "Sony A1 II"),
    _default(392, "ILCE-9M3", "Sony A9 III"),
    _default(407, "ILCE-7M5", "Sony A7 V"),
    _default(410, "ILCE-7RM6", "Sony A7R VI"),
)
BODY_BY_ID = {b.model_id: b for b in BODIES}

EXIFTOOL_SOURCE = "exiftool/exiftool lib/Image/ExifTool/Sony.pm @ 2200871d9cef988051d2a99d67df3bda6cbb30a8 (SonyModelID table)"


def body_name_of_profile(profile_id: str) -> str:
    """Display body of the measured camera behind a profile ("Sony A7 III")."""
    return PROFILE_BY_ID[profile_id].body


def siblings_of(profile_id: str) -> list[Body]:
    """Same-sensor bodies that share a profile (each has a recorded ColorMatrix check)."""
    return [b for b in BODIES if b.profile == profile_id and b.match == "sensor_family"]


#: The measured result of every pairing that was checked: model ID -> largest per-coefficient difference
#: between the sibling's and the profile body's LibRaw ColorMatrix after row normalisation
#: (``crates/ym-profiles/tests/py/sibling_matrices.py``, which also verifies these numbers with ``--verify``).
#: ``0.0`` means LibRaw holds the same matrix for both bodies.
SIBLING_MAX_ABS_DIFF: dict[int, float] = {
    381: 0.0,     # ILCE-7C     -> ILCE-7M3
    386: 0.0,     # ILCE-7RM3A  -> ILCE-7RM3
    387: 0.0,     # ILCE-7RM4A  -> ILCE-7RM4
    390: 0.0934,  # ILCE-7RM5   -> ILCE-7RM4  (fails: a newer sensor generation with its own matrix)
    396: 0.0934,  # ILCE-7CR    -> ILCE-7RM4  (fails)
    385: 0.0,     # ILME-FX3    -> ILCE-7SM3
    393: 0.0,     # ZV-E1       -> ILCE-7SM3
    376: 0.0,     # ILCE-9M2    -> ILCE-9
    379: 0.0,     # ILCE-6100   -> ILCE-6400
    378: 0.0,     # ILCE-6600   -> ILCE-6400
    402: 0.0,     # ILCE-6400A  -> ILCE-6400
    403: 0.0,     # ILCE-6100A  -> ILCE-6400
}

#: Bodies that were considered as same-sensor siblings and failed the check. They keep the profile of the
#: pairing (the A7R IV profile is the default, so rendered bytes are unchanged) but with match kind ``default``;
#: the failed check is recorded in the profile's provenance and in the body's registry notes (docs/03).
REJECTED_SIBLINGS: dict[int, str] = {390: "sony-ilce-7rm4", 396: "sony-ilce-7rm4"}


def rejected_siblings_of(profile_id: str) -> list[Body]:
    """Bodies whose same-sensor pairing with ``profile_id`` failed the ColorMatrix check."""
    by_id = {b.model_id: b for b in BODIES}
    return [by_id[i] for i, pid in REJECTED_SIBLINGS.items() if pid == profile_id]


def sibling_check_result(model_id: int) -> str:
    """The text recorded as ``result`` of the provenance check of one pairing."""
    diff = SIBLING_MAX_ABS_DIFF[model_id]
    if diff <= SIBLING_LIMIT:
        return f"pass: max |diff| {diff:.4f} per coefficient (limit {SIBLING_LIMIT}); {SIBLING_TOOL}"
    return (
        f"fail: max |diff| {diff:.4f} per coefficient (limit {SIBLING_LIMIT}); {SIBLING_TOOL}. "
        "The pairing does not ship: the body has match kind default and still renders with this profile as the stand-in"
    )


@dataclass(frozen=True)
class Sensor:
    """Published sensor geometry of one body (spec ``docs/algorithms/03`` section 2.2).

    ``width_mm`` is the width of the sensor as the maker states it; ``width_px`` is the width of the largest
    3:2 still image the camera records from the full sensor (Sony's "Image size", full frame L), which is the
    horizontal count of the effective pixels. A crop mode does not change the pitch. ``kind`` says how the
    numbers were read: ``maker`` for a Sony specification page quoted by the search excerpt, ``restated`` for
    a retailer or reference site that restates Sony's specification (the Sony page itself was not seen).
    """

    width_mm: float
    width_px: int
    kind: str
    source: str

    @property
    def pitch_um(self) -> float:
        return 1000.0 * self.width_mm / self.width_px


#: How the numbers below were read. Sony's hosts are not reachable from the build environment and the fetch tool
#: could not open them, so each entry was read from the excerpt a web search returned for the Sony page; the
#: pages themselves were not fetched. Date of the searches: 2026-10-13.
SENSOR_READ_NOTE = (
    "read 2026-10-13 from web-search excerpts of the pages named; the pages were not fetched "
    "(Sony hosts are not reachable from the build environment); verify against the page before relying on the value"
)


def _maker(width_mm: float, width_px: int, *pages: str) -> Sensor:
    return Sensor(width_mm, width_px, "maker", "Sony specification page(s): " + "; ".join(pages))


def _restated(width_mm: float, width_px: int, *pages: str) -> Sensor:
    return Sensor(width_mm, width_px, "restated", "Sony specification as restated by: " + "; ".join(pages))


#: Sensor geometry by model ID. A body without an entry has no published pixel width that could be read
#: (ILME-FX3: Sony's page gives its stills only as "12M"), and uses the EXIF crop-factor fallback.
SENSORS: dict[int, Sensor] = {
    # --- 35.6 x 23.8 mm, 6000 x 4000 ---
    363: _restated(35.6, 6000, "en.wikipedia.org/wiki/Sony_%CE%B17_III (35.6 x 23.8 mm, 6000 x 4000)",
                   "store.sony.co.th ILCE-7M3 specifications (35mm full frame, 24.2 MP effective)"),
    381: _maker(35.6, 6000, "sony.co.th/en/interchangeable-lens-cameras/products/ilce-7c/spec (L 6000 x 4000 (24M))",
                "store.sony.com.sg/blogs/6821926863003_40202286170267/6821926863003__specifications (35.6 x 23.8 mm)"),
    358: _maker(35.6, 6000, "sony.com.tw/en/electronics/interchangeable-lens-cameras/ilce-9/specifications "
                "(35mm full frame 35.6 x 23.8 mm, 24.2 MP effective; 16:9 L 6000 x 3376)",
                "en.wikipedia.org/wiki/Sony_%CE%B19 (6000 x 4000 at 3:2)"),
    376: _maker(35.6, 6000, "sony.co.in/electronics/interchangeable-lens-cameras/ilce-9m2/specifications",
                "sony.com.sg/electronics/interchangeable-lens-cameras/ilce-9m2/specifications "
                "(35.6 x 23.8 mm; L 6000 x 4000 (24M))"),
    392: _maker(35.6, 6000, "sony.com.sg/electronics/interchangeable-lens-cameras/ilce-9m3/specifications "
                "(L 6000 x 4000 (24M))", "store.sony.com.sg/blogs/7805100327067_43884466143387 (35.6 x 23.8 mm)"),
    # --- 35.6 x 23.8 mm, 4240 x 2832 ---
    350: _maker(35.6, 4240, "sony.co.id/en/electronics/interchangeable-lens-cameras/ilce-7sm2/specifications "
                "(35mm full frame 35.6 x 23.8 mm, L 4240 x 2832 (12M))"),
    383: _restated(35.6, 4240, "bhphotovideo.com Sony a7S III (12.1 MP, 4240 x 2832)",
                   "en.wikipedia.org/wiki/Sony_%CE%B17S_III (35.6 x 23.8 mm; Sony product page: 35mm full frame "
                   "(35.6x23.8mm))"),
    393: _maker(35.6, 4240, "sony-mea.com/en/electronics/support/e-mount-body-zv-e-series/zv-e1/specifications "
                "(35.6 x 23.8 mm, L 4240 x 2832 (12 M))"),
    # --- 35.9 x 23.9 mm, 7008 x 4672 ---
    388: _restated(35.9, 7008, "bhphotovideo.com Sony a7 IV (35.9 x 23.9 mm, 7008 x 4672)",
                   "imaging-resource.com Sony a7 IV (35.90 x 23.90 mm, 7008 x 4672)"),
    397: _restated(35.9, 7008, "bhphotovideo.com Sony a7C II (35.9 x 23.9 mm, 7008 x 4672)"),
    407: _maker(35.9, 7008, "sony.com.au/electronics/interchangeable-lens-cameras/ilce-7m5/specifications "
                "(35.9 x 23.9 mm, L 7008 x 4672 (33 M))"),
    # --- 35.9 x 24.0 mm ---
    347: _maker(35.9, 7952, "sony.co.id/en/electronics/interchangeable-lens-cameras/ilce-7rm2/specifications "
                "(35.9 x 24.0 mm, L 7952 x 5304 (42M))"),
    362: _maker(35.9, 7952, "sony.com.tw/en/electronics/interchangeable-lens-cameras/ilce-7rm3/specifications "
                "(35.9 x 24.0 mm, L 7952 x 5304 (42M))"),
    386: _maker(35.9, 7952, "sony.ca/en/electronics/interchangeable-lens-cameras/ilce-7rm3a/specifications "
                "(35.9 x 24.0 mm, L 7952 x 5304 (42M))"),
    384: _restated(35.9, 8640, "en.wikipedia.org/wiki/Sony_%CE%B11 (35.9 x 24.0 mm, 8640 x 5760)",
                   "imaging-resource.com Sony a1 (35.9 x 24.0 mm, 8640 x 5760)",
                   "sony.co.th/en/interchangeable-lens-cameras/products/ilce-1/spec (50.1 MP effective)"),
    400: _maker(35.9, 8640, "sony.com.tw/en/electronics/interchangeable-lens-cameras/ilce-1m2/specifications "
                "(35.9 x 24.0 mm, L 8640 x 5760 (50 M))"),
    410: _maker(35.9, 9984, "sony.co.th/en/electronics/interchangeable-lens-cameras/ilce-7rm6/specifications "
                "(35.9 x 24.0 mm, L 9984 x 6656 (66 M))"),
    # --- 35.7 x 23.8 mm, 9504 x 6336 ---
    375: _maker(35.7, 9504, "sony.com.tw/en/electronics/interchangeable-lens-cameras/ilce-7rm4/specifications "
                "(35.7 x 23.8 mm, L 9504 x 6336 (60M))"),
    387: _maker(35.7, 9504, "sony.ca/en/electronics/interchangeable-lens-cameras/ilce-7rm4a/specifications "
                "(35.7 x 23.8 mm, L 9504 x 6336 (60M))"),
    390: _maker(35.7, 9504, "sony.co.th/en/interchangeable-lens-cameras/products/ilce-7rm5/spec "
                "(35.7 x 23.8 mm, L 9504 x 6336 (60M))"),
    396: _maker(35.7, 9504, "sony.com.my/electronics/interchangeable-lens-cameras/ilce-7cr/specifications "
                "(35.7 x 23.8 mm, L 9504 x 6336 (60 M))"),
    # --- APS-C 23.5 x 15.6 mm, 6000 x 4000 ---
    371: _maker(23.5, 6000, "store.sony.co.th ILCE-6400 specifications (APS-C type 23.5 x 15.6 mm, "
                "L 6000 x 4000 (24M))"),
    379: _maker(23.5, 6000, "sony.co.id/en/electronics/interchangeable-lens-cameras/ilce-6100/specifications "
                "(APS-C type 23.5 x 15.6 mm, 24.2 MP, L 6000 x 4000)"),
    378: _maker(23.5, 6000, "sony.com.sg/electronics/interchangeable-lens-cameras/ilce-6600/specifications "
                "(APS-C type 23.5 x 15.6 mm, L 6000 x 4000 (24M))"),
    402: _maker(23.5, 6000, "sonylatvija.com/electronics/interchangeable-lens-cameras/ilce-6400a/specifications "
                "(APS-C type 23.5 x 15.6 mm, L 6000 x 4000 (24M))"),
    403: _maker(23.5, 6000, "sony.fr/electronics/appareils-photo-a-objectifs-interchangeables/ilce-6100a/specifications "
                "(APS-C 23.5 x 15.6 mm, 24.2 MP effective, 3:2 L 24M); the 6000 px width is the ILCE-6100 page's "
                "(sony.ca/en/electronics/interchangeable-lens-cameras/ilce-6100/specifications)"),
    # --- 1.0-type 13.2 x 8.8 mm ---
    341: _maker(13.2, 5472, "sony-asia.com/electronics/support/compact-cameras-dsc-rx-series/dsc-rx100m4/specifications "
                "(1.0-type 13.2 x 8.8 mm); sony.co.in/electronics/support/compact-cameras-dsc-rx-series/dsc-rx100m4/"
                "specifications (5,472 x 3,648 (20M))"),
}


def _self_check() -> None:
    ids = [b.model_id for b in BODIES]
    assert len(ids) == len(set(ids)), "duplicate model id"
    models = [b.model for b in BODIES]
    assert len(models) == len(set(models)), "duplicate model string"
    assert {b.profile for b in BODIES} <= set(PROFILE_BY_ID)
    for p in PROFILES:
        exact = [b for b in BODIES if b.profile == p.id and b.match == "exact"]
        assert len(exact) == 1 and exact[0].model == p.model, p.id
        assert exact[0].body == p.body, p.id


def _sensor_self_check() -> None:
    by_id = {b.model_id: b for b in BODIES}
    assert set(SENSORS) <= set(by_id), "sensor entry for an unknown model ID"
    for model_id, sensor in SENSORS.items():
        assert 0.5 <= sensor.pitch_um <= 20.0, (by_id[model_id].model, sensor.pitch_um)
        assert sensor.kind in ("maker", "restated")
    # Bodies that share a sensor share its pitch (within 2 %): a typo in one entry shows here.
    for b in BODIES:
        if b.match != "sensor_family" or b.model_id not in SENSORS:
            continue
        exact = next(e for e in BODIES if e.profile == b.profile and e.match == "exact")
        if exact.model_id in SENSORS:
            ratio = SENSORS[b.model_id].pitch_um / SENSORS[exact.model_id].pitch_um
            assert abs(ratio - 1.0) < 0.02, (b.model, exact.model, ratio)
    pairings = {b.model_id for b in BODIES if b.match == "sensor_family"} | set(REJECTED_SIBLINGS)
    assert set(SIBLING_MAX_ABS_DIFF) == pairings, "every same-sensor pairing (shipped or rejected) needs a recorded check"
    for model_id, profile_id in REJECTED_SIBLINGS.items():
        assert by_id[model_id].match == "default" and SIBLING_MAX_ABS_DIFF[model_id] > SIBLING_LIMIT
        assert profile_id == DEFAULT_PROFILE
    for model_id, diff in SIBLING_MAX_ABS_DIFF.items():
        if model_id not in REJECTED_SIBLINGS:
            assert diff <= SIBLING_LIMIT, (by_id[model_id].model, diff)


_self_check()
_sensor_self_check()
