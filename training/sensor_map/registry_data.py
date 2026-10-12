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
SIBLING_CHECK = "pending: no converted samples available"


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
    _family(390, "ILCE-7RM5", "Sony A7R V", "sony-ilce-7rm4"),
    _family(396, "ILCE-7CR", "Sony A7CR", "sony-ilce-7rm4"),
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

EXIFTOOL_SOURCE = "exiftool/exiftool lib/Image/ExifTool/Sony.pm @ 2200871d9cef988051d2a99d67df3bda6cbb30a8 (SonyModelID table)"


def body_name_of_profile(profile_id: str) -> str:
    """Display body of the measured camera behind a profile ("Sony A7 III")."""
    return PROFILE_BY_ID[profile_id].body


def siblings_of(profile_id: str) -> list[Body]:
    """Same-sensor bodies that share a profile (their ColorMatrix check is pending)."""
    return [b for b in BODIES if b.profile == profile_id and b.match == "sensor_family"]


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


_self_check()
