"""What gets generated: the recipes of the default (CI) set and of the ``--full`` set.

A recipe is one raw mosaic: scene, light, body, ISO, CFA phase and bit depth, plus the ARW variants ``golden synth``
writes from it. The recipe list is the single source of file names; the Rust writer and ``tools/golden/manifest.json``
derive theirs from it (``manifest()``), and a test keeps the committed manifest in step.

Default set (about 37 mosaics, 61 ARW files, under 300 MB, a few minutes):

=========  =========================================================================================  ======
group      what                                                                                       mosaics
=========  =========================================================================================  ======
lights     A7R IV, ISO 400, all eight lights and the mixed scene, lossless L                          9
bodies     six bodies, D 6500 K, ISO 400: every 14-bit variant (uncompressed, cRAW, lossless L and    5 (+1)
           M/S) and a 12-bit mosaic written as packed 12-bit                                          6
iso        A7R IV, D 6500 K, ISO 100, 800, 3200, 12800                                                4
cfa        A7R IV, D 6500 K, ISO 400, GRBG, BGGR, GBRG: uncompressed, lossless L and cRAW             3
cross      the other five bodies under tungsten 2850 K and LED-B3, ISO 400, the body's usual variant  10
=========  =========================================================================================  ======

``--full`` adds the 6240 x 4160 full-size scene (performance), the complete light x body cross product at ISO 400
(every light under every body, the body's usual variant) and the evaluation scene (99 CIE CES, 24 Ohta and 15 NIST
patches in the chart; CC BY-SA data, so those outputs live in ``testdata/`` only).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from training.synth import spectra as sp

#: Variants written from a 14-bit mosaic and from a 12-bit mosaic.
VARIANTS_14 = ("uncompressed14", "craw", "lossless-l", "lossless-ms")
VARIANTS_12 = ("packed12",)
ALL_VARIANTS = VARIANTS_14 + VARIANTS_12

SCENE_SCALE = {"chart": 1.0, "fullsize": 5.0, "eval": 1.0}
#: First capture time; each recipe is 7 s after the previous one, so no two files form a burst.
BASE_TIME = (2026, 10, 12, 12, 0, 0)


@dataclass(frozen=True)
class Recipe:
    """One raw mosaic and the ARW variants to write from it."""

    scene: str  # "chart" | "fullsize" | "eval"
    light: str  # a SceneLight token
    body: str  # EXIF model
    iso: int
    cfa: str = "RGGB"
    bits: int = 14
    variants: tuple[str, ...] = ()
    ev_comp: float = 0.0
    group: str = ""

    @property
    def stem(self) -> str:
        """File stem of the mosaic and the sidecar: ``scene__light__model__isoN__cfa__14b``."""
        return f"{self.scene}__{self.light}__{self.body}__iso{self.iso}__{self.cfa.lower()}__{self.bits}b"

    @property
    def tier(self) -> str:
        return "synthetic-eval" if self.scene == "eval" else "synthetic"

    def arw_names(self) -> list[str]:
        return [f"{self.stem}__{v}.ARW" for v in self.variants]

    def to_json(self) -> dict:
        d = asdict(self)
        d["variants"] = list(self.variants)
        return d


def _merge(recipes: list[Recipe]) -> list[Recipe]:
    """One recipe per stem, the variants of repeated ones united (in ``ALL_VARIANTS`` order)."""
    by_stem: dict[str, Recipe] = {}
    for r in recipes:
        if r.stem in by_stem:
            old = by_stem[r.stem]
            union = tuple(v for v in ALL_VARIANTS if v in set(old.variants) | set(r.variants))
            by_stem[r.stem] = Recipe(**{**asdict(old), "variants": union})
        else:
            by_stem[r.stem] = r
    return list(by_stem.values())


def default_recipes() -> list[Recipe]:
    """The default set (CI): see the module docstring."""
    lights = list(sp.single_lights()) + ["mixed"]
    a7r4 = "ILCE-7RM4"
    others = [m for m in sp.BODIES if m != a7r4]
    out: list[Recipe] = []
    out += [Recipe("chart", lt, a7r4, 400, variants=("lossless-l",), group="lights") for lt in lights]
    for model in sp.BODIES:
        out.append(Recipe("chart", "D6500", model, 400, variants=VARIANTS_14, group="bodies"))
        out.append(Recipe("chart", "D6500", model, 400, bits=12, variants=VARIANTS_12, group="bodies"))
    out += [Recipe("chart", "D6500", a7r4, iso, variants=("lossless-l",), group="iso") for iso in (100, 800, 3200, 12800)]
    out += [
        Recipe("chart", "D6500", a7r4, 400, cfa=cfa, variants=("uncompressed14", "lossless-l", "craw"), group="cfa")
        for cfa in ("GRBG", "BGGR", "GBRG")
    ]
    for model in others:
        for lt in ("bb2850", "LEDB3"):
            out.append(Recipe("chart", lt, model, 400, variants=(sp.BODIES[model].typical_variant,), group="cross"))
    return _merge(out)


def full_extra_recipes() -> list[Recipe]:
    """What ``--full`` adds: the full-size scene, the rest of the light x body cross product and the evaluation scene."""
    a7r4, a7m3 = "ILCE-7RM4", "ILCE-7M3"
    out = [
        Recipe("fullsize", "D5500", a7m3, 100, variants=("uncompressed14", "lossless-l", "craw"), group="full"),
        Recipe("fullsize", "D5500", a7m3, 3200, variants=("lossless-l",), group="full"),
        Recipe("fullsize", "D5500", a7m3, 100, bits=12, variants=("packed12",), group="full"),
    ]
    lights = list(sp.single_lights())
    # every light under every body (merged with the default set's share of the product)
    out += [
        Recipe("chart", lt, model, 400, variants=(body.typical_variant,), group="cross-full")
        for model, body in sp.BODIES.items()
        for lt in lights + ["mixed"]
    ]
    out += [Recipe("eval", lt, a7r4, 100, variants=("lossless-l",), group="eval") for lt in lights]
    for model in (m for m in sp.BODIES if m != a7r4):
        for lt in ("D6500", "bb2850"):
            out.append(Recipe("eval", lt, model, 100, variants=("lossless-l",), group="eval"))
    return _merge(out)


def recipes(full: bool = False) -> list[Recipe]:
    """The default recipes, then (with ``full``) the extra ones; a stem in both appears once, in its first place, with
    the variants of both (so adding ``--full`` never changes a default file or its capture time)."""
    return _merge(default_recipes() + (full_extra_recipes() if full else []))


def capture_time(index: int) -> str:
    """EXIF DateTime string of recipe number ``index``."""
    import datetime as dt

    t = dt.datetime(*BASE_TIME) + dt.timedelta(seconds=7 * index)
    return t.strftime("%Y:%m:%d %H:%M:%S")


def manifest_set(recipe: Recipe, variant: str, default_files: set[tuple[str, str]]) -> str:
    """Which ``synthetic.sh`` run writes the file: ``default`` (the CI set), ``full`` (only with ``--full``: the
    full-size scene and the rest of the light x body product) or ``eval`` (``--full``, CIE data, local only).
    A golden runner that selects ``set == "default"`` must find exactly the files of a plain ``synthetic.sh`` run."""
    if recipe.scene == "eval":
        return "eval"
    return "default" if (recipe.stem, variant) in default_files else "full"


def manifest(full: bool = True) -> dict:
    """The synthetic part of ``tools/golden/manifest.json`` as a dict (``entries``), one per ARW file."""
    default_files = {(r.stem, v) for r in default_recipes() for v in r.variants}
    entries = []
    for r in recipes(full):
        body = sp.BODIES[r.body]
        for variant in r.variants:
            name = f"{r.stem}__{variant}.ARW"
            entries.append(
                {
                    "id": f"synthetic/{r.stem}__{variant}",
                    "tier": r.tier,
                    "set": manifest_set(r, variant, default_files),
                    "group": r.group,
                    "file": f"synthetic/arw/{name}",
                    "sidecar": f"synthetic/scenes/{r.stem}.json",
                    "scene": r.scene,
                    "light": r.light,
                    "body": r.body,
                    "model_id": body.model_id,
                    "iso": r.iso,
                    "cfa": r.cfa,
                    "bits": r.bits,
                    "variant": variant,
                    "settings": {},
                    "expected_hash": "",
                    "reference": "",
                }
            )
    return {"entries": entries}
