"""A SHA-256 of what a JSON document says, not of how it is written.

The bytes of ``profile.json`` change whenever its ``provenance`` block does (the commit of the fit
script, the hashes of its sources, the results of the sibling checks), although not one number the
runtime reads has moved. A test that pins the bytes then fails after a harmless rebuild of the
registry. This digest ignores the writing (white space, ``1`` against ``1.0``, the order of keys) and
can leave out top-level keys, so a reference file can say "made from this profile, provenance aside".

The Rust side (``crates/ym-color/tests/spectral.rs``, ``content_digest``) feeds the same bytes:

* ``null`` is ``00``, ``false`` is ``01``, ``true`` is ``02``;
* a number is ``03`` and its 8 bytes as a little-endian IEEE double (an integer is converted first);
* a string is ``04``, its length in UTF-8 bytes as a little-endian u32, and the bytes;
* an array is ``05``, its length as a little-endian u32, and its elements in order;
* an object is ``06``, its number of keys as a little-endian u32, and for each key in sorted order
  (UTF-8 byte order) the key as a string and then its value.
"""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Iterable
from typing import Any


# The top-level key of a ``profile.json`` that no reader of its numbers depends on.
PROFILE_SKIPS = ("provenance",)


def _feed(h: "hashlib._Hash", value: Any) -> None:
    if value is None:
        h.update(b"\x00")
    elif value is False:
        h.update(b"\x01")
    elif value is True:
        h.update(b"\x02")
    elif isinstance(value, (int, float)):
        h.update(b"\x03" + struct.pack("<d", float(value)))
    elif isinstance(value, str):
        raw = value.encode("utf-8")
        h.update(b"\x04" + struct.pack("<I", len(raw)) + raw)
    elif isinstance(value, (list, tuple)):
        h.update(b"\x05" + struct.pack("<I", len(value)))
        for item in value:
            _feed(h, item)
    elif isinstance(value, dict):
        h.update(b"\x06" + struct.pack("<I", len(value)))
        for key in sorted(value, key=lambda k: k.encode("utf-8")):
            _feed(h, key)
            _feed(h, value[key])
    else:
        raise TypeError(f"not a JSON value: {type(value).__name__}")


def digest(value: Any, skip_top_level: Iterable[str] = ()) -> str:
    """The hex SHA-256 of ``value`` (a parsed JSON document) without the listed top-level keys."""
    skip = set(skip_top_level)
    if skip:
        if not isinstance(value, dict):
            raise TypeError("keys can only be left out of an object")
        value = {k: v for k, v in value.items() if k not in skip}
    h = hashlib.sha256()
    _feed(h, value)
    return h.hexdigest()
