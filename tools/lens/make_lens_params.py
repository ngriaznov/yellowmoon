#!/usr/bin/env python3
"""Writes the hex of a darktable 4.6.1 `lens` module parameter blob (module version 10) that selects
the "embedded metadata" correction method, for `darktable_oracle.sh --lens-params`.

darktable-cli has no switch for the lens module and does not enable it by itself for Sony files,
but it accepts a history item. The blob is the module's `dt_iop_lens_params_t` (356 bytes, little
endian, every field 4 bytes except the two 128-byte strings); the layout was read off darktable's
public module source (`src/iop/lens.cc`, parameters v10) and then proven by use: with this blob
`darktable-cli` applies the correction, and the radial profiles it produces match the Sony model of
`lens_model.py` (see `record_oracle.py`).

  make_lens_params.py [--flags all|vignetting|distortion|ca|N] [--algorithm 2|1] OUT.hex

`--flags` selects the corrections (darktable's bit mask: 1 CA, 2 vignetting, 4 distortion; default
all = 7). `--algorithm` is the embedded-metadata algorithm version (2 is darktable 4.4's improved one
and the default of the module; 1 is the older one, kept to be able to reproduce its numbers).
"""
import argparse
import struct

FLAGS = {"all": 7, "ca": 1, "vignetting": 2, "distortion": 4}


def params_blob(flags=7, algorithm=2):
    """The 356-byte `dt_iop_lens_params_t`."""
    md_version = {1: 0, 2: 1}[algorithm]  # DT_IOP_LENS_EMBEDDED_METADATA_VERSION_1 = 0, _2 = 1
    b = struct.pack("<iii", 0, flags, 0)                         # method = embedded metadata, modify_flags, inverse
    b += struct.pack("<fffff", 1.0, 1.0, 0.0, 0.0, 1000.0)       # scale, crop, focal, aperture, distance
    b += struct.pack("<i", 1)                                    # target_geom = rectilinear
    b += bytes(128) + bytes(128)                                 # camera, lens
    b += struct.pack("<i", 0)                                    # tca_override
    b += struct.pack("<ff", 1.0, 1.0)                            # tca_r, tca_b
    b += struct.pack("<ffff", 1.0, 1.0, 1.0, 1.0)                # cor_dist_ft, cor_vig_ft, cor_ca_r_ft, cor_ca_b_ft
    b += struct.pack("<f", 0.0)                                  # scale_md_v1 (0: autoscale, v1 only)
    b += struct.pack("<i", md_version)                           # md_version
    b += struct.pack("<f", 1.0)                                  # scale_md
    b += struct.pack("<i", 1)                                    # has_been_set
    b += struct.pack("<fff", 0.0, 0.5, 0.5)                      # v_strength, v_radius, v_steepness (manual vignette)
    b += struct.pack("<ff", 0.0, 0.0)                            # reserved
    assert len(b) == 356, len(b)
    return b


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--flags", default="all")
    ap.add_argument("--algorithm", type=int, choices=(1, 2), default=2)
    ap.add_argument("out")
    a = ap.parse_args()
    flags = FLAGS[a.flags] if a.flags in FLAGS else int(a.flags)
    with open(a.out, "w") as f:
        f.write(params_blob(flags, a.algorithm).hex() + "\n")


if __name__ == "__main__":
    main()
