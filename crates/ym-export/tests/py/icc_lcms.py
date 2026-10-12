"""Reads the patch grid of exported files and converts each patch with LittleCMS.

For every file given, the script extracts the embedded ICC profile, decodes the centre pixel of
each cell of the patch grid, and converts the codes

* to sRGB, 8 bit, with Pillow's ImageCms (relative colorimetric), for 8-bit files, and
* to CIE XYZ (D50 PCS, doubles) and to sRGB with the C library through ctypes, for every file.

Output: JSON on stdout, {file: {"bits", "codes", "xyz", "srgb_pillow", "srgb_lcms", "profile_id"}}.
Usage: icc_lcms.py <columns> <rows> <cell_px> <file>...
"""
import ctypes
import ctypes.util
import hashlib
import io
import json
import sys

import numpy as np
from PIL import Image, ImageCms

INTENT_RELATIVE_COLORIMETRIC = 1
TYPE_RGB_8 = 0x40019
TYPE_RGB_16 = 0x4001A
TYPE_XYZ_DBL = 0x490018
FLAGS_NOOPTIMIZE = 0x0100


def load_lcms():
    """The system LittleCMS 2 through ctypes, or None when it is not installed (Pillow bundles its own copy)."""
    for name in (ctypes.util.find_library("lcms2"), "liblcms2.so.2", "liblcms2.2.dylib", "liblcms2.dylib"):
        if not name:
            continue
        try:
            lib = ctypes.CDLL(name)
            break
        except OSError:
            continue
    else:
        return None
    lib.cmsOpenProfileFromMem.restype = ctypes.c_void_p
    lib.cmsOpenProfileFromMem.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    lib.cmsCreateXYZProfile.restype = ctypes.c_void_p
    lib.cmsCreate_sRGBProfile.restype = ctypes.c_void_p
    lib.cmsCreateTransform.restype = ctypes.c_void_p
    lib.cmsCreateTransform.argtypes = [
        ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
    ]
    lib.cmsDoTransform.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32]
    lib.cmsDeleteTransform.argtypes = [ctypes.c_void_p]
    lib.cmsCloseProfile.argtypes = [ctypes.c_void_p]
    return lib


def read_pixels(path):
    """Returns (codes[H, W, 3] as uint8/uint16, bits, icc bytes)."""
    lower = path.lower()
    if lower.endswith((".tif", ".tiff")):
        import tifffile

        with tifffile.TiffFile(path) as tf:
            page = tf.pages[0]
            icc = bytes(page.tags[34675].value)
            try:
                data = page.asarray()
            except ValueError:
                # LZW needs the optional imagecodecs package; Pillow (libtiff) reads 8-bit files without it.
                data = np.asarray(Image.open(path).convert("RGB"))
        return data, data.dtype.itemsize * 8, icc
    if lower.endswith(".png"):
        return read_png(path)
    im = Image.open(path)
    return np.asarray(im.convert("RGB")), 8, bytes(im.info["icc_profile"])


def read_png(path):
    """A small PNG reader for non-interlaced 8/16-bit RGB files (Pillow reads 16-bit RGB as 8-bit)."""
    import struct
    import zlib

    data = open(path, "rb").read()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    pos, idat, icc, header = 8, b"", None, None
    while pos < len(data):
        (length,) = struct.unpack(">I", data[pos : pos + 4])
        kind = data[pos + 4 : pos + 8]
        body = data[pos + 8 : pos + 8 + length]
        if kind == b"IHDR":
            header = struct.unpack(">IIBBBBB", body)
        elif kind == b"iCCP":
            name_end = body.index(b"\0")
            icc = zlib.decompress(body[name_end + 2 :])
        elif kind == b"IDAT":
            idat += body
        pos += 12 + length
    w, h, depth, color_type, _, _, interlace = header
    assert color_type == 2 and interlace == 0 and depth in (8, 16), header
    bpp = 3 * depth // 8
    stride = w * bpp
    raw = zlib.decompress(idat)
    rows = []
    prev = np.zeros(stride, dtype=np.int32)
    for y in range(h):
        ft = raw[y * (stride + 1)]
        line = np.frombuffer(raw[y * (stride + 1) + 1 : (y + 1) * (stride + 1)], dtype=np.uint8).astype(np.int32)
        cur = np.zeros(stride, dtype=np.int32)
        for i in range(stride):
            a = cur[i - bpp] if i >= bpp else 0
            b = prev[i]
            c = prev[i - bpp] if i >= bpp else 0
            if ft == 0:
                pred = 0
            elif ft == 1:
                pred = a
            elif ft == 2:
                pred = b
            elif ft == 3:
                pred = (a + b) // 2
            else:
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                pred = a if pa <= pb and pa <= pc else (b if pb <= pc else c)
            cur[i] = (line[i] + pred) & 255
        rows.append(cur)
        prev = cur
    arr = np.array(rows, dtype=np.uint8)
    if depth == 16:
        arr = (arr[:, 0::2].astype(np.uint16) << 8) | arr[:, 1::2]
    return arr.reshape(h, w, 3), depth, icc


def main():
    cols, rows, cell = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    lib = load_lcms()
    out = {}
    for path in sys.argv[4:]:
        pixels, bits, icc = read_pixels(path)
        h, w = pixels.shape[:2]
        assert (w, h) == (cols * cell, rows * cell), (w, h)
        centres = pixels[cell // 2 :: cell, cell // 2 :: cell, :][:rows, :cols, :].reshape(-1, 3)
        fmt_in = TYPE_RGB_8 if bits == 8 else TYPE_RGB_16
        n = centres.shape[0]
        flat = np.ascontiguousarray(centres, dtype=np.uint8 if bits == 8 else np.uint16)
        entry = {
            "bits": bits,
            "codes": flat.astype(int).tolist(),
            "profile_size": len(icc),
            "icc_sha256": hashlib.sha256(icc).hexdigest(),
            "c_library": lib is not None,
        }
        if lib is not None:
            buf = ctypes.create_string_buffer(icc, len(icc))
            src = lib.cmsOpenProfileFromMem(buf, len(icc))
            assert src, "LittleCMS could not read the embedded profile"
            xyz_profile = lib.cmsCreateXYZProfile()
            srgb_profile = lib.cmsCreate_sRGBProfile()

            xform = lib.cmsCreateTransform(src, fmt_in, xyz_profile, TYPE_XYZ_DBL, INTENT_RELATIVE_COLORIMETRIC, FLAGS_NOOPTIMIZE)
            assert xform, "no transform to XYZ"
            xyz = np.zeros((n, 3), dtype=np.float64)
            lib.cmsDoTransform(xform, flat.ctypes.data, xyz.ctypes.data, n)
            lib.cmsDeleteTransform(xform)

            # NOOPTIMIZE keeps LittleCMS from replacing the profile maths with its fixed-point matrix-shaper
            # approximation, which is off by up to two codes next to the gamut edge (and more on extreme
            # out-of-gamut dark patches); the "default" entry shows what that optimised path gives.
            srgb_lcms = np.zeros((n, 3), dtype=np.uint8)
            srgb_default = np.zeros((n, 3), dtype=np.uint8)
            for flags, dest in ((FLAGS_NOOPTIMIZE, srgb_lcms), (0, srgb_default)):
                xform = lib.cmsCreateTransform(src, fmt_in, srgb_profile, TYPE_RGB_8, INTENT_RELATIVE_COLORIMETRIC, flags)
                assert xform, "no transform to sRGB"
                lib.cmsDoTransform(xform, flat.ctypes.data, dest.ctypes.data, n)
                lib.cmsDeleteTransform(xform)
            entry["xyz"] = xyz.tolist()
            entry["srgb_lcms"] = srgb_lcms.astype(int).tolist()
            entry["srgb_lcms_default"] = srgb_default.astype(int).tolist()
            for p in (src, xyz_profile, srgb_profile):
                lib.cmsCloseProfile(p)

        if bits == 8:
            src_p = ImageCms.ImageCmsProfile(io.BytesIO(icc))
            dst_p = ImageCms.createProfile("sRGB")
            strip = Image.fromarray(flat.reshape(1, n, 3), "RGB")
            for key, flags in (("srgb_pillow", ImageCms.Flags.NOOPTIMIZE), ("srgb_pillow_default", ImageCms.Flags.NONE)):
                transform = ImageCms.buildTransform(
                    src_p, dst_p, "RGB", "RGB", renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC, flags=flags
                )
                converted = ImageCms.applyTransform(strip, transform)
                entry[key] = np.asarray(converted).reshape(n, 3).astype(int).tolist()
            entry["profile_description"] = ImageCms.getProfileDescription(src_p).strip()
        out[path] = entry
    json.dump(out, sys.stdout)


if __name__ == "__main__":
    main()
