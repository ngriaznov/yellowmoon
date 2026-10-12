# 04 · ARW decoding

Crate: `ym-raw`. Our own decoder, Apache-2.0. You may read LibRaw, rawler, rawspeed and ExifTool to learn the format; write the code yourself and verify it against them as oracles (`tools/oracle`). Background: `research/notes/sony-arw-formats.md`.

## Output

```rust
pub struct RawImage {
    pub meta: RawMetadata,             // camera, exposure (ISO, shutter, aperture, EC), lens, timestamps, orientation
    pub layout: Layout,                // Bayer { pattern: RGGB.., active_area, crop } | Ycbcr { subsampling }
    pub data: Plane<u16>,              // undithered sensor values, linear after the camera curve
    pub black: [f32; 4], pub white: [f32; 4],   // per CFA position
    pub wb_as_shot: Option<[f32; 4]>,  // from SR2 0x7313 (RGGB) or 0x7303 (GRBG)
    pub qstep: Option<Plane<f16>>,     // quantisation step per pixel (cRAW and ARW6 only)
    pub lens: Option<LensTags>,        // 0x7032 vignetting, 0x7035 CA, 0x7037 distortion
    pub tone_curve: Option<[u16; 4]>,  // 0x7010 knots, for cRAW
    pub preview: Option<JpegRef>,      // embedded full-size JPEG location (for fast thumbnails)
}
```

Never dither. rawler dithers its curve output; that is wrong here, because the quantisation map feeds the denoise network.

## Container

- ARW is little-endian TIFF. IFD0 holds the thumbnail and EXIF pointer; the raw image is in a sub-IFD (`SubIFDs`, 0x014A) with `NewSubFileType = 0`. Older files may place it differently; walk all IFDs and pick the one with Sony raw compression and the largest dimensions.
- EXIF sub-IFD and the Sony maker note (0x927C) carry ISO, exposure, lens and the model ID (0xB001). Some maker-note tags (0x9050, 0x94xx) are enciphered with a byte substitution cipher and model-specific layouts; read only what we need (bit depth hints, RAWFileType 0x2029 if useful) and treat failures as "unknown".
- ISO: prefer Sony's maker-note ISO over EXIF `ISOSpeedRatings` (16-bit, saturates at 65535). Map extended ISOs (below base) to base gain for noise purposes.

### SR2 private data (black, white, white balance)

1. IFD0 tag `DNGPrivateData` (0xC634) holds a 4-byte little-endian offset to the **SR2Private** IFD.
2. SR2Private contains `SR2SubIFDOffset` (0x7200), `SR2SubIFDLength` (0x7201) and `SR2SubIFDKey` (0x7221, 4 bytes LE).
3. Decrypt `length/4` little-endian 32-bit words starting at the offset with Sony's pad cipher: seed four words by iterating `key = key·48828125 + 1` (wrapping u32); fix up word 3 from words 0 and 2; extend to 127 words with the shift-xor recurrence over words p−4, p−2, p−3, p−1; byte-swap the pad words; then for each data word advance the pad (`pad[p] = pad[p+1] ^ pad[p+65]`, indices mod 128) and XOR. Write it from this description and verify against both oracles on every sample.
4. Parse the decrypted bytes as an IFD (offsets relative to the SR2 sub-IFD start). Tags:
   - `0x7310` BlackLevel (4 values per CFA position), falling back to `0x7300` (older);
   - `0x787F` WhiteLevel (per channel);
   - `0x7313` WB levels in RGGB order, falling back to `0x7303` (GRBG).
5. Typical modern values: black 512, white 16383 (A1, 14-bit). 12-bit modes are scaled: check the bit depth and that black < white.

## Decoders by TIFF compression

| Compression / layout | Variant | Decode |
|---|---|---|
| 32767, bits 12 | ARW 2 12-bit uncompressed (packed) | Unpack 12-bit little-endian samples row by row. |
| 32767, bits 8 | cRAW (lossy) | Block decoder below; also used by A100's ARW 1 when bits differ (A100 uses a Huffman variant; support it only if a sample is in the corpus). |
| 1 | 14-bit (or 12-bit) uncompressed | Read u16 LE samples; respect `BitsPerSample`. |
| 7, SamplesPerPixel 1, Photometric CFA | Lossless L | Tiled LJPEG-92 below. |
| 7, Photometric YCbCr (6), SamplesPerPixel 3 | Lossless M / S | Reduced path below. |
| 32766 | ARW 6.0 Compressed RAW 2 (lossy wavelet) | Gated, below. |

### cRAW

Each row stores `width` bytes. Every 32 pixels form two interleaved 16-pixel blocks (even columns and odd columns, i.e. same CFA colour). Each block is 128 bits, read LSB-first:

- 11-bit `max`, 11-bit `min`, 4-bit index of the max pixel, 4-bit index of the min pixel;
- 14 × 7-bit deltas for the other pixels: `value = min(2047, (delta << shift) + min)` where `shift = max(0, bitlen(max − min) − 7)`.

Then expand through the camera curve: tag 0x7010 holds 4 knots (each `(v >> 2) & 0xFFF`); with `c0 = 0`, `c5 = 4095`, the curve's slope doubles at each knot (1, 2, 4, 8, 16). Index the curve with `value << 1`. The curve output is on a 14-bit scale (rawler's convention); LibRaw additionally shifts it right by 2 and uses 12-bit black and white levels. Pick one convention, keep `data`, black, white and `qstep` in the same units, and convert when comparing with an oracle.

**Quantisation map:** for the two exact pixels (max and min) the step is the curve's local step; for delta-coded pixels it is `2^shift` in the 11-bit domain times the curve's local slope. Emit it as `qstep` (in output DN). The denoise network uses it to suppress cRAW's "blondie" stripes within the known intervals.

### Lossless L (tiled LJPEG-92)

- Tiles from `TileWidth`/`TileLength`/`TileOffsets` (512×512 on current bodies).
- Each tile is a lossless JPEG (ITU T.81 process 14) stream that declares 4 components and half the width: each decoded row of 4-component pixels holds two consecutive sensor rows of the 2×2 Bayer quad. Unpack into two rows of `TileWidth` samples. Verify the component-to-quad order on samples from at least three bodies.
- Write our own LJPEG-92 decoder (predictors 1–7, Huffman tables from DHT, restart markers). The `ljpeg` crate is LGPL; don't use it.
- Decode tiles in parallel.

### Lossless M and S (reduced path)

- These are camera-processed: white-balanced, demosaiced, chroma-subsampled YCbCr (4:2:0 or 4:2:2), losslessly coded as LJPEG tiles with 3 components. They are not Bayer.
- Decode tiles, upsample chroma, convert YCbCr → camera RGB with the inverse of the camera's transform (check LibRaw's `sonycc.cpp` for the matrix and any scaling, then implement and verify against the oracle), then **undo** the camera's white-balance gains (SR2 0x7313) so the result is unbalanced camera RGB.
- The pipeline skips highlight reconstruction on the mosaic and the demosaic network; it runs the denoise-only variant of the network if available, else a light classical denoise. The sensor map runs as usual.
- The UI tags these photos REDUCED with the note: "This file was saved in the camera as Lossless M or S. The camera has already processed it, so colour emulation is approximate. Use Lossless L or uncompressed for best results."

### ARW 6.0 Compressed RAW 2 (gated)

- A7 V (Compressed, Compressed HQ) and A7R VI (Compressed, HD). Tiled (up to 16 tiles), packetised wavelet codec built on Sony's LLVC (SMPTE RDD 34: reversible 5/3 lifting, variable-length coding) with Sony deviations. A green-mean plane plus residuals; R and B as colour-difference planes; 12-bit codes and a companding curve whose constants are not stored in the file.
- Permissive reference: lightcraft (Rust, MIT/Apache-2.0; 14-bit only). LibRaw master (LGPL/CDDL) as an oracle.
- **Gate:** [HUMAN] M2-H3: email Sony about IP for RDD 34-based decoding. We ship the ARW 6.0 decoder only after a satisfactory answer, or after a legal opinion Mykyta accepts. Until then: detect, and show "ARW 6.0 compressed files from the A7 V and A7R VI aren't supported yet. Lossless files from these cameras work. We're working on it." Build the decoder behind a cargo feature `arw6` that is off in release builds until the gate clears.
- Quantisation map: derive from the wavelet quantiser per subband if feasible; otherwise a per-tile constant from the bitrate mode. Test 12-bit files once a sample exists (none public as of Oct 2026).

## Lens tags

Raw sub-IFD tags 0x7032 (vignetting), 0x7035 (chromatic aberration), 0x7037 (distortion): up to 16 knots each, as signed 16-bit values preceded by a count. Return them raw; the maths lives in `docs/05-raw-processing.md`. If a tag is missing (older bodies, adapted lenses), lens corrections are skipped for that part and the RAW tab says "No lens data in this file".

## Errors

Every failure returns a typed error with a plain message:

- `Unsupported { camera, variant }`: a known variant we don't decode yet (ARW 6.0 until gated).
- `NotArw`: SRF, SR2, ARQ, or not a TIFF ("This isn't an ARW file. Yellowmoon opens Sony ARW files only."). ARQ gets its own message: "Pixel-shift ARQ files aren't supported. Open the individual ARW frames instead."
- `Corrupt { detail }`: truncated or malformed. Fail that file and keep the session going; never show partially decoded pixels.

## Performance targets

- Metadata probe (no pixels): < 20 ms per file.
- Full decode, 50 MP lossless L on an M1: < 400 ms using all decode threads; uncompressed: < 150 ms (memory-bound).
- Embedded JPEG preview extraction: < 30 ms; used for the start screen's first thumbnails before our own render replaces them.

## Tests

- **Oracle CI (`tools/oracle`):** for every file in the corpus, our decoded `data`, black, white and WB must equal the oracles' values. Compare `data` against LibRaw's `raw_image` (undithered). rawler dithers cRAW output, so for cRAW compare with rawler only within ±1 curve step, and exactly elsewhere. Any other mismatch fails the build. ARW 6.0 is compared with LibRaw master and lightcraft.
- **Corpus:** RawDB (rawdb.dnglab.org, CC0) and raw.pixls.us (CC0 files only, per-file licence check), fetched by `tools/fetch/samples.sh` into `testdata/arw/`. Gaps to fill [HUMAN]: FX2, RX10 V, 12-bit ARW 6.0, A1 Lossless M/S.
- **Fuzzing:** `cargo fuzz` targets for the TIFF walker, SR2 parser, cRAW, LJPEG and the M/S path; no panics, no unbounded allocation (cap allocations by declared dimensions ≤ 120 MP).
- **Quantisation map:** synthetic test: encode a known image with our own cRAW encoder (test-only), decode, and check every true value lies within ±qstep/2 of the decoded value.
