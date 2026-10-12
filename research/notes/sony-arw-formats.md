## Sony ARW formats for Yellowmoon: research report (11 Oct 2026)

**Method.** I read the source of LibRaw (master 9 Oct 2026, with full git history), dnglab/rawler (main at the v0.8.0 bump, 30 Aug 2026), rawspeed (10 Oct 2026) and ExifTool 13.59. Local clones were used (not included). The proxy blocked raw.pixls.us (RPU), so I could not list it directly. RPU status below is inferred from rawspeed's sample policy and from forum and PR posts.

### Variant table

| Variant (TIFF compression) | Bodies | Lossy? | Bayer? | Public docs | LibRaw / rawler / rawspeed | Samples |
|---|---|---|---|---|---|---|
| ARW 1.0 (32767, 12-bit Huffman) | A100 | Lossless | Yes | dcraw/LibRaw `sony_arw_load_raw` [1] | ✓ / ✓ / ✓ | RawDB, CC0 [19] |
| ARW 2.x 12-bit uncompressed (32767, bps 12) | A700/A850/A900 era | Lossless | Yes | rawler `arw.rs` comments [4] | ✓ / ✓ / ✓ | Unverified |
| cRAW (32767, 8 bpp: 11-bit base + 7-bit deltas, 4-segment curve from tag 0x7010) | A700 (2007) to A1 II, A9 III, RX1R III, ZV-E10 II. RX10 V (2026) too, inferred from file size and rawler decoding it. Some modes are 12-bit. | **Lossy** | Yes | lclevy [9], RawDigger/Dietz cited there | ✓ / ✓ / ✓ | RawDB CC0, many bodies |
| 14-bit uncompressed (1) | From A7R II/A7S II/RX1R II (ARW 2.3.2, 2015); the A7 V has none | Lossless | Yes | trivial | ✓ / ✓ / ✓ | RawDB CC0 |
| Lossless L (7): 512×512 LJPEG92 tiles, two Bayer rows packed into one 4-component row | A1 (2021) onward: A7 IV, A7R V, 6700, A7C II/CR, ZV-E1, A9 III, A1 II, RX1R III, A7 V, A7R VI, RX10 V. ARW 4.0 itself first appears on the A7S III [8]. | Lossless | Yes | rawler `decode_ljpeg` comments [4] | ✓ (0.21+) / ✓ / ✓ | RawDB CC0 (A1, A7R V, A9 III, A1 II, A7 V, A7R VI, RX1R III, RX10 V) |
| Lossless M/S (7, Photometric 6 = YCbCr, 3 samples, 4:2:0/4:2:2, white balance already applied) | A7R V (first), A9 III, A1 II, RX1R III; A1 via firmware (unverified) | Lossless coding of camera-processed data | **No** (demosaiced, scaled YCbCr) | dnglab #263/#269 [5], LibRaw `sonycc.cpp` [1] | ✓ (since 0.22, Dec 2025, as "pseudo-RAW") / ✓ / **✗** [7] | RawDB CC0 (A7R V "yuv" set, A9 III, A1 II, RX1R III). A1 set has no M/S. |
| "A1 30 fps YUV420 burst" | Claimed only in lclevy's table [9] | — | — | — | No decoder has a separate path | **Unverified.** The only YUV420 path anywhere is the M/S one (dnglab #269), so this looks like a mix-up. |
| ARW 5.0 / 5.0.1 | A9 III (5.0); A1 FW 2.0 and A1 II (5.0.1) [8][9] | — | — | — | Same codecs as above | No new pixel codec. Changes are metadata only: new tags, shifted makernote layouts. |
| **ARW 6.0 "Compressed RAW 2" (32766)** | **A7 V** (Compressed and Compressed HQ), **A7R VI** (Compressed and HD) | **Lossy** | Bayer, rebuilt from planes | SMPTE RDD 34 [10] plus LibRaw `sony_arw6.cpp` [1] and lightcraft docs [11] | **master only** (merged 18 Jul 2026, not in 0.22.2) / **stub** (`unimplemented!()`) / **✗** | RawDB CC0: A7 V (FF and APS-C, C and HQ), A7R VI (C and HD, portrait and landscape). A7 V set also on RPU [18]. No 12-bit ARW6 sample anywhere. |

**Licences.** LibRaw is LGPL-2.1 or CDDL-1.0. rawler is LGPL-2.1. rawspeed is LGPL-2.1+. lclevy/sony_raw is GPL-3.0. lightcraft is MIT/Apache-2.0, the only permissively licensed ARW6 decoder. RPU's new samples are CC0, but some older ones are CC BY-NC-SA. RawDB sets checked here are all CC0 per their records. rawspeed treats a camera without the `no-samples` flag as having RPU samples. On that basis the A7 V, A7R VI, A1 II, A9 III, RX1R III, FX2 and ZV-E10 II have RPU samples; the RX10 V does not ("unknown-no-samples"). FX2 has no RawDB set.

### Bodies released 2024–2026
- **A9 III, A1 II, RX1R III:** uncompressed, lossless L/M/S, cRAW.
- **ZV-E10 II:** 12-bit cRAW and lossless.
- **FX2:** LibRaw and rawspeed support it; its raw modes are unverified.
- **A7 V** (announced 2 Dec 2025): only Lossless, Compressed (HQ) and Compressed. No uncompressed, no M/S [12].
- **A7R VI** (May 2026): Lossless, Compressed, Compressed HQ. Whether it also offers uncompressed or M/S is unverified.
- **RX10 V** (July 2026): lossless and cRAW.
- Also ILX-LR1 (industrial) and the "A" regional variants (FX3A and others), which are aliases.

### The A7 V format (confidence: high)
- Both lossy modes write TIFF compression 32766. ExifTool reports `RAWFileType = 3 "Compressed RAW 2"` for both [8]. darktable 5.6.2 (4 Oct 2026) lists "ARW 6.0 compressed and compressed (HQ)" as missing [7].
- It is a tiled (up to 16 tiles), packetised wavelet codec:
  - Built on Sony's LLVC (SMPTE RDD 34: reversible 5/3 lifting with variable-length coding), with "two Sony deviations" [10][11].
  - Coded as a green-mean plane plus residuals, with R and B as colour-difference planes.
  - Uses 12-bit codes and a companding curve that LibRaw hard-codes (`llvc3_curve`, black 1024, max 39002).
  - The decoded output is a Bayer CFA, but the data is **lossy**.
- Third-party support as of October 2026:
  - **Adobe:** Lightroom Classic 15.2.1 / Camera Raw (18 Mar 2026) supports both lossy modes [14]; DNG Converter works too.
  - **Capture One:** official note says Lossless only [15]. A blog claim of full support is unverified.
  - **DxO:** Lossless only (PhotoLab 9.5); lossy still unsupported in June 2026 [16].
  - **LibRaw:** master only. PR #824 tested HQ files only; standard Compressed is unverified [2].
  - **rawler:** stub. **rawspeed/darktable:** no.
  - **RawTherapee:** no in releases, which bundle LibRaw 0.22.2 [17].
  - **lightcraft (Rust, MIT/Apache):** merged a clean-room decoder on 11 Oct 2026. It handles 14-bit only, and A7 V was tested by a reviewer only [11].

### Encryption and obfuscation
- Nothing new. The SR2SubIFD (white balance, black and white levels) still uses Sony's known 32-bit-key cipher; see rawspeed `SonyDecrypt` and rawler `sony_decrypt`.
- The enciphered makernote tags 0x9050/0x94xx use the same substitution cipher on the A7 V and A7R VI, but with model-specific offset shifts [8].
- ARW6 pixel data is not encrypted. It is undocumented, and several constants are not stored in the file metadata.

### Risks for an "every Sony ARW" promise
1. **ARW6 provenance and patents:**
   - LibRaw's decoder is LGPL/CDDL and credits AI assistance (GLM-5).
   - RDD 34 tells implementers to send IP enquiries to Sony.
   - lightcraft is the only permissively licensed implementation, and it is one day old.
2. **ARW6 has hard-coded or fitted constants:** the curve, black/white levels and a ×2 output scale. Header version values vary between files. APS-C heights are not multiples of 16 (4372). Portrait orientation matters for the lossy files. The 12-bit variant is untested.
3. **M/S files are not Bayer:** they are white-balanced, chroma-subsampled YCbCr. CFA-level sensor emulation cannot apply to them.
4. **Lossy cRAW and ARW6 artefacts:** cRAW posterises; ARW6 has wavelet artefacts.
5. **Firmware changes formats:** an update can bump the ARW version or add modes, as A1 FW 2.0 did (ARW 5.0.1). Makernote offsets differ per model.
6. **Sample gaps:** FX2 (no RawDB set), RX10 V (not on RPU), 12-bit ARW6, A1 M/S. Expect to shoot or borrow these.
7. **Scope:** SRF and SR2 (F828/V3, R1) are not ARW. Exclude them explicitly, as well as ARQ.

### Sources
1. LibRaw source: https://github.com/LibRaw/LibRaw (`src/decoders/sony_arw6.cpp`, `sonycc.cpp`, `src/metadata/tiff.cpp`; tags 0.22.0 = 9 Jan, 0.22.1 = 6 Apr, 0.22.2 = 15 Jul 2026)
2. https://github.com/LibRaw/LibRaw/pull/824
3. https://github.com/LibRaw/LibRaw/issues/776
4. https://github.com/dnglab/dnglab (`rawler/src/decoders/arw.rs`, `rawler/src/decompressors/arw6/mod.rs`)
5. https://github.com/dnglab/dnglab/pull/269 and https://github.com/dnglab/dnglab/issues/263
6. https://github.com/darktable-org/rawspeed (`ArwDecoder.cpp`, `data/cameras.xml`, `docs/ReferenceSampleArchive.rst`)
7. https://newreleases.io/project/github/darktable-org/darktable/release/release-5.6.2 and https://www.darktable.org/2026/02/darktable-5.4.1-released/
8. https://github.com/exiftool/exiftool/blob/master/lib/Image/ExifTool/Sony.pm
9. https://github.com/lclevy/sony_raw
10. https://pub.smpte.org/latest/rdd34/rdd34-2015.pdf
11. https://github.com/storytold/lightcraft/pull/625
12. https://helpguide.sony.net/ilc/2540/v1/en/contents/251h_raw_file_type.html
13. https://helpguide.sony.net/ilc/2440/v1/en/contents/201h_raw_file_type.html
14. https://community.adobe.com/announcements-673/lightroom-classic-15-2-1-is-now-live-with-camera-raw-support-1554249
15. https://support.captureone.com/hc/de/community/posts/33724105853597-Notes-on-Sony-a7-V-support
16. https://forum.dxo.com/t/sony-a7v-compressed-and-hq-raw-not-supported/54584 and https://forum.dxo.com/t/sony-a7-v-raw-files/54073
17. https://discuss.pixls.us/t/no-showing-of-ilce-7m5-raw-pictures-in-rawtherapee-5-13-rc1/59354 and https://github.com/RawTherapee/RawTherapee/pull/7727
18. https://github.com/darktable-org/darktable/issues/19986
19. RawDB set records: https://rawdb.dnglab.org/api/sets/Sony/ILCE-7M5 (likewise /ILCE-7RM6, /ILCE-1, /ILCE-1M2, /ILCE-9M3, /ILCE-7RM5, /DSC-RX1RM3, /DSC-RX10M5, /DSLR-A100)
20. https://en.wikipedia.org/wiki/Sony_%CE%B17R_VI and https://sansmirror.com/cameras/camera-database/sony-mirrorless-cameras-2/sony-a7-mark-v-camera-speci.html
21. https://forum.dxo.com/t/support-for-sony-dsc-rx-10m5-raw/55781

**Worth remembering for the Yellowmoon project:** RawDB (rawdb.dnglab.org) has a JSON API that WebFetch can read, with CC0 Sony sets that include ARW6 files. lightcraft (MIT/Apache Rust) is the only permissively licensed ARW6 decoder.