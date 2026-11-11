# Nikon NEF (and NRW) decoding for an independent Apache-2.0 decoder — research notes (as of 2026-11-11)

Reading note for the report writer. "Cited Findings" are things I read on the linked page during this session. Anything I know only from prior knowledge and could not re-verify in this session is placed under "Inferences" or "Gaps" and labelled **[from memory, unverified]**. Several of the source pages are large and were read through a summarising fetch tool, so byte-level details should be re-checked against a real file with `exiftool -v4` and against the CI oracle before being written into a spec.

---

## 1. NEF container structure and the makernote tags a raw processor needs

### Takeaway
NEF is a TIFF/EP-like container whose decoding-critical parameters live in the Nikon makernote, and ExifTool's tag tables are the most complete public documentation of them. The white-balance block (ColorBalance, 0x0097), ShotInfo and LensData are encrypted in most bodies with a key derived from SerialNumber (0x001d) and ShutterCount (0x00a7).

### Cited Findings
- NEF is "TIFF-based, close to TIFF/EP or DNG"; a large JPEG preview is in a SubIFD of the main IFD (type = thumbnail) or, in older files, in the MakerNote IFD. — [libopenraw NEF format notes](https://libopenraw.pages.freedesktop.org/formats/nef)
- TIFF Compression values seen in NEF raw IFDs: 1 = uncompressed (D1; D1X has a 0.5 pixel ratio, two rows for one); 32769 = packed data with padding (if `col % 10 == 9` an extra 8 bits are skipped); 34713 = Nikon compressed (Huffman tree plus quantisation/linearisation table). — [libopenraw NEF format notes](https://libopenraw.pages.freedesktop.org/formats/nef)
- D100 quirk: some D100 files tag uncompressed packed data as compressed (34713). — [libopenraw NEF format notes](https://libopenraw.pages.freedesktop.org/formats/nef)
- Makernote tags a raw processor needs, per ExifTool: 0x000c WB_RBLevels (rational64u[4]); 0x001d SerialNumber ("this value is used as a key to decrypt other information"); 0x003d BlackLevel (int16u[4]); 0x0045 CropArea (int16u[4], "left, top, width, height"); 0x004f ColorTemperatureAuto; 0x008c ContrastCurve; 0x0093 NEFCompression; 0x0096 NEFLinearizationTable; 0x0097 ColorBalance (versioned); 0x0098 LensData (versioned); 0x00a7 ShutterCount ("This value is used as a key to decrypt other information"; "includes both mechanical and electronic shutter activations for models with this feature"); 0x0e22 NEFBitDepth (int16u[4]: `12 0 0 0` = 12-bit, `14 0 0 0` = 14-bit, `8 8 8 0`, `16 16 16 0` for RGB TIFF-style data); 0x0011 PreviewIFD. — [ExifTool Nikon tags](https://exiftool.org/TagNames/Nikon.html)
- NEFCompression (0x0093) values: 1 Lossy (type 1); 2 Uncompressed; 3 Lossless; 4 Lossy (type 2); 5 Striped packed 12 bits; 6 Uncompressed (reduced to 12 bit); 7 Unpacked 12 bits; 8 Small; 9 Packed 12 bits; 10 Packed 14 bits; 13 High Efficiency; 14 High Efficiency*. — [ExifTool Nikon tags](https://exiftool.org/TagNames/Nikon.html)
- Newer bodies also carry NEFCompression at index 10 of a separate makernote block 0x0051 (MakerNotes0x51), with FirmwareVersion51 at index 0. — [ExifTool Nikon tags](https://exiftool.org/TagNames/Nikon.html)
- ColorBalance (0x0097) is versioned and the version selects the channel order of the WB levels: ColorBalance1 = WB_RBGGLevels, ColorBalance2 = WB_RGGBLevels, ColorBalance3 = WB_RGBGLevels, ColorBalance4 = WB_GRBGLevels (all int16u[4]). Versions map as 0100→CB1, 0102→CB2, 0103→CB3, 0205→CB2, 0209→CB4, 02xx→CB2, 0211→CB4, 0213→CB2, 0215→CB4, 0219→CB2, plus "Unknown" variants. — [ExifTool Nikon tags](https://exiftool.org/TagNames/Nikon.html)
- Encryption notes quoted from ExifTool: ColorBalance2 "This information is encrypted for most camera models."; ShotInfo "This information is encrypted for ShotInfoVersion 02xx, and some tags are only valid for specific models."; LensData01 "Nikon encrypts the LensData information below if LensDataVersion is 0201 or higher" (decryption algorithm is known); ShotInfoZ9 tags are "extracted from encrypted data in images from the Z9". — [ExifTool Nikon tags](https://exiftool.org/TagNames/Nikon.html)
- The quantisation/linearisation tables are at makernote tags 0x8c and 0x96; dcraw effectively uses only 0x96. — [libopenraw NEF format notes](https://libopenraw.pages.freedesktop.org/formats/nef)
- One independent Rust implementation (the `bioformats` crate's `tiff/nikon.rs`) parses tag 150 (0x96) into: a lossless flag, four vertical predictors, a curve, and a split value, and reads the split value at a fixed byte offset 562 of the tag. — [docs.rs bioformats nikon.rs](https://docs.rs/crate/bioformats/0.1.3/source/src/tiff/nikon.rs)
- Z-series files carry an additional Nikon block in SubIFD1 tag 0xc7d5 (NEFInfo) with sub-blocks for distortion and vignette information (see §6). — [ExifTool Nikon tags](https://exiftool.org/TagNames/Nikon.html)
- LensData versions documented: 0100, 0101, 0201, 0204, 0400, 0402, 0403, 0800 (Z-mount); LensData0800 includes FocusDistance (index 11) and MaxApertureAtMaxFocal (index 18). — [ExifTool Nikon tags](https://exiftool.org/TagNames/Nikon.html)

### Inferences
- **[from memory, unverified in this session]** Typical modern layout: IFD0 = 160x120 RGB thumbnail plus EXIF/makernote pointers; SubIFD0 = full-size JPEG preview; SubIFD1 = raw CFA (Compression 34713 or 1, CFAPattern, BitsPerSample, StripOffsets, ActiveArea-like crop comes from makernote 0x0045); newer Z files add a third SubIFD. The makernote starts with `Nikon\0` + 2 version bytes + 2 padding bytes and then its **own embedded TIFF header** (own byte order), and all makernote offsets are relative to that inner header. Verify with `exiftool -v4`; ExifTool's page is the citation target ([ExifTool Nikon tags](https://exiftool.org/TagNames/Nikon.html)).
- **[from memory, unverified]** The 0x0097 decryption scheme widely described in public sources: key byte A = XOR of the four bytes of ShutterCount; key byte B = a fold of the numeric serial number's low byte (non-numeric serials fall back to model constants, e.g. 0x22 for D50, 0x60 otherwise); two fixed 256-byte substitution tables seed a simple stream cipher (`ci = xlat0[serial & 0xff]`, `cj = xlat1[key]`, `ck = 0x60`, then per byte `cj += ci * ck++; out = in ^ cj`). Decryption starts at a version-dependent offset inside the block (e.g. 284 for 0x0204+ type blocks). The two xlat tables are pure data that appear in ExifTool, exiv2, dcraw, LibRaw and rawspeed; a clean-room decoder still needs those 512 bytes, which are functional constants rather than expressive code (legal inference, not legal advice).
- **[from memory, unverified]** Tag 0x96 layout as implemented by every open decoder: bytes 0–1 = version pair (`0x44 0x20` = 12-bit lossy; `0x44 0x10`/`0x44 0x20` family lossy with split; `0x46 0x30` = lossless; `0x49 0x58` = D3100-era variant with different predictor stepping), then 4 × u16 vertical predictors (2 rows × 2 columns), then u16 curve entry count `csize`, then `csize` u16 curve points which are linearly interpolated with step `max / (csize-1)` when `csize` is small; when version is `0x44 0x20` and step > 0, a u16 "split row" is read at offset 562. The bioformats source above corroborates the 562 offset and the field set.
- Because ExifTool marks 0x8c/0x96 as `undef!` blobs with no layout, the byte-level layout is *not* documented in any permissive prose source I found; it is only expressed in code (dcraw, LibRaw, rawspeed, rawler, libopenraw, bioformats).

### Gaps
- I did not find a permissively licensed prose specification of the NEF IFD layout or of the 0x96 block; libopenraw's page defers to dcraw. Nikon publishes no format specification.
- I could not read the last ~55 k characters of the ExifTool Nikon page (offset 200000+), which may hold further notes (e.g. NikonCapture, NEFInfo details, NRW-specific tags).
- NRW: libopenraw only says NRW is a Coolpix variant (P6000, P7000) with non-standard raw storage, "TBD" ([libopenraw](https://libopenraw.pages.freedesktop.org/formats/nef)). **[from memory]** NRW WB is in different makernote tags (0x0014 NRWData) and is not relevant to current Z bodies; NRW can reasonably be out of scope.
- Optical-black/masked-area geometry per body is not documented in any source I read; open decoders carry per-camera crop tables.

---

## 2. Compression variants across the range

### Takeaway
A classic NEF decoder must handle: uncompressed (several packings), Nikon Huffman-predictive lossless, Nikon lossy (same Huffman coder plus a linearisation curve, optionally with a mid-image "split"), small/medium NEF, and — on Z9, Z8, Zf, Z6III, Z5II, Z50II, ZR — High Efficiency / High Efficiency* (NEFCompression 13/14), which is intoPIX TicoRAW and is a wholly different (wavelet) codec. On the HE-capable bodies the only non-HE choice is lossless compressed.

### Cited Findings
- "Raw image data is compressed by using a lossy encoding followed by non-adaptive Huffman compression"; the lossy step maps ADC values through an encoding table stored in the file; lossless NEF skips the encoding table. — [Bill Claff, NEF Compression (prepared 2006, updated 2021-01-28)](https://photonstophotos.net/NikonInfo/NEF_Compression.htm)
- Eight lossy encoding-table sizes exist: 567, 683, 689, 769, 1025, 2753, 3073, 4097 levels. 12-bit: D300/D300S/D3/D700 → 689; D5300, D500, D600, D610, D7000, D7100, D7500, D3S, D3X, D4, D800/E, D850, Df → 769; D7200, D750, D810, D5500, D5600 → 1025; D780 and Z6/Z7 → 1025 (variant 2). 14-bit: D300/D300S/D3/D700 → 2753; D5300, D500, D600/610, D7000, D7100, D7500, D3S, D3X, D4, D800/E, D850, Df, D5100, D5200, Coolpix A → 3073; D7200, D750, D810, D5500, D5600 → 4097; D780 and Z6/Z7 → 4097 (variant 2). — [Bill Claff, NEF Compression](https://photonstophotos.net/NikonInfo/NEF_Compression.htm)
- Each table is 1:1 up to a "first missing value" (e.g. 107 for the 567 table, 216 for 683, 240 for 689, 114 for 769, 400 for 1025, 640 for 2753, 450 for 3073, 1552 for 4097), i.e. shadows are stored exactly and only highlights are quantised. — [Bill Claff, NEF Compression](https://photonstophotos.net/NikonInfo/NEF_Compression.htm); corroborated for D5300 14-bit ("1-to-1 below 450 DN") by [RawDigger forum](https://www.rawdigger.com/node/335?page=1)
- Some 14-bit files are "lossy after split" and use a separate split Huffman table (sample hunted from a D750). — [pixls.us "Nikon: a specific raw sample wanted"](https://discuss.pixls.us/t/nikon-a-specific-raw-sample-wanted/5483/9)
- Z8: "NEF (RAW): 14 bit; choose from lossless compression, high efficiency★, and high efficiency options" — there is no uncompressed and no 12-bit option. — [Nikon Z8 product page](https://imaging.nikon.com/imaging/lineup/mirrorless/z_8/); [Nikon Z8 online manual, specifications](https://onlinemanual.nikonimglib.com/z8/en/specifications_364.html)
- Z9 has the same three choices only ("there are only three choices in the RAW recording menu"). — [Photography Life, Z9 lossless vs HE](https://photographylife.com/nikon-z9-lossless-vs-high-efficiency-raw-files)
- Z6III offers Lossless compression, High efficiency★ and High efficiency. — [Nikon Z6III online manual, RAW recording](https://onlinemanual.nikonimglib.com/z6III/da/psm_raw_recording_133.html)
- intoPIX states TicoRAW "has been added to Nikon Z9, Z8, Z6 III, and Z5 II cameras" for stills and video (N-RAW). — [intoPIX TicoRAW page](https://www.intopix.com/ticoraw)
- Z50II HE files and Zf HE files are reported unreadable by open tools, i.e. those bodies also offer HE. — [pixls.us "Z50II and darktable 5.2.0"](https://discuss.pixls.us/t/z50ii-and-darktable-5-2-0/50822); [pixls.us "Nikon ZF and NEF/HEIF"](https://discuss.pixls.us/t/nikon-zf-and-nef-heif/47342)
- A user accidentally reset a Z6III and found the NEFs were then saved as HE* — anecdotal evidence about what the camera falls to after a reset. — [pixls.us "Beating Nikon High Efficiency NEFs", 2026-07-20](https://discuss.pixls.us/t/beating-nikon-high-efficieny-nefs/59347)
- raw.pixls.us asks contributors for Nikon samples in "all the combinations of compression and bitness settings, plus all the raw sizes": 12-bit and 14-bit, each compressed / lossless compressed / uncompressed, plus small and medium NEF. — [raw.pixls.us](https://raw.pixls.us)

### Inferences
- **[from memory, unverified]** The Huffman stage uses six fixed code tables (12-bit lossy, 12-bit lossy-after-split, 12-bit lossless, 14-bit lossy, 14-bit lossy-after-split, 14-bit lossless); each symbol is a bit-length category followed by that many difference bits (JPEG-lossless style), predictor = pixel two columns to the left (same CFA colour), with the first two pixels of each row predicted from the 2×2 vertical predictors carried in 0x96. The tables are small data arrays present in every open decoder; they must be re-derived/transcribed as data, not copied as code.
- **[from memory, unverified]** sNEF/"RAW S" on D4S/D810/D5 is not CFA data: it is 4:2:2 YCbCr, uncompressed 12-bit packed, already white-balanced and tone-curved (NEFCompression 8 = "Small"). D850/D780/Z7 "Medium/Small" raw are instead 12-bit lossless-compressed files produced by in-camera resampling. For a spectral-emulation pipeline that needs true CFA data, sNEF should be rejected rather than supported.
- **[from memory, unverified]** Generation map: D1–D2 era uncompressed or lossy only; D3/D300 onward offer uncompressed / lossless / lossy at 12 or 14-bit; consumer bodies (D3x00, D5x00, Z50, Zfc, Z30) are lossy-or-lossless compressed only, mostly fixed; Z5/Z6/Z6II/Z7/Z7II offer uncompressed, lossless and lossy at 12/14-bit; EXPEED 7 bodies (Z9, Z8, Zf, Z6III, Z5II, Z50II, ZR) drop uncompressed and lossy and offer only lossless + HE* + HE, 14-bit only.
- Share readable without HE: a decoder with lossless + lossy + uncompressed covers 100 % of files from Z5, Z6, Z6II, Z7, Z7II, Z50, Zfc, Z30 and all DSLRs, and the lossless-mode files of the EXPEED 7 bodies. The unreadable fraction equals the share of EXPEED 7 owners who select HE/HE*. I found no statistic for that share (see Gaps); forum traffic suggests HE* is popular with Z8/Z9 action shooters because files are roughly half the size.

### Gaps
- **Factory default RAW recording mode per body** (lossless vs HE*) — not confirmed from Nikon manuals in this session. The Z6III reset anecdote suggests HE* may be what a reset yields on that body, but one forum post is weak evidence; check each body's reference manual "defaults" table.
- No published data on what fraction of Z-mount NEFs in the wild are HE/HE*. Any percentage would be a guess.
- Exact compression/bit-depth menu per body (table for all bodies the brief lists) was not verified against Nikon manuals; the generation map above is from memory.
- Whether Z50II, Zf, Z5II, ZR HE bitstreams differ from the Z9/Z8 ones (profile, bit depth) is undocumented; the pixls decoder author saw Z6III files fail at first and decode later (see §3).

---

## 3. TicoRAW / High Efficiency status: open decoders, patents, how others handle it, Apple fallback

### Takeaway
As of November 2026 I found no released open-source or permissively licensed HE/HE* decoder: LibRaw, rawspeed/darktable, RawTherapee and ART do not decode it, and the one community reverse-engineering effort (pixls.us, April–May 2026) stalled without publishing a repository. The codec is intoPIX's proprietary, patent-encumbered TicoRAW; commercial converters (Adobe, DxO, Nikon NX Studio) decode it, and Apple's system RAW engine now lists the HE-capable bodies, so a macOS app can fall back to Apple's decoder — but that yields a demosaiced image, not CFA data.

### Cited Findings
**What it is**
- "Nikon High Efficiency NEF formats basically stuff an intoPIX's TicoRAW data package into the usual Nikon NEF container." — [Photography Life, HE vs lossless](https://photographylife.com/nikon-he-versus-lossless-compressed-nef-files)
- intoPIX announced TicoRAW in the Z9's High Efficiency RAW recording. — [intoPIX blog](https://www.intopix.com/blogs/post/TicoRAW-technology-added-with-High-Efficiency-RAW-recording-of-Nikon-Z-9-flagship-mirrorless-camera)
- intoPIX distinguishes two formats: **TicoRAW**, its proprietary default profile used by all its RAW IP cores/SDKs, and a "JPEG XS RAW edition" implementing the RAW (Bayer) profile of ISO/IEC 21122. It states TicoRAW is "covered by one or more claims of patent", that it "holds several essential patents" in the JPEG XS RAW profile, and that "your product relies on intoPIX inventions" whichever format is chosen. — [intoPIX TicoRAW page](https://www.intopix.com/ticoraw)
- A FastTicoRAW SDK exists for x86-64, CUDA, ARM64 and Apple silicon; "Evaluation versions, IP cores for FPGA or ASIC and software libraries are fully available"; no price or licence terms are published — contact sales. — [intoPIX TicoRAW page](https://www.intopix.com/ticoraw)
- An intoPIX TicoRAW flyer (Sept 2023) carries the disclaimer that no licence is granted by implication under any intoPIX patent. — [intoPIX TICO-RAW flyer](https://ja.intopix.com/Ressources/Flyers/TicoRAW/EN/intoPIX-TICO-RAW_Flyer_Intel.pdf)

**Open-source status**
- "High efficiency NEF compression is not supported by any open source raw decoding library (at least that I'm aware of)." — [pixls.us "Nikon Z Series High Efficiency RAW Support"](https://discuss.pixls.us/t/nikon-z-series-high-efficiency-raw-support/50428)
- darktable release notes from 4.4.1 onward list Nikon high-efficiency NEFs under unsupported compression modes. — [darktable changelog mirror](https://www.softpedia.com/progChangelog/darktable-Changelog-256980.html); a 2023 forum post claiming darktable 4.6 would support HE conflicts with this and is not borne out by later reports — [pixls.us](https://discuss.pixls.us/t/darktable-nightly-and-newly-supported-file-formats/40817)
- July 2026: Z6III HE* file "does not open in the latest Darktable nightly"; workaround was Adobe DNG Converter 18 under Wine (v14 failed), producing a DNG darktable reads (18.5 MB HE* NEF → 25.3 MB DNG). — [pixls.us "Beating Nikon High Efficiency NEFs", 2026-07-20](https://discuss.pixls.us/t/beating-nikon-high-efficieny-nefs/59347)
- April 2026: RawTherapee (which uses LibRaw for such files) does not open Z8 HE* files while DxO PhotoLab does. — [pixls.us "Nikon HE* decoder" p.3](https://discuss.pixls.us/t/nikon-he-decoder/56955?page=3)
- ART cannot read HE NEFs. — [pixls.us "ART can not read Nikon High Efficiency RAW files"](https://discuss.pixls.us/t/art-can-not-read-nikon-high-efficiency-raw-files/53870)
- A LibRaw-based viewer vendor concluded LibRaw does not support HE NEF. — [Tonfotos community](https://community.tonfotos.com/d/1094-support-for-nef-files-when-high-efficiency-option-is-activated)
- Community reverse-engineering attempt: "piratenpanda" (Benjamin), working with an LLM for about a month on Z9 samples, reached a viewable image by 2026-04-06 with wrong colours and artefacts; the pipeline involves wavelet transforms, precincts, an inverse multi-component transform and a "star-tetrix"-like step (linking the format to the JPEG XS sensor-compression design); Z6III files decoded by 2026-05-03 with the same colour/fine-detail defects; on 2026-05-08 he wrote "I'm not getting anywhere at this stage" and paused. No repository URL or licence was posted in the pages I read. — [pixls.us "Nikon HE* decoder" p.1](https://discuss.pixls.us/t/nikon-he-decoder/56955); [p.3](https://discuss.pixls.us/t/nikon-he-decoder/56955?page=3)
- In that thread: a user reports having asked Nikon for help with an open decoder and Nikon "could not provide one"; the EFF was asked for advice on 2026-04-06 and had not replied by 2026-04-29; forum participants (not lawyers) say the concern is patents rather than copyright, that describing a patented method is fine while shipping working code is "a grey area", and that intoPIX — not Nikon — is the licensor. — [pixls.us "Nikon HE* decoder" p.1](https://discuss.pixls.us/t/nikon-he-decoder/56955); [p.3](https://discuss.pixls.us/t/nikon-he-decoder/56955?page=3)
- Forum claim that intoPIX earns from licensing the decoding SDK as well as encoders and insists its SDK be used. — [pixls.us HE support thread](https://discuss.pixls.us/t/nikon-z-series-high-efficiency-raw-support/50428?page=2) (forum opinion, not a primary source)

**Commercial and Apple handling**
- Adobe DNG Converter converts HE NEFs to decoded DNGs readable by darktable. — [pixls.us HE support thread](https://discuss.pixls.us/t/nikon-z-series-high-efficiency-raw-support/50428)
- Apollo One and FastRawViewer added HE/HE* support by bypassing Apple's built-in raw support. — reported by [zsystemuser (Thom Hogan)](https://www.zsystemuser.com/nikon-z-system-news-and/2025-nikon-z-system-newsvie/apple-woefully-behind-on.html) via search snippet; article body not retrieved
- Historically Apple supported Z8/Z9 lossless-compressed NEF only; HE files did not display in Photos/Finder/Preview on Sonoma. — [Apple Communities](https://discussions.apple.com/thread/255260020); [Apple Communities](https://discussions.apple.com/thread/255894628)
- A Nikon-focused site reports that macOS Tahoe 26.4 (March 2026) began supporting High efficiency* and High efficiency files system-wide (passing through to Photos, Affinity Photo 2, RAW Power). — [zsystemuser "Versions of Nikon Software"](https://zsystemuser.com/accessories/versions-of-nikon-software.html) (seen via search snippet; page body not retrieved); a forum thread title says "MacOS Tahoe 26.4 Supports Viewing of Nikon HE*/HE RAW Thumbnails" — [bcgforums](https://bcgforums.com/goto/post?id=587961)
- Apple's "Digital camera RAW formats supported by iOS 26, iPadOS 26, macOS Tahoe 26, visionOS 26" page (dated 2026-10-06 as fetched) lists Nikon Z30, Z5, Z5II, Z50, Z50II, Z6, Z6II, Z6III, Z7, Z7II, Z8, Z9, Zf, Zfc, ZR and all DSLRs named in the brief (D850, D780, D750, D810, D500, D7500); as fetched, no Nikon entry carried a footnote, while footnotes 1 "Lossless RAW format only" and 4 "Lossless compressed format only" exist on the page. — [Apple Support 122870](https://support.apple.com/en-us/122870); **contradicted by** Apple Communities posts saying most Nikon Z models on that list carry footnote 4 "Lossless compressed format only" (Z6III, Z5II named) — [Apple Communities KR](https://discussions-kr-prz.apple.com/thread/256162681), [Apple Communities DE](https://communities.apple.com/de/thread/256344272). The difference may be the page changing after 26.4, or a fetch-tool error; re-check the page by eye.
- CIRAWFilter exposes `supportedCameraModels`, `decoderVersion`, `linearSpaceFilter` (a filter applied "while it is in linear space") and `draftModeEnabled`; WWDC26 session 305 describes the RAW pipeline stages (demosaic, denoise, convolution, colour adjustments) and opting in to "RAW 9" via `decoderVersion`. — [Apple CIRAWFilter docs](https://developer.apple.com/documentation/coreimage/cirawfilter.md); [WWDC26 session 305](https://developer.apple.com/videos/play/wwdc2026/305/)

### Inferences
- Legal position for Yellowmoon (inference, not legal advice): there is no copyright obstacle to an independent HE decoder, but intoPIX asserts patents over TicoRAW and offers no public patent licence; shipping a decoder in an Apache-2.0 project (whose licence includes a contributor patent grant the project cannot make for third-party patents) carries unquantified patent risk. LibRaw's and rawspeed's five-year absence of HE support, despite both teams' record of reverse-engineering, is consistent with technical difficulty plus this risk.
- Using Apple's system decoder is legally clean: the app calls a public OS API and ships no codec. **[from memory, unverified against the doc page this session]** CIRAWFilter's `outputImage` is a demosaiced, colour-processed CIImage; the most "raw" a caller can get is scene-linear demosaiced RGB (by disabling boost, sharpening, noise reduction, gamut mapping, lens correction, and reading in a linear working space or via `linearSpaceFilter`). There is no public API returning undemosaiced CFA data, and Apple has already applied its own black subtraction, WB and camera→working-space matrix.
- Consequences for Yellowmoon if it used that fallback: (a) the spectral camera model would have to start from Apple's linear RGB instead of camera RGB, so the per-camera spectral profile cannot be applied identically; (b) output would depend on the macOS version (`decoderVersion`), breaking hard rule 8 (determinism) and hard rule 6 (CLI/app bit-identical, all processing in Rust crates); (c) Linux CI could not exercise it. The fallback would therefore need to be an explicitly labelled, non-golden path — or HE files could simply be rejected with a message pointing to Adobe DNG Converter / shooting lossless.
- The DNG route is the other clean option: Adobe DNG Converter decodes HE to a standard (lossless-JPEG or JPEG XL) CFA DNG, so a DNG reader would give Yellowmoon true mosaic data for HE files without touching TicoRAW.

### Gaps
- Could not confirm from a primary Apple source that macOS 26.4 decodes HE/HE*; evidence is one enthusiast site plus forum titles, and Apple's list as fetched shows no lossless-only footnote on Nikon (conflicting with forum reports). Needs a hands-on test on a Mac (owner task).
- Could not retrieve the CIRAWFilter doc body (JS-rendered); the demosaiced-only statement is from prior knowledge.
- No information on intoPIX SDK pricing, whether a decode-only licence is available to an open-source project, or whether any TicoRAW patent numbers/expiry dates are listed (intoPIX links a patents page I did not fetch).
- No statement from LibRaw or rawspeed maintainers found in this session giving their reason for not supporting HE.
- Whether the pixls.us author later published code (after May 2026) is unknown; nothing surfaced in searches.
- How Capture One, DxO and Lightroom obtained decoding (intoPIX SDK vs. own implementation) is not public.

---

## 4. Sensor facts and raw pre-processing quirks

### Takeaway
Most Nikon bodies in scope use Sony Semiconductor sensors, but usually Nikon-specific variants rather than the exact part in a Sony Alpha body, so few have a published-spectral-curve "sibling" in the Sony set; identifications come from teardowns and rumour sites and are only partly reliable.

### Cited Findings
- Z9 sensor is Sony IMX609AQJ (TechInsights teardown, via Nikon Rumors). — [Nikon Rumors, mirrorless sensor list, 2025-09-28](https://nikonrumors.com/2025/09/28/who-produces-the-sensors-for-nikons-mirrorless-cameras.aspx/)
- Z7/Z7II: Sony IMX309BQJ (BSI); D850: IMX309AQJ, with a note that the die itself may be TowerJazz-made. — [Nikon Rumors, "Confirmed: the sensor inside the Nikon Z7 camera is made by Sony"](https://nikonrumors.com/2021/01/24/confirmed-the-sensor-inside-the-nikon-z7-camera-is-made-by-sony.aspx/); [Nikon Rumors DSLR sensor list](https://nikonrumors.com/?p=100118)
- Z6III and ZR: Sony IMX820AQJ, 24.5 MP partially stacked (the list originally said Tower and was corrected: "the Nikon Z6III sensor is made by Sony"). — [Nikon Rumors, 2025-09-28](https://nikonrumors.com/2025/09/28/who-produces-the-sensors-for-nikons-mirrorless-cameras.aspx/)
- Same list: Z5 (24.3 MP FSI, Sony), Z6II (24.5 MP BSI, Sony — the page attaches "IMX820" to the Z6II row, which looks like an error), Z8 (45.7 MP stacked, Sony, inferred from Z9), Zf and Z5II (24.5 MP BSI, Sony, unsourced), Zfc/Z30 (20.9 MP, Sony, flagged "possibly incorrect"), Z50/Z50II (20.9 MP, Tower Semiconductor, flagged "possibly incorrect"). The page says: "the above information is not official, and some sensor manufacturers may be incorrect." — [Nikon Rumors, 2025-09-28](https://nikonrumors.com/2025/09/28/who-produces-the-sensors-for-nikons-mirrorless-cameras.aspx/)
- Another secondary list describes Z6, Z6II, Z5 sensors as "similar to IMX128". — [Nikon Rumors DSLR sensor list](https://nikonrumors.com/?p=100118) (conflicts with the common attribution of the Z6 to IMX410; see Gaps)
- Bit depth: Z8 NEF is 14-bit only; Z6III stills use 14-bit readout. — [Nikon Z8 specifications](https://onlinemanual.nikonimglib.com/z8/en/specifications_364.html); [DPReview Z6III review](https://www.dpreview.com/reviews/8674594944)
- NEFBitDepth tag distinguishes 12- and 14-bit CFA files. — [ExifTool Nikon tags](https://exiftool.org/TagNames/Nikon.html)
- Lossy NEF quantises highlights only (tables 1:1 below a threshold); the author calls it "visually lossless". — [Bill Claff, NEF Compression](https://photonstophotos.net/NikonInfo/NEF_Compression.htm)
- A known darktable issue thread exists for wrong white balance on Z6 files, i.e. WB extraction on Z bodies has tripped open implementers. — [pixls.us "Darktable Nikon Z6 white balance error"](https://discuss.pixls.us/t/darktable-nikon-z6-white-balance-error/10919/2)

### Inferences
- **[from memory, unverified]** All Nikon interchangeable-lens bodies in the brief use a conventional 2×2 Bayer CFA (no X-Trans/quad-Bayer); CFA phase is given by the TIFF CFAPattern tag and is typically RGGB for most modern bodies, with exceptions in older DSLRs (GBRG/BGGR) and shifts when the camera crops (DX crop on FX bodies), so the pattern must be read per file, not per model.
- **[from memory, unverified]** Sony-sibling candidates for borrowing a published spectral profile: Z6/Z6II (24.5 MP BSI, widely reported as IMX410-class) ↔ Sony A7 III; Zf/Z5II (24.5 MP BSI) ↔ same family; D810 (36 MP, IMX094-class) ↔ Sony A7R; D750/D610/Z5 (24 MP FSI, IMX128-class) ↔ roughly Sony A7/A7 II generation; D850/Z7/Z7II (45.7 MP IMX309) has **no** Sony body; Z9/Z8 (IMX609, 45.7 MP stacked) has **no** Sony body (A1 is a different 50 MP part); D500/D7500/Z50/Zfc/Z30 (20.9 MP DX) has **no** Sony body. Even where the silicon is shared, Nikon specifies its own CFA dyes, microlenses and IR-cut/low-pass stack, so a Sony body's spectral sensitivities are only an approximation — the IR-cut filter alone changes the red channel's long-wavelength roll-off. Spectral data measured on the Nikon body itself is preferable where a public database has it.
- **[from memory, unverified]** Raw pre-processing quirks widely reported for Nikon: (1) on many DSLRs (D300/D700/D7000 era and later) the camera digitally scales R and B raw values before writing the NEF ("white-balance pre-conditioning"), visible as gaps/combing in R/B histograms; it is baked in and needs no decoder action but affects noise modelling; (2) hot-pixel suppression is applied to raw data for long exposures and cannot be disabled; (3) older bodies (D70–D300) clip black to zero in camera (no pedestal), so the black level is 0 and read-noise is half-clipped, whereas newer bodies (D7100/D5300 onward and all Z) keep a pedestal recorded in makernote 0x003d (commonly 1008 at 14-bit on Z bodies, 600 on D850-era, scaled /4 for 12-bit); (4) Z6/Z7-family on-sensor PDAF rows are interpolated/scaled in camera and can show as horizontal banding in deep shadow lifts; (5) Z bodies apply extra digital scaling at some ISOs. These should be sourced from photonstophotos.net and DPReview before being cited in a spec.
- White level is not stored in the file for most NEFs; open decoders carry per-camera, sometimes per-ISO, white-level tables (inference from the existence of rawspeed's `cameras.xml` — LGPL data, must not be copied; values need to be measured from public sample files).

### Gaps
- No authoritative (Nikon or teardown-report) source confirming the sensor part for Z5, Z6, Z6II, Zf, Z5II, Zfc, Z30, Z50, Z50II, D780, D750, D810, D500, D7500. The two Nikon Rumors lists contradict each other and themselves (Z6 "similar to IMX128" vs the common IMX410 attribution; "IMX820" on the Z6II row).
- Did not retrieve photonstophotos/DPReview pages documenting WB pre-scaling, hot-pixel suppression, PDAF banding or black-level values; all quirk statements above are from memory.
- Per-body CFA phase, black level and white level tables were not compiled; they must be derived from sample files (with rawler/LibRaw as CI oracle only).
- Availability of public spectral sensitivity measurements for Nikon bodies (e.g. in rawtoaces-data or other open databases) was outside the fetches I made; the V1 data rule makes this the deciding factor for which Nikon bodies could be supported at all.

---

## 5. Permissively licensed references for NEF decoding

### Takeaway
There is no permissively licensed, maintained NEF decoder I could verify; the usable permissive assets are documentation (ExifTool tag tables, Bill Claff's notes, libopenraw's format notes) and CC0 sample files, while every mature decoder (dcraw lineage, LibRaw, rawspeed, rawler/dnglab, libopenraw, exiv2) is copyleft or otherwise restricted and can serve only as reading material and CI oracle.

### Cited Findings
- ExifTool's Nikon tag documentation covers the makernote tag IDs, value tables, ColorBalance/LensData/ShotInfo versions and encryption notes used throughout §1. — [ExifTool Nikon tags](https://exiftool.org/TagNames/Nikon.html)
- exiv2 publishes an equivalent Nikon tag reference. — [exiv2 Nikon tags](https://exiv2.org/tags-nikon.html)
- libopenraw publishes short NEF/NRW format notes (compression values 1 / 32769 / 34713, D100 quirk, 0x8c/0x96 tables) and points to dcraw for the algorithm; the page shows only "© 2025 libopenraw" and no document licence. — [libopenraw NEF format notes](https://libopenraw.pages.freedesktop.org/formats/nef)
- Bill Claff's "NEF Compression" note documents the lossy tables per body. — [photonstophotos](https://photonstophotos.net/NikonInfo/NEF_Compression.htm)
- A historical write-up of the D70 NEF format/compression exists (Fazal Majid). — [majid.info](https://blog.majid.info/is-the-nikon-d70-nef-raw-format-truly-lossless/)
- A Rust crate `bioformats` contains a Nikon decoder module (`src/tiff/nikon.rs`, a port in the Bio-Formats lineage, whose Java original has a `NikonCodec`). — [docs.rs bioformats](https://docs.rs/crate/bioformats/0.1.3/source/src/tiff/nikon.rs); [Bio-Formats NikonCodec API](https://downloads.openmicroscopy.org/bio-formats/5.2.2/api/loci/formats/codec/NikonCodec.html); [Bio-Formats NEF format page](https://bio-formats.readthedocs.io/en/v7.2.0/formats/nef.html)
- A Rust users' forum thread discusses writing a NEF decoder from scratch. — [users.rust-lang.org](https://users.rust-lang.org/t/decoding-of-nikon-raw-format-nef/48261)

### Inferences
- **Licences, [from memory — verify each in the repo's LICENSE file before relying on it]:**
  - ExifTool: "same terms as Perl itself", i.e. dual Artistic License 1.0 / GPL (v1 or later). Reading the tag documentation as a factual reference is fine; copying its Perl code or verbatim tables into an Apache-2.0 crate is not clearly compatible (Artistic 1.0 is not on the usual Apache-compatible list).
  - dcraw: Dave Coffin's own licence — free use for most of the file, with GPL-or-commercial conditions on certain marked sections; treated by the project rules as no-copy.
  - LibRaw: dual LGPL-2.1 / CDDL-1.0. rawspeed: LGPL-2.1+. darktable: GPL-3.0+. RawTherapee and ART: GPL-3.0. rawler/dnglab: LGPL-2.1. rawloader (pedrocr): LGPL-2.1. libopenraw: LGPL-3.0+. exiv2: GPL-2.0+. librtprocess: GPL-3.0. All: read-only / oracle-only under AGENTS.md rule 1.
  - Bio-Formats (Java): GPL-2.0+ for the main formats, with a BSD-2 subset (`formats-bsd`); which side `NikonCodec` sits on, and the licence of the Rust `bioformats` crate, were not checked — treat as copyleft until verified.
  - Adobe DNG SDK: Adobe's own permissive "DNG SDK License" (redistribution in source/binary allowed with notice); it contains **no NEF decoder** — it is useful only for the lossless-JPEG codec, DNG opcodes (WarpRectilinear, FixVignetteRadial) and the colour model. The DNG specification itself carries a royalty-free patent licence for compliant implementations.
  - Nikon SDK (Imaging SDK / "NEF codec" downloads): proprietary binary under a Nikon licence agreement, Windows/macOS only, no source; not usable in an Apache-2.0 Rust core and not a format specification.
  - Graphite's `rawkit` (MIT/Apache-2.0) is a clean permissive Rust raw decoder but, as far as I know, Sony-ARW only; `quickraw` (Rust) and `imagepipe` are LGPL. I know of no MIT/BSD/Apache NEF decoder in Rust, Go or C that is maintained and covers the Huffman variants; small hobby implementations on GitHub are mostly transliterations of dcraw and inherit its terms in practice.
- Practical clean-room route: take tag semantics from ExifTool docs; take the algorithm description (predictive Huffman with category codes, curve, split) from prose sources and from *reading* copyleft code; re-derive the Huffman code tables and xlat cipher tables as data; validate bit-exactly against rawler/LibRaw in `tools/oracle`.

### Gaps
- Licences listed under Inferences were not re-fetched in this session; each should be confirmed from the upstream LICENSE file before appearing in `docs/14-licensing.md`.
- Licence of the Rust `bioformats` crate and of the specific Bio-Formats Java `NikonCodec` file not verified.
- I found no Wikipedia-grade or academic prose description of the NEF Huffman tables or the WB cipher; these appear to exist only in code.
- Did not locate Nikon's SDK licence text.

---

## 6. Lens correction metadata in NEF

### Takeaway
Z-mount NEFs embed distortion and vignetting polynomial coefficients in a Nikon-private block (NEFInfo, SubIFD1 tag 0xc7d5) that ExifTool now names, and makernote tags record whether the corrections were enabled; the mathematical model was reverse-engineered by the community rather than documented by Nikon, and lateral-CA parameters are not documented at all.

### Cited Findings
- NEFInfo (SubIFD1 tag 0xc7d5) → 0x0005 DistortionInfo: DistortionCorrectionVersion (string[4], offset 0); DistortionCorrection (int8u, offset 4: 0 No Lens Attached, 1 On (Optional), 2 Off, 3 On (Required)); RadialDistortionCoefficient1/2/3 (rational64s at offsets 20, 28, 36). — [ExifTool Nikon tags](https://exiftool.org/TagNames/Nikon.html)
- NEFInfo → 0x0006 VignetteInfo: VignetteCorrectionVersion (offset 0); VignetteCoefficient1/2/3 (rational64s at offsets 36, 52, 68). — [ExifTool Nikon tags](https://exiftool.org/TagNames/Nikon.html)
- Makernote 0x002a VignetteControl (0 Off, 1 Low, 3 Normal, 5 High); 0x002b DistortInfo with AutoDistortionControl at index 4 (0 Off, 1 On, 2 On (underwater)). — [ExifTool Nikon tags](https://exiftool.org/TagNames/Nikon.html)
- ExifTool's tables show no chromatic-aberration coefficient tags. — [ExifTool Nikon tags](https://exiftool.org/TagNames/Nikon.html)
- The distortion/vignetting blocks were identified by ExifTool forum users from Z6 files; a contributor suggested the three radial coefficients may match the parameters of Adobe's DNG WarpRectilinear opcode. — [ExifTool forum, "Nikon Z6 NEF – Extract: Lens Distortion / Vignetting / Chromatic Aberration"](https://exiftool.org/forum/index.php?msg=71840)
- Community reverse-engineering threads exist for the Z-series correction model (parts 1 and 2) and a proposed darktable module using embedded manufacturer corrections. — [pixls.us "Reverse engineering Nikon Z-series lens correction"](https://discuss.pixls.us/t/reverse-engineering-nikon-z-series-lens-correction/36733); [part 2](https://discuss.pixls.us/t/reverse-engineering-nikon-z-series-lens-correction-part-2/41635?page=2); [manufacturer lens correction module](https://discuss.pixls.us/t/manufacturer-lens-correction-module-for-darktable/21543?page=3)
- darktable's Lensfun-based path depends on the Z-mount focus-distance tag (Exif.NikonLd4.FocusDistance); 4.2 did not read it and users reported vignetting correction failing after 4.4. — [pixls.us "Lens Correction not fixing Vignetting"](https://discuss.pixls.us/t/lens-correction-not-fixing-vignetting/38363)
- Lightroom/ACR apply a "built-in" profile for Z lenses and honour the in-camera vignette-control level. — [Photography Life](https://photographylife.com/nikon-z-lightroom-profile)

### Inferences
- The coefficients are per-shot values (already evaluated for the focal length/focus distance/aperture of that frame), which is why three numbers suffice; this matches the "On (Required)" flag for lenses designed around software distortion correction. An implementer needs only a radial polynomial warp and a radial gain — both already standard for DNG opcodes.
- **[from memory, unverified]** darktable 4.4+ "embedded metadata" lens correction supports Sony/Fuji/Olympus/Panasonic-style embedded data and DNG opcodes; Nikon Z support was discussed in the threads above but I did not confirm it was merged.
- F-mount DSLR NEFs carry no correction coefficients (in-camera correction used a firmware lens database), so DSLR files would need Lensfun-style external profiles (Lensfun database is CC BY-SA 3.0; the library LGPL-3.0 — **[from memory]**).

### Gaps
- The exact polynomial form (which radius normalisation; whether coefficients multiply r², r⁴, r⁶; what the unnamed rationals between the listed offsets mean — VignetteInfo offsets step by 16, implying interleaved unknown values) was not extracted; the pixls threads would need to be read in full.
- No documented source for lateral chromatic aberration parameters in NEF.
- Conflict between sources on where the data sits (makernote 0x002b vs SubIFD1 0xc7d5 → 0x0005); my reading is that 0x002b holds the on/off setting and 0xc7d5 holds coefficients, but this should be confirmed on a real Z file.

---

## 7. Public sample NEF files for tests

### Takeaway
raw.pixls.us is the canonical source: mostly CC0, rsync-able, with SHA-256 file lists, and it deliberately collects every Nikon compression × bit-depth × size combination; HE/HE* samples there are uncertain, with the readily available HE samples being forum uploads under other licences.

### Cited Findings
- "All new samples are in Public Domain under CC0 1.0 license (85+% of samples and counting), however some older samples are still under more restrictive CC BY-NC-SA 4.0 license." — [rawspeed Reference Sample Archive](https://darktable-org.github.io/rawspeed/ReferenceSampleArchive.html)
- Full set: `https://raw.pixls.us/data/`; hand-picked "masterset": `https://raw.pixls.us/data-unique/` (about 1/22 of the disk footprint and 1/44 of the sample count of the full set; public-domain samples only); fetch with `rsync://raw.pixls.us/data-unique/`; each set has a top-level `filelist.sha256` (sha256sum format) and `timestamp.txt`. — [rawspeed Reference Sample Archive](https://darktable-org.github.io/rawspeed/ReferenceSampleArchive.html)
- The site requests, for Nikon, every combination of 12/14-bit × compressed / lossless / uncompressed, plus small and medium NEF. — [raw.pixls.us](https://raw.pixls.us)
- HE/HE* samples in the wild: Z9 raws from photographylife.com were used for reverse engineering; forum members posted Z6III HE* files, Z8 ColorChecker shots and a Z8 ISO 64–12800 series (April–May 2026). — [pixls.us "Nikon HE* decoder"](https://discuss.pixls.us/t/nikon-he-decoder/56955); [p.3](https://discuss.pixls.us/t/nikon-he-decoder/56955?page=3)
- A Z6III HE* NEF plus its Adobe-converted DNG is posted under CC BY-SA 4.0. — [pixls.us Play Raw, 2026-07-20](https://discuss.pixls.us/t/beating-nikon-high-efficieny-nefs/59347)
- Photography Life hosts Z9 lossless vs HE comparison files. — [Photography Life](https://photographylife.com/nikon-z9-lossless-vs-high-efficiency-raw-files)

### Inferences
- The existing `tools/fetch/` pattern (download + SHA-256) maps directly onto raw.pixls.us: pin individual file URLs and hashes from `filelist.sha256`, and filter to CC0 entries (licence is shown per sample on the site). The CC BY-NC-SA remainder is fine for local/CI testing but should be avoided for anything redistributed.
- The pixls.us pair (HE* NEF + Adobe DNG of the same frame) is a ready-made reference for a "reject HE with a helpful message" test and, if a DNG reader is ever added, for an HE-via-DNG path. CC BY-SA 4.0 is acceptable for a downloaded test input that is never committed.
- Other sources **[from memory, unverified licences]**: DPReview and Imaging Resource sample galleries (all-rights-reserved, download for personal evaluation — not suitable for CI redistribution but usable by fetch scripts only if their terms allow automated download, which is doubtful); rawsamples.ch (old bodies only, licence unclear).

### Gaps
- Did not enumerate which Nikon bodies/modes are actually present on raw.pixls.us (needs a crawl of the site's camera list or `filelist.sha256`); in particular whether any HE/HE* files were accepted (the site's coverage is validated against rawspeed, which cannot decode them).
- No current coverage percentage; the last published figure (about 77 %) is from 2017 — [pixls.us blog](https://pixls.us/blog/2017/09/keep-the-raws-coming).

---

## 8. Implementation effort relative to ARW, and typical pitfalls

### Takeaway
Classic NEF (uncompressed + lossless + lossy Huffman) is a moderate, well-trodden job comparable in codec complexity to ARW but with more metadata hazards (encrypted WB, makernote-relative offsets, per-model quirks); HE/HE* is a separate, much larger and legally risky project that no open implementation has completed.

### Cited Findings
- The classic codec is a "lossy encoding followed by non-adaptive Huffman compression" with a small set of fixed table sizes — i.e. one entropy coder with table variants, not a family of unrelated codecs. — [Bill Claff, NEF Compression](https://photonstophotos.net/NikonInfo/NEF_Compression.htm)
- Known pitfalls reported by implementers/users: D100 files mislabelled as compressed; D1X non-square pixels; padding bytes in packed 12-bit rows (Compression 32769). — [libopenraw NEF format notes](https://libopenraw.pages.freedesktop.org/formats/nef)
- "Lossy after split" 14-bit files needed a specifically hunted sample to get right. — [pixls.us sample request thread](https://discuss.pixls.us/t/nikon-a-specific-raw-sample-wanted/5483/9)
- A batch of NEFs that failed in darktable and RawTherapee but opened in Lightroom/Photoshop shows that edge-case files still surface. — [pixls.us](https://discuss.pixls.us/t/darktable-unable-to-open-a-number-of-nefs-in-darktable-rawtherapee-but-ok-in-lr-and-ps/21048?page=2)
- White balance on Z6 files was wrong in darktable at launch. — [pixls.us](https://discuss.pixls.us/t/darktable-nikon-z6-white-balance-error/10919/2)
- New bodies (Z50II) needed explicit support additions in darktable 5.2 even for non-HE files. — [pixls.us "Z50II and darktable 5.2.0"](https://discuss.pixls.us/t/z50ii-and-darktable-5-2-0/50822)
- HE: about one month of LLM-assisted reverse engineering produced a viewable but incorrect image and then stalled. — [pixls.us "Nikon HE* decoder"](https://discuss.pixls.us/t/nikon-he-decoder/56955?page=3)

### Inferences
- Rough sizing (inference from the structure above and from the size of the corresponding modules in open decoders, **[from memory]**: rawspeed's NefDecoder + NikonDecompressor and rawler's nef module are each on the order of 1–1.5 k lines, similar to their ARW counterparts):
  - TIFF/IFD walking is reusable from the ARW decoder; the new part is the nested makernote TIFF with its own base offset and byte order.
  - Codec work: one bit-reader + Huffman category decoder with six tables, 2-column-back prediction, curve lookup, split handling; plus trivial unpackers (12/14-bit packed, padded, big/little-endian). ARW by comparison has ARW2 block-delta lossy, lossless-JPEG tiles (ARW 4/5) and uncompressed — so NEF's codec surface is about the same or slightly smaller.
  - Metadata work is larger than ARW: versioned + encrypted ColorBalance (four channel orders, several offsets), fallback to 0x000c WB_RBLevels, black level in 0x003d with 12/14-bit scaling, white levels not in file, CropArea, CFA phase per file, sNEF detection and rejection, HE detection and rejection.
  - Per-camera data (white level, crop/masked areas, colour profile) is the real long tail, and for Yellowmoon the binding constraint is spectral data availability per Nikon body, not the decoder.
- Pitfall list for a spec (mix of cited and remembered): makernote offsets relative to inner TIFF header; non-numeric or absent serial numbers when deriving the WB key; ShutterCount tag absent on some files (then decryption impossible — fall back to 0x000c); 12-bit files whose black level is stored at 14-bit scale; lossy files whose curve has fewer entries than 2^bits and needs interpolation; the split row switching Huffman tables mid-image; BitsPerSample vs NEFBitDepth disagreement; multiple SubIFDs where the largest is not necessarily the raw one (JPEG preview can be large) — select by NewSubfileType/Compression; DX-crop and other in-camera crops changing dimensions and CFA phase; pixel-shift and multiple-exposure NEFs; files rewritten by Nikon Transfer/NX Studio or Capture NX (added NikonCapture blocks, moved offsets); zero-length or truncated strips from card errors.
- Determinism (hard rule 8) is easy for classic NEF: the codec is integer-exact, so bit-exact comparison against rawler/LibRaw in the oracle is feasible, as with ARW.

### Gaps
- No published effort estimates from implementers; sizing above is my inference.
- Line counts of rawspeed/rawler modules are from memory and were not re-measured.
- I did not review LibRaw/rawspeed issue trackers directly in this session (tool budget); pitfalls are from forum threads and prior knowledge.
