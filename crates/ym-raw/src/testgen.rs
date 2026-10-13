//! Test-only synthetic ARW writer (uncompressed, cRAW and LJPEG encoders) that produces files for decoder, oracle and pipeline tests; compiled for tests or with the `testgen` feature (`docs/04-raw-decoding.md` section Tests, `docs/13-testing.md`).
//!
//! There are no real ARW files in this repository's test environment, so every container, metadata
//! and decoder test starts from a file produced here. [`ArwBuilder`] lays a file out the way a Sony
//! body does and is written independently of the reader in `tiff`, `exif`, `makernote` and `sr2`
//! (only the pad cipher in `sr2::apply_pad` and the byte cipher in `makernote` are shared; the
//! LibRaw oracle test checks both against LibRaw's own implementation).
//!
//! # Layout of a generated file
//!
//! ```text
//! header "II*\0", IFD0 at 8
//! IFD0     Make, Model, Orientation, resolution, Software, DateTime,
//!          SubIFDs -> raw IFD, YCbCrPositioning, ExifIFD, GPSInfo (optional),
//!          DNGPrivateData -> SR2Private (4 bytes, little-endian offset)
//! IFD1     the thumbnail: Compression 6, JPEGInterchangeFormat + Length
//! ExifIFD  ExposureTime, FNumber, ISOSpeedRatings, SensitivityType, RecommendedExposureIndex,
//!          DateTimeOriginal and offsets, ExposureBiasValue, FocalLength, MakerNote, LensModel, ...
//! GPS IFD  (optional)
//! MakerNote IFD without a header (older layouts add "SONY DSC \0\0\0" in front): PreviewImage
//!          0x2001 (32-byte header + JPEG, a generated gradient or the caller's own JPEG),
//!          FileFormat 0xB000, SonyModelID 0xB001, enciphered table 0x9405 or 0x9416 holding
//!          SonyISO, the exposure time (as a 1/256-stop word and as a fraction), the F-number and,
//!          in 0x9416, the focal length
//! SR2Private  0x7200 offset, 0x7201 length, 0x7221 key
//! raw IFD  NewSubFileType 0, size, BitsPerSample, Compression, Photometric 32803, strips or
//!          tiles, CFARepeatPatternDim, CFAPattern, 0x7000, 0x7010, 0x7032, 0x7035, 0x7037,
//!          0x7038, 0x74C7, 0x74C8 (and, optionally, 0x7310 / 0x7313)
//! thumbnail JPEG
//! SR2 sub-IFD, encrypted: black 0x7310 (or 0x7300), WB 0x7313 (or 0x7303), white 0x787F
//! raw data (strips or tiles), last, as in the real files
//! ```
//!
//! # Payloads
//!
//! The raw data is produced by a [`RawPayload`]. [`Uncompressed`] writes 16-bit little-endian
//! words (TIFF compression 1) and [`ContainerOnly`] writes a correct container description with
//! filler bytes for the compressed variants, enough for probe and dispatch tests. The decoder
//! task implements [`RawPayload`] for 12-bit packed, cRAW, lossless L and lossless M/S and
//! passes it to [`ArwBuilder::payload`].
//!
//! Everything here is deterministic: no clocks, no randomness except the seeded
//! [`SplitMix64`].

use crate::layout::{CfaPattern, ChromaSubsampling, Rect};
use crate::makernote::{encipher_byte, sony_word_for_iso};
use crate::sr2;

pub mod craw_enc;
pub mod ljpeg_enc;
pub mod lossless;

pub use craw_enc::{CrawEncoder, Rounding};
pub use ljpeg_enc::HuffmanChoice;
pub use lossless::{LosslessL, LosslessMs};

/// A small deterministic random number generator (SplitMix64) for test content.
#[derive(Clone, Debug)]
pub struct SplitMix64(pub u64);

impl SplitMix64 {
    /// The next 64 random bits.
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A value in `0..bound` (bound must be non-zero).
    pub fn below(&mut self, bound: u64) -> u64 {
        self.next_u64() % bound.max(1)
    }
}

/// Makes a deterministic test mosaic: a smooth diagonal gradient plus noise, between `black` and
/// `white` (inclusive), `width * height` values in row-major order.
pub fn gradient_mosaic(width: u32, height: u32, black: u16, white: u16, seed: u64) -> Vec<u16> {
    let mut rng = SplitMix64(seed);
    let span = u64::from(white.saturating_sub(black)).max(1);
    let (w, h) = (u64::from(width.max(1)), u64::from(height.max(1)));
    let mut out = Vec::with_capacity(width as usize * height as usize);
    for y in 0..u64::from(height) {
        for x in 0..u64::from(width) {
            let base = (x * 3 + y * 2) * span / (w * 3 + h * 2);
            let noise = rng.below(span / 32 + 1);
            let value = (u64::from(black) + base / 2 + noise).min(u64::from(white));
            out.push(value as u16);
        }
    }
    out
}

/// Makes a deterministic camera-RGB test image: a smooth diagonal gradient per channel plus a
/// little noise, between `low` and `high`, `3 * width * height` values (R, G, B per pixel).
pub fn gradient_rgb(width: u32, height: u32, low: u16, high: u16, seed: u64) -> Vec<u16> {
    let mut rng = SplitMix64(seed);
    let span = u64::from(high.saturating_sub(low)).max(1);
    let (w, h) = (u64::from(width.max(1)), u64::from(height.max(1)));
    let mut out = Vec::with_capacity(width as usize * height as usize * 3);
    for y in 0..u64::from(height) {
        for x in 0..u64::from(width) {
            for c in 0..3u64 {
                let phase = match c {
                    0 => x * 3 + y,
                    1 => x + y * 3,
                    _ => (w - 1 - x) * 2 + y * 2,
                };
                let base = phase * span / (w * 3 + h * 3);
                let noise = rng.below(span / 64 + 1);
                out.push((u64::from(low) + base + noise).min(u64::from(high)) as u16);
            }
        }
    }
    out
}

/// Encodes a small colour gradient as a baseline JPEG (`FF D8 FF DB ...`), without any
/// dependency.
///
/// The image is made of 8 x 8 blocks that carry only a DC coefficient (a flat colour per block),
/// which is a complete, valid JPEG: three components, 4:4:4, one quantisation table, small custom
/// Huffman tables (DC categories 0 to 11 with 4-bit codes, AC with the single symbol "end of
/// block"). `seed` perturbs the colours so different previews differ. The stream starts with the
/// quantisation table like the previews Sony cameras write (ExifTool only accepts a maker-note
/// preview that starts `FF D8 FF DB`).
pub fn tiny_jpeg(width: u16, height: u16, seed: u64) -> Vec<u8> {
    let (blocks_x, blocks_y) = (
        usize::from(width).div_ceil(8),
        usize::from(height).div_ceil(8),
    );
    let mut rng = SplitMix64(seed);

    let mut out = vec![0xFF, 0xD8];
    // DQT: table 0, 8-bit, every step 16.
    out.extend_from_slice(&[0xFF, 0xDB, 0x00, 0x43, 0x00]);
    out.extend(std::iter::repeat_n(16u8, 64));
    // SOF0: 8-bit samples, three components with 1x1 sampling and quantisation table 0.
    out.extend_from_slice(&[0xFF, 0xC0, 0x00, 0x11, 0x08]);
    out.extend_from_slice(&height.to_be_bytes());
    out.extend_from_slice(&width.to_be_bytes());
    out.extend_from_slice(&[0x03, 0x01, 0x11, 0x00, 0x02, 0x11, 0x00, 0x03, 0x11, 0x00]);
    // DHT: DC table 0 (twelve 4-bit codes for categories 0 to 11), AC table 0 (EOB only).
    out.extend_from_slice(&[0xFF, 0xC4, 0x00, 0x1F, 0x00]);
    out.extend_from_slice(&[0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    out.extend(0u8..12);
    out.extend_from_slice(&[0xFF, 0xC4, 0x00, 0x14, 0x10]);
    out.extend_from_slice(&[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    out.push(0x00);
    // SOS: all three components use table 0 for DC and AC; baseline spectral selection.
    out.extend_from_slice(&[
        0xFF, 0xDA, 0x00, 0x0C, 0x03, 0x01, 0x00, 0x02, 0x00, 0x03, 0x00,
    ]);
    out.extend_from_slice(&[0x00, 0x3F, 0x00]);

    let mut bits = BitWriter::default();
    let mut previous = [0i32; 3];
    for by in 0..blocks_y {
        for bx in 0..blocks_x {
            let noise = rng.below(9) as i32 - 4;
            // Quantised DC values in -64..=63 (a step of 16 with the DCT's factor of 8).
            let dc = [
                (bx * 90 / blocks_x.max(1)) as i32 - 45 + noise,
                (by * 70 / blocks_y.max(1)) as i32 - 35,
                ((bx + by) * 40 / (blocks_x + blocks_y).max(1)) as i32 - 20 - noise,
            ];
            for (component, value) in dc.into_iter().enumerate() {
                let diff = value - previous[component];
                previous[component] = value;
                let category = 32 - diff.unsigned_abs().leading_zeros();
                bits.put(category, 4);
                if category > 0 {
                    // Negative differences are stored as the one's complement.
                    let magnitude = if diff < 0 { diff - 1 } else { diff };
                    bits.put((magnitude as u32) & ((1 << category) - 1), category);
                }
                bits.put(0, 1); // AC: end of block
            }
        }
    }
    out.extend_from_slice(&bits.finish());
    out.extend_from_slice(&[0xFF, 0xD9]);
    out
}

/// Pads a JPEG to at least `min_len` bytes with `COM` (comment) segments placed right after the
/// first segment, so the stream still starts `FF D8 FF DB`. Returns the JPEG unchanged when it is
/// already long enough. Real previews are a few hundred kilobytes; this makes timing tests
/// realistic.
pub fn pad_jpeg_to(jpeg: Vec<u8>, min_len: usize) -> Vec<u8> {
    if jpeg.len() >= min_len || jpeg.get(..4) != Some(&[0xFF, 0xD8, 0xFF, 0xDB]) {
        return jpeg;
    }
    let first_len = usize::from(u16::from_be_bytes([jpeg[4], jpeg[5]]));
    let split = 4 + first_len;
    let mut out = jpeg[..split].to_vec();
    let mut missing = min_len - jpeg.len();
    while missing > 0 {
        // A COM segment is FF FE, a 16-bit length (counting itself) and the comment.
        let data = missing.saturating_sub(4).clamp(1, 65_533);
        out.extend_from_slice(&[0xFF, 0xFE]);
        out.extend_from_slice(&((data + 2) as u16).to_be_bytes());
        out.extend(std::iter::repeat_n(b' ', data));
        missing = missing.saturating_sub(data + 4);
    }
    out.extend_from_slice(&jpeg[split..]);
    out
}

/// `(width, height)` from the first frame header of a JPEG, or `None` when it has none.
fn jpeg_dimensions(jpeg: &[u8]) -> Option<(u16, u16)> {
    // `pos` is the length field of the frame header, then the precision (1 byte), height and width.
    let (_, pos) = crate::metadata::jpeg_frame_header(jpeg)?;
    let height = u16::from_be_bytes([*jpeg.get(pos + 3)?, *jpeg.get(pos + 4)?]);
    let width = u16::from_be_bytes([*jpeg.get(pos + 5)?, *jpeg.get(pos + 6)?]);
    Some((width, height))
}

/// Writes bits most significant first with JPEG byte stuffing (`FF` is followed by `00`).
#[derive(Default)]
struct BitWriter {
    bytes: Vec<u8>,
    accumulator: u32,
    count: u32,
}

impl BitWriter {
    fn put(&mut self, value: u32, bits: u32) {
        for i in (0..bits).rev() {
            self.accumulator = (self.accumulator << 1) | ((value >> i) & 1);
            self.count += 1;
            if self.count == 8 {
                self.flush_byte();
            }
        }
    }

    fn flush_byte(&mut self) {
        let byte = self.accumulator as u8;
        self.bytes.push(byte);
        if byte == 0xFF {
            self.bytes.push(0x00);
        }
        self.accumulator = 0;
        self.count = 0;
    }

    /// Pads the last byte with one bits and returns the stream.
    fn finish(mut self) -> Vec<u8> {
        while self.count != 0 {
            self.put(1, 1);
        }
        self.bytes
    }
}

// ---------------------------------------------------------------------------------------------
// Payloads
// ---------------------------------------------------------------------------------------------

/// How the raw image appears in the raw IFD.
#[derive(Clone, Debug)]
pub struct PayloadDescription {
    /// TIFF `Compression`.
    pub compression: u16,
    /// TIFF `BitsPerSample`.
    pub bits_per_sample: u16,
    /// TIFF `PhotometricInterpretation`.
    pub photometric: u16,
    /// TIFF `SamplesPerPixel`.
    pub samples_per_pixel: u16,
    /// Tile size, or `None` for a single strip.
    pub tile: Option<(u32, u32)>,
    /// `SonyRawFileType` (tag 0x7000).
    pub sony_raw_file_type: Option<u16>,
    /// `YCbCrSubSampling` (tag 0x0212) for camera-processed files.
    pub ycbcr_subsampling: Option<(u16, u16)>,
}

/// Something that turns a mosaic into the bytes of the raw image.
pub trait RawPayload: Send + Sync {
    /// The raw IFD description of the data this payload writes.
    fn describe(&self) -> PayloadDescription;

    /// Encodes `mosaic` (`width * height` sensor values, row-major, on the scale of the file's
    /// black and white levels) into one byte vector per strip or tile. A strip payload returns one
    /// vector. A tiled payload returns tiles left to right, top to bottom, as many as
    /// `ceil(width / tw) * ceil(height / th)`.
    fn encode(&self, mosaic: &[u16], width: u32, height: u32) -> Vec<Vec<u8>>;
}

/// Uncompressed 16-bit little-endian words: TIFF compression 1.
#[derive(Clone, Copy, Debug)]
pub struct Uncompressed {
    /// `BitsPerSample`: 14 for the A7R III and later, 12 for the 12-bit mode. Values must fit.
    pub bits: u16,
}

impl RawPayload for Uncompressed {
    fn describe(&self) -> PayloadDescription {
        PayloadDescription {
            compression: 1,
            bits_per_sample: self.bits,
            photometric: 32803,
            samples_per_pixel: 1,
            tile: None,
            sony_raw_file_type: Some(if self.bits == 12 { 1 } else { 0 }),
            ycbcr_subsampling: None,
        }
    }

    fn encode(&self, mosaic: &[u16], _width: u32, _height: u32) -> Vec<Vec<u8>> {
        vec![mosaic.iter().flat_map(|v| v.to_le_bytes()).collect()]
    }
}

/// Packed 12-bit samples (ARW 2, TIFF compression 32767 with 12 bits): three bytes hold two
/// samples, the first in the low 12 bits.
#[derive(Clone, Copy, Debug, Default)]
pub struct Packed12;

impl RawPayload for Packed12 {
    fn describe(&self) -> PayloadDescription {
        PayloadDescription {
            compression: 32767,
            bits_per_sample: 12,
            photometric: 32803,
            samples_per_pixel: 1,
            tile: None,
            sony_raw_file_type: Some(1),
            ycbcr_subsampling: None,
        }
    }

    fn encode(&self, mosaic: &[u16], width: u32, _height: u32) -> Vec<Vec<u8>> {
        assert_eq!(
            width % 2,
            0,
            "packed 12-bit rows hold an even number of samples"
        );
        let mut out = Vec::with_capacity(mosaic.len() * 3 / 2);
        for pair in mosaic.chunks_exact(2) {
            let (a, b) = (pair[0], pair[1]);
            assert!(a < 4096 && b < 4096, "12-bit samples must be below 4096");
            out.push((a & 0xFF) as u8);
            out.push((((a >> 8) & 0x0F) | ((b & 0x0F) << 4)) as u8);
            out.push((b >> 4) as u8);
        }
        vec![out]
    }
}

/// ARW 1: Huffman-coded 12-bit samples (TIFF compression 32767, 12 bits, a byte count that fits
/// no uncompressed layout), as the A200 to A390 write them.
#[derive(Clone, Copy, Debug, Default)]
pub struct Arw1;

impl RawPayload for Arw1 {
    fn describe(&self) -> PayloadDescription {
        PayloadDescription {
            compression: 32767,
            bits_per_sample: 12,
            photometric: 32803,
            samples_per_pixel: 1,
            tile: None,
            sony_raw_file_type: None,
            ycbcr_subsampling: None,
        }
    }

    fn encode(&self, mosaic: &[u16], width: u32, height: u32) -> Vec<Vec<u8>> {
        vec![crate::decode::arw1::encode_stream(
            mosaic,
            width as usize,
            height as usize,
        )]
    }
}

/// A correct container description with filler data, for variants whose encoders are written by
/// the decoder task. The probe only reads the description and the chunk sizes, so this is enough
/// to test dispatch, error messages and the CLI.
#[derive(Clone, Debug)]
pub struct ContainerOnly {
    desc: PayloadDescription,
    chunk_bytes: ChunkBytes,
    first_chunk_prefix: Vec<u8>,
}

#[derive(Clone, Copy, Debug)]
enum ChunkBytes {
    /// Total bytes = width * height * numerator / denominator (spread over the chunks).
    PerPixel { numerator: u64, denominator: u64 },
    /// Every tile has this many bytes.
    PerTile(usize),
}

impl ContainerOnly {
    /// cRAW: compression 32767, one byte per pixel and, as in every real cRAW file, BitsPerSample
    /// 12 (`ContainerOnly::other(32767, 8)` gives the 8 bits the format description uses).
    pub fn craw() -> Self {
        Self {
            desc: PayloadDescription {
                compression: 32767,
                bits_per_sample: 12,
                photometric: 32803,
                samples_per_pixel: 1,
                tile: None,
                sony_raw_file_type: Some(2),
                ycbcr_subsampling: None,
            },
            chunk_bytes: ChunkBytes::PerPixel {
                numerator: 1,
                denominator: 1,
            },
            first_chunk_prefix: Vec::new(),
        }
    }

    /// Packed 12-bit uncompressed (ARW 2): compression 32767, 12 bits, 1.5 bytes per pixel.
    pub fn packed12() -> Self {
        Self {
            desc: PayloadDescription {
                compression: 32767,
                bits_per_sample: 12,
                photometric: 32803,
                samples_per_pixel: 1,
                tile: None,
                sony_raw_file_type: Some(1),
                ycbcr_subsampling: None,
            },
            chunk_bytes: ChunkBytes::PerPixel {
                numerator: 3,
                denominator: 2,
            },
            first_chunk_prefix: Vec::new(),
        }
    }

    /// Lossless L: compression 7, 14 bits, 512 x 512 tiles.
    pub fn lossless_l() -> Self {
        Self {
            desc: PayloadDescription {
                compression: 7,
                bits_per_sample: 14,
                photometric: 32803,
                samples_per_pixel: 1,
                tile: Some((512, 512)),
                sony_raw_file_type: Some(3),
                ycbcr_subsampling: None,
            },
            chunk_bytes: ChunkBytes::PerTile(300),
            first_chunk_prefix: Vec::new(),
        }
    }

    /// Lossless M or S: compression 7, YCbCr, 3 samples, 512 x 512 tiles. The first tile starts
    /// with a minimal lossless-JPEG header (`SOI`, `SOF3`) whose first component has the sampling
    /// of `subsampling`. When `with_tag` is false the `YCbCrSubSampling` tag is left out so the
    /// reader has to look at that header.
    pub fn lossless_ms(subsampling: ChromaSubsampling, with_tag: bool) -> Self {
        let (h, v) = match subsampling {
            ChromaSubsampling::Yuv444 => (1u8, 1u8),
            ChromaSubsampling::Yuv422 => (2, 1),
            ChromaSubsampling::Yuv420 => (2, 2),
        };
        let mut prefix = vec![0xFF, 0xD8, 0xFF, 0xC3, 0, 17, 14, 0, 8, 0, 8, 3];
        prefix.extend_from_slice(&[1, (h << 4) | v, 0, 2, 0x11, 0, 3, 0x11, 0]);
        Self {
            desc: PayloadDescription {
                compression: 7,
                bits_per_sample: 14,
                photometric: 6,
                samples_per_pixel: 3,
                tile: Some((512, 512)),
                sony_raw_file_type: Some(3),
                ycbcr_subsampling: with_tag.then_some(match subsampling {
                    ChromaSubsampling::Yuv444 => (1, 1),
                    ChromaSubsampling::Yuv422 => (2, 1),
                    ChromaSubsampling::Yuv420 => (2, 2),
                }),
            },
            chunk_bytes: ChunkBytes::PerTile(300),
            first_chunk_prefix: prefix,
        }
    }

    /// ARW 6.0 Compressed RAW 2: compression 32766, `bits` bits, one strip.
    pub fn arw6(bits: u16) -> Self {
        Self {
            desc: PayloadDescription {
                compression: 32766,
                bits_per_sample: bits,
                photometric: 32803,
                samples_per_pixel: 1,
                tile: None,
                sony_raw_file_type: Some(6),
                ycbcr_subsampling: None,
            },
            chunk_bytes: ChunkBytes::PerPixel {
                numerator: 1,
                denominator: 2,
            },
            first_chunk_prefix: Vec::new(),
        }
    }

    /// Any other compression and bit depth, one strip of one byte per pixel.
    pub fn other(compression: u16, bits: u16) -> Self {
        Self {
            desc: PayloadDescription {
                compression,
                bits_per_sample: bits,
                photometric: 32803,
                samples_per_pixel: 1,
                tile: None,
                sony_raw_file_type: None,
                ycbcr_subsampling: None,
            },
            chunk_bytes: ChunkBytes::PerPixel {
                numerator: 1,
                denominator: 1,
            },
            first_chunk_prefix: Vec::new(),
        }
    }

    /// Pixel-shift ARQ: four samples per pixel, 16-bit words, photometric 32892.
    pub fn arq() -> Self {
        Self {
            desc: PayloadDescription {
                compression: 1,
                bits_per_sample: 16,
                photometric: 32892,
                samples_per_pixel: 4,
                tile: None,
                sony_raw_file_type: None,
                ycbcr_subsampling: None,
            },
            chunk_bytes: ChunkBytes::PerPixel {
                numerator: 8,
                denominator: 1,
            },
            first_chunk_prefix: Vec::new(),
        }
    }
}

impl RawPayload for ContainerOnly {
    fn describe(&self) -> PayloadDescription {
        self.desc.clone()
    }

    fn encode(&self, _mosaic: &[u16], width: u32, height: u32) -> Vec<Vec<u8>> {
        let mut chunks: Vec<Vec<u8>> = match (self.desc.tile, self.chunk_bytes) {
            (Some((tw, th)), ChunkBytes::PerTile(n)) => {
                let count = width.div_ceil(tw) * height.div_ceil(th);
                (0..count).map(|_| vec![0u8; n]).collect()
            }
            (
                None,
                ChunkBytes::PerPixel {
                    numerator,
                    denominator,
                },
            ) => {
                let total = u64::from(width) * u64::from(height) * numerator / denominator;
                vec![vec![0u8; total as usize]]
            }
            (
                Some((tw, th)),
                ChunkBytes::PerPixel {
                    numerator,
                    denominator,
                },
            ) => {
                let count = u64::from(width.div_ceil(tw)) * u64::from(height.div_ceil(th));
                let each =
                    u64::from(width) * u64::from(height) * numerator / denominator / count.max(1);
                (0..count).map(|_| vec![0u8; each as usize]).collect()
            }
            (None, ChunkBytes::PerTile(n)) => vec![vec![0u8; n]],
        };
        if let Some(first) = chunks.first_mut() {
            let n = self.first_chunk_prefix.len().min(first.len());
            first[..n].copy_from_slice(&self.first_chunk_prefix[..n]);
        }
        chunks
    }
}

// ---------------------------------------------------------------------------------------------
// The builder
// ---------------------------------------------------------------------------------------------

/// Which enciphered maker-note table carries `SonyISO`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SonyIsoTable {
    /// Tag 0x9405 (E-mount bodies up to the A7R IV).
    Tag9405,
    /// Tag 0x9416 (A7S III and newer).
    Tag9416,
    /// No enciphered table.
    None,
}

/// How the SR2 block is written.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sr2Variant {
    /// Tags 0x7310 (black), 0x7313 (white balance, RGGB) and 0x787F (white).
    Modern,
    /// Tags 0x7300 (black), 0x7303 (white balance, GRBG) and 0x787F (white).
    Legacy,
    /// No SR2 block and no DNGPrivateData.
    Absent,
}

/// Where the thumbnail's `JPEGInterchangeFormat` tags go.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ThumbnailIfd {
    /// In IFD1 (the usual place).
    Ifd1,
    /// In IFD0 (some writers do this).
    Ifd0,
}

/// Lens-correction tables as stored: a count followed by the knots.
///
/// Tags 0x7032 and 0x7037 lead with `n`, the number of knots (16). Tag 0x7035 leads with `2n`
/// (32): `n` knots for red, then `n` for blue (`ym_gpu::cpu::lens::LensTags::from_stored` reads it
/// that way).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LensTables {
    /// Tag 0x7032: the count `n` and `n` knots (17 values by default).
    pub vignetting: Option<Vec<i16>>,
    /// Tag 0x7035: the count `2n` and `2n` knots (33 values by default).
    pub chromatic_aberration: Option<Vec<i16>>,
    /// Tag 0x7037: the count `n` and `n` knots (17 values by default).
    pub distortion: Option<Vec<i16>>,
}

impl Default for LensTables {
    fn default() -> Self {
        let ramp = |n: usize, start: i16, step: i16| -> Vec<i16> {
            let mut v = vec![16];
            v.extend((0..n).map(|i| start + step * i as i16));
            v
        };
        Self {
            vignetting: Some(ramp(16, -120, 9)),
            // The count is 2n (32): n knots for red, then n for blue, as the ILCE-9 dump stores them.
            chromatic_aberration: Some({
                let mut v = vec![32];
                v.extend((0..32).map(|i| (i as i16 - 16) * 3));
                v
            }),
            distortion: Some(ramp(16, 5, -11)),
        }
    }
}

/// GPS contents for the optional GPS directory.
#[derive(Clone, Debug, PartialEq)]
pub struct GpsSpec {
    /// Latitude in degrees (negative south).
    pub latitude_deg: f64,
    /// Longitude in degrees (negative west).
    pub longitude_deg: f64,
    /// Altitude in metres (negative below sea level).
    pub altitude_m: f64,
}

/// The sample model IDs of the bodies in `docs/03-camera-profiles.md`.
pub fn model_id_for(model: &str) -> Option<u16> {
    Some(match model {
        "DSC-RX100M4" => 341,
        "ILCE-7RM2" => 347,
        "ILCE-7SM2" => 350,
        "ILCE-9" => 358,
        "ILCE-7RM3" => 362,
        "ILCE-7M3" => 363,
        "ILCE-6400" => 371,
        "ILCE-7RM4" => 375,
        "ILCE-9M2" => 376,
        "ILCE-6600" => 378,
        "ILCE-6100" => 379,
        "ILCE-7C" => 381,
        "ILCE-7SM3" => 383,
        "ILCE-1" => 384,
        "ILME-FX3" => 385,
        "ILCE-7RM3A" => 386,
        "ILCE-7RM4A" => 387,
        "ILCE-7M4" => 388,
        "ILCE-7RM5" => 390,
        "ILCE-9M3" => 392,
        "ZV-E1" => 393,
        "ILCE-7CR" => 396,
        "ILCE-7CM2" => 397,
        "ILCE-1M2" => 400,
        "ILCE-6400A" => 402,
        "ILCE-6100A" => 403,
        "ILCE-7M5" => 407,
        "ILCE-7RM6" => 410,
        _ => return None,
    })
}

/// What the maker note's `PreviewImage` (tag 0x2001) holds.
#[derive(Clone, Debug, PartialEq, Eq)]
enum MakerPreview {
    /// No tag.
    Omitted,
    /// A gradient from [`tiny_jpeg`] of this width and height.
    Generated(u16, u16),
    /// This JPEG, byte for byte.
    Custom(Vec<u8>),
}

/// Builds a synthetic ARW file the way a Sony body lays it out. See the module documentation.
///
/// Every setter returns `self`. Only the fields set explicitly differ from the defaults, which
/// describe an ILCE-7RM4 style file: 14-bit uncompressed, black 512, white 16300, SR2 block,
/// lens tables, thumbnail and 0x2001 preview, ISO 400 at 1/250 s f/4.
#[derive(Clone)]
pub struct ArwBuilder {
    make: String,
    model: String,
    model_id: Option<u16>,
    software: String,
    width: u32,
    height: u32,
    orientation: u16,
    date_time: String,
    subsec: String,
    utc_offset: Option<String>,
    iso: u32,
    exposure_time: (u32, u32),
    f_number: (u32, u32),
    exposure_comp: (i32, i32),
    focal_length: (u32, u32),
    lens_model: Option<String>,
    black: [u16; 4],
    white: [u16; 3],
    wb: Option<[u16; 4]>,
    sr2: Sr2Variant,
    sr2_key: u32,
    raw_ifd_levels: bool,
    cfa: CfaPattern,
    tone_curve: Option<[u16; 4]>,
    lens: LensTables,
    active_area: Option<(u32, u32)>,
    crop: Option<Rect>,
    thumbnail: Option<(u16, u16)>,
    thumbnail_ifd: ThumbnailIfd,
    preview: MakerPreview,
    preview_min_bytes: usize,
    iso_table: Option<SonyIsoTable>,
    sony_iso_word: Option<u16>,
    raw_ifd_white_level: Option<u16>,
    lossless_thumbnail: bool,
    file_format: [u8; 4],
    maker_note_header: Vec<u8>,
    exif_iso_ratings: bool,
    gps: Option<GpsSpec>,
    payload: Option<std::sync::Arc<dyn RawPayload>>,
    sony_raw_file_type: Option<u16>,
    omit_bits_per_sample: bool,
}

impl ArwBuilder {
    /// A builder for a `width` x `height` file from the camera `model` (for example
    /// `"ILCE-7RM4"`). Width and height must be even.
    pub fn new(model: &str, width: u32, height: u32) -> Self {
        Self {
            make: "SONY".to_owned(),
            model: model.to_owned(),
            model_id: model_id_for(model),
            software: format!("{model} v1.00"),
            width,
            height,
            orientation: 1,
            date_time: "2026:10:04 12:03:44".to_owned(),
            subsec: "07".to_owned(),
            utc_offset: Some("+02:00".to_owned()),
            iso: 400,
            exposure_time: (1, 250),
            f_number: (40, 10),
            exposure_comp: (-7, 10),
            focal_length: (350, 10),
            lens_model: Some("FE 24-70mm F2.8 GM".to_owned()),
            black: [512; 4],
            white: [16300; 3],
            wb: Some([2043, 1024, 1024, 1755]),
            sr2: Sr2Variant::Modern,
            sr2_key: 0x0D3F_91A7,
            raw_ifd_levels: false,
            cfa: CfaPattern::RGGB,
            tone_curve: None,
            lens: LensTables::default(),
            active_area: None,
            crop: None,
            thumbnail: Some((160, 120)),
            thumbnail_ifd: ThumbnailIfd::Ifd1,
            preview: MakerPreview::Generated(320, 214),
            preview_min_bytes: 0,
            iso_table: None,
            sony_iso_word: None,
            raw_ifd_white_level: None,
            lossless_thumbnail: false,
            file_format: [3, 2, 0, 0],
            // Modern ARW files carry no header in front of the maker-note directory (ExifTool's
            // `MakerNoteSony5`); `maker_note_header` writes the older `SONY DSC` layout.
            maker_note_header: Vec::new(),
            exif_iso_ratings: true,
            gps: None,
            payload: None,
            sony_raw_file_type: None,
            omit_bits_per_sample: false,
        }
    }

    /// Leaves the `BitsPerSample` tag out of the raw directory (for detection tests).
    pub fn omit_bits_per_sample(mut self, yes: bool) -> Self {
        self.omit_bits_per_sample = yes;
        self
    }

    /// Sets the `Make` string.
    pub fn make(mut self, make: &str) -> Self {
        self.make = make.to_owned();
        self
    }

    /// Sets the Sony model ID written to maker note tag 0xB001.
    pub fn model_id(mut self, id: u16) -> Self {
        self.model_id = Some(id);
        self
    }

    /// Leaves the model ID out of the maker note.
    pub fn without_model_id(mut self) -> Self {
        self.model_id = None;
        self
    }

    /// Sets the `Software` string.
    pub fn software(mut self, software: &str) -> Self {
        self.software = software.to_owned();
        self
    }

    /// Sets the EXIF orientation (1 to 8).
    pub fn orientation(mut self, orientation: u16) -> Self {
        self.orientation = orientation;
        self
    }

    /// Sets `DateTimeOriginal` (`YYYY:MM:DD HH:MM:SS`), its sub-second digits and the UTC offset
    /// string (`None` omits the `OffsetTime*` tags).
    pub fn capture_time(mut self, date_time: &str, subsec: &str, utc_offset: Option<&str>) -> Self {
        self.date_time = date_time.to_owned();
        self.subsec = subsec.to_owned();
        self.utc_offset = utc_offset.map(str::to_owned);
        self
    }

    /// Sets the ISO. The EXIF `ISOSpeedRatings` (16 bit) saturates at 65535; the recommended
    /// exposure index and the maker-note table hold the exact value.
    pub fn iso(mut self, iso: u32) -> Self {
        self.iso = iso;
        self
    }

    /// Sets the exposure time as a fraction of a second.
    pub fn exposure_time(mut self, numerator: u32, denominator: u32) -> Self {
        self.exposure_time = (numerator, denominator);
        self
    }

    /// Sets the F-number as a fraction (`40, 10` for f/4).
    pub fn f_number(mut self, numerator: u32, denominator: u32) -> Self {
        self.f_number = (numerator, denominator);
        self
    }

    /// Sets the exposure compensation as a signed fraction of an EV.
    pub fn exposure_comp(mut self, numerator: i32, denominator: i32) -> Self {
        self.exposure_comp = (numerator, denominator);
        self
    }

    /// Sets the focal length as a fraction of a millimetre.
    pub fn focal_length(mut self, numerator: u32, denominator: u32) -> Self {
        self.focal_length = (numerator, denominator);
        self
    }

    /// Sets the lens model (`None` leaves the tag out).
    pub fn lens_model(mut self, lens: Option<&str>) -> Self {
        self.lens_model = lens.map(str::to_owned);
        self
    }

    /// Sets the black level per colour (R, G1, G2, B) as written to the SR2 block.
    pub fn black(mut self, black: [u16; 4]) -> Self {
        self.black = black;
        self
    }

    /// Sets the white level per colour (R, G, B) as written to the SR2 block.
    pub fn white(mut self, white: [u16; 3]) -> Self {
        self.white = white;
        self
    }

    /// Sets the as-shot white balance (R, G1, G2, B), or `None` to leave it out.
    pub fn wb(mut self, wb: Option<[u16; 4]>) -> Self {
        self.wb = wb;
        self
    }

    /// Chooses how the SR2 block is written.
    pub fn sr2(mut self, variant: Sr2Variant) -> Self {
        self.sr2 = variant;
        self
    }

    /// Sets the SR2 cipher key.
    pub fn sr2_key(mut self, key: u32) -> Self {
        self.sr2_key = key;
        self
    }

    /// Also writes black level (0x7310) and white balance (0x7313) as plain tags in the raw IFD,
    /// as newer bodies do.
    pub fn raw_ifd_levels(mut self, yes: bool) -> Self {
        self.raw_ifd_levels = yes;
        self
    }

    /// Sets the colour filter array pattern written to the raw IFD.
    pub fn cfa(mut self, cfa: CfaPattern) -> Self {
        self.cfa = cfa;
        self
    }

    /// Writes the cRAW tone-curve knots (tag 0x7010, four 16-bit values).
    pub fn tone_curve(mut self, knots: Option<[u16; 4]>) -> Self {
        self.tone_curve = knots;
        self
    }

    /// Replaces the lens-correction tables.
    pub fn lens_tables(mut self, tables: LensTables) -> Self {
        self.lens = tables;
        self
    }

    /// Writes `SonyRawImageSize` (tag 0x7038): the valid image area inside the readout.
    pub fn active_area(mut self, width: u32, height: u32) -> Self {
        self.active_area = Some((width, height));
        self
    }

    /// Writes `SonyCropTopLeft` and `SonyCropSize` (tags 0x74C7, 0x74C8).
    pub fn crop(mut self, crop: Rect) -> Self {
        self.crop = Some(crop);
        self
    }

    /// Sets the size of the thumbnail JPEG (`None` omits it).
    pub fn thumbnail(mut self, size: Option<(u16, u16)>) -> Self {
        self.thumbnail = size;
        self
    }

    /// Puts the thumbnail tags in IFD0 or IFD1.
    pub fn thumbnail_ifd(mut self, ifd: ThumbnailIfd) -> Self {
        self.thumbnail_ifd = ifd;
        self
    }

    /// Sets the size of the larger JPEG in maker note tag 0x2001 (`None` omits it). The picture is
    /// a gradient from [`tiny_jpeg`]. The last call of `preview` and [`ArwBuilder::preview_jpeg`]
    /// wins.
    pub fn preview(mut self, size: Option<(u16, u16)>) -> Self {
        self.preview = match size {
            Some((width, height)) => MakerPreview::Generated(width, height),
            None => MakerPreview::Omitted,
        };
        self
    }

    /// Puts this JPEG in maker note tag 0x2001 instead of a generated gradient, byte for byte
    /// (after [`ArwBuilder::preview_min_bytes`] padding, which only applies when the JPEG starts
    /// `FF D8 FF DB` as Sony previews do and is shorter than the minimum). The 32-byte header in
    /// front of it gets the JPEG's own length, width and height. The last call of
    /// [`ArwBuilder::preview`] and `preview_jpeg` wins.
    pub fn preview_jpeg(mut self, jpeg: Vec<u8>) -> Self {
        self.preview = MakerPreview::Custom(jpeg);
        self
    }

    /// Pads the maker-note preview with JPEG comment segments to at least `bytes` bytes (real
    /// previews are a few hundred kilobytes).
    pub fn preview_min_bytes(mut self, bytes: usize) -> Self {
        self.preview_min_bytes = bytes;
        self
    }

    /// Chooses the enciphered maker-note table that holds `SonyISO` (the default depends on the
    /// model).
    pub fn iso_table(mut self, table: SonyIsoTable) -> Self {
        self.iso_table = Some(table);
        self
    }

    /// Writes this word as `SonyISO` in the enciphered table instead of the one that matches
    /// [`ArwBuilder::iso`]. Real cameras store the sensitivity they measured, which differs a
    /// little from the nominal ISO in the EXIF tags (the ILCE-9 in the lclevy/sony_raw dump has
    /// word 2890, ISO 2619, in a file whose EXIF ISO is 2500).
    pub fn sony_iso_word(mut self, word: u16) -> Self {
        self.sony_iso_word = Some(word);
        self
    }

    /// Writes the DNG `WhiteLevel` (tag 0xC61D) into the raw IFD, as many Sony bodies do (the
    /// ILCE-9 writes 16383 there next to an SR2 white level of 15360).
    pub fn raw_ifd_white_level(mut self, white: u16) -> Self {
        self.raw_ifd_white_level = Some(white);
        self
    }

    /// Marks the thumbnail JPEG as coded with a lossless process (its frame header becomes SOF3),
    /// like the "RGB preview" Sony keeps next to Lossless M and S data. The stream is not
    /// decodable; only the marker is real.
    pub fn lossless_thumbnail(mut self, yes: bool) -> Self {
        self.lossless_thumbnail = yes;
        self
    }

    /// Sets the ARW version bytes of maker note tag 0xB000.
    pub fn file_format(mut self, bytes: [u8; 4]) -> Self {
        self.file_format = bytes;
        self
    }

    /// Sets the bytes in front of the maker-note directory. The default is none, as in modern ARW
    /// files; `b"SONY DSC \0\0\0"` gives the layout of older bodies and JPEG files.
    pub fn maker_note_header(mut self, header: &[u8]) -> Self {
        self.maker_note_header = header.to_vec();
        self
    }

    /// Whether to write EXIF `ISOSpeedRatings` (on by default).
    pub fn exif_iso_ratings(mut self, yes: bool) -> Self {
        self.exif_iso_ratings = yes;
        self
    }

    /// Adds a GPS directory.
    pub fn gps(mut self, gps: GpsSpec) -> Self {
        self.gps = Some(gps);
        self
    }

    /// Overrides `SonyRawFileType` (tag 0x7000); `None` leaves the payload's value.
    pub fn sony_raw_file_type(mut self, value: Option<u16>) -> Self {
        self.sony_raw_file_type = value;
        self
    }

    /// Chooses how the raw data is encoded. The default is [`Uncompressed`] with 14 bits.
    pub fn payload(mut self, payload: impl RawPayload + 'static) -> Self {
        self.payload = Some(std::sync::Arc::new(payload));
        self
    }

    /// Shorthand for [`ArwBuilder::payload`] with [`Uncompressed`] of `bits` bits.
    pub fn uncompressed(self, bits: u16) -> Self {
        self.payload(Uncompressed { bits })
    }

    /// Writes the container for a compressed variant with filler data (see [`ContainerOnly`]).
    pub fn container_only(self, payload: ContainerOnly) -> Self {
        self.payload(payload)
    }

    /// Writes a cRAW file: the encoder's payload plus its tone curve in tag 0x7010.
    pub fn craw(self, encoder: CrawEncoder) -> Self {
        self.tone_curve(Some(encoder.knots)).payload(encoder)
    }

    /// Writes a Lossless M or S file: the payload plus the SR2 white balance that matches the
    /// gains it applied (the `mosaic` given to [`ArwBuilder::build`] is ignored).
    pub fn lossless_ms(self, payload: LosslessMs) -> Self {
        let wb = payload.sr2_wb();
        self.wb(Some(wb)).payload(payload)
    }

    /// Builds the file. `mosaic` holds `width * height` sensor values; it is ignored by payloads
    /// that write filler.
    ///
    /// # Panics
    ///
    /// Panics if `mosaic` is shorter than `width * height` for a payload that needs it. This is
    /// test-support code.
    pub fn build(&self, mosaic: &[u16]) -> Vec<u8> {
        let payload: std::sync::Arc<dyn RawPayload> = self
            .payload
            .clone()
            .unwrap_or_else(|| std::sync::Arc::new(Uncompressed { bits: 14 }));
        let desc = payload.describe();
        let chunks = payload.encode(mosaic, self.width, self.height);
        assemble(self, &desc, &chunks)
    }

    /// Builds the file and writes it to `path`.
    pub fn write_to(&self, path: &std::path::Path, mosaic: &[u16]) -> std::io::Result<()> {
        std::fs::write(path, self.build(mosaic))
    }

    /// The default ISO table for this model: 0x9416 for the A7S III and newer, 0x9405 before.
    fn resolved_iso_table(&self) -> SonyIsoTable {
        self.iso_table.unwrap_or(match self.model.as_str() {
            "ILCE-1" | "ILCE-1M2" | "ILCE-7M4" | "ILCE-7SM3" | "ILCE-7CM2" | "ILCE-7RM5"
            | "ILCE-7CR" | "ILCE-9M3" | "ILCE-7M5" | "ILCE-7RM6" | "ILME-FX3" | "ZV-E1" => {
                SonyIsoTable::Tag9416
            }
            _ => SonyIsoTable::Tag9405,
        })
    }
}

// ---------------------------------------------------------------------------------------------
// TIFF assembly
// ---------------------------------------------------------------------------------------------

const BYTE: u16 = 1;
const ASCII: u16 = 2;
const SHORT: u16 = 3;
const LONG: u16 = 4;
const RATIONAL: u16 = 5;
const UNDEFINED: u16 = 7;
const SSHORT: u16 = 8;
const SRATIONAL: u16 = 10;

/// One directory entry being written: tag, type, count and the little-endian value bytes.
#[derive(Clone, Debug)]
struct Ent {
    tag: u16,
    ty: u16,
    count: u32,
    data: Vec<u8>,
    /// The value lives in another block at this absolute offset (`data` is then empty).
    external: Option<u32>,
}

impl Ent {
    fn bytes(tag: u16, ty: u16, data: Vec<u8>, count: u32) -> Self {
        Self {
            tag,
            ty,
            count,
            data,
            external: None,
        }
    }

    /// An entry whose value of `count` items of type `ty` is stored elsewhere, at `offset`.
    fn external(tag: u16, ty: u16, count: u32, offset: u32) -> Self {
        Self {
            tag,
            ty,
            count,
            data: Vec::new(),
            external: Some(offset),
        }
    }

    fn byte(tag: u16, v: &[u8]) -> Self {
        Self::bytes(tag, BYTE, v.to_vec(), v.len() as u32)
    }

    fn undefined(tag: u16, v: &[u8]) -> Self {
        Self::bytes(tag, UNDEFINED, v.to_vec(), v.len() as u32)
    }

    fn ascii(tag: u16, text: &str) -> Self {
        let mut data = text.as_bytes().to_vec();
        data.push(0);
        let count = data.len() as u32;
        Self::bytes(tag, ASCII, data, count)
    }

    fn short(tag: u16, v: &[u16]) -> Self {
        Self::bytes(
            tag,
            SHORT,
            v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            v.len() as u32,
        )
    }

    fn sshort(tag: u16, v: &[i16]) -> Self {
        Self::bytes(
            tag,
            SSHORT,
            v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            v.len() as u32,
        )
    }

    fn long(tag: u16, v: &[u32]) -> Self {
        Self::bytes(
            tag,
            LONG,
            v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            v.len() as u32,
        )
    }

    fn rational(tag: u16, v: &[(u32, u32)]) -> Self {
        Self::bytes(
            tag,
            RATIONAL,
            v.iter()
                .flat_map(|(n, d)| n.to_le_bytes().into_iter().chain(d.to_le_bytes()))
                .collect(),
            v.len() as u32,
        )
    }

    fn srational(tag: u16, v: &[(i32, i32)]) -> Self {
        Self::bytes(
            tag,
            SRATIONAL,
            v.iter()
                .flat_map(|(n, d)| n.to_le_bytes().into_iter().chain(d.to_le_bytes()))
                .collect(),
            v.len() as u32,
        )
    }
}

/// Serialises a directory placed at file position `at`: the entry table (sorted by tag), the
/// next-IFD offset, then every value that does not fit in four bytes.
fn serialize_ifd(at: usize, entries: &[Ent], next: u32) -> Vec<u8> {
    let mut sorted = entries.to_vec();
    sorted.sort_by_key(|e| e.tag);
    let table_len = 2 + sorted.len() * 12 + 4;
    let mut out = Vec::new();
    out.extend_from_slice(&(sorted.len() as u16).to_le_bytes());
    let mut tail: Vec<u8> = Vec::new();
    for e in &sorted {
        out.extend_from_slice(&e.tag.to_le_bytes());
        out.extend_from_slice(&e.ty.to_le_bytes());
        out.extend_from_slice(&e.count.to_le_bytes());
        if let Some(offset) = e.external {
            out.extend_from_slice(&offset.to_le_bytes());
        } else if e.data.len() <= 4 {
            out.extend_from_slice(&e.data);
            out.extend(std::iter::repeat_n(0u8, 4 - e.data.len()));
        } else {
            let offset = (at + table_len + tail.len()) as u32;
            out.extend_from_slice(&offset.to_le_bytes());
            tail.extend_from_slice(&e.data);
            if tail.len() % 2 == 1 {
                tail.push(0);
            }
        }
    }
    out.extend_from_slice(&next.to_le_bytes());
    out.extend_from_slice(&tail);
    out
}

fn align4(v: &mut Vec<u8>) {
    while !v.len().is_multiple_of(4) {
        v.push(0);
    }
}

/// File positions of every block, filled in by a first measuring pass.
#[derive(Clone, Copy, Debug, Default)]
struct Plan {
    ifd0: usize,
    ifd1: usize,
    exif: usize,
    gps: usize,
    maker: usize,
    sr2_private: usize,
    raw_ifd: usize,
    thumbnail: usize,
    sr2_block: usize,
    payload: usize,
}

/// The blocks of a file, each already serialised for the positions in a [`Plan`].
struct Blocks {
    ifd0: Vec<u8>,
    ifd1: Vec<u8>,
    exif: Vec<u8>,
    gps: Vec<u8>,
    maker: Vec<u8>,
    sr2_private: Vec<u8>,
    raw_ifd: Vec<u8>,
    thumbnail: Vec<u8>,
    sr2_block: Vec<u8>,
}

fn assemble(b: &ArwBuilder, desc: &PayloadDescription, chunks: &[Vec<u8>]) -> Vec<u8> {
    let payload_lens: Vec<usize> = chunks.iter().map(Vec::len).collect();
    // Pass 1 measures the blocks with dummy offsets; their sizes do not depend on the offsets.
    let measured = blocks(b, desc, &payload_lens, &Plan::default());
    let mut plan = Plan::default();
    let mut pos = 8usize;
    let mut place = |len: usize| -> usize {
        let at = pos;
        pos += len.next_multiple_of(4);
        at
    };
    plan.ifd0 = place(measured.ifd0.len());
    plan.ifd1 = place(measured.ifd1.len());
    plan.exif = place(measured.exif.len());
    plan.gps = place(measured.gps.len());
    plan.maker = place(measured.maker.len());
    plan.sr2_private = place(measured.sr2_private.len());
    plan.raw_ifd = place(measured.raw_ifd.len());
    plan.thumbnail = place(measured.thumbnail.len());
    plan.sr2_block = place(measured.sr2_block.len());
    plan.payload = pos;
    // Pass 2 writes the real offsets.
    let real = blocks(b, desc, &payload_lens, &plan);
    if std::env::var_os("YM_DEBUG_PLAN").is_some() {
        eprintln!("plan {plan:?}");
        for (name, m, r) in [
            ("ifd0", &measured.ifd0, &real.ifd0),
            ("ifd1", &measured.ifd1, &real.ifd1),
            ("exif", &measured.exif, &real.exif),
            ("gps", &measured.gps, &real.gps),
            ("maker", &measured.maker, &real.maker),
            ("sr2p", &measured.sr2_private, &real.sr2_private),
            ("raw", &measured.raw_ifd, &real.raw_ifd),
            ("thumb", &measured.thumbnail, &real.thumbnail),
            ("sr2b", &measured.sr2_block, &real.sr2_block),
        ] {
            eprintln!("{name}: measured {} real {}", m.len(), r.len());
        }
    }

    let mut file = b"II*\0".to_vec();
    file.extend_from_slice(&8u32.to_le_bytes());
    for (at, mut block) in [
        (plan.ifd0, real.ifd0),
        (plan.ifd1, real.ifd1),
        (plan.exif, real.exif),
        (plan.gps, real.gps),
        (plan.maker, real.maker),
        (plan.sr2_private, real.sr2_private),
        (plan.raw_ifd, real.raw_ifd),
        (plan.thumbnail, real.thumbnail),
        (plan.sr2_block, real.sr2_block),
    ] {
        debug_assert_eq!(file.len(), at, "block placed where the plan said");
        align4(&mut block);
        file.extend_from_slice(&block);
    }
    debug_assert_eq!(file.len(), plan.payload);
    for chunk in chunks {
        file.extend_from_slice(chunk);
        // Keep every chunk 2-byte aligned; the byte counts in the IFD stay exact.
        if file.len() % 2 == 1 {
            file.push(0);
        }
    }
    file
}

/// The payload chunk positions for a plan: each chunk starts at an even offset.
fn chunk_positions(plan: &Plan, lens: &[usize]) -> Vec<u32> {
    let mut at = plan.payload;
    lens.iter()
        .map(|len| {
            let here = at as u32;
            at += len + (len % 2);
            here
        })
        .collect()
}

fn blocks(
    b: &ArwBuilder,
    desc: &PayloadDescription,
    payload_lens: &[usize],
    plan: &Plan,
) -> Blocks {
    let has_sr2 = b.sr2 != Sr2Variant::Absent;
    let mut thumbnail_jpeg = b
        .thumbnail
        .map(|(w, h)| tiny_jpeg(w, h, 1))
        .unwrap_or_default();
    if b.lossless_thumbnail
        && let Some(at) = thumbnail_jpeg.windows(2).position(|w| w == [0xFF, 0xC0])
    {
        thumbnail_jpeg[at + 1] = 0xC3;
    }
    let have_thumb = b.thumbnail.is_some();

    // --- IFD0 ---------------------------------------------------------------------------------
    let mut ifd0 = vec![
        Ent::ascii(0x010F, &b.make),
        Ent::ascii(0x0110, &b.model),
        Ent::short(0x0112, &[b.orientation]),
        Ent::rational(0x011A, &[(72, 1)]),
        Ent::rational(0x011B, &[(72, 1)]),
        Ent::short(0x0128, &[2]),
        Ent::ascii(0x0131, &b.software),
        Ent::ascii(0x0132, &b.date_time),
        Ent::long(0x014A, &[plan.raw_ifd as u32]),
        Ent::short(0x0213, &[2]),
        Ent::long(0x8769, &[plan.exif as u32]),
    ];
    if b.gps.is_some() {
        ifd0.push(Ent::long(0x8825, &[plan.gps as u32]));
    }
    if has_sr2 {
        ifd0.push(Ent::byte(0xC634, &(plan.sr2_private as u32).to_le_bytes()));
    }
    if have_thumb && b.thumbnail_ifd == ThumbnailIfd::Ifd0 {
        ifd0.push(Ent::short(0x0103, &[6]));
        ifd0.push(Ent::long(0x0201, &[plan.thumbnail as u32]));
        ifd0.push(Ent::long(0x0202, &[thumbnail_jpeg.len() as u32]));
    }
    let has_ifd1 = have_thumb && b.thumbnail_ifd == ThumbnailIfd::Ifd1;
    let ifd1_at = if has_ifd1 { plan.ifd1 as u32 } else { 0 };
    let ifd0_bytes = serialize_ifd(plan.ifd0, &ifd0, ifd1_at);

    // --- IFD1 (thumbnail) ---------------------------------------------------------------------
    let ifd1_bytes = if has_ifd1 {
        serialize_ifd(
            plan.ifd1,
            &[
                Ent::short(0x0103, &[6]),
                Ent::rational(0x011A, &[(72, 1)]),
                Ent::rational(0x011B, &[(72, 1)]),
                Ent::short(0x0128, &[2]),
                Ent::long(0x0201, &[plan.thumbnail as u32]),
                Ent::long(0x0202, &[thumbnail_jpeg.len() as u32]),
                Ent::short(0x0213, &[2]),
            ],
            0,
        )
    } else {
        Vec::new()
    };

    // --- EXIF IFD -----------------------------------------------------------------------------
    let maker = maker_note(b, plan);
    let mut exif = vec![
        Ent::rational(0x829A, &[b.exposure_time]),
        Ent::rational(0x829D, &[b.f_number]),
        Ent::short(0x8822, &[2]),
        Ent::short(0x8830, &[2]),
        Ent::long(0x8832, &[b.iso]),
        Ent::undefined(0x9000, b"0231"),
        Ent::ascii(0x9003, &b.date_time),
        Ent::ascii(0x9004, &b.date_time),
        Ent::srational(0x9204, &[b.exposure_comp]),
        Ent::short(0x9207, &[5]),
        Ent::short(0x9209, &[0]),
        Ent::rational(0x920A, &[b.focal_length]),
        Ent::external(0x927C, UNDEFINED, maker.len() as u32, plan.maker as u32),
        Ent::ascii(0x9291, &b.subsec),
        Ent::short(0xA001, &[1]),
        Ent::short(
            0xA405,
            &[(b.focal_length.0 / b.focal_length.1.max(1)) as u16],
        ),
    ];
    if b.exif_iso_ratings {
        exif.push(Ent::short(0x8827, &[b.iso.min(65_535) as u16]));
    }
    if let Some(offset) = &b.utc_offset {
        exif.push(Ent::ascii(0x9010, offset));
        exif.push(Ent::ascii(0x9011, offset));
        exif.push(Ent::ascii(0x9012, offset));
    }
    if let Some(lens) = &b.lens_model {
        exif.push(Ent::ascii(0xA433, "Sony"));
        exif.push(Ent::ascii(0xA434, lens));
        exif.push(Ent::rational(
            0xA432,
            &[(240, 10), (700, 10), (28, 10), (28, 10)],
        ));
    }
    let exif_bytes = serialize_ifd(plan.exif, &exif, 0);

    // --- GPS ----------------------------------------------------------------------------------
    let gps_bytes = match &b.gps {
        Some(gps) => {
            let dms = |deg: f64| -> Vec<(u32, u32)> {
                let a = deg.abs();
                let d = a.floor();
                let m = ((a - d) * 60.0).floor();
                let s = ((a - d) * 60.0 - m) * 60.0;
                vec![
                    (d as u32, 1),
                    (m as u32, 1),
                    ((s * 1000.0).round() as u32, 1000),
                ]
            };
            serialize_ifd(
                plan.gps,
                &[
                    Ent::byte(0x0000, &[2, 3, 0, 0]),
                    Ent::ascii(0x0001, if gps.latitude_deg < 0.0 { "S" } else { "N" }),
                    Ent::rational(0x0002, &dms(gps.latitude_deg)),
                    Ent::ascii(0x0003, if gps.longitude_deg < 0.0 { "W" } else { "E" }),
                    Ent::rational(0x0004, &dms(gps.longitude_deg)),
                    Ent::byte(0x0005, &[u8::from(gps.altitude_m < 0.0)]),
                    Ent::rational(
                        0x0006,
                        &[((gps.altitude_m.abs() * 100.0).round() as u32, 100)],
                    ),
                    Ent::rational(0x0007, &[(10, 1), (20, 1), (30, 1)]),
                    Ent::ascii(0x001D, "2026:10:04"),
                ],
                0,
            )
        }
        None => Vec::new(),
    };

    // --- SR2Private and the encrypted block ---------------------------------------------------
    let (sr2_private, sr2_block) = if has_sr2 {
        let plain = sr2_plain(b, plan.sr2_block);
        let mut padded = plain;
        while !padded.len().is_multiple_of(4) {
            padded.push(0);
        }
        let length = padded.len() as u32;
        sr2::apply_pad(&mut padded, b.sr2_key);
        let private = serialize_ifd(
            plan.sr2_private,
            &[
                Ent::long(0x7200, &[plan.sr2_block as u32]),
                Ent::long(0x7201, &[length]),
                Ent::undefined(0x7221, &b.sr2_key.to_le_bytes()),
            ],
            0,
        );
        (private, padded)
    } else {
        (Vec::new(), Vec::new())
    };

    // --- raw IFD ------------------------------------------------------------------------------
    let positions = chunk_positions(plan, payload_lens);
    let counts: Vec<u32> = payload_lens.iter().map(|l| *l as u32).collect();
    let mut raw = vec![
        Ent::long(0x00FE, &[0]),
        Ent::long(0x0100, &[b.width]),
        Ent::long(0x0101, &[b.height]),
        Ent::short(0x0103, &[desc.compression]),
        Ent::short(0x0106, &[desc.photometric]),
        Ent::short(0x0115, &[desc.samples_per_pixel]),
        Ent::short(0x828D, &[2, 2]),
        Ent::byte(0x828E, &b.cfa.tiff_codes()),
    ];
    if !b.omit_bits_per_sample {
        raw.push(Ent::short(0x0102, &[desc.bits_per_sample]));
    }
    match desc.tile {
        Some((tw, th)) => {
            // Real Sony files with tiles (the ILCE-7M4 file in metadata-extractor-images) carry
            // the strip tags as well, describing the whole tile area as one strip of the full
            // image height; rawler looks for `StripOffsets` to find the raw directory.
            let first = positions.first().copied().unwrap_or(0);
            let end = positions
                .iter()
                .zip(&counts)
                .map(|(p, c)| p + c)
                .max()
                .unwrap_or(first);
            raw.push(Ent::long(0x0111, &[first]));
            raw.push(Ent::long(0x0116, &[b.height]));
            raw.push(Ent::long(0x0117, &[end - first]));
            raw.push(Ent::long(0x0142, &[tw]));
            raw.push(Ent::long(0x0143, &[th]));
            raw.push(Ent::long(0x0144, &positions));
            raw.push(Ent::long(0x0145, &counts));
        }
        None => {
            raw.push(Ent::long(0x0111, &positions));
            raw.push(Ent::long(0x0116, &[b.height]));
            raw.push(Ent::long(0x0117, &counts));
        }
    }
    if let Some((h, v)) = desc.ycbcr_subsampling {
        raw.push(Ent::short(0x0212, &[h, v]));
    }
    if let Some(t) = b.sony_raw_file_type.or(desc.sony_raw_file_type) {
        raw.push(Ent::short(0x7000, &[t]));
    }
    if let Some(curve) = b.tone_curve {
        raw.push(Ent::short(0x7010, &curve));
    }
    if let Some(v) = &b.lens.vignetting {
        raw.push(Ent::sshort(0x7032, v));
    }
    if let Some(v) = &b.lens.chromatic_aberration {
        raw.push(Ent::sshort(0x7035, v));
    }
    if let Some(v) = &b.lens.distortion {
        raw.push(Ent::sshort(0x7037, v));
    }
    if let Some((w, h)) = b.active_area {
        raw.push(Ent::long(0x7038, &[w, h]));
    }
    if b.raw_ifd_levels {
        raw.push(Ent::short(0x7310, &b.black));
        if let Some(wb) = b.wb {
            raw.push(Ent::sshort(0x7313, &wb.map(|v| v as i16)));
        }
    }
    if let Some(white) = b.raw_ifd_white_level {
        raw.push(Ent::short(0xC61D, &[white]));
    }
    if let Some(crop) = b.crop {
        raw.push(Ent::long(0x74C7, &[crop.x, crop.y]));
        raw.push(Ent::long(0x74C8, &[crop.width, crop.height]));
    }
    let raw_bytes = serialize_ifd(plan.raw_ifd, &raw, 0);

    Blocks {
        ifd0: ifd0_bytes,
        ifd1: ifd1_bytes,
        exif: exif_bytes,
        gps: gps_bytes,
        maker,
        sr2_private,
        raw_ifd: raw_bytes,
        thumbnail: thumbnail_jpeg,
        sr2_block,
    }
}

/// The plain (decrypted) SR2 sub-IFD for a block placed at `at`; offsets are absolute.
fn sr2_plain(b: &ArwBuilder, at: usize) -> Vec<u8> {
    let mut entries = vec![Ent::short(0x787F, &b.white)];
    match b.sr2 {
        Sr2Variant::Modern => {
            entries.push(Ent::short(0x7310, &b.black));
            if let Some(wb) = b.wb {
                // Real files store this tag as signed 16-bit values.
                entries.push(Ent::sshort(0x7313, &wb.map(|v| v as i16)));
            }
        }
        Sr2Variant::Legacy => {
            entries.push(Ent::short(0x7300, &b.black));
            if let Some(wb) = b.wb {
                // Stored G, R, B, G.
                entries.push(Ent::short(0x7303, &[wb[1], wb[0], wb[3], wb[2]]));
            }
        }
        Sr2Variant::Absent => {}
    }
    let mut block = serialize_ifd(at, &entries, 0);
    // Real blocks carry many more tags after these; LibRaw's bounds check rejects a value that
    // ends exactly at the end of the block, so leave some room.
    block.extend_from_slice(&[0u8; 96]);
    block
}

/// The word Sony stores for an exposure time: `round(256 * (16 - log2(seconds)))`, so that
/// `2^(16 - word / 256)` is the time (ExifTool's `SonyExposureTime2`). A zero time (bulb) is 0.
fn sony_exposure_word(numerator: u32, denominator: u32) -> u16 {
    if numerator == 0 {
        return 0;
    }
    let seconds = f64::from(numerator) / f64::from(denominator.max(1));
    (256.0 * (16.0 - seconds.log2()))
        .round()
        .clamp(0.0, f64::from(u16::MAX)) as u16
}

/// The word Sony stores for an F-number: `round(256 * (16 + 2 * log2(f)))` (ExifTool's
/// `SonyFNumber`: `2^((word / 256 - 16) / 2)`).
fn sony_f_number_word(numerator: u32, denominator: u32) -> u16 {
    let f = f64::from(numerator.max(1)) / f64::from(denominator.max(1));
    (256.0 * (16.0 + 2.0 * f.log2()))
        .round()
        .clamp(0.0, f64::from(u16::MAX)) as u16
}

/// A fraction squeezed into two 16-bit words (ExifTool's `rational32u`): reduced, and scaled down
/// together when either part is above 65535.
fn rational_u16(numerator: u32, denominator: u32) -> (u16, u16) {
    fn gcd(a: u32, b: u32) -> u32 {
        if b == 0 { a } else { gcd(b, a % b) }
    }
    // A zero denominator is read as 1, as in `sony_exposure_word`.
    let denominator = denominator.max(1);
    let g = gcd(numerator, denominator).max(1);
    let (mut n, mut d) = (numerator / g, denominator / g);
    let largest = n.max(d);
    if largest > u32::from(u16::MAX) {
        let scale = f64::from(u16::MAX) / f64::from(largest);
        n = (f64::from(n) * scale).round() as u32;
        d = ((f64::from(d) * scale).round() as u32).max(1);
    }
    (n as u16, d as u16)
}

/// The enciphered table 0x9405 (`older` is true) or 0x9416, as stored: the plain bytes at the
/// offsets ExifTool's `Tag9405b` and `Tag9416` tables read, every byte put through Sony's cipher
/// ([`encipher_byte`]); the reader undoes it with [`crate::makernote::decipher_byte`].
///
/// | Field | `Tag9405b` | `Tag9416` | Value |
/// |---|---|---|---|
/// | `SonyISO` | 0x04 | 0x04 | `u16`, [`sony_word_for_iso`] of the builder's ISO (or its [`ArwBuilder::sony_iso_word`]) |
/// | `SonyExposureTime2` | 0x0e | 0x0a | `u16`, [`sony_exposure_word`] |
/// | `ExposureTime` | 0x10 | 0x0c | two `u16`, numerator and denominator |
/// | `SonyFNumber` | 0x14 | 0x10 | `u16`, [`sony_f_number_word`] |
/// | `FocalLength` | none | 0x71 | `u16`, tenths of a millimetre |
///
/// The older table has no focal length field. Bytes not listed are zero, and the first byte of
/// 0x9405 is the one that tells the reader (and ExifTool) which layout it is.
fn sony_table(b: &ArwBuilder, older: bool) -> Vec<u8> {
    let mut plain = vec![0u8; if older { 0x100 } else { 0x200 }];
    let mut put = |at: usize, value: u16| plain[at..at + 2].copy_from_slice(&value.to_le_bytes());
    let iso_word = b.sony_iso_word.unwrap_or_else(|| sony_word_for_iso(b.iso));
    let time_word = sony_exposure_word(b.exposure_time.0, b.exposure_time.1);
    let (time_num, time_den) = rational_u16(b.exposure_time.0, b.exposure_time.1);
    let f_word = sony_f_number_word(b.f_number.0, b.f_number.1);
    let focal_tenths = (f64::from(b.focal_length.0) * 10.0 / f64::from(b.focal_length.1.max(1)))
        .round()
        .clamp(0.0, f64::from(u16::MAX)) as u16;
    if older {
        put(0x04, iso_word);
        put(0x0E, time_word);
        put(0x10, time_num);
        put(0x12, time_den);
        put(0x14, f_word);
    } else {
        put(0x04, iso_word);
        put(0x0A, time_word);
        put(0x0C, time_num);
        put(0x0E, time_den);
        put(0x10, f_word);
        put(0x71, focal_tenths);
    }
    if older {
        // Stored first byte 0x8B (the layout of the A7 III to A7R IV and A9 bodies).
        plain[0] = crate::makernote::decipher_byte(0x8B);
    }
    plain.iter().map(|byte| encipher_byte(*byte)).collect()
}

/// The maker note value: 12-byte header, the directory, then its out-of-line values. Offsets are
/// absolute, so the block's own file position matters.
fn maker_note(b: &ArwBuilder, plan: &Plan) -> Vec<u8> {
    let mut entries = Vec::new();
    if let Some(id) = b.model_id {
        entries.push(Ent::short(0xB001, &[id]));
    }
    entries.push(Ent::byte(0xB000, &b.file_format));
    let preview = match &b.preview {
        MakerPreview::Omitted => None,
        MakerPreview::Generated(w, h) => Some((tiny_jpeg(*w, *h, 2), *w, *h)),
        MakerPreview::Custom(jpeg) => {
            let (w, h) = jpeg_dimensions(jpeg).unwrap_or((0, 0));
            Some((jpeg.clone(), w, h))
        }
    };
    if let Some((jpeg, w, h)) = preview {
        let jpeg = pad_jpeg_to(jpeg, b.preview_min_bytes);
        let mut data = Vec::with_capacity(32 + jpeg.len());
        data.extend_from_slice(&(jpeg.len() as u32).to_le_bytes());
        data.extend_from_slice(&u32::from(w).to_le_bytes());
        data.extend_from_slice(&u32::from(h).to_le_bytes());
        data.extend_from_slice(&[0u8; 8]);
        data.extend_from_slice(&u32::from(w).to_le_bytes());
        data.extend_from_slice(&u32::from(h).to_le_bytes());
        data.extend_from_slice(&[0u8; 4]);
        data.extend_from_slice(&jpeg);
        entries.push(Ent::undefined(0x2001, &data));
    }
    match b.resolved_iso_table() {
        SonyIsoTable::Tag9405 => entries.push(Ent::undefined(0x9405, &sony_table(b, true))),
        SonyIsoTable::Tag9416 => entries.push(Ent::undefined(0x9416, &sony_table(b, false))),
        SonyIsoTable::None => {}
    }
    let header_len = b.maker_note_header.len();
    let mut out = b.maker_note_header.clone();
    out.extend_from_slice(&serialize_ifd(plan.maker + header_len, &entries, 0));
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::RawError;
    use crate::layout::DecodeKind;
    use crate::metadata::{LevelSource, RawStorage, probe_bytes};

    fn mosaic(w: u32, h: u32) -> Vec<u16> {
        gradient_mosaic(w, h, 512, 16_000, 7)
    }

    #[test]
    fn splitmix_is_deterministic_and_spread_out() {
        let mut a = SplitMix64(42);
        let mut b = SplitMix64(42);
        let first: Vec<u64> = (0..4).map(|_| a.next_u64()).collect();
        let second: Vec<u64> = (0..4).map(|_| b.next_u64()).collect();
        assert_eq!(first, second);
        assert_ne!(first[0], first[1]);
        // Reference value of SplitMix64 for seed 0.
        assert_eq!(SplitMix64(0).next_u64(), 0xE220_A839_7B1D_CDAF);
    }

    #[test]
    fn gradient_mosaic_stays_within_the_levels_and_is_reproducible() {
        let a = gradient_mosaic(64, 32, 512, 16_000, 9);
        assert_eq!(a, gradient_mosaic(64, 32, 512, 16_000, 9));
        assert_ne!(a, gradient_mosaic(64, 32, 512, 16_000, 10));
        assert!(a.iter().all(|v| (512..=16_000).contains(v)));
        assert_eq!(a.len(), 64 * 32);
    }

    /// Decodes the entropy-coded data of `tiny_jpeg` with an independent minimal decoder (the
    /// tables are fixed) and returns the DC values per block and component.
    fn decode_dc(jpeg: &[u8], blocks: usize) -> Vec<[i32; 3]> {
        let sos = jpeg
            .windows(2)
            .position(|w| w == [0xFF, 0xDA])
            .expect("SOS marker");
        let header_len = usize::from(u16::from_be_bytes([jpeg[sos + 2], jpeg[sos + 3]]));
        let mut data = Vec::new();
        let mut i = sos + 2 + header_len;
        while i + 1 < jpeg.len() {
            if jpeg[i] == 0xFF && jpeg[i + 1] == 0xD9 {
                break;
            }
            data.push(jpeg[i]);
            i += if jpeg[i] == 0xFF { 2 } else { 1 }; // skip the stuffed zero
        }
        let mut position = 0usize;
        let mut take = |n: usize| -> i32 {
            let mut v = 0i32;
            for _ in 0..n {
                let byte = data[position / 8];
                v = (v << 1) | i32::from((byte >> (7 - position % 8)) & 1);
                position += 1;
            }
            v
        };
        let mut previous = [0i32; 3];
        let mut out = Vec::new();
        for _ in 0..blocks {
            for p in previous.iter_mut() {
                let category = take(4) as usize;
                let diff = if category == 0 {
                    0
                } else {
                    let raw = take(category);
                    if raw < (1 << (category - 1)) {
                        raw - (1 << category) + 1
                    } else {
                        raw
                    }
                };
                *p += diff;
                assert_eq!(take(1), 0, "end of block");
            }
            out.push(previous);
        }
        out
    }

    #[test]
    fn tiny_jpeg_entropy_data_decodes_to_the_gradient_it_encodes() {
        // A tiny independent decoder reads back every block's DC values.
        let jpeg = tiny_jpeg(40, 24, 3);
        let dc = decode_dc(&jpeg, 5 * 3);
        assert_eq!(dc.len(), 15);
        // The luma gradient rises left to right (the noise is at most 4 steps).
        assert!(dc[4][0] > dc[0][0] + 20, "{dc:?}");
        // Every value is a legal quantised DC.
        assert!(
            dc.iter().flatten().all(|v| (-64..=63).contains(v)),
            "{dc:?}"
        );
        // Different seeds give different pictures; the same seed gives the same bytes.
        assert_ne!(jpeg, tiny_jpeg(40, 24, 4));
        assert_eq!(jpeg, tiny_jpeg(40, 24, 3));
    }

    #[test]
    fn padding_keeps_the_jpeg_valid_and_reaches_the_size() {
        let plain = tiny_jpeg(64, 48, 1);
        for target in [
            0,
            plain.len() - 1,
            plain.len(),
            plain.len() + 1,
            plain.len() + 5,
            200_000,
        ] {
            let padded = pad_jpeg_to(plain.clone(), target);
            assert!(padded.len() >= target.max(plain.len()), "target {target}");
            assert_eq!(&padded[..4], &[0xFF, 0xD8, 0xFF, 0xDB]);
            assert_eq!(&padded[padded.len() - 2..], &[0xFF, 0xD9]);
            // The picture data is untouched: decoding still gives the same DC values.
            assert_eq!(
                decode_dc(&padded, 8 * 6),
                decode_dc(&plain, 8 * 6),
                "target {target}"
            );
        }
        // Not a Sony-style JPEG: returned as is.
        assert_eq!(pad_jpeg_to(vec![1, 2, 3], 100), vec![1, 2, 3]);
    }

    #[test]
    fn tiny_jpeg_has_the_marker_structure_of_a_baseline_jpeg() {
        let jpeg = tiny_jpeg(33, 17, 1);
        // Walk the segments up to SOS.
        let mut markers = Vec::new();
        let mut pos = 2;
        while jpeg[pos] == 0xFF && jpeg[pos + 1] != 0xDA {
            markers.push(jpeg[pos + 1]);
            pos += 2 + usize::from(u16::from_be_bytes([jpeg[pos + 2], jpeg[pos + 3]]));
        }
        assert_eq!(markers, [0xDB, 0xC0, 0xC4, 0xC4]);
        assert_eq!(jpeg[pos + 1], 0xDA);
        // SOF0 carries the size, big endian.
        let sof = jpeg
            .windows(2)
            .position(|w| w == [0xFF, 0xC0])
            .expect("SOF0");
        assert_eq!(u16::from_be_bytes([jpeg[sof + 5], jpeg[sof + 6]]), 17);
        assert_eq!(u16::from_be_bytes([jpeg[sof + 7], jpeg[sof + 8]]), 33);
        // 33 x 17 needs 5 x 3 blocks; the entropy data is not empty and ends with EOI.
        assert_eq!(&jpeg[jpeg.len() - 2..], &[0xFF, 0xD9]);
    }

    #[test]
    fn tiny_jpeg_is_a_jpeg_of_the_requested_size() {
        let jpeg = tiny_jpeg(32, 24, 1);
        // SOI then the quantisation table, as in Sony previews.
        assert_eq!(&jpeg[..4], &[0xFF, 0xD8, 0xFF, 0xDB]);
        assert_eq!(&jpeg[jpeg.len() - 2..], &[0xFF, 0xD9]);
        assert_ne!(jpeg, tiny_jpeg(32, 24, 2));
    }

    #[test]
    fn a_default_file_round_trips_through_the_probe() {
        let file = ArwBuilder::new("ILCE-7RM4", 64, 48).build(&mosaic(64, 48));
        let meta = probe_bytes(&file).expect("probe");
        assert_eq!(meta.make, "SONY");
        assert_eq!(meta.model, "ILCE-7RM4");
        assert_eq!(meta.model_id, Some(375));
        assert_eq!(meta.arw_version, Some([3, 2, 0, 0]));
        assert_eq!(meta.software.as_deref(), Some("ILCE-7RM4 v1.00"));
        assert_eq!((meta.width, meta.height), (64, 48));
        assert_eq!(meta.bits_per_sample, 14);
        assert_eq!(meta.compression, 1);
        assert_eq!(meta.decode, DecodeKind::Uncompressed14);
        assert_eq!(meta.cfa, CfaPattern::RGGB);
        assert_eq!(meta.black, [512.0; 4]);
        assert_eq!(meta.white, [16300.0; 4]);
        assert_eq!(meta.black_source, LevelSource::Sr2);
        assert_eq!(meta.white_source, LevelSource::Sr2);
        assert_eq!(meta.wb_as_shot, Some([2043.0, 1024.0, 1024.0, 1755.0]));
        assert_eq!(meta.iso, Some(400));
        assert_eq!(meta.exposure_time_s, Some(1.0 / 250.0));
        assert_eq!(meta.f_number, Some(4.0));
        assert!((meta.exposure_comp_ev.unwrap() + 0.7).abs() < 1e-12);
        assert_eq!(meta.focal_length_mm, Some(35.0));
        assert_eq!(meta.lens_model.as_deref(), Some("FE 24-70mm F2.8 GM"));
        assert_eq!(meta.orientation, 1);
        let t = meta.capture_time.expect("capture time");
        assert_eq!(t.to_iso8601(), "2026-10-04T12:03:44.070+02:00");
        assert!(meta.preview.is_some());
        assert!(meta.lens_corrections.is_some());
        assert!(meta.gps.is_none());
        match &meta.storage {
            RawStorage::Strips {
                offsets,
                byte_counts,
                ..
            } => {
                assert_eq!(offsets.len(), 1);
                assert_eq!(byte_counts, &[64 * 48 * 2]);
                // The data is last in the file and is the mosaic as little-endian words.
                let start = offsets[0] as usize;
                assert_eq!(start + 64 * 48 * 2, file.len());
                assert_eq!(
                    u16::from_le_bytes([file[start], file[start + 1]]),
                    mosaic(64, 48)[0]
                );
            }
            other => panic!("expected strips, got {other:?}"),
        }
    }

    #[test]
    fn sr2_block_really_is_encrypted_in_the_file() {
        let plain = ArwBuilder::new("ILCE-7M3", 32, 32)
            .black([777; 4])
            .build(&mosaic(32, 32));
        // The plaintext black level 777 as little-endian words must not appear in the file...
        let needle = [
            777u16.to_le_bytes(),
            777u16.to_le_bytes(),
            777u16.to_le_bytes(),
            777u16.to_le_bytes(),
        ]
        .concat();
        assert!(!plain.windows(needle.len()).any(|w| w == needle.as_slice()));
        // ...yet the probe reads it back.
        assert_eq!(probe_bytes(&plain).expect("probe").black, [777.0; 4]);
    }

    #[test]
    fn legacy_sr2_tags_round_trip_with_grbg_reordering() {
        let file = ArwBuilder::new("ILCE-7M3", 32, 32)
            .sr2(Sr2Variant::Legacy)
            .wb(Some([2200, 1024, 1030, 1700]))
            .black([256; 4])
            .build(&mosaic(32, 32));
        let meta = probe_bytes(&file).expect("probe");
        assert_eq!(meta.wb_as_shot, Some([2200.0, 1024.0, 1030.0, 1700.0]));
        assert_eq!(meta.black, [256.0; 4]);
    }

    #[test]
    fn without_sr2_the_levels_fall_back_to_the_raw_ifd_then_defaults() {
        let raw_ifd = ArwBuilder::new("ILCE-1", 32, 32)
            .sr2(Sr2Variant::Absent)
            .raw_ifd_levels(true)
            .black([600; 4])
            .wb(Some([2100, 1024, 1024, 1650]))
            .build(&mosaic(32, 32));
        let meta = probe_bytes(&raw_ifd).expect("probe");
        assert_eq!(meta.black, [600.0; 4]);
        assert_eq!(meta.black_source, LevelSource::RawIfd);
        assert_eq!(meta.wb_source, Some(LevelSource::RawIfd));
        assert_eq!(meta.white_source, LevelSource::Default);
        assert_eq!(meta.white, [16383.0; 4]);

        let nothing = ArwBuilder::new("ILCE-1", 32, 32)
            .sr2(Sr2Variant::Absent)
            .build(&mosaic(32, 32));
        let meta = probe_bytes(&nothing).expect("probe");
        assert_eq!(meta.black_source, LevelSource::Default);
        assert_eq!(meta.black, [512.0; 4]);
        assert_eq!(meta.wb_as_shot, None);
    }

    #[test]
    fn twelve_bit_data_scales_14_bit_sr2_levels() {
        let file = ArwBuilder::new("ILCE-7RM4", 32, 32)
            .uncompressed(12)
            .white([16383; 3])
            .build(&gradient_mosaic(32, 32, 128, 4000, 3));
        let meta = probe_bytes(&file).expect("probe");
        assert_eq!(meta.decode, DecodeKind::Uncompressed12);
        assert_eq!(meta.black, [128.0; 4]);
        assert_eq!(meta.white, [4095.0; 4]);
    }

    /// Modern ARW files have no `SONY DSC` header in front of the maker-note directory (ExifTool:
    /// `MakerNoteSony5`, "used in SR2 and ARW images"; the ILCE-9 in the lclevy/sony_raw dump
    /// starts its maker note with `63 00`, the entry count). The reader must find the model ID,
    /// the ARW version, `SonyISO` and the preview either way.
    #[test]
    fn maker_notes_with_and_without_the_sony_header_read_the_same() {
        let read = |header: &[u8]| {
            let file = ArwBuilder::new("ILCE-7M4", 64, 48)
                .maker_note_header(header)
                .iso(3200)
                .build(&mosaic(64, 48));
            let meta = probe_bytes(&file).expect("probe");
            (
                meta.model_id,
                meta.arw_version,
                meta.iso,
                meta.iso_source,
                meta.preview.map(|p| p.length),
            )
        };
        let headerless = read(&[]);
        assert_eq!(headerless.0, Some(388));
        assert_eq!(headerless.1, Some([3, 2, 0, 0]));
        assert_eq!(headerless.2, Some(3200));
        assert_eq!(headerless.3, Some(crate::exif::IsoSource::SonyMakerNote));
        assert!(headerless.4.is_some(), "maker-note preview");
        for header in [
            &b"SONY DSC \0\0\0"[..],
            b"SONY CAM \0\0\0",
            b"SONY MOBILE\0",
        ] {
            assert_eq!(read(header), headerless, "{header:?}");
        }
    }

    #[test]
    fn a_headerless_maker_note_is_only_read_for_sony_family_makes() {
        let build = |make: &str, model: &str| {
            ArwBuilder::new(model, 64, 48)
                .make(make)
                .model_id(358)
                .build(&mosaic(64, 48))
        };
        // A Sony-written raw IFD (the 0x70xx tags) keeps a file readable under another make, but
        // another make's maker note is not interpreted as Sony's.
        let other = probe_bytes(&build("NIKON CORPORATION", "D850")).expect("probe");
        assert_eq!(other.model_id, None);
        assert_eq!(other.arw_version, None);
        for (make, model) in [
            ("SONY", "ILCE-9"),
            ("Sony", "ILCE-9"),
            ("Hasselblad", "Lunar"),
            ("HASSELBLAD", "Stellar II"),
            ("Hasselblad", "Lusso"),
            ("Hasselblad", "HV"),
        ] {
            let meta = probe_bytes(&build(make, model)).expect("probe");
            assert_eq!(meta.model_id, Some(358), "{make} {model}");
        }
        // A Hasselblad medium-format body is not a Sony.
        let h6d = probe_bytes(&build("Hasselblad", "H6D-100c")).expect("probe");
        assert_eq!(h6d.model_id, None);
    }

    /// Values from the real ILCE-9 file (firmware 0.01) in github.com/lclevy/sony_raw
    /// (`SONY_A9_DSC04458.txt`, an ExifTool 12.41 dump). Everything the dump shows for the tags we
    /// read is reproduced here, on a smaller image: the headerless maker note, the SR2 white
    /// level 15360 next to the raw IFD's `WhiteLevel` 16383, 0x7313 stored as signed 16-bit, the
    /// measured `SonyISO` word 2890 (ISO 2619) next to EXIF ISO 2500, and the lens tables.
    #[test]
    fn a_file_laid_out_like_the_lclevy_ilce_9_dump_reads_as_the_dump_says() {
        let lens = LensTables {
            vignetting: Some(vec![
                16, 0, 32, 128, 288, 480, 768, 1120, 1536, 2016, 2592, 3232, 4000, 4896, 5984,
                7296, 8672,
            ]),
            chromatic_aberration: Some(vec![
                32, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 640, 640, 640, 896,
                896, 768, 768, 768, 768, 640, 640, 640, 640, 640, 640, 512, 512, 384, 256, 128,
            ]),
            distortion: Some(vec![
                16, 13, 0, -25, -56, -99, -149, -208, -270, -334, -395, -458, -512, -557, -584,
                -593, -571,
            ]),
        };
        let file = ArwBuilder::new("ILCE-9", 96, 64)
            .software("ILCE-9 v0.01")
            .capture_time("2017:04:20 14:14:46", "", Some("-05:00"))
            .iso(2500)
            .sony_iso_word(2890)
            .exposure_time(1, 60)
            .f_number(56, 10)
            .exposure_comp(0, 10)
            .focal_length(240, 10)
            .lens_model(None)
            .file_format([3, 3, 3, 0])
            .black([512; 4])
            .white([15360; 3])
            .wb(Some([2364, 1024, 1024, 1728]))
            .raw_ifd_levels(true)
            .raw_ifd_white_level(16383)
            .tone_curve(Some([8000, 10400, 12900, 14100]))
            .lens_tables(lens.clone())
            .crop(Rect {
                x: 12,
                y: 12,
                width: 72,
                height: 40,
            })
            .build(&gradient_mosaic(96, 64, 512, 15_000, 4));
        let meta = probe_bytes(&file).expect("probe");
        assert_eq!(
            (meta.make.as_str(), meta.model.as_str()),
            ("SONY", "ILCE-9")
        );
        assert_eq!(meta.model_id, Some(358));
        assert_eq!(meta.arw_version_name().as_deref(), Some("ARW 2.3.3"));
        assert_eq!(meta.software.as_deref(), Some("ILCE-9 v0.01"));
        assert_eq!(meta.decode, DecodeKind::Uncompressed14);
        // The maker note's measured sensitivity wins over the nominal EXIF ISO 2500.
        assert_eq!(meta.iso, Some(2619));
        assert_eq!(meta.iso_source, Some(crate::exif::IsoSource::SonyMakerNote));
        assert_eq!(meta.exposure_time_s, Some(1.0 / 60.0));
        assert_eq!(meta.f_number, Some(5.6));
        assert_eq!(meta.focal_length_mm, Some(24.0));
        let taken = meta.capture_time.expect("capture time");
        assert_eq!(taken.to_iso8601(), "2017-04-20T14:14:46-05:00");
        // SR2 first for the white level (15360), the raw IFD's 16383 only as a fallback.
        assert_eq!(
            (meta.white, meta.white_source),
            ([15360.0; 4], LevelSource::Sr2)
        );
        assert_eq!(
            (meta.black, meta.black_source),
            ([512.0; 4], LevelSource::Sr2)
        );
        assert_eq!(meta.wb_as_shot, Some([2364.0, 1024.0, 1024.0, 1728.0]));
        assert_eq!(meta.tone_curve, Some([8000, 10400, 12900, 14100]));
        let found = meta.lens_corrections.expect("lens tables");
        assert_eq!(found.vignetting.expect("vig").raw, lens.vignetting.unwrap());
        let ca = found.chromatic_aberration.expect("ca");
        assert_eq!((ca.count(), ca.knots().len()), (Some(32), 32));
        assert_eq!(
            found.distortion.expect("dist").raw,
            lens.distortion.unwrap()
        );
        assert_eq!(
            meta.crop,
            Some(Rect {
                x: 12,
                y: 12,
                width: 72,
                height: 40
            })
        );

        // Without the SR2 block the white level comes from the raw IFD.
        let no_sr2 = ArwBuilder::new("ILCE-9", 96, 64)
            .sr2(Sr2Variant::Absent)
            .raw_ifd_levels(true)
            .raw_ifd_white_level(16383)
            .build(&gradient_mosaic(96, 64, 512, 15_000, 4));
        let meta = probe_bytes(&no_sr2).expect("probe");
        assert_eq!(
            (meta.white, meta.white_source),
            ([16383.0; 4], LevelSource::RawIfd)
        );
        assert_eq!(meta.black_source, LevelSource::RawIfd);
    }

    #[test]
    fn iso_sources_in_order_of_preference() {
        // High ISO: EXIF ISOSpeedRatings saturates, the maker note keeps the real value.
        let file = ArwBuilder::new("ILCE-1", 32, 32)
            .iso(102_400)
            .build(&mosaic(32, 32));
        let meta = probe_bytes(&file).expect("probe");
        assert_eq!(meta.iso, Some(102_400));
        assert_eq!(meta.iso_source, Some(crate::exif::IsoSource::SonyMakerNote));

        // Without the enciphered table: the 32-bit recommended exposure index.
        let file = ArwBuilder::new("ILCE-1", 32, 32)
            .iso(102_400)
            .iso_table(SonyIsoTable::None)
            .build(&mosaic(32, 32));
        let meta = probe_bytes(&file).expect("probe");
        assert_eq!(meta.iso, Some(102_400));
        assert_eq!(
            meta.iso_source,
            Some(crate::exif::IsoSource::ExifRecommendedExposureIndex)
        );

        // The 9405 table of older bodies works too.
        let file = ArwBuilder::new("ILCE-7M3", 32, 32)
            .iso(3200)
            .build(&mosaic(32, 32));
        let meta = probe_bytes(&file).expect("probe");
        assert_eq!(
            (meta.iso, meta.iso_source),
            (Some(3200), Some(crate::exif::IsoSource::SonyMakerNote))
        );
    }

    #[test]
    fn model_ids_come_from_the_table_and_can_be_overridden_or_dropped() {
        for (model, id) in [
            ("ILCE-1", 384),
            ("ILCE-7M3", 363),
            ("ILCE-6400", 371),
            ("DSC-RX100M4", 341),
        ] {
            let file = ArwBuilder::new(model, 32, 32).build(&mosaic(32, 32));
            assert_eq!(
                probe_bytes(&file).expect("probe").model_id,
                Some(id),
                "{model}"
            );
        }
        let file = ArwBuilder::new("ILCE-1", 32, 32)
            .model_id(999)
            .build(&mosaic(32, 32));
        assert_eq!(probe_bytes(&file).expect("probe").model_id, Some(999));
        let file = ArwBuilder::new("ILCE-1", 32, 32)
            .without_model_id()
            .build(&mosaic(32, 32));
        assert_eq!(probe_bytes(&file).expect("probe").model_id, None);
        assert_eq!(model_id_for("not-a-camera"), None);
    }

    #[test]
    fn hasselblad_rebadges_are_read_as_arw() {
        let file = ArwBuilder::new("Lunar", 32, 32)
            .make("Hasselblad")
            .model_id(358)
            .maker_note_header(b"VHAB     \0\0\0")
            .build(&mosaic(32, 32));
        let meta = probe_bytes(&file).expect("probe");
        assert_eq!(meta.make, "Hasselblad");
        assert_eq!(meta.model, "Lunar");
        assert_eq!(meta.model_id, Some(358));
        assert_eq!(meta.decode, DecodeKind::Uncompressed14);
    }

    #[test]
    fn the_cam_header_variant_is_read() {
        let file = ArwBuilder::new("ILCE-7M3", 32, 32)
            .maker_note_header(b"SONY CAM \0\0\0")
            .build(&mosaic(32, 32));
        assert_eq!(probe_bytes(&file).expect("probe").model_id, Some(363));
    }

    #[test]
    fn container_only_payloads_are_classified() {
        let cases: [(ContainerOnly, DecodeKind); 6] = [
            (ContainerOnly::craw(), DecodeKind::Craw),
            (ContainerOnly::packed12(), DecodeKind::Uncompressed12),
            (ContainerOnly::lossless_l(), DecodeKind::LosslessL),
            (
                ContainerOnly::lossless_ms(ChromaSubsampling::Yuv420, true),
                DecodeKind::LosslessMS {
                    subsampling: ChromaSubsampling::Yuv420,
                },
            ),
            (ContainerOnly::arw6(12), DecodeKind::Arw6),
            (
                ContainerOnly::other(9999, 14),
                DecodeKind::Unknown {
                    compression: 9999,
                    bits: 14,
                },
            ),
        ];
        for (payload, expected) in cases {
            let file = ArwBuilder::new("ILCE-7RM5", 1024, 768)
                .container_only(payload)
                .build(&[]);
            let meta = probe_bytes(&file).expect("probe");
            assert_eq!(meta.decode, expected);
        }
    }

    /// Every real cRAW file says BitsPerSample 12 (A500, A5000, NEX-5N, NEX-7, ILCE-7M3 in
    /// metadata-extractor-images) and stores one byte per pixel. LibRaw reads black 512 and white
    /// 15360 from them, on the 14-bit scale of the decoded data.
    #[test]
    fn a_craw_file_as_the_cameras_write_it_is_craw_with_14_bit_levels() {
        let file = ArwBuilder::new("ILCE-7M3", 96, 64)
            .container_only(ContainerOnly::craw())
            .tone_curve(Some([8000, 10_400, 12_900, 14_100]))
            .black([512; 4])
            .white([15_360; 3])
            .build(&[]);
        let meta = probe_bytes(&file).expect("probe");
        assert_eq!(meta.bits_per_sample, 12, "as the tag says");
        assert_eq!(meta.compression, 32_767);
        assert_eq!(meta.raw_file_type, Some(2));
        assert_eq!(meta.decode, DecodeKind::Craw);
        assert_eq!(meta.black, [512.0; 4]);
        assert_eq!(meta.white, [15_360.0; 4]);
        assert_eq!(meta.tone_curve, Some([8000, 10_400, 12_900, 14_100]));
        // The format description's 8 bits, and a missing BitsPerSample, are cRAW too.
        for payload in [
            ContainerOnly::other(32_767, 8),
            ContainerOnly::other(32_767, 0),
        ] {
            let file = ArwBuilder::new("ILCE-7M3", 96, 64)
                .container_only(payload)
                .build(&[]);
            assert_eq!(probe_bytes(&file).expect("probe").decode, DecodeKind::Craw);
        }
    }

    #[test]
    fn tiled_payloads_record_tile_geometry() {
        let file = ArwBuilder::new("ILCE-1", 1100, 700)
            .container_only(ContainerOnly::lossless_l())
            .build(&[]);
        let meta = probe_bytes(&file).expect("probe");
        match meta.storage {
            RawStorage::Tiles {
                tile_width,
                tile_length,
                offsets,
                byte_counts,
            } => {
                assert_eq!((tile_width, tile_length), (512, 512));
                // 1100 / 512 -> 3 columns, 700 / 512 -> 2 rows.
                assert_eq!(offsets.len(), 6);
                assert_eq!(byte_counts.len(), 6);
            }
            other => panic!("expected tiles, got {other:?}"),
        }
    }

    #[test]
    fn lossless_ms_subsampling_comes_from_the_tag_or_the_jpeg_header() {
        for (subsampling, with_tag) in [
            (ChromaSubsampling::Yuv420, true),
            (ChromaSubsampling::Yuv422, true),
            (ChromaSubsampling::Yuv444, true),
            (ChromaSubsampling::Yuv422, false),
            (ChromaSubsampling::Yuv444, false),
        ] {
            let file = ArwBuilder::new("ILCE-7RM5", 640, 480)
                .container_only(ContainerOnly::lossless_ms(subsampling, with_tag))
                .build(&[]);
            let meta = probe_bytes(&file).expect("probe");
            assert_eq!(
                meta.decode,
                DecodeKind::LosslessMS { subsampling },
                "tag {with_tag}"
            );
            assert!(matches!(
                meta.layout(),
                crate::layout::Layout::Ycbcr { subsampling: s } if s == subsampling
            ));
        }
    }

    #[test]
    fn arq_is_reported_as_arq() {
        let file = ArwBuilder::new("ILCE-7RM4", 64, 48)
            .container_only(ContainerOnly::arq())
            .build(&[]);
        assert!(matches!(probe_bytes(&file), Err(RawError::Arq)));
    }

    #[test]
    fn thumbnail_in_ifd0_is_found_too() {
        let file = ArwBuilder::new("ILCE-7M3", 32, 32)
            .thumbnail_ifd(ThumbnailIfd::Ifd0)
            .preview(None)
            .build(&mosaic(32, 32));
        let meta = probe_bytes(&file).expect("probe");
        let jpeg = meta.preview.expect("thumbnail");
        assert_eq!(&file[jpeg.offset as usize..][..3], &[0xFF, 0xD8, 0xFF]);
    }

    #[test]
    fn the_largest_jpeg_wins_as_preview() {
        let file = ArwBuilder::new("ILCE-7M3", 32, 32)
            .thumbnail(Some((40, 30)))
            .preview(Some((400, 300)))
            .build(&mosaic(32, 32));
        let meta = probe_bytes(&file).expect("probe");
        let jpeg = meta.preview.expect("preview");
        // The 400 x 300 JPEG is far bigger than the 40 x 30 one.
        assert!(jpeg.length > 1500, "{}", jpeg.length);
        let bytes = &file[jpeg.offset as usize..][..jpeg.length as usize];
        assert_eq!(&bytes[..3], &[0xFF, 0xD8, 0xFF]);
        assert_eq!(&bytes[bytes.len() - 2..], &[0xFF, 0xD9]);

        let only_thumb = ArwBuilder::new("ILCE-7M3", 32, 32)
            .preview(None)
            .build(&mosaic(32, 32));
        let small = probe_bytes(&only_thumb)
            .expect("probe")
            .preview
            .expect("thumbnail");
        assert!(small.length < jpeg.length);

        let none = ArwBuilder::new("ILCE-7M3", 32, 32)
            .preview(None)
            .thumbnail(None)
            .build(&mosaic(32, 32));
        assert_eq!(probe_bytes(&none).expect("probe").preview, None);
    }

    /// The ILCE-7M4 file in metadata-extractor-images holds three JPEGs: a 160 x 120 thumbnail
    /// (IFD1), the 1616 x 1080 preview (IFD0) and a 7008 x 4672 full-size one (IFD2, 1.7 MB). The
    /// first thumbnails want the preview: the smallest JPEG that is at least 1024 px on its long
    /// edge, else the largest one.
    #[test]
    fn the_preview_is_the_smallest_jpeg_big_enough_to_show() {
        let jpeg_of = |file: &[u8]| {
            let meta = probe_bytes(file).expect("probe");
            let found = meta.preview.expect("a preview");
            file[found.offset as usize..][..found.length as usize].to_vec()
        };
        // Preview 1616 x 1080 next to a much bigger JPEG (standing in for the full-size one) and
        // next to the small thumbnail: the 1616 x 1080 one wins over both.
        let file = ArwBuilder::new("ILCE-7M4", 32, 32)
            .preview(Some((1616, 1080)))
            .thumbnail(Some((3000, 2000)))
            .build(&mosaic(32, 32));
        assert_eq!(jpeg_of(&file), tiny_jpeg(1616, 1080, 2));
        let file = ArwBuilder::new("ILCE-7M4", 32, 32)
            .preview(Some((1616, 1080)))
            .thumbnail(Some((160, 120)))
            .build(&mosaic(32, 32));
        assert_eq!(jpeg_of(&file), tiny_jpeg(1616, 1080, 2));
        // A portrait preview is measured by its long edge too.
        let file = ArwBuilder::new("ILCE-7M4", 32, 32)
            .preview(Some((1080, 1616)))
            .thumbnail(Some((2000, 3000)))
            .build(&mosaic(32, 32));
        assert_eq!(jpeg_of(&file), tiny_jpeg(1080, 1616, 2));
        // Only a big one next to a small one: the big one is the preview.
        let file = ArwBuilder::new("ILCE-7M4", 32, 32)
            .preview(Some((800, 600)))
            .thumbnail(Some((3000, 2000)))
            .build(&mosaic(32, 32));
        assert_eq!(jpeg_of(&file), tiny_jpeg(3000, 2000, 1));
        // Nothing reaches 1024 px: the largest JPEG.
        let file = ArwBuilder::new("ILCE-7M4", 32, 32)
            .preview(Some((800, 600)))
            .thumbnail(Some((160, 120)))
            .build(&mosaic(32, 32));
        assert_eq!(jpeg_of(&file), tiny_jpeg(800, 600, 2));
    }

    /// Sony files with Lossless M and S data carry an extra 8-bit "RGB preview" coded as lossless
    /// JPEG (LibRaw skips that directory when it picks the raw image). It is bigger than the real
    /// preview, so a "largest JPEG" rule would pick a stream no viewer can show.
    #[test]
    fn a_lossless_jpeg_is_never_the_preview() {
        let file = ArwBuilder::new("ILCE-1", 32, 32)
            .thumbnail(Some((600, 400)))
            .lossless_thumbnail(true)
            .preview(Some((320, 214)))
            .build(&mosaic(32, 32));
        let meta = probe_bytes(&file).expect("probe");
        let jpeg = meta.preview.expect("the maker-note preview");
        let bytes = &file[jpeg.offset as usize..][..jpeg.length as usize];
        assert_eq!(&bytes[..4], &[0xFF, 0xD8, 0xFF, 0xDB]);
        assert!(
            bytes.windows(2).any(|w| w == [0xFF, 0xC0]),
            "a baseline frame header"
        );
        assert!(!bytes.windows(2).any(|w| w == [0xFF, 0xC3]));
        // With nothing else to show, there is no preview at all.
        let only = ArwBuilder::new("ILCE-1", 32, 32)
            .thumbnail(Some((600, 400)))
            .lossless_thumbnail(true)
            .preview(None)
            .build(&mosaic(32, 32));
        assert_eq!(probe_bytes(&only).expect("probe").preview, None);
    }

    #[test]
    fn lens_tags_tone_curve_and_areas_round_trip() {
        let tables = LensTables {
            vignetting: Some(vec![16, -5, 0, 5]),
            chromatic_aberration: None,
            distortion: Some((0..17).map(|i| i as i16 * -3).collect()),
        };
        let file = ArwBuilder::new("ILCE-1", 128, 96)
            .lens_tables(tables.clone())
            .tone_curve(Some([1000, 4000, 8000, 12000]))
            .active_area(120, 90)
            .crop(Rect {
                x: 4,
                y: 2,
                width: 100,
                height: 80,
            })
            .build(&mosaic(128, 96));
        let meta = probe_bytes(&file).expect("probe");
        let lens = meta.lens_corrections.expect("lens tags");
        assert_eq!(lens.vignetting.expect("vig").raw, vec![16, -5, 0, 5]);
        assert!(lens.chromatic_aberration.is_none());
        let distortion = lens.distortion.expect("dist");
        assert_eq!(distortion.count(), Some(0));
        assert_eq!(distortion.knots().len(), 16);
        assert_eq!(meta.tone_curve, Some([1000, 4000, 8000, 12000]));
        assert_eq!(
            meta.active_area,
            Some(Rect {
                x: 0,
                y: 0,
                width: 120,
                height: 90
            })
        );
        assert_eq!(
            meta.crop,
            Some(Rect {
                x: 4,
                y: 2,
                width: 100,
                height: 80
            })
        );

        let bare = ArwBuilder::new("ILCE-1", 128, 96)
            .lens_tables(LensTables {
                vignetting: None,
                chromatic_aberration: None,
                distortion: None,
            })
            .build(&mosaic(128, 96));
        let meta = probe_bytes(&bare).expect("probe");
        assert!(meta.lens_corrections.is_none());
        assert!(meta.tone_curve.is_none());
        assert!(meta.active_area.is_none() && meta.crop.is_none());
    }

    #[test]
    fn rectangles_outside_the_readout_are_ignored() {
        let file = ArwBuilder::new("ILCE-1", 64, 64)
            .active_area(100, 100)
            .crop(Rect {
                x: 60,
                y: 0,
                width: 10,
                height: 10,
            })
            .build(&mosaic(64, 64));
        let meta = probe_bytes(&file).expect("probe");
        assert_eq!(meta.active_area, None);
        assert_eq!(meta.crop, None);
    }

    #[test]
    fn gps_round_trips() {
        let file = ArwBuilder::new("ILCE-7M3", 32, 32)
            .gps(GpsSpec {
                latitude_deg: -33.8688,
                longitude_deg: 151.2093,
                altitude_m: 58.5,
            })
            .build(&mosaic(32, 32));
        let gps = probe_bytes(&file).expect("probe").gps.expect("gps");
        assert!((gps.latitude_deg.unwrap() + 33.8688).abs() < 1e-4);
        assert!((gps.longitude_deg.unwrap() - 151.2093).abs() < 1e-4);
        assert_eq!(gps.altitude_m, Some(58.5));
        assert_eq!(gps.utc_time.as_deref(), Some("2026-10-04T10:20:30Z"));
    }

    #[test]
    fn orientation_and_capture_time_setters_round_trip() {
        let file = ArwBuilder::new("ILCE-7M3", 32, 32)
            .orientation(6)
            .capture_time("2020:02:29 23:59:58", "5", None)
            .build(&mosaic(32, 32));
        let meta = probe_bytes(&file).expect("probe");
        assert_eq!(meta.orientation, 6);
        let t = meta.capture_time.expect("time");
        assert_eq!(t.to_iso8601(), "2020-02-29T23:59:58.500");
    }

    #[test]
    fn cfa_patterns_map_levels_to_raster_positions() {
        let file = ArwBuilder::new("ILCE-7M3", 32, 32)
            .cfa(CfaPattern::BGGR)
            .black([10, 20, 30, 40])
            .white([16000, 16100, 16200])
            .wb(Some([2000, 1000, 1010, 1800]))
            .build(&mosaic(32, 32));
        let meta = probe_bytes(&file).expect("probe");
        assert_eq!(meta.cfa, CfaPattern::BGGR);
        assert_eq!(meta.black, [40.0, 20.0, 30.0, 10.0]);
        assert_eq!(meta.white, [16200.0, 16100.0, 16100.0, 16000.0]);
        assert_eq!(meta.wb_as_shot, Some([1800.0, 1000.0, 1010.0, 2000.0]));
    }

    #[test]
    fn the_default_lens_tables_lead_with_n_and_the_ca_table_with_2n() {
        let tables = LensTables::default();
        let ca = tables.chromatic_aberration.as_ref().expect("ca");
        // 0x7035 holds red then blue: the count is 2n, and so is the number of knots after it.
        assert_eq!((ca[0], ca.len()), (32, 33));
        for table in [&tables.vignetting, &tables.distortion] {
            let table = table.as_ref().expect("table");
            assert_eq!((table[0], table.len()), (16, 17));
        }
        // The reader hands the table back as stored: count 2n = 32 and 32 knots.
        let file = ArwBuilder::new("ILCE-7RM4", 64, 48).build(&mosaic(64, 48));
        let found = probe_bytes(&file)
            .expect("probe")
            .lens_corrections
            .expect("lens tables");
        let ca = found.chromatic_aberration.expect("ca");
        assert_eq!((ca.count(), ca.knots().len()), (Some(32), 32));
        assert_eq!(ca.raw, tables.chromatic_aberration.unwrap());
    }

    /// Finds the stored (enciphered) bytes of the maker-note table `plain_len` bytes long that
    /// starts with `stored_prefix`, and returns them deciphered.
    fn table_in(file: &[u8], stored_prefix: &[u8], len: usize) -> Vec<u8> {
        let at = file
            .windows(stored_prefix.len())
            .position(|w| w == stored_prefix)
            .expect("the table is in the file");
        file[at..at + len]
            .iter()
            .map(|b| crate::makernote::decipher_byte(*b))
            .collect()
    }

    fn word(table: &[u8], at: usize) -> u16 {
        u16::from_le_bytes([table[at], table[at + 1]])
    }

    #[test]
    fn the_9416_table_holds_iso_exposure_time_f_number_and_focal_length() {
        let builder = ArwBuilder::new("ILCE-7M4", 64, 48)
            .iso(3200)
            .exposure_time(1, 125)
            .f_number(28, 10)
            .focal_length(500, 10);
        assert_eq!(builder.resolved_iso_table(), SonyIsoTable::Tag9416);
        let file = builder.build(&mosaic(64, 48));
        let stored = sony_table(&builder, false);
        assert_eq!(stored.len(), 0x200);
        // The file carries the stored bytes, and deciphering them gives the fields.
        let plain = table_in(&file, &stored[..0x80], 0x200);
        assert_eq!(word(&plain, 0x04), sony_word_for_iso(3200));
        let seconds = 2f64.powf(16.0 - f64::from(word(&plain, 0x0A)) / 256.0);
        assert!((seconds * 125.0 - 1.0).abs() < 0.003, "{seconds}");
        assert_eq!((word(&plain, 0x0C), word(&plain, 0x0E)), (1, 125));
        let f = 2f64.powf((f64::from(word(&plain, 0x10)) / 256.0 - 16.0) / 2.0);
        assert!((f - 2.8).abs() < 0.01, "{f}");
        assert_eq!(word(&plain, 0x71), 500);
        // The reader takes the ISO from it.
        let meta = probe_bytes(&file).expect("probe");
        assert_eq!(meta.iso, Some(3200));
        assert_eq!(meta.iso_source, Some(crate::exif::IsoSource::SonyMakerNote));
        // Only enciphered bytes are in the file: the plain focal length is not.
        assert!(
            !file
                .windows(0x80)
                .any(|w| w == &plain[..0x80] && plain[..0x80] != stored[..0x80])
        );
    }

    #[test]
    fn the_9405_table_holds_iso_exposure_time_and_f_number() {
        let builder = ArwBuilder::new("ILCE-7M3", 64, 48)
            .iso(800)
            .exposure_time(1, 4000)
            .f_number(56, 10)
            .focal_length(240, 10);
        assert_eq!(builder.resolved_iso_table(), SonyIsoTable::Tag9405);
        let file = builder.build(&mosaic(64, 48));
        let stored = sony_table(&builder, true);
        assert_eq!(stored.len(), 0x100);
        // The first stored byte is one of the layouts the reader and ExifTool accept.
        assert_eq!(stored[0], 0x8B);
        let plain = table_in(&file, &stored[..0x80], 0x100);
        assert_eq!(word(&plain, 0x04), sony_word_for_iso(800));
        let seconds = 2f64.powf(16.0 - f64::from(word(&plain, 0x0E)) / 256.0);
        assert!((seconds * 4000.0 - 1.0).abs() < 0.01, "{seconds}");
        assert_eq!((word(&plain, 0x10), word(&plain, 0x12)), (1, 4000));
        let f = 2f64.powf((f64::from(word(&plain, 0x14)) / 256.0 - 16.0) / 2.0);
        assert!((f - 5.6).abs() < 0.02, "{f}");
        let meta = probe_bytes(&file).expect("probe");
        assert_eq!(meta.iso, Some(800));
        assert_eq!(meta.iso_source, Some(crate::exif::IsoSource::SonyMakerNote));
    }

    #[test]
    fn exposure_words_and_fractions_cover_the_extremes() {
        // Whole stops: 1 s is 16 * 256, each halving adds 256.
        assert_eq!(sony_exposure_word(1, 1), 4096);
        assert_eq!(sony_exposure_word(1, 2), 4096 + 256);
        assert_eq!(sony_exposure_word(30, 1), 2840); // 256 * (16 - log2(30)) = 2839.9
        assert_eq!(sony_exposure_word(0, 1), 0, "bulb");
        assert_eq!(sony_f_number_word(1, 1), 4096);
        assert_eq!(sony_f_number_word(4, 1), 4096 + 1024);
        // Fractions that do not fit 16 bits are scaled down together.
        assert_eq!(rational_u16(1, 250), (1, 250));
        assert_eq!(rational_u16(30, 1), (30, 1));
        // A zero time is bulb; a zero denominator is read as 1 and never leaves a zero denominator behind.
        assert_eq!(rational_u16(0, 1), (0, 1));
        assert_eq!(rational_u16(7, 0), (7, 1));
        assert_eq!(rational_u16(4, 8000), (1, 2000));
        let (n, d) = rational_u16(100_000, 4_000_000);
        assert_eq!((n, d), (1, 40));
        let (n, d) = rational_u16(1, 200_000);
        assert_eq!(n, 0, "a time below 1/65535 s rounds to zero");
        assert_eq!(d, 65_535);
    }

    #[test]
    fn a_custom_preview_jpeg_is_embedded_byte_for_byte() {
        let jpeg = tiny_jpeg(222, 111, 77);
        let file = ArwBuilder::new("ILCE-7M4", 64, 48)
            .preview_jpeg(jpeg.clone())
            .build(&mosaic(64, 48));
        let meta = probe_bytes(&file).expect("probe");
        let found = meta.preview.expect("preview");
        assert_eq!(
            &file[found.offset as usize..][..found.length as usize],
            &jpeg[..]
        );
        // The 32-byte header in front of it carries the JPEG's own length, width and height.
        let header = &file[found.offset as usize - 32..found.offset as usize];
        let field = |at: usize| u32::from_le_bytes(header[at..at + 4].try_into().unwrap());
        assert_eq!(
            (field(0), field(4), field(8)),
            (jpeg.len() as u32, 222, 111)
        );
        assert_eq!((field(20), field(24)), (222, 111));
        // The last call wins: dropping the preview, or asking for a generated one.
        let none = ArwBuilder::new("ILCE-7M4", 64, 48)
            .preview_jpeg(jpeg.clone())
            .preview(None)
            .thumbnail(None)
            .build(&mosaic(64, 48));
        assert_eq!(probe_bytes(&none).expect("probe").preview, None);
        let generated = ArwBuilder::new("ILCE-7M4", 64, 48)
            .preview_jpeg(jpeg.clone())
            .preview(Some((222, 111)))
            .build(&mosaic(64, 48));
        let found = probe_bytes(&generated)
            .expect("probe")
            .preview
            .expect("preview");
        assert_eq!(
            &generated[found.offset as usize..][..found.length as usize],
            &tiny_jpeg(222, 111, 2)[..]
        );
        // A padded size minimum applies to a custom JPEG too, and keeps it a valid JPEG.
        let padded = ArwBuilder::new("ILCE-7M4", 64, 48)
            .preview_jpeg(jpeg.clone())
            .preview_min_bytes(jpeg.len() + 5000)
            .build(&mosaic(64, 48));
        let found = probe_bytes(&padded)
            .expect("probe")
            .preview
            .expect("preview");
        assert!(found.length as usize >= jpeg.len() + 5000);
        let bytes = &padded[found.offset as usize..][..found.length as usize];
        assert_eq!(super::jpeg_dimensions(bytes), Some((222, 111)));
        assert_eq!(&bytes[bytes.len() - 2..], &[0xFF, 0xD9]);
        // A JPEG without a frame header still embeds; its header sizes are zero.
        let odd = ArwBuilder::new("ILCE-7M4", 64, 48)
            .preview_jpeg(vec![0xFF, 0xD8, 0xFF, 0xDB, 0, 2, 0xFF, 0xD9])
            .build(&mosaic(64, 48));
        assert!(probe_bytes(&odd).is_ok());
    }

    #[test]
    fn a_stable_byte_image_for_a_fixed_configuration() {
        // The writer is deterministic: the same builder gives the same bytes.
        let a = ArwBuilder::new("ILCE-7RM4", 32, 32).build(&mosaic(32, 32));
        let b = ArwBuilder::new("ILCE-7RM4", 32, 32).build(&mosaic(32, 32));
        assert_eq!(a, b);
        assert_eq!(&a[..4], b"II*\0");
    }
}
