//! ARRI Look Files: `.alf4`, `.alf4c` and `.aml` (`docs/08-looks.md` section ARRI Look Files).
//!
//! An ARRI Look File is a small binary container that holds a creative look for ARRI cameras: an
//! optional ASC CDL and a 3D LUT. Yellowmoon reads three of them and turns each into the
//! [`LookInput`] / [`Cdl`] / [`Lut3d`] triple that [`crate::look::LookTransform`] applies:
//!
//! | Container | Magic | Version | What it holds | Becomes |
//! |---|---|---|---|---|
//! | `.alf4` (ALEXA 35) | `3FLA` | 3 | CDL, then one LogC4 to LogC4 3D LUT (the CMT) | [`LookInput::LogC4`] |
//! | `.alf4c` (custom colour management) | `4FLA` | 5 | CDL and up to three LUTs: CMT, SDR and HDR display transforms | the CMT as [`LookInput::LogC4`]; the display transforms are not used |
//! | `.aml` (ALF-2) | `KOLA` | 3 | CDL, then one 3D LUT from LogC3 to Rec.709 (video rendering included) | [`LookInput::LogC3ToRec709`] |
//!
//! The formats are ARRI's, undocumented as byte layouts. The layouts below were worked out from
//! ARRI's public guides and from files whose contents we know: the free Look Library downloads and
//! files ARRI's own tool wrote from LUTs of ours (`docs/14-licensing.md`). No ARRI code or data is
//! involved; users import their own files at runtime.
//!
//! # Layouts
//!
//! All integers are little endian. `crc32` is the zlib/IEEE CRC-32.
//!
//! ```text
//! .alf4     0x000  "3FLA"            0x004 u32 version (3)        0x008 u32 file size
//!           0x00C  u32 crc32 of [0x10, EOF)
//!           0x010  u32 note length L (<= 1024)   0x014 u32 parameter table offset   0x018 u32 parameter count
//!           0x01C  64 bytes type id (ARRI's library files: "com.arri.blackbox.alf4"; ART CMD writes zeros)
//!           0x05C  u32 block region offset  0x060 u32 0  0x064 u32 block count  0x068 u32 first block offset
//!           0x100  1024 bytes look NOTE, UTF-8, L bytes used
//!           0x500  parameter table: N x { u32 id, f32 value }
//!           block  256 bytes descriptor:
//!                    +0x000 u16 colour-space id length   +0x002 64 bytes colour-space id
//!                    +0x042 u16 LUT size n               +0x044 u32 scale (65535)
//!                    +0x048 u16 source file name length  +0x04A 64 bytes source file name
//!                    +0x08A u32 payload offset
//!                  then 6 n^3 bytes: u16 triples (R, G, B), node index (r*n + g)*n + b (blue fastest), value = code / scale
//!
//! .alf4c    0x000  "4FLA"            0x004 u32 version (5)        0x008 u32 file size
//!           0x00C  u32 crc32 of [0x10, EOF)
//!           0x010  u32 note length L   0x014 u32 parameter table offset   0x018 u32 parameter count   0x01C u32 present blocks
//!           0x020  3 slots (CMT, SDR, HDR) of 5 x u32: descriptor offset, payload offset, payload size, extra offset, extra size
//!                  (an absent block is an all-zero slot)
//!           0x100  note (L bytes), then the parameter table, then for each present slot:
//!                    256 bytes descriptor: +0x000 u32 colour-space id length, +0x004 64 bytes id, +0x044 u32 n, +0x048 u32 scale,
//!                    +0x04C u32 source name length, +0x050 64 bytes name, +0x090 112 bytes design-data key
//!                    6 n^3 payload as above, then the design-data zip (extra size bytes) when there is one
//!
//! .aml      0x000  "KOLA"            0x004 u32 version (3)        0x008 u32 crc32 of [0x0C, EOF)
//!           0x00C  u32 parameter table offset  0x010 u32 parameter count  0x014 u32 LUT offset  0x018 u32 LUT size n
//!           0x01E  u16 name field length  0x022 u16 colour-space id length
//!           0x100  name (NUL terminated, <= 0x440 bytes)    0x540  64 bytes colour-space id
//!           0x580  parameter table, then the LUT payload as above (scale 65535)
//! ```
//!
//! # Parameters
//!
//! The parameter table holds more entries than a look needs and its order differs between files, so
//! entries are looked up **by id**; unknown ids are ignored. The ASC CDL ids are inferred from the
//! files' structure (no file we have seen carries a CDL other than the identity), so treat them as
//! "best reading": slope `3011 3012 3013` (R, G, B), offset `3001 3002 3003`, power `3021 3022 3023`,
//! saturation `3040`. A CDL that is the identity is dropped. Look Intensity (id `10006`, inferred) is
//! not used: looks open at 100 % like every look (`docs/08-looks.md`).
//!
//! # Validation
//!
//! [`parse`] and [`parse_file`] never panic. They check the magic, the version, the size field, the
//! CRC-32, that every offset and length lies inside the file, that the LUT size is 2 to 129, and
//! cap the counts and string lengths. Every problem is an [`AlfError`] whose text is a sentence for
//! the person importing the file.
//!
//! # Writing
//!
//! Yellowmoon only reads look files. [`AlfFile::to_bytes`] and the constructors exist so tests can
//! make fixtures from their own LUTs and CDLs, and so a parsed file can be rebuilt byte for byte
//! to prove the layout above.

use crate::cdl::{Cdl, CdlError};
use crate::consts::Vec3;
use crate::cube::{MAX_3D_SIZE, MIN_3D_SIZE};
use crate::look::LookInput;
use crate::lut::Lut3d;

/// Largest look file that is read, in bytes. A 129-point LUT is 12.9 MB; three of them and a design
/// archive fit comfortably.
pub const MAX_FILE_BYTES: usize = 64 * 1024 * 1024;

/// The colour-space id of a look that maps LogC4 / AWG4 to LogC4 / AWG4 (the CMT of an `.alf4`).
pub const LOGC4_COLORSPACE: &str = "com.arri.colorspace.awg4.d65.logcv4";
/// The colour-space id of Rec.709 with the BT.1886 (gamma 2.4) curve, the target of an `.aml`.
pub const REC709_COLORSPACE: &str = "com.apple.proapps.colorspace.rec1886";
/// What every other colour-space id of a display target starts with.
const APPLE_COLORSPACE_PREFIX: &str = "com.apple.proapps.colorspace.";

/// Parameter ids (see the module documentation: inferred, looked up by id).
pub mod param {
    /// CDL slope for R, G, B.
    pub const SLOPE: [u32; 3] = [3011, 3012, 3013];
    /// CDL offset for R, G, B.
    pub const OFFSET: [u32; 3] = [3001, 3002, 3003];
    /// CDL power for R, G, B.
    pub const POWER: [u32; 3] = [3021, 3022, 3023];
    /// CDL saturation.
    pub const SATURATION: u32 = 3040;
    /// Look Intensity (not used).
    pub const INTENSITY: u32 = 10006;
}

const ALF4_MAGIC: [u8; 4] = *b"3FLA";
const ALF4C_MAGIC: [u8; 4] = *b"4FLA";
const AML_MAGIC: [u8; 4] = *b"KOLA";
const ALF4_VERSION: u32 = 3;
const ALF4C_VERSION: u32 = 5;
const AML_VERSION: u32 = 3;

/// Most parameters a table may hold (real files have 11 or 29).
const MAX_PARAMS: usize = 4096;
/// The ID and file-name fields of a LUT descriptor are 64 bytes.
const NAME_FIELD: usize = 64;
/// The note field of an `.alf4` is 1024 bytes.
const ALF4_NOTE_FIELD: usize = 1024;
/// Longest note of an `.alf4c` (it is not padded; the parameter table follows it).
const ALF4C_NOTE_MAX: usize = 64 * 1024;
/// The name field of an `.aml`, 0x100 to 0x540.
const AML_NAME_FIELD: usize = 0x440;
/// The design-data key field of an `.alf4c` descriptor, +0x90 to +0x100.
const DESIGN_KEY_FIELD: usize = 0x70;
/// Most blocks an `.alf4` may announce (it has one; the first is read).
const ALF4_MAX_BLOCKS: u32 = 64;

/// Which container a look file is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AlfKind {
    /// `.alf4`: ALEXA 35 look (CDL and CMT).
    Alf4,
    /// `.alf4c`: custom colour management (CDL, CMT, SDR and HDR display transforms).
    Alf4c,
    /// `.aml`: ALF-2 look for LogC3 cameras (CDL and a LUT with the Rec.709 rendering in it).
    Aml,
}

impl AlfKind {
    /// The file extension, without the dot.
    #[must_use]
    pub fn extension(self) -> &'static str {
        match self {
            Self::Alf4 => "alf4",
            Self::Alf4c => "alf4c",
            Self::Aml => "aml",
        }
    }

    /// The kind a file with these first bytes is, if any.
    #[must_use]
    pub fn sniff(bytes: &[u8]) -> Option<Self> {
        match bytes.get(..4)? {
            m if m == ALF4_MAGIC => Some(Self::Alf4),
            m if m == ALF4C_MAGIC => Some(Self::Alf4c),
            m if m == AML_MAGIC => Some(Self::Aml),
            _ => None,
        }
    }
}

impl std::fmt::Display for AlfKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, ".{}", self.extension())
    }
}

/// Why a look file could not be read. The messages are sentences for the person importing it.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum AlfError {
    /// The first bytes are none of the three markers.
    #[error(
        "It is not an ARRI Look File. A .alf4, .alf4c or .aml file starts with a marker this file does not have."
    )]
    NotALookFile,
    /// A version this build does not read.
    #[error(
        "It is version {found} of the {kind} format. Yellowmoon reads version {supported}. Update Yellowmoon, or export the look again."
    )]
    UnsupportedVersion {
        /// The container.
        kind: AlfKind,
        /// The version in the file.
        found: u32,
        /// The version this build reads.
        supported: u32,
    },
    /// More than [`MAX_FILE_BYTES`].
    #[error("It is {bytes} bytes, far larger than a look file ({limit} bytes at most).")]
    TooLarge {
        /// Size of the input.
        bytes: usize,
        /// The limit.
        limit: usize,
    },
    /// Shorter than its own header, or its size field disagrees with its length.
    #[error("It is cut short or damaged: its size does not match. Download it again.")]
    Truncated,
    /// The CRC-32 does not match.
    #[error("Its checksum does not match, so the file is damaged. Download it again.")]
    BadChecksum,
    /// A table, string or LUT lies outside the file, or a count is absurd.
    #[error("It is damaged: a table points outside the file. Download it again.")]
    Layout,
    /// The LUT is smaller than 2 or larger than 129 points per side.
    #[error("Its LUT has {0} points per side. Yellowmoon reads 2 to 129.")]
    LutSize(u32),
    /// The code scale of the LUT is 0 or above 65535.
    #[error("Its LUT uses a value scale of {0}, which Yellowmoon does not read.")]
    Scale(u32),
    /// A CDL value is NaN or infinite.
    #[error(
        "Its colour correction (CDL) holds a value that is not a number, so the file is damaged."
    )]
    CdlNotFinite,
    /// A CDL value is outside what an ASC CDL allows.
    #[error(
        "Its colour correction (CDL) is outside the allowed ranges. Slope and saturation cannot be negative and power must be above zero."
    )]
    CdlRange,
    /// An `.alf4c` without a creative look.
    #[error(
        "It holds only display transforms and no creative look (CMT), so there is nothing to import."
    )]
    NoCmt,
    /// A look for a colour space other than LogC4.
    #[error(
        "Its look is made for another colour space ({0}). Yellowmoon can use ARRI Look Files made for LogC4."
    )]
    NotLogC4(String),
    /// A look made for a display other than Rec.709.
    #[error("It is made for {0}. Yellowmoon can use ARRI Look Files made for Rec.709.")]
    NotRec709(String),
    /// An `.aml` with no LUT (only ARRI's look parameters).
    #[error("It has no LUT, only ARRI's look parameters, so there is nothing to import.")]
    NoLut,
    /// A field of a file being written is longer than its place in the layout.
    #[error("The {0} is too long for the file format.")]
    FieldTooLong(&'static str),
}

/// One entry of a parameter table.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Param {
    /// The parameter id.
    pub id: u32,
    /// Its value.
    pub value: f32,
}

/// One LUT of an `.alf4` or `.alf4c`, as stored.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LutBlock {
    /// The colour-space id the LUT is made for.
    pub colorspace: String,
    /// Points per side.
    pub size: usize,
    /// The code that stands for 1.0 (65535 in every file seen).
    pub scale: u32,
    /// The name of the LUT file the block was made from.
    pub source_name: String,
    /// The key of the design-data archive (`.alf4c` only; empty when there is none).
    pub design_key: String,
    /// `6 * size^3` bytes: little-endian u16 triples, blue varying fastest, exactly as stored.
    pub payload: Vec<u8>,
    /// The design-data archive stored after the payload (`.alf4c` only), as bytes.
    pub extra: Vec<u8>,
}

impl LutBlock {
    /// A block from a cube over `[0, 1]`: `colorspace` is the id, `source_name` the file name it
    /// is said to come from. Codes are `floor(v * 65535 + 0.5)`, the rule ARRI's tool uses, and
    /// values outside `[0, 1]` clip.
    #[must_use]
    pub fn from_lut3d(lut: &Lut3d, colorspace: &str, source_name: &str) -> Self {
        let n = lut.size();
        let mut payload = Vec::with_capacity(6 * n * n * n);
        for r in 0..n {
            for g in 0..n {
                for b in 0..n {
                    let node = lut.data()[r + n * (g + n * b)];
                    for c in node {
                        payload.extend_from_slice(&encode_code(c).to_le_bytes());
                    }
                }
            }
        }
        Self {
            colorspace: colorspace.to_owned(),
            size: n,
            scale: 65535,
            source_name: source_name.to_owned(),
            design_key: String::new(),
            payload,
            extra: Vec::new(),
        }
    }

    /// The cube this block holds, red varying fastest, over `[0, 1]`.
    ///
    /// # Errors
    ///
    /// [`AlfError::LutSize`], [`AlfError::Scale`] or [`AlfError::Layout`] when the block is not
    /// what a parsed file would hold.
    pub fn to_lut3d(&self) -> Result<Lut3d, AlfError> {
        decode_lut(self.size, self.scale, &self.payload)
    }
}

fn encode_code(v: f64) -> u16 {
    // `as` saturates, and a NaN becomes 0; the cube's own values are finite.
    (v * 65535.0 + 0.5).floor().clamp(0.0, 65535.0) as u16
}

/// An `.alf4`, field by field.
#[derive(Debug, Clone, PartialEq)]
pub struct Alf4 {
    /// The type id string of ARRI's library files; empty in files written by ART CMD.
    pub type_id: String,
    /// The look note (not a name).
    pub note: String,
    /// The parameter table, in stored order.
    pub params: Vec<Param>,
    /// The CMT.
    pub lut: LutBlock,
}

/// An `.alf4c`, field by field.
#[derive(Debug, Clone, PartialEq)]
pub struct Alf4c {
    /// The note (the name of the custom colour management, where ART keeps it, is not known).
    pub note: String,
    /// The parameter table, in stored order.
    pub params: Vec<Param>,
    /// The creative look, if the file has one.
    pub cmt: Option<LutBlock>,
    /// The SDR display transform.
    pub sdr: Option<LutBlock>,
    /// The HDR display transform.
    pub hdr: Option<LutBlock>,
}

/// An `.aml`, field by field.
#[derive(Debug, Clone, PartialEq)]
pub struct Aml {
    /// The name inside the file (the name of the cube it was made from).
    pub name: String,
    /// The name-length field as stored: the length plus one in ARRI's library files, the length
    /// in files from ART CMD.
    pub name_field_len: u16,
    /// The colour-space id of the display the LUT renders to.
    pub colorspace: String,
    /// The parameter table, in stored order.
    pub params: Vec<Param>,
    /// Points per side; 0 for a file without a LUT.
    pub size: usize,
    /// The LUT payload, like [`LutBlock::payload`]; empty without a LUT.
    pub payload: Vec<u8>,
}

/// A look file's containers, field by field: what [`parse_file`] reads and [`AlfFile::to_bytes`]
/// writes.
#[derive(Debug, Clone, PartialEq)]
pub enum AlfFile {
    /// An `.alf4`.
    Alf4(Alf4),
    /// An `.alf4c`.
    Alf4c(Alf4c),
    /// An `.aml`.
    Aml(Aml),
}

/// A look file read into what Yellowmoon applies.
#[derive(Debug, Clone, PartialEq)]
pub struct AlfLook {
    /// The container.
    pub kind: AlfKind,
    /// The encoding the look expects: [`LookInput::LogC4`] or [`LookInput::LogC3ToRec709`].
    pub input: LookInput,
    /// The file's CDL, applied before the LUT; `None` when it is the identity.
    pub cdl: Option<Cdl>,
    /// The LUT, red varying fastest over `[0, 1]`, at the size the file stores it.
    pub lut: Lut3d,
    /// The note of an `.alf4` or `.alf4c` (empty for an `.aml`).
    pub note: String,
    /// An `.alf4c` carries SDR and HDR display transforms of its own. They are not used.
    pub display_transforms_ignored: bool,
}

// ---- CRC-32 -------------------------------------------------------------------------------------

const fn crc_table() -> [u32; 256] {
    let mut table = [0_u32; 256];
    let mut i = 0;
    while i < 256 {
        let mut c = i as u32;
        let mut k = 0;
        while k < 8 {
            c = if c & 1 != 0 {
                0xEDB8_8320 ^ (c >> 1)
            } else {
                c >> 1
            };
            k += 1;
        }
        table[i] = c;
        i += 1;
    }
    table
}

static CRC_TABLE: [u32; 256] = crc_table();

/// The zlib/IEEE CRC-32 of `bytes`.
#[must_use]
pub fn crc32(bytes: &[u8]) -> u32 {
    let mut c = 0xFFFF_FFFF_u32;
    for &b in bytes {
        c = CRC_TABLE[((c ^ u32::from(b)) & 0xFF) as usize] ^ (c >> 8);
    }
    !c
}

// ---- Reading ------------------------------------------------------------------------------------

/// Bounds-checked little-endian reads.
struct Reader<'a> {
    bytes: &'a [u8],
}

impl<'a> Reader<'a> {
    fn slice(&self, offset: usize, len: usize) -> Result<&'a [u8], AlfError> {
        let end = offset.checked_add(len).ok_or(AlfError::Layout)?;
        self.bytes.get(offset..end).ok_or(AlfError::Layout)
    }

    fn u16(&self, offset: usize) -> Result<u16, AlfError> {
        let b = self.slice(offset, 2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }

    fn u32(&self, offset: usize) -> Result<u32, AlfError> {
        let b = self.slice(offset, 4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    /// A u32 that is an offset or a length.
    fn usize(&self, offset: usize) -> Result<usize, AlfError> {
        usize::try_from(self.u32(offset)?).map_err(|_| AlfError::Layout)
    }

    /// `len` bytes at `offset` as text, up to the first NUL.
    fn text(&self, offset: usize, len: usize) -> Result<String, AlfError> {
        let b = self.slice(offset, len)?;
        let end = b.iter().position(|&c| c == 0).unwrap_or(b.len());
        Ok(String::from_utf8_lossy(&b[..end]).into_owned())
    }

    fn params(&self, offset: usize, count: usize) -> Result<Vec<Param>, AlfError> {
        if count > MAX_PARAMS {
            return Err(AlfError::Layout);
        }
        let table = self.slice(offset, count * 8)?;
        Ok(table
            .chunks_exact(8)
            .map(|e| Param {
                id: u32::from_le_bytes([e[0], e[1], e[2], e[3]]),
                value: f32::from_le_bytes([e[4], e[5], e[6], e[7]]),
            })
            .collect())
    }
}

fn check_size(n: u32) -> Result<usize, AlfError> {
    let n_usize = usize::try_from(n).map_err(|_| AlfError::LutSize(n))?;
    if (MIN_3D_SIZE..=MAX_3D_SIZE).contains(&n_usize) {
        Ok(n_usize)
    } else {
        Err(AlfError::LutSize(n))
    }
}

fn check_scale(scale: u32) -> Result<u32, AlfError> {
    if (1..=65535).contains(&scale) {
        Ok(scale)
    } else {
        Err(AlfError::Scale(scale))
    }
}

/// The payload of a LUT of `size` points: `6 * size^3` bytes at `offset`.
fn payload_at(r: &Reader<'_>, offset: usize, size: usize) -> Result<Vec<u8>, AlfError> {
    Ok(r.slice(offset, 6 * size * size * size)?.to_vec())
}

fn decode_lut(size: usize, scale: u32, payload: &[u8]) -> Result<Lut3d, AlfError> {
    let size_u32 = u32::try_from(size).unwrap_or(u32::MAX);
    check_size(size_u32)?;
    let scale = f64::from(check_scale(scale)?);
    if payload.len() != 6 * size * size * size {
        return Err(AlfError::Layout);
    }
    let mut data: Vec<Vec3> = vec![[0.0; 3]; size * size * size];
    for r in 0..size {
        for g in 0..size {
            for b in 0..size {
                let at = 6 * ((r * size + g) * size + b);
                let code = |c: usize| {
                    f64::from(u16::from_le_bytes([
                        payload[at + 2 * c],
                        payload[at + 2 * c + 1],
                    ])) / scale
                };
                // The file runs blue fastest; `Lut3d` runs red fastest.
                data[r + size * (g + size * b)] = [code(0), code(1), code(2)];
            }
        }
    }
    Lut3d::new(size, [0.0; 3], [1.0; 3], data).map_err(|_| AlfError::Layout)
}

fn check_header(bytes: &[u8], kind: AlfKind, version: u32) -> Result<(), AlfError> {
    if bytes.len() > MAX_FILE_BYTES {
        return Err(AlfError::TooLarge {
            bytes: bytes.len(),
            limit: MAX_FILE_BYTES,
        });
    }
    if bytes.len() < 0x10 {
        return Err(AlfError::Truncated);
    }
    let r = Reader { bytes };
    let found = r.u32(4)?;
    if found != version {
        return Err(AlfError::UnsupportedVersion {
            kind,
            found,
            supported: version,
        });
    }
    Ok(())
}

/// Reads the size field (`.alf4`, `.alf4c`) and the CRC-32 at 0x0C over `[0x10, EOF)`.
fn check_size_and_crc(bytes: &[u8]) -> Result<(), AlfError> {
    let r = Reader { bytes };
    if r.usize(8)? != bytes.len() {
        return Err(AlfError::Truncated);
    }
    if r.u32(0x0C)? != crc32(&bytes[0x10..]) {
        return Err(AlfError::BadChecksum);
    }
    Ok(())
}

fn parse_alf4(bytes: &[u8]) -> Result<Alf4, AlfError> {
    check_header(bytes, AlfKind::Alf4, ALF4_VERSION)?;
    check_size_and_crc(bytes)?;
    let r = Reader { bytes };
    let note_len = r.usize(0x10)?;
    let table = r.usize(0x14)?;
    let count = r.usize(0x18)?;
    if note_len > ALF4_NOTE_FIELD {
        return Err(AlfError::Layout);
    }
    let type_id = r.text(0x1C, NAME_FIELD)?;
    let blocks = r.u32(0x64)?;
    let first = r.usize(0x68)?;
    if blocks == 0 || blocks > ALF4_MAX_BLOCKS {
        return Err(AlfError::Layout);
    }
    let note = String::from_utf8_lossy(r.slice(0x100, note_len)?).into_owned();
    let params = r.params(table, count)?;

    r.slice(first, 0x100)?;
    let cs_len = usize::from(r.u16(first)?);
    let name_len = usize::from(r.u16(first + 0x48)?);
    if cs_len > NAME_FIELD || name_len > NAME_FIELD {
        return Err(AlfError::Layout);
    }
    let colorspace = String::from_utf8_lossy(r.slice(first + 2, cs_len)?).into_owned();
    let size = check_size(u32::from(r.u16(first + 0x42)?))?;
    let scale = check_scale(r.u32(first + 0x44)?)?;
    let source_name = String::from_utf8_lossy(r.slice(first + 0x4A, name_len)?).into_owned();
    let payload = payload_at(&r, r.usize(first + 0x8A)?, size)?;
    Ok(Alf4 {
        type_id,
        note,
        params,
        lut: LutBlock {
            colorspace,
            size,
            scale,
            source_name,
            design_key: String::new(),
            payload,
            extra: Vec::new(),
        },
    })
}

fn parse_alf4c_block(r: &Reader<'_>, slot: usize) -> Result<Option<LutBlock>, AlfError> {
    let slot_bytes = r.slice(slot, 20)?;
    if slot_bytes.iter().all(|&b| b == 0) {
        return Ok(None);
    }
    let desc = r.usize(slot)?;
    let data = r.usize(slot + 4)?;
    let data_len = r.usize(slot + 8)?;
    let extra = r.usize(slot + 12)?;
    let extra_len = r.usize(slot + 16)?;
    r.slice(desc, 0x100)?;
    let cs_len = r.usize(desc)?;
    let name_len = r.usize(desc + 0x4C)?;
    if cs_len > NAME_FIELD || name_len > NAME_FIELD {
        return Err(AlfError::Layout);
    }
    let colorspace = String::from_utf8_lossy(r.slice(desc + 4, cs_len)?).into_owned();
    let size = check_size(r.u32(desc + 0x44)?)?;
    let scale = check_scale(r.u32(desc + 0x48)?)?;
    let source_name = String::from_utf8_lossy(r.slice(desc + 0x50, name_len)?).into_owned();
    let design_key = r.text(desc + 0x90, DESIGN_KEY_FIELD)?;
    if data_len != 6 * size * size * size {
        return Err(AlfError::Layout);
    }
    let payload = payload_at(r, data, size)?;
    let extra = if extra_len > 0 {
        r.slice(extra, extra_len)?.to_vec()
    } else {
        Vec::new()
    };
    Ok(Some(LutBlock {
        colorspace,
        size,
        scale,
        source_name,
        design_key,
        payload,
        extra,
    }))
}

fn parse_alf4c(bytes: &[u8]) -> Result<Alf4c, AlfError> {
    check_header(bytes, AlfKind::Alf4c, ALF4C_VERSION)?;
    check_size_and_crc(bytes)?;
    let r = Reader { bytes };
    let note_len = r.usize(0x10)?;
    let table = r.usize(0x14)?;
    let count = r.usize(0x18)?;
    if note_len > ALF4C_NOTE_MAX {
        return Err(AlfError::Layout);
    }
    let note = String::from_utf8_lossy(r.slice(0x100, note_len)?).into_owned();
    let params = r.params(table, count)?;
    let cmt = parse_alf4c_block(&r, 0x20)?;
    let sdr = parse_alf4c_block(&r, 0x20 + 20)?;
    let hdr = parse_alf4c_block(&r, 0x20 + 40)?;
    Ok(Alf4c {
        note,
        params,
        cmt,
        sdr,
        hdr,
    })
}

fn parse_aml(bytes: &[u8]) -> Result<Aml, AlfError> {
    check_header(bytes, AlfKind::Aml, AML_VERSION)?;
    let r = Reader { bytes };
    if r.u32(8)? != crc32(&bytes[0x0C..]) {
        return Err(AlfError::BadChecksum);
    }
    let table = r.usize(0x0C)?;
    let count = r.usize(0x10)?;
    let lut_offset = r.usize(0x14)?;
    let size_field = r.u32(0x18)?;
    let name_field_len = r.u16(0x1E)?;
    let cs_len = usize::from(r.u16(0x22)?);
    if cs_len > NAME_FIELD {
        return Err(AlfError::Layout);
    }
    let name = r.text(0x100, usize::from(name_field_len).min(AML_NAME_FIELD))?;
    let colorspace = String::from_utf8_lossy(r.slice(0x540, cs_len)?).into_owned();
    let params = r.params(table, count)?;
    // A size of zero, or nothing after the parameter table, is a look without a LUT.
    let (size, payload) = if size_field == 0 || lut_offset == bytes.len() {
        (0, Vec::new())
    } else {
        let size = check_size(size_field)?;
        (size, payload_at(&r, lut_offset, size)?)
    };
    Ok(Aml {
        name,
        name_field_len,
        colorspace,
        params,
        size,
        payload,
    })
}

/// Reads a look file's container, checking everything the module documentation lists.
///
/// # Errors
///
/// An [`AlfError`] for a file that is not a look file, is of another version, is damaged, or whose
/// LUT size is out of range.
pub fn parse_file(bytes: &[u8]) -> Result<AlfFile, AlfError> {
    match AlfKind::sniff(bytes).ok_or(AlfError::NotALookFile)? {
        AlfKind::Alf4 => parse_alf4(bytes).map(AlfFile::Alf4),
        AlfKind::Alf4c => parse_alf4c(bytes).map(AlfFile::Alf4c),
        AlfKind::Aml => parse_aml(bytes).map(AlfFile::Aml),
    }
}

/// Reads a look file into what Yellowmoon applies ([`parse_file`] then [`AlfFile::look`]).
///
/// # Errors
///
/// As [`parse_file`] and [`AlfFile::look`].
pub fn parse(bytes: &[u8]) -> Result<AlfLook, AlfError> {
    parse_file(bytes)?.look()
}

// ---- From container to look ---------------------------------------------------------------------

fn param_value(params: &[Param], id: u32, default: f32) -> f32 {
    params
        .iter()
        .find(|p| p.id == id)
        .map_or(default, |p| p.value)
}

/// The CDL a parameter table holds, or `None` when it is the identity.
fn cdl_of(params: &[Param]) -> Result<Option<Cdl>, AlfError> {
    let triple = |ids: [u32; 3], default: f32| -> Vec3 {
        ids.map(|id| f64::from(param_value(params, id, default)))
    };
    let slope = triple(param::SLOPE, 1.0);
    let offset = triple(param::OFFSET, 0.0);
    let power = triple(param::POWER, 1.0);
    let saturation = f64::from(param_value(params, param::SATURATION, 1.0));
    let cdl = Cdl::new(slope, offset, power, saturation).map_err(|e| match e {
        CdlError::NotFinite => AlfError::CdlNotFinite,
        _ => AlfError::CdlRange,
    })?;
    Ok((!cdl.is_identity()).then_some(cdl))
}

/// A display target from a colour-space id, for the message: `rec2100.d65.pq` is "a Rec.2020 PQ
/// display".
fn display_label(colorspace: &str) -> String {
    let id = colorspace
        .strip_prefix(APPLE_COLORSPACE_PREFIX)
        .unwrap_or(colorspace)
        .to_ascii_lowercase();
    let label = if id.starts_with("rec2100") && id.ends_with("pq") {
        "Rec.2020 PQ"
    } else if id.starts_with("rec2100") && id.ends_with("hlg") {
        "Rec.2020 HLG"
    } else if id.starts_with("rec2020") {
        "Rec.2020"
    } else if id.starts_with("p3") {
        "P3"
    } else {
        ""
    };
    if !label.is_empty() {
        format!("a {label} display")
    } else if colorspace.is_empty() {
        "an unknown display".to_owned()
    } else {
        format!("a display Yellowmoon does not know ({colorspace})")
    }
}

fn logc4_block(block: &LutBlock) -> Result<(), AlfError> {
    if block.colorspace.eq_ignore_ascii_case(LOGC4_COLORSPACE) {
        Ok(())
    } else {
        Err(AlfError::NotLogC4(if block.colorspace.is_empty() {
            "none given".to_owned()
        } else {
            block.colorspace.clone()
        }))
    }
}

impl AlfFile {
    /// Which container this is.
    #[must_use]
    pub fn kind(&self) -> AlfKind {
        match self {
            Self::Alf4(_) => AlfKind::Alf4,
            Self::Alf4c(_) => AlfKind::Alf4c,
            Self::Aml(_) => AlfKind::Aml,
        }
    }

    /// The look this file holds, as Yellowmoon applies it (see the module documentation for what
    /// each container becomes).
    ///
    /// # Errors
    ///
    /// [`AlfError::NoCmt`], [`AlfError::NotLogC4`], [`AlfError::NotRec709`], [`AlfError::NoLut`],
    /// [`AlfError::CdlNotFinite`] or [`AlfError::CdlRange`] for a file that has nothing Yellowmoon
    /// can use, and the LUT errors of [`LutBlock::to_lut3d`].
    pub fn look(&self) -> Result<AlfLook, AlfError> {
        match self {
            Self::Alf4(f) => {
                logc4_block(&f.lut)?;
                Ok(AlfLook {
                    kind: AlfKind::Alf4,
                    input: LookInput::LogC4,
                    cdl: cdl_of(&f.params)?,
                    lut: f.lut.to_lut3d()?,
                    note: f.note.clone(),
                    display_transforms_ignored: false,
                })
            }
            Self::Alf4c(f) => {
                let cmt = f.cmt.as_ref().ok_or(AlfError::NoCmt)?;
                logc4_block(cmt)?;
                Ok(AlfLook {
                    kind: AlfKind::Alf4c,
                    input: LookInput::LogC4,
                    cdl: cdl_of(&f.params)?,
                    lut: cmt.to_lut3d()?,
                    note: f.note.clone(),
                    display_transforms_ignored: f.sdr.is_some() || f.hdr.is_some(),
                })
            }
            Self::Aml(f) => {
                if f.size == 0 {
                    return Err(AlfError::NoLut);
                }
                if !f.colorspace.eq_ignore_ascii_case(REC709_COLORSPACE) {
                    return Err(AlfError::NotRec709(display_label(&f.colorspace)));
                }
                Ok(AlfLook {
                    kind: AlfKind::Aml,
                    input: LookInput::LogC3ToRec709,
                    cdl: cdl_of(&f.params)?,
                    lut: decode_lut(f.size, 65535, &f.payload)?,
                    note: String::new(),
                    display_transforms_ignored: false,
                })
            }
        }
    }
}

// ---- Writing ------------------------------------------------------------------------------------

fn put_u16(out: &mut [u8], offset: usize, v: u16) {
    out[offset..offset + 2].copy_from_slice(&v.to_le_bytes());
}

fn put_u32(out: &mut [u8], offset: usize, v: u32) {
    out[offset..offset + 4].copy_from_slice(&v.to_le_bytes());
}

fn put_len(out: &mut [u8], offset: usize, len: usize) -> Result<(), AlfError> {
    put_u32(
        out,
        offset,
        u32::try_from(len).map_err(|_| AlfError::FieldTooLong("file"))?,
    );
    Ok(())
}

fn put_text(
    out: &mut [u8],
    offset: usize,
    text: &str,
    field: usize,
    what: &'static str,
) -> Result<(), AlfError> {
    if text.len() > field {
        return Err(AlfError::FieldTooLong(what));
    }
    out[offset..offset + text.len()].copy_from_slice(text.as_bytes());
    Ok(())
}

fn param_bytes(params: &[Param]) -> Vec<u8> {
    let mut out = Vec::with_capacity(params.len() * 8);
    for p in params {
        out.extend_from_slice(&p.id.to_le_bytes());
        out.extend_from_slice(&p.value.to_le_bytes());
    }
    out
}

/// The 11 parameters ART CMD writes for `.alf4` and `.alf4c`, in its order.
fn cdl_params(cdl: Option<&Cdl>) -> Vec<Param> {
    let cdl = cdl.copied().unwrap_or_default();
    let mut out = Vec::with_capacity(11);
    let mut push = |id: u32, value: f64| {
        out.push(Param {
            id,
            value: value as f32,
        });
    };
    for c in 0..3 {
        push(param::SLOPE[c], cdl.slope[c]);
    }
    for c in 0..3 {
        push(param::OFFSET[c], cdl.offset[c]);
    }
    for c in 0..3 {
        push(param::POWER[c], cdl.power[c]);
    }
    push(param::SATURATION, cdl.saturation);
    push(param::INTENSITY, 1.0);
    out
}

/// The 29 parameters of ARRI's library `.aml` files: the CDL, then the video look parameters at
/// their defaults, in the library's order.
fn aml_params(cdl: Option<&Cdl>) -> Vec<Param> {
    let cdl = cdl.copied().unwrap_or_default();
    let mut out = Vec::with_capacity(29);
    let mut push = |id: u32, value: f64| {
        out.push(Param {
            id,
            value: value as f32,
        });
    };
    for c in 0..3 {
        push(param::SLOPE[c], cdl.slope[c]);
    }
    for c in 0..3 {
        push(param::OFFSET[c], cdl.offset[c]);
    }
    for c in 0..3 {
        push(param::POWER[c], cdl.power[c]);
    }
    push(param::SATURATION, cdl.saturation);
    push(4002, 0.5);
    push(4001, 0.5);
    push(5040, 1.0);
    push(5051, 1.0);
    for id in 5061..=5072 {
        push(id, 1.0);
    }
    for id in 5073..=5075 {
        push(id, 0.0);
    }
    out
}

fn finish_crc(out: &mut [u8], crc_at: usize, from: usize) {
    let crc = crc32(&out[from..]);
    put_u32(out, crc_at, crc);
}

impl AlfFile {
    /// An `.alf4` as ART CMD writes it from a LogC4 to LogC4 cube: `lut` over `[0, 1]`, the CDL as
    /// parameters (identity when `None`), a note and the name of the cube it is said to come from.
    #[must_use]
    pub fn alf4(lut: &Lut3d, cdl: Option<&Cdl>, note: &str, source_name: &str) -> Self {
        Self::Alf4(Alf4 {
            type_id: String::new(),
            note: note.to_owned(),
            params: cdl_params(cdl),
            lut: LutBlock::from_lut3d(lut, LOGC4_COLORSPACE, source_name),
        })
    }

    /// An `.alf4c` with the given CMT (or none) and display transforms. The SDR and HDR blocks are
    /// made from `sdr` and `hdr`, Rec.709 and PQ.
    #[must_use]
    pub fn alf4c(
        cmt: Option<&Lut3d>,
        cdl: Option<&Cdl>,
        sdr: Option<&Lut3d>,
        hdr: Option<&Lut3d>,
        note: &str,
    ) -> Self {
        Self::Alf4c(Alf4c {
            note: note.to_owned(),
            params: cdl_params(cdl),
            cmt: cmt.map(|l| LutBlock::from_lut3d(l, LOGC4_COLORSPACE, "cmt.cube")),
            sdr: sdr.map(|l| LutBlock::from_lut3d(l, REC709_COLORSPACE, "sdr.cube")),
            hdr: hdr.map(|l| {
                LutBlock::from_lut3d(l, "com.apple.proapps.colorspace.rec2100.d65.pq", "hdr.cube")
            }),
        })
    }

    /// An `.aml` for a Rec.709 target, with the parameters of ARRI's library files. ART CMD
    /// resamples every cube to 33 points; this stores `lut` as given, which readers accept at any
    /// size from 2 to 129.
    #[must_use]
    pub fn aml(lut: &Lut3d, cdl: Option<&Cdl>, name: &str) -> Self {
        let block = LutBlock::from_lut3d(lut, REC709_COLORSPACE, name);
        Self::Aml(Aml {
            name: name.to_owned(),
            name_field_len: u16::try_from(name.len()).unwrap_or(u16::MAX),
            colorspace: REC709_COLORSPACE.to_owned(),
            params: aml_params(cdl),
            size: block.size,
            payload: block.payload,
        })
    }

    /// The bytes of the file: the size and the CRC-32 are computed, everything else is written as
    /// held, so a file read with [`parse_file`] is rebuilt byte for byte.
    ///
    /// # Errors
    ///
    /// [`AlfError::FieldTooLong`] for a string longer than its place in the layout.
    pub fn to_bytes(&self) -> Result<Vec<u8>, AlfError> {
        match self {
            Self::Alf4(f) => write_alf4(f),
            Self::Alf4c(f) => write_alf4c(f),
            Self::Aml(f) => write_aml(f),
        }
    }
}

fn write_alf4(f: &Alf4) -> Result<Vec<u8>, AlfError> {
    let table = param_bytes(&f.params);
    let block = 0x500 + table.len();
    let mut out = vec![0_u8; block + 0x100];
    out[0..4].copy_from_slice(&ALF4_MAGIC);
    put_u32(&mut out, 4, ALF4_VERSION);
    put_len(&mut out, 0x10, f.note.len())?;
    put_u32(&mut out, 0x14, 0x500);
    put_len(&mut out, 0x18, f.params.len())?;
    put_text(&mut out, 0x1C, &f.type_id, NAME_FIELD, "type id")?;
    put_len(&mut out, 0x5C, block)?;
    put_u32(&mut out, 0x64, 1);
    put_len(&mut out, 0x68, block)?;
    put_text(&mut out, 0x100, &f.note, ALF4_NOTE_FIELD, "note")?;
    out[0x500..0x500 + table.len()].copy_from_slice(&table);
    let l = &f.lut;
    put_u16(
        &mut out,
        block,
        u16::try_from(l.colorspace.len()).map_err(|_| AlfError::FieldTooLong("colour-space id"))?,
    );
    put_text(
        &mut out,
        block + 2,
        &l.colorspace,
        NAME_FIELD,
        "colour-space id",
    )?;
    put_u16(
        &mut out,
        block + 0x42,
        u16::try_from(l.size)
            .map_err(|_| AlfError::LutSize(u32::try_from(l.size).unwrap_or(u32::MAX)))?,
    );
    put_u32(&mut out, block + 0x44, l.scale);
    put_u16(
        &mut out,
        block + 0x48,
        u16::try_from(l.source_name.len()).map_err(|_| AlfError::FieldTooLong("source name"))?,
    );
    put_text(
        &mut out,
        block + 0x4A,
        &l.source_name,
        NAME_FIELD,
        "source name",
    )?;
    put_len(&mut out, block + 0x8A, block + 0x100)?;
    out.extend_from_slice(&l.payload);
    let len = out.len();
    put_len(&mut out, 8, len)?;
    finish_crc(&mut out, 0x0C, 0x10);
    Ok(out)
}

fn write_alf4c(f: &Alf4c) -> Result<Vec<u8>, AlfError> {
    let table = param_bytes(&f.params);
    let mut pos = 0x100 + f.note.len() + table.len();
    let mut head = vec![0_u8; pos];
    let mut body: Vec<u8> = Vec::new();
    let slots = [&f.cmt, &f.sdr, &f.hdr];
    let mut present = 0_u32;
    for (i, slot) in slots.into_iter().enumerate() {
        let Some(l) = slot else { continue };
        present += 1;
        let desc = pos;
        let mut d = vec![0_u8; 0x100];
        put_len(&mut d, 0, l.colorspace.len())?;
        put_text(&mut d, 4, &l.colorspace, NAME_FIELD, "colour-space id")?;
        put_len(&mut d, 0x44, l.size)?;
        put_u32(&mut d, 0x48, l.scale);
        put_len(&mut d, 0x4C, l.source_name.len())?;
        put_text(&mut d, 0x50, &l.source_name, NAME_FIELD, "source name")?;
        put_text(
            &mut d,
            0x90,
            &l.design_key,
            DESIGN_KEY_FIELD,
            "design-data key",
        )?;
        body.extend_from_slice(&d);
        body.extend_from_slice(&l.payload);
        let data = desc + 0x100;
        pos = data + l.payload.len();
        let (extra, extra_len) = if l.extra.is_empty() {
            (0, 0)
        } else {
            let at = pos;
            body.extend_from_slice(&l.extra);
            pos += l.extra.len();
            (at, l.extra.len())
        };
        let at = 0x20 + 20 * i;
        put_len(&mut head, at, desc)?;
        put_len(&mut head, at + 4, data)?;
        put_len(&mut head, at + 8, l.payload.len())?;
        put_len(&mut head, at + 12, extra)?;
        put_len(&mut head, at + 16, extra_len)?;
    }
    head[0..4].copy_from_slice(&ALF4C_MAGIC);
    put_u32(&mut head, 4, ALF4C_VERSION);
    put_len(&mut head, 0x10, f.note.len())?;
    put_len(&mut head, 0x14, 0x100 + f.note.len())?;
    put_len(&mut head, 0x18, f.params.len())?;
    put_u32(&mut head, 0x1C, present);
    head[0x100..0x100 + f.note.len()].copy_from_slice(f.note.as_bytes());
    head[0x100 + f.note.len()..].copy_from_slice(&table);
    head.extend_from_slice(&body);
    let len = head.len();
    put_len(&mut head, 8, len)?;
    finish_crc(&mut head, 0x0C, 0x10);
    Ok(head)
}

fn write_aml(f: &Aml) -> Result<Vec<u8>, AlfError> {
    let table = param_bytes(&f.params);
    let lut_at = 0x580 + table.len();
    let mut out = vec![0_u8; lut_at];
    out[0..4].copy_from_slice(&AML_MAGIC);
    put_u32(&mut out, 4, AML_VERSION);
    put_u32(&mut out, 0x0C, 0x580);
    put_len(&mut out, 0x10, f.params.len())?;
    put_len(&mut out, 0x14, lut_at)?;
    put_len(&mut out, 0x18, f.size)?;
    put_u16(&mut out, 0x1E, f.name_field_len);
    put_u16(
        &mut out,
        0x22,
        u16::try_from(f.colorspace.len()).map_err(|_| AlfError::FieldTooLong("colour-space id"))?,
    );
    put_text(&mut out, 0x100, &f.name, AML_NAME_FIELD, "name")?;
    put_text(
        &mut out,
        0x540,
        &f.colorspace,
        NAME_FIELD,
        "colour-space id",
    )?;
    out[0x580..0x580 + table.len()].copy_from_slice(&table);
    out.extend_from_slice(&f.payload);
    finish_crc(&mut out, 8, 0x0C);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc32_matches_the_standard_check_value() {
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
        assert_eq!(crc32(b""), 0);
    }

    #[test]
    fn codes_round_half_up() {
        assert_eq!(encode_code(0.0), 0);
        assert_eq!(encode_code(1.0), 65535);
        assert_eq!(encode_code(0.5), 32768);
        assert_eq!(encode_code(-0.2), 0);
        assert_eq!(encode_code(1.7), 65535);
    }

    #[test]
    fn display_labels_read_as_words() {
        assert_eq!(
            display_label("com.apple.proapps.colorspace.rec2100.d65.pq"),
            "a Rec.2020 PQ display"
        );
        assert_eq!(
            display_label("com.apple.proapps.colorspace.rec2100.d65.hlg"),
            "a Rec.2020 HLG display"
        );
        assert_eq!(
            display_label("com.apple.proapps.colorspace.rec2020.d65.gamma24"),
            "a Rec.2020 display"
        );
        assert_eq!(
            display_label("com.apple.proapps.colorspace.p3.dci.gamma26"),
            "a P3 display"
        );
        assert_eq!(display_label(""), "an unknown display");
        assert!(display_label("x.y").contains("x.y"));
    }
}
