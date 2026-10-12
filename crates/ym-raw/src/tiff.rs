//! Little- and big-endian TIFF/IFD walker over a bounded byte source: IFD0, SubIFDs, EXIF and GPS directories, strip and tile tags (`docs/04-raw-decoding.md` section Container).
//!
//! The walker never reads a whole file. It asks a [`ByteSource`] for the few bytes it needs (the
//! header, the directories, the value of one tag) through positional reads, so a metadata probe of
//! a 100 MB file touches a few hundred kilobytes at most. It is written to survive hostile input:
//! directory loops, offsets outside the file, absurd counts and truncated files produce errors or
//! skipped directories, never panics or unbounded allocations. The limits are the `MAX_*` and
//! [`READ_BUDGET`] constants below.

use std::cell::Cell;
use std::collections::HashSet;
use std::fs::File;
use std::io;
use std::path::Path;
use std::rc::Rc;

use crate::error::RawError;

/// The most entries read from one directory. Real Sony directories hold fewer than 200.
pub const MAX_IFD_ENTRIES: usize = 2048;
/// The largest value read into memory for one tag (8 MiB). The SR2 block and the Sony maker note
/// are far smaller; pixel data and previews are never read through this path.
pub const MAX_VALUE_BYTES: u64 = 8 << 20;
/// The most directories visited in one walk.
pub const MAX_IFDS: usize = 64;
/// The deepest directory nesting followed (IFD0, SubIFD, and their children).
pub const MAX_DEPTH: u8 = 4;
/// The most top-level directories followed through the "next IFD" chain.
pub const MAX_ROOT_IFDS: usize = 8;
/// The most bytes one [`Tiff`] reads through [`Tiff::read_vec`] over its lifetime (64 MiB).
/// Bounds the work done on a file whose directories all point at big values.
pub const READ_BUDGET: u64 = 64 << 20;
/// The most array elements read from one tag by the typed readers.
pub const MAX_ARRAY_ELEMENTS: usize = 1 << 20;

/// TIFF field types.
pub mod field_type {
    /// 8-bit unsigned.
    pub const BYTE: u16 = 1;
    /// NUL-terminated text.
    pub const ASCII: u16 = 2;
    /// 16-bit unsigned.
    pub const SHORT: u16 = 3;
    /// 32-bit unsigned.
    pub const LONG: u16 = 4;
    /// Two 32-bit unsigned: numerator, denominator.
    pub const RATIONAL: u16 = 5;
    /// 8-bit signed.
    pub const SBYTE: u16 = 6;
    /// Uninterpreted bytes.
    pub const UNDEFINED: u16 = 7;
    /// 16-bit signed.
    pub const SSHORT: u16 = 8;
    /// 32-bit signed.
    pub const SLONG: u16 = 9;
    /// Two 32-bit signed: numerator, denominator.
    pub const SRATIONAL: u16 = 10;
    /// 32-bit IEEE float.
    pub const FLOAT: u16 = 11;
    /// 64-bit IEEE float.
    pub const DOUBLE: u16 = 12;
    /// 32-bit directory offset.
    pub const IFD: u16 = 13;
}

/// Well-known TIFF, EXIF and GPS tag numbers used by the container parser.
pub mod tag {
    /// `NewSubfileType`.
    pub const NEW_SUBFILE_TYPE: u16 = 0x00FE;
    /// `ImageWidth`.
    pub const IMAGE_WIDTH: u16 = 0x0100;
    /// `ImageLength`.
    pub const IMAGE_LENGTH: u16 = 0x0101;
    /// `BitsPerSample`.
    pub const BITS_PER_SAMPLE: u16 = 0x0102;
    /// `Compression`.
    pub const COMPRESSION: u16 = 0x0103;
    /// `PhotometricInterpretation`.
    pub const PHOTOMETRIC: u16 = 0x0106;
    /// `Make`.
    pub const MAKE: u16 = 0x010F;
    /// `Model`.
    pub const MODEL: u16 = 0x0110;
    /// `StripOffsets`.
    pub const STRIP_OFFSETS: u16 = 0x0111;
    /// `Orientation`.
    pub const ORIENTATION: u16 = 0x0112;
    /// `SamplesPerPixel`.
    pub const SAMPLES_PER_PIXEL: u16 = 0x0115;
    /// `RowsPerStrip`.
    pub const ROWS_PER_STRIP: u16 = 0x0116;
    /// `StripByteCounts`.
    pub const STRIP_BYTE_COUNTS: u16 = 0x0117;
    /// `Software`.
    pub const SOFTWARE: u16 = 0x0131;
    /// `DateTime`.
    pub const DATE_TIME: u16 = 0x0132;
    /// `TileWidth`.
    pub const TILE_WIDTH: u16 = 0x0142;
    /// `TileLength`.
    pub const TILE_LENGTH: u16 = 0x0143;
    /// `TileOffsets`.
    pub const TILE_OFFSETS: u16 = 0x0144;
    /// `TileByteCounts`.
    pub const TILE_BYTE_COUNTS: u16 = 0x0145;
    /// `SubIFDs`.
    pub const SUB_IFDS: u16 = 0x014A;
    /// `JPEGInterchangeFormat` (thumbnail offset).
    pub const JPEG_INTERCHANGE_FORMAT: u16 = 0x0201;
    /// `JPEGInterchangeFormatLength`.
    pub const JPEG_INTERCHANGE_FORMAT_LENGTH: u16 = 0x0202;
    /// `YCbCrSubSampling`.
    pub const YCBCR_SUBSAMPLING: u16 = 0x0212;
    /// `CFARepeatPatternDim`.
    pub const CFA_REPEAT_PATTERN_DIM: u16 = 0x828D;
    /// `CFAPattern`.
    pub const CFA_PATTERN: u16 = 0x828E;
    /// `ExposureTime`.
    pub const EXPOSURE_TIME: u16 = 0x829A;
    /// `FNumber`.
    pub const F_NUMBER: u16 = 0x829D;
    /// `ExifIFD` pointer.
    pub const EXIF_IFD: u16 = 0x8769;
    /// `GPSInfo` (GPS IFD pointer).
    pub const GPS_IFD: u16 = 0x8825;
    /// `ISOSpeedRatings`.
    pub const ISO_SPEED_RATINGS: u16 = 0x8827;
    /// `SensitivityType`.
    pub const SENSITIVITY_TYPE: u16 = 0x8830;
    /// `StandardOutputSensitivity`.
    pub const STANDARD_OUTPUT_SENSITIVITY: u16 = 0x8831;
    /// `RecommendedExposureIndex`.
    pub const RECOMMENDED_EXPOSURE_INDEX: u16 = 0x8832;
    /// `ISOSpeed`.
    pub const ISO_SPEED: u16 = 0x8833;
    /// `DateTimeOriginal`.
    pub const DATE_TIME_ORIGINAL: u16 = 0x9003;
    /// `OffsetTimeOriginal`.
    pub const OFFSET_TIME_ORIGINAL: u16 = 0x9011;
    /// `OffsetTime`.
    pub const OFFSET_TIME: u16 = 0x9010;
    /// `ExposureBiasValue`.
    pub const EXPOSURE_BIAS_VALUE: u16 = 0x9204;
    /// `FocalLength`.
    pub const FOCAL_LENGTH: u16 = 0x920A;
    /// `MakerNote`.
    pub const MAKER_NOTE: u16 = 0x927C;
    /// `SubSecTimeOriginal`.
    pub const SUB_SEC_TIME_ORIGINAL: u16 = 0x9291;
    /// `FocalLengthIn35mmFilm`.
    pub const FOCAL_LENGTH_35MM: u16 = 0xA405;
    /// `LensMake`.
    pub const LENS_MAKE: u16 = 0xA433;
    /// `LensModel`.
    pub const LENS_MODEL: u16 = 0xA434;
    /// `DNGVersion`.
    pub const DNG_VERSION: u16 = 0xC612;
    /// `DNGPrivateData`: in an ARW, four bytes holding the offset of the SR2Private directory.
    pub const DNG_PRIVATE_DATA: u16 = 0xC634;
    /// `DefaultCropOrigin`.
    pub const DEFAULT_CROP_ORIGIN: u16 = 0xC61F;
    /// `DefaultCropSize`.
    pub const DEFAULT_CROP_SIZE: u16 = 0xC620;
}

/// Byte order of a TIFF structure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ByteOrder {
    /// `II`, Intel: least significant byte first. Every ARW uses this.
    Little,
    /// `MM`, Motorola: most significant byte first.
    Big,
}

impl ByteOrder {
    /// Reads a `u16` at `at`, or `None` when `bytes` is too short.
    pub fn u16(self, bytes: &[u8], at: usize) -> Option<u16> {
        let raw: [u8; 2] = bytes.get(at..at.checked_add(2)?)?.try_into().ok()?;
        Some(match self {
            Self::Little => u16::from_le_bytes(raw),
            Self::Big => u16::from_be_bytes(raw),
        })
    }

    /// Reads a `u32` at `at`, or `None` when `bytes` is too short.
    pub fn u32(self, bytes: &[u8], at: usize) -> Option<u32> {
        let raw: [u8; 4] = bytes.get(at..at.checked_add(4)?)?.try_into().ok()?;
        Some(match self {
            Self::Little => u32::from_le_bytes(raw),
            Self::Big => u32::from_be_bytes(raw),
        })
    }

    /// Reads a `u64` at `at`, or `None` when `bytes` is too short.
    pub fn u64(self, bytes: &[u8], at: usize) -> Option<u64> {
        let raw: [u8; 8] = bytes.get(at..at.checked_add(8)?)?.try_into().ok()?;
        Some(match self {
            Self::Little => u64::from_le_bytes(raw),
            Self::Big => u64::from_be_bytes(raw),
        })
    }
}

/// Random-access read-only bytes: a file, or a slice in memory.
///
/// Sources are `Sync` so the decoders can read tiles from several threads at once (positional
/// reads on a file and reads from a slice need no locking).
pub trait ByteSource: Sync {
    /// Total length in bytes.
    fn len(&self) -> u64;

    /// True when the source has no bytes.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Fills `buf` with the bytes at `offset`. Fails with [`io::ErrorKind::UnexpectedEof`] when
    /// the range reaches past the end.
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<()>;
}

fn eof() -> io::Error {
    io::Error::from(io::ErrorKind::UnexpectedEof)
}

/// Bytes in memory.
#[derive(Clone, Copy, Debug)]
pub struct MemSource<'a>(pub &'a [u8]);

impl ByteSource for MemSource<'_> {
    fn len(&self) -> u64 {
        self.0.len() as u64
    }

    fn read_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        let start = usize::try_from(offset).map_err(|_| eof())?;
        let end = start.checked_add(buf.len()).ok_or_else(eof)?;
        buf.copy_from_slice(self.0.get(start..end).ok_or_else(eof)?);
        Ok(())
    }
}

impl ByteSource for Vec<u8> {
    fn len(&self) -> u64 {
        self.as_slice().len() as u64
    }

    fn read_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        MemSource(self.as_slice()).read_at(offset, buf)
    }
}

/// A file read with positional reads. The file is never read as a whole and never written.
#[derive(Debug)]
pub struct FileSource {
    file: File,
    len: u64,
}

impl FileSource {
    /// Opens `path` read-only.
    pub fn open(path: &Path) -> io::Result<Self> {
        let file = File::open(path)?;
        let len = file.metadata()?.len();
        Ok(Self { file, len })
    }
}

impl ByteSource for FileSource {
    fn len(&self) -> u64 {
        self.len
    }

    fn read_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        let end = offset.checked_add(buf.len() as u64).ok_or_else(eof)?;
        if end > self.len {
            return Err(eof());
        }
        read_exact_at(&self.file, buf, offset)
    }
}

#[cfg(unix)]
fn read_exact_at(file: &File, buf: &mut [u8], offset: u64) -> io::Result<()> {
    use std::os::unix::fs::FileExt;
    file.read_exact_at(buf, offset)
}

#[cfg(windows)]
fn read_exact_at(file: &File, mut buf: &mut [u8], mut offset: u64) -> io::Result<()> {
    use std::os::windows::fs::FileExt;
    while !buf.is_empty() {
        let n = file.seek_read(buf, offset)?;
        if n == 0 {
            return Err(eof());
        }
        offset += n as u64;
        buf = &mut buf[n..];
    }
    Ok(())
}

/// One 12-byte directory entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Entry {
    /// Tag number.
    pub tag: u16,
    /// Field type (see [`field_type`]).
    pub ty: u16,
    /// Number of values (not bytes).
    pub count: u32,
    /// The last four bytes of the entry: the value itself when it fits, otherwise an offset.
    value: [u8; 4],
    order: ByteOrder,
}

impl Entry {
    /// Parses the 12 bytes of a directory entry.
    pub fn parse(bytes: &[u8], order: ByteOrder) -> Option<Self> {
        Some(Self {
            tag: order.u16(bytes, 0)?,
            ty: order.u16(bytes, 2)?,
            count: order.u32(bytes, 4)?,
            value: bytes.get(8..12)?.try_into().ok()?,
            order,
        })
    }

    /// Size in bytes of one value of this entry's type, or `None` for an unknown type.
    pub fn type_size(&self) -> Option<u64> {
        match self.ty {
            field_type::BYTE | field_type::ASCII | field_type::SBYTE | field_type::UNDEFINED => {
                Some(1)
            }
            field_type::SHORT | field_type::SSHORT => Some(2),
            field_type::LONG | field_type::SLONG | field_type::FLOAT | field_type::IFD => Some(4),
            field_type::RATIONAL | field_type::SRATIONAL | field_type::DOUBLE => Some(8),
            _ => None,
        }
    }

    /// Total size of the value in bytes, or `None` for an unknown type.
    pub fn byte_len(&self) -> Option<u64> {
        self.type_size().map(|size| size * u64::from(self.count))
    }

    /// True when the value is stored in the entry itself (four bytes or fewer).
    pub fn is_inline(&self) -> bool {
        self.byte_len().is_some_and(|n| n <= 4)
    }

    /// The four value bytes exactly as stored.
    pub fn raw_value(&self) -> [u8; 4] {
        self.value
    }

    /// The value field read as an offset in the entry's byte order.
    pub fn offset(&self) -> u32 {
        self.order.u32(&self.value, 0).unwrap_or(0)
    }

    /// The first value read as an unsigned integer without touching the source: only for entries
    /// whose value is stored inline and whose type is an integer type. `None` otherwise.
    pub fn inline_uint(&self) -> Option<u32> {
        if self.count == 0 || !self.is_inline() {
            return None;
        }
        match self.ty {
            field_type::BYTE | field_type::UNDEFINED => Some(u32::from(self.value[0])),
            field_type::SHORT => self.order.u16(&self.value, 0).map(u32::from),
            field_type::LONG | field_type::IFD => self.order.u32(&self.value, 0),
            _ => None,
        }
    }
}

/// One parsed image file directory.
#[derive(Clone, Debug)]
pub struct Ifd {
    /// Position of the directory in the source.
    pub offset: u64,
    /// The entries in file order.
    pub entries: Vec<Entry>,
    /// Offset of the next directory in the chain, 0 when there is none.
    pub next: u32,
}

impl Ifd {
    /// The first entry with tag `tag`.
    pub fn get(&self, tag: u16) -> Option<&Entry> {
        self.entries.iter().find(|e| e.tag == tag)
    }
}

/// What a directory is for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IfdKind {
    /// A top-level directory; the number counts from 0 (IFD0, IFD1, ...).
    Root(u8),
    /// A directory reached through `SubIFDs` (0x014A).
    Sub,
    /// The EXIF directory (0x8769).
    Exif,
    /// The GPS directory (0x8825).
    Gps,
}

/// A directory found by [`Tiff::walk`], with how it was reached.
#[derive(Clone, Debug)]
pub struct IfdNode {
    /// The directory.
    pub ifd: Ifd,
    /// What kind of directory it is.
    pub kind: IfdKind,
    /// Index in the walk result of the directory that points to this one, `None` for roots.
    pub parent: Option<usize>,
}

/// A directory reader over a [`ByteSource`]: byte order, value-offset bias and a read budget.
pub struct Tiff<'a> {
    src: &'a dyn ByteSource,
    order: ByteOrder,
    bias: u64,
    budget: Rc<Cell<u64>>,
}

impl<'a> Tiff<'a> {
    /// A reader for directories without a TIFF header (the decrypted SR2 block, the Sony maker
    /// note). `bias` is subtracted from every value offset read from an entry: the decrypted SR2
    /// block stores absolute file offsets but lives in its own buffer, so its bias is the file
    /// offset where the block starts.
    pub fn headerless(src: &'a dyn ByteSource, order: ByteOrder, bias: u64) -> Self {
        Self {
            src,
            order,
            bias,
            budget: Rc::new(Cell::new(READ_BUDGET)),
        }
    }

    /// A reader over the same source in another byte order (the SR2Private directory is always
    /// little endian). The value-offset bias is kept and the read budget is shared, so
    /// [`READ_BUDGET`] bounds everything read through the original and its reordered copies.
    pub fn reordered(&self, order: ByteOrder) -> Tiff<'a> {
        Tiff {
            src: self.src,
            order,
            bias: self.bias,
            budget: Rc::clone(&self.budget),
        }
    }

    /// Reads the 8-byte TIFF header and returns the reader and the offset of IFD0.
    ///
    /// Fails with [`RawError::NotArw`] when the bytes are not a classic TIFF (wrong magic,
    /// BigTIFF, too short) and with [`RawError::Corrupt`] when IFD0 lies outside the file.
    pub fn open(src: &'a dyn ByteSource) -> Result<(Self, u64), RawError> {
        let mut header = [0u8; 8];
        if src.len() < 8 {
            return Err(RawError::NotArw);
        }
        // A real read failure is an I/O error; a file that ends early is simply not a TIFF.
        match src.read_at(0, &mut header) {
            Ok(()) => {}
            Err(error) if error.kind() == io::ErrorKind::UnexpectedEof => {
                return Err(RawError::NotArw);
            }
            Err(error) => return Err(RawError::Io(error)),
        }
        let order = match &header[..2] {
            b"II" => ByteOrder::Little,
            b"MM" => ByteOrder::Big,
            _ => return Err(RawError::NotArw),
        };
        if order.u16(&header, 2) != Some(42) {
            return Err(RawError::NotArw);
        }
        let first = u64::from(order.u32(&header, 4).ok_or(RawError::NotArw)?);
        if first < 8 || first + 2 > src.len() {
            return Err(RawError::corrupt(
                "the first directory lies outside the file",
            ));
        }
        Ok((Self::headerless(src, order, 0), first))
    }

    /// The byte order of the structure.
    pub fn order(&self) -> ByteOrder {
        self.order
    }

    /// Total length of the underlying source.
    pub fn source_len(&self) -> u64 {
        self.src.len()
    }

    /// Reads `len` bytes at `offset`, within the size, per-value and total read limits.
    pub fn read_vec(&self, offset: u64, len: u64) -> Result<Vec<u8>, RawError> {
        if len > MAX_VALUE_BYTES {
            return Err(RawError::corrupt(format!(
                "a value of {len} bytes is larger than the {MAX_VALUE_BYTES} byte limit"
            )));
        }
        let end = offset
            .checked_add(len)
            .ok_or_else(|| RawError::corrupt("an offset overflows"))?;
        if end > self.src.len() {
            return Err(RawError::corrupt("a value lies beyond the end of the file"));
        }
        let remaining = self.budget.get();
        if len > remaining {
            return Err(RawError::corrupt(
                "the file's directories ask for too much data",
            ));
        }
        self.budget.set(remaining - len);
        // `len` is at most 8 MiB here, so the cast and the allocation are bounded.
        let mut buf = vec![0u8; len as usize];
        self.src.read_at(offset, &mut buf)?;
        Ok(buf)
    }

    /// Reads the directory at `offset` (a position in the source).
    ///
    /// Fails when the entry table does not fit in the source, has no entries or more than
    /// [`MAX_IFD_ENTRIES`], or when most entries carry impossible field types (the bytes are not
    /// a directory).
    pub fn read_ifd(&self, offset: u64) -> Result<Ifd, RawError> {
        let mut count_bytes = [0u8; 2];
        self.src
            .read_at(offset, &mut count_bytes)
            .map_err(|_| RawError::corrupt("a directory lies outside the file"))?;
        let count = usize::from(
            self.order
                .u16(&count_bytes, 0)
                .ok_or_else(|| RawError::corrupt("a directory header is cut off"))?,
        );
        if count == 0 || count > MAX_IFD_ENTRIES {
            return Err(RawError::corrupt(format!(
                "a directory claims {count} entries"
            )));
        }
        let table = self.read_vec(offset + 2, count as u64 * 12)?;
        let mut entries = Vec::with_capacity(count);
        let mut bad_types = 0usize;
        for chunk in table.chunks_exact(12) {
            let Some(entry) = Entry::parse(chunk, self.order) else {
                continue;
            };
            if entry.type_size().is_none() {
                bad_types += 1;
            }
            entries.push(entry);
        }
        if bad_types * 2 > entries.len() {
            return Err(RawError::corrupt("a directory holds invalid entries"));
        }
        let mut next_bytes = [0u8; 4];
        let next = match self
            .src
            .read_at(offset + 2 + count as u64 * 12, &mut next_bytes)
        {
            Ok(()) => self.order.u32(&next_bytes, 0).unwrap_or(0),
            Err(_) => 0,
        };
        Ok(Ifd {
            offset,
            entries,
            next,
        })
    }

    /// Translates the offset field of an entry into a position in the source.
    fn value_position(&self, entry: &Entry) -> Result<u64, RawError> {
        u64::from(entry.offset())
            .checked_sub(self.bias)
            .ok_or_else(|| RawError::corrupt("a value offset points before its directory"))
    }

    /// The first `max_bytes` bytes of an entry's value (or all of it when shorter).
    pub fn entry_prefix(&self, entry: &Entry, max_bytes: u64) -> Result<Vec<u8>, RawError> {
        let total = entry
            .byte_len()
            .ok_or_else(|| RawError::corrupt("a tag has an unknown field type"))?;
        let wanted = total.min(max_bytes);
        if total <= 4 {
            // `wanted <= 4`, so the cast cannot truncate.
            return Ok(entry.value[..wanted as usize].to_vec());
        }
        self.read_vec(self.value_position(entry)?, wanted)
    }

    /// The whole value of an entry as bytes (at most [`MAX_VALUE_BYTES`]).
    pub fn entry_bytes(&self, entry: &Entry) -> Result<Vec<u8>, RawError> {
        self.entry_prefix(entry, MAX_VALUE_BYTES)
    }

    /// The position and size in the source of an entry's value, for values that are read later
    /// in pieces (the Sony maker note, JPEG previews). Inline values have no position.
    pub fn entry_location(&self, entry: &Entry) -> Result<Option<(u64, u64)>, RawError> {
        let total = entry
            .byte_len()
            .ok_or_else(|| RawError::corrupt("a tag has an unknown field type"))?;
        if total <= 4 {
            return Ok(None);
        }
        Ok(Some((self.value_position(entry)?, total)))
    }

    /// Reads up to `max` leading values of an integer tag (BYTE, SHORT, LONG, IFD, UNDEFINED) as
    /// unsigned numbers. Signed types are rejected; use [`Tiff::ints`] for those.
    pub fn uints(&self, entry: &Entry, max: usize) -> Result<Vec<u64>, RawError> {
        let size = match entry.ty {
            field_type::BYTE | field_type::UNDEFINED => 1,
            field_type::SHORT => 2,
            field_type::LONG | field_type::IFD => 4,
            _ => return Err(RawError::corrupt("a tag is not an unsigned integer")),
        };
        let n = (entry.count as usize).min(max.min(MAX_ARRAY_ELEMENTS));
        let bytes = self.entry_prefix(entry, n as u64 * size)?;
        let order = self.order;
        Ok(bytes
            .chunks_exact(size as usize)
            .map(|c| match size {
                1 => u64::from(c[0]),
                2 => u64::from(order.u16(c, 0).unwrap_or(0)),
                _ => u64::from(order.u32(c, 0).unwrap_or(0)),
            })
            .collect())
    }

    /// Reads up to `max` leading values of an integer tag (any BYTE, SHORT or LONG type, signed
    /// or not) as signed numbers.
    pub fn ints(&self, entry: &Entry, max: usize) -> Result<Vec<i64>, RawError> {
        let (size, signed) = match entry.ty {
            field_type::BYTE | field_type::UNDEFINED => (1, false),
            field_type::SBYTE => (1, true),
            field_type::SHORT => (2, false),
            field_type::SSHORT => (2, true),
            field_type::LONG | field_type::IFD => (4, false),
            field_type::SLONG => (4, true),
            _ => return Err(RawError::corrupt("a tag is not an integer")),
        };
        let n = (entry.count as usize).min(max.min(MAX_ARRAY_ELEMENTS));
        let bytes = self.entry_prefix(entry, n as u64 * size)?;
        let order = self.order;
        Ok(bytes
            .chunks_exact(size as usize)
            .map(|c| match (size, signed) {
                (1, false) => i64::from(c[0]),
                (1, true) => i64::from(c[0] as i8),
                (2, false) => i64::from(order.u16(c, 0).unwrap_or(0)),
                (2, true) => i64::from(order.u16(c, 0).unwrap_or(0) as i16),
                (_, false) => i64::from(order.u32(c, 0).unwrap_or(0)),
                (_, true) => i64::from(order.u32(c, 0).unwrap_or(0) as i32),
            })
            .collect())
    }

    /// Reads up to `max` leading values of any numeric tag as floating point numbers. Rationals
    /// become `numerator / denominator`; a zero denominator gives NaN.
    pub fn floats(&self, entry: &Entry, max: usize) -> Result<Vec<f64>, RawError> {
        match entry.ty {
            field_type::RATIONAL | field_type::SRATIONAL => {
                let n = (entry.count as usize).min(max.min(MAX_ARRAY_ELEMENTS));
                let bytes = self.entry_prefix(entry, n as u64 * 8)?;
                let order = self.order;
                let signed = entry.ty == field_type::SRATIONAL;
                Ok(bytes
                    .chunks_exact(8)
                    .map(|c| {
                        let a = order.u32(c, 0).unwrap_or(0);
                        let b = order.u32(c, 4).unwrap_or(0);
                        let (num, den) = if signed {
                            (f64::from(a as i32), f64::from(b as i32))
                        } else {
                            (f64::from(a), f64::from(b))
                        };
                        if den == 0.0 { f64::NAN } else { num / den }
                    })
                    .collect())
            }
            field_type::FLOAT => {
                let n = (entry.count as usize).min(max.min(MAX_ARRAY_ELEMENTS));
                let bytes = self.entry_prefix(entry, n as u64 * 4)?;
                let order = self.order;
                Ok(bytes
                    .chunks_exact(4)
                    .map(|c| f64::from(f32::from_bits(order.u32(c, 0).unwrap_or(0))))
                    .collect())
            }
            field_type::DOUBLE => {
                let n = (entry.count as usize).min(max.min(MAX_ARRAY_ELEMENTS));
                let bytes = self.entry_prefix(entry, n as u64 * 8)?;
                let order = self.order;
                Ok(bytes
                    .chunks_exact(8)
                    .map(|c| f64::from_bits(order.u64(c, 0).unwrap_or(0)))
                    .collect())
            }
            _ => Ok(self
                .ints(entry, max)?
                .into_iter()
                .map(|v| v as f64)
                .collect()),
        }
    }

    /// Reads a text tag: the bytes up to the first NUL, as UTF-8 (lossy), with surrounding
    /// whitespace removed. Values longer than 4 KiB are cut.
    pub fn ascii(&self, entry: &Entry) -> Result<String, RawError> {
        if !matches!(
            entry.ty,
            field_type::ASCII | field_type::BYTE | field_type::UNDEFINED
        ) {
            return Err(RawError::corrupt("a tag is not text"));
        }
        let bytes = self.entry_prefix(entry, 4096)?;
        let end = bytes.iter().position(|b| *b == 0).unwrap_or(bytes.len());
        Ok(String::from_utf8_lossy(&bytes[..end]).trim().to_owned())
    }

    /// Walks IFD0, its "next IFD" chain and every SubIFD, EXIF and GPS directory below them.
    ///
    /// IFD0 must parse (its error is returned). Everything else is best effort: a directory that
    /// is out of range, malformed or already visited is skipped. At most [`MAX_IFDS`] directories
    /// are read and nesting stops at [`MAX_DEPTH`].
    pub fn walk(&self, first_ifd: u64) -> Result<Vec<IfdNode>, RawError> {
        let mut nodes: Vec<(IfdNode, u8)> = Vec::new();
        let mut visited: HashSet<u64> = HashSet::new();

        let ifd0 = self.read_ifd(first_ifd)?;
        visited.insert(first_ifd);
        let mut next = ifd0.next;
        nodes.push((
            IfdNode {
                ifd: ifd0,
                kind: IfdKind::Root(0),
                parent: None,
            },
            0,
        ));
        // The chain of top-level directories (IFD1 holds the thumbnail).
        let mut root_index = 1u8;
        while next != 0 && usize::from(root_index) < MAX_ROOT_IFDS && nodes.len() < MAX_IFDS {
            let offset = u64::from(next);
            if !visited.insert(offset) {
                break;
            }
            let Ok(ifd) = self.read_ifd(offset) else {
                break;
            };
            next = ifd.next;
            nodes.push((
                IfdNode {
                    ifd,
                    kind: IfdKind::Root(root_index),
                    parent: None,
                },
                0,
            ));
            root_index += 1;
        }

        // Children, breadth first. `nodes` grows while we iterate by index.
        let mut index = 0;
        while index < nodes.len() {
            let depth = nodes[index].1;
            if depth < MAX_DEPTH {
                let children = [
                    (tag::SUB_IFDS, IfdKind::Sub),
                    (tag::EXIF_IFD, IfdKind::Exif),
                    (tag::GPS_IFD, IfdKind::Gps),
                ];
                for (child_tag, kind) in children {
                    let Some(entry) = nodes[index].0.ifd.get(child_tag).copied() else {
                        continue;
                    };
                    let Ok(offsets) = self.uints(&entry, 64) else {
                        continue;
                    };
                    for offset in offsets {
                        if nodes.len() >= MAX_IFDS {
                            break;
                        }
                        if offset < 8 || !visited.insert(offset) {
                            continue;
                        }
                        let Ok(ifd) = self.read_ifd(offset) else {
                            continue;
                        };
                        nodes.push((
                            IfdNode {
                                ifd,
                                kind,
                                parent: Some(index),
                            },
                            depth + 1,
                        ));
                    }
                }
            }
            index += 1;
        }
        Ok(nodes.into_iter().map(|(node, _)| node).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tiny TIFF builder for the walker tests: little-endian, entries given as raw tuples.
    struct Raw(Vec<u8>);

    impl Raw {
        fn header(first: u32) -> Self {
            let mut v = b"II*\0".to_vec();
            v.extend_from_slice(&first.to_le_bytes());
            Self(v)
        }

        fn pad_to(&mut self, len: usize) {
            self.0.resize(len, 0);
        }

        fn ifd(&mut self, entries: &[(u16, u16, u32, [u8; 4])], next: u32) {
            self.0
                .extend_from_slice(&(entries.len() as u16).to_le_bytes());
            for (tag, ty, count, value) in entries {
                self.0.extend_from_slice(&tag.to_le_bytes());
                self.0.extend_from_slice(&ty.to_le_bytes());
                self.0.extend_from_slice(&count.to_le_bytes());
                self.0.extend_from_slice(value);
            }
            self.0.extend_from_slice(&next.to_le_bytes());
        }
    }

    fn long(v: u32) -> [u8; 4] {
        v.to_le_bytes()
    }

    #[test]
    fn open_rejects_non_tiff_and_bigtiff_as_not_arw() {
        for bytes in [
            &b""[..],
            &b"II*"[..],
            &b"\xff\xd8\xff\xe0\0\x10JFIF\0"[..],
            &b"II+\0\x08\0\0\0\0\0\0\0\0\0\0\0"[..],
            &b"XX*\0\x08\0\0\0\0\0"[..],
        ] {
            let source = MemSource(bytes);
            let result = Tiff::open(&source);
            assert!(matches!(result, Err(RawError::NotArw)), "{bytes:?}");
        }
    }

    #[test]
    fn open_rejects_a_first_directory_outside_the_file() {
        let bytes = Raw::header(4096).0;
        assert!(matches!(Tiff::open(&bytes), Err(RawError::Corrupt { .. })));
        // Offset below the header is as bad.
        let bytes = Raw::header(4).0;
        assert!(matches!(Tiff::open(&bytes), Err(RawError::Corrupt { .. })));
    }

    #[test]
    fn reads_inline_and_offset_values_in_both_byte_orders() {
        // Little endian: SHORT x2 inline, ASCII "SONY\0" by offset, RATIONAL by offset.
        let mut raw = Raw::header(8);
        let table_end = 8 + 2 + 3 * 12 + 4;
        raw.ifd(
            &[
                (0x0001, field_type::SHORT, 2, [1, 0, 2, 0]),
                (0x010F, field_type::ASCII, 5, long(table_end as u32)),
                (0x829A, field_type::RATIONAL, 1, long(table_end as u32 + 8)),
            ],
            0,
        );
        raw.0.extend_from_slice(b"SONY\0\0\0\0");
        raw.0.extend_from_slice(&1u32.to_le_bytes());
        raw.0.extend_from_slice(&250u32.to_le_bytes());
        let (tiff, first) = Tiff::open(&raw.0).expect("header");
        let ifd = tiff.read_ifd(first).expect("ifd");
        assert_eq!(tiff.uints(ifd.get(1).expect("tag 1"), 8).unwrap(), [1, 2]);
        assert_eq!(tiff.ascii(ifd.get(0x010F).expect("make")).unwrap(), "SONY");
        let t = tiff.floats(ifd.get(0x829A).expect("time"), 4).unwrap();
        assert_eq!(t, [1.0 / 250.0]);

        // Big endian: one SHORT inline, left justified.
        let mut be = b"MM\0*\0\0\0\x08".to_vec();
        be.extend_from_slice(&1u16.to_be_bytes());
        be.extend_from_slice(&0x0112u16.to_be_bytes());
        be.extend_from_slice(&3u16.to_be_bytes());
        be.extend_from_slice(&1u32.to_be_bytes());
        be.extend_from_slice(&[0, 6, 0, 0]);
        be.extend_from_slice(&0u32.to_be_bytes());
        let (tiff, first) = Tiff::open(&be).expect("header");
        assert_eq!(tiff.order(), ByteOrder::Big);
        let ifd = tiff.read_ifd(first).expect("ifd");
        assert_eq!(
            tiff.uints(ifd.get(0x0112).expect("orientation"), 1)
                .unwrap(),
            [6]
        );
    }

    #[test]
    fn signed_and_float_readers_decode_their_types() {
        let mut raw = Raw::header(8);
        let base = 8 + 2 + 3 * 12 + 4;
        raw.ifd(
            &[
                (1, field_type::SSHORT, 3, long(base as u32)),
                (2, field_type::SRATIONAL, 1, long(base as u32 + 8)),
                (3, field_type::SBYTE, 2, [0xFF, 0x02, 0, 0]),
            ],
            0,
        );
        raw.pad_to(base);
        for v in [-5i16, 7, -32768, 0] {
            raw.0.extend_from_slice(&v.to_le_bytes());
        }
        raw.0.extend_from_slice(&(-1i32).to_le_bytes());
        raw.0.extend_from_slice(&3i32.to_le_bytes());
        let (tiff, first) = Tiff::open(&raw.0).expect("header");
        let ifd = tiff.read_ifd(first).expect("ifd");
        assert_eq!(
            tiff.ints(ifd.get(1).expect("t1"), 16).unwrap(),
            [-5, 7, -32768]
        );
        let r = tiff.floats(ifd.get(2).expect("t2"), 4).unwrap();
        assert!((r[0] + 1.0 / 3.0).abs() < 1e-12);
        assert_eq!(tiff.ints(ifd.get(3).expect("t3"), 4).unwrap(), [-1, 2]);
        // Unsigned reader refuses signed types.
        assert!(tiff.uints(ifd.get(1).expect("t1"), 4).is_err());
    }

    #[test]
    fn zero_denominator_rational_is_nan() {
        let mut raw = Raw::header(8);
        let base = 8 + 2 + 12 + 4;
        raw.ifd(&[(1, field_type::RATIONAL, 1, long(base as u32))], 0);
        raw.0.extend_from_slice(&5u32.to_le_bytes());
        raw.0.extend_from_slice(&0u32.to_le_bytes());
        let (tiff, first) = Tiff::open(&raw.0).expect("header");
        let ifd = tiff.read_ifd(first).expect("ifd");
        assert!(tiff.floats(ifd.get(1).expect("t1"), 1).unwrap()[0].is_nan());
    }

    #[test]
    fn values_outside_the_file_are_errors_not_panics() {
        let mut raw = Raw::header(8);
        raw.ifd(
            &[
                (1, field_type::LONG, 4, long(1_000_000)),
                (2, field_type::BYTE, 0xFFFF_FFFF, long(8)),
                (3, field_type::LONG, u32::MAX, long(u32::MAX)),
            ],
            0,
        );
        let (tiff, first) = Tiff::open(&raw.0).expect("header");
        let ifd = tiff.read_ifd(first).expect("ifd");
        for tag in [1, 2, 3] {
            let result = tiff.uints(ifd.get(tag).expect("tag"), 1 << 20);
            assert!(matches!(result, Err(RawError::Corrupt { .. })), "tag {tag}");
        }
    }

    #[test]
    fn absurd_entry_counts_do_not_allocate() {
        // A directory claiming 65535 entries in a 30-byte file.
        let mut bytes = Raw::header(8).0;
        bytes.extend_from_slice(&u16::MAX.to_le_bytes());
        bytes.resize(30, 0);
        let (tiff, first) = Tiff::open(&bytes).expect("header");
        assert!(tiff.read_ifd(first).is_err());
        // Zero entries is rejected too.
        let mut bytes = Raw::header(8).0;
        bytes.extend_from_slice(&[0, 0, 0, 0, 0, 0]);
        let (tiff, first) = Tiff::open(&bytes).expect("header");
        assert!(tiff.read_ifd(first).is_err());
    }

    #[test]
    fn walk_follows_chain_subifds_exif_and_gps() {
        // IFD0 -> next IFD1; IFD0 has SubIFDs (2 offsets), Exif and GPS pointers.
        let ifd0_at = 8usize;
        let ifd0_len = 2 + 4 * 12 + 4;
        let ifd1_at = ifd0_at + ifd0_len;
        let ifd1_len = 2 + 12 + 4;
        let sub_a = ifd1_at + ifd1_len;
        let sub_b = sub_a + 18;
        let exif = sub_b + 18;
        let gps = exif + 18;
        let offsets_at = gps + 18;
        let mut raw = Raw::header(8);
        raw.ifd(
            &[
                (tag::SUB_IFDS, field_type::LONG, 2, long(offsets_at as u32)),
                (tag::EXIF_IFD, field_type::LONG, 1, long(exif as u32)),
                (tag::GPS_IFD, field_type::LONG, 1, long(gps as u32)),
                (tag::MAKE, field_type::BYTE, 4, *b"SONY"),
            ],
            ifd1_at as u32,
        );
        raw.ifd(&[(tag::COMPRESSION, field_type::SHORT, 1, [6, 0, 0, 0])], 0);
        for _ in 0..4 {
            raw.ifd(&[(tag::IMAGE_WIDTH, field_type::LONG, 1, long(1))], 0);
        }
        raw.0.extend_from_slice(&(sub_a as u32).to_le_bytes());
        raw.0.extend_from_slice(&(sub_b as u32).to_le_bytes());
        let (tiff, first) = Tiff::open(&raw.0).expect("header");
        let nodes = tiff.walk(first).expect("walk");
        let kinds: Vec<_> = nodes.iter().map(|n| n.kind).collect();
        assert_eq!(
            kinds,
            [
                IfdKind::Root(0),
                IfdKind::Root(1),
                IfdKind::Sub,
                IfdKind::Sub,
                IfdKind::Exif,
                IfdKind::Gps
            ]
        );
        assert_eq!(nodes[2].parent, Some(0));
    }

    #[test]
    fn walk_survives_directory_loops() {
        // IFD0 points to itself as next IFD, as SubIFD and as Exif IFD.
        let mut raw = Raw::header(8);
        raw.ifd(
            &[
                (tag::SUB_IFDS, field_type::LONG, 1, long(8)),
                (tag::EXIF_IFD, field_type::LONG, 1, long(8)),
            ],
            8,
        );
        let (tiff, first) = Tiff::open(&raw.0).expect("header");
        let nodes = tiff.walk(first).expect("walk");
        assert_eq!(nodes.len(), 1);
    }

    #[test]
    fn walk_skips_bad_children_but_requires_ifd0() {
        let mut raw = Raw::header(8);
        raw.ifd(
            &[
                (tag::SUB_IFDS, field_type::LONG, 1, long(5_000_000)),
                (tag::EXIF_IFD, field_type::LONG, 1, long(2)),
            ],
            9_999_999,
        );
        let (tiff, first) = Tiff::open(&raw.0).expect("header");
        assert_eq!(tiff.walk(first).expect("walk").len(), 1);
        // A bogus IFD0 is an error.
        assert!(tiff.walk(9_000).is_err());
    }

    #[test]
    fn walk_stops_at_the_directory_and_depth_limits() {
        // A chain of SubIFDs 10 deep: each directory points at the next.
        let mut raw = Raw::header(8);
        let len = 2 + 12 + 4;
        for i in 0..10u32 {
            let next_at = 8 + (i + 1) * len as u32;
            raw.ifd(&[(tag::SUB_IFDS, field_type::LONG, 1, long(next_at))], 0);
        }
        let (tiff, first) = Tiff::open(&raw.0).expect("header");
        let nodes = tiff.walk(first).expect("walk");
        assert!(nodes.len() <= usize::from(MAX_DEPTH) + 1, "{}", nodes.len());
    }

    #[test]
    fn read_budget_bounds_total_reads() {
        let bytes = vec![0u8; 1 << 20];
        let tiff = Tiff::headerless(&bytes, ByteOrder::Little, 0);
        let mut total = 0u64;
        let mut failed = false;
        for _ in 0..200 {
            match tiff.read_vec(0, 1 << 20) {
                Ok(_) => total += 1 << 20,
                Err(_) => {
                    failed = true;
                    break;
                }
            }
        }
        assert!(failed, "budget never ran out");
        assert!(total <= READ_BUDGET);
    }

    #[test]
    fn reordered_readers_share_one_read_budget() {
        let bytes = vec![0u8; 1 << 20];
        let tiff = Tiff::headerless(&bytes, ByteOrder::Little, 0);
        let other = tiff.reordered(ByteOrder::Big);
        let mut total = 0u64;
        // Alternate between the two readers: together they still get only READ_BUDGET bytes.
        for i in 0..200 {
            let reader = if i % 2 == 0 { &tiff } else { &other };
            match reader.read_vec(0, 1 << 20) {
                Ok(_) => total += 1 << 20,
                Err(_) => break,
            }
        }
        assert!(total <= READ_BUDGET, "{total} bytes read");
        assert!(other.read_vec(0, 1 << 20).is_err());
        assert!(tiff.read_vec(0, 1 << 20).is_err());
    }

    #[test]
    fn read_vec_rejects_oversized_and_out_of_range_reads() {
        let bytes = vec![0u8; 100];
        let tiff = Tiff::headerless(&bytes, ByteOrder::Little, 0);
        assert!(tiff.read_vec(90, 20).is_err());
        assert!(tiff.read_vec(u64::MAX, 2).is_err());
        assert!(tiff.read_vec(0, MAX_VALUE_BYTES + 1).is_err());
        assert_eq!(tiff.read_vec(10, 5).expect("in range").len(), 5);
    }

    #[test]
    fn bias_translates_absolute_offsets_into_the_buffer() {
        // A headerless directory whose value offset is an absolute file position (1000 + 18).
        let mut buf = Vec::new();
        buf.extend_from_slice(&1u16.to_le_bytes());
        buf.extend_from_slice(&0x7313u16.to_le_bytes());
        buf.extend_from_slice(&field_type::SHORT.to_le_bytes());
        buf.extend_from_slice(&4u32.to_le_bytes());
        buf.extend_from_slice(&1018u32.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        for v in [2000u16, 1024, 1024, 1700] {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        let tiff = Tiff::headerless(&buf, ByteOrder::Little, 1000);
        let ifd = tiff.read_ifd(0).expect("ifd");
        let values = tiff.uints(ifd.get(0x7313).expect("wb"), 8).unwrap();
        assert_eq!(values, [2000, 1024, 1024, 1700]);
        // A bias larger than the offset is an error.
        let tiff = Tiff::headerless(&buf, ByteOrder::Little, 5000);
        let ifd = tiff.read_ifd(0).expect("ifd");
        assert!(tiff.uints(ifd.get(0x7313).expect("wb"), 8).is_err());
    }

    #[test]
    fn ascii_stops_at_nul_and_trims() {
        let mut raw = Raw::header(8);
        let base = 8 + 2 + 12 + 4;
        raw.ifd(&[(1, field_type::ASCII, 12, long(base as u32))], 0);
        raw.0.extend_from_slice(b"  ILCE-1\0xyz ");
        let (tiff, first) = Tiff::open(&raw.0).expect("header");
        let ifd = tiff.read_ifd(first).expect("ifd");
        assert_eq!(tiff.ascii(ifd.get(1).expect("t1")).unwrap(), "ILCE-1");
    }

    #[test]
    fn float_and_double_values_decode_in_both_byte_orders() {
        for order in [ByteOrder::Little, ByteOrder::Big] {
            let put32 = |v: u32| match order {
                ByteOrder::Little => v.to_le_bytes(),
                ByteOrder::Big => v.to_be_bytes(),
            };
            let put64 = |v: u64| match order {
                ByteOrder::Little => v.to_le_bytes(),
                ByteOrder::Big => v.to_be_bytes(),
            };
            let put16 = |v: u16| match order {
                ByteOrder::Little => v.to_le_bytes(),
                ByteOrder::Big => v.to_be_bytes(),
            };
            let mut bytes = match order {
                ByteOrder::Little => b"II*\0".to_vec(),
                ByteOrder::Big => b"MM\0*".to_vec(),
            };
            bytes.extend_from_slice(&put32(8));
            bytes.extend_from_slice(&put16(2));
            let data_at = 8 + 2 + 2 * 12 + 4;
            for (tag, ty, count, value) in [
                (1u16, field_type::FLOAT, 1u32, put32(1.5f32.to_bits())),
                (2, field_type::DOUBLE, 1, put32(data_at as u32)),
            ] {
                bytes.extend_from_slice(&put16(tag));
                bytes.extend_from_slice(&put16(ty));
                bytes.extend_from_slice(&put32(count));
                bytes.extend_from_slice(&value);
            }
            bytes.extend_from_slice(&put32(0));
            bytes.extend_from_slice(&put64((-2.25f64).to_bits()));
            let (tiff, first) = Tiff::open(&bytes).expect("header");
            let ifd = tiff.read_ifd(first).expect("ifd");
            assert_eq!(tiff.floats(ifd.get(1).expect("float"), 4).unwrap(), [1.5]);
            assert_eq!(
                tiff.floats(ifd.get(2).expect("double"), 4).unwrap(),
                [-2.25]
            );
            assert_eq!(order.u64(&put64(7), 0), Some(7));
            assert_eq!(order.u64(&[0; 7], 0), None);
        }
    }

    #[test]
    fn directories_full_of_impossible_types_are_not_directories() {
        let mut raw = Raw::header(8);
        raw.ifd(
            &[
                (1, 0, 1, [0; 4]),
                (2, 99, 1, [0; 4]),
                (3, 200, 1, [0; 4]),
                (4, field_type::SHORT, 1, [1, 0, 0, 0]),
            ],
            0,
        );
        let (tiff, first) = Tiff::open(&raw.0).expect("header");
        assert!(matches!(
            tiff.read_ifd(first),
            Err(RawError::Corrupt { .. })
        ));

        // One odd entry among good ones is tolerated and kept.
        let mut raw = Raw::header(8);
        raw.ifd(
            &[
                (1, 99, 1, [0; 4]),
                (2, field_type::SHORT, 1, [1, 0, 0, 0]),
                (3, field_type::SHORT, 1, [2, 0, 0, 0]),
            ],
            0,
        );
        let (tiff, first) = Tiff::open(&raw.0).expect("header");
        let ifd = tiff.read_ifd(first).expect("ifd");
        assert_eq!(ifd.entries.len(), 3);
        assert_eq!(ifd.get(1).expect("odd entry").byte_len(), None);
        // Reading a value of an unknown type is an error, not a guess.
        assert!(tiff.uints(ifd.get(1).expect("odd entry"), 1).is_err());
        assert!(tiff.entry_bytes(ifd.get(1).expect("odd entry")).is_err());
    }

    #[test]
    fn entry_location_is_none_for_inline_values_and_text_readers_reject_numbers() {
        let mut raw = Raw::header(8);
        let base = 8 + 2 + 2 * 12 + 4;
        raw.ifd(
            &[
                (1, field_type::SHORT, 1, [5, 0, 0, 0]),
                (2, field_type::LONG, 4, long(base as u32)),
            ],
            0,
        );
        raw.0.extend_from_slice(&[0u8; 16]);
        let (tiff, first) = Tiff::open(&raw.0).expect("header");
        let ifd = tiff.read_ifd(first).expect("ifd");
        assert_eq!(tiff.entry_location(ifd.get(1).expect("t1")).unwrap(), None);
        assert_eq!(
            tiff.entry_location(ifd.get(2).expect("t2")).unwrap(),
            Some((base as u64, 16))
        );
        assert!(tiff.ascii(ifd.get(1).expect("t1")).is_err());
        assert!(tiff.floats(ifd.get(2).expect("t2"), 4).is_ok());
    }

    #[test]
    fn the_top_level_chain_is_limited() {
        // Twelve top-level directories in a row; only MAX_ROOT_IFDS are followed.
        let len = 2 + 12 + 4;
        let mut raw = Raw::header(8);
        for i in 0..12u32 {
            let next = if i == 11 { 0 } else { 8 + (i + 1) * len as u32 };
            raw.ifd(
                &[(tag::COMPRESSION, field_type::SHORT, 1, [6, 0, 0, 0])],
                next,
            );
        }
        let (tiff, first) = Tiff::open(&raw.0).expect("header");
        let nodes = tiff.walk(first).expect("walk");
        assert_eq!(nodes.len(), MAX_ROOT_IFDS);
        assert_eq!(nodes[7].kind, IfdKind::Root(7));
    }

    #[test]
    fn open_reports_real_read_errors_as_io_not_not_arw() {
        struct Failing;
        impl ByteSource for Failing {
            fn len(&self) -> u64 {
                100
            }
            fn read_at(&self, _offset: u64, _buf: &mut [u8]) -> io::Result<()> {
                Err(io::Error::from(io::ErrorKind::PermissionDenied))
            }
        }
        assert!(matches!(Tiff::open(&Failing), Err(RawError::Io(_))));
    }

    #[test]
    fn file_source_reads_positionally_and_reports_eof() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("f.bin");
        std::fs::write(&path, b"0123456789").expect("write");
        let source = FileSource::open(&path).expect("open");
        assert_eq!(ByteSource::len(&source), 10);
        let mut buf = [0u8; 3];
        source.read_at(4, &mut buf).expect("read");
        assert_eq!(&buf, b"456");
        let error = source.read_at(8, &mut buf).expect_err("past the end");
        assert_eq!(error.kind(), io::ErrorKind::UnexpectedEof);
        assert!(FileSource::open(&dir.path().join("missing")).is_err());
    }

    #[test]
    fn inline_uint_only_answers_for_inline_integers() {
        let short = Entry::parse(
            &[1, 0, 3, 0, 1, 0, 0, 0, 0x34, 0x12, 0, 0],
            ByteOrder::Little,
        )
        .expect("entry");
        assert_eq!(short.inline_uint(), Some(0x1234));
        let big =
            Entry::parse(&[1, 0, 3, 0, 9, 0, 0, 0, 0, 0, 0, 0], ByteOrder::Little).expect("entry");
        assert_eq!(big.inline_uint(), None);
        let text = Entry::parse(&[1, 0, 2, 0, 2, 0, 0, 0, b'A', 0, 0, 0], ByteOrder::Little)
            .expect("entry");
        assert_eq!(text.inline_uint(), None);
    }
}
