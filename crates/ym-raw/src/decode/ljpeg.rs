//! Our own lossless JPEG decoder (ITU T.81 process 14, "LJPEG") and the Sony Lossless L tile layout (`docs/04-raw-decoding.md` section Lossless L).
//!
//! # The codec
//!
//! [`decode`] reads one lossless JPEG stream (`SOI`, `DHT`, `DRI`, `SOF3`, `SOS`, entropy-coded
//! data, `EOI`): 2 to 16 bits of precision, one to four components with any sampling factors from
//! 1 to 4, the seven predictors of T.81 table H.1, a point transform, Huffman tables of any shape
//! (codes up to 16 bits) and restart intervals (`DRI` and `RSTn`). It returns one plane per
//! component at the component's own resolution. Streams with several scans, arithmetic coding,
//! hierarchical or lossy processes are refused.
//!
//! Prediction follows T.81 section H.1.2.1 per component plane: the first sample of the scan and of
//! every restart interval is predicted from `2^(P - Pt - 1)`, the rest of the first line from the
//! sample on the left, the first sample of later lines from the sample above, everything else with
//! the selected predictor. Differences wrap modulo 2^16.
//!
//! # Sony's 4:2:0 variant
//!
//! Lossless M and S tiles are ordinary 3-component streams with one exception that LibRaw and
//! rawler both implement (rawler calls it a probably broken Sony encoding): in 4:2:0 the first luma
//! sample of an MCU row, which has a row above it, is predicted from the first sample of the
//! *previous MCU row* (two rows up), not from the row directly above. [`Options::sony_420_quirk`]
//! turns that on; the other three luma samples of the MCU and both chroma samples follow T.81.
//!
//! # Lossless L tiles
//!
//! Each 512 x 512 tile of a Lossless L file is a stream that declares four components, half the tile
//! width and half its height. A decoded row of four-component "pixels" holds two consecutive sensor
//! rows of the 2 x 2 quad: component 0 is the sensor sample at `(2j, 2i)`, component 1 at
//! `(2j, 2i + 1)`, component 2 at `(2j + 1, 2i)` and component 3 at `(2j + 1, 2i + 1)`
//! ([`decode_lossless_l`]). Tiles are independent, so they are decoded in parallel.
//!
//! The decoder allocates only after the header has been checked against [`Options::max_samples`],
//! never indexes outside its planes whatever the input says, and stops at the end of the data:
//! corrupt streams give [`RawError::Corrupt`], never a panic.

use rayon::prelude::*;

use crate::error::RawError;
use crate::layout::{Plane, check_dimensions};
use crate::metadata::{RawMetadata, RawStorage};
use crate::tiff::ByteSource;

/// Most components a stream may declare (T.81 allows 4 in a scan).
pub const MAX_COMPONENTS: usize = 4;

/// Largest number of samples one stream may decode to by default (64 Mi samples = 128 MiB).
pub const DEFAULT_MAX_SAMPLES: usize = 1 << 26;

/// Largest tile (in sensor samples) a tiled file may declare: Sony writes 512 x 512 = 262 144.
pub const MAX_TILE_SAMPLES: usize = 1 << 24;

/// Bits of the fast Huffman lookup table.
const LOOKAHEAD: u32 = 10;

/// Decoder switches.
#[derive(Clone, Copy, Debug)]
pub struct Options {
    /// Refuse streams that would decode to more than this many samples (all components).
    pub max_samples: usize,
    /// Apply Sony's 4:2:0 first-column prediction (see the module documentation).
    pub sony_420_quirk: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            max_samples: DEFAULT_MAX_SAMPLES,
            sony_420_quirk: false,
        }
    }
}

/// One component of a frame header.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ComponentInfo {
    /// Component identifier (`Ci`).
    pub id: u8,
    /// Horizontal sampling factor (`Hi`).
    pub h: u8,
    /// Vertical sampling factor (`Vi`).
    pub v: u8,
    /// Index of the Huffman table this component's differences use (`Td` of the scan header).
    pub table: u8,
}

/// The parsed headers of a stream: everything up to the first byte of entropy-coded data.
#[derive(Clone, Debug)]
pub struct Header {
    /// Sample precision `P` in bits.
    pub precision: u8,
    /// Frame width `X` in luma samples.
    pub width: usize,
    /// Frame height `Y` in lines.
    pub height: usize,
    /// The components in frame order, which is also scan order.
    pub components: Vec<ComponentInfo>,
    /// Predictor selection `Ss` (1 to 7).
    pub predictor: u8,
    /// Point transform `Pt` (`Al`).
    pub point_transform: u8,
    /// Restart interval in MCUs (0 = none).
    pub restart_interval: usize,
    scan_start: usize,
    tables: [Option<HuffTable>; 4],
}

impl Header {
    /// Largest horizontal sampling factor.
    pub fn h_max(&self) -> usize {
        self.components
            .iter()
            .map(|c| usize::from(c.h))
            .max()
            .unwrap_or(1)
    }

    /// Largest vertical sampling factor.
    pub fn v_max(&self) -> usize {
        self.components
            .iter()
            .map(|c| usize::from(c.v))
            .max()
            .unwrap_or(1)
    }

    /// MCUs per row.
    pub fn mcus_x(&self) -> usize {
        self.width.div_ceil(self.h_max())
    }

    /// MCU rows.
    pub fn mcus_y(&self) -> usize {
        self.height.div_ceil(self.v_max())
    }
}

/// One decoded component at its own resolution (padded to whole MCUs).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ComponentPlane {
    /// Horizontal sampling factor.
    pub h: u8,
    /// Vertical sampling factor.
    pub v: u8,
    /// Samples per row (`mcus_x * h`).
    pub width: usize,
    /// Rows (`mcus_y * v`).
    pub height: usize,
    /// `width * height` samples, with the point transform already applied.
    pub data: Vec<u16>,
}

/// A decoded stream.
#[derive(Clone, Debug)]
pub struct Decoded {
    /// Sample precision `P`.
    pub precision: u8,
    /// Frame width `X`.
    pub width: usize,
    /// Frame height `Y`.
    pub height: usize,
    /// One plane per component, in frame order.
    pub planes: Vec<ComponentPlane>,
}

// ---------------------------------------------------------------------------------------------
// Huffman tables
// ---------------------------------------------------------------------------------------------

/// A Huffman table for difference categories 0 to 16.
#[derive(Clone, Debug)]
struct HuffTable {
    /// `(length << 8) | symbol` for codes up to `LOOKAHEAD` bits, indexed by the next
    /// `LOOKAHEAD` bits; 0 marks "longer code".
    lookup: Vec<u16>,
    /// Largest code of each length (index 1 to 16), or -1 when no code has that length.
    max_code: [i32; 17],
    /// First code of each length.
    first_code: [i32; 17],
    /// Index into `symbols` of the first code of each length.
    first_index: [usize; 17],
    symbols: Vec<u8>,
}

impl HuffTable {
    fn build(counts: &[u8; 16], symbols: &[u8]) -> Result<Self, RawError> {
        let total: usize = counts.iter().map(|c| usize::from(*c)).sum();
        if total == 0 || total > 256 || symbols.len() != total {
            return Err(RawError::corrupt(
                "a lossless JPEG Huffman table has a bad symbol count",
            ));
        }
        if symbols.iter().any(|s| *s > 16) {
            return Err(RawError::corrupt(
                "a lossless JPEG Huffman table has a difference category above 16",
            ));
        }
        let mut lookup = vec![0u16; 1 << LOOKAHEAD];
        let mut max_code = [-1i32; 17];
        let mut first_code = [0i32; 17];
        let mut first_index = [0usize; 17];
        let mut code = 0i32;
        let mut index = 0usize;
        for length in 1..=16usize {
            let count = i32::from(counts[length - 1]);
            first_code[length] = code;
            first_index[length] = index;
            if code + count > (1 << length) {
                return Err(RawError::corrupt(
                    "a lossless JPEG Huffman table has more codes than its lengths allow",
                ));
            }
            for k in 0..count {
                if length as u32 <= LOOKAHEAD {
                    let symbol = u16::from(symbols[index + k as usize]);
                    let shift = LOOKAHEAD - length as u32;
                    let base = ((code + k) as usize) << shift;
                    for entry in &mut lookup[base..base + (1usize << shift)] {
                        *entry = ((length as u16) << 8) | symbol;
                    }
                }
            }
            if count > 0 {
                max_code[length] = code + count - 1;
            }
            code = (code + count) << 1;
            index += count as usize;
        }
        Ok(Self {
            lookup,
            max_code,
            first_code,
            first_index,
            symbols: symbols.to_vec(),
        })
    }
}

// ---------------------------------------------------------------------------------------------
// Header parsing
// ---------------------------------------------------------------------------------------------

fn be16(data: &[u8], at: usize) -> Option<usize> {
    Some(usize::from(u16::from_be_bytes([
        *data.get(at)?,
        *data.get(at + 1)?,
    ])))
}

/// A frame header as read: precision, width, height and `(id, h, v)` of each component.
type FrameHeader = (u8, usize, usize, Vec<(u8, u8, u8)>);

/// Parses the markers of a lossless JPEG stream up to and including the start-of-scan header.
pub fn parse_header(data: &[u8]) -> Result<Header, RawError> {
    let bad = |what: &str| RawError::corrupt(format!("lossless JPEG: {what}"));
    if data.get(..2) != Some(&[0xFF, 0xD8]) {
        return Err(bad("the stream does not start with a JPEG marker"));
    }
    let mut pos = 2usize;
    let mut frame: Option<FrameHeader> = None;
    let mut tables: [Option<HuffTable>; 4] = [None, None, None, None];
    let mut restart_interval = 0usize;
    loop {
        // Find the next marker: 0xFF, then a non-0xFF byte (fill bytes are skipped).
        while *data.get(pos).ok_or_else(|| bad("the headers end early"))? != 0xFF {
            pos += 1;
        }
        while *data.get(pos).ok_or_else(|| bad("the headers end early"))? == 0xFF {
            pos += 1;
        }
        let marker = *data.get(pos).ok_or_else(|| bad("the headers end early"))?;
        pos += 1;
        match marker {
            0x00 | 0x01 | 0xD0..=0xD8 => continue,
            0xD9 => return Err(bad("end of image before the scan")),
            _ => {}
        }
        let length = be16(data, pos).ok_or_else(|| bad("a marker is cut off"))?;
        if length < 2 || pos + length > data.len() {
            return Err(bad("a marker segment is cut off"));
        }
        let body = &data[pos + 2..pos + length];
        match marker {
            0xC3 => {
                if frame.is_some() {
                    return Err(bad("two frame headers"));
                }
                if body.len() < 6 {
                    return Err(bad("the frame header is too short"));
                }
                let precision = body[0];
                let height = be16(body, 1).unwrap_or(0);
                let width = be16(body, 3).unwrap_or(0);
                let count = usize::from(body[5]);
                if !(2..=16).contains(&precision) {
                    return Err(bad("the sample precision is not between 2 and 16 bits"));
                }
                if width == 0 || height == 0 {
                    return Err(bad("the frame has no pixels (a zero width or height)"));
                }
                if !(1..=MAX_COMPONENTS).contains(&count) || body.len() < 6 + 3 * count {
                    return Err(bad("the frame header has a bad component count"));
                }
                let mut components = Vec::with_capacity(count);
                for k in 0..count {
                    let id = body[6 + 3 * k];
                    let sampling = body[7 + 3 * k];
                    let (h, v) = (sampling >> 4, sampling & 0x0F);
                    if !(1..=4).contains(&h) || !(1..=4).contains(&v) {
                        return Err(bad("a sampling factor is not between 1 and 4"));
                    }
                    if components.iter().any(|(cid, _, _)| *cid == id) {
                        return Err(bad("two components share an identifier"));
                    }
                    components.push((id, h, v));
                }
                frame = Some((precision, width, height, components));
            }
            0xC0..=0xCF if !matches!(marker, 0xC4 | 0xC8 | 0xCC) => {
                return Err(bad(
                    "the stream uses a coding process other than lossless Huffman (SOF3)",
                ));
            }
            0xC4 => {
                let mut at = 0usize;
                while at < body.len() {
                    let tc_th = body[at];
                    let (class, id) = (tc_th >> 4, usize::from(tc_th & 0x0F));
                    let counts: [u8; 16] = body
                        .get(at + 1..at + 17)
                        .and_then(|s| s.try_into().ok())
                        .ok_or_else(|| bad("a Huffman table is cut off"))?;
                    let total: usize = counts.iter().map(|c| usize::from(*c)).sum();
                    let symbols = body
                        .get(at + 17..at + 17 + total)
                        .ok_or_else(|| bad("a Huffman table is cut off"))?;
                    at += 17 + total;
                    if class == 0 && id < 4 {
                        tables[id] = Some(HuffTable::build(&counts, symbols)?);
                    } else if class > 1 || id > 3 {
                        return Err(bad("a Huffman table has a bad class or number"));
                    }
                }
            }
            0xDD => {
                restart_interval =
                    be16(body, 0).ok_or_else(|| bad("the restart interval is cut off"))?;
            }
            0xDA => {
                let (precision, width, height, comps) = frame
                    .take()
                    .ok_or_else(|| bad("the scan comes before the frame header"))?;
                let ns = usize::from(*body.first().ok_or_else(|| bad("the scan is cut off"))?);
                if ns != comps.len() {
                    return Err(bad(
                        "the stream has several scans or a partial scan, which is not supported",
                    ));
                }
                if body.len() < 4 + 2 * ns {
                    return Err(bad("the scan header is cut off"));
                }
                let mut components = Vec::with_capacity(ns);
                for k in 0..ns {
                    let id = body[1 + 2 * k];
                    let td = body[2 + 2 * k] >> 4;
                    let (_, h, v) = comps
                        .iter()
                        .copied()
                        .find(|(cid, _, _)| *cid == id)
                        .ok_or_else(|| bad("the scan names an unknown component"))?;
                    if td > 3 {
                        return Err(bad("the scan names a bad Huffman table"));
                    }
                    if tables[usize::from(td)].is_none() {
                        return Err(bad("the scan uses a Huffman table that was never defined"));
                    }
                    components.push(ComponentInfo {
                        id,
                        h,
                        v,
                        table: td,
                    });
                }
                let predictor = body[1 + 2 * ns];
                let end = body[2 + 2 * ns];
                let point_transform = body[3 + 2 * ns] & 0x0F;
                if !(1..=7).contains(&predictor) || end != 0 {
                    return Err(bad("the scan header is not that of a lossless scan"));
                }
                if point_transform >= precision {
                    return Err(bad("the point transform is not below the precision"));
                }
                return Ok(Header {
                    precision,
                    width,
                    height,
                    components,
                    predictor,
                    point_transform,
                    restart_interval,
                    scan_start: pos + length,
                    tables,
                });
            }
            _ => {} // APPn, COM, DQT and friends carry nothing the decoder needs.
        }
        pos += length;
    }
}

// ---------------------------------------------------------------------------------------------
// Entropy-coded data
// ---------------------------------------------------------------------------------------------

/// The entropy-coded segments of a scan with byte stuffing removed. `bytes` ends with 16 zero
/// bytes so the bit reader never needs a bounds check near the end of a segment.
struct Scan {
    bytes: Vec<u8>,
    /// `(start, end)` of each restart interval in `bytes`.
    segments: Vec<(usize, usize)>,
}

const SCAN_PADDING: usize = 16;

fn split_scan(data: &[u8]) -> Scan {
    let mut bytes = Vec::with_capacity(data.len() + SCAN_PADDING);
    let mut segments = Vec::new();
    let mut seg_start = 0usize;
    let mut pos = 0usize;
    loop {
        // Copy the run up to the next 0xFF.
        let run = data[pos..].iter().position(|b| *b == 0xFF);
        let Some(run) = run else {
            bytes.extend_from_slice(&data[pos..]);
            segments.push((seg_start, bytes.len()));
            break;
        };
        bytes.extend_from_slice(&data[pos..pos + run]);
        pos += run;
        // Skip fill bytes (a run of 0xFF) to find what follows.
        let mut next = pos + 1;
        while data.get(next) == Some(&0xFF) {
            next += 1;
        }
        match data.get(next) {
            Some(0x00) => {
                bytes.push(0xFF);
                pos = next + 1;
            }
            Some(0xD0..=0xD7) => {
                segments.push((seg_start, bytes.len()));
                seg_start = bytes.len();
                pos = next + 1;
            }
            // EOI, any other marker, or the end of the data.
            _ => {
                segments.push((seg_start, bytes.len()));
                break;
            }
        }
    }
    bytes.resize(bytes.len() + SCAN_PADDING, 0);
    Scan { bytes, segments }
}

/// MSB-first bit reader over unstuffed bytes. Reads past the end return zero bits; the caller
/// checks [`BitReader::overrun`] once a segment is done.
struct BitReader<'a> {
    data: &'a [u8],
    pos: usize,
    bits: u64,
    nbits: u32,
    error: bool,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            pos: 0,
            bits: 0,
            nbits: 0,
            error: false,
        }
    }

    #[inline(always)]
    fn refill(&mut self) {
        if let Some(chunk) = self.data.get(self.pos..self.pos + 8) {
            let mut word = [0u8; 8];
            word.copy_from_slice(chunk);
            let word = u64::from_be_bytes(word);
            self.bits |= word >> self.nbits;
            let advance = (63 - self.nbits) >> 3;
            self.pos += advance as usize;
            self.nbits += advance * 8;
        } else {
            while self.nbits <= 56 {
                let byte = self.data.get(self.pos).copied().unwrap_or(0);
                self.bits |= u64::from(byte) << (56 - self.nbits);
                self.pos += 1;
                self.nbits += 8;
            }
        }
    }

    /// Bits consumed so far (including zero bits invented past the end).
    fn consumed_bits(&self) -> usize {
        self.pos * 8 - self.nbits as usize
    }

    #[inline(always)]
    fn consume(&mut self, n: u32) {
        self.bits <<= n;
        self.nbits -= n;
    }

    /// Decodes one Huffman symbol (a difference category). Sets `error` for a code that no
    /// table entry matches and returns 0.
    #[inline(always)]
    fn symbol(&mut self, table: &HuffTable) -> u32 {
        if self.nbits < 32 {
            self.refill();
        }
        let entry = table.lookup[(self.bits >> (64 - LOOKAHEAD)) as usize];
        if entry != 0 {
            self.consume(u32::from(entry >> 8));
            return u32::from(entry & 0xFF);
        }
        let peek = (self.bits >> 48) as i32;
        for length in (LOOKAHEAD as usize + 1)..=16 {
            let code = peek >> (16 - length);
            if table.max_code[length] >= 0 && code <= table.max_code[length] {
                let index = table.first_index[length] + (code - table.first_code[length]) as usize;
                self.consume(length as u32);
                return u32::from(table.symbols.get(index).copied().unwrap_or(0));
            }
        }
        self.error = true;
        self.consume(16);
        0
    }

    /// Reads the difference that follows a category symbol (T.81 table H.2 and F.12).
    #[inline(always)]
    fn difference(&mut self, category: u32) -> i32 {
        match category {
            0 => 0,
            16 => 32768,
            n => {
                if self.nbits < 16 {
                    self.refill();
                }
                let value = (self.bits >> (64 - n)) as i32;
                self.consume(n);
                if value < (1 << (n - 1)) {
                    value - (1 << n) + 1
                } else {
                    value
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Decoding
// ---------------------------------------------------------------------------------------------

#[inline(always)]
fn predict(predictor: u8, ra: i32, rb: i32, rc: i32) -> i32 {
    match predictor {
        1 => ra,
        2 => rb,
        3 => rc,
        4 => ra + rb - rc,
        5 => ra + ((rb - rc) >> 1),
        6 => rb + ((ra - rc) >> 1),
        _ => (ra + rb) >> 1,
    }
}

/// Decodes the stream in `data` (a whole lossless JPEG).
pub fn decode(data: &[u8], options: &Options) -> Result<Decoded, RawError> {
    let header = parse_header(data)?;
    decode_scan(&header, data, options)
}

/// Decodes the MCU rows `rows` of one restart interval when every component has sampling 1 x 1
/// (Lossless L, 4:4:4): the common case, written with row slices and a compile-time component
/// count instead of per-sample index arithmetic. Same prediction rules as the general loop in
/// [`decode_scan`].
fn decode_unit_rows(
    reader: &mut BitReader,
    tables: &[&HuffTable],
    planes: &mut [ComponentPlane],
    rows: std::ops::Range<usize>,
    base: i32,
    predictor: u8,
) -> Result<(), RawError> {
    match planes.len() {
        1 => decode_unit_rows_n::<1>(reader, tables, planes, rows, base, predictor),
        2 => decode_unit_rows_n::<2>(reader, tables, planes, rows, base, predictor),
        3 => decode_unit_rows_n::<3>(reader, tables, planes, rows, base, predictor),
        _ => decode_unit_rows_n::<4>(reader, tables, planes, rows, base, predictor),
    }
}

fn decode_unit_rows_n<const N: usize>(
    reader: &mut BitReader,
    tables: &[&HuffTable],
    planes: &mut [ComponentPlane],
    rows: std::ops::Range<usize>,
    base: i32,
    predictor: u8,
) -> Result<(), RawError> {
    let internal = || RawError::corrupt("lossless JPEG: internal component count mismatch");
    let tables: [&HuffTable; N] = tables.try_into().map_err(|_| internal())?;
    let width = planes[0].width;
    let first_row = rows.start;
    for y in rows {
        let lines: Vec<(&[u16], &mut [u16])> = planes
            .iter_mut()
            .map(|plane| {
                let (head, tail) = plane.data.split_at_mut(y * width);
                let head: &[u16] = head;
                let prev = if y > 0 {
                    &head[(y - 1) * width..]
                } else {
                    &[][..]
                };
                (prev, &mut tail[..width])
            })
            .collect();
        let mut lines: [(&[u16], &mut [u16]); N] = lines.try_into().map_err(|_| internal())?;
        // The last sample of every component, as the left neighbour of the next one.
        let mut left = [0i32; N];
        if y == first_row {
            // The first sample of every component is predicted from 2^(P - Pt - 1), the rest of
            // the line from the sample on the left.
            for c in 0..N {
                let category = reader.symbol(tables[c]);
                let v = (base + reader.difference(category)) & 0xFFFF;
                lines[c].1[0] = v as u16;
                left[c] = v;
            }
        } else {
            // The first sample of a later line is predicted from the sample above it.
            for c in 0..N {
                let category = reader.symbol(tables[c]);
                let v = (i32::from(lines[c].0[0]) + reader.difference(category)) & 0xFFFF;
                lines[c].1[0] = v as u16;
                left[c] = v;
            }
        }
        if y == first_row || predictor == 1 {
            for x in 1..width {
                for c in 0..N {
                    let category = reader.symbol(tables[c]);
                    let v = (left[c] + reader.difference(category)) & 0xFFFF;
                    left[c] = v;
                    lines[c].1[x] = v as u16;
                }
            }
        } else {
            for x in 1..width {
                for c in 0..N {
                    let category = reader.symbol(tables[c]);
                    let diff = reader.difference(category);
                    let (prev, cur) = &mut lines[c];
                    let px = predict(
                        predictor,
                        i32::from(cur[x - 1]),
                        i32::from(prev[x]),
                        i32::from(prev[x - 1]),
                    );
                    cur[x] = (px + diff) as u16;
                }
            }
        }
        if reader.error {
            return Err(RawError::corrupt(
                "lossless JPEG: the data holds a code that is not in the Huffman table",
            ));
        }
    }
    Ok(())
}

/// Decodes the entropy-coded data of an already parsed `header`; `data` is the same buffer that
/// was parsed.
pub fn decode_scan(header: &Header, data: &[u8], options: &Options) -> Result<Decoded, RawError> {
    let corrupt = |what: &str| RawError::corrupt(format!("lossless JPEG: {what}"));
    let (mcus_x, mcus_y) = (header.mcus_x(), header.mcus_y());
    let total_mcus = mcus_x * mcus_y;
    let comps = &header.components;

    // Allocation guard: sizes come from the header.
    let mut samples = 0usize;
    let mut planes: Vec<ComponentPlane> = Vec::with_capacity(comps.len());
    for c in comps {
        let width = mcus_x * usize::from(c.h);
        let height = mcus_y * usize::from(c.v);
        samples = samples.saturating_add(width.saturating_mul(height));
        if samples > options.max_samples {
            return Err(corrupt("the frame is larger than the decoder accepts"));
        }
        planes.push(ComponentPlane {
            h: c.h,
            v: c.v,
            width,
            height,
            data: Vec::new(),
        });
    }
    for plane in &mut planes {
        plane.data = vec![0u16; plane.width * plane.height];
    }

    let scan = split_scan(data.get(header.scan_start..).unwrap_or(&[]));
    let interval = if header.restart_interval == 0 {
        total_mcus
    } else {
        header.restart_interval
    };
    let needed = total_mcus.div_ceil(interval);
    if scan.segments.len() < needed {
        return Err(corrupt(
            "the data has fewer restart intervals than the image needs",
        ));
    }

    let precision = u32::from(header.precision);
    let pt = u32::from(header.point_transform);
    let base_prediction = 1i32 << (precision - pt - 1);
    let predictor = header.predictor;

    let all_tables: Vec<&HuffTable> = comps
        .iter()
        .map(|c| header.tables[usize::from(c.table)].as_ref())
        .collect::<Option<_>>()
        .ok_or_else(|| corrupt("a Huffman table is missing"))?;

    // Streams whose components all have sampling 1 x 1 and whose restart intervals are whole MCU
    // rows (Lossless L, 4:4:4) take a faster path with the same results.
    let unit_tables: Option<&[&HuffTable]> = (comps.iter().all(|c| c.h == 1 && c.v == 1)
        && interval.is_multiple_of(mcus_x))
    .then_some(&all_tables[..]);

    for (index, &(start, end)) in scan.segments.iter().take(needed).enumerate() {
        let mcu_start = index * interval;
        let mcu_end = (mcu_start + interval).min(total_mcus);
        let mut reader = BitReader::new(&scan.bytes[start..]);
        let first_mcu_row = mcu_start / mcus_x;
        if let Some(tables) = unit_tables {
            decode_unit_rows(
                &mut reader,
                tables,
                &mut planes,
                first_mcu_row..mcu_end.div_ceil(mcus_x),
                base_prediction,
                predictor,
            )?;
            if reader.consumed_bits() > (end - start) * 8 {
                return Err(corrupt("the entropy-coded data ends before the image does"));
            }
            continue;
        }
        let mut fresh = [true; MAX_COMPONENTS];
        for mcu in mcu_start..mcu_end {
            let (my, mx) = (mcu / mcus_x, mcu % mcus_x);
            for (ci, (info, table)) in comps.iter().zip(&all_tables).enumerate() {
                let plane = &mut planes[ci];
                let (h, v, pw) = (usize::from(info.h), usize::from(info.v), plane.width);
                let first_line = first_mcu_row * v;
                for j in 0..v {
                    let y = my * v + j;
                    for i in 0..h {
                        let x = mx * h + i;
                        let category = reader.symbol(table);
                        let diff = reader.difference(category);
                        let at = y * pw + x;
                        let px = if fresh[ci] {
                            fresh[ci] = false;
                            base_prediction
                        } else if y == first_line {
                            i32::from(plane.data[at - 1])
                        } else if x == 0 {
                            if options.sony_420_quirk && ci == 0 && v == 2 && j == 0 && y >= 2 {
                                i32::from(plane.data[(y - 2) * pw])
                            } else {
                                i32::from(plane.data[at - pw])
                            }
                        } else {
                            let ra = i32::from(plane.data[at - 1]);
                            let rb = i32::from(plane.data[at - pw]);
                            let rc = i32::from(plane.data[at - pw - 1]);
                            predict(predictor, ra, rb, rc)
                        };
                        plane.data[at] = (px + diff) as u16;
                    }
                }
            }
            if reader.error {
                return Err(corrupt(
                    "the data holds a code that is not in the Huffman table",
                ));
            }
        }
        if reader.consumed_bits() > (end - start) * 8 {
            return Err(corrupt("the entropy-coded data ends before the image does"));
        }
    }

    if pt > 0 {
        for plane in &mut planes {
            for value in &mut plane.data {
                *value <<= pt;
            }
        }
    }
    Ok(Decoded {
        precision: header.precision,
        width: header.width,
        height: header.height,
        planes,
    })
}

// ---------------------------------------------------------------------------------------------
// Sony tiled files
// ---------------------------------------------------------------------------------------------

/// The tile geometry of a tiled raw image, checked against the file size.
#[derive(Clone, Copy, Debug)]
pub struct TileGrid {
    /// Tile width in samples.
    pub tile_width: usize,
    /// Tile height in rows.
    pub tile_length: usize,
    /// Tiles across.
    pub across: usize,
    /// Tiles down.
    pub down: usize,
}

/// Reads the tile geometry from the metadata of a tiled file.
pub fn tile_grid(meta: &RawMetadata) -> Result<(TileGrid, &[u64], &[u64]), RawError> {
    let RawStorage::Tiles {
        tile_width,
        tile_length,
        offsets,
        byte_counts,
    } = &meta.storage
    else {
        return Err(RawError::corrupt(
            "a lossless file must store its raw data in tiles",
        ));
    };
    let (tw, tl) = (*tile_width as usize, *tile_length as usize);
    if tw == 0 || tl == 0 || tw > 16384 || tl > 16384 || tw * tl > MAX_TILE_SAMPLES {
        return Err(RawError::corrupt("the tile size is out of range"));
    }
    let across = (meta.width as usize).div_ceil(tw);
    let down = (meta.height as usize).div_ceil(tl);
    if offsets.len() != across * down || byte_counts.len() != offsets.len() {
        return Err(RawError::corrupt("the tile count does not match the image"));
    }
    Ok((
        TileGrid {
            tile_width: tw,
            tile_length: tl,
            across,
            down,
        },
        offsets,
        byte_counts,
    ))
}

/// Reads one tile's bytes from the file.
pub fn read_tile(
    src: &dyn ByteSource,
    offset: u64,
    count: u64,
    limit: u64,
) -> Result<Vec<u8>, RawError> {
    if count == 0 || count > limit {
        return Err(RawError::corrupt("a tile has a bad length"));
    }
    let end = offset
        .checked_add(count)
        .ok_or_else(|| RawError::corrupt("a tile offset overflows"))?;
    if end > src.len() {
        return Err(RawError::corrupt("a tile lies beyond the end of the file"));
    }
    let mut buf = vec![0u8; count as usize];
    src.read_at(offset, &mut buf)?;
    Ok(buf)
}

/// The most compressed bytes one tile may hold: ten times its raw size is far beyond anything a
/// lossless coder produces.
fn tile_byte_limit(grid: &TileGrid) -> u64 {
    (grid.tile_width * grid.tile_length * 2 * 4) as u64 + (1 << 16)
}

/// Decodes a Sony Lossless L file (tiled, four components at half width and height per tile) into
/// the full sensor mosaic.
pub fn decode_lossless_l(src: &dyn ByteSource, meta: &RawMetadata) -> Result<Plane<u16>, RawError> {
    let (grid, offsets, counts) = tile_grid(meta)?;
    let (width, height) = (meta.width as usize, meta.height as usize);
    check_dimensions(width, height)?;
    if grid.tile_width % 2 != 0 || grid.tile_length % 2 != 0 {
        return Err(RawError::corrupt("the tile size is odd"));
    }
    let mut plane = Plane::<u16>::try_new(width, height)?;
    let limit = tile_byte_limit(&grid);
    let options = Options {
        max_samples: grid.tile_width * grid.tile_length * 2,
        sony_420_quirk: false,
    };

    // One task per tile row; inside it the tiles of the row decode in parallel and are copied into
    // the band of rows they cover.
    plane
        .data
        .par_chunks_mut(width * grid.tile_length)
        .enumerate()
        .try_for_each(|(tile_row, band)| -> Result<(), RawError> {
            let rows_in_band = band.len() / width;
            let tiles: Vec<Vec<u16>> = (0..grid.across)
                .into_par_iter()
                .map(|tile_col| {
                    let k = tile_row * grid.across + tile_col;
                    let bytes = read_tile(src, offsets[k], counts[k], limit)?;
                    decode_l_tile(&bytes, &grid, &options)
                })
                .collect::<Result<_, _>>()?;
            for (tile_col, tile) in tiles.iter().enumerate() {
                let x0 = tile_col * grid.tile_width;
                let copy = grid.tile_width.min(width - x0);
                for row in 0..rows_in_band {
                    band[row * width + x0..row * width + x0 + copy].copy_from_slice(
                        &tile[row * grid.tile_width..row * grid.tile_width + copy],
                    );
                }
            }
            Ok(())
        })?;
    Ok(plane)
}

/// Decodes one Lossless L tile into `tile_length` rows of `tile_width` sensor samples.
pub fn decode_l_tile(
    bytes: &[u8],
    grid: &TileGrid,
    options: &Options,
) -> Result<Vec<u16>, RawError> {
    let (half_w, half_l) = (grid.tile_width / 2, grid.tile_length / 2);
    // Check the declared geometry before anything is allocated.
    let header = parse_header(bytes)?;
    if header.components.len() != 4
        || header.width != half_w
        || header.height != half_l
        || header.components.iter().any(|c| c.h != 1 || c.v != 1)
    {
        return Err(RawError::corrupt(
            "a Lossless L tile does not declare four components at half the tile size",
        ));
    }
    let decoded = decode_scan(&header, bytes, options)?;
    let [c0, c1, c2, c3] = &decoded.planes[..] else {
        return Err(RawError::corrupt("a Lossless L tile lost a component"));
    };
    let mut out = vec![0u16; grid.tile_width * grid.tile_length];
    for j in 0..half_l {
        let (top, bottom) = out[2 * j * grid.tile_width..(2 * j + 2) * grid.tile_width]
            .split_at_mut(grid.tile_width);
        let src = j * c0.width..j * c0.width + half_w;
        for (i, (((a, b), c), d)) in c0.data[src.clone()]
            .iter()
            .zip(&c1.data[src.clone()])
            .zip(&c2.data[src.clone()])
            .zip(&c3.data[src])
            .enumerate()
        {
            top[2 * i] = *a;
            top[2 * i + 1] = *b;
            bottom[2 * i] = *c;
            bottom[2 * i + 1] = *d;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testgen::SplitMix64;
    use crate::testgen::ljpeg_enc::{EncodeParams, HuffmanChoice, encode};

    /// A smooth plane with noise, `precision` bits wide, whose samples are multiples of
    /// `1 << point_transform`.
    fn test_plane(
        width: usize,
        height: usize,
        precision: u8,
        point_transform: u8,
        seed: u64,
    ) -> Vec<u16> {
        let mut rng = SplitMix64(seed);
        let max = (1u32 << precision) - 1;
        let mut out = Vec::with_capacity(width * height);
        for y in 0..height {
            for x in 0..width {
                let base = (x * 37 + y * 53) as u32 % (max / 2).max(1);
                let noise = rng.below(u64::from(max / 16).max(1)) as u32;
                let value = (max / 4 + base + noise).min(max) as u16;
                out.push(value >> point_transform << point_transform);
            }
        }
        out
    }

    fn planes_for(params: &EncodeParams, seed: u64) -> Vec<Vec<u16>> {
        (0..params.sampling.len())
            .map(|c| {
                let (w, h) = params.plane_size(c);
                test_plane(
                    w,
                    h,
                    params.precision,
                    params.point_transform,
                    seed + c as u64,
                )
            })
            .collect()
    }

    fn round_trip(params: &EncodeParams, seed: u64) {
        let planes = planes_for(params, seed);
        let bytes = encode(&planes, params);
        let decoded = decode(
            &bytes,
            &Options {
                sony_420_quirk: params.sony_420_quirk,
                ..Options::default()
            },
        )
        .unwrap_or_else(|e| panic!("{params:?}: {e}"));
        assert_eq!(decoded.precision, params.precision);
        assert_eq!(
            (decoded.width, decoded.height),
            (params.width, params.height)
        );
        for (c, (got, want)) in decoded.planes.iter().zip(&planes).enumerate() {
            assert_eq!(&got.data, want, "component {c} of {params:?}");
        }
    }

    fn params(
        sampling: &[(u8, u8)],
        predictor: u8,
        precision: u8,
        restart_rows: usize,
    ) -> EncodeParams {
        let mut p = EncodeParams {
            precision,
            predictor,
            point_transform: 0,
            restart_interval: 0,
            sony_420_quirk: false,
            huffman: HuffmanChoice::Optimal,
            width: 37,
            height: 23,
            sampling: sampling.to_vec(),
        };
        p.restart_interval = restart_rows * p.mcus_x();
        p
    }

    #[test]
    fn a_stream_assembled_by_hand_decodes_to_the_samples_it_codes() {
        // Table: symbols 0, 1, 2 with 2-bit codes 00, 01, 10. 2 x 2 samples at 8 bits,
        // predictor 1: 128 129 / 131 130. Differences: 0, +1, +3 (above), -1 (left).
        // Bits: 00 | 01 1 | 10 11 | 01 0 = 0001 1101 1010, padded with ones: 1D AF.
        let mut stream = vec![0xFF, 0xD8];
        stream.extend_from_slice(&[0xFF, 0xC4, 0x00, 22, 0x00]);
        stream.extend_from_slice(&[0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        stream.extend_from_slice(&[0, 1, 2]);
        stream.extend_from_slice(&[0xFF, 0xC3, 0x00, 11, 8, 0, 2, 0, 2, 1, 1, 0x11, 0]);
        stream.extend_from_slice(&[0xFF, 0xDA, 0x00, 8, 1, 1, 0x00, 1, 0, 0]);
        stream.extend_from_slice(&[0x1D, 0xAF, 0xFF, 0xD9]);
        let decoded = decode(&stream, &Options::default()).expect("valid stream");
        assert_eq!(decoded.planes[0].data, vec![128, 129, 131, 130]);
    }

    #[test]
    fn every_predictor_round_trips_for_one_and_four_components() {
        for predictor in 1..=7 {
            round_trip(&params(&[(1, 1)], predictor, 14, 0), 11);
            round_trip(&params(&[(1, 1); 4], predictor, 14, 0), 12);
            round_trip(&params(&[(1, 1); 3], predictor, 12, 0), 13);
        }
    }

    #[test]
    fn precisions_from_two_to_sixteen_bits_round_trip() {
        for precision in [2u8, 5, 8, 10, 12, 14, 15, 16] {
            round_trip(&params(&[(1, 1); 2], 4, precision, 0), 20);
        }
    }

    #[test]
    fn restart_markers_round_trip_with_every_predictor() {
        for predictor in 1..=7 {
            for rows in [1usize, 3] {
                round_trip(&params(&[(1, 1); 4], predictor, 14, rows), 30);
            }
        }
    }

    #[test]
    fn point_transforms_shift_the_samples_back() {
        for pt in [1u8, 2, 4] {
            let mut p = params(&[(1, 1); 3], 6, 14, 0);
            p.point_transform = pt;
            round_trip(&p, 40);
        }
    }

    #[test]
    fn skewed_huffman_tables_with_sixteen_bit_codes_round_trip() {
        for predictor in [1u8, 7] {
            let mut p = params(&[(1, 1); 4], predictor, 14, 0);
            p.huffman = HuffmanChoice::Skewed;
            round_trip(&p, 50);
        }
    }

    #[test]
    fn subsampled_components_round_trip_with_and_without_the_sony_quirk() {
        for quirk in [false, true] {
            for sampling in [
                vec![(2u8, 2u8), (1, 1), (1, 1)],
                vec![(2, 1), (1, 1), (1, 1)],
                vec![(1, 2), (1, 1), (1, 1)],
                vec![(4, 3), (2, 1), (1, 2)],
            ] {
                for restart_rows in [0usize, 2] {
                    let mut p = params(&sampling, 1, 16, restart_rows);
                    p.sony_420_quirk = quirk;
                    round_trip(&p, 60);
                    p.predictor = 5;
                    round_trip(&p, 61);
                }
            }
        }
    }

    #[test]
    fn the_sony_quirk_changes_the_prediction_and_only_in_420() {
        let p = params(&[(2, 2), (1, 1), (1, 1)], 1, 14, 0);
        let planes = planes_for(&p, 70);
        let normal = encode(&planes, &p);
        let quirk = encode(
            &planes,
            &EncodeParams {
                sony_420_quirk: true,
                ..p.clone()
            },
        );
        assert_ne!(normal, quirk);
        // Decoding the quirk stream without the quirk gives different luma.
        let wrong = decode(&quirk, &Options::default()).expect("still a valid stream");
        assert_ne!(wrong.planes[0].data, planes[0]);
        // 4:2:2 has no quirk to apply.
        let p422 = params(&[(2, 1), (1, 1), (1, 1)], 1, 14, 0);
        let planes = planes_for(&p422, 71);
        let a = encode(&planes, &p422);
        let b = encode(
            &planes,
            &EncodeParams {
                sony_420_quirk: true,
                ..p422
            },
        );
        assert_eq!(a, b);
    }

    #[test]
    fn single_row_and_single_column_images_work() {
        for (w, h) in [(1usize, 1usize), (1, 9), (9, 1), (2, 2)] {
            let mut p = params(&[(1, 1); 2], 4, 12, 0);
            p.width = w;
            p.height = h;
            round_trip(&p, 80);
        }
    }

    #[test]
    fn stuffed_ff_bytes_survive() {
        // A flat plane of 0xFFFF-ish values makes long runs of one bits.
        let mut p = params(&[(1, 1)], 1, 16, 0);
        p.width = 64;
        p.height = 8;
        let planes = vec![vec![0xFFFFu16; 64 * 8]];
        let bytes = encode(&planes, &p);
        assert!(bytes.windows(2).any(|w| w == [0xFF, 0x00]) || bytes.len() < 60);
        let decoded = decode(&bytes, &Options::default()).expect("decodes");
        assert_eq!(decoded.planes[0].data, planes[0]);
    }

    #[test]
    fn broken_streams_are_errors_not_panics() {
        let p = params(&[(1, 1); 4], 1, 14, 0);
        let planes = planes_for(&p, 90);
        let good = encode(&planes, &p);
        // Every truncation of the stream.
        for cut in (0..good.len()).step_by(7) {
            let _ = decode(&good[..cut], &Options::default());
        }
        // Truncation inside the data is reported, not accepted.
        assert!(decode(&good[..good.len() / 2], &Options::default()).is_err());
        // Random byte flips never panic.
        let mut rng = SplitMix64(5);
        for _ in 0..300 {
            let mut bytes = good.clone();
            for _ in 0..3 {
                let at = rng.below(bytes.len() as u64) as usize;
                bytes[at] = rng.next_u64() as u8;
            }
            let _ = decode(&bytes, &Options::default());
        }
        // Pure noise and tiny inputs.
        for len in 0..40usize {
            let noise: Vec<u8> = (0..len).map(|_| rng.next_u64() as u8).collect();
            assert!(decode(&noise, &Options::default()).is_err());
        }
    }

    #[test]
    fn header_errors_are_reported() {
        let p = params(&[(1, 1)], 1, 14, 0);
        let good = encode(&planes_for(&p, 91), &p);
        let find = |marker: u8| good.windows(2).position(|w| w == [0xFF, marker]).unwrap();
        // Not lossless: change SOF3 to SOF0.
        let mut bytes = good.clone();
        bytes[find(0xC3) + 1] = 0xC0;
        assert!(decode(&bytes, &Options::default()).is_err());
        // Zero width.
        let mut bytes = good.clone();
        let sof = find(0xC3);
        bytes[sof + 7] = 0;
        bytes[sof + 8] = 0;
        assert!(decode(&bytes, &Options::default()).is_err());
        // Predictor 0.
        let mut bytes = good.clone();
        let sos = find(0xDA);
        bytes[sos + 2 + 2 + 1 + 2] = 0;
        assert!(decode(&bytes, &Options::default()).is_err());
        // Frame larger than the budget.
        let tiny = Options {
            max_samples: 10,
            ..Options::default()
        };
        assert!(decode(&good, &tiny).is_err());
    }

    #[test]
    fn missing_restart_intervals_are_corrupt() {
        let p = params(&[(1, 1); 2], 1, 14, 2);
        let planes = planes_for(&p, 92);
        let good = encode(&planes, &p);
        // Drop everything after the first RST marker.
        let rst = good
            .windows(2)
            .position(|w| w[0] == 0xFF && (0xD0..=0xD7).contains(&w[1]))
            .expect("the stream has restart markers");
        let mut cut = good[..rst].to_vec();
        cut.extend_from_slice(&[0xFF, 0xD9]);
        assert!(decode(&cut, &Options::default()).is_err());
    }
}
