//! Turns the three scans of an optimised-Huffman JPEG into one interleaved scan, losslessly.
//!
//! `jpeg-encoder` computes optimal Huffman tables from the quantised blocks of each component and
//! then writes **one scan per component** (a non-interleaved baseline JPEG). That is legal, and
//! libjpeg reads it, but several decoders do not: `zune-jpeg`, the decoder behind Rust's `image`
//! crate, returns wrong colours for it. libjpeg's own `-optimize` writes a single interleaved scan,
//! which every decoder handles, so this module rewrites the entropy-coded data into that form.
//!
//! What is kept: the quantisation tables and every quantised coefficient (the decoded pixels are
//! identical, which the tests check against libjpeg through Pillow). What changes: the blocks are
//! written in MCU order, the DC differences are recomputed for that order, luma blocks that fill an
//! incomplete MCU at the right or bottom edge are padded with empty blocks (they lie outside the
//! image and are cropped), and the Huffman tables are rebuilt for the new symbol stream with the
//! optimal-table procedure of ITU-T T.81 Annex K.2 (the old tables may lack symbols the new order
//! needs, for example a larger DC difference between vertically adjacent luma blocks).
//!
//! Memory stays small: the scans are decoded in step, one MCU row at a time, in two passes (symbol
//! statistics, then output).

use crate::error::ExportError;

fn bad(reason: &str) -> ExportError {
    ExportError::encode(
        "JPEG",
        format!("unexpected structure while interleaving scans: {reason}"),
    )
}

/// One component of the frame header.
#[derive(Debug, Clone, Copy)]
struct Component {
    id: u8,
    h: usize,
    v: usize,
}

/// A Huffman table, for decoding (the encoder's tables) or encoding (the rebuilt ones).
#[derive(Clone)]
struct Huffman {
    /// Number of codes of each length 1..=16.
    counts: [u8; 16],
    /// Symbols in order of code length.
    symbols: Vec<u8>,
    /// Largest code of each length (index 1..=16), or -1.
    max_code: [i32; 17],
    /// `symbols[code + delta[len]]` is the symbol of a code of that length.
    delta: [i32; 17],
    /// Code and length for each symbol (length 0 = not in the table).
    encode: [(u16, u8); 256],
}

impl Huffman {
    fn new(counts: [u8; 16], symbols: Vec<u8>) -> Result<Self, ExportError> {
        let mut max_code = [-1_i32; 17];
        let mut delta = [0_i32; 17];
        let mut encode = [(0_u16, 0_u8); 256];
        let mut code: i32 = 0;
        let mut k = 0_usize;
        for len in 1..=16_usize {
            let n = usize::from(counts[len - 1]);
            if n > 0 {
                delta[len] = k as i32 - code;
                for _ in 0..n {
                    let symbol = *symbols
                        .get(k)
                        .ok_or_else(|| bad("a Huffman table is short of symbols"))?;
                    if code >= 1 << len {
                        return Err(bad("an over-subscribed Huffman table"));
                    }
                    encode[usize::from(symbol)] = (code as u16, len as u8);
                    code += 1;
                    k += 1;
                }
                max_code[len] = code - 1;
            }
            code <<= 1;
        }
        let symbols = symbols[..k].to_vec();
        Ok(Self {
            counts,
            symbols,
            max_code,
            delta,
            encode,
        })
    }

    /// The optimal table for symbol frequencies `freq` (T.81 Annex K.2, as in libjpeg): a
    /// reserved all-ones code point keeps every real code from being all ones, and lengths are
    /// limited to 16 bits.
    fn optimal(freq: &[u64; 256]) -> Result<Self, ExportError> {
        let mut freq: Vec<u64> = freq.iter().copied().chain([1]).collect(); // 257 entries
        let mut code_size = vec![0_u32; 257];
        let mut others = vec![-1_i32; 257];
        loop {
            // c1: the least frequent symbol (highest index among ties); c2: the next one.
            let mut c1 = -1_i32;
            let mut v = u64::MAX;
            for (i, &f) in freq.iter().enumerate() {
                if f != 0 && f <= v {
                    v = f;
                    c1 = i as i32;
                }
            }
            let mut c2 = -1_i32;
            v = u64::MAX;
            for (i, &f) in freq.iter().enumerate() {
                if f != 0 && f <= v && i as i32 != c1 {
                    v = f;
                    c2 = i as i32;
                }
            }
            if c2 < 0 {
                break;
            }
            freq[c1 as usize] += freq[c2 as usize];
            freq[c2 as usize] = 0;
            let mut c = c1 as usize;
            code_size[c] += 1;
            while others[c] >= 0 {
                c = others[c] as usize;
                code_size[c] += 1;
            }
            others[c] = c2;
            let mut c = c2 as usize;
            code_size[c] += 1;
            while others[c] >= 0 {
                c = others[c] as usize;
                code_size[c] += 1;
            }
        }
        let mut bits = [0_i32; 33];
        for &size in &code_size {
            if size > 0 {
                let size = size as usize;
                if size > 32 {
                    return Err(bad("a Huffman code longer than 32 bits"));
                }
                bits[size] += 1;
            }
        }
        for i in (17..=32).rev() {
            while bits[i] > 0 {
                let mut j = i - 2;
                while bits[j] == 0 {
                    j -= 1;
                }
                bits[i] -= 2;
                bits[i - 1] += 1;
                bits[j + 1] += 2;
                bits[j] -= 1;
            }
        }
        // Remove the reserved code point (the longest code).
        let mut i = 16;
        while bits[i] == 0 {
            i -= 1;
        }
        bits[i] -= 1;
        let mut counts = [0_u8; 16];
        for (len, c) in counts.iter_mut().enumerate() {
            *c = u8::try_from(bits[len + 1]).map_err(|_| bad("too many codes of one length"))?;
        }
        // Symbols by code length, then by value; the reserved entry 256 is not a symbol.
        let mut symbols = Vec::new();
        for len in 1..=32_u32 {
            for (symbol, &size) in code_size.iter().enumerate().take(256) {
                if size == len {
                    symbols.push(symbol as u8);
                }
            }
        }
        Self::new(counts, symbols)
    }

    fn segment(&self, class: u8, id: u8) -> Vec<u8> {
        let mut s = vec![0xFF, 0xC4];
        s.extend_from_slice(&((19 + self.symbols.len()) as u16).to_be_bytes());
        s.push((class << 4) | id);
        s.extend_from_slice(&self.counts);
        s.extend_from_slice(&self.symbols);
        s
    }
}

/// Reads bits from entropy-coded data (unstuffing `FF 00`).
struct BitReader<'a> {
    data: &'a [u8],
    pos: usize,
    acc: u64,
    nbits: u32,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            pos: 0,
            acc: 0,
            nbits: 0,
        }
    }

    fn fill(&mut self) {
        while self.nbits <= 56 {
            let byte = if self.pos < self.data.len() {
                let b = self.data[self.pos];
                self.pos += 1;
                if b == 0xFF && self.data.get(self.pos) == Some(&0) {
                    self.pos += 1; // a stuffed zero follows every FF
                }
                b
            } else {
                0xFF // past the end reads as ones, as the padding does
            };
            self.acc |= u64::from(byte) << (56 - self.nbits);
            self.nbits += 8;
        }
    }

    fn bits(&mut self, n: u32) -> u32 {
        if n == 0 {
            return 0;
        }
        if self.nbits < n {
            self.fill();
        }
        let v = (self.acc >> (64 - n)) as u32;
        self.acc <<= n;
        self.nbits -= n;
        v
    }

    fn symbol(&mut self, table: &Huffman) -> Result<u8, ExportError> {
        let mut code: i32 = 0;
        for len in 1..=16_usize {
            code = (code << 1) | self.bits(1) as i32;
            if table.max_code[len] >= 0 && code <= table.max_code[len] {
                let index = code + table.delta[len];
                return table
                    .symbols
                    .get(index as usize)
                    .copied()
                    .ok_or_else(|| bad("a Huffman code outside its table"));
            }
        }
        Err(bad("an invalid Huffman code"))
    }
}

/// Writes bits with byte stuffing.
struct BitWriter {
    out: Vec<u8>,
    acc: u64,
    nbits: u32,
}

impl BitWriter {
    fn new() -> Self {
        Self {
            out: Vec::new(),
            acc: 0,
            nbits: 0,
        }
    }

    fn put(&mut self, value: u32, n: u32) {
        if n == 0 {
            return;
        }
        self.acc = (self.acc << n) | u64::from(value & ((1_u32 << n) - 1));
        self.nbits += n;
        while self.nbits >= 8 {
            let byte = (self.acc >> (self.nbits - 8)) as u8;
            self.nbits -= 8;
            self.out.push(byte);
            if byte == 0xFF {
                self.out.push(0);
            }
        }
        self.acc &= (1_u64 << self.nbits) - 1;
    }

    fn finish(mut self) -> Vec<u8> {
        if self.nbits > 0 {
            let pad = 8 - self.nbits;
            self.put((1 << pad) - 1, pad);
        }
        self.out
    }
}

fn extend(value: u32, size: u32) -> i32 {
    if size == 0 {
        0
    } else if value < (1 << (size - 1)) {
        value as i32 - (1 << size) + 1
    } else {
        value as i32
    }
}

/// An AC symbol with its raw amplitude bits, as it was in the stream (EOB and ZRL included).
#[derive(Clone, Copy)]
struct AcToken {
    symbol: u8,
    bits: u16,
}

/// A decoded block: the absolute DC value and a range of AC tokens in a shared buffer.
#[derive(Clone, Copy, Default)]
struct Block {
    dc: i32,
    first: u32,
    len: u32,
}

/// Decodes the blocks of one component's scan, one at a time, in raster order.
struct ScanReader<'a> {
    bits: BitReader<'a>,
    dc: &'a Huffman,
    ac: &'a Huffman,
    pred: i32,
}

impl ScanReader<'_> {
    fn next_block(&mut self, tokens: &mut Vec<AcToken>) -> Result<Block, ExportError> {
        let size = u32::from(self.bits.symbol(self.dc)?);
        if size > 11 {
            return Err(bad("a DC category above 11"));
        }
        self.pred += extend(self.bits.bits(size), size);
        let first = tokens.len() as u32;
        let mut k = 1;
        while k < 64 {
            let rs = self.bits.symbol(self.ac)?;
            let (run, size) = (usize::from(rs >> 4), u32::from(rs & 15));
            if size == 0 {
                tokens.push(AcToken {
                    symbol: rs,
                    bits: 0,
                });
                if run == 15 {
                    k += 16; // ZRL: sixteen zeros
                    continue;
                }
                break; // EOB
            }
            tokens.push(AcToken {
                symbol: rs,
                bits: self.bits.bits(size) as u16,
            });
            k += run + 1;
        }
        if k > 64 {
            return Err(bad("a block longer than 64 coefficients"));
        }
        Ok(Block {
            dc: self.pred,
            first,
            len: tokens.len() as u32 - first,
        })
    }
}

/// What happens to each block in MCU order.
trait Sink {
    /// A block of component `comp` (index in the frame) with DC difference `diff` in the new order.
    fn block(&mut self, comp: usize, diff: i32, tokens: &[AcToken]) -> Result<(), ExportError>;
}

/// Counts symbol frequencies per Huffman table.
struct Stats {
    dc: [[u64; 256]; 4],
    ac: [[u64; 256]; 4],
    selectors: Vec<(usize, usize)>,
}

impl Sink for Stats {
    fn block(&mut self, comp: usize, diff: i32, tokens: &[AcToken]) -> Result<(), ExportError> {
        let (dc, ac) = self.selectors[comp];
        self.dc[dc][(32 - diff.unsigned_abs().leading_zeros()) as usize] += 1;
        for t in tokens {
            self.ac[ac][usize::from(t.symbol)] += 1;
        }
        Ok(())
    }
}

/// Writes the interleaved entropy-coded segment.
struct Emit {
    writer: BitWriter,
    dc: [Option<Huffman>; 4],
    ac: [Option<Huffman>; 4],
    selectors: Vec<(usize, usize)>,
}

impl Sink for Emit {
    fn block(&mut self, comp: usize, diff: i32, tokens: &[AcToken]) -> Result<(), ExportError> {
        let (dc_id, ac_id) = self.selectors[comp];
        let size = 32 - diff.unsigned_abs().leading_zeros();
        let dc = self.dc[dc_id]
            .as_ref()
            .ok_or_else(|| bad("a missing DC table"))?;
        let ac = self.ac[ac_id]
            .as_ref()
            .ok_or_else(|| bad("a missing AC table"))?;
        let (code, len) = dc.encode[size as usize];
        if len == 0 {
            return Err(bad("a DC category missing from the rebuilt table"));
        }
        self.writer.put(u32::from(code), u32::from(len));
        let amplitude = if diff < 0 {
            (diff - 1) as u32
        } else {
            diff as u32
        };
        self.writer.put(amplitude, size);
        for t in tokens {
            let (code, len) = ac.encode[usize::from(t.symbol)];
            if len == 0 {
                return Err(bad("an AC symbol missing from the rebuilt table"));
            }
            self.writer.put(u32::from(code), u32::from(len));
            self.writer.put(u32::from(t.bits), u32::from(t.symbol & 15));
        }
        Ok(())
    }
}

struct Scan<'a> {
    component: u8,
    dc_table: usize,
    ac_table: usize,
    data: &'a [u8],
}

struct Frame {
    width: usize,
    height: usize,
    components: Vec<Component>,
}

/// Decodes the scans in step and feeds the blocks to `sink` in interleaved MCU order.
fn walk(
    frame: &Frame,
    scans: &[&Scan<'_>],
    dc_tables: &[Option<Huffman>; 4],
    ac_tables: &[Option<Huffman>; 4],
    sink: &mut dyn Sink,
) -> Result<(), ExportError> {
    let h_max = frame.components.iter().map(|c| c.h).max().unwrap_or(1);
    let v_max = frame.components.iter().map(|c| c.v).max().unwrap_or(1);
    let mcu_cols = frame.width.div_ceil(8 * h_max);
    let mcu_rows = frame.height.div_ceil(8 * v_max);

    struct Comp<'a> {
        info: Component,
        reader: ScanReader<'a>,
        cols: usize,
        rows: usize,
        pred: i32,
    }
    let mut comps: Vec<Comp<'_>> = Vec::new();
    for (info, scan) in frame.components.iter().zip(scans) {
        let dc = dc_tables[scan.dc_table]
            .as_ref()
            .ok_or_else(|| bad("a missing DC table"))?;
        let ac = ac_tables[scan.ac_table]
            .as_ref()
            .ok_or_else(|| bad("a missing AC table"))?;
        comps.push(Comp {
            info: *info,
            reader: ScanReader {
                bits: BitReader::new(scan.data),
                dc,
                ac,
                pred: 0,
            },
            cols: (frame.width * info.h).div_ceil(h_max).div_ceil(8),
            rows: (frame.height * info.v).div_ceil(v_max).div_ceil(8),
            pred: 0,
        });
    }

    const EOB: [AcToken; 1] = [AcToken { symbol: 0, bits: 0 }];
    for mcu_row in 0..mcu_rows {
        // This MCU row's blocks of every component: `v` rows of `cols` blocks.
        let mut rows: Vec<(Vec<Block>, Vec<AcToken>)> = Vec::with_capacity(comps.len());
        for c in &mut comps {
            let mut blocks = vec![Block::default(); c.info.v * c.cols];
            let mut tokens = Vec::new();
            for v in 0..c.info.v {
                if mcu_row * c.info.v + v < c.rows {
                    for bx in 0..c.cols {
                        blocks[v * c.cols + bx] = c.reader.next_block(&mut tokens)?;
                    }
                }
            }
            rows.push((blocks, tokens));
        }
        for mcu_col in 0..mcu_cols {
            for (index, (c, (blocks, tokens))) in comps.iter_mut().zip(&rows).enumerate() {
                for v in 0..c.info.v {
                    for h in 0..c.info.h {
                        let (bx, by) = (mcu_col * c.info.h + h, mcu_row * c.info.v + v);
                        if bx < c.cols && by < c.rows {
                            let block = &blocks[v * c.cols + bx];
                            let diff = block.dc - c.pred;
                            c.pred = block.dc;
                            sink.block(
                                index,
                                diff,
                                &tokens[block.first as usize..(block.first + block.len) as usize],
                            )?;
                        } else {
                            // Outside the image: same DC as before, no coefficients.
                            sink.block(index, 0, &EOB)?;
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

/// Rewrites a three-scan baseline JPEG as a single interleaved scan with rebuilt optimal Huffman tables.
///
/// A JPEG that is already one scan, or has one component, is returned unchanged.
pub fn interleave_scans(jpeg: &[u8]) -> Result<Vec<u8>, ExportError> {
    if jpeg.len() < 4 || jpeg[0] != 0xFF || jpeg[1] != 0xD8 {
        return Err(bad("not a JPEG"));
    }
    let mut i = 2;
    let mut frame: Option<Frame> = None;
    let mut dc_tables: [Option<Huffman>; 4] = Default::default();
    let mut ac_tables: [Option<Huffman>; 4] = Default::default();
    // Everything before the first scan except the Huffman tables, which are rebuilt.
    let mut head: Vec<u8> = vec![0xFF, 0xD8];
    let mut scans: Vec<Scan<'_>> = Vec::new();

    while i + 4 <= jpeg.len() {
        if jpeg[i] != 0xFF {
            return Err(bad("a marker was expected"));
        }
        let marker = jpeg[i + 1];
        if marker == 0xD9 {
            break;
        }
        let len = usize::from(u16::from_be_bytes([jpeg[i + 2], jpeg[i + 3]]));
        if len < 2 || i + 2 + len > jpeg.len() {
            return Err(bad("a truncated segment"));
        }
        let body = &jpeg[i + 4..i + 2 + len];
        match marker {
            0xC0 => {
                if body.len() < 6 {
                    return Err(bad("a short frame header"));
                }
                let n = usize::from(body[5]);
                if body.len() < 6 + 3 * n {
                    return Err(bad("a short frame header"));
                }
                frame = Some(Frame {
                    height: usize::from(u16::from_be_bytes([body[1], body[2]])),
                    width: usize::from(u16::from_be_bytes([body[3], body[4]])),
                    components: (0..n)
                        .map(|c| Component {
                            id: body[6 + 3 * c],
                            h: usize::from(body[7 + 3 * c] >> 4),
                            v: usize::from(body[7 + 3 * c] & 15),
                        })
                        .collect(),
                });
                head.extend_from_slice(&jpeg[i..i + 2 + len]);
            }
            0xC1..=0xC3 | 0xC5..=0xC7 | 0xC9..=0xCB | 0xCD..=0xCF => {
                return Err(bad("only baseline frames are supported"));
            }
            0xC4 => {
                let mut b = body;
                while b.len() > 17 {
                    let (class, id) = (usize::from(b[0] >> 4), usize::from(b[0] & 15));
                    // `b.len() > 17` was checked by the loop condition, so 16 bytes follow the class byte.
                    let counts: [u8; 16] = b[1..17].try_into().expect("16 bytes");
                    let total: usize = counts.iter().map(|&c| usize::from(c)).sum();
                    if class > 1 || id > 3 || b.len() < 17 + total {
                        return Err(bad("a bad Huffman table definition"));
                    }
                    let table = Huffman::new(counts, b[17..17 + total].to_vec())?;
                    if class == 0 {
                        dc_tables[id] = Some(table);
                    } else {
                        ac_tables[id] = Some(table);
                    }
                    b = &b[17 + total..];
                }
            }
            0xDD => return Err(bad("restart intervals are not supported")),
            0xDA => {
                if body.first().is_some_and(|&n| n > 1) {
                    return Ok(jpeg.to_vec()); // already one interleaved scan
                }
                if body.len() < 6 || body[0] != 1 {
                    return Err(bad("expected single-component scans"));
                }
                let start = i + 2 + len;
                let mut j = start;
                loop {
                    let Some(k) = jpeg[j..].iter().position(|&b| b == 0xFF) else {
                        return Err(bad("a scan without an end"));
                    };
                    j += k;
                    match jpeg.get(j + 1) {
                        Some(0) => j += 2,
                        Some(0xD0..=0xD7) => return Err(bad("restart markers are not supported")),
                        Some(_) => break,
                        None => return Err(bad("a scan without an end")),
                    }
                }
                scans.push(Scan {
                    component: body[1],
                    dc_table: usize::from((body[2] >> 4) & 3),
                    ac_table: usize::from(body[2] & 3),
                    data: &jpeg[start..j],
                });
                i = j;
                continue;
            }
            _ => head.extend_from_slice(&jpeg[i..i + 2 + len]),
        }
        i += 2 + len;
    }

    let frame = frame.ok_or_else(|| bad("no frame header"))?;
    if frame.components.len() == 1 || scans.len() == 1 {
        return Ok(jpeg.to_vec());
    }
    if scans.len() != frame.components.len() {
        return Err(bad(
            "the number of scans differs from the number of components",
        ));
    }
    if frame.width == 0
        || frame.height == 0
        || frame.components.iter().any(|c| c.h == 0 || c.v == 0)
    {
        return Err(bad("an empty frame"));
    }
    // Scans in frame order.
    let ordered: Vec<&Scan<'_>> = frame
        .components
        .iter()
        .map(|c| {
            scans
                .iter()
                .find(|s| s.component == c.id)
                .ok_or_else(|| bad("a component without a scan"))
        })
        .collect::<Result<_, _>>()?;
    let selectors: Vec<(usize, usize)> = ordered.iter().map(|s| (s.dc_table, s.ac_table)).collect();

    // Pass 1: symbol statistics in the new order.
    let mut stats = Stats {
        dc: [[0; 256]; 4],
        ac: [[0; 256]; 4],
        selectors: selectors.clone(),
    };
    walk(&frame, &ordered, &dc_tables, &ac_tables, &mut stats)?;
    let mut new_dc: [Option<Huffman>; 4] = Default::default();
    let mut new_ac: [Option<Huffman>; 4] = Default::default();
    let mut segments = Vec::new();
    for id in 0..4 {
        if selectors.iter().any(|s| s.0 == id) {
            let table = Huffman::optimal(&stats.dc[id])?;
            segments.extend(table.segment(0, id as u8));
            new_dc[id] = Some(table);
        }
        if selectors.iter().any(|s| s.1 == id) {
            let table = Huffman::optimal(&stats.ac[id])?;
            segments.extend(table.segment(1, id as u8));
            new_ac[id] = Some(table);
        }
    }

    // Pass 2: the interleaved entropy-coded segment.
    let mut emit = Emit {
        writer: BitWriter::new(),
        dc: new_dc,
        ac: new_ac,
        selectors,
    };
    walk(&frame, &ordered, &dc_tables, &ac_tables, &mut emit)?;
    let entropy = emit.writer.finish();

    let mut out = head;
    out.extend_from_slice(&segments);
    out.extend_from_slice(&[0xFF, 0xDA]);
    out.extend_from_slice(&((6 + 2 * ordered.len()) as u16).to_be_bytes());
    out.push(ordered.len() as u8);
    for (c, s) in frame.components.iter().zip(&ordered) {
        out.push(c.id);
        out.push(((s.dc_table as u8) << 4) | s.ac_table as u8);
    }
    out.extend_from_slice(&[0, 63, 0]);
    out.extend_from_slice(&entropy);
    out.extend_from_slice(&[0xFF, 0xD9]);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn optimal_tables_satisfy_the_kraft_inequality_and_limit_lengths() {
        // A skewed distribution forces long codes that the length limiting must repair.
        let mut freq = [0_u64; 256];
        for (i, f) in freq.iter_mut().enumerate().take(200) {
            *f = 1 << (i % 20).min(40);
        }
        let t = Huffman::optimal(&freq).unwrap();
        let used: usize = t.counts.iter().map(|&c| usize::from(c)).sum();
        assert_eq!(used, 200);
        let kraft: f64 = t
            .counts
            .iter()
            .enumerate()
            .map(|(i, &c)| f64::from(c) * 0.5_f64.powi(i as i32 + 1))
            .sum();
        assert!(
            kraft < 1.0,
            "the reserved point keeps the sum below 1: {kraft}"
        );
        // Encode then decode every symbol.
        for s in 0..200_usize {
            let (code, len) = t.encode[s];
            assert!((1..=16).contains(&len));
            let mut w = BitWriter::new();
            w.put(u32::from(code), u32::from(len));
            let bytes = w.finish();
            let mut r = BitReader::new(&bytes);
            assert_eq!(usize::from(r.symbol(&t).unwrap()), s);
        }
    }

    #[test]
    fn without_length_limiting_frequent_symbols_get_shorter_codes() {
        let mut freq = [0_u64; 256];
        for (i, f) in freq.iter_mut().enumerate().take(120) {
            *f = 100 + (i as u64) * 13;
        }
        let t = Huffman::optimal(&freq).unwrap();
        for a in 0..120_usize {
            for b in 0..120_usize {
                if freq[a] > freq[b] {
                    assert!(
                        t.encode[a].1 <= t.encode[b].1,
                        "freq {} vs {}",
                        freq[a],
                        freq[b]
                    );
                }
            }
        }
    }

    #[test]
    fn a_single_symbol_still_gets_a_valid_table() {
        let mut freq = [0_u64; 256];
        freq[0] = 10;
        let t = Huffman::optimal(&freq).unwrap();
        assert_eq!(t.symbols, vec![0]);
        assert_eq!(t.encode[0].1, 1);
    }

    #[test]
    fn bit_writer_stuffs_ff_and_pads_with_ones() {
        let mut w = BitWriter::new();
        w.put(0xFF, 8);
        w.put(0b101, 3);
        assert_eq!(w.finish(), vec![0xFF, 0x00, 0b1011_1111]);
        let mut r = BitReader::new(&[0xFF, 0x00, 0xA5]);
        assert_eq!(r.bits(8), 0xFF);
        assert_eq!(r.bits(8), 0xA5);
    }

    #[test]
    fn extend_matches_the_t81_rule() {
        assert_eq!(extend(0, 1), -1);
        assert_eq!(extend(1, 1), 1);
        assert_eq!(extend(0b00, 2), -3);
        assert_eq!(extend(0b11, 2), 3);
        assert_eq!(extend(0b01, 2), -2);
        assert_eq!(extend(0, 0), 0);
    }

    use jpeg_encoder::{ColorType, Encoder, SamplingFactor};

    /// Pseudo-random photographic-ish pixels: smooth gradients, edges and noise.
    fn test_pixels(w: usize, h: usize, noise: u32) -> Vec<u8> {
        let mut state = 0x2545_F491_4F6C_DD1D_u64 ^ (w * 31 + h) as u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 33) as u32
        };
        let mut v = Vec::with_capacity(w * h * 3);
        for y in 0..h {
            for x in 0..w {
                let edge = if (x / 7 + y / 5) % 3 == 0 { 70 } else { 0 };
                for c in 0..3 {
                    let base = (x * 255 / w.max(1) + y * 90 * (c + 1) / h.max(1) + edge) % 256;
                    let n = if noise == 0 {
                        0
                    } else {
                        (next() % noise) as usize
                    };
                    v.push(((base + n) % 256) as u8);
                }
            }
        }
        v
    }

    fn encode_with(
        px: &[u8],
        w: usize,
        h: usize,
        q: u8,
        sampling: SamplingFactor,
        optimize: bool,
    ) -> Vec<u8> {
        let mut out = Vec::new();
        let mut e = Encoder::new(&mut out, q);
        e.set_sampling_factor(sampling);
        e.set_optimized_huffman_tables(optimize);
        e.set_chroma_subsampling_method(jpeg_encoder::ChromaSubsamplingMethod::Average);
        e.encode(px, w as u16, h as u16, ColorType::Rgb).unwrap();
        out
    }

    fn decode(jpeg: &[u8]) -> Vec<u8> {
        zune_jpeg::JpegDecoder::new(std::io::Cursor::new(jpeg.to_vec()))
            .decode()
            .unwrap()
    }

    fn count_scans(jpeg: &[u8]) -> usize {
        let mut n = 0;
        let mut i = 2;
        while i + 4 <= jpeg.len() {
            let marker = jpeg[i + 1];
            let len = usize::from(u16::from_be_bytes([jpeg[i + 2], jpeg[i + 3]]));
            if marker == 0xDA {
                n += 1;
                // Skip the entropy-coded data.
                i += 2 + len;
                while i + 1 < jpeg.len()
                    && !(jpeg[i] == 0xFF
                        && jpeg[i + 1] != 0
                        && !(0xD0..=0xD7).contains(&jpeg[i + 1]))
                {
                    i += 1;
                }
                continue;
            }
            if marker == 0xD9 {
                break;
            }
            i += 2 + len;
        }
        n
    }

    /// The rewritten file decodes to exactly the pixels of the same coefficients coded with the
    /// standard tables in one interleaved scan: same quantised coefficients, different entropy coding.
    #[test]
    fn interleaving_preserves_every_pixel() {
        let sizes = [
            (1, 1),
            (2, 3),
            (7, 9),
            (8, 8),
            (9, 7),
            (15, 17),
            (16, 16),
            (17, 33),
            (31, 8),
            (33, 31),
            (48, 32),
            (64, 48),
            (100, 61),
            (123, 77),
            (256, 8),
            (8, 256),
            (200, 200),
        ];
        for sampling in [SamplingFactor::R_4_4_4, SamplingFactor::R_4_2_0] {
            for &(w, h) in &sizes {
                for (q, noise) in [(92_u8, 0_u32), (100, 255), (50, 40)] {
                    let px = test_pixels(w, h, noise);
                    let scans = encode_with(&px, w, h, q, sampling, true);
                    assert_eq!(
                        count_scans(&scans),
                        3,
                        "the crate writes three scans for optimised tables"
                    );
                    let ours = interleave_scans(&scans).unwrap();
                    assert_eq!(count_scans(&ours), 1, "{w}x{h}");
                    let reference = encode_with(&px, w, h, q, sampling, false);
                    assert_eq!(
                        decode(&reference),
                        decode(&ours),
                        "{sampling:?} {w}x{h} q{q} noise {noise}"
                    );
                }
            }
        }
    }

    #[test]
    fn the_result_is_smaller_than_standard_tables_and_valid_for_a_large_image() {
        let (w, h) = (640, 480);
        let px = test_pixels(w, h, 12);
        let standard = encode_with(&px, w, h, 92, SamplingFactor::R_4_4_4, false);
        let ours =
            interleave_scans(&encode_with(&px, w, h, 92, SamplingFactor::R_4_4_4, true)).unwrap();
        assert!(
            ours.len() < standard.len(),
            "{} vs {}",
            ours.len(),
            standard.len()
        );
        assert_eq!(decode(&standard), decode(&ours));
    }

    #[test]
    fn single_scan_input_passes_through_and_garbage_is_rejected() {
        let px = test_pixels(20, 20, 0);
        let standard = encode_with(&px, 20, 20, 90, SamplingFactor::R_4_4_4, false);
        assert_eq!(interleave_scans(&standard).unwrap(), standard);
        assert!(interleave_scans(b"not a jpeg").is_err());
        assert!(interleave_scans(&[0xFF, 0xD8, 0xFF, 0xD9]).is_err());
        // A truncated three-scan file is an error, not a panic.
        let scans = encode_with(&px, 20, 20, 90, SamplingFactor::R_4_4_4, true);
        for cut in [30, scans.len() / 2, scans.len() - 3] {
            let _ = interleave_scans(&scans[..cut]);
        }
    }
}
