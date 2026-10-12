//! Test-only lossless JPEG (ITU T.81 process 14) encoder.
//!
//! Produces streams for every shape the decoder in `decode::ljpeg` accepts: one to four
//! components with any sampling factors, predictors 1 to 7, a point transform, restart intervals
//! and Huffman tables generated from the data (the optimal length-limited procedure of T.81
//! Annex K.2) or deliberately skewed so that codes reach 16 bits and exercise the decoder's slow
//! path. The prediction rules are written out here a second time, independently of the decoder,
//! from T.81 section H.1.2.1; the LibRaw oracle checks both against real-world behaviour.

/// How the Huffman tables are chosen.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HuffmanChoice {
    /// Optimal for the data (shortest codes for the most frequent categories).
    Optimal,
    /// A table whose frequencies fall off geometrically, so the rarely used categories get codes
    /// up to 16 bits long and the common ones are not necessarily short.
    Skewed,
}

/// Parameters of one stream.
#[derive(Clone, Debug)]
pub struct EncodeParams {
    /// Sample precision `P`.
    pub precision: u8,
    /// Predictor 1 to 7.
    pub predictor: u8,
    /// Point transform `Pt`; input samples must be multiples of `2^Pt` to survive.
    pub point_transform: u8,
    /// Restart interval in MCUs (0 = none). Must be a multiple of the MCUs per row.
    pub restart_interval: usize,
    /// Use Sony's 4:2:0 first-column prediction (component 0 with `v == 2`).
    pub sony_420_quirk: bool,
    /// Huffman table choice.
    pub huffman: HuffmanChoice,
    /// Frame width `X` in luma samples.
    pub width: usize,
    /// Frame height `Y` in lines.
    pub height: usize,
    /// `(h, v)` sampling factors of each component.
    pub sampling: Vec<(u8, u8)>,
}

impl EncodeParams {
    fn h_max(&self) -> usize {
        self.sampling
            .iter()
            .map(|(h, _)| usize::from(*h))
            .max()
            .unwrap_or(1)
    }

    fn v_max(&self) -> usize {
        self.sampling
            .iter()
            .map(|(_, v)| usize::from(*v))
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

    /// Dimensions `(width, height)` of the plane component `c` must be given as.
    pub fn plane_size(&self, c: usize) -> (usize, usize) {
        let (h, v) = self.sampling[c];
        (
            self.mcus_x() * usize::from(h),
            self.mcus_y() * usize::from(v),
        )
    }
}

/// A Huffman table in the form of a DHT segment.
#[derive(Clone, Debug)]
struct Table {
    /// Number of codes of each length 1 to 16.
    bits: [u8; 16],
    /// Symbols in code order.
    values: Vec<u8>,
    /// `(code, length)` per symbol 0 to 16.
    codes: [(u32, u32); 17],
}

/// Generates length-limited Huffman code lengths from symbol frequencies (T.81 Annex K.2). Symbol
/// 256 is a reserved symbol that keeps the all-ones code unused.
fn make_table(frequencies: &[u64; 17]) -> Table {
    let mut freq = [0i64; 257];
    for (slot, f) in freq.iter_mut().zip(frequencies) {
        *slot = *f as i64;
    }
    freq[256] = 1;
    let mut code_size = [0usize; 257];
    let mut others = [-1i32; 257];
    loop {
        let mut c1 = -1i32;
        let mut least = i64::MAX;
        for (i, f) in freq.iter().enumerate() {
            if *f != 0 && *f <= least {
                least = *f;
                c1 = i as i32;
            }
        }
        let mut c2 = -1i32;
        least = i64::MAX;
        for (i, f) in freq.iter().enumerate() {
            if *f != 0 && *f <= least && i as i32 != c1 {
                least = *f;
                c2 = i as i32;
            }
        }
        if c2 < 0 {
            break;
        }
        freq[c1 as usize] += freq[c2 as usize];
        freq[c2 as usize] = 0;
        let mut a = c1 as usize;
        code_size[a] += 1;
        while others[a] >= 0 {
            a = others[a] as usize;
            code_size[a] += 1;
        }
        others[a] = c2;
        let mut b = c2 as usize;
        code_size[b] += 1;
        while others[b] >= 0 {
            b = others[b] as usize;
            code_size[b] += 1;
        }
    }
    let mut counts = [0i32; 40];
    for size in code_size {
        if size > 0 {
            counts[size] += 1;
        }
    }
    for i in (17..40).rev() {
        while counts[i] > 0 {
            let mut j = i - 2;
            while counts[j] == 0 {
                j -= 1;
            }
            counts[i] -= 2;
            counts[i - 1] += 1;
            counts[j + 1] += 2;
            counts[j] -= 1;
        }
    }
    // Remove the reserved symbol's code (the longest).
    let mut i = 16;
    while counts[i] == 0 {
        i -= 1;
    }
    counts[i] -= 1;

    // Symbols sorted by code length, then value; the reserved symbol is dropped.
    let mut values = Vec::new();
    for length in 1..=39usize {
        for (symbol, size) in code_size.iter().enumerate().take(17) {
            if *size == length {
                values.push(symbol as u8);
            }
        }
    }
    let mut bits = [0u8; 16];
    for (slot, count) in bits.iter_mut().zip(&counts[1..17]) {
        *slot = *count as u8;
    }
    // After the length limiting the symbols still sort in order of their original lengths, which
    // is what Annex K.2 prescribes; assign canonical codes.
    let mut codes = [(0u32, 0u32); 17];
    let mut code = 0u32;
    let mut k = 0usize;
    for length in 1..=16u32 {
        for _ in 0..bits[length as usize - 1] {
            codes[usize::from(values[k])] = (code, length);
            code += 1;
            k += 1;
        }
        code <<= 1;
    }
    Table {
        bits,
        values,
        codes,
    }
}

/// MSB-first bit writer with JPEG byte stuffing.
struct BitWriter {
    out: Vec<u8>,
    acc: u64,
    nbits: u32,
}

impl BitWriter {
    fn put(&mut self, value: u32, bits: u32) {
        if bits == 0 {
            return;
        }
        self.acc = (self.acc << bits) | u64::from(value & ((1u32 << bits) - 1));
        self.nbits += bits;
        while self.nbits >= 8 {
            let byte = (self.acc >> (self.nbits - 8)) as u8;
            self.out.push(byte);
            if byte == 0xFF {
                self.out.push(0);
            }
            self.nbits -= 8;
        }
        self.acc &= (1u64 << self.nbits) - 1;
    }

    /// Pads to a byte boundary with one bits.
    fn align(&mut self) {
        if self.nbits > 0 {
            let pad = 8 - self.nbits;
            self.put((1 << pad) - 1, pad);
        }
    }
}

/// Predictor value of T.81 table H.1 for samples `ra` (left), `rb` (above) and `rc` (above left).
fn predictor_value(selection: u8, ra: i32, rb: i32, rc: i32) -> i32 {
    match selection {
        1 => ra,
        2 => rb,
        3 => rc,
        4 => ra + rb - rc,
        5 => ra + ((rb - rc) >> 1),
        6 => rb + ((ra - rc) >> 1),
        7 => (ra + rb) / 2,
        _ => panic!("predictor {selection} does not exist"),
    }
}

/// One coded difference: component, category and the extra bits.
#[derive(Clone, Copy)]
struct Coded {
    component: u8,
    category: u8,
    extra: u32,
}

/// Encodes `planes` (one per component, sized by [`EncodeParams::plane_size`]) as a lossless JPEG.
///
/// # Panics
///
/// Panics on inconsistent parameters or plane sizes; this is test-support code.
pub fn encode(planes: &[Vec<u16>], params: &EncodeParams) -> Vec<u8> {
    assert_eq!(planes.len(), params.sampling.len());
    let (mcus_x, mcus_y) = (params.mcus_x(), params.mcus_y());
    let total = mcus_x * mcus_y;
    let interval = if params.restart_interval == 0 {
        total
    } else {
        assert_eq!(
            params.restart_interval % mcus_x,
            0,
            "restart intervals are whole MCU rows"
        );
        params.restart_interval
    };
    let pt = u32::from(params.point_transform);
    let base = 1i32 << (u32::from(params.precision) - pt - 1);

    // Pass 1: differences in stream order, grouped by restart interval.
    let mut intervals: Vec<Vec<Coded>> = Vec::new();
    let mut frequencies = vec![[0u64; 17]; planes.len()];
    let mut mcu = 0usize;
    while mcu < total {
        let end = (mcu + interval).min(total);
        let first_mcu_row = mcu / mcus_x;
        let mut fresh = vec![true; planes.len()];
        let mut coded = Vec::new();
        for m in mcu..end {
            let (my, mx) = (m / mcus_x, m % mcus_x);
            for (c, plane) in planes.iter().enumerate() {
                let (h, v) = (
                    usize::from(params.sampling[c].0),
                    usize::from(params.sampling[c].1),
                );
                let (pw, _) = params.plane_size(c);
                let sample = |x: usize, y: usize| i32::from(plane[y * pw + x] >> pt);
                for j in 0..v {
                    let y = my * v + j;
                    for i in 0..h {
                        let x = mx * h + i;
                        let prediction = if fresh[c] {
                            fresh[c] = false;
                            base
                        } else if y == first_mcu_row * v {
                            sample(x - 1, y)
                        } else if x == 0 {
                            if params.sony_420_quirk && c == 0 && v == 2 && j == 0 && y >= 2 {
                                sample(0, y - 2)
                            } else {
                                sample(0, y - 1)
                            }
                        } else {
                            predictor_value(
                                params.predictor,
                                sample(x - 1, y),
                                sample(x, y - 1),
                                sample(x - 1, y - 1),
                            )
                        };
                        let diff = (((sample(x, y) - prediction) + 32768) & 0xFFFF) - 32768;
                        let magnitude = diff.unsigned_abs();
                        let category = if diff == 0 {
                            0
                        } else {
                            32 - magnitude.leading_zeros()
                        };
                        let extra = if diff >= 0 {
                            diff as u32
                        } else {
                            (diff + (1 << category) - 1) as u32
                        };
                        frequencies[c][category as usize] += 1;
                        coded.push(Coded {
                            component: c as u8,
                            category: category as u8,
                            extra,
                        });
                    }
                }
            }
        }
        intervals.push(coded);
        mcu = end;
    }

    // Tables: one per component.
    let tables: Vec<Table> = frequencies
        .iter()
        .map(|f| match params.huffman {
            HuffmanChoice::Optimal => make_table(f),
            HuffmanChoice::Skewed => {
                // Frequencies that double with every category force a chain-shaped tree (the
                // length limit then trims it to 16 bits). They run backwards, so the small
                // categories that real data uses most get the longest codes, which exercises the
                // decoder's slow path.
                let mut skewed = [0u64; 17];
                for (i, slot) in skewed.iter_mut().enumerate() {
                    *slot = 1 << i;
                }
                skewed.reverse();
                make_table(&skewed)
            }
        })
        .collect();

    // Headers.
    let mut out = vec![0xFF, 0xD8];
    for (id, table) in tables.iter().enumerate() {
        out.extend_from_slice(&[0xFF, 0xC4]);
        let length = 2 + 1 + 16 + table.values.len();
        out.extend_from_slice(&(length as u16).to_be_bytes());
        out.push(id as u8);
        out.extend_from_slice(&table.bits);
        out.extend_from_slice(&table.values);
    }
    out.extend_from_slice(&[0xFF, 0xC3]);
    out.extend_from_slice(&((8 + 3 * planes.len()) as u16).to_be_bytes());
    out.push(params.precision);
    out.extend_from_slice(&(params.height as u16).to_be_bytes());
    out.extend_from_slice(&(params.width as u16).to_be_bytes());
    out.push(planes.len() as u8);
    for (c, (h, v)) in params.sampling.iter().enumerate() {
        out.extend_from_slice(&[c as u8 + 1, (h << 4) | v, 0]);
    }
    if params.restart_interval > 0 {
        out.extend_from_slice(&[0xFF, 0xDD, 0, 4]);
        out.extend_from_slice(&(params.restart_interval as u16).to_be_bytes());
    }
    out.extend_from_slice(&[0xFF, 0xDA]);
    out.extend_from_slice(&((6 + 2 * planes.len()) as u16).to_be_bytes());
    out.push(planes.len() as u8);
    for c in 0..planes.len() {
        out.extend_from_slice(&[c as u8 + 1, (c as u8) << 4]);
    }
    out.extend_from_slice(&[params.predictor, 0, params.point_transform]);

    // Entropy-coded data.
    let mut writer = BitWriter {
        out,
        acc: 0,
        nbits: 0,
    };
    let last = intervals.len() - 1;
    for (n, coded) in intervals.iter().enumerate() {
        for item in coded {
            let (code, length) =
                tables[usize::from(item.component)].codes[usize::from(item.category)];
            assert!(length > 0, "category {} has no code", item.category);
            writer.put(code, length);
            if item.category > 0 && item.category < 16 {
                writer.put(item.extra, u32::from(item.category));
            }
        }
        writer.align();
        if n != last {
            writer.out.extend_from_slice(&[0xFF, 0xD0 + (n % 8) as u8]);
        }
    }
    let mut out = writer.out;
    out.extend_from_slice(&[0xFF, 0xD9]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn skewed_tables_reach_sixteen_bit_codes() {
        let table = make_table(&{
            let mut f = [0u64; 17];
            for (i, slot) in f.iter_mut().enumerate() {
                *slot = 1 << i;
            }
            f
        });
        assert!(table.bits[15] > 0, "{:?}", table.bits);
        // Kraft sum stays below one so the all-ones code is free.
        let kraft: f64 = table
            .bits
            .iter()
            .enumerate()
            .map(|(i, n)| f64::from(*n) / f64::from(1u32 << (i + 1)))
            .sum();
        assert!(kraft < 1.0, "{kraft}");
    }

    #[test]
    fn a_single_category_gets_a_one_bit_code() {
        let mut f = [0u64; 17];
        f[3] = 100;
        let table = make_table(&f);
        assert_eq!(table.values, vec![3]);
        assert_eq!(table.bits[0], 1);
    }
}
