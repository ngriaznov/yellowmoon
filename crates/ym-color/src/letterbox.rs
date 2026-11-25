//! Black-bar (letterbox and pillarbox) removal for a reference picture (`docs/34-reference-grade.md` section 6.5, RG-05).
//!
//! A reference still often carries black bars the photo does not have, and a bar is not part of the colour of the picture. The rule
//! takes **uniform bars only**: rows and columns are examined from each edge inward, and a bar is the run of lines that are
//! uniform and nearly black, **with a sharp step where the picture starts**. A night frame is full of dark lines that are picture,
//! so darkness alone is not a bar.
//!
//! The rule, with every constant (section 16.5 gives where each came from):
//!
//! 1. **A bar line** is a row, or a column between the bars already found at the top and the bottom, whose 8-bit codes (the
//!    channels together) have a standard deviation below [`LINE_STD_MAX`] and a mean below [`LINE_MEAN_MAX`] (the POC's two
//!    thresholds). The first run of such lines from an edge is the strict run.
//! 2. **The ringing margin.** JPEG rings next to a hard edge: the last lines of a bar, up to one 4:2:0 block (16 lines) before the
//!    picture, are no longer flat. After the strict run the rule looks on through up to [`RINGING_MARGIN_PX`] lines, as long as they
//!    stay dark (mean below [`RINGING_MEAN_MAX`]), for the picture's edge.
//! 3. **The step.** The picture's edge is the first line in that window whose mean is at least [`STEP_MEAN_CODES`] above the line
//!    before it. No such line, no bar: a night frame's dark lines rise gradually (the largest rise between two lines in the
//!    window on 195 local pictures was 7.0 codes; `docs/34` section 16.5), a real bar meets the picture in one line.
//! 4. **The size.** The bar is at least [`MIN_BAR_PX`] lines and [`MIN_BAR_FRACTION`] of the side, and at most
//!    [`MAX_BAR_FRACTION`] of it (the POC's 35 %). A longer dark run is not a bar and nothing is removed from that side.
//!
//! A picture in which every row (or, between the bars, every column) is a bar line has nothing in it: the bars are not removed and
//! [`find_bars`] says [`Letterbox::Empty`], which the loader reports as "nothing inside the bars".
//!
//! The sums are exact integers, so the verdict on a line does not depend on rounding, and the result is the same on every machine.

/// A bar line's standard deviation of codes is below this (the POC's value, 8-bit codes, channels together).
pub const LINE_STD_MAX: f64 = 2.0;
/// A bar line's mean code is below this (the POC's value).
pub const LINE_MEAN_MAX: f64 = 16.0;
/// A bar is never more than this share of a side (the POC's value).
pub const MAX_BAR_FRACTION: f64 = 0.35;
/// A bar is at least this share of a side.
pub const MIN_BAR_FRACTION: f64 = 0.01;
/// A bar is at least this many lines, whatever the side.
pub const MIN_BAR_PX: usize = 4;
/// How many lines after the strict run are searched for the picture's edge (a 16-line 4:2:0 JPEG block).
pub const RINGING_MARGIN_PX: usize = 16;
/// The picture's first line has a mean at least this many codes above the line before it.
pub const STEP_MEAN_CODES: f64 = 16.0;
/// A line in the ringing window may have a mean up to this and still be taken for the bar's own (the colour that 4:2:0 sampling
/// and the ripples carry into the last lines of a bar); a line above it with no step ends the search.
pub const RINGING_MEAN_MAX: f64 = 32.0;

/// What a sample's 8-bit code is: `u8` is its own, `u16` is scaled to 0 to 255 by rounding.
pub trait Sample: Copy + Send + Sync {
    /// The 8-bit code of this sample.
    fn code8(self) -> u32;
}

impl Sample for u8 {
    fn code8(self) -> u32 {
        u32::from(self)
    }
}

impl Sample for u16 {
    fn code8(self) -> u32 {
        (u32::from(self) * 255 + 32_767) / 65_535
    }
}

/// How many lines of each side are bars.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Bars {
    /// Rows at the top.
    pub top: usize,
    /// Rows at the bottom.
    pub bottom: usize,
    /// Columns at the left.
    pub left: usize,
    /// Columns at the right.
    pub right: usize,
}

impl Bars {
    /// Whether any bar was found.
    #[must_use]
    pub fn any(&self) -> bool {
        self.top + self.bottom + self.left + self.right > 0
    }
}

/// The verdict on a picture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Letterbox {
    /// These bars (possibly none) are to be removed.
    Bars(Bars),
    /// Every line is a bar line: there is no picture inside the bars.
    Empty,
}

/// The sums of a line's 8-bit codes.
#[derive(Clone, Copy, Default)]
struct Line {
    sum: u64,
    sum_sq: u64,
}

impl Line {
    fn add(&mut self, code: u32) {
        let c = u64::from(code);
        self.sum += c;
        self.sum_sq += c * c;
    }
}

/// Whether a line of `n` codes is a bar line (exact integer arithmetic: `var < 4` is `n * sum_sq - sum^2 < 4 n^2`).
fn is_bar_line(l: Line, n: u64) -> bool {
    let limit_var = (LINE_STD_MAX * LINE_STD_MAX) as u64;
    let var_n2 = (n * l.sum_sq).saturating_sub(l.sum * l.sum);
    var_n2 < limit_var * n * n && (l.sum as f64) < LINE_MEAN_MAX * n as f64
}

fn mean(l: Line, n: u64) -> f64 {
    l.sum as f64 / n as f64
}

/// The bar at the start of `lines` (the lines of one side from its edge inward, each of `n` codes): its length (0 for none), and
/// whether every line is a bar line.
fn bar_length(lines: &dyn Fn(usize) -> Line, len: usize, n: u64) -> (usize, bool) {
    let mut k = 0;
    while k < len && is_bar_line(lines(k), n) {
        k += 1;
    }
    if k == len {
        return (0, true);
    }
    if k == 0 {
        return (0, false);
    }
    let cap = (len as f64 * MAX_BAR_FRACTION).floor() as usize;
    let min = MIN_BAR_PX.max((len as f64 * MIN_BAR_FRACTION).ceil() as usize);
    // The picture's edge: the first line from the strict run's end, within the ringing margin, that steps up from the line before
    // it; every line between has to stay dark.
    let last = (k + RINGING_MARGIN_PX).min(len - 1);
    let mut previous = mean(lines(k - 1), n);
    for j in k..=last {
        let m = mean(lines(j), n);
        if m - previous >= STEP_MEAN_CODES {
            return (if (min..=cap).contains(&j) { j } else { 0 }, false);
        }
        if m >= RINGING_MEAN_MAX {
            break;
        }
        previous = m;
    }
    (0, false)
}

/// Finds the black bars of a `width` x `height` picture of `channels` interleaved samples a pixel (1 for grey, 3 for RGB, rows top
/// to bottom).
///
/// # Panics
///
/// When `data` is shorter than `width * height * channels`.
#[must_use]
pub fn find_bars<T: Sample>(data: &[T], width: usize, height: usize, channels: usize) -> Letterbox {
    assert!(data.len() >= width * height * channels);
    if width == 0 || height == 0 || channels == 0 {
        return Letterbox::Empty;
    }
    let stride = width * channels;
    // The rows.
    let mut rows = vec![Line::default(); height];
    for (y, line) in rows.iter_mut().enumerate() {
        for s in &data[y * stride..(y + 1) * stride] {
            line.add(s.code8());
        }
    }
    let n_row = stride as u64;
    let (top, all_rows) = bar_length(&|i| rows[i], height, n_row);
    if all_rows {
        return Letterbox::Empty;
    }
    let (bottom, _) = bar_length(&|i| rows[height - 1 - i], height, n_row);
    // The columns, between the bars found at the top and the bottom.
    let (y0, y1) = (top, height - bottom);
    let mut cols = vec![Line::default(); width];
    for y in y0..y1 {
        let row = &data[y * stride..(y + 1) * stride];
        for (x, px) in row.chunks_exact(channels).enumerate() {
            for s in px {
                cols[x].add(s.code8());
            }
        }
    }
    let n_col = ((y1 - y0) * channels) as u64;
    let (left, all_cols) = bar_length(&|i| cols[i], width, n_col);
    if all_cols {
        return Letterbox::Empty;
    }
    let (right, _) = bar_length(&|i| cols[width - 1 - i], width, n_col);
    Letterbox::Bars(Bars {
        top,
        bottom,
        left,
        right,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A small deterministic generator (SplitMix64) for the synthetic pictures.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        /// Uniform in -1..1.
        fn signed(&mut self) -> f64 {
            (self.next() >> 11) as f64 / (1u64 << 52) as f64 - 1.0
        }
    }

    /// A textured picture: a gradient of mean about `level` with noise of about `noise` codes.
    fn content(w: usize, h: usize, level: f64, noise: f64, rng: &mut Rng) -> Vec<u8> {
        let mut d = Vec::with_capacity(w * h * 3);
        for y in 0..h {
            for x in 0..w {
                for c in 0..3 {
                    let v = level
                        + 40.0 * (x as f64 / w as f64 - 0.5)
                        + 20.0 * (y as f64 / h as f64 - 0.5)
                        + 4.0 * c as f64
                        + noise * rng.signed();
                    d.push(v.clamp(0.0, 255.0) as u8);
                }
            }
        }
        d
    }

    /// `picture` of `w x h` inside bars of the given widths, the bars black.
    fn with_bars(picture: &[u8], w: usize, h: usize, b: Bars) -> (Vec<u8>, usize, usize) {
        let (ow, oh) = (w + b.left + b.right, h + b.top + b.bottom);
        let mut d = vec![0u8; ow * oh * 3];
        for y in 0..h {
            let at = ((y + b.top) * ow + b.left) * 3;
            d[at..at + w * 3].copy_from_slice(&picture[y * w * 3..(y + 1) * w * 3]);
        }
        (d, ow, oh)
    }

    #[test]
    fn a_letterboxed_picture_is_trimmed_exactly() {
        let mut rng = Rng(1);
        let pic = content(320, 134, 110.0, 20.0, &mut rng);
        for bars in [
            Bars {
                top: 53,
                bottom: 53,
                ..Bars::default()
            },
            Bars {
                top: 20,
                bottom: 31,
                ..Bars::default()
            },
            Bars {
                top: 12,
                ..Bars::default()
            },
            Bars {
                bottom: 40,
                ..Bars::default()
            },
        ] {
            let (d, w, h) = with_bars(&pic, 320, 134, bars);
            assert_eq!(find_bars(&d, w, h, 3), Letterbox::Bars(bars));
        }
    }

    #[test]
    fn a_pillarboxed_picture_is_trimmed_exactly() {
        let mut rng = Rng(2);
        let pic = content(240, 320, 100.0, 25.0, &mut rng);
        let bars = Bars {
            left: 60,
            right: 77,
            ..Bars::default()
        };
        let (d, w, h) = with_bars(&pic, 240, 320, bars);
        assert_eq!(find_bars(&d, w, h, 3), Letterbox::Bars(bars));
        // A window box: bars on all four sides.
        let bars = Bars {
            top: 24,
            bottom: 24,
            left: 50,
            right: 50,
        };
        let (d, w, h) = with_bars(&pic, 240, 320, bars);
        assert_eq!(find_bars(&d, w, h, 3), Letterbox::Bars(bars));
    }

    #[test]
    fn a_picture_without_bars_loses_nothing() {
        let mut rng = Rng(3);
        let pic = content(300, 200, 90.0, 20.0, &mut rng);
        assert_eq!(
            find_bars(&pic, 300, 200, 3),
            Letterbox::Bars(Bars::default())
        );
        // Grey, and 16-bit.
        let grey: Vec<u8> = pic.iter().step_by(3).copied().collect();
        assert_eq!(
            find_bars(&grey, 300, 200, 1),
            Letterbox::Bars(Bars::default())
        );
        let wide: Vec<u16> = pic.iter().map(|&v| u16::from(v) * 257).collect();
        assert_eq!(
            find_bars(&wide, 300, 200, 3),
            Letterbox::Bars(Bars::default())
        );
    }

    #[test]
    fn sixteen_bit_pictures_are_read_by_their_eight_bit_codes() {
        let mut rng = Rng(4);
        let pic = content(200, 100, 120.0, 20.0, &mut rng);
        let bars = Bars {
            top: 30,
            bottom: 30,
            ..Bars::default()
        };
        let (d, w, h) = with_bars(&pic, 200, 100, bars);
        let wide: Vec<u16> = d.iter().map(|&v| u16::from(v) * 257).collect();
        assert_eq!(find_bars(&wide, w, h, 3), Letterbox::Bars(bars));
    }

    /// A night frame: a dark sky with sensor noise that rises towards the horizon, a lit town at the bottom; a quarter of the rows
    /// are dark enough and flat enough to pass the POC's two tests, and none of them is a bar.
    #[test]
    fn a_night_frame_with_many_dark_rows_is_not_trimmed() {
        let (w, h) = (400usize, 300usize);
        let mut rng = Rng(5);
        let mut d = Vec::with_capacity(w * h * 3);
        for y in 0..h {
            // The mean climbs from 0.3 to 14 and the noise from 0.3 to 1.9 over the rows.
            let t = y as f64 / (h as f64 * 0.8);
            for _ in 0..w {
                for _ in 0..3 {
                    let v = if y < (h * 4) / 5 {
                        0.3 + 14.0 * t.min(1.0).powi(3)
                            + (0.3 + 1.6 * t.min(1.0)) * rng.signed() * 1.7
                    } else {
                        90.0 + 60.0 * rng.signed()
                    };
                    d.push(v.clamp(0.0, 255.0).round() as u8);
                }
            }
        }
        // The POC's two thresholds alone would take a long run of rows from the top.
        let strict = (0..h)
            .take_while(|&y| {
                let row = &d[y * w * 3..(y + 1) * w * 3];
                let n = row.len() as f64;
                let m = row.iter().map(|&v| f64::from(v)).sum::<f64>() / n;
                let var = row.iter().map(|&v| (f64::from(v) - m).powi(2)).sum::<f64>() / n;
                var.sqrt() < LINE_STD_MAX && m < LINE_MEAN_MAX
            })
            .count();
        assert!(
            strict > 100,
            "the frame must tempt the POC's rule: {strict}"
        );
        assert_eq!(find_bars(&d, w, h, 3), Letterbox::Bars(Bars::default()));
    }

    /// A night frame with a dark picture edge that is sharper than a ramp but not a step: a left edge of black set that fades in
    /// about 24 columns.
    #[test]
    fn a_dark_edge_that_fades_is_not_a_bar() {
        let (w, h) = (300usize, 200usize);
        let mut rng = Rng(6);
        let mut d = vec![0u8; w * h * 3];
        for y in 0..h {
            for x in 0..w {
                let fade = ((x as f64 - 90.0) / 24.0).clamp(0.0, 1.0);
                for c in 0..3 {
                    let v = 0.4 + fade * (70.0 + 50.0 * rng.signed()) + 0.5 * rng.signed();
                    d[(y * w + x) * 3 + c] = v.clamp(0.0, 255.0).round() as u8;
                }
            }
        }
        assert_eq!(find_bars(&d, w, h, 3), Letterbox::Bars(Bars::default()));
    }

    #[test]
    fn a_flat_dark_picture_is_not_trimmed_into_nothing() {
        // Black, near black and a dark grey with a little noise: every line is a bar line, no bar is taken, and the picture is
        // reported as empty for the loader to refuse.
        let mut rng = Rng(7);
        for level in [0.0, 6.0, 11.0] {
            let d: Vec<u8> = (0..120 * 90 * 3)
                .map(|_| (level + 0.6 * rng.signed()).clamp(0.0, 255.0).round() as u8)
                .collect();
            assert_eq!(find_bars(&d, 120, 90, 3), Letterbox::Empty, "{level}");
        }
        // Black bars round a picture that is itself flat and dark: the columns between the rows' bars are all dark.
        let d = vec![0u8; 60 * 40 * 3];
        assert_eq!(find_bars(&d, 60, 40, 3), Letterbox::Empty);
    }

    #[test]
    fn a_dark_run_longer_than_the_limit_is_not_a_bar() {
        let mut rng = Rng(8);
        // 130 of 200 rows are black above a bright picture: 65 % of the side.
        let pic = content(100, 70, 130.0, 20.0, &mut rng);
        let (d, w, h) = with_bars(
            &pic,
            100,
            70,
            Bars {
                top: 130,
                ..Bars::default()
            },
        );
        assert_eq!(find_bars(&d, w, h, 3), Letterbox::Bars(Bars::default()));
        // Exactly at the limit (70 of 200 rows) it is one.
        let pic = content(100, 130, 130.0, 20.0, &mut rng);
        let (d, w, h) = with_bars(
            &pic,
            100,
            130,
            Bars {
                top: 70,
                ..Bars::default()
            },
        );
        assert_eq!(
            find_bars(&d, w, h, 3),
            Letterbox::Bars(Bars {
                top: 70,
                ..Bars::default()
            })
        );
        let (d, w, h) = with_bars(
            &pic,
            100,
            130,
            Bars {
                top: 71,
                ..Bars::default()
            },
        );
        assert_eq!(find_bars(&d, w, h, 3), Letterbox::Bars(Bars::default()));
    }

    #[test]
    fn a_sliver_is_not_a_bar() {
        let mut rng = Rng(9);
        let pic = content(200, 200, 110.0, 20.0, &mut rng);
        for top in [1, 2, 3] {
            let (d, w, h) = with_bars(
                &pic,
                200,
                200,
                Bars {
                    top,
                    ..Bars::default()
                },
            );
            assert_eq!(
                find_bars(&d, w, h, 3),
                Letterbox::Bars(Bars::default()),
                "{top}"
            );
        }
        let (d, w, h) = with_bars(
            &pic,
            200,
            200,
            Bars {
                top: 4,
                ..Bars::default()
            },
        );
        assert_eq!(
            find_bars(&d, w, h, 3),
            Letterbox::Bars(Bars {
                top: 4,
                ..Bars::default()
            })
        );
        // One per cent of a tall side is the floor there: on a side of 1213 rows, 13 pass and 12 do not.
        let tall = content(40, 1200, 110.0, 20.0, &mut rng);
        let (d, w, h) = with_bars(
            &tall,
            40,
            1200,
            Bars {
                top: 13,
                ..Bars::default()
            },
        );
        assert!(matches!(find_bars(&d, w, h, 3), Letterbox::Bars(b) if b.top == 13));
        let (d, w, h) = with_bars(
            &tall,
            40,
            1200,
            Bars {
                top: 12,
                ..Bars::default()
            },
        );
        assert!(matches!(find_bars(&d, w, h, 3), Letterbox::Bars(b) if b.top == 0));
    }

    #[test]
    fn a_bar_with_a_few_noisy_lines_before_the_picture_is_taken_whole() {
        // The ringing of JPEG: the last lines of the bar are no longer flat. The step finds the picture's first line.
        let mut rng = Rng(10);
        let (w, bar, ph) = (160usize, 40usize, 100usize);
        let pic = content(w, ph, 150.0, 25.0, &mut rng);
        for ring in [0usize, 1, 5, 8, 14, 16] {
            let bars = Bars {
                top: bar,
                bottom: bar,
                ..Bars::default()
            };
            let (mut d, ow, oh) = with_bars(&pic, w, ph, bars);
            for edge_row in [bar - 1, bar + ph] {
                for i in 0..ring {
                    let y = if edge_row < bar {
                        edge_row - i
                    } else {
                        edge_row + i
                    };
                    // Decaying ripples with a mean of a few codes and a spread above 2.
                    let amp = 9.0 * 0.8f64.powi(i as i32);
                    for x in 0..ow {
                        for c in 0..3 {
                            let v = 4.0 + amp * rng.signed();
                            d[(y * ow + x) * 3 + c] = v.clamp(0.0, 255.0).round() as u8;
                        }
                    }
                }
            }
            assert_eq!(
                find_bars(&d, ow, oh, 3),
                Letterbox::Bars(bars),
                "ring {ring}"
            );
        }
    }

    #[test]
    fn a_bar_next_to_a_dark_picture_is_left_alone() {
        // The picture's first lines are about as dark as the bar: no step, no bar (the lines are harmless).
        let mut rng = Rng(11);
        let pic: Vec<u8> = (0..160 * 100 * 3)
            .map(|_| (9.0 + 6.0 * rng.signed()).round() as u8)
            .collect();
        let bars = Bars {
            top: 30,
            bottom: 30,
            ..Bars::default()
        };
        let (d, w, h) = with_bars(&pic, 160, 100, bars);
        assert_eq!(find_bars(&d, w, h, 3), Letterbox::Bars(Bars::default()));
    }
}
