//! `.cube` parser and validator for 1D and 3D LUTs, including `LUT_3D_INPUT_RANGE` (`docs/02-colour-pipeline.md` section C10, `docs/08-looks.md`).
//!
//! Reads the two common dialects:
//!
//! * **Adobe / IRIDAS**: `TITLE "name"`, `LUT_3D_SIZE n` (or `LUT_1D_SIZE n`), `DOMAIN_MIN r g b`,
//!   `DOMAIN_MAX r g b`.
//! * **DaVinci Resolve**: `LUT_1D_SIZE`, `LUT_3D_SIZE`, `LUT_1D_INPUT_RANGE lo hi` and
//!   `LUT_3D_INPUT_RANGE lo hi`. A file may hold a 1D table **followed by** a 3D table; the 1D
//!   table is then a shaper applied before the cube.
//!
//! Both: `#` comments (whole line or trailing), blank lines, LF, CRLF or CR line ends, a UTF-8
//! byte-order mark, case-insensitive keywords, data rows of three finite numbers, red varying
//! fastest. A 3D size must be 2 to 129 ([`MAX_3D_SIZE`]); a 1D size 2 to 65536 ([`MAX_1D_SIZE`]).
//!
//! # Domains
//!
//! For a lone 3D table, `DOMAIN_MIN/MAX` or `LUT_3D_INPUT_RANGE` set the cube's domain (default
//! `[0, 1]`). For a lone 1D table, `DOMAIN_MIN/MAX` or `LUT_1D_INPUT_RANGE`. With both tables,
//! `DOMAIN_MIN/MAX` and `LUT_1D_INPUT_RANGE` apply to the shaper (the first stage) and
//! `LUT_3D_INPUT_RANGE` to the cube. Giving two keywords for the same table is an error unless
//! they agree.
//!
//! # Robustness
//!
//! [`parse`] never panics. Allocation is bounded by the declared size (at most 129^3 rows) and by
//! the input length, files over [`MAX_FILE_BYTES`] are refused, and every problem is a typed
//! [`CubeError`] with the 1-based line number where there is one. This is the fuzz target in
//! `crates/ym-color/fuzz`.

use crate::consts::Vec3;
use crate::lut::{Lut, Lut1d, Lut3d};

/// Smallest 3D size accepted.
pub const MIN_3D_SIZE: usize = 2;
/// Largest 3D size accepted.
pub const MAX_3D_SIZE: usize = 129;
/// Smallest 1D size accepted.
pub const MIN_1D_SIZE: usize = 2;
/// Largest 1D size accepted.
pub const MAX_1D_SIZE: usize = 65_536;
/// Largest file [`parse_bytes`] will read, in bytes (a 129-point cube is about 60 MB).
pub const MAX_FILE_BYTES: usize = 256 * 1024 * 1024;
/// Titles are truncated to this many characters.
pub const MAX_TITLE_CHARS: usize = 1024;

/// The most arguments any keyword takes (`DOMAIN_MIN` and `DOMAIN_MAX` take three numbers).
const MAX_KEYWORD_ARGS: usize = 3;
/// Byte length of the longest keyword (`LUT_3D_INPUT_RANGE`, 18) rounded up; longer words are unknown.
const MAX_KEYWORD_BYTES: usize = 32;

/// Why a `.cube` file was rejected.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum CubeError {
    /// The input is larger than [`MAX_FILE_BYTES`].
    #[error("the file is {bytes} bytes, more than the {limit} byte limit")]
    TooLarge {
        /// Size of the input.
        bytes: usize,
        /// The limit.
        limit: usize,
    },
    /// The input is not valid UTF-8 text.
    #[error("the file is not valid UTF-8 text")]
    NotUtf8,
    /// A keyword this parser does not know.
    #[error("line {line}: unknown keyword {keyword:?}")]
    UnknownKeyword {
        /// 1-based line number.
        line: usize,
        /// The keyword as written.
        keyword: String,
    },
    /// A keyword with the wrong number or kind of arguments.
    #[error("line {line}: {keyword} needs {expected}")]
    BadArguments {
        /// 1-based line number.
        line: usize,
        /// The keyword.
        keyword: &'static str,
        /// What it expects, for the message.
        expected: &'static str,
    },
    /// The same keyword (or two keywords for the same setting) twice.
    #[error("line {line}: {keyword} is given more than once")]
    Duplicate {
        /// 1-based line number.
        line: usize,
        /// The keyword.
        keyword: &'static str,
    },
    /// A LUT size outside the supported range.
    #[error("line {line}: {keyword} {size} is outside {min} to {max}")]
    SizeOutOfRange {
        /// 1-based line number.
        line: usize,
        /// The keyword.
        keyword: &'static str,
        /// The size as written (saturated if it did not fit).
        size: u64,
        /// Smallest allowed.
        min: usize,
        /// Largest allowed.
        max: usize,
    },
    /// A keyword line after the first data row.
    #[error("line {line}: header keyword after LUT data")]
    KeywordAfterData {
        /// 1-based line number.
        line: usize,
    },
    /// A data row before any `LUT_1D_SIZE` or `LUT_3D_SIZE`.
    #[error("line {line}: LUT data before LUT_1D_SIZE or LUT_3D_SIZE")]
    DataBeforeSize {
        /// 1-based line number.
        line: usize,
    },
    /// A data row that is not exactly three numbers.
    #[error("line {line}: a data row must be three numbers")]
    BadDataRow {
        /// 1-based line number.
        line: usize,
    },
    /// A number that is NaN or infinite (or overflows to infinity).
    #[error("line {line}: value is not finite")]
    NonFinite {
        /// 1-based line number.
        line: usize,
    },
    /// More data rows than the sizes declare.
    #[error("line {line}: more data rows than the declared {expected}")]
    TooManyEntries {
        /// 1-based line number of the first extra row.
        line: usize,
        /// Rows expected in total.
        expected: usize,
    },
    /// Fewer data rows than the sizes declare.
    #[error("the file has {found} data rows, {expected} expected")]
    TooFewEntries {
        /// Rows expected in total.
        expected: usize,
        /// Rows found.
        found: usize,
    },
    /// Neither `LUT_1D_SIZE` nor `LUT_3D_SIZE` was given.
    #[error("the file has no LUT_1D_SIZE or LUT_3D_SIZE")]
    MissingSize,
    /// A domain where the minimum is not below the maximum, or two domain keywords that disagree.
    #[error("the input domain is empty or contradictory")]
    BadDomain,
    /// The file is a valid 1D LUT but a 3D LUT is required.
    #[error("the file holds only a 1D LUT; a 3D LUT is required")]
    NoThreeDimensionalLut,
}

/// A parsed `.cube` file.
#[derive(Debug, Clone, PartialEq)]
pub struct CubeFile {
    /// `TITLE`, when present (truncated to [`MAX_TITLE_CHARS`]).
    pub title: Option<String>,
    /// The 1D table: a shaper when `lut3d` is present, otherwise the whole LUT.
    pub lut1d: Option<Lut1d>,
    /// The 3D table.
    pub lut3d: Option<Lut3d>,
}

impl CubeFile {
    /// Converts to a shaper-plus-cube [`Lut`].
    ///
    /// # Errors
    ///
    /// [`CubeError::NoThreeDimensionalLut`] when the file only has a 1D table.
    pub fn into_lut(self) -> Result<Lut, CubeError> {
        match self.lut3d {
            Some(cube) => Ok(Lut {
                title: self.title,
                shaper: self.lut1d,
                cube,
            }),
            None => Err(CubeError::NoThreeDimensionalLut),
        }
    }
}

/// Parses `.cube` text and requires a 3D LUT (a 1D shaper in front of it is allowed).
///
/// # Errors
///
/// Any [`CubeError`]; [`CubeError::NoThreeDimensionalLut`] for a 1D-only file.
pub fn parse_lut(text: &str) -> Result<Lut, CubeError> {
    parse(text)?.into_lut()
}

/// Parses raw bytes: checks the size limit and UTF-8, then [`parse`].
///
/// # Errors
///
/// [`CubeError::TooLarge`], [`CubeError::NotUtf8`] or anything [`parse`] returns.
pub fn parse_bytes(bytes: &[u8]) -> Result<CubeFile, CubeError> {
    if bytes.len() > MAX_FILE_BYTES {
        return Err(CubeError::TooLarge {
            bytes: bytes.len(),
            limit: MAX_FILE_BYTES,
        });
    }
    let text = std::str::from_utf8(bytes).map_err(|_| CubeError::NotUtf8)?;
    parse(text)
}

/// Iterates `(1-based line number, text)` over LF, CRLF and bare-CR line ends without allocating.
struct Lines<'a> {
    rest: &'a str,
    line: usize,
    done: bool,
}

impl<'a> Lines<'a> {
    fn new(text: &'a str) -> Self {
        Self {
            rest: text,
            line: 0,
            done: false,
        }
    }
}

impl<'a> Iterator for Lines<'a> {
    type Item = (usize, &'a str);

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        self.line += 1;
        match self.rest.find(['\n', '\r']) {
            Some(i) => {
                let bytes = self.rest.as_bytes();
                let skip = if bytes[i] == b'\r' && bytes.get(i + 1) == Some(&b'\n') {
                    2
                } else {
                    1
                };
                let line = &self.rest[..i];
                self.rest = &self.rest[i + skip..];
                Some((self.line, line))
            }
            None => {
                self.done = true;
                Some((self.line, self.rest))
            }
        }
    }
}

fn looks_numeric(word: &str) -> bool {
    word.parse::<f64>().is_ok()
        || word.starts_with(|c: char| c.is_ascii_digit() || matches!(c, '+' | '-' | '.'))
}

fn parse_f64(token: &str) -> Option<f64> {
    token.parse::<f64>().ok()
}

/// Parses `n` finite numbers from `tokens`, or `None` (wrong count, not a number, or not finite).
fn numbers<const N: usize>(tokens: &[&str]) -> Option<[f64; N]> {
    if tokens.len() != N {
        return None;
    }
    let mut out = [0.0; N];
    for (slot, t) in out.iter_mut().zip(tokens) {
        let v = parse_f64(t)?;
        if !v.is_finite() {
            return None;
        }
        *slot = v;
    }
    Some(out)
}

/// A keyword that sets a scalar size: parses and range-checks it.
fn parse_size(
    line: usize,
    keyword: &'static str,
    args: &[&str],
    min: usize,
    max: usize,
) -> Result<usize, CubeError> {
    let [token] = args else {
        return Err(CubeError::BadArguments {
            line,
            keyword,
            expected: "one integer",
        });
    };
    let value: u64 = match token.parse::<u64>() {
        Ok(v) => v,
        // A well-formed integer too big for u64 is out of range; anything else is a bad argument.
        Err(e) if matches!(e.kind(), std::num::IntErrorKind::PosOverflow) => u64::MAX,
        Err(_) => {
            return Err(CubeError::BadArguments {
                line,
                keyword,
                expected: "one integer",
            });
        }
    };
    match usize::try_from(value) {
        Ok(v) if (min..=max).contains(&v) => Ok(v),
        _ => Err(CubeError::SizeOutOfRange {
            line,
            keyword,
            size: value,
            min,
            max,
        }),
    }
}

fn parse_range(line: usize, keyword: &'static str, args: &[&str]) -> Result<(f64, f64), CubeError> {
    let bad = CubeError::BadArguments {
        line,
        keyword,
        expected: "two finite numbers",
    };
    let [lo, hi] = numbers::<2>(args).ok_or(bad)?;
    Ok((lo, hi))
}

fn parse_domain(line: usize, keyword: &'static str, args: &[&str]) -> Result<Vec3, CubeError> {
    numbers::<3>(args).ok_or(CubeError::BadArguments {
        line,
        keyword,
        expected: "three finite numbers",
    })
}

fn title_of(rest: &str) -> String {
    let rest = rest.trim();
    let inner = match (rest.find('"'), rest.rfind('"')) {
        (Some(a), Some(b)) if b > a => &rest[a + 1..b],
        _ => rest.trim_matches('"'),
    };
    inner.chars().take(MAX_TITLE_CHARS).collect()
}

/// Which part of the header a domain keyword set.
#[derive(Default)]
struct Domains {
    min: Option<Vec3>,
    max: Option<Vec3>,
    range_1d: Option<(f64, f64)>,
    range_3d: Option<(f64, f64)>,
}

fn resolve_domain(
    explicit: Option<(Option<Vec3>, Option<Vec3>)>,
    range: Option<(f64, f64)>,
) -> Result<(Vec3, Vec3), CubeError> {
    let (mut lo, mut hi) = ([0.0; 3], [1.0; 3]);
    let mut seen_range = false;
    if let Some((min, max)) = explicit {
        if let Some(m) = min {
            lo = m;
        }
        if let Some(m) = max {
            hi = m;
        }
        seen_range = min.is_some() || max.is_some();
    }
    if let Some((a, b)) = range {
        if seen_range {
            // Two ways of saying the same thing must agree.
            let (explicit_lo, explicit_hi) = (lo, hi);
            if explicit_lo != [a; 3] || explicit_hi != [b; 3] {
                return Err(CubeError::BadDomain);
            }
        }
        lo = [a; 3];
        hi = [b; 3];
    }
    if (0..3).all(|c| lo[c] < hi[c]) {
        Ok((lo, hi))
    } else {
        Err(CubeError::BadDomain)
    }
}

/// Parses `.cube` text. See the module documentation for the accepted dialects.
///
/// # Errors
///
/// A [`CubeError`] describing the first problem found.
pub fn parse(text: &str) -> Result<CubeFile, CubeError> {
    let text = text.strip_prefix('\u{feff}').unwrap_or(text);
    let mut title: Option<String> = None;
    let mut size_1d: Option<usize> = None;
    let mut size_3d: Option<usize> = None;
    let mut domains = Domains::default();
    let mut rows: Vec<Vec3> = Vec::new();
    let mut expected_rows = 0_usize;
    let mut capacity_known = false;
    let mut first_data_line = 0_usize;

    for (line, content) in Lines::new(text) {
        let trimmed = content.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let first_word = trimmed.split_ascii_whitespace().next().unwrap_or("");
        let is_title = first_word.eq_ignore_ascii_case("TITLE");
        // TITLE keeps everything after the keyword (a '#' may sit inside the quotes); all other
        // lines lose a trailing comment.
        let body = if is_title {
            trimmed
        } else {
            trimmed.split('#').next().unwrap_or("").trim()
        };
        if body.is_empty() {
            continue;
        }
        let mut words = body.split_ascii_whitespace();
        let word = words.next().unwrap_or("");

        // A data row starts with something that looks like a number (digits, a sign or a point), or
        // parses as one (`nan`, `inf`: rejected below as non-finite with a clear message).
        if looks_numeric(word) {
            if !capacity_known {
                return Err(CubeError::DataBeforeSize { line });
            }
            if first_data_line == 0 {
                first_data_line = line;
            }
            let (Some(g_text), Some(b_text), None) = (words.next(), words.next(), words.next())
            else {
                return Err(CubeError::BadDataRow { line });
            };
            let (Some(r), Some(g), Some(b)) =
                (parse_f64(word), parse_f64(g_text), parse_f64(b_text))
            else {
                return Err(CubeError::BadDataRow { line });
            };
            if !(r.is_finite() && g.is_finite() && b.is_finite()) {
                return Err(CubeError::NonFinite { line });
            }
            if rows.len() >= expected_rows {
                return Err(CubeError::TooManyEntries {
                    line,
                    expected: expected_rows,
                });
            }
            rows.push([r, g, b]);
            continue;
        }

        if first_data_line != 0 {
            return Err(CubeError::KeywordAfterData { line });
        }
        // Keep at most one argument more than any keyword takes (DOMAIN_* takes three): a surplus
        // argument is still seen, but a hostile line of millions of words costs no memory.
        let args: Vec<&str> = words.take(MAX_KEYWORD_ARGS + 1).collect();
        let args = args.as_slice();
        // No keyword is longer than this, so a longer first word is unknown without copying it.
        let upper = if word.len() <= MAX_KEYWORD_BYTES {
            word.to_ascii_uppercase()
        } else {
            String::new()
        };
        match upper.as_str() {
            "TITLE" => {
                if title.is_some() {
                    return Err(CubeError::Duplicate {
                        line,
                        keyword: "TITLE",
                    });
                }
                title = Some(title_of(&trimmed[word.len()..]));
            }
            "LUT_1D_SIZE" => {
                if size_1d.is_some() {
                    return Err(CubeError::Duplicate {
                        line,
                        keyword: "LUT_1D_SIZE",
                    });
                }
                size_1d = Some(parse_size(
                    line,
                    "LUT_1D_SIZE",
                    args,
                    MIN_1D_SIZE,
                    MAX_1D_SIZE,
                )?);
            }
            "LUT_3D_SIZE" => {
                if size_3d.is_some() {
                    return Err(CubeError::Duplicate {
                        line,
                        keyword: "LUT_3D_SIZE",
                    });
                }
                size_3d = Some(parse_size(
                    line,
                    "LUT_3D_SIZE",
                    args,
                    MIN_3D_SIZE,
                    MAX_3D_SIZE,
                )?);
            }
            "DOMAIN_MIN" => {
                if domains.min.is_some() {
                    return Err(CubeError::Duplicate {
                        line,
                        keyword: "DOMAIN_MIN",
                    });
                }
                domains.min = Some(parse_domain(line, "DOMAIN_MIN", args)?);
            }
            "DOMAIN_MAX" => {
                if domains.max.is_some() {
                    return Err(CubeError::Duplicate {
                        line,
                        keyword: "DOMAIN_MAX",
                    });
                }
                domains.max = Some(parse_domain(line, "DOMAIN_MAX", args)?);
            }
            "LUT_1D_INPUT_RANGE" => {
                if domains.range_1d.is_some() {
                    return Err(CubeError::Duplicate {
                        line,
                        keyword: "LUT_1D_INPUT_RANGE",
                    });
                }
                domains.range_1d = Some(parse_range(line, "LUT_1D_INPUT_RANGE", args)?);
            }
            "LUT_3D_INPUT_RANGE" => {
                if domains.range_3d.is_some() {
                    return Err(CubeError::Duplicate {
                        line,
                        keyword: "LUT_3D_INPUT_RANGE",
                    });
                }
                domains.range_3d = Some(parse_range(line, "LUT_3D_INPUT_RANGE", args)?);
            }
            _ => {
                return Err(CubeError::UnknownKeyword {
                    line,
                    keyword: word.chars().take(64).collect(),
                });
            }
        }

        // Whenever a size keyword has been seen, the row budget (and so the allocation bound) is known.
        if size_1d.is_some() || size_3d.is_some() {
            capacity_known = true;
            expected_rows = size_1d.unwrap_or(0) + size_3d.map_or(0, |n| n * n * n);
        }
    }

    if !capacity_known {
        return Err(CubeError::MissingSize);
    }
    if rows.len() != expected_rows {
        return Err(CubeError::TooFewEntries {
            expected: expected_rows,
            found: rows.len(),
        });
    }

    let explicit = if domains.min.is_some() || domains.max.is_some() {
        Some((domains.min, domains.max))
    } else {
        None
    };
    let (lut1d, lut3d) = match (size_1d, size_3d) {
        (Some(n1), Some(n3)) => {
            // Shaper first (DOMAIN_* and LUT_1D_INPUT_RANGE), then the cube (LUT_3D_INPUT_RANGE).
            let (lo1, hi1) = resolve_domain(explicit, domains.range_1d)?;
            let (lo3, hi3) = resolve_domain(None, domains.range_3d)?;
            let cube_rows = rows.split_off(n1);
            let shaper = Lut1d::new(n1, lo1, hi1, rows).map_err(|_| CubeError::BadDomain)?;
            let cube = Lut3d::new(n3, lo3, hi3, cube_rows).map_err(|_| CubeError::BadDomain)?;
            (Some(shaper), Some(cube))
        }
        (Some(n1), None) => {
            if domains.range_3d.is_some() {
                return Err(CubeError::BadDomain);
            }
            let (lo, hi) = resolve_domain(explicit, domains.range_1d)?;
            (
                Some(Lut1d::new(n1, lo, hi, rows).map_err(|_| CubeError::BadDomain)?),
                None,
            )
        }
        (None, Some(n3)) => {
            if domains.range_1d.is_some() {
                return Err(CubeError::BadDomain);
            }
            let (lo, hi) = resolve_domain(explicit, domains.range_3d)?;
            (
                None,
                Some(Lut3d::new(n3, lo, hi, rows).map_err(|_| CubeError::BadDomain)?),
            )
        }
        (None, None) => return Err(CubeError::MissingSize),
    };
    Ok(CubeFile {
        title,
        lut1d,
        lut3d,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity_text(n: usize) -> String {
        Lut3d::identity(n).unwrap().to_cube_string("id")
    }

    #[test]
    fn round_trips_a_written_cube_exactly() {
        let lut = Lut3d::bake(5, [-0.1, 0.0, 0.0], [1.2, 1.0, 1.5], |c| {
            [c[0] * c[1] + 0.1, c[2].sqrt(), (c[0] * 3.0).sin()]
        })
        .unwrap();
        let parsed = parse(&lut.to_cube_string("round trip")).unwrap();
        assert_eq!(parsed.title.as_deref(), Some("round trip"));
        assert_eq!(parsed.lut3d.as_ref(), Some(&lut));
        assert!(parsed.lut1d.is_none());
    }

    #[test]
    fn accepts_crlf_cr_bom_comments_and_lowercase() {
        let base = identity_text(2);
        let crlf = base.replace('\n', "\r\n");
        let cr = base.replace('\n', "\r");
        let bom = format!("\u{feff}{base}");
        let noisy = format!(
            "# header\n\n{}  # trailing\n",
            base.replace("DOMAIN_MAX 1 1 1", "domain_max 1 1 1")
        );
        for text in [crlf, cr, bom, noisy] {
            assert_eq!(parse(&text).unwrap().lut3d.unwrap().size(), 2);
        }
    }

    #[test]
    fn reports_line_numbers_for_both_line_ends() {
        let lf = "LUT_3D_SIZE 2\nBOGUS 1\n0 0 0\n";
        assert!(matches!(
            parse(lf),
            Err(CubeError::UnknownKeyword { line: 2, .. })
        ));
        let after = "LUT_3D_SIZE 2\n0 0 0\nbad row here\n";
        assert!(matches!(
            parse(after),
            Err(CubeError::KeywordAfterData { line: 3 })
        ));
        let crlf = "LUT_3D_SIZE 2\r\n0 0 0\r\n1 2\r\n";
        assert!(matches!(
            parse(crlf),
            Err(CubeError::BadDataRow { line: 3 })
        ));
    }
}
