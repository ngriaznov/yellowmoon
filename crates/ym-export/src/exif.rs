//! EXIF, built by us from a plain struct (`docs/09-export.md` section Metadata).
//!
//! The writers never copy the source file's EXIF: [`ExifInfo`] carries only what the export is
//! allowed to say (camera, lens, exposure, capture time, optional location), so nothing else (serial
//! numbers, maker notes, thumbnails) can leak. The orientation is always written as 1 because the
//! pixels are upright. With "Remove location from exports" on, [`ExifInfo::tables`] leaves out the GPS
//! directory.
//!
//! [`ExifTables`] is the neutral form: three lists of entries (IFD0 descriptive tags, the Exif IFD and
//! the GPS IFD). [`build_exif_blob`] turns it into a standalone little-endian TIFF structure, which
//! is the payload of a JPEG `APP1 Exif` segment and of the PNG `eXIf` chunk; [`write_directory`]
//! writes the same entries into a `tiff` crate directory (TIFF and DNG files).

use std::io::{Seek, Write};

use serde::{Deserialize, Serialize};
use tiff::encoder::{DirectoryEncoder, Rational as TiffRational, SRational as TiffSRational};
use tiff::encoder::{TiffKindStandard, TiffValue};
use tiff::tags::{Tag, Type};

use crate::error::ExportError;
use crate::options::ColorSpace;

/// An unsigned fraction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Rational {
    /// Numerator.
    pub num: u32,
    /// Denominator, never 0.
    pub den: u32,
}

/// A signed fraction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SRational {
    /// Numerator.
    pub num: i32,
    /// Denominator, never 0.
    pub den: i32,
}

/// The best fraction for `x` with numerator below `max_num` and denominator below `max_den`
/// (continued fractions); exact when the value is a short fraction.
fn best_fraction(x: f64, max_num: u64, max_den: u64) -> Option<(u64, u64)> {
    if !x.is_finite() || x < 0.0 {
        return None;
    }
    let (mut h0, mut k0, mut h1, mut k1) = (0_u64, 1_u64, 1_u64, 0_u64);
    let mut frac = x;
    for _ in 0..40 {
        let a = frac.floor();
        if a > u64::MAX as f64 / 4.0 {
            break;
        }
        let a_u = a as u64;
        let (Some(h2), Some(k2)) = (
            a_u.checked_mul(h1).and_then(|v| v.checked_add(h0)),
            a_u.checked_mul(k1).and_then(|v| v.checked_add(k0)),
        ) else {
            break;
        };
        if h2 > max_num || k2 > max_den {
            break;
        }
        (h0, k0, h1, k1) = (h1, k1, h2, k2);
        let rem = frac - a;
        if rem < 1e-12 || (x - h1 as f64 / k1 as f64).abs() <= x * 1e-10 {
            break;
        }
        frac = 1.0 / rem;
    }
    (k1 != 0).then_some((h1, k1))
}

impl Rational {
    /// The fraction nearest to `x` (0 or more) with a denominator up to `max_den`. `None` if `x` is
    /// negative, not finite, or too large.
    #[must_use]
    pub fn from_f64(x: f64, max_den: u32) -> Option<Self> {
        let (n, d) = best_fraction(x, u64::from(u32::MAX), u64::from(max_den))?;
        Some(Self {
            num: u32::try_from(n).ok()?,
            den: u32::try_from(d).ok()?,
        })
    }

    /// `round(x * den) / den`, for values printed with a fixed number of decimals (f-number, focal length).
    #[must_use]
    pub fn from_decimal(x: f64, den: u32) -> Option<Self> {
        if !x.is_finite() || x < 0.0 {
            return None;
        }
        let n = (x * f64::from(den)).round();
        (n <= f64::from(u32::MAX)).then_some(Self { num: n as u32, den })
    }

    /// The value as a float.
    #[must_use]
    pub fn as_f64(self) -> f64 {
        f64::from(self.num) / f64::from(self.den)
    }
}

impl SRational {
    /// The fraction nearest to `x` with a denominator up to `max_den`.
    #[must_use]
    pub fn from_f64(x: f64, max_den: u32) -> Option<Self> {
        let (n, d) = best_fraction(x.abs(), i32::MAX as u64, u64::from(max_den))?;
        let n = i32::try_from(n).ok()?;
        Some(Self {
            num: if x < 0.0 { -n } else { n },
            den: i32::try_from(d).ok()?,
        })
    }

    /// The value as a float.
    #[must_use]
    pub fn as_f64(self) -> f64 {
        f64::from(self.num) / f64::from(self.den)
    }
}

/// The capture time as the camera recorded it (local time, no time zone database involved).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CaptureTime {
    /// Year, 1900 to 9999.
    pub year: u16,
    /// Month, 1 to 12.
    pub month: u8,
    /// Day, 1 to 31.
    pub day: u8,
    /// Hour, 0 to 23.
    pub hour: u8,
    /// Minute, 0 to 59.
    pub minute: u8,
    /// Second, 0 to 59 (60 is accepted for a leap second).
    pub second: u8,
    /// Milliseconds, 0 to 999, when the camera recorded sub-seconds.
    pub millisecond: Option<u16>,
    /// Offset from UTC in minutes (for example +120), when the camera recorded it.
    pub utc_offset_minutes: Option<i16>,
}

impl CaptureTime {
    /// Whether every field is in range.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        (1900..=9999).contains(&self.year)
            && (1..=12).contains(&self.month)
            && (1..=31).contains(&self.day)
            && self.hour < 24
            && self.minute < 60
            && self.second <= 60
            && self.millisecond.is_none_or(|ms| ms < 1000)
            && self
                .utc_offset_minutes
                .is_none_or(|o| (-14 * 60..=14 * 60).contains(&o))
    }

    /// `YYYY:MM:DD HH:MM:SS`, the EXIF form.
    #[must_use]
    pub fn exif_string(&self) -> String {
        format!(
            "{:04}:{:02}:{:02} {:02}:{:02}:{:02}",
            self.year, self.month, self.day, self.hour, self.minute, self.second
        )
    }

    /// `+HH:MM`, the EXIF `OffsetTime` form.
    #[must_use]
    pub fn offset_string(&self) -> Option<String> {
        self.utc_offset_minutes.map(|o| {
            let sign = if o < 0 { '-' } else { '+' };
            format!("{sign}{:02}:{:02}", o.abs() / 60, o.abs() % 60)
        })
    }

    /// `YYYY-MM-DDTHH:MM:SS[.mmm][+HH:MM]`, the XMP (ISO 8601) form.
    #[must_use]
    pub fn xmp_string(&self) -> String {
        let mut s = format!(
            "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}",
            self.year, self.month, self.day, self.hour, self.minute, self.second
        );
        if let Some(ms) = self.millisecond {
            s.push_str(&format!(".{ms:03}"));
        }
        if let Some(offset) = self.offset_string() {
            s.push_str(&offset);
        }
        s
    }

    /// `YYYY-MM-DD`, for the `{date}` file name token.
    #[must_use]
    pub fn date_string(&self) -> String {
        format!("{:04}-{:02}-{:02}", self.year, self.month, self.day)
    }
}

/// A GPS position in decimal degrees.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Gps {
    /// Latitude, -90 (south) to 90 (north).
    pub latitude_deg: f64,
    /// Longitude, -180 (west) to 180 (east).
    pub longitude_deg: f64,
    /// Altitude above sea level in metres (negative below), when known.
    pub altitude_m: Option<f64>,
}

/// What an export may say about the shot. All fields are optional.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ExifInfo {
    /// Camera make ("SONY").
    pub make: Option<String>,
    /// Camera model ("ILCE-1").
    pub model: Option<String>,
    /// Lens make.
    pub lens_make: Option<String>,
    /// Lens model ("FE 24-70mm F2.8 GM II").
    pub lens_model: Option<String>,
    /// Exposure time in seconds.
    pub exposure_time_s: Option<f64>,
    /// F-number (2.8).
    pub f_number: Option<f64>,
    /// ISO speed.
    pub iso: Option<u32>,
    /// Focal length in millimetres.
    pub focal_length_mm: Option<f64>,
    /// Focal length in 35 mm film terms.
    pub focal_length_35mm: Option<u16>,
    /// Exposure compensation in EV.
    pub exposure_bias_ev: Option<f64>,
    /// When the photo was taken.
    pub capture_time: Option<CaptureTime>,
    /// Where the photo was taken.
    pub gps: Option<Gps>,
}

/// The value of one EXIF/TIFF entry.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    /// ASCII text (the writer adds the terminating NUL).
    Ascii(String),
    /// Unsigned bytes.
    Byte(Vec<u8>),
    /// Raw bytes with no defined type (EXIF version, components configuration).
    Undefined(Vec<u8>),
    /// 16-bit unsigned integers.
    Short(Vec<u16>),
    /// 32-bit unsigned integers.
    Long(Vec<u32>),
    /// Unsigned fractions.
    Rational(Vec<Rational>),
    /// Signed fractions.
    SRational(Vec<SRational>),
}

/// One IFD entry: a tag and its value.
#[derive(Debug, Clone, PartialEq)]
pub struct Entry {
    /// The tag number.
    pub tag: u16,
    /// The value.
    pub value: Value,
}

impl Entry {
    fn new(tag: u16, value: Value) -> Self {
        Self { tag, value }
    }
}

/// Everything about the output file the EXIF refers to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OutputFacts {
    /// Output width in pixels.
    pub width: u32,
    /// Output height in pixels.
    pub height: u32,
    /// The embedded colour space, `None` for a DNG.
    pub color_space: Option<ColorSpace>,
    /// Whether the pixels are stored as YCbCr (a JPEG): then `ComponentsConfiguration` is Y, Cb, Cr
    /// and IFD0 gets the `YCbCrPositioning` tag the Exif specification requires for JPEG.
    pub ycbcr: bool,
}

/// The entries of the three directories.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ExifTables {
    /// IFD0 descriptive tags: Make, Model, Software, DateTime. Structure tags (orientation,
    /// resolution, strips) are added by the file writers.
    pub ifd0: Vec<Entry>,
    /// The Exif sub-IFD.
    pub exif: Vec<Entry>,
    /// The GPS sub-IFD, empty when there is no location or it is removed.
    pub gps: Vec<Entry>,
}

// Tag numbers used below.
const MAKE: u16 = 271;
const MODEL: u16 = 272;
const ORIENTATION: u16 = 274;
const X_RESOLUTION: u16 = 282;
const Y_RESOLUTION: u16 = 283;
const RESOLUTION_UNIT: u16 = 296;
const SOFTWARE: u16 = 305;
const DATE_TIME: u16 = 306;
/// IFD0 pointer to the Exif IFD.
pub const EXIF_IFD_POINTER: u16 = 34665;
/// IFD0 pointer to the GPS IFD.
pub const GPS_IFD_POINTER: u16 = 34853;

fn ascii(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_ascii() && !c.is_ascii_control() {
                c
            } else {
                '?'
            }
        })
        .collect::<String>()
        .trim()
        .to_owned()
}

fn non_empty(s: &Option<String>) -> Option<String> {
    s.as_deref().map(ascii).filter(|t| !t.is_empty())
}

impl ExifInfo {
    /// Builds the entries for an output file. `software` is written as the `Software` tag.
    /// With `remove_location`, no GPS directory is produced.
    #[must_use]
    pub fn tables(&self, facts: &OutputFacts, software: &str, remove_location: bool) -> ExifTables {
        let mut t = ExifTables::default();
        if let Some(make) = non_empty(&self.make) {
            t.ifd0.push(Entry::new(MAKE, Value::Ascii(make)));
        }
        if let Some(model) = non_empty(&self.model) {
            t.ifd0.push(Entry::new(MODEL, Value::Ascii(model)));
        }
        let software = ascii(software);
        if !software.is_empty() {
            t.ifd0.push(Entry::new(SOFTWARE, Value::Ascii(software)));
        }
        if facts.ycbcr {
            // Centred chroma siting, as the JPEG encoder writes it.
            t.ifd0.push(Entry::new(531, Value::Short(vec![1])));
        }
        let capture = self.capture_time.filter(CaptureTime::is_valid);
        if let Some(c) = capture {
            t.ifd0
                .push(Entry::new(DATE_TIME, Value::Ascii(c.exif_string())));
        }

        let e = &mut t.exif;
        if let Some(r) = self
            .exposure_time_s
            .and_then(|s| Rational::from_f64(s, 1_000_000))
        {
            e.push(Entry::new(33434, Value::Rational(vec![r])));
        }
        if let Some(r) = self.f_number.and_then(|f| Rational::from_decimal(f, 10)) {
            e.push(Entry::new(33437, Value::Rational(vec![r])));
        }
        if let Some(iso) = self.iso {
            e.push(Entry::new(34855, Value::Short(vec![iso.min(65535) as u16])));
            if iso > 65535 {
                // SensitivityType 3 = ISO speed, with the real value in ISOSpeed.
                e.push(Entry::new(34864, Value::Short(vec![3])));
                e.push(Entry::new(34867, Value::Long(vec![iso])));
            }
        }
        e.push(Entry::new(36864, Value::Undefined(b"0232".to_vec())));
        if let Some(c) = capture {
            let stamp = c.exif_string();
            e.push(Entry::new(36867, Value::Ascii(stamp.clone())));
            e.push(Entry::new(36868, Value::Ascii(stamp)));
            if let Some(offset) = c.offset_string() {
                for tag in [36880, 36881, 36882] {
                    e.push(Entry::new(tag, Value::Ascii(offset.clone())));
                }
            }
            if let Some(ms) = c.millisecond {
                let sub = format!("{ms:03}");
                e.push(Entry::new(37520, Value::Ascii(sub.clone())));
                e.push(Entry::new(37521, Value::Ascii(sub.clone())));
                e.push(Entry::new(37522, Value::Ascii(sub)));
            }
        }
        if facts.ycbcr {
            e.push(Entry::new(37121, Value::Undefined(vec![1, 2, 3, 0])));
        } else if facts.color_space.is_some() {
            e.push(Entry::new(37121, Value::Undefined(vec![4, 5, 6, 0])));
        }
        if let Some(r) = self
            .exposure_bias_ev
            .and_then(|v| SRational::from_f64(v, 1000))
        {
            e.push(Entry::new(37380, Value::SRational(vec![r])));
        }
        if let Some(r) = self
            .focal_length_mm
            .and_then(|f| Rational::from_decimal(f, 10))
        {
            e.push(Entry::new(37386, Value::Rational(vec![r])));
        }
        e.push(Entry::new(40960, Value::Undefined(b"0100".to_vec())));
        if let Some(space) = facts.color_space {
            e.push(Entry::new(
                40961,
                Value::Short(vec![space.exif_color_space()]),
            ));
        }
        e.push(Entry::new(40962, Value::Long(vec![facts.width])));
        e.push(Entry::new(40963, Value::Long(vec![facts.height])));
        if let Some(mm) = self.focal_length_35mm {
            e.push(Entry::new(41989, Value::Short(vec![mm])));
        }
        if let Some(make) = non_empty(&self.lens_make) {
            e.push(Entry::new(42035, Value::Ascii(make)));
        }
        if let Some(model) = non_empty(&self.lens_model) {
            e.push(Entry::new(42036, Value::Ascii(model)));
        }

        if !remove_location && let Some(g) = &self.gps {
            t.gps = gps_entries(g);
        }
        t
    }
}

/// `(degrees, minutes, seconds)` as three fractions for `value` (non-negative degrees). The seconds
/// are in ten-thousandths. The split is done on whole units, so a value a hair below a minute never
/// produces "60 seconds".
fn dms(value: f64) -> Vec<Rational> {
    const SECOND: u64 = 10_000;
    const MINUTE: u64 = 60 * SECOND;
    const DEGREE: u64 = 60 * MINUTE;
    // At most 180 degrees * 36,000,000: far inside u64, and `as` saturates for anything else.
    let total = (value.max(0.0) * DEGREE as f64).round() as u64;
    let (deg, rest) = (total / DEGREE, total % DEGREE);
    vec![
        Rational {
            num: u32::try_from(deg).unwrap_or(u32::MAX),
            den: 1,
        },
        Rational {
            num: (rest / MINUTE) as u32,
            den: 1,
        },
        Rational {
            num: (rest % MINUTE) as u32,
            den: SECOND as u32,
        },
    ]
}

fn gps_entries(g: &Gps) -> Vec<Entry> {
    let lat_ok = g.latitude_deg.is_finite() && g.latitude_deg.abs() <= 90.0;
    let lon_ok = g.longitude_deg.is_finite() && g.longitude_deg.abs() <= 180.0;
    if !(lat_ok && lon_ok) {
        return Vec::new();
    }
    let mut v = vec![
        Entry::new(0, Value::Byte(vec![2, 3, 0, 0])),
        Entry::new(
            1,
            Value::Ascii(if g.latitude_deg < 0.0 { "S" } else { "N" }.to_owned()),
        ),
        Entry::new(2, Value::Rational(dms(g.latitude_deg.abs()))),
        Entry::new(
            3,
            Value::Ascii(if g.longitude_deg < 0.0 { "W" } else { "E" }.to_owned()),
        ),
        Entry::new(4, Value::Rational(dms(g.longitude_deg.abs()))),
    ];
    if let Some(alt) = g.altitude_m.filter(|a| a.is_finite())
        && let Some(r) = Rational::from_decimal(alt.abs(), 100)
    {
        v.push(Entry::new(5, Value::Byte(vec![u8::from(alt < 0.0)])));
        v.push(Entry::new(6, Value::Rational(vec![r])));
    }
    v
}

// ---- standalone TIFF blob ---------------------------------------------------------------

impl Value {
    fn type_code(&self) -> u16 {
        match self {
            Self::Byte(_) => 1,
            Self::Ascii(_) => 2,
            Self::Short(_) => 3,
            Self::Long(_) => 4,
            Self::Rational(_) => 5,
            Self::Undefined(_) => 7,
            Self::SRational(_) => 10,
        }
    }

    /// Count and little-endian bytes.
    fn encode(&self) -> (u32, Vec<u8>) {
        match self {
            Self::Ascii(s) => {
                let mut b = s.as_bytes().to_vec();
                b.push(0);
                (b.len() as u32, b)
            }
            Self::Byte(v) | Self::Undefined(v) => (v.len() as u32, v.clone()),
            Self::Short(v) => (
                v.len() as u32,
                v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            ),
            Self::Long(v) => (
                v.len() as u32,
                v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            ),
            Self::Rational(v) => (
                v.len() as u32,
                v.iter()
                    .flat_map(|r| [r.num.to_le_bytes(), r.den.to_le_bytes()].concat())
                    .collect(),
            ),
            Self::SRational(v) => (
                v.len() as u32,
                v.iter()
                    .flat_map(|r| [r.num.to_le_bytes(), r.den.to_le_bytes()].concat())
                    .collect(),
            ),
        }
    }
}

struct Layout {
    entries: Vec<Entry>,
    start: u32,
}

impl Layout {
    fn ifd_len(&self) -> u32 {
        2 + 12 * self.entries.len() as u32 + 4
    }

    fn data_len(&self) -> u32 {
        self.entries
            .iter()
            .map(|e| {
                let (_, bytes) = e.value.encode();
                if bytes.len() > 4 {
                    (bytes.len() as u32 + 1) & !1
                } else {
                    0
                }
            })
            .sum()
    }

    fn end(&self) -> u32 {
        self.start + self.ifd_len() + self.data_len()
    }

    fn write(&self, out: &mut Vec<u8>) {
        debug_assert_eq!(out.len() as u32, self.start);
        out.extend_from_slice(&(self.entries.len() as u16).to_le_bytes());
        let mut data_offset = self.start + self.ifd_len();
        let mut data = Vec::new();
        for e in &self.entries {
            let (count, bytes) = e.value.encode();
            out.extend_from_slice(&e.tag.to_le_bytes());
            out.extend_from_slice(&e.value.type_code().to_le_bytes());
            out.extend_from_slice(&count.to_le_bytes());
            if bytes.len() <= 4 {
                let mut inline = [0_u8; 4];
                inline[..bytes.len()].copy_from_slice(&bytes);
                out.extend_from_slice(&inline);
            } else {
                out.extend_from_slice(&data_offset.to_le_bytes());
                data.extend_from_slice(&bytes);
                if bytes.len() % 2 == 1 {
                    data.push(0);
                }
                data_offset += (bytes.len() as u32 + 1) & !1;
            }
        }
        out.extend_from_slice(&0_u32.to_le_bytes()); // no next IFD
        out.extend_from_slice(&data);
    }
}

/// Builds the standalone EXIF payload: a little-endian TIFF structure with IFD0, the Exif IFD and,
/// if present, the GPS IFD. IFD0 gets the descriptive tags of `tables` plus Orientation 1 (upright
/// pixels) and a resolution of 300 dpi.
///
/// For JPEG put `b"Exif\0\0"` in front; PNG's `eXIf` chunk takes it as is.
#[must_use]
pub fn build_exif_blob(tables: &ExifTables) -> Vec<u8> {
    let mut ifd0 = tables.ifd0.clone();
    ifd0.push(Entry::new(ORIENTATION, Value::Short(vec![1])));
    ifd0.push(Entry::new(
        X_RESOLUTION,
        Value::Rational(vec![Rational { num: 300, den: 1 }]),
    ));
    ifd0.push(Entry::new(
        Y_RESOLUTION,
        Value::Rational(vec![Rational { num: 300, den: 1 }]),
    ));
    ifd0.push(Entry::new(RESOLUTION_UNIT, Value::Short(vec![2])));
    let has_exif = !tables.exif.is_empty();
    let has_gps = !tables.gps.is_empty();
    if has_exif {
        ifd0.push(Entry::new(EXIF_IFD_POINTER, Value::Long(vec![0])));
    }
    if has_gps {
        ifd0.push(Entry::new(GPS_IFD_POINTER, Value::Long(vec![0])));
    }
    ifd0.sort_by_key(|e| e.tag);
    let mut exif = tables.exif.clone();
    exif.sort_by_key(|e| e.tag);
    let mut gps = tables.gps.clone();
    gps.sort_by_key(|e| e.tag);

    // Lay the directories out one after another, then patch the two pointers in IFD0.
    let l0 = Layout {
        entries: ifd0,
        start: 8,
    };
    let exif_start = l0.end();
    let l_exif = Layout {
        entries: if has_exif { exif } else { Vec::new() },
        start: exif_start,
    };
    let gps_start = if has_exif { l_exif.end() } else { exif_start };
    let l_gps = Layout {
        entries: if has_gps { gps } else { Vec::new() },
        start: gps_start,
    };
    let mut l0 = l0;
    for e in &mut l0.entries {
        if e.tag == EXIF_IFD_POINTER {
            e.value = Value::Long(vec![exif_start]);
        } else if e.tag == GPS_IFD_POINTER {
            e.value = Value::Long(vec![gps_start]);
        }
    }

    let mut out = Vec::with_capacity(l_gps.end() as usize);
    out.extend_from_slice(b"II");
    out.extend_from_slice(&42_u16.to_le_bytes());
    out.extend_from_slice(&8_u32.to_le_bytes());
    l0.write(&mut out);
    if has_exif {
        l_exif.write(&mut out);
    }
    if has_gps {
        l_gps.write(&mut out);
    }
    out
}

/// The `Exif\0\0`-prefixed form for a JPEG `APP1` segment.
#[must_use]
pub fn build_exif_app1_payload(tables: &ExifTables) -> Vec<u8> {
    let mut v = b"Exif\0\0".to_vec();
    v.extend_from_slice(&build_exif_blob(tables));
    v
}

// ---- tiff crate adapter -----------------------------------------------------------------

/// An UNDEFINED-typed value for the `tiff` crate's tag writer.
pub(crate) struct Undefined<'a>(pub &'a [u8]);

impl TiffValue for Undefined<'_> {
    const BYTE_LEN: u8 = 1;
    const FIELD_TYPE: Type = Type::UNDEFINED;

    fn count(&self) -> usize {
        self.0.len()
    }

    fn data(&self) -> std::borrow::Cow<'_, [u8]> {
        std::borrow::Cow::Borrowed(self.0)
    }
}

/// The `tiff` crate's directory encoder for standard (32-bit offset) TIFF.
pub(crate) type Dir<'a, W> = DirectoryEncoder<'a, W, TiffKindStandard>;

/// Pads the file so that the next value written starts on an even offset.
///
/// The TIFF specification wants every value that does not fit in the entry at a word boundary;
/// the `tiff` crate writes values back to back, so odd offsets happen whenever an odd-sized value
/// comes first.
pub(crate) fn align<W: Write + Seek>(dir: &mut Dir<'_, W>) -> Result<(), ExportError> {
    let at = dir.write_data(&[0_u8][..])?;
    if at % 2 == 0 {
        dir.write_data(&[0_u8][..])?;
    }
    Ok(())
}

/// Pads the file with one zero byte if the next write would start on an odd offset.
///
/// The `tiff` crate writes a directory right where the previous data ended, so after an odd-length
/// text value the directory itself would start at an odd offset, which the TIFF specification forbids.
fn pad_to_even<W: Write + Seek>(dir: &mut Dir<'_, W>) -> Result<(), ExportError> {
    let at = dir.write_data(&[] as &[u8])?;
    if at % 2 == 1 {
        dir.write_data(&[0_u8][..])?;
    }
    Ok(())
}

/// Writes the directory (and its entries) to the file, starting it on a word boundary.
pub(crate) fn finish<W: Write + Seek>(mut dir: Dir<'_, W>) -> Result<(), ExportError> {
    pad_to_even(&mut dir)?;
    dir.finish()?;
    Ok(())
}

/// Like [`finish`], and returns the directory's offset for a pointer tag in another directory.
pub(crate) fn finish_with_offset<W: Write + Seek>(mut dir: Dir<'_, W>) -> Result<u32, ExportError> {
    pad_to_even(&mut dir)?;
    Ok(dir.finish_with_offsets()?.offset)
}

/// Writes tag `number` with `value`, aligning an out-of-line value to a word boundary first.
pub(crate) fn put<W: Write + Seek, T: TiffValue>(
    dir: &mut Dir<'_, W>,
    number: u16,
    value: T,
) -> Result<(), ExportError> {
    if value.bytes() > 4 {
        align(dir)?;
    }
    dir.write_tag(Tag::Unknown(number), value)?;
    Ok(())
}

/// Writes `entries` as tags of `dir`, in the types the entries carry.
pub fn write_directory<W: Write + Seek>(
    dir: &mut DirectoryEncoder<'_, W, TiffKindStandard>,
    entries: &[Entry],
) -> Result<(), ExportError> {
    for e in entries {
        let n = e.tag;
        match &e.value {
            Value::Ascii(s) => put(dir, n, s.as_str())?,
            Value::Byte(v) => put(dir, n, &v[..])?,
            Value::Undefined(v) => put(dir, n, Undefined(v))?,
            Value::Short(v) => put(dir, n, &v[..])?,
            Value::Long(v) => put(dir, n, &v[..])?,
            Value::Rational(v) => {
                let t: Vec<TiffRational> = v
                    .iter()
                    .map(|r| TiffRational { n: r.num, d: r.den })
                    .collect();
                put(dir, n, &t[..])?;
            }
            Value::SRational(v) => {
                let t: Vec<TiffSRational> = v
                    .iter()
                    .map(|r| TiffSRational { n: r.num, d: r.den })
                    .collect();
                put(dir, n, &t[..])?;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fractions_are_exact_for_camera_values() {
        for (value, want) in [
            (1.0 / 250.0, (1, 250)),
            (1.0 / 8000.0, (1, 8000)),
            (1.0 / 32000.0, (1, 32000)),
            (1.0 / 3.0, (1, 3)),
            (0.3, (3, 10)),
            (0.4, (2, 5)),
            (30.0, (30, 1)),
            (2.5, (5, 2)),
            (1.0, (1, 1)),
        ] {
            let r = Rational::from_f64(value, 1_000_000).unwrap();
            assert_eq!((r.num, r.den), want, "{value}");
        }
        assert!(Rational::from_f64(-1.0, 100).is_none());
        assert!(Rational::from_f64(f64::NAN, 100).is_none());
        assert_eq!(
            Rational::from_decimal(2.8, 10).unwrap(),
            Rational { num: 28, den: 10 }
        );
        assert_eq!(
            Rational::from_decimal(105.0, 10).unwrap(),
            Rational { num: 1050, den: 10 }
        );
        for (value, want) in [
            (0.0, (0, 1)),
            (-1.0, (-1, 1)),
            (1.0 / 3.0, (1, 3)),
            (-0.7, (-7, 10)),
            (2.0 / 3.0, (2, 3)),
        ] {
            let r = SRational::from_f64(value, 1000).unwrap();
            assert_eq!((r.num, r.den), want, "{value}");
        }
    }

    #[test]
    fn capture_time_formats() {
        let t = CaptureTime {
            year: 2026,
            month: 10,
            day: 4,
            hour: 14,
            minute: 22,
            second: 31,
            millisecond: Some(120),
            utc_offset_minutes: Some(-330),
        };
        assert!(t.is_valid());
        assert_eq!(t.exif_string(), "2026:10:04 14:22:31");
        assert_eq!(t.offset_string().unwrap(), "-05:30");
        assert_eq!(t.xmp_string(), "2026-10-04T14:22:31.120-05:30");
        assert_eq!(t.date_string(), "2026-10-04");
        assert!(!CaptureTime { month: 13, ..t }.is_valid());
    }

    fn sample() -> ExifInfo {
        ExifInfo {
            make: Some("SONY".into()),
            model: Some("ILCE-1".into()),
            lens_make: Some("Sony".into()),
            lens_model: Some("FE 24-70mm F2.8 GM II".into()),
            exposure_time_s: Some(1.0 / 250.0),
            f_number: Some(2.8),
            iso: Some(400),
            focal_length_mm: Some(35.0),
            focal_length_35mm: Some(35),
            exposure_bias_ev: Some(-0.7),
            capture_time: Some(CaptureTime {
                year: 2026,
                month: 10,
                day: 4,
                hour: 14,
                minute: 22,
                second: 31,
                millisecond: None,
                utc_offset_minutes: Some(120),
            }),
            gps: Some(Gps {
                latitude_deg: 38.7223,
                longitude_deg: -9.1393,
                altitude_m: Some(12.5),
            }),
        }
    }

    const FACTS: OutputFacts = OutputFacts {
        width: 3000,
        height: 2000,
        color_space: Some(ColorSpace::DisplayP3),
        ycbcr: false,
    };

    #[test]
    fn remove_location_drops_the_gps_directory() {
        let info = sample();
        let with = info.tables(&FACTS, "Yellowmoon 0.0.1", false);
        let without = info.tables(&FACTS, "Yellowmoon 0.0.1", true);
        assert!(!with.gps.is_empty());
        assert!(without.gps.is_empty());
        let blob = build_exif_blob(&without);
        // No GPS pointer (tag 34853 = 0x8825) and no latitude reference in the output.
        assert!(
            !blob.windows(2).any(|w| w == 34853_u16.to_le_bytes()) || {
                // The two bytes could match inside other data; check the IFD0 entry list instead.
                let n = u16::from_le_bytes([blob[8], blob[9]]) as usize;
                !(0..n).any(|i| u16::from_le_bytes([blob[10 + 12 * i], blob[11 + 12 * i]]) == 34853)
            }
        );
        let blob_with = build_exif_blob(&with);
        let n = u16::from_le_bytes([blob_with[8], blob_with[9]]) as usize;
        assert!(
            (0..n).any(
                |i| u16::from_le_bytes([blob_with[10 + 12 * i], blob_with[11 + 12 * i]]) == 34853
            )
        );
    }

    #[test]
    fn tables_carry_the_documented_fields() {
        let t = sample().tables(&FACTS, "Yellowmoon 0.0.1", false);
        let find = |entries: &[Entry], tag: u16| {
            entries
                .iter()
                .find(|e| e.tag == tag)
                .map(|e| e.value.clone())
        };
        assert_eq!(find(&t.ifd0, 271), Some(Value::Ascii("SONY".into())));
        assert_eq!(find(&t.ifd0, 272), Some(Value::Ascii("ILCE-1".into())));
        assert_eq!(
            find(&t.ifd0, 305),
            Some(Value::Ascii("Yellowmoon 0.0.1".into()))
        );
        assert_eq!(
            find(&t.ifd0, 306),
            Some(Value::Ascii("2026:10:04 14:22:31".into()))
        );
        assert_eq!(
            find(&t.exif, 33434),
            Some(Value::Rational(vec![Rational { num: 1, den: 250 }]))
        );
        assert_eq!(
            find(&t.exif, 33437),
            Some(Value::Rational(vec![Rational { num: 28, den: 10 }]))
        );
        assert_eq!(find(&t.exif, 34855), Some(Value::Short(vec![400])));
        assert_eq!(
            find(&t.exif, 37380),
            Some(Value::SRational(vec![SRational { num: -7, den: 10 }]))
        );
        assert_eq!(find(&t.exif, 40961), Some(Value::Short(vec![0xFFFF])));
        assert_eq!(find(&t.exif, 40962), Some(Value::Long(vec![3000])));
        assert_eq!(find(&t.exif, 36880), Some(Value::Ascii("+02:00".into())));
        assert_eq!(
            find(&t.exif, 42036),
            Some(Value::Ascii("FE 24-70mm F2.8 GM II".into()))
        );
        assert_eq!(find(&t.gps, 1), Some(Value::Ascii("N".into())));
        assert_eq!(find(&t.gps, 3), Some(Value::Ascii("W".into())));
    }

    #[test]
    fn high_iso_uses_the_iso_speed_tag() {
        let info = ExifInfo {
            iso: Some(102_400),
            ..ExifInfo::default()
        };
        let t = info.tables(&FACTS, "", false);
        assert!(
            t.exif
                .iter()
                .any(|e| e.tag == 34855 && e.value == Value::Short(vec![65535]))
        );
        assert!(
            t.exif
                .iter()
                .any(|e| e.tag == 34867 && e.value == Value::Long(vec![102_400]))
        );
    }

    #[test]
    fn non_ascii_text_is_replaced_not_corrupted() {
        let info = ExifInfo {
            model: Some("Ünïcode".into()),
            ..ExifInfo::default()
        };
        let t = info.tables(&FACTS, "", false);
        assert_eq!(t.ifd0[0].value, Value::Ascii("?n?code".into()));
    }

    #[test]
    fn invalid_gps_and_times_are_dropped() {
        let info = ExifInfo {
            gps: Some(Gps {
                latitude_deg: 123.0,
                longitude_deg: 0.0,
                altitude_m: None,
            }),
            capture_time: Some(CaptureTime {
                year: 2026,
                month: 0,
                day: 1,
                hour: 0,
                minute: 0,
                second: 0,
                millisecond: None,
                utc_offset_minutes: None,
            }),
            ..ExifInfo::default()
        };
        let t = info.tables(&FACTS, "", false);
        assert!(t.gps.is_empty());
        assert!(t.exif.iter().all(|e| e.tag != 36867));
    }

    #[test]
    fn blob_is_a_well_formed_little_endian_tiff() {
        let t = sample().tables(&FACTS, "Yellowmoon 0.0.1", false);
        let blob = build_exif_blob(&t);
        assert_eq!(&blob[0..4], b"II*\0");
        assert_eq!(u32::from_le_bytes(blob[4..8].try_into().unwrap()), 8);
        // Walk IFD0, follow both pointers and check each directory's data stays inside the blob.
        let read_ifd = |start: usize| -> Vec<(u16, u16, u32, [u8; 4])> {
            let n = u16::from_le_bytes([blob[start], blob[start + 1]]) as usize;
            (0..n)
                .map(|i| {
                    let e = &blob[start + 2 + 12 * i..start + 14 + 12 * i];
                    (
                        u16::from_le_bytes([e[0], e[1]]),
                        u16::from_le_bytes([e[2], e[3]]),
                        u32::from_le_bytes(e[4..8].try_into().unwrap()),
                        e[8..12].try_into().unwrap(),
                    )
                })
                .collect()
        };
        let ifd0 = read_ifd(8);
        assert!(ifd0.windows(2).all(|w| w[0].0 < w[1].0), "tags ascending");
        let orientation = ifd0.iter().find(|e| e.0 == 274).unwrap();
        assert_eq!(u16::from_le_bytes([orientation.3[0], orientation.3[1]]), 1);
        let exif_off = u32::from_le_bytes(ifd0.iter().find(|e| e.0 == 34665).unwrap().3) as usize;
        let gps_off = u32::from_le_bytes(ifd0.iter().find(|e| e.0 == 34853).unwrap().3) as usize;
        let exif = read_ifd(exif_off);
        let gps = read_ifd(gps_off);
        assert!(exif.windows(2).all(|w| w[0].0 < w[1].0));
        assert_eq!(gps[0].0, 0);
        for (tag, ty, count, value) in ifd0.iter().chain(&exif).chain(&gps) {
            let unit = match ty {
                1 | 2 | 7 => 1,
                3 => 2,
                4 => 4,
                5 | 10 => 8,
                other => panic!("type {other} for tag {tag}"),
            };
            if unit * *count as usize > 4 {
                let off = u32::from_le_bytes(*value) as usize;
                assert_eq!(off % 2, 0, "tag {tag} data is word aligned");
                assert!(
                    off + unit * *count as usize <= blob.len(),
                    "tag {tag} data inside the blob"
                );
            }
        }
        // The APP1 form is the same bytes behind the Exif header.
        let app1 = build_exif_app1_payload(&t);
        assert_eq!(&app1[0..6], b"Exif\0\0");
        assert_eq!(&app1[6..], &blob[..]);
        assert_eq!(blob, build_exif_blob(&t), "deterministic");
    }

    #[test]
    fn dms_conversion() {
        let r = dms(38.7223);
        assert_eq!((r[0].num, r[1].num), (38, 43));
        let seconds = f64::from(r[2].num) / f64::from(r[2].den);
        assert!((seconds - 20.28).abs() < 1e-3, "{seconds}");
        // Whole numbers of degrees and the limits.
        let whole = |v: f64| dms(v).iter().map(|r| r.num).collect::<Vec<_>>();
        assert_eq!(whole(0.0), vec![0, 0, 0]);
        assert_eq!(whole(90.0), vec![90, 0, 0]);
        assert_eq!(whole(180.0), vec![180, 0, 0]);
        // A value a hair below a whole degree or minute carries instead of writing 60 minutes or 60 seconds.
        assert_eq!(whole(38.999_999_999), vec![39, 0, 0]);
        assert_eq!(
            whole(12.0 + 59.0 / 60.0 + 59.999_999_9 / 3600.0),
            vec![13, 0, 0]
        );
        let r = dms(10.0 + 30.0 / 60.0 + 59.999_94 / 3600.0);
        assert_eq!((r[0].num, r[1].num, r[2].num), (10, 30, 599_999));
    }

    proptest::proptest! {
        /// Degrees, minutes and seconds always add up to the value, with minutes and seconds below 60.
        #[test]
        fn dms_is_always_normalised(value in 0.0_f64..=180.0) {
            let r = dms(value);
            proptest::prop_assert!(r[1].num < 60);
            proptest::prop_assert!(r[2].num < 600_000, "{} seconds", r[2].num);
            let back = f64::from(r[0].num) + f64::from(r[1].num) / 60.0 + f64::from(r[2].num) / 36_000_000.0;
            proptest::prop_assert!((back - value).abs() < 1e-7, "{value} -> {back}");
        }
    }

    proptest::proptest! {
        /// A fraction with a large denominator limit is accurate to better than one part in a million for camera-sized values.
        #[test]
        fn fractions_are_accurate(x in 1.0e-5_f64..1.0e4) {
            let r = Rational::from_f64(x, 1_000_000).unwrap();
            proptest::prop_assert!(r.den >= 1);
            proptest::prop_assert!(((r.as_f64() - x) / x).abs() < 1e-5, "{x} -> {}/{}", r.num, r.den);
            let s = SRational::from_f64(-x, 1_000_000).unwrap();
            proptest::prop_assert!(((s.as_f64() + x) / x).abs() < 1e-5);
        }
    }
}
