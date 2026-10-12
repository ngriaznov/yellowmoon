//! EXIF reader: make, model, exposure time, aperture, ISO, timestamps and orientation (`docs/04-raw-decoding.md` section Container).
//!
//! # Tags read
//!
//! From IFD0 (the first top-level directory that has them): `Make` 0x010F, `Model` 0x0110,
//! `Software` 0x0131, `Orientation` 0x0112, and the pointers `ExifIFD` 0x8769 and `GPSInfo`
//! 0x8825.
//!
//! From the EXIF directory: `ExposureTime` 0x829A, `FNumber` 0x829D, `ExposureBiasValue` 0x9204,
//! `FocalLength` 0x920A, `FocalLengthIn35mmFilm` 0xA405, `LensModel` 0xA434, `LensMake` 0xA433,
//! `DateTimeOriginal` 0x9003 with `SubSecTimeOriginal` 0x9291 and `OffsetTimeOriginal` 0x9011
//! (`OffsetTime` 0x9010 when that is missing), and the maker note 0x927C.
//!
//! ISO comes from these EXIF tags (and from the Sony maker note, see [`resolve_iso`]):
//! `ISOSpeedRatings` 0x8827 (16 bit, saturates at 65535), `SensitivityType` 0x8830,
//! `StandardOutputSensitivity` 0x8831, `RecommendedExposureIndex` 0x8832 and `ISOSpeed` 0x8833
//! (all 32 bit).
//!
//! From the GPS directory: `GPSLatitudeRef/Latitude` 0x0001/2, `GPSLongitudeRef/Longitude`
//! 0x0003/4, `GPSAltitudeRef/Altitude` 0x0005/6, `GPSTimeStamp` 0x0007 and `GPSDateStamp` 0x001D.

use serde::{Deserialize, Serialize};

use crate::tiff::{Entry, Ifd, IfdKind, IfdNode, Tiff, tag};

/// Where the ISO value came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IsoSource {
    /// `SonyISO` in an enciphered maker-note table (0x9416 or 0x9405). Never saturates.
    SonyMakerNote,
    /// EXIF `ISOSpeed` 0x8833 (32 bit).
    ExifIsoSpeed,
    /// EXIF `RecommendedExposureIndex` 0x8832 (32 bit).
    ExifRecommendedExposureIndex,
    /// EXIF `ISOSpeedRatings` 0x8827 (16 bit): saturates at 65535, used last.
    ExifIsoSpeedRatings,
}

impl IsoSource {
    /// A short stable name for text output.
    pub fn name(self) -> &'static str {
        match self {
            Self::SonyMakerNote => "sony_maker_note",
            Self::ExifIsoSpeed => "exif_iso_speed",
            Self::ExifRecommendedExposureIndex => "exif_recommended_exposure_index",
            Self::ExifIsoSpeedRatings => "exif_iso_speed_ratings",
        }
    }
}

/// The EXIF tags that carry a sensitivity.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct IsoTags {
    /// `ISOSpeedRatings` 0x8827.
    pub speed_ratings: Option<u32>,
    /// `SensitivityType` 0x8830.
    pub sensitivity_type: Option<u16>,
    /// `StandardOutputSensitivity` 0x8831.
    pub standard_output_sensitivity: Option<u32>,
    /// `RecommendedExposureIndex` 0x8832.
    pub recommended_exposure_index: Option<u32>,
    /// `ISOSpeed` 0x8833.
    pub iso_speed: Option<u32>,
}

/// The largest ISO value accepted from any source (a million; the highest Sony setting is 409600).
pub const MAX_PLAUSIBLE_ISO: u32 = 1_000_000;

/// Picks the ISO to report, preferring values that cannot saturate.
///
/// Order: the Sony maker-note ISO; then, depending on `SensitivityType` (2 and 4 mean the
/// recommended exposure index is the camera's ISO, anything else the ISO speed), `ISOSpeed` 0x8833
/// and `RecommendedExposureIndex` 0x8832; last the 16-bit `ISOSpeedRatings` 0x8827. Zero and
/// absurd values (above [`MAX_PLAUSIBLE_ISO`]) are skipped.
pub fn resolve_iso(sony: Option<u32>, tags: &IsoTags) -> Option<(u32, IsoSource)> {
    let sane = |v: u32| (1..=MAX_PLAUSIBLE_ISO).contains(&v);
    if let Some(v) = sony.filter(|v| sane(*v)) {
        return Some((v, IsoSource::SonyMakerNote));
    }
    let speed = tags
        .iso_speed
        .filter(|v| sane(*v))
        .map(|v| (v, IsoSource::ExifIsoSpeed));
    let rei = tags
        .recommended_exposure_index
        .filter(|v| sane(*v))
        .map(|v| (v, IsoSource::ExifRecommendedExposureIndex));
    let long_tags = if matches!(tags.sensitivity_type, Some(2 | 4)) {
        rei.or(speed)
    } else {
        speed.or(rei)
    };
    long_tags.or_else(|| {
        tags.speed_ratings
            .filter(|v| sane(*v))
            .map(|v| (v, IsoSource::ExifIsoSpeedRatings))
    })
}

/// The moment the shutter was released, as written by the camera: local wall-clock time and,
/// when the camera records it, the offset from UTC.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CaptureTime {
    /// Four-digit year.
    pub year: u16,
    /// Month, 1 to 12.
    pub month: u8,
    /// Day of the month, 1 to 31.
    pub day: u8,
    /// Hour, 0 to 23.
    pub hour: u8,
    /// Minute, 0 to 59.
    pub minute: u8,
    /// Second, 0 to 60.
    pub second: u8,
    /// Fraction of a second in milliseconds (from `SubSecTimeOriginal`), when present.
    pub subsec_ms: Option<u16>,
    /// Offset of local time from UTC in minutes (from `OffsetTimeOriginal`), when present.
    pub utc_offset_minutes: Option<i16>,
}

impl CaptureTime {
    /// Parses an EXIF date string (`YYYY:MM:DD HH:MM:SS`), an optional `SubSecTime` digit string
    /// and an optional `OffsetTime` string (`+HH:MM`). Returns `None` for malformed or all-zero
    /// dates. A malformed sub-second or offset string is dropped, not an error.
    pub fn parse_exif(datetime: &str, subsec: Option<&str>, offset: Option<&str>) -> Option<Self> {
        let b = datetime.trim().as_bytes();
        if b.len() < 19 {
            return None;
        }
        let digits = |range: std::ops::Range<usize>| -> Option<u32> {
            let part = b.get(range)?;
            if part.iter().all(u8::is_ascii_digit) {
                part.iter()
                    .try_fold(0u32, |acc, d| Some(acc * 10 + u32::from(d - b'0')))
            } else {
                None
            }
        };
        let separators_ok = matches!(b[4], b':' | b'-')
            && matches!(b[7], b':' | b'-')
            && matches!(b[10], b' ' | b'T')
            && b[13] == b':'
            && b[16] == b':';
        if !separators_ok {
            return None;
        }
        let (year, month, day) = (digits(0..4)?, digits(5..7)?, digits(8..10)?);
        let (hour, minute, second) = (digits(11..13)?, digits(14..16)?, digits(17..19)?);
        if year < 1900 || !(1..=12).contains(&month) || !(1..=31).contains(&day) {
            return None;
        }
        if hour > 23 || minute > 59 || second > 60 {
            return None;
        }
        Some(Self {
            year: year as u16,
            month: month as u8,
            day: day as u8,
            hour: hour as u8,
            minute: minute as u8,
            second: second as u8,
            subsec_ms: subsec.and_then(parse_subsec_ms),
            utc_offset_minutes: offset.and_then(parse_utc_offset),
        })
    }

    /// `YYYY:MM:DD HH:MM:SS`, the EXIF form.
    pub fn to_exif_string(&self) -> String {
        format!(
            "{:04}:{:02}:{:02} {:02}:{:02}:{:02}",
            self.year, self.month, self.day, self.hour, self.minute, self.second
        )
    }

    /// ISO 8601: `YYYY-MM-DDTHH:MM:SS`, then `.mmm` when a sub-second part is known, then
    /// `+HH:MM` when the offset is known.
    pub fn to_iso8601(&self) -> String {
        let mut text = format!(
            "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}",
            self.year, self.month, self.day, self.hour, self.minute, self.second
        );
        if let Some(ms) = self.subsec_ms {
            text.push_str(&format!(".{ms:03}"));
        }
        if let Some(minutes) = self.utc_offset_minutes {
            let sign = if minutes < 0 { '-' } else { '+' };
            let abs = minutes.unsigned_abs();
            text.push_str(&format!("{sign}{:02}:{:02}", abs / 60, abs % 60));
        }
        text
    }
}

/// The first three digits of a `SubSecTime` string as milliseconds ("5" is 500 ms).
fn parse_subsec_ms(text: &str) -> Option<u16> {
    let digits: String = text
        .trim()
        .chars()
        .take_while(char::is_ascii_digit)
        .take(3)
        .collect();
    if digits.is_empty() {
        return None;
    }
    let value: u16 = digits.parse().ok()?;
    Some(value * 10u16.pow(3 - digits.len() as u32))
}

/// `+HH:MM` or `-HH:MM` to minutes; offsets beyond +-14 hours are rejected.
fn parse_utc_offset(text: &str) -> Option<i16> {
    let b = text.trim().as_bytes();
    if b.len() != 6 || b[3] != b':' || !matches!(b[0], b'+' | b'-') {
        return None;
    }
    let num = |r: std::ops::Range<usize>| -> Option<i16> {
        let s = std::str::from_utf8(b.get(r)?).ok()?;
        s.parse().ok()
    };
    let minutes = num(1..3)? * 60 + num(4..6)?;
    if num(4..6)? > 59 || minutes > 14 * 60 {
        return None;
    }
    Some(if b[0] == b'-' { -minutes } else { minutes })
}

/// Position and time from the GPS directory. Every field is `None` when the camera had no fix.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct GpsInfo {
    /// Latitude in degrees, positive north.
    pub latitude_deg: Option<f64>,
    /// Longitude in degrees, positive east.
    pub longitude_deg: Option<f64>,
    /// Altitude in metres above sea level (negative below).
    pub altitude_m: Option<f64>,
    /// UTC time of the fix, `YYYY-MM-DDTHH:MM:SSZ`.
    pub utc_time: Option<String>,
}

/// Everything the EXIF directories tell us.
#[derive(Clone, Debug, Default)]
pub struct ExifInfo {
    /// `Make`.
    pub make: Option<String>,
    /// `Model`.
    pub model: Option<String>,
    /// `Software` (the firmware, for example `ILCE-7RM4 v1.10`).
    pub software: Option<String>,
    /// `Orientation` (1 to 8).
    pub orientation: Option<u16>,
    /// `ExposureTime` in seconds.
    pub exposure_time_s: Option<f64>,
    /// `FNumber`.
    pub f_number: Option<f64>,
    /// `ExposureBiasValue` in EV.
    pub exposure_comp_ev: Option<f64>,
    /// `FocalLength` in millimetres.
    pub focal_length_mm: Option<f64>,
    /// `FocalLengthIn35mmFilm` in millimetres.
    pub focal_length_35mm: Option<u16>,
    /// `LensModel`.
    pub lens_model: Option<String>,
    /// `LensMake`.
    pub lens_make: Option<String>,
    /// `DateTimeOriginal` with sub-second and offset.
    pub capture_time: Option<CaptureTime>,
    /// The sensitivity tags.
    pub iso: IsoTags,
    /// The `MakerNote` entry, to hand to [`crate::makernote::parse`].
    pub maker_note: Option<Entry>,
    /// The GPS directory contents when the file has one.
    pub gps: Option<GpsInfo>,
}

/// Reads the EXIF fields from the directories found by [`Tiff::walk`].
pub fn read(tiff: &Tiff, nodes: &[IfdNode]) -> ExifInfo {
    let mut info = ExifInfo::default();

    // Make, model and friends: the first top-level directory that has the tag.
    let roots = || {
        nodes
            .iter()
            .filter(|n| matches!(n.kind, IfdKind::Root(_)))
            .map(|n| &n.ifd)
    };
    info.make = roots().find_map(|ifd| text(tiff, ifd, tag::MAKE));
    info.model = roots().find_map(|ifd| text(tiff, ifd, tag::MODEL));
    info.software = roots().find_map(|ifd| text(tiff, ifd, tag::SOFTWARE));
    info.orientation = roots()
        .find_map(|ifd| uint(tiff, ifd, tag::ORIENTATION))
        .and_then(|v| u16::try_from(v).ok())
        .filter(|v| (1..=8).contains(v));

    let exif_ifds = || {
        nodes
            .iter()
            .filter(|n| n.kind == IfdKind::Exif)
            .map(|n| &n.ifd)
    };
    info.exposure_time_s = exif_ifds()
        .find_map(|ifd| float(tiff, ifd, tag::EXPOSURE_TIME))
        .filter(|v| *v > 0.0);
    info.f_number = exif_ifds()
        .find_map(|ifd| float(tiff, ifd, tag::F_NUMBER))
        .filter(|v| *v > 0.0);
    info.exposure_comp_ev = exif_ifds().find_map(|ifd| float(tiff, ifd, tag::EXPOSURE_BIAS_VALUE));
    info.focal_length_mm = exif_ifds()
        .find_map(|ifd| float(tiff, ifd, tag::FOCAL_LENGTH))
        .filter(|v| *v > 0.0);
    info.focal_length_35mm = exif_ifds()
        .find_map(|ifd| uint(tiff, ifd, tag::FOCAL_LENGTH_35MM))
        .and_then(|v| u16::try_from(v).ok())
        .filter(|v| *v > 0);
    info.lens_model = exif_ifds()
        .find_map(|ifd| text(tiff, ifd, tag::LENS_MODEL))
        .filter(|s| !s.is_empty());
    info.lens_make = exif_ifds()
        .find_map(|ifd| text(tiff, ifd, tag::LENS_MAKE))
        .filter(|s| !s.is_empty());

    if let Some(datetime) = exif_ifds().find_map(|ifd| text(tiff, ifd, tag::DATE_TIME_ORIGINAL)) {
        let subsec = exif_ifds().find_map(|ifd| text(tiff, ifd, tag::SUB_SEC_TIME_ORIGINAL));
        let offset = exif_ifds()
            .find_map(|ifd| text(tiff, ifd, tag::OFFSET_TIME_ORIGINAL))
            .or_else(|| exif_ifds().find_map(|ifd| text(tiff, ifd, tag::OFFSET_TIME)));
        info.capture_time =
            CaptureTime::parse_exif(&datetime, subsec.as_deref(), offset.as_deref());
    }
    if info.capture_time.is_none() {
        // Cameras that omit the EXIF original time still write IFD0 `DateTime`.
        info.capture_time = roots()
            .find_map(|ifd| text(tiff, ifd, tag::DATE_TIME))
            .and_then(|s| CaptureTime::parse_exif(&s, None, None));
    }

    info.iso = IsoTags {
        speed_ratings: exif_ifds().find_map(|ifd| uint(tiff, ifd, tag::ISO_SPEED_RATINGS)),
        sensitivity_type: exif_ifds()
            .find_map(|ifd| uint(tiff, ifd, tag::SENSITIVITY_TYPE))
            .and_then(|v| u16::try_from(v).ok()),
        standard_output_sensitivity: exif_ifds()
            .find_map(|ifd| uint(tiff, ifd, tag::STANDARD_OUTPUT_SENSITIVITY)),
        recommended_exposure_index: exif_ifds()
            .find_map(|ifd| uint(tiff, ifd, tag::RECOMMENDED_EXPOSURE_INDEX)),
        iso_speed: exif_ifds().find_map(|ifd| uint(tiff, ifd, tag::ISO_SPEED)),
    };

    info.maker_note = exif_ifds().find_map(|ifd| ifd.get(tag::MAKER_NOTE).copied());
    info.gps = nodes
        .iter()
        .find(|n| n.kind == IfdKind::Gps)
        .map(|n| read_gps(tiff, &n.ifd));
    info
}

fn text(tiff: &Tiff, ifd: &Ifd, tag: u16) -> Option<String> {
    let entry = ifd.get(tag)?;
    tiff.ascii(entry).ok().filter(|s| !s.is_empty())
}

fn uint(tiff: &Tiff, ifd: &Ifd, tag: u16) -> Option<u32> {
    let entry = ifd.get(tag)?;
    let value = *tiff.uints(entry, 1).ok()?.first()?;
    u32::try_from(value).ok()
}

fn float(tiff: &Tiff, ifd: &Ifd, tag: u16) -> Option<f64> {
    let entry = ifd.get(tag)?;
    tiff.floats(entry, 1)
        .ok()?
        .first()
        .copied()
        .filter(|v| v.is_finite())
}

fn floats(tiff: &Tiff, ifd: &Ifd, tag: u16, n: usize) -> Option<Vec<f64>> {
    let values = tiff.floats(ifd.get(tag)?, n).ok()?;
    (values.len() == n && values.iter().all(|v| v.is_finite())).then_some(values)
}

fn read_gps(tiff: &Tiff, ifd: &Ifd) -> GpsInfo {
    let dms = |values: &[f64]| values[0] + values[1] / 60.0 + values[2] / 3600.0;
    let signed = |tag_ref: u16, tag_value: u16, negative: &str| -> Option<f64> {
        let value = dms(&floats(tiff, ifd, tag_value, 3)?);
        let hemisphere = text(tiff, ifd, tag_ref).unwrap_or_default();
        Some(if hemisphere.eq_ignore_ascii_case(negative) {
            -value
        } else {
            value
        })
    };
    let latitude_deg = signed(0x0001, 0x0002, "S").filter(|v| v.abs() <= 90.0);
    let longitude_deg = signed(0x0003, 0x0004, "W").filter(|v| v.abs() <= 180.0);
    let altitude_m = float(tiff, ifd, 0x0006).map(|alt| {
        let below = ifd
            .get(0x0005)
            .and_then(|e| tiff.uints(e, 1).ok())
            .and_then(|v| v.first().copied())
            == Some(1);
        if below { -alt } else { alt }
    });
    let utc_time = (|| {
        let date = text(tiff, ifd, 0x001D)?;
        let time = floats(tiff, ifd, 0x0007, 3)?;
        let stamp = CaptureTime::parse_exif(
            &format!(
                "{} {:02}:{:02}:{:02}",
                date.trim(),
                time[0] as u32,
                time[1] as u32,
                time[2] as u32
            ),
            None,
            None,
        )?;
        Some(format!("{}Z", stamp.to_iso8601()))
    })();
    GpsInfo {
        latitude_deg,
        longitude_deg,
        altitude_m,
        utc_time,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_exif_dates_with_subsec_and_offset() {
        let t = CaptureTime::parse_exif("2026:10:04 12:03:44", Some("07"), Some("+02:00"))
            .expect("time");
        assert_eq!((t.year, t.month, t.day), (2026, 10, 4));
        assert_eq!((t.hour, t.minute, t.second), (12, 3, 44));
        assert_eq!(t.subsec_ms, Some(70));
        assert_eq!(t.utc_offset_minutes, Some(120));
        assert_eq!(t.to_exif_string(), "2026:10:04 12:03:44");
        assert_eq!(t.to_iso8601(), "2026-10-04T12:03:44.070+02:00");
        let t = CaptureTime::parse_exif("2026:01:02 03:04:05", None, Some("-05:30")).expect("time");
        assert_eq!(t.to_iso8601(), "2026-01-02T03:04:05-05:30");
        let t = CaptureTime::parse_exif("2026:01:02 03:04:05", Some("123456"), None).expect("time");
        assert_eq!(t.subsec_ms, Some(123));
    }

    #[test]
    fn rejects_malformed_and_zero_dates() {
        for text in [
            "",
            "0000:00:00 00:00:00",
            "2026:13:01 00:00:00",
            "2026:10:04 25:00:00",
            "2026:10:04 12:61:00",
            "2026-10-04",
            "abcd:10:04 12:03:44",
            "2026:10:04x12:03:44",
        ] {
            assert_eq!(CaptureTime::parse_exif(text, None, None), None, "{text:?}");
        }
        // Bad sub-second and offset strings are dropped, the date survives.
        let t = CaptureTime::parse_exif("2026:10:04 12:03:44", Some("xx"), Some("+25:00"))
            .expect("time");
        assert_eq!(t.subsec_ms, None);
        assert_eq!(t.utc_offset_minutes, None);
        let t =
            CaptureTime::parse_exif("2026:10:04 12:03:44", Some(""), Some("+0200")).expect("time");
        assert_eq!(t.subsec_ms, None);
        assert_eq!(t.utc_offset_minutes, None);
    }

    #[test]
    fn subsec_scales_to_milliseconds() {
        assert_eq!(parse_subsec_ms("5"), Some(500));
        assert_eq!(parse_subsec_ms("05"), Some(50));
        assert_eq!(parse_subsec_ms("005"), Some(5));
        assert_eq!(parse_subsec_ms("9999"), Some(999));
        assert_eq!(parse_subsec_ms(""), None);
    }

    #[test]
    fn iso_prefers_the_maker_note_then_32_bit_tags_then_ratings() {
        let tags = IsoTags {
            speed_ratings: Some(65_535),
            sensitivity_type: Some(2),
            standard_output_sensitivity: None,
            recommended_exposure_index: Some(102_400),
            iso_speed: None,
        };
        assert_eq!(
            resolve_iso(Some(102_405), &tags),
            Some((102_405, IsoSource::SonyMakerNote))
        );
        assert_eq!(
            resolve_iso(None, &tags),
            Some((102_400, IsoSource::ExifRecommendedExposureIndex))
        );
        let ratings_only = IsoTags {
            speed_ratings: Some(800),
            ..IsoTags::default()
        };
        assert_eq!(
            resolve_iso(None, &ratings_only),
            Some((800, IsoSource::ExifIsoSpeedRatings))
        );
        assert_eq!(resolve_iso(None, &IsoTags::default()), None);
    }

    #[test]
    fn sensitivity_type_chooses_between_the_two_32_bit_tags() {
        let both = |t: Option<u16>| IsoTags {
            sensitivity_type: t,
            recommended_exposure_index: Some(1600),
            iso_speed: Some(3200),
            ..IsoTags::default()
        };
        assert_eq!(resolve_iso(None, &both(Some(2))).map(|v| v.0), Some(1600));
        assert_eq!(resolve_iso(None, &both(Some(4))).map(|v| v.0), Some(1600));
        assert_eq!(resolve_iso(None, &both(Some(3))).map(|v| v.0), Some(3200));
        assert_eq!(resolve_iso(None, &both(None)).map(|v| v.0), Some(3200));
    }

    #[test]
    fn zero_and_absurd_iso_values_are_skipped() {
        let tags = IsoTags {
            speed_ratings: Some(400),
            recommended_exposure_index: Some(0),
            iso_speed: Some(u32::MAX),
            ..IsoTags::default()
        };
        assert_eq!(
            resolve_iso(Some(0), &tags),
            Some((400, IsoSource::ExifIsoSpeedRatings))
        );
    }

    #[test]
    fn iso_source_names_are_stable() {
        assert_eq!(IsoSource::SonyMakerNote.name(), "sony_maker_note");
        assert_eq!(
            IsoSource::ExifIsoSpeedRatings.name(),
            "exif_iso_speed_ratings"
        );
    }
}
