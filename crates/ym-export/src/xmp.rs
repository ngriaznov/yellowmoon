//! The XMP packet every export carries (`docs/09-export.md` section Metadata).
//!
//! `xmp:CreatorTool` is `Yellowmoon <version>`; our own properties live in the `ym` namespace
//! `https://yellowmoon.app/ns/1.0/` so any exported file can be traced back to the profile, look,
//! display transform, models and settings that made it. The packet is deterministic: the dates are
//! the capture time, never the time of export, and properties appear in a fixed order.

use serde::{Deserialize, Serialize};

use crate::error::ExportError;
use crate::exif::CaptureTime;

/// The `ym` XMP namespace URI.
pub const YM_NAMESPACE: &str = "https://yellowmoon.app/ns/1.0/";

/// The prefix of the XMP namespace in packets and in the JPEG `APP1` header (with a trailing NUL).
pub const XMP_APP1_HEADER: &[u8] = b"http://ns.adobe.com/xap/1.0/\0";

/// The `ym:` properties of `docs/09`. Every field is optional; a missing field is left out.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct YmInfo {
    /// `ym:PipelineVersion`, the semver of the processing chain (`1.0.0`).
    pub pipeline_version: Option<String>,
    /// `ym:Profile`, the camera profile id (`sony-ilce-7rm4`).
    pub profile: Option<String>,
    /// `ym:ProfileVersion` (`1.0.0`).
    pub profile_version: Option<String>,
    /// `ym:ProfileMatch`, how the profile was chosen (`exact`, `sensor family`, `default`).
    pub profile_match: Option<String>,
    /// `ym:Look`, the look id (`builtin:neutral`, `arri:3000:…`).
    pub look: Option<String>,
    /// `ym:LookSHA256`, the SHA-256 of the look's file.
    pub look_sha256: Option<String>,
    /// `ym:LookIntensity`, 0 to 100.
    pub look_intensity_pct: Option<u32>,
    /// `ym:DisplayTransform` (`cine-p3`).
    pub display_transform: Option<String>,
    /// `ym:Models`, the networks used (`jdd 1.2.0 (sha256…)`).
    pub models: Option<String>,
    /// `ym:Auto`, which Auto made the photo's automatic values: `rule`, `apple`, `claude` or `agy`
    /// (`docs/19` section 9, `docs/22` section 14.2).
    pub auto: Option<String>,
    /// `ym:AutoAdvisor`, what the provider said about the photo: compact JSON with sorted keys, only
    /// when a provider acted on it (`docs/22` section 14.2). Replaces `ym:AppleAuto`, which is no
    /// longer written.
    pub auto_advisor: Option<String>,
    /// `ym:FilmData`, which film data made the picture: for each live Film node its stock and paper, the version of the
    /// profile data and the engine's version, with the credit (`docs/24` section 6.2). Left out for a photo without one.
    #[serde(default)]
    pub film_data: Option<String>,
    /// `ym:Settings`, the resolved develop settings; written as compact JSON.
    pub settings: Option<serde_json::Value>,
    /// `ym:AutoFields` (`docs/30` section 12): which Auto fields were still Auto, and the light a picked white resolved to; compact
    /// JSON with sorted keys. Written after `ym:Settings` and only with it: a preset made from the file puts Auto back where
    /// `ym:Settings` is pinned.
    #[serde(default)]
    pub auto_fields: Option<String>,
}

/// What goes into the XMP packet.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct XmpInfo {
    /// The app version for `xmp:CreatorTool` (`0.0.1` gives `Yellowmoon 0.0.1`).
    pub app_version: String,
    /// The capture time, used for `xmp:CreateDate`, `xmp:ModifyDate` and `xmp:MetadataDate`.
    pub capture_time: Option<CaptureTime>,
    /// The `ym:` properties.
    pub ym: YmInfo,
}

impl XmpInfo {
    /// The `xmp:CreatorTool` string.
    #[must_use]
    pub fn creator_tool(&self) -> String {
        if self.app_version.is_empty() {
            "Yellowmoon".to_owned()
        } else {
            format!("Yellowmoon {}", self.app_version)
        }
    }
}

/// Escapes text for XML 1.0: the five special characters, and control characters are dropped.
fn escape(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            '\t' | '\n' | '\r' => out.push(c),
            c if c.is_control() => {}
            c => out.push(c),
        }
    }
    out
}

fn property(out: &mut String, name: &str, value: &str) {
    out.push_str("   <");
    out.push_str(name);
    out.push('>');
    out.push_str(&escape(value));
    out.push_str("</");
    out.push_str(name);
    out.push_str(">\n");
}

/// Builds the XMP packet for a file of type `mime` (`image/jpeg`, `image/tiff`, `image/png`,
/// `image/dng`, the type Adobe writes).
pub fn build_xmp(info: &XmpInfo, mime: &str) -> Result<Vec<u8>, ExportError> {
    let mut x = String::new();
    x.push_str("<?xpacket begin=\"\u{feff}\" id=\"W5M0MpCehiHzreSzNTczkc9d\"?>\n");
    x.push_str(&format!(
        "<x:xmpmeta xmlns:x=\"adobe:ns:meta/\" x:xmptk=\"{}\">\n",
        escape(&info.creator_tool())
    ));
    x.push_str(" <rdf:RDF xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\">\n");
    x.push_str("  <rdf:Description rdf:about=\"\"\n");
    x.push_str("    xmlns:xmp=\"http://ns.adobe.com/xap/1.0/\"\n");
    x.push_str("    xmlns:dc=\"http://purl.org/dc/elements/1.1/\"\n");
    x.push_str(&format!("    xmlns:ym=\"{YM_NAMESPACE}\">\n"));
    property(&mut x, "xmp:CreatorTool", &info.creator_tool());
    if let Some(t) = info.capture_time.filter(CaptureTime::is_valid) {
        let stamp = t.xmp_string();
        property(&mut x, "xmp:CreateDate", &stamp);
        property(&mut x, "xmp:ModifyDate", &stamp);
        property(&mut x, "xmp:MetadataDate", &stamp);
    }
    property(&mut x, "dc:format", mime);
    let ym = &info.ym;
    let intensity = ym.look_intensity_pct.map(|pct| pct.to_string());
    // Fixed order, as in docs/09.
    let fields: [(&str, Option<&str>); 12] = [
        ("ym:PipelineVersion", ym.pipeline_version.as_deref()),
        ("ym:Profile", ym.profile.as_deref()),
        ("ym:ProfileVersion", ym.profile_version.as_deref()),
        ("ym:ProfileMatch", ym.profile_match.as_deref()),
        ("ym:Look", ym.look.as_deref()),
        ("ym:LookSHA256", ym.look_sha256.as_deref()),
        ("ym:LookIntensity", intensity.as_deref()),
        ("ym:DisplayTransform", ym.display_transform.as_deref()),
        ("ym:Models", ym.models.as_deref()),
        ("ym:Auto", ym.auto.as_deref()),
        ("ym:AutoAdvisor", ym.auto_advisor.as_deref()),
        ("ym:FilmData", ym.film_data.as_deref()),
    ];
    for (name, value) in fields {
        if let Some(v) = value {
            property(&mut x, name, v);
        }
    }
    if let Some(settings) = &ym.settings {
        let json = serde_json::to_string(settings).map_err(|e| {
            ExportError::encode("XMP", format!("the settings aren't valid JSON ({e})"))
        })?;
        property(&mut x, "ym:Settings", &json);
        if let Some(fields) = ym.auto_fields.as_deref() {
            property(&mut x, "ym:AutoFields", fields);
        }
    }
    x.push_str("  </rdf:Description>\n");
    x.push_str(" </rdf:RDF>\n");
    x.push_str("</x:xmpmeta>\n");
    x.push_str("<?xpacket end=\"w\"?>");
    Ok(x.into_bytes())
}

// ---------------------------------------------------------------------------------------------
// Reading the properties back (`docs/30` section 12)
// ---------------------------------------------------------------------------------------------

/// What [`read_ym`] finds in an export's XMP.
#[derive(Debug, Clone, PartialEq)]
pub struct YmRead {
    /// `xmp:CreatorTool` (`Yellowmoon 0.2.4`).
    pub creator_tool: Option<String>,
    /// `ym:PipelineVersion`.
    pub pipeline_version: Option<String>,
    /// `ym:Settings`, the pinned settings document.
    pub settings: serde_json::Value,
    /// `ym:AutoFields`, the property's text; `None` for a file made before P-06.
    pub auto_fields: Option<String>,
}

/// The text of the property `name` (`ym:Settings`) in `packet`, with the XML escapes undone.
fn property_text(packet: &str, name: &str) -> Option<String> {
    let open = format!("<{name}>");
    let start = packet.find(&open)? + open.len();
    let end = start + packet[start..].find(&format!("</{name}>"))?;
    Some(unescape(&packet[start..end]))
}

fn unescape(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(at) = rest.find('&') {
        out.push_str(&rest[..at]);
        rest = &rest[at..];
        let replaced = [
            ("&amp;", '&'),
            ("&lt;", '<'),
            ("&gt;", '>'),
            ("&quot;", '"'),
            ("&apos;", '\''),
        ]
        .into_iter()
        .find(|(entity, _)| rest.starts_with(entity));
        match replaced {
            Some((entity, c)) => {
                out.push(c);
                rest = &rest[entity.len()..];
            }
            None => {
                out.push('&');
                rest = &rest[1..];
            }
        }
    }
    out.push_str(rest);
    out
}

/// The XMP packets of a JPEG: the `APP1` segments that start with the XMP header, in file order.
fn jpeg_packets(bytes: &[u8]) -> Vec<String> {
    let mut out = Vec::new();
    let mut at = 2;
    while at + 4 <= bytes.len() && bytes[at] == 0xFF {
        let marker = bytes[at + 1];
        if marker == 0xFF {
            at += 1;
            continue;
        }
        // The start of the scan, or the end: the segments are over.
        if marker == 0xDA || marker == 0xD9 {
            break;
        }
        if (0xD0..=0xD8).contains(&marker) || marker == 0x01 {
            at += 2;
            continue;
        }
        let len = usize::from(u16::from_be_bytes([bytes[at + 2], bytes[at + 3]]));
        let body = bytes
            .get(at + 4..(at + 2 + len).min(bytes.len()))
            .unwrap_or(&[]);
        if marker == 0xE1
            && let Some(packet) = body.strip_prefix(XMP_APP1_HEADER)
        {
            out.push(String::from_utf8_lossy(packet).into_owned());
        }
        at += 2 + len;
    }
    out
}

/// The XMP packet of a TIFF or DNG: tag 700 of the first directory.
fn tiff_packets(bytes: &[u8]) -> Vec<String> {
    let big = match bytes.get(..2) {
        Some(b"II") => false,
        Some(b"MM") => true,
        _ => return Vec::new(),
    };
    let u16_at = |at: usize| -> Option<u16> {
        let b: [u8; 2] = bytes.get(at..at + 2)?.try_into().ok()?;
        Some(if big {
            u16::from_be_bytes(b)
        } else {
            u16::from_le_bytes(b)
        })
    };
    let u32_at = |at: usize| -> Option<u32> {
        let b: [u8; 4] = bytes.get(at..at + 4)?.try_into().ok()?;
        Some(if big {
            u32::from_be_bytes(b)
        } else {
            u32::from_le_bytes(b)
        })
    };
    let find = || -> Option<String> {
        if u16_at(2)? != 42 {
            return None;
        }
        let ifd = u32_at(4)? as usize;
        let count = usize::from(u16_at(ifd)?);
        for n in 0..count {
            let entry = ifd + 2 + n * 12;
            if u16_at(entry)? != 700 {
                continue;
            }
            // BYTE (1), ASCII (2) and UNDEFINED (7) are one byte an item.
            let len = u32_at(entry + 4)? as usize;
            let data = if len <= 4 {
                bytes.get(entry + 8..entry + 8 + len)?
            } else {
                let at = u32_at(entry + 8)? as usize;
                bytes.get(at..at.checked_add(len)?)?
            };
            return Some(String::from_utf8_lossy(data).into_owned());
        }
        None
    };
    find().into_iter().collect()
}

/// The XMP packet of a PNG: the `iTXt` chunk with the keyword `XML:com.adobe.xmp`.
fn png_packets(bytes: &[u8]) -> Vec<String> {
    let Ok(reader) = png::Decoder::new(std::io::Cursor::new(bytes)).read_info() else {
        return Vec::new();
    };
    reader
        .info()
        .utf8_text
        .iter()
        .filter(|c| c.keyword == crate::png_writer::XMP_ITXT_KEYWORD)
        .filter_map(|c| c.get_text().ok())
        .collect()
}

/// Reads the `ym:` properties of a JPEG, TIFF, PNG or DNG that Yellowmoon exported (`docs/30` section 12): the `APP1` segment of
/// a JPEG, tag 700 of a TIFF or DNG, the `XML:com.adobe.xmp` chunk of a PNG. It is our own reader and takes only the properties
/// it needs; a file that holds another tool's packet before ours is read all the same (the first packet with `ym:Settings`).
///
/// Returns `Ok(None)` for a file that is not an image of these kinds or has no `ym:Settings` (not an export, or a JPEG whose
/// settings were too long to record).
///
/// # Errors
///
/// The error of reading the file.
pub fn read_ym(path: &std::path::Path) -> std::io::Result<Option<YmRead>> {
    Ok(read_ym_bytes(&std::fs::read(path)?))
}

/// [`read_ym`] for a file's bytes.
#[must_use]
pub fn read_ym_bytes(bytes: &[u8]) -> Option<YmRead> {
    let packets = if bytes.starts_with(&[0xFF, 0xD8]) {
        jpeg_packets(bytes)
    } else if bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        png_packets(bytes)
    } else {
        tiff_packets(bytes)
    };
    packets.iter().find_map(|packet| {
        let settings = serde_json::from_str(&property_text(packet, "ym:Settings")?).ok()?;
        Some(YmRead {
            creator_tool: property_text(packet, "xmp:CreatorTool"),
            pipeline_version: property_text(packet, "ym:PipelineVersion"),
            settings,
            auto_fields: property_text(packet, "ym:AutoFields"),
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> XmpInfo {
        XmpInfo {
            app_version: "0.0.1".into(),
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
            ym: YmInfo {
                pipeline_version: Some("1.0.0".into()),
                profile: Some("sony-ilce-7rm4".into()),
                profile_version: Some("1.0.0".into()),
                profile_match: Some("default".into()),
                look: Some("arri:3000:film <warm> & \"cool\"".into()),
                look_sha256: Some("ab".repeat(32)),
                look_intensity_pct: Some(100),
                display_transform: Some("cine-p3".into()),
                models: Some("jdd 1.2.0 (sha256:ff)".into()),
                auto: Some("apple".into()),
                auto_advisor: Some(
                    r#"{"model":"AFM 3 Core","provider":"apple","result":"answered"}"#.into(),
                ),
                film_data: Some("kodak_portra_400 on kodak_portra_endura; profiles 0.3.2".into()),
                settings: Some(
                    serde_json::json!({ "exposure": { "manual_ev": 0.3 }, "look": "x" }),
                ),
                auto_fields: Some(r#"{"input":{"exposure":true},"main":{"black":true}}"#.into()),
            },
        }
    }

    #[test]
    fn packet_has_creator_tool_and_every_ym_property() {
        let xmp = String::from_utf8(build_xmp(&sample(), "image/jpeg").unwrap()).unwrap();
        assert!(xmp.starts_with("<?xpacket begin=\"\u{feff}\" id=\"W5M0MpCehiHzreSzNTczkc9d\"?>"));
        assert!(xmp.ends_with("<?xpacket end=\"w\"?>"));
        assert!(xmp.contains("<xmp:CreatorTool>Yellowmoon 0.0.1</xmp:CreatorTool>"));
        assert!(xmp.contains("xmlns:ym=\"https://yellowmoon.app/ns/1.0/\""));
        for name in [
            "PipelineVersion",
            "Profile",
            "ProfileVersion",
            "ProfileMatch",
            "Look",
            "LookSHA256",
            "LookIntensity",
            "DisplayTransform",
            "Models",
            "Auto",
            "AutoAdvisor",
            "FilmData",
            "Settings",
            "AutoFields",
        ] {
            assert!(xmp.contains(&format!("<ym:{name}>")), "{name}");
        }
        // In the order of docs/09: ym:Auto and ym:AutoAdvisor after ym:Models, ym:Settings last;
        // ym:AppleAuto is no longer written (docs/22 section 14.2).
        let at = |name: &str| xmp.find(&format!("<ym:{name}>")).unwrap();
        assert!(at("Models") < at("Auto") && at("Auto") < at("AutoAdvisor"));
        assert!(at("AutoAdvisor") < at("FilmData") && at("FilmData") < at("Settings"));
        assert!(
            at("Settings") < at("AutoFields"),
            "AutoFields follows Settings"
        );
        assert!(!xmp.contains("AppleAuto"));
        assert!(xmp.contains("<ym:Auto>apple</ym:Auto>"));
        assert!(xmp.contains(
            "<ym:AutoAdvisor>{&quot;model&quot;:&quot;AFM 3 Core&quot;,&quot;provider&quot;:&quot;apple&quot;,&quot;result&quot;:&quot;answered&quot;}</ym:AutoAdvisor>"
        ));
        assert!(xmp.contains("<ym:LookIntensity>100</ym:LookIntensity>"));
        // Compact JSON with sorted keys, escaped for XML.
        assert!(xmp.contains(
            "<ym:Settings>{&quot;exposure&quot;:{&quot;manual_ev&quot;:0.3},&quot;look&quot;:&quot;x&quot;}</ym:Settings>"
        ));
        assert!(xmp.contains("film &lt;warm&gt; &amp; &quot;cool&quot;"));
        // Timestamps are the capture time.
        assert!(xmp.contains("<xmp:CreateDate>2026-10-04T14:22:31+02:00</xmp:CreateDate>"));
        assert!(xmp.contains("<xmp:ModifyDate>2026-10-04T14:22:31+02:00</xmp:ModifyDate>"));
    }

    #[test]
    fn packet_is_deterministic_and_omits_missing_fields() {
        let a = build_xmp(&sample(), "image/png").unwrap();
        let b = build_xmp(&sample(), "image/png").unwrap();
        assert_eq!(a, b);
        let bare = String::from_utf8(
            build_xmp(
                &XmpInfo {
                    app_version: "9.9.9".into(),
                    ..XmpInfo::default()
                },
                "image/tiff",
            )
            .unwrap(),
        )
        .unwrap();
        assert!(bare.contains("Yellowmoon 9.9.9"));
        assert!(!bare.contains("ym:Look"));
        assert!(!bare.contains("CreateDate"));
    }

    #[test]
    fn control_characters_never_reach_the_packet() {
        let mut info = sample();
        info.ym.profile = Some("a\u{0}b\u{1b}c".into());
        let xmp = String::from_utf8(build_xmp(&info, "image/jpeg").unwrap()).unwrap();
        assert!(xmp.contains("<ym:Profile>abc</ym:Profile>"));
    }
}
