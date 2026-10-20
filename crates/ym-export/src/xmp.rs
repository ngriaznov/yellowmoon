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
    /// `ym:DisplayTransform` (`aces2-sdr100-p3lim`, `arri-display:…`).
    pub display_transform: Option<String>,
    /// `ym:Models`, the networks used (`jdd 1.2.0 (sha256…)`).
    pub models: Option<String>,
    /// `ym:Auto`, which Auto made the photo's automatic values: `rule` or `apple` (`docs/19`
    /// section 9).
    pub auto: Option<String>,
    /// `ym:AppleAuto`, what Apple Auto said about the photo: compact JSON with sorted keys, only when
    /// Apple Auto was on for it (`docs/19` section 9).
    pub apple_auto: Option<String>,
    /// `ym:Settings`, the resolved develop settings; written as compact JSON.
    pub settings: Option<serde_json::Value>,
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
    let fields: [(&str, Option<&str>); 11] = [
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
        ("ym:AppleAuto", ym.apple_auto.as_deref()),
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
    }
    x.push_str("  </rdf:Description>\n");
    x.push_str(" </rdf:RDF>\n");
    x.push_str("</x:xmpmeta>\n");
    x.push_str("<?xpacket end=\"w\"?>");
    Ok(x.into_bytes())
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
                display_transform: Some("aces2-sdr100-p3lim".into()),
                models: Some("jdd 1.2.0 (sha256:ff)".into()),
                auto: Some("apple".into()),
                apple_auto: Some(r#"{"model":"AFM 3 Core","result":"answered"}"#.into()),
                settings: Some(
                    serde_json::json!({ "exposure": { "manual_ev": 0.3 }, "look": "x" }),
                ),
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
            "AppleAuto",
            "Settings",
        ] {
            assert!(xmp.contains(&format!("<ym:{name}>")), "{name}");
        }
        // In the order of docs/09: ym:Auto and ym:AppleAuto after ym:Models, ym:Settings last.
        let at = |name: &str| xmp.find(&format!("<ym:{name}>")).unwrap();
        assert!(at("Models") < at("Auto") && at("Auto") < at("AppleAuto"));
        assert!(at("AppleAuto") < at("Settings"));
        assert!(xmp.contains("<ym:Auto>apple</ym:Auto>"));
        assert!(xmp.contains(
            "<ym:AppleAuto>{&quot;model&quot;:&quot;AFM 3 Core&quot;,&quot;result&quot;:&quot;answered&quot;}</ym:AppleAuto>"
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
