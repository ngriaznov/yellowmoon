//! `ym_export::xmp::read_ym` reads back what the writers put in a JPEG, a TIFF, a PNG and a DNG (`docs/30` section 12): the
//! settings, `ym:AutoFields`, the creator tool and the pipeline version, also when another tool's packet comes first.

mod common;

use ym_export::dng::{BaselineExposure, DngProfile};
use ym_export::xmp::{read_ym, read_ym_bytes};
use ym_export::*;

const AUTO_FIELDS: &str = r#"{"input":{"exposure":true},"main":{"black":false},"picked_white":{"kelvin":5300.0,"tint":4.0}}"#;

fn meta() -> Metadata {
    let mut m = common::metadata();
    m.xmp.ym.settings = Some(serde_json::json!({
        "nodes": { "items": [{ "id": "input", "note": "a <b> & \"c\" 'd'" }] }
    }));
    m.xmp.ym.auto_fields = Some(AUTO_FIELDS.to_owned());
    m
}

fn encode(format: ExportFormat, meta: &Metadata) -> Vec<u8> {
    let (w, h) = (24, 16);
    let display = common::display_scene(w, h);
    let raw = common::awg4_scene(w, h);
    let profile = DngProfile::identity();
    let dng = (format == ExportFormat::Dng).then(|| DngInputs {
        preview: ImageRef::new(w, h, &display, PixelEncoding::DisplayLinear).unwrap(),
        orientation: 1,
        crop: None,
        baseline_exposure: BaselineExposure::default(),
        profile: &profile,
    });
    let (image, encoding) = if dng.is_some() {
        (&raw, PixelEncoding::SceneLinearAwg4)
    } else {
        (&display, PixelEncoding::DisplayLinear)
    };
    let req = ExportRequest {
        pixels: ImageRef::new(w, h, image, encoding).unwrap(),
        format,
        color_space: ColorSpace::Srgb,
        size: Size::Full,
        metadata: meta,
        remove_location: false,
        dng,
    };
    encode_to_vec(&req, &ExportHooks::none()).unwrap().0
}

fn formats() -> Vec<(&'static str, ExportFormat)> {
    vec![
        (
            "jpeg",
            ExportFormat::Jpeg {
                quality: JpegQuality::new(90).unwrap(),
            },
        ),
        (
            "tiff",
            ExportFormat::Tiff {
                bits: BitDepth::Eight,
                compression: TiffCompression::Zip,
            },
        ),
        (
            "png",
            ExportFormat::Png {
                bits: BitDepth::Eight,
            },
        ),
        ("dng", ExportFormat::Dng),
    ]
}

#[test]
fn what_the_writers_wrote_is_what_read_ym_reads() {
    let meta = meta();
    for (name, format) in formats() {
        let bytes = encode(format, &meta);
        let read = read_ym_bytes(&bytes).unwrap_or_else(|| panic!("{name}: nothing read"));
        assert_eq!(
            read.settings,
            meta.xmp.ym.settings.clone().unwrap(),
            "{name}"
        );
        assert_eq!(read.auto_fields.as_deref(), Some(AUTO_FIELDS), "{name}");
        assert_eq!(
            read.creator_tool.as_deref(),
            Some("Yellowmoon 0.0.1"),
            "{name}"
        );
        assert_eq!(read.pipeline_version.as_deref(), Some("1.0.0"), "{name}");
    }
}

#[test]
fn a_file_is_read_from_its_path_and_a_file_without_settings_says_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("a.jpg");
    std::fs::write(&path, encode(formats().remove(0).1, &meta())).unwrap();
    assert!(read_ym(&path).unwrap().is_some());
    // An older export has no ym:AutoFields.
    let mut old = meta();
    old.xmp.ym.auto_fields = None;
    let bytes = encode(formats().remove(2).1, &old);
    let read = read_ym_bytes(&bytes).unwrap();
    assert_eq!(read.auto_fields, None);
    // No ym:Settings (not an export, or settings left out): nothing, and AutoFields goes with them.
    let mut none = meta();
    none.xmp.ym.settings = None;
    for (name, format) in formats() {
        let bytes = encode(format, &none);
        assert!(read_ym_bytes(&bytes).is_none(), "{name}");
        assert!(
            !String::from_utf8_lossy(&bytes).contains("ym:AutoFields"),
            "{name}"
        );
    }
    assert!(read_ym_bytes(b"{\"nodes\":{}}").is_none());
    assert!(read_ym_bytes(&[]).is_none());
    assert!(read_ym(&dir.path().join("missing.jpg")).is_err());
}

#[test]
fn a_packet_from_another_tool_before_ours_is_skipped() {
    let bytes = encode(formats().remove(0).1, &meta());
    // A JPEG with a foreign XMP APP1 right after SOI.
    let mut packet = b"http://ns.adobe.com/xap/1.0/\0".to_vec();
    packet.extend_from_slice(b"<x:xmpmeta><xmp:CreatorTool>Other</xmp:CreatorTool></x:xmpmeta>");
    let mut file = vec![0xFF, 0xD8, 0xFF, 0xE1];
    file.extend_from_slice(&u16::try_from(packet.len() + 2).unwrap().to_be_bytes());
    file.extend_from_slice(&packet);
    file.extend_from_slice(&bytes[2..]);
    let read = read_ym_bytes(&file).expect("ours is found");
    assert_eq!(read.creator_tool.as_deref(), Some("Yellowmoon 0.0.1"));
}
