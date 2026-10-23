//! The camera's focus point and AF-area setting in the Sony maker note (`docs/20-auto-crop.md` §4):
//! tags 0x2027 FocusLocation, 0x204A FocusLocation2, 0x2037 FocusFrameSize and 0x201C
//! AFAreaModeSetting, read from synthetic files and checked against ExifTool on the real files in
//! `testdata/` when both are there.

use std::path::Path;
use std::process::Command;

use ym_raw::makernote::SonyFocus;
use ym_raw::metadata::{probe, probe_bytes};
use ym_raw::testgen::{ArwBuilder, gradient_mosaic};

fn file(builder: ArwBuilder) -> Vec<u8> {
    builder.build(&gradient_mosaic(64, 48, 512, 16000, 7))
}

#[test]
fn the_focus_point_its_box_and_the_af_area_setting_are_read() {
    let bytes = file(
        ArwBuilder::new("ILCE-7M4", 64, 48)
            .focus_location([7008, 4672, 3306, 3124])
            .focus_frame_size([153, 156, 257])
            .af_area_setting(3),
    );
    let meta = probe_bytes(&bytes).expect("probe");
    assert_eq!(
        meta.focus,
        Some(SonyFocus {
            frame: [7008, 4672],
            point: [3306, 3124],
            box_size: Some([153, 156]),
        })
    );
    assert_eq!(meta.af_area_setting, Some(3));
}

#[test]
fn focus_location2_wins_and_an_invalid_box_is_none() {
    let bytes = file(
        ArwBuilder::new("ILCE-1", 64, 48)
            .focus_location([5616, 3744, 2965, 1872])
            .focus_location2([5616, 3744, 2966, 1872])
            .focus_frame_size([122, 124, 0]),
    );
    let meta = probe_bytes(&bytes).expect("probe");
    assert_eq!(
        meta.focus,
        Some(SonyFocus {
            frame: [5616, 3744],
            point: [2966, 1872],
            box_size: None,
        })
    );
    assert_eq!(meta.af_area_setting, None);
}

#[test]
fn files_without_the_tags_have_no_focus() {
    let meta = probe_bytes(&file(ArwBuilder::new("NEX-7", 64, 48))).expect("probe");
    assert_eq!(meta.focus, None);
    assert_eq!(meta.af_area_setting, None);
}

/// The values ExifTool prints with `-n` for one tag, as integers.
fn exiftool_numbers(path: &Path, tag: &str) -> Option<Vec<u32>> {
    let output = Command::new("exiftool")
        .args(["-s3", "-n", &format!("-{tag}")])
        .arg(path)
        .output()
        .ok()?;
    let text = String::from_utf8(output.stdout).ok()?;
    let text = text.trim();
    if text.is_empty() {
        return None;
    }
    text.split_whitespace().map(|v| v.parse().ok()).collect()
}

#[test]
fn the_real_files_agree_with_exiftool() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata");
    let files: Vec<_> = ["real/files", "readme-photos/src"]
        .iter()
        .filter_map(|dir| std::fs::read_dir(root.join(dir)).ok())
        .flatten()
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e.eq_ignore_ascii_case("arw")))
        .collect();
    if files.is_empty() || Command::new("exiftool").arg("-ver").output().is_err() {
        eprintln!("SKIP: no real ARW files in testdata/ or no exiftool on PATH");
        return;
    }
    let mut checked = 0;
    for path in &files {
        let meta = probe(path).expect("probe");
        let location = exiftool_numbers(path, "FocusLocation2")
            .or_else(|| exiftool_numbers(path, "FocusLocation"))
            .filter(|v| v.len() == 4);
        let box_size = exiftool_numbers(path, "FocusFrameSize")
            .filter(|v| v.len() == 3 && v[2] != 0)
            .map(|v| [v[0] as u16, v[1] as u16]);
        let expected = location.map(|v| SonyFocus {
            frame: [v[0] as u16, v[1] as u16],
            point: [v[2] as u16, v[3] as u16],
            box_size,
        });
        assert_eq!(meta.focus, expected, "{}", path.display());
        let setting = exiftool_numbers(path, "AFAreaModeSetting").map(|v| v[0] as u8);
        assert_eq!(meta.af_area_setting, setting, "{}", path.display());
        checked += 1;
    }
    eprintln!("checked {checked} real files against ExifTool");
}
