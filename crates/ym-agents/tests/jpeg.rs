//! The picture's JPEG (`docs/22` section 6.1): decodes, one baseline scan, no metadata, deterministic.

mod common;

use common::gradient;
use ym_agents::jpeg::{JpegError, encode};
use ym_core::apple_auto::AdvisorPicture;

/// The markers of a JPEG with the length of each segment, up to the first scan.
fn segments(bytes: &[u8]) -> Vec<(u8, usize)> {
    let mut out = Vec::new();
    let mut at = 2;
    while at + 4 <= bytes.len() && bytes[at] == 0xFF {
        let marker = bytes[at + 1];
        let len = usize::from(bytes[at + 2]) << 8 | usize::from(bytes[at + 3]);
        out.push((marker, len));
        if marker == 0xDA {
            break;
        }
        at += 2 + len;
    }
    out
}

#[test]
fn the_jpeg_decodes_to_the_picture_within_six() {
    let picture = gradient(256, 192);
    let jpeg = encode(&picture).unwrap();
    assert_eq!(&jpeg[..2], [0xFF, 0xD8], "starts with SOI");
    assert_eq!(&jpeg[jpeg.len() - 2..], [0xFF, 0xD9], "ends with EOI");

    let mut decoder = zune_jpeg::JpegDecoder::new(std::io::Cursor::new(jpeg.clone()));
    let pixels = decoder.decode().unwrap();
    let info = decoder.info().unwrap();
    assert_eq!(
        (usize::from(info.width), usize::from(info.height)),
        (256, 192)
    );
    assert_eq!(pixels.len(), picture.rgb.len());
    let worst = pixels
        .iter()
        .zip(&picture.rgb)
        .map(|(a, b)| (i32::from(*a) - i32::from(*b)).abs())
        .max()
        .unwrap();
    assert!(worst <= 6, "worst error {worst}");
}

#[test]
fn it_is_one_baseline_scan_at_444_with_no_metadata() {
    let jpeg = encode(&gradient(64, 48)).unwrap();
    let segs = segments(&jpeg);
    let markers: Vec<u8> = segs.iter().map(|(m, _)| *m).collect();
    // One scan, baseline (SOF0), no progressive (SOF2), no EXIF/ICC/XMP (APP1, APP2), no comment.
    assert_eq!(markers.iter().filter(|m| **m == 0xDA).count(), 1);
    assert!(markers.contains(&0xC0), "baseline frame");
    assert!(!markers.contains(&0xC2), "not progressive");
    assert!(!markers.contains(&0xE1) && !markers.contains(&0xE2) && !markers.contains(&0xFE));
    // The frame header: three components, each sampled 1x1 (4:4:4).
    let sof = jpeg.windows(2).position(|w| w == [0xFF, 0xC0]).unwrap();
    let frame = &jpeg[sof + 4..];
    assert_eq!(frame[5], 3, "three components");
    for component in 0..3 {
        assert_eq!(
            frame[6 + component * 3 + 1],
            0x11,
            "component {component} sampled 1x1"
        );
    }
}

#[test]
fn the_same_picture_gives_the_same_bytes() {
    let picture = gradient(100, 75);
    assert_eq!(encode(&picture).unwrap(), encode(&picture).unwrap());
}

#[test]
fn a_picture_that_is_not_width_by_height_by_three_is_refused() {
    let bad = AdvisorPicture {
        width: 10,
        height: 10,
        rgb: vec![0; 299],
    };
    assert!(matches!(
        encode(&bad),
        Err(JpegError::Size {
            expected: 300,
            actual: 299,
            ..
        })
    ));
    let empty = AdvisorPicture {
        width: 0,
        height: 5,
        rgb: Vec::new(),
    };
    assert!(matches!(encode(&empty), Err(JpegError::Empty { .. })));
    let huge = AdvisorPicture {
        width: 70_000,
        height: 1,
        rgb: vec![0; 210_000],
    };
    assert!(matches!(encode(&huge), Err(JpegError::TooLarge { .. })));
}
