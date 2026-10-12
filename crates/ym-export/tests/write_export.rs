//! `write_export`, end to end: names, destination folders, If-exists modes, cancel, progress,
//! sizes, metadata, and the guarantee that originals are never touched.

mod common;

use std::fs;
use std::path::Path;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU32, Ordering};

use ym_export::naming::{NameFields, Placement};
use ym_export::*;

fn request<'a>(
    image: &'a [f32],
    w: u32,
    h: u32,
    format: ExportFormat,
    meta: &'a Metadata,
) -> ExportRequest<'a> {
    ExportRequest {
        pixels: ImageRef::new(w, h, image, PixelEncoding::DisplayLinear).unwrap(),
        format,
        color_space: ColorSpace::Srgb,
        size: Size::Full,
        metadata: meta,
        remove_location: false,
        dng: None,
    }
}

fn fields() -> NameFields {
    NameFields {
        name: "DSC04204".into(),
        look: "neutral".into(),
        date: Some("2026-10-04".into()),
        seq: 3,
        camera: "ILCE-1".into(),
    }
}

fn jpeg() -> ExportFormat {
    ExportFormat::Jpeg {
        quality: JpegQuality::default(),
    }
}

#[test]
fn each_format_lands_in_the_destination_under_the_template_name() {
    let tmp = tempfile::tempdir().unwrap();
    let image = common::display_scene(48, 32);
    let meta = common::metadata();
    let cases = [
        (
            ExportFormat::Jpeg {
                quality: JpegQuality::default(),
            },
            "jpg",
        ),
        (
            ExportFormat::Tiff {
                bits: BitDepth::Sixteen,
                compression: TiffCompression::Lzw,
            },
            "tif",
        ),
        (
            ExportFormat::Png {
                bits: BitDepth::Eight,
            },
            "png",
        ),
    ];
    for (format, ext) in cases {
        let options = ExportOptions {
            destination: Some(tmp.path().join("out")),
            name_template: "{date}_{name}_{look}_{seq}".into(),
            ..ExportOptions::default()
        };
        let dest = ExportDestination::resolve(&options, &fields(), None, None).unwrap();
        let out = write_export(
            &request(&image, 48, 32, format, &meta),
            &dest,
            &ExportHooks::none(),
        )
        .unwrap();
        let expected = tmp
            .path()
            .join("out")
            .join(format!("2026-10-04_DSC04204_neutral_003.{ext}"));
        assert_eq!(out.path(), expected);
        assert!(matches!(out.placement, Placement::Written(_)));
        assert_eq!((out.width, out.height), (48, 32));
        assert_eq!(out.bytes, fs::metadata(&expected).unwrap().len());
        assert!(out.bytes > 100);
    }
    let names: Vec<_> = fs::read_dir(tmp.path().join("out"))
        .unwrap()
        .map(|e| e.unwrap().file_name())
        .collect();
    assert_eq!(names.len(), 3, "no temporary files left: {names:?}");
}

#[test]
fn default_destination_is_the_export_folder_next_to_the_source_folder() {
    let tmp = tempfile::tempdir().unwrap();
    let source_dir = tmp.path().join("Lisbon");
    fs::create_dir_all(&source_dir).unwrap();
    let source = source_dir.join("DSC04204.ARW");
    fs::write(&source, b"original bytes").unwrap();
    let image = common::display_scene(24, 16);
    let meta = common::metadata();
    let options = ExportOptions::default();
    let dest =
        ExportDestination::resolve(&options, &fields(), Some(&source), Some(tmp.path())).unwrap();
    let out = write_export(
        &request(&image, 24, 16, jpeg(), &meta),
        &dest,
        &ExportHooks::none(),
    )
    .unwrap();
    let expected = fs::canonicalize(tmp.path())
        .unwrap()
        .join("Lisbon export")
        .join("DSC04204_neutral.jpg");
    assert_eq!(out.path(), expected);
    assert_eq!(
        fs::read(&source).unwrap(),
        b"original bytes",
        "the original is untouched"
    );
}

#[test]
fn if_exists_modes() {
    let tmp = tempfile::tempdir().unwrap();
    let image = common::display_scene(24, 16);
    let meta = common::metadata();
    let run = |mode: IfExists| {
        let options = ExportOptions {
            destination: Some(tmp.path().to_path_buf()),
            if_exists: mode,
            ..ExportOptions::default()
        };
        let dest = ExportDestination::resolve(&options, &fields(), None, None).unwrap();
        write_export(
            &request(&image, 24, 16, jpeg(), &meta),
            &dest,
            &ExportHooks::none(),
        )
        .unwrap()
    };
    let first = run(IfExists::KeepBoth);
    assert_eq!(first.path(), tmp.path().join("DSC04204_neutral.jpg"));
    let second = run(IfExists::KeepBoth);
    assert_eq!(second.path(), tmp.path().join("DSC04204_neutral 2.jpg"));
    let third = run(IfExists::KeepBoth);
    assert_eq!(third.path(), tmp.path().join("DSC04204_neutral 3.jpg"));
    let skipped = run(IfExists::Skip);
    assert!(matches!(skipped.placement, Placement::Skipped(_)));
    assert_eq!(skipped.path(), tmp.path().join("DSC04204_neutral.jpg"));
    assert_eq!(skipped.bytes, 0);
    // Replace swaps the first file's content.
    let before = fs::read(tmp.path().join("DSC04204_neutral.jpg")).unwrap();
    let other = common::awg4_scene(24, 16); // different pixels so the bytes differ
    let img: Vec<f32> = other.iter().map(|v| v.clamp(0.0, 1.0)).collect();
    let options = ExportOptions {
        destination: Some(tmp.path().to_path_buf()),
        if_exists: IfExists::Replace,
        ..ExportOptions::default()
    };
    let dest = ExportDestination::resolve(&options, &fields(), None, None).unwrap();
    let replaced = write_export(
        &request(&img, 24, 16, jpeg(), &meta),
        &dest,
        &ExportHooks::none(),
    )
    .unwrap();
    assert!(matches!(replaced.placement, Placement::Replaced(_)));
    assert_ne!(
        fs::read(tmp.path().join("DSC04204_neutral.jpg")).unwrap(),
        before
    );
    assert_eq!(fs::read_dir(tmp.path()).unwrap().count(), 3);
}

#[test]
fn cancel_removes_the_partial_file_and_leaves_existing_files_alone() {
    let tmp = tempfile::tempdir().unwrap();
    let (w, h) = (900, 700);
    let image = vec![0.3_f32; (w * h * 3) as usize];
    let meta = common::metadata();
    let existing = tmp.path().join("DSC04204_neutral.png");
    fs::write(&existing, b"keep me").unwrap();
    for format in [
        ExportFormat::Png {
            bits: BitDepth::Sixteen,
        },
        ExportFormat::Tiff {
            bits: BitDepth::Sixteen,
            compression: TiffCompression::Lzw,
        },
        ExportFormat::Jpeg {
            quality: JpegQuality::default(),
        },
    ] {
        let polls = AtomicU32::new(0);
        // Cancel at the third poll: after the first band of pixels was converted and written.
        let cancel = || polls.fetch_add(1, Ordering::Relaxed) >= 2;
        let hooks = ExportHooks::new(Some(&cancel), None);
        let options = ExportOptions {
            destination: Some(tmp.path().to_path_buf()),
            if_exists: IfExists::Replace,
            ..ExportOptions::default()
        };
        let dest = ExportDestination::resolve(&options, &fields(), None, None).unwrap();
        let ext = format.extension();
        let result = write_export(&request(&image, w, h, format, &meta), &dest, &hooks);
        assert!(
            matches!(result, Err(ExportError::Cancelled)),
            "{ext}: {result:?}"
        );
        assert!(
            polls.load(Ordering::Relaxed) >= 3,
            "{ext}: cancel was polled during the work, not only at the start"
        );
        let names: Vec<_> = fs::read_dir(tmp.path())
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(
            names,
            vec!["DSC04204_neutral.png".to_owned()],
            "{ext}: only the pre-existing file remains"
        );
    }
    assert_eq!(fs::read(&existing).unwrap(), b"keep me");
}

#[test]
fn progress_goes_from_zero_to_one_without_going_back() {
    let tmp = tempfile::tempdir().unwrap();
    let image = common::display_scene(300, 200);
    let meta = common::metadata();
    for (format, size) in [
        (jpeg(), Size::Full),
        (jpeg(), Size::LongEdge { px: 120 }),
        (
            ExportFormat::Tiff {
                bits: BitDepth::Eight,
                compression: TiffCompression::Zip,
            },
            Size::Full,
        ),
        (
            ExportFormat::Png {
                bits: BitDepth::Sixteen,
            },
            Size::LongEdge { px: 150 },
        ),
    ] {
        let seen = Mutex::new(Vec::<f32>::new());
        let progress = |f: f32| seen.lock().unwrap().push(f);
        let hooks = ExportHooks::new(None, Some(&progress));
        let mut req = request(&image, 300, 200, format, &meta);
        req.size = size;
        let dest = ExportDestination {
            dir: tmp.path().to_path_buf(),
            file_stem: format!("p_{}_{:?}", format.extension(), size)
                .replace([' ', '{', '}', ':'], "_"),
            if_exists: IfExists::Replace,
            source_path: None,
        };
        write_export(&req, &dest, &hooks).unwrap();
        let v = seen.into_inner().unwrap();
        assert!(!v.is_empty());
        assert!(v.iter().all(|f| (0.0..=1.0).contains(f)), "{v:?}");
        assert!(
            v.windows(2).all(|w| w[0] <= w[1] + 1e-6),
            "{} {size:?} not monotone: {v:?}",
            format.extension()
        );
        assert!(
            (v.last().unwrap() - 1.0).abs() < 1e-6,
            "{} ends at 1: {v:?}",
            format.extension()
        );
    }
}

#[test]
fn long_edge_resizes_the_file() {
    let tmp = tempfile::tempdir().unwrap();
    let image = common::display_scene(400, 300);
    let meta = common::metadata();
    let mut req = request(
        &image,
        400,
        300,
        ExportFormat::Png {
            bits: BitDepth::Eight,
        },
        &meta,
    );
    req.size = Size::LongEdge { px: 100 };
    let dest = ExportDestination {
        dir: tmp.path().to_path_buf(),
        file_stem: "small".into(),
        if_exists: IfExists::Replace,
        source_path: None,
    };
    let out = write_export(&req, &dest, &ExportHooks::none()).unwrap();
    assert_eq!((out.width, out.height), (100, 75));
    let decoder = png::Decoder::new(std::io::BufReader::new(fs::File::open(out.path()).unwrap()));
    let reader = decoder.read_info().unwrap();
    assert_eq!((reader.info().width, reader.info().height), (100, 75));
    // The EXIF dimensions describe the resized image.
    let meta_tags = reader.info().exif_metadata.clone().unwrap();
    let ifd0 = common::read_ifd(
        &meta_tags,
        u32::from_le_bytes(meta_tags[4..8].try_into().unwrap()) as usize,
    );
    let exif = common::read_ifd(&meta_tags, ifd0.one(34665) as usize);
    assert_eq!((exif.one(40962), exif.one(40963)), (100, 75));
}

#[test]
fn a_target_that_is_the_original_is_refused() {
    let tmp = tempfile::tempdir().unwrap();
    let original = tmp.path().join("photo.jpg");
    fs::write(&original, b"original").unwrap();
    let image = common::display_scene(16, 16);
    let meta = common::metadata();
    for mode in [IfExists::Replace, IfExists::Skip] {
        let dest = ExportDestination {
            dir: tmp.path().to_path_buf(),
            file_stem: "photo".into(),
            if_exists: mode,
            source_path: Some(original.clone()),
        };
        let r = write_export(
            &request(&image, 16, 16, jpeg(), &meta),
            &dest,
            &ExportHooks::none(),
        );
        assert!(
            matches!(r, Err(ExportError::WouldOverwriteOriginal(_))),
            "{mode:?}: {r:?}"
        );
    }
    assert_eq!(fs::read(&original).unwrap(), b"original");
    assert_eq!(fs::read_dir(tmp.path()).unwrap().count(), 1);
}

/// The GPS directory of the JPEG's Exif segment, if there is one.
fn jpeg_has_gps(file: &[u8]) -> bool {
    let mut i = 2;
    while i + 4 <= file.len() {
        let marker = file[i + 1];
        let len = usize::from(u16::from_be_bytes([file[i + 2], file[i + 3]]));
        if marker == 0xE1 && &file[i + 4..i + 10] == b"Exif\0\0" {
            let tiff = &file[i + 10..i + 2 + len];
            let ifd0 = common::read_ifd(
                tiff,
                u32::from_le_bytes(tiff[4..8].try_into().unwrap()) as usize,
            );
            return ifd0.has(34853);
        }
        i += 2 + len;
    }
    panic!("no Exif segment");
}

#[test]
fn location_is_written_unless_removed() {
    let image = common::display_scene(32, 32);
    let meta = common::metadata();
    let with = encode_to_vec(
        &request(&image, 32, 32, jpeg(), &meta),
        &ExportHooks::none(),
    )
    .unwrap()
    .0;
    assert!(jpeg_has_gps(&with));
    let without = encode_to_vec(
        &request(&image, 32, 32, jpeg(), &meta).remove_location(true),
        &ExportHooks::none(),
    )
    .unwrap()
    .0;
    assert!(!jpeg_has_gps(&without));
    // Nothing of the position survives anywhere in the file (latitude 38 deg 43 min).
    let no_gps = common::metadata();
    let mut stripped = no_gps.clone();
    stripped.exif.gps = None;
    let other = encode_to_vec(
        &request(&image, 32, 32, jpeg(), &stripped),
        &ExportHooks::none(),
    )
    .unwrap()
    .0;
    assert_eq!(
        without, other,
        "remove_location gives the same bytes as having no location"
    );
}

#[test]
fn the_exif_never_carries_more_than_the_plain_struct() {
    // The orientation is 1 and there is no thumbnail or maker note, whatever the input.
    let image = common::display_scene(16, 16);
    let meta = common::metadata();
    let file = encode_to_vec(
        &request(&image, 16, 16, jpeg(), &meta),
        &ExportHooks::none(),
    )
    .unwrap()
    .0;
    let mut i = 2;
    while i + 4 <= file.len() {
        let marker = file[i + 1];
        let len = usize::from(u16::from_be_bytes([file[i + 2], file[i + 3]]));
        if marker == 0xE1 && &file[i + 4..i + 10] == b"Exif\0\0" {
            let tiff = &file[i + 10..i + 2 + len];
            let ifd0 = common::read_ifd(
                tiff,
                u32::from_le_bytes(tiff[4..8].try_into().unwrap()) as usize,
            );
            assert_eq!(ifd0.one(274), 1);
            assert_eq!(ifd0.next, 0, "no IFD1 thumbnail");
            let exif = common::read_ifd(tiff, ifd0.one(34665) as usize);
            assert!(!exif.has(37500), "no maker note");
            return;
        }
        i += 2 + len;
    }
    panic!("no Exif segment");
}

#[test]
fn estimate_is_in_the_right_range_for_real_files() {
    let image = common::display_scene(512, 384);
    let meta = common::metadata();
    for format in [
        ExportFormat::Tiff {
            bits: BitDepth::Sixteen,
            compression: TiffCompression::None,
        },
        ExportFormat::Tiff {
            bits: BitDepth::Eight,
            compression: TiffCompression::None,
        },
    ] {
        let (bytes, _) = encode_to_vec(
            &request(&image, 512, 384, format, &meta),
            &ExportHooks::none(),
        )
        .unwrap();
        let est = estimate_bytes(512, 384, format, Size::Full);
        let ratio = est as f64 / bytes.len() as f64;
        assert!(
            (0.99..1.02).contains(&ratio),
            "uncompressed TIFF estimate is exact up to metadata: {ratio}"
        );
    }
}

#[allow(dead_code)]
fn unused(_: &Path) {}
