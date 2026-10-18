//! ExifTool as a second independent oracle for the tags LibRaw does not expose: Sony model ID,
//! ARW version, `SonyISO`, the SR2 values, lens-correction tables, tone curve, previews.
//!
//! The test needs ExifTool: either `exiftool` on `PATH` or, in this repository's build
//! environment, a clone of github.com/exiftool/exiftool with the path of its `exiftool` script in
//! `YM_EXIFTOOL` (it is run with `perl`). Without either it prints a skip message and passes,
//! unless `YM_REQUIRE_ORACLE=1` is set. ExifTool is a Perl program run as a separate process; it
//! is only ever a test oracle.

use std::path::Path;
use std::process::Command;

use serde_json::Value;
use ym_raw::exif::{ExposureProgram, FlashInfo, MeteringMode, SceneCaptureType};
use ym_raw::makernote::{CreativeStyle, DroMode, SceneMode};
use ym_raw::metadata::{probe, read_preview};
use ym_raw::testgen::{ArwBuilder, SonyIsoTable, ThumbnailIfd, gradient_mosaic};
use ym_raw::tiff::{field_type, tag};

fn exiftool() -> Option<Command> {
    let candidates: Vec<Command> = match std::env::var_os("YM_EXIFTOOL") {
        Some(path) => {
            let mut perl = Command::new("perl");
            perl.arg(path);
            vec![perl]
        }
        None => vec![Command::new("exiftool")],
    };
    candidates.into_iter().find_map(|mut command| {
        let output = command.arg("-ver").output().ok()?;
        output.status.success().then(|| {
            // Rebuild a command without the `-ver` argument.
            match std::env::var_os("YM_EXIFTOOL") {
                Some(path) => {
                    let mut perl = Command::new("perl");
                    perl.arg(path);
                    perl
                }
                None => Command::new("exiftool"),
            }
        })
    })
}

fn oracle_ready() -> bool {
    if exiftool().is_some() {
        return true;
    }
    if std::env::var_os("YM_REQUIRE_ORACLE").is_some() {
        panic!("YM_REQUIRE_ORACLE is set but ExifTool is not available");
    }
    eprintln!("SKIP oracle_exiftool: ExifTool is not available (set YM_EXIFTOOL)");
    false
}

/// `exiftool -j -n -G1 -a -u FILE` as one JSON object (numbers are not pretty-printed).
fn tags(path: &Path) -> serde_json::Map<String, Value> {
    let output = exiftool()
        .expect("exiftool")
        .args(["-j", "-n", "-G1", "-a", "-u"])
        .arg(path)
        .output()
        .expect("run exiftool");
    let value: Value = serde_json::from_slice(&output.stdout)
        .unwrap_or_else(|e| panic!("ExifTool output is not JSON ({e})"));
    value[0].as_object().expect("one object per file").clone()
}

fn binary(path: &Path, tag: &str) -> Vec<u8> {
    exiftool()
        .expect("exiftool")
        .args(["-b", &format!("-{tag}")])
        .arg(path)
        .output()
        .expect("run exiftool")
        .stdout
}

fn numbers(value: &Value) -> Vec<f64> {
    match value {
        Value::String(text) => text
            .split_whitespace()
            .map(|t| t.parse().unwrap_or_else(|_| panic!("number expected: {t}")))
            .collect(),
        Value::Number(n) => vec![n.as_f64().expect("f64")],
        other => panic!("unexpected value {other}"),
    }
}

/// `2026:10:04 12:03:44.07+02:00` -> `2026-10-04T12:03:44.070+02:00`.
fn iso_from_exiftool(text: &str) -> String {
    let (date, rest) = text.split_once(' ').expect("date and time");
    let date = date.replace(':', "-");
    let (time, tail) = rest.split_at(8);
    let (fraction, offset) = match tail.strip_prefix('.') {
        Some(after) => {
            let digits: String = after.chars().take_while(char::is_ascii_digit).collect();
            let offset = after[digits.len()..].to_owned();
            (format!(".{digits:0<3}"), offset)
        }
        None => (String::new(), tail.to_owned()),
    };
    format!("{date}T{time}{fraction}{offset}")
}

fn get<'a>(tags: &'a serde_json::Map<String, Value>, key: &str) -> &'a Value {
    tags.get(key).unwrap_or_else(|| {
        panic!(
            "ExifTool did not report {key}; it reported: {:?}",
            tags.keys().collect::<Vec<_>>()
        )
    })
}

#[test]
fn exiftool_agrees_with_the_reader_on_synthetic_files() {
    if !oracle_ready() {
        return;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    let bodies: [(&str, ArwBuilder, u32, u32); 6] = [
        (
            "ILCE-7RM4",
            ArwBuilder::new("ILCE-7RM4", 128, 96).iso(1600),
            128,
            96,
        ),
        (
            "ILCE-1",
            ArwBuilder::new("ILCE-1", 160, 96)
                .iso(25_600)
                .raw_ifd_levels(true)
                .black([510, 512, 514, 516])
                .white([16200, 16250, 16300])
                .wb(Some([2310, 1024, 1030, 1480]))
                .tone_curve(Some([1000, 4000, 8000, 12_000]))
                .exposure_time(1, 4000)
                .f_number(28, 10)
                .focal_length(850, 10),
            160,
            96,
        ),
        (
            "ILCE-6400",
            ArwBuilder::new("ILCE-6400", 96, 64)
                .iso(100)
                .exposure_time(30, 1)
                .capture_time("2024:12:31 23:59:59", "123", Some("-05:30")),
            96,
            64,
        ),
        (
            "DSC-RX100M4",
            ArwBuilder::new("DSC-RX100M4", 96, 64).iso(125),
            96,
            64,
        ),
        // The older maker-note layout with the 12-byte header (every body above writes the
        // modern, headerless one).
        (
            "ILCE-7M3 with the SONY DSC header",
            ArwBuilder::new("ILCE-7M3", 96, 64)
                .maker_note_header(b"SONY DSC \0\0\0")
                .iso(800),
            96,
            64,
        ),
        // Laid out like the real ILCE-9 file in github.com/lclevy/sony_raw: measured SonyISO
        // 2619 next to EXIF ISO 2500, SR2 white 15360 next to the raw IFD's WhiteLevel 16383,
        // 0x7313 as signed 16-bit values.
        (
            "ILCE-9 as in the lclevy dump",
            ArwBuilder::new("ILCE-9", 96, 64)
                .iso(2500)
                .sony_iso_word(2890)
                .white([15360; 3])
                .wb(Some([2364, 1024, 1024, 1728]))
                .raw_ifd_levels(true)
                .raw_ifd_white_level(16383)
                .file_format([3, 3, 3, 0])
                .tone_curve(Some([8000, 10_400, 12_900, 14_100])),
            96,
            64,
        ),
    ];
    for (index, (name, builder, w, h)) in bodies.into_iter().enumerate() {
        let path = dir.path().join(format!("{index}.ARW"));
        builder
            .write_to(&path, &gradient_mosaic(w, h, 600, 16_000, 9))
            .expect("write");
        let ours = probe(&path).unwrap_or_else(|e| panic!("{name}: {e}"));
        let t = tags(&path);

        assert_eq!(get(&t, "File:FileType"), "ARW", "{name}");
        assert_eq!(
            get(&t, "IFD0:Make").as_str(),
            Some(ours.make.as_str()),
            "{name}"
        );
        assert_eq!(
            get(&t, "IFD0:Model").as_str(),
            Some(ours.model.as_str()),
            "{name}"
        );
        assert_eq!(
            get(&t, "Sony:SonyModelID").as_u64(),
            ours.model_id.map(u64::from),
            "{name}: model ID"
        );
        let version = numbers(get(&t, "Sony:FileFormat"));
        let version: Vec<u8> = version.iter().map(|v| *v as u8).collect();
        assert_eq!(
            Some(<[u8; 4]>::try_from(version).expect("4 bytes")),
            ours.arw_version,
            "{name}"
        );

        // ISO: ExifTool reports SonyISO from the same enciphered tables.
        assert_eq!(
            numbers(get(&t, "Sony:SonyISO"))[0].round() as u32,
            ours.iso.expect("iso"),
            "{name}: SonyISO"
        );

        // Raw image geometry and layout.
        assert_eq!(
            get(&t, "SubIFD:ImageWidth").as_u64(),
            Some(u64::from(ours.width)),
            "{name}"
        );
        assert_eq!(
            get(&t, "SubIFD:ImageHeight").as_u64(),
            Some(u64::from(ours.height)),
            "{name}"
        );
        assert_eq!(
            get(&t, "SubIFD:BitsPerSample").as_u64(),
            Some(u64::from(ours.bits_per_sample))
        );
        assert_eq!(
            get(&t, "SubIFD:Compression").as_u64(),
            Some(u64::from(ours.compression))
        );
        assert_eq!(
            get(&t, "SubIFD:PhotometricInterpretation").as_u64(),
            Some(u64::from(ours.photometric))
        );
        let (offsets, counts) = ours.storage.chunks();
        assert_eq!(
            numbers(get(&t, "SubIFD:StripOffsets")),
            offsets.iter().map(|v| *v as f64).collect::<Vec<_>>()
        );
        assert_eq!(
            numbers(get(&t, "SubIFD:StripByteCounts")),
            counts.iter().map(|v| *v as f64).collect::<Vec<_>>()
        );

        // Exposure.
        let close = |a: f64, b: f64| (a - b).abs() <= 1e-9 * b.abs().max(1.0);
        assert!(
            close(
                numbers(get(&t, "ExifIFD:ExposureTime"))[0],
                ours.exposure_time_s.expect("time")
            ),
            "{name}"
        );
        assert!(
            close(
                numbers(get(&t, "ExifIFD:FNumber"))[0],
                ours.f_number.expect("f")
            ),
            "{name}"
        );
        assert!(
            close(
                numbers(get(&t, "ExifIFD:FocalLength"))[0],
                ours.focal_length_mm.expect("fl")
            ),
            "{name}"
        );
        assert!(
            close(
                numbers(get(&t, "ExifIFD:ExposureCompensation"))[0],
                ours.exposure_comp_ev.expect("ec")
            ),
            "{name}"
        );
        assert_eq!(
            get(&t, "ExifIFD:LensModel").as_str(),
            ours.lens_model.as_deref(),
            "{name}"
        );
        let taken = ours.capture_time.expect("time");
        assert_eq!(
            get(&t, "ExifIFD:DateTimeOriginal").as_str(),
            Some(taken.to_exif_string().as_str()),
            "{name}"
        );
        // ExifTool's composite combines the date, sub-second digits and offset.
        let theirs_time = iso_from_exiftool(
            get(&t, "Composite:SubSecDateTimeOriginal")
                .as_str()
                .expect("SubSecDateTimeOriginal"),
        );
        assert_eq!(theirs_time, taken.to_iso8601(), "{name}: capture time");

        // SR2 values.
        let black = numbers(get(&t, "SR2SubIFD:BlackLevel"));
        let ours_black: Vec<f64> = ours.black.iter().map(|v| f64::from(*v)).collect();
        assert_eq!(black, ours_black, "{name}: black level");
        let wb = numbers(get(&t, "SR2SubIFD:WB_RGGBLevels"));
        let ours_wb: Vec<f64> = ours
            .wb_as_shot
            .expect("wb")
            .iter()
            .map(|v| f64::from(*v))
            .collect();
        assert_eq!(wb, ours_wb, "{name}: white balance");
        let white = numbers(get(&t, "SR2SubIFD:WhiteLevel"));
        // ExifTool lists R, G, B; ours is R, G1, G2, B.
        assert_eq!(
            white,
            vec![
                f64::from(ours.white[0]),
                f64::from(ours.white[1]),
                f64::from(ours.white[3])
            ],
            "{name}: white level"
        );

        // Lens tables and the tone curve, exactly as stored.
        let lens = ours.lens_corrections.as_ref().expect("lens tables");
        for (key, table) in [
            (
                "SubIFD:VignettingCorrParams",
                lens.vignetting.as_ref().expect("v"),
            ),
            (
                "SubIFD:ChromaticAberrationCorrParams",
                lens.chromatic_aberration.as_ref().expect("c"),
            ),
            (
                "SubIFD:DistortionCorrParams",
                lens.distortion.as_ref().expect("d"),
            ),
        ] {
            let theirs = numbers(get(&t, key));
            let ours_raw: Vec<f64> = table.raw.iter().map(|v| f64::from(*v)).collect();
            assert_eq!(theirs, ours_raw, "{name}: {key}");
        }
        if let Some(curve) = ours.tone_curve {
            let theirs = numbers(get(&t, "SubIFD:SonyToneCurve"));
            assert_eq!(
                theirs,
                curve.iter().map(|v| f64::from(*v)).collect::<Vec<_>>(),
                "{name}"
            );
        }

        // The preview the reader picks is the maker-note PreviewImage ExifTool extracts.
        let preview = read_preview(&path).expect("probe").expect("preview");
        let theirs = binary(&path, "PreviewImage");
        assert_eq!(theirs.len(), preview.len(), "{name}: PreviewImage length");
        assert!(theirs == preview, "{name}: PreviewImage bytes differ");
    }
}

#[test]
fn exiftool_finds_the_same_thumbnail_the_reader_does_when_there_is_no_larger_preview() {
    if !oracle_ready() {
        return;
    }
    // ExifTool takes ThumbnailImage from IFD1 only, so only that placement is compared; the
    // reader also accepts the tags in IFD0 (tested against LibRaw, which accepts both).
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("ifd1.ARW");
    ArwBuilder::new("ILCE-7M3", 64, 48)
        .thumbnail(Some((200, 150)))
        .thumbnail_ifd(ThumbnailIfd::Ifd1)
        .preview(None)
        .write_to(&path, &gradient_mosaic(64, 48, 600, 16_000, 2))
        .expect("write");
    let ours = read_preview(&path).expect("probe").expect("thumbnail");
    let theirs = binary(&path, "ThumbnailImage");
    assert_eq!(theirs.len(), ours.len(), "thumbnail length");
    assert!(
        theirs == ours,
        "ExifTool's ThumbnailImage differs from the reader's preview"
    );
}

/// The enciphered maker-note tables carry the exposure time, the F-number and (0x9416 only) the
/// focal length the builder was given, at the offsets ExifTool's `Tag9405b` and `Tag9416` tables
/// read. ExifTool's `[Sony]` group reports them next to `SonyISO`, which the reader also takes.
#[test]
fn exiftool_reads_exposure_and_focal_length_from_the_enciphered_tables() {
    if !oracle_ready() {
        return;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    // (name, builder, table has a focal length)
    let cases: [(&str, ArwBuilder, bool); 4] = [
        (
            "ILCE-7M4 (0x9416)",
            ArwBuilder::new("ILCE-7M4", 96, 64)
                .iso(3200)
                .exposure_time(1, 125)
                .f_number(28, 10)
                .focal_length(500, 10),
            true,
        ),
        (
            "ILCE-1 (0x9416), long exposure",
            ArwBuilder::new("ILCE-1", 96, 64)
                .iso(100)
                .exposure_time(30, 1)
                .f_number(110, 10)
                .focal_length(1050, 10),
            true,
        ),
        (
            "ILCE-7M3 (0x9405)",
            ArwBuilder::new("ILCE-7M3", 96, 64)
                .iso(800)
                .exposure_time(1, 4000)
                .f_number(56, 10)
                .focal_length(240, 10),
            false,
        ),
        (
            "ILCE-7RM4 (0x9405), ISO word of a measured sensitivity",
            ArwBuilder::new("ILCE-7RM4", 96, 64)
                .iso(2500)
                .sony_iso_word(2890)
                .exposure_time(1, 60)
                .f_number(40, 10)
                .focal_length(240, 10),
            false,
        ),
    ];
    for (index, (name, builder, has_focal_length)) in cases.into_iter().enumerate() {
        let path = dir.path().join(format!("tables{index}.ARW"));
        builder
            .write_to(&path, &gradient_mosaic(96, 64, 600, 16_000, 5))
            .expect("write");
        let ours = probe(&path).unwrap_or_else(|e| panic!("{name}: {e}"));
        let t = tags(&path);

        // Our reader and ExifTool agree on the sensitivity, and both read it from the table.
        assert_eq!(
            ours.iso_source,
            Some(ym_raw::exif::IsoSource::SonyMakerNote),
            "{name}"
        );
        assert_eq!(
            numbers(get(&t, "Sony:SonyISO"))[0].round() as u32,
            ours.iso.expect("iso"),
            "{name}: SonyISO"
        );

        // The exposure time is stored twice: as a 1/256 stop word and as a fraction of 16 bits.
        let seconds = ours.exposure_time_s.expect("exposure time");
        let word = numbers(get(&t, "Sony:SonyExposureTime2"))[0];
        assert!(
            (word / seconds - 1.0).abs() < 0.0028,
            "{name}: SonyExposureTime2 {word} s against {seconds} s"
        );
        let fraction = numbers(get(&t, "Sony:ExposureTime"))[0];
        assert!(
            // ExifTool prints numbers with eight significant digits.
            (fraction / seconds - 1.0).abs() < 1e-6,
            "{name}: Sony ExposureTime {fraction} s against {seconds} s"
        );

        // The F-number is a 1/256 step of 2 * log2(f).
        let f_number = ours.f_number.expect("f-number");
        let key = if has_focal_length {
            "Sony:SonyFNumber2"
        } else {
            "Sony:SonyFNumber"
        };
        let theirs = numbers(get(&t, key))[0];
        assert!(
            (theirs / f_number - 1.0).abs() < 0.003,
            "{name}: {key} {theirs} against {f_number}"
        );

        // Tag9416 has a focal length field; Tag9405b has none.
        if has_focal_length {
            assert_eq!(
                numbers(get(&t, "Sony:FocalLength"))[0],
                ours.focal_length_mm.expect("focal length"),
                "{name}: Sony FocalLength"
            );
        } else {
            assert!(!t.contains_key("Sony:FocalLength"), "{name}");
        }
    }
}

#[test]
fn exiftool_agrees_with_the_reader_on_the_exposure_context_and_sony_codes() {
    if !oracle_ready() {
        return;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    let short = |v: u16| v.to_le_bytes().to_vec();
    let long = |v: u32| v.to_le_bytes().to_vec();
    let name16 = |s: &str| {
        let mut bytes = s.as_bytes().to_vec();
        bytes.resize(16, 0);
        bytes
    };
    // (name, builder, the codes the file holds, checked against ExifTool and the reader)
    let cases: [(&str, ArwBuilder); 2] = [
        (
            "ILCE-7M3, plain tags",
            ArwBuilder::new("ILCE-7M3", 96, 64)
                .exif_entry(
                    tag::BRIGHTNESS_VALUE,
                    field_type::SRATIONAL,
                    1,
                    [(-3i32).to_le_bytes(), 2i32.to_le_bytes()].concat(),
                )
                .exif_entry(tag::FLASH, field_type::SHORT, 1, short(0x19))
                .exif_entry(tag::EXPOSURE_PROGRAM, field_type::SHORT, 1, short(3))
                .exif_entry(tag::SCENE_CAPTURE_TYPE, field_type::SHORT, 1, short(2))
                .maker_entry(
                    ym_raw::makernote::TAG_CREATIVE_STYLE,
                    field_type::ASCII,
                    16,
                    name16("Vivid"),
                )
                .maker_entry(
                    ym_raw::makernote::TAG_DRO_MODE,
                    field_type::LONG,
                    1,
                    long(3),
                )
                .maker_entry(
                    ym_raw::makernote::TAG_SCENE_MODE,
                    field_type::LONG,
                    1,
                    long(6),
                ),
        ),
        (
            "ILCE-7M4, Creative Look code only in the 0x9416 table",
            ArwBuilder::new("ILCE-7M4", 96, 64)
                .iso_table(SonyIsoTable::Tag9416)
                .creative_style_9416(15),
        ),
    ];
    for (index, (name, builder)) in cases.into_iter().enumerate() {
        let path = dir.path().join(format!("context{index}.ARW"));
        builder
            .write_to(&path, &gradient_mosaic(96, 64, 600, 16_000, 5))
            .expect("write");
        let ours = probe(&path).unwrap_or_else(|e| panic!("{name}: {e}"));
        let t = tags(&path);
        let code = |key: &str| numbers(get(&t, key))[0];

        if index == 0 {
            assert_eq!(code("ExifIFD:BrightnessValue"), -1.5, "{name}");
            assert_eq!(ours.brightness_value, Some(-1.5), "{name}");
            assert_eq!(code("ExifIFD:MeteringMode"), 5.0, "{name}");
            assert_eq!(
                ours.metering_mode,
                Some(MeteringMode::Multi),
                "{name}: metering"
            );
            assert_eq!(code("ExifIFD:Flash"), 25.0, "{name}");
            assert_eq!(
                ours.flash,
                Some(FlashInfo {
                    fired: true,
                    raw: 0x19
                }),
                "{name}: flash"
            );
            assert_eq!(code("ExifIFD:ExposureProgram"), 3.0, "{name}");
            assert_eq!(
                ours.exposure_program,
                Some(ExposureProgram::AperturePriorityAe),
                "{name}: exposure program"
            );
            assert_eq!(code("ExifIFD:SceneCaptureType"), 2.0, "{name}");
            assert_eq!(
                ours.scene_capture_type,
                Some(SceneCaptureType::Portrait),
                "{name}: scene capture type"
            );
            assert_eq!(code("Sony:DynamicRangeOptimizer"), 3.0, "{name}");
            assert_eq!(ours.dro_mode, Some(DroMode::Auto), "{name}: DRO");
            assert_eq!(
                get(&t, "Sony:CreativeStyle").as_str(),
                Some("Vivid"),
                "{name}: CreativeStyle"
            );
            assert_eq!(
                ours.creative_style,
                Some(CreativeStyle::Vivid),
                "{name}: creative style"
            );
            assert_eq!(code("Sony:SceneMode"), 6.0, "{name}");
            assert_eq!(ours.scene_mode, Some(SceneMode::Landscape), "{name}: scene");
        } else {
            // ExifTool reads the 0x9416 byte at offset 55 as CreativeStyle 15 (FL).
            assert_eq!(code("Sony:CreativeStyle"), 15.0, "{name}");
            assert_eq!(
                ours.creative_style,
                Some(CreativeStyle::Other("FL".to_owned())),
                "{name}: creative style"
            );
        }
    }
}
