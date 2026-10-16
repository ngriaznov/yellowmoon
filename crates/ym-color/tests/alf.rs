//! ARRI Look Files (`ym_color::alf`): every container built by our own writer from our own LUTs
//! and CDLs, read back, damaged in every way the reader checks, and rebuilt byte for byte.
//!
//! No ARRI file or value is used anywhere here.

use ym_color::alf::{
    self, AlfError, AlfFile, AlfKind, LOGC4_COLORSPACE, MAX_FILE_BYTES, Param, REC709_COLORSPACE,
    crc32, param,
};
use ym_color::cdl::Cdl;
use ym_color::look::{LookInput, LookTransform};
use ym_color::lut::{Lut, Lut3d};

/// Half a code of 16 bits: the most a stored value can differ from the cube it was made from.
const HALF_CODE: f64 = 0.5 / 65535.0 + 1e-12;

/// A cube whose three channels depend on all three inputs in different ways, so a swapped axis
/// order shows.
fn cube(n: usize) -> Lut3d {
    Lut3d::bake(n, [0.0; 3], [1.0; 3], |c| {
        [
            (c[0] * 0.9 + 0.05 * c[2]).clamp(0.0, 1.0),
            c[1] * c[1] * 0.8 + 0.1 * c[0],
            (c[2] * 0.5 + c[0] * 0.25 + 0.2 * c[1]).clamp(0.0, 1.0),
        ]
    })
    .expect("bake")
}

fn strong_cdl() -> Cdl {
    Cdl::new([1.1, 1.2, 1.3], [0.01, 0.02, 0.03], [0.9, 1.1, 1.3], 1.5).expect("cdl")
}

fn max_diff(a: &Lut3d, b: &Lut3d) -> f64 {
    assert_eq!(a.size(), b.size());
    a.data()
        .iter()
        .zip(b.data())
        .flat_map(|(x, y)| (0..3).map(move |c| (x[c] - y[c]).abs()))
        .fold(0.0, f64::max)
}

fn close(a: &Cdl, b: &Cdl) {
    for c in 0..3 {
        assert!((a.slope[c] - b.slope[c]).abs() < 1e-6, "slope {a:?} {b:?}");
        assert!(
            (a.offset[c] - b.offset[c]).abs() < 1e-6,
            "offset {a:?} {b:?}"
        );
        assert!((a.power[c] - b.power[c]).abs() < 1e-6, "power {a:?} {b:?}");
    }
    assert!((a.saturation - b.saturation).abs() < 1e-6);
}

/// Recomputes the checksum of a file whose bytes a test changed on purpose, so the reader gets as
/// far as the check that test is about.
fn fix_crc(bytes: &mut [u8]) {
    let (at, from) = if &bytes[..4] == b"KOLA" {
        (8, 0x0C)
    } else {
        (0x0C, 0x10)
    };
    let crc = crc32(&bytes[from..]);
    bytes[at..at + 4].copy_from_slice(&crc.to_le_bytes());
}

fn put_u32(bytes: &mut [u8], at: usize, v: u32) {
    bytes[at..at + 4].copy_from_slice(&v.to_le_bytes());
}

fn alf4(n: usize, cdl: Option<&Cdl>) -> Vec<u8> {
    AlfFile::alf4(&cube(n), cdl, "a note", "look.cube")
        .to_bytes()
        .expect("write")
}

fn aml(n: usize, cdl: Option<&Cdl>) -> Vec<u8> {
    AlfFile::aml(&cube(n), cdl, "look.cube")
        .to_bytes()
        .expect("write")
}

fn alf4c(n: usize, cdl: Option<&Cdl>) -> Vec<u8> {
    AlfFile::alf4c(
        Some(&cube(n)),
        cdl,
        Some(&cube(3)),
        Some(&cube(5)),
        "custom",
    )
    .to_bytes()
    .expect("write")
}

#[test]
fn every_container_reads_back_at_its_own_size() {
    for n in [2, 17, 33, 65] {
        let source = cube(n);
        for (kind, bytes, input) in [
            (AlfKind::Alf4, alf4(n, None), LookInput::LogC4),
            (AlfKind::Alf4c, alf4c(n, None), LookInput::LogC4),
            (AlfKind::Aml, aml(n, None), LookInput::LogC3ToRec709),
        ] {
            let look = alf::parse(&bytes).unwrap_or_else(|e| panic!("{kind} {n}: {e}"));
            assert_eq!(look.kind, kind);
            assert_eq!(look.input, input, "{kind} {n}");
            assert_eq!(look.lut.size(), n, "{kind} {n}");
            assert!(look.cdl.is_none(), "{kind} {n}: identity CDL is dropped");
            let d = max_diff(&look.lut, &source);
            assert!(d <= HALF_CODE, "{kind} {n}: {d:e}");
        }
    }
}

#[test]
fn nodes_are_transposed_from_blue_fastest_to_red_fastest() {
    // A look that only moves red by the value of blue: a swapped axis would show at once.
    let n = 5;
    let lut = Lut3d::bake(n, [0.0; 3], [1.0; 3], |c| [c[2], c[0] * 0.5, c[1]]).expect("bake");
    for bytes in [
        AlfFile::alf4(&lut, None, "", "x.cube").to_bytes().unwrap(),
        AlfFile::aml(&lut, None, "x.cube").to_bytes().unwrap(),
    ] {
        let look = alf::parse(&bytes).unwrap();
        for (r, g, b) in [(1, 0, 0), (0, 1, 0), (0, 0, 1), (4, 2, 3), (3, 4, 1)] {
            let step = 1.0 / (n - 1) as f64;
            let want = [b as f64 * step, r as f64 * step * 0.5, g as f64 * step];
            let got = look.lut.entry(r, g, b).unwrap();
            for c in 0..3 {
                assert!(
                    (got[c] - want[c]).abs() <= HALF_CODE,
                    "{r}{g}{b} {got:?} {want:?}"
                );
            }
        }
    }
}

#[test]
fn a_cdl_comes_out_as_written_and_an_identity_cdl_is_dropped() {
    let cdl = strong_cdl();
    for (kind, bytes) in [
        (AlfKind::Alf4, alf4(5, Some(&cdl))),
        (AlfKind::Alf4c, alf4c(5, Some(&cdl))),
        (AlfKind::Aml, aml(5, Some(&cdl))),
    ] {
        let look = alf::parse(&bytes).unwrap();
        close(&look.cdl.unwrap_or_else(|| panic!("{kind}: no CDL")), &cdl);
    }
    let identity = Cdl::identity();
    for bytes in [alf4(5, Some(&identity)), aml(5, Some(&identity))] {
        assert!(alf::parse(&bytes).unwrap().cdl.is_none());
    }
}

#[test]
fn parameters_are_found_by_id_in_any_order_and_unknown_ids_are_ignored() {
    let cdl = strong_cdl();
    let AlfFile::Alf4(mut file) = AlfFile::alf4(&cube(3), Some(&cdl), "", "x.cube") else {
        unreachable!()
    };
    file.params.reverse();
    file.params.insert(
        3,
        Param {
            id: 9999,
            value: f32::NAN,
        },
    );
    file.params.push(Param {
        id: 4_000_000_000,
        value: -7.0,
    });
    let bytes = AlfFile::Alf4(file).to_bytes().unwrap();
    close(&alf::parse(&bytes).unwrap().cdl.unwrap(), &cdl);

    // An id that is missing is the identity value; the first of two entries for one id wins.
    let AlfFile::Alf4(mut file) = AlfFile::alf4(&cube(3), None, "", "x.cube") else {
        unreachable!()
    };
    file.params
        .retain(|p| p.id != param::SATURATION && p.id != param::SLOPE[1]);
    file.params.insert(
        0,
        Param {
            id: param::SLOPE[0],
            value: 2.0,
        },
    );
    let look = alf::parse(&AlfFile::Alf4(file).to_bytes().unwrap()).unwrap();
    let cdl = look.cdl.unwrap();
    assert_eq!(cdl.slope, [2.0, 1.0, 1.0]);
    assert_eq!(cdl.saturation, 1.0);

    // A file with no parameters at all is the plain LUT.
    let AlfFile::Alf4(mut file) = AlfFile::alf4(&cube(3), None, "", "x.cube") else {
        unreachable!()
    };
    file.params.clear();
    assert!(
        alf::parse(&AlfFile::Alf4(file).to_bytes().unwrap())
            .unwrap()
            .cdl
            .is_none()
    );
}

#[test]
fn the_look_renders_like_the_same_cube_with_the_same_cdl() {
    // An imported look and a `.cube` with a CDL in front are one look.
    let cdl = strong_cdl();
    let lut = cube(17);
    let look = alf::parse(&alf4(17, Some(&cdl))).unwrap();
    let from_file =
        LookTransform::new(look.input, look.cdl, Some(Lut::from_cube(look.lut))).unwrap();
    // The cube the file was made from, through the same CDL as the file holds (f32 precision).
    let file_cdl = from_file.cdl().copied().unwrap();
    let from_cube =
        LookTransform::new(LookInput::LogC4, Some(file_cdl), Some(Lut::from_cube(lut))).unwrap();
    let mut worst = 0.0_f64;
    for i in 0..400 {
        let t = f64::from(i) / 399.0;
        let rgb = [t, (t * 7.0).fract(), (1.0 - t) * 0.9 + 0.05];
        let a = from_file.apply_own_encoding(rgb);
        let b = from_cube.apply_own_encoding(rgb);
        for c in 0..3 {
            worst = worst.max((a[c] - b[c]).abs());
        }
    }
    assert!(worst < 4e-5, "{worst:e}");
}

#[test]
fn files_rebuild_byte_for_byte_from_what_was_read() {
    let cdl = strong_cdl();
    let mut files = vec![
        alf4(2, None),
        alf4(17, Some(&cdl)),
        alf4(33, None),
        alf4c(17, Some(&cdl)),
        aml(33, None),
        aml(17, Some(&cdl)),
    ];
    // An `.alf4c` without a CMT, one with a design-data archive, an ARRI-style type id and an
    // `.aml` name length that counts the NUL.
    files.push(
        AlfFile::alf4c(None, None, Some(&cube(3)), Some(&cube(3)), "")
            .to_bytes()
            .unwrap(),
    );
    let AlfFile::Alf4c(mut design) =
        AlfFile::alf4c(Some(&cube(3)), None, Some(&cube(3)), Some(&cube(3)), "n")
    else {
        unreachable!()
    };
    let sdr = design.sdr.as_mut().unwrap();
    sdr.extra = b"PK\x03\x04 pretend archive".to_vec();
    sdr.design_key = "a-key".to_owned();
    files.push(AlfFile::Alf4c(design).to_bytes().unwrap());
    let AlfFile::Alf4(mut library) = AlfFile::alf4(&cube(3), None, "text", "x.cube") else {
        unreachable!()
    };
    library.type_id = "com.arri.blackbox.alf4".to_owned();
    files.push(AlfFile::Alf4(library).to_bytes().unwrap());
    let AlfFile::Aml(mut library) = AlfFile::aml(&cube(3), None, "look.cube") else {
        unreachable!()
    };
    library.name_field_len += 1;
    files.push(AlfFile::Aml(library).to_bytes().unwrap());

    for (i, bytes) in files.iter().enumerate() {
        let parsed = alf::parse_file(bytes).unwrap_or_else(|e| panic!("file {i}: {e}"));
        assert_eq!(&parsed.to_bytes().unwrap(), bytes, "file {i}");
    }
}

#[test]
fn an_alf4c_keeps_its_cmt_and_sets_its_display_transforms_aside() {
    let look = alf::parse(&alf4c(9, None)).unwrap();
    assert_eq!(look.kind, AlfKind::Alf4c);
    assert_eq!(look.input, LookInput::LogC4);
    assert!(look.display_transforms_ignored);
    assert_eq!(
        look.lut.size(),
        9,
        "the CMT, not the 3- and 5-point transforms"
    );
    assert_eq!(look.note, "custom");
    assert!(
        !alf::parse(&alf4(3, None))
            .unwrap()
            .display_transforms_ignored
    );
    // A CMT-only file has no display transforms to set aside.
    let bytes = AlfFile::alf4c(Some(&cube(3)), None, None, None, "")
        .to_bytes()
        .unwrap();
    assert!(!alf::parse(&bytes).unwrap().display_transforms_ignored);
}

// ---- Errors -------------------------------------------------------------------------------------

#[test]
fn files_that_are_not_look_files_say_so() {
    for bytes in [
        &b""[..],
        &b"3FL"[..],
        &b"LUT_3D_SIZE 2\n0 0 0\n"[..],
        &b"<?xml version=\"1.0\"?><adicam/>"[..],
        &[0_u8; 64][..],
        &b"PK\x03\x04 zip"[..],
    ] {
        assert_eq!(alf::parse(bytes), Err(AlfError::NotALookFile), "{bytes:?}");
    }
    assert_eq!(AlfKind::sniff(b"3FLAxxxx"), Some(AlfKind::Alf4));
    assert_eq!(AlfKind::sniff(b"4FLAxxxx"), Some(AlfKind::Alf4c));
    assert_eq!(AlfKind::sniff(b"KOLAxxxx"), Some(AlfKind::Aml));
    assert_eq!(AlfKind::sniff(b"KOL"), None);
}

#[test]
fn another_version_is_refused_by_name() {
    for (mut bytes, kind) in [
        (alf4(3, None), AlfKind::Alf4),
        (alf4c(3, None), AlfKind::Alf4c),
        (aml(3, None), AlfKind::Aml),
    ] {
        let supported = if kind == AlfKind::Alf4c { 5 } else { 3 };
        put_u32(&mut bytes, 4, 9);
        fix_crc(&mut bytes);
        assert_eq!(
            alf::parse(&bytes),
            Err(AlfError::UnsupportedVersion {
                kind,
                found: 9,
                supported
            })
        );
    }
    // The text tells the person what to do.
    let text = AlfError::UnsupportedVersion {
        kind: AlfKind::Alf4,
        found: 9,
        supported: 3,
    }
    .to_string();
    assert!(
        text.contains("version 9") && text.contains(".alf4"),
        "{text}"
    );
}

#[test]
fn a_file_cut_short_or_changed_is_damaged() {
    for bytes in [alf4(3, None), alf4c(3, None)] {
        // The size field disagrees with the length.
        assert_eq!(
            alf::parse(&bytes[..bytes.len() - 1]),
            Err(AlfError::Truncated)
        );
        let mut longer = bytes.clone();
        longer.push(0);
        assert_eq!(alf::parse(&longer), Err(AlfError::Truncated));
        assert_eq!(alf::parse(&bytes[..12]), Err(AlfError::Truncated));
    }
    // An `.aml` has no size field: a cut file fails its checksum.
    let bytes = aml(3, None);
    assert_eq!(
        alf::parse(&bytes[..bytes.len() - 6]),
        Err(AlfError::BadChecksum)
    );
    // One changed byte anywhere after the checksum is found.
    for bytes in [alf4(3, None), alf4c(3, None), aml(3, None)] {
        for at in [0x20, 0x100, bytes.len() / 2, bytes.len() - 1] {
            let mut damaged = bytes.clone();
            damaged[at] ^= 0x10;
            assert_eq!(
                alf::parse(&damaged),
                Err(AlfError::BadChecksum),
                "byte {at:#x}"
            );
        }
    }
}

#[test]
fn tables_that_point_outside_the_file_are_refused() {
    // Each change is followed by a fresh checksum, so the reader gets to the table itself.
    let cases: [(&str, Vec<u8>, usize, u32); 7] = [
        ("alf4 parameter table", alf4(3, None), 0x14, 0x00FF_0000),
        ("alf4 parameter count", alf4(3, None), 0x18, 4_000_000),
        ("alf4 block", alf4(3, None), 0x68, 0x00FF_0000),
        ("alf4 note", alf4(3, None), 0x10, 2000),
        ("alf4c parameter table", alf4c(3, None), 0x14, 0x00FF_0000),
        ("alf4c descriptor", alf4c(3, None), 0x20, 0x00FF_0000),
        ("aml LUT offset", aml(3, None), 0x14, 0x00FF_0000),
    ];
    for (what, mut bytes, at, v) in cases {
        put_u32(&mut bytes, at, v);
        fix_crc(&mut bytes);
        assert_eq!(alf::parse(&bytes), Err(AlfError::Layout), "{what}");
    }
    // An offset that would overflow when added to a length.
    let mut bytes = alf4(3, None);
    put_u32(&mut bytes, 0x14, u32::MAX);
    fix_crc(&mut bytes);
    assert_eq!(alf::parse(&bytes), Err(AlfError::Layout));
    // A payload that runs past the end of the file.
    let mut bytes = alf4(3, None);
    let block = 0x500 + 8 * 11;
    let len = bytes.len() as u32;
    put_u32(&mut bytes, block + 0x8A, len - 10);
    fix_crc(&mut bytes);
    assert_eq!(alf::parse(&bytes), Err(AlfError::Layout));
}

#[test]
fn a_lut_size_outside_2_to_129_is_refused() {
    let block = 0x500 + 8 * 11;
    for n in [0_u16, 1, 130, 255, 65535] {
        let mut bytes = alf4(2, None);
        bytes[block + 0x42..block + 0x44].copy_from_slice(&n.to_le_bytes());
        fix_crc(&mut bytes);
        assert_eq!(
            alf::parse(&bytes),
            Err(AlfError::LutSize(u32::from(n))),
            ".alf4 {n}"
        );
    }
    for n in [0_u32, 1, 130, 1 << 20, u32::MAX] {
        // For an `.alf4c` the size sits in the CMT's descriptor.
        let mut bytes = alf4c(2, None);
        let desc = u32::from_le_bytes(bytes[0x20..0x24].try_into().unwrap()) as usize;
        put_u32(&mut bytes, desc + 0x44, n);
        fix_crc(&mut bytes);
        assert_eq!(alf::parse(&bytes), Err(AlfError::LutSize(n)), ".alf4c {n}");
    }
    for n in [1_u32, 130, 1 << 20, u32::MAX] {
        let mut bytes = aml(2, None);
        put_u32(&mut bytes, 0x18, n);
        fix_crc(&mut bytes);
        assert_eq!(alf::parse(&bytes), Err(AlfError::LutSize(n)), ".aml {n}");
    }
    // The smallest and the largest size are read.
    assert_eq!(alf::parse(&alf4(2, None)).unwrap().lut.size(), 2);
    let big = AlfFile::alf4(&Lut3d::identity(129).unwrap(), None, "", "big.cube");
    let look = alf::parse(&big.to_bytes().unwrap()).unwrap();
    assert_eq!(look.lut.size(), 129);
    assert!(
        (look.lut.entry(128, 64, 0).unwrap()[1] - 0.5).abs() <= HALF_CODE,
        "identity survives at 129 points"
    );
}

#[test]
fn a_scale_of_zero_or_above_65535_is_refused() {
    let block = 0x500 + 8 * 11;
    for scale in [0_u32, 65536] {
        let mut bytes = alf4(2, None);
        put_u32(&mut bytes, block + 0x44, scale);
        fix_crc(&mut bytes);
        assert_eq!(alf::parse(&bytes), Err(AlfError::Scale(scale)));
    }
}

fn alf4_with(params: Vec<Param>) -> AlfFile {
    let AlfFile::Alf4(mut f) = AlfFile::alf4(&cube(2), None, "", "x") else {
        unreachable!()
    };
    f.params = params;
    AlfFile::Alf4(f)
}

fn aml_with(params: Vec<Param>) -> AlfFile {
    let AlfFile::Aml(mut f) = AlfFile::aml(&cube(2), None, "x") else {
        unreachable!()
    };
    f.params = params;
    AlfFile::Aml(f)
}

#[test]
fn bad_colour_correction_is_refused() {
    for (what, id, value) in [
        ("NaN slope", param::SLOPE[0], f32::NAN),
        ("infinite offset", param::OFFSET[2], f32::INFINITY),
        ("NaN saturation", param::SATURATION, f32::NAN),
        ("-inf power", param::POWER[1], f32::NEG_INFINITY),
    ] {
        for make in [alf4_with, aml_with] {
            let file = make(vec![Param { id, value }]);
            let parsed = alf::parse_file(&file.to_bytes().unwrap()).unwrap();
            assert_eq!(parsed.look(), Err(AlfError::CdlNotFinite), "{what}");
        }
    }
    for (what, id, value) in [
        ("negative slope", param::SLOPE[1], -0.5),
        ("zero power", param::POWER[0], 0.0),
        ("negative power", param::POWER[2], -1.0),
        ("negative saturation", param::SATURATION, -0.1),
    ] {
        let AlfFile::Alf4(mut f) = AlfFile::alf4(&cube(2), None, "", "x") else {
            unreachable!()
        };
        f.params = vec![Param { id, value }];
        let parsed = alf::parse_file(&AlfFile::Alf4(f).to_bytes().unwrap()).unwrap();
        assert_eq!(parsed.look(), Err(AlfError::CdlRange), "{what}");
    }
    // A bad value in an id that is not a CDL value does not matter.
    let AlfFile::Alf4(mut f) = AlfFile::alf4(&cube(2), None, "", "x") else {
        unreachable!()
    };
    f.params.push(Param {
        id: param::INTENSITY,
        value: f32::NAN,
    });
    assert!(alf::parse(&AlfFile::Alf4(f).to_bytes().unwrap()).is_ok());
}

#[test]
fn an_alf4c_needs_a_creative_look_for_logc4() {
    let none = AlfFile::alf4c(None, None, Some(&cube(3)), Some(&cube(3)), "")
        .to_bytes()
        .unwrap();
    assert_eq!(alf::parse(&none), Err(AlfError::NoCmt));
    let text = AlfError::NoCmt.to_string();
    assert!(
        text.contains("no creative look") && text.contains("nothing to import"),
        "{text}"
    );

    let AlfFile::Alf4c(mut f) = AlfFile::alf4c(Some(&cube(3)), None, None, None, "") else {
        unreachable!()
    };
    f.cmt.as_mut().unwrap().colorspace = "com.apple.proapps.colorspace.rec1886".to_owned();
    let bytes = AlfFile::Alf4c(f.clone()).to_bytes().unwrap();
    assert_eq!(
        alf::parse(&bytes),
        Err(AlfError::NotLogC4(
            "com.apple.proapps.colorspace.rec1886".to_owned()
        ))
    );
    f.cmt.as_mut().unwrap().colorspace = String::new();
    assert!(matches!(
        alf::parse(&AlfFile::Alf4c(f).to_bytes().unwrap()),
        Err(AlfError::NotLogC4(_))
    ));

    // The same rule for an `.alf4`, and the id is compared without regard to case.
    let AlfFile::Alf4(mut f) = AlfFile::alf4(&cube(3), None, "", "x") else {
        unreachable!()
    };
    f.lut.colorspace = LOGC4_COLORSPACE.to_uppercase();
    assert!(alf::parse(&AlfFile::Alf4(f.clone()).to_bytes().unwrap()).is_ok());
    f.lut.colorspace = "com.arri.colorspace.awg3.d65.logcv3".to_owned();
    assert!(matches!(
        alf::parse(&AlfFile::Alf4(f).to_bytes().unwrap()),
        Err(AlfError::NotLogC4(_))
    ));
}

#[test]
fn an_aml_must_be_made_for_rec709_and_hold_a_lut() {
    for (id, label) in [
        (
            "com.apple.proapps.colorspace.rec2100.d65.pq",
            "a Rec.2020 PQ display",
        ),
        (
            "com.apple.proapps.colorspace.rec2100.d65.hlg",
            "a Rec.2020 HLG display",
        ),
        (
            "com.apple.proapps.colorspace.rec2020.d65.gamma24",
            "a Rec.2020 display",
        ),
        (
            "com.apple.proapps.colorspace.p3.d65.gamma26",
            "a P3 display",
        ),
    ] {
        let AlfFile::Aml(mut f) = AlfFile::aml(&cube(3), None, "x") else {
            unreachable!()
        };
        f.colorspace = id.to_owned();
        let err = alf::parse(&AlfFile::Aml(f).to_bytes().unwrap()).unwrap_err();
        assert_eq!(err, AlfError::NotRec709(label.to_owned()), "{id}");
        assert_eq!(
            err.to_string(),
            format!("It is made for {label}. Yellowmoon can use ARRI Look Files made for Rec.709.")
        );
    }
    let AlfFile::Aml(mut f) = AlfFile::aml(&cube(3), None, "x") else {
        unreachable!()
    };
    assert_eq!(f.colorspace, REC709_COLORSPACE);
    // No LUT: a size of zero, or nothing after the parameter table.
    f.size = 0;
    f.payload.clear();
    assert_eq!(
        alf::parse(&AlfFile::Aml(f).to_bytes().unwrap()),
        Err(AlfError::NoLut)
    );
    let mut bytes = aml(3, None);
    bytes.truncate(0x580 + 8 * 29);
    fix_crc(&mut bytes);
    assert_eq!(alf::parse(&bytes), Err(AlfError::NoLut));
}

#[test]
fn messages_are_sentences_for_the_person_importing() {
    let all = [
        AlfError::NotALookFile,
        AlfError::Truncated,
        AlfError::BadChecksum,
        AlfError::Layout,
        AlfError::LutSize(300),
        AlfError::Scale(0),
        AlfError::CdlNotFinite,
        AlfError::CdlRange,
        AlfError::NoCmt,
        AlfError::NotLogC4("x".into()),
        AlfError::NotRec709("a P3 display".into()),
        AlfError::NoLut,
        AlfError::TooLarge { bytes: 1, limit: 2 },
        AlfError::UnsupportedVersion {
            kind: AlfKind::Aml,
            found: 1,
            supported: 3,
        },
    ];
    for e in all {
        let text = e.to_string();
        assert!(text.starts_with(|c: char| c.is_uppercase()), "{text}");
        assert!(text.ends_with('.'), "{text}");
        assert!(!text.contains('!'), "{text}");
    }
}

#[test]
fn a_file_over_the_limit_is_refused_before_it_is_read() {
    let big = vec![b'3'; MAX_FILE_BYTES + 1];
    // Not even the marker is looked at twice: the size comes first for a known marker.
    let mut known = big.clone();
    known[..4].copy_from_slice(b"3FLA");
    assert!(matches!(alf::parse(&known), Err(AlfError::TooLarge { .. })));
    assert_eq!(alf::parse(&big), Err(AlfError::NotALookFile));
}

// ---- Robustness ---------------------------------------------------------------------------------

/// A small deterministic generator for the damage below.
struct XorShift(u64);

impl XorShift {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

#[test]
fn damaged_files_never_panic() {
    let cdl = strong_cdl();
    let sources = [alf4(2, Some(&cdl)), alf4c(2, None), aml(2, Some(&cdl))];
    let mut rng = XorShift(0x9E37_79B9_7F4A_7C15);
    for bytes in &sources {
        // Every length.
        for len in 0..bytes.len() {
            let _ = alf::parse(&bytes[..len]);
        }
        // Random changes, with and without a repaired checksum (so the structure is reached).
        for round in 0..3000 {
            let mut damaged = bytes.clone();
            for _ in 0..=(rng.next() % 4) {
                let at = (rng.next() as usize) % damaged.len().min(0x700);
                match rng.next() % 4 {
                    0 => damaged[at] = rng.next() as u8,
                    1 => damaged[at] ^= 1 << (rng.next() % 8),
                    2 => {
                        let end = (at + 4).min(damaged.len());
                        for b in &mut damaged[at..end] {
                            *b = 0xFF;
                        }
                    }
                    _ => {
                        let end = (at + 4).min(damaged.len());
                        for b in &mut damaged[at..end] {
                            *b = 0;
                        }
                    }
                }
            }
            if round % 2 == 0 && damaged.len() >= 16 && AlfKind::sniff(&damaged).is_some() {
                fix_crc(&mut damaged);
            }
            if let Ok(look) = alf::parse(&damaged) {
                // Whatever still reads is a usable look.
                assert!((2..=129).contains(&look.lut.size()));
            }
            // And whatever still reads as a container is written again as a file that reads back
            // and is written as the same bytes (the third property of the fuzz target; compared as
            // bytes because a NaN parameter is not equal to itself).
            // (A string that was not UTF-8 reads with replacement characters and is not what was in
            // the file, so a rebuilt file may differ from the one read; those are left out.)
            if let Ok(file) = alf::parse_file(&damaged)
                && !format!("{file:?}").contains('\u{FFFD}')
                && let Ok(rebuilt) = file.to_bytes()
            {
                let again = alf::parse_file(&rebuilt).expect("a written file reads back");
                assert_eq!(again.to_bytes().unwrap(), rebuilt);
            }
        }
    }
}

/// Writes the seed corpus of the `alf_parse` fuzz target (`crates/ym-color/fuzz`): one small valid
/// file of each container, and an `.alf4c` with design data and one without a CMT.
/// `YM_WRITE_ALF_SEEDS=fuzz/corpus/alf_parse cargo test -p ym-color --test alf seeds -- --ignored`
/// (the path is relative to `crates/ym-color`, where cargo runs the test).
#[test]
#[ignore = "writes files: set YM_WRITE_ALF_SEEDS to the corpus folder"]
fn write_the_fuzz_seeds() {
    let Some(dir) = std::env::var_os("YM_WRITE_ALF_SEEDS").map(std::path::PathBuf::from) else {
        eprintln!("YM_WRITE_ALF_SEEDS is not set: nothing written");
        return;
    };
    std::fs::create_dir_all(&dir).expect("folder");
    let AlfFile::Alf4c(mut design) = AlfFile::alf4c(
        Some(&cube(2)),
        None,
        Some(&cube(2)),
        Some(&cube(2)),
        "design",
    ) else {
        unreachable!()
    };
    let sdr = design.sdr.as_mut().unwrap();
    sdr.extra = b"PK\x03\x04 design".to_vec();
    sdr.design_key = "key".to_owned();
    let files: [(&str, Vec<u8>); 6] = [
        ("alf4_2.alf4", alf4(2, Some(&strong_cdl()))),
        ("alf4c_2.alf4c", alf4c(2, None)),
        ("aml_2.aml", aml(2, Some(&strong_cdl()))),
        (
            "alf4c_design.alf4c",
            AlfFile::Alf4c(design).to_bytes().unwrap(),
        ),
        (
            "alf4c_no_cmt.alf4c",
            AlfFile::alf4c(None, None, Some(&cube(2)), Some(&cube(2)), "")
                .to_bytes()
                .unwrap(),
        ),
        ("aml_no_lut.aml", {
            let AlfFile::Aml(mut bare) = AlfFile::aml(&cube(2), None, "x") else {
                unreachable!()
            };
            bare.size = 0;
            bare.payload.clear();
            AlfFile::Aml(bare).to_bytes().unwrap()
        }),
    ];
    for (name, bytes) in files {
        std::fs::write(dir.join(name), bytes).expect("write");
    }
}
