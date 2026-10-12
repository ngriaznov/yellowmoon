//! M1-10: the .cube parser: valid dialects, every malformed-input error, bounded allocation and
//! "never panics" properties (the cargo-fuzz target in `crates/ym-color/fuzz` runs the same entry point).

use proptest::prelude::*;
use ym_color::cube::{
    CubeError, MAX_1D_SIZE, MAX_3D_SIZE, MAX_FILE_BYTES, parse, parse_bytes, parse_lut,
};
use ym_color::lut::Lut3d;

fn ident2() -> String {
    "LUT_3D_SIZE 2\n0 0 0\n1 0 0\n0 1 0\n1 1 0\n0 0 1\n1 0 1\n0 1 1\n1 1 1\n".to_string()
}

#[test]
fn minimal_3d_cube() {
    let f = parse(&ident2()).unwrap();
    let lut = f.lut3d.unwrap();
    assert_eq!(lut.size(), 2);
    assert_eq!(lut.domain_min(), [0.0; 3]);
    assert_eq!(lut.domain_max(), [1.0; 3]);
    assert!(f.lut1d.is_none() && f.title.is_none());
    // Red varies fastest.
    assert_eq!(lut.entry(1, 0, 0), Some([1.0, 0.0, 0.0]));
    assert_eq!(lut.entry(0, 1, 0), Some([0.0, 1.0, 0.0]));
    assert_eq!(lut.entry(0, 0, 1), Some([0.0, 0.0, 1.0]));
}

#[test]
fn title_variants() {
    for (line, want) in [
        ("TITLE \"Warm look\"", "Warm look"),
        ("TITLE Warm look", "Warm look"),
        ("title \"a # not a comment\"", "a # not a comment"),
        ("TITLE \"\"", ""),
        ("TITLE   \"padded\"   ", "padded"),
    ] {
        let f = parse(&format!("{line}\n{}", ident2())).unwrap();
        assert_eq!(f.title.as_deref(), Some(want), "{line}");
    }
    // Overlong titles are truncated, not rejected.
    let long = "x".repeat(5000);
    let f = parse(&format!("TITLE \"{long}\"\n{}", ident2())).unwrap();
    assert_eq!(f.title.unwrap().chars().count(), 1024);
}

#[test]
fn comments_blank_lines_whitespace_and_line_ends() {
    let body = "# made by hand\n\n  LUT_3D_SIZE\t2   # size\n\t0 0 0\n1 0 0   # red\n0 1 0\n1 1 0\n\n0 0 1\n1 0 1\n0 1 1\n1 1 1  \n# end";
    for text in [
        body.to_string(),
        body.replace('\n', "\r\n"),
        body.replace('\n', "\r"),
        format!("\u{feff}{body}"),
    ] {
        assert_eq!(parse(&text).unwrap().lut3d.unwrap().size(), 2);
    }
    // Lowercase keywords.
    assert!(parse(&ident2().to_lowercase()).is_ok());
}

#[test]
fn number_formats() {
    let text =
        "LUT_3D_SIZE 2\n+0.0 -0 .5\n1e0 1E-1 0.25e+1\n0 0 0\n0 0 0\n0 0 0\n0 0 0\n0 0 0\n0 0 0\n";
    let lut = parse(text).unwrap().lut3d.unwrap();
    assert_eq!(lut.data()[0], [0.0, -0.0, 0.5]);
    assert_eq!(lut.data()[1], [1.0, 0.1, 2.5]);
}

#[test]
fn domain_forms() {
    // Adobe: DOMAIN_MIN / DOMAIN_MAX per channel.
    let adobe = format!(
        "DOMAIN_MIN -0.1 -0.2 0\nDOMAIN_MAX 1.5 1.25 2\n{}",
        ident2()
    );
    let lut = parse(&adobe).unwrap().lut3d.unwrap();
    assert_eq!(lut.domain_min(), [-0.1, -0.2, 0.0]);
    assert_eq!(lut.domain_max(), [1.5, 1.25, 2.0]);
    // Only a maximum given: the minimum defaults to 0.
    let only_max = format!("DOMAIN_MAX 2 2 2\n{}", ident2());
    let lut = parse(&only_max).unwrap().lut3d.unwrap();
    assert_eq!((lut.domain_min(), lut.domain_max()), ([0.0; 3], [2.0; 3]));
    // Resolve: LUT_3D_INPUT_RANGE.
    let resolve = format!("LUT_3D_INPUT_RANGE -0.25 4.0\n{}", ident2());
    let lut = parse(&resolve).unwrap().lut3d.unwrap();
    assert_eq!((lut.domain_min(), lut.domain_max()), ([-0.25; 3], [4.0; 3]));
    // Both spellings agreeing is fine, disagreeing is an error.
    let agree = format!(
        "DOMAIN_MIN 0 0 0\nDOMAIN_MAX 3 3 3\nLUT_3D_INPUT_RANGE 0 3\n{}",
        ident2()
    );
    assert!(parse(&agree).is_ok());
    let clash = format!(
        "DOMAIN_MIN 0 0 0\nDOMAIN_MAX 3 3 3\nLUT_3D_INPUT_RANGE 0 4\n{}",
        ident2()
    );
    assert_eq!(parse(&clash), Err(CubeError::BadDomain));
    // Empty or reversed domains.
    for bad in [
        "DOMAIN_MIN 1 0 0\nDOMAIN_MAX 1 1 1\n",
        "DOMAIN_MIN 0 0 0\nDOMAIN_MAX 0 1 1\n",
        "LUT_3D_INPUT_RANGE 2 1\n",
        "LUT_3D_INPUT_RANGE 1 1\n",
    ] {
        assert_eq!(
            parse(&format!("{bad}{}", ident2())),
            Err(CubeError::BadDomain),
            "{bad}"
        );
    }
    // LUT_1D_INPUT_RANGE has no meaning without a 1D table.
    assert_eq!(
        parse(&format!("LUT_1D_INPUT_RANGE 0 1\n{}", ident2())),
        Err(CubeError::BadDomain)
    );
}

#[test]
fn one_d_only_and_shaper_plus_cube() {
    let one_d = "TITLE \"curve\"\nLUT_1D_SIZE 3\nDOMAIN_MIN 0 0 0\nDOMAIN_MAX 2 2 2\n0 0 0\n0.5 0.25 0.125\n1 1 1\n";
    let f = parse(one_d).unwrap();
    let l1 = f.lut1d.as_ref().unwrap();
    assert!(f.lut3d.is_none());
    assert_eq!((l1.size(), l1.domain_max()), (3, [2.0; 3]));
    assert_eq!(l1.sample([1.0, 1.0, 1.0]), [0.5, 0.25, 0.125]);
    assert_eq!(l1.sample([-3.0, 9.0, f64::NAN]), [0.0, 1.0, 0.0]);
    // A 1D-only file is not a usable look LUT.
    assert_eq!(parse_lut(one_d), Err(CubeError::NoThreeDimensionalLut));

    let both = format!(
        "LUT_1D_SIZE 2\nLUT_3D_SIZE 2\nLUT_1D_INPUT_RANGE -1 1\nLUT_3D_INPUT_RANGE 0 0.5\n0 0 0\n1 1 1\n{}",
        ident2().lines().skip(1).collect::<Vec<_>>().join("\n")
    );
    let lut = parse_lut(&both).unwrap();
    let shaper = lut.shaper.as_ref().unwrap();
    assert_eq!(
        (shaper.size(), shaper.domain_min(), shaper.domain_max()),
        (2, [-1.0; 3], [1.0; 3])
    );
    assert_eq!((lut.cube.size(), lut.cube.domain_max()), (2, [0.5; 3]));
    // Shaper maps x in [-1, 1] to [0, 1]; the cube then sees it over [0, 0.5] (clamped above 0.5).
    assert_eq!(
        lut.sample([0.0, 0.0, 0.0]),
        lut.cube.sample([0.5, 0.5, 0.5])
    );
    // DOMAIN_* with both tables applies to the shaper.
    let dom = "LUT_1D_SIZE 2\nLUT_3D_SIZE 2\nDOMAIN_MIN -2 -2 -2\nDOMAIN_MAX 2 2 2\n0 0 0\n1 1 1\n"
        .to_string()
        + &ident2().lines().skip(1).collect::<Vec<_>>().join("\n");
    let lut = parse_lut(&dom).unwrap();
    assert_eq!(lut.shaper.unwrap().domain_min(), [-2.0; 3]);
    assert_eq!(lut.cube.domain_max(), [1.0; 3]);
}

#[test]
fn size_limits() {
    assert_eq!(MAX_3D_SIZE, 129);
    assert_eq!(MAX_1D_SIZE, 65_536);
    for bad in [
        "0",
        "1",
        "130",
        "1000",
        "99999999999999999999999999",
        "-2",
        "2.5",
        "abc",
        "0x10",
        "",
    ] {
        let r = parse(&format!("LUT_3D_SIZE {bad}\n"));
        assert!(
            matches!(
                r,
                Err(CubeError::SizeOutOfRange { .. } | CubeError::BadArguments { .. })
            ),
            "LUT_3D_SIZE {bad:?}: {r:?}"
        );
    }
    assert!(matches!(
        parse("LUT_3D_SIZE 130\n"),
        Err(CubeError::SizeOutOfRange {
            size: 130,
            min: 2,
            max: 129,
            ..
        })
    ));
    assert!(matches!(
        parse("LUT_1D_SIZE 65537\n"),
        Err(CubeError::SizeOutOfRange { size: 65537, .. })
    ));
    assert!(matches!(
        parse("LUT_1D_SIZE 1\n"),
        Err(CubeError::SizeOutOfRange { size: 1, .. })
    ));
    assert!(matches!(
        parse("LUT_3D_SIZE 2 3\n"),
        Err(CubeError::BadArguments { .. })
    ));
    // The limits themselves parse (header only: rows are missing, which is the next error).
    assert!(matches!(
        parse("LUT_3D_SIZE 129\n"),
        Err(CubeError::TooFewEntries {
            expected: 2_146_689,
            found: 0
        })
    ));
    assert!(matches!(
        parse("LUT_3D_SIZE 2\n"),
        Err(CubeError::TooFewEntries {
            expected: 8,
            found: 0
        })
    ));
    assert!(matches!(
        parse("LUT_1D_SIZE 65536\n"),
        Err(CubeError::TooFewEntries {
            expected: 65_536,
            ..
        })
    ));
}

#[test]
fn full_size_129_cube_parses() {
    let n = 129;
    let mut text = format!("LUT_3D_SIZE {n}\n");
    text.reserve(n * n * n * 6);
    for _ in 0..n * n * n {
        text.push_str("0 0 0\n");
    }
    let lut = parse(&text).unwrap().lut3d.unwrap();
    assert_eq!((lut.size(), lut.data().len()), (129, 2_146_689));
}

#[test]
fn malformed_inputs_get_typed_errors() {
    let cases: Vec<(String, &str)> = vec![
        (String::new(), "MissingSize"),
        ("# nothing here\n".into(), "MissingSize"),
        ("TITLE \"x\"\n".into(), "MissingSize"),
        ("0 0 0\n".into(), "DataBeforeSize"),
        ("LUT_3D_SIZE 2\n0 0 0\n1 0 0\n".into(), "TooFewEntries"),
        (format!("{}0 0 0\n", ident2()), "TooManyEntries"),
        ("LUT_3D_SIZE 2\n0 0\n".into(), "BadDataRow"),
        ("LUT_3D_SIZE 2\n0 0 0 0\n".into(), "BadDataRow"),
        ("LUT_3D_SIZE 2\n0 0 x\n".into(), "BadDataRow"),
        ("LUT_3D_SIZE 2\n0,5 0,5 0,5\n".into(), "BadDataRow"),
        ("LUT_3D_SIZE 2\nnan 0 0\n".into(), "NonFinite"),
        ("LUT_3D_SIZE 2\n0 inf 0\n".into(), "NonFinite"),
        ("LUT_3D_SIZE 2\n0 0 -Infinity\n".into(), "NonFinite"),
        ("LUT_3D_SIZE 2\n1e999 0 0\n".into(), "NonFinite"),
        ("LUT_3D_SIZE 2\nLUT_3D_SIZE 2\n".into(), "Duplicate"),
        ("TITLE a\nTITLE b\nLUT_3D_SIZE 2\n".into(), "Duplicate"),
        (
            "DOMAIN_MIN 0 0 0\nDOMAIN_MIN 0 0 0\nLUT_3D_SIZE 2\n".into(),
            "Duplicate",
        ),
        ("LUT_3D_SIZE 2\nFOO 1\n".into(), "UnknownKeyword"),
        (
            "LUT_3D_SIZE 2\nLUT_IN_VIDEO_RANGE\n".into(),
            "UnknownKeyword",
        ),
        ("DOMAIN_MIN 0 0\nLUT_3D_SIZE 2\n".into(), "BadArguments"),
        ("DOMAIN_MAX 1 1 nan\nLUT_3D_SIZE 2\n".into(), "BadArguments"),
        (
            "LUT_3D_INPUT_RANGE 0\nLUT_3D_SIZE 2\n".into(),
            "BadArguments",
        ),
        ("LUT_3D_SIZE\n".into(), "BadArguments"),
        (
            "LUT_3D_SIZE 2\n0 0 0\nDOMAIN_MIN 0 0 0\n".into(),
            "KeywordAfterData",
        ),
    ];
    for (text, want) in cases {
        let got = parse(&text).expect_err(&format!("{text:?} should fail"));
        let name = format!("{got:?}");
        assert!(
            name.starts_with(want),
            "{text:?}: expected {want}, got {name}"
        );
        // Messages are readable and never empty.
        assert!(!got.to_string().is_empty());
    }
    assert!(
        parse("LUT_3D_SIZE 2\n0 0 0\nnan 0 0\n")
            .unwrap_err()
            .to_string()
            .contains("line 3")
    );
}

#[test]
fn bytes_entry_point() {
    assert_eq!(parse_bytes(&[0xff, 0xfe, 0x00]), Err(CubeError::NotUtf8));
    assert!(parse_bytes(ident2().as_bytes()).is_ok());
    let big = vec![b' '; MAX_FILE_BYTES + 1];
    assert!(matches!(parse_bytes(&big), Err(CubeError::TooLarge { .. })));
    // Exactly at the limit is accepted as input (and rejected as a LUT for having no size).
    let edge = vec![b' '; MAX_FILE_BYTES];
    assert_eq!(parse_bytes(&edge), Err(CubeError::MissingSize));
}

#[test]
fn allocation_is_bounded_by_the_input() {
    // A header that promises 129^3 rows with almost no data returns promptly and holds little memory.
    let t = std::time::Instant::now();
    for _ in 0..200 {
        assert!(matches!(
            parse("LUT_3D_SIZE 129\n0 0 0\n"),
            Err(CubeError::TooFewEntries { found: 1, .. })
        ));
    }
    assert!(t.elapsed().as_secs() < 5);
    // Rows beyond the declared count stop the parse at the first extra row.
    let mut text = ident2();
    for _ in 0..10_000 {
        text.push_str("0 0 0\n");
    }
    assert!(matches!(
        parse(&text),
        Err(CubeError::TooManyEntries {
            line: 10,
            expected: 8
        })
    ));
}

#[test]
fn parsed_cubes_equal_written_ones() {
    let lut = Lut3d::bake(4, [-0.5, 0.0, 0.25], [2.0, 1.0, 3.25], |c| {
        [c[0] * 1.0e-3, -c[1], c[2].sin()]
    })
    .unwrap();
    let back = parse(&lut.to_cube_string("t")).unwrap().lut3d.unwrap();
    assert_eq!(back, lut);
}

fn valid_templates() -> Vec<String> {
    vec![
        ident2(),
        Lut3d::bake(3, [0.0; 3], [1.0; 3], |c| [c[0] * c[1], c[2], 1.0 - c[0]]).unwrap().to_cube_string("t3"),
        "TITLE \"c\"\nLUT_1D_SIZE 4\nDOMAIN_MIN 0 0 0\nDOMAIN_MAX 1 1 1\n0 0 0\n0.2 0.2 0.2\n0.7 0.7 0.7\n1 1 1\n".into(),
        "LUT_1D_SIZE 2\nLUT_3D_SIZE 2\nLUT_1D_INPUT_RANGE 0 1\nLUT_3D_INPUT_RANGE 0 1\n0 0 0\n1 1 1\n0 0 0\n1 0 0\n0 1 0\n1 1 0\n0 0 1\n1 0 1\n0 1 1\n1 1 1\n".into(),
    ]
}

/// Any outcome is fine except a panic; an `Ok` result must be self-consistent.
fn check_consistent(text: &str) {
    if let Ok(f) = parse(text) {
        assert!(f.lut1d.is_some() || f.lut3d.is_some());
        if let Some(c) = &f.lut3d {
            assert!((2..=MAX_3D_SIZE).contains(&c.size()));
            assert_eq!(c.data().len(), c.size().pow(3));
            assert!(c.data().iter().all(|v| v.iter().all(|x| x.is_finite())));
            assert!((0..3).all(|k| c.domain_min()[k] < c.domain_max()[k]));
        }
        if let Some(l) = &f.lut1d {
            assert_eq!(l.data().len(), l.size());
            assert!(l.data().iter().all(|v| v.iter().all(|x| x.is_finite())));
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(512))]

    #[test]
    fn arbitrary_text_never_panics(s in ".{0,400}") {
        check_consistent(&s);
    }

    #[test]
    fn arbitrary_bytes_never_panic(b in proptest::collection::vec(any::<u8>(), 0..400)) {
        let _ = parse_bytes(&b);
    }

    #[test]
    fn mutated_valid_files_never_panic(
        which in 0usize..4,
        edits in proptest::collection::vec((0usize..2000, any::<u8>(), 0u8..3), 1..8),
    ) {
        let mut bytes = valid_templates()[which].clone().into_bytes();
        for (pos, byte, kind) in edits {
            if bytes.is_empty() { break; }
            let at = pos % bytes.len();
            match kind {
                0 => bytes[at] = byte,
                1 => { bytes.remove(at); }
                _ => bytes.insert(at, byte),
            }
        }
        let text = String::from_utf8_lossy(&bytes).into_owned();
        check_consistent(&text);
    }

    #[test]
    fn token_soup_never_panics(
        words in proptest::collection::vec(
            prop_oneof![
                Just("LUT_3D_SIZE".to_string()), Just("LUT_1D_SIZE".to_string()), Just("DOMAIN_MIN".to_string()),
                Just("DOMAIN_MAX".to_string()), Just("TITLE".to_string()), Just("LUT_3D_INPUT_RANGE".to_string()),
                Just("LUT_1D_INPUT_RANGE".to_string()), Just("2".to_string()), Just("3".to_string()), Just("0".to_string()),
                Just("1".to_string()), Just("-1".to_string()), Just("nan".to_string()), Just("#".to_string()),
                Just("\n".to_string()), Just("\r\n".to_string()), Just(" ".to_string()), Just("0.5".to_string()),
            ],
            0..120,
        ),
    ) {
        check_consistent(&words.join(" "));
        check_consistent(&words.concat());
    }
}

/// A longer property run for machines without nightly (so without cargo-fuzz):
/// `cargo test -p ym-color --test cube --release -- --ignored`.
mod long_run {
    use super::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(200_000))]

        #[test]
        #[ignore = "long property run (about a minute in release); the cargo-fuzz target is the main fuzzer"]
        fn mutated_valid_files_never_panic_long(
            which in 0usize..4,
            edits in proptest::collection::vec((0usize..4000, any::<u8>(), 0u8..4), 1..24),
        ) {
            let mut bytes = valid_templates()[which].clone().into_bytes();
            for (pos, byte, kind) in edits {
                if bytes.is_empty() { break; }
                let at = pos % bytes.len();
                match kind {
                    0 => bytes[at] = byte,
                    1 => { bytes.remove(at); }
                    2 => bytes.insert(at, byte),
                    _ => { let copy = bytes[at..bytes.len().min(at + 12)].to_vec(); bytes.splice(at..at, copy); }
                }
            }
            let text = String::from_utf8_lossy(&bytes).into_owned();
            check_consistent(&text);
        }
    }
}
