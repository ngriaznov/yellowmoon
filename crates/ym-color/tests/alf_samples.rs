//! `ym_color::alf` against look files that ARRI's look-builder (ART CMD) wrote from our own
//! synthetic cubes. **Ignored by default**: the files are produced with ARRI's tool on a
//! developer's machine and are not in the repository.
//!
//! ```text
//! YM_ALF_SAMPLES=/path/to/folder cargo test -p ym-color --test alf_samples -- --ignored --nocapture
//! ```
//!
//! The folder holds `out/` (the nine files) and `samples/` (the cubes they were made from, and
//! `design.zip`). Every LUT must equal its cube within one rounding step of 16 bits (0.5 / 65535),
//! except the `.aml` made from a 17-point cube, which ART resamples to 33 points (checked against
//! our tetrahedral sampling within two codes), and every file must be rebuilt byte for byte by
//! [`AlfFile::to_bytes`] from what [`parse_file`] read.

use std::fs;
use std::path::{Path, PathBuf};

use ym_color::alf::{self, AlfError, AlfFile, AlfKind, LutBlock, REC709_COLORSPACE, parse_file};
use ym_color::cube;
use ym_color::look::LookInput;
use ym_color::lut::Lut3d;

/// One rounding step of 16 bits, plus rounding slack.
const ONE_CODE_HALF: f64 = 0.5 / 65535.0 + 1e-9;

fn folder() -> Option<PathBuf> {
    let dir = std::env::var_os("YM_ALF_SAMPLES").map(PathBuf::from);
    if dir.is_none() {
        eprintln!("YM_ALF_SAMPLES is not set: nothing to check");
    }
    dir
}

fn cube_of(dir: &Path, name: &str) -> Lut3d {
    let text = fs::read(dir.join("samples").join(name)).expect("sample cube");
    cube::parse_bytes(&text)
        .expect("cube parses")
        .lut3d
        .expect("3D cube")
}

fn max_diff(a: &Lut3d, b: &Lut3d) -> f64 {
    assert_eq!(a.size(), b.size());
    a.data()
        .iter()
        .zip(b.data())
        .flat_map(|(x, y)| (0..3).map(move |c| (x[c] - y[c]).abs()))
        .fold(0.0, f64::max)
}

fn check_block(label: &str, block: &LutBlock, dir: &Path, cube_name: &str, colorspace: &str) {
    let ours = block.to_lut3d().expect("decodes");
    let cube = cube_of(dir, cube_name);
    let d = max_diff(&ours, &cube);
    eprintln!(
        "{label}: n={} vs {cube_name}: max abs diff {d:.3e} (bound {:.3e}), colour space {}",
        ours.size(),
        0.5 / 65535.0,
        block.colorspace
    );
    assert!(d <= ONE_CODE_HALF, "{label}: {d:e}");
    assert_eq!(block.colorspace, colorspace, "{label}");
    assert_eq!(block.source_name, cube_name, "{label}");
}

#[test]
#[ignore = "needs look files written by ART CMD: set YM_ALF_SAMPLES (see the file header)"]
fn the_nine_files_from_the_look_builder_match_our_cubes_and_rebuild_exactly() {
    let Some(dir) = folder() else {
        return;
    };
    let out = dir.join("out");
    let read = |name: &str| fs::read(out.join(name)).unwrap_or_else(|e| panic!("{name}: {e}"));
    const LOGC4: &str = "com.arri.colorspace.awg4.d65.logcv4";
    const REC709: &str = "com.apple.proapps.colorspace.rec1886";
    let mut parsed = 0;

    // .alf4 at three sizes.
    for (name, cube_name, note) in [
        ("alf4_cmt33.alf4", "cmt33.cube", ""),
        ("alf4_cmt65.alf4", "cmt65.cube", ""),
        ("alf4_cmt17_note.alf4", "cmt17.cube", "A note for the look"),
    ] {
        let bytes = read(name);
        let file = parse_file(&bytes).expect(name);
        let AlfFile::Alf4(alf4) = &file else {
            panic!("{name} is not an .alf4")
        };
        assert_eq!(alf4.note, note, "{name}");
        assert!(alf4.type_id.is_empty(), "{name}: ART CMD writes no type id");
        check_block(name, &alf4.lut, &dir, cube_name, LOGC4);
        let look = alf::parse(&bytes).expect(name);
        assert_eq!(look.input, LookInput::LogC4);
        assert!(look.cdl.is_none(), "{name}: identity CDL");
        assert_eq!(
            file.to_bytes().expect("write"),
            bytes,
            "{name}: rebuilt bytes"
        );
        parsed += 1;
    }

    // .alf4c: CMT 17 with SDR 33 and HDR 65.
    {
        let name = "alf4c_cmt17_sdr33_hdr65.alf4c";
        let bytes = read(name);
        let file = parse_file(&bytes).expect(name);
        let AlfFile::Alf4c(f) = &file else {
            panic!("{name} is not an .alf4c")
        };
        assert_eq!(f.note, "CMT SDR HDR");
        check_block(
            name,
            f.cmt.as_ref().expect("cmt"),
            &dir,
            "cmt17.cube",
            LOGC4,
        );
        check_block(
            name,
            f.sdr.as_ref().expect("sdr"),
            &dir,
            "sdr33.cube",
            REC709,
        );
        check_block(
            name,
            f.hdr.as_ref().expect("hdr"),
            &dir,
            "hdr65.cube",
            "com.apple.proapps.colorspace.rec2100.d65.pq",
        );
        let look = alf::parse(&bytes).expect(name);
        assert_eq!(look.lut.size(), 17);
        assert!(look.display_transforms_ignored);
        assert_eq!(
            file.to_bytes().expect("write"),
            bytes,
            "{name}: rebuilt bytes"
        );
        parsed += 1;
    }

    // .alf4c without a CMT: refused in plain words, but a valid container.
    {
        let name = "alf4c_nocmt.alf4c";
        let bytes = read(name);
        let file = parse_file(&bytes).expect(name);
        let AlfFile::Alf4c(f) = &file else {
            panic!("{name} is not an .alf4c")
        };
        assert!(f.cmt.is_none());
        check_block(
            name,
            f.sdr.as_ref().expect("sdr"),
            &dir,
            "sdr33.cube",
            REC709,
        );
        check_block(
            name,
            f.hdr.as_ref().expect("hdr"),
            &dir,
            "hdr33.cube",
            "com.apple.proapps.colorspace.rec2100.d65.hlg",
        );
        assert_eq!(alf::parse(&bytes), Err(AlfError::NoCmt));
        assert_eq!(
            file.to_bytes().expect("write"),
            bytes,
            "{name}: rebuilt bytes"
        );
        parsed += 1;
    }

    // .alf4c with design data on the SDR transform.
    {
        let name = "alf4c_design.alf4c";
        let bytes = read(name);
        let file = parse_file(&bytes).expect(name);
        let AlfFile::Alf4c(f) = &file else {
            panic!("{name} is not an .alf4c")
        };
        check_block(
            name,
            f.cmt.as_ref().expect("cmt"),
            &dir,
            "cmt33.cube",
            LOGC4,
        );
        let sdr = f.sdr.as_ref().expect("sdr");
        check_block(
            name,
            sdr,
            &dir,
            "sdr17.cube",
            "com.apple.proapps.colorspace.rec2020.d65.gamma24",
        );
        assert_eq!(sdr.design_key, "our-key");
        assert_eq!(
            sdr.extra,
            fs::read(dir.join("samples/design.zip")).expect("design.zip"),
            "the design data is stored byte for byte"
        );
        assert!(alf::parse(&bytes).expect(name).display_transforms_ignored);
        assert_eq!(
            file.to_bytes().expect("write"),
            bytes,
            "{name}: rebuilt bytes"
        );
        parsed += 1;
    }

    // .aml: 33 and 65 (even nodes) for Rec.709, 17 (resampled) for Rec.2020 PQ.
    for (name, cube_name) in [
        ("aml_709_33.aml", "sdr33.cube"),
        ("aml_709_65.aml", "sdr65.cube"),
    ] {
        let bytes = read(name);
        let file = parse_file(&bytes).expect(name);
        let AlfFile::Aml(f) = &file else {
            panic!("{name} is not an .aml")
        };
        assert_eq!(
            (f.size, f.colorspace.as_str()),
            (33, REC709_COLORSPACE),
            "{name}"
        );
        let ours = alf::parse(&bytes).expect(name);
        assert_eq!(ours.input, LookInput::LogC3ToRec709);
        let cube = cube_of(&dir, cube_name);
        let n = cube.size();
        let step = (n - 1) / 32;
        let mut d = 0.0_f64;
        for r in 0..33 {
            for g in 0..33 {
                for b in 0..33 {
                    let a = ours.lut.entry(r, g, b).expect("node");
                    let c = cube.entry(r * step, g * step, b * step).expect("node");
                    for k in 0..3 {
                        d = d.max((a[k] - c[k]).abs());
                    }
                }
            }
        }
        eprintln!(
            "{name}: 33 points vs the nodes of {cube_name} every {step}: max abs diff {d:.3e}"
        );
        assert!(d <= ONE_CODE_HALF, "{name}: {d:e}");
        assert_eq!(
            file.to_bytes().expect("write"),
            bytes,
            "{name}: rebuilt bytes"
        );
        parsed += 1;
    }
    {
        let name = "aml_2020pq_17.aml";
        let bytes = read(name);
        let file = parse_file(&bytes).expect(name);
        let AlfFile::Aml(f) = &file else {
            panic!("{name} is not an .aml")
        };
        assert_eq!(f.colorspace, "com.apple.proapps.colorspace.rec2100.d65.pq");
        assert_eq!(f.size, 33);
        let err = alf::parse(&bytes).unwrap_err();
        assert_eq!(err, AlfError::NotRec709("a Rec.2020 PQ display".to_owned()));
        eprintln!("{name}: refused: {err}");
        // The LUT is ART's 17 -> 33 resampling: tetrahedral, within two codes of ours.
        let cube = cube_of(&dir, "sdr17.cube");
        let lut = LutBlock {
            colorspace: f.colorspace.clone(),
            size: f.size,
            scale: 65535,
            source_name: String::new(),
            design_key: String::new(),
            payload: f.payload.clone(),
            extra: Vec::new(),
        }
        .to_lut3d()
        .expect("decodes");
        let mut d = 0.0_f64;
        for r in 0..33 {
            for g in 0..33 {
                for b in 0..33 {
                    let at = [r, g, b].map(|i| i as f64 / 32.0);
                    let want = cube.sample(at);
                    let got = lut.entry(r, g, b).expect("node");
                    for k in 0..3 {
                        d = d.max((got[k] - want[k]).abs());
                    }
                }
            }
        }
        eprintln!(
            "{name}: 17 -> 33 against our tetrahedral sampling: max abs diff {d:.3e} (bound {:.3e})",
            2.0 / 65535.0
        );
        assert!(d <= 2.0 / 65535.0 + 1e-9, "{name}: {d:e}");
        assert_eq!(
            file.to_bytes().expect("write"),
            bytes,
            "{name}: rebuilt bytes"
        );
        parsed += 1;
    }
    assert_eq!(parsed, 9);
    assert_eq!(AlfKind::Aml.extension(), "aml");
    eprintln!("9 of 9 files parsed, within bounds, and rebuilt byte for byte");
}
