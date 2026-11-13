//! The NEX-5N and NEX-7 get the lens corrections now that their SR2 lens tables are read
//! (`docs/04-raw-decoding.md` section Lens tags): vignetting (R3), chromatic aberration (R5b) and
//! distortion (R9) all have a table to work from, and the distortion crop makes the picture smaller.
//!
//! Local files (`tools/fetch/real-eval.sh`, never committed or used in CI): the tests say SKIPPED and pass
//! without them, unless `YM_REQUIRE_REAL_FILES` is set.

use std::path::{Path, PathBuf};

use ym_core::pipeline::raw::has_vignetting_profile;
use ym_gpu::cpu::lens::{LensGeometry, LensSwitches, LensTags};
use ym_raw::metadata::LensTable;

fn real_file(name: &str) -> Option<PathBuf> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../testdata/real/files")
        .join(name);
    if path.is_file() {
        return Some(path);
    }
    assert!(
        std::env::var_os("YM_REQUIRE_REAL_FILES").is_none(),
        "YM_REQUIRE_REAL_FILES is set but {} is missing",
        path.display()
    );
    eprintln!("SKIPPED: {} is not on this machine", path.display());
    None
}

fn stored(t: &Option<LensTable>) -> Option<&[i16]> {
    t.as_ref().map(|t| t.raw.as_slice())
}

fn all_three_corrections_switch_on(file: &str) {
    let Some(path) = real_file(file) else {
        return;
    };
    let raw = ym_raw::decode::decode(&path).expect("decode");
    let (w, h) = (raw.pixel_width(), raw.pixel_height());

    // R3: a usable vignetting table, with the switch on and with it off.
    assert!(has_vignetting_profile(&raw, true), "{file}: vignetting");
    assert!(!has_vignetting_profile(&raw, false), "{file}: switch off");

    // The tables as the pipeline cuts them: 11 knots, 22 for the CA (red then blue).
    let lens = raw.lens.as_ref().expect("lens tables");
    let tags = LensTags::from_stored(
        stored(&lens.vignetting),
        stored(&lens.chromatic_aberration),
        stored(&lens.distortion),
    )
    .expect("tables");
    assert_eq!(tags.vignetting.len(), 11, "{file}");
    assert_eq!(tags.ca.len(), 22, "{file}");
    assert_eq!(tags.distortion.len(), 11, "{file}");

    // R5b and R9: a geometry with distortion and both CA curves, and a crop inside the frame.
    let geometry = LensGeometry::new(w, h, &tags, LensSwitches::default()).expect("geometry");
    assert!(geometry.distortion_knots().is_some(), "{file}: distortion");
    assert!(
        geometry.ca_knots().is_some(),
        "{file}: chromatic aberration"
    );
    assert!(!geometry.is_identity(), "{file}");
    let (cw, ch) = geometry.with_auto_crop().output_size();
    assert!(
        cw < w && ch < h,
        "{file}: the distortion crop {cw}x{ch} of {w}x{h}"
    );

    // The switches take them off again.
    let off = LensGeometry::new(
        w,
        h,
        &tags,
        LensSwitches {
            lens_corrections: false,
            chromatic_aberration: false,
        },
    )
    .expect("geometry");
    assert!(off.is_identity(), "{file}");
}

#[test]
fn the_nex_7_gets_vignetting_chromatic_aberration_and_distortion_corrections() {
    all_three_corrections_switch_on("NEX-7.ARW");
}

#[test]
fn the_nex_5n_gets_vignetting_chromatic_aberration_and_distortion_corrections() {
    all_three_corrections_switch_on("NEX-5N.ARW");
}
