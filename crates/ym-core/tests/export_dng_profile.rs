//! The DNG export writes the fitted profile (`docs/09`, DNG with the look baked in): the tone curve and look table of
//! `data/dng/profile.json`, embedded in `ym-export`, and `BaselineExposure::Auto` with the offset they were fitted with.
//! `ExportContext::new` is what the app and the CLI both use, so these are the bytes of both.
mod common;

use std::path::Path;

use common::tiffmini::Tiff;
use common::*;
use ym_core::export::{
    ExportContext, ExportControl, ExportParams, ExportPhoto, ExportReport, export_photo,
};
use ym_core::session::{OpenOptions, Session};
use ym_export::dng::{DngProfile, FittedProfile};
use ym_export::{ExportOptions, FormatKind};

fn export_dng(ctx: &ExportContext, out: &Path) -> (ExportReport, Vec<u8>) {
    let src = tempfile::tempdir().expect("source folder");
    let scene = Scene::new(
        240,
        160,
        light("sony-ilce-7rm4", 5500.0),
        Layout::Patchwork,
        0.4,
    )
    .noise(2e-4, 1e-5);
    scene
        .builder("ILCE-7RM4")
        .write_to(&src.path().join("DSC00001.ARW"), &scene.mosaic())
        .expect("write the synthetic ARW");
    let session = Session::new(base_env());
    let opened = session.open(&[src.path().to_path_buf()], OpenOptions::default());
    assert_eq!(opened.added.len(), 1, "one photo added");
    settle(&session);
    let photo = ExportPhoto::from_session(&session, opened.added[0]).expect("export photo");
    let options = ExportOptions {
        format: FormatKind::Dng,
        destination: Some(out.to_path_buf()),
        ..ExportOptions::default()
    };
    let report = export_photo(
        ctx,
        &photo,
        &ExportParams::new(options),
        &ExportControl::none(),
    )
    .expect("export a DNG");
    let bytes = std::fs::read(&report.path).expect("read the DNG");
    (report, bytes)
}

/// The value of an `SRATIONAL` tag with one value.
fn srational(tiff: &Tiff<'_>, ifd: usize, tag: u16) -> f64 {
    let entry = tiff.find(ifd, tag).expect("the tag");
    assert_eq!(
        (entry.typ, entry.count),
        (10, 1),
        "tag {tag} is one SRATIONAL"
    );
    let d = tiff.data(&entry);
    let (n, den) = (
        i32::from_le_bytes(d[..4].try_into().expect("4 bytes")),
        i32::from_le_bytes(d[4..8].try_into().expect("4 bytes")),
    );
    f64::from(n) / f64::from(den)
}

#[test]
fn a_dng_carries_the_fitted_profile_and_a_baseline_exposure_that_is_never_negative() {
    let (home, out) = (
        tempfile::tempdir().expect("home"),
        tempfile::tempdir().expect("out"),
    );
    let ctx = export_context(home.path());
    let (report, bytes) = export_dng(&ctx, out.path());
    let fitted = FittedProfile::embedded();
    let tiff = Tiff::new(&bytes).expect("a DNG");
    assert!(tiff.le, "the writer is little endian");
    let ifd0 = tiff.ifd0();

    // The profile in the file is the embedded one, not the identity.
    let dims = tiff.uints(&tiff.find(ifd0, 50981).expect("ProfileLookTableDims"));
    assert_eq!(dims, fitted.profile.look_table.dims.to_vec());
    assert_ne!(dims, vec![2, 2, 2], "not the identity table");
    let curve = tiff.find(ifd0, 50940).expect("ProfileToneCurve");
    assert_eq!(curve.count as usize, fitted.profile.tone_curve.len() * 2);
    assert_eq!(
        tiff.uint(ifd0, 51108),
        Some(fitted.profile.look_table.encoding),
        "ProfileLookTableEncoding"
    );
    let table = tiff.find(ifd0, 50982).expect("ProfileLookTableData");
    assert_eq!(table.count as usize, fitted.profile.look_table.data.len());

    // The white level is floored at 2^-offset, so BaselineExposure is never negative; it is
    // log2(x_white) + offset + the display placement (the app's ACES 2 path brightens the scene by
    // 2^DISPLAY_PLACEMENT_EV before the transform that the profile was fitted to).
    let dng = report.dng.expect("a DNG report");
    let floor = f64::from(fitted.baseline_exposure().white_floor());
    assert_eq!(floor, 16.0);
    assert!(f64::from(dng.x_white) >= floor, "x_white {}", dng.x_white);
    assert!(
        dng.baseline_exposure_ev >= 0.0,
        "{}",
        dng.baseline_exposure_ev
    );
    let want = f64::from(dng.x_white).log2()
        + fitted.fit_offset_ev
        + ym_color::display::DISPLAY_PLACEMENT_EV;
    assert!((dng.baseline_exposure_ev - want).abs() < 1e-9);
    let written = srational(&tiff, ifd0, 50730);
    assert!(
        (written - dng.baseline_exposure_ev).abs() < 1e-6,
        "BaselineExposure {written} against the report {}",
        dng.baseline_exposure_ev
    );
}

#[test]
fn another_profile_can_still_be_given_to_the_context() {
    let (home, out) = (
        tempfile::tempdir().expect("home"),
        tempfile::tempdir().expect("out"),
    );
    let identity = FittedProfile {
        profile: DngProfile::identity(),
        fit_offset_ev: 0.0,
    };
    let ctx = export_context(home.path()).with_dng_profile(identity);
    let (report, bytes) = export_dng(&ctx, out.path());
    let tiff = Tiff::new(&bytes).expect("a DNG");
    let dims = tiff.uints(&tiff.find(tiff.ifd0(), 50981).expect("dims"));
    assert_eq!(dims, vec![2, 2, 2]);
    // With no offset the floor is 1.0 and the exposure is log2(x_white) plus the display placement.
    let dng = report.dng.expect("a DNG report");
    assert!(dng.x_white >= 1.0);
    let want = f64::from(dng.x_white).log2() + ym_color::display::DISPLAY_PLACEMENT_EV;
    assert!((dng.baseline_exposure_ev - want).abs() < 1e-9);
}
