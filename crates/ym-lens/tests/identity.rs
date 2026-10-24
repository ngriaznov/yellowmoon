//! All three amounts at 0 are no lens, whatever the lens says (`docs/21` section 2).

mod common;

use common::{apply_tile, camera, noise, plan, planes_for, whole};
use ym_lens::{Axis, LensAmounts, LensKind, PixelRect, apply, halo_px};

#[test]
fn zero_amounts_return_the_input_bit_for_bit() {
    let (w, h) = (300, 200);
    let data = noise(w, h, 5);
    for kind in LensKind::ALL {
        let p = plan(kind, 0.0, 0.0, 0.0, w, h, Axis::X);
        assert!(p.is_identity());
        let planes = planes_for(&p);
        assert_eq!(halo_px(&p, &planes, 1), 0);
        assert_eq!(p.key(), "lens:none");
        assert_eq!(p.planes_key(), "lens-planes:none");
        assert!(!p.needs_blur_map());
        // The whole frame, and a tile cut from the middle.
        let out = apply(
            &p,
            &planes,
            1,
            &whole(w, h, &data),
            PixelRect::new(0, 0, w, h),
        )
        .unwrap();
        assert_eq!(out, data, "{kind}");
        let rect = PixelRect::new(37, 11, 120, 90);
        let tile = apply_tile(&p, &planes, 1, &data, w, h, rect);
        assert_eq!(tile, common::crop(&data, w, h, rect), "{kind}");
        // The warp is the identity too.
        let warp = p.warp(w as f64, h as f64);
        assert!(warp.is_identity());
        assert_eq!(warp.zoom(), 1.0);
        assert_eq!(warp.canvas_to_photo(12.5, 99.25), (12.5, 99.25));
        assert_eq!(warp.photo_to_canvas(12.5, 99.25), (12.5, 99.25));
    }
}

#[test]
fn identity_ignores_the_lens_in_its_keys() {
    let keys: Vec<String> = LensKind::ALL
        .iter()
        .map(|k| plan(*k, 0.0, 0.0, 0.0, 300, 200, Axis::X).key())
        .collect();
    assert!(keys.iter().all(|k| k == &keys[0]));
    // The amounts' own check: not identity as soon as one is above 0.
    assert!(
        !LensAmounts {
            kind: LensKind::Vintage,
            flares: 0.0,
            edges: 0.0,
            anamorphic: 0.5
        }
        .is_identity()
    );
    assert!(
        LensAmounts {
            kind: LensKind::Vintage,
            flares: 0.0,
            edges: 0.0,
            anamorphic: 0.0
        }
        .is_identity()
    );
}

#[test]
fn flares_alone_change_the_picture_through_the_highlight_optics() {
    // Flares above 0 is a lens even without Edges and Anamorphic: the field optics have nothing to do, but L3
    // takes its fraction of the light and gives it back as glow, veil, ghosts and starbursts. At this size the
    // glow's core is a single pixel, so a tile needs no context.
    let (w, h) = (300, 200);
    let data = noise(w, h, 6);
    let p = plan(LensKind::Classic, 60.0, 0.0, 0.0, w, h, Axis::X);
    assert!(!p.is_identity());
    let planes = planes_for(&p);
    assert_eq!(halo_px(&p, &planes, 1), 0);
    let out = apply(
        &p,
        &planes,
        1,
        &whole(w, h, &data),
        PixelRect::new(0, 0, w, h),
    )
    .unwrap();
    assert_ne!(out, data);
    assert!(out.iter().all(|v| v.is_finite() && *v >= 0.0));
    // The picture keeps 1 - a of itself and gets the flare light on top: every pixel stays within the range
    // of the input and what the budget can add.
    let a = p.light_fraction();
    assert!(a > 0.0 && a < 1.0);
    let (lo, hi) = data
        .iter()
        .fold((f32::MAX, f32::MIN), |(lo, hi), v| (lo.min(*v), hi.max(*v)));
    for (o, z) in out.iter().zip(&data) {
        let o = f64::from(*o);
        assert!(o >= (1.0 - a) * f64::from(*z) - 1e-6);
        assert!(o <= (1.0 - a) * f64::from(*z) + 8.0 * f64::from(hi), "{o}");
    }
    assert!(lo > 0.0);
    // Flares 0 with the other amounts at 0 is no lens at all.
    let zero = plan(LensKind::Classic, 0.0, 0.0, 0.0, w, h, Axis::X);
    assert!(zero.is_identity());
    let _ = camera();
}
