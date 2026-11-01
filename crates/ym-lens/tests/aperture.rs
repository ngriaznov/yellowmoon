//! The aperture law and the headroom's density gate of Edges (`docs/21` section 4, L2, *As built (the
//! aperture)*; the owner's decision of 2026-11-01, "Follow aperture (Recommended)"):
//!
//! * the photo's f-number sets the field optics' width (`(N_d / N)^1.5`, floor 0.1) and darkening
//!   (`(N_d / N)^0.5`), both exactly 1 without an f-number and at or wider than the design's;
//! * a crowd of clipped patches (petals) is a surface and gets no headroom, a clipped star alone keeps it;
//! * with both active, tiles of any size equal the whole frame bit for bit and a repeat run is identical.

#![allow(clippy::needless_range_loop)]

mod common;

use std::sync::OnceLock;

use common::{apply_tile, camera, library, repo_root, tiles, whole};
use ym_lens::{
    Axis, FrameInfo, LensAmounts, LensKind, LensLibrary, LensPlan, LensPlanes, PixelRect, apply,
    field_halo_px, l2_input,
};

/// The shipped lenses (the synthetic library is designed at f/2 throughout; these differ).
fn shipped() -> &'static LensLibrary {
    static LIB: OnceLock<LensLibrary> = OnceLock::new();
    LIB.get_or_init(|| {
        LensLibrary::load(&repo_root().join("data")).expect("the shipped lens data load")
    })
}

fn plan_in(
    lib: &LensLibrary,
    kind: LensKind,
    (flares, edges, anamorphic): (f64, f64, f64),
    (w, h): (usize, usize),
    f_number: Option<f64>,
) -> LensPlan {
    let amounts = LensAmounts {
        kind,
        flares,
        edges,
        anamorphic,
    };
    let frame = FrameInfo {
        width: w,
        height: h,
        axis: Axis::X,
        f_number,
        clip: [1.0; 3],
    };
    LensPlan::new(lib, &amounts, frame, camera()).expect("plan")
}

// ---------------------------------------------------------------------------------------------
// The law
// ---------------------------------------------------------------------------------------------

#[test]
fn the_factors_follow_the_decision() {
    let at = |f: Option<f64>| {
        plan_in(
            library(),
            LensKind::Vintage,
            (0.0, 50.0, 0.0),
            (600, 400),
            f,
        )
    };
    // The synthetic lenses are designed at f/2: a photo at f/8 has (2/8)^1.5 = 1/8 of the width and (2/8)^0.5 =
    // 1/2 of the darkening.
    let p = at(Some(8.0));
    assert!((p.aperture_width_factor() - 0.125).abs() < 1e-12);
    assert!((p.aperture_darkening_factor() - 0.5).abs() < 1e-12);
    // Without an f-number, at the design's and wider: exactly 1.
    for f in [None, Some(2.0), Some(1.8), Some(1.0), Some(0.5)] {
        let p = at(f);
        assert_eq!(
            p.aperture_width_factor().to_bits(),
            1.0f64.to_bits(),
            "{f:?}"
        );
        assert_eq!(
            p.aperture_darkening_factor().to_bits(),
            1.0f64.to_bits(),
            "{f:?}"
        );
    }
    // Far stopped down: the width stops at a tenth, the darkening goes on.
    let far = at(Some(64.0));
    assert_eq!(far.aperture_width_factor(), 0.1);
    assert!((far.aperture_darkening_factor() - (2.0f64 / 64.0).sqrt()).abs() < 1e-12);
    // Both fall as the lens is stopped down, and never rise above 1.
    let (mut width, mut dark) = (1.0f64, 1.0f64);
    for tenths in 10..=450 {
        let p = at(Some(f64::from(tenths) / 10.0));
        assert!(p.aperture_width_factor() <= width && p.aperture_width_factor() >= 0.1);
        assert!(p.aperture_darkening_factor() <= dark);
        width = p.aperture_width_factor();
        dark = p.aperture_darkening_factor();
    }
    // The shipped lenses carry their own design aperture: Triplet is f/4, Classic f/2.
    let shipped_at = |kind, f| plan_in(shipped(), kind, (0.0, 50.0, 0.0), (600, 400), Some(f));
    assert_eq!(
        shipped_at(LensKind::Triplet, 4.0).aperture_width_factor(),
        1.0
    );
    assert!(
        (shipped_at(LensKind::Triplet, 8.0).aperture_width_factor() - 0.5f64.powf(1.5)).abs()
            < 1e-12
    );
    assert!((shipped_at(LensKind::Classic, 8.0).aperture_width_factor() - 0.125).abs() < 1e-12);
}

#[test]
fn the_f_number_is_in_the_field_key_and_flares_are_not() {
    let at = |flares: f64, f: Option<f64>| {
        plan_in(
            library(),
            LensKind::Vintage,
            (flares, 50.0, 20.0),
            (600, 400),
            f,
        )
    };
    let base = at(30.0, Some(5.6));
    assert_eq!(base.field_key(), at(30.0, Some(5.6)).field_key());
    // The field optics follow the aperture, so the f-number moves the key of Z ...
    assert_ne!(base.field_key(), at(30.0, Some(8.0)).field_key());
    assert_ne!(base.field_key(), at(30.0, None).field_key());
    // ... and a Flares move keeps it.
    assert_eq!(base.field_key(), at(70.0, Some(5.6)).field_key());
    // The planes do not read the f-number (the light map and the blur map are the photo's).
    assert_eq!(base.planes_key(), at(30.0, Some(8.0)).planes_key());
}

// ---------------------------------------------------------------------------------------------
// The pictures
// ---------------------------------------------------------------------------------------------

/// The frame of the blossom scenes (the prototype's).
const W: usize = 1800;
const H: usize = 1200;

/// A scene like a cherry tree in bloom: a dim textured ground, optionally a canopy of 900 small clipped
/// discs (radius 2 to 6 px, value 1.4 over a clip level of 1) over `x` 300 to 1200 and `y` 100 to 600, and
/// optionally a clipped star of 5 x 5 pixels alone at (1500, 900). Planar RGB.
fn scene(canopy: bool, star: bool) -> Vec<f32> {
    let mut data = vec![0.0f32; 3 * W * H];
    let mut s = 0x9E37_79B9_7F4A_7C15u64;
    let mut rnd = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 40) as f32 / (1u64 << 24) as f32
    };
    let n = W * H;
    for i in 0..n {
        let v = 0.15 + 0.3 * rnd();
        for c in 0..3 {
            data[c * n + i] = v * [1.0, 0.95, 0.9][c];
        }
    }
    if canopy {
        for _ in 0..900 {
            let (cx, cy) = (
                (300.0 + 900.0 * rnd()) as i64,
                (100.0 + 500.0 * rnd()) as i64,
            );
            let r = 2 + (rnd() * 5.0) as i64;
            for y in cy - r..=cy + r {
                for x in cx - r..=cx + r {
                    if (x - cx).pow(2) + (y - cy).pow(2) <= r * r {
                        let i = y as usize * W + x as usize;
                        for c in 0..3 {
                            data[c * n + i] = 1.4;
                        }
                    }
                }
            }
        }
    }
    if star {
        for y in 900..905 {
            for x in 1500..1505 {
                for c in 0..3 {
                    data[c * n + y * W + x] = 1.6;
                }
            }
        }
    }
    data
}

/// The planes of a plan from the scene's lens plane, the scene reduced by `k` with box means.
fn planes_of_scene(plan: &LensPlan, data: &[f32]) -> LensPlanes {
    let k = plan.lens_plane_scale();
    let (pw, ph) = (W.div_ceil(k), H.div_ceil(k));
    let mut plane = vec![0.0f32; 3 * pw * ph];
    for c in 0..3 {
        for y in 0..ph {
            for x in 0..pw {
                let (mut sum, mut count) = (0.0f32, 0.0f32);
                for dy in 0..k {
                    for dx in 0..k {
                        let (sx, sy) = (x * k + dx, y * k + dy);
                        if sx < W && sy < H {
                            sum += data[c * W * H + sy * W + sx];
                            count += 1.0;
                        }
                    }
                }
                plane[c * pw * ph + y * pw + x] = sum / count;
            }
        }
    }
    LensPlanes::build(plan, &whole(pw, ph, &plane), None, &|| false).expect("planes")
}

/// The headroom L2 spreads, per pixel of the frame.
fn headroom(plan: &LensPlan, planes: &LensPlanes, data: &[f32]) -> Vec<f32> {
    let l2 =
        l2_input(plan, planes, 1, &whole(W, H, data), None).expect("the scene has clipped light");
    assert_eq!(l2.rect, PixelRect::new(0, 0, W, H));
    l2.head[..W * H].to_vec()
}

/// The largest value of `head` over a rectangle.
fn peak(head: &[f32], x: std::ops::Range<usize>, y: std::ops::Range<usize>) -> f32 {
    let mut m = 0.0f32;
    for j in y {
        for i in x.clone() {
            m = m.max(head[j * W + i]);
        }
    }
    m
}

/// A crowd of clipped petals is a surface: none of its patches is given the headroom of a star, where the dark
/// gaps between them would be lifted to two or three times the clip level and turn flat white. A star alone
/// keeps all of its headroom, whatever happens at the other side of the frame.
#[test]
fn a_crowd_of_clipped_patches_gets_no_headroom_and_an_isolated_star_keeps_it() {
    let plan = plan_in(
        shipped(),
        LensKind::Classic,
        (0.0, 60.0, 0.0),
        (W, H),
        Some(8.0),
    );
    let both = scene(true, true);
    let star_only = scene(false, true);
    let with_crowd = headroom(&plan, &planes_of_scene(&plan, &both), &both);
    let alone = headroom(&plan, &planes_of_scene(&plan, &star_only), &star_only);
    // The star alone has a headroom of many clip levels. With the canopy in the picture it keeps its headroom
    // too (a share of it: the photo's cap scale, which L3 sets from the energy of all its lights, falls with
    // the canopy's, and the gate leaves that alone; the unit test of the light map has the full EV).
    let star = peak(&alone, 1500..1505, 900..905);
    assert!(star > 10.0, "the star's headroom is {star} clip levels");
    let with = peak(&with_crowd, 1500..1505, 900..905);
    assert!(
        with > 0.1,
        "the star keeps its headroom with the canopy: {with}"
    );
    // The canopy (its inner part: the patches at its edge see fewer neighbours): nothing.
    let crowd = peak(&with_crowd, 450..1050, 200..500);
    assert_eq!(crowd, 0.0, "the canopy's headroom is {crowd}");
}

/// The width of the field optics follows the aperture: a photo taken at f/8 through a lens designed at f/2 is
/// moved by an eighth of the spread, so it stays nearer to the picture than the same photo taken wide open.
/// At or below the design's f-number, and without an f-number, the bytes are the traced lens's.
#[test]
fn a_stopped_down_photo_is_moved_less_and_a_wide_open_photo_is_the_traced_lens() {
    let (w, h) = (600usize, 400usize);
    let data = common::noise(w, h, 5);
    let at = |f: Option<f64>| plan_in(library(), LensKind::Default, (0.0, 40.0, 0.0), (w, h), f);
    let planes = common::planes_for(&at(None));
    let rect = PixelRect::new(0, 0, w, h);
    let render = |f: Option<f64>| {
        let plan = at(f);
        apply_tile(&plan, &planes, 1, &data, w, h, rect)
    };
    let traced = render(None);
    for f in [2.0, 1.4] {
        let out = render(Some(f));
        assert!(
            out.iter()
                .zip(&traced)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "f/{f} is the traced lens"
        );
    }
    let distance = |out: &[f32]| -> f64 {
        out.iter()
            .zip(&data)
            .map(|(a, b)| f64::from((a - b).abs()))
            .sum::<f64>()
            / out.len() as f64
    };
    let (open, f4, f8, f16) = (
        distance(&traced),
        distance(&render(Some(4.0))),
        distance(&render(Some(8.0))),
        distance(&render(Some(16.0))),
    );
    assert!(open > 0.02, "the lens moves the picture: {open}");
    assert!(
        open > f4 && f4 > f8 && f8 > f16,
        "wide open {open}, f/4 {f4}, f/8 {f8}, f/16 {f16}"
    );
    // The halo of a stopped-down photo is no larger than the wide-open one's.
    let halo = |f| field_halo_px(&at(f), &planes, 1);
    assert!(halo(Some(8.0)) <= halo(None));
}

/// Tiles of any size equal the whole frame bit for bit with both active, and a second run of the whole frame
/// is the first, for the three settings of the prototype's check.
#[test]
fn tiles_equal_the_whole_frame_and_a_repeat_run_is_identical_with_the_aperture_and_the_gate() {
    let data = scene(true, true);
    let mut found = 0;
    for (kind, flares, edges) in [
        (LensKind::Classic, 35.0, 60.0),
        (LensKind::Default, 0.0, 50.0),
        (LensKind::Vintage, 45.0, 100.0),
    ] {
        let plan = plan_in(shipped(), kind, (flares, edges, 0.0), (W, H), Some(8.0));
        let planes = planes_of_scene(&plan, &data);
        let whole_rect = PixelRect::new(0, 0, W, H);
        let a = apply(&plan, &planes, 1, &whole(W, H, &data), whole_rect).expect("apply");
        let b = apply(&plan, &planes, 1, &whole(W, H, &data), whole_rect).expect("apply again");
        assert!(
            a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()),
            "{kind}: a repeat run differs"
        );
        for rect in tiles(W, H, 3, 2) {
            let tile = apply_tile(&plan, &planes, 1, &data, W, H, rect);
            for c in 0..3 {
                for y in 0..rect.h {
                    for x in 0..rect.w {
                        let from_tile = tile[c * rect.w * rect.h + y * rect.w + x];
                        let from_whole = a[c * W * H + (rect.y + y) * W + rect.x + x];
                        assert_eq!(
                            from_tile.to_bits(),
                            from_whole.to_bits(),
                            "{kind} {flares}/{edges}: tile {rect} at ({x}, {y}) channel {c}"
                        );
                    }
                }
            }
            found += 1;
        }
    }
    assert_eq!(found, 18);
}
