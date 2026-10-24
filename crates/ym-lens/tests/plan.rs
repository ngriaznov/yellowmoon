//! Plans, keys, planes and the rules around them (`docs/21` sections 3 and 5).

mod common;

use common::{camera, curves_path, frame, library, plan, planes_for, whole};
use ym_color::spectral::SpectralCurves;
use ym_lens::{
    Axis, CameraResponse, FrameInfo, LensAmounts, LensError, LensKind, LensPlan, LensPlanes,
    ReshapeReason, ReshapeReport,
};

fn plan_of(kind: LensKind, amounts: (f64, f64, f64), (w, h): (usize, usize)) -> LensPlan {
    plan(kind, amounts.0, amounts.1, amounts.2, w, h, Axis::X)
}

#[test]
fn the_lens_plane_scale_follows_the_long_edge() {
    let scale =
        |w: usize, h: usize| plan_of(LensKind::Default, (0.0, 0.0, 0.0), (w, h)).lens_plane_scale();
    assert_eq!(scale(6000, 4000), 4);
    assert_eq!(scale(4000, 6000), 4);
    assert_eq!(scale(8640, 5760), 8);
    assert_eq!(scale(8192, 6144), 8);
    assert_eq!(scale(2048, 1365), 2);
    assert_eq!(scale(4095, 2730), 2);
    assert_eq!(scale(4096, 2730), 4);
    assert_eq!(scale(1500, 1000), 2, "under 2048 px long gives 2");
    assert_eq!(scale(1024, 700), 2);
    assert_eq!(
        scale(1023, 700),
        1,
        "tiny test frames keep the full resolution"
    );
    assert_eq!(scale(600, 400), 1);
    // The lens plane keeps at least 1024 px on its long edge once the frame is that large.
    for long in [2048usize, 3000, 6000, 8640, 12000, 16384] {
        let k = scale(long, long * 2 / 3);
        assert!(long / k >= 1024 || k == 2, "{long}: k = {k}");
    }
}

#[test]
fn the_blur_map_is_needed_by_anamorphic_and_by_swirling_lenses() {
    let needs = |kind, a: (f64, f64, f64)| plan_of(kind, a, (600, 400)).needs_blur_map();
    // Default has neither swirl nor cat's-eye.
    assert!(!needs(LensKind::Default, (0.0, 100.0, 0.0)));
    assert!(!needs(LensKind::Default, (100.0, 0.0, 0.0)));
    assert!(needs(LensKind::Default, (0.0, 0.0, 1.0)));
    // Petzval swirls: Edges asks for it, Flares alone does not.
    assert!(needs(LensKind::Petzval, (0.0, 1.0, 0.0)));
    assert!(!needs(LensKind::Petzval, (100.0, 0.0, 0.0)));
    assert!(needs(LensKind::Vintage, (0.0, 20.0, 0.0)));
    assert!(!needs(LensKind::Default, (0.0, 0.0, 0.0)));
}

#[test]
fn amounts_outside_the_range_are_refused() {
    let lib = library();
    let f = frame(600, 400, Axis::X);
    for bad in [
        LensAmounts {
            kind: LensKind::Default,
            flares: -1.0,
            edges: 0.0,
            anamorphic: 0.0,
        },
        LensAmounts {
            kind: LensKind::Default,
            flares: 0.0,
            edges: 100.5,
            anamorphic: 0.0,
        },
        LensAmounts {
            kind: LensKind::Default,
            flares: 0.0,
            edges: 0.0,
            anamorphic: f64::NAN,
        },
        LensAmounts {
            kind: LensKind::Default,
            flares: f64::INFINITY,
            edges: 0.0,
            anamorphic: 0.0,
        },
    ] {
        assert!(
            matches!(
                LensPlan::new(lib, &bad, f, camera()),
                Err(LensError::Amount { .. })
            ),
            "{bad:?}"
        );
    }
    let ok = LensAmounts {
        kind: LensKind::Default,
        flares: 100.0,
        edges: 100.0,
        anamorphic: 100.0,
    };
    assert!(LensPlan::new(lib, &ok, f, camera()).is_ok());
    assert!(matches!(
        LensPlan::new(lib, &ok, FrameInfo { width: 0, ..f }, camera()),
        Err(LensError::Frame(_))
    ));
    assert!(matches!(
        LensPlan::new(
            lib,
            &ok,
            FrameInfo {
                f_number: Some(-2.0),
                ..f
            },
            camera()
        ),
        Err(LensError::Frame(_))
    ));
    assert!(
        LensPlan::new(
            lib,
            &ok,
            FrameInfo {
                f_number: Some(2.8),
                ..f
            },
            camera()
        )
        .is_ok()
    );
}

#[test]
fn keys_say_what_the_passes_depend_on() {
    let base = plan_of(LensKind::Vintage, (30.0, 60.0, 20.0), (3000, 2000));
    assert_eq!(
        base.key(),
        plan_of(LensKind::Vintage, (30.0, 60.0, 20.0), (3000, 2000)).key()
    );
    // Every input of the per-pixel passes changes the key.
    let others = [
        plan_of(LensKind::Petzval, (30.0, 60.0, 20.0), (3000, 2000)),
        plan_of(LensKind::Vintage, (31.0, 60.0, 20.0), (3000, 2000)),
        plan_of(LensKind::Vintage, (30.0, 61.0, 20.0), (3000, 2000)),
        plan_of(LensKind::Vintage, (30.0, 60.0, 21.0), (3000, 2000)),
        plan_of(LensKind::Vintage, (30.0, 60.0, 20.0), (3000, 2001)),
        plan(LensKind::Vintage, 30.0, 60.0, 20.0, 3000, 2000, Axis::Y),
        LensPlan::new(
            library(),
            base.amounts(),
            FrameInfo {
                f_number: Some(8.0),
                ..*base.frame()
            },
            camera(),
        )
        .unwrap(),
        LensPlan::new(
            library(),
            base.amounts(),
            *base.frame(),
            &CameraResponse::neutral(),
        )
        .unwrap(),
        LensPlan::new(
            library(),
            base.amounts(),
            FrameInfo {
                clip: [0.9, 1.0, 1.0],
                ..*base.frame()
            },
            camera(),
        )
        .unwrap(),
    ];
    for (i, o) in others.iter().enumerate() {
        assert_ne!(o.key(), base.key(), "variant {i}");
    }
    // The planes' key leaves out the amounts and the f-number.
    assert_eq!(
        base.planes_key(),
        plan_of(LensKind::Vintage, (31.0, 10.0, 90.0), (3000, 2000)).planes_key()
    );
    assert_eq!(base.planes_key(), others[6].planes_key());
    // But not the lens, the frame, the axis, the camera or the clip levels.
    for i in [0usize, 4, 5, 7, 8] {
        assert_ne!(others[i].planes_key(), base.planes_key(), "variant {i}");
    }
    let curves = SpectralCurves::from_path(curves_path("Sony_ILCE-6400_380_780_5.json")).unwrap();
    let other_camera = CameraResponse::from_curves(&curves).unwrap();
    let p = LensPlan::new(library(), base.amounts(), *base.frame(), &other_camera).unwrap();
    assert_ne!(p.planes_key(), base.planes_key());
    assert_ne!(p.key(), base.key());
}

#[test]
fn the_light_budget_follows_the_spec() {
    let lens = library().get(LensKind::Sonnar).unwrap();
    let att = library().attachment();
    let p = plan_of(LensKind::Sonnar, (100.0, 0.0, 0.0), (600, 400));
    let st = p.flare_strengths();
    let (lens_part, att_part) = (st.lens_fraction(), st.attachment_fraction());
    // The strengths only raise what the data hold (the synthetic lens is above every floor).
    assert!(lens_part >= lens.flare_fraction() - 1e-12);
    assert!(att_part >= att.flare_fraction() - 1e-12);
    assert!((p.light_fraction() - lens_part).abs() < 1e-12);
    // Half of Flares, full Anamorphic: the streak and the attachment's ghosts at their full fractions.
    let p = plan_of(LensKind::Sonnar, (50.0, 0.0, 100.0), (600, 400));
    assert!((p.light_fraction() - 0.5 * (lens_part + att_part)).abs() < 1e-12);
    // Half of Anamorphic scales the attachment's part by (q - 1) / (Q_max - 1) = 0.5.
    let p = plan_of(LensKind::Sonnar, (100.0, 0.0, 50.0), (600, 400));
    assert!((p.light_fraction() - (lens_part + 0.5 * att_part)).abs() < 1e-12);
    assert_eq!(
        plan_of(LensKind::Sonnar, (0.0, 100.0, 100.0), (600, 400)).light_fraction(),
        0.0
    );
    // The budget stays below 1 for every lens at the largest amounts (the loader checks the data's).
    for kind in LensKind::ALL {
        assert!(plan_of(kind, (100.0, 100.0, 100.0), (600, 400)).light_fraction() < 1.0);
    }
}

#[test]
fn the_strengths_raise_what_the_data_hold_to_the_floors_and_never_lower_it() {
    use ym_lens::consts::{
        ATTACHMENT_GHOST_FLOOR, GHOST_FRACTION_FLOOR, GLOW_FRACTION_FLOOR, STARBURST_FRACTION_FLOOR,
    };
    let lib = ym_lens::LensLibrary::load(&common::repo_root().join("data")).unwrap();
    for kind in LensKind::ALL {
        let amounts = LensAmounts {
            kind,
            flares: 100.0,
            edges: 0.0,
            anamorphic: 100.0,
        };
        let p = LensPlan::new(&lib, &amounts, common::frame(600, 400, Axis::X), camera()).unwrap();
        let (lens, att, st) = (p.lens(), p.attachment(), p.flare_strengths());
        assert!(st.glow >= lens.flare.glow.fraction && st.glow >= GLOW_FRACTION_FLOOR - 1e-15);
        assert_eq!(
            st.veil, lens.flare.veil.fraction,
            "{kind}: the veil is the data's"
        );
        assert_eq!(
            st.streak, att.streak.fraction,
            "{kind}: the streak is the data's"
        );
        for (g, f) in lens.flare.ghosts.iter().zip(&st.ghosts) {
            assert!(*f >= g.fraction, "{kind}: a ghost was lowered");
        }
        for (g, f) in att.ghosts.iter().zip(&st.attachment_ghosts) {
            assert!(*f >= g.fraction, "{kind}: an attachment ghost was lowered");
        }
        let sum: f64 = st.ghosts.iter().sum();
        assert!(
            sum >= GHOST_FRACTION_FLOOR * (1.0 - 1e-12),
            "{kind}: ghosts {sum}"
        );
        let att_sum: f64 = st.attachment_ghosts.iter().sum();
        assert!(
            att_sum >= ATTACHMENT_GHOST_FLOOR * (1.0 - 1e-12),
            "{kind}: {att_sum}"
        );
        if lens.flare.starburst.fraction > 0.0 {
            assert!(st.burst >= STARBURST_FRACTION_FLOOR - 1e-15);
        } else {
            assert_eq!(
                st.burst, 0.0,
                "{kind}: a lens without a starburst stays without"
            );
        }
        // A lens whose ghosts are already above the floor keeps them as they are (Petzval: bare glass).
        if lens.flare.ghosts.iter().map(|g| g.fraction).sum::<f64>() >= GHOST_FRACTION_FLOOR {
            for (g, f) in lens.flare.ghosts.iter().zip(&st.ghosts) {
                assert_eq!(*f, g.fraction, "{kind}");
            }
        }
        // The small ghosts get most of what the floor adds: the smallest ghost's gain is the largest.
        let gains: Vec<(f64, f64)> = lens
            .flare
            .ghosts
            .iter()
            .zip(&st.ghosts)
            .map(|(g, f)| (g.s, f / g.fraction))
            .collect();
        let smallest = gains.iter().min_by(|a, b| a.0.total_cmp(&b.0)).unwrap();
        let largest = gains.iter().max_by(|a, b| a.0.total_cmp(&b.0)).unwrap();
        assert!(smallest.1 >= largest.1 - 1e-9, "{kind}: {gains:?}");
        // The whole budget stays small: every lens keeps at least 80 % of the picture's light at the top.
        assert!(p.light_fraction() < 0.2, "{kind}: {}", p.light_fraction());
    }
}

#[test]
fn the_derived_constants_follow_the_spec() {
    let att = library().attachment();
    let p = plan_of(LensKind::Default, (0.0, 0.0, 0.0), (600, 400));
    assert_eq!(
        (
            p.q(),
            p.field_amount(),
            p.alpha(),
            p.side_softness(),
            p.fringe_factor()
        ),
        (1.0, 0.0, 1.0, 0.0, 1.0)
    );
    let p = plan_of(LensKind::Default, (0.0, 40.0, 100.0), (600, 400));
    assert_eq!(p.q(), att.q_max);
    let extra = att.q_max - 1.0;
    assert!((p.alpha() - (1.0 + att.field.kappa * extra)).abs() < 1e-12);
    assert!((p.side_softness() - att.field.side_softness * extra).abs() < 1e-12);
    assert!((p.fringe_factor() - (1.0 + att.field.fringe * extra)).abs() < 1e-12);
    // e' = e + (1 - e) kappa (q - 1).
    assert!((p.field_amount() - (0.4 + 0.6 * att.field.kappa * extra)).abs() < 1e-12);
    // At Anamorphic 0 it is Edges.
    let p = plan_of(LensKind::Default, (0.0, 40.0, 0.0), (600, 400));
    assert!((p.field_amount() - 0.4).abs() < 1e-15);
}

#[test]
fn planes_are_built_from_the_lens_plane_and_can_be_cancelled() {
    let p = plan_of(LensKind::Vintage, (0.0, 50.0, 50.0), (6000, 4000));
    let k = p.lens_plane_scale();
    assert_eq!(k, 4);
    let (w, h) = (1500usize, 1000usize);
    let data = vec![0.4f32; 3 * w * h];
    let planes = LensPlanes::build(&p, &whole(w, h, &data), Some((0.4, 0.6)), &|| false).unwrap();
    assert_eq!(planes.key(), p.planes_key());
    assert_eq!(planes.focus(), Some((0.4, 0.6)));
    assert!(planes.bytes() > 0);
    assert_eq!(planes.lens_plane_size(), (w, h));
    // A flat plane has no blur to reshape: the planes say so (L1 asked, found little).
    assert_eq!(
        planes.report(),
        ReshapeReport {
            reshaped: false,
            reason: Some(ReshapeReason::LittleBlur)
        }
    );
    assert!(
        planes
            .bokeh()
            .is_some_and(|b| !b.active() && b.balls().is_empty())
    );
    assert!(planes.serves(&p));
    // Flares alone ask for no blur map, whatever the lens.
    let flares = plan_of(LensKind::Vintage, (50.0, 0.0, 0.0), (6000, 4000));
    let none = LensPlanes::build(&flares, &whole(w, h, &data), None, &|| false).unwrap();
    assert_eq!(
        none.report(),
        ReshapeReport {
            reshaped: false,
            reason: Some(ReshapeReason::NotAsked)
        }
    );
    assert!(none.bokeh().is_none());
    // A plane of another size, a plane that is only part of the frame, a focus point outside the frame.
    let small = vec![0.4f32; 3 * 750 * 500];
    assert!(matches!(
        LensPlanes::build(&p, &whole(750, 500, &small), None, &|| false),
        Err(LensError::Frame(_))
    ));
    let mut part = whole(w, h, &data);
    part.x0 = 10;
    part.frame_width = w + 10;
    assert!(LensPlanes::build(&p, &part, None, &|| false).is_err());
    assert!(matches!(
        LensPlanes::build(&p, &whole(w, h, &data), Some((1.5, 0.5)), &|| false),
        Err(LensError::Frame(_))
    ));
    // Cancelling ends the build.
    assert!(matches!(
        LensPlanes::build(&p, &whole(w, h, &data), None, &|| true),
        Err(LensError::Cancelled)
    ));
    let calls = std::cell::Cell::new(0);
    let r = LensPlanes::build(&p, &whole(w, h, &data), None, &|| {
        calls.set(calls.get() + 1);
        calls.get() > 1
    });
    assert!(matches!(r, Err(LensError::Cancelled)));
    // An identity plan needs no plane at all.
    let id = plan_of(LensKind::Vintage, (0.0, 0.0, 0.0), (6000, 4000));
    let empty = LensPlanes::build(&id, &whole(1, 1, &[0.0; 3]), None, &|| false).unwrap();
    assert_eq!(empty.key(), "lens-planes:none");
    let _ = planes_for(&id);
}
