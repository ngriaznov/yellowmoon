//! The colour of the light the field stages move (`docs/21` section 4, *As built (the light's colour)*): a light
//! clipped in every channel at the clip pattern, or in green alone, spreads light of its own colour in L1 and L2,
//! not the cast of the white balance's clip levels.

#![allow(clippy::needless_range_loop)]

mod common;
mod scenes;

use common::{apply_tile, camera, library, tiles, whole};
use scenes::{Noise, balls_over_texture_scene};
use ym_lens::{
    Axis, FrameInfo, LensAmounts, LensKind, LensPlan, LensPlanes, PixelRect, apply_field, lit_input,
};

/// The clip levels `g0` of a body whose white balance puts 2.6 on red and 1.95 on blue: green clips first.
const G0: [f32; 3] = [2.6, 1.0, 1.95];

const W: usize = 600;
const H: usize = 400;

fn plan_with_clip(
    kind: LensKind,
    flares: f64,
    edges: f64,
    anamorphic: f64,
    clip: [f32; 3],
    w: usize,
    h: usize,
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
        f_number: None,
        clip,
    };
    LensPlan::new(library(), &amounts, frame, camera()).expect("plan")
}

/// A frame of a neutral dim ground (the same in every channel of the balanced space) with a disc of `value` at
/// `(cx, cy)`, radius `r`.
fn scene(value: [f32; 3], cx: f64, cy: f64, r: f64) -> Vec<f32> {
    let n = W * H;
    let mut data = vec![0.02f32; 3 * n];
    for y in 0..H {
        for x in 0..W {
            let (dx, dy) = (x as f64 + 0.5 - cx, y as f64 + 0.5 - cy);
            if dx * dx + dy * dy <= r * r {
                for c in 0..3 {
                    data[c * n + y * W + x] = value[c];
                }
            }
        }
    }
    data
}

/// The light the disc adds to the field stages' output, per channel, summed over the rectangle: the picture with the
/// disc minus the picture without it.
fn spread_light(plan: &LensPlan, value: [f32; 3], cx: f64, cy: f64, r: f64) -> [f64; 3] {
    let planes = common::planes_for(plan);
    let with = scene(value, cx, cy, r);
    let without = scene([0.02; 3], cx, cy, r);
    let whole_frame = PixelRect::new(0, 0, W, H);
    let a = apply_field(plan, &planes, 1, &whole(W, H, &with), whole_frame).expect("with");
    let b = apply_field(plan, &planes, 1, &whole(W, H, &without), whole_frame).expect("without");
    let n = W * H;
    let mut sum = [0.0f64; 3];
    for c in 0..3 {
        for i in 0..n {
            sum[c] += f64::from(a[c * n + i]) - f64::from(b[c * n + i]);
        }
    }
    sum
}

/// The chromaticity of a light: each channel over the mean of the three (1 1 1 is neutral).
fn chroma(sum: [f64; 3]) -> [f64; 3] {
    let m = (sum[0] + sum[1] + sum[2]) / 3.0;
    [sum[0] / m, sum[1] / m, sum[2] / m]
}

#[test]
fn a_disc_at_the_clip_pattern_spreads_neutral_light() {
    // Highlight recovery off: every channel at its own clip level, which is a magenta cast after the white
    // balance (2.6, 1, 1.95). A light of that value is white; its spread must be too.
    for (kind, edges, anamorphic) in [
        (LensKind::Vintage, 100.0, 0.0),
        (LensKind::Petzval, 100.0, 0.0),
        (LensKind::Default, 100.0, 100.0),
    ] {
        let plan = plan_with_clip(kind, 0.0, edges, anamorphic, G0, W, H);
        // Near the right edge, where the spread functions are widest.
        let got = chroma(spread_light(&plan, G0, 520.0, 180.0, 5.0));
        for (c, v) in got.iter().enumerate() {
            assert!(
                (v - 1.0).abs() < 0.01,
                "{kind} {edges}/{anamorphic}: chromaticity {got:?}, channel {c}"
            );
        }
    }
}

#[test]
fn a_disc_clipped_in_green_alone_spreads_green_at_the_mean_of_the_others() {
    // The star of the review: green at its clip level, red and blue still measuring. Green is a lower bound,
    // so the light it spreads has green at the mean of the others' (and red and blue as they are).
    let value = [1.4f32, 1.1, 1.7];
    for (kind, edges) in [(LensKind::Vintage, 100.0), (LensKind::Petzval, 100.0)] {
        let plan = plan_with_clip(kind, 0.0, edges, 0.0, G0, W, H);
        let sum = spread_light(&plan, value, 520.0, 180.0, 5.0);
        let others = 0.5 * (sum[0] + sum[2]);
        assert!(
            (sum[1] / others - 1.0).abs() < 0.01,
            "{kind}: red {:.4} green {:.4} blue {:.4}",
            sum[0],
            sum[1],
            sum[2]
        );
        // Red and blue keep their ratio: 1.4 to 1.7 less the ground.
        let want = (1.4 - 0.02) / (1.7 - 0.02);
        assert!((sum[0] / sum[2] / want - 1.0).abs() < 0.01);
    }
}

#[test]
fn a_light_with_a_colour_of_its_own_keeps_it() {
    // A sodium lamp: green clipped, red above, blue far under. Nothing to raise: the mean of the others is under
    // green's own value.
    let value = [2.2f32, 1.2, 0.2];
    let plan = plan_with_clip(LensKind::Vintage, 0.0, 100.0, 0.0, G0, W, H);
    let sum = spread_light(&plan, value, 520.0, 180.0, 5.0);
    let want = [2.2 - 0.02, 1.2 - 0.02, 0.2 - 0.02];
    let want = chroma([want[0], want[1], want[2]].map(f64::from));
    let got = chroma(sum);
    for c in 0..3 {
        assert!(
            (got[c] / want[c] - 1.0).abs() < 0.01,
            "{got:?} against {want:?}"
        );
    }
}

#[test]
fn the_light_the_lens_leaves_alone_is_not_touched() {
    // Flares alone and no lens move nothing: the picture is the input, bit for bit, with a saturated pixel too.
    let value = [1.4f32, 1.1, 1.7];
    let data = scene(value, 300.0, 200.0, 4.0);
    let flares_only = plan_with_clip(LensKind::Vintage, 60.0, 0.0, 0.0, G0, W, H);
    assert!(lit_input(&flares_only, &whole(W, H, &data)).is_none());
    // Nothing saturated: nothing to give.
    let dim = vec![0.3f32; 3 * W * H];
    let edges = plan_with_clip(LensKind::Vintage, 0.0, 60.0, 0.0, G0, W, H);
    assert!(lit_input(&edges, &whole(W, H, &dim)).is_none());
    // With a saturated pixel the lit input differs from the picture there and nowhere else.
    let lit = lit_input(&edges, &whole(W, H, &data)).expect("a saturated light");
    let n = W * H;
    let mut changed = 0;
    for i in 0..n {
        let same = (0..3).all(|c| lit[c * n + i] == data[c * n + i]);
        if !same {
            changed += 1;
            assert!(lit[n + i] > data[n + i], "only green is raised");
            assert_eq!(lit[i], data[i]);
            assert_eq!(lit[2 * n + i], data[2 * n + i]);
        }
    }
    assert!(changed > 20, "the disc's pixels changed: {changed}");
}

#[test]
fn tiles_of_the_field_stages_equal_the_whole_frame_with_saturated_lights() {
    let mut data = scene([1.4, 1.1, 1.7], 120.0, 90.0, 6.0);
    let n = W * H;
    // A second light at the clip pattern and a coloured one.
    for y in 250..258 {
        for x in 430..438 {
            for c in 0..3 {
                data[c * n + y * W + x] = G0[c];
            }
        }
    }
    for y in 300..306 {
        for x in 60..66 {
            data[y * W + x] = 2.2;
            data[n + y * W + x] = 1.2;
            data[2 * n + y * W + x] = 0.2;
        }
    }
    // Two stars the headroom reaches: one clipped in every channel, one in green alone.
    for (cx, cy, v) in [(250.0, 70.0, [3.1f32, 3.0, 3.0]), (330.0, 140.0, [1.4, 1.1, 1.7])] {
        for y in 0..H {
            for x in 0..W {
                let (dx, dy) = (x as f64 + 0.5 - cx, y as f64 + 0.5 - cy);
                if dx * dx + dy * dy <= 2.0 * 2.0 {
                    for c in 0..3 {
                        data[c * n + y * W + x] = v[c];
                    }
                }
            }
        }
    }
    let plan = plan_with_clip(LensKind::Petzval, 0.0, 80.0, 30.0, G0, W, H);
    let planes = LensPlanes::build(&plan, &whole(W, H, &data), None, &|| false).expect("planes");
    let whole_out = apply_tile(&plan, &planes, 1, &data, W, H, PixelRect::new(0, 0, W, H));
    for t in tiles(W, H, 3, 2) {
        let got = apply_tile(&plan, &planes, 1, &data, W, H, t);
        for c in 0..3 {
            for y in 0..t.h {
                for x in 0..t.w {
                    let a = got[c * t.w * t.h + y * t.w + x];
                    let b = whole_out[c * W * H + (t.y + y) * W + t.x + x];
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "tile {t}: channel {c} at ({x}, {y})"
                    );
                }
            }
        }
    }
}

#[test]
fn balls_at_the_clip_pattern_are_redrawn_in_neutral_light() {
    // The scene's lights are (4.0, 3.4, 2.6): a clip pattern when g0 is that. L1 redraws them from the plane's
    // balls, which must carry the light's colour (neutral), not the cast.
    let clip = [4.0f32, 3.4, 2.6];
    let sc = balls_over_texture_scene(1000, 667, 9.0, &[9.0; 6], Noise::LOW, 4);
    let (w, h) = (sc.img.w, sc.img.h);
    let plan = plan_with_clip(LensKind::Default, 0.0, 0.0, 100.0, clip, w, h);
    let planes =
        LensPlanes::build(&plan, &whole(w, h, &sc.img.data), sc.focus, &|| false).expect("planes");
    let bp = planes.bokeh().expect("the plan reshapes bokeh");
    let (bw, bh) = bp.size();
    let mut checked = 0;
    // The scene's lights (a bright patch of the blurred texture can be a ball too, and has its own colour).
    for &(cx, cy, r) in &sc.discs {
        let Some(b) = bp
            .balls()
            .iter()
            .find(|b| (b.x * bw as f64 - cx).hypot(b.y * bh as f64 - cy) < 0.6 * r)
        else {
            continue;
        };
        checked += 1;
        let got = chroma(b.energy);
        for c in 0..3 {
            // A rim pixel is a blend of the light and the ground, not the clip pattern: the cast is under 0.04
            // against 0.2 raw (4.0 : 3.4 : 2.6 is 1.2 : 1.02 : 0.78).
            assert!(
                (got[c] - 1.0).abs() < 0.04,
                "the ball of the light at ({cx:.0}, {cy:.0}) carries {got:?}"
            );
        }
    }
    assert!(checked >= 4, "found the balls of {checked} lights");
}

/// The light the stars' scene adds over the ground-only scene to the field stages' output, per channel, summed
/// over the annulus `lo..hi` pixels from `(cx, cy)`. The planes are built from each scene (the clipped lights'
/// headroom is made from the photo).
fn annulus_light(plan: &LensPlan, scene: &[f32], ground: &[f32], c: (f64, f64), lo: f64, hi: f64) -> [f64; 3] {
    let run = |data: &[f32]| {
        let planes = LensPlanes::build(plan, &whole(W, H, data), None, &|| false).expect("planes");
        apply_field(
            plan,
            &planes,
            1,
            &whole(W, H, data),
            PixelRect::new(0, 0, W, H),
        )
        .expect("field")
    };
    let (a, b) = (run(scene), run(ground));
    let n = W * H;
    let mut sum = [0.0f64; 3];
    for y in 0..H {
        for x in 0..W {
            let r = (x as f64 + 0.5 - c.0).hypot(y as f64 + 0.5 - c.1);
            if r >= lo && r < hi {
                for ch in 0..3 {
                    sum[ch] += f64::from(a[ch * n + y * W + x]) - f64::from(b[ch * n + y * W + x]);
                }
            }
        }
    }
    sum
}

/// A small clipped light with a violet fringe: a core of 9 pixels at the neutral level a recovered clipped
/// photosite has, a ring of unclipped violet pixels round it that holds more light than the core does.
fn violet_star(cx: f64, cy: f64) -> Vec<f32> {
    let n = W * H;
    let mut data = vec![0.02f32; 3 * n];
    for y in 0..H {
        for x in 0..W {
            let r = (x as f64 + 0.5 - cx).hypot(y as f64 + 0.5 - cy);
            let v = if r <= 1.6 {
                [3.1, 3.0, 3.0]
            } else if r <= 4.0 {
                // The fringe: blue and red over green, 1.2 to 1.6 clip levels of green's.
                [1.1, 0.35, 1.8]
            } else {
                continue;
            };
            for c in 0..3 {
                data[c * n + y * W + x] = v[c];
            }
        }
    }
    data
}

#[test]
fn a_small_clipped_light_with_a_violet_fringe_spreads_white_light() {
    // The star of 60855: the fringe's pixels hold more light than the core's, in violet. Spread as the picture
    // records it the halo is mostly the fringe's colour; with the headroom of the clipped core, which is
    // neutral, the halo is the core's. At 6 to 30 px, where the lens's spread is wide near the frame's edge.
    let (cx, cy) = (500.5, 170.5);
    let star = violet_star(cx, cy);
    let ground = vec![0.02f32; 3 * W * H];
    for (kind, edges, anamorphic) in [
        (LensKind::Vintage, 100.0, 0.0),
        (LensKind::Petzval, 100.0, 0.0),
        (LensKind::Vintage, 100.0, 100.0),
    ] {
        let plan = plan_with_clip(kind, 0.0, edges, anamorphic, G0, W, H);
        let got = chroma(annulus_light(&plan, &star, &ground, (cx, cy), 6.0, 30.0));
        for (c, v) in got.iter().enumerate() {
            assert!(
                (v - 1.0).abs() < 0.05,
                "{kind} {edges}/{anamorphic}: the added light is {got:?}, channel {c}"
            );
        }
    }
    // Without a clipped core the fringe's colour spreads: the test would not tell the two apart otherwise.
    let ring_only = {
        let mut d = star.clone();
        let n = W * H;
        for y in 0..H {
            for x in 0..W {
                if (x as f64 + 0.5 - cx).hypot(y as f64 + 0.5 - cy) <= 1.6 {
                    for c in 0..3 {
                        d[c * n + y * W + x] = 1.1 * [1.0, 0.32, 1.64][c];
                    }
                }
            }
        }
        d
    };
    let plan = plan_with_clip(LensKind::Vintage, 0.0, 100.0, 0.0, G0, W, H);
    let got = chroma(annulus_light(&plan, &ring_only, &ground, (cx, cy), 6.0, 30.0));
    assert!(
        got[1] < 0.8 && got[2] > 1.2,
        "a violet light spreads violet: {got:?}"
    );
}

#[test]
fn a_clipped_light_gets_no_headroom_where_the_lens_does_not_move_it_and_none_in_a_surface() {
    // The planes of a photo with a window 120 pixels across at the clip pattern: no headroom, the light
    // spreads as the picture records it, so a pixel's output cannot exceed the brightest value near it.
    let mut data = vec![0.02f32; 3 * W * H];
    let n = W * H;
    for y in 100..220 {
        for x in 300..420 {
            for c in 0..3 {
                data[c * n + y * W + x] = G0[c];
            }
        }
    }
    let plan = plan_with_clip(LensKind::Vintage, 0.0, 100.0, 0.0, G0, W, H);
    let planes = LensPlanes::build(&plan, &whole(W, H, &data), None, &|| false).expect("planes");
    let out = apply_field(&plan, &planes, 1, &whole(W, H, &data), PixelRect::new(0, 0, W, H))
        .expect("field");
    let most = G0.iter().fold(0.0f32, |m, v| m.max(*v));
    assert!(
        out.iter().all(|v| *v <= most * 1.001),
        "the window is a surface: nothing in the output exceeds its own level"
    );
}
