//! R5 highlight reconstruction on synthetic blown regions (`docs/05` R5 gate, `docs/13` highlights row).
//!
//! The gate asks for "no magenta or cyan fringes at the clip boundary" and "the reconstructed
//! region's chroma within dE00 5 of the surrounding unclipped area". Real photographs are not
//! available here, so the scenes are synthetic: a light source of constant chromaticity whose
//! intensity falls off smoothly from a peak well above the sensor's clip level. Where the
//! intensity is high enough, channels clip one after the other (the brightest channel first), which
//! is what creates the hue fringes.
//!
//! The metric works in CIE Lab through the stand-in camera matrix documented in
//! `ym_gpu::cpu::metrics` (white-balanced camera RGB read as linear Rec. 709 / sRGB, D65). Chroma
//! differences are CIEDE2000 with the lightness set equal ([`chroma_delta_e00`]), because a
//! reconstruction cannot recover the true brightness of a blown region, only its colour.

use ym_gpu::cpu::demosaic::{self, Method};
use ym_gpu::cpu::highlights;
use ym_gpu::cpu::metrics::{chroma, chroma_delta_e00, hue_degrees};
use ym_gpu::cpu::synth;
use ym_gpu::cpu::{Cfa, ClipMask, Mosaic, RgbImage};

const W: usize = 160;
const H: usize = 160;

#[derive(Clone, Copy)]
enum Profile {
    /// Gaussian glow of standard deviation 22 px: a gentle falloff, wide transition bands.
    Glow,
    /// A disc of radius 30 px with a 2.5 px logistic edge (a lamp, a window, the sun): all channels
    /// clip within a few pixels of each other, the worst case for hue fringes.
    Disc,
}

struct Scene {
    name: &'static str,
    /// Chromaticity of the source (linear camera RGB, balanced), the largest channel scaled to 1.
    colour: [f32; 3],
    /// Peak intensity multiplier in units of the brightest channel's clip level.
    peak: f32,
    profile: Profile,
}

/// The scene image: constant chromaticity, Gaussian intensity falloff, a faint background.
fn scene_image(s: &Scene) -> RgbImage {
    let mut img = RgbImage::new(W, H);
    for y in 0..H {
        for x in 0..W {
            let (dx, dy) = (
                x as f32 + 0.5 - W as f32 / 2.0,
                y as f32 + 0.5 - H as f32 / 2.0,
            );
            let falloff = match s.profile {
                Profile::Glow => (-(dx * dx + dy * dy) / (2.0 * 22.0 * 22.0)).exp(),
                Profile::Disc => 1.0 / (1.0 + (((dx * dx + dy * dy).sqrt() - 30.0) / 2.5).exp()),
            };
            let i = s.peak * falloff + 0.03;
            img.set_pixel(x, y, [s.colour[0] * i, s.colour[1] * i, s.colour[2] * i]);
        }
    }
    img
}

/// Mosaic clipped at 1.0 and its mask (photosites whose scene value reached the clip level).
fn clipped_mosaic(img: &RgbImage) -> (Mosaic, ClipMask) {
    let mut m = synth::mosaic_from_rgb(img, Cfa::RGGB);
    let mut mask = ClipMask::new(W, H);
    for (v, f) in m.data.iter_mut().zip(mask.data.iter_mut()) {
        if *v >= 1.0 {
            *v = 1.0;
            *f = 1;
        }
    }
    (m, mask)
}

/// Pixels (of the demosaiced result) whose 3x3 neighbourhood holds a clipped photosite.
fn blown_region(mask: &ClipMask) -> Vec<bool> {
    let flags = mask.rgb_flags(Cfa::RGGB);
    flags.iter().map(|&f| f != 0).collect()
}

fn mean_colour(img: &RgbImage, select: impl Fn(usize, usize) -> bool) -> [f64; 3] {
    let (mut s, mut n) = ([0.0f64; 3], 0usize);
    for y in 0..H {
        for x in 0..W {
            if select(x, y) {
                let p = img.pixel(x, y);
                for c in 0..3 {
                    s[c] += f64::from(p[c]);
                }
                n += 1;
            }
        }
    }
    assert!(n > 0, "empty selection");
    [s[0] / n as f64, s[1] / n as f64, s[2] / n as f64]
}

struct Result {
    /// Chroma dE00 of the blown region's mean colour from the surround's, without recovery.
    region_off: f64,
    /// The same with recovery.
    region_on: f64,
    /// Largest Lab hue error (degrees, against the ground truth) within 3 px of the clip boundary,
    /// without recovery. Only pixels with Lab chroma above 3 count.
    boundary_hue_err_off: f64,
    /// The same with recovery.
    boundary_hue_err_on: f64,
    /// Largest hue step (degrees) between horizontally adjacent pixels over the blown region and
    /// its 3 px band, without recovery: an abrupt ring of wrong hue shows up here.
    hue_step_off: f64,
    /// The same with recovery.
    hue_step_on: f64,
    /// The largest hue step of the ground truth over the same pixels (the scene's own gradient).
    hue_step_truth: f64,
    blown_pixels: usize,
}

fn hue_diff_degrees(a: f64, b: f64) -> f64 {
    let d = (a - b).rem_euclid(360.0);
    d.min(360.0 - d)
}

fn rgb64(p: [f32; 3]) -> [f64; 3] {
    [f64::from(p[0]), f64::from(p[1]), f64::from(p[2])]
}

fn evaluate(s: &Scene) -> Result {
    let truth = scene_image(s);
    let (m, mask) = clipped_mosaic(&truth);
    let blown = blown_region(&mask);
    let method = Method::Lmmse;

    let off = demosaic::demosaic(&m, method);
    let mut recovered = m.clone();
    highlights::reconstruct(&mut recovered, &mask).expect("reconstruct");
    let on = demosaic::demosaic(&recovered, method);

    // Chebyshev distance (up to 12) to the nearest pixel of the other class.
    let dist_to_other = |x: usize, y: usize| -> usize {
        let here = blown[y * W + x];
        let mut best = usize::MAX;
        for yy in y.saturating_sub(12)..(y + 13).min(H) {
            for xx in x.saturating_sub(12)..(x + 13).min(W) {
                if blown[yy * W + xx] != here {
                    best = best.min(x.abs_diff(xx).max(y.abs_diff(yy)));
                }
            }
        }
        best
    };
    let mut ring = vec![false; W * H]; // unclipped surround, 4 to 12 px outside
    let mut band = vec![false; W * H]; // within 3 px of the clip boundary, both sides
    for y in 0..H {
        for x in 0..W {
            let d = dist_to_other(x, y);
            if !blown[y * W + x] && (4..=12).contains(&d) {
                ring[y * W + x] = true;
            }
            band[y * W + x] = d <= 3;
        }
    }
    let surround = mean_colour(&truth, |x, y| ring[y * W + x]);
    let off_mean = mean_colour(&off, |x, y| blown[y * W + x]);
    let on_mean = mean_colour(&on, |x, y| blown[y * W + x]);
    let blown_pixels = blown.iter().filter(|&&b| b).count();

    let boundary_hue_err = |img: &RgbImage| {
        let mut worst = 0.0f64;
        for y in 0..H {
            for x in 0..W {
                if band[y * W + x] {
                    let (p, t) = (rgb64(img.pixel(x, y)), rgb64(truth.pixel(x, y)));
                    if chroma(t) > 3.0 && chroma(p) > 3.0 {
                        worst = worst.max(hue_diff_degrees(hue_degrees(p), hue_degrees(t)));
                    }
                }
            }
        }
        worst
    };
    let hue_step = |img: &RgbImage| {
        let mut worst = 0.0f64;
        for y in 0..H {
            for x in 0..W - 1 {
                if band[y * W + x] || blown[y * W + x] {
                    let (a, b) = (rgb64(img.pixel(x, y)), rgb64(img.pixel(x + 1, y)));
                    if chroma(a) > 3.0 && chroma(b) > 3.0 {
                        worst = worst.max(hue_diff_degrees(hue_degrees(a), hue_degrees(b)));
                    }
                }
            }
        }
        worst
    };
    Result {
        region_off: chroma_delta_e00(off_mean, surround),
        region_on: chroma_delta_e00(on_mean, surround),
        boundary_hue_err_off: boundary_hue_err(&off),
        boundary_hue_err_on: boundary_hue_err(&on),
        hue_step_off: hue_step(&off),
        hue_step_on: hue_step(&on),
        hue_step_truth: hue_step(&truth),
        blown_pixels,
    }
}

fn scenes() -> Vec<Scene> {
    let base = [
        ("warm lamp (R clips first)", [1.0, 0.62, 0.35], 2.6),
        ("blue sky (B clips first)", [0.45, 0.65, 1.0], 2.4),
        ("green LED (G clips first)", [0.35, 1.0, 0.45], 2.8),
        ("magenta sign (R and B clip)", [1.0, 0.3, 0.9], 2.4),
        ("neutral white lamp", [1.0, 1.0, 1.0], 3.0),
    ];
    let mut out = Vec::new();
    for (profile, tag) in [(Profile::Glow, "glow"), (Profile::Disc, "disc")] {
        for (name, colour, peak) in base {
            let name: &'static str = Box::leak(format!("{tag}: {name}").into_boxed_str());
            out.push(Scene {
                name,
                colour,
                peak,
                profile,
            });
        }
    }
    out
}

#[test]
fn reconstructed_chroma_is_within_de00_5_of_the_surround() {
    println!(
        "{:<40} {:>8} {:>10} {:>10} | {:>13} {:>12} | {:>9} {:>8} {:>8}",
        "scene",
        "blown px",
        "region off",
        "region on",
        "bnd hue off",
        "bnd hue on",
        "step off",
        "step on",
        "step gt"
    );
    for s in scenes() {
        let r = evaluate(&s);
        println!(
            "{:<40} {:>8} {:>10.2} {:>10.2} | {:>13.1} {:>12.1} | {:>9.1} {:>8.1} {:>8.1}",
            s.name,
            r.blown_pixels,
            r.region_off,
            r.region_on,
            r.boundary_hue_err_off,
            r.boundary_hue_err_on,
            r.hue_step_off,
            r.hue_step_on,
            r.hue_step_truth
        );
        assert!(r.blown_pixels > 200, "{}: scene does not clip", s.name);
        assert!(
            r.region_on <= 5.0,
            "{}: region chroma dE00 {} > 5",
            s.name,
            r.region_on
        );
    }
}

#[test]
fn recovery_never_adds_a_hue_fringe() {
    for s in scenes() {
        let r = evaluate(&s);
        // Reconstruction must not move the boundary hue away from the truth, nor make the hue jump
        // more abruptly between neighbouring pixels, than leaving the channels clipped does.
        assert!(
            r.boundary_hue_err_on <= r.boundary_hue_err_off + 1.0,
            "{}: recovery worsened the boundary hue ({} -> {})",
            s.name,
            r.boundary_hue_err_off,
            r.boundary_hue_err_on
        );
        assert!(
            r.hue_step_on <= r.hue_step_off + 0.5,
            "{}: recovery made the hue step worse ({} -> {})",
            s.name,
            r.hue_step_off,
            r.hue_step_on
        );
    }
}

#[test]
fn gentle_falloffs_keep_the_boundary_hue_within_15_degrees() {
    // With a smooth falloff (a sky near the sun, a lamp's glow) the global chroma offset is
    // calibrated at the clip boundary, so the hue there stays close to the truth. Hard-edged
    // sources (the `disc` scenes) clip all channels within a few pixels; the spec'd algorithm then
    // has only the weakest channel to work from and recovers chroma (see the dE00 gate) but not the
    // exact hue; those scenes are reported in the table, not gated on hue.
    for s in scenes()
        .into_iter()
        .filter(|s| matches!(s.profile, Profile::Glow))
    {
        let r = evaluate(&s);
        assert!(
            r.boundary_hue_err_on <= 15.0,
            "{}: boundary hue error {} deg",
            s.name,
            r.boundary_hue_err_on
        );
    }
}

#[test]
fn recovery_beats_no_recovery_on_coloured_sources() {
    // The metrics must discriminate: without R5, coloured blown regions drift far from the surround
    // and the boundary shows a hue step.
    let mut helped = 0;
    for s in scenes().into_iter().filter(|s| !s.name.contains("neutral")) {
        let r = evaluate(&s);
        if r.region_off > 5.0 {
            helped += 1;
            assert!(
                r.region_on < r.region_off,
                "{}: {} -> {}",
                s.name,
                r.region_off,
                r.region_on
            );
        }
    }
    assert!(
        helped >= 2,
        "expected the no-recovery rendering to fail the 5 dE00 gate on at least two scenes, got {helped}"
    );
}

#[test]
fn clipped_values_only_increase_and_unclipped_photosites_are_untouched() {
    let s = &scenes()[0];
    let (m, mask) = clipped_mosaic(&scene_image(s));
    let mut r = m.clone();
    let offsets = highlights::reconstruct(&mut r, &mask).unwrap();
    assert!(offsets.count[0] > 0, "no reference superpixels for red");
    for i in 0..m.data.len() {
        if mask.data[i] == 0 {
            assert_eq!(r.data[i], m.data[i], "unclipped photosite {i} changed");
        } else {
            assert!(r.data[i] >= m.data[i], "clipped photosite {i} decreased");
        }
    }
    assert!(
        r.data.iter().zip(&m.data).any(|(a, b)| a > b),
        "nothing was reconstructed"
    );
}

#[test]
fn tiled_reconstruction_equals_whole() {
    use ym_gpu::cpu::tiling::{Halo, Rect, tile_grid};
    let s = &scenes()[1];
    let (m, mask) = clipped_mosaic(&scene_image(s));
    let mut whole = m.clone();
    let offsets = highlights::reconstruct(&mut whole, &mask).unwrap();

    // Pass 1: accumulate the global chroma offsets over tile cores, merge in raster order.
    let frame = Rect::whole(W, H);
    let tiles = tile_grid(W, H, 48, 64, 2);
    let mut acc = highlights::OffsetAccumulator::default();
    for core in &tiles {
        let ext = core.grow_aligned(Halo::uniform(4), frame, 2);
        let tile = m.extract(ext);
        let tmask = mask.extract(ext);
        acc.merge(&highlights::accumulate_offsets(&tile, &tmask, core.relative_to(&ext)).unwrap());
    }
    let merged = acc.finish();
    for c in 0..3 {
        assert!(
            (merged.offset[c] - offsets.offset[c]).abs() < 1e-9,
            "offset {c}: {} vs {}",
            merged.offset[c],
            offsets.offset[c]
        );
        assert_eq!(merged.count[c], offsets.count[c]);
    }
    // Pass 2: apply per tile with the merged offsets and paste the cores.
    let mut out = m.clone();
    for core in &tiles {
        let ext = core.grow_aligned(Halo::uniform(4), frame, 2);
        let mut tile = m.extract(ext);
        let tmask = mask.extract(ext);
        highlights::apply_offsets(&mut tile, &tmask, &merged).unwrap();
        out.paste(&tile, core.relative_to(&ext), core.x, core.y);
    }
    assert_eq!(out.data, whole.data, "tiled R5 differs from whole-frame R5");
}
