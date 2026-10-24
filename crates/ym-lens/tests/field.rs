//! L2, field optics (`docs/21` section 4): falloff, kernels, lateral colour, side softness, tiles.

mod common;

use common::{
    apply_textured, camera, frame, library, noise, plan, planes_for, region, tiles, whole,
};
use ym_lens::{
    Axis, ImageRef, LensAmounts, LensError, LensKind, LensPlan, LensPlanes, PixelRect, apply,
    halo_px,
};

/// The falloff gain of a pixel, from the public lens data: `1 - e' (1 - T(rho'))`.
fn expected_gain(
    kind: LensKind,
    edges: f64,
    anamorphic: f64,
    axis: Axis,
    (fw, fh): (usize, usize),
    (px, py): (usize, usize),
) -> f64 {
    let lens = library().get(kind).unwrap();
    let att = library().attachment();
    let extra = (att.q_max - 1.0) * anamorphic / 100.0;
    let e = edges / 100.0;
    let e_field = e + (1.0 - e) * att.field.kappa * extra;
    let alpha = 1.0 + att.field.kappa * extra;
    let h = 0.5 * ((fw * fw + fh * fh) as f64).sqrt();
    let u = (px as f64 + 0.5 - 0.5 * fw as f64) / h;
    let v = (py as f64 + 0.5 - 0.5 * fh as f64) / h;
    let (xa, ya) = if axis == Axis::X { (u, v) } else { (v, u) };
    let rho = ((alpha * xa).powi(2) + ya * ya).sqrt().min(1.0);
    let r = &lens.field.radii;
    let s = (0..8).find(|s| rho <= r[s + 1]).unwrap_or(7);
    let t = (rho - r[s]) / (r[s + 1] - r[s]);
    let illum = (1.0 - t) * lens.field.relative_illumination[s]
        + t * lens.field.relative_illumination[s + 1];
    1.0 - e_field * (1.0 - illum)
}

#[test]
fn a_flat_field_comes_out_as_the_falloff_gain() {
    let flat = [0.3f32, 0.6, 0.9];
    for (axis, (w, h)) in [(Axis::X, (600, 400)), (Axis::Y, (400, 600))] {
        let mut data = Vec::new();
        for v in flat {
            data.extend(std::iter::repeat_n(v, w * h));
        }
        for kind in LensKind::ALL {
            for (edges, anamorphic) in [
                (100.0, 0.0),
                (45.0, 0.0),
                (100.0, 100.0),
                (0.0, 100.0),
                (30.0, 55.0),
            ] {
                let p = plan(kind, 0.0, edges, anamorphic, w, h, axis);
                let planes = planes_for(&p);
                let out = apply(
                    &p,
                    &planes,
                    1,
                    &whole(w, h, &data),
                    PixelRect::new(0, 0, w, h),
                )
                .unwrap();
                let mut worst = 0.0f64;
                for y in (0..h).step_by(7) {
                    for x in (0..w).step_by(5) {
                        let g = expected_gain(kind, edges, anamorphic, axis, (w, h), (x, y));
                        for (c, v) in flat.iter().enumerate() {
                            let got = f64::from(out[c * w * h + y * w + x]);
                            let want = g * f64::from(*v);
                            worst = worst.max(((got - want) / want).abs());
                        }
                    }
                }
                assert!(
                    worst < 1e-6,
                    "{kind} edges {edges} anamorphic {anamorphic} {axis:?}: worst relative error {worst:e}"
                );
            }
        }
    }
}

#[test]
fn the_gain_is_one_at_the_centre_and_falls_towards_the_corners() {
    let (w, h) = (600, 400);
    let mut data = vec![0.5f32; 3 * w * h];
    data[0] = 0.5;
    let p = plan(LensKind::Petzval, 0.0, 100.0, 0.0, w, h, Axis::X);
    let planes = planes_for(&p);
    let out = apply(
        &p,
        &planes,
        1,
        &whole(w, h, &data),
        PixelRect::new(0, 0, w, h),
    )
    .unwrap();
    let at = |x: usize, y: usize| out[y * w + x] / 0.5;
    assert!((at(300, 200) - 1.0).abs() < 1e-3);
    assert!(at(300, 200) > at(100, 100) && at(100, 100) > at(0, 0));
    assert!(
        (at(0, 0) - 0.45).abs() < 0.02,
        "Petzval's corner gain is 1 - 0.55 = 0.45, found {}",
        at(0, 0)
    );
}

#[test]
fn edges_zero_with_anamorphic_zero_changes_nothing() {
    let (w, h) = (300, 200);
    let data = noise(w, h, 3);
    for kind in LensKind::ALL {
        let p = plan(kind, 70.0, 0.0, 0.0, w, h, Axis::X);
        let planes = planes_for(&p);
        assert_eq!(halo_px(&p, &planes, 1), 0);
        assert_eq!(p.field_amount(), 0.0);
        let out = apply(
            &p,
            &planes,
            1,
            &whole(w, h, &data),
            PixelRect::new(0, 0, w, h),
        )
        .unwrap();
        assert_eq!(out, data);
    }
}

#[test]
fn the_attachment_acts_alone_at_edges_zero() {
    // Anamorphic above 0 with Edges at 0: the oval falloff, the side softness and the fringe still act.
    let (w, h) = (600, 400);
    let data = vec![0.5f32; 3 * w * h];
    let p = plan(LensKind::Default, 0.0, 0.0, 100.0, w, h, Axis::X);
    assert!(p.field_amount() > 0.1 && p.field_amount() < 0.2);
    let planes = planes_for(&p);
    assert!(halo_px(&p, &planes, 1) > 0);
    let out = apply(
        &p,
        &planes,
        1,
        &whole(w, h, &data),
        PixelRect::new(0, 0, w, h),
    )
    .unwrap();
    let at = |x: usize, y: usize| f64::from(out[y * w + x]) / 0.5;
    assert!(
        (at(300, 200) - expected_gain(LensKind::Default, 0.0, 100.0, Axis::X, (w, h), (300, 200)))
            .abs()
            < 1e-6
    );
    // Darker towards the left and right edges than towards the top and bottom at the same distance.
    assert!(
        at(10, 200) < at(300, 10) - 0.01,
        "{} vs {}",
        at(10, 200),
        at(300, 10)
    );
}

/// The centroid of one channel of a lens's table at radius index `r`, in table units (x radial, y tangential),
/// computed from the public data with the camera's weights.
fn table_centroid(kind: LensKind, r: usize, c: usize) -> [f64; 2] {
    let lens = library().get(kind).unwrap();
    let w = camera().weights()[c];
    let n = 33;
    let centre = |k: usize| -1.0 + (2 * k + 1) as f64 / n as f64;
    let mut total = 0.0;
    let (mut cx, mut cy) = (0.0, 0.0);
    for (j, wj) in w.iter().enumerate() {
        for (i, v) in lens.psf(r, j).iter().enumerate() {
            let m = wj * f64::from(*v);
            total += m;
            cx += m * centre(i % n);
            cy += m * centre(i / n);
        }
    }
    [cx / total, cy / total]
}

#[test]
fn lateral_colour_shifts_each_channel_by_its_table_centroid() {
    // An affine image is read back at the kernel's centroid: out(p) = gain * I(p + centroid). Pixels of a large
    // frame, cut out as a small tile, have kernels of 15 to 40 px.
    let (fw, fh) = (6000usize, 4000usize);
    let kind = LensKind::Vintage;
    let p = plan(kind, 0.0, 100.0, 0.0, fw, fh, Axis::X);
    let planes = planes_for(&p);
    let slope = [
        [1.0 / 6000.0, 0.5 / 4000.0],
        [0.8 / 6000.0, 1.0 / 4000.0],
        [0.6 / 6000.0, 0.7 / 4000.0],
    ];
    let value =
        |c: usize, x: f64, y: f64| slope[c][0] * x + slope[c][1] * y + 0.1 * (c as f64 + 1.0);
    let lens = library().get(kind).unwrap();
    let h = 0.5 * ((fw * fw + fh * fh) as f64).sqrt();
    for (px, py) in [
        (5200usize, 800usize),
        (4600, 3300),
        (900, 700),
        (1300, 3300),
    ] {
        let out_rect = PixelRect::new(px, py, 1, 1);
        let need = out_rect.grown(halo_px(&p, &planes, 1), fw, fh);
        let mut buf = Vec::new();
        for c in 0..3 {
            for y in need.y..need.y + need.h {
                for x in need.x..need.x + need.w {
                    buf.push(value(c, x as f64 + 0.5, y as f64 + 0.5) as f32);
                }
            }
        }
        let input = ImageRef {
            width: need.w,
            height: need.h,
            x0: need.x,
            y0: need.y,
            frame_width: fw,
            frame_height: fh,
            data: &buf,
        };
        let out = apply(&p, &planes, 1, &input, out_rect).unwrap();

        // The kernel at this pixel, from the public data.
        let u = (px as f64 + 0.5 - 3000.0) / h;
        let v = (py as f64 + 0.5 - 2000.0) / h;
        let rho = (u * u + v * v).sqrt();
        let s = (0..8)
            .find(|s| rho <= lens.field.radii[*s + 1])
            .unwrap_or(7);
        let t = (rho - lens.field.radii[s]) / (lens.field.radii[s + 1] - lens.field.radii[s]);
        let blend = |a: f64, b: f64| (1.0 - t) * a + t * b;
        let omega = (blend(
            lens.field.psf_half_width[s],
            lens.field.psf_half_width[s + 1],
        ))
        .min(0.012)
            * h;
        let rr = (u * u + v * v).sqrt();
        let (cs, sn) = (u / rr, v / rr);
        let gain = 1.0
            - (1.0
                - blend(
                    lens.field.relative_illumination[s],
                    lens.field.relative_illumination[s + 1],
                ));
        assert!(omega > 12.0, "pixel ({px},{py}) has a kernel of {omega} px");
        let mut shifts = Vec::new();
        for c in 0..3 {
            let (ca, cb) = (table_centroid(kind, s, c), table_centroid(kind, s + 1, c));
            let (cx, cy) = (blend(ca[0], cb[0]), blend(ca[1], cb[1]));
            // Table x is radial (cs, sn), table y is tangential (sn, -cs).
            let (dx, dy) = (omega * (cx * cs + cy * sn), omega * (cx * sn - cy * cs));
            let want = gain * value(c, px as f64 + 0.5 + dx, py as f64 + 0.5 + dy);
            let got = f64::from(out[c]);
            // The ramp's slope is below 1.5e-4 per pixel, so a centroid error of e px is an error of 1.5e-4 e.
            // The error in pixels, to the ramp's mean slope per pixel (the resampling to 17 x 17 cells moves a
            // centroid by under 0.003 px here).
            let err_px = (got - want).abs() / (slope[c][0].hypot(slope[c][1]) / 2f64.sqrt());
            assert!(
                err_px < 0.02,
                "pixel ({px},{py}) channel {c}: {got} vs {want}"
            );
            shifts.push((dx, dy));
        }
        // The red and blue shifts are really apart (the test would pass trivially on a neutral camera).
        let apart =
            ((shifts[0].0 - shifts[2].0).powi(2) + (shifts[0].1 - shifts[2].1).powi(2)).sqrt();
        assert!(
            apart > 0.5,
            "red and blue are {apart} px apart at ({px},{py})"
        );
    }
}

/// Where the lens data put the centroid of each channel's kernel at pixel `(px, py)` (no attachment), in pixels
/// from the pixel centre, and the kernel's half-width: computed from the public data only.
fn expected_kernel(
    kind: LensKind,
    (fw, fh): (usize, usize),
    (px, py): (usize, usize),
) -> (f64, [[f64; 2]; 3]) {
    let lens = library().get(kind).unwrap();
    let h = 0.5 * ((fw * fw + fh * fh) as f64).sqrt();
    let u = (px as f64 + 0.5 - 0.5 * fw as f64) / h;
    let v = (py as f64 + 0.5 - 0.5 * fh as f64) / h;
    let rho = (u * u + v * v).sqrt();
    let s = (0..8)
        .find(|s| rho <= lens.field.radii[*s + 1])
        .unwrap_or(7);
    let t = (rho - lens.field.radii[s]) / (lens.field.radii[s + 1] - lens.field.radii[s]);
    let blend = |a: f64, b: f64| (1.0 - t) * a + t * b;
    let omega = blend(
        lens.field.psf_half_width[s],
        lens.field.psf_half_width[s + 1],
    )
    .min(0.012)
        * h;
    let (cs, sn) = (u / rho, v / rho);
    let shifts = std::array::from_fn(|c| {
        let (ca, cb) = (table_centroid(kind, s, c), table_centroid(kind, s + 1, c));
        let (cx, cy) = (blend(ca[0], cb[0]), blend(ca[1], cb[1]));
        // Table x is radial (cs, sn), table y is tangential (sn, -cs).
        [omega * (cx * cs + cy * sn), omega * (cx * sn - cy * cs)]
    });
    (omega, shifts)
}

#[test]
fn the_response_at_a_pixel_to_every_input_pixel_is_its_kernel() {
    // The kernel of one output pixel, measured by feeding unit impulses one by one: it sums to 1 (before the
    // gain) and has the centroid of the projected table, turned to the pixel's direction and scaled to its width.
    let (fw, fh) = (1500usize, 1000usize);
    let kind = LensKind::Petzval;
    let p = plan(kind, 0.0, 100.0, 0.0, fw, fh, Axis::X);
    let planes = planes_for(&p);
    let halo = halo_px(&p, &planes, 1);
    for (px, py) in [(900usize, 520usize), (1150, 640), (1400, 900), (300, 250)] {
        let (omega, shifts) = expected_kernel(kind, (fw, fh), (px, py));
        assert!(omega > 1.0, "pixel ({px},{py}) has a kernel of {omega} px");
        let gain = expected_gain(kind, 100.0, 0.0, Axis::X, (fw, fh), (px, py));
        let reach = (omega * 2f64.sqrt()).ceil() as usize + 2;
        let out_rect = PixelRect::new(px, py, 1, 1);
        let need = out_rect.grown(halo, fw, fh);
        let mut sums = [0.0f64; 3];
        let mut moments = [[0.0f64; 2]; 3];
        for y in py - reach..=py + reach {
            for x in px - reach..=px + reach {
                let mut buf = vec![0.0f32; 3 * need.w * need.h];
                for c in 0..3 {
                    buf[c * need.w * need.h + (y - need.y) * need.w + (x - need.x)] = 1.0;
                }
                let input = ImageRef {
                    width: need.w,
                    height: need.h,
                    x0: need.x,
                    y0: need.y,
                    frame_width: fw,
                    frame_height: fh,
                    data: &buf,
                };
                let out = apply(&p, &planes, 1, &input, out_rect).unwrap();
                for c in 0..3 {
                    // Channels are independent: every channel got the same impulse.
                    let k = f64::from(out[c]) / gain;
                    sums[c] += k;
                    moments[c][0] += k * (x as f64 - px as f64);
                    moments[c][1] += k * (y as f64 - py as f64);
                }
            }
        }
        for c in 0..3 {
            assert!(
                (sums[c] - 1.0).abs() < 1e-6,
                "({px},{py}) channel {c}: the kernel sums to {}",
                sums[c]
            );
            let (mx, my) = (moments[c][0] / sums[c], moments[c][1] / sums[c]);
            assert!(
                (mx - shifts[c][0]).abs() < 0.05 && (my - shifts[c][1]).abs() < 0.05,
                "({px},{py}) channel {c}: centroid ({mx:.3}, {my:.3}) against ({:.3}, {:.3})",
                shifts[c][0],
                shifts[c][1]
            );
        }
    }
}

/// The spread (RMS distance from the centroid) of the green response to a unit impulse at `(x, y)`.
fn impulse_spread(
    p: &LensPlan,
    planes: &LensPlans,
    (fw, fh): (usize, usize),
    (x, y): (usize, usize),
) -> f64 {
    let half = 40usize;
    let win = PixelRect::new(x - half, y - half, 2 * half + 1, 2 * half + 1);
    let halo = halo_px(p, planes, 1);
    let need = win.grown(halo, fw, fh);
    let mut buf = vec![0.0f32; 3 * need.w * need.h];
    buf[need.w * need.h + (y - need.y) * need.w + (x - need.x)] = 1.0;
    let input = ImageRef {
        width: need.w,
        height: need.h,
        x0: need.x,
        y0: need.y,
        frame_width: fw,
        frame_height: fh,
        data: &buf,
    };
    let out = apply(p, planes, 1, &input, win).unwrap();
    let g = &out[win.w * win.h..2 * win.w * win.h];
    let total: f64 = g.iter().map(|v| f64::from(*v)).sum();
    let (mut mx, mut my) = (0.0, 0.0);
    for (i, v) in g.iter().enumerate() {
        mx += f64::from(*v) * (i % win.w) as f64;
        my += f64::from(*v) * (i / win.w) as f64;
    }
    let (mx, my) = (mx / total, my / total);
    let mut var = 0.0;
    for (i, v) in g.iter().enumerate() {
        var +=
            f64::from(*v) * (((i % win.w) as f64 - mx).powi(2) + ((i / win.w) as f64 - my).powi(2));
    }
    (var / total).sqrt()
}

type LensPlans = LensPlanes;

#[test]
fn anamorphic_side_softness_widens_the_left_and_right_more_than_the_top_and_bottom() {
    let (fw, fh) = (6000, 4000);
    let h = 0.5 * ((fw * fw + fh * fh) as f64).sqrt();
    // Same round field radius 0.5 on the left and at the top.
    let left = (3000 - (0.5 * h) as usize, 2000);
    let top = (3000, 2000 - (0.5 * h) as usize);
    for kind in [LensKind::Default, LensKind::Triplet] {
        let p = plan(kind, 0.0, 0.0, 100.0, fw, fh, Axis::X);
        let planes = planes_for(&p);
        let (sl, st) = (
            impulse_spread(&p, &planes, (fw, fh), left),
            impulse_spread(&p, &planes, (fw, fh), top),
        );
        assert!(sl > 1.15 * st, "{kind}: left {sl}, top {st}");
        // Without the attachment the lens is the same in both directions (the PSF is turned to the radial).
        let p = plan(kind, 0.0, 100.0, 0.0, fw, fh, Axis::X);
        let planes = planes_for(&p);
        let (sl, st) = (
            impulse_spread(&p, &planes, (fw, fh), left),
            impulse_spread(&p, &planes, (fw, fh), top),
        );
        assert!((sl / st - 1.0).abs() < 0.05, "{kind}: left {sl}, top {st}");
    }
    // The axis is the buffer's y when the frame says so: the roles swap.
    let p = plan(LensKind::Default, 0.0, 0.0, 100.0, fw, fh, Axis::Y);
    let planes = planes_for(&p);
    let (sl, st) = (
        impulse_spread(&p, &planes, (fw, fh), left),
        impulse_spread(&p, &planes, (fw, fh), top),
    );
    assert!(st > 1.1 * sl, "axis Y: left {sl}, top {st}");
}

#[test]
fn an_impulse_keeps_its_energy() {
    let (fw, fh) = (1500, 1000);
    let p = plan(LensKind::Sonnar, 0.0, 100.0, 0.0, fw, fh, Axis::X);
    let planes = planes_for(&p);
    for (x, y) in [(750usize, 500usize), (1100, 300), (200, 800)] {
        let half = 30usize;
        let win = PixelRect::new(x - half, y - half, 2 * half + 1, 2 * half + 1);
        let need = win.grown(halo_px(&p, &planes, 1), fw, fh);
        let mut buf = vec![0.0f32; 3 * need.w * need.h];
        buf[(y - need.y) * need.w + (x - need.x)] = 1.0;
        let input = ImageRef {
            width: need.w,
            height: need.h,
            x0: need.x,
            y0: need.y,
            frame_width: fw,
            frame_height: fh,
            data: &buf,
        };
        let out = apply(&p, &planes, 1, &input, win).unwrap();
        let total: f64 = out[..win.w * win.h].iter().map(|v| f64::from(*v)).sum();
        // The falloff gain (about 1 to 0.7) times a kernel mass of 1, spread by a slowly changing kernel.
        let gain = expected_gain(LensKind::Sonnar, 100.0, 0.0, Axis::X, (fw, fh), (x, y));
        assert!(
            (total / gain - 1.0).abs() < 0.05,
            "({x},{y}): {total} vs gain {gain}"
        );
    }
}

#[test]
fn tiles_equal_the_whole_frame_bit_for_bit() {
    for (axis, k, (fw, fh), full) in [
        (Axis::X, 1usize, (600usize, 400usize), (600usize, 400usize)),
        (Axis::Y, 1, (400, 600), (400, 600)),
        (Axis::X, 4, (600, 400), (2400, 1600)),
        (Axis::Y, 4, (400, 600), (1600, 2400)),
    ] {
        for (kind, (flares, edges, anamorphic)) in [
            (LensKind::Petzval, (40.0, 100.0, 100.0)),
            (LensKind::Detuned, (0.0, 70.0, 0.0)),
            (LensKind::Default, (0.0, 0.0, 60.0)),
        ] {
            let p = plan(kind, flares, edges, anamorphic, full.0, full.1, axis);
            let planes = planes_for(&p);
            let all = PixelRect::new(0, 0, fw, fh);
            let reference = apply_textured(&p, &planes, k, fw, fh, all);
            for (nx, ny) in [(2, 2), (3, 3)] {
                for t in tiles(fw, fh, nx, ny) {
                    let got = apply_textured(&p, &planes, k, fw, fh, t);
                    for c in 0..3 {
                        for y in 0..t.h {
                            let want = &reference[c * fw * fh + (t.y + y) * fw + t.x..][..t.w];
                            assert_eq!(
                                &got[c * t.w * t.h + y * t.w..][..t.w],
                                want,
                                "{kind} {axis:?} k={k} {nx}x{ny} tile {t:?} channel {c} row {y}"
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn tiles_equal_the_region_when_the_kernels_read_reduced_levels() {
    // A 3000 x 2000 frame has kernels of up to 21 px; they read the 2x and 4x copies. The region at the frame's
    // corner touches the edge, where the input is replicated.
    let (fw, fh) = (3000usize, 2000usize);
    let p = plan(LensKind::Petzval, 0.0, 100.0, 100.0, fw, fh, Axis::X);
    let planes = planes_for(&p);
    assert!(halo_px(&p, &planes, 1) > 30);
    let r = PixelRect::new(2300, 0, 700, 500);
    let reference = apply_textured(&p, &planes, 1, fw, fh, r);
    for (nx, ny) in [(2, 2), (3, 3)] {
        for t in tiles(r.w, r.h, nx, ny) {
            let abs = PixelRect::new(r.x + t.x, r.y + t.y, t.w, t.h);
            let got = apply_textured(&p, &planes, 1, fw, fh, abs);
            for c in 0..3 {
                for y in 0..t.h {
                    let want = &reference[c * r.w * r.h + (t.y + y) * r.w + t.x..][..t.w];
                    assert_eq!(
                        &got[c * t.w * t.h + y * t.w..][..t.w],
                        want,
                        "{nx}x{ny} tile {t:?} channel {c} row {y}"
                    );
                }
            }
        }
    }
}

#[test]
fn two_runs_and_any_thread_count_give_the_same_bits() {
    let (w, h) = (400usize, 300usize);
    let p = plan(LensKind::Triplet, 30.0, 90.0, 80.0, w, h, Axis::X);
    let planes = planes_for(&p);
    let data = noise(w, h, 11);
    let run = || {
        apply(
            &p,
            &planes,
            1,
            &whole(w, h, &data),
            PixelRect::new(0, 0, w, h),
        )
        .unwrap()
    };
    let a = run();
    assert_eq!(a, run());
    for threads in [1usize, 3] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        assert_eq!(a, pool.install(run), "{threads} threads");
    }
}

#[test]
fn an_input_without_the_halo_is_refused() {
    let (w, h) = (600usize, 400usize);
    let p = plan(LensKind::Vintage, 0.0, 100.0, 0.0, w, h, Axis::X);
    let planes = planes_for(&p);
    let halo = halo_px(&p, &planes, 1);
    assert!(halo > 0);
    let out = PixelRect::new(200, 100, 100, 100);
    let tight = region(out.grown(halo - 1, w, h));
    let rect = out.grown(halo - 1, w, h);
    let input = ImageRef {
        width: rect.w,
        height: rect.h,
        x0: rect.x,
        y0: rect.y,
        frame_width: w,
        frame_height: h,
        data: &tight,
    };
    assert!(matches!(
        apply(&p, &planes, 1, &input, out),
        Err(LensError::InputTooSmall { .. })
    ));
    // An input that stops at the frame edge is enough there.
    let edge = PixelRect::new(0, 0, 50, 50);
    let rect = edge.grown(halo, w, h);
    let buf = region(rect);
    let input = ImageRef {
        width: rect.w,
        height: rect.h,
        x0: 0,
        y0: 0,
        frame_width: w,
        frame_height: h,
        data: &buf,
    };
    assert!(apply(&p, &planes, 1, &input, edge).is_ok());
}

#[test]
fn mismatched_plans_planes_and_frames_are_refused() {
    let (w, h) = (600usize, 400usize);
    let p = plan(LensKind::Vintage, 0.0, 100.0, 0.0, w, h, Axis::X);
    let planes = planes_for(&p);
    let data = noise(w, h, 2);
    let all = PixelRect::new(0, 0, w, h);
    // Planes of another lens.
    let other = plan(LensKind::Petzval, 0.0, 100.0, 0.0, w, h, Axis::X);
    assert!(matches!(
        apply(&p, &planes_for(&other), 1, &whole(w, h, &data), all),
        Err(LensError::Frame(_))
    ));
    // An input whose frame is not the plan's frame at that scale.
    assert!(matches!(
        apply(&p, &planes, 2, &whole(w, h, &data), all),
        Err(LensError::Frame(_))
    ));
    assert!(matches!(
        apply(&p, &planes, 0, &whole(w, h, &data), all),
        Err(LensError::Frame(_))
    ));
    // An output outside the frame.
    assert!(
        apply(
            &p,
            &planes,
            1,
            &whole(w, h, &data),
            PixelRect::new(500, 300, 200, 200)
        )
        .is_err()
    );
    // Data of the wrong length.
    assert!(apply(&p, &planes, 1, &whole(w, h, &data[..100]), all).is_err());
    let _ = (frame(w, h, Axis::X), LensAmounts::default());
}

/// Where an affine ramp is read back by the pass at pixel `(px, py)` of a `fw x fh` frame: the shift of each
/// channel against the pixel centre, along x, in pixels.
fn ramp_shifts(p: &LensPlan, (fw, fh): (usize, usize), (px, py): (usize, usize)) -> [f64; 3] {
    let planes = planes_for(p);
    let slope = 1.0 / 6000.0;
    let value = |c: usize, x: f64| slope * x * (1.0 + 0.1 * c as f64) + 0.2;
    let out_rect = PixelRect::new(px, py, 1, 1);
    let need = out_rect.grown(halo_px(p, &planes, 1), fw, fh);
    let mut buf = Vec::new();
    for c in 0..3 {
        for _y in need.y..need.y + need.h {
            for x in need.x..need.x + need.w {
                buf.push(value(c, x as f64 + 0.5) as f32);
            }
        }
    }
    let input = ImageRef {
        width: need.w,
        height: need.h,
        x0: need.x,
        y0: need.y,
        frame_width: fw,
        frame_height: fh,
        data: &buf,
    };
    let out = apply(p, &planes, 1, &input, out_rect).unwrap();
    let gain = expected_gain(
        p.kind(),
        p.edges() * 100.0,
        (p.q() - 1.0) / (library().attachment().q_max - 1.0) * 100.0,
        p.frame().axis,
        (fw, fh),
        (px, py),
    );
    std::array::from_fn(|c| {
        let v = f64::from(out[c]) / gain;
        // Invert the ramp.
        ((v - 0.2) / (slope * (1.0 + 0.1 * c as f64))) - (px as f64 + 0.5)
    })
}

#[test]
fn anamorphic_fringe_boosts_the_axis_part_of_the_lateral_colour() {
    // The same plan on two libraries that differ only in the attachment's fringe (1 and 0): at Anamorphic 100
    // (q - 1 = 1) the axis shifts of the channels are twice as large with the fringe.
    let dir = tempfile::tempdir().unwrap();
    ym_lens::test_support::write_synthetic_lenses(dir.path()).unwrap();
    let path = dir.path().join("anamorphic.json");
    let mut att: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    att["field"]["fringe"] = serde_json::json!(0.0);
    std::fs::write(&path, serde_json::to_string_pretty(&att).unwrap()).unwrap();
    ym_lens::test_support::write_sums(dir.path()).unwrap();
    let without = ym_lens::LensLibrary::from_dir(dir.path()).unwrap();

    let (fw, fh) = (6000usize, 4000usize);
    let amounts = LensAmounts {
        kind: LensKind::Vintage,
        flares: 0.0,
        edges: 100.0,
        anamorphic: 100.0,
    };
    let frame = frame(fw, fh, Axis::X);
    let with = LensPlan::new(library(), &amounts, frame, camera()).unwrap();
    let plain = LensPlan::new(&without, &amounts, frame, camera()).unwrap();
    assert_eq!(with.fringe_factor(), 2.0);
    assert_eq!(plain.fringe_factor(), 1.0);
    // On the horizontal centre line the radial direction is x: the whole radial shift is along the axis.
    let (a, b) = (
        ramp_shifts(&with, (fw, fh), (5300, 2000)),
        ramp_shifts(&plain, (fw, fh), (5300, 2000)),
    );
    let apart = |s: [f64; 3]| s[0] - s[2];
    assert!(
        apart(b).abs() > 1.0,
        "red and blue are {} px apart without the fringe",
        apart(b)
    );
    let ratio = apart(a) / apart(b);
    assert!(
        (ratio - 2.0).abs() < 0.03,
        "ratio {ratio}: {a:?} against {b:?}"
    );
}
