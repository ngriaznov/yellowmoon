//! C13: display transfer functions, quantisation and the deterministic blue-noise dither.

use ym_color::encode::blue_noise::{self, SIDE, TEXELS};
use ym_color::encode::{
    OutputSpace, bt1886_eotf, bt1886_inverse, dither_8bit, dither_threshold, encode_8bit_dithered,
    encode_quantized, max_code, quantize, srgb_decode, srgb_encode,
};

#[test]
fn srgb_matches_colour_science() {
    // colour-science 0.4.7 eotf_inverse_sRGB / eotf_sRGB (pasted from a local run).
    for (x, want) in [
        (0.0, 0.0),
        (0.001, 0.012920000000000001),
        (0.01, 0.09985282273412832),
        (0.18, 0.46135612950044164),
        (0.5, 0.7353569830524495),
        (1.0, 1.0),
        (2.0, 1.3532560461493863),
    ] {
        assert!((srgb_encode(x) - want).abs() < 1e-14, "encode {x}");
    }
    for (e, want) in [
        (0.0, 0.0),
        (0.2, 0.033104766570885055),
        (0.5, 0.21404114048223255),
        (1.0, 1.0),
    ] {
        assert!((srgb_decode(e) - want).abs() < 1e-14, "decode {e}");
    }
    // At the join the two published constants disagree by ~2e-9 (a property of the standard).
    assert!((srgb_decode(0.04045) - 0.0031308072830676845).abs() < 1e-8);
}

#[test]
fn srgb_is_monotone_and_round_trips() {
    let mut prev = srgb_encode(-0.01);
    for k in -9..=20_000 {
        let x = f64::from(k) / 10_000.0;
        let e = srgb_encode(x);
        assert!(e >= prev);
        prev = e;
        assert!((srgb_decode(e) - x).abs() < 3e-9, "x={x}");
    }
    assert_eq!(srgb_encode(-0.25), -srgb_encode(0.25));
    assert_eq!(srgb_encode(0.0), 0.0);
}

#[test]
fn bt1886_is_a_pure_power() {
    for x in [0.0, 0.001, 0.18, 0.5, 1.0] {
        assert!((bt1886_inverse(x) - x.powf(1.0 / 2.4)).abs() < 1e-15);
        assert!((bt1886_eotf(bt1886_inverse(x)) - x).abs() < 1e-14);
    }
    assert!((bt1886_inverse(0.18) - 0.489_437_09).abs() < 1e-7);
    assert_eq!(bt1886_inverse(-0.5), -bt1886_inverse(0.5));
}

#[test]
fn output_spaces_pick_the_right_transfer() {
    let x = 0.18;
    assert_eq!(OutputSpace::Srgb.encode_channel(x), srgb_encode(x));
    assert_eq!(OutputSpace::DisplayP3.encode_channel(x), srgb_encode(x));
    assert_eq!(OutputSpace::Rec2020.encode_channel(x), bt1886_inverse(x));
    assert_eq!(
        OutputSpace::Srgb.encode_clamped([-1.0, 0.5, 7.0]),
        [0.0, srgb_encode(0.5), 1.0]
    );
    assert_eq!(
        OutputSpace::Srgb.encode_clamped([f64::NAN, 0.0, 1.0])[0],
        0.0
    );
}

#[test]
fn quantization_levels() {
    assert_eq!(
        (max_code(8), max_code(10), max_code(16)),
        (255, 1023, 65535)
    );
    for (v, bits, want) in [
        (0.0, 8, 0),
        (1.0, 8, 255),
        (0.5, 8, 128),
        (0.5, 10, 512),
        (0.5, 16, 32768),
        (1.0 / 255.0, 8, 1),
        (0.4999, 8, 127),
    ] {
        assert_eq!(quantize(v, bits), want, "v={v} bits={bits}");
    }
    // 10 bit: one code value is 1/1023.
    for code in 0..=1023_u16 {
        assert_eq!(quantize(f64::from(code) / 1023.0, 10), code);
    }
    assert_eq!(
        encode_quantized([1.0, 0.0, 0.18], OutputSpace::Srgb, 10)[0],
        1023
    );
    assert_eq!(
        encode_quantized([0.18, 0.18, 0.18], OutputSpace::Srgb, 8),
        [118, 118, 118]
    );
}

// ---- Blue noise ----------------------------------------------------------------------------------

fn table() -> Vec<u16> {
    // Read back through the public threshold: rank = threshold * 4096 - 0.5.
    (0..TEXELS)
        .map(|i| (dither_threshold(i % SIDE, i / SIDE) * TEXELS as f64 - 0.5).round() as u16)
        .collect()
}

fn fnv1a(values: &[u16]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for v in values {
        for b in v.to_le_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

#[test]
fn the_embedded_texture_is_a_permutation_of_all_ranks() {
    let t = table();
    let mut seen = vec![false; TEXELS];
    for r in &t {
        assert!(!seen[usize::from(*r)]);
        seen[usize::from(*r)] = true;
    }
    assert!(seen.iter().all(|&s| s));
}

#[test]
fn the_embedded_texture_is_what_the_generator_makes() {
    // Regenerates the texture from scratch (seeded void-and-cluster) and compares with the table.
    let generated = blue_noise::generate();
    assert_eq!(generated, table());
    assert_eq!(blue_noise::SEED, 0x5945_4c4c_4f57_4d4f);
    // A fingerprint of the shipped table: a change here changes every dithered export (bump pipeline_version).
    println!("blue-noise table fingerprint: {:#018x}", fnv1a(&generated));
    assert_eq!(fnv1a(&generated), BLUE_NOISE_FINGERPRINT);
}

/// FNV-1a of the table as little-endian u16 (see the test above).
const BLUE_NOISE_FINGERPRINT: u64 = 0x993f_ccd5_c6db_3aa9;

#[test]
fn the_texture_is_blue() {
    // Radial power spectrum of the threshold texture: strongly suppressed at low frequencies.
    let n = SIDE;
    let t: Vec<f64> = (0..TEXELS)
        .map(|i| dither_threshold(i % n, i / n) - 0.5)
        .collect();
    // Naive separable DFT: rows then columns (64 is small).
    let cos: Vec<f64> = (0..n * n)
        .map(|k| (2.0 * std::f64::consts::PI * (k % n) as f64 / n as f64 * (k / n) as f64).cos())
        .collect();
    let sin: Vec<f64> = (0..n * n)
        .map(|k| (2.0 * std::f64::consts::PI * (k % n) as f64 / n as f64 * (k / n) as f64).sin())
        .collect();
    let dft1 = |re: &[f64], im: &[f64], stride: usize, offset: usize, kfreq: usize| -> (f64, f64) {
        let (mut a, mut b) = (0.0, 0.0);
        for j in 0..n {
            let (c, s) = (cos[kfreq * n + (j % n)], sin[kfreq * n + (j % n)]);
            // cos table indexed [freq][pos]
            let (r, i) = (re[offset + j * stride], im[offset + j * stride]);
            a += r * c + i * s;
            b += i * c - r * s;
        }
        (a, b)
    };
    let zeros = vec![0.0; TEXELS];
    // Rows (along x).
    let mut re = vec![0.0; TEXELS];
    let mut im = vec![0.0; TEXELS];
    for y in 0..n {
        for kx in 0..n {
            let (a, b) = dft1(&t, &zeros, 1, y * n, kx);
            re[y * n + kx] = a;
            im[y * n + kx] = b;
        }
    }
    // Columns (along y).
    let mut power = vec![0.0; TEXELS];
    for kx in 0..n {
        for ky in 0..n {
            let (a, b) = dft1(&re, &im, n, kx, ky);
            power[ky * n + kx] = a * a + b * b;
        }
    }
    let radial = |lo: f64, hi: f64| -> f64 {
        let (mut sum, mut count) = (0.0, 0);
        for ky in 0..n {
            for kx in 0..n {
                let fx = if kx > n / 2 {
                    kx as f64 - n as f64
                } else {
                    kx as f64
                };
                let fy = if ky > n / 2 {
                    ky as f64 - n as f64
                } else {
                    ky as f64
                };
                let r = fx.hypot(fy);
                if r >= lo && r < hi {
                    sum += power[ky * n + kx];
                    count += 1;
                }
            }
        }
        sum / f64::from(count)
    };
    let low = radial(1.0, 5.0);
    let mid = radial(5.0, 12.0);
    let high = radial(16.0, 33.0);
    println!(
        "blue-noise spectrum: low {low:.3e}, mid {mid:.3e}, high {high:.3e}, low/high {:.4}",
        low / high
    );
    // White noise would give low / high ~ 1.
    assert!(
        low < 0.05 * high,
        "low-frequency power {low:e} vs high {high:e}"
    );
    assert!(mid < high);
}

#[test]
fn dither_is_deterministic_and_position_only() {
    for (x, y) in [(0, 0), (5, 9), (63, 63), (64, 0), (0, 64), (200, 131)] {
        let a = dither_8bit(0.37, x, y);
        assert_eq!(a, dither_8bit(0.37, x, y));
        assert_eq!(a, dither_8bit(0.37, x % 64, y % 64));
        assert_eq!(
            dither_threshold(x, y),
            dither_threshold(x + 64 * 3, y + 64 * 7)
        );
    }
    // Same linear input, same pixel: same bytes, however often it runs.
    let px = [0.2, 0.05, 0.9];
    let first = encode_8bit_dithered(px, OutputSpace::DisplayP3, 77, 12);
    for _ in 0..10 {
        assert_eq!(
            first,
            encode_8bit_dithered(px, OutputSpace::DisplayP3, 77, 12)
        );
    }
    // All three channels use the same threshold: equal inputs give equal outputs.
    let gray = encode_8bit_dithered([0.3, 0.3, 0.3], OutputSpace::Srgb, 10, 20);
    assert!(gray[0] == gray[1] && gray[1] == gray[2]);
}

#[test]
fn dither_stays_within_one_code_value_and_is_exact_on_integers() {
    for level in 0..=2550 {
        let v = f64::from(level) / 2550.0;
        let exact = v * 255.0;
        for (x, y) in [(0, 0), (13, 40), (63, 63), (31, 2)] {
            let d = f64::from(dither_8bit(v, x, y));
            assert!((d - exact).abs() < 1.0 + 1e-9, "v={v}: {d}");
        }
    }
    for code in 0..=255_u32 {
        for (x, y) in [(0, 0), (13, 40), (63, 63)] {
            assert_eq!(u32::from(dither_8bit(f64::from(code) / 255.0, x, y)), code);
        }
    }
}

#[test]
fn dithered_means_are_unbiased() {
    // Over one 64 x 64 tile the thresholds are the 4096 evenly spaced values in (0, 1), so the mean of a
    // dithered constant equals the unquantised value to within 1/4096 of a code value.
    let mut worst = 0.0_f64;
    for level in 0..=1020 {
        let v = f64::from(level) / 1020.0;
        let mut sum = 0.0;
        for y in 0..64 {
            for x in 0..64 {
                sum += f64::from(dither_8bit(v, x, y));
            }
        }
        let mean = sum / 4096.0;
        worst = worst.max((mean - v * 255.0).abs());
    }
    println!("max |mean(dithered) - exact| over a 64x64 tile: {worst:e} code values");
    assert!(worst < 2.5e-4, "{worst:e}");

    // A ramp across many tiles: local means follow the ramp (mean over each 64-wide column band).
    let width = 1024;
    let mut worst_band = 0.0_f64;
    for band in 0..width / 64 {
        let (mut sum, mut exact) = (0.0, 0.0);
        for y in 0..64 {
            for x in band * 64..(band + 1) * 64 {
                let v = x as f64 / (width - 1) as f64;
                sum += f64::from(dither_8bit(v, x, y));
                exact += v * 255.0;
            }
        }
        worst_band = worst_band.max(((sum - exact) / 4096.0).abs());
    }
    println!("max band-mean error on a ramp: {worst_band:e} code values");
    assert!(worst_band < 0.05, "{worst_band:e}");
}

#[test]
fn undithered_rounding_is_biased_where_dither_is_not() {
    // At 0.25 of a code value the plain rounding is always 0 (mean error 0.25); the dither averages to 0.25.
    let v = 0.25 / 255.0;
    assert_eq!(quantize(v, 8), 0);
    let mean: f64 = (0..4096)
        .map(|i| f64::from(dither_8bit(v, i % 64, i / 64)))
        .sum::<f64>()
        / 4096.0;
    assert!((mean - 0.25).abs() < 1e-3, "{mean}");
}

#[test]
fn rec709_look_output_converts_to_each_output_space() {
    use ym_color::gamut::{LinearRec709, matrices};
    let rec709 = LinearRec709([0.3, 0.5, 0.1]);
    assert_eq!(OutputSpace::Srgb.from_linear_rec709(rec709), rec709.0);
    let p3 = OutputSpace::DisplayP3.from_linear_rec709(rec709);
    let want = ym_color::consts::linalg::mul_vec(&matrices().rec709_to_p3, rec709.0);
    assert_eq!(p3, want);
    // White maps to white in every space (same D65 white).
    for space in [
        OutputSpace::Srgb,
        OutputSpace::DisplayP3,
        OutputSpace::Rec2020,
    ] {
        for c in space.from_linear_rec709(LinearRec709([1.0; 3])) {
            assert!((c - 1.0).abs() < 1e-14);
        }
    }
}
