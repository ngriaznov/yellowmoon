//! The resize against Pillow's Lanczos filter (`docs/09-export.md` section Tests: Lanczos in
//! linear light), and the linear-light property itself.

mod common;

use ym_color::encode::OutputSpace;
use ym_export::pixels::{ExportHooks, ImageRef, PixelEncoding};
use ym_export::resize::resize_lanczos3;

fn noisy_image(w: u32, h: u32) -> Vec<f32> {
    let mut rng = common::Rng(0xA5A5_5A5A_1234_5678 ^ u64::from(w * 977 + h));
    let mut v = Vec::new();
    for y in 0..h {
        for x in 0..w {
            for c in 0..3 {
                let smooth =
                    0.5 + 0.4 * ((x as f32 * 0.21 + c as f32).sin() * (y as f32 * 0.13).cos());
                let edge = if (x / 9 + y / 7) % 2 == 0 { 0.2 } else { 0.0 };
                v.push(smooth + edge + (rng.unit() - 0.5) * 0.2);
            }
        }
    }
    v
}

#[test]
fn matches_pillows_lanczos_on_float_images() {
    if !common::require_python_modules(&["numpy", "PIL"]) {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let mut worst = 0.0_f32;
    for (w, h, ow, oh) in [
        (97, 61, 40, 25),
        (100, 80, 33, 27),
        (64, 64, 21, 13),
        (50, 30, 49, 29),
        (31, 17, 8, 5),
        (200, 120, 199, 7),
    ] {
        let data = noisy_image(w, h);
        let src = dir.path().join("in.f32");
        let dst = dir.path().join("out.f32");
        std::fs::write(
            &src,
            data.iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<u8>>(),
        )
        .unwrap();
        common::run_python(
            &common::tests_py("resize_ref.py"),
            &[
                src.to_str().unwrap(),
                &w.to_string(),
                &h.to_string(),
                &ow.to_string(),
                &oh.to_string(),
                dst.to_str().unwrap(),
            ],
        );
        let reference: Vec<f32> = std::fs::read(&dst)
            .unwrap()
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let img = ImageRef::new(w, h, &data, PixelEncoding::DisplayLinear).unwrap();
        let ours = resize_lanczos3(&img, ow, oh, OutputSpace::Srgb, &ExportHooks::none()).unwrap();
        assert_eq!(ours.rgb.len(), reference.len());
        for (i, (a, b)) in ours.rgb.iter().zip(&reference).enumerate() {
            let d = (a - b).abs();
            worst = worst.max(d);
            assert!(
                d < 2e-5,
                "{w}x{h} -> {ow}x{oh} sample {i}: ours {a} Pillow {b}"
            );
        }
    }
    eprintln!("Lanczos-3 resize vs Pillow (float32): worst difference {worst:.2e}");
}

#[test]
fn resizing_in_linear_light_differs_from_resizing_code_values() {
    // Alternating black and white pixels: the linear average is 0.5 light, which is sRGB code 0.735.
    let (w, h) = (64_u32, 8_u32);
    let mut linear = vec![0.0_f32; (w * h * 3) as usize];
    for y in 0..h as usize {
        for x in (0..w as usize).step_by(2) {
            let i = (y * w as usize + x) * 3;
            linear[i..i + 3].fill(1.0);
        }
    }
    let img = ImageRef::new(w, h, &linear, PixelEncoding::DisplayLinear).unwrap();
    let small = resize_lanczos3(&img, 4, 1, OutputSpace::Srgb, &ExportHooks::none()).unwrap();
    let code = ym_color::encode::srgb_encode(f64::from(small.rgb[3]));
    assert!(
        (code - 0.735).abs() < 0.01,
        "encoded after a linear-light resize: {code}"
    );
    // Resizing the encoded values and decoding them afterwards would give 0.5^2.2 = 0.2 light instead.
    assert!(small.rgb[3] > 0.45);
}
