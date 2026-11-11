//! The 24 MP benchmark of `docs/27` section 8: one face of realistic size, mock models, all three
//! tools at 60. Run with `cargo bench -p ym-retouch --features mock`.

use criterion::{Criterion, criterion_group, criterion_main};
use ym_retouch::mock::{self, MockProposer};
use ym_retouch::{
    Face, LinearAwg4Image, Models, Rect, RetouchCache, SkinSettings, face_masks, prepare_skin,
    retouch_skin,
};

const FULL: (u32, u32) = (6000, 4000);

fn picture(w: u32, h: u32) -> LinearAwg4Image {
    let mut data = Vec::with_capacity((w * h) as usize);
    for y in 0..h {
        for x in 0..w {
            let mut n = x.wrapping_mul(0x9e37_79b1) ^ y.wrapping_mul(0x85eb_ca6b);
            n ^= n >> 15;
            n = n.wrapping_mul(0x2c1b_3c6d);
            let noise = ((n & 0xffff) as f32 / 65535.0 - 0.5) * 0.01;
            let l = 0.25 + 0.1 * (x as f32 / w as f32) + 0.05 * (y as f32 / h as f32) + noise;
            data.push([l, l * 0.78, l * 0.62]);
        }
    }
    LinearAwg4Image {
        width_px: w,
        height_px: h,
        data,
    }
}

/// Area downscale by an integer factor.
fn downscale(img: &LinearAwg4Image, f: u32) -> LinearAwg4Image {
    let (w, h) = (img.width_px / f, img.height_px / f);
    let mut data = Vec::with_capacity((w * h) as usize);
    for y in 0..h {
        for x in 0..w {
            let mut acc = [0.0f32; 3];
            for dy in 0..f {
                for dx in 0..f {
                    let p = img.data[((y * f + dy) * img.width_px + x * f + dx) as usize];
                    for c in 0..3 {
                        acc[c] += p[c];
                    }
                }
            }
            let k = 1.0 / (f * f) as f32;
            data.push([acc[0] * k, acc[1] * k, acc[2] * k]);
        }
    }
    LinearAwg4Image {
        width_px: w,
        height_px: h,
        data,
    }
}

fn models() -> Models {
    Models {
        proposer: Box::new(MockProposer::scale(192, 0.9)),
        ..mock::models(vec![Rect::new(2600, 1500, 60, 60)])
    }
}

fn bench(c: &mut Criterion) {
    let face = Face {
        rect: Rect::new(2100, 500, 1500, 1800),
        scale: 1800.0 / 6000.0,
    };
    let settings = SkinSettings {
        blemish: 60.0,
        dark_circles: 60.0,
        even_skin: 60.0,
        texture: 0.0,
        include_neck_blemish: true,
        include_neck_even_skin: true,
    };
    let faces = [(face, settings)];
    let full = picture(FULL.0, FULL.1);
    let preview = downscale(&full, 4);
    let models = models();

    let mut group = c.benchmark_group("retouch_24mp_one_face");
    group.sample_size(10);
    group.bench_function("masks", |b| {
        b.iter(|| face_masks(&models, &full, &face, &settings, FULL, None).unwrap());
    });
    group.bench_function("prepare_masks_and_map", |b| {
        b.iter(|| {
            let mut cache = RetouchCache::new(1, 1);
            prepare_skin(&full, &faces, &models, &mut cache).unwrap();
            cache
        });
    });

    let mut warm = RetouchCache::new(1, 1);
    prepare_skin(&full, &faces, &models, &mut warm).unwrap();
    group.bench_function("apply_full_warm_cache", |b| {
        b.iter(|| retouch_skin(&full, FULL, &faces, 1.0, &models, &mut warm, None).unwrap());
    });
    group.bench_function("apply_scale_0.25_warm_cache", |b| {
        b.iter(|| retouch_skin(&preview, FULL, &faces, 0.25, &models, &mut warm, None).unwrap());
    });
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
