//! Large random sweep of the ACES 2 output transform against OpenColorIO (development check).
//!
//! Ignored by default: it needs files produced by the vector script.
//!
//! ```text
//! python3 tools/vectors/aces2_vectors.py --sweep 2000000 --sweep-out /tmp/aces2_sweep
//! YM_ACES2_SWEEP_DIR=/tmp/aces2_sweep cargo test --release -p ym-color --test aces2_sweep -- --ignored --nocapture
//! ```
//!
//! Prints the maximum, 99.9th and 99th percentile error in 10-bit code values per output and lists
//! the worst samples. The 1 code value gate applies as in `aces2_vectors.rs`.
//!
//! OpenColorIO's float32 code returns NaN (written as black) for a thin band of very bright,
//! slightly coloured input where the tone-scaled lightness lands on the display maximum (see the
//! module notes of `ym_color::aces2`). Samples where the reference is exactly black and the
//! transform gives near-white are counted as "reference glitches" and left out of the statistics;
//! the count and the inputs are printed so they cannot grow unnoticed.

use std::path::{Path, PathBuf};

use ym_color::aces2::{Aces2Output, Aces2Transform};

const CODE_VALUES_10BIT: f64 = 1023.0;

fn read_f32_triplets(path: &Path) -> Vec<[f64; 3]> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    assert_eq!(bytes.len() % 12, 0);
    bytes
        .chunks_exact(12)
        .map(|c| {
            let f = |i: usize| {
                f32::from_le_bytes([c[4 * i], c[4 * i + 1], c[4 * i + 2], c[4 * i + 3]]) as f64
            };
            [f(0), f(1), f(2)]
        })
        .collect()
}

fn code_error(a: [f64; 3], b: [f64; 3]) -> f64 {
    (0..3)
        .map(|k| (a[k] - b[k]).abs() * CODE_VALUES_10BIT)
        .fold(0.0, f64::max)
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    sorted[((sorted.len() as f64 * p).ceil() as usize).clamp(1, sorted.len()) - 1]
}

/// Reference glitch signature: exactly black from OpenColorIO while ours is close to white.
fn is_reference_glitch(ours: [f64; 3], expected: [f64; 3]) -> bool {
    expected == [0.0; 3] && ours.iter().all(|&c| c > 0.9)
}

#[test]
#[ignore = "needs the sweep files from tools/vectors/aces2_vectors.py --sweep (set YM_ACES2_SWEEP_DIR)"]
fn sweep_matches_ocio() {
    let Ok(dir) = std::env::var("YM_ACES2_SWEEP_DIR") else {
        println!("YM_ACES2_SWEEP_DIR is not set; nothing to compare (see the module docs)");
        return;
    };
    let dir = PathBuf::from(dir);
    let mut failures = Vec::new();

    for output in Aces2Output::ALL {
        let name = output.name();
        let input = read_f32_triplets(&dir.join(format!("{name}.in.f32")));
        let expected = read_f32_triplets(&dir.join(format!("{name}.enc.f32")));
        let transform = Aces2Transform::new(output);

        // Error per sample, NaN for reference glitches.
        let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
        let chunk = input.len().div_ceil(threads);
        let all_errors: Vec<f64> = std::thread::scope(|s| {
            let handles: Vec<_> = input
                .chunks(chunk)
                .zip(expected.chunks(chunk))
                .map(|(inp, exp)| {
                    let transform = &transform;
                    s.spawn(move || {
                        inp.iter()
                            .zip(exp)
                            .map(|(i, e)| {
                                let ours = transform.transform_ap0_unclamped(*i);
                                if is_reference_glitch(ours, *e) {
                                    f64::NAN
                                } else {
                                    code_error(ours, e.map(|c| c.max(0.0)))
                                }
                            })
                            .collect::<Vec<f64>>()
                    })
                })
                .collect();
            handles
                .into_iter()
                .flat_map(|h| h.join().expect("worker"))
                .collect()
        });

        let glitch_inputs: Vec<[f64; 3]> = all_errors
            .iter()
            .zip(&input)
            .filter(|(e, _)| e.is_nan())
            .map(|(_, i)| *i)
            .collect();
        let mut sorted: Vec<f64> = all_errors.iter().copied().filter(|e| !e.is_nan()).collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).expect("no NaN left"));
        let max = *sorted.last().expect("samples");
        let over = sorted.iter().filter(|&&e| e > 1.0).count();
        println!(
            "{name:<11} {} samples: max {:.4}  p99.9 {:.4}  p99 {:.4}  median {:.5}  over 1 code value: {over}  reference glitches: {} {:?}",
            sorted.len(),
            max,
            percentile(&sorted, 0.999),
            percentile(&sorted, 0.99),
            percentile(&sorted, 0.5),
            glitch_inputs.len(),
            glitch_inputs
        );

        let mut worst: Vec<usize> = (0..all_errors.len())
            .filter(|&i| !all_errors[i].is_nan())
            .collect();
        worst.sort_by(|&a, &b| all_errors[b].partial_cmp(&all_errors[a]).expect("no NaN"));
        for &i in worst.iter().take(5) {
            println!(
                "    #{i}: {:.4}  in {:?}\n        ours {:?}\n        ocio {:?}",
                all_errors[i],
                input[i],
                transform.transform_ap0_unclamped(input[i]),
                expected[i]
            );
        }
        if max > 1.0 {
            failures.push(format!("{name}: max {max:.4}"));
        }
        // A few glitches are expected; a growing number means something else broke.
        if glitch_inputs.len() * 100_000 > input.len() {
            failures.push(format!(
                "{name}: {} reference glitches",
                glitch_inputs.len()
            ));
        }
    }
    assert!(failures.is_empty(), "gate exceeded: {failures:?}");
}
