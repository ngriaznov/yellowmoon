//! Asks Apple's on-device model about a synthetic picture, the way Apple Auto asks about a photo
//! (`docs/19-apple-auto.md` section 13, "On Mykyta's Mac").
//!
//! ```text
//! cargo run -p ym-apple --release --example probe
//! ```
//!
//! Prints whether Apple Auto is available here (and why not), the model and macOS version, then
//! asks twice about the same 768 × 512 picture (a smooth gradient with a darker disc) and prints the
//! steps, the scene, the time each request took and whether the second answer repeated the first
//! (greedy sampling should repeat). `YELLOWMOON_TEST_APPLE_AUTO=fake` runs it against the test
//! advisor. Exits with 0 when Apple Auto is not available, 1 when a request fails.

use std::process::ExitCode;
use std::time::Instant;

use ym_core::apple_auto::{AdvisorAnswer, AdvisorPicture, AutoSteps, Availability};

const WIDTH: usize = 768;
const HEIGHT: usize = 512;

fn main() -> ExitCode {
    let advisor = ym_apple::system_advisor();
    println!(
        "bridge built: {}{}",
        if ym_apple::BRIDGE_BUILT { "yes" } else { "no" },
        match std::env::var(ym_apple::TEST_ADVISOR_ENV) {
            Ok(value) => format!(" ({}={value})", ym_apple::TEST_ADVISOR_ENV),
            Err(_) => String::new(),
        }
    );
    let info = match advisor.availability() {
        Availability::Available(info) => info,
        Availability::Unavailable(reason) => {
            println!(
                "Apple Auto is not available on this computer ({}). [{reason:?}]",
                reason.words()
            );
            return ExitCode::SUCCESS;
        }
    };
    println!("available: {}", info.version_string());

    let picture = synthetic_picture();
    println!(
        "asking about a {}×{} picture (gradient with a darker disc)…",
        picture.width, picture.height
    );
    let mut answers: Vec<AdvisorAnswer> = Vec::new();
    for attempt in 1..=2 {
        let started = Instant::now();
        let result = advisor.ask(&picture);
        let seconds = started.elapsed().as_secs_f64();
        match result {
            Ok(answer) => {
                println!(
                    "answer {attempt} in {seconds:.2} s: {} · scene \"{}\"",
                    describe(answer.steps),
                    answer.scene
                );
                answers.push(answer);
            }
            Err(error) => {
                println!(
                    "request {attempt} failed after {seconds:.2} s: {error} [{:?}]",
                    error.kind
                );
                return ExitCode::FAILURE;
            }
        }
    }
    if let [first, second] = answers.as_slice() {
        if first == second {
            println!("repeated: yes (the same steps and scene)");
        } else if first.steps == second.steps {
            println!("repeated: the steps, not the scene");
        } else {
            println!("repeated: no");
        }
    }
    ExitCode::SUCCESS
}

/// The steps in words, e.g. `exposure +1 · white 0 · contrast +1 · saturation 0`.
fn describe(steps: AutoSteps) -> String {
    let signed = |step: i8| {
        if step == 0 {
            "0".to_owned()
        } else {
            format!("{step:+}")
        }
    };
    format!(
        "exposure {} · white {} · contrast {} · saturation {}",
        signed(steps.exposure),
        signed(steps.white),
        signed(steps.contrast),
        signed(steps.saturation)
    )
}

/// A smooth gradient (a cool, brighter top to a warm, darker bottom) with a darker disc a little
/// right of centre: 8-bit sRGB, 3 bytes per pixel, rows top to bottom, as Apple Auto sends.
fn synthetic_picture() -> AdvisorPicture {
    let mut rgb = Vec::with_capacity(WIDTH * HEIGHT * 3);
    let (cx, cy) = (0.6 * WIDTH as f64, 0.55 * HEIGHT as f64);
    let radius = 0.25 * HEIGHT as f64;
    for y in 0..HEIGHT {
        let fy = y as f64 / (HEIGHT - 1) as f64;
        for x in 0..WIDTH {
            let fx = x as f64 / (WIDTH - 1) as f64;
            let mut pixel = [
                150.0 + 60.0 * fy + 20.0 * fx,
                170.0 + 10.0 * fy,
                220.0 - 110.0 * fy - 20.0 * fx,
            ];
            let distance = ((x as f64 - cx).powi(2) + (y as f64 - cy).powi(2)).sqrt();
            // A soft edge over 4 pixels.
            let inside = ((radius - distance) / 4.0).clamp(0.0, 1.0);
            for channel in &mut pixel {
                *channel *= 1.0 - 0.6 * inside;
            }
            // In 0…255 after the clamp, so the cast cannot truncate.
            rgb.extend(pixel.map(|v| v.round().clamp(0.0, 255.0) as u8));
        }
    }
    AdvisorPicture {
        width: WIDTH,
        height: HEIGHT,
        rgb,
    }
}
