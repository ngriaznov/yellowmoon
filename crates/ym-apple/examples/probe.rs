//! Asks Apple's on-device model about a synthetic picture, the way Apple Auto asks about a photo
//! (`docs/19-apple-auto.md` section 13, "On Mykyta's Mac").
//!
//! ```text
//! cargo run -p ym-apple --release --example probe [-- INSTRUCTIONS_FILE]
//! ```
//!
//! Prints whether Apple Auto is available here (and why not), the model and macOS version, then
//! asks twice about the same 768 × 512 picture (a smooth gradient with a darker disc), with the
//! default instructions or those of the file given, and prints every step (`-` for a field the model
//! left empty), the scene, the plan, the time each request took and whether the second answer
//! repeated the first (greedy sampling should repeat). `YELLOWMOON_TEST_APPLE_AUTO=fake` runs it
//! against the test advisor. Exits with 0 when Apple Auto is not available, 1 when a request fails
//! or the file cannot be read.

use std::process::ExitCode;
use std::time::Instant;

use ym_core::apple_auto::{
    AdvisorAnswer, AdvisorPicture, AutoSteps, Availability, DEFAULT_INSTRUCTIONS,
    effective_instructions,
};

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
    let own = match std::env::args_os().nth(1) {
        Some(path) => match std::fs::read_to_string(&path) {
            Ok(text) => text,
            Err(e) => {
                println!("can't read {}: {e}", path.to_string_lossy());
                return ExitCode::FAILURE;
            }
        },
        None => String::new(),
    };
    let instructions = effective_instructions(&own);
    println!(
        "available: {}",
        info.with_instructions(instructions).version_string()
    );
    println!(
        "instructions: {}",
        if instructions == DEFAULT_INSTRUCTIONS {
            "the default"
        } else {
            "from the file"
        }
    );

    let picture = synthetic_picture();
    println!(
        "asking about a {}×{} picture (gradient with a darker disc)…",
        picture.width, picture.height
    );
    let mut answers: Vec<AdvisorAnswer> = Vec::new();
    for attempt in 1..=2 {
        let started = Instant::now();
        let result = advisor.ask(&picture, instructions);
        let seconds = started.elapsed().as_secs_f64();
        match result {
            Ok(answer) => {
                println!(
                    "answer {attempt} in {seconds:.2} s: {} · scene \"{}\" · plan \"{}\"",
                    describe(answer.steps),
                    answer.scene,
                    answer.plan
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
            println!("repeated: yes (the same steps, scene and plan)");
        } else if first.steps == second.steps {
            println!("repeated: the steps, not the words");
        } else {
            println!("repeated: no");
        }
    }
    ExitCode::SUCCESS
}

/// Every step in words, `-` for a field left empty, e.g. `exposure +1 · highlights - · … ·
/// shadow colour teal 2 · highlight colour -`.
fn describe(steps: AutoSteps) -> String {
    let signed = |step: Option<i8>| match step {
        None => "-".to_owned(),
        Some(0) => "0".to_owned(),
        Some(s) => format!("{s:+}"),
    };
    let mut parts: Vec<String> = steps
        .numeric()
        .iter()
        .map(|(name, value, _)| format!("{name} {}", signed(*value)))
        .collect();
    let color = |name: Option<&str>, strength: Option<i8>| match (name, strength) {
        (None, _) => "-".to_owned(),
        (Some(n), None) => n.to_owned(),
        (Some(n), Some(s)) => format!("{n} {s}"),
    };
    parts.push(format!(
        "shadow colour {}",
        color(
            steps.shadow_color.map(|c| c.name()),
            steps.shadow_color_strength
        )
    ));
    parts.push(format!(
        "highlight colour {}",
        color(
            steps.highlight_color.map(|c| c.name()),
            steps.highlight_color_strength
        )
    ));
    parts.join(" · ")
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
