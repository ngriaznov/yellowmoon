//! What the white lift estimator makes of real files: the per-photo numbers of `docs/02` C12 step 1b
//! (the amount `A`, the anchor, how high-key the photo is, the display code of the brightest tones before
//! and after the lift), one JSON object per line.
//!
//! ```text
//! cargo run -p ym-core --example white_lift_report -- testdata/real/files testdata/readme-photos/src
//! ```
//!
//! Each argument is a folder or an ARW file. The numbers are the automatic estimates a session makes when
//! the photo is opened (the Signature look as the reference, the default roll-off, the photo's own
//! exposure); no image is rendered. `ym render --white-offset -3 ...` renders the picture without the
//! lift, and the same command without the option renders it with the automatic one.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use ym_core::session::{OpenOptions, Session, SessionEnv};

fn main() {
    let paths: Vec<PathBuf> = std::env::args().skip(1).map(PathBuf::from).collect();
    assert!(!paths.is_empty(), "give a folder or ARW files");
    let session = Session::new(SessionEnv::from_data_dir().expect("the data folder"));
    // One photo at a time, so the burst median never joins files that are not one scene.
    for path in &paths {
        let files: Vec<PathBuf> = if path.is_dir() {
            let mut v: Vec<PathBuf> = std::fs::read_dir(path)
                .expect("read the folder")
                .flatten()
                .map(|e| e.path())
                .filter(|p| p.extension().is_some_and(|x| x.eq_ignore_ascii_case("arw")))
                .collect();
            v.sort();
            v
        } else {
            vec![path.clone()]
        };
        for file in files {
            let started = Instant::now();
            let out = session.open(std::slice::from_ref(&file), OpenOptions::default());
            assert!(session.wait_idle(Duration::from_secs(900)));
            let Some(&id) = out.added.first() else {
                eprintln!("{}: not opened", file.display());
                continue;
            };
            let view = session.view(id).expect("view");
            let w = view.estimates.white;
            println!(
                "{}",
                serde_json::json!({
                    "file": file.file_name().and_then(|n| n.to_str()),
                    "exposure_ev": view.estimates.exposure.ev,
                    "white_ev": w.amount_ev,
                    "anchor_ev": w.anchor_ev,
                    "high_key": w.high_key,
                    "highlight_ev": w.highlight_ev,
                    "code_before": w.code_before,
                    "target_code": w.target_code,
                    "code_after": w.code_after,
                    "preview_pixels": w.pixels,
                    "open_ms": started.elapsed().as_millis(),
                })
            );
            session.close();
        }
    }
}
