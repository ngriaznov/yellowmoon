//! `ym render --reference` (`docs/34-reference-grade.md` section 8, task RG-09).
//!
//! The command fits a Reference Grade node to a picture **with the code the app uses**: the node is put into the photo's settings
//! (a node added after Main when the graph has none, as Add node does), the reference is chosen with
//! `Core::reference_choose_file` and `Core::reference_fit` writes the fitted fields as one patch. The export that follows is
//! the ordinary one, so `ym` and the app write the same bytes for the same photo, reference and settings (`parity.rs`).
//!
//! A reference is a file only: a photo of the filmstrip has no meaning here.

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use serde_json::{Value, json};
use ym_core::reference::ReferencePicture;
use ym_core::reference_fit::{FitEvent, FitFailure, FitOutcome, FitStatus};
use ym_core::session::PhotoId;
use ym_core::settings::SettingsPatch;
use ym_gpu::queue::CancelToken;

use crate::console::{Console, bar};
use crate::exit::Failure;
use crate::report::number;
use crate::runtime::{Runtime, interrupted};
use crate::text::{lossy, path_text};

/// The flags of `--reference`, read once.
#[derive(Debug, Clone)]
pub struct ReferenceRun<'a> {
    /// The picture to match.
    pub file: &'a Path,
    /// `--reference-strength`, in percent.
    pub strength: Option<f64>,
    /// `--reference-brightness`, in percent.
    pub brightness: Option<f64>,
    /// `--no-skin-protection`.
    pub no_skin_protection: bool,
    /// `--reference-node`.
    pub node: Option<u32>,
}

/// Why a photo was not fitted.
#[derive(Debug)]
pub enum Stop {
    /// A mistake in the command line or the settings (exit 2): the run ends before anything is exported.
    Usage(String),
    /// This photo could not be fitted: the error `kind` and its sentence, and the detail for `--verbose`.
    Photo {
        /// The `error.kind` of `--json`.
        kind: &'static str,
        /// What went wrong and what to do.
        message: String,
        /// The detail for the log.
        detail: Option<String>,
    },
    /// Ctrl-C ended the fit: nothing was changed.
    Interrupted,
}

/// What a fit leaves for the report of a photo.
#[derive(Debug)]
pub struct Fitted {
    /// The `reference` object of `--json`.
    pub json: Value,
    /// The notes of the card ("No skin was found, so Skin protection did nothing.", the reference's own).
    pub notes: Vec<String>,
    /// The line for a person ("Fitted to the reference.").
    pub status_text: &'static str,
}

/// The sentence for a reference file that is not there.
const MISSING: &str =
    "The reference can't be used: this file was not found. Check the path and try again.";

/// Checks the reference file before any photo is opened: it is read, decoded and measured as a fit does, so a file that cannot
/// be used is a usage error (exit 2) with the text of `docs/34` section 9.3 after "The reference can't be used:".
///
/// # Errors
///
/// A [`Failure`] with exit code 2.
pub fn check_file(path: &Path) -> Result<(), Failure> {
    if !path.exists() {
        return Err(Failure::usage(MISSING));
    }
    ReferencePicture::from_file(path)
        .map(drop)
        .map_err(|e| Failure::usage(FitFailure::Reference(e).to_string()))
}

/// The element names of the fit's report, in the order of `docs/34` section 4b.
fn found_json(outcome: &FitOutcome) -> Value {
    let share = |x: f64| json!((x * 10_000.0).round() / 10_000.0);
    Value::Array(
        outcome
            .found
            .iter()
            .map(|e| {
                json!({
                    "element": e.element,
                    "photo_share": share(e.photo_share),
                    "reference_share": share(e.reference_share),
                })
            })
            .collect(),
    )
}

/// Fits the Reference Grade node of photo `id` to the reference and returns what it did. The photo's settings hold the fitted node
/// afterwards; a failure and a Ctrl-C leave them as they were (the node added for the fit stays, as the app leaves an added node).
///
/// # Errors
///
/// [`Stop`].
pub fn fit_photo(
    rt: &Runtime,
    id: PhotoId,
    run: &ReferenceRun<'_>,
    console: &mut Console,
) -> Result<Fitted, Stop> {
    let core = &rt.core;
    let photo_failure = |message: String| Stop::Photo {
        kind: "reference",
        message,
        detail: None,
    };
    let now = core
        .session
        .view(id)
        .map_err(|e| photo_failure(e.to_string()))?
        .settings;
    let mut next = now.clone();
    // Which node: the one the graph has, the one named, or a new one after Main.
    let node = match (run.node, next.nodes.references.len()) {
        (Some(n), _) => {
            if !next.nodes.references.contains_key(&n) {
                return Err(Stop::Usage(format!(
                    "Node {n} is not a Reference Grade node."
                )));
            }
            n
        }
        (None, 0) => next
            .nodes
            .insert_reference_grade_after_main()
            .map_err(|e| photo_failure(e.to_string()))?,
        (None, 1) => next
            .nodes
            .references
            .keys()
            .next()
            .copied()
            .unwrap_or_default(),
        (None, _) => {
            return Err(Stop::Usage(
                "The graph has more than one Reference Grade node. Say which with --reference-node."
                    .to_owned(),
            ));
        }
    };
    if let Some(n) = next.nodes.reference_grade_mut(node) {
        if let Some(strength) = run.strength {
            n.strength = strength;
        }
        if let Some(brightness) = run.brightness {
            n.match_brightness = brightness;
        }
        if run.no_skin_protection {
            n.skin_protection = false;
        }
    }
    let patch = SettingsPatch::between(&now, &next);
    if !patch.is_empty() {
        core.update_settings(id, &patch)
            .map_err(|e| photo_failure(e.to_string()))?;
    }
    core.reference_choose_file(id, node, run.file, None)
        .map_err(|e| failure_of(&e))?;

    let cancel = CancelToken::new();
    let done = AtomicBool::new(false);
    let started = Instant::now();
    let outcome = std::thread::scope(|scope| {
        // Ctrl-C stops the fit between candidates, as Cancel does in the app.
        scope.spawn(|| {
            while !done.load(Ordering::Acquire) {
                if interrupted() {
                    cancel.cancel();
                    break;
                }
                std::thread::sleep(Duration::from_millis(40));
            }
        });
        let result = core.reference_fit(id, node, &cancel, &mut |event| match event {
            FitEvent::Scene => console.progress(event.text()),
            FitEvent::Search {
                fraction,
                search,
                of,
                ..
            } => console.progress(format!(
                "Fitting… {search} of {of} {}",
                bar(fraction.clamp(0.0, 1.0), 20)
            )),
        });
        done.store(true, Ordering::Release);
        result
    });
    console.finish_progress();
    let outcome = match outcome {
        Ok(o) => o,
        Err(FitFailure::Cancelled) if interrupted() => return Err(Stop::Interrupted),
        Err(e) => return Err(failure_of(&e)),
    };
    let elapsed = started.elapsed();
    let applied = core
        .session
        .view(id)
        .ok()
        .and_then(|v| v.settings.nodes.reference_grade(node).cloned());
    let status = match outcome.status {
        FitStatus::Fitted => "fitted",
        FitStatus::AlreadyClose => "already_close",
    };
    let result = &outcome.result;
    let start = match result.start {
        ym_color::refgrade::search::Start::Neutral => "neutral",
        ym_color::refgrade::search::Start::Look => "look",
    };
    let json = json!({
        "file": lossy(run.file),
        "node": node,
        "status": status,
        "text": outcome.status.text(),
        "strength": applied.as_ref().map_or(Value::Null, |n| number(n.strength)),
        "skin_protection": applied.as_ref().map(|n| n.skin_protection),
        "skin_found": outcome.skin_found,
        "brightness": {
            "percent": applied.as_ref().map_or(Value::Null, |n| number(n.match_brightness)),
            "moved_ev": number(outcome.brightness.moved_ev),
            "limited": outcome.brightness.limited,
        },
        "start": start,
        "elements": found_json(&outcome),
        "candidates": result.evals,
        "time_ms": u64::try_from(elapsed.as_millis()).unwrap_or(u64::MAX),
    });
    Ok(Fitted {
        json,
        notes: outcome.notes.clone(),
        status_text: outcome.status.text(),
    })
}

/// The photo's failure for a [`FitFailure`]: a missing scene model is a `mask` error as it is for a render; the rest is `reference`.
fn failure_of(e: &FitFailure) -> Stop {
    match e {
        FitFailure::SceneModelMissing => Stop::Photo {
            kind: "mask",
            message: e.to_string(),
            detail: None,
        },
        FitFailure::Cancelled => Stop::Interrupted,
        FitFailure::Failed(detail) => Stop::Photo {
            kind: "reference",
            message: e.to_string(),
            detail: Some(detail.clone()),
        },
        other => Stop::Photo {
            kind: "reference",
            message: other.to_string(),
            detail: None,
        },
    }
}

/// The line printed for a photo that was fitted.
#[must_use]
pub fn status_line(input: &Path, fitted: &Fitted) -> String {
    format!("{}: {}", path_text(input), fitted.status_text)
}
