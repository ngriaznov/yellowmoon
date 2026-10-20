//! Apple Auto in the core (`docs/19`): Auto contrast and saturation in the settings, the estimates
//! with and without an answer, burst medians, the worker's order, the viewer that never waits, the
//! exports that do, failures and retries, the preference both ways, Re-run, closing, the picture the
//! model sees, the XMP and the Core API. Runs on any machine with the deterministic
//! [`FakeAdvisor`] and a scripted advisor; the Swift bridge has its own tests in `ym-apple`.

mod common;

use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::Receiver;
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant};

use common::tiffmini::{xmp_of, xmp_property};
use common::*;
use serde_json::{Value, json};
use ym_core::app::{Core, CoreOptions};
use ym_core::apple_auto::{
    AdvisorAnswer, AdvisorError, AdvisorInfo, AdvisorPicture, AppleAuto, AppleAutoMode,
    AutoAdvisor, AutoSteps, Availability, FailureKind, FakeAdvisor, PICTURE_LONG_EDGE_PX,
    UnavailableReason,
};
use ym_core::auto::{AutoSource, tables_of};
use ym_core::export::ExportContext;
use ym_core::export_queue::{ExportEvent, ExportJob, ExportQueue, ExportSink};
use ym_core::pipeline::{AdvisorPictures, PhotoSource, RenderOptions, Renderer};
use ym_core::preferences::PreferencesPatch;
use ym_core::session::{
    AppleAutoEnv, ApplyScope, EventSink, OpenOptions, PhotoId, PictureInput, PictureRenderer,
    Session, SessionEvent, ThumbnailInput, ThumbnailRenderer,
};
use ym_core::settings::{AutoOr, DevelopSettings, PercentSetting, SettingsPatch};
use ym_export::ExportOptions;
use ym_gpu::queue::CancelToken;

const MODEL: &str = "ILCE-7RM4";
const PROFILE: &str = "sony-ilce-7rm4";

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Steps with the four steps of prompt version 1 set and every other step keep (no colours).
fn steps(exposure: i8, white: i8, contrast: i8, saturation: i8) -> AutoSteps {
    AutoSteps {
        exposure: Some(exposure),
        white: Some(white),
        contrast: Some(contrast),
        saturation: Some(saturation),
        ..AutoSteps::keep()
    }
}

fn scene(level: f64) -> Scene {
    Scene::new(384, 288, light(PROFILE, 5200.0), Layout::Patchwork, level).noise(2e-4, 1e-6)
}

/// Photo number `k`, taken `k` minutes after noon: no two photos of these tests make a burst.
fn write_photo(dir: &Path, name: &str, k: usize) -> PathBuf {
    let s = scene(0.2 + 0.05 * (k % 8) as f64);
    let path = dir.join(name);
    s.builder(MODEL)
        .capture_time(&format!("2026:10:04 12:{:02}:00", k % 60), "00", None)
        .write_to(&path, &s.mosaic())
        .expect("write the photo");
    path
}

/// `n` photos, `DSC00001.ARW` on, minutes apart (no bursts).
fn write_photos(dir: &Path, n: usize) {
    for k in 0..n {
        write_photo(dir, &format!("DSC{:05}.ARW", k + 1), k);
    }
}

// ---------------------------------------------------------------------------------------------
// A scripted advisor and a cheap picture renderer
// ---------------------------------------------------------------------------------------------

/// Holds the advisor's calls until opened; `only` holds the calls about one photo alone, and the
/// calls about the photos in `free` always pass.
struct Gate {
    state: Mutex<(bool, usize)>,
    only: Mutex<Option<u8>>,
    free: Mutex<Vec<u8>>,
    changed: Condvar,
}

impl Gate {
    fn new(open: bool) -> Self {
        Self {
            state: Mutex::new((open, 0)),
            only: Mutex::new(None),
            free: Mutex::new(Vec::new()),
            changed: Condvar::new(),
        }
    }

    /// Lets the calls about these photos pass, held ones included, while the gate stays closed to
    /// the others.
    fn free(&self, photos: &[PhotoId]) {
        lock(&self.free).extend(photos.iter().map(|p| p.0 as u8));
        let _st = lock(&self.state);
        self.changed.notify_all();
    }

    fn close(&self, only: Option<u8>) {
        *lock(&self.only) = only;
        lock(&self.state).0 = false;
    }

    fn open(&self) {
        lock(&self.state).0 = true;
        self.changed.notify_all();
    }

    fn pass(&self, photo: u8) {
        if lock(&self.only).is_some_and(|only| only != photo) {
            return;
        }
        let free = || lock(&self.free).contains(&photo);
        let mut st = lock(&self.state);
        if st.0 || free() {
            return;
        }
        st.1 += 1;
        self.changed.notify_all();
        while !st.0 && !free() {
            st = self
                .changed
                .wait(st)
                .unwrap_or_else(PoisonError::into_inner);
        }
        st.1 -= 1;
    }

    /// Waits until a call is held.
    fn wait_parked(&self) {
        let deadline = Instant::now() + Duration::from_secs(60);
        let mut st = lock(&self.state);
        while st.1 == 0 {
            assert!(Instant::now() < deadline, "no call reached the advisor");
            st = self
                .changed
                .wait_timeout(st, Duration::from_millis(50))
                .unwrap_or_else(PoisonError::into_inner)
                .0;
        }
    }
}

type Reply = Result<AutoSteps, FailureKind>;

/// An advisor that answers what its script says, photo by photo (the photo is the first byte of
/// [`StubPictures`]' picture), records the photos it was asked about and can be held at a gate.
struct Scripted {
    availability: Mutex<Availability>,
    script: Mutex<HashMap<u8, VecDeque<Reply>>>,
    fallback: Mutex<Reply>,
    calls: Mutex<Vec<u8>>,
    /// The instructions of each call, in order.
    instructions: Mutex<Vec<String>>,
    gate: Gate,
}

impl Scripted {
    fn info() -> AdvisorInfo {
        AdvisorInfo::new("Scripted", "27.0 (27A1)")
    }

    fn new(fallback: Reply) -> Arc<Self> {
        Arc::new(Self {
            availability: Mutex::new(Availability::Available(Self::info())),
            script: Mutex::new(HashMap::new()),
            fallback: Mutex::new(fallback),
            calls: Mutex::new(Vec::new()),
            instructions: Mutex::new(Vec::new()),
            gate: Gate::new(true),
        })
    }

    fn answer(&self, photo: PhotoId, replies: impl IntoIterator<Item = Reply>) {
        lock(&self.script)
            .entry(photo.0 as u8)
            .or_default()
            .extend(replies);
    }

    fn set_available(&self, available: bool) {
        *lock(&self.availability) = if available {
            Availability::Available(Self::info())
        } else {
            Availability::Unavailable(UnavailableReason::AppleIntelligenceOff)
        };
    }

    fn calls(&self) -> Vec<u8> {
        lock(&self.calls).clone()
    }

    fn calls_about(&self, photo: PhotoId) -> usize {
        self.calls().iter().filter(|&&p| p == photo.0 as u8).count()
    }
}

impl AutoAdvisor for Scripted {
    fn availability(&self) -> Availability {
        lock(&self.availability).clone()
    }

    fn ask(
        &self,
        picture: &AdvisorPicture,
        instructions: &str,
    ) -> Result<AdvisorAnswer, AdvisorError> {
        let photo = picture.rgb[0];
        lock(&self.calls).push(photo);
        lock(&self.instructions).push(instructions.to_owned());
        self.gate.pass(photo);
        let reply = lock(&self.script)
            .get_mut(&photo)
            .and_then(VecDeque::pop_front)
            .unwrap_or_else(|| *lock(&self.fallback));
        reply
            .map(|steps| AdvisorAnswer {
                steps,
                scene: format!("Photo {photo}"),
                plan: format!("Plan {photo}"),
            })
            .map_err(|kind| AdvisorError::new(kind, "scripted"))
    }
}

/// A picture that only says which photo it is: two pixels of the photo's id.
#[derive(Default)]
struct StubPictures {
    renders: AtomicUsize,
}

impl PictureRenderer for StubPictures {
    fn render(&self, input: &PictureInput<'_>) -> Result<AdvisorPicture, String> {
        self.renders.fetch_add(1, Ordering::SeqCst);
        Ok(AdvisorPicture {
            width: 2,
            height: 1,
            rgb: vec![input.id.0 as u8; 6],
        })
    }
}

/// A thumbnail renderer that renders nothing and counts.
#[derive(Default)]
struct CountingThumbnails {
    calls: AtomicUsize,
}

impl ThumbnailRenderer for CountingThumbnails {
    fn render(&self, _input: &ThumbnailInput<'_>) -> Result<Vec<u8>, String> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(vec![1, 2, 3])
    }
}

struct Rig {
    session: Session,
    events: Receiver<SessionEvent>,
    pictures: Arc<StubPictures>,
    dir: tempfile::TempDir,
}

impl Rig {
    /// A session with the scripted advisor (retries 5 ms apart) and `n` photos written but not
    /// opened yet.
    fn new(advisor: Arc<Scripted>, n: usize, thumbnails: bool) -> Self {
        let dir = tempfile::tempdir().expect("dir");
        write_photos(dir.path(), n);
        Self::in_dir(advisor, dir, thumbnails)
    }

    /// A session with the scripted advisor over the photos already in `dir`.
    fn in_dir(advisor: Arc<Scripted>, dir: tempfile::TempDir, thumbnails: bool) -> Self {
        let pictures = Arc::new(StubPictures::default());
        let (sink, events) = EventSink::channel();
        let mut env = base_env().with_events(sink).with_apple_auto(AppleAutoEnv {
            retry_pause: Duration::from_millis(5),
            ..AppleAutoEnv::new(advisor, pictures.clone())
        });
        if thumbnails {
            env = env.with_thumbnails(Arc::new(CountingThumbnails::default()));
        }
        Self {
            session: Session::new(env),
            events,
            pictures,
            dir,
        }
    }

    fn open(&self) -> Vec<PhotoId> {
        self.session
            .open(&[self.dir.path().to_path_buf()], OpenOptions::default())
            .added
    }

    fn settle(&self) {
        assert!(
            self.session.wait_idle(Duration::from_secs(120)),
            "the session went idle"
        );
    }

    fn state(&self, id: PhotoId) -> AppleAuto {
        self.session.estimates(id).expect("estimates").apple.clone()
    }

    fn drain(&self) -> Vec<SessionEvent> {
        self.events.try_iter().collect()
    }
}

fn changed_ids(events: &[SessionEvent]) -> Vec<PhotoId> {
    events
        .iter()
        .filter_map(|e| match e {
            SessionEvent::EstimatesChanged { id } => Some(*id),
            _ => None,
        })
        .collect()
}

fn thumbnails_of(events: &[SessionEvent], id: PhotoId) -> usize {
    events
        .iter()
        .filter(|e| matches!(e, SessionEvent::ThumbnailReady(t) if t.id == id))
        .count()
}

// ---------------------------------------------------------------------------------------------
// Settings
// ---------------------------------------------------------------------------------------------

#[test]
fn contrast_and_saturation_are_auto_or_manual_and_read_old_numbers() {
    let neutral = ym_looks::library::LookId::neutral();
    // The default is Auto, written as the tagged form.
    let d = DevelopSettings::default();
    assert_eq!((d.contrast, d.saturation), (AutoOr::Auto, AutoOr::Auto));
    let v = serde_json::to_value(&d).unwrap();
    assert_eq!(v["contrast"], json!({ "kind": "auto" }));
    assert_eq!(v["saturation"], json!({ "kind": "auto" }));
    // The new forms.
    let (s, _) = DevelopSettings::from_json(
        r#"{ "contrast": { "kind": "manual", "value": 12 }, "saturation": { "kind": "auto" } }"#,
        &neutral,
    )
    .unwrap();
    assert_eq!(s.contrast, PercentSetting::manual_value(12.0));
    assert_eq!(s.saturation, AutoOr::Auto);
    assert_eq!(
        serde_json::to_value(s.contrast).unwrap(),
        json!({ "kind": "manual", "value": 12.0 })
    );
    // Old documents: a number is Manual, 0 (the old default, also -0) is Auto, and the range clamps.
    for (text, contrast, saturation) in [
        (
            r#"{ "contrast": 10, "saturation": -20 }"#,
            PercentSetting::manual_value(10.0),
            PercentSetting::manual_value(-20.0),
        ),
        (
            r#"{ "contrast": 0, "saturation": -0.0 }"#,
            AutoOr::Auto,
            AutoOr::Auto,
        ),
        (
            r#"{ "contrast": 250, "saturation": { "kind": "manual", "value": -400 } }"#,
            PercentSetting::manual_value(100.0),
            PercentSetting::manual_value(-100.0),
        ),
    ] {
        let (s, _) = DevelopSettings::from_json(text, &neutral).unwrap();
        assert_eq!((s.contrast, s.saturation), (contrast, saturation), "{text}");
    }
    // Wrong shapes are refused.
    for bad in [
        r#"{ "contrast": "high" }"#,
        r#"{ "contrast": { "kind": "manual" } }"#,
        r#"{ "saturation": { "kind": "manual", "value": 1, "ev": 1 } }"#,
        r#"{ "contrast": null }"#,
    ] {
        assert!(DevelopSettings::from_json(bad, &neutral).is_err(), "{bad}");
    }
    // A manual value that is not a number is an error, not a clamp.
    let s = DevelopSettings {
        contrast: PercentSetting::manual_value(f64::NAN),
        ..DevelopSettings::default()
    };
    assert!(s.validated().is_err());
}

#[test]
fn patches_take_either_form_and_reset_is_auto() {
    let mut s = DevelopSettings::default();
    for (text, want) in [
        (
            r#"{ "contrast": 30, "saturation": { "kind": "manual", "value": -10 } }"#,
            (
                PercentSetting::manual_value(30.0),
                PercentSetting::manual_value(-10.0),
            ),
        ),
        (
            r#"{ "contrast": { "kind": "auto" } }"#,
            (AutoOr::Auto, PercentSetting::manual_value(-10.0)),
        ),
        (r#"{ "saturation": 0 }"#, (AutoOr::Auto, AutoOr::Auto)),
        (
            r#"{ "contrast": 140 }"#,
            (PercentSetting::manual_value(100.0), AutoOr::Auto),
        ),
    ] {
        let patch: SettingsPatch = serde_json::from_str(text).unwrap();
        s.apply_patch(&patch, None).unwrap();
        assert_eq!((s.contrast, s.saturation), want, "{text}");
    }
    // A patch writes the tagged form and an absent field is no change.
    let patch: SettingsPatch = serde_json::from_str(r#"{ "contrast": 5 }"#).unwrap();
    assert_eq!(
        serde_json::to_value(&patch).unwrap(),
        json!({ "contrast": { "kind": "manual", "value": 5.0 } })
    );
    let none: SettingsPatch = serde_json::from_str(r#"{ "contrast": null }"#).unwrap();
    assert!(none.is_empty());
    // Re-run resets both with exposure and white while Apple Auto acts on the photo, and leaves
    // them alone otherwise.
    let edited = DevelopSettings {
        contrast: PercentSetting::manual_value(10.0),
        saturation: PercentSetting::manual_value(-10.0),
        exposure: ym_core::settings::ExposureSetting::manual_ev(1.0),
        ..DevelopSettings::default()
    };
    let mut s = edited.clone();
    s.reset_automatic(true);
    assert!(s.contrast.is_auto() && s.saturation.is_auto() && s.exposure.is_auto());
    let mut s = edited.clone();
    s.reset_automatic(false);
    assert!(s.exposure.is_auto());
    assert_eq!(
        (s.contrast, s.saturation),
        (edited.contrast, edited.saturation)
    );
}

#[test]
fn auto_contrast_and_saturation_resolve_and_pin_to_the_estimates_values() {
    let dir = tempfile::tempdir().unwrap();
    let (_, est, profile) = estimates_of_file(&scene(0.3).write(dir.path(), "a.ARW", MODEL));
    // Without Apple Auto they resolve to 0, the old default.
    let r = ym_core::settings::ResolvedSettings::resolve(
        PhotoId(1),
        &DevelopSettings::default(),
        &est,
        &profile,
        None,
    )
    .unwrap();
    assert_eq!((r.values.contrast, r.values.saturation), (0.0, 0.0));
    assert!(r.auto.contrast && r.auto.saturation);
    let pinned = r.pinned_settings();
    assert_eq!(pinned.contrast, PercentSetting::manual_value(0.0));
    assert_eq!(pinned.saturation, PercentSetting::manual_value(0.0));
    // With an answer they resolve to the table's values; a manual value wins.
    let answered = est.with_apple(
        Some(steps(0, 0, 2, -1)),
        AppleAuto::Answered {
            answer: AdvisorAnswer {
                steps: steps(0, 0, 2, -1),
                scene: String::new(),
                plan: String::new(),
            },
            info: Scripted::info(),
        },
        tables_of(&profile),
    );
    let s = DevelopSettings {
        saturation: PercentSetting::manual_value(7.0),
        ..DevelopSettings::default()
    };
    let r = ym_core::settings::ResolvedSettings::resolve(PhotoId(1), &s, &answered, &profile, None)
        .unwrap();
    assert_eq!((r.values.contrast, r.values.saturation), (30.0, 7.0));
    assert!(r.auto.contrast && !r.auto.saturation);
    assert_eq!(
        r.pinned_settings().contrast,
        PercentSetting::manual_value(30.0)
    );
    assert_eq!(
        r.pinned_settings().saturation,
        PercentSetting::manual_value(7.0)
    );
    // The grade gets the resolved values.
    let g = ym_core::pipeline::color::grade_params(&s, &answered);
    assert_eq!((g.contrast, g.saturation), (30.0, 7.0));
    // And the summary says so.
    assert_eq!(r.estimates.source, AutoSource::Apple);
    assert_eq!(
        (r.estimates.contrast, r.estimates.saturation),
        (30.0, -15.0)
    );
}

// ---------------------------------------------------------------------------------------------
// The estimates
// ---------------------------------------------------------------------------------------------

#[test]
fn estimates_without_an_answer_are_the_rule_bit_for_bit() {
    let dir = tempfile::tempdir().unwrap();
    let (_, est, profile) = estimates_of_file(&scene(0.3).write(dir.path(), "a.ARW", MODEL));
    assert_eq!(est.apple, AppleAuto::Off);
    assert_eq!(est.source(), AutoSource::Rule);
    assert_eq!((est.contrast, est.saturation), (0.0, 0.0));
    assert_eq!(est.color.white_shift_mired, 0.0);
    // Failed, pending or unavailable: only the state differs.
    for state in [
        AppleAuto::Pending,
        AppleAuto::Unavailable {
            reason: UnavailableReason::NotAMac,
        },
        AppleAuto::Failed {
            kind: FailureKind::Timeout,
            message: String::new(),
            info: Scripted::info(),
        },
    ] {
        let e = est.with_apple(None, state.clone(), tables_of(&profile));
        assert_eq!(e.apple, state);
        assert_eq!(format!("{:?}", e.color), format!("{:?}", est.color));
        assert_eq!(e.exposure, est.exposure);
        assert_eq!(e.readout, est.readout);
        assert_eq!(e.source(), AutoSource::Rule);
        // Keep on every step is Apple's source and the rule's numbers.
        let keep = est.with_apple(Some(AutoSteps::default()), state, tables_of(&profile));
        assert_eq!(keep.source(), AutoSource::Apple);
        assert_eq!(keep.exposure, est.exposure);
        assert_eq!(keep.readout, est.readout);
        assert_eq!(keep.color.white_shift_mired, 0.0);
    }
}

#[test]
fn an_answer_moves_exposure_white_contrast_and_saturation() {
    let dir = tempfile::tempdir().unwrap();
    let (_, est, profile) = estimates_of_file(&scene(0.3).write(dir.path(), "a.ARW", MODEL));
    let warmer = est.with_apple(
        Some(steps(2, 1, 1, -2)),
        AppleAuto::Pending,
        tables_of(&profile),
    );
    assert!((warmer.exposure.ev - (est.exposure.ev + 0.6)).abs() < 1e-12);
    assert_eq!(warmer.color.ev_auto, warmer.exposure.ev);
    assert_eq!(warmer.exposure.own_ev, est.exposure.own_ev);
    assert_eq!(warmer.color.white_shift_mired, -10.0);
    assert_eq!((warmer.contrast, warmer.saturation), (15.0, -30.0));
    // The readout shows the nudged white: a warmer picture is a higher Kelvin.
    assert!(
        warmer.readout.kelvin_k > est.readout.kelvin_k,
        "{:?} vs {:?}",
        warmer.readout,
        est.readout
    );
    let cooler = est.with_apple(
        Some(steps(0, -2, 0, 0)),
        AppleAuto::Pending,
        tables_of(&profile),
    );
    assert!(cooler.readout.kelvin_k < est.readout.kelvin_k);
    // The white lift is not sized again.
    assert_eq!(warmer.white, est.white);
    // The summary.
    let s = warmer.summary();
    assert_eq!(s.kelvin, warmer.readout.kelvin_k);
    assert_eq!(s.exposure_ev, warmer.exposure.ev);
    let v = serde_json::to_value(&s).unwrap();
    assert_eq!(v["source"], json!("apple"));
    assert_eq!(v["apple"], json!({ "state": "pending" }));
    assert_eq!(
        (v["contrast"].clone(), v["saturation"].clone()),
        (json!(15.0), json!(-30.0))
    );
}

#[test]
fn an_answer_moves_highlights_shadows_black_the_tint_and_the_wheels() {
    use ym_core::apple_auto::{HighlightColor, ShadowColor};
    let dir = tempfile::tempdir().unwrap();
    let (_, est, profile) = estimates_of_file(&scene(0.3).write(dir.path(), "a.ARW", MODEL));
    let graded = AutoSteps {
        highlights: Some(-2),
        shadows: Some(1),
        blacks: Some(-1),
        tint: Some(2),
        shadow_color: Some(ShadowColor::Teal),
        shadow_color_strength: Some(2),
        highlight_color: Some(HighlightColor::Warm),
        highlight_color_strength: Some(1),
        ..AutoSteps::default()
    };
    let e = est.with_apple(Some(graded), AppleAuto::Pending, tables_of(&profile));
    assert_eq!((e.highlights, e.shadows, e.black), (-40.0, 15.0, -15.0));
    assert_eq!(e.balance_shadows, [-0.866 * 0.55, -0.5 * 0.55]);
    assert_eq!(e.balance_highlights, [0.866 * 0.3, 0.5 * 0.3]);
    // The tint nudge: more magenta reads a higher tint, the Kelvin and the exposure stay.
    assert_eq!(e.color.tint_shift, 10.0);
    assert_eq!(e.color.white_shift_mired, 0.0);
    assert!(
        (e.readout.tint - (est.readout.tint + 10.0).min(50.0)).abs() < 0.5,
        "{:?} vs {:?}",
        e.readout,
        est.readout
    );
    assert_eq!(e.readout.kelvin_k, est.readout.kelvin_k);
    assert_eq!(e.exposure, est.exposure);
    // Unset steps keep the values without Apple Auto.
    assert_eq!((e.contrast, e.saturation), (0.0, 0.0));
    // The settings resolve against them; a manual value wins; the grade takes them.
    let settings = DevelopSettings::default();
    let r = ym_core::settings::ResolvedSettings::resolve(PhotoId(1), &settings, &e, &profile, None)
        .unwrap();
    assert_eq!(
        (r.values.highlights, r.values.shadows, r.values.black),
        (-40.0, 15.0, -15.0)
    );
    assert_eq!(r.values.color_balance.shadows, e.balance_shadows);
    assert_eq!(r.values.color_balance.highlights, e.balance_highlights);
    assert_eq!(r.values.color_balance.midtones, [0.0, 0.0]);
    assert!(r.auto.highlights && r.auto.balance_shadows && r.auto.balance_highlights);
    let g = ym_core::pipeline::color::grade_params(&settings, &e);
    assert_eq!((g.highlights, g.shadows, g.black), (-40.0, 15.0, -15.0));
    assert_eq!(
        [g.balance_shadows.a, g.balance_shadows.b],
        e.balance_shadows
    );
    let mut manual = settings.clone();
    manual.highlights = PercentSetting::manual_value(5.0);
    manual.color_balance.shadows = ym_core::settings::BalanceSetting::manual_point([0.1, 0.2]);
    let g = ym_core::pipeline::color::grade_params(&manual, &e);
    assert_eq!(g.highlights, 5.0);
    assert_eq!([g.balance_shadows.a, g.balance_shadows.b], [0.1, 0.2]);
    // Pinned settings record the numbers and points they resolved to.
    let pinned = r.pinned_settings();
    assert_eq!(pinned.highlights, PercentSetting::manual_value(-40.0));
    assert_eq!(
        pinned.color_balance.shadows,
        ym_core::settings::BalanceSetting::manual_point(e.balance_shadows)
    );
    // The summary carries the Auto values and the steps in effect.
    let v = serde_json::to_value(e.summary()).unwrap();
    assert_eq!(
        (
            v["highlights"].clone(),
            v["shadows"].clone(),
            v["black"].clone()
        ),
        (json!(-40.0), json!(15.0), json!(-15.0))
    );
    assert_eq!(v["apple_steps"]["highlights"], json!(-2));
    assert_eq!(v["apple_steps"]["exposure"], json!(null));
    assert_eq!(v["apple_steps"]["shadow_color"], json!("teal"));
}

#[test]
fn steps_the_model_left_empty_set_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let (_, est, profile) = estimates_of_file(&scene(0.3).write(dir.path(), "a.ARW", MODEL));
    // Only the contrast was filled: every other value is the rule's, bit for bit.
    let only = AutoSteps {
        contrast: Some(-1),
        ..AutoSteps::default()
    };
    let e = est.with_apple(Some(only), AppleAuto::Pending, tables_of(&profile));
    assert_eq!(e.contrast, -15.0);
    assert_eq!(e.exposure, est.exposure);
    assert_eq!(format!("{:?}", e.color), format!("{:?}", est.color));
    assert_eq!(e.readout, est.readout);
    assert_eq!(
        (e.highlights, e.shadows, e.black, e.saturation),
        (0.0, 0.0, 0.0, 0.0)
    );
    assert_eq!(
        (e.balance_shadows, e.balance_highlights),
        ([0.0; 2], [0.0; 2])
    );
    assert_eq!(e.source(), AutoSource::Apple);
}

#[test]
fn shadow_and_highlight_colours_take_the_named_hue_through_the_grade() {
    use ym_color::grade::{ColorBalance, Grade, GradeParams, logc4_at_stops};
    use ym_color::logc::LogC4;
    use ym_core::apple_auto::{HighlightColor, ShadowColor};
    // The hue in degrees of a small offset from grey (HSV's hue of grey plus the offset).
    fn hue(d: [f64; 3]) -> f64 {
        let (r, g, b) = (d[0], d[1], d[2]);
        let max = r.max(g).max(b);
        let min = r.min(g).min(b);
        let c = max - min;
        assert!(c > 1e-4, "no colour: {d:?}");
        let h = if max == r {
            60.0 * ((g - b) / c)
        } else if max == g {
            60.0 * ((b - r) / c + 2.0)
        } else {
            60.0 * ((r - g) / c + 4.0)
        };
        (h + 360.0) % 360.0
    }
    // A neutral grey ramp through C9 with the wheel at the colour's point: the offset at the
    // zone's centre (3 stops below or above grey) has the colour's hue, within 5 degrees: teal
    // between cyan and blue, blue, green, amber (orange), magenta; warm (orange), gold (yellow),
    // cool (blue), pink (between magenta and red).
    let check = |shadows: [f64; 2], highlights: [f64; 2], stops: f64, want: f64, name: &str| {
        let grade = Grade::new(&GradeParams {
            balance_shadows: ColorBalance {
                a: shadows[0],
                b: shadows[1],
            },
            balance_highlights: ColorBalance {
                a: highlights[0],
                b: highlights[1],
            },
            ..GradeParams::default()
        });
        for s in [-6.0, -3.0, 0.0, 3.0, 6.0] {
            let l = logc4_at_stops(s);
            let out = grade.apply(LogC4([l, l, l]), l).0;
            let d = [out[0] - l, out[1] - l, out[2] - l];
            if s == stops {
                let h = hue(d);
                let off = ((h - want + 540.0) % 360.0 - 180.0).abs();
                assert!(off < 5.0, "{name}: hue {h} at {s} stops, wanted {want}");
            }
        }
    };
    let point = |d: [f64; 2]| [0.55 * d[0], 0.55 * d[1]];
    for (c, want) in [
        (ShadowColor::Teal, 210.0),
        (ShadowColor::Blue, 240.0),
        (ShadowColor::Green, 120.0),
        (ShadowColor::Amber, 30.0),
        (ShadowColor::Magenta, 300.0),
    ] {
        check(point(c.direction()), [0.0; 2], -3.0, want, c.name());
    }
    for (c, want) in [
        (HighlightColor::Warm, 30.0),
        (HighlightColor::Gold, 60.0),
        (HighlightColor::Cool, 240.0),
        (HighlightColor::Pink, 330.0),
    ] {
        check([0.0; 2], point(c.direction()), 3.0, want, c.name());
    }
}

// ---------------------------------------------------------------------------------------------
// The worker
// ---------------------------------------------------------------------------------------------

#[test]
fn the_viewer_never_waits_and_the_answer_arrives_with_estimates_changed() {
    let advisor = Scripted::new(Ok(steps(1, 1, 1, -1)));
    advisor.gate.close(None);
    let rig = Rig::new(advisor.clone(), 1, false);
    let id = rig.open()[0];
    advisor.gate.wait_parked();
    // The advisor is held: the photo is served with the rule, at once.
    let started = Instant::now();
    let before = rig.session.view(id).expect("view");
    let resolved = rig.session.settings(id).expect("settings");
    assert!(started.elapsed() < Duration::from_secs(2));
    assert_eq!(before.estimates.apple, AppleAuto::Pending);
    assert_eq!(before.estimates.source(), AutoSource::Rule);
    assert_eq!(resolved.values.contrast, 0.0);
    assert!(!rig.session.estimates_final(id).unwrap(), "an export waits");
    assert!(
        !rig.session.wait_idle(Duration::from_millis(200)),
        "`ym` waits for the answer"
    );
    rig.drain();

    advisor.gate.open();
    rig.settle();
    let after = rig.session.view(id).expect("view");
    assert!(matches!(after.estimates.apple, AppleAuto::Answered { .. }));
    assert_eq!(after.estimates.source(), AutoSource::Apple);
    assert!((after.estimates.exposure.ev - (before.estimates.exposure.ev + 0.3)).abs() < 1e-12);
    assert_eq!(after.estimates.color.white_shift_mired, -10.0);
    assert!(after.estimates.readout.kelvin_k > before.estimates.readout.kelvin_k);
    assert!(after.estimates_version > before.estimates_version);
    assert!(rig.session.estimates_final(id).unwrap());
    let r = rig.session.settings(id).unwrap();
    assert_eq!((r.values.contrast, r.values.saturation), (15.0, -15.0));
    assert_eq!(r.values.exposure_ev, after.estimates.exposure.ev);
    assert!(r.auto.exposure && r.auto.contrast && r.auto.saturation);
    assert_eq!(changed_ids(&rig.drain()), vec![id]);
    // One request, one picture (the development job's).
    assert_eq!(advisor.calls(), vec![id.0 as u8]);
    assert_eq!(rig.pictures.renders.load(Ordering::SeqCst), 1);
    assert_eq!(rig.session.apple_auto_asks(), 1);

    // The rendered picture follows: the same photo before and after differs.
    let renderer = Renderer::neutral_only().expect("renderer");
    let raw = rig.session.raw_image(id).unwrap();
    let thumb = |v: &ym_core::session::PhotoView| {
        renderer
            .thumbnail_png(&PhotoSource::from_view(v, raw.clone()), &CancelToken::new())
            .unwrap()
    };
    let (a, b) = (thumb(&before), thumb(&after));
    renderer.forget(id);
    assert_ne!(a, b);
}

#[test]
fn the_photo_on_screen_goes_first_then_an_exports_photos_then_the_others_in_order() {
    let advisor = Scripted::new(Ok(AutoSteps::keep()));
    advisor.set_available(false);
    let rig = Rig::new(advisor.clone(), 4, false);
    let ids = rig.open();
    rig.settle();
    // Not available: nobody is asked, every photo says why and is final.
    assert!(advisor.calls().is_empty());
    for id in &ids {
        assert_eq!(
            rig.state(*id),
            AppleAuto::Unavailable {
                reason: UnavailableReason::AppleIntelligenceOff
            }
        );
        assert!(rig.session.estimates_final(*id).unwrap());
    }
    rig.session.set_current(ids[2]).unwrap();
    rig.session.prioritise_apple_auto(&[ids[3], ids[0]]);
    advisor.set_available(true);
    rig.session.refresh_apple_auto();
    rig.settle();
    let want: Vec<u8> = [ids[2], ids[3], ids[0], ids[1]]
        .iter()
        .map(|id| id.0 as u8)
        .collect();
    assert_eq!(advisor.calls(), want);
    // The worker rendered each picture itself (they were not rendered while it was unavailable).
    assert_eq!(rig.pictures.renders.load(Ordering::SeqCst), 4);
    for id in &ids {
        assert!(matches!(rig.state(*id), AppleAuto::Answered { .. }));
    }
}

#[test]
fn failures_use_the_rule_and_say_why_and_only_waiting_is_retried() {
    let advisor = Scripted::new(Ok(AutoSteps::keep()));
    let rig = Rig::new(advisor.clone(), 6, false);
    // The ids are given in filmstrip order, from 1.
    let id = |k: u32| PhotoId(k);
    advisor.answer(id(1), [Err(FailureKind::Timeout)]);
    advisor.answer(id(2), [Err(FailureKind::Refused)]);
    advisor.answer(id(3), [Ok(steps(4, 0, 0, 0))]);
    advisor.answer(id(4), [Err(FailureKind::RateLimited); 3]);
    advisor.answer(
        id(5),
        [Err(FailureKind::RateLimited), Ok(steps(1, 0, 0, 0))],
    );
    advisor.answer(
        id(6),
        [
            Err(FailureKind::ConcurrentRequests),
            Err(FailureKind::ConcurrentRequests),
            Ok(steps(-1, 0, 0, 0)),
        ],
    );
    let ids = rig.open();
    assert_eq!(ids, (1..=6).map(id).collect::<Vec<_>>());
    rig.settle();
    let failed = |k: u32| match rig.state(id(k)) {
        AppleAuto::Failed { kind, info, .. } => {
            assert_eq!(info, Scripted::info());
            Some(kind)
        }
        _ => None,
    };
    assert_eq!(failed(1), Some(FailureKind::Timeout));
    assert_eq!(failed(2), Some(FailureKind::Refused));
    assert_eq!(failed(3), Some(FailureKind::BadAnswer));
    assert_eq!(failed(4), Some(FailureKind::RateLimited));
    assert!(matches!(rig.state(id(5)), AppleAuto::Answered { .. }));
    assert!(matches!(rig.state(id(6)), AppleAuto::Answered { .. }));
    // Asked once, except the waiting kinds: twice more at most.
    let asks: Vec<usize> = (1..=6).map(|k| advisor.calls_about(id(k))).collect();
    assert_eq!(asks, vec![1, 1, 1, 3, 2, 3]);
    // A failed photo is the rule, and final.
    for k in 1..=4 {
        let e = rig.session.estimates(id(k)).unwrap();
        assert_eq!(e.source(), AutoSource::Rule);
        assert_eq!(
            (e.contrast, e.saturation, e.color.white_shift_mired),
            (0.0, 0.0, 0.0)
        );
        assert_eq!(e.exposure.ev, e.exposure.own_ev);
        assert!(rig.session.estimates_final(id(k)).unwrap());
    }
    let five = rig.session.estimates(id(5)).unwrap();
    assert!((five.exposure.ev - (five.exposure.own_ev + 0.3)).abs() < 1e-12);
    // The documented pause between tries.
    let default = AppleAutoEnv::new(advisor.clone(), Arc::new(StubPictures::default()));
    assert_eq!(default.retry_pause, Duration::from_secs(2));
}

#[test]
fn a_burst_shares_the_median_of_its_answered_steps_and_waits_for_every_frame() {
    let dir = tempfile::tempdir().unwrap();
    let frame = |name: &str, sec: &str, level: f64| {
        let s = scene(level);
        s.builder(MODEL)
            .capture_time(&format!("2026:10:04 12:03:{sec}"), "00", None)
            .write_to(&dir.path().join(name), &s.mosaic())
            .unwrap();
    };
    frame("a.ARW", "10", 0.10);
    frame("b.ARW", "10", 0.20);
    frame("c.ARW", "11", 0.40);
    frame("d.ARW", "40", 0.10);
    let advisor = Scripted::new(Ok(AutoSteps::keep()));
    advisor.set_available(false);
    let pictures = Arc::new(StubPictures::default());
    let session = Session::new(base_env().with_apple_auto(AppleAutoEnv {
        retry_pause: Duration::from_millis(5),
        ..AppleAutoEnv::new(advisor.clone(), pictures)
    }));
    let ids = session
        .open(&[dir.path().to_path_buf()], OpenOptions::default())
        .added;
    assert!(session.wait_idle(Duration::from_secs(120)));
    let (a, b, c, d) = (ids[0], ids[1], ids[2], ids[3]);
    advisor.answer(a, [Ok(steps(1, 1, 2, 0))]);
    advisor.answer(b, [Ok(steps(3, -1, -2, 1))]);
    advisor.answer(c, [Err(FailureKind::Refused)]);
    advisor.answer(d, [Ok(steps(-1, 0, 0, 0))]);
    // Hold c: a and b answer first (a is on screen, then filmstrip order).
    advisor.gate.close(Some(c.0 as u8));
    advisor.set_available(true);
    session.refresh_apple_auto();
    advisor.gate.wait_parked();
    assert!(matches!(
        session.estimates(a).unwrap().apple,
        AppleAuto::Answered { .. }
    ));
    // a has its answer, but its burst does not: an export of a waits for c.
    assert!(!session.estimates_final(a).unwrap());
    advisor.gate.open();
    assert!(session.wait_idle(Duration::from_secs(60)));
    for id in &ids {
        assert!(session.estimates_final(*id).unwrap());
    }
    // a, b, c: field by field, the lower middle of a's and b's steps; c failed and is left out of
    // the median, but takes the burst's steps.
    let shared = steps(1, -1, -2, 0);
    for id in [a, b, c] {
        let e = session.estimates(id).unwrap();
        assert_eq!(e.apple_steps, Some(shared), "{id}");
        assert_eq!(e.source(), AutoSource::Apple);
        assert!((e.exposure.ev - (e.exposure.own_ev + 0.3)).abs() < 1e-12);
        assert_eq!(e.color.white_shift_mired, 10.0);
        assert_eq!(e.contrast, -30.0);
    }
    assert!(matches!(
        session.estimates(c).unwrap().apple,
        AppleAuto::Failed {
            kind: FailureKind::Refused,
            ..
        }
    ));
    // d is alone.
    let e = session.estimates(d).unwrap();
    assert_eq!(e.apple_steps, Some(steps(-1, 0, 0, 0)));
    assert!((e.exposure.ev - (e.exposure.own_ev - 0.3)).abs() < 1e-12);
}

#[test]
fn the_preference_switches_both_ways_without_asking_again() {
    let advisor = Scripted::new(Ok(steps(1, 0, 1, 0)));
    let rig = Rig::new(advisor.clone(), 1, false);
    let id = rig.open()[0];
    rig.settle();
    let answered = rig.session.estimates(id).unwrap();
    assert_eq!(answered.source(), AutoSource::Apple);
    rig.drain();
    // Off: back to the rule at once, the viewers told, nothing asked.
    rig.session.set_apple_auto_preference(false);
    let off = rig.session.estimates(id).unwrap();
    assert_eq!(off.apple, AppleAuto::Off);
    assert_eq!(off.source(), AutoSource::Rule);
    assert_eq!(off.exposure.ev, off.exposure.own_ev);
    assert_eq!(off.contrast, 0.0);
    assert_eq!(changed_ids(&rig.drain()), vec![id]);
    assert!(!rig.session.apple_auto_status().active);
    assert!(rig.session.estimates_final(id).unwrap());
    // A photo opened while it is off is not asked.
    let extra = tempfile::tempdir().unwrap();
    write_photo(extra.path(), "e.ARW", 30);
    let e = rig
        .session
        .open(&[extra.path().to_path_buf()], OpenOptions::default())
        .added[0];
    rig.settle();
    assert_eq!(rig.state(e), AppleAuto::Off);
    assert_eq!(advisor.calls().len(), 1);
    // On: the answer kept is used again, and only the photo without one is asked.
    rig.session.set_apple_auto_preference(true);
    rig.settle();
    assert_eq!(
        rig.session.estimates(id).unwrap().exposure,
        answered.exposure
    );
    assert!(matches!(rig.state(e), AppleAuto::Answered { .. }));
    assert_eq!(advisor.calls(), vec![id.0 as u8, e.0 as u8]);
    // The run's mode overrides the preference.
    rig.session.set_apple_auto_preference(false);
    rig.session.set_apple_auto_mode(AppleAutoMode::On);
    assert!(rig.session.apple_auto_active());
    assert_eq!(
        rig.session.estimates(id).unwrap().source(),
        AutoSource::Apple
    );
    rig.session.set_apple_auto_mode(AppleAutoMode::Off);
    assert_eq!(rig.state(id), AppleAuto::Off);
}

#[test]
fn rerun_asks_again_and_resets_contrast_and_saturation() {
    let advisor = Scripted::new(Ok(steps(0, 0, 1, 1)));
    let rig = Rig::new(advisor.clone(), 1, false);
    let id = rig.open()[0];
    rig.settle();
    rig.session
        .update_settings(
            id,
            &SettingsPatch {
                contrast: Some(PercentSetting::manual_value(40.0)),
                saturation: Some(PercentSetting::manual_value(-5.0)),
                ..SettingsPatch::default()
            },
        )
        .unwrap();
    advisor.answer(id, [Ok(steps(0, 0, -1, 0))]);
    let r = rig.session.rerun_auto(id).unwrap();
    assert!(r.settings.contrast.is_auto() && r.settings.saturation.is_auto());
    rig.settle();
    assert_eq!(advisor.calls_about(id), 2);
    let r = rig.session.settings(id).unwrap();
    assert_eq!((r.values.contrast, r.values.saturation), (-15.0, 0.0));

    // While Apple Auto does not act on the photo (the preference off, or not available), the
    // Automatic card has no Contrast and Saturation rows and Re-run leaves them as they are, as
    // before Apple Auto; exposure still goes back to Auto.
    let edit = SettingsPatch {
        exposure: Some(ym_core::settings::ExposureSetting::manual_ev(1.0)),
        contrast: Some(PercentSetting::manual_value(40.0)),
        saturation: Some(PercentSetting::manual_value(-5.0)),
        ..SettingsPatch::default()
    };
    let kept = |r: &ym_core::settings::ResolvedSettings| {
        assert!(r.auto.exposure);
        assert_eq!(r.settings.contrast, PercentSetting::manual_value(40.0));
        assert_eq!(r.settings.saturation, PercentSetting::manual_value(-5.0));
        assert_eq!((r.values.contrast, r.values.saturation), (40.0, -5.0));
    };
    rig.session.set_apple_auto_preference(false);
    assert_eq!(rig.state(id), AppleAuto::Off);
    rig.session.update_settings(id, &edit).unwrap();
    kept(&rig.session.rerun_auto(id).unwrap());
    rig.session.set_apple_auto_preference(true);
    advisor.set_available(false);
    rig.session.refresh_apple_auto();
    assert!(matches!(rig.state(id), AppleAuto::Unavailable { .. }));
    rig.session.update_settings(id, &edit).unwrap();
    kept(&rig.session.rerun_auto(id).unwrap());
    // Available again, the photo is waiting for its new answer (pending): Re-run resets them.
    let asked = advisor.calls().len();
    advisor.gate.close(None);
    advisor.set_available(true);
    rig.session.refresh_apple_auto();
    advisor.gate.wait_parked();
    assert_eq!(rig.state(id), AppleAuto::Pending);
    rig.session.update_settings(id, &edit).unwrap();
    let r = rig.session.rerun_auto(id).unwrap();
    assert!(r.settings.contrast.is_auto() && r.settings.saturation.is_auto());
    advisor.gate.open();
    rig.settle();
    assert!(advisor.calls().len() > asked);
}

#[test]
fn copy_and_paste_keep_auto_so_each_photo_keeps_its_own_answer() {
    let advisor = Scripted::new(Ok(AutoSteps::keep()));
    let rig = Rig::new(advisor.clone(), 2, false);
    advisor.answer(PhotoId(1), [Ok(steps(0, 0, 2, 0))]);
    advisor.answer(PhotoId(2), [Ok(steps(0, 0, -1, 0))]);
    let ids = rig.open();
    rig.settle();
    rig.session
        .update_settings(
            ids[0],
            &SettingsPatch {
                saturation: Some(PercentSetting::manual_value(12.0)),
                ..SettingsPatch::default()
            },
        )
        .unwrap();
    let patch = rig.session.copy_settings(ids[0]).unwrap();
    assert_eq!(patch.contrast, Some(AutoOr::Auto));
    assert_eq!(patch.saturation, Some(PercentSetting::manual_value(12.0)));
    rig.session.paste_settings(&[ids[1]], &patch).unwrap();
    let b = rig.session.settings(ids[1]).unwrap();
    assert_eq!((b.values.contrast, b.values.saturation), (-15.0, 12.0));
}

#[test]
fn an_answer_that_moves_the_values_renders_the_thumbnail_again_and_keep_does_not() {
    let advisor = Scripted::new(Ok(AutoSteps::keep()));
    let rig = Rig::new(advisor.clone(), 2, true);
    advisor.answer(PhotoId(1), [Ok(steps(1, 0, 0, 0))]);
    let ids = rig.open();
    rig.settle();
    let events = rig.drain();
    assert_eq!(thumbnails_of(&events, ids[0]), 2, "the first and a new one");
    assert_eq!(thumbnails_of(&events, ids[1]), 1, "keep changes no pixel");
    // Both say the state changed (the Automatic card's line follows).
    let changed = changed_ids(&events);
    assert!(changed.contains(&ids[0]) && changed.contains(&ids[1]));
}

#[test]
fn closing_the_session_stops_the_worker_and_drops_the_answer() {
    let advisor = Scripted::new(Ok(steps(1, 0, 0, 0)));
    advisor.gate.close(None);
    let rig = Rig::new(advisor.clone(), 1, false);
    rig.open();
    advisor.gate.wait_parked();
    rig.session.close();
    rig.drain();
    advisor.gate.open();
    // The request in flight ends; nothing reaches the closed session.
    std::thread::sleep(Duration::from_millis(300));
    assert!(changed_ids(&rig.drain()).is_empty());
    // The session opens again and the worker asks about the new photos.
    let id = rig.open()[0];
    rig.settle();
    assert!(matches!(rig.state(id), AppleAuto::Answered { .. }));
    // Dropping a session whose worker is idle is quick.
    let started = Instant::now();
    drop(rig);
    assert!(started.elapsed() < Duration::from_secs(5));
}

#[test]
fn without_an_advisor_there_is_no_worker_and_nothing_changes() {
    let (session, _rx) = session_with_events();
    let dir = tempfile::tempdir().unwrap();
    write_photos(dir.path(), 2);
    let ids = session
        .open(&[dir.path().to_path_buf()], OpenOptions::default())
        .added;
    settle(&session);
    for id in ids {
        let e = session.estimates(id).unwrap();
        assert_eq!(e.apple, AppleAuto::Off);
        assert_eq!(e.source(), AutoSource::Rule);
        assert!(session.estimates_final(id).unwrap());
    }
    let status = session.apple_auto_status();
    assert!(!status.available && !status.active && status.reason.is_none());
    assert_eq!(session.apple_auto_asks(), 0);
}

// ---------------------------------------------------------------------------------------------
// The picture
// ---------------------------------------------------------------------------------------------

#[test]
fn the_picture_is_768_px_upright_srgb_and_the_same_every_time() {
    let dir = tempfile::tempdir().unwrap();
    let big = Scene::new(1600, 1066, light(PROFILE, 4300.0), Layout::Patchwork, 0.3)
        .noise(2e-4, 1e-6)
        .seed(3);
    let path = dir.path().join("big.ARW");
    big.builder(MODEL)
        .orientation(6)
        .write_to(&path, &big.mosaic())
        .unwrap();
    let small = scene(0.3).write(dir.path(), "small.ARW", MODEL);
    let env = base_env();
    let pictures =
        AdvisorPictures::new(env.reference_look.clone(), &RenderOptions::default()).unwrap();
    let picture_of = |path: &Path, id: u32| {
        let (raw, est, profile) = estimates_of_file(path);
        let camera = ym_core::camera::PhotoCamera::new(&env.profiles, &raw.meta, None);
        let (raw, est) = (Arc::new(raw), Arc::new(est));
        let not = || false;
        pictures
            .picture(&PictureInput {
                id: PhotoId(id),
                raw: &raw,
                camera: &camera,
                profile: &profile,
                estimates: &est,
                cancelled: &not,
            })
            .unwrap()
    };
    let a = picture_of(&path, 1);
    // The camera turned it a quarter: the picture is upright, 768 px on its long edge.
    assert_eq!(a.height, PICTURE_LONG_EDGE_PX, "{} x {}", a.width, a.height);
    assert!((500..=520).contains(&a.width), "{} x {}", a.width, a.height);
    assert_eq!(a.rgb.len(), a.width * a.height * 3);
    // Deterministic: the same bytes again (and under another id).
    assert_eq!(picture_of(&path, 1), a);
    assert_eq!(picture_of(&path, 7), a);
    // A smaller photo is not enlarged.
    let s = picture_of(&small, 2);
    assert!(
        s.width <= 384 && s.height <= 288 && s.width > 370,
        "{} x {}",
        s.width,
        s.height
    );
    // It is a picture: not black, not one flat colour.
    let mean = a.rgb.iter().map(|v| f64::from(*v)).sum::<f64>() / a.rgb.len() as f64;
    assert!(mean > 20.0 && mean < 235.0, "{mean}");
    assert!(a.rgb.iter().any(|v| *v != a.rgb[0]));
    // Cancelled before it starts.
    let (raw, est, profile) = estimates_of_file(&small);
    let camera = ym_core::camera::PhotoCamera::new(&env.profiles, &raw.meta, None);
    let yes = || true;
    assert!(
        pictures
            .picture(&PictureInput {
                id: PhotoId(3),
                raw: &Arc::new(raw),
                camera: &camera,
                profile: &profile,
                estimates: &Arc::new(est),
                cancelled: &yes,
            })
            .is_err()
    );
}

// ---------------------------------------------------------------------------------------------
// The core: status, preference, exports
// ---------------------------------------------------------------------------------------------

struct CoreRig {
    core: Core,
    events: Receiver<ExportEvent>,
    _app: tempfile::TempDir,
    _home: tempfile::TempDir,
    photos: tempfile::TempDir,
}

fn core_rig(advisor: Option<Arc<dyn AutoAdvisor>>, n: usize) -> CoreRig {
    let app = tempfile::tempdir().expect("app");
    let home = tempfile::tempdir().expect("home");
    let (sink, events) = ExportSink::channel();
    let mut options = CoreOptions::new(data_root(), app.path().to_path_buf());
    options.export_events = sink;
    options.decode_threads = 2;
    options.thumbnails = false;
    options.home = Some(home.path().to_path_buf());
    options.advisor = advisor;
    let core = Core::open(options).expect("core");
    let photos = tempfile::tempdir().expect("photos");
    write_photos(photos.path(), n);
    CoreRig {
        core,
        events,
        _app: app,
        _home: home,
        photos,
    }
}

impl CoreRig {
    fn open(&self) -> Vec<PhotoId> {
        self.core
            .session
            .open(&[self.photos.path().to_path_buf()], OpenOptions::default())
            .added
    }

    /// Exports every photo as JPEG into `dest` and returns the files' bytes, by name.
    fn export(&self, dest: &Path) -> Vec<(String, Vec<u8>)> {
        self.start_export(dest);
        self.wait_export();
        let mut files: Vec<(String, Vec<u8>)> = std::fs::read_dir(dest)
            .unwrap()
            .flatten()
            .map(|e| {
                (
                    e.file_name().to_string_lossy().into_owned(),
                    std::fs::read(e.path()).unwrap(),
                )
            })
            .collect();
        files.sort();
        files
    }

    fn start_export(&self, dest: &Path) {
        self.core
            .export_start(&ExportJob {
                scope: ApplyScope::All,
                options: ExportOptions {
                    destination: Some(dest.to_path_buf()),
                    ..ExportOptions::default()
                },
            })
            .expect("export queued");
    }

    fn wait_export(&self) {
        let deadline = Instant::now() + Duration::from_secs(180);
        loop {
            let left = deadline.saturating_duration_since(Instant::now());
            if self.events.recv_timeout(left).expect("export finished") == ExportEvent::Busy(false)
            {
                return;
            }
        }
    }
}

fn unescape(xml: &str) -> String {
    xml.replace("&quot;", "\"")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&apos;", "'")
        .replace("&amp;", "&")
}

#[test]
fn the_status_and_the_preference_reach_the_session() {
    let none = core_rig(None, 0);
    let s = none.core.apple_auto_status();
    assert!(!s.available && !s.active && s.reason.is_none() && s.info.is_none());
    assert!(s.preference);
    let fake = core_rig(Some(Arc::new(FakeAdvisor)), 1);
    let s = fake.core.apple_auto_status();
    assert!(s.available && s.active && s.preference);
    assert_eq!(s.info, Some(FakeAdvisor::info()));
    let sha = ym_core::apple_auto::InstructionsId::default().instructions_sha256;
    assert_eq!(
        serde_json::to_value(&s).unwrap(),
        json!({
            "available": true, "reason": null,
            "info": {
                "model": "Test advisor", "os": "test",
                "instructions": "default", "instructions_sha256": sha,
            },
            "preference": true, "mode": "from_preference", "active": true,
            "prompt": 3, "table": 2,
            "instructions": "default", "instructions_sha256": sha,
            "own_instructions": "",
            "default_instructions": ym_core::apple_auto::DEFAULT_INSTRUCTIONS,
            "instructions_max_chars": 2000,
        })
    );
    let id = fake.open()[0];
    assert!(fake.core.session.wait_idle(Duration::from_secs(120)));
    assert!(matches!(
        fake.core.session.estimates(id).unwrap().apple,
        AppleAuto::Answered { .. }
    ));
    // The preference, saved and applied at once.
    let prefs = fake
        .core
        .preferences_set(&PreferencesPatch {
            apple_auto: Some(false),
            ..PreferencesPatch::default()
        })
        .unwrap();
    assert!(!prefs.apple_auto);
    assert!(!fake.core.apple_auto_status().active);
    assert_eq!(
        fake.core.session.estimates(id).unwrap().apple,
        AppleAuto::Off
    );
    // `ym --apple-auto on` for one run.
    fake.core.set_apple_auto_mode(AppleAutoMode::On);
    assert!(fake.core.apple_auto_status().active);
    // A core opened later reads the preference from the file.
    let app = fake._app.path().to_path_buf();
    let mut options = CoreOptions::new(data_root(), app);
    options.advisor = Some(Arc::new(FakeAdvisor));
    let again = Core::open(options).unwrap();
    let s = again.apple_auto_status();
    assert!(!s.preference && !s.active);
}

#[test]
fn exports_with_the_fake_carry_its_answer_and_are_the_same_bytes_twice() {
    let fake = core_rig(Some(Arc::new(FakeAdvisor)), 2);
    // Exported at once: the export waits for the answers.
    fake.open();
    let dest = tempfile::tempdir().unwrap();
    // The session is still developing: wait until every photo has estimates (an export needs them).
    while fake
        .core
        .session
        .photo_ids()
        .iter()
        .any(|id| fake.core.session.estimates(*id).is_none())
    {
        std::thread::sleep(Duration::from_millis(20));
    }
    let first = fake.export(&dest.path().join("one"));
    let second = fake.export(&dest.path().join("two"));
    assert_eq!(first.len(), 2);
    assert_eq!(
        first, second,
        "two exports with the fake on, the same bytes"
    );
    for (name, bytes) in &first {
        let xmp = xmp_of(bytes).expect("xmp");
        assert_eq!(
            xmp_property(&xmp, "ym:Auto").as_deref(),
            Some("apple"),
            "{name}"
        );
        let apple: Value =
            serde_json::from_str(&unescape(&xmp_property(&xmp, "ym:AppleAuto").unwrap())).unwrap();
        assert_eq!(apple["result"], json!("answered"));
        assert_eq!(apple["model"], json!("Test advisor"));
        assert_eq!(apple["os"], json!("test"));
        assert_eq!(apple["scene"], json!("Test picture"));
        assert_eq!(
            (apple["prompt"].clone(), apple["table"].clone()),
            (json!(3), json!(2))
        );
        assert_eq!(apple["plan"], json!("Test plan: You"));
        assert_eq!(apple["instructions"], json!("default"));
        // ym:Settings pins Auto contrast to the table's value of the answer.
        let settings: Value =
            serde_json::from_str(&unescape(&xmp_property(&xmp, "ym:Settings").unwrap())).unwrap();
        let contrast = [-30.0, -15.0, 0.0, 15.0, 30.0]
            [(apple["steps"]["contrast"].as_i64().unwrap() + 2) as usize];
        assert_eq!(
            settings["contrast"],
            json!({ "kind": "manual", "value": contrast })
        );
        assert_eq!(settings["white_balance"], json!({ "kind": "auto" }));
    }

    // Without an advisor: `ym:Auto` is the rule and there is no `ym:AppleAuto`.
    let none = core_rig(None, 1);
    none.open();
    assert!(none.core.session.wait_idle(Duration::from_secs(120)));
    let files = none.export(&dest.path().join("rule"));
    let xmp = xmp_of(&files[0].1).unwrap();
    assert_eq!(xmp_property(&xmp, "ym:Auto").as_deref(), Some("rule"));
    assert!(xmp_property(&xmp, "ym:AppleAuto").is_none());
    let models = xmp.find("<ym:DisplayTransform>").unwrap();
    assert!(models < xmp.find("<ym:Auto>").unwrap());
    assert!(xmp.find("<ym:Auto>").unwrap() < xmp.find("<ym:Settings>").unwrap());
}

#[test]
fn an_export_waits_for_the_answer_and_uses_it() {
    let advisor = Scripted::new(Ok(steps(2, 0, 0, 0)));
    advisor.gate.close(None);
    let rig = core_rig(Some(advisor.clone()), 1);
    let id = rig.open()[0];
    advisor.gate.wait_parked();
    let dest = tempfile::tempdir().unwrap();
    rig.start_export(dest.path());
    // Held: the export does not finish.
    let quiet = Instant::now() + Duration::from_millis(500);
    while Instant::now() < quiet {
        if let Ok(ExportEvent::Busy(false)) = rig.events.recv_timeout(Duration::from_millis(50)) {
            panic!("the export finished before the answer");
        }
    }
    advisor.gate.open();
    rig.wait_export();
    let file: PathBuf = std::fs::read_dir(dest.path())
        .unwrap()
        .flatten()
        .next()
        .unwrap()
        .path();
    let xmp = xmp_of(&std::fs::read(file).unwrap()).unwrap();
    assert_eq!(xmp_property(&xmp, "ym:Auto").as_deref(), Some("apple"));
    let settings: Value =
        serde_json::from_str(&unescape(&xmp_property(&xmp, "ym:Settings").unwrap())).unwrap();
    let e = rig.core.session.estimates(id).unwrap();
    assert!((e.exposure.ev - (e.exposure.own_ev + 0.6)).abs() < 1e-12);
    assert_eq!(settings["exposure"]["ev"].as_f64(), Some(e.exposure.ev));
    // The core rendered the picture (the real one, 384 x 288 here) on its own renderer.
    assert_eq!(advisor.calls().len(), 1);
}

#[test]
fn a_failed_photo_exports_with_the_rule_and_says_why() {
    let advisor = Scripted::new(Err(FailureKind::Guardrail));
    let rig = core_rig(Some(advisor), 1);
    rig.open();
    assert!(rig.core.session.wait_idle(Duration::from_secs(120)));
    let dest = tempfile::tempdir().unwrap();
    let files = rig.export(dest.path());
    let xmp = xmp_of(&files[0].1).unwrap();
    assert_eq!(xmp_property(&xmp, "ym:Auto").as_deref(), Some("rule"));
    assert_eq!(
        unescape(&xmp_property(&xmp, "ym:AppleAuto").unwrap()),
        format!(
            r#"{{"instructions":"default","instructions_sha256":"{}","model":"Scripted","os":"27.0 (27A1)","prompt":3,"reason":"guardrail","result":"failed","table":2}}"#,
            ym_core::apple_auto::InstructionsId::default().instructions_sha256
        )
    );
}

#[test]
fn new_instructions_ask_every_photo_again_and_the_model_reads_them() {
    let fake = core_rig(Some(Arc::new(FakeAdvisor)), 2);
    let ids = fake.open();
    assert!(fake.core.session.wait_idle(Duration::from_secs(120)));
    let plan = |id: PhotoId| match fake.core.session.estimates(id).unwrap().apple.clone() {
        AppleAuto::Answered { answer, info } => (answer.plan, info.instructions.instructions),
        other => panic!("{other:?}"),
    };
    use ym_core::apple_auto::InstructionsKind;
    for id in &ids {
        assert_eq!(
            plan(*id),
            ("Test plan: You".to_owned(), InstructionsKind::Default)
        );
    }
    // Too long: refused with a plain message, nothing saved, nothing asked again.
    let long = PreferencesPatch {
        apple_auto_instructions: Some("x".repeat(2001)),
        ..PreferencesPatch::default()
    };
    let err = fake.core.preferences_set(&long).unwrap_err().to_string();
    assert!(err.contains("At most 2000"), "{err}");
    assert_eq!(fake.core.preferences_get().apple_auto_instructions, "");
    // Edited: saved, every photo asked again with them (the fake echoes their first word).
    let edited = PreferencesPatch {
        apple_auto_instructions: Some("Moody night grade, please.".to_owned()),
        ..PreferencesPatch::default()
    };
    fake.core.preferences_set(&edited).unwrap();
    assert!(fake.core.session.wait_idle(Duration::from_secs(120)));
    for id in &ids {
        assert_eq!(
            plan(*id),
            ("Test plan: Moody".to_owned(), InstructionsKind::Edited)
        );
    }
    let status = fake.core.apple_auto_status();
    assert_eq!(status.instructions.instructions, InstructionsKind::Edited);
    assert_eq!(status.own_instructions, "Moody night grade, please.");
    // Back to the default (empty text, or the default text itself): saved as "", asked again.
    let reset = PreferencesPatch {
        apple_auto_instructions: Some(ym_core::apple_auto::DEFAULT_INSTRUCTIONS.to_owned()),
        ..PreferencesPatch::default()
    };
    let saved = fake.core.preferences_set(&reset).unwrap();
    assert_eq!(saved.apple_auto_instructions, "");
    assert!(fake.core.session.wait_idle(Duration::from_secs(120)));
    for id in &ids {
        assert_eq!(
            plan(*id),
            ("Test plan: You".to_owned(), InstructionsKind::Default)
        );
    }
}

#[test]
fn an_answer_to_old_instructions_is_dropped_and_the_photo_asked_again() {
    let advisor = Scripted::new(Ok(steps(1, 0, 0, 0)));
    advisor.gate.close(None);
    let rig = Rig::new(advisor.clone(), 1, false);
    let id = rig.open()[0];
    advisor.gate.wait_parked();
    // The instructions change while the first request is held.
    rig.session.set_apple_auto_instructions("Bright and airy.");
    advisor.gate.open();
    rig.settle();
    let seen = lock(&advisor.instructions).clone();
    assert_eq!(seen.len(), 2, "{seen:?}");
    assert_eq!(seen[0], ym_core::apple_auto::DEFAULT_INSTRUCTIONS);
    assert_eq!(seen[1], "Bright and airy.");
    match rig.state(id) {
        AppleAuto::Answered { info, .. } => assert_eq!(
            info.instructions,
            ym_core::apple_auto::InstructionsId::of("Bright and airy.")
        ),
        other => panic!("{other:?}"),
    }
    // The same text again changes nothing and asks nothing.
    rig.session.set_apple_auto_instructions("Bright and airy.");
    rig.settle();
    assert_eq!(lock(&advisor.instructions).len(), 2);
}

#[test]
fn an_exports_photo_moves_up_with_its_burst_so_the_export_waits_for_its_burst_alone() {
    // a.ARW and z.ARW are one burst (a second apart), at the two ends of the filmstrip; twelve
    // photos minutes apart lie between them. a is on screen and is exported alone: its export waits
    // for z's answer too, so z must move up with it (`docs/19` section 6).
    let advisor = Scripted::new(Ok(steps(1, 0, 0, 0)));
    // Not available while the photos develop, so that nothing is asked before a is on screen.
    advisor.set_available(false);
    advisor.gate.close(None);
    let session = Arc::new(Session::new(base_env().with_apple_auto(AppleAutoEnv {
        retry_pause: Duration::from_millis(5),
        ..AppleAutoEnv::new(advisor.clone(), Arc::new(StubPictures::default()))
    })));
    let dir = tempfile::tempdir().unwrap();
    let frame = |name: &str, sec: &str| {
        let s = scene(0.3);
        s.builder(MODEL)
            .capture_time(&format!("2026:10:04 12:03:{sec}"), "00", None)
            .write_to(&dir.path().join(name), &s.mosaic())
            .unwrap();
    };
    frame("a.ARW", "10");
    frame("z.ARW", "11");
    for k in 0..12 {
        write_photo(dir.path(), &format!("m{k:02}.ARW"), 10 + k);
    }
    let ids = session
        .open(&[dir.path().to_path_buf()], OpenOptions::default())
        .added;
    assert_eq!(ids.len(), 14);
    let (a, z) = (ids[0], ids[13]);
    assert!(session.wait_idle(Duration::from_secs(120)));
    assert_eq!(session.state().current_index, Some(0));
    // The photo on screen is asked first; it is held until the export is queued.
    advisor.set_available(true);
    session.refresh_apple_auto();
    advisor.gate.wait_parked();
    assert_eq!(advisor.calls(), vec![a.0 as u8]);
    let home = tempfile::tempdir().unwrap();
    let dest = tempfile::tempdir().unwrap();
    let ctx = Arc::new(
        ExportContext::new(Renderer::neutral_only().expect("renderer"))
            .with_home(Some(home.path().to_path_buf())),
    );
    let (sink, events) = ExportSink::channel();
    let queue = ExportQueue::new(ctx, sink);
    queue
        .start(
            &session,
            &ExportJob {
                scope: ApplyScope::This,
                options: ExportOptions {
                    destination: Some(dest.path().to_path_buf()),
                    ..ExportOptions::default()
                },
            },
            false,
        )
        .expect("export queued");
    // Only the burst is let through: the export must finish without any other photo's answer.
    advisor.gate.free(&[a, z]);
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        let left = deadline.saturating_duration_since(Instant::now());
        match events.recv_timeout(left) {
            Ok(ExportEvent::Busy(false)) => break,
            Ok(_) => {}
            Err(_) => panic!(
                "the export waited for other photos than its burst; asked about {:?}",
                advisor.calls()
            ),
        }
    }
    // About its burst's requests: a, z, and at most the next photo, held at the gate.
    let calls = advisor.calls();
    assert_eq!(calls[..2], [a.0 as u8, z.0 as u8], "{calls:?}");
    assert!(calls.len() <= 3, "{calls:?}");
    assert_eq!(
        session.estimates(a).unwrap().apple_steps,
        Some(steps(1, 0, 0, 0))
    );
    assert_eq!(std::fs::read_dir(dest.path()).unwrap().count(), 1);
    advisor.gate.open();
}

/// A sink that panics the first time the Apple Auto worker sends `estimates_changed` from its own
/// thread (when an answer arrives), and forwards every event.
fn panicking_sink() -> (EventSink, Receiver<SessionEvent>) {
    let (tx, rx) = std::sync::mpsc::channel();
    let tx = Mutex::new(tx);
    let armed = std::sync::atomic::AtomicBool::new(true);
    let sink = EventSink::new(move |e| {
        let worker = std::thread::current().name() == Some("ym-apple-auto");
        if worker
            && matches!(e, SessionEvent::EstimatesChanged { .. })
            && armed.swap(false, Ordering::SeqCst)
        {
            panic!("the sink broke");
        }
        let _ = lock(&tx).send(e);
    });
    (sink, rx)
}

#[test]
fn a_panic_after_the_answer_fails_that_photo_and_the_worker_goes_on() {
    // The answer of the first photo arrives and the event sink panics on the worker's thread: the
    // request ends, the photo is failed (kind other) and uses the rule, the second photo is still
    // asked and answered, and waiting for the session returns.
    let advisor = Scripted::new(Ok(steps(2, 1, 1, 1)));
    let pictures = Arc::new(StubPictures::default());
    let (sink, events) = panicking_sink();
    let session = Session::new(base_env().with_events(sink).with_apple_auto(AppleAutoEnv {
        retry_pause: Duration::from_millis(5),
        ..AppleAutoEnv::new(advisor.clone(), pictures)
    }));
    let dir = tempfile::tempdir().unwrap();
    write_photos(dir.path(), 2);
    let ids = session
        .open(&[dir.path().to_path_buf()], OpenOptions::default())
        .added;
    assert!(
        session.wait_idle(Duration::from_secs(60)),
        "waiting for the session returns after the panic"
    );
    // The photo asked first (the photo on screen, or the other when it developed first) is the one
    // whose answer met the panic.
    assert_eq!(advisor.calls().len(), 2);
    let broken = PhotoId(u32::from(advisor.calls()[0]));
    let other = *ids.iter().find(|id| **id != broken).unwrap();
    let first = session.estimates(broken).unwrap();
    match &first.apple {
        AppleAuto::Failed {
            kind,
            message,
            info,
        } => {
            assert_eq!(*kind, FailureKind::Other);
            assert!(message.contains("the sink broke"), "{message}");
            assert_eq!(*info, Scripted::info());
        }
        other => panic!("the first photo is failed, not {other:?}"),
    }
    assert_eq!(first.source(), AutoSource::Rule);
    assert_eq!(first.exposure.ev, first.exposure.own_ev);
    assert_eq!((first.contrast, first.saturation), (0.0, 0.0));
    assert!(session.estimates_final(broken).unwrap());
    // The worker is alive: the other photo has its answer.
    assert!(matches!(
        session.estimates(other).unwrap().apple,
        AppleAuto::Answered { .. }
    ));
    assert!(session.apple_auto_active());
    // The viewers were told about the failure.
    let changed = changed_ids(&events.try_iter().collect::<Vec<_>>());
    assert!(changed.contains(&broken));
}

// ---------------------------------------------------------------------------------------------
// Removing photos from the filmstrip, and Undo (`docs/01` Session, `docs/19` section 6)
// ---------------------------------------------------------------------------------------------

/// Waits until the decode pool has nothing queued or running (the worker may still be asking).
fn wait_developed(session: &Session) {
    let deadline = Instant::now() + Duration::from_secs(120);
    while session.pending_jobs() > 0 {
        assert!(Instant::now() < deadline, "the photos were developed");
        std::thread::sleep(Duration::from_millis(5));
    }
}

/// `a.ARW`, `b.ARW` and `c.ARW` one burst (a second apart), `d.ARW` alone a half minute later.
fn write_burst_of_three_and_one(dir: &Path) {
    let frame = |name: &str, sec: &str, level: f64| {
        let s = scene(level);
        s.builder(MODEL)
            .capture_time(&format!("2026:10:04 12:03:{sec}"), "00", None)
            .write_to(&dir.join(name), &s.mosaic())
            .unwrap();
    };
    frame("a.ARW", "10", 0.10);
    frame("b.ARW", "10", 0.20);
    frame("c.ARW", "11", 0.40);
    frame("d.ARW", "40", 0.10);
}

impl Rig {
    /// Develops the photos while Apple Auto is not available, so that nobody is asked yet and the order of the
    /// questions is the documented one (the photo on screen, then the others in filmstrip order); then makes it
    /// available with the advisor's gate closed to `only` (every photo, or the one), and waits until a request is
    /// held there.
    fn hold_the_first_request(&self, advisor: &Scripted, only: Option<u8>) {
        self.settle();
        assert!(
            advisor.calls().is_empty(),
            "nobody was asked while it was unavailable"
        );
        advisor.gate.close(only);
        advisor.set_available(true);
        self.session.refresh_apple_auto();
        advisor.gate.wait_parked();
    }
}

/// A rig over `n` photos (no bursts) that is held at the first request, with `fallback` as the model's answer.
fn held_rig(fallback: Reply, n: usize) -> (Arc<Scripted>, Rig, Vec<PhotoId>) {
    let advisor = Scripted::new(fallback);
    advisor.set_available(false);
    let rig = Rig::new(advisor.clone(), n, false);
    let ids = rig.open();
    rig.hold_the_first_request(&advisor, None);
    (advisor, rig, ids)
}

#[test]
fn a_removed_photo_leaves_the_queue_and_its_answer_is_dropped_when_it_arrives() {
    let (advisor, rig, ids) = held_rig(Ok(steps(1, 0, 0, 0)), 4);
    let (a, b, c, d) = (ids[0], ids[1], ids[2], ids[3]);
    // The photo on screen is held at the advisor; c is the photo an export waits for (next in the order).
    assert_eq!(advisor.calls(), vec![a.0 as u8]);
    rig.session.prioritise_apple_auto(&[c]);
    rig.drain();

    let out = rig.session.remove(&[a, c]);
    assert_eq!(out.removed, vec![a, c]);
    assert_eq!(rig.session.restorable(), vec![a, c]);
    assert!(rig.session.estimates(a).is_none() && rig.session.estimates(c).is_none());
    // Nothing is told about the photos that left.
    let told = changed_ids(&rig.drain());
    assert!(!told.contains(&a) && !told.contains(&c), "{told:?}");

    // The held request returns: its answer belongs to a photo that is not in the session, and is dropped.
    advisor.gate.open();
    rig.settle();
    // b and d are asked, in filmstrip order; c never is, and a only the once, before it left.
    assert_eq!(advisor.calls(), vec![a.0 as u8, b.0 as u8, d.0 as u8]);
    assert!(matches!(rig.state(b), AppleAuto::Answered { .. }));
    assert!(matches!(rig.state(d), AppleAuto::Answered { .. }));
    assert!(
        rig.session.estimates(a).is_none(),
        "the answer found no place to go"
    );
    assert_eq!(
        rig.session.restorable(),
        vec![a, c],
        "the removal is still there"
    );

    // Undo: both come back without an answer (the late one was dropped; c was never asked) and go back to the queue,
    // the photo that was on screen first.
    advisor.gate.close(None);
    let back = rig.session.restore_removed();
    assert_eq!(back.restored, vec![a, c]);
    advisor.gate.wait_parked();
    assert_eq!(rig.state(a), AppleAuto::Pending);
    assert_eq!(rig.state(c), AppleAuto::Pending);
    assert!(
        !rig.session.estimates_final(c).unwrap(),
        "an export of c waits again"
    );
    advisor.gate.open();
    rig.settle();
    assert_eq!(advisor.calls_about(a), 2);
    assert_eq!(advisor.calls_about(c), 1);
    for id in [a, c] {
        assert!(matches!(rig.state(id), AppleAuto::Answered { .. }), "{id}");
        assert!(rig.session.estimates_final(id).unwrap());
    }
}

#[test]
fn a_removed_photo_drops_its_picture_and_renders_it_again_when_it_is_asked() {
    // Both photos are developed with Apple Auto on: the development jobs render the pictures (hard rule 4: in memory only).
    let advisor = Scripted::new(Ok(steps(1, 0, 0, 0)));
    advisor.gate.close(None);
    let rig = Rig::new(advisor.clone(), 2, false);
    let ids = rig.open();
    advisor.gate.wait_parked();
    wait_developed(&rig.session);
    assert_eq!(rig.pictures.renders.load(Ordering::SeqCst), 2);
    // One photo is held at the advisor; the other waits behind it with its picture ready.
    let asked = advisor.calls()[0];
    let waiting = *ids.iter().find(|id| id.0 as u8 != asked).unwrap();

    rig.session.remove(&[waiting]);
    assert_eq!(rig.session.restore_removed().restored, vec![waiting]);
    advisor.gate.open();
    rig.settle();
    // The picture was dropped with the photo, so the worker made it again for the question.
    assert_eq!(rig.pictures.renders.load(Ordering::SeqCst), 3);
    assert!(matches!(rig.state(waiting), AppleAuto::Answered { .. }));
}

#[test]
fn an_answer_that_arrives_after_a_removal_and_its_undo_is_dropped_too() {
    let (advisor, rig, ids) = held_rig(Ok(steps(1, 0, 0, 0)), 2);
    let a = ids[0];
    // The first request about a is answered with exposure +3, the second with +1.
    advisor.answer(a, [Ok(steps(3, 0, 0, 0)), Ok(steps(1, 0, 0, 0))]);
    assert_eq!(advisor.calls(), vec![a.0 as u8]);

    // a leaves and comes back while the model is still working on it: a new incarnation, without an answer.
    rig.session.remove(&[a]);
    assert_eq!(rig.session.restore_removed().restored, vec![a]);
    assert_eq!(rig.state(a), AppleAuto::Pending);
    advisor.gate.open();
    rig.settle();

    // The first answer was about the photo before it left and is not used; the photo was asked again.
    assert_eq!(advisor.calls_about(a), 2);
    assert_eq!(
        rig.session.estimates(a).unwrap().apple_steps,
        Some(steps(1, 0, 0, 0))
    );
}

#[test]
fn a_photo_comes_back_with_its_answer_or_its_failure_and_is_not_asked_again() {
    let advisor = Scripted::new(Ok(steps(1, 1, 0, 0)));
    let rig = Rig::new(advisor.clone(), 3, false);
    let ids = rig.open();
    let (a, b, c) = (ids[0], ids[1], ids[2]);
    advisor.answer(b, [Err(FailureKind::Refused)]);
    rig.session.refresh_apple_auto();
    rig.settle();
    assert_eq!(advisor.calls().len(), 3);
    let before: Vec<_> = ids
        .iter()
        .map(|id| rig.session.estimates(*id).unwrap())
        .collect();
    assert!(matches!(before[0].apple, AppleAuto::Answered { .. }));
    assert!(matches!(
        before[1].apple,
        AppleAuto::Failed {
            kind: FailureKind::Refused,
            ..
        }
    ));
    let served: Vec<_> = ids
        .iter()
        .map(|id| serde_json::to_value(rig.session.settings(*id).unwrap()).unwrap())
        .collect();

    rig.session.remove(&[a, b]);
    // An export of the photo that stays does not wait for photos that are gone.
    assert!(rig.session.estimates_final(c).unwrap());
    let back = rig.session.restore_removed();
    assert_eq!(back.restored, vec![a, b]);
    for (i, id) in ids.iter().enumerate() {
        let now = rig.session.estimates(*id).unwrap();
        assert_eq!(now.apple, before[i].apple, "{id}");
        assert_eq!(now.apple_steps, before[i].apple_steps, "{id}");
        assert_eq!(now.exposure.ev, before[i].exposure.ev, "{id}");
        assert_eq!(
            serde_json::to_value(rig.session.settings(*id).unwrap()).unwrap(),
            served[i],
            "{id}"
        );
        assert!(rig.session.estimates_final(*id).unwrap());
    }
    rig.settle();
    assert_eq!(advisor.calls().len(), 3, "nobody was asked again");
    assert_eq!(rig.session.apple_auto_asks(), 3);
}

#[test]
fn removing_a_frame_of_a_burst_makes_the_steps_of_the_frames_left_again_and_tells_them() {
    let dir = tempfile::tempdir().unwrap();
    write_burst_of_three_and_one(dir.path());
    let advisor = Scripted::new(Ok(AutoSteps::keep()));
    let rig = Rig::in_dir(advisor.clone(), dir, true);
    let ids = rig.open();
    let (a, b, c, d) = (ids[0], ids[1], ids[2], ids[3]);
    // Exposure steps +1, +3, +3 in the burst: the middle one is +3; without c, the lower middle of +1 and +3 is +1.
    advisor.answer(a, [Ok(steps(1, 0, 0, 0))]);
    advisor.answer(b, [Ok(steps(3, 0, 0, 0))]);
    advisor.answer(c, [Ok(steps(3, 0, 0, 0))]);
    advisor.answer(d, [Ok(steps(-2, 0, 0, 0))]);
    rig.settle();
    let exposure_step = |id: PhotoId| {
        rig.session
            .estimates(id)
            .unwrap()
            .apple_steps
            .unwrap()
            .exposure
    };
    for id in [a, b, c] {
        assert_eq!(exposure_step(id), Some(3), "{id}");
    }
    let ev_added = |id: PhotoId| {
        let e = rig.session.estimates(id).unwrap();
        e.exposure.ev - e.exposure.own_ev
    };
    assert!((ev_added(a) - 1.0).abs() < 1e-12);
    rig.drain();

    let out = rig.session.remove(&[c]);
    // The frames left share what is left: +1, the way the burst median exposure is made again (`docs/19` section 5).
    assert_eq!(exposure_step(a), Some(1));
    assert_eq!(exposure_step(b), Some(1));
    assert!((ev_added(a) - 0.3).abs() < 1e-12 && (ev_added(b) - 0.3).abs() < 1e-12);
    assert_eq!(
        out.estimates_changed,
        vec![a, b],
        "d is alone and did not move"
    );
    assert_eq!(exposure_step(d), Some(-2));
    rig.settle();
    let events = rig.drain();
    let mut told = changed_ids(&events);
    told.sort();
    assert_eq!(told, vec![a, b], "each once, and not c or d");
    // Their pictures changed, so their thumbnails are made again (and none is made for the photo that left).
    assert_eq!(thumbnails_of(&events, a), 1, "{events:?}");
    assert_eq!(thumbnails_of(&events, b), 1);
    assert_eq!(thumbnails_of(&events, c) + thumbnails_of(&events, d), 0);

    // Undo: c joins the burst again with its own answer, and the median is +3 for the three.
    let back = rig.session.restore_removed();
    assert_eq!(back.restored, vec![c]);
    assert_eq!(
        back.estimates_changed,
        vec![a, b],
        "the restored photo is not in it"
    );
    for id in [a, b, c] {
        assert_eq!(exposure_step(id), Some(3), "{id}");
    }
    assert!((ev_added(c) - 1.0).abs() < 1e-12);
    rig.settle();
    assert_eq!(advisor.calls().len(), 4, "nobody was asked again");
}

#[test]
fn a_burst_waiting_for_a_frame_that_is_removed_is_complete_without_it() {
    let dir = tempfile::tempdir().unwrap();
    write_burst_of_three_and_one(dir.path());
    let advisor = Scripted::new(Ok(steps(1, 0, 0, 0)));
    advisor.set_available(false);
    let rig = Rig::in_dir(advisor.clone(), dir, false);
    let ids = rig.open();
    let (a, b, c) = (ids[0], ids[1], ids[2]);
    // c is held: a and b answer first (the photo on screen, then filmstrip order), and the burst is not complete.
    rig.hold_the_first_request(&advisor, Some(c.0 as u8));
    assert!(matches!(rig.state(a), AppleAuto::Answered { .. }));
    assert!(matches!(rig.state(b), AppleAuto::Answered { .. }));
    assert!(!rig.session.estimates_final(a).unwrap(), "a waits for c");
    rig.drain();

    // c leaves while the model is working on it: a and b are final at once, and the answer for c is dropped when it comes.
    rig.session.remove(&[c]);
    assert!(rig.session.estimates_final(a).unwrap());
    assert!(rig.session.estimates_final(b).unwrap());
    advisor.gate.open();
    rig.settle();
    assert!(rig.session.estimates(c).is_none());
    assert_eq!(advisor.calls_about(c), 1);
    assert!(
        !changed_ids(&rig.drain()).contains(&c),
        "nothing about c is sent"
    );
}

/// An export queue of its own over a session that is not a [`Core`], with one job queued.
struct QueuedExport {
    _queue: ExportQueue,
    _home: tempfile::TempDir,
    dest: tempfile::TempDir,
    events: Receiver<ExportEvent>,
}

/// Queues an export of `scope` into a new folder.
fn queue_for(session: &Arc<Session>, scope: ApplyScope) -> QueuedExport {
    let home = tempfile::tempdir().unwrap();
    let dest = tempfile::tempdir().unwrap();
    let ctx = Arc::new(
        ExportContext::new(Renderer::neutral_only().expect("renderer"))
            .with_home(Some(home.path().to_path_buf())),
    );
    let (sink, events) = ExportSink::channel();
    let queue = ExportQueue::new(ctx, sink);
    queue
        .start(
            session,
            &ExportJob {
                scope,
                options: ExportOptions {
                    destination: Some(dest.path().to_path_buf()),
                    ..ExportOptions::default()
                },
            },
            false,
        )
        .expect("export queued");
    QueuedExport {
        _queue: queue,
        _home: home,
        dest,
        events,
    }
}

impl QueuedExport {
    /// Whether the export has finished, looking at the events that came so far.
    fn finished(&self) -> bool {
        self.events
            .try_iter()
            .any(|e| e == ExportEvent::Busy(false))
    }

    fn wait_finished(&self, what: &str) {
        let deadline = Instant::now() + Duration::from_secs(60);
        loop {
            let left = deadline.saturating_duration_since(Instant::now());
            match self.events.recv_timeout(left) {
                Ok(ExportEvent::Busy(false)) => return,
                Ok(_) => {}
                Err(_) => panic!("the export hung: {what}"),
            }
        }
    }

    fn files(&self) -> usize {
        std::fs::read_dir(self.dest.path()).unwrap().count()
    }
}

#[test]
fn an_export_waiting_for_a_removed_frame_or_photo_does_not_hang() {
    let dir = tempfile::tempdir().unwrap();
    write_burst_of_three_and_one(dir.path());
    let advisor = Scripted::new(Ok(steps(1, 0, 0, 0)));
    let session = Arc::new(Session::new(base_env().with_apple_auto(AppleAutoEnv {
        retry_pause: Duration::from_millis(5),
        ..AppleAutoEnv::new(advisor.clone(), Arc::new(StubPictures::default()))
    })));
    let ids = session
        .open(&[dir.path().to_path_buf()], OpenOptions::default())
        .added;
    assert!(session.wait_idle(Duration::from_secs(120)));
    let (a, b, c, d) = (ids[0], ids[1], ids[2], ids[3]);
    // Everything is answered once; new instructions ask every photo again, and the first request is held.
    advisor.gate.close(None);
    session.set_apple_auto_instructions("Moody and dark.");
    advisor.gate.wait_parked();
    assert_eq!(advisor.calls().last(), Some(&(a.0 as u8)));

    // 1. An export of a (This photo) waits for a, b and c: the burst. Only a is let through, once the export is queued.
    let export = queue_for(&session, ApplyScope::This);
    advisor.gate.free(&[a]);
    std::thread::sleep(Duration::from_millis(300));
    assert!(!export.finished(), "it waits for b and c");
    // b and c leave (b is the request in flight): the burst is a alone, and the export goes on without them.
    session.remove(&[b, c]);
    export.wait_finished("a burst whose other frames were removed");
    assert_eq!(export.files(), 1);

    // 2. An export of a and d waits for d's new answer (the worker is still inside b's request); both leave meanwhile,
    //    and the export finishes with the settings it was queued with.
    let export = queue_for(&session, ApplyScope::All);
    std::thread::sleep(Duration::from_millis(300));
    assert!(!export.finished(), "it waits for d");
    session.remove(&[d, a]);
    export.wait_finished("photos removed while their export waited");
    assert_eq!(export.files(), 2);
    advisor.gate.open();
}

#[test]
fn new_instructions_while_a_photo_is_out_drop_its_answer_too() {
    let advisor = Scripted::new(Ok(steps(1, 0, 0, 0)));
    let rig = Rig::new(advisor.clone(), 3, false);
    let ids = rig.open();
    let (a, b, c) = (ids[0], ids[1], ids[2]);
    rig.settle();
    assert_eq!(advisor.calls().len(), 3);

    // b is out when the instructions change: a and c are asked again, b was not in the session to be.
    rig.session.remove(&[b]);
    rig.session.set_apple_auto_instructions("Bright and airy.");
    rig.settle();
    assert_eq!(advisor.calls_about(a), 2);
    assert_eq!(advisor.calls_about(c), 2);
    assert_eq!(advisor.calls_about(b), 1);

    // Its answer was made under the old instructions, so it is not brought back: b is asked with the new ones.
    assert_eq!(rig.session.restore_removed().restored, vec![b]);
    rig.settle();
    assert_eq!(advisor.calls_about(b), 2);
    let seen = lock(&advisor.instructions).clone();
    assert_eq!(seen.last().map(String::as_str), Some("Bright and airy."));
    match rig.state(b) {
        AppleAuto::Answered { info, .. } => assert_eq!(
            info.instructions,
            ym_core::apple_auto::InstructionsId::of("Bright and airy.")
        ),
        other => panic!("{other:?}"),
    }

    // The same instructions all along keep the answer: an undo with nothing changed in between asks nobody.
    let asks = rig.session.apple_auto_asks();
    rig.session.remove(&[b]);
    rig.session.restore_removed();
    rig.settle();
    assert_eq!(rig.session.apple_auto_asks(), asks);
}

#[test]
fn an_export_after_a_removal_and_its_undo_is_the_same_bytes_with_the_same_answers() {
    let rig = core_rig(Some(Arc::new(FakeAdvisor)), 3);
    let ids = rig.open();
    assert!(rig.core.session.wait_idle(Duration::from_secs(120)));
    let first = tempfile::tempdir().unwrap();
    let before = rig.export(first.path());
    assert_eq!(before.len(), 3);
    let asks = rig.core.session.apple_auto_asks();

    let out = rig.core.remove_photos(&[ids[1]]);
    assert_eq!(out.removed, vec![ids[1]]);
    // The photos that stay export alone: two files.
    let alone = tempfile::tempdir().unwrap();
    assert_eq!(rig.export(alone.path()).len(), 2);
    assert_eq!(rig.core.restore_photos().restored, vec![ids[1]]);
    assert!(rig.core.session.wait_idle(Duration::from_secs(120)));
    assert_eq!(
        rig.core.session.apple_auto_asks(),
        asks,
        "nobody was asked again"
    );

    let second = tempfile::tempdir().unwrap();
    let after = rig.export(second.path());
    assert_eq!(before, after, "the photo came back with its answer");
    for (_, bytes) in &after {
        let xmp = xmp_of(bytes).unwrap();
        assert_eq!(xmp_property(&xmp, "ym:Auto").as_deref(), Some("apple"));
    }
}
