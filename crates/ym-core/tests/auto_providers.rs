//! The Auto providers in the core (`docs/22` sections 9 to 11 and 16): which provider acts, answers kept
//! per photo and per provider, the photo on screen first, workers by concurrency, the sign-out rule,
//! the model that answered, change notifications, cancelling, and the preference patch with its consent.
//! Runs on any machine with scripted in-process advisors; the two programs have their own tests in
//! `ym-agents` and `ym-cli`.

mod common;

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant};

use common::*;
use ym_core::app::{Core, CoreOptions};
use ym_core::apple_auto::{
    AdvisorAnswer, AdvisorError, AdvisorInfo, AdvisorPicture, AppleAuto, AutoAdvisor, AutoSteps,
    Availability, FailureKind, UnavailableReason,
};
use ym_core::auto_provider::{
    AdvisorDetails, AutoChoice, AutoConsent, AutoProvider, ChangeListener, ProviderAdvisors,
    ProviderState, ToolInfo,
};
use ym_core::export_queue::ExportJob;
use ym_core::preferences::PreferencesPatch;
use ym_core::session::{
    AppleAutoEnv, ApplyScope, EventSink, OpenOptions, PhotoId, PictureInput, PictureRenderer,
    Session, SessionEvent,
};
use ym_export::ExportOptions;

const MODEL: &str = "ILCE-7RM4";
const PROFILE: &str = "sony-ilce-7rm4";

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

/// `n` photos taken minutes apart (no bursts) in a new folder.
fn write_photos(n: usize) -> tempfile::TempDir {
    let dir = tempfile::tempdir().expect("dir");
    for k in 0..n {
        let s = Scene::new(
            384,
            288,
            light(PROFILE, 5200.0),
            Layout::Patchwork,
            0.2 + 0.05 * (k % 8) as f64,
        )
        .noise(2e-4, 1e-6);
        s.builder(MODEL)
            .capture_time(&format!("2026:10:04 12:{:02}:00", k % 60), "00", None)
            .write_to(
                &dir.path().join(format!("DSC{:05}.ARW", k + 1)),
                &s.mosaic(),
            )
            .expect("write the photo");
    }
    dir
}

/// A picture that only says which photo it is: two pixels of the photo's id.
struct StubPictures;

impl PictureRenderer for StubPictures {
    fn render(&self, input: &PictureInput<'_>) -> Result<AdvisorPicture, String> {
        Ok(AdvisorPicture {
            width: 2,
            height: 1,
            rgb: vec![input.id.0 as u8; 6],
        })
    }
}

/// A scripted advisor: records which photos it was asked about, answers with its own steps (so the
/// answers of two providers can be told apart), can be held at a gate, fail, sign out, say which model
/// answered and tell its listeners about a change.
struct Lab {
    provider: AutoProvider,
    concurrency: usize,
    /// The exposure step it answers, so that each provider's answer is its own.
    exposure: i8,
    availability: Mutex<Availability>,
    calls: Mutex<Vec<u8>>,
    instructions: Mutex<Vec<String>>,
    running: AtomicUsize,
    peak: AtomicUsize,
    open: Mutex<bool>,
    wake: Condvar,
    cancels: AtomicUsize,
    failures: Mutex<VecDeque<FailureKind>>,
    /// A failure of kind `not_signed_in` also makes the advisor say it is signed out, as the programs do.
    sign_out_on_not_signed_in: bool,
    served_model: Mutex<Option<String>>,
    listeners: Mutex<Vec<ChangeListener>>,
    details: Mutex<AdvisorDetails>,
    /// How many times the core asked it to probe.
    refreshes: AtomicUsize,
    /// How many times the core settled it, and how long each settle blocks (up to its timeout).
    settles: AtomicUsize,
    settle_block: Mutex<Duration>,
    /// How long a request takes to end after a cancel (a program is told to stop, then reaped).
    wind_down: Mutex<Duration>,
    /// The timeouts of the `wait_idle` calls and how many requests were still running at each.
    idle_waits: Mutex<Vec<(Duration, usize)>>,
    /// `wait_idle` gives up at once (a program that cannot be stopped).
    stuck: std::sync::atomic::AtomicBool,
}

impl Lab {
    fn info(provider: AutoProvider) -> AdvisorInfo {
        match provider {
            AutoProvider::Claude | AutoProvider::Agy => AdvisorInfo::cli(
                provider,
                "sonnet",
                ToolInfo {
                    version: "2.1.310".to_owned(),
                    api: Some("firstParty".to_owned()),
                    auth: None,
                },
            ),
            _ => AdvisorInfo::new("Lab model", "27.0 (27A1)"),
        }
    }

    fn new(provider: AutoProvider, exposure: i8) -> Arc<Self> {
        Arc::new(Self {
            provider,
            concurrency: provider.concurrency(),
            exposure,
            availability: Mutex::new(Availability::Available(Self::info(provider))),
            calls: Mutex::new(Vec::new()),
            instructions: Mutex::new(Vec::new()),
            running: AtomicUsize::new(0),
            peak: AtomicUsize::new(0),
            open: Mutex::new(true),
            wake: Condvar::new(),
            cancels: AtomicUsize::new(0),
            failures: Mutex::new(VecDeque::new()),
            sign_out_on_not_signed_in: true,
            served_model: Mutex::new(None),
            listeners: Mutex::new(Vec::new()),
            details: Mutex::new(AdvisorDetails::default()),
            refreshes: AtomicUsize::new(0),
            settles: AtomicUsize::new(0),
            settle_block: Mutex::new(Duration::ZERO),
            wind_down: Mutex::new(Duration::ZERO),
            idle_waits: Mutex::new(Vec::new()),
            stuck: std::sync::atomic::AtomicBool::new(false),
        })
    }

    fn with_concurrency(provider: AutoProvider, exposure: i8, concurrency: usize) -> Arc<Self> {
        let mut lab = Arc::try_unwrap(Self::new(provider, exposure))
            .ok()
            .expect("one owner");
        lab.concurrency = concurrency;
        Arc::new(lab)
    }

    fn close(&self) {
        *lock(&self.open) = false;
    }

    fn open(&self) {
        *lock(&self.open) = true;
        self.wake.notify_all();
    }

    fn calls(&self) -> Vec<u8> {
        lock(&self.calls).clone()
    }

    fn asks_about(&self, photo: PhotoId) -> usize {
        self.calls().iter().filter(|p| **p == photo.0 as u8).count()
    }

    fn set_availability(&self, availability: Availability) {
        *lock(&self.availability) = availability;
        self.tell_listeners();
    }

    fn tell_listeners(&self) {
        let listeners: Vec<ChangeListener> = lock(&self.listeners).clone();
        for l in listeners {
            l();
        }
    }

    fn fail_next(&self, kinds: impl IntoIterator<Item = FailureKind>) {
        lock(&self.failures).extend(kinds);
    }

    /// Waits until `n` requests are inside `ask` at once.
    fn wait_running(&self, n: usize) {
        let deadline = Instant::now() + Duration::from_secs(60);
        while self.running.load(Ordering::SeqCst) < n {
            assert!(Instant::now() < deadline, "{n} requests never ran at once");
            std::thread::sleep(Duration::from_millis(5));
        }
    }
}

impl AutoAdvisor for Lab {
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
        let now = self.running.fetch_add(1, Ordering::SeqCst) + 1;
        self.peak.fetch_max(now, Ordering::SeqCst);
        let cancels = self.cancels.load(Ordering::SeqCst);
        {
            let mut open = lock(&self.open);
            while !*open && self.cancels.load(Ordering::SeqCst) == cancels {
                open = self
                    .wake
                    .wait_timeout(open, Duration::from_millis(20))
                    .unwrap_or_else(PoisonError::into_inner)
                    .0;
            }
        }
        let cancelled = self.cancels.load(Ordering::SeqCst) != cancels;
        if cancelled {
            // The program is told to stop and reaped: it is still running meanwhile.
            std::thread::sleep(*lock(&self.wind_down));
        }
        self.running.fetch_sub(1, Ordering::SeqCst);
        if cancelled {
            return Err(AdvisorError::new(FailureKind::Other, "cancelled"));
        }
        // The scripted failures go first, in order. A program that is signed out fails every request, and
        // so does this one while its availability reads signed out: whichever of two requests that run at
        // once is the first to meet the sign-out, the other meets it too. Both are decided under the
        // failures lock, so that no request slips in between the first one's failure and the sign-out it
        // causes.
        let (failure, signed_out_now) = {
            let mut failures = lock(&self.failures);
            match failures.pop_front() {
                Some(kind) => {
                    let sign_out =
                        kind == FailureKind::NotSignedIn && self.sign_out_on_not_signed_in;
                    if sign_out {
                        *lock(&self.availability) =
                            Availability::Unavailable(UnavailableReason::SignedOut);
                    }
                    (Some(kind), sign_out)
                }
                None if matches!(
                    *lock(&self.availability),
                    Availability::Unavailable(UnavailableReason::SignedOut)
                ) =>
                {
                    (Some(FailureKind::NotSignedIn), false)
                }
                None => (None, false),
            }
        };
        if signed_out_now {
            self.tell_listeners();
        }
        if let Some(kind) = failure {
            return Err(AdvisorError::new(kind, "scripted"));
        }
        Ok(AdvisorAnswer {
            steps: AutoSteps {
                exposure: Some(self.exposure),
                ..AutoSteps::keep()
            },
            scene: format!("Photo {photo}"),
            plan: format!("{} plan", self.provider.name()),
            served_model: lock(&self.served_model).clone(),
        })
    }

    fn concurrency(&self) -> usize {
        self.concurrency
    }

    fn cancel(&self) {
        self.cancels.fetch_add(1, Ordering::SeqCst);
        let _open = lock(&self.open);
        self.wake.notify_all();
    }

    fn refresh(&self) {
        self.refreshes.fetch_add(1, Ordering::SeqCst);
    }

    fn settle(&self, timeout: Duration) {
        self.settles.fetch_add(1, Ordering::SeqCst);
        let block = *lock(&self.settle_block);
        std::thread::sleep(block.min(timeout));
    }

    fn wait_idle(&self, timeout: Duration) -> bool {
        lock(&self.idle_waits).push((timeout, self.running.load(Ordering::SeqCst)));
        if self.stuck.load(Ordering::SeqCst) {
            return false;
        }
        let deadline = Instant::now() + timeout;
        while self.running.load(Ordering::SeqCst) > 0 {
            if Instant::now() >= deadline {
                return false;
            }
            std::thread::sleep(Duration::from_millis(2));
        }
        true
    }

    fn on_change(&self, listener: ChangeListener) {
        lock(&self.listeners).push(listener);
    }

    fn details(&self) -> AdvisorDetails {
        lock(&self.details).clone()
    }
}

struct Rig {
    session: Session,
    events: std::sync::mpsc::Receiver<SessionEvent>,
    dir: tempfile::TempDir,
}

impl Rig {
    /// A session with these advisors, the preference and the consent, retries 5 ms apart, over `n` photos
    /// written but not opened yet.
    fn new(
        advisors: ProviderAdvisors,
        preference: AutoProvider,
        consent: AutoConsent,
        n: usize,
    ) -> Self {
        let dir = write_photos(n);
        let (sink, events) = EventSink::channel();
        let env = base_env().with_events(sink).with_apple_auto(AppleAutoEnv {
            preference,
            consent,
            retry_pause: Duration::from_millis(5),
            ..AppleAutoEnv::new(advisors, Arc::new(StubPictures))
        });
        Self {
            session: Session::new(env),
            events,
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

    fn exposure_step(&self, id: PhotoId) -> Option<i8> {
        self.session
            .estimates(id)
            .expect("estimates")
            .apple_steps
            .and_then(|s| s.exposure)
    }

    fn drain(&self) -> Vec<SessionEvent> {
        self.events.try_iter().collect()
    }
}

fn advisors(apple: &Arc<Lab>, claude: &Arc<Lab>, agy: Option<&Arc<Lab>>) -> ProviderAdvisors {
    let mut a = ProviderAdvisors::default()
        .with(AutoProvider::Apple, apple.clone())
        .with(AutoProvider::Claude, claude.clone());
    if let Some(agy) = agy {
        a = a.with(AutoProvider::Agy, agy.clone());
    }
    a
}

const CLAUDE_OK: AutoConsent = AutoConsent {
    claude: true,
    agy: false,
};

#[test]
fn switching_provider_asks_the_photos_without_an_answer_and_keeps_the_others() {
    let apple = Lab::new(AutoProvider::Apple, -1);
    // One at a time, so that "the photo on screen first" is the first and only request in flight.
    let claude = Lab::with_concurrency(AutoProvider::Claude, 2, 1);
    let rig = Rig::new(
        advisors(&apple, &claude, None),
        AutoProvider::Apple,
        CLAUDE_OK,
        3,
    );
    let ids = rig.open();
    rig.settle();
    assert_eq!(apple.calls().len(), 3);
    assert!(
        claude.calls().is_empty(),
        "a provider that is not chosen is not asked"
    );
    for id in &ids {
        assert_eq!(rig.exposure_step(*id), Some(-1), "Apple's answer");
        assert_eq!(rig.session.estimates(*id).unwrap().source().name(), "apple");
        assert_eq!(
            rig.session.estimates(*id).unwrap().provider,
            AutoProvider::Apple
        );
    }
    // Choose Claude Code, the photo on screen being the last one: it is asked first, then the others in
    // order, and Apple's answers stay.
    rig.session.set_current(ids[2]).unwrap();
    rig.drain();
    claude.close();
    rig.session.set_auto_preference(AutoProvider::Claude);
    claude.wait_running(1);
    assert_eq!(
        claude.calls(),
        vec![ids[2].0 as u8],
        "the photo on screen first"
    );
    // Meanwhile the photos show Claude's state, pending, with the rule's values.
    assert_eq!(rig.state(ids[0]), AppleAuto::Pending);
    assert_eq!(rig.exposure_step(ids[0]), None);
    claude.open();
    rig.settle();
    assert_eq!(
        claude.calls(),
        vec![ids[2].0 as u8, ids[0].0 as u8, ids[1].0 as u8]
    );
    for id in &ids {
        assert_eq!(rig.exposure_step(*id), Some(2), "Claude's answer");
        assert_eq!(
            rig.session.estimates(*id).unwrap().source().name(),
            "claude"
        );
        match rig.state(*id) {
            AppleAuto::Answered { info, .. } => assert_eq!(info.provider, AutoProvider::Claude),
            other => panic!("{other:?}"),
        }
    }
    assert_eq!(apple.calls().len(), 3, "Apple was not asked again");
    // Switching back reuses Apple's answers: nothing is asked again.
    rig.session.set_auto_preference(AutoProvider::Apple);
    rig.settle();
    assert_eq!(apple.calls().len(), 3);
    assert_eq!(claude.calls().len(), 3);
    for id in &ids {
        assert_eq!(rig.exposure_step(*id), Some(-1));
    }
    // The viewers were told about each change of values.
    assert!(!rig.drain().is_empty());
    // And Default Auto shows the rule while both answers stay.
    rig.session.set_auto_preference(AutoProvider::Yellowmoon);
    assert_eq!(rig.state(ids[0]), AppleAuto::Off);
    assert_eq!(
        rig.session.estimates(ids[0]).unwrap().source().name(),
        "rule"
    );
    assert_eq!(
        rig.session.estimates(ids[0]).unwrap().provider,
        AutoProvider::Yellowmoon
    );
    rig.session.set_auto_preference(AutoProvider::Claude);
    rig.settle();
    assert_eq!(claude.calls().len(), 3);
    assert_eq!(apple.calls().len(), 3);
}

#[test]
fn new_instructions_drop_every_providers_answers_and_cancel_running_requests() {
    let apple = Lab::new(AutoProvider::Apple, -1);
    let claude = Lab::new(AutoProvider::Claude, 2);
    let rig = Rig::new(
        advisors(&apple, &claude, None),
        AutoProvider::Apple,
        CLAUDE_OK,
        2,
    );
    let ids = rig.open();
    rig.settle();
    rig.session.set_auto_preference(AutoProvider::Claude);
    rig.settle();
    assert_eq!((apple.calls().len(), claude.calls().len()), (2, 2));
    // New instructions: both providers' answers are dropped, the chosen one asks again with them.
    claude.close();
    rig.session.set_apple_auto_instructions("Make it moody.");
    claude.wait_running(2);
    assert_eq!(
        claude.calls().len(),
        4,
        "both photos asked again, two at once"
    );
    // New instructions while that request runs: it is cancelled, and its answer dropped.
    let cancels = claude.cancels.load(Ordering::SeqCst);
    rig.session.set_apple_auto_instructions("Make it bright.");
    assert!(
        claude.cancels.load(Ordering::SeqCst) > cancels,
        "running requests are cancelled"
    );
    claude.open();
    rig.settle();
    let seen = lock(&claude.instructions).clone();
    assert!(seen.contains(&"Make it bright.".to_owned()));
    for id in &ids {
        match rig.state(*id) {
            AppleAuto::Answered { info, .. } => assert_eq!(
                info.instructions,
                ym_core::apple_auto::InstructionsId::of("Make it bright.")
            ),
            other => panic!("{other:?}"),
        }
    }
    // Apple's answers were dropped too: switching to it asks again with the new instructions.
    let before = apple.calls().len();
    rig.session.set_auto_preference(AutoProvider::Apple);
    rig.settle();
    assert_eq!(apple.calls().len(), before + 2);
    assert!(
        lock(&apple.instructions)[before..]
            .iter()
            .all(|i| i == "Make it bright.")
    );
}

#[test]
fn an_answer_that_arrives_after_the_provider_changed_is_kept_for_the_provider_asked() {
    let apple = Lab::new(AutoProvider::Apple, -1);
    let claude = Lab::new(AutoProvider::Claude, 2);
    let rig = Rig::new(
        advisors(&apple, &claude, None),
        AutoProvider::Claude,
        CLAUDE_OK,
        1,
    );
    claude.close();
    let id = rig.open()[0];
    claude.wait_running(1);
    // Switch to Apple while Claude Code's request runs: Apple answers at once.
    rig.session.set_auto_preference(AutoProvider::Apple);
    rig.settle_without_claude(id);
    assert_eq!(rig.exposure_step(id), Some(-1));
    // Claude's answer arrives now: it is stored for Claude, not shown.
    claude.open();
    let deadline = Instant::now() + Duration::from_secs(30);
    while claude.running.load(Ordering::SeqCst) > 0 {
        assert!(Instant::now() < deadline);
        std::thread::sleep(Duration::from_millis(5));
    }
    std::thread::sleep(Duration::from_millis(100));
    assert_eq!(rig.exposure_step(id), Some(-1), "still Apple's");
    // Back to Claude Code: its answer is there, and nothing is asked again.
    rig.session.set_auto_preference(AutoProvider::Claude);
    rig.settle();
    assert_eq!(rig.exposure_step(id), Some(2));
    assert_eq!(claude.asks_about(id), 1);
}

impl Rig {
    /// Waits for Apple's answer to the photo while Claude's request is still held.
    fn settle_without_claude(&self, id: PhotoId) {
        let deadline = Instant::now() + Duration::from_secs(60);
        while !matches!(self.state(id), AppleAuto::Answered { .. }) {
            assert!(Instant::now() < deadline, "Apple did not answer");
            std::thread::sleep(Duration::from_millis(5));
        }
    }
}

#[test]
fn two_workers_ask_two_photos_at_once_and_never_the_same_one_twice() {
    let apple = Lab::new(AutoProvider::Apple, -1);
    let claude = Lab::with_concurrency(AutoProvider::Claude, 1, 2);
    assert_eq!(advisors(&apple, &claude, None).max_concurrency(), 2);
    claude.close();
    let rig = Rig::new(
        advisors(&apple, &claude, None),
        AutoProvider::Claude,
        CLAUDE_OK,
        4,
    );
    let ids = rig.open();
    claude.wait_running(2);
    // Exactly two are in flight, and they are two different photos; no third is started while they wait.
    std::thread::sleep(Duration::from_millis(200));
    let calls = claude.calls();
    assert_eq!(calls.len(), 2, "{calls:?}");
    assert_ne!(calls[0], calls[1]);
    assert_eq!(claude.peak.load(Ordering::SeqCst), 2);
    claude.open();
    rig.settle();
    let mut all = claude.calls();
    all.sort_unstable();
    let mut want: Vec<u8> = ids.iter().map(|i| i.0 as u8).collect();
    want.sort_unstable();
    assert_eq!(all, want, "one request per photo");
    assert_eq!(
        claude.peak.load(Ordering::SeqCst),
        2,
        "never more than the advisor takes"
    );
    // Apple's advisor takes one at a time even in a session whose largest advisor takes two.
    rig.session.set_auto_preference(AutoProvider::Apple);
    rig.settle();
    assert_eq!(apple.peak.load(Ordering::SeqCst), 1);
    assert_eq!(apple.calls().len(), 4);
}

#[test]
fn only_apple_is_retried() {
    let apple = Lab::new(AutoProvider::Apple, -1);
    let claude = Lab::new(AutoProvider::Claude, 2);
    apple.fail_next([FailureKind::RateLimited, FailureKind::RateLimited]);
    claude.fail_next([FailureKind::RateLimited]);
    let rig = Rig::new(
        advisors(&apple, &claude, None),
        AutoProvider::Apple,
        CLAUDE_OK,
        1,
    );
    let id = rig.open()[0];
    rig.settle();
    // Apple: two waits and then the answer.
    assert_eq!(apple.asks_about(id), 3);
    assert!(matches!(rig.state(id), AppleAuto::Answered { .. }));
    // Claude Code: the first failure is the photo's failure, no retry.
    rig.session.set_auto_preference(AutoProvider::Claude);
    rig.settle();
    assert_eq!(claude.asks_about(id), 1);
    assert!(matches!(
        rig.state(id),
        AppleAuto::Failed {
            kind: FailureKind::RateLimited,
            ..
        }
    ));
    let e = rig.session.estimates(id).unwrap();
    assert_eq!(e.apple_steps, None, "a failed photo uses the rule");
    // Re-run asks again.
    rig.session.rerun_auto(id).unwrap();
    rig.settle();
    assert!(matches!(rig.state(id), AppleAuto::Answered { .. }));
    assert_eq!(claude.asks_about(id), 2);
}

#[test]
fn a_sign_out_during_a_request_abandons_it_and_fails_nothing() {
    let apple = Lab::new(AutoProvider::Apple, -1);
    let claude = Lab::new(AutoProvider::Claude, 2);
    // The first request meets the sign-out. Claude Code runs two requests at once, so the second photo
    // may be asked at the same moment, before or after the sign-out is read; the Lab fails it too, as a
    // program that is signed out fails every request. Whatever the order, the rule below holds.
    claude.fail_next([FailureKind::NotSignedIn]);
    let rig = Rig::new(
        advisors(&apple, &claude, None),
        AutoProvider::Claude,
        CLAUDE_OK,
        2,
    );
    let ids = rig.open();
    rig.settle();
    // The advisor said it is signed out: the photos that met the sign-out are not failed, nothing more is
    // asked (one request, or two when both were asked before the sign-out was read; never one photo twice),
    // and every photo reads as unavailable (the rule's values), so nothing waits.
    let asked_before: Vec<usize> = ids.iter().map(|id| claude.asks_about(*id)).collect();
    assert!(
        (1..=2).contains(&claude.calls().len()) && asked_before.iter().all(|n| *n <= 1),
        "{:?}",
        claude.calls()
    );
    for id in &ids {
        assert_eq!(
            rig.state(*id),
            AppleAuto::Unavailable {
                reason: UnavailableReason::SignedOut
            }
        );
        assert!(rig.session.estimates_final(*id).unwrap());
    }
    let status = rig.session.auto_status();
    assert_eq!(
        status.provider(AutoProvider::Claude).unwrap().state,
        ProviderState::SignedOut
    );
    assert_eq!(status.effective, AutoProvider::Yellowmoon);
    // Signed in again: the advisor tells, the session reads it again, and every photo is asked exactly
    // once more than it was before (a photo never asked until now is asked for the first time).
    claude.set_availability(Availability::Available(Lab::info(AutoProvider::Claude)));
    rig.session.refresh_apple_auto();
    rig.settle();
    for (id, before) in ids.iter().zip(&asked_before) {
        assert_eq!(
            claude.asks_about(*id),
            before + 1,
            "photo {id:?}: {:?}",
            claude.calls()
        );
        assert!(
            matches!(rig.state(*id), AppleAuto::Answered { .. }),
            "{:?}",
            rig.state(*id)
        );
    }
}

#[test]
fn a_not_signed_in_failure_with_the_provider_still_available_fails_the_photo() {
    let apple = Lab::new(AutoProvider::Apple, -1);
    let mut claude = Arc::try_unwrap(Lab::new(AutoProvider::Claude, 2))
        .ok()
        .expect("owner");
    claude.sign_out_on_not_signed_in = false;
    let claude = Arc::new(claude);
    claude.fail_next([FailureKind::NotSignedIn]);
    let rig = Rig::new(
        advisors(&apple, &claude, None),
        AutoProvider::Claude,
        CLAUDE_OK,
        1,
    );
    let id = rig.open()[0];
    rig.settle();
    assert!(matches!(
        rig.state(id),
        AppleAuto::Failed {
            kind: FailureKind::NotSignedIn,
            ..
        }
    ));
}

#[test]
fn the_model_that_answered_is_the_photos_model() {
    let apple = Lab::new(AutoProvider::Apple, -1);
    let claude = Lab::new(AutoProvider::Claude, 2);
    let rig = Rig::new(
        advisors(&apple, &claude, None),
        AutoProvider::Claude,
        CLAUDE_OK,
        2,
    );
    let ids = rig.open();
    rig.settle();
    // Until a request names a model, the alias asked for is the info's model.
    match rig.state(ids[0]) {
        AppleAuto::Answered { info, .. } => assert_eq!(info.model, "sonnet"),
        other => panic!("{other:?}"),
    }
    *lock(&claude.served_model) = Some("claude-sonnet-5-5".to_owned());
    rig.session.rerun_auto(ids[0]).unwrap();
    rig.settle();
    match rig.state(ids[0]) {
        AppleAuto::Answered { info, answer } => {
            assert_eq!(info.model, "claude-sonnet-5-5");
            assert_eq!(answer.served_model.as_deref(), Some("claude-sonnet-5-5"));
            assert_eq!(info.provider, AutoProvider::Claude);
            assert_eq!(
                info.tool.as_ref().map(|t| t.version.as_str()),
                Some("2.1.310")
            );
            assert!(
                info.version_string()
                    .starts_with("Claude Code 2.1.310 · claude-sonnet-5-5")
            );
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn a_cloud_provider_acts_only_with_its_consent_or_when_the_run_names_it() {
    let apple = Lab::new(AutoProvider::Apple, -1);
    let claude = Lab::new(AutoProvider::Claude, 2);
    let rig = Rig::new(
        advisors(&apple, &claude, None),
        AutoProvider::Claude,
        AutoConsent::default(),
        1,
    );
    let id = rig.open()[0];
    rig.settle();
    // The saved choice is Claude Code, but nothing agreed to it: Default Auto acts, and says why.
    assert_eq!(
        rig.state(id),
        AppleAuto::Unavailable {
            reason: UnavailableReason::NotConsented
        }
    );
    assert!(claude.calls().is_empty());
    assert!(!rig.session.apple_auto_active());
    assert_eq!(
        rig.session.auto_status().effective,
        AutoProvider::Yellowmoon
    );
    // Consent saved: it acts and asks.
    rig.session.set_auto_consent(CLAUDE_OK);
    rig.settle();
    assert!(matches!(rig.state(id), AppleAuto::Answered { .. }));
    assert_eq!(rig.session.auto_status().effective, AutoProvider::Claude);
    // Taking the consent away stops it acting and keeps the answer.
    rig.session.set_auto_consent(AutoConsent::default());
    assert_eq!(
        rig.state(id),
        AppleAuto::Unavailable {
            reason: UnavailableReason::NotConsented
        }
    );
    // A run that names the provider is the consent.
    rig.session
        .set_auto_choice(AutoChoice::Provider(AutoProvider::Claude));
    assert!(matches!(rig.state(id), AppleAuto::Answered { .. }));
    assert_eq!(claude.calls().len(), 1, "the kept answer was used");
    let status = rig.session.auto_status();
    assert_eq!(
        (status.preference, status.chosen, status.effective),
        (
            AutoProvider::Claude,
            AutoProvider::Claude,
            AutoProvider::Claude
        )
    );
    // The apple lab never ran.
    assert!(apple.calls().is_empty());
}

#[test]
fn a_provider_without_an_advisor_reads_as_not_installed() {
    let apple = Lab::new(AutoProvider::Apple, -1);
    let advisors = ProviderAdvisors::default().with(AutoProvider::Apple, apple.clone());
    let rig = Rig::new(
        advisors,
        AutoProvider::Agy,
        AutoConsent {
            claude: true,
            agy: true,
        },
        1,
    );
    let id = rig.open()[0];
    rig.settle();
    assert_eq!(
        rig.state(id),
        AppleAuto::Unavailable {
            reason: UnavailableReason::NotInstalled
        }
    );
    let status = rig.session.auto_status();
    assert!(status.visible, "Apple is available, so the dropdown shows");
    assert_eq!(
        status.provider(AutoProvider::Agy).unwrap().state,
        ProviderState::NotInstalled
    );
    assert!(status.provider(AutoProvider::Agy).unwrap().experimental);
    assert_eq!(status.effective, AutoProvider::Yellowmoon);
    // Apple's answers were not asked for: the chosen provider is Antigravity.
    assert!(apple.calls().is_empty());
}

#[test]
fn the_status_has_the_four_providers_in_order() {
    let apple = Lab::new(AutoProvider::Apple, -1);
    let claude = Lab::new(AutoProvider::Claude, 2);
    let agy = Lab::new(AutoProvider::Agy, 0);
    claude.set_availability(Availability::Unavailable(UnavailableReason::SignedOut));
    *lock(&claude.details) = AdvisorDetails {
        path: Some(PathBuf::from("/x/claude")),
        tool_version: Some("2.1.200".to_owned()),
        ..AdvisorDetails::default()
    };
    agy.set_availability(Availability::Unavailable(UnavailableReason::TooOld));
    let rig = Rig::new(
        advisors(&apple, &claude, Some(&agy)),
        AutoProvider::Apple,
        AutoConsent::default(),
        0,
    );
    let s = rig.session.auto_status();
    let order: Vec<_> = s.providers.iter().map(|p| p.provider).collect();
    assert_eq!(order, AutoProvider::ALL);
    let state = |p| s.provider(p).unwrap();
    assert_eq!(state(AutoProvider::Apple).state, ProviderState::Available);
    assert_eq!(state(AutoProvider::Claude).state, ProviderState::SignedOut);
    assert_eq!(
        state(AutoProvider::Claude).tool_version.as_deref(),
        Some("2.1.200")
    );
    assert_eq!(
        state(AutoProvider::Claude).path.as_deref(),
        Some(Path::new("/x/claude"))
    );
    assert!(state(AutoProvider::Claude).listed && !state(AutoProvider::Claude).selectable);
    assert_eq!(state(AutoProvider::Agy).state, ProviderState::TooOld);
    assert!(s.visible && s.active && s.effective == AutoProvider::Apple);
    // The same through the session's own readers.
    assert_eq!(
        rig.session
            .advisor_details(AutoProvider::Claude)
            .unwrap()
            .tool_version
            .as_deref(),
        Some("2.1.200")
    );
    assert_eq!(rig.session.chosen_provider(), AutoProvider::Apple);
    assert!(
        rig.session
            .advisor_details(AutoProvider::Yellowmoon)
            .is_none()
    );
}

#[test]
fn cancel_auto_and_the_sessions_end_cancel_running_requests() {
    let apple = Lab::new(AutoProvider::Apple, -1);
    let claude = Lab::new(AutoProvider::Claude, 2);
    claude.close();
    let rig = Rig::new(
        advisors(&apple, &claude, None),
        AutoProvider::Claude,
        CLAUDE_OK,
        1,
    );
    rig.open();
    claude.wait_running(1);
    let before = claude.cancels.load(Ordering::SeqCst);
    rig.session.cancel_auto();
    assert!(
        claude.cancels.load(Ordering::SeqCst) > before,
        "cancel_auto reaches the advisor"
    );
    // The held request ends; a cancelled request stores nothing (see
    // `cancel_auto_waits_for_the_programs_and_asks_nothing_more`).
    claude.open();
    let deadline = Instant::now() + Duration::from_secs(30);
    while claude.running.load(Ordering::SeqCst) > 0 {
        assert!(Instant::now() < deadline);
        std::thread::sleep(Duration::from_millis(5));
    }
    // Closing the session cancels as well.
    let claude2 = Lab::new(AutoProvider::Claude, 2);
    claude2.close();
    let rig2 = Rig::new(
        advisors(&apple, &claude2, None),
        AutoProvider::Claude,
        CLAUDE_OK,
        1,
    );
    rig2.open();
    claude2.wait_running(1);
    let before = claude2.cancels.load(Ordering::SeqCst);
    rig2.session.close();
    assert!(
        claude2.cancels.load(Ordering::SeqCst) > before,
        "close cancels"
    );
    // So does dropping it.
    let claude3 = Lab::new(AutoProvider::Claude, 2);
    claude3.close();
    let before = claude3.cancels.load(Ordering::SeqCst);
    {
        let rig3 = Rig::new(
            advisors(&apple, &claude3, None),
            AutoProvider::Claude,
            CLAUDE_OK,
            1,
        );
        rig3.open();
        claude3.wait_running(1);
    }
    assert!(
        claude3.cancels.load(Ordering::SeqCst) > before,
        "drop cancels"
    );
}

/// `Session::cancel_auto` (the app quitting, `ym` on Ctrl-C) tells every advisor to stop and then waits
/// until its programs are reaped (`AutoAdvisor::wait_idle`), so the process may end right after it. The
/// requests it cancelled store nothing, and nothing new is asked afterwards (`docs/22` section 9.4).
#[test]
fn cancel_auto_waits_for_the_programs_and_asks_nothing_more() {
    let apple = Lab::new(AutoProvider::Apple, -1);
    let claude = Lab::new(AutoProvider::Claude, 2);
    claude.close();
    *lock(&claude.wind_down) = Duration::from_millis(300);
    // Three photos, two requests at once: the third would be asked as soon as one ended.
    let rig = Rig::new(
        advisors(&apple, &claude, None),
        AutoProvider::Claude,
        CLAUDE_OK,
        3,
    );
    let ids = rig.open();
    claude.wait_running(2);
    let before = claude.cancels.load(Ordering::SeqCst);
    let started = Instant::now();
    rig.session.cancel_auto();
    // It returned only after the programs were gone: the cancelled requests (which take 300 ms to end,
    // as a program told to stop does) are over.
    assert!(claude.cancels.load(Ordering::SeqCst) > before);
    assert_eq!(
        claude.running.load(Ordering::SeqCst),
        0,
        "cancel_auto returned while a request was still running"
    );
    assert!(
        started.elapsed() >= Duration::from_millis(250),
        "it waited: {:?}",
        started.elapsed()
    );
    let waits = lock(&claude.idle_waits).clone();
    assert_eq!(waits.len(), 1, "{waits:?}");
    assert!(
        waits[0].0 <= ym_core::session::AUTO_CANCEL_WAIT,
        "the wait is bounded: {waits:?}"
    );
    assert!(waits[0].1 >= 1, "it was asked after the cancel: {waits:?}");
    // Nothing is asked after it, and the cancelled requests stored nothing (not even a failure).
    claude.open();
    std::thread::sleep(Duration::from_millis(300));
    assert_eq!(claude.calls().len(), 2, "no third request after the cancel");
    assert!(
        rig.session.wait_idle(Duration::from_secs(60)),
        "a halted session does not wait for answers that will not come"
    );
    for id in ids {
        assert_eq!(rig.state(id), AppleAuto::Pending, "photo {id:?}");
    }
}

/// The wait is bounded: a program that cannot be stopped does not keep the application from quitting.
#[test]
fn cancel_auto_gives_up_when_a_program_does_not_stop() {
    let apple = Lab::new(AutoProvider::Apple, -1);
    let claude = Lab::new(AutoProvider::Claude, 2);
    claude.close();
    claude.stuck.store(true, Ordering::SeqCst);
    let rig = Rig::new(
        advisors(&apple, &claude, None),
        AutoProvider::Claude,
        CLAUDE_OK,
        1,
    );
    rig.open();
    claude.wait_running(1);
    let started = Instant::now();
    rig.session.cancel_auto();
    assert!(started.elapsed() < Duration::from_secs(3), "it returned");
    let waits = lock(&claude.idle_waits).clone();
    assert_eq!(waits.len(), 1);
    // The bound is `AUTO_CANCEL_WAIT`, less the time that went before the advisor was asked. It covers
    // the 2 s of grace a program that ignores `SIGTERM` has (`ym-agents`' `TERM_GRACE`) plus 2 s.
    assert!(ym_core::session::AUTO_CANCEL_WAIT >= Duration::from_secs(2) + Duration::from_secs(2));
    assert!(
        waits[0].0 <= ym_core::session::AUTO_CANCEL_WAIT
            && waits[0].0 > ym_core::session::AUTO_CANCEL_WAIT - Duration::from_millis(500),
        "{waits:?}"
    );
    claude.open();
}

/// The session's end cancels the requests and waits for the programs as well (bounded), so that `ym`,
/// which ends the process right after the core is dropped, leaves nothing behind. `close()` does not
/// wait: the person is at the keyboard there.
#[test]
fn dropping_the_session_waits_for_the_programs() {
    let apple = Lab::new(AutoProvider::Apple, -1);
    let claude = Lab::new(AutoProvider::Claude, 2);
    claude.close();
    *lock(&claude.wind_down) = Duration::from_millis(200);
    {
        let rig = Rig::new(
            advisors(&apple, &claude, None),
            AutoProvider::Claude,
            CLAUDE_OK,
            1,
        );
        rig.open();
        claude.wait_running(1);
        rig.session.close();
        assert!(
            lock(&claude.idle_waits).is_empty(),
            "close() does not wait for the programs"
        );
    }
    assert_eq!(
        claude.running.load(Ordering::SeqCst),
        0,
        "the session's drop returned while a request was running"
    );
    assert_eq!(lock(&claude.idle_waits).len(), 1);
}

/// `ProviderAdvisors::halt` is what `ym`'s Ctrl-C handler calls with no session at hand: the session that
/// holds the registry then asks and probes nothing more, and `wait_idle` shares one deadline.
#[test]
fn a_halted_registry_stops_the_session_asking_and_probing() {
    let apple = Lab::new(AutoProvider::Apple, -1);
    let claude = Lab::new(AutoProvider::Claude, 2);
    let agy = Lab::new(AutoProvider::Agy, 3);
    claude.close();
    let registry = advisors(&apple, &claude, Some(&agy));
    let rig = Rig::new(registry.clone(), AutoProvider::Claude, CLAUDE_OK, 3);
    rig.open();
    claude.wait_running(2);
    let refreshes = claude.refreshes.load(Ordering::SeqCst);
    registry.halt();
    assert!(registry.is_halted());
    assert!(registry.wait_idle(Duration::from_secs(5)));
    let waits = lock(&claude.idle_waits).clone();
    let agy_waits = lock(&agy.idle_waits).clone();
    assert!(
        agy_waits[0].0 <= waits[0].0,
        "one shared deadline: {waits:?} then {agy_waits:?}"
    );
    claude.open();
    std::thread::sleep(Duration::from_millis(200));
    assert_eq!(claude.calls().len(), 2, "nothing is asked after the halt");
    // Nothing probes either: the session's reads and the registry's refresh do nothing.
    rig.session.refresh_apple_auto();
    registry.refresh(None);
    assert_eq!(claude.refreshes.load(Ordering::SeqCst), refreshes);
    assert_eq!(agy.refreshes.load(Ordering::SeqCst), 0);
}

// ---------------------------------------------------------------------------------------------
// Which advisors are probed (docs/22 sections 3.3, 10.2, 13.1)
// ---------------------------------------------------------------------------------------------

/// A saved cloud choice that nothing agreed to (the file says `claude`, no consent) is never probed: the
/// core only reads what the advisor last said. The photo reads `not_consented` every time, not
/// `checking` at first and `not_consented` once a probe had finished. Consent, or the run naming the
/// provider, starts the probes.
#[test]
fn a_saved_cloud_choice_nobody_agreed_to_is_never_probed() {
    let apple = Lab::new(AutoProvider::Apple, -1);
    let claude = Lab::new(AutoProvider::Claude, 2);
    // What an advisor that was never asked to probe says.
    claude.set_availability(Availability::Unavailable(UnavailableReason::Checking));
    let rig = core_rig(
        advisors(&apple, &claude, None),
        AutoChoice::FromPreference,
        Some(r#"{ "auto_provider": "claude" }"#),
    );
    let core = &rig.core;
    assert_eq!(claude.refreshes.load(Ordering::SeqCst), 0, "Core::open");
    let dir = write_photos(1);
    let id = core
        .session
        .open(&[dir.path().to_path_buf()], OpenOptions::default())
        .added[0];
    assert!(core.session.wait_idle(Duration::from_secs(120)));
    assert_eq!(
        core.session.estimates(id).expect("estimates").apple,
        AppleAuto::Unavailable {
            reason: UnavailableReason::NotConsented
        }
    );
    // Every way the core reads the chosen provider again: the preferences, the status the webview reads,
    // a re-read, an export, a settle. None starts a probe, and an export does not wait for one.
    core.preferences_set(&PreferencesPatch {
        remove_location_from_exports: Some(true),
        ..PreferencesPatch::default()
    })
    .expect("preferences");
    let status = core.auto_status_cached();
    assert_eq!(status.effective, AutoProvider::Yellowmoon);
    assert_eq!(core.session.refresh_apple_auto(), None);
    let out = tempfile::tempdir().expect("out");
    core.export_start(&ExportJob {
        scope: ApplyScope::All,
        options: ExportOptions {
            destination: Some(out.path().to_path_buf()),
            ..ExportOptions::default()
        },
    })
    .expect("export queued");
    let deadline = Instant::now() + Duration::from_secs(120);
    while core.export_busy() {
        assert!(Instant::now() < deadline, "the export finished");
        std::thread::sleep(Duration::from_millis(20));
    }
    assert_eq!(
        claude.settles.load(Ordering::SeqCst),
        0,
        "an export settles nothing"
    );
    assert_eq!(claude.refreshes.load(Ordering::SeqCst), 0);
    assert_eq!(apple.refreshes.load(Ordering::SeqCst), 0);
    assert_eq!(
        core.session.estimates(id).expect("estimates").apple,
        AppleAuto::Unavailable {
            reason: UnavailableReason::NotConsented
        },
        "the same every time"
    );
    // Opening Preferences is the one place that asks every advisor.
    let _ = core.auto_status();
    assert_eq!(claude.refreshes.load(Ordering::SeqCst), 1);
    // The consent is saved: the program is probed from then on.
    let before = claude.refreshes.load(Ordering::SeqCst);
    core.preferences_set(&patch(AutoProvider::Claude, Some(CLAUDE_OK)))
        .expect("consent");
    assert!(claude.refreshes.load(Ordering::SeqCst) > before);
}

#[test]
fn a_run_that_names_the_provider_probes_it() {
    let apple = Lab::new(AutoProvider::Apple, -1);
    let claude = Lab::new(AutoProvider::Claude, 2);
    let rig = core_rig(
        advisors(&apple, &claude, None),
        AutoChoice::Provider(AutoProvider::Claude),
        None,
    );
    assert!(
        claude.refreshes.load(Ordering::SeqCst) >= 1,
        "naming it is the consent for the run"
    );
    assert_eq!(apple.refreshes.load(Ordering::SeqCst), 0);
    // Default Auto, named: nobody is probed.
    let (before, apple_before) = (
        claude.refreshes.load(Ordering::SeqCst),
        apple.refreshes.load(Ordering::SeqCst),
    );
    rig.core
        .set_auto_choice(AutoChoice::Provider(AutoProvider::Yellowmoon));
    assert_eq!(claude.refreshes.load(Ordering::SeqCst), before);
    assert_eq!(apple.refreshes.load(Ordering::SeqCst), apple_before);
}

/// An advisor that says its availability changed makes the core read what the advisors say. It does not
/// ask any advisor to probe (`docs/22` section 3.3: no process), so a request-time sign-out and the
/// probe that would undo it cannot feed each other.
#[test]
fn a_change_notification_starts_no_probe_and_no_loop() {
    let apple = Lab::new(AutoProvider::Apple, -1);
    let claude = Lab::new(AutoProvider::Claude, 2);
    claude.set_availability(Availability::Unavailable(UnavailableReason::Checking));
    let rig = core_rig(
        advisors(&apple, &claude, None),
        AutoChoice::FromPreference,
        Some(r#"{ "auto_provider": "claude", "auto_consent": { "claude": true } }"#),
    );
    let core = &rig.core;
    let told = Arc::new(AtomicUsize::new(0));
    let t = Arc::clone(&told);
    core.on_auto_changed(move || {
        t.fetch_add(1, Ordering::SeqCst);
    });
    let (claude_before, apple_before) = (
        claude.refreshes.load(Ordering::SeqCst),
        apple.refreshes.load(Ordering::SeqCst),
    );
    assert!(claude_before >= 1, "Core::open probes the chosen provider");
    // The probe finishes: the listener reads the answer, and starts nothing.
    claude.set_availability(Availability::Available(Lab::info(AutoProvider::Claude)));
    assert_eq!(told.load(Ordering::SeqCst), 1);
    assert!(core.session.apple_auto_active(), "the session read it");
    let dir = write_photos(1);
    let id = core
        .session
        .open(&[dir.path().to_path_buf()], OpenOptions::default())
        .added[0];
    assert!(core.session.wait_idle(Duration::from_secs(120)));
    let after_open = claude.refreshes.load(Ordering::SeqCst);
    assert_eq!(claude.calls().len(), 1);
    // The request-time sign-out: the advisor says so and tells; the listener reads it and nothing
    // else happens: no probe, so nothing says "available" again behind its back, and the photo is not
    // asked again.
    claude.fail_next([FailureKind::NotSignedIn]);
    core.session.set_apple_auto_instructions("Cooler, please.");
    assert!(core.session.wait_idle(Duration::from_secs(120)));
    assert!(!core.session.apple_auto_active(), "signed out");
    assert_eq!(told.load(Ordering::SeqCst), 2);
    assert_eq!(
        claude.calls().len(),
        2,
        "asked again once, with the new instructions"
    );
    assert_eq!(
        claude.refreshes.load(Ordering::SeqCst),
        after_open,
        "the listener started no probe"
    );
    assert_eq!(apple.refreshes.load(Ordering::SeqCst), apple_before);
    assert_eq!(
        core.session.estimates(id).expect("estimates").apple,
        AppleAuto::Unavailable {
            reason: UnavailableReason::SignedOut
        }
    );
    // The provider is back (a probe the person asked for, or the sign-in): the photo is asked again.
    claude.set_availability(Availability::Available(Lab::info(AutoProvider::Claude)));
    assert!(core.session.wait_idle(Duration::from_secs(120)));
    assert_eq!(claude.calls().len(), 3);
    assert!(matches!(
        core.session.estimates(id).expect("estimates").apple,
        AppleAuto::Answered { .. }
    ));
}

/// Settling waits for the provider the run uses, not for every advisor: a slow start-up probe of another
/// program must not eat the 5 s an export in the first seconds waits (`docs/22` section 3.3).
#[test]
fn settling_waits_for_the_chosen_provider_only() {
    let apple = Lab::new(AutoProvider::Apple, -1);
    let claude = Lab::new(AutoProvider::Claude, 2);
    let agy = Lab::new(AutoProvider::Agy, 3);
    *lock(&agy.settle_block) = Duration::from_secs(3);
    let registry = advisors(&apple, &claude, Some(&agy));
    let rig = core_rig(
        registry.clone(),
        AutoChoice::Provider(AutoProvider::Claude),
        None,
    );
    let started = Instant::now();
    rig.core.settle_auto(Duration::from_secs(5));
    assert!(
        started.elapsed() < Duration::from_secs(1),
        "the chosen advisor was settled alone: {:?}",
        started.elapsed()
    );
    assert_eq!(claude.settles.load(Ordering::SeqCst), 1);
    assert_eq!(agy.settles.load(Ordering::SeqCst), 0);
    // The chosen one that is slow is waited for, up to the timeout.
    *lock(&claude.settle_block) = Duration::from_millis(400);
    let started = Instant::now();
    rig.core.settle_auto(Duration::from_secs(5));
    assert!(started.elapsed() >= Duration::from_millis(350));
    assert_eq!(agy.settles.load(Ordering::SeqCst), 0);
    // `doctor` refreshes every advisor and settles every advisor.
    let started = Instant::now();
    registry.settle(Duration::from_secs(5));
    assert!(started.elapsed() >= Duration::from_secs(3));
    assert_eq!(agy.settles.load(Ordering::SeqCst), 1);
}

// ---------------------------------------------------------------------------------------------
// The core: the preference patch, the listener and the status
// ---------------------------------------------------------------------------------------------

struct CoreRig {
    core: Core,
    _app: tempfile::TempDir,
    _home: tempfile::TempDir,
}

fn core_rig(advisors: ProviderAdvisors, choice: AutoChoice, prefs: Option<&str>) -> CoreRig {
    let app = tempfile::tempdir().expect("app");
    let home = tempfile::tempdir().expect("home");
    if let Some(prefs) = prefs {
        std::fs::write(app.path().join("preferences.json"), prefs).expect("preferences");
    }
    let mut options = CoreOptions::new(data_root(), app.path().to_path_buf());
    options.thumbnails = false;
    options.decode_threads = 2;
    options.home = Some(home.path().to_path_buf());
    options.advisors = advisors;
    options.auto_choice = choice;
    CoreRig {
        core: Core::open(options).expect("core"),
        _app: app,
        _home: home,
    }
}

fn patch(provider: AutoProvider, consent: Option<AutoConsent>) -> PreferencesPatch {
    PreferencesPatch {
        auto_provider: Some(provider),
        auto_consent: consent,
        ..PreferencesPatch::default()
    }
}

#[test]
fn the_preference_patch_reaches_the_session_and_the_consent_rule_holds() {
    let apple = Lab::new(AutoProvider::Apple, -1);
    let claude = Lab::new(AutoProvider::Claude, 2);
    let rig = core_rig(
        advisors(&apple, &claude, None),
        AutoChoice::FromPreference,
        None,
    );
    let core = &rig.core;
    let dir = write_photos(1);
    core.session
        .open(&[dir.path().to_path_buf()], OpenOptions::default());
    assert!(core.session.wait_idle(Duration::from_secs(120)));
    assert_eq!(core.auto_status().effective, AutoProvider::Apple);
    // Without its consent a cloud provider is refused, and nothing changes.
    let err = core
        .preferences_set(&patch(AutoProvider::Claude, None))
        .unwrap_err()
        .to_string();
    assert_eq!(err, "Say yes to sending photos to Claude Code first.");
    assert_eq!(core.preferences_get().auto_provider, AutoProvider::Apple);
    assert_eq!(core.auto_status().effective, AutoProvider::Apple);
    // With it in the same patch: saved and applied at once, the photo asked.
    let saved = core
        .preferences_set(&patch(AutoProvider::Claude, Some(CLAUDE_OK)))
        .unwrap();
    assert_eq!(
        (saved.auto_provider, saved.auto_consent),
        (AutoProvider::Claude, CLAUDE_OK)
    );
    assert!(core.session.wait_idle(Duration::from_secs(120)));
    let s = core.auto_status();
    assert_eq!(
        (s.preference, s.chosen, s.effective),
        (
            AutoProvider::Claude,
            AutoProvider::Claude,
            AutoProvider::Claude
        )
    );
    assert_eq!(s.consent, CLAUDE_OK);
    assert_eq!(claude.calls().len(), 1);
    // Choosing Default Auto or Apple is always allowed.
    core.preferences_set(&patch(AutoProvider::Yellowmoon, None))
        .unwrap();
    assert!(!core.auto_status().active);
    core.preferences_set(&patch(AutoProvider::Apple, None))
        .unwrap();
    assert_eq!(core.auto_status().effective, AutoProvider::Apple);
    // A file with a cloud choice and no consent acts as Default Auto.
    let from_file = core_rig(
        advisors(&apple, &claude, None),
        AutoChoice::FromPreference,
        Some(r#"{ "auto_provider": "claude" }"#),
    );
    let s = from_file.core.auto_status();
    assert_eq!(s.preference, AutoProvider::Claude);
    assert_eq!(s.effective, AutoProvider::Yellowmoon);
    assert!(s.visible);
    // The old switch off reads as Default Auto.
    let migrated = core_rig(
        advisors(&apple, &claude, None),
        AutoChoice::FromPreference,
        Some(r#"{ "apple_auto": false }"#),
    );
    assert_eq!(
        migrated.core.auto_status().preference,
        AutoProvider::Yellowmoon
    );
}

#[test]
fn a_run_that_names_the_provider_does_not_need_the_saved_consent() {
    let apple = Lab::new(AutoProvider::Apple, -1);
    let claude = Lab::new(AutoProvider::Claude, 2);
    let rig = core_rig(
        advisors(&apple, &claude, None),
        AutoChoice::Provider(AutoProvider::Claude),
        None,
    );
    let s = rig.core.auto_status();
    assert_eq!(s.choice, AutoChoice::Provider(AutoProvider::Claude));
    assert_eq!(s.effective, AutoProvider::Claude);
    assert_eq!(s.preference, AutoProvider::Apple, "nothing was saved");
    assert_eq!(
        rig.core.preferences_get().auto_consent,
        AutoConsent::default()
    );
    // Changing the run's choice acts at once.
    rig.core
        .set_auto_choice(AutoChoice::Provider(AutoProvider::Yellowmoon));
    assert_eq!(rig.core.auto_status().effective, AutoProvider::Yellowmoon);
    rig.core.set_auto_choice(AutoChoice::FromPreference);
    assert_eq!(rig.core.auto_status().effective, AutoProvider::Apple);
}

#[test]
fn a_change_notification_reaches_the_session_and_the_cores_callbacks() {
    let apple = Lab::new(AutoProvider::Apple, -1);
    let claude = Lab::new(AutoProvider::Claude, 2);
    claude.set_availability(Availability::Unavailable(UnavailableReason::Checking));
    let rig = core_rig(
        advisors(&apple, &claude, None),
        AutoChoice::FromPreference,
        Some(r#"{ "auto_provider": "claude", "auto_consent": { "claude": true } }"#),
    );
    let core = &rig.core;
    let told = Arc::new(AtomicUsize::new(0));
    let t = Arc::clone(&told);
    core.on_auto_changed(move || {
        t.fetch_add(1, Ordering::SeqCst);
    });
    assert_eq!(
        core.auto_status().effective,
        AutoProvider::Yellowmoon,
        "still checking"
    );
    assert!(!core.session.apple_auto_active());
    // The probe finishes: the advisor tells, the session reads the availability again before the callback
    // runs, so a webview that asks the status then sees the change.
    let seen = Arc::new(Mutex::new(None));
    let (s, probe) = (Arc::clone(&seen), claude.clone());
    core.on_auto_changed(move || {
        *lock(&s) = Some(probe.availability());
    });
    claude.set_availability(Availability::Available(Lab::info(AutoProvider::Claude)));
    assert_eq!(told.load(Ordering::SeqCst), 1);
    assert!(lock(&seen).is_some());
    assert!(
        core.session.apple_auto_active(),
        "the session read it without being asked"
    );
    let s = core.auto_status();
    assert_eq!(
        s.provider(AutoProvider::Claude).unwrap().state,
        ProviderState::Available
    );
    assert_eq!(s.effective, AutoProvider::Claude);
    // And a photo opened now is asked at once.
    let dir = write_photos(1);
    core.session
        .open(&[dir.path().to_path_buf()], OpenOptions::default());
    assert!(core.session.wait_idle(Duration::from_secs(120)));
    assert_eq!(claude.calls().len(), 1);
    // An advisor that signs out is told the same way.
    claude.set_availability(Availability::Unavailable(UnavailableReason::SignedOut));
    assert_eq!(told.load(Ordering::SeqCst), 2);
    assert!(!core.session.apple_auto_active());
}

#[test]
fn the_core_cancels_and_settles() {
    let apple = Lab::new(AutoProvider::Apple, -1);
    let claude = Lab::new(AutoProvider::Claude, 2);
    claude.close();
    let rig = core_rig(
        advisors(&apple, &claude, None),
        AutoChoice::Provider(AutoProvider::Claude),
        None,
    );
    let core = &rig.core;
    let dir = write_photos(1);
    core.session
        .open(&[dir.path().to_path_buf()], OpenOptions::default());
    claude.wait_running(1);
    let before = claude.cancels.load(Ordering::SeqCst);
    core.cancel_auto();
    assert!(claude.cancels.load(Ordering::SeqCst) > before);
    // Nothing probes in these advisors, so settling returns at once.
    let started = Instant::now();
    core.settle_auto(Duration::from_secs(5));
    assert!(started.elapsed() < Duration::from_secs(4));
    claude.open();
    core.shutdown();
}

#[test]
fn without_advisors_the_core_says_so_and_changes_nothing() {
    let rig = core_rig(
        ProviderAdvisors::default(),
        AutoChoice::FromPreference,
        None,
    );
    let s = rig.core.auto_status();
    assert!(!s.visible && !s.active);
    assert_eq!(s.effective, AutoProvider::Yellowmoon);
    for p in &s.providers[1..] {
        assert_eq!(p.state, ProviderState::NotInstalled, "{:?}", p.provider);
    }
    // The choice and the consent are known and saved all the same.
    let saved = rig
        .core
        .preferences_set(&patch(AutoProvider::Claude, Some(CLAUDE_OK)))
        .unwrap();
    assert_eq!(saved.auto_provider, AutoProvider::Claude);
    let s = rig.core.auto_status();
    assert_eq!((s.preference, s.consent), (AutoProvider::Claude, CLAUDE_OK));
    assert_eq!(s.effective, AutoProvider::Yellowmoon);
}
