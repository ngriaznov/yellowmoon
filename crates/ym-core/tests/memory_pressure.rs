//! Memory pressure and the idle unload of the models (`docs/26-hardening.md` P4-4, P4-5, H-14).
//!
//! What is checked, with levels and a clock the tests give (no real memory pressure, no sleeping for correctness):
//!
//! * **Warning** drops exactly what P4-5 lists: the cached mosaics of the neighbours, the finished-frame caches of the photos
//!   that are not on screen, and the models. The photo on screen keeps every cache it has, a model that is running keeps its
//!   weights until the inference ends, and neither the device's pool nor the background work is touched.
//! * **Critical** does that too, then trims the device's pool and pauses the optional background work until the level falls.
//! * **The notice** shows once per stay above normal, goes away at normal, and is for critical only.
//! * **The poll thread** reads a source we script, acts on each reading, and is gone after `shutdown` and after a drop.
//! * **The idle unload**: after the idle time the next use loads again (the count of loads is 2), and a use in the meantime
//!   starts the time again.
mod common;

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::time::{Duration, Instant};

use common::*;
use ym_core::app::{Core, CoreOptions};
use ym_core::memory::{MEMORY_NOTICE_SENTENCE, PressureLevel, PressureSource};
use ym_core::pipeline::{
    Clock, DepthModel, FrameOptions, PhotoSource, Unload, Unloadable, ViewportRect, ViewportState,
};
use ym_core::session::{OpenOptions, PhotoId};
use ym_gpu::queue::CancelToken;

const MODEL: &str = "ILCE-7RM4";
const PROFILE: &str = "sony-ilce-7rm4";

/// A clock a test moves by hand, and the model it drives.
#[derive(Clone)]
struct Hand(Arc<AtomicU64>);

impl Hand {
    fn new() -> Self {
        Self(Arc::new(AtomicU64::new(0)))
    }
    fn clock(&self) -> Clock {
        let t = Arc::clone(&self.0);
        Arc::new(move || Duration::from_secs(t.load(Ordering::SeqCst)))
    }
    fn advance(&self, secs: u64) {
        self.0.fetch_add(secs, Ordering::SeqCst);
    }
}

/// A fake depth model whose weights load and unload by `hand`; `entered` is told when a run starts and `gate` (when given)
/// holds the run until the test sends on it.
fn depth_model(
    hand: &Hand,
    entered: Option<mpsc::Sender<()>>,
    hold: Option<Arc<Mutex<mpsc::Receiver<()>>>>,
) -> Arc<DepthModel> {
    DepthModel::fake_unloading(
        "memory",
        Arc::new(move |w, h, _rgb, _| {
            if let Some(tx) = &entered {
                let _ = tx.send(());
            }
            if let Some(rx) = &hold {
                let _ = rx
                    .lock()
                    .expect("the hold")
                    .recv_timeout(Duration::from_secs(60));
            }
            let (nw, nh) = ym_ml::depth::capped_network_size(w, h);
            let (cw, ch) = (nw / ym_ml::depth::PATCH, nh / ym_ml::depth::PATCH);
            Ok((cw, ch, vec![0.5; cw * ch]))
        }),
        hand.clock(),
        Duration::from_secs(300),
    )
}

/// Four photos of one folder, a Core over them, and the photos' ids in filmstrip order.
struct Rig {
    core: Core,
    ids: Vec<PhotoId>,
    _photos: tempfile::TempDir,
    _app: tempfile::TempDir,
}

fn rig(configure: impl FnOnce(&mut CoreOptions)) -> Rig {
    let photos = tempfile::tempdir().expect("photos");
    for n in 0..4u64 {
        let scene = Scene::new(600, 400, light(PROFILE, 5500.0), Layout::Patchwork, 0.4)
            .noise(0.0, 0.0)
            .seed(n + 1);
        scene.write(photos.path(), &format!("DSC0000{}.ARW", n + 1), MODEL);
    }
    let app = tempfile::tempdir().expect("app root");
    let mut o = CoreOptions::new(data_root(), app.path().to_path_buf());
    o.decode_threads = 2;
    o.thumbnails = false;
    configure(&mut o);
    let core = Core::open(o).expect("core");
    let opened = core
        .session
        .open(&[photos.path().to_path_buf()], OpenOptions::default());
    assert_eq!(opened.added.len(), 4);
    assert!(core.session.wait_idle(Duration::from_secs(120)));
    Rig {
        ids: opened.added,
        core,
        _photos: photos,
        _app: app,
    }
}

fn wait_until(what: &str, secs: u64, mut done: impl FnMut() -> bool) {
    let end = Instant::now() + Duration::from_secs(secs);
    while !done() {
        assert!(Instant::now() < end, "timed out waiting for {what}");
        std::thread::sleep(Duration::from_millis(5));
    }
}

fn source(core: &Core, id: PhotoId) -> PhotoSource {
    PhotoSource::from_session(&core.session, id).expect("source")
}

/// Makes the photos' caches: a frame of each in turn (the last is on screen), the models asked, and the mosaics cached around
/// the photo on screen.
fn fill(rig: &Rig, on_screen: usize) {
    let core = &rig.core;
    let state = ViewportState::new(ViewportRect::cell(300.0, 200.0, 1.0));
    // The renderer keeps three photos' graphs: the neighbours of the one on screen, then that photo last.
    let mut order: Vec<usize> = [on_screen.wrapping_sub(1), on_screen + 1]
        .into_iter()
        .filter(|i| *i < rig.ids.len())
        .collect();
    order.push(on_screen);
    for i in order {
        let id = rig.ids[i];
        core.renderer.navigate(id);
        core.session.set_current(id).expect("current");
        core.renderer
            .viewport_frame(
                &source(core, id),
                &state,
                &FrameOptions::default(),
                &CancelToken::new(),
            )
            .expect("a frame");
    }
    assert!(core.session.wait_idle(Duration::from_secs(120)));
}

#[test]
fn warning_drops_exactly_what_the_spec_lists_and_the_photo_on_screen_keeps_its_caches() {
    let hand = Hand::new();
    let model = depth_model(&hand, None, None);
    let r = rig(|o| o.render.depth_model = Some(Arc::clone(&model)));
    let core = &r.core;
    let (a, b, c, d) = (r.ids[0], r.ids[1], r.ids[2], r.ids[3]);
    // B on screen; A and C are its neighbours; D is far.
    fill(&r, 1);
    // The model has been used.
    core.renderer
        .depth_map(&source(core, b), &CancelToken::new())
        .expect("a map");
    assert!(model.loaded());

    let cached_before = core.session.cached_ids();
    assert!(
        cached_before.contains(&b) && cached_before.contains(&a) && cached_before.contains(&c),
        "the window of B is cached: {cached_before:?}"
    );
    let bytes = |id: PhotoId| core.renderer.graph(id).bytes();
    let on_screen_before = bytes(b);
    assert!(on_screen_before > 0, "the photo on screen has caches");
    // The photos that are not on screen have caches too (A and C are two of the three graphs the renderer keeps).
    assert!(bytes(a) > 0 && bytes(c) > 0);
    let used_before = core.renderer.ledger().used();

    let report = core.trim_caches(PressureLevel::Warning);

    // The neighbours' mosaics are gone; the photo on screen's stays.
    assert_eq!(
        core.session.cached_ids(),
        vec![b],
        "only B's mosaic is kept"
    );
    // The finished-frame caches of the photos that are not on screen are gone, the on-screen photo's are as they were.
    assert_eq!(bytes(a), 0);
    assert_eq!(bytes(c), 0);
    assert_eq!(
        bytes(b),
        on_screen_before,
        "the photo on screen keeps every cache"
    );
    assert!(report.off_screen_photos >= 2);
    assert!(core.renderer.ledger().used() < used_before);
    // The model is unloaded (idle by nobody's clock: a warning does not wait).
    assert!(!model.loaded());
    assert_eq!(report.models, vec![("Depth", Unload::Unloaded)]);
    // Warning leaves the device and the background work alone.
    assert_eq!(report.gpu_bytes, 0);
    assert!(!core.background_gate().is_paused());
    // The next use loads the model again.
    core.renderer
        .depth_map(&source(core, d), &CancelToken::new())
        .expect("a map");
    assert_eq!(model.loads(), 2);

    // Normal frees nothing.
    let before = (core.session.cached_ids(), bytes(b));
    let report = core.trim_caches(PressureLevel::Normal);
    assert_eq!(report, ym_core::memory::TrimReport::default());
    assert_eq!((core.session.cached_ids(), bytes(b)), before);
}

#[test]
fn a_model_that_is_running_keeps_its_weights_and_goes_when_the_inference_ends() {
    let hand = Hand::new();
    let (entered_tx, entered) = mpsc::channel();
    let (release, hold) = mpsc::channel();
    let model = depth_model(&hand, Some(entered_tx), Some(Arc::new(Mutex::new(hold))));
    let r = rig(|o| o.render.depth_model = Some(Arc::clone(&model)));
    let core = Arc::new(r.core);
    let src = source(&core, r.ids[0]);
    let running = {
        let core = Arc::clone(&core);
        std::thread::spawn(move || core.renderer.depth_map(&src, &CancelToken::new()).is_ok())
    };
    entered
        .recv_timeout(Duration::from_secs(60))
        .expect("the inference started");

    // A warning while it runs: the weights stay, the unload is remembered.
    let report = core.trim_caches(PressureLevel::Warning);
    assert_eq!(report.models, vec![("Depth", Unload::InUse)]);
    assert!(model.loaded(), "an inference is using the weights");
    assert_eq!(model.unloads(), 0);
    // A sweep does not take them either.
    assert_eq!(core.sweep_idle_models(), 0);

    release.send(()).expect("let it finish");
    assert!(
        running.join().expect("the inference"),
        "it finished with its weights"
    );
    // The first sweep after it let go unloads them: no clock has moved.
    assert_eq!(core.sweep_idle_models(), 1);
    assert!(!model.loaded());
    assert_eq!(model.loads(), 1);
}

#[test]
fn an_idle_model_unloads_and_the_next_use_loads_again_and_a_use_starts_the_time_again() {
    let hand = Hand::new();
    let model = depth_model(&hand, None, None);
    let r = rig(|o| o.render.depth_model = Some(Arc::clone(&model)));
    let core = &r.core;
    let map = |i: usize| {
        // Each photo has no map yet, so each call asks the model.
        core.renderer
            .depth_map(&source(core, r.ids[i]), &CancelToken::new())
            .expect("a map");
    };
    map(0);
    assert_eq!((model.loads(), model.loaded()), (1, true));

    // 200 s: not idle yet.
    hand.advance(200);
    assert_eq!(core.sweep_idle_models(), 0);
    // A use at 200 s starts the time again: at 400 s it has been idle for 200 s only.
    map(1);
    hand.advance(200);
    assert_eq!(core.sweep_idle_models(), 0);
    assert!(model.loaded());
    assert_eq!(model.loads(), 1);
    // 300 s since the use: unloaded, and the next use loads again.
    hand.advance(100);
    assert_eq!(core.sweep_idle_models(), 1);
    assert!(!model.loaded());
    map(2);
    assert_eq!(model.loads(), 2, "the next use loaded the weights again");
}

#[test]
fn a_model_registered_with_the_core_joins_the_warning_and_the_sweep() {
    struct Counting {
        trims: AtomicUsize,
        sweeps: AtomicUsize,
    }
    impl Unloadable for Counting {
        fn name(&self) -> &'static str {
            "Counting"
        }
        fn sweep(&self) -> Unload {
            self.sweeps.fetch_add(1, Ordering::SeqCst);
            Unload::NotLoaded
        }
        fn trim(&self) -> Unload {
            self.trims.fetch_add(1, Ordering::SeqCst);
            Unload::Unloaded
        }
        fn loaded(&self) -> bool {
            true
        }
    }
    let r = rig(|_| {});
    let model = Arc::new(Counting {
        trims: AtomicUsize::new(0),
        sweeps: AtomicUsize::new(0),
    });
    let as_dyn: Arc<dyn Unloadable> = Arc::clone(&model) as Arc<dyn Unloadable>;
    r.core.register_unloadable(Arc::clone(&as_dyn));
    r.core.register_unloadable(as_dyn);
    let report = r.core.trim_caches(PressureLevel::Warning);
    assert_eq!(
        report.models,
        vec![("Counting", Unload::Unloaded)],
        "once, not twice"
    );
    r.core.sweep_idle_models();
    assert_eq!(
        (
            model.trims.load(Ordering::SeqCst),
            model.sweeps.load(Ordering::SeqCst)
        ),
        (1, 1)
    );
}

/// What a listener of the notice heard.
fn notices(core: &Core) -> Arc<Mutex<Vec<Option<String>>>> {
    let heard = Arc::new(Mutex::new(Vec::new()));
    let h = Arc::clone(&heard);
    core.on_memory_notice(move |text| h.lock().expect("heard").push(text.map(str::to_owned)));
    heard
}

#[test]
fn critical_pauses_the_background_work_until_the_level_falls_and_warning_does_not() {
    let r = rig(|_| {});
    let core = &r.core;
    let gate = core.background_gate();
    let step = core.apply_pressure(PressureLevel::Normal);
    assert!(!step.trimmed && !step.paused && step.notice.is_none());

    let step = core.apply_pressure(PressureLevel::Warning);
    assert!(step.trimmed, "the level rose");
    assert!(!step.paused && !gate.is_paused(), "warning pauses nothing");
    assert_eq!(gate.pauses(), 0);

    let step = core.apply_pressure(PressureLevel::Critical);
    assert!(step.trimmed && step.paused && gate.is_paused());
    assert_eq!(gate.pauses(), 1);

    // The same reading again changes nothing and trims nothing.
    let step = core.apply_pressure(PressureLevel::Critical);
    assert!(!step.trimmed && step.paused);
    assert_eq!(gate.pauses(), 1);

    // Down to warning: the jobs resume (the gate is for critical only), and nothing is trimmed on the way down.
    let step = core.apply_pressure(PressureLevel::Warning);
    assert!(!step.trimmed && !step.paused && !gate.is_paused());
    assert_eq!(gate.resumes(), 1);

    // Critical again, then normal.
    core.apply_pressure(PressureLevel::Critical);
    assert!(gate.is_paused());
    core.apply_pressure(PressureLevel::Normal);
    assert!(!gate.is_paused());
    assert_eq!((gate.pauses(), gate.resumes()), (2, 2));
}

#[test]
fn prefetches_wait_while_critical_and_run_when_the_level_falls() {
    let r = rig(|_| {});
    let core = &r.core;
    let (a, b, c, d) = (r.ids[0], r.ids[1], r.ids[2], r.ids[3]);
    core.session.set_current(a).expect("current");
    assert!(core.session.wait_idle(Duration::from_secs(60)));
    // The window of A is A and B. Critical drops B's mosaic and pauses the pool's optional work.
    core.apply_pressure(PressureLevel::Critical);
    assert_eq!(core.session.cached_ids(), vec![a]);
    let started = core.session.jobs_started();
    // Moving to C wants B, C and D in the window: those are prefetches, and they must wait.
    core.session.set_current(c).expect("current");
    assert!(
        core.session.pending_jobs() > 0,
        "the prefetches are queued, not dropped"
    );
    assert_eq!(
        core.session.jobs_started(),
        started,
        "and none of them started while the gate is paused"
    );
    assert!(
        core.session
            .cached_ids()
            .iter()
            .all(|id| ![b, d].contains(id))
    );
    // The level falls: they run.
    core.apply_pressure(PressureLevel::Normal);
    assert!(core.session.wait_idle(Duration::from_secs(60)));
    assert!(core.session.jobs_started() > started);
    let cached = core.session.cached_ids();
    assert!(
        cached.contains(&b) && cached.contains(&c) && cached.contains(&d),
        "{cached:?}"
    );
}

#[test]
fn the_notice_shows_once_per_stay_above_normal_and_only_for_critical() {
    let r = rig(|_| {});
    let core = &r.core;
    let heard = notices(core);
    let say = |level| core.apply_pressure(level).notice;

    assert_eq!(say(PressureLevel::Normal), None);
    // Warning shows nothing.
    assert_eq!(say(PressureLevel::Warning), None);
    assert!(heard.lock().unwrap().is_empty());
    // Into critical: shown once, with the words of the spec.
    assert_eq!(say(PressureLevel::Critical), Some(true));
    assert_eq!(
        say(PressureLevel::Critical),
        None,
        "not again while it stays critical"
    );
    assert_eq!(say(PressureLevel::Critical), None);
    // Down to warning and up again: the notice never went away, so it does not show again.
    assert_eq!(say(PressureLevel::Warning), None);
    assert_eq!(say(PressureLevel::Critical), None);
    // Normal takes it away; the next entry shows it again.
    assert_eq!(say(PressureLevel::Normal), Some(false));
    assert_eq!(say(PressureLevel::Normal), None);
    assert_eq!(say(PressureLevel::Critical), Some(true));
    // Straight from normal to critical, with no warning between, shows it too.
    assert_eq!(say(PressureLevel::Normal), Some(false));
    assert_eq!(say(PressureLevel::Critical), Some(true));

    let heard = heard.lock().unwrap().clone();
    let words = Some(MEMORY_NOTICE_SENTENCE.to_owned());
    assert_eq!(heard, vec![words.clone(), None, words.clone(), None, words]);
    assert_eq!(
        MEMORY_NOTICE_SENTENCE,
        "Your Mac is short of memory, so Yellowmoon paused background work. Close other apps to speed it up again."
    );
}

/// A source that answers the levels it is given, then the last one for ever, and counts the readings.
fn scripted(levels: &[PressureLevel]) -> (PressureSource, Arc<()>) {
    let script = Mutex::new(VecDeque::from(levels.to_vec()));
    let last = *levels.last().expect("a level");
    let probe = Arc::new(());
    let held = Arc::clone(&probe);
    let source = PressureSource::new(move || {
        let _held = &held;
        script.lock().expect("script").pop_front().unwrap_or(last)
    });
    (source, probe)
}

#[test]
fn the_poll_thread_acts_on_each_reading_and_ends_with_shutdown() {
    use PressureLevel::{Critical, Normal, Warning};
    let r = rig(|_| {});
    let core = &r.core;
    let heard = notices(core);
    let (source, probe) = scripted(&[Warning, Critical, Critical, Normal]);
    // A short interval, so the script is read quickly; `memory_polls` says when the readings are done.
    core.start_memory_watch(source, Duration::from_millis(1))
        .expect("the thread starts");
    wait_until("four readings", 60, || core.memory_polls() >= 4);
    // After the script the source says normal for ever: the notice came and went, once.
    wait_until("the notice to go", 60, || heard.lock().unwrap().len() == 2);
    assert_eq!(
        heard.lock().unwrap().clone(),
        vec![Some(MEMORY_NOTICE_SENTENCE.to_owned()), None]
    );
    assert!(!core.background_gate().is_paused());
    assert_eq!(core.background_gate().pauses(), 1);

    assert_eq!(Arc::strong_count(&probe), 2, "the thread holds the source");
    core.shutdown();
    // `shutdown` joined the thread: it let go of the source, and it can read no more.
    assert_eq!(Arc::strong_count(&probe), 1, "the thread is gone");
    let polls = core.memory_polls();
    assert_eq!(core.memory_polls(), polls);
}

#[test]
fn a_thread_waiting_out_a_long_interval_stops_at_once_and_a_dropped_core_leaves_none() {
    let r = rig(|_| {});
    let (source, probe) = scripted(&[PressureLevel::Normal]);
    r.core
        .start_memory_watch(source.clone(), Duration::from_secs(3600))
        .expect("the thread starts");
    drop(source);
    wait_until("the first reading", 60, || r.core.memory_polls() >= 1);
    // It is now asleep for an hour; stopping it must not wait for that.
    let began = Instant::now();
    r.core.stop_memory_watch();
    assert!(began.elapsed() < Duration::from_secs(30));
    assert_eq!(
        Arc::strong_count(&probe),
        1,
        "the thread let go of the source"
    );

    // A core dropped with its thread running ends it too.
    let (source, probe) = scripted(&[PressureLevel::Normal]);
    r.core
        .start_memory_watch(source, Duration::from_secs(3600))
        .expect("the thread starts");
    let Rig { core, .. } = r;
    wait_until("the first reading", 60, || core.memory_polls() >= 2);
    drop(core);
    assert_eq!(
        Arc::strong_count(&probe),
        1,
        "dropping the core ended the thread"
    );
}

#[test]
fn critical_trims_the_pool_and_warning_does_not() {
    use ym_core::app::gpu::GpuStages;
    use ym_gpu::device::{Gpu, GpuConfig};
    use ym_gpu::queue::{JobSpec, Priority};

    let Ok(gpu) = Gpu::start(GpuConfig {
        poison: true,
        ..GpuConfig::default()
    }) else {
        eprintln!("* GPU TEST SKIPPED in memory_pressure: no adapter opens");
        return;
    };
    let gpu = Arc::new(gpu);
    let r = rig(|o| {
        o.render.gpu = Some(Arc::clone(&gpu));
        o.render.gpu_stages = GpuStages::ALL;
    });
    fill(&r, 1);
    let pool = || {
        gpu.run(JobSpec::new(Priority::Viewport), |ctx, _| {
            Ok(ctx.pool().stats())
        })
        .expect("pool stats")
    };
    let free =
        |s: &ym_gpu::pool::PoolStats| s.allocated_bytes - s.in_use_bytes - s.quarantined_bytes;
    let before = pool();
    assert!(
        free(&before) > 0,
        "the frames left free buffers in the pool: {before:?}"
    );

    // Warning does not trim the pool: nothing is destroyed (the caches it drops hand their buffers to the pool's free list, so
    // the free bytes may grow, and the allocation stays).
    let report = r.core.trim_caches(PressureLevel::Warning);
    assert_eq!(report.gpu_bytes, 0);
    let after_warning = pool();
    assert_eq!(after_warning.allocated_bytes, before.allocated_bytes);
    assert!(free(&after_warning) >= free(&before));

    // Critical gives the free buffers back.
    let report = r.core.trim_caches(PressureLevel::Critical);
    // At least what was free (the raw words of the other photos went into the free list first).
    assert!(report.gpu_bytes >= free(&after_warning), "{report:?}");
    assert_eq!(free(&pool()), 0);
}
