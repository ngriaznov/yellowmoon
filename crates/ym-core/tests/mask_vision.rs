//! Subject and People masks through the render graph (`docs/23-nodes.md` sections 3.10, 4.5 and 7.6,
//! `docs/algorithms/06-masks.md` section 5.3, task N-19): a row's raster `S0` is the union of the instances one Vision request
//! finds in the Input's picture. The requests need a Mac, so these tests play the Mac with a `FakeVision` provider (and
//! "a Mac that does not support it" and "not a Mac" with the others): they check everything around Vision, on every platform.
//! What Vision itself makes is checked on a Mac by `crates/ym-apple` and by CI's macOS jobs.
mod common;

use std::io::Write as _;
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use common::*;
use serde_json::{Value, json};
use ym_color::display::OutputTarget;
use ym_core::app::gpu::GpuStages;
use ym_core::app::{Core, CoreOptions};
use ym_core::pipeline::{
    ComputeDevice, FrameOptions, FullImage, FullRequest, FullTarget, MaskStyle, MaskView,
    NeutralOnly, PhotoSource, RenderOptions, RenderedFrame, Renderer, ViewportRect, ViewportState,
    VisionPhase,
};
use ym_core::session::PhotoId;
use ym_core::settings::{
    DevelopSettings, MaskKind, MaskSupport, MaskUnavailable, SettingsPatch, mask_kinds_with,
    mask_unavailable_with,
};
use ym_core::vision::{
    FakeAnswer, FakeVision, NoVision, VisionError, VisionKind, VisionMask, VisionMasks,
    VisionPicture, VisionProvider, VisionStatus, VisionSupport,
};
use ym_gpu::device::{Gpu, GpuConfig, gpu_required_by_env};
use ym_gpu::queue::CancelToken;

const MODEL: &str = "ILCE-7RM4";
const PROFILE: &str = "sony-ilce-7rm4";
/// The paper around a photo in a Mask only frame.
const PAPER: [u8; 3] = [40, 60, 90];
const CELL: f64 = 300.0;

/// What the sentences of `docs/23` section 3.10 say, word for word.
const SUBJECT_SENTENCE: &str = "Node 2 has a Subject mask, and this computer cannot make one. Use a Mac that supports Subject masks, or remove the mask.";
const PEOPLE_SENTENCE: &str = "Node 2 has a People mask, and this computer cannot make one. Use a Mac that supports People masks, or remove the mask.";

// ---- helpers -----------------------------------------------------------------------------------------

/// A provider that counts, remembers the pictures it was given, and can be held shut: a request blocks inside the provider
/// until `open` is called, as Vision does for a second or two.
#[derive(Debug)]
struct Probe {
    inner: FakeVision,
    open: Mutex<bool>,
    cv: Condvar,
    pictures: Mutex<Vec<(VisionKind, VisionPicture)>>,
}

impl Probe {
    fn new(inner: FakeVision) -> Arc<Self> {
        Arc::new(Self {
            inner,
            open: Mutex::new(true),
            cv: Condvar::new(),
            pictures: Mutex::new(Vec::new()),
        })
    }

    fn shut(&self) {
        *self.open.lock().unwrap() = false;
    }

    fn open(&self) {
        *self.open.lock().unwrap() = true;
        self.cv.notify_all();
    }

    fn pictures(&self) -> Vec<(VisionKind, VisionPicture)> {
        self.pictures.lock().unwrap().clone()
    }
}

impl VisionProvider for Probe {
    fn probe(&self) -> VisionSupport {
        self.inner.probe()
    }

    fn mask(&self, kind: VisionKind, picture: &VisionPicture) -> Result<VisionMask, VisionError> {
        self.pictures.lock().unwrap().push((kind, picture.clone()));
        let mut open = self.open.lock().unwrap();
        while !*open {
            open = self.cv.wait(open).unwrap();
        }
        drop(open);
        self.inner.mask(kind, picture)
    }
}

fn options(provider: Option<Arc<dyn VisionProvider>>, tile: usize) -> RenderOptions {
    RenderOptions {
        vision: provider.map(VisionMasks::new),
        data_root: Some(data_root()),
        background: PAPER,
        full_tile_px: tile,
        ..RenderOptions::default()
    }
}

fn renderer_with(provider: Option<Arc<dyn VisionProvider>>, tile: usize) -> Renderer {
    Renderer::new(Arc::new(NeutralOnly), options(provider, tile)).expect("render thread")
}

fn renderer(provider: Arc<dyn VisionProvider>) -> Renderer {
    renderer_with(Some(provider), RenderOptions::default().full_tile_px)
}

fn fake() -> Arc<FakeVision> {
    Arc::new(FakeVision::new())
}

/// Input, Main, node 2 (with these rows and an exposure trim of a stop), Output.
fn graph(rows: Vec<Value>) -> DevelopSettings {
    let doc = json!({ "nodes": { "items": [
        { "id": 1 },
        { "id": 2, "kind": "corrector", "pos": [1.0, 0.0], "trim": { "exposure_ev": 1.0 },
          "masks": { "invert": false, "items": rows } },
    ], "links": [
        { "from": { "node": "input", "port": "picture" }, "to": { "node": 1, "port": "picture" } },
        { "from": { "node": 1, "port": "picture" }, "to": { "node": 2, "port": "picture" } },
        { "from": { "node": 2, "port": "picture" }, "to": { "node": "output", "port": "picture" } },
    ] } });
    serde_json::from_value::<DevelopSettings>(doc)
        .expect("a document")
        .validated()
        .expect("a valid graph")
        .0
}

fn row(kind: &str, extra: Value) -> Value {
    let mut row = json!({ "kind": kind, "mode": "add" });
    if let (Some(r), Some(e)) = (row.as_object_mut(), extra.as_object()) {
        r.extend(e.clone());
    }
    row
}

fn subject() -> Value {
    row("subject", json!({}))
}

fn people() -> Value {
    row("people", json!({}))
}

fn mask_view(node: u32) -> ViewportState {
    let mut st = ViewportState::new(ViewportRect::cell(CELL, CELL, 1.0));
    st.mask_view = Some(MaskView {
        node,
        style: MaskStyle::MaskOnly,
    });
    st
}

fn picture_view() -> ViewportState {
    ViewportState::new(ViewportRect::cell(300.0, 200.0, 1.0))
}

fn frame(r: &Renderer, src: &PhotoSource, st: &ViewportState) -> RenderedFrame {
    r.viewport_frame(src, st, &FrameOptions::default(), &CancelToken::new())
        .expect("frame")
}

fn coverage(f: &RenderedFrame) -> f64 {
    let v: Vec<f64> = f
        .rgba
        .chunks(4)
        .filter(|p| p[..3] != PAPER)
        .map(|p| f64::from(p[0]) / 255.0)
        .collect();
    v.iter().sum::<f64>() / v.len().max(1) as f64
}

fn export_image(r: &Renderer, src: &PhotoSource) -> FullImage {
    r.render_full(
        src,
        &FullRequest {
            target: FullTarget::DisplayLinear(OutputTarget::Srgb),
        },
        &CancelToken::new(),
    )
    .expect("render")
}

fn wait_until(what: &str, secs: u64, mut done: impl FnMut() -> bool) {
    let end = Instant::now() + Duration::from_secs(secs);
    while !done() {
        assert!(Instant::now() < end, "timed out waiting for {what}");
        std::thread::sleep(Duration::from_millis(20));
    }
}

/// The row appears: the worker is wanted, and the test waits until it has ended its work for the photo.
fn make_ready(r: &Renderer, src: &PhotoSource) {
    r.vision_prepare(src);
    wait_until("the Vision worker", 60, || !r.vision_working(src.id));
}

/// A photo with a bright disc on a darker ground (its brightness centroid is the disc), at `cx` of the width.
fn disc_photo_at(cx_frac: f64) -> Developed {
    let (w, h) = (1200u32, 800u32);
    let scene = Scene::new(w, h, light(PROFILE, 5500.0), Layout::Uniform, 0.08).noise(2e-4, 1e-6);
    let mut mosaic = scene.mosaic();
    let code = (scene.black + 0.6 * (scene.white_level - scene.black)) as u16;
    let (cx, cy, r) = (
        cx_frac * f64::from(w),
        0.5 * f64::from(h),
        0.2 * f64::from(h),
    );
    for y in 0..h {
        for x in 0..w {
            if (f64::from(x) + 0.5 - cx).hypot(f64::from(y) + 0.5 - cy) <= r {
                mosaic[(y * w + x) as usize] = code;
            }
        }
    }
    let dir = tempfile::tempdir().expect("temp dir");
    scene
        .builder(MODEL)
        .write_to(&dir.path().join("DSC00001.ARW"), &mosaic)
        .expect("write the synthetic ARW");
    Developed::open(dir)
}

fn disc_photo() -> Developed {
    disc_photo_at(0.45)
}

/// The photo with node 2 holding these rows.
fn with_rows(src: &PhotoSource, rows: Vec<Value>) -> PhotoSource {
    src.with_settings(graph(rows))
}

// ---- the kinds and the probe ---------------------------------------------------------------------------

#[test]
fn mask_kinds_follow_the_probe_in_the_order_of_the_menu() {
    use MaskKind::*;
    let everything = [
        Object,
        Subject,
        People,
        Background,
        Linear,
        Radial,
        Brush,
        ColorRange,
        BrightnessRange,
    ];
    // The pure lists, for each answer the probe can give.
    let both = MaskSupport::new(true, VisionSupport::BOTH);
    assert_eq!(mask_kinds_with(both), everything);
    assert!(mask_unavailable_with(both).is_empty());
    let no_model = MaskSupport::new(false, VisionSupport::BOTH);
    assert_eq!(mask_kinds_with(no_model)[0], Subject);
    assert_eq!(
        mask_unavailable_with(no_model),
        [(Object, MaskUnavailable::WeightsMissing)]
    );
    let one = |subject, people| MaskSupport::new(true, VisionSupport { subject, people });
    let no = VisionStatus::Unsupported(MaskUnavailable::NotSupported);
    assert_eq!(
        mask_kinds_with(one(VisionStatus::Supported, no)),
        [
            Object,
            Subject,
            Background,
            Linear,
            Radial,
            Brush,
            ColorRange,
            BrightnessRange
        ]
    );
    assert_eq!(
        mask_kinds_with(one(no, VisionStatus::Supported)),
        [
            Object,
            People,
            Background,
            Linear,
            Radial,
            Brush,
            ColorRange,
            BrightnessRange
        ]
    );
    for why in [
        MaskUnavailable::NotAMac,
        MaskUnavailable::MacosTooOld,
        MaskUnavailable::NotSupported,
    ] {
        let support = MaskSupport::new(true, VisionSupport::neither(why));
        assert_eq!(
            mask_kinds_with(support),
            [
                Object,
                Background,
                Linear,
                Radial,
                Brush,
                ColorRange,
                BrightnessRange
            ],
            "{why:?}"
        );
        assert_eq!(
            mask_unavailable_with(support),
            [(Subject, why), (People, why)],
            "{why:?}"
        );
    }

    // A renderer says the same, from its provider's probe.
    let r = renderer(fake());
    // (Object is not in a renderer made with no SAM model.)
    assert_eq!(r.mask_kinds()[..2], [Subject, People]);
    let r = renderer(Arc::new(
        FakeVision::new().with_support(VisionSupport::neither(MaskUnavailable::MacosTooOld)),
    ));
    assert!(!r.mask_kinds().contains(&Subject) && !r.mask_kinds().contains(&People));
    // Not a Mac: no provider at all, and the provider that says so.
    let bare = renderer_with(None, 1024);
    assert_eq!(
        bare.mask_kinds(),
        [
            Background,
            Linear,
            Radial,
            Brush,
            ColorRange,
            BrightnessRange
        ]
    );
    let none = renderer(Arc::new(NoVision));
    assert_eq!(none.mask_kinds(), bare.mask_kinds());
    assert_eq!(none.vision_support(), VisionSupport::none_here());
}

#[test]
fn the_probe_runs_once_per_run_however_often_it_is_asked() {
    let provider = fake();
    let r = renderer(provider.clone() as Arc<dyn VisionProvider>);
    assert_eq!(
        provider.probes(),
        0,
        "nothing is probed until something asks"
    );
    for _ in 0..4 {
        let _ = r.mask_kinds();
        let _ = r.vision_support();
    }
    let d = disc_photo();
    let src = with_rows(&d.source(), vec![subject(), people()]);
    make_ready(&r, &src);
    let _ = r.unmakeable_mask(&src.settings);
    assert_eq!(provider.probes(), 1);
    // A second renderer built on the same wrapper reads the same answer.
    let masks = VisionMasks::new(provider.clone() as Arc<dyn VisionProvider>);
    let _ = masks.support();
    let _ = masks.support();
    assert_eq!(
        provider.probes(),
        2,
        "one more for the new wrapper, one for each"
    );
}

#[test]
fn a_request_the_probe_refused_is_never_asked_for() {
    let provider = Probe::new(FakeVision::new().with_support(VisionSupport {
        subject: VisionStatus::Supported,
        people: VisionStatus::Unsupported(MaskUnavailable::NotSupported),
    }));
    let r = renderer(provider.clone());
    let d = disc_photo();
    let src = with_rows(&d.source(), vec![subject(), people()]);
    make_ready(&r, &src);
    let kinds: Vec<VisionKind> = provider.pictures().iter().map(|(k, _)| *k).collect();
    assert_eq!(kinds, [VisionKind::Subject]);
    // An export of the photo is refused for the People row alone.
    assert_eq!(
        r.unmakeable_mask(&src.settings).as_deref(),
        Some(PEOPLE_SENTENCE)
    );
    let subject_only = with_rows(&d.source(), vec![subject()]);
    assert_eq!(r.unmakeable_mask(&subject_only.settings), None);
}

// ---- the sentences of section 3.10 --------------------------------------------------------------------

#[test]
fn a_row_this_computer_cannot_make_is_the_mask_error_with_the_sentence_of_the_spec() {
    let d = disc_photo();
    let subject_row = with_rows(&d.source(), vec![subject()]);
    let people_row = with_rows(&d.source(), vec![people()]);
    for (what, provider) in [
        ("no provider", None),
        (
            "not a Mac",
            Some(Arc::new(NoVision) as Arc<dyn VisionProvider>),
        ),
        (
            "macOS too old",
            Some(Arc::new(
                FakeVision::new()
                    .with_support(VisionSupport::neither(MaskUnavailable::MacosTooOld)),
            ) as Arc<dyn VisionProvider>),
        ),
        (
            "a Mac that cannot run the requests",
            Some(Arc::new(
                FakeVision::new()
                    .with_support(VisionSupport::neither(MaskUnavailable::NotSupported)),
            ) as Arc<dyn VisionProvider>),
        ),
    ] {
        let r = renderer_with(provider, 1024);
        assert_eq!(
            r.unmakeable_mask(&subject_row.settings).as_deref(),
            Some(SUBJECT_SENTENCE),
            "{what}"
        );
        assert_eq!(
            r.unmakeable_mask(&people_row.settings).as_deref(),
            Some(PEOPLE_SENTENCE),
            "{what}"
        );
    }
    // The node's number is filled in.
    let s = ym_core::settings::unmakeable_mask_sentence(5, MaskKind::Subject).expect("sentence");
    assert_eq!(
        s,
        "Node 5 has a Subject mask, and this computer cannot make one. Use a Mac that supports Subject masks, or remove the mask."
    );
    // A row with the eye off, and a computer that can, are no problem.
    let off = with_rows(
        &d.source(),
        vec![row("subject", json!({ "enabled": false }))],
    );
    assert_eq!(
        renderer_with(None, 1024).unmakeable_mask(&off.settings),
        None
    );
    assert_eq!(
        renderer(fake()).unmakeable_mask(&subject_row.settings),
        None
    );
    // The row's own words in the Mask tab.
    assert_eq!(
        ym_core::settings::unsupported_row_sentence(MaskKind::Subject).as_deref(),
        Some("Subject masks need a Mac that supports them")
    );
    assert_eq!(
        ym_core::settings::unsupported_row_sentence(MaskKind::People).as_deref(),
        Some("People masks need a Mac that supports them")
    );
}

#[test]
fn a_view_shows_an_unsupported_row_selecting_nothing_and_the_photo_renders() {
    let d = disc_photo();
    let plain = frame(&renderer_with(None, 1024), &d.source(), &picture_view());
    for provider in [
        None,
        Some(Arc::new(NoVision) as Arc<dyn VisionProvider>),
        Some(Arc::new(
            FakeVision::new().with_support(VisionSupport::neither(MaskUnavailable::NotSupported)),
        ) as Arc<dyn VisionProvider>),
    ] {
        let r = renderer_with(provider, 1024);
        let src = with_rows(&d.source(), vec![subject(), people()]);
        // The node's trim of a stop acts only through its mask, which is empty: the photo is the plain photo.
        let f = frame(&r, &src, &picture_view());
        assert_eq!(f.rgba, plain.rgba);
        // And nothing asks for a mask: the worker does not run.
        r.vision_prepare(&src);
        assert!(!r.vision_working(src.id));
        assert_eq!(coverage(&frame(&r, &src, &mask_view(2))), 0.0);
    }
}

// ---- a fake provider's mask goes through the raster pass --------------------------------------------

#[test]
fn a_viewer_never_waits_the_row_selects_nothing_until_the_mask_is_made() {
    let provider = Probe::new(FakeVision::new());
    let r = renderer(provider.clone());
    let heard = Arc::new(Mutex::new(Vec::<(PhotoId, VisionPhase)>::new()));
    {
        let heard = Arc::clone(&heard);
        r.set_vision_listener(Arc::new(move |id, phase| {
            heard.lock().unwrap().push((id, phase));
        }));
    }
    let d = disc_photo();
    let plain = frame(&r, &d.source(), &picture_view());
    let src = with_rows(&d.source(), vec![subject()]);
    provider.shut();

    // A frame with the row: it comes back at once, with the row selecting nothing, and the worker is asked for the mask.
    let t = Instant::now();
    let first = frame(&r, &src, &picture_view());
    assert!(
        t.elapsed() < Duration::from_secs(20),
        "the frame waited ({:?})",
        t.elapsed()
    );
    assert_eq!(first.rgba, plain.rgba, "the row selected nothing");
    assert!(
        r.vision_working(src.id),
        "the worker was asked for the mask"
    );
    assert!(
        r.finding_masks(src.id, &src.settings),
        "the app says Finding… (viewport_busy.finding)"
    );
    // Frames while the request is running are quick and still empty; they ask for nothing more.
    for _ in 0..3 {
        let again = frame(&r, &src, &picture_view());
        assert_eq!(again.rgba, plain.rgba);
    }
    wait_until("the request to reach Vision", 60, || {
        !provider.pictures().is_empty()
    });
    assert_eq!(provider.pictures().len(), 1, "one request, not one a frame");
    assert!(r.finding_masks(src.id, &src.settings));

    // Vision answers: the worker ends, the listener hears it, and the next frame has the mask.
    provider.open();
    wait_until("the worker to end", 60, || !r.vision_working(src.id));
    assert!(!r.finding_masks(src.id, &src.settings));
    let made = frame(&r, &src, &picture_view());
    assert!(made.rgba != plain.rgba, "the row acts");
    let c = coverage(&frame(&r, &src, &mask_view(2)));
    assert!(c > 0.02 && c < 0.7, "the mask covers {c}");
    assert_eq!(
        *heard.lock().unwrap(),
        [
            (src.id, VisionPhase::Started),
            (
                src.id,
                VisionPhase::Finished {
                    nothing_found: vec![]
                }
            )
        ]
    );
    assert_eq!(r.vision_makes(src.id), 1);
}

#[test]
fn the_mask_is_kept_in_memory_by_the_picture_and_the_request_and_grading_never_makes_it_again() {
    let provider = fake();
    let r = renderer(provider.clone() as Arc<dyn VisionProvider>);
    let d = disc_photo();
    let src = with_rows(&d.source(), vec![subject()]);
    make_ready(&r, &src);
    assert_eq!(provider.masks_made(), 1);
    let base = coverage(&frame(&r, &src, &mask_view(2)));
    assert!(base > 0.02);
    // The same row again, other frames, other views: the cache serves them.
    let _ = frame(&r, &src, &picture_view());
    let _ = frame(&r, &src, &mask_view(2));
    // Grading a node, changing the look, the row's own sliders and another row's mode move nothing of the mask.
    let mut graded = src.settings.clone();
    graded
        .nodes
        .corrector_mut(2)
        .expect("node 2")
        .trim
        .exposure_ev = -2.0;
    graded.output_mut().look_intensity = 40.0;
    if let Some(row) = graded
        .nodes
        .corrector_mut(2)
        .expect("node 2")
        .masks
        .items
        .first_mut()
        && let ym_core::settings::MaskShape::Subject(f) = &mut row.shape
    {
        f.feather = 0.3;
        f.grow = 0.2;
    }
    let graded = src.with_settings(graded);
    r.vision_prepare(&graded);
    assert!(
        !r.vision_working(src.id),
        "nothing to make: the key is the picture's"
    );
    let _ = frame(&r, &graded, &mask_view(2));
    assert_eq!(provider.masks_made(), 1, "the mask was made once");
    // The Input's exposure moves the picture: the mask is made again, from the new picture.
    let mut brighter = src.settings.clone();
    brighter.input_mut().exposure = ym_core::settings::AutoOr::manual_ev(1.0);
    let brighter = src.with_settings(brighter);
    // Until it is made the viewer keeps the latest mask it has: the mask stays put during a drag.
    let during = coverage(&frame(&r, &brighter, &mask_view(2)));
    assert!(
        (during - base).abs() < 1e-9,
        "the old mask stood in: {during} against {base}"
    );
    wait_until("the new mask", 60, || {
        provider.masks_made() == 2 && !r.vision_working(src.id)
    });
    assert_eq!(provider.masks_made(), 2);
    assert!(coverage(&frame(&r, &brighter, &mask_view(2))) > 0.02);
}

#[test]
fn the_picture_vision_reads_is_the_inputs_picture_the_one_the_object_mask_reads() {
    let provider = Probe::new(FakeVision::new());
    let r = renderer(provider.clone());
    let d = disc_photo();
    let src = with_rows(&d.source(), vec![subject(), people()]);
    make_ready(&r, &src);
    let seen = provider.pictures();
    assert_eq!(seen.len(), 2);
    let (w, h, rgb) = r
        .object_picture_of(&src, &CancelToken::new())
        .expect("the picture");
    for (kind, picture) in &seen {
        assert_eq!(
            (picture.width, picture.height),
            (w, h),
            "{kind:?} reads the Object mask's picture"
        );
        assert_eq!(picture.rgb, rgb, "{kind:?}");
    }
    assert_eq!(w.max(h), 1024, "1024 px on the long edge");
    // No node's grade and no look reach it.
    let mut graded = src.settings.clone();
    graded
        .nodes
        .corrector_mut(2)
        .expect("node 2")
        .trim
        .exposure_ev = 3.0;
    graded.output_mut().output_sharpening = 80.0;
    let (_, _, again) = r
        .object_picture_of(&src.with_settings(graded), &CancelToken::new())
        .expect("the picture");
    assert_eq!(again, rgb);
}

#[test]
fn subject_and_people_are_separate_requests_with_separate_planes() {
    let provider = fake();
    let r = renderer(provider.clone() as Arc<dyn VisionProvider>);
    let d = disc_photo();
    let both = with_rows(&d.source(), vec![subject(), people()]);
    make_ready(&r, &both);
    assert_eq!(provider.masks_made(), 2);
    let s = r
        .vision_plane(&both, VisionKind::Subject, &CancelToken::new())
        .expect("subject")
        .expect("a plane");
    let p = r
        .vision_plane(&both, VisionKind::People, &CancelToken::new())
        .expect("people")
        .expect("a plane");
    assert_ne!(s.key, p.key);
    assert_ne!(s.data, p.data);
    assert_eq!((s.width, s.height), (p.width, p.height));
    assert_eq!(s.width.max(s.height), 1024);
    assert_eq!(provider.masks_made(), 2, "vision_plane read the cache");
    // The rows act on their own.
    let only_people = with_rows(&d.source(), vec![people()]);
    let a = coverage(&frame(&r, &only_people, &mask_view(2)));
    let b = coverage(&frame(
        &r,
        &with_rows(&d.source(), vec![subject()]),
        &mask_view(2),
    ));
    assert!(a > 0.0 && b > 0.0 && (a - b).abs() > 0.005, "{a} {b}");
}

// ---- nothing found, a failure ---------------------------------------------------------------------

#[test]
fn nothing_found_gives_an_empty_mask_and_is_reported_not_retried() {
    let provider =
        Probe::new(FakeVision::new().answering(VisionKind::People, FakeAnswer::NothingFound));
    let r = renderer(provider.clone());
    let heard = Arc::new(Mutex::new(Vec::<VisionPhase>::new()));
    {
        let heard = Arc::clone(&heard);
        r.set_vision_listener(Arc::new(move |_, phase| heard.lock().unwrap().push(phase)));
    }
    let d = disc_photo();
    let plain = frame(&r, &d.source(), &picture_view());
    let src = with_rows(&d.source(), vec![people()]);
    make_ready(&r, &src);
    assert_eq!(r.vision_nothing_found(src.id), [MaskKind::People]);
    assert_eq!(
        heard.lock().unwrap().last(),
        Some(&VisionPhase::Finished {
            nothing_found: vec![MaskKind::People]
        })
    );
    // S0 = 0: the row selects nothing and the node changes nothing, in a view and in an export.
    assert_eq!(frame(&r, &src, &picture_view()).rgba, plain.rgba);
    assert_eq!(coverage(&frame(&r, &src, &mask_view(2))), 0.0);
    let exported = export_image(&r, &src);
    let unmasked = export_image(&r, &d.source());
    assert_eq!(exported.rgb, unmasked.rgb);
    // Invert on a row that selected nothing is the whole photo (`docs/23` section 4.2).
    let inverted = with_rows(&d.source(), vec![row("people", json!({ "invert": true }))]);
    assert!((coverage(&frame(&r, &inverted, &mask_view(2))) - 1.0).abs() < 1e-9);
    // Not asked again: the answer is kept like a mask.
    r.vision_prepare(&src);
    assert!(!r.vision_working(src.id));
    assert_eq!(provider.pictures().len(), 1);
    // A request that found something next to it does not hide it.
    let both = with_rows(&d.source(), vec![subject(), people()]);
    make_ready(&r, &both);
    assert_eq!(r.vision_nothing_found(src.id), [MaskKind::People]);
    assert!(
        coverage(&frame(
            &r,
            &with_rows(&d.source(), vec![subject()]),
            &mask_view(2)
        )) > 0.0
    );
}

#[test]
fn a_request_that_fails_selects_nothing_in_a_view_is_not_retried_and_fails_an_export() {
    let provider = Probe::new(FakeVision::new().answering(VisionKind::Subject, FakeAnswer::Fail));
    let r = renderer(provider.clone());
    let d = disc_photo();
    let plain = frame(&r, &d.source(), &picture_view());
    let src = with_rows(&d.source(), vec![subject()]);
    make_ready(&r, &src);
    assert_eq!(frame(&r, &src, &picture_view()).rgba, plain.rgba);
    assert_eq!(
        r.vision_nothing_found(src.id),
        [],
        "a failure is not 'nothing found'"
    );
    // A redraw does not ask again (it would make a loop with the page's redraw).
    r.vision_prepare(&src);
    assert!(!r.vision_working(src.id));
    assert_eq!(provider.pictures().len(), 1);
    // An export asks itself, and fails with the mask error and a plain sentence.
    let error = r
        .render_full(
            &src,
            &FullRequest {
                target: FullTarget::DisplayLinear(OutputTarget::Srgb),
            },
            &CancelToken::new(),
        )
        .expect_err("the export fails");
    assert!(
        matches!(&error, ym_core::pipeline::RenderError::Mask(m)
            if m == "The Subject mask can't be made: The fake Vision failed. Try again, or remove the mask."),
        "{error:?}"
    );
}

// ---- exports ------------------------------------------------------------------------------------------

#[test]
fn an_export_makes_its_own_mask_and_reuses_the_viewers_only_for_the_same_picture() {
    let provider = fake();
    let r = renderer(provider.clone() as Arc<dyn VisionProvider>);
    let d = disc_photo();
    let src = with_rows(&d.source(), vec![subject()]);
    // No viewer has made it: the export makes it, once for the export however many tiles it has.
    let first = export_image(&r, &src);
    assert_eq!(provider.masks_made(), 1);
    assert!(
        first.rgb != export_image(&r, &d.source()).rgb,
        "the row acts"
    );
    assert!(!r.vision_working(src.id), "an export starts no worker");
    // The viewer's mask, of the same picture, serves the next export.
    make_ready(&r, &src);
    let n = provider.masks_made();
    let second = export_image(&r, &src);
    assert_eq!(
        provider.masks_made(),
        n,
        "the viewer's mask served the export"
    );
    assert_eq!(second.rgb, first.rgb, "and gave the same bytes");
}

#[test]
fn an_export_is_the_same_bytes_in_two_renderers_and_in_tiles() {
    let d = disc_photo();
    let rows = vec![
        row(
            "subject",
            json!({ "grow": 0.2, "feather": 0.25, "refine": 0.6 }),
        ),
        row("people", json!({ "mode": "subtract", "feather": 0.1 })),
    ];
    let src = with_rows(&d.source(), rows);
    let whole = export_image(&renderer_with(Some(fake()), 4096), &src);
    let again = export_image(&renderer_with(Some(fake()), 4096), &src);
    assert!(
        again.rgb == whole.rgb,
        "a second renderer gives other bytes"
    );
    // Tiles equal the whole frame bit for bit, the halos of Grow, Feather and Refine edges included.
    for tile in [64, 100, 192] {
        let tiled = export_image(&renderer_with(Some(fake()), tile), &src);
        assert_eq!((tiled.width, tiled.height), (whole.width, whole.height));
        assert!(
            tiled.rgb == whole.rgb,
            "tiles of {tile} px differ from the whole frame"
        );
    }
    // The rows act.
    let none = export_image(&renderer_with(Some(fake()), 4096), &d.source());
    assert!(none.rgb != whole.rgb);
    // And a viewer's frames are the same in two renderers.
    let (a, b) = (renderer(fake()), renderer(fake()));
    make_ready(&a, &src);
    make_ready(&b, &src);
    assert!(frame(&a, &src, &mask_view(2)).rgba == frame(&b, &src, &mask_view(2)).rgba);
}

// ---- the app's path: Core, paste ----------------------------------------------------------------------

#[test]
fn the_core_asks_for_the_masks_when_the_row_appears_and_a_paste_works_them_out_again() {
    let provider = Probe::new(FakeVision::new());
    let app = tempfile::tempdir().expect("app root");
    let mut o = CoreOptions::new(data_root(), app.path().to_path_buf());
    o.decode_threads = 2;
    o.thumbnails = false;
    o.render.vision = Some(VisionMasks::new(provider.clone()));
    let core = Core::open(o).expect("core");
    // The menu lists the kinds the probe allows.
    // (After Object where the SAM weights are there: the order of the Add mask menu.)
    let kinds: Vec<MaskKind> = core
        .mask_kinds()
        .into_iter()
        .filter(|k| *k != MaskKind::Object)
        .collect();
    assert_eq!(kinds[..2], [MaskKind::Subject, MaskKind::People]);
    assert!(
        core.mask_unavailable()
            .iter()
            .all(|(k, _)| !matches!(k, MaskKind::Subject | MaskKind::People))
    );

    // Two photos with their bright discs in different places.
    let (left, right) = (disc_photo_at(0.3), disc_photo_at(0.7));
    let mut ids = Vec::new();
    for d in [&left, &right] {
        let opened = core.session.open(
            &[d.dir.path().to_path_buf()],
            ym_core::session::OpenOptions::default(),
        );
        assert!(core.session.wait_idle(Duration::from_secs(120)));
        ids.push(opened.added[0]);
    }
    let (a, b) = (ids[0], ids[1]);
    core.session.set_current(a).expect("current");
    core.renderer.navigate(a);
    assert!(!core.renderer.vision_working(a), "no row, no worker");

    // The patch that makes the row asks for the mask in the background.
    let patch: SettingsPatch = serde_json::from_value(
        json!({ "nodes": { "1": { "masks": { "items": [ { "kind": "subject" } ] } } } }),
    )
    .expect("a patch");
    core.update_settings(a, &patch).expect("the row is added");
    assert!(core.renderer.vision_working(a), "the core asked at once");
    wait_until("the worker", 60, || !core.renderer.vision_working(a));
    assert_eq!(provider.pictures().len(), 1);
    let src_a = PhotoSource::from_session(&core.session, a).expect("source");
    let plane_a = core
        .renderer
        .vision_plane(&src_a, VisionKind::Subject, &CancelToken::new())
        .expect("a plane")
        .expect("found");

    // Copy, then paste onto the other photo: the row is the same, its mask is worked out again from that photo's picture.
    let copied = core.copy_settings(a).expect("copy");
    core.session.set_current(b).expect("current");
    core.renderer.navigate(b);
    assert!(!core.renderer.vision_working(b));
    let report = core.paste_settings(&[b], &copied).expect("paste");
    assert_eq!(report.pasted.len(), 1);
    assert!(
        core.renderer.vision_working(b) || core.renderer.vision_makes(b) > 0,
        "the paste asked for the other photo's mask"
    );
    wait_until("the worker", 60, || !core.renderer.vision_working(b));
    assert_eq!(provider.pictures().len(), 2, "one request for each photo");
    let src_b = PhotoSource::from_session(&core.session, b).expect("source");
    assert_eq!(
        src_b.settings.main().masks.items.len(),
        1,
        "the pasted photo has the row"
    );
    let plane_b = core
        .renderer
        .vision_plane(&src_b, VisionKind::Subject, &CancelToken::new())
        .expect("a plane")
        .expect("found");
    assert_ne!(plane_a.key, plane_b.key, "another picture, another mask");
    // The fake centres the subject on the brightness centroid (the ground's light pulls it to the middle): the disc at 30 % of
    // the width against the one at 70 %.
    let centroid = |p: &ym_color::mask_raster::SourcePlane| -> f64 {
        let (mut sum, mut sx) = (0.0f64, 0.0f64);
        for y in 0..p.height {
            for x in 0..p.width {
                let v = f64::from(p.data[y * p.width + x]);
                sum += v;
                sx += v * (x as f64 + 0.5) / p.width as f64;
            }
        }
        sx / sum
    };
    let (ca, cb) = (centroid(&plane_a), centroid(&plane_b));
    assert!(
        ca < 0.5 && cb > 0.5 && cb - ca > 0.03,
        "centroids {ca} and {cb}: the paste used the new photo's pixels"
    );
    assert_eq!(
        provider.pictures()[1].1.rgb,
        core.renderer
            .object_picture_of(&src_b, &CancelToken::new())
            .expect("picture")
            .2
    );
}

// ---- the device ------------------------------------------------------------------------------------------

fn gpu() -> Option<Arc<Gpu>> {
    match Gpu::start(GpuConfig {
        poison: true,
        ..GpuConfig::default()
    }) {
        Ok(g) => Some(Arc::new(g)),
        Err(e) => {
            assert!(
                !gpu_required_by_env(),
                "YM_REQUIRE_GPU is set but no GPU adapter could be opened: {e}"
            );
            let _ = writeln!(
                std::io::stderr(),
                "\n* GPU TESTS SKIPPED in mask_vision: {e}\n* (set WGPU_BACKEND=vulkan and install mesa-vulkan-drivers)\n"
            );
            None
        }
    }
}

/// A frame and a Mask only frame on the device and on the CPU with Subject and People rows: the plane is the provider's, so
/// what is compared is the raster pass, the colour pass and the encode on the device against the CPU reference, within the
/// bounds of N-13.
#[test]
fn the_device_makes_the_same_frames_and_masks_as_the_cpu_with_subject_and_people_rows() {
    let Some(gpu) = gpu() else { return };
    let d = disc_photo();
    let src = d.source();
    let mut o = options(Some(fake()), RenderOptions::default().full_tile_px);
    o.gpu = Some(gpu);
    o.gpu_stages = GpuStages::ALL
        .difference(GpuStages::RAW_CHAIN)
        .difference(GpuStages::RAW_FRONT);
    let on_gpu = Renderer::new(Arc::new(NeutralOnly), o).expect("render thread");
    let cpu = renderer(fake());
    for (what, rows) in [
        ("a subject", vec![subject()]),
        ("people", vec![people()]),
        (
            "a subject, people taken away",
            vec![subject(), row("people", json!({ "mode": "subtract" }))],
        ),
        (
            "all of it",
            vec![row(
                "subject",
                json!({ "grow": 0.3, "feather": 0.3, "refine": 0.7, "invert": true }),
            )],
        ),
    ] {
        let s = with_rows(&src, rows);
        for r in [&cpu, &on_gpu] {
            make_ready(r, &s);
        }
        for (name, st) in [("picture", picture_view()), ("mask only", mask_view(2))] {
            let device = frame(&on_gpu, &s, &st);
            assert_eq!(device.report.device, ComputeDevice::Gpu, "{what}, {name}");
            let reference = frame(&cpu, &s, &st);
            if name == "mask only" {
                let m = coverage(&reference);
                assert!(m > 0.005 && m < 0.995, "{what}: the mask is flat ({m})");
            }
            let (mut worst, mut sum) = (0u8, 0u64);
            for (x, y) in device.rgba.iter().zip(&reference.rgba) {
                let d = x.abs_diff(*y);
                worst = worst.max(d);
                sum += u64::from(d);
            }
            let mean = sum as f64 / device.rgba.len() as f64;
            eprintln!(
                "[vision] {what}, {name}: the device differs from the CPU by {worst} codes, mean {mean:.4}"
            );
            assert!(
                worst <= 3 && mean < 0.3,
                "{what}, {name}: {worst} codes, mean {mean}"
            );
        }
    }
}
