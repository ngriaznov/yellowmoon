//! Shared helpers for the ym-core integration tests: a loaded environment and synthetic ARW scenes
//! with a known light, a known brightness and a known noise model.
#![allow(dead_code)]

pub mod dng_looks;
pub mod tiffmini;

use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use ym_core::auto::IlluminantModels;
use ym_core::session::{
    EventSink, Session, SessionEnv, SessionEvent, ThumbnailInput, ThumbnailRenderer,
};
use ym_profiles::registry::ProfileStore;
use ym_raw::testgen::{ArwBuilder, SplitMix64};

/// A writer that appends to a shared buffer (the log capture below).
struct LogWriter(Arc<std::sync::Mutex<Vec<u8>>>);

impl std::io::Write for LogWriter {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// Everything the process has logged at info level and above (the application's default level)
/// since the first call, as the application's log file would hold it. The first call installs the
/// subscriber, so a test calls it before the work it wants to look at. Global, because the export
/// worker logs on its own thread; tests that run at the same time share it, so look for what is
/// yours (a temporary folder's name is unique).
pub fn captured_log() -> String {
    static LOG: OnceLock<Arc<std::sync::Mutex<Vec<u8>>>> = OnceLock::new();
    let buffer = LOG.get_or_init(|| {
        let buffer = Arc::new(std::sync::Mutex::new(Vec::new()));
        let shared = Arc::clone(&buffer);
        let _ = tracing_subscriber::fmt()
            .with_ansi(false)
            .with_max_level(tracing::Level::INFO)
            .with_writer(move || LogWriter(Arc::clone(&shared)))
            .try_init();
        buffer
    });
    let bytes = buffer
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    String::from_utf8_lossy(&bytes).into_owned()
}

/// The workspace `data/` folder (debug builds find it through `CARGO_MANIFEST_DIR`).
pub fn data_root() -> PathBuf {
    ym_core::data_dir::find_data_dir().expect("the data folder")
}

/// The environment of the tests: real profiles and noise priors, **no** illuminant models (every
/// photo takes the as-shot fallback, whatever `data/models/illum` holds), no events, two decode
/// threads.
pub fn base_env() -> SessionEnv {
    static ENV: OnceLock<SessionEnv> = OnceLock::new();
    ENV.get_or_init(|| {
        let mut env = SessionEnv::load(&data_root()).expect("load the data folder");
        env.illuminants = Arc::new(IlluminantModels::none());
        env.decode_threads = 2;
        env
    })
    .clone()
}

/// The profile store of the tests.
pub fn profiles() -> Arc<ProfileStore> {
    base_env().profiles
}

/// A session over `base_env()` with a channel for its events.
pub fn session_with_events() -> (Session, std::sync::mpsc::Receiver<SessionEvent>) {
    let (sink, rx) = EventSink::channel();
    (Session::new(base_env().with_events(sink)), rx)
}

/// Waits for the session to finish its jobs (fails the test after a minute).
pub fn settle(session: &Session) {
    assert!(
        session.wait_idle(Duration::from_secs(60)),
        "the decode pool did not go idle"
    );
}

/// What a synthetic scene looks like.
#[derive(Clone, Copy, Debug)]
pub enum Layout {
    /// One flat neutral level.
    Uniform,
    /// A horizontal ramp from 10 percent of the level to the level.
    Ramp,
    /// 6 x 4 flat patches, mostly neutral at different levels, some coloured.
    Patchwork,
}

/// A synthetic raw scene: a known light (camera white), a known brightness and Poisson-Gaussian
/// noise of a known `(a, b)`. All numbers are normalised units (after R1, before white balance).
#[derive(Clone, Debug)]
pub struct Scene {
    pub width: u32,
    pub height: u32,
    /// Camera RGB of the light, green = 1, in the camera's raw scale.
    pub white: [f64; 3],
    pub layout: Layout,
    /// Scene luminance of the brightest neutral area (a neutral pixel of luminance `L` has the
    /// camera RGB `0.9 L w / max(w)`).
    pub level: f64,
    /// Shot-noise gain.
    pub a: f64,
    /// Read-noise variance.
    pub b: f64,
    pub seed: u64,
    pub black: f64,
    pub white_level: f64,
}

/// The colours of the coloured patches of `Layout::Patchwork` (reflectance per channel).
const COLORED: [[f64; 3]; 4] = [
    [1.0, 0.55, 0.40],
    [0.35, 0.80, 0.35],
    [0.40, 0.50, 0.90],
    [0.90, 0.80, 0.30],
];

impl Scene {
    pub fn new(width: u32, height: u32, white: [f64; 3], layout: Layout, level: f64) -> Self {
        Self {
            width,
            height,
            white,
            layout,
            level,
            a: 0.0,
            b: 0.0,
            seed: 1,
            black: 512.0,
            white_level: 16300.0,
        }
    }

    pub fn noise(mut self, a: f64, b: f64) -> Self {
        self.a = a;
        self.b = b;
        self
    }

    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// The noiseless normalised signal of channel `c` (0 R, 1 G, 2 B) at a pixel.
    pub fn signal(&self, x: u32, y: u32, c: usize) -> f64 {
        let peak = self.white.iter().copied().fold(0.0, f64::max);
        let w = self.white[c] / peak;
        let (lum, rho) = match self.layout {
            Layout::Uniform => (self.level, [1.0; 3]),
            Layout::Ramp => {
                let t = f64::from(x) / f64::from(self.width.max(2) - 1);
                (self.level * (0.1 + 0.9 * t), [1.0; 3])
            }
            Layout::Patchwork => {
                let (px, py) = (x * 6 / self.width, y * 4 / self.height);
                let k = (py * 6 + px) as usize;
                let m = 0.12 + 0.88 * ((k * 7 % 24) as f64 / 23.0);
                let rho = if k % 5 == 3 {
                    COLORED[(k / 5) % COLORED.len()]
                } else {
                    [1.0; 3]
                };
                (self.level * m, rho)
            }
        };
        0.9 * lum * rho[c] * w
    }

    /// The mosaic (RGGB), `width * height` raw values.
    pub fn mosaic(&self) -> Vec<u16> {
        let mut rng = SplitMix64(self.seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
        let range = self.white_level - self.black;
        let mut out = Vec::with_capacity(self.width as usize * self.height as usize);
        for y in 0..self.height {
            for x in 0..self.width {
                let c = [0usize, 1, 1, 2][((y & 1) * 2 + (x & 1)) as usize];
                let s = self.signal(x, y, c);
                let sigma = (self.a * s.max(0.0) + self.b).sqrt();
                let v = (s + sigma * gaussian(&mut rng)).clamp(-0.03, 1.0);
                out.push((self.black + v * range).round().clamp(0.0, 65535.0) as u16);
            }
        }
        out
    }

    /// The SR2 as-shot white balance of a camera that got the light right: `[R, G1, G2, B]`
    /// multipliers with green 1024.
    pub fn as_shot_wb(&self) -> [u16; 4] {
        wb_levels(self.white)
    }

    /// A builder for this scene on `model`, with the as-shot white balance of the true light, ISO
    /// 400 and no exposure compensation. Add the payload and write it with [`Scene::write`].
    pub fn builder(&self, model: &str) -> ArwBuilder {
        ArwBuilder::new(model, self.width, self.height)
            .black([self.black as u16; 4])
            .white([self.white_level as u16; 3])
            .wb(Some(self.as_shot_wb()))
            .exposure_comp(0, 10)
            .iso(400)
    }

    /// Writes the scene as an uncompressed 14-bit ARW file and returns its path.
    pub fn write(&self, dir: &Path, name: &str, model: &str) -> PathBuf {
        let path = dir.join(name);
        self.builder(model)
            .write_to(&path, &self.mosaic())
            .expect("write the synthetic ARW");
        path
    }
}

/// The SR2 white-balance levels (multipliers, green 1024) that neutralise `white`.
pub fn wb_levels(white: [f64; 3]) -> [u16; 4] {
    let g = |c: usize| (1024.0 * white[1] / white[c]).round() as u16;
    [g(0), 1024, 1024, g(2)]
}

/// A standard normal value (Box-Muller).
pub fn gaussian(rng: &mut SplitMix64) -> f64 {
    let u1 = ((rng.next_u64() >> 11) as f64 + 1.0) / (1u64 << 53) as f64;
    let u2 = (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// The camera white (green = 1) of a blackbody or daylight at `cct_k` on a profile's locus.
pub fn light(profile_id: &str, cct_k: f64) -> [f64; 3] {
    let store = profiles();
    store
        .profile_by_name(profile_id)
        .expect("profile")
        .locus
        .rgb_at(cct_k)
}

/// Decodes a synthetic file and computes its automatic estimates with the as-shot fallback (no
/// illuminant model), with the profile the registry gives its camera.
pub fn estimates_of_file(
    path: &Path,
) -> (
    ym_raw::decode::RawImage,
    ym_core::auto::AutoEstimates,
    Arc<ym_profiles::profile::Profile>,
) {
    let env = base_env();
    let raw = ym_raw::decode::decode(path).expect("decode the synthetic file");
    let camera = ym_core::camera::PhotoCamera::new(&env.profiles, &raw.meta, None);
    let profile = env.profiles.resolve(&camera.matched).clone();
    let est = ym_core::auto::AutoEstimates::compute(
        &raw,
        &ym_core::auto::AutoInputs {
            profile: &profile,
            illuminant_model: None,
            priors: &env.priors,
        },
    )
    .expect("estimates");
    (raw, est, profile)
}

// ---------------------------------------------------------------------------------------------
// Rendering helpers
// ---------------------------------------------------------------------------------------------

/// A developed session with one synthetic photo, ready to render.
pub struct Developed {
    pub session: Session,
    pub id: ym_core::session::PhotoId,
    pub dir: tempfile::TempDir,
}

impl Developed {
    /// Writes `scene` as an ARW of `model`, opens it, waits for the estimates.
    pub fn new(scene: &Scene, model: &str) -> Self {
        let dir = tempfile::tempdir().expect("temp dir");
        scene.write(dir.path(), "DSC00001.ARW", model);
        Self::open(dir)
    }

    /// Opens the single ARW in `dir`.
    pub fn open(dir: tempfile::TempDir) -> Self {
        let session = Session::new(base_env());
        let out = session.open(
            &[dir.path().to_path_buf()],
            ym_core::session::OpenOptions::default(),
        );
        assert_eq!(out.added.len(), 1, "one photo added");
        settle(&session);
        let id = out.added[0];
        Self { session, id, dir }
    }

    /// The photo as the renderer takes it, with the session's current settings.
    pub fn source(&self) -> ym_core::pipeline::PhotoSource {
        let view = self.session.view(self.id).expect("view");
        let raw = self.session.raw_image(self.id).expect("raw image");
        ym_core::pipeline::PhotoSource::from_view(&view, raw)
    }
}

/// A night scene of defocused lights (discs of 1.5 % of the width on a dark ground, 4 x 6 of them): the blur the
/// bokeh detector finds, so L1 (`docs/21` section 4) reshapes the photo when a slider asks for it.
pub fn bokeh_developed(w: u32, h: u32, model: &str, profile: &str) -> Developed {
    let scene = Scene::new(w, h, light(profile, 5500.0), Layout::Uniform, 0.02).noise(2e-4, 1e-6);
    let mut mosaic = scene.mosaic();
    let code = (scene.black + 0.45 * (scene.white_level - scene.black)) as u16;
    let radius = 0.015 * f64::from(w);
    for fy in [0.18, 0.4, 0.62, 0.84] {
        for fx in [0.1, 0.26, 0.42, 0.58, 0.74, 0.9] {
            let (cx, cy) = (fx * f64::from(w), fy * f64::from(h));
            for y in 0..h {
                for x in 0..w {
                    let d = (f64::from(x) + 0.5 - cx).hypot(f64::from(y) + 0.5 - cy);
                    if d <= radius {
                        mosaic[(y * w + x) as usize] = code;
                    }
                }
            }
        }
    }
    let dir = tempfile::tempdir().expect("temp dir");
    scene
        .builder(model)
        .write_to(&dir.path().join("DSC00001.ARW"), &mosaic)
        .expect("write the synthetic ARW");
    Developed::open(dir)
}

/// The mean of channel `c` (0 R, 1 G, 2 B) of an RGBA frame over the pixels of a rectangle.
pub fn mean_rgba(
    frame: &ym_core::pipeline::RenderedFrame,
    x0: usize,
    y0: usize,
    w: usize,
    h: usize,
) -> [f64; 3] {
    let mut sum = [0.0f64; 3];
    for y in y0..y0 + h {
        for x in x0..x0 + w {
            let p = &frame.rgba[(y * frame.width + x) * 4..][..3];
            for c in 0..3 {
                sum[c] += f64::from(p[c]);
            }
        }
    }
    let n = (w * h) as f64;
    [sum[0] / n, sum[1] / n, sum[2] / n]
}

// ---------------------------------------------------------------------------------------------
// Export helpers
// ---------------------------------------------------------------------------------------------

/// An export context over a renderer that knows only the Neutral look, with `home` as the home
/// folder (so the fallback destination stays inside the test's temporary folder).
pub fn export_context(home: &Path) -> ym_core::export::ExportContext {
    ym_core::export::ExportContext::new(
        ym_core::pipeline::Renderer::neutral_only().expect("render thread"),
    )
    .with_home(Some(home.to_path_buf()))
}

/// The files directly inside `dir`, sorted by name (names only).
pub fn names_in(dir: &Path) -> Vec<String> {
    let mut v: Vec<String> = std::fs::read_dir(dir)
        .expect("read dir")
        .map(|e| e.expect("entry").file_name().to_string_lossy().into_owned())
        .collect();
    v.sort();
    v
}

/// A thumbnail hook that parks the `hold`-th thumbnail it is asked for (1-based, in call order)
/// until [`GateThumbnails::release`]. With one decode thread the thread is stuck there, so exactly
/// the photos before that one are estimated and the rest are still waiting.
pub struct GateThumbnails {
    hold: usize,
    calls: std::sync::atomic::AtomicUsize,
    open: std::sync::Mutex<bool>,
    changed: std::sync::Condvar,
}

impl GateThumbnails {
    pub fn new(hold: usize) -> Arc<Self> {
        Arc::new(Self {
            hold,
            calls: std::sync::atomic::AtomicUsize::new(0),
            open: std::sync::Mutex::new(false),
            changed: std::sync::Condvar::new(),
        })
    }

    /// Thumbnails asked for so far (the parked one included).
    pub fn calls(&self) -> usize {
        self.calls.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Lets the parked thumbnail go.
    pub fn release(&self) {
        *self.open.lock().unwrap() = true;
        self.changed.notify_all();
    }

    /// Waits until the `hold`-th thumbnail is parked (fails the test after a minute).
    pub fn wait_parked(&self) {
        let deadline = std::time::Instant::now() + Duration::from_secs(60);
        while self.calls() < self.hold {
            assert!(
                std::time::Instant::now() < deadline,
                "no thumbnail was parked"
            );
            std::thread::sleep(Duration::from_millis(5));
        }
    }
}

impl ThumbnailRenderer for GateThumbnails {
    fn render(&self, input: &ThumbnailInput<'_>) -> Result<Vec<u8>, String> {
        let n = self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;
        if n == self.hold {
            let mut open = self.open.lock().unwrap();
            while !*open && !(input.cancelled)() {
                open = self
                    .changed
                    .wait_timeout(open, Duration::from_millis(20))
                    .unwrap()
                    .0;
            }
        }
        Ok(vec![0])
    }
}
