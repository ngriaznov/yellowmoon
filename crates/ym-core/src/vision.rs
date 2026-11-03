//! Subject and People masks from the computer's own Vision framework (`docs/23-nodes.md` sections 3.10 and 4.5,
//! `docs/algorithms/06-masks.md` section 5.3).
//!
//! Apple's Vision framework makes the masks, on this Mac and nowhere else (hard rule 7: nothing is sent anywhere). The
//! code that calls it is Swift behind a macOS gate in `ym-apple`; this module is what the core knows about it, and
//! it is the same on every platform:
//!
//! * [`VisionProvider`] is the seam. It answers [`probe`](VisionProvider::probe) (can this computer run the two
//!   requests, each tried on a 64 x 64 picture) and makes the **union of the instances** of a request on a picture
//!   ([`mask`](VisionProvider::mask)). `ym-apple` implements it with Vision; the tests implement it with
//!   [`FakeVision`], so Linux can play "a Mac that supports it", "a Mac that does not" and "not a Mac".
//! * [`VisionMasks`] is what a renderer holds: the provider and its probe answer, **asked once and kept** (as Apple
//!   Auto's availability is), so a Mac that cannot run the requests is asked once per run.
//! * [`VisionPicture`] is the picture handed to Vision: the Input's picture at 1024 px on the long edge, 8 bits,
//!   sRGB, in the oriented photo frame (`docs/algorithms/06` section 5.3), the same one the Object mask reads.
//!
//! Nothing here stores anything: the masks live in the photo's graph in memory (`pipeline::graph::vision`).

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};

use crate::apple_auto::AdvisorPicture;
use crate::settings::{MaskKind, MaskUnavailable};

/// The picture Vision reads: 8-bit sRGB, 3 bytes a pixel, rows top to bottom, no padding
/// (`AdvisorPicture` is that picture; Apple Auto's model and SAM read the same kind).
pub type VisionPicture = AdvisorPicture;

/// The two Vision requests the core uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum VisionKind {
    /// `VNGenerateForegroundInstanceMaskRequest`: the main thing in the photo.
    Subject,
    /// `VNGeneratePersonInstanceMaskRequest`: each person.
    People,
}

impl VisionKind {
    /// Both kinds, Subject first (the Add mask menu's order).
    pub const ALL: [VisionKind; 2] = [VisionKind::Subject, VisionKind::People];

    /// The mask kind of the row that asks for this request.
    #[must_use]
    pub fn mask_kind(self) -> MaskKind {
        match self {
            Self::Subject => MaskKind::Subject,
            Self::People => MaskKind::People,
        }
    }

    /// The request a mask kind stands for; `None` for the kinds Vision does not make.
    #[must_use]
    pub fn of(kind: MaskKind) -> Option<Self> {
        match kind {
            MaskKind::Subject => Some(Self::Subject),
            MaskKind::People => Some(Self::People),
            _ => None,
        }
    }

    /// The kind's name in the interface ("Subject", "People").
    #[must_use]
    pub fn label(self) -> &'static str {
        self.mask_kind().label()
    }

    /// A tag that goes into cache keys.
    fn tag(self) -> u64 {
        match self {
            Self::Subject => 1,
            Self::People => 2,
        }
    }
}

/// Whether this computer can run one Vision request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VisionStatus {
    /// The request ran on a 64 x 64 picture.
    Supported,
    /// It cannot, and why: [`MaskUnavailable::NotAMac`] off macOS, [`MaskUnavailable::MacosTooOld`] on a macOS older
    /// than 14, [`MaskUnavailable::NotSupported`] for a Mac whose request failed (a virtual Mac, a build without the
    /// Vision code).
    Unsupported(MaskUnavailable),
}

impl VisionStatus {
    /// Whether the request can be made.
    #[must_use]
    pub fn is_supported(self) -> bool {
        matches!(self, Self::Supported)
    }

    /// The reason it cannot be made, `None` when it can.
    #[must_use]
    pub fn reason(self) -> Option<MaskUnavailable> {
        match self {
            Self::Supported => None,
            Self::Unsupported(why) => Some(why),
        }
    }
}

/// What the probe found out: Subject and People are reported separately (a Mac may run one and not the other).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VisionSupport {
    /// `VNGenerateForegroundInstanceMaskRequest`.
    pub subject: VisionStatus,
    /// `VNGeneratePersonInstanceMaskRequest`.
    pub people: VisionStatus,
}

impl VisionSupport {
    /// Both requests run.
    pub const BOTH: Self = Self {
        subject: VisionStatus::Supported,
        people: VisionStatus::Supported,
    };

    /// Neither runs, for this reason.
    #[must_use]
    pub const fn neither(why: MaskUnavailable) -> Self {
        Self {
            subject: VisionStatus::Unsupported(why),
            people: VisionStatus::Unsupported(why),
        }
    }

    /// What a build with no Vision code says on this platform: `not_a_mac` off macOS, `not_supported` on a Mac (the SDK
    /// did not have the requests, or the code was switched off). Also what a renderer made without a provider says.
    #[must_use]
    pub const fn none_here() -> Self {
        Self::neither(if cfg!(target_os = "macos") {
            MaskUnavailable::NotSupported
        } else {
            MaskUnavailable::NotAMac
        })
    }

    /// The status of one request.
    #[must_use]
    pub fn of(self, kind: VisionKind) -> VisionStatus {
        match kind {
            VisionKind::Subject => self.subject,
            VisionKind::People => self.people,
        }
    }

    /// The requests that run.
    #[must_use]
    pub fn kinds(self) -> Vec<VisionKind> {
        VisionKind::ALL
            .into_iter()
            .filter(|&k| self.of(k).is_supported())
            .collect()
    }
}

/// The union mask of a request: values in [0, 1] over the picture, row by row, as the picture's size.
#[derive(Debug, Clone, PartialEq)]
pub struct VisionPlane {
    /// Width in pixels (the picture's).
    pub width: usize,
    /// Height in pixels (the picture's).
    pub height: usize,
    /// `width * height` values, 0 where Vision found no instance and 1 inside one.
    pub data: Vec<f32>,
}

/// What a request made of a picture.
#[derive(Debug, Clone, PartialEq)]
pub enum VisionMask {
    /// Vision found no instance: `S0` is 0 and the row says "Nothing found" (`docs/algorithms/06` section 5.3).
    NothingFound,
    /// The union of all the instances.
    Plane(VisionPlane),
}

/// Why a request made no mask.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum VisionError {
    /// This computer cannot run the request (the probe said so, or macOS stopped being able to).
    #[error("This computer cannot make {} masks.", .0.label())]
    Unavailable(VisionKind, MaskUnavailable),
    /// The picture handed over is not a picture (empty, or its bytes do not fit its size).
    #[error("The picture for the {} mask is not usable.", .0.label())]
    BadPicture(VisionKind),
    /// Vision failed on this picture; the text is Apple's.
    #[error("{0}")]
    Failed(String),
}

/// Whoever makes the masks: Vision through `ym-apple`, or a test's [`FakeVision`].
///
/// Both methods may be called from any thread. [`mask`](Self::mask) blocks the calling thread for one request (a
/// second or two on a large picture): the core calls it from a background worker, or from an export's own thread,
/// never from a thread a viewer waits on.
pub trait VisionProvider: Send + Sync + std::fmt::Debug {
    /// Whether this computer can run each request. The core asks once per run and keeps the answer
    /// ([`VisionMasks::support`]); a provider need not cache.
    fn probe(&self) -> VisionSupport;

    /// The union of the instances `kind` finds in `picture`, at the picture's size. The same picture on the same
    /// computer gives the same plane.
    ///
    /// # Errors
    ///
    /// [`VisionError`] when the request cannot run or fails.
    fn mask(&self, kind: VisionKind, picture: &VisionPicture) -> Result<VisionMask, VisionError>;
}

/// The provider of a computer with no Vision code: nothing is supported, for the reason of the platform. What
/// `YELLOWMOON_TEST_VISION=none` selects, and what a renderer without a provider acts like.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoVision;

impl VisionProvider for NoVision {
    fn probe(&self) -> VisionSupport {
        VisionSupport::none_here()
    }

    fn mask(&self, kind: VisionKind, _picture: &VisionPicture) -> Result<VisionMask, VisionError> {
        Err(VisionError::Unavailable(
            kind,
            VisionSupport::none_here()
                .of(kind)
                .reason()
                .unwrap_or(MaskUnavailable::NotSupported),
        ))
    }
}

/// A provider and the answer of its probe, asked once. What the renderer's options hold
/// (`RenderOptions::vision`).
pub struct VisionMasks {
    provider: Arc<dyn VisionProvider>,
    support: OnceLock<VisionSupport>,
}

impl std::fmt::Debug for VisionMasks {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VisionMasks")
            .field("provider", &self.provider)
            .field("support", &self.support.get())
            .finish()
    }
}

impl VisionMasks {
    /// Wraps a provider. Nothing is probed until [`support`](Self::support) is asked.
    #[must_use]
    pub fn new(provider: Arc<dyn VisionProvider>) -> Arc<Self> {
        Arc::new(Self {
            provider,
            support: OnceLock::new(),
        })
    }

    /// Whether this computer can run each request: the probe, run the first time this is asked and never again.
    /// Concurrent first callers wait for the one probe.
    #[must_use]
    pub fn support(&self) -> VisionSupport {
        *self.support.get_or_init(|| self.provider.probe())
    }

    /// The probe's answer if it has run, `None` while it has not: for a thread that must not wait for it.
    #[must_use]
    pub fn support_if_known(&self) -> Option<VisionSupport> {
        self.support.get().copied()
    }

    /// The requests among `kinds` that the probe has not refused. While the probe has not run, all of them: this never
    /// waits for it, so a viewer can call it; whoever acts on the answer must be ready to meet a refusal later
    /// (the background worker asks the probe first and drops what it refuses).
    #[must_use]
    pub fn not_refused(&self, mut kinds: Vec<VisionKind>) -> Vec<VisionKind> {
        if let Some(known) = self.support_if_known() {
            kinds.retain(|&k| known.of(k).is_supported());
        }
        kinds
    }

    /// The requests this computer can run.
    #[must_use]
    pub fn kinds(&self) -> Vec<VisionKind> {
        self.support().kinds()
    }

    /// The mask `kind` makes of `picture`. A request the probe said no to is refused without asking the provider.
    ///
    /// # Errors
    ///
    /// [`VisionError::Unavailable`] for such a request, [`VisionError::BadPicture`] for a picture whose bytes do not
    /// fit its size, and the provider's own errors. A plane that does not fit the picture is a
    /// [`VisionError::Failed`].
    pub fn mask(
        &self,
        kind: VisionKind,
        picture: &VisionPicture,
    ) -> Result<VisionMask, VisionError> {
        if let Some(why) = self.support().of(kind).reason() {
            return Err(VisionError::Unavailable(kind, why));
        }
        let fits = picture.width > 0
            && picture.height > 0
            && picture
                .width
                .checked_mul(picture.height)
                .and_then(|n| n.checked_mul(3))
                == Some(picture.rgb.len());
        if !fits {
            return Err(VisionError::BadPicture(kind));
        }
        let made = self.provider.mask(kind, picture)?;
        if let VisionMask::Plane(p) = &made
            && (p.width != picture.width
                || p.height != picture.height
                || p.data.len() != p.width * p.height)
        {
            return Err(VisionError::Failed(format!(
                "The {} mask has the wrong size.",
                kind.label()
            )));
        }
        Ok(made)
    }
}

/// The codes of a plane's values: the f32 mask held to [0, 1] (NaN is 0) and rounded once to a u16 code, the plane
/// the raster pass reads (`docs/algorithms/06` section 5.1, step 1).
#[must_use]
pub fn plane_codes(plane: &VisionPlane) -> Vec<u16> {
    plane
        .data
        .iter()
        .map(|&v| {
            let v = if v.is_nan() { 0.0 } else { v.clamp(0.0, 1.0) };
            ym_color::mask_raster::quantise(f64::from(v))
        })
        .collect()
}

/// The key of the plane a request makes of a picture: equal keys are equal planes. `picture` is the hash of what the
/// picture was made from.
#[must_use]
pub fn plane_key(picture: u64, kind: VisionKind) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for v in [picture, 0x5649_5349_4f4e_0000 | kind.tag()] {
        for byte in v.to_le_bytes() {
            h ^= u64::from(byte);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

// ---------------------------------------------------------------------------------------------
// A provider for the tests
// ---------------------------------------------------------------------------------------------

/// How a [`FakeVision`] answers for one request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FakeAnswer {
    /// A soft mask made from the picture (below).
    Mask,
    /// No instance: [`VisionMask::NothingFound`].
    NothingFound,
    /// Vision fails: [`VisionError::Failed`] ("The fake Vision failed.").
    Fail,
}

/// A provider for tests, the examples and `YELLOWMOON_TEST_VISION=fake`: it makes up masks from the picture, so a
/// test sees a mask that depends on the pixels and is the same every time (no Vision, no Mac needed), and it counts
/// what it was asked.
///
/// *Subject* is a soft disc of radius 0.22 of the long edge centred on the picture's brightness centroid; *People* is a
/// soft ellipse a third as wide, centred on the lower-left third of the picture. Both edges are a smoothstep over
/// 4 % of the long edge, so the masks are soft like Vision's.
#[derive(Debug)]
pub struct FakeVision {
    support: VisionSupport,
    subject: FakeAnswer,
    people: FakeAnswer,
    probes: AtomicUsize,
    masks: AtomicUsize,
}

impl Default for FakeVision {
    fn default() -> Self {
        Self::new()
    }
}

impl FakeVision {
    /// A Mac that supports both requests and finds a mask for each.
    #[must_use]
    pub fn new() -> Self {
        Self {
            support: VisionSupport::BOTH,
            subject: FakeAnswer::Mask,
            people: FakeAnswer::Mask,
            probes: AtomicUsize::new(0),
            masks: AtomicUsize::new(0),
        }
    }

    /// A computer whose probe says this.
    #[must_use]
    pub fn with_support(mut self, support: VisionSupport) -> Self {
        self.support = support;
        self
    }

    /// How a request answers once it runs.
    #[must_use]
    pub fn answering(mut self, kind: VisionKind, answer: FakeAnswer) -> Self {
        match kind {
            VisionKind::Subject => self.subject = answer,
            VisionKind::People => self.people = answer,
        }
        self
    }

    /// How many times the probe ran.
    #[must_use]
    pub fn probes(&self) -> usize {
        self.probes.load(Ordering::SeqCst)
    }

    /// How many times a mask was asked for (the calls that reached the provider).
    #[must_use]
    pub fn masks_made(&self) -> usize {
        self.masks.load(Ordering::SeqCst)
    }
}

impl VisionProvider for FakeVision {
    fn probe(&self) -> VisionSupport {
        self.probes.fetch_add(1, Ordering::SeqCst);
        self.support
    }

    fn mask(&self, kind: VisionKind, picture: &VisionPicture) -> Result<VisionMask, VisionError> {
        self.masks.fetch_add(1, Ordering::SeqCst);
        let answer = match kind {
            VisionKind::Subject => self.subject,
            VisionKind::People => self.people,
        };
        match answer {
            FakeAnswer::NothingFound => return Ok(VisionMask::NothingFound),
            FakeAnswer::Fail => {
                return Err(VisionError::Failed("The fake Vision failed.".to_owned()));
            }
            FakeAnswer::Mask => {}
        }
        let (w, h) = (picture.width, picture.height);
        let long = w.max(h) as f64;
        // The brightness centroid, summed in a fixed order.
        let (mut sum, mut sx, mut sy) = (0.0f64, 0.0f64, 0.0f64);
        for y in 0..h {
            for x in 0..w {
                let i = 3 * (y * w + x);
                let luma = 0.2126 * f64::from(picture.rgb[i])
                    + 0.7152 * f64::from(picture.rgb[i + 1])
                    + 0.0722 * f64::from(picture.rgb[i + 2]);
                sum += luma;
                sx += luma * (x as f64 + 0.5);
                sy += luma * (y as f64 + 0.5);
            }
        }
        let (cx, cy, rx, ry) = match kind {
            VisionKind::Subject if sum > 0.0 => (sx / sum, sy / sum, 0.22 * long, 0.22 * long),
            VisionKind::Subject => (0.5 * w as f64, 0.5 * h as f64, 0.22 * long, 0.22 * long),
            VisionKind::People => (0.30 * w as f64, 0.62 * h as f64, 0.075 * long, 0.2 * long),
        };
        let edge = 0.04 * long;
        let mut data = Vec::with_capacity(w * h);
        for y in 0..h {
            for x in 0..w {
                let dx = (x as f64 + 0.5 - cx) / rx;
                let dy = (y as f64 + 0.5 - cy) / ry;
                // Distance outside the ellipse, in pixels of the smaller radius.
                let d = (dx.hypot(dy) - 1.0) * rx.min(ry);
                let t = (0.5 - d / edge).clamp(0.0, 1.0);
                data.push((t * t * (3.0 - 2.0 * t)) as f32);
            }
        }
        Ok(VisionMask::Plane(VisionPlane {
            width: w,
            height: h,
            data,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn picture(w: usize, h: usize, value: u8) -> VisionPicture {
        VisionPicture {
            width: w,
            height: h,
            rgb: vec![value; 3 * w * h],
        }
    }

    #[test]
    fn the_probe_runs_once_however_often_it_is_asked() {
        let fake = Arc::new(FakeVision::new());
        let masks = VisionMasks::new(fake.clone());
        assert_eq!(fake.probes(), 0, "nothing is probed on construction");
        for _ in 0..5 {
            assert_eq!(masks.support(), VisionSupport::BOTH);
        }
        assert_eq!(fake.probes(), 1);
    }

    #[test]
    fn a_request_the_probe_refused_never_reaches_the_provider() {
        let fake = Arc::new(FakeVision::new().with_support(VisionSupport {
            subject: VisionStatus::Supported,
            people: VisionStatus::Unsupported(MaskUnavailable::NotSupported),
        }));
        let masks = VisionMasks::new(fake.clone());
        assert_eq!(masks.kinds(), [VisionKind::Subject]);
        let err = masks
            .mask(VisionKind::People, &picture(8, 8, 90))
            .unwrap_err();
        assert_eq!(
            err,
            VisionError::Unavailable(VisionKind::People, MaskUnavailable::NotSupported)
        );
        assert_eq!(fake.masks_made(), 0);
        assert!(masks.mask(VisionKind::Subject, &picture(8, 8, 90)).is_ok());
        assert_eq!(fake.masks_made(), 1);
    }

    #[test]
    fn a_picture_whose_bytes_do_not_fit_is_refused_before_the_provider() {
        let fake = Arc::new(FakeVision::new());
        let masks = VisionMasks::new(fake.clone());
        let mut bad = picture(4, 4, 10);
        bad.rgb.pop();
        assert_eq!(
            masks.mask(VisionKind::Subject, &bad),
            Err(VisionError::BadPicture(VisionKind::Subject))
        );
        assert_eq!(
            masks.mask(VisionKind::Subject, &picture(0, 4, 10)),
            Err(VisionError::BadPicture(VisionKind::Subject))
        );
        assert_eq!(fake.masks_made(), 0);
    }

    #[test]
    fn the_fake_makes_a_soft_mask_that_follows_the_picture_and_repeats() {
        let fake = FakeVision::new();
        // A dark picture with a bright block at the right: the subject's centroid is at the block.
        let mut p = picture(64, 48, 10);
        for y in 10..30 {
            for x in 44..60 {
                let i = 3 * (y * 64 + x);
                p.rgb[i..i + 3].copy_from_slice(&[240, 240, 240]);
            }
        }
        let VisionMask::Plane(a) = fake.mask(VisionKind::Subject, &p).unwrap() else {
            panic!("a plane");
        };
        let VisionMask::Plane(b) = fake.mask(VisionKind::Subject, &p).unwrap() else {
            panic!("a plane");
        };
        assert_eq!(a, b, "the same picture gives the same plane");
        assert_eq!((a.width, a.height), (64, 48));
        assert!(a.data.iter().all(|v| (0.0..=1.0).contains(v)));
        let at = |x: usize, y: usize| a.data[y * 64 + x];
        assert!(at(52, 20) > 0.99, "inside the disc at the block");
        assert!(at(2, 2) < 0.01, "far from it");
        assert!(a.data.iter().any(|&v| v > 0.02 && v < 0.98), "soft edge");
    }

    #[test]
    fn the_codes_are_the_values_rounded_once_and_held_to_the_range() {
        let plane = VisionPlane {
            width: 5,
            height: 1,
            data: vec![0.0, 1.0, 0.5, f32::NAN, 2.0],
        };
        assert_eq!(plane_codes(&plane), [0, 65535, 32768, 0, 65535]);
    }

    #[test]
    fn plane_keys_differ_by_picture_and_by_kind() {
        let a = plane_key(1, VisionKind::Subject);
        assert_eq!(a, plane_key(1, VisionKind::Subject));
        assert_ne!(a, plane_key(2, VisionKind::Subject));
        assert_ne!(a, plane_key(1, VisionKind::People));
    }

    #[test]
    fn no_vision_says_why_for_the_platform() {
        let support = NoVision.probe();
        let why = if cfg!(target_os = "macos") {
            MaskUnavailable::NotSupported
        } else {
            MaskUnavailable::NotAMac
        };
        assert_eq!(support, VisionSupport::neither(why));
        assert!(support.kinds().is_empty());
        assert!(matches!(
            NoVision.mask(VisionKind::Subject, &picture(2, 2, 0)),
            Err(VisionError::Unavailable(VisionKind::Subject, w)) if w == why
        ));
    }
}
