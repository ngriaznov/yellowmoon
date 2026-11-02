//! Subject and People masks from Apple's Vision framework (`docs/23-nodes.md` sections 3.10 and 4.5,
//! `docs/algorithms/06-masks.md` section 5.3).
//!
//! [`probe`] says whether this computer can run each of the two requests (each is tried on a 64 × 64 picture, once per
//! process), and [`mask`] makes the union of the instances a request finds in a picture. [`system_vision`] gives the app
//! and `ym` the [`VisionProvider`] of this computer, which the core wraps and asks (`ym_core::vision`), or, when the
//! tests ask for one with `YELLOWMOON_TEST_VISION`, a fake one (`fake`) or one that supports nothing (`none`).
//!
//! The Swift (`swift/VisionMasks.swift`) is compiled for every Mac whose SDK has the macOS 14 Vision requests, both
//! architectures (`build.rs`); elsewhere this module is a stub that reports `not_a_mac` (not a Mac) or `not_supported`
//! (a Mac built without the bridge). Vision runs on this Mac and sends nothing anywhere (hard rule 7); the masks are not
//! stored (hard rule 4): the core keeps them in the photo's memory.

use std::sync::{Arc, OnceLock};

use ym_core::vision::{
    FakeVision, NoVision, VisionError, VisionKind, VisionMask, VisionPicture, VisionProvider,
    VisionSupport,
};

/// The environment variable that selects a test provider, like `YELLOWMOON_TEST_APPLE_AUTO` selects Apple Auto's.
pub const TEST_VISION_ENV: &str = "YELLOWMOON_TEST_VISION";

/// The value of [`TEST_VISION_ENV`] that selects [`FakeVision`]: a computer that supports both requests and makes up a
/// soft mask from the picture, on any platform. The tests that need Subject and People on Linux set it.
pub const TEST_VISION_FAKE: &str = "fake";

/// The value of [`TEST_VISION_ENV`] that selects [`NoVision`]: nothing is supported, for the reason a build without the
/// bridge gives on this platform. Tests that expect the sentences of `docs/23` section 3.10 set it, so that they give
/// the same answer on a Mac that supports Vision.
pub const TEST_VISION_NONE: &str = "none";

/// Whether this build carries the Vision bridge (a Mac whose SDK has the requests). Without it [`AppleVision`] supports
/// nothing.
pub const VISION_BUILT: bool = cfg!(ym_apple_vision);

/// The side of the picture the probe runs on, in pixels.
pub const PROBE_PX: usize = crate::vision_abi::PROBE_PX;

/// The provider the app and `ym` hand to the core: [`FakeVision`] when `YELLOWMOON_TEST_VISION=fake` is set,
/// [`NoVision`] when it is `none`, else [`AppleVision`].
#[must_use]
pub fn system_vision() -> Arc<dyn VisionProvider> {
    let test_vision = std::env::var(TEST_VISION_ENV).ok();
    vision_for(test_vision.as_deref())
}

/// The provider for a value of [`TEST_VISION_ENV`] (`None` when it is not set).
fn vision_for(test_vision: Option<&str>) -> Arc<dyn VisionProvider> {
    match test_vision {
        Some(TEST_VISION_FAKE) => Arc::new(FakeVision::new()),
        Some(TEST_VISION_NONE) => Arc::new(NoVision),
        _ => Arc::new(AppleVision),
    }
}

/// Whether this computer can run each request, asked of Vision the first time and kept for the process: the Swift runs
/// each request on a 64 × 64 picture (`#available(macOS 14.0, *)`), so a Mac that cannot (an older macOS, a virtual Mac)
/// says so once. Without the bridge: `not_a_mac` off macOS, `not_supported` on a Mac.
#[must_use]
pub fn probe() -> VisionSupport {
    static SUPPORT: OnceLock<VisionSupport> = OnceLock::new();
    *SUPPORT.get_or_init(platform::probe)
}

/// The union of the instances `kind` finds in `picture` (8-bit sRGB, the Input's picture at 1024 px on the long edge),
/// at the picture's size: [`VisionMask::NothingFound`] when Vision finds none. Blocks the calling thread for the
/// request. The same picture on the same Mac gives the same plane; another macOS version may answer differently.
///
/// # Errors
///
/// [`VisionError::Unavailable`] where the request cannot run, [`VisionError::BadPicture`] for a picture whose bytes do
/// not fit its size, [`VisionError::Failed`] with Apple's message when Vision fails.
pub fn mask(kind: VisionKind, picture: &VisionPicture) -> Result<VisionMask, VisionError> {
    platform::mask(kind, picture)
}

/// Apple's Vision framework, on this Mac.
#[derive(Debug, Clone, Copy, Default)]
pub struct AppleVision;

impl VisionProvider for AppleVision {
    fn probe(&self) -> VisionSupport {
        probe()
    }

    fn mask(&self, kind: VisionKind, picture: &VisionPicture) -> Result<VisionMask, VisionError> {
        mask(kind, picture)
    }
}

/// With the Swift bridge: Vision answers.
#[cfg(ym_apple_vision)]
mod platform {
    pub(crate) use crate::vision_bridge::{mask, probe};
}

/// Without the Swift bridge: nothing is supported, for the reason of this platform.
#[cfg(not(ym_apple_vision))]
mod platform {
    use ym_core::vision::{
        NoVision, VisionError, VisionKind, VisionMask, VisionPicture, VisionProvider, VisionSupport,
    };

    pub(crate) fn probe() -> VisionSupport {
        NoVision.probe()
    }

    pub(crate) fn mask(
        kind: VisionKind,
        picture: &VisionPicture,
    ) -> Result<VisionMask, VisionError> {
        NoVision.mask(kind, picture)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ym_core::settings::MaskUnavailable;
    use ym_core::vision::VisionStatus;

    fn picture(w: usize, h: usize) -> VisionPicture {
        VisionPicture {
            width: w,
            height: h,
            rgb: vec![90; 3 * w * h],
        }
    }

    #[test]
    fn the_test_value_selects_the_provider() {
        assert_eq!(vision_for(Some("fake")).probe(), VisionSupport::BOTH);
        assert_eq!(vision_for(Some("none")).probe(), VisionSupport::none_here());
        // Anything else is Vision itself, whatever this computer says.
        for value in [
            None,
            Some(""),
            Some("FAKE"),
            Some("off"),
            Some(" fake"),
            Some("NONE"),
        ] {
            assert_ne!(
                format!("{:?}", vision_for(value)),
                format!("{:?}", FakeVision::new()),
                "{value:?}"
            );
            assert_eq!(
                format!("{:?}", vision_for(value)),
                "AppleVision",
                "{value:?}"
            );
        }
        assert_eq!(format!("{:?}", vision_for(Some("none"))), "NoVision");
    }

    /// Without the Swift bridge, Subject and People are not supported, for the reason of this platform, and a request is
    /// refused with it. With it the probe is Vision's own answer: a reason it gives is never the build's.
    #[test]
    fn this_platform_says_why_or_answers_for_itself() {
        let support = probe();
        println!("Subject and People on this computer: {support:?}");
        assert_eq!(VISION_BUILT, cfg!(ym_apple_vision));
        if cfg!(ym_apple_vision) {
            // Supported, macOS too old, or not supported (a virtual Mac): never "not a Mac".
            for kind in VisionKind::ALL {
                assert_ne!(
                    support.of(kind),
                    VisionStatus::Unsupported(MaskUnavailable::NotAMac)
                );
            }
        } else {
            assert_eq!(support, VisionSupport::none_here());
            let why = support.subject.reason().expect("a reason");
            assert_eq!(
                mask(VisionKind::Subject, &picture(8, 8)),
                Err(VisionError::Unavailable(VisionKind::Subject, why))
            );
        }
        // The answer is the same the second time (it is asked once).
        assert_eq!(probe(), support);
    }

    #[test]
    fn a_picture_that_does_not_fit_never_reaches_vision() {
        let mut bad = picture(4, 3);
        bad.rgb.pop();
        let error = mask(VisionKind::People, &bad).unwrap_err();
        // With the bridge the picture is refused first; without it every request is unavailable.
        if cfg!(ym_apple_vision) {
            assert_eq!(error, VisionError::BadPicture(VisionKind::People));
        } else {
            assert!(matches!(error, VisionError::Unavailable(..)), "{error:?}");
        }
    }

    // ---- on a Mac whose SDK had the requests (CI's macOS jobs): Vision itself ---------------------------------

    /// A 1024 px picture a segmentation model may find something in: a soft, out-of-focus ground and a sharp, shaded,
    /// striped ball with a hard edge in the middle. Deterministic: no random numbers.
    #[cfg(ym_apple_vision)]
    fn test_picture() -> VisionPicture {
        let (w, h) = (1024usize, 683usize);
        let mut rgb = vec![0u8; 3 * w * h];
        let (cx, cy, r) = (0.5 * w as f64, 0.52 * h as f64, 0.27 * h as f64);
        for y in 0..h {
            for x in 0..w {
                let (fx, fy) = (x as f64 / w as f64, y as f64 / h as f64);
                // The ground: a slow green-to-brown gradient with a little low-frequency texture.
                let wave = 0.04 * ((fx * 9.0).sin() * (fy * 7.0).cos());
                let mut px = [
                    255.0 * (0.30 + 0.25 * fy + wave),
                    255.0 * (0.42 + 0.10 * fx - 0.12 * fy + wave),
                    255.0 * (0.22 + 0.05 * fx + wave),
                ];
                let (dx, dy) = (x as f64 + 0.5 - cx, y as f64 + 0.5 - cy);
                let d = dx.hypot(dy);
                if d < r {
                    // The ball: orange with a highlight, white stripes and a darker rim.
                    let shade = 0.55 + 0.45 * (1.0 - d / r).powf(0.6) - 0.15 * (dx + dy) / r;
                    let stripe = if ((dx * 0.06 + dy * 0.03).sin()) > 0.55 {
                        1.0
                    } else {
                        0.0
                    };
                    px = [
                        255.0 * (0.95 * shade * (1.0 - 0.7 * stripe) + 0.9 * stripe),
                        255.0 * (0.45 * shade * (1.0 - 0.7 * stripe) + 0.9 * stripe),
                        255.0 * (0.08 * shade * (1.0 - 0.7 * stripe) + 0.9 * stripe),
                    ];
                }
                let i = 3 * (y * w + x);
                for (out, value) in rgb[i..i + 3].iter_mut().zip(px) {
                    *out = value.clamp(0.0, 255.0).round() as u8;
                }
            }
        }
        VisionPicture {
            width: w,
            height: h,
            rgb,
        }
    }

    /// The probe on this Mac: each request ran on a 64 x 64 picture, or the reason it did not. A virtual Mac (a CI runner)
    /// may not run them: the answer is printed plainly and the test passes, unless `YM_REQUIRE_VISION` is set.
    #[cfg(ym_apple_vision)]
    #[test]
    fn the_probe_answers_for_each_request_on_this_mac() {
        let support = probe();
        println!(
            "Vision on this Mac: Subject {:?}, People {:?}",
            support.subject, support.people
        );
        // A second ask is the process's cached answer.
        assert_eq!(probe(), support);
        for kind in VisionKind::ALL {
            match support.of(kind) {
                VisionStatus::Supported => println!("{}: supported", kind.label()),
                VisionStatus::Unsupported(why) => {
                    println!(
                        "{}: NOT SUPPORTED on this Mac ({})",
                        kind.label(),
                        why.wire()
                    );
                }
            }
        }
        if std::env::var_os("YM_REQUIRE_VISION").is_some() {
            assert_eq!(support, VisionSupport::BOTH, "YM_REQUIRE_VISION is set");
        }
    }

    /// A real request on a test picture, twice: the same picture on the same Mac gives equal bytes (`docs/23` section 5.10; the
    /// Vision row of the gates). What Vision finds in a synthetic picture is its own business, so the test accepts a mask and
    /// "nothing found", prints which, and checks the plane when there is one.
    #[cfg(ym_apple_vision)]
    #[test]
    fn a_request_on_a_test_picture_gives_equal_bytes_twice() {
        let support = probe();
        let picture = test_picture();
        for kind in VisionKind::ALL {
            if !support.of(kind).is_supported() {
                println!("{} is not supported on this Mac: skipped", kind.label());
                continue;
            }
            let started = std::time::Instant::now();
            let first = mask(kind, &picture).expect("the request runs");
            let ms = started.elapsed().as_secs_f64() * 1000.0;
            let second = mask(kind, &picture).expect("the request runs again");
            assert_eq!(
                first,
                second,
                "{}: the same picture gave two masks",
                kind.label()
            );
            match &first {
                VisionMask::NothingFound => {
                    println!(
                        "{}: Vision found nothing in the test picture ({ms:.0} ms)",
                        kind.label()
                    );
                }
                VisionMask::Plane(plane) => {
                    assert_eq!((plane.width, plane.height), (picture.width, picture.height));
                    assert_eq!(plane.data.len(), plane.width * plane.height);
                    assert!(
                        plane.data.iter().all(|v| (0.0..=1.0).contains(v)),
                        "values in [0, 1]"
                    );
                    let covered = plane.data.iter().map(|&v| f64::from(v)).sum::<f64>()
                        / plane.data.len() as f64;
                    println!(
                        "{}: a mask covering {:.1} % of the test picture ({ms:.0} ms)",
                        kind.label(),
                        100.0 * covered
                    );
                    assert!(
                        covered > 0.0,
                        "a plane with nothing in it should be NothingFound"
                    );
                    assert!(
                        covered < 1.0,
                        "a mask of the whole picture is not a subject"
                    );
                }
            }
        }
    }

    /// A flat grey picture has no subject and no person: whatever Vision says, it says it twice, and a plane is a plane.
    #[cfg(ym_apple_vision)]
    #[test]
    fn a_flat_picture_gives_the_same_answer_twice() {
        let flat = picture(256, 192);
        for kind in VisionKind::ALL {
            if !probe().of(kind).is_supported() {
                continue;
            }
            let a = mask(kind, &flat).expect("the request runs");
            assert_eq!(a, mask(kind, &flat).expect("again"), "{}", kind.label());
            println!(
                "{} on a flat picture: {}",
                kind.label(),
                if a == VisionMask::NothingFound {
                    "nothing found"
                } else {
                    "a mask"
                }
            );
        }
    }
}
