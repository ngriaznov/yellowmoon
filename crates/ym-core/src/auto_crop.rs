//! Auto crop in the core (`docs/20-auto-crop.md`): the camera's hints from the raw metadata, the
//! request for one aspect and orientation, and the answer the app and `ym` show
//! ([`CropSuggestion`]). The analysis itself is `ym_compose`'s; [`crate::session::Session::suggest_crop`]
//! renders the picture and keeps the analysis per photo.

use serde::{Deserialize, Serialize};
use ym_compose::{Analysis, CameraHint, FrameRequest, Ratio, Straighten};
use ym_raw::metadata::RawMetadata;

use crate::pipeline::geometry::{FIT_MARGIN_PX, fit_inside, quarter_turns_of, sensor_to_oriented};
use crate::settings::{CropAspect, NormRect, Orientation};

/// The straightener's answer as the app and `ym` name it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StraightenKind {
    /// Within 0.2° of level: the angle is 0.
    Level,
    /// Levelled by [`CropSuggestion::angle_deg`].
    Rotate,
    /// No horizon or verticals that agree, or a tilt beyond 10°: the angle is 0.
    Unsure,
}

/// Where the subject came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SubjectSource {
    /// The camera's focus point (Sony maker note).
    Camera,
    /// The picture alone.
    Picture,
}

/// One proposed frame.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CropFrame {
    /// The frame on the canvas, as `Crop.rect`.
    pub rect: NormRect,
    /// Its score (0 for the kept frame), rounded to 4 decimals.
    pub score: f64,
}

/// What Auto crop proposes for one photo, aspect and orientation (`docs/20` §3).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CropSuggestion {
    /// The straightener's answer.
    pub straighten: StraightenKind,
    /// The straighten angle the frames are made for: the proposal's when `straighten` is
    /// [`StraightenKind::Rotate`], else 0.
    pub angle_deg: f64,
    /// Best first, at most 3, never empty.
    pub frames: Vec<CropFrame>,
    /// The first frame is the largest frame of the aspect at that angle.
    pub keep: bool,
    /// Where the subject came from.
    pub subject: SubjectSource,
}

/// How far from the centre, as a fraction of the short side, a placed AF area must be to count as
/// placed on purpose (`docs/20` §4).
const PLACED_MIN_OFFSET: f64 = 0.10;

/// The camera's hints for a photo, in the space of its picture (the camera's orientation applied,
/// `docs/20` §4).
#[must_use]
pub fn camera_hint(meta: &RawMetadata) -> CameraHint {
    let Some(focus) = meta.focus else {
        return CameraHint::default();
    };
    let [fw, fh] = focus.frame.map(f64::from);
    let [x, y] = focus.point.map(f64::from);
    if fw <= 0.0 || fh <= 0.0 || x <= 0.0 || y <= 0.0 || x >= fw || y >= fh {
        return CameraHint::default();
    }
    // Sony writes the centre when it does not know the point.
    if (x - fw / 2.0).abs() <= 1.0 && (y - fh / 2.0).abs() <= 1.0 {
        return CameraHint::default();
    }
    let orientation = meta.orientation;
    let point = sensor_to_oriented(orientation, [x / fw, y / fh]);
    let swaps = (5..=8).contains(&orientation);
    let box_size = focus.box_size.map(|[w, h]| {
        let (w, h) = (f64::from(w) / fw, f64::from(h) / fh);
        if swaps { [h, w] } else { [w, h] }
    });
    // Offset from the centre in units of the short side (fractions scaled by the frame's sides).
    let short = fw.min(fh);
    let offset = (x - fw / 2.0).hypot(y - fh / 2.0) / short;
    let placed = placed_af_area(&meta.model, meta.af_area_setting) && offset >= PLACED_MIN_OFFSET;
    CameraHint {
        point: Some(point),
        box_size,
        placed,
    }
}

/// Whether `AFAreaModeSetting` says the photographer placed the AF area: Flexible Spot, Zone or
/// Expanded Flexible Spot (`docs/20` §4; the values differ between body families).
fn placed_af_area(model: &str, setting: Option<u8>) -> bool {
    let Some(v) = setting else {
        return false;
    };
    let ilce_like = ["NEX-", "ILCE-", "ILME-", "ZV-"]
        .iter()
        .any(|p| model.starts_with(p));
    if ilce_like {
        matches!(v, 3 | 4 | 11 | 12)
    } else if model.starts_with("ILCA-") {
        matches!(v, 4 | 8 | 12)
    } else {
        false
    }
}

/// The ratio of an aspect on a canvas of `canvas_w × canvas_h` (`docs/20` §7 step 2).
#[must_use]
pub fn ratio_for(aspect: CropAspect, canvas_w: f64, canvas_h: f64) -> Ratio {
    let landscape = canvas_w >= canvas_h;
    match aspect {
        CropAspect::Original | CropAspect::Free => Ratio::Canvas,
        CropAspect::Square => Ratio::Fixed(1.0),
        CropAspect::FourByFive => Ratio::Fixed(if landscape { 5.0 / 4.0 } else { 4.0 / 5.0 }),
        CropAspect::SixteenByNine => Ratio::Fixed(if landscape { 16.0 / 9.0 } else { 9.0 / 16.0 }),
    }
}

/// The suggestion for one aspect and orientation from a photo's analysis.
///
/// `photo_px` is the photo's size in pixels as the camera turned it (`PhotoSummary::width_px` and
/// `height_px`; 0 when unknown, then the picture's size stands in). The analysis made its frames on
/// the picture, whose sides are rounded to whole pixels; each frame is then fitted inside the photo
/// straightened at its full size ([`fit_inside`]), so that an Auto crop never leaves an empty corner
/// (the owner's rule of 2026-10-22, "No border in photos"; `docs/20` section 7 step 6).
#[must_use]
pub fn suggestion(
    analysis: &Analysis,
    aspect: CropAspect,
    orientation: Orientation,
    photo_px: (u32, u32),
) -> CropSuggestion {
    let quarter_turns = quarter_turns_of(orientation);
    let (w, h) = analysis.picture_size();
    let (w, h) = if photo_px.0 > 0 && photo_px.1 > 0 {
        (f64::from(photo_px.0), f64::from(photo_px.1))
    } else {
        (w as f64, h as f64)
    };
    let (cw, ch) = if quarter_turns % 2 == 1 {
        (h, w)
    } else {
        (w, h)
    };
    let angle_deg = analysis.angle_deg();
    let proposal = analysis.propose(&FrameRequest {
        quarter_turns,
        ratio: ratio_for(aspect, cw, ch),
        angle_deg,
    });
    let straighten = match analysis.straighten() {
        Straighten::Level => StraightenKind::Level,
        Straighten::Rotate { .. } => StraightenKind::Rotate,
        Straighten::Unsure => StraightenKind::Unsure,
    };
    CropSuggestion {
        straighten,
        angle_deg,
        frames: proposal
            .frames
            .iter()
            .map(|f| {
                let rect = NormRect {
                    x: f.rect.x,
                    y: f.rect.y,
                    width: f.rect.width,
                    height: f.rect.height,
                };
                CropFrame {
                    rect: rounded(fit_inside(rect, angle_deg, cw, ch, FIT_MARGIN_PX)),
                    score: f.score,
                }
            })
            .collect(),
        keep: proposal.keep,
        subject: if analysis.subject_from_camera() {
            SubjectSource::Camera
        } else {
            SubjectSource::Picture
        },
    }
}

/// A frame rounded to 6 decimals as `ym_compose` rounds its own (a millionth of the side, far
/// inside [`FIT_MARGIN_PX`]); a frame that is already rounded keeps its bits.
fn rounded(r: NormRect) -> NormRect {
    let round = |v: f64| (v * 1e6).round() / 1e6;
    NormRect {
        x: round(r.x),
        y: round(r.y),
        width: round(r.width),
        height: round(r.height),
    }
}

#[cfg(test)]
mod tests {
    use ym_raw::makernote::SonyFocus;
    use ym_raw::testgen::ArwBuilder;

    use super::*;

    fn meta(model: &str, orientation: u16) -> RawMetadata {
        let bytes = ArwBuilder::new(model, 16, 16)
            .orientation(orientation)
            .build(&[1000u16; 16 * 16]);
        ym_raw::metadata::probe_bytes(&bytes).expect("probe")
    }

    fn with_focus(mut m: RawMetadata, point: [u16; 2], box_size: Option<[u16; 2]>) -> RawMetadata {
        m.focus = Some(SonyFocus {
            frame: [6000, 4000],
            point,
            box_size,
        });
        m
    }

    #[test]
    fn no_tags_no_hint_and_the_centre_is_no_point() {
        assert_eq!(camera_hint(&meta("ILCE-7M3", 1)), CameraHint::default());
        let centre = with_focus(meta("ILCE-7M3", 1), [3000, 2000], Some([150, 150]));
        assert_eq!(camera_hint(&centre), CameraHint::default());
        let zero = with_focus(meta("ILCE-7M3", 1), [0, 0], None);
        assert_eq!(camera_hint(&zero), CameraHint::default());
    }

    #[test]
    fn the_point_turns_with_the_cameras_orientation() {
        // Upright: as stored.
        let h = camera_hint(&with_focus(
            meta("ILCE-7M3", 1),
            [1500, 1000],
            Some([600, 400]),
        ));
        assert_eq!(h.point, Some([0.25, 0.25]));
        assert_eq!(h.box_size, Some([0.1, 0.1]));
        // Orientation 8 (the camera turned left, shown turned 90° counter-clockwise): the sensor's
        // top-left quarter lands at the bottom left; the box's sides swap.
        let h = camera_hint(&with_focus(
            meta("ILCE-7M3", 8),
            [1500, 1000],
            Some([600, 200]),
        ));
        assert_eq!(h.point, Some(sensor_to_oriented(8, [0.25, 0.25])));
        assert_eq!(h.point, Some([0.25, 0.75]));
        assert_eq!(h.box_size, Some([0.05, 0.1]));
    }

    #[test]
    fn a_flexible_spot_away_from_the_centre_is_placed_on_purpose() {
        let mut m = with_focus(meta("ILCE-7M3", 1), [1000, 1000], None);
        m.af_area_setting = Some(3);
        assert!(camera_hint(&m).placed);
        // Wide is the camera's choice, not the photographer's.
        m.af_area_setting = Some(0);
        assert!(!camera_hint(&m).placed);
        // Flexible Spot near the centre says nothing about the framing.
        let mut near = with_focus(meta("ILCE-7M3", 1), [3100, 2050], None);
        near.af_area_setting = Some(3);
        assert!(!camera_hint(&near).placed);
        // ILCA numbers its modes differently.
        let mut a = with_focus(meta("ILCA-99M2", 1), [1000, 1000], None);
        a.af_area_setting = Some(4);
        assert!(camera_hint(&a).placed);
        a.af_area_setting = Some(3);
        assert!(!camera_hint(&a).placed);
    }

    #[test]
    fn ratios_follow_the_canvas() {
        assert_eq!(ratio_for(CropAspect::Original, 3.0, 2.0), Ratio::Canvas);
        assert_eq!(ratio_for(CropAspect::Free, 2.0, 3.0), Ratio::Canvas);
        assert_eq!(ratio_for(CropAspect::Square, 3.0, 2.0), Ratio::Fixed(1.0));
        assert_eq!(
            ratio_for(CropAspect::FourByFive, 3.0, 2.0),
            Ratio::Fixed(1.25)
        );
        assert_eq!(
            ratio_for(CropAspect::FourByFive, 2.0, 3.0),
            Ratio::Fixed(0.8)
        );
        assert_eq!(
            ratio_for(CropAspect::SixteenByNine, 2.0, 3.0),
            Ratio::Fixed(9.0 / 16.0)
        );
    }

    /// A 768 x 512 picture of a sky over the sea whose horizon is tilted by `roll_deg` (clockwise,
    /// y down), 4 x 4 samples per pixel, with a little fixed texture.
    fn tilted_sea(roll_deg: f64) -> Vec<u8> {
        let (w, h) = (768usize, 512usize);
        let (s, c) = roll_deg.to_radians().sin_cos();
        let texture = |u: f64, v: f64| 0.02 * ((u * 0.37).sin() * (v * 0.53).cos());
        let mut out = Vec::with_capacity(w * h * 3);
        for y in 0..h {
            for x in 0..w {
                let mut acc = [0.0; 3];
                for sy in 0..4 {
                    for sx in 0..4 {
                        let qx = x as f64 + (f64::from(sx) + 0.5) / 4.0 - w as f64 / 2.0;
                        let qy = y as f64 + (f64::from(sy) + 0.5) / 4.0 - h as f64 / 2.0;
                        // The scene's own coordinates: the picture turned back by the roll.
                        let (u, v) = (c * qx + s * qy, -s * qx + c * qy);
                        let t = texture(u, v);
                        let col = if v < 20.0 {
                            [0.62 + t, 0.76 + t, 0.93 + t]
                        } else {
                            [0.10 + t, 0.27 + t, 0.42 + t]
                        };
                        for k in 0..3 {
                            acc[k] += col[k] / 16.0;
                        }
                    }
                }
                out.extend(acc.map(|a| (a.clamp(0.0, 1.0) * 255.0).round() as u8));
            }
        }
        out
    }

    #[test]
    fn a_straightened_frame_leaves_no_empty_corner_at_the_photos_size() {
        use crate::pipeline::geometry::Geometry;
        let rgb = tilted_sea(3.0);
        let picture = ym_compose::Picture::new(768, 512, &rgb).expect("picture");
        let analysis = ym_compose::analyse(&picture, &CameraHint::default());
        assert!(
            matches!(analysis.straighten(), Straighten::Rotate { .. }),
            "{:?}",
            analysis.straighten()
        );
        // The photo is not quite the picture's shape (6048 x 4024 is 1.503 : 1): the frames are
        // fitted at the photo's own size.
        let (w, h) = (6048usize, 4024usize);
        for orientation in [Orientation::Deg0, Orientation::Deg90] {
            for aspect in [
                CropAspect::Original,
                CropAspect::Square,
                CropAspect::SixteenByNine,
            ] {
                let s = suggestion(&analysis, aspect, orientation, (w as u32, h as u32));
                assert_eq!(s.straighten, StraightenKind::Rotate);
                assert!(s.angle_deg != 0.0);
                for f in &s.frames {
                    let geometry = Geometry {
                        exif_orientation: 1,
                        quarter_turns: quarter_turns_of(orientation),
                        angle_deg: s.angle_deg,
                        crop: f.rect,
                        pixel_exact: false,
                    }
                    .at_full_resolution();
                    let out = geometry.output_size(w, h);
                    let grid = geometry.grid((w, h), out);
                    let (ow, oh) = out;
                    let filled = (0..ow)
                        .all(|x| grid.source(x, 0).is_some() && grid.source(x, oh - 1).is_some())
                        && (0..oh).all(|y| {
                            grid.source(0, y).is_some() && grid.source(ow - 1, y).is_some()
                        });
                    assert!(filled, "{aspect:?} {orientation:?} {s:?}");
                }
            }
        }
    }

    #[test]
    fn the_answer_reads_as_the_spec_writes_it() {
        let s = CropSuggestion {
            straighten: StraightenKind::Rotate,
            angle_deg: -1.4,
            frames: vec![CropFrame {
                rect: NormRect::FULL,
                score: 0.0,
            }],
            keep: true,
            subject: SubjectSource::Camera,
        };
        assert_eq!(
            serde_json::to_string(&s).unwrap(),
            r#"{"straighten":"rotate","angle_deg":-1.4,"frames":[{"rect":{"x":0.0,"y":0.0,"width":1.0,"height":1.0},"score":0.0}],"keep":true,"subject":"camera"}"#
        );
    }
}
