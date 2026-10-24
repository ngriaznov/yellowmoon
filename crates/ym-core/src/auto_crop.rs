//! Auto crop in the core (`docs/20-auto-crop.md`): the request for one aspect and orientation, and
//! the answer the app and `ym` show ([`CropSuggestion`]). The analysis itself is `ym_compose`'s, on
//! the crop model's crop (`ym_ml::crop_net`); [`crate::session::Session::suggest_crop`] renders the
//! picture, runs the model and keeps the analysis per photo.

use serde::{Deserialize, Serialize};
use ym_compose::{Analysis, FrameRequest, Ratio, Straighten};

use crate::pipeline::geometry::{FIT_MARGIN_PX, fit_inside, quarter_turns_of};
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

/// Where the subject came from. The crop model looks at the picture alone, so the answer is always
/// [`SubjectSource::Picture`]; `Camera` stays in the wire for the classical proposer's answers and
/// a later model that reads the camera's focus point (`docs/20` §11).
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
    /// Its intersection over union with the crop model's crop, rounded to 4 decimals.
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

/// The ratio of an aspect on a canvas of `canvas_w × canvas_h` (`docs/20` §7 step 3): Free is the
/// crop model's own shape.
#[must_use]
pub fn ratio_for(aspect: CropAspect, canvas_w: f64, canvas_h: f64) -> Ratio {
    let landscape = canvas_w >= canvas_h;
    match aspect {
        CropAspect::Original => Ratio::Canvas,
        CropAspect::Free => Ratio::Free,
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
        subject: SubjectSource::Picture,
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
    use super::*;

    #[test]
    fn ratios_follow_the_canvas() {
        assert_eq!(ratio_for(CropAspect::Original, 3.0, 2.0), Ratio::Canvas);
        assert_eq!(ratio_for(CropAspect::Free, 2.0, 3.0), Ratio::Free);
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
        let analysis = ym_compose::analyse(&picture, [0.08, 0.02, 0.97, 0.9]);
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
                CropAspect::Free,
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
                        warp: None,
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
            subject: SubjectSource::Picture,
        };
        assert_eq!(
            serde_json::to_string(&s).unwrap(),
            r#"{"straighten":"rotate","angle_deg":-1.4,"frames":[{"rect":{"x":0.0,"y":0.0,"width":1.0,"height":1.0},"score":0.0}],"keep":true,"subject":"picture"}"#
        );
    }
}
