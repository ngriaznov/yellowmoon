//! R9 lens geometry: distortion (0x7037) and lateral chromatic aberration (0x7035) displacement maps (`docs/05-raw-processing.md` section R9).
//!
//! Sony stores up to 16 signed 16-bit knots per tag. Knot `i` of `n` sits at the normalised radius
//! `(i + 0.5) / (n - 1)`, where `r = 1` is the half-diagonal of the active area, and the factors
//! are read by linear interpolation (held at the end values), exactly as darktable 4.6.1 does
//! (`tools/lens/` measures that against `darktable-cli`, see [`LinearKnots`]):
//!
//! * **Distortion** (0x7037): `F_i = d_i * 2^-14 + 1`. The output pixel at radius `r` is taken from
//!   the source at radius `r_s = r * F(r)`.
//! * **Lateral CA** (0x7035): 2n values, red first then blue. `K_c,i = ca_i * 2^-21 + 1`. Red and
//!   blue sample at `r_s * K_c(r)`, green at `r_s`. `K_c` is a function of the output radius `r`.
//!
//! Coordinates: continuous, pixel `i` covers `[i, i + 1)` and its centre is `i + 0.5`; the centre of
//! the frame is `(width / 2, height / 2)`; `r` is the distance from it over half the diagonal of
//! the *source* frame. A source position `s` is *valid* when it lies inside the hull of the pixel
//! centres, `0.5 <= s <= size - 0.5` (taps beyond the edge replicate the edge pixel).
//!
//! The output of the stage is cropped to the largest centred rectangle of valid pixels with the
//! source's aspect ratio ([`LensGeometry::auto_crop`]); output coordinates in this module are those
//! of the cropped image, origin at the crop's top-left. (darktable instead zooms the corrected
//! image up by the largest factor in its table so that it fills the frame; `tools/lens/` fits that
//! zoom out before comparing.)
//!
//! The factors come from public reverse-engineering, not a Sony document (`docs/05`).

use super::spline::LinearKnots;
use super::{CpuError, Placement, Rect};

/// The raw knots of the three Sony lens tags. An empty vector means the tag is absent.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LensTags {
    /// Tag 0x7032, vignetting knots.
    pub vignetting: Vec<i16>,
    /// Tag 0x7035, chromatic aberration: `2n` values, `n` for red then `n` for blue.
    pub ca: Vec<i16>,
    /// Tag 0x7037, distortion knots.
    pub distortion: Vec<i16>,
}

impl LensTags {
    /// Builds the tags from the three tables exactly as the file stores them: the count first, then
    /// the knots (`ym_raw::LensTable::raw`). Tags 0x7032 and 0x7037 start with `n`, the number of
    /// knots; tag 0x7035 starts with `2n` (red knots, then blue). Values beyond the stated count are
    /// ignored. A missing tag (`None`) gives an empty vector; a table that is empty, shorter than its
    /// count, has an odd CA count or fewer than two knots per curve is an error (darktable, which
    /// reads the same tags, silently drops all three corrections in that case).
    pub fn from_stored(
        vignetting: Option<&[i16]>,
        ca: Option<&[i16]>,
        distortion: Option<&[i16]>,
    ) -> Result<LensTags, CpuError> {
        Ok(LensTags {
            vignetting: stored_knots("0x7032 vignetting", vignetting, 1)?,
            ca: stored_knots("0x7035 chromatic aberration", ca, 2)?,
            distortion: stored_knots("0x7037 distortion", distortion, 1)?,
        })
    }
}

/// The knots of one stored table whose count is `curves` times the number of knots per curve.
fn stored_knots(
    tag: &'static str,
    stored: Option<&[i16]>,
    curves: usize,
) -> Result<Vec<i16>, CpuError> {
    let Some(stored) = stored else {
        return Ok(Vec::new());
    };
    let count = usize::try_from(stored.first().copied().unwrap_or(-1))
        .map_err(|_| CpuError::Mismatch(format!("lens tag {tag} has no valid count")))?;
    if count % curves != 0 || stored.len() < 1 + count {
        return Err(CpuError::Mismatch(format!(
            "lens tag {tag}: count {count} does not fit the {} stored values",
            stored.len()
        )));
    }
    if count / curves < 2 {
        return Err(CpuError::TooFewKnots {
            tag,
            got: count / curves,
            min: 2,
        });
    }
    Ok(stored[1..=count].to_vec())
}

/// The two RAW-tab switches that affect the lens stages (`decode.lens_corrections`,
/// `decode.chromatic_aberration`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LensSwitches {
    /// Distortion and vignetting.
    pub lens_corrections: bool,
    /// Lateral chromatic aberration only.
    pub chromatic_aberration: bool,
}

impl Default for LensSwitches {
    fn default() -> Self {
        LensSwitches {
            lens_corrections: true,
            chromatic_aberration: true,
        }
    }
}

/// The distortion factor `F_i` of one raw knot.
pub fn distortion_factor(d: i16) -> f64 {
    f64::from(d) * 2f64.powi(-14) + 1.0
}

/// The CA factor `K_i` of one raw knot.
pub fn ca_factor(c: i16) -> f64 {
    f64::from(c) * 2f64.powi(-21) + 1.0
}

/// The geometry of R9 for one frame: distortion, CA and the crop.
#[derive(Clone, Debug)]
pub struct LensGeometry {
    frame_w: usize,
    frame_h: usize,
    distortion: Option<LinearKnots>,
    ca_red: Option<LinearKnots>,
    ca_blue: Option<LinearKnots>,
    crop: Rect,
}

impl LensGeometry {
    /// Builds the maps for a `frame_w x frame_h` source from the tags and switches. The crop is the
    /// whole frame; call [`with_auto_crop`](Self::with_auto_crop) for the R9 crop.
    ///
    /// `lens_corrections = false` ignores the distortion tag; `chromatic_aberration = false`
    /// ignores the CA tag. A CA tag with an odd number of values is rejected.
    pub fn new(
        frame_w: usize,
        frame_h: usize,
        tags: &LensTags,
        switches: LensSwitches,
    ) -> Result<LensGeometry, CpuError> {
        let distortion = if switches.lens_corrections && !tags.distortion.is_empty() {
            if tags.distortion.len() < 2 {
                return Err(CpuError::TooFewKnots {
                    tag: "0x7037 distortion",
                    got: tags.distortion.len(),
                    min: 2,
                });
            }
            let ys: Vec<f64> = tags
                .distortion
                .iter()
                .map(|&d| distortion_factor(d))
                .collect();
            Some(LinearKnots::sony(&ys)?)
        } else {
            None
        };
        let (mut ca_red, mut ca_blue) = (None, None);
        if switches.chromatic_aberration && !tags.ca.is_empty() {
            if !tags.ca.len().is_multiple_of(2) || tags.ca.len() < 4 {
                return Err(CpuError::TooFewKnots {
                    tag: "0x7035 chromatic aberration (2n values, n >= 2)",
                    got: tags.ca.len(),
                    min: 4,
                });
            }
            let n = tags.ca.len() / 2;
            let red: Vec<f64> = tags.ca[..n].iter().map(|&c| ca_factor(c)).collect();
            let blue: Vec<f64> = tags.ca[n..].iter().map(|&c| ca_factor(c)).collect();
            ca_red = Some(LinearKnots::sony(&red)?);
            ca_blue = Some(LinearKnots::sony(&blue)?);
        }
        Ok(LensGeometry {
            frame_w,
            frame_h,
            distortion,
            ca_red,
            ca_blue,
            crop: Rect::whole(frame_w, frame_h),
        })
    }

    /// A geometry with no correction at all (identity map, whole-frame crop).
    pub fn identity(frame_w: usize, frame_h: usize) -> LensGeometry {
        LensGeometry {
            frame_w,
            frame_h,
            distortion: None,
            ca_red: None,
            ca_blue: None,
            crop: Rect::whole(frame_w, frame_h),
        }
    }

    /// Whether the geometry moves any pixel.
    pub fn is_identity(&self) -> bool {
        self.distortion.is_none() && self.ca_red.is_none()
    }

    /// Source frame width in pixels.
    pub fn frame_width(&self) -> usize {
        self.frame_w
    }

    /// Source frame height in pixels.
    pub fn frame_height(&self) -> usize {
        self.frame_h
    }

    /// The crop rectangle in uncropped output coordinates (the source frame's pixel grid).
    pub fn crop(&self) -> Rect {
        self.crop
    }

    /// Sets the crop (uncropped output coordinates).
    pub fn with_crop(mut self, crop: Rect) -> LensGeometry {
        self.crop = crop;
        self
    }

    /// Size of the cropped output.
    pub fn output_size(&self) -> (usize, usize) {
        (self.crop.w, self.crop.h)
    }

    /// Applies [`auto_crop`](Self::auto_crop).
    pub fn with_auto_crop(mut self) -> LensGeometry {
        self.crop = self.auto_crop();
        self
    }

    fn placement(&self) -> Placement {
        Placement::whole(self.frame_w, self.frame_h)
    }

    /// Source position (continuous) for colour channel `c` of the output pixel at *uncropped*
    /// output pixel centre `(px, py)` (continuous coordinates, e.g. `x + 0.5`).
    #[inline]
    pub fn source_position(&self, c: usize, px: f64, py: f64) -> (f64, f64) {
        let place = self.placement();
        let (cx, cy) = place.centre();
        let (dx, dy) = (px - cx, py - cy);
        let r = dx.hypot(dy) / place.half_diagonal();
        let mut scale = self.distortion.as_ref().map_or(1.0, |t| t.eval(r));
        match c {
            0 => scale *= self.ca_red.as_ref().map_or(1.0, |t| t.eval(r)),
            2 => scale *= self.ca_blue.as_ref().map_or(1.0, |t| t.eval(r)),
            _ => {}
        }
        (cx + dx * scale, cy + dy * scale)
    }

    /// Source positions of the three colour channels (red, green, blue) for the output pixel
    /// centre `(px, py)`, sharing one radius computation.
    #[inline]
    pub fn source_positions(&self, px: f64, py: f64) -> [(f64, f64); 3] {
        let place = self.placement();
        let (cx, cy) = place.centre();
        let (dx, dy) = (px - cx, py - cy);
        let r = dx.hypot(dy) / place.half_diagonal();
        let f = self.distortion.as_ref().map_or(1.0, |t| t.eval(r));
        let kr = self.ca_red.as_ref().map_or(1.0, |t| t.eval(r));
        let kb = self.ca_blue.as_ref().map_or(1.0, |t| t.eval(r));
        [
            (cx + dx * f * kr, cy + dy * f * kr),
            (cx + dx * f, cy + dy * f),
            (cx + dx * f * kb, cy + dy * f * kb),
        ]
    }

    /// Source position for the pixel `(x, y)` of the cropped output.
    #[inline]
    pub fn source_position_cropped(&self, c: usize, x: usize, y: usize) -> (f64, f64) {
        self.source_position(
            c,
            (self.crop.x + x) as f64 + 0.5,
            (self.crop.y + y) as f64 + 0.5,
        )
    }

    fn valid(&self, sx: f64, sy: f64) -> bool {
        sx >= 0.5 && sy >= 0.5 && sx <= self.frame_w as f64 - 0.5 && sy <= self.frame_h as f64 - 0.5
    }

    /// Whether every pixel on the border of the uncropped-output rectangle `rect` samples a valid
    /// source position in all three channels (the interior is covered because the maps are radial
    /// and monotone in the output radius).
    fn rect_is_valid(&self, rect: Rect) -> bool {
        let check = |x: usize, y: usize| {
            (0..3).all(|c| {
                let (sx, sy) = self.source_position(c, x as f64 + 0.5, y as f64 + 0.5);
                self.valid(sx, sy)
            })
        };
        for x in rect.x..rect.x1() {
            if !check(x, rect.y) || !check(x, rect.y1() - 1) {
                return false;
            }
        }
        for y in rect.y..rect.y1() {
            if !check(rect.x, y) || !check(rect.x1() - 1, y) {
                return false;
            }
        }
        true
    }

    /// The largest centred rectangle of valid output pixels with the aspect ratio of the frame
    /// (width and height rounded to even numbers, in uncropped output coordinates).
    pub fn auto_crop(&self) -> Rect {
        let whole = Rect::whole(self.frame_w, self.frame_h);
        if self.is_identity() || self.rect_is_valid(whole) {
            return whole;
        }
        let (w, h) = (self.frame_w, self.frame_h);
        let make = |k: usize| {
            // k is the crop width in units of two pixels.
            let cw = (2 * k).min(w);
            let ch = ((cw as f64 * h as f64 / w as f64 / 2.0).round() as usize * 2).clamp(2, h);
            let x = (w - cw) / 2;
            let y = (h - ch) / 2;
            Rect::new(x, y, cw, ch)
        };
        let (mut lo, mut hi) = (1usize, w / 2); // lo is always valid (tiny rectangle at the centre)
        while lo < hi {
            let mid = (lo + hi).div_ceil(2);
            if self.rect_is_valid(make(mid)) {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        make(lo)
    }

    /// The source rectangle (source frame coordinates, clamped to the frame) that must be
    /// available to resample the cropped-output rectangle `out`: the bounding box of every source
    /// position of every channel, widened by the Catmull-Rom support (two pixels each side) and one
    /// pixel of safety.
    pub fn source_rect(&self, out: Rect) -> Rect {
        if out.is_empty() {
            return Rect::new(0, 0, 0, 0);
        }
        let (mut x_lo, mut x_hi, mut y_lo, mut y_hi) = (f64::MAX, f64::MIN, f64::MAX, f64::MIN);
        let mut take = |x: usize, y: usize| {
            for c in 0..3 {
                let (sx, sy) = self.source_position_cropped(c, x, y);
                x_lo = x_lo.min(sx);
                x_hi = x_hi.max(sx);
                y_lo = y_lo.min(sy);
                y_hi = y_hi.max(sy);
            }
        };
        for x in out.x..out.x1() {
            take(x, out.y);
            take(x, out.y1() - 1);
        }
        for y in out.y..out.y1() {
            take(out.x, y);
            take(out.x1() - 1, y);
        }
        let mut y = out.y;
        while y < out.y1() {
            let mut x = out.x;
            while x < out.x1() {
                take(x, y);
                x += 8;
            }
            y += 8;
        }
        // Taps: floor(s - 0.5) - 1 ..= floor(s - 0.5) + 2.
        let x0 = (x_lo - 0.5).floor() - 1.0 - 1.0;
        let y0 = (y_lo - 0.5).floor() - 1.0 - 1.0;
        let x1 = (x_hi - 0.5).floor() + 2.0 + 1.0;
        let y1 = (y_hi - 0.5).floor() + 2.0 + 1.0;
        let clamp = |v: f64, n: usize| v.clamp(0.0, n as f64 - 1.0) as usize;
        let (xa, xb) = (clamp(x0, self.frame_w), clamp(x1, self.frame_w));
        let (ya, yb) = (clamp(y0, self.frame_h), clamp(y1, self.frame_h));
        Rect::new(xa, ya, xb - xa + 1, yb - ya + 1)
    }

    /// The largest displacement (pixels) of any channel over the whole cropped output: the number
    /// the scheduler quotes as "max displacement" (`docs/05` tiling table).
    pub fn max_displacement_px(&self) -> f64 {
        let mut worst = 0.0f64;
        let step = 16usize;
        let mut y = 0;
        while y < self.crop.h {
            let mut x = 0;
            while x < self.crop.w {
                for c in 0..3 {
                    let (sx, sy) = self.source_position_cropped(c, x, y);
                    let (ox, oy) = (
                        (self.crop.x + x) as f64 + 0.5,
                        (self.crop.y + y) as f64 + 0.5,
                    );
                    worst = worst.max((sx - ox).hypot(sy - oy));
                }
                x += step;
            }
            y += step;
        }
        worst
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn barrel_tags() -> LensTags {
        // Source radius 1% larger than the output radius at the corner: the correction of barrel
        // distortion (the lens pulled the corners in).
        LensTags {
            distortion: vec![0, 20, 60, 120, 200, 300, 420, 560, 740],
            ca: vec![],
            vignetting: vec![],
        }
    }

    #[test]
    fn stored_tables_are_trimmed_to_their_count() {
        // 0x7032 and 0x7037 lead with n, 0x7035 with 2n; padding after the knots is ignored.
        let vig = [4i16, 0, 100, 300, 600, 0, 0];
        let ca = [8i16, 0, 10, 30, 60, 0, -10, -30, -60, 99];
        let dist = [3i16, 0, 50, 120];
        let t = LensTags::from_stored(Some(&vig), Some(&ca), Some(&dist)).unwrap();
        assert_eq!(t.vignetting, vec![0, 100, 300, 600]);
        assert_eq!(t.ca, vec![0, 10, 30, 60, 0, -10, -30, -60]);
        assert_eq!(t.distortion, vec![0, 50, 120]);
        // Missing tags are empty, not an error.
        assert_eq!(
            LensTags::from_stored(None, None, None).unwrap(),
            LensTags::default()
        );
        // The CA count is 2n: a file that leads with n (the mistake `lensgen` once made) is refused.
        assert!(LensTags::from_stored(None, Some(&[3, 0, 1, 2, 3]), None).is_err());
        // Too short for its count, empty, negative count, one knot only.
        assert!(LensTags::from_stored(Some(&[16, 0, 1]), None, None).is_err());
        assert!(LensTags::from_stored(None, None, Some(&[])).is_err());
        assert!(LensTags::from_stored(None, None, Some(&[-3, 0, 1, 2])).is_err());
        assert!(LensTags::from_stored(Some(&[1, 5]), None, None).is_err());
        // The tables of the fixtures in tools/lens (16 knots; the CA tag leads with 32) make a geometry.
        let mut ca32 = vec![32i16];
        ca32.extend((0..32).map(|i| i as i16));
        let mut d16 = vec![16i16];
        d16.extend((0..16).map(|i| 10 * i as i16));
        let t = LensTags::from_stored(None, Some(&ca32), Some(&d16)).unwrap();
        assert_eq!((t.ca.len(), t.distortion.len()), (32, 16));
        LensGeometry::new(300, 200, &t, LensSwitches::default()).unwrap();
    }

    #[test]
    fn knot_factors_follow_the_documented_scaling() {
        assert_eq!(distortion_factor(0), 1.0);
        assert!((distortion_factor(16384) - 2.0).abs() < 1e-12);
        assert!((distortion_factor(-1638) - (1.0 - 1638.0 / 16384.0)).abs() < 1e-12);
        assert!((ca_factor(2097) - (1.0 + 2097.0 / 2097152.0)).abs() < 1e-15);
    }

    #[test]
    fn identity_maps_pixel_centres_to_themselves() {
        let geo = LensGeometry::identity(100, 60);
        for c in 0..3 {
            assert_eq!(geo.source_position(c, 12.5, 40.5), (12.5, 40.5));
        }
        assert_eq!(geo.auto_crop(), Rect::whole(100, 60));
    }

    #[test]
    fn distortion_scales_the_radius_by_f() {
        let geo = LensGeometry::new(400, 300, &barrel_tags(), LensSwitches::default()).unwrap();
        let place = Placement::whole(400, 300);
        let (cx, cy) = place.centre();
        // A point on the diagonal at r = 0.5.
        let hd = place.half_diagonal();
        let (px, py) = (cx + 0.5 * hd * 0.8, cy + 0.5 * hd * 0.6);
        let (sx, sy) = geo.source_position(1, px, py);
        let rs = (sx - cx).hypot(sy - cy) / hd;
        let f = geo.distortion.as_ref().unwrap().eval(0.5);
        assert!((rs - 0.5 * f).abs() < 1e-9);
        // Direction preserved.
        assert!(((sy - cy) / (sx - cx) - 0.75).abs() < 1e-9);
    }

    #[test]
    fn ca_moves_red_and_blue_relative_to_green() {
        let tags = LensTags {
            ca: vec![0, 100, 300, 600, 0, -100, -300, -600],
            ..Default::default()
        };
        let geo = LensGeometry::new(400, 300, &tags, LensSwitches::default()).unwrap();
        let (px, py) = (400.0, 300.0); // the exact corner: r = 1
        let g = geo.source_position(1, px, py);
        let r = geo.source_position(0, px, py);
        let b = geo.source_position(2, px, py);
        assert_eq!(g, (px, py));
        assert!(r.0 > g.0 && b.0 < g.0, "red outward, blue inward");
        let (cx, cy) = (200.0, 150.0);
        let k = (r.0 - cx) / (px - cx);
        // Four knots sit at 1/6, 1/2, 5/6 and 7/6: the corner (r = 1) is halfway between the last two.
        let want = 0.5 * (ca_factor(300) + ca_factor(600));
        assert!(
            (k - want).abs() < 1e-12,
            "K_red at the corner {k} vs {want}"
        );
        assert!((k - ((r.1 - cy) / (py - cy))).abs() < 1e-12);
        let kb = (b.0 - cx) / (px - cx);
        assert!((kb - 0.5 * (ca_factor(-300) + ca_factor(-600))).abs() < 1e-12);
    }

    #[test]
    fn switches_disable_their_part() {
        let tags = LensTags {
            distortion: vec![0, 500, 1000],
            ca: vec![0, 500, 0, -500],
            vignetting: vec![],
        };
        let off = LensGeometry::new(
            100,
            80,
            &tags,
            LensSwitches {
                lens_corrections: false,
                chromatic_aberration: false,
            },
        )
        .unwrap();
        assert!(off.is_identity());
        let ca_only = LensGeometry::new(
            100,
            80,
            &tags,
            LensSwitches {
                lens_corrections: false,
                chromatic_aberration: true,
            },
        )
        .unwrap();
        assert!(ca_only.distortion.is_none() && ca_only.ca_red.is_some());
        let dist_only = LensGeometry::new(
            100,
            80,
            &tags,
            LensSwitches {
                lens_corrections: true,
                chromatic_aberration: false,
            },
        )
        .unwrap();
        assert!(dist_only.distortion.is_some() && dist_only.ca_red.is_none());
        assert!(
            LensGeometry::new(
                100,
                80,
                &LensTags {
                    ca: vec![1, 2, 3],
                    ..Default::default()
                },
                LensSwitches::default()
            )
            .is_err()
        );
    }

    #[test]
    fn auto_crop_is_valid_maximal_and_keeps_the_aspect_ratio() {
        let geo = LensGeometry::new(600, 400, &barrel_tags(), LensSwitches::default()).unwrap();
        let crop = geo.auto_crop();
        assert!(crop.w < 600 && crop.h < 400);
        assert!(geo.rect_is_valid(crop));
        assert!((crop.w as f64 / crop.h as f64 - 1.5).abs() < 0.01);
        // Centred.
        assert!((crop.x as isize * 2 + crop.w as isize - 600).abs() <= 1);
        // One step (two pixels) larger must be invalid.
        let bigger = Rect::new(crop.x - 1, crop.y - 1, crop.w + 2, crop.h + 2);
        assert!(!geo.rect_is_valid(bigger), "crop {crop:?} is not maximal");
        // Pincushion correction (F < 1) needs no crop.
        let pin = LensTags {
            distortion: vec![0, -20, -60, -120, -200],
            ..Default::default()
        };
        let geo = LensGeometry::new(600, 400, &pin, LensSwitches::default()).unwrap();
        assert_eq!(geo.auto_crop(), Rect::whole(600, 400));
    }

    #[test]
    fn source_rect_contains_every_sample_footprint() {
        let tags = LensTags {
            distortion: vec![0, 20, 60, 120, 200, 300, 420, 560, 740],
            ca: vec![0, 40, 90, 160, 250, 0, -40, -90, -160, -250],
            vignetting: vec![],
        };
        let geo = LensGeometry::new(300, 200, &tags, LensSwitches::default())
            .unwrap()
            .with_auto_crop();
        let (cw, ch) = geo.output_size();
        let out = Rect::new(cw / 3, ch / 4, cw / 3, ch / 2);
        let need = geo.source_rect(out);
        for y in out.y..out.y1() {
            for x in out.x..out.x1() {
                for c in 0..3 {
                    let (sx, sy) = geo.source_position_cropped(c, x, y);
                    let (i0, j0) = (
                        (sx - 0.5).floor() as isize - 1,
                        (sy - 0.5).floor() as isize - 1,
                    );
                    assert!(
                        i0 >= need.x as isize && (i0 + 3) < need.x1() as isize,
                        "x footprint at ({x},{y}) c{c}"
                    );
                    assert!(
                        j0 >= need.y as isize && (j0 + 3) < need.y1() as isize,
                        "y footprint at ({x},{y}) c{c}"
                    );
                }
            }
        }
    }
}
