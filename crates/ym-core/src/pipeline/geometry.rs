//! Geometry: the camera's EXIF orientation, the user's quarter turns, the straighten angle and the
//! crop, applied when the finished pixels are sampled for display or export (`docs/01` section
//! Render graph, row "Geometry"; `docs/11` section Crop).
//!
//! # The model (the frontend's, `src/fixtures/grade.ts: previewSvg`)
//!
//! 1. The pixel buffer `B` is in sensor orientation (after the lens stage's auto crop).
//! 2. The camera's EXIF orientation (1 to 8) turns it into the *oriented photo* `O1`, the space
//!    `white_pick` lives in.
//! 3. The user's `orientation` (0, 90, 180, 270 degrees clockwise) turns `O1` into the *canvas*
//!    `O2`. All later coordinates are on this canvas, whose size is the size of the photo after both
//!    turns.
//! 4. The straighten angle rotates the photo about the centre of the canvas, clockwise for positive
//!    angles, and the canvas keeps its size (corners of the canvas may be empty).
//! 5. The crop is a rectangle of the canvas, normalised to `[0, 1]²`. The output is that rectangle.
//!
//! Everything is a composition of affine maps and fraction permutations, so one [`Grid`] maps any
//! output pixel to a position in the buffer, at whatever resolution the buffer and the output
//! have. Positions are continuous: pixel `i` covers `[i, i + 1)` and has its centre at `i + 0.5`.
//!
//! The dihedral part is exact: with no straighten angle and an output grid that lines up with the
//! buffer, [`Grid::source`] returns pixel centres and sampling copies pixels.

use ym_gpu::cpu::sampler::SamplerGrid;

use crate::auto::display_to_sensor;
use crate::settings::{Crop, DevelopSettings, NormRect, Orientation};

/// Largest straighten angle that counts as "none" (radians would be overkill: this is degrees).
const ANGLE_EPS_DEG: f64 = 1e-9;

/// Maps a point of the photo as the camera oriented it to the same spot on the sensor (both as
/// fractions): [`display_to_sensor`] under the name this module uses.
#[must_use]
pub fn oriented_to_sensor(exif_orientation: u16, p: [f64; 2]) -> [f64; 2] {
    display_to_sensor(exif_orientation, p)
}

/// The inverse of [`oriented_to_sensor`]: a point of the sensor (as a fraction of the image area)
/// to the same spot on the photo as the camera oriented it.
#[must_use]
pub fn sensor_to_oriented(exif_orientation: u16, [x, y]: [f64; 2]) -> [f64; 2] {
    match exif_orientation {
        2 => [1.0 - x, y],
        3 => [1.0 - x, 1.0 - y],
        4 => [x, 1.0 - y],
        5 => [y, x],
        6 => [1.0 - y, x],
        7 => [1.0 - y, 1.0 - x],
        8 => [y, 1.0 - x],
        _ => [x, y],
    }
}

/// The user's turn: a point of the canvas `O2` to the same spot of the oriented photo `O1`
/// (fractions). `quarter_turns` is the number of clockwise quarter turns the user applied.
#[must_use]
pub fn canvas_to_oriented(quarter_turns: u8, [u, v]: [f64; 2]) -> [f64; 2] {
    match quarter_turns % 4 {
        1 => [v, 1.0 - u],
        2 => [1.0 - u, 1.0 - v],
        3 => [1.0 - v, u],
        _ => [u, v],
    }
}

/// The inverse of [`canvas_to_oriented`]: a point of the oriented photo to the canvas.
#[must_use]
pub fn oriented_to_canvas(quarter_turns: u8, [a, b]: [f64; 2]) -> [f64; 2] {
    match quarter_turns % 4 {
        1 => [1.0 - b, a],
        2 => [1.0 - a, 1.0 - b],
        3 => [b, 1.0 - a],
        _ => [a, b],
    }
}

/// A 2 x 3 affine map `(x, y) -> (a x + b y + c, d x + e y + f)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Affine {
    /// Row-major `[[a, b, c], [d, e, f]]`.
    pub m: [[f64; 3]; 2],
}

impl Affine {
    /// The identity map.
    pub const IDENTITY: Affine = Affine {
        m: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    };

    /// `(x, y) -> (sx x + tx, sy y + ty)`.
    #[must_use]
    pub const fn scale_translate(sx: f64, sy: f64, tx: f64, ty: f64) -> Affine {
        Affine {
            m: [[sx, 0.0, tx], [0.0, sy, ty]],
        }
    }

    /// Rotation by `angle_deg` clockwise (y down) about `(cx, cy)`.
    #[must_use]
    pub fn rotation_about(angle_deg: f64, cx: f64, cy: f64) -> Affine {
        let (s, c) = angle_deg.to_radians().sin_cos();
        // p' = c0 + R (p - c0), R = [[c, -s], [s, c]] (clockwise on a y-down canvas).
        Affine {
            m: [[c, -s, cx - c * cx + s * cy], [s, c, cy - s * cx - c * cy]],
        }
    }

    /// Applies the map.
    #[inline]
    #[must_use]
    pub fn apply(&self, x: f64, y: f64) -> (f64, f64) {
        (
            self.m[0][0] * x + self.m[0][1] * y + self.m[0][2],
            self.m[1][0] * x + self.m[1][1] * y + self.m[1][2],
        )
    }

    /// The map that applies `self` first and then `then`.
    #[must_use]
    pub fn then(&self, then: &Affine) -> Affine {
        let a = &then.m;
        let b = &self.m;
        let mut m = [[0.0; 3]; 2];
        for (r, row) in m.iter_mut().enumerate() {
            row[0] = a[r][0] * b[0][0] + a[r][1] * b[1][0];
            row[1] = a[r][0] * b[0][1] + a[r][1] * b[1][1];
            row[2] = a[r][0] * b[0][2] + a[r][1] * b[1][2] + a[r][2];
        }
        Affine { m }
    }

    /// The inverse map, or `None` for a singular one.
    #[must_use]
    pub fn inverse(&self) -> Option<Affine> {
        let [[a, b, c], [d, e, f]] = self.m;
        let det = a * e - b * d;
        if !det.is_finite() || det.abs() < 1e-300 {
            return None;
        }
        let (ia, ib, id, ie) = (e / det, -b / det, -d / det, a / det);
        Some(Affine {
            m: [[ia, ib, -(ia * c + ib * f)], [id, ie, -(id * c + ie * f)]],
        })
    }
}

/// The geometry of a photo: the EXIF orientation of the file and the user's settings.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Geometry {
    /// EXIF orientation, 1 to 8 (anything else counts as 1).
    pub exif_orientation: u16,
    /// Clockwise quarter turns the user applied (`settings.orientation`).
    pub quarter_turns: u8,
    /// Straighten angle in degrees, clockwise positive.
    pub angle_deg: f64,
    /// The crop rectangle on the canvas, normalised (the whole canvas when there is no crop).
    pub crop: NormRect,
    /// Work on the pixel grid of the buffer ([`Geometry::at_full_resolution`]): the crop starts on
    /// a whole pixel and an output of the crop's own size is a window on the pixels, not a
    /// resample.
    pub pixel_exact: bool,
}

impl Geometry {
    /// The geometry the settings ask for on a file with EXIF orientation `exif_orientation`.
    #[must_use]
    pub fn new(exif_orientation: u16, settings: &DevelopSettings) -> Self {
        let (angle_deg, crop) = match &settings.crop {
            Some(Crop {
                rect, angle_deg, ..
            }) => (*angle_deg, *rect),
            None => (0.0, NormRect::FULL),
        };
        Self {
            exif_orientation,
            quarter_turns: quarter_turns_of(settings.orientation),
            angle_deg,
            crop,
            pixel_exact: false,
        }
    }

    /// No turn, no straightening, no crop: the buffer as it is, in the camera's orientation.
    #[must_use]
    pub fn camera_only(exif_orientation: u16) -> Self {
        Self {
            exif_orientation,
            quarter_turns: 0,
            angle_deg: 0.0,
            crop: NormRect::FULL,
            pixel_exact: false,
        }
    }

    /// The same geometry for a buffer at the final resolution (the 100 % view and exports), where
    /// pixels are what the person looks at.
    ///
    /// The crop rectangle is a fraction of the canvas, so its edges fall between pixels. At a
    /// reduced scale that does not matter; at full resolution a fractional origin would make the
    /// picture a bicubic resample of itself shifted by up to half a pixel (visibly soft, and not
    /// the pixels the uncropped export holds). With [`Geometry::pixel_exact`] set the crop origin is
    /// rounded to a whole pixel (kept inside the canvas) and [`Geometry::grid`] maps an output of
    /// exactly [`Geometry::output_size`] one to one, so that without a straighten angle sampling
    /// copies pixels.
    #[must_use]
    pub fn at_full_resolution(mut self) -> Self {
        self.pixel_exact = true;
        self
    }

    /// The crop as a window of whole pixels on the canvas of a `buf_w x buf_h` buffer:
    /// `(x, y, width, height)`, the origin rounded to the nearest pixel and kept inside the canvas,
    /// the size [`Geometry::output_size`].
    #[must_use]
    pub fn crop_window(&self, buf_w: usize, buf_h: usize) -> (usize, usize, usize, usize) {
        let (cw, ch) = self.canvas_size(buf_w, buf_h);
        let (ow, oh) = self.output_size(buf_w, buf_h);
        let origin = |fraction: f64, canvas: usize, out: usize| {
            (fraction * canvas as f64)
                .round()
                .clamp(0.0, canvas.saturating_sub(out) as f64) as usize
        };
        (
            origin(self.crop.x, cw, ow),
            origin(self.crop.y, ch, oh),
            ow,
            oh,
        )
    }

    /// The same picture in the buffer's own orientation: no EXIF turn, no turn of the user's, the
    /// canvas is the buffer. This is the geometry of a DNG, whose pixels stay as the sensor wrote
    /// them while the `Orientation` tag says how to show them (`docs/09`: "90° steps go into the
    /// TIFF Orientation tag without rotating pixels").
    ///
    /// With `keep_crop` the crop and the straighten angle go with it: the crop window (the whole
    /// pixels of [`Geometry::crop_window`], mapped through the turns on the pixel lattice, so that
    /// both views hold exactly the same pixels) becomes the crop of the buffer, and the angle keeps
    /// its sense on the canvas (a mirrored orientation turns it the other way round in the
    /// buffer). Without it the result is the whole buffer, never straightened.
    #[must_use]
    pub fn upright(&self, buf_w: usize, buf_h: usize, keep_crop: bool) -> Geometry {
        let plain = Geometry {
            exif_orientation: 1,
            quarter_turns: 0,
            angle_deg: 0.0,
            crop: NormRect::FULL,
            pixel_exact: self.pixel_exact,
        };
        if !keep_crop {
            return plain;
        }
        let (cw, ch) = self.canvas_size(buf_w, buf_h);
        let (x0, y0, ow, oh) = self.crop_window(buf_w, buf_h);
        let to_buffer = |x: usize, y: usize| {
            let o = canvas_to_oriented(
                self.quarter_turns,
                [x as f64 / cw as f64, y as f64 / ch as f64],
            );
            let s = oriented_to_sensor(self.exif_orientation, o);
            ((s[0] * buf_w as f64).round(), (s[1] * buf_h as f64).round())
        };
        let (ax, ay) = to_buffer(x0, y0);
        let (bx, by) = to_buffer(x0 + ow, y0 + oh);
        let (lx, hx) = (ax.min(bx), ax.max(bx));
        let (ly, hy) = (ay.min(by), ay.max(by));
        let mirrored = matches!(self.exif_orientation, 2 | 4 | 5 | 7);
        Geometry {
            angle_deg: if mirrored {
                -self.angle_deg
            } else {
                self.angle_deg
            },
            crop: NormRect {
                x: lx / buf_w as f64,
                y: ly / buf_h as f64,
                width: (hx - lx) / buf_w as f64,
                height: (hy - ly) / buf_h as f64,
            },
            ..plain
        }
    }

    /// Whether the buffer's width and height are swapped on the canvas.
    #[must_use]
    pub fn swaps_sides(&self) -> bool {
        ((5..=8).contains(&self.exif_orientation)) != (self.quarter_turns % 2 == 1)
    }

    /// The size of the canvas for a buffer of `buf_w x buf_h`.
    #[must_use]
    pub fn canvas_size(&self, buf_w: usize, buf_h: usize) -> (usize, usize) {
        if self.swaps_sides() {
            (buf_h, buf_w)
        } else {
            (buf_w, buf_h)
        }
    }

    /// The size of the cropped output at the buffer's own resolution, at least 1 x 1.
    #[must_use]
    pub fn output_size(&self, buf_w: usize, buf_h: usize) -> (usize, usize) {
        let (cw, ch) = self.canvas_size(buf_w, buf_h);
        (
            ((self.crop.width * cw as f64).round() as usize).max(1),
            ((self.crop.height * ch as f64).round() as usize).max(1),
        )
    }

    /// Whether the straighten angle is zero.
    #[must_use]
    pub fn is_unrotated(&self) -> bool {
        self.angle_deg.abs() < ANGLE_EPS_DEG
    }

    /// The map from *final image* pixels (the cropped output at the buffer's resolution
    /// `buf_w x buf_h`, centres at `i + 0.5`) to canvas pixels, and from canvas pixels to buffer
    /// positions. Returns the two halves of [`Grid`].
    fn chains(&self, buf_w: usize, buf_h: usize) -> (Affine, Affine) {
        let (cw, ch) = self.canvas_size(buf_w, buf_h);
        let (cwf, chf) = (cw as f64, ch as f64);
        // Final image pixels -> canvas pixels before straightening: add the crop origin.
        let (ox, oy) = if self.pixel_exact {
            let (x, y, _, _) = self.crop_window(buf_w, buf_h);
            (x as f64, y as f64)
        } else {
            (self.crop.x * cwf, self.crop.y * chf)
        };
        let crop = Affine::scale_translate(1.0, 1.0, ox, oy);
        // Output pixel q samples the photo at p = c + R(-angle)(q - c).
        let straighten = Affine::rotation_about(-self.angle_deg, cwf * 0.5, chf * 0.5);
        (crop, straighten)
    }

    /// A grid for an output of `out_w x out_h` pixels that covers the crop rectangle, sampling a
    /// buffer of `buf_w x buf_h` pixels.
    #[must_use]
    pub fn grid(&self, buf: (usize, usize), out: (usize, usize)) -> Grid {
        let (cw, ch) = self.canvas_size(buf.0, buf.1);
        let (crop_w, crop_h) = if self.pixel_exact && out == self.output_size(buf.0, buf.1) {
            // The output is the crop's own size in whole pixels: one output pixel per canvas pixel.
            (out.0 as f64, out.1 as f64)
        } else {
            (
                (self.crop.width * cw as f64).max(1e-9),
                (self.crop.height * ch as f64).max(1e-9),
            )
        };
        // Output pixel (centre at i + 0.5) -> position inside the crop rectangle, in canvas pixels.
        let view = Affine::scale_translate(
            crop_w / out.0.max(1) as f64,
            crop_h / out.1.max(1) as f64,
            0.0,
            0.0,
        );
        self.grid_from(buf, view)
    }

    /// A grid for an arbitrary map `view_to_final` from output pixels to *final image pixels at
    /// the buffer's resolution* (what a zoomed and panned viewport needs).
    #[must_use]
    pub fn grid_from(&self, buf: (usize, usize), view_to_final: Affine) -> Grid {
        let (crop, straighten) = self.chains(buf.0, buf.1);
        let to_canvas = view_to_final.then(&crop).then(&straighten);
        let (cw, ch) = self.canvas_size(buf.0, buf.1);
        Grid {
            geometry: *self,
            to_canvas,
            canvas: (cw as f64, ch as f64),
            buffer: (buf.0 as f64, buf.1 as f64),
        }
    }

    /// Final image pixels (at the buffer's resolution) to a point of the photo as the camera
    /// oriented it, as fractions: where a viewport click lands in the space `white_pick` uses
    /// before the lens stage is accounted for. `None` outside the straightened photo.
    #[must_use]
    pub fn final_to_oriented(&self, buf: (usize, usize), x: f64, y: f64) -> Option<[f64; 2]> {
        let (crop, straighten) = self.chains(buf.0, buf.1);
        let (cx, cy) = crop.then(&straighten).apply(x, y);
        let (cw, ch) = self.canvas_size(buf.0, buf.1);
        let (u, v) = (cx / cw as f64, cy / ch as f64);
        if !(0.0..=1.0).contains(&u) || !(0.0..=1.0).contains(&v) {
            return None;
        }
        Some(canvas_to_oriented(self.quarter_turns, [u, v]))
    }
}

/// The user's orientation as clockwise quarter turns.
#[must_use]
pub fn quarter_turns_of(o: Orientation) -> u8 {
    match o {
        Orientation::Deg0 => 0,
        Orientation::Deg90 => 1,
        Orientation::Deg180 => 2,
        Orientation::Deg270 => 3,
    }
}

/// Maps output pixels to positions in a pixel buffer (see the module documentation).
#[derive(Debug, Clone, Copy)]
pub struct Grid {
    geometry: Geometry,
    /// Output pixel -> canvas pixels (crop and straighten applied).
    to_canvas: Affine,
    canvas: (f64, f64),
    buffer: (f64, f64),
}

impl Grid {
    /// The buffer position (continuous, pixel `i` centred at `i + 0.5`) that output pixel
    /// `(x, y)` samples, or `None` when it falls outside the photo (an empty corner of a
    /// straightened canvas, or outside the image in a viewport).
    #[inline]
    #[must_use]
    pub fn source(&self, x: usize, y: usize) -> Option<(f64, f64)> {
        self.source_at(x as f64 + 0.5, y as f64 + 0.5)
    }

    /// As [`Grid::source`] for a continuous output position.
    #[inline]
    #[must_use]
    pub fn source_at(&self, ox: f64, oy: f64) -> Option<(f64, f64)> {
        let (px, py) = self.to_canvas.apply(ox, oy);
        // A small tolerance keeps the edge pixels of an exact grid inside.
        let tol = 1e-6;
        if px < -tol || py < -tol || px > self.canvas.0 + tol || py > self.canvas.1 + tol {
            return None;
        }
        let (u, v) = (px / self.canvas.0, py / self.canvas.1);
        let o = canvas_to_oriented(self.geometry.quarter_turns, [u, v]);
        let s = oriented_to_sensor(self.geometry.exif_orientation, o);
        Some((
            snap(s[0].clamp(0.0, 1.0) * self.buffer.0),
            snap(s[1].clamp(0.0, 1.0) * self.buffer.1),
        ))
    }

    /// The same grid as plain data for the sampler of `ym-gpu` ([`SamplerGrid`], the CPU reference of
    /// the device kernel and, since G-05, the implementation `sample_to_camera` calls): every number
    /// the position chain reads, nothing else, so the two cannot disagree about where a pixel samples.
    #[must_use]
    pub fn sampler_grid(&self) -> SamplerGrid {
        SamplerGrid::new(
            self.to_canvas.m,
            self.canvas,
            self.buffer,
            self.geometry.exif_orientation,
            self.geometry.quarter_turns,
            self.geometry.is_unrotated(),
        )
    }

    /// The smallest rectangle of the buffer (integer pixels, `x0 <= x < x1`) that holds every
    /// position the output rectangle `[ox, ox + ow) x [oy, oy + oh)` samples, widened by `margin`
    /// pixels and clipped to the buffer. `None` when the whole rectangle lies outside the photo.
    #[must_use]
    pub fn source_bounds(
        &self,
        ox: usize,
        oy: usize,
        ow: usize,
        oh: usize,
        margin: usize,
    ) -> Option<(usize, usize, usize, usize)> {
        let corners = [
            (ox as f64, oy as f64),
            ((ox + ow) as f64, oy as f64),
            (ox as f64, (oy + oh) as f64),
            ((ox + ow) as f64, (oy + oh) as f64),
        ];
        let mut lo = (f64::INFINITY, f64::INFINITY);
        let mut hi = (f64::NEG_INFINITY, f64::NEG_INFINITY);
        let mut any = false;
        for (cx, cy) in corners {
            // Use the unclamped canvas position for the box, then map; corners of a rotated
            // rectangle may lie outside the canvas while the rectangle still overlaps it.
            let (px, py) = self.to_canvas.apply(cx, cy);
            let (u, v) = (px / self.canvas.0, py / self.canvas.1);
            let o = canvas_to_oriented(self.geometry.quarter_turns, [u, v]);
            let s = oriented_to_sensor(self.geometry.exif_orientation, o);
            let (bx, by) = (s[0] * self.buffer.0, s[1] * self.buffer.1);
            lo = (lo.0.min(bx), lo.1.min(by));
            hi = (hi.0.max(bx), hi.1.max(by));
            any = true;
        }
        if !any {
            return None;
        }
        let m = margin as f64;
        let x0 = (lo.0 - m).floor().max(0.0);
        let y0 = (lo.1 - m).floor().max(0.0);
        let x1 = (hi.0 + m).ceil().min(self.buffer.0);
        let y1 = (hi.1 + m).ceil().min(self.buffer.1);
        if x1 <= x0 || y1 <= y0 {
            return None;
        }
        Some((x0 as usize, y0 as usize, x1 as usize, y1 as usize))
    }

    /// How far one step along the output's x axis and one along its y axis move in the buffer, in
    /// buffer pixels. The canvas has the buffer's pixel size (it only swaps the sides), so these
    /// are the lengths of the columns of the output-to-canvas map, the same everywhere.
    #[must_use]
    pub fn step_lengths(&self) -> (f64, f64) {
        let m = &self.to_canvas.m;
        (m[0][0].hypot(m[1][0]), m[0][1].hypot(m[1][1]))
    }

    /// Whether the grid is an exact pixel copy: no rotation, one buffer pixel per output pixel and
    /// the output's pixel centres on the buffer's (it may still turn or mirror the image).
    #[must_use]
    pub fn is_pixel_copy(&self) -> bool {
        if !self.geometry.is_unrotated() {
            return false;
        }
        let (sx, sy) = self.step_lengths();
        let one = |v: f64| (v - 1.0).abs() < 1e-9;
        let (px, py) = self.to_canvas.apply(0.5, 0.5);
        let aligned = |v: f64| (v - 0.5 - (v - 0.5).round()).abs() < 1e-9;
        one(sx) && one(sy) && aligned(px) && aligned(py)
    }
}

/// Rounds a coordinate that is within 1e-9 of an integer or half integer, so exact grids stay
/// exact through the floating point maps.
#[inline]
fn snap(v: f64) -> f64 {
    let r = (v * 2.0).round() / 2.0;
    if (v - r).abs() < 1e-9 { r } else { v }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::settings::DevelopSettings;

    /// Independent forward maps of the 8 EXIF orientations on a w x h buffer, as the pixel
    /// coordinates in the oriented image of buffer pixel (x, y). The oriented image has size
    /// (ow, oh).
    fn forward(exif: u16, x: usize, y: usize, w: usize, h: usize) -> (usize, usize) {
        match exif {
            2 => (w - 1 - x, y),
            3 => (w - 1 - x, h - 1 - y),
            4 => (x, h - 1 - y),
            5 => (y, x),
            6 => (h - 1 - y, x),
            7 => (h - 1 - y, w - 1 - x),
            8 => (y, w - 1 - x),
            _ => (x, y),
        }
    }

    /// One more clockwise quarter turn of an image of size (w, h): pixel (x, y) -> (h - 1 - y, x).
    fn turn(x: usize, y: usize, h: usize) -> (usize, usize) {
        (h - 1 - y, x)
    }

    #[test]
    fn exif_and_user_turns_agree_with_an_independent_forward_map() {
        let (w, h) = (7usize, 4usize);
        for exif in 1..=8u16 {
            for turns in 0..4u8 {
                let settings = DevelopSettings {
                    orientation: match turns {
                        0 => Orientation::Deg0,
                        1 => Orientation::Deg90,
                        2 => Orientation::Deg180,
                        _ => Orientation::Deg270,
                    },
                    ..DevelopSettings::default()
                };
                let g = Geometry::new(exif, &settings);
                let (cw, ch) = g.canvas_size(w, h);
                assert_eq!(g.output_size(w, h), (cw, ch));
                let grid = g.grid((w, h), (cw, ch));
                assert!(grid.is_pixel_copy(), "exif {exif} turns {turns}");
                for by in 0..h {
                    for bx in 0..w {
                        // Where does the buffer pixel end up on the canvas?
                        let (mut px, mut py) = forward(exif, bx, by, w, h);
                        let (mut ow, mut oh) = if (5..=8).contains(&exif) {
                            (h, w)
                        } else {
                            (w, h)
                        };
                        for _ in 0..turns {
                            (px, py) = turn(px, py, oh);
                            std::mem::swap(&mut ow, &mut oh);
                        }
                        assert_eq!((ow, oh), (cw, ch));
                        let (sx, sy) = grid.source(px, py).expect("inside");
                        assert_eq!(
                            (sx, sy),
                            (bx as f64 + 0.5, by as f64 + 0.5),
                            "exif {exif} turns {turns} pixel ({bx},{by})"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn sensor_and_oriented_fractions_are_inverse() {
        for exif in 1..=8u16 {
            for &(u, v) in &[(0.1, 0.2), (0.9, 0.35), (0.5, 0.5), (0.0, 1.0)] {
                let s = oriented_to_sensor(exif, [u, v]);
                let back = sensor_to_oriented(exif, s);
                assert!(
                    (back[0] - u).abs() < 1e-12 && (back[1] - v).abs() < 1e-12,
                    "{exif}"
                );
            }
        }
        for turns in 0..4u8 {
            let c = oriented_to_canvas(turns, [0.2, 0.7]);
            let o = canvas_to_oriented(turns, c);
            assert!((o[0] - 0.2).abs() < 1e-12 && (o[1] - 0.7).abs() < 1e-12);
        }
    }

    #[test]
    fn straightening_rotates_clockwise_about_the_centre() {
        // A canvas 100 x 100 straightened by +90 degrees: the content turns clockwise, so the
        // content point (60, 50) (right of centre) appears below the centre, at (50, 60).
        let g = Geometry {
            angle_deg: 90.0,
            ..Geometry::camera_only(1)
        };
        let grid = g.grid((100, 100), (100, 100));
        // Output pixel (50, 60) samples the photo at (60, 50).
        let (sx, sy) = grid.source_at(50.0, 60.0).expect("inside");
        assert!(
            (sx - 60.0).abs() < 1e-9 && (sy - 50.0).abs() < 1e-9,
            "{sx} {sy}"
        );
        // A corner of the canvas stays inside after a quarter turn of a square.
        assert!(grid.source_at(1.0, 1.0).is_some());
        // 45 degrees: the empty corner of the canvas has no source.
        let g45 = Geometry {
            angle_deg: 45.0,
            ..Geometry::camera_only(1)
        };
        let grid45 = g45.grid((100, 100), (100, 100));
        assert!(grid45.source_at(2.0, 2.0).is_none());
        assert!(grid45.source_at(50.0, 50.0).is_some());
    }

    #[test]
    fn the_crop_selects_a_rectangle_of_the_canvas() {
        let g = Geometry {
            crop: NormRect {
                x: 0.25,
                y: 0.5,
                width: 0.5,
                height: 0.25,
            },
            ..Geometry::camera_only(1)
        };
        assert_eq!(g.output_size(200, 100), (100, 25));
        let grid = g.grid((200, 100), (100, 25));
        assert!(grid.is_pixel_copy());
        assert_eq!(grid.step_lengths(), (1.0, 1.0));
        assert_eq!(grid.source(0, 0), Some((50.5, 50.5)));
        assert_eq!(grid.source(99, 24), Some((149.5, 74.5)));
        // The same crop sampled into a half-size output samples every other pixel.
        let half = g.grid((200, 100), (50, 12));
        assert!(!half.is_pixel_copy());
        let (sx, sy) = half.step_lengths();
        assert!(
            (sx - 2.0).abs() < 1e-9 && (sy - 25.0 / 12.0).abs() < 1e-9,
            "{sx} {sy}"
        );
        let (x, _) = half.source(1, 0).expect("inside");
        assert!((x - (50.0 + 1.5 * 100.0 / 50.0)).abs() < 1e-9);
    }

    #[test]
    fn viewport_grid_composes_with_the_geometry() {
        // A 2x zoom, panned so that final pixel (10, 20) is at the viewport's (0, 0) corner.
        let g = Geometry {
            quarter_turns: 1,
            ..Geometry::camera_only(1)
        };
        let view = Affine::scale_translate(0.5, 0.5, 10.0, 20.0);
        let grid = g.grid_from((40, 30), view);
        // Canvas is 30 x 40 (swapped). Viewport (0, 0) centre -> final (10.25, 20.25).
        let (sx, sy) = grid.source(0, 0).expect("inside");
        // Final (10.25, 20.25) on the 30 x 40 canvas -> oriented fractions (v, 1 - u) with
        // u = 10.25 / 30 and v = 20.25 / 40.
        let (u, v) = (10.25 / 30.0, 20.25 / 40.0);
        assert!((sx - v * 40.0).abs() < 1e-9 && (sy - (1.0 - u) * 30.0).abs() < 1e-9);
    }

    #[test]
    fn source_bounds_cover_the_rectangle_and_clip() {
        let g = Geometry {
            angle_deg: 10.0,
            ..Geometry::camera_only(1)
        };
        let grid = g.grid((200, 100), (200, 100));
        let (x0, y0, x1, y1) = grid.source_bounds(50, 20, 40, 30, 2).expect("bounds");
        for y in 20..50 {
            for x in 50..90 {
                let (sx, sy) = grid.source(x, y).expect("inside");
                assert!(
                    sx >= x0 as f64 && sx <= x1 as f64,
                    "x {sx} not in {x0}..{x1}"
                );
                assert!(
                    sy >= y0 as f64 && sy <= y1 as f64,
                    "y {sy} not in {y0}..{y1}"
                );
            }
        }
        // A rectangle wholly outside a rotated canvas has no bounds only when it is off the
        // buffer; one at the far corner still clips to the buffer.
        let all = grid.source_bounds(0, 0, 200, 100, 5).expect("bounds");
        assert!(all.2 <= 200 && all.3 <= 100);
    }

    #[test]
    fn affine_algebra() {
        let a = Affine::scale_translate(2.0, 3.0, 1.0, -1.0);
        let r = Affine::rotation_about(30.0, 5.0, 7.0);
        let both = a.then(&r);
        let (x, y) = (1.25, -3.5);
        let direct = {
            let (ax, ay) = a.apply(x, y);
            r.apply(ax, ay)
        };
        let composed = both.apply(x, y);
        assert!((direct.0 - composed.0).abs() < 1e-12 && (direct.1 - composed.1).abs() < 1e-12);
        let inv = both.inverse().expect("invertible");
        let back = inv.apply(composed.0, composed.1);
        assert!((back.0 - x).abs() < 1e-12 && (back.1 - y).abs() < 1e-12);
        // The rotation fixes its centre.
        let c = r.apply(5.0, 7.0);
        assert!((c.0 - 5.0).abs() < 1e-12 && (c.1 - 7.0).abs() < 1e-12);
    }

    #[test]
    fn the_upright_geometry_samples_the_same_buffer_positions_as_the_baked_one() {
        // The DNG's pixels stay in the buffer's orientation; the baked geometry turns them. Every
        // pixel of the turned crop must be a pixel of the upright crop, through the dihedral map,
        // and sample the same place of the buffer (also with a straighten angle, whose sense a
        // mirrored orientation reverses). Odd sizes and crops that fall between pixels.
        let (w, h) = (37usize, 23usize);
        let crops = [
            NormRect::FULL,
            NormRect {
                x: 0.2573,
                y: 0.1931,
                width: 0.4981,
                height: 0.5113,
            },
            NormRect {
                x: 0.5,
                y: 0.0,
                width: 0.5,
                height: 0.3,
            },
        ];
        for exif in 1..=8u16 {
            for turns in 0..4u8 {
                for angle in [0.0, 7.5, -13.0] {
                    for crop in crops {
                        let baked = Geometry {
                            exif_orientation: exif,
                            quarter_turns: turns,
                            angle_deg: angle,
                            crop,
                            pixel_exact: true,
                        };
                        let up = baked.upright(w, h, true);
                        assert_eq!((up.exif_orientation, up.quarter_turns), (1, 0));
                        let (bw, bh) = baked.output_size(w, h);
                        let (uw, uh) = up.output_size(w, h);
                        let swapped = (5..=8).contains(&exif) != (turns % 2 == 1);
                        assert_eq!(
                            (uw, uh),
                            if swapped { (bh, bw) } else { (bw, bh) },
                            "exif {exif} turns {turns} crop {crop:?}: the window keeps its size"
                        );
                        let gb = baked.grid((w, h), (bw, bh));
                        let gu = up.grid((w, h), (uw, uh));
                        for y in 0..uh {
                            for x in 0..uw {
                                // The upright pixel in the oriented image, then the user's turns.
                                let (mut px, mut py) = forward(exif, x, y, uw, uh);
                                let (mut ow, mut oh) = if (5..=8).contains(&exif) {
                                    (uh, uw)
                                } else {
                                    (uw, uh)
                                };
                                for _ in 0..turns {
                                    (px, py) = turn(px, py, oh);
                                    std::mem::swap(&mut ow, &mut oh);
                                }
                                assert_eq!((ow, oh), (bw, bh));
                                let a = gu.source(x, y);
                                let b = gb.source(px, py);
                                match (a, b) {
                                    (None, None) => {}
                                    (Some(a), Some(b)) => assert!(
                                        (a.0 - b.0).abs() < 1e-6 && (a.1 - b.1).abs() < 1e-6,
                                        "exif {exif} turns {turns} angle {angle} crop {crop:?} \
                                         pixel ({x},{y}): {a:?} vs {b:?}"
                                    ),
                                    other => panic!(
                                        "exif {exif} turns {turns} angle {angle} pixel ({x},{y}): {other:?}"
                                    ),
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn the_dng_orientation_tag_turns_the_upright_pixels_into_the_baked_picture() {
        // A DNG holds the buffer's pixels and an `Orientation` tag from the writer's
        // `compose_orientation(camera EXIF orientation, user turn)`. A reader that applies the tag
        // to the pixels must show what the baked export shows: checked here for all eight camera
        // orientations and the four user turns, against the TIFF definition of the tag (the
        // independent forward maps of this module).
        let (w, h) = (9usize, 5usize);
        for exif in 1..=8u16 {
            for turns in 0..4u8 {
                let tag = ym_export::dng::compose_orientation(exif, 90 * u16::from(turns))
                    .expect("a valid orientation");
                let baked = Geometry {
                    exif_orientation: exif,
                    quarter_turns: turns,
                    ..Geometry::camera_only(1)
                };
                let (cw, ch) = baked.canvas_size(w, h);
                let grid = baked.grid((w, h), (cw, ch));
                for y in 0..h {
                    for x in 0..w {
                        // Where the tag puts buffer pixel (x, y) of the unturned image.
                        let (px, py) = forward(tag, x, y, w, h);
                        let (sx, sy) = grid.source(px, py).expect("inside");
                        assert_eq!(
                            (sx, sy),
                            (x as f64 + 0.5, y as f64 + 0.5),
                            "exif {exif} turns {turns} (tag {tag}) pixel ({x},{y})"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn the_upright_geometry_without_the_crop_is_the_whole_buffer_unturned() {
        let settings = DevelopSettings {
            orientation: Orientation::Deg90,
            crop: Some(Crop {
                rect: NormRect {
                    x: 0.1,
                    y: 0.1,
                    width: 0.5,
                    height: 0.5,
                },
                angle_deg: 11.0,
                aspect: crate::settings::CropAspect::Free,
            }),
            ..DevelopSettings::default()
        };
        let g = Geometry::new(6, &settings).at_full_resolution();
        let up = g.upright(60, 40, false);
        assert_eq!(up.output_size(60, 40), (60, 40));
        assert!(up.is_unrotated());
        assert!(
            up.at_full_resolution()
                .grid((60, 40), (60, 40))
                .is_pixel_copy()
        );
        // The crop window of the user's crop, on the canvas, in whole pixels.
        let (x, y, w, h) = g.crop_window(60, 40);
        assert_eq!(g.output_size(60, 40), (w, h));
        assert!(
            x + w <= 40 && y + h <= 60,
            "inside the turned canvas (40 x 60): {x} {y} {w} {h}"
        );
    }

    /// The sampler of `ym-gpu` reads the grid as plain data ([`Grid::sampler_grid`]): the position
    /// chain, the inside test and the kernel choice are the grid's own, for every orientation, turn
    /// and straighten angle.
    #[test]
    fn the_sampler_grid_is_this_grid_as_plain_data() {
        let (w, h) = (64usize, 48usize);
        for exif in 1..=8u16 {
            for turns in 0..4u8 {
                for angle in [0.0, 3.5, -12.0] {
                    let settings = DevelopSettings {
                        orientation: match turns {
                            0 => Orientation::Deg0,
                            1 => Orientation::Deg90,
                            2 => Orientation::Deg180,
                            _ => Orientation::Deg270,
                        },
                        crop: Some(Crop {
                            rect: NormRect {
                                x: 0.05,
                                y: 0.1,
                                width: 0.8,
                                height: 0.7,
                            },
                            angle_deg: angle,
                            aspect: crate::settings::CropAspect::Free,
                        }),
                        ..DevelopSettings::default()
                    };
                    let g = Geometry::new(exif, &settings);
                    for out in [(40usize, 30usize), (90, 70), (23, 17)] {
                        let grid = g.grid((w, h), out);
                        let sg = grid.sampler_grid();
                        assert_eq!(sg.is_pixel_copy(), grid.is_pixel_copy());
                        assert_eq!(sg.step_lengths(), grid.step_lengths());
                        for (x, y) in [(0, 0), (out.0 - 1, 3), (7, out.1 - 1), (out.0 / 2, out.1 / 2)] {
                            assert_eq!(
                                sg.source(x, y),
                                grid.source(x, y),
                                "exif {exif} turns {turns} angle {angle} out {out:?} pixel ({x}, {y})"
                            );
                        }
                        assert_eq!(
                            sg.source_at(1.25, 2.75),
                            grid.source_at(1.25, 2.75),
                            "exif {exif} turns {turns} angle {angle}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn final_to_oriented_inverts_the_grid_for_pure_orientation() {
        let (w, h) = (40usize, 30usize);
        for exif in 1..=8u16 {
            let g = Geometry::camera_only(exif);
            let (cw, ch) = g.canvas_size(w, h);
            let grid = g.grid((w, h), (cw, ch));
            for &(fx, fy) in &[(3.5, 4.5), (cw as f64 - 0.5, ch as f64 - 0.5)] {
                let o = g.final_to_oriented((w, h), fx, fy).expect("inside");
                let s = oriented_to_sensor(exif, o);
                let (bx, by) = grid.source_at(fx, fy).expect("inside");
                assert!((s[0] * w as f64 - bx).abs() < 1e-9 && (s[1] * h as f64 - by).abs() < 1e-9);
            }
        }
    }
}
