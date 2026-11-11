//! The face parts of `docs/27` section 4 from the mesh landmarks and the face-skin probability of a segmenter (S-10): the
//! maths of the real `FaceParser`, with no network in it. Anything that gives 478 landmarks and a [`SkinClass`] can use it;
//! `crate::mediapipe` (feature `mediapipe`) gives them from Face Mesh V2 and Selfie Multiclass.
//!
//! Coordinates are continuous image coordinates: pixel `(i, j)` covers `[i, i + 1) x [j, j + 1)`.
//!
//! **Skin without ears** ([`skin_mask`]), the recipe of `docs/27` section 1:
//!
//! 1. the *face-skin probability* of the segmenter (its face-skin class includes the eyes, brows, lips and ears and reaches
//!    the hairline), at full resolution by bilinear sampling;
//! 2. times the *face oval* of the mesh, with a *forehead band*: the oval's upper vertices move up by up to
//!    [`FOREHEAD_BAND`] of the face's height, so skin the mesh stops short of is kept where the segmenter says it is skin
//!    (the ears and the neck stay outside the oval, which is how "no neck" and "without ears" are met);
//! 3. minus the *eyes*, *brows*, *lips* and *nostrils* from the mesh polygons, each grown by a small margin;
//! 4. a per-face *colour trim at the edges*: a Gaussian of the skin's colour is fitted to the solid core of the mask, and in
//!    a band along the mask's edge the pixels far from it (hair, shadow under the jaw, a collar) are taken out;
//! 5. the *guided filter* of `docs/algorithms/06` (Refine edges), radius [`GUIDE_RADIUS`] of the face's height, pulls the
//!    edge to the picture's own edge.
//!
//! **Eyes** ([`eyes_mask`]) are the two lid-contour polygons; **mouth interior** ([`mouth_interior_mask`]) is the inner lip
//! contour (nothing when the lips are closed).
//!
//! # Determinism
//!
//! Pure functions of the landmarks, the probabilities and the picture. Reductions run in a fixed order; passes over pixels
//! write disjoint rows.

use rayon::prelude::*;

use crate::error::Result;
use crate::face_raster::{fill_polygon, offset_polygon, region_of, render_mesh};
use crate::face_tables::texture;
use crate::face_topology::{
    FACE_OVAL, INNER_LIPS, LEFT_EYE, LEFT_EYEBROW, OUTER_LIPS, RIGHT_EYE, RIGHT_EYEBROW,
};
use crate::guided::guided_filter;
use crate::models::MeshTable;
use crate::types::{BoundedMask, LinearAwg4Image, Mask, Rect};

/// The number of landmarks of Face Mesh V2: 468 mesh points and 10 for the irises.
pub const LANDMARK_COUNT: usize = 478;
/// How far the forehead may reach above the mesh's oval, as a fraction of the face's height (chin to top of the oval).
pub const FOREHEAD_BAND: f64 = 0.16;
/// The margins the cut-out polygons grow by, as fractions of the face's height.
pub const EYE_MARGIN: f64 = 0.012;
/// See [`EYE_MARGIN`]: the brows.
pub const BROW_MARGIN: f64 = 0.010;
/// See [`EYE_MARGIN`]: the lips.
pub const LIP_MARGIN: f64 = 0.008;
/// See [`EYE_MARGIN`]: the nostrils.
pub const NOSTRIL_MARGIN: f64 = 0.004;
/// The width of the band along the mask's edge where the colour trim works, as a fraction of the face's height.
pub const TRIM_BAND: f64 = 0.07;
/// The colour distance (in standard deviations of the fitted Gaussian) where the trim starts and where it is complete.
pub const TRIM_FROM_SIGMA: f64 = 3.0;
/// See [`TRIM_FROM_SIGMA`].
pub const TRIM_TO_SIGMA: f64 = 5.0;
/// The thinnest opening of the mouth that counts as open, as a fraction of the face's height.
pub const MOUTH_OPEN_MIN: f64 = 0.004;
/// The radius of the guided filter, as a fraction of the face's height.
pub const GUIDE_RADIUS: f64 = 0.02;
/// The regularisation of the guided filter (`docs/algorithms/06`: 3.2e-4 is Refine edges at 50 %).
pub const GUIDE_EPS: f32 = 3.2e-4;
/// The most colour samples taken from the core to fit the trim's Gaussian.
const TRIM_SAMPLES: usize = 20_000;

/// The nostrils: the slit of the right nostril (on the left of the picture) and of the left one, as landmark indices.
const RIGHT_NOSTRIL: [u16; 5] = [98, 240, 75, 60, 99];
const LEFT_NOSTRIL: [u16; 5] = [327, 460, 305, 290, 328];

/// The 478 landmarks of one face in continuous image coordinates.
#[derive(Debug, Clone, PartialEq)]
pub struct FaceGeometry {
    points: Vec<[f64; 2]>,
}

impl FaceGeometry {
    /// Wraps the landmarks (at least [`LANDMARK_COUNT`], all finite); `None` otherwise.
    pub fn new(points: Vec<[f64; 2]>) -> Option<Self> {
        (points.len() >= LANDMARK_COUNT
            && points.iter().all(|p| p[0].is_finite() && p[1].is_finite()))
        .then_some(Self { points })
    }

    /// All landmarks.
    pub fn points(&self) -> &[[f64; 2]] {
        &self.points
    }

    fn ring(&self, ids: &[u16]) -> Vec<[f64; 2]> {
        ids.iter().map(|&i| self.points[usize::from(i)]).collect()
    }

    /// The face's height: the distance from the chin (152) to the top of the oval (10).
    pub fn height_px(&self) -> f64 {
        let (a, b) = (self.points[10], self.points[152]);
        (a[0] - b[0]).hypot(a[1] - b[1]).max(1.0)
    }

    /// The unit vector that points up the face, from the chin to the top of the oval.
    pub fn up(&self) -> [f64; 2] {
        let (a, b) = (self.points[10], self.points[152]);
        let h = self.height_px();
        [(a[0] - b[0]) / h, (a[1] - b[1]) / h]
    }

    /// The face oval, its upper vertices moved up by the forehead band.
    pub fn skin_polygon(&self) -> Vec<[f64; 2]> {
        let up = self.up();
        let band = FOREHEAD_BAND * self.height_px();
        let along = |p: [f64; 2]| p[0] * up[0] + p[1] * up[1];
        let eyes = [33usize, 263, 133, 362];
        let eye_level =
            eyes.iter().map(|&i| along(self.points[i])).sum::<f64>() / eyes.len() as f64;
        let top = along(self.points[10]);
        self.ring(&FACE_OVAL)
            .into_iter()
            .map(|p| {
                let w = ((along(p) - eye_level) / (top - eye_level).max(1e-6)).clamp(0.0, 1.0);
                let d = band * w * w;
                [p[0] + up[0] * d, p[1] + up[1] * d]
            })
            .collect()
    }

    /// The two eyes (lid contours).
    pub fn eye_polygons(&self) -> [Vec<[f64; 2]>; 2] {
        [self.ring(&RIGHT_EYE), self.ring(&LEFT_EYE)]
    }

    /// The two brows.
    pub fn brow_polygons(&self) -> [Vec<[f64; 2]>; 2] {
        [self.ring(&RIGHT_EYEBROW), self.ring(&LEFT_EYEBROW)]
    }

    /// The outer contour of the lips.
    pub fn lip_polygon(&self) -> Vec<[f64; 2]> {
        self.ring(&OUTER_LIPS)
    }

    /// The inner contour of the lips: the opening of the mouth.
    pub fn mouth_polygon(&self) -> Vec<[f64; 2]> {
        self.ring(&INNER_LIPS)
    }

    /// The two nostrils.
    pub fn nostril_polygons(&self) -> [Vec<[f64; 2]>; 2] {
        [self.ring(&RIGHT_NOSTRIL), self.ring(&LEFT_NOSTRIL)]
    }
}

/// The face-skin probability of a square crop of the picture (the segmenter's output), `size x size` values `0..1`.
#[derive(Debug, Clone, PartialEq)]
pub struct SkinClass {
    /// The crop's centre in continuous image coordinates.
    pub center_px: [f64; 2],
    /// The crop's side.
    pub side_px: f64,
    /// The crop's rotation in radians: its x axis points along `(cos, sin)` in the picture.
    pub rotation_rad: f64,
    /// The crop's side in pixels of the probability map.
    pub size: usize,
    /// `size * size` probabilities, row-major; pixel `i` of the crop sits at `u = i / size` of the side.
    pub probs: Vec<f32>,
}

impl SkinClass {
    /// The probability at the continuous image position `(x, y)`, bilinear, held to the crop's edge; 0 outside the crop.
    pub fn at(&self, x: f64, y: f64) -> f32 {
        let (s, c) = self.rotation_rad.sin_cos();
        let (dx, dy) = (x - self.center_px[0], y - self.center_px[1]);
        let u = (dx * c + dy * s) / self.side_px + 0.5;
        let v = (-dx * s + dy * c) / self.side_px + 0.5;
        if !(-0.02..=1.02).contains(&u) || !(-0.02..=1.02).contains(&v) {
            return 0.0;
        }
        let n = self.size as f64;
        let (px, py) = (u * n, v * n);
        let (fx, fy) = (px.floor(), py.floor());
        let (tx, ty) = ((px - fx) as f32, (py - fy) as f32);
        let at = |xx: i64, yy: i64| {
            let xx = xx.clamp(0, self.size as i64 - 1) as usize;
            let yy = yy.clamp(0, self.size as i64 - 1) as usize;
            self.probs[yy * self.size + xx]
        };
        let (x0, y0) = (fx as i64, fy as i64);
        let top = at(x0, y0) * (1.0 - tx) + at(x0 + 1, y0) * tx;
        let bottom = at(x0, y0 + 1) * (1.0 - tx) + at(x0 + 1, y0 + 1) * tx;
        top * (1.0 - ty) + bottom * ty
    }
}

/// A plane of `f32` over a region of the image.
struct Plane {
    region: Rect,
    data: Vec<f32>,
}

impl Plane {
    fn zeros(region: Rect) -> Self {
        Self {
            region,
            data: vec![0.0; region.width_px as usize * region.height_px as usize],
        }
    }

    fn width(&self) -> usize {
        self.region.width_px as usize
    }

    /// Raises the plane to the coverage of `poly` where that is larger (a union of polygons).
    fn union_polygon(&mut self, poly: &[[f64; 2]], margin_px: f64) {
        let poly = offset_polygon(poly, margin_px);
        let Some(sub) = region_of(&poly, 1.0, (u32::MAX / 2, u32::MAX / 2)) else {
            return;
        };
        // The part of the polygon's box that is inside this plane.
        let r = self.region;
        let (x0, y0) = (sub.x_px.max(r.x_px), sub.y_px.max(r.y_px));
        let (x1, y1) = (
            (i64::from(sub.x_px) + i64::from(sub.width_px)).min(r.right_px()),
            (i64::from(sub.y_px) + i64::from(sub.height_px)).min(r.bottom_px()),
        );
        if i64::from(x0) >= x1 || i64::from(y0) >= y1 {
            return;
        }
        let window = Rect::new(
            x0,
            y0,
            (x1 - i64::from(x0)) as u32,
            (y1 - i64::from(y0)) as u32,
        );
        let cov = fill_polygon(&poly, window);
        let w = self.width();
        for row in 0..window.height_px as usize {
            let dst = (window.y_px - r.y_px) as usize + row;
            for col in 0..window.width_px as usize {
                let d = &mut self.data[dst * w + (window.x_px - r.x_px) as usize + col];
                *d = d.max(cov[row * window.width_px as usize + col]);
            }
        }
    }
}

fn to_u8(plane: &[f32]) -> Vec<u8> {
    plane
        .iter()
        .map(|&v| (v * 255.0 + 0.5).clamp(0.0, 255.0) as u8)
        .collect()
}

fn bounded(region: Rect, plane: &[f32]) -> Result<Option<BoundedMask>> {
    let mask = Mask::new(region.width_px, region.height_px, to_u8(plane))?;
    Ok(BoundedMask::new(region, mask)?.crop_to_nonzero())
}

fn smoothstep(e0: f64, e1: f64, x: f64) -> f64 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// The skin colour feature of a pixel: log brightness and two log chromaticity ratios (linear light).
fn colour_feature(p: [f32; 3]) -> [f64; 3] {
    let c = p.map(|v| f64::from(v.max(1e-4)));
    [c[1].ln(), (c[0] / c[1]).ln(), (c[2] / c[1]).ln()]
}

/// The Gaussian of the skin's colour: mean and the inverse covariance (a Cholesky solve done once).
struct SkinColour {
    mean: [f64; 3],
    /// The lower Cholesky factor of the covariance.
    chol: [[f64; 3]; 3],
}

impl SkinColour {
    /// Fits the Gaussian to the pixels of `image` where `core` is set, taking at most [`TRIM_SAMPLES`] of them at a fixed
    /// stride; `None` when there are too few.
    fn fit(image: &LinearAwg4Image, plane: &Plane, core: &[bool]) -> Option<Self> {
        let w = plane.width();
        let positions: Vec<usize> = (0..core.len()).filter(|&i| core[i]).collect();
        if positions.len() < 64 {
            return None;
        }
        let stride = positions.len().div_ceil(TRIM_SAMPLES);
        let feats: Vec<[f64; 3]> = positions
            .iter()
            .step_by(stride)
            .map(|&i| {
                let (x, y) = (
                    plane.region.x_px as usize + i % w,
                    plane.region.y_px as usize + i / w,
                );
                colour_feature(image.data[y * image.width_px as usize + x])
            })
            .collect();
        let n = feats.len() as f64;
        let mut mean = [0.0; 3];
        for f in &feats {
            for k in 0..3 {
                mean[k] += f[k] / n;
            }
        }
        let mut cov = [[0.0f64; 3]; 3];
        for f in &feats {
            for a in 0..3 {
                for b in 0..3 {
                    cov[a][b] += (f[a] - mean[a]) * (f[b] - mean[b]) / n;
                }
            }
        }
        // A floor on the variances keeps a flat patch from making every other colour infinitely far.
        for (k, row) in cov.iter_mut().enumerate() {
            row[k] += 1e-4;
        }
        let mut l = [[0.0f64; 3]; 3];
        for i in 0..3 {
            for j in 0..=i {
                let mut s = cov[i][j];
                for (a, b) in l[i].iter().zip(&l[j]).take(j) {
                    s -= a * b;
                }
                if i == j {
                    if s <= 0.0 {
                        return None;
                    }
                    l[i][i] = s.sqrt();
                } else {
                    l[i][j] = s / l[j][j];
                }
            }
        }
        Some(Self { mean, chol: l })
    }

    /// The Mahalanobis distance of a pixel from the skin colour, in standard deviations.
    fn distance(&self, p: [f32; 3]) -> f64 {
        let f = colour_feature(p);
        let d = [
            f[0] - self.mean[0],
            f[1] - self.mean[1],
            f[2] - self.mean[2],
        ];
        // Solve L z = d.
        let z0 = d[0] / self.chol[0][0];
        let z1 = (d[1] - self.chol[1][0] * z0) / self.chol[1][1];
        let z2 = (d[2] - self.chol[2][0] * z0 - self.chol[2][1] * z1) / self.chol[2][2];
        (z0 * z0 + z1 * z1 + z2 * z2).sqrt()
    }
}

/// A box mean of radius `r` of a plane (clipped at its edge), the cost independent of `r`.
fn box_blur(plane: &[f32], w: usize, h: usize, r: usize) -> Vec<f32> {
    let sw = w + 1;
    let mut sat = vec![0.0f64; sw * (h + 1)];
    for y in 0..h {
        let mut row = 0.0f64;
        for x in 0..w {
            row += f64::from(plane[y * w + x]);
            sat[(y + 1) * sw + x + 1] = sat[y * sw + x + 1] + row;
        }
    }
    let mut out = vec![0.0f32; w * h];
    out.par_chunks_mut(w).enumerate().for_each(|(y, line)| {
        let (y0, y1) = (y.saturating_sub(r), (y + r + 1).min(h));
        for (x, v) in line.iter_mut().enumerate() {
            let (x0, x1) = (x.saturating_sub(r), (x + r + 1).min(w));
            let sum = sat[y1 * sw + x1] - sat[y0 * sw + x1] - sat[y1 * sw + x0] + sat[y0 * sw + x0];
            *v = (sum / ((x1 - x0) * (y1 - y0)) as f64) as f32;
        }
    });
    out
}

/// The skin of the face without the ears (`docs/27` 5.2): see the module documentation. `None` when nothing of the face is
/// inside the image or the segmenter finds no skin in it.
///
/// # Errors
///
/// [`crate::RetouchError`] for an image whose data does not match its size.
pub fn skin_mask(
    image: &LinearAwg4Image,
    geometry: &FaceGeometry,
    class: &SkinClass,
) -> Result<Option<BoundedMask>> {
    image.validate()?;
    let size = geometry.height_px();
    let oval = geometry.skin_polygon();
    let size_px = (image.width_px, image.height_px);
    let Some(region) = region_of(&oval, GUIDE_RADIUS * size + 2.0, size_px) else {
        return Ok(None);
    };
    let (w, h) = (region.width_px as usize, region.height_px as usize);
    let (ox, oy) = (f64::from(region.x_px), f64::from(region.y_px));

    // Steps 1 and 2: the segmenter's probability inside the oval with its forehead band.
    let oval_cover = fill_polygon(&oval, region);
    let mut m = Plane::zeros(region);
    m.data
        .par_chunks_mut(w)
        .enumerate()
        .for_each(|(row, line)| {
            let y = oy + row as f64 + 0.5;
            for (col, v) in line.iter_mut().enumerate() {
                let c = oval_cover[row * w + col];
                if c > 0.0 {
                    *v = c * class.at(ox + col as f64 + 0.5, y);
                }
            }
        });
    drop(oval_cover);

    // Step 3: the features come out.
    let mut cut = Plane::zeros(region);
    for p in geometry.eye_polygons() {
        cut.union_polygon(&p, EYE_MARGIN * size);
    }
    for p in geometry.brow_polygons() {
        cut.union_polygon(&p, BROW_MARGIN * size);
    }
    cut.union_polygon(&geometry.lip_polygon(), LIP_MARGIN * size);
    for p in geometry.nostril_polygons() {
        cut.union_polygon(&p, NOSTRIL_MARGIN * size);
    }
    for (v, &c) in m.data.iter_mut().zip(&cut.data) {
        *v *= 1.0 - c;
    }
    drop(cut);

    // Step 4: the colour trim in a band along the edge.
    let band = ((TRIM_BAND * size).round() as usize).max(1);
    let blurred = box_blur(&m.data, w, h, band);
    let core: Vec<bool> = m
        .data
        .iter()
        .zip(&blurred)
        .map(|(&v, &b)| v > 0.95 && b > 0.9)
        .collect();
    if let Some(colour) = SkinColour::fit(image, &m, &core) {
        let iw = image.width_px as usize;
        m.data
            .par_chunks_mut(w)
            .zip(blurred.par_chunks(w))
            .enumerate()
            .for_each(|(row, (line, blur))| {
                let y = region.y_px as usize + row;
                for (col, (v, &b)) in line.iter_mut().zip(blur).enumerate() {
                    if *v <= 0.0 {
                        continue;
                    }
                    // 1 along the edge, 0 well inside.
                    let edge = 1.0 - smoothstep(0.85, 0.99, f64::from(b));
                    if edge <= 0.0 {
                        continue;
                    }
                    let d = colour.distance(image.data[y * iw + region.x_px as usize + col]);
                    let trim = smoothstep(TRIM_FROM_SIGMA, TRIM_TO_SIGMA, d);
                    *v *= (1.0 - edge * trim) as f32;
                }
            });
    }
    drop(blurred);

    // Step 5: the guided filter, with the picture's luma (encoded, 0..1) as the guide.
    let iw = image.width_px as usize;
    let mut guide = vec![0.0f32; w * h];
    guide.par_chunks_mut(w).enumerate().for_each(|(row, line)| {
        let y = region.y_px as usize + row;
        for (col, g) in line.iter_mut().enumerate() {
            let p = image.data[y * iw + region.x_px as usize + col];
            let luma = 0.2126 * p[0].max(0.0) + 0.7152 * p[1].max(0.0) + 0.0722 * p[2].max(0.0);
            *g = luma.sqrt().min(1.0);
        }
    });
    let radius = ((GUIDE_RADIUS * size).round() as usize).max(1);
    let mut refined = guided_filter(&m.data, &guide, w, h, radius, GUIDE_EPS);
    // The filter follows the picture's edges and may pull skin back over a lid or a lip; the polygons themselves (no
    // margin) are cut once more, so the eyes, brows, lips and nostrils are never skin.
    let mut holes = Plane::zeros(region);
    for p in geometry
        .eye_polygons()
        .iter()
        .chain(&geometry.brow_polygons())
    {
        holes.union_polygon(p, 0.0);
    }
    holes.union_polygon(&geometry.lip_polygon(), 0.0);
    for p in geometry.nostril_polygons() {
        holes.union_polygon(&p, 0.0);
    }
    for (v, &c) in refined.iter_mut().zip(&holes.data) {
        *v *= 1.0 - c;
    }
    bounded(region, &refined)
}

/// The eyes: the two lid contours (`FacePart::Eyes`). `None` when they are outside the image.
///
/// # Errors
///
/// [`crate::RetouchError`] from the mask constructors (never for a valid size).
pub fn eyes_mask(geometry: &FaceGeometry, size_px: (u32, u32)) -> Result<Option<BoundedMask>> {
    union_mask(&geometry.eye_polygons(), 0.0, size_px)
}

/// The inside of the mouth: the inner lip contour (`FacePart::MouthInterior`). `None` when the lips are closed or the mouth
/// is outside the image.
///
/// # Errors
///
/// [`crate::RetouchError`] from the mask constructors (never for a valid size).
pub fn mouth_interior_mask(
    geometry: &FaceGeometry,
    size_px: (u32, u32),
) -> Result<Option<BoundedMask>> {
    let polygon = geometry.mouth_polygon();
    // The opening's height along the face's own up axis; closed lips leave less than 0.4 % of the face's height.
    let up = geometry.up();
    let (mut lo, mut hi) = (f64::MAX, f64::MIN);
    for p in &polygon {
        let a = p[0] * up[0] + p[1] * up[1];
        lo = lo.min(a);
        hi = hi.max(a);
    }
    if hi - lo < MOUTH_OPEN_MIN * geometry.height_px() {
        return Ok(None);
    }
    union_mask(&[polygon], 0.0, size_px)
}

/// The union of polygons, each grown by `margin_px`, as a mask on a region clipped to the image.
fn union_mask(
    polygons: &[Vec<[f64; 2]>],
    margin_px: f64,
    size_px: (u32, u32),
) -> Result<Option<BoundedMask>> {
    let all: Vec<[f64; 2]> = polygons.iter().flatten().copied().collect();
    let Some(region) = region_of(&all, margin_px.abs() + 2.0, size_px) else {
        return Ok(None);
    };
    let mut plane = Plane::zeros(region);
    for p in polygons {
        plane.union_polygon(p, margin_px);
    }
    bounded(region, &plane.data)
}

/// One of the two tables of the mesh drawn at the face's landmarks, clipped to the image: a mask of zeros (one pixel) when
/// the face is outside it.
///
/// # Errors
///
/// [`crate::RetouchError`] from the mask constructors (never for a valid size).
pub fn mesh_table_mask(
    geometry: &FaceGeometry,
    table: MeshTable,
    size_px: (u32, u32),
) -> Result<BoundedMask> {
    let pts: Vec<[f64; 2]> = geometry.points().to_vec();
    match render_mesh(&pts, texture(table), size_px)? {
        Some(mask) => Ok(mask),
        None => BoundedMask::new(Rect::new(0, 0, 1, 1), Mask::zeros(1, 1)?),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::face_topology::UV;

    /// A synthetic face: the 468 vertices on the model's texture layout scaled to `size` pixels (a front view, up is up),
    /// the ten iris points at the eye centres.
    fn flat_geometry(size: f64, x0: f64, y0: f64) -> FaceGeometry {
        let mut pts: Vec<[f64; 2]> = UV
            .iter()
            .map(|uv| [x0 + f64::from(uv[0]) * size, y0 + f64::from(uv[1]) * size])
            .collect();
        for centre in [[33usize, 133], [263, 362]] {
            let c = [
                (pts[centre[0]][0] + pts[centre[1]][0]) / 2.0,
                (pts[centre[0]][1] + pts[centre[1]][1]) / 2.0,
            ];
            pts.extend(std::iter::repeat_n(c, 5));
        }
        FaceGeometry::new(pts).expect("geometry")
    }

    const SKIN: [f32; 3] = [0.45, 0.30, 0.22];

    fn picture(w: u32, h: u32, f: impl Fn(u32, u32) -> [f32; 3]) -> LinearAwg4Image {
        let data = (0..h)
            .flat_map(|y| (0..w).map(move |x| (x, y)))
            .map(|(x, y)| f(x, y))
            .collect();
        LinearAwg4Image::new(w, h, data).expect("image")
    }

    /// The segmenter says skin everywhere in a crop that covers the image.
    fn all_skin(w: u32, h: u32) -> SkinClass {
        SkinClass {
            center_px: [f64::from(w) / 2.0, f64::from(h) / 2.0],
            side_px: f64::from(w.max(h)) * 1.5,
            rotation_rad: 0.0,
            size: 16,
            probs: vec![1.0; 256],
        }
    }

    fn at(m: &BoundedMask, x: f64, y: f64) -> u8 {
        m.mask.get(
            x.floor() as i64 - i64::from(m.region.x_px),
            y.floor() as i64 - i64::from(m.region.y_px),
        )
    }

    #[test]
    fn geometry_needs_all_the_landmarks_and_finds_the_face_axes() {
        assert!(FaceGeometry::new(vec![[0.0, 0.0]; 100]).is_none());
        let mut bad = vec![[1.0, 1.0]; 478];
        bad[5][0] = f64::NAN;
        assert!(FaceGeometry::new(bad).is_none());
        let g = flat_geometry(300.0, 50.0, 40.0);
        let up = g.up();
        assert!(up[1] < -0.99 && up[0].abs() < 0.05, "{up:?}");
        assert!(
            (g.height_px() - 0.847 * 300.0).abs() < 4.0,
            "{}",
            g.height_px()
        );
        // The forehead band raises the top of the oval and leaves the jaw where it was.
        let top_oval = g.points()[10][1];
        let polygon = g.skin_polygon();
        let top = polygon.iter().map(|p| p[1]).fold(f64::MAX, f64::min);
        assert!(
            top < top_oval - 0.1 * g.height_px(),
            "{top} against {top_oval}"
        );
        let chin = polygon.iter().map(|p| p[1]).fold(f64::MIN, f64::max);
        assert!((chin - g.points()[152][1]).abs() < 1e-9);
    }

    #[test]
    fn the_skin_keeps_the_cheeks_and_the_forehead_and_loses_the_features_the_ears_and_the_background()
     {
        let g = flat_geometry(300.0, 50.0, 40.0);
        let img = picture(400, 380, |_, _| SKIN);
        let class = all_skin(400, 380);
        let m = skin_mask(&img, &g, &class).expect("mask").expect("skin");
        let p = g.points();
        for (i, name) in [
            (205usize, "right cheek"),
            (425, "left cheek"),
            (152, "chin"),
            (9, "forehead"),
        ] {
            // A little inside the point so the chin is inside the oval.
            let (x, y) = (p[i][0], p[i][1] - if i == 152 { 6.0 } else { 0.0 });
            assert!(at(&m, x, y) > 200, "{name}: {}", at(&m, x, y));
        }
        // Eyes, brows, lips and nostrils are out.
        for (a, b, name) in [
            (159usize, 145usize, "right eye"),
            (386, 374, "left eye"),
            (13, 14, "mouth"),
            (0, 17, "lips"),
        ] {
            let (x, y) = ((p[a][0] + p[b][0]) / 2.0, (p[a][1] + p[b][1]) / 2.0);
            assert!(at(&m, x, y) < 30, "{name}: {}", at(&m, x, y));
        }
        let brow = ((p[105][0] + p[107][0]) / 2.0, (p[105][1] + p[107][1]) / 2.0);
        assert!(
            at(&m, brow.0, brow.1) < 100,
            "brow {}",
            at(&m, brow.0, brow.1)
        );
        // Outside the oval: the ears (far left of the cheek point 234) and the background.
        assert_eq!(at(&m, p[234][0] - 30.0, p[234][1]), 0);
        assert_eq!(at(&m, p[454][0] + 30.0, p[454][1]), 0);
        // The forehead band: above the top of the oval but inside the band the skin goes on.
        let above = p[10][1] - 0.05 * g.height_px();
        assert!(at(&m, p[10][0], above) > 150, "{}", at(&m, p[10][0], above));
        // And stops at the band's end.
        assert_eq!(at(&m, p[10][0], p[10][1] - 0.25 * g.height_px()), 0);
    }

    #[test]
    fn no_skin_class_means_no_skin_and_a_face_outside_the_picture_means_none() {
        let g = flat_geometry(300.0, 50.0, 40.0);
        let img = picture(400, 380, |_, _| SKIN);
        let mut class = all_skin(400, 380);
        class.probs = vec![0.0; 256];
        assert!(skin_mask(&img, &g, &class).expect("mask").is_none());
        let far = flat_geometry(100.0, 2000.0, 2000.0);
        assert!(
            skin_mask(&img, &far, &all_skin(400, 380))
                .expect("mask")
                .is_none()
        );
        assert!(eyes_mask(&far, (400, 380)).expect("mask").is_none());
    }

    #[test]
    fn the_colour_trim_takes_a_collar_out_of_the_edge_but_not_a_spot_inside() {
        let g = flat_geometry(300.0, 50.0, 40.0);
        let p = g.points().to_vec();
        let chin_y = p[152][1];
        let blue = [0.05f32, 0.08, 0.30];
        // A blue collar across the bottom of the oval (below 92 % of the way down), and a red spot on the cheek.
        let (sx, sy) = (p[205][0] + 12.0, p[205][1]);
        let with = picture(400, 380, |x, y| {
            let (fx, fy) = (f64::from(x) + 0.5, f64::from(y) + 0.5);
            if fy > chin_y - 20.0 {
                blue
            } else if (fx - sx).hypot(fy - sy) < 5.0 {
                [0.6, 0.12, 0.10]
            } else {
                SKIN
            }
        });
        let without = picture(400, 380, |_, _| SKIN);
        let class = all_skin(400, 380);
        let a = skin_mask(&with, &g, &class).expect("mask").expect("skin");
        let b = skin_mask(&without, &g, &class)
            .expect("mask")
            .expect("skin");
        let (x, y) = (p[152][0], chin_y - 10.0);
        assert!(at(&b, x, y) > 150, "{}", at(&b, x, y));
        assert!(
            at(&a, x, y) < at(&b, x, y) / 2,
            "collar kept: {} against {}",
            at(&a, x, y),
            at(&b, x, y)
        );
        // The spot is inside the face, far from the edge: the trim leaves it (the guided filter may soften it a little).
        assert!(at(&a, sx, sy) > 200, "spot {}", at(&a, sx, sy));
    }

    #[test]
    fn the_guided_filter_pulls_the_edge_of_the_skin_to_the_edge_of_the_picture() {
        // The picture's skin (lit) ends at a straight cut inside the oval, with a darker surface of the same hue to its left;
        // the segmenter's probability falls over 8 pixels (the soft edge of a 256-pixel map) around the cut and does not
        // know where the edge is.
        let g = flat_geometry(300.0, 50.0, 40.0);
        let p = g.points().to_vec();
        let cut_x = p[205][0] - 10.0;
        let img = picture(400, 380, |x, _| {
            let lit = if f64::from(x) + 0.5 < cut_x {
                0.35
            } else {
                1.0
            };
            SKIN.map(|c| c * lit)
        });
        let size = 100usize;
        let probs: Vec<f32> = (0..size * size)
            .map(|i| {
                let x = (i % size) as f64 / size as f64 * 400.0;
                ((x - (cut_x - 4.0)) / 8.0).clamp(0.0, 1.0) as f32
            })
            .collect();
        let class = SkinClass {
            center_px: [200.0, 190.0],
            side_px: 400.0,
            rotation_rad: 0.0,
            size,
            probs,
        };
        let m = skin_mask(&img, &g, &class).expect("mask").expect("skin");
        let y = p[205][1] + 25.0;
        let (left, right) = (at(&m, cut_x - 3.0, y), at(&m, cut_x + 3.0, y));
        // The ramp alone would give 0.125 and 0.875 there (32 and 223).
        assert!(left < 20, "left of the cut {left}");
        assert!(right > 205, "right of the cut {right}");
        // A step at the picture's edge: across two pixels the mask rises by more than the ramp (64) would give.
        let step = i32::from(at(&m, cut_x + 1.0, y)) - i32::from(at(&m, cut_x - 1.0, y));
        assert!(step > 110, "step {step}");
    }

    #[test]
    fn eyes_and_mouth_masks() {
        let g = flat_geometry(300.0, 50.0, 40.0);
        let p = g.points();
        let eyes = eyes_mask(&g, (400, 380)).expect("mask").expect("eyes");
        for (a, b) in [(159usize, 145usize), (386, 374)] {
            let (x, y) = ((p[a][0] + p[b][0]) / 2.0, (p[a][1] + p[b][1]) / 2.0);
            assert!(at(&eyes, x, y) > 250, "{}", at(&eyes, x, y));
        }
        assert_eq!(at(&eyes, p[205][0], p[205][1]), 0);
        assert_eq!(
            at(
                &eyes,
                (p[133][0] + p[362][0]) / 2.0,
                (p[133][1] + p[362][1]) / 2.0
            ),
            0,
            "between the eyes"
        );
        // The synthetic face has its lips apart by the canonical layout's own distance between the inner contours.
        let mouth = mouth_interior_mask(&g, (400, 380));
        let open = mouth.expect("mask");
        if let Some(m) = open {
            assert!(at(&m, (p[13][0] + p[14][0]) / 2.0, (p[13][1] + p[14][1]) / 2.0) > 200);
            assert_eq!(at(&m, p[205][0], p[205][1]), 0);
        }
        // Closed lips: bring the lower inner contour up to the upper one.
        let mut closed = g.points().to_vec();
        for &i in &[95usize, 88, 178, 87, 14, 317, 402, 318, 324] {
            closed[i][1] = closed[13][1];
        }
        for &i in &[82usize, 81, 80, 191, 415, 310, 311, 312, 13, 78, 308] {
            closed[i][1] = closed[13][1];
        }
        let closed = FaceGeometry::new(closed).expect("geometry");
        assert!(
            mouth_interior_mask(&closed, (400, 380))
                .expect("mask")
                .is_none()
        );
    }

    #[test]
    fn the_mesh_tables_are_drawn_at_the_landmarks() {
        let g = flat_geometry(300.0, 50.0, 40.0);
        let p = g.points();
        let under = mesh_table_mask(&g, MeshTable::UnderEye, (400, 380)).expect("mask");
        let non_uniform =
            mesh_table_mask(&g, MeshTable::InvertedSkinUniformity, (400, 380)).expect("mask");
        // Below the lower lid, the crescent; at the cheek, nothing. The nostril is non-uniform, the cheek is not.
        let below = (p[145][0], p[145][1] + 0.03 * 300.0);
        assert!(
            at(&under, below.0, below.1) > 200,
            "{}",
            at(&under, below.0, below.1)
        );
        assert_eq!(at(&under, p[205][0], p[205][1]), 0);
        assert!(at(&non_uniform, p[98][0], p[98][1]) > 200);
        assert!(at(&non_uniform, p[205][0], p[205][1]) < 25);
        let far = flat_geometry(100.0, 2000.0, 2000.0);
        let none = mesh_table_mask(&far, MeshTable::UnderEye, (400, 380)).expect("mask");
        assert!(none.mask.is_all_zero());
    }

    #[test]
    fn the_masks_do_not_depend_on_the_thread_count() {
        let g = flat_geometry(300.0, 50.0, 40.0);
        let img = picture(400, 380, |x, y| {
            [
                SKIN[0] + (x % 7) as f32 * 0.01,
                SKIN[1],
                SKIN[2] + (y % 5) as f32 * 0.01,
            ]
        });
        let class = all_skin(400, 380);
        let one = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("pool");
        let four = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .expect("pool");
        let run = || {
            (
                skin_mask(&img, &g, &class).expect("mask"),
                mesh_table_mask(&g, MeshTable::UnderEye, (400, 380)).expect("mask"),
            )
        };
        assert_eq!(one.install(run), four.install(run));
    }
}
