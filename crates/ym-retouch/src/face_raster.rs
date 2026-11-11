//! Rasterisers for the face geometry (`docs/27` S-09): polygons with exact horizontal coverage and mesh triangles with a
//! texture.
//!
//! Coordinates are continuous image coordinates: pixel `(i, j)` covers `[i, i + 1) x [j, j + 1)`. Every function is a pure
//! function of its arguments; rows are processed in parallel and write disjoint rows, so the result does not depend on the
//! thread count.
//!
//! * [`fill_polygon`] fills a simple polygon (even-odd) and returns the coverage of every pixel, `0..1`: eight sub-scanlines
//!   per row and the exact overlap of each span with the pixel, so an edge is anti-aliased to a ninth of a step vertically
//!   and exactly horizontally.
//! * [`render_mesh`] draws the triangles of the canonical face mesh with the texture coordinates of the model: each pixel
//!   takes the value of a [`Texture`] at its interpolated `(u, v)`, weighted by the share of the pixel the triangle covers
//!   (a 4 x 4 grid of samples on the pixels the edges cross, the whole pixel elsewhere). The shares of neighbouring triangles
//!   add up to one inside the mesh, so there are no seams; the outer border of the mesh is anti-aliased.

use rayon::prelude::*;

use crate::error::Result;
use crate::face_topology::{TRIANGLES, UV, VERTICES};
use crate::types::{BoundedMask, Mask, Rect};

/// Sub-scanlines per pixel row of [`fill_polygon`].
const SUB: usize = 8;
/// Samples per side of the coverage grid on the pixels an edge of a triangle crosses.
const GRID: usize = 4;
/// Rows of a band of [`render_mesh`].
const BAND_ROWS: usize = 32;

/// The sign of the doubled signed area `(b - a) x (c - a)` (picture coordinates, y down) of the triangles that face the
/// camera in the canonical winding. Triangles with the other sign are back faces and are not drawn.
pub const FRONT_FACING_SIGN: f64 = -1.0;

/// Coverage `0..1` of the polygon `poly` (even-odd rule) on the pixels of `region`, row-major, `width * height` values.
///
/// A polygon with fewer than three points covers nothing.
pub fn fill_polygon(poly: &[[f64; 2]], region: Rect) -> Vec<f32> {
    let (w, h) = (region.width_px as usize, region.height_px as usize);
    let mut out = vec![0.0f32; w * h];
    if poly.len() < 3 || w == 0 || h == 0 {
        return out;
    }
    let (ox, oy) = (f64::from(region.x_px), f64::from(region.y_px));
    // Edges as (x0, y0, x1, y1) with y0 < y1 in region coordinates, plus their y range for a quick reject.
    let mut edges = Vec::with_capacity(poly.len());
    for i in 0..poly.len() {
        let (a, b) = (poly[i], poly[(i + 1) % poly.len()]);
        let (a, b) = ([a[0] - ox, a[1] - oy], [b[0] - ox, b[1] - oy]);
        if a[1] == b[1] {
            continue;
        }
        let (lo, hi) = if a[1] < b[1] { (a, b) } else { (b, a) };
        edges.push((lo[0], lo[1], hi[0], hi[1]));
    }
    let step = 1.0 / SUB as f64;
    out.par_chunks_mut(w).enumerate().for_each(|(row, line)| {
        let mut direct = vec![0.0f32; w];
        let mut diff = vec![0.0f32; w + 1];
        let mut xs: Vec<f64> = Vec::with_capacity(edges.len());
        let top = row as f64;
        if !edges.iter().any(|e| e.1 < top + 1.0 && e.3 > top) {
            return;
        }
        for s in 0..SUB {
            let y = top + (s as f64 + 0.5) * step;
            xs.clear();
            for &(x0, y0, x1, y1) in &edges {
                if y0 <= y && y < y1 {
                    xs.push(x0 + (y - y0) * (x1 - x0) / (y1 - y0));
                }
            }
            xs.sort_by(f64::total_cmp);
            for pair in xs.chunks_exact(2) {
                add_span(&mut direct, &mut diff, pair[0], pair[1], step as f32);
            }
        }
        let mut run = 0.0f32;
        for (x, v) in line.iter_mut().enumerate() {
            run += diff[x];
            *v = (direct[x] + run).clamp(0.0, 1.0);
        }
    });
    out
}

/// Adds the exact overlap of the span `[x0, x1)` with every pixel, times `weight`: the two end pixels directly, the pixels
/// between them through the difference array.
fn add_span(direct: &mut [f32], diff: &mut [f32], x0: f64, x1: f64, weight: f32) {
    let w = direct.len() as f64;
    let (x0, x1) = (x0.clamp(0.0, w), x1.clamp(0.0, w));
    if x1 <= x0 {
        return;
    }
    let (i0, i1) = (x0.floor() as usize, x1.floor() as usize);
    if i0 == i1 {
        direct[i0.min(direct.len() - 1)] += (x1 - x0) as f32 * weight;
        return;
    }
    direct[i0] += ((i0 + 1) as f64 - x0) as f32 * weight;
    if i1 < direct.len() {
        direct[i1] += (x1 - i1 as f64) as f32 * weight;
    }
    // Whole pixels i0 + 1 .. i1.
    diff[i0 + 1] += weight;
    diff[i1] -= weight;
}

/// The polygon with every vertex moved `d` pixels along the outward bisector of its two edges (`d` negative shrinks it).
/// A polygon of fewer than three points, or with a repeated point, is returned as it is.
pub fn offset_polygon(poly: &[[f64; 2]], d: f64) -> Vec<[f64; 2]> {
    let n = poly.len();
    if n < 3 || d == 0.0 {
        return poly.to_vec();
    }
    // Orientation by the signed area: positive in picture coordinates (y down) is clockwise on screen.
    let area: f64 = (0..n)
        .map(|i| {
            let (a, b) = (poly[i], poly[(i + 1) % n]);
            a[0] * b[1] - b[0] * a[1]
        })
        .sum();
    let sign = if area >= 0.0 { 1.0 } else { -1.0 };
    let normal = |a: [f64; 2], b: [f64; 2]| -> [f64; 2] {
        let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
        let len = dx.hypot(dy);
        if len == 0.0 {
            return [0.0, 0.0];
        }
        // Outward for a positive area (y down): the right-hand normal (dy, -dx).
        [sign * dy / len, -sign * dx / len]
    };
    (0..n)
        .map(|i| {
            let (p, q, r) = (poly[(i + n - 1) % n], poly[i], poly[(i + 1) % n]);
            let (n0, n1) = (normal(p, q), normal(q, r));
            let bis = [n0[0] + n1[0], n0[1] + n1[1]];
            let len2 = bis[0] * bis[0] + bis[1] * bis[1];
            if len2 < 1e-12 {
                return q;
            }
            // Miter length 1 / cos(half angle) = 2 / |n0 + n1|, limited to 2.
            let scale = (2.0 / len2.sqrt()).min(2.0) / len2.sqrt() * d;
            [q[0] + bis[0] * scale, q[1] + bis[1] * scale]
        })
        .collect()
}

/// The centre of the bounding box of `poly`.
pub fn polygon_center(poly: &[[f64; 2]]) -> [f64; 2] {
    let mut b = [f64::MAX, f64::MAX, f64::MIN, f64::MIN];
    for p in poly {
        b = [
            b[0].min(p[0]),
            b[1].min(p[1]),
            b[2].max(p[0]),
            b[3].max(p[1]),
        ];
    }
    [(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0]
}

/// The smallest pixel rectangle that holds all `points`, grown by `margin_px` and clipped to an image of `size_px`; `None`
/// when nothing of it is inside the image.
pub fn region_of(points: &[[f64; 2]], margin_px: f64, size_px: (u32, u32)) -> Option<Rect> {
    if points.is_empty()
        || points
            .iter()
            .any(|p| !p[0].is_finite() || !p[1].is_finite())
    {
        return None;
    }
    let mut b = [f64::MAX, f64::MAX, f64::MIN, f64::MIN];
    for p in points {
        b = [
            b[0].min(p[0]),
            b[1].min(p[1]),
            b[2].max(p[0]),
            b[3].max(p[1]),
        ];
    }
    let x0 = ((b[0] - margin_px).floor()).max(0.0);
    let y0 = ((b[1] - margin_px).floor()).max(0.0);
    let x1 = ((b[2] + margin_px).ceil()).min(f64::from(size_px.0));
    let y1 = ((b[3] + margin_px).ceil()).min(f64::from(size_px.1));
    if x1 <= x0 || y1 <= y0 {
        return None;
    }
    Some(Rect::new(
        x0 as i32,
        y0 as i32,
        (x1 - x0) as u32,
        (y1 - y0) as u32,
    ))
}

/// A square table of values `0..1` over `(u, v)` in `0..1`, sampled bilinearly between texel centres.
#[derive(Debug, Clone, PartialEq)]
pub struct Texture {
    size: usize,
    data: Vec<f32>,
}

impl Texture {
    /// A texture of `size x size` texels, each the value of `f` at the texel's centre `(u, v)`.
    pub fn from_fn(size: usize, f: impl Fn(f64, f64) -> f64) -> Self {
        let size = size.max(2);
        let mut data = Vec::with_capacity(size * size);
        for j in 0..size {
            for i in 0..size {
                let (u, v) = (
                    (i as f64 + 0.5) / size as f64,
                    (j as f64 + 0.5) / size as f64,
                );
                data.push(f(u, v).clamp(0.0, 1.0) as f32);
            }
        }
        Self { size, data }
    }

    /// The side in texels.
    pub fn size(&self) -> usize {
        self.size
    }

    /// The value at `(u, v)`, held to the edge outside `0..1`.
    pub fn sample(&self, u: f64, v: f64) -> f32 {
        let n = self.size as f64;
        let (x, y) = (u * n - 0.5, v * n - 0.5);
        let (fx, fy) = (x.floor(), y.floor());
        let (tx, ty) = ((x - fx) as f32, (y - fy) as f32);
        let at = |xx: i64, yy: i64| {
            let xx = xx.clamp(0, self.size as i64 - 1) as usize;
            let yy = yy.clamp(0, self.size as i64 - 1) as usize;
            self.data[yy * self.size + xx]
        };
        let (x0, y0) = (fx as i64, fy as i64);
        let top = at(x0, y0) * (1.0 - tx) + at(x0 + 1, y0) * tx;
        let bottom = at(x0, y0 + 1) * (1.0 - tx) + at(x0 + 1, y0 + 1) * tx;
        top * (1.0 - ty) + bottom * ty
    }
}

/// One triangle ready to draw.
struct Prepared {
    /// Vertices in image coordinates.
    p: [[f64; 2]; 3],
    /// Texture coordinates of the vertices.
    uv: [[f64; 2]; 3],
    /// The doubled signed area (the sign of the facing).
    area2: f64,
    /// Rows of the image the triangle touches: first and one past the last.
    rows: (i64, i64),
    /// Columns: first and one past the last.
    cols: (i64, i64),
}

fn edge(a: [f64; 2], b: [f64; 2], p: [f64; 2]) -> f64 {
    (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
}

impl Prepared {
    /// The barycentric weights of `p`, or `None` when it is outside.
    fn weights(&self, p: [f64; 2]) -> Option<[f64; 3]> {
        let [a, b, c] = self.p;
        let s = self.area2.signum();
        let (w0, w1, w2) = (s * edge(b, c, p), s * edge(c, a, p), s * edge(a, b, p));
        if w0 >= 0.0 && w1 >= 0.0 && w2 >= 0.0 {
            let k = 1.0 / (w0 + w1 + w2);
            Some([w0 * k, w1 * k, w2 * k])
        } else {
            None
        }
    }

    fn uv_at(&self, w: [f64; 3]) -> [f64; 2] {
        [
            w[0] * self.uv[0][0] + w[1] * self.uv[1][0] + w[2] * self.uv[2][0],
            w[0] * self.uv[0][1] + w[1] * self.uv[1][1] + w[2] * self.uv[2][1],
        ]
    }

    /// Adds the triangle's share of the pixel `(x, y)` times the texture value to `acc`.
    fn shade(&self, x: i64, y: i64, tex: &Texture, acc: &mut f32) {
        let (fx, fy) = (x as f64, y as f64);
        let corners = [
            [fx, fy],
            [fx + 1.0, fy],
            [fx, fy + 1.0],
            [fx + 1.0, fy + 1.0],
        ];
        if corners.iter().all(|&c| self.weights(c).is_some()) {
            if let Some(w) = self.weights([fx + 0.5, fy + 0.5]) {
                let uv = self.uv_at(w);
                *acc += tex.sample(uv[0], uv[1]);
            }
            return;
        }
        let mut count = 0u32;
        let mut sum_uv = [0.0f64; 2];
        for j in 0..GRID {
            for i in 0..GRID {
                let p = [
                    fx + (i as f64 + 0.5) / GRID as f64,
                    fy + (j as f64 + 0.5) / GRID as f64,
                ];
                if let Some(w) = self.weights(p) {
                    let uv = self.uv_at(w);
                    sum_uv[0] += uv[0];
                    sum_uv[1] += uv[1];
                    count += 1;
                }
            }
        }
        if count > 0 {
            let share = count as f32 / (GRID * GRID) as f32;
            let n = f64::from(count);
            *acc += share * tex.sample(sum_uv[0] / n, sum_uv[1] / n);
        }
    }
}

/// Draws the mesh into a mask: `points` are the 468 vertices (at least) in image coordinates, `texture` gives the value at
/// each pixel's interpolated texture coordinates, `size_px` is the image size. Only triangles that face the camera are drawn.
/// The mask covers the part of the mesh inside the image; `None` when there is none.
///
/// # Errors
///
/// [`crate::RetouchError::Size`] never in practice; the error type is for the mask constructor.
pub fn render_mesh(
    points: &[[f64; 2]],
    texture: &Texture,
    size_px: (u32, u32),
) -> Result<Option<BoundedMask>> {
    if points.len() < VERTICES {
        return Ok(None);
    }
    let Some(region) = region_of(&points[..VERTICES], 1.0, size_px) else {
        return Ok(None);
    };
    let (w, h) = (region.width_px as usize, region.height_px as usize);
    let (ox, oy) = (f64::from(region.x_px), f64::from(region.y_px));
    let mut prepared = Vec::with_capacity(TRIANGLES.len());
    for tri in &TRIANGLES {
        let idx = tri.map(usize::from);
        let p = idx.map(|i| [points[i][0] - ox, points[i][1] - oy]);
        let area2 = edge(p[0], p[1], p[2]);
        if !area2.is_finite() || area2 * FRONT_FACING_SIGN <= 1e-9 {
            continue;
        }
        let uv = idx.map(|i| [f64::from(UV[i][0]), f64::from(UV[i][1])]);
        let min = |k: usize| p.iter().map(|q| q[k]).fold(f64::MAX, f64::min);
        let max = |k: usize| p.iter().map(|q| q[k]).fold(f64::MIN, f64::max);
        let cols = (
            min(0).floor().max(0.0) as i64,
            (max(0).ceil() as i64).min(w as i64),
        );
        let rows = (
            min(1).floor().max(0.0) as i64,
            (max(1).ceil() as i64).min(h as i64),
        );
        if cols.0 >= cols.1 || rows.0 >= rows.1 {
            continue;
        }
        prepared.push(Prepared {
            p,
            uv,
            area2,
            rows,
            cols,
        });
    }
    let mut acc = vec![0.0f32; w * h];
    acc.par_chunks_mut(w * BAND_ROWS)
        .enumerate()
        .for_each(|(band, chunk)| {
            let y_lo = (band * BAND_ROWS) as i64;
            let y_hi = y_lo + (chunk.len() / w) as i64;
            // Triangles in index order, so every pixel adds its shares in the same order on any thread count.
            for t in prepared
                .iter()
                .filter(|t| t.rows.0 < y_hi && t.rows.1 > y_lo)
            {
                for y in t.rows.0.max(y_lo)..t.rows.1.min(y_hi) {
                    let line = &mut chunk[(y - y_lo) as usize * w..][..w];
                    for x in t.cols.0..t.cols.1 {
                        t.shade(x, y, texture, &mut line[x as usize]);
                    }
                }
            }
        });
    let data: Vec<u8> = acc
        .iter()
        .map(|&v| (v * 255.0 + 0.5).clamp(0.0, 255.0) as u8)
        .collect();
    let mask = Mask::new(region.width_px, region.height_px, data)?;
    Ok(Some(BoundedMask::new(region, mask)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(cov: &[f32], w: usize, x: usize, y: usize) -> f32 {
        cov[y * w + x]
    }

    #[test]
    fn a_rectangle_covers_whole_pixels_and_fractions_at_its_edges() {
        let region = Rect::new(0, 0, 10, 8);
        let poly = [[2.5, 1.0], [6.0, 1.0], [6.0, 4.25], [2.5, 4.25]];
        let c = fill_polygon(&poly, region);
        assert!(
            (at(&c, 10, 2, 2) - 0.5).abs() < 1e-6,
            "left edge half covered"
        );
        assert!((at(&c, 10, 3, 2) - 1.0).abs() < 1e-6);
        assert!((at(&c, 10, 5, 2) - 1.0).abs() < 1e-6);
        assert_eq!(at(&c, 10, 6, 2), 0.0);
        assert!(
            (at(&c, 10, 4, 4) - 0.25).abs() < 1e-6,
            "bottom edge a quarter"
        );
        assert_eq!(at(&c, 10, 4, 5), 0.0);
        let total: f32 = c.iter().sum();
        assert!((total - 3.5 * 3.25).abs() < 1e-3, "area {total}");
    }

    #[test]
    fn a_triangle_and_a_region_that_cuts_it_keep_their_area() {
        let tri = [[1.0, 1.0], [9.0, 2.0], [4.0, 8.0]];
        let whole = fill_polygon(&tri, Rect::new(0, 0, 10, 10));
        let area = 0.5 * ((9.0 - 1.0) * (8.0 - 1.0) - (4.0 - 1.0) * (2.0 - 1.0));
        let total: f32 = whole.iter().sum();
        assert!(
            (f64::from(total) - area).abs() < 0.05,
            "{total} against {area}"
        );
        // The same polygon seen through a window: the window's pixels equal the whole's.
        let part = fill_polygon(&tri, Rect::new(3, 2, 4, 5));
        for y in 0..5 {
            for x in 0..4 {
                assert_eq!(part[y * 4 + x], whole[(y + 2) * 10 + x + 3]);
            }
        }
        assert!(
            fill_polygon(&tri[..2], Rect::new(0, 0, 4, 4))
                .iter()
                .all(|&v| v == 0.0)
        );
    }

    #[test]
    fn offsetting_grows_and_shrinks_a_square_by_the_distance() {
        let sq = [[10.0, 10.0], [20.0, 10.0], [20.0, 20.0], [10.0, 20.0]];
        for poly in [sq.to_vec(), sq.iter().rev().copied().collect::<Vec<_>>()] {
            let bigger = offset_polygon(&poly, 2.0);
            let area = |p: &[[f64; 2]]| {
                let r = Rect::new(0, 0, 40, 40);
                fill_polygon(p, r)
                    .iter()
                    .map(|&v| f64::from(v))
                    .sum::<f64>()
            };
            assert!((area(&bigger) - 196.0).abs() < 1.0, "{}", area(&bigger));
            assert!((area(&offset_polygon(&poly, -2.0)) - 36.0).abs() < 1.0);
        }
        assert_eq!(offset_polygon(&sq[..2], 3.0), sq[..2].to_vec());
    }

    #[test]
    fn regions_are_clipped_to_the_image() {
        let r = region_of(&[[-5.0, 3.2], [12.5, 9.9]], 1.0, (10, 10)).expect("region");
        assert_eq!((r.x_px, r.y_px, r.width_px, r.height_px), (0, 2, 10, 8));
        assert!(region_of(&[[20.0, 20.0]], 1.0, (10, 10)).is_none());
        assert!(region_of(&[[f64::NAN, 1.0]], 1.0, (10, 10)).is_none());
        assert!(region_of(&[], 1.0, (10, 10)).is_none());
    }

    #[test]
    fn a_texture_is_sampled_between_texel_centres() {
        let t = Texture::from_fn(4, |u, _| u);
        assert!((t.sample(0.5, 0.5) - 0.5).abs() < 1e-6);
        assert!((t.sample(0.125, 0.3) - 0.125).abs() < 1e-6);
        assert!(
            (t.sample(-1.0, 0.3) - 0.125).abs() < 1e-6,
            "held to the edge"
        );
        assert!((t.sample(2.0, 0.3) - 0.875).abs() < 1e-6);
    }

    /// A flat mesh: the 468 vertices on the canonical UV layout scaled to `size` pixels, so a constant texture is drawn
    /// where the triangles are and the picture of the texture is the atlas.
    fn flat_mesh(size: f64, x0: f64, y0: f64) -> Vec<[f64; 2]> {
        // The model's front faces have the sign FRONT_FACING_SIGN; the UV layout has it too (checked below).
        UV.iter()
            .map(|uv| [x0 + f64::from(uv[0]) * size, y0 + f64::from(uv[1]) * size])
            .chain(std::iter::repeat_n([x0, y0], 10))
            .collect()
    }

    #[test]
    fn the_mesh_draws_a_constant_texture_without_seams_and_with_a_soft_border() {
        let tex = Texture::from_fn(8, |_, _| 1.0);
        let pts = flat_mesh(200.0, 20.0, 15.0);
        let m = render_mesh(&pts, &tex, (260, 240))
            .expect("render")
            .expect("mask");
        let (w, h) = (m.mask.width_px as usize, m.mask.height_px as usize);
        assert!(w > 150 && h > 150, "{w} x {h}");
        // The centre of the face is solid; no pixel inside the mesh is a seam (a share below 255 only on the border).
        let centre = m.mask.data[(h / 2) * w + w / 2];
        assert_eq!(centre, 255);
        let partial = m.mask.data.iter().filter(|&&v| v > 0 && v < 255).count();
        let solid = m.mask.data.iter().filter(|&&v| v == 255).count();
        assert!(solid > 10_000, "{solid}");
        // Partial pixels are the border only: a small share of the filled area.
        assert!(partial * 8 < solid, "{partial} partial of {solid}");
        // Inside, away from the border, every pixel of the face oval's interior is solid (no seam along shared edges).
        let mut seams = 0;
        for y in h / 2 - 20..h / 2 + 20 {
            for x in w / 2 - 20..w / 2 + 20 {
                if m.mask.data[y * w + x] != 255 {
                    seams += 1;
                }
            }
        }
        assert_eq!(seams, 0);
    }

    #[test]
    fn the_mesh_reads_the_texture_at_the_uv_of_each_pixel_and_is_deterministic() {
        // A texture that is the u coordinate: the mask rises left to right across the face.
        let tex = Texture::from_fn(64, |u, _| u);
        let pts = flat_mesh(300.0, 10.0, 10.0);
        let a = render_mesh(&pts, &tex, (330, 330))
            .expect("render")
            .expect("mask");
        let b = render_mesh(&pts, &tex, (330, 330))
            .expect("render")
            .expect("mask");
        assert_eq!(a, b);
        let (w, h) = (a.mask.width_px as usize, a.mask.height_px as usize);
        let row = &a.mask.data[(h / 2) * w..][..w];
        let (x_left, x_right) = (w / 4, 3 * w / 4);
        assert!(
            row[x_right] > row[x_left] + 60,
            "{} {}",
            row[x_left],
            row[x_right]
        );
        // At the image position the u coordinate puts the value: u = (x - 10) / 300.
        let x = (a.region.x_px as usize) + w / 2;
        let want = ((x as f64 + 0.5 - 10.0) / 300.0 * 255.0).round() as i32;
        let got = i32::from(row[w / 2]);
        assert!((got - want).abs() <= 3, "{got} against {want}");
    }

    #[test]
    fn back_faces_a_missing_mesh_and_an_outside_mesh_draw_nothing() {
        let tex = Texture::from_fn(4, |_, _| 1.0);
        // Mirrored: every triangle flips its facing, so nothing is drawn.
        let mirrored: Vec<[f64; 2]> = flat_mesh(100.0, 0.0, 0.0)
            .iter()
            .map(|p| [100.0 - p[0], p[1]])
            .collect();
        let m = render_mesh(&mirrored, &tex, (120, 120)).expect("render");
        assert!(m.is_none() || m.expect("mask").mask.is_all_zero());
        assert!(
            render_mesh(&[[0.0, 0.0]; 5], &tex, (10, 10))
                .expect("render")
                .is_none()
        );
        let far = flat_mesh(100.0, 500.0, 500.0);
        assert!(
            render_mesh(&far, &tex, (120, 120))
                .expect("render")
                .is_none()
        );
    }
}
