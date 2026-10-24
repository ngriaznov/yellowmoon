//! The source list of L3: the highlights of the lens plane as compact lights (`docs/21` section 4, L3).
//!
//! A pixel of the lens plane belongs to a source when its highlight `H` reaches [`SOURCE_FLOOR`] clip levels in
//! some channel. The sources are the 8-connected components of those pixels; a component larger than
//! [`SOURCE_CELL_HD`] half-diagonals is split into cells of that size on a grid fixed to the plane. Each source
//! has its energy per channel, its centroid and its second moments; the list holds the [`SOURCES_MAX`] brightest,
//! ties by position, in that order.
//!
//! Everything is computed in a fixed order in `f64`.

use std::collections::HashMap;

use crate::consts::{SOURCE_CELL_HD, SOURCE_FLOOR, SOURCES_MAX};
use crate::error::LensError;
use crate::image::ImageRef;
use crate::l3::geometry::Frame;
use crate::l3::restore::Restore;
use crate::plan::Axis;

/// One compact light, in canvas coordinates (`along`, `across`) and half-diagonal units.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Source {
    /// The centroid.
    pub(crate) pos: [f64; 2],
    /// The highlights summed over the source, per channel, times the area of a plane pixel in half-diagonals
    /// squared: the light the source sends into its ghosts and streak.
    pub(crate) energy: [f64; 3],
    /// The second moment along the axis about the centroid, in half-diagonals squared, including the footprint
    /// of a plane pixel.
    pub(crate) var_along: f64,
    /// The second moment across the axis.
    pub(crate) var_across: f64,
    /// The larger side of the source's bounding box, in half-diagonals.
    pub(crate) extent: f64,
}

impl Source {
    /// The source's total energy over the channels.
    pub(crate) fn total(&self) -> f64 {
        self.energy[0] + self.energy[1] + self.energy[2]
    }
}

/// Union-find with path halving; the root of a set is its smallest member, so labels follow the raster order.
struct Dsu {
    parent: Vec<u32>,
}

impl Dsu {
    fn find(&mut self, mut a: u32) -> u32 {
        while self.parent[a as usize] != a {
            let p = self.parent[a as usize];
            self.parent[a as usize] = self.parent[p as usize];
            a = self.parent[a as usize];
        }
        a
    }

    fn union(&mut self, a: u32, b: u32) {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra != rb {
            // The smaller index becomes the root.
            let (lo, hi) = if ra < rb { (ra, rb) } else { (rb, ra) };
            self.parent[hi as usize] = lo;
        }
    }
}

/// What a cell of a component has gathered.
#[derive(Clone, Copy)]
struct Acc {
    weight: f64,
    energy: [f64; 3],
    sx: f64,
    sy: f64,
    sxx: f64,
    syy: f64,
    x0: usize,
    x1: usize,
    y0: usize,
    y1: usize,
}

impl Acc {
    fn new(x: usize, y: usize) -> Self {
        Acc {
            weight: 0.0,
            energy: [0.0; 3],
            sx: 0.0,
            sy: 0.0,
            sxx: 0.0,
            syy: 0.0,
            x0: x,
            x1: x,
            y0: y,
            y1: y,
        }
    }
}

/// The highlights of one pixel of the plane, as numbers.
fn highlight_at(plane: &ImageRef, rs: &Restore, i: usize, j: usize) -> [f64; 3] {
    let n = plane.width * plane.height;
    let at = j * plane.width + i;
    let x = [
        f64::from(plane.data[at]),
        f64::from(plane.data[n + at]),
        f64::from(plane.data[2 * n + at]),
    ];
    rs.highlight(x)
}

/// The sources of the lens plane `plane` (the whole frame at the lens plane's scale).
///
/// `cancel` is polled once per row of the plane.
pub(crate) fn find(
    plane: &ImageRef,
    rs: &Restore,
    axis: Axis,
    cancel: &dyn Fn() -> bool,
) -> Result<Vec<Source>, LensError> {
    let (w, h) = (plane.width, plane.height);
    let frame = Frame::new(w, h, axis);
    let half_diag = frame.half_diag;

    // The mask of source pixels.
    let mut mask = vec![false; w * h];
    let mut any = false;
    for j in 0..h {
        if cancel() {
            return Err(LensError::Cancelled);
        }
        for i in 0..w {
            let hl = highlight_at(plane, rs, i, j);
            if rs.level(hl) >= SOURCE_FLOOR {
                mask[j * w + i] = true;
                any = true;
            }
        }
    }
    if !any {
        return Ok(Vec::new());
    }

    // 8-connected components, labelled by their first pixel in raster order.
    let mut dsu = Dsu {
        parent: (0..(w * h) as u32).collect(),
    };
    for j in 0..h {
        for i in 0..w {
            let p = j * w + i;
            if !mask[p] {
                continue;
            }
            let p = p as u32;
            if i > 0 && mask[j * w + i - 1] {
                dsu.union(p, p - 1);
            }
            if j > 0 {
                let up = (j - 1) * w + i;
                if mask[up] {
                    dsu.union(p, up as u32);
                }
                if i > 0 && mask[up - 1] {
                    dsu.union(p, (up - 1) as u32);
                }
                if i + 1 < w && mask[up + 1] {
                    dsu.union(p, (up + 1) as u32);
                }
            }
        }
    }
    if cancel() {
        return Err(LensError::Cancelled);
    }
    // The bounding box of each component, by its root.
    let mut bbox: HashMap<u32, (usize, usize, usize, usize)> = HashMap::new();
    let mut label = vec![0u32; w * h];
    for j in 0..h {
        for i in 0..w {
            let p = j * w + i;
            if !mask[p] {
                continue;
            }
            let root = dsu.find(p as u32);
            label[p] = root;
            let e = bbox.entry(root).or_insert((i, i, j, j));
            e.0 = e.0.min(i);
            e.1 = e.1.max(i);
            e.2 = e.2.min(j);
            e.3 = e.3.max(j);
        }
    }
    let cell_px = (SOURCE_CELL_HD * half_diag).max(1.0);
    let too_large: HashMap<u32, bool> = bbox
        .iter()
        .map(|(root, (x0, x1, y0, y1))| {
            let extent = ((x1 - x0 + 1).max(y1 - y0 + 1)) as f64 / half_diag;
            (*root, extent > SOURCE_CELL_HD)
        })
        .collect();

    // Sums per (component, cell), in raster order; `slots` keeps the order of first appearance.
    type Key = (u32, i64, i64);
    let mut index: HashMap<Key, usize> = HashMap::new();
    let mut slots: Vec<Acc> = Vec::new();
    let mut last: Option<(Key, usize)> = None;
    for j in 0..h {
        if cancel() {
            return Err(LensError::Cancelled);
        }
        for i in 0..w {
            let p = j * w + i;
            if !mask[p] {
                continue;
            }
            let root = label[p];
            let key: Key = if too_large.get(&root).copied().unwrap_or(false) {
                (
                    root,
                    libm::floor(i as f64 / cell_px) as i64,
                    libm::floor(j as f64 / cell_px) as i64,
                )
            } else {
                (root, 0, 0)
            };
            let slot = match last {
                Some((k, s)) if k == key => s,
                _ => {
                    let s = *index.entry(key).or_insert_with(|| {
                        slots.push(Acc::new(i, j));
                        slots.len() - 1
                    });
                    last = Some((key, s));
                    s
                }
            };
            let hl = highlight_at(plane, rs, i, j);
            let weight = hl[0] + hl[1] + hl[2];
            let (x, y) = (i as f64 + 0.5, j as f64 + 0.5);
            let acc = &mut slots[slot];
            acc.weight += weight;
            for (e, h) in acc.energy.iter_mut().zip(&hl) {
                *e += h;
            }
            acc.sx += weight * x;
            acc.sy += weight * y;
            acc.sxx += weight * x * x;
            acc.syy += weight * y * y;
            acc.x0 = acc.x0.min(i);
            acc.x1 = acc.x1.max(i);
            acc.y0 = acc.y0.min(j);
            acc.y1 = acc.y1.max(j);
        }
    }

    // The sources, brightest first; equal energies keep the raster order of the cells' first pixels.
    let area = 1.0 / (half_diag * half_diag);
    let mut sources: Vec<(Source, f64, f64)> = slots
        .iter()
        .filter(|a| a.weight > 0.0)
        .map(|a| {
            let (cx, cy) = (a.sx / a.weight, a.sy / a.weight);
            let var_x = (a.sxx / a.weight - cx * cx).max(0.0) + 1.0 / 12.0;
            let var_y = (a.syy / a.weight - cy * cy).max(0.0) + 1.0 / 12.0;
            let (along, across) = frame.canvas_at(cx, cy);
            let (var_along, var_across) = match axis {
                Axis::X => (var_x, var_y),
                Axis::Y => (var_y, var_x),
            };
            let extent = ((a.x1 - a.x0 + 1).max(a.y1 - a.y0 + 1)) as f64 / half_diag;
            let source = Source {
                pos: [along, across],
                energy: [a.energy[0] * area, a.energy[1] * area, a.energy[2] * area],
                var_along: var_along * area,
                var_across: var_across * area,
                extent,
            };
            (source, cy, cx)
        })
        .collect();
    sources.sort_by(|a, b| {
        b.0.total()
            .total_cmp(&a.0.total())
            .then(a.1.total_cmp(&b.1))
            .then(a.2.total_cmp(&b.2))
    });
    sources.truncate(SOURCES_MAX);
    Ok(sources.into_iter().map(|(s, _, _)| s).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::amounts::{LensAmounts, LensKind};
    use crate::plan::{FrameInfo, LensPlan};
    use crate::spectral::CameraResponse;
    use crate::test_support::write_synthetic_lenses;

    fn plan(w: usize, h: usize, axis: Axis) -> LensPlan {
        let dir = tempfile::tempdir().unwrap();
        write_synthetic_lenses(dir.path()).unwrap();
        let lib = crate::data::LensLibrary::from_dir(dir.path()).unwrap();
        let amounts = LensAmounts {
            kind: LensKind::Classic,
            flares: 100.0,
            edges: 0.0,
            anamorphic: 0.0,
        };
        let frame = FrameInfo {
            width: w,
            height: h,
            axis,
            f_number: None,
            clip: [1.0; 3],
        };
        LensPlan::new(&lib, &amounts, frame, &CameraResponse::neutral()).unwrap()
    }

    fn scene(w: usize, h: usize, lights: &[(usize, usize, usize, usize, f32)]) -> Vec<f32> {
        let mut data = vec![0.02f32; 3 * w * h];
        for (x0, y0, lw, lh, v) in lights {
            for c in 0..3 {
                for y in *y0..y0 + lh {
                    for x in *x0..x0 + lw {
                        data[c * w * h + y * w + x] = *v;
                    }
                }
            }
        }
        data
    }

    fn image(w: usize, h: usize, data: &[f32]) -> ImageRef<'_> {
        ImageRef {
            width: w,
            height: h,
            x0: 0,
            y0: 0,
            frame_width: w,
            frame_height: h,
            data,
        }
    }

    #[test]
    fn a_clipped_patch_is_one_source_with_the_restored_energy() {
        let (w, h) = (600, 400);
        let data = scene(w, h, &[(450, 120, 3, 3, 1.0)]);
        let p = plan(w, h, Axis::X);
        let rs = Restore::new(&p, crate::consts::CLIP_RAMP_PLANE);
        let list = find(&image(w, h, &data), &rs, Axis::X, &|| false).unwrap();
        assert_eq!(list.len(), 1);
        let s = list[0];
        let frame = Frame::new(w, h, Axis::X);
        // Nine pixels at 1.0 times 2^3, each 1 / H^2 of a half-diagonal squared.
        let want = 9.0 * 8.0 / (frame.half_diag * frame.half_diag);
        for c in 0..3 {
            assert!((s.energy[c] - want).abs() < 1e-12 * want, "{:?}", s.energy);
        }
        // The centroid is the patch's centre: pixel 451.5, 121.5.
        assert!((s.pos[0] - frame.u(451.5)).abs() < 1e-12);
        assert!((s.pos[1] - frame.v(121.5)).abs() < 1e-12);
        assert!((s.extent - 3.0 / frame.half_diag).abs() < 1e-12);
        // A 3 x 3 patch: variance 2/3 + 1/12 of a pixel.
        let var = (2.0 / 3.0 + 1.0 / 12.0) / (frame.half_diag * frame.half_diag);
        assert!((s.var_along - var).abs() < 1e-12 && (s.var_across - var).abs() < 1e-12);
    }

    #[test]
    fn the_axis_decides_which_coordinate_is_along() {
        let (w, h) = (600, 400);
        let data = scene(w, h, &[(450, 120, 3, 3, 1.0)]);
        let rs = Restore::new(&plan(w, h, Axis::Y), crate::consts::CLIP_RAMP_PLANE);
        let list = find(&image(w, h, &data), &rs, Axis::Y, &|| false).unwrap();
        let frame = Frame::new(w, h, Axis::Y);
        assert!((list[0].pos[0] - frame.v(121.5)).abs() < 1e-12);
        assert!((list[0].pos[1] - frame.u(451.5)).abs() < 1e-12);
    }

    #[test]
    fn separate_patches_are_separate_sources_and_touching_ones_are_one() {
        let (w, h) = (600, 400);
        let data = scene(
            w,
            h,
            &[
                (20, 20, 2, 2, 1.0),
                (100, 60, 2, 2, 0.8),
                // Touches the patch at (100, 60) only at a corner: 8-connected.
                (102, 62, 2, 2, 0.8),
            ],
        );
        let rs = Restore::new(&plan(w, h, Axis::X), crate::consts::CLIP_RAMP_PLANE);
        let list = find(&image(w, h, &data), &rs, Axis::X, &|| false).unwrap();
        assert_eq!(list.len(), 2);
        // The clipped (restored) patch is the brighter one and comes first.
        assert!(list[0].total() > list[1].total());
        let frame = Frame::new(w, h, Axis::X);
        assert!((list[0].pos[0] - frame.u(21.0)).abs() < 1e-12);
    }

    #[test]
    fn a_large_component_is_split_into_cells_and_the_list_is_capped() {
        let (w, h) = (600, 400);
        let p = plan(w, h, Axis::X);
        let rs = Restore::new(&p, crate::consts::CLIP_RAMP_PLANE);
        // One big clipped block: 400 x 200 px, far larger than 0.02 half-diagonals (about 14 px here).
        let data = scene(w, h, &[(100, 100, 400, 200, 1.0)]);
        let list = find(&image(w, h, &data), &rs, Axis::X, &|| false).unwrap();
        assert!(
            list.len() > 100 && list.len() <= SOURCES_MAX,
            "{}",
            list.len()
        );
        let frame = Frame::new(w, h, Axis::X);
        let cell = SOURCE_CELL_HD * frame.half_diag;
        for s in &list {
            assert!(s.extent <= cell / frame.half_diag + 1.0 / frame.half_diag);
        }
        // Sorted by energy.
        assert!(list.windows(2).all(|p| p[0].total() >= p[1].total()));
        // Cancel is honoured.
        assert!(matches!(
            find(&image(w, h, &data), &rs, Axis::X, &|| true),
            Err(LensError::Cancelled)
        ));
    }

    #[test]
    fn a_dark_plane_has_no_sources() {
        let (w, h) = (100, 80);
        let data = scene(w, h, &[]);
        let rs = Restore::new(&plan(w, h, Axis::X), crate::consts::CLIP_RAMP_PLANE);
        assert!(
            find(&image(w, h, &data), &rs, Axis::X, &|| false)
                .unwrap()
                .is_empty()
        );
    }
}
