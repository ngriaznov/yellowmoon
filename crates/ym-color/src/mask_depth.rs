//! The Depth mask's maths on the CPU reference (`docs/23-nodes.md` section 4.8, `docs/algorithms/06-masks.md`
//! section 5.8): the rank normalisation of a relative depth map, the gradient that makes a plane from it, Pick depth,
//! the Focus plane's median and the Show depth plane.
//!
//! A [`DepthMap`] is what the network (`ym_ml::depth`) makes once per photo, `d` (larger is nearer), with the
//! **mid-ranks** `k` of its cells made when the map is, except that the tie group of the smallest value is the far
//! end (`k = 0`, the sky). Everything after that works on `k`, in integers or in f64, so the cut of a plane is the
//! same on every machine and never runs the network:
//!
//! * the rank `r = k / (2 (n - 1))` and the depth position `z = 1 - r` ([`DepthMap::z`], [`DepthMap::r`]);
//! * the gradient `S0(z) = clamp(1 - z / f, 0, 1)` of a far limit `f`: 1 at the nearest cell, 0 at depth `f` and beyond,
//!   and with `f = 1` the Show depth picture itself ([`depth_gradient`], [`depth_s0`]);
//! * the plane a Depth row makes ([`DepthSource`], [`depth_plane`]), kept in the cache Background's planes use;
//! * Pick depth ([`pick_far`], [`DepthMap::r_at`]) and the Focus plane ([`DepthMap::focus_z`]).
//!
//! The cut is f64 on the CPU only and the plane is rounded once to a u16 code ([`quantise`]), so the device reads
//! the same bytes (`docs/algorithms/06` section 5.1, step 6).

use std::sync::Arc;

use crate::mask_raster::{SourcePlane, cached_plane, quantise};

/// The least far limit, as a share of the photo's cells, so that the division of the gradient is safe: a limit under it
/// reads as it (`docs/algorithms/06` section 5.8).
pub const DEPTH_FAR_MIN: f64 = 0.02;
/// How far behind the clicked depth the fade ends: Pick depth and Focus plane set the far limit to `z + DEPTH_PICK_MARGIN`
/// (at most 1), so the fade ends a little behind the subject and the subject stays in the mask.
pub const DEPTH_PICK_MARGIN: f64 = 0.1;
/// The side of the Focus plane's box, as a share of the photo's long edge, when the camera gave no size.
pub const FOCUS_BOX_DEFAULT: f64 = 0.04;
/// The least side of the Focus plane's box, in cells of the map.
pub const FOCUS_BOX_MIN_CELLS: usize = 3;
/// The version of the rank and mask rules. It is part of the map's key (`docs/23` section 4.8) and of every plane's.
/// 2: the smallest tie group is the far end, and the band's ramps are centred on its bounds (D-H2, 2026-11-07).
/// 3: the mask is a gradient from the nearest cell to a far limit; the band is gone (the owner, 2026-11-08).
pub const DEPTH_RULES_VERSION: u32 = 3;

/// Why a set of values is not a depth map.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum DepthMapError {
    /// The values are not `width x height`, or the map is empty.
    #[error("the depth map has {found} values for a {width} x {height} grid")]
    Size {
        /// Cells across.
        width: usize,
        /// Cells down.
        height: usize,
        /// Values given.
        found: usize,
    },
    /// A value is not finite.
    #[error("the depth map holds a value that is not finite")]
    NotFinite,
}

/// The depth map of a photo: the network's relative inverse depth `d` (larger is nearer) over the **photo frame**
/// (the cells cover it uniformly), with the mid-rank `k` of every cell (`docs/algorithms/06` section 5.8, step 1).
#[derive(Clone, PartialEq)]
pub struct DepthMap {
    /// What the map was made from: equal keys are the same cells. The core gives every map it makes a key of its own.
    pub key: u64,
    /// Cells across.
    pub width: usize,
    /// Cells down.
    pub height: usize,
    /// The network's values, row-major, finite.
    pub d: Vec<f32>,
    /// The mid-rank of every cell, `2 less + eq - 1`, 0 to `2 (n - 1)`; 0 for the group of the smallest value.
    pub k: Vec<u32>,
}

impl std::fmt::Debug for DepthMap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DepthMap")
            .field("key", &self.key)
            .field("width", &self.width)
            .field("height", &self.height)
            .finish_non_exhaustive()
    }
}

/// The mid-ranks of `d`: for a cell `i`, `less` is the number of cells below it and `eq` the number equal to it (itself
/// included), and `k = 2 less + eq - 1`, except that the tie group of the smallest value (`less = 0`) is the far end,
/// `k = 0` (the sky: large flat areas come out of the network as exactly the smallest value). A single smallest cell
/// has `k = 0` either way, and a map of one value is all far end. Equality is equality of the values with `-0.0` read
/// as `+0.0`; the counts come from one sort, so cells with equal values have equal `k` whatever order they are listed
/// in. `d` must be finite.
#[must_use]
pub fn depth_ranks(d: &[f32]) -> Vec<u32> {
    // -0.0 is +0.0 (`x + 0.0` turns -0.0 into +0.0 and changes nothing else).
    let mut sorted: Vec<f32> = d.iter().map(|&v| v + 0.0).collect();
    sorted.sort_unstable_by(f32::total_cmp);
    d.iter()
        .map(|&v| {
            let v = v + 0.0;
            let less = sorted.partition_point(|s| s.total_cmp(&v).is_lt());
            let up_to = sorted.partition_point(|s| s.total_cmp(&v).is_le());
            if less == 0 {
                return 0;
            }
            // The counts are below 2^31 for any map the network makes (770 x 518 cells is 4e5).
            (2 * less + (up_to - less) - 1) as u32
        })
        .collect()
}

impl DepthMap {
    /// Makes a map from its values, with its ranks.
    ///
    /// # Errors
    ///
    /// [`DepthMapError::Size`] for a grid that is empty or does not hold `d`, and [`DepthMapError::NotFinite`] for a
    /// value that is NaN or infinite.
    pub fn new(key: u64, width: usize, height: usize, d: Vec<f32>) -> Result<Self, DepthMapError> {
        if width == 0 || height == 0 || d.len() != width * height {
            return Err(DepthMapError::Size {
                width,
                height,
                found: d.len(),
            });
        }
        if d.iter().any(|v| !v.is_finite()) {
            return Err(DepthMapError::NotFinite);
        }
        let k = depth_ranks(&d);
        Ok(Self {
            key,
            width,
            height,
            d,
            k,
        })
    }

    /// The number of cells.
    #[must_use]
    pub fn len(&self) -> usize {
        self.k.len()
    }

    /// Whether the map has no cell (it cannot be made so).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.k.is_empty()
    }

    /// `2 (n - 1)`, the largest mid-rank (0 for a map of one cell).
    fn k_max(&self) -> u64 {
        2 * (self.len().max(1) as u64 - 1)
    }

    /// The rank `r = k / (2 (n - 1))` of cell `i`, 1 for the nearest: one division of two integers, correctly rounded;
    /// 0 (the far end) for a map of one cell.
    #[must_use]
    pub fn r(&self, i: usize) -> f64 {
        match self.k_max() {
            0 => 0.0,
            m => f64::from(self.k[i]) / m as f64,
        }
    }

    /// The depth position `z = 1 - r` of cell `i` (0 is the nearest part of the photo), the share of the photo that is
    /// nearer with ties counted as half: `(2 greater + eq - 1) / (2 (n - 1))`, made from integers so that `r + z` is 1
    /// to the last bit. The group of the smallest value is exactly 1, and so is a map of one cell.
    #[must_use]
    pub fn z(&self, i: usize) -> f64 {
        match self.k_max() {
            0 => 1.0,
            m => (m - u64::from(self.k[i])) as f64 / m as f64,
        }
    }

    /// The rank `r` at a point of the photo frame (`x`, `y` are shares of the width and height): bilinear at
    /// `(u, v) = (x w - 0.5, y h - 0.5)` with the four taps held to the map (`docs/algorithms/06` section 5.8,
    /// step 4). A point outside the photo is held to its edge.
    #[must_use]
    pub fn r_at(&self, x: f64, y: f64) -> f64 {
        let (w, h) = (self.width, self.height);
        let u = unit_or(x, 0.5) * w as f64 - 0.5;
        let v = unit_or(y, 0.5) * h as f64 - 0.5;
        let (u0, v0) = (u.floor(), v.floor());
        let (fx, fy) = (u - u0, v - v0);
        let tap = |cx: f64, cy: f64| {
            let cx = (cx.max(0.0) as usize).min(w - 1);
            let cy = (cy.max(0.0) as usize).min(h - 1);
            self.r(cy * w + cx)
        };
        let top = tap(u0, v0) * (1.0 - fx) + tap(u0 + 1.0, v0) * fx;
        let bottom = tap(u0, v0 + 1.0) * (1.0 - fx) + tap(u0 + 1.0, v0 + 1.0) * fx;
        top * (1.0 - fy) + bottom * fy
    }

    /// The median depth position `z` of the cells whose centres lie inside a box (the Focus plane,
    /// `docs/algorithms/06` section 5.8, step 5). The box is centred on `(cx, cy)` and `bw x bh` wide, all shares of the
    /// photo frame; it is made at least [`FOCUS_BOX_MIN_CELLS`] cells on each side and cut to the photo. The cells are
    /// sorted by `k`; the median is the middle `k` of an odd count and `(k_a + k_b) / 2` of the two middle ones of an
    /// even count; when no centre is inside, the cell that holds the centre is the answer.
    #[must_use]
    pub fn focus_z(&self, cx: f64, cy: f64, bw: f64, bh: f64) -> f64 {
        let (w, h) = (self.width, self.height);
        let cx = unit_or(cx, 0.5);
        let cy = unit_or(cy, 0.5);
        let bw = if bw.is_finite() { bw } else { 0.0 }.max(FOCUS_BOX_MIN_CELLS as f64 / w as f64);
        let bh = if bh.is_finite() { bh } else { 0.0 }.max(FOCUS_BOX_MIN_CELLS as f64 / h as f64);
        let (x0, x1) = ((cx - bw / 2.0).max(0.0), (cx + bw / 2.0).min(1.0));
        let (y0, y1) = ((cy - bh / 2.0).max(0.0), (cy + bh / 2.0).min(1.0));
        // A cell `i` has its centre at `(i + 0.5) / w`: inside when `x0 <= (i + 0.5) / w <= x1`.
        let cells = |lo: f64, hi: f64, n: usize| {
            let first = (lo * n as f64 - 0.5).ceil().max(0.0);
            let last = (hi * n as f64 - 0.5).floor().min(n as f64 - 1.0);
            (first, last)
        };
        let (fx, lx) = cells(x0, x1, w);
        let (fy, ly) = cells(y0, y1, h);
        let mut ks: Vec<u32> = Vec::new();
        if fx <= lx && fy <= ly {
            for y in fy as usize..=ly as usize {
                for x in fx as usize..=lx as usize {
                    ks.push(self.k[y * w + x]);
                }
            }
        }
        if ks.is_empty() {
            let x = ((cx * w as f64).floor() as usize).min(w - 1);
            let y = ((cy * h as f64).floor() as usize).min(h - 1);
            ks.push(self.k[y * w + x]);
        }
        ks.sort_unstable();
        let n = ks.len();
        let km = if n % 2 == 1 {
            f64::from(ks[n / 2])
        } else {
            (f64::from(ks[n / 2 - 1]) + f64::from(ks[n / 2])) / 2.0
        };
        match self.k_max() {
            0 => 1.0,
            m => (m as f64 - km) / m as f64,
        }
    }

    /// The Show depth plane: the rank `r` of every cell as a u16 code, white (1) for the nearest
    /// (`docs/algorithms/06` section 5.8, step 6). `key` names the plane.
    #[must_use]
    pub fn show_plane(&self, key: u64) -> SourcePlane {
        SourcePlane {
            key,
            width: self.width,
            height: self.height,
            data: (0..self.len()).map(|i| quantise(self.r(i))).collect(),
        }
    }
}

/// `x` held to [0, 1]; a NaN reads as `default`.
fn unit_or(x: f64, default: f64) -> f64 {
    if x.is_nan() {
        default
    } else {
        x.clamp(0.0, 1.0)
    }
}

/// The far limit as the gradient reads it: held to [`DEPTH_FAR_MIN`]..1; a NaN reads as the default, 1.
#[must_use]
pub fn depth_far(far: f64) -> f64 {
    if far.is_nan() {
        1.0
    } else {
        far.clamp(DEPTH_FAR_MIN, 1.0)
    }
}

/// The mask at depth position `z` for the far limit `far` (already held by [`depth_far`]): `clamp(1 - z / far, 0, 1)`,
/// linear, 1 at the nearest cell (`z = 0`) and 0 at `z = far` and beyond. With `far = 1` it is `1 - z`, the Show depth
/// picture.
#[must_use]
pub fn depth_gradient(z: f64, far: f64) -> f64 {
    (1.0 - z / far).clamp(0.0, 1.0)
}

/// `S0` of a Depth row from the map's ranks: the gradient per cell in f64 and rounded once to a u16 code.
#[must_use]
pub fn depth_s0(map: &DepthMap, far: f64) -> Vec<u16> {
    let far = depth_far(far);
    (0..map.len())
        .map(|i| quantise(depth_gradient(map.z(i), far)))
        .collect()
}

/// What makes a Depth row's `S0`: the photo's depth map and the row's far limit.
#[derive(Debug, Clone, PartialEq)]
pub struct DepthSource {
    /// The photo's depth map.
    pub map: Arc<DepthMap>,
    /// Where the mask fades out, a share of the photo's depth order, 0.02 to 1 (1 is the whole depth).
    pub far: f64,
}

impl DepthSource {
    /// What identifies the plane: the map and the far limit after the clamping (FNV-1a over their bits, with a tag of its
    /// own so that no Background plane shares a key), never Feather, Grow or Refine edges, which act after the plane.
    #[must_use]
    pub fn key(&self) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for bits in [
            0x0000_00de_97d0_0001,
            u64::from(DEPTH_RULES_VERSION),
            self.map.key,
            depth_far(self.far).to_bits(),
        ] {
            for byte in bits.to_le_bytes() {
                h ^= u64::from(byte);
                h = h.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
        h
    }
}

/// The plane of a Depth row, from the small in-memory cache (never on disk) that Background's planes use when the same
/// map was cut with the same limit lately, else made now ([`depth_s0`]): moving `far` cuts a new plane from the cached
/// ranks and never runs the network, and Feather, Grow and Refine edges, which are not in the key, never cut one.
#[must_use]
pub fn depth_plane(src: &DepthSource) -> Arc<SourcePlane> {
    depth_plane_among(src, 0)
}

/// [`depth_plane`] for a render with `live` raster rows: the cache keeps at least `max(16, live)` planes.
#[must_use]
pub fn depth_plane_among(src: &DepthSource, live: usize) -> Arc<SourcePlane> {
    let key = src.key();
    cached_plane(key, live, || SourcePlane {
        key,
        width: src.map.width,
        height: src.map.height,
        data: depth_s0(&src.map, src.far),
    })
}

/// The far limit Pick depth and Focus plane set for the depth position `z_p`: `min(1, z_p + DEPTH_PICK_MARGIN)`, so the
/// mask is full at the nearest and fades to none a little behind the picked depth (`docs/algorithms/06` section 5.8,
/// step 4). Not rounded; the least limit [`DEPTH_FAR_MIN`] applies, and a NaN reads as the middle of the photo.
#[must_use]
pub fn pick_far(z_p: f64) -> f64 {
    depth_far((unit_or(z_p, 0.5) + DEPTH_PICK_MARGIN).min(1.0))
}

/// Pick depth: the far limit for the depth under the point `(x, y)` of the photo frame (`docs/algorithms/06`
/// section 5.8, step 4).
#[must_use]
pub fn pick_depth(map: &DepthMap, x: f64, y: f64) -> f64 {
    pick_far(1.0 - map.r_at(x, y))
}
