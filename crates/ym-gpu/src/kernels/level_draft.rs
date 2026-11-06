//! The draft of a zoom (`docs/17-gpu.md` section 19): a rectangle of a finished RGBA8 picture resampled, bilinear, into a
//! cell-sized RGBA8 frame, so that a zoom shows the nearest whole-image level at once and the exact frame replaces it when
//! it is rendered. The arithmetic is integer, so every device gives the bytes of the CPU reference [`resample_rgba8`].

use bytemuck::{Pod, Zeroable};

use super::shader::KernelId;
use super::util::GROUP;
use crate::device::GpuError;
use crate::pool::PooledBuffer;
use crate::stream::PassList;

/// Where the pixels of a draft read: output pixel `(i, j)` reads the source at `x = sx (i + 0.5) + tx`, `y = sy (j + 0.5) + ty`,
/// in source pixel edges (the source pixel `k` spans `[k, k + 1)`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DraftMap {
    /// Source pixels per output pixel, horizontally.
    pub sx: f32,
    /// Where the left edge of the output is in the source.
    pub tx: f32,
    /// Source pixels per output pixel, vertically.
    pub sy: f32,
    /// Where the top edge of the output is in the source.
    pub ty: f32,
}

/// Fraction bits of the fixed-point positions and steps (1/16384 of a source pixel).
const FRAC_BITS: u32 = 14;
/// Fraction bits that are dropped to make the 8-bit mixing weight.
const WEIGHT_SHIFT: u32 = FRAC_BITS - 8;

/// A [`DraftMap`] in integers, the only form the kernel and the CPU reference read: output pixel `i` reads the source at
/// `x = sx * i + ox`, in 1/16384 of a source pixel (`i32`). The step has the same 14 fraction bits as the position, so the
/// drift over a 8192-pixel frame is at most a quarter of a source pixel. Both sides get the numbers from [`DraftFixed::new`]
/// and then use integer arithmetic only, so a compiler's fast math (fused multiply-adds, re-association) cannot change a
/// result: every device gives the CPU's bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DraftFixed {
    /// Horizontal step per output pixel.
    pub sx: i32,
    /// Horizontal position of output pixel 0 (its centre, minus half a source pixel for the bilinear taps).
    pub ox: i32,
    /// Vertical step per output pixel.
    pub sy: i32,
    /// Vertical position of output row 0.
    pub oy: i32,
}

impl DraftFixed {
    /// Converts `map` for a frame of `dst_size` pixels, once, on the CPU, in `f64`.
    ///
    /// The positions of every pixel of the frame stay inside `i32` (the largest is `sx * (width - 1) + ox`). A viewer draft
    /// has steps in [1/8, 8], frames up to 8192 pixels wide and levels up to about 12000 pixels wide, so its positions stay
    /// under 2^30 + 2^28.
    ///
    /// # Errors
    ///
    /// [`GpuError::InvalidInput`] for a map that is not finite or whose positions do not fit (the caller makes no draft).
    pub fn new(map: DraftMap, dst_size: (usize, usize)) -> Result<Self, GpuError> {
        let one = f64::from(1u32 << FRAC_BITS);
        let limit = f64::from(i32::MAX);
        let conv = |v: f64| -> Result<i32, GpuError> {
            let f = (v * one).round();
            if f.is_finite() && f.abs() <= limit {
                Ok(f as i32)
            } else {
                Err(GpuError::InvalidInput(format!(
                    "draft: the map value {v} does not fit 1/{one} pixel positions"
                )))
            }
        };
        let (sx, tx) = (f64::from(map.sx), f64::from(map.tx));
        let (sy, ty) = (f64::from(map.sy), f64::from(map.ty));
        let fixed = Self {
            sx: conv(sx)?,
            ox: conv(0.5 * sx + tx - 0.5)?,
            sy: conv(sy)?,
            oy: conv(0.5 * sy + ty - 0.5)?,
        };
        // Position of the last pixel; the others lie between it and the first.
        let fits = |step: i32, origin: i32, n: usize| -> bool {
            let last = i64::from(step) * (n.max(1) as i64 - 1) + i64::from(origin);
            i32::try_from(last).is_ok()
        };
        if !fits(fixed.sx, fixed.ox, dst_size.0) || !fits(fixed.sy, fixed.oy, dst_size.1) {
            return Err(GpuError::InvalidInput(format!(
                "draft: a {} x {} frame at steps {} and {} leaves the integer positions",
                dst_size.0, dst_size.1, map.sx, map.sy
            )));
        }
        Ok(fixed)
    }
}

/// The uniform block of `level_draft.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DraftU {
    /// Source width and height, output width and height.
    size: [u32; 4],
    /// `sx`, `ox`, `sy`, `oy` ([`DraftFixed`]).
    map: [i32; 4],
}

/// Records the resampling of `src` (`src_size` pixels, one `u32` each, row-major) into `dst` (`dst_size` pixels).
///
/// # Errors
///
/// [`GpuError::InvalidInput`] for an empty picture, a buffer smaller than its pixels, or a map whose positions do not fit
/// ([`DraftFixed::new`]); shader errors.
pub fn encode_level_draft(
    passes: &mut PassList<'_>,
    src: &PooledBuffer,
    src_size: (usize, usize),
    map: DraftMap,
    dst: &PooledBuffer,
    dst_size: (usize, usize),
) -> Result<(), GpuError> {
    let (sb, db) = (
        4 * src_size.0 as u64 * src_size.1 as u64,
        4 * dst_size.0 as u64 * dst_size.1 as u64,
    );
    if sb == 0 || db == 0 || src.size() < sb || dst.size() < db {
        return Err(GpuError::InvalidInput(format!(
            "draft: a {} x {} source of {} bytes into a {} x {} frame of {} bytes",
            src_size.0,
            src_size.1,
            src.size(),
            dst_size.0,
            dst_size.1,
            dst.size()
        )));
    }
    let f = DraftFixed::new(map, dst_size)?;
    let u = DraftU {
        size: [
            src_size.0 as u32,
            src_size.1 as u32,
            dst_size.0 as u32,
            dst_size.1 as u32,
        ],
        map: [f.sx, f.ox, f.sy, f.oy],
    };
    let ub = passes.uniform("draft uniform", &u)?;
    passes.dispatch(
        KernelId::LevelDraft,
        "level draft",
        &[(0, ub.binding()), (1, src.binding()), (2, dst.binding())],
        (
            (dst_size.0 as u32).div_ceil(GROUP),
            (dst_size.1 as u32).div_ceil(GROUP),
            1,
        ),
    )?;
    Ok(())
}

/// The CPU reference of the kernel: the same integer arithmetic. `src` is `4 * src_size.0 * src_size.1` bytes of RGBA, the
/// result `4 * dst_size.0 * dst_size.1`.
///
/// Per axis: `u = step * i + origin` (`i32`), the left tap `u >> 14` (arithmetic, so negatives floor), the weight of the
/// right tap `(u & 16383) >> 6` (0 to 255, of 256); taps clamp to the picture. The mix is
/// `(top * (256 - wb) + bottom * wb + 32768) >> 16` with `top = p00 * (256 - wa) + p10 * wa`; the largest value is
/// `255 * 65536 + 32768`, far inside `u32`.
///
/// # Errors
///
/// [`GpuError::InvalidInput`] for a map whose positions do not fit ([`DraftFixed::new`]).
pub fn resample_rgba8(
    src: &[u8],
    src_size: (usize, usize),
    map: DraftMap,
    dst_size: (usize, usize),
) -> Result<Vec<u8>, GpuError> {
    let f = DraftFixed::new(map, dst_size)?;
    let (sw, sh) = src_size;
    let at = |x: i32, y: i32| -> [u32; 4] {
        let x = x.clamp(0, sw as i32 - 1) as usize;
        let y = y.clamp(0, sh as i32 - 1) as usize;
        let p = &src[4 * (y * sw + x)..4 * (y * sw + x) + 4];
        [
            u32::from(p[0]),
            u32::from(p[1]),
            u32::from(p[2]),
            u32::from(p[3]),
        ]
    };
    let weight = |u: i32| ((u & ((1 << FRAC_BITS) - 1)) >> WEIGHT_SHIFT) as u32;
    let mut out = Vec::with_capacity(4 * dst_size.0 * dst_size.1);
    for j in 0..dst_size.1 {
        let v = f.sy.wrapping_mul(j as i32).wrapping_add(f.oy);
        let (y0, wb) = (v >> FRAC_BITS, weight(v));
        for i in 0..dst_size.0 {
            let u = f.sx.wrapping_mul(i as i32).wrapping_add(f.ox);
            let (x0, wa) = (u >> FRAC_BITS, weight(u));
            let (p00, p10, p01, p11) = (
                at(x0, y0),
                at(x0 + 1, y0),
                at(x0, y0 + 1),
                at(x0 + 1, y0 + 1),
            );
            for c in 0..4 {
                let top = p00[c] * (256 - wa) + p10[c] * wa;
                let bottom = p01[c] * (256 - wa) + p11[c] * wa;
                out.push(((top * (256 - wb) + bottom * wb + 32768) >> 16) as u8);
            }
        }
    }
    Ok(out)
}

/// The WGSL of the draft kernel.
pub(super) fn body() -> String {
    let mut s = String::from(include_str!("../shaders/level_draft.wgsl"));
    s.push_str(&format!(
        "\n@compute @workgroup_size({GROUP}, {GROUP}, 1)\n\
         fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{ draft_main(gid); }}\n"
    ));
    s
}

#[cfg(test)]
mod tests {
    use std::mem::{offset_of, size_of};

    use super::*;
    use crate::kernels::prep::tests::uniform_layout;

    #[test]
    fn the_uniform_block_has_the_layout_of_its_rust_struct() {
        let (fields, span) = uniform_layout(KernelId::LevelDraft, "pu");
        let want = [
            ("size", offset_of!(DraftU, size)),
            ("map", offset_of!(DraftU, map)),
        ];
        assert_eq!(fields.len(), want.len());
        for ((name, offset), (want_name, want_offset)) in fields.iter().zip(want) {
            assert_eq!(name, want_name);
            assert_eq!(*offset as usize, want_offset, "offset of {name}");
        }
        assert_eq!(span as usize, size_of::<DraftU>());
    }

    #[test]
    fn the_identity_map_copies_and_a_half_pixel_mixes() {
        let src: Vec<u8> = (0..4 * 4 * 2).map(|i| (i * 7 % 256) as u8).collect();
        let id = DraftMap {
            sx: 1.0,
            tx: 0.0,
            sy: 1.0,
            ty: 0.0,
        };
        assert_eq!(resample_rgba8(&src, (4, 2), id, (4, 2)).unwrap(), src);
        // Half a pixel to the right: the mean of two neighbours (rounded half up).
        let shifted = DraftMap { tx: 0.5, ..id };
        let out = resample_rgba8(&src, (4, 2), shifted, (3, 2)).unwrap();
        let want = (u32::from(src[0]) + u32::from(src[4])).div_ceil(2) as u8;
        assert_eq!(out[0], want);
    }

    #[test]
    fn the_fixed_point_map_at_the_range_limits() {
        let map = |sx: f32, tx: f32| DraftMap {
            sx,
            tx,
            sy: sx,
            ty: tx,
        };
        // The identity is exact.
        let f = DraftFixed::new(map(1.0, 0.0), (10, 10)).unwrap();
        assert_eq!((f.sx, f.ox), (16384, 0));
        // The largest viewer map: step 8, 8192 pixels, a level of 12000 pixels (positions up to 8 * 8191 + 12000).
        let f = DraftFixed::new(map(8.0, 12000.0), (8192, 8192)).unwrap();
        assert_eq!(f.sx, 8 * 16384);
        let last = i64::from(f.sx) * 8191 + i64::from(f.ox);
        assert!(last < 1 << 31 && last > 1 << 30, "{last}");
        // The smallest step.
        let f = DraftFixed::new(map(0.125, -3.5), (8192, 1)).unwrap();
        assert_eq!(f.sx, 2048);
        // Past the integers, or not a number: refused, never wrapped.
        assert!(DraftFixed::new(map(8.0, 100_000.0), (8192, 8192)).is_err());
        assert!(DraftFixed::new(map(8.0, 0.0), (400_000, 1)).is_err());
        assert!(DraftFixed::new(map(f32::NAN, 0.0), (4, 4)).is_err());
        assert!(DraftFixed::new(map(1.0, f32::INFINITY), (4, 4)).is_err());
        assert!(resample_rgba8(&[0; 4], (1, 1), map(1.0, 1e9), (1, 1)).is_err());
    }
}
