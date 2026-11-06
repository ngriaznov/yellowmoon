//! The draft of a zoom (`docs/17-gpu.md` section 19): a rectangle of a finished RGBA8 picture resampled, bilinear, into a
//! cell-sized RGBA8 frame, so that a zoom shows the nearest whole-image level at once and the exact frame replaces it when
//! it is rendered. The CPU reference is [`resample_rgba8`].

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

/// The uniform block of `level_draft.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DraftU {
    /// Source width and height, output width and height.
    size: [u32; 4],
    /// `sx`, `tx`, `sy`, `ty`.
    map: [f32; 4],
}

/// Records the resampling of `src` (`src_size` pixels, one `u32` each, row-major) into `dst` (`dst_size` pixels).
///
/// # Errors
///
/// [`GpuError::InvalidInput`] for an empty picture or a buffer smaller than its pixels; shader errors.
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
    let u = DraftU {
        size: [
            src_size.0 as u32,
            src_size.1 as u32,
            dst_size.0 as u32,
            dst_size.1 as u32,
        ],
        map: [map.sx, map.tx, map.sy, map.ty],
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

/// The CPU reference of the kernel: the same arithmetic in `f32`, in the same order. `src` is `4 * src_size.0 * src_size.1` bytes
/// of RGBA, the result `4 * dst_size.0 * dst_size.1`.
#[must_use]
pub fn resample_rgba8(
    src: &[u8],
    src_size: (usize, usize),
    map: DraftMap,
    dst_size: (usize, usize),
) -> Vec<u8> {
    let (sw, sh) = src_size;
    let at = |x: i64, y: i64| -> [f32; 4] {
        let x = x.clamp(0, sw as i64 - 1) as usize;
        let y = y.clamp(0, sh as i64 - 1) as usize;
        let p = &src[4 * (y * sw + x)..4 * (y * sw + x) + 4];
        [
            f32::from(p[0]),
            f32::from(p[1]),
            f32::from(p[2]),
            f32::from(p[3]),
        ]
    };
    let mut out = Vec::with_capacity(4 * dst_size.0 * dst_size.1);
    for j in 0..dst_size.1 {
        for i in 0..dst_size.0 {
            let u = map.sx * (i as f32 + 0.5) + map.tx - 0.5;
            let v = map.sy * (j as f32 + 0.5) + map.ty - 0.5;
            let (fu, fv) = (u.floor(), v.floor());
            let (a, b) = (u - fu, v - fv);
            let (x0, y0) = (fu as i64, fv as i64);
            let (p00, p10, p01, p11) = (at(x0, y0), at(x0 + 1, y0), at(x0, y0 + 1), at(x0 + 1, y0 + 1));
            for c in 0..4 {
                let top = p00[c] * (1.0 - a) + p10[c] * a;
                let bottom = p01[c] * (1.0 - a) + p11[c] * a;
                let value = (top * (1.0 - b) + bottom * b + 0.5).floor().clamp(0.0, 255.0);
                out.push(value as u8);
            }
        }
    }
    out
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
        assert_eq!(resample_rgba8(&src, (4, 2), id, (4, 2)), src);
        // Half a pixel to the right: the mean of two neighbours (rounded half up).
        let shifted = DraftMap { tx: 0.5, ..id };
        let out = resample_rgba8(&src, (4, 2), shifted, (3, 2));
        let want = ((u32::from(src[0]) + u32::from(src[4]) + 1) / 2) as u8;
        assert_eq!(out[0], want);
    }
}
