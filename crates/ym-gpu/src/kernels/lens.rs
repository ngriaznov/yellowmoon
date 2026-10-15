//! R9, the lens resample, on the device (G-07, `docs/05-raw-processing.md` section R9).
//!
//! The CPU reference is [`crate::cpu::resample::resample_lens`]: one Catmull-Rom sample per colour
//! plane at the source position of [`LensGeometry`] (distortion for all three planes, the lateral CA
//! for red and blue when the geometry keeps it in R9), edge pixels replicated. The device forms the
//! position as the output position plus the displacement `d (F K - 1)`, so no float holds a position
//! of the order of the frame size (D4): the offset from the frame centre is an exact integer, the
//! radius is taken from an exact integer sum of squares, and the tables enter as `F_i - 1` and
//! `K_i - 1` (exact in `f32`: a 16-bit knot times a power of two). A tile of any size equals the
//! whole frame bit for bit. This also serves the binned fit view, which resamples the image at 1/k
//! with the geometry of the smaller frame (`RawPlan::lens_at`).

use bytemuck::{Pod, Zeroable};

use super::shader::KernelId;
use super::util::GROUP;
use crate::cpu::lens::LensGeometry;
use crate::cpu::tiling::Rect;
use crate::cpu::{Placement, RgbImage};
use crate::device::{GpuContext, GpuError};
use crate::image::GpuImage;
use crate::stream::PassList;

/// The most knots a table takes (the uniform holds 32 of each).
pub const MAX_KNOTS: usize = 32;

/// The uniform block of `lens.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LensU {
    /// The runtime zero of the barrier (always +0.0 bits).
    zero: [u32; 4],
    /// Source width, height, row stride, plane length.
    src: [u32; 4],
    /// Source column and row of the first output pixel, output width, height.
    out: [i32; 4],
    /// Output row stride, plane length.
    dst: [u32; 4],
    /// Doubled offset of the first pixel from the frame centre (x, y), knots of the distortion, of
    /// one CA curve.
    fr: [i32; 4],
    /// The CA is applied here (1) or was applied on the mosaic (0).
    flags: [u32; 4],
    /// `(n_distortion - 1) / hypot`, `(n_ca - 1) / hypot`.
    k: [f32; 4],
    dist: [[f32; 4]; 8],
    car: [[f32; 4]; 8],
    cab: [[f32; 4]; 8],
}

/// A lens geometry as the shader reads it.
#[derive(Clone, Debug, PartialEq)]
pub struct LensTable {
    frame_w: usize,
    frame_h: usize,
    crop: Rect,
    /// `F_i - 1`.
    dist: Vec<f32>,
    /// `K_red,i - 1` and `K_blue,i - 1` when the geometry has a CA table.
    ca: Option<(Vec<f32>, Vec<f32>)>,
    ca_in_r9: bool,
}

impl LensTable {
    /// The table of `geo`.
    ///
    /// # Errors
    ///
    /// [`GpuError::InvalidInput`] for a table with more than [`MAX_KNOTS`] knots.
    pub fn from_geometry(geo: &LensGeometry) -> Result<LensTable, GpuError> {
        let minus_one = |ys: &[f64]| -> Result<Vec<f32>, GpuError> {
            if ys.len() > MAX_KNOTS {
                return Err(GpuError::InvalidInput(format!(
                    "lens: a table of {} knots (the kernel takes at most {MAX_KNOTS})",
                    ys.len()
                )));
            }
            Ok(ys.iter().map(|&y| (y - 1.0) as f32).collect())
        };
        Ok(LensTable {
            frame_w: geo.frame_width(),
            frame_h: geo.frame_height(),
            crop: geo.crop(),
            dist: geo
                .distortion_knots()
                .map(minus_one)
                .transpose()?
                .unwrap_or_default(),
            ca: geo
                .ca_knots()
                .map(|(r, b)| Ok::<_, GpuError>((minus_one(r)?, minus_one(b)?)))
                .transpose()?,
            ca_in_r9: geo.ca_in_r9(),
        })
    }

    /// The size of the cropped output.
    #[must_use]
    pub fn output_size(&self) -> (usize, usize) {
        (self.crop.w, self.crop.h)
    }
}

fn pad(v: &[f32]) -> [[f32; 4]; 8] {
    let mut out = [[0.0f32; 4]; 8];
    for (i, x) in v.iter().take(MAX_KNOTS).enumerate() {
        out[i / 4][i % 4] = *x;
    }
    out
}

/// The largest storage binding (bytes) the pass needs for a source of `rect` pixels: its three
/// planes.
#[must_use]
pub fn max_binding_bytes(rect: Rect) -> u64 {
    GpuImage::buffer_bytes(rect.w, rect.h, 3)
}

/// Records R9 for the rectangle `out` of the cropped output of `table` (device in, device out), from
/// the `window` of `src` (a rectangle of the source image that holds the source positions of `out`
/// with the Catmull-Rom support, `LensGeometry::source_rect`; samples beyond it replicate its edge).
/// The result is `out.w x out.h` with the placement `(out.x, out.y)` in a frame of the cropped size.
///
/// # Errors
///
/// [`GpuError::InvalidInput`] for a source that is not three planes or whose frame is not the
/// geometry's, an empty window or output, or an output outside the cropped image; the pool's errors;
/// shader errors.
pub fn encode_lens(
    passes: &mut PassList<'_>,
    src: &GpuImage,
    window: Rect,
    table: &LensTable,
    out: Rect,
) -> Result<GpuImage, GpuError> {
    let ctx = passes.context();
    if src.planes != 3 {
        return Err(GpuError::InvalidInput(format!(
            "lens: the source has {} planes, not 3",
            src.planes
        )));
    }
    if (src.place.frame_width, src.place.frame_height) != (table.frame_w, table.frame_h) {
        return Err(GpuError::InvalidInput(format!(
            "lens: the source frame is {} x {}, the geometry's {} x {}",
            src.place.frame_width, src.place.frame_height, table.frame_w, table.frame_h
        )));
    }
    if window.is_empty() || window.x1() > src.width || window.y1() > src.height {
        return Err(GpuError::InvalidInput(format!(
            "lens: the window {window:?} is empty or outside the {} x {} source",
            src.width, src.height
        )));
    }
    let (cw, ch) = table.output_size();
    if out.is_empty() || out.x1() > cw || out.y1() > ch {
        return Err(GpuError::InvalidInput(format!(
            "lens: the output {out:?} is empty or outside the {cw} x {ch} cropped image"
        )));
    }
    let diag = (table.frame_w as f64).hypot(table.frame_h as f64);
    if (table.frame_w as u64).pow(2) + (table.frame_h as u64).pow(2) >= 1 << 30 {
        return Err(GpuError::InvalidInput(format!(
            "lens: a {} x {} frame is too large for the 32-bit radius",
            table.frame_w, table.frame_h
        )));
    }
    let dst = GpuImage::alloc(
        ctx,
        "lens rgb",
        out.w,
        out.h,
        3,
        Placement {
            x0: out.x,
            y0: out.y,
            frame_width: cw,
            frame_height: ch,
        },
    )?;
    let nd = table.dist.len();
    let (car, cab, nc) = match (&table.ca, table.ca_in_r9) {
        (Some((r, b)), true) => (pad(r), pad(b), r.len().min(b.len())),
        _ => ([[0.0; 4]; 8], [[0.0; 4]; 8], 0),
    };
    let ox = src.place.x0 as i64 + window.x as i64;
    let oy = src.place.y0 as i64 + window.y as i64;
    let u = LensU {
        zero: [0; 4],
        // The window is addressed inside the image: its origin is added by the kernel through `out`.
        src: [
            src.width as u32,
            src.height as u32,
            src.stride as u32,
            src.plane_len() as u32,
        ],
        out: [
            (table.crop.x as i64 + out.x as i64 - ox + window.x as i64) as i32,
            (table.crop.y as i64 + out.y as i64 - oy + window.y as i64) as i32,
            out.w as i32,
            out.h as i32,
        ],
        dst: [dst.stride as u32, dst.plane_len() as u32, 0, 0],
        fr: [
            2 * (table.crop.x + out.x) as i32 + 1 - table.frame_w as i32,
            2 * (table.crop.y + out.y) as i32 + 1 - table.frame_h as i32,
            nd as i32,
            nc as i32,
        ],
        flags: [u32::from(nc > 0), 0, 0, 0],
        k: [
            if nd > 1 { (nd - 1) as f32 / diag as f32 } else { 0.0 },
            if nc > 1 { (nc - 1) as f32 / diag as f32 } else { 0.0 },
            0.0,
            0.0,
        ],
        dist: pad(&table.dist),
        car,
        cab,
    };
    let ub = passes.uniform("lens uniform", &u)?;
    passes.dispatch(
        KernelId::Lens,
        "lens r9",
        &[(0, ub.binding()), (1, src.buf.binding()), (2, dst.buf.binding())],
        (
            (out.w as u32).div_ceil(GROUP),
            (out.h as u32).div_ceil(GROUP),
            1,
        ),
    )?;
    Ok(dst)
}

/// R9 on the host: uploads `src`, resamples the rectangle `out` of the cropped output of `geo` and
/// reads it back. The GPU counterpart of [`crate::cpu::resample::resample_lens`].
///
/// # Errors
///
/// As [`encode_lens`], plus the readback's.
pub fn lens(
    ctx: &GpuContext,
    src: &RgbImage,
    geo: &LensGeometry,
    out: Rect,
) -> Result<RgbImage, GpuError> {
    let table = LensTable::from_geometry(geo)?;
    let input = GpuImage::upload_rgb(ctx, "lens source", src)?;
    let mut passes = PassList::new(ctx, "lens");
    let img = encode_lens(
        &mut passes,
        &input,
        Rect::whole(src.width, src.height),
        &table,
        out,
    )?;
    let read = img.encode_read(&mut passes)?;
    let done = passes.submit()?.finish()?;
    let data = img.take(&done, read)?;
    Ok(RgbImage {
        width: img.width,
        height: img.height,
        place: img.place,
        data,
    })
}

/// The WGSL of the lens kernel.
pub(super) fn body() -> String {
    let mut s = String::new();
    s.push_str(include_str!("../shaders/front_opaque.wgsl"));
    s.push_str(include_str!("../shaders/lens.wgsl"));
    s.push_str(&format!(
        "\n@compute @workgroup_size({GROUP}, {GROUP}, 1)\n\
         fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{ lens_main(gid); }}\n"
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
        let (fields, span) = uniform_layout(KernelId::Lens, "pu");
        let want = [
            ("zero", offset_of!(LensU, zero)),
            ("src", offset_of!(LensU, src)),
            ("out", offset_of!(LensU, out)),
            ("dst", offset_of!(LensU, dst)),
            ("fr", offset_of!(LensU, fr)),
            ("flags", offset_of!(LensU, flags)),
            ("k", offset_of!(LensU, k)),
            ("dist", offset_of!(LensU, dist)),
            ("car", offset_of!(LensU, car)),
            ("cab", offset_of!(LensU, cab)),
        ];
        assert_eq!(fields.len(), want.len());
        for ((name, offset), (want_name, want_offset)) in fields.iter().zip(want) {
            assert_eq!(name, want_name);
            assert_eq!(*offset as usize, want_offset, "offset of {name}");
        }
        assert_eq!(span as usize, size_of::<LensU>());
    }
}
