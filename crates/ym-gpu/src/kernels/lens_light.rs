//! The input of L1 and L2 with the colour of the light it holds, and L2's input with the headroom of the small
//! clipped lights, on the device (`docs/21` section 4, *As built (the light's colour)* and *As built (the
//! headroom of a clipped light)*).
//!
//! The CPU references are `ym_lens::lit_input` and the input L2 reads (`light::l2_input`): every pixel goes
//! through `Restore::light` (the clip pattern's cast taken off, then each saturated channel raised to the mean of
//! the channels that still measure, and the pixel's white light weight), which `lens_light_colour.wgsl`
//! computes in `f32`. A function of one pixel and its place, so a tile and the whole frame give the same bits.
//! The restored light of L3 (`lens_flare.wgsl`) includes the same function.
//!
//! One kernel, three modes: **0** the light's colour alone (L1's input); **1** the extra light L2 spreads, the
//! light's colour and the headroom of the clipped lights less the picture (when L1 leaves the picture alone);
//! **2** the same where L1 left the pixel as it is, and the picture L2 gathers beside it (after L1).

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
pub use ym_lens::LightPass;
use ym_lens::consts::{
    CLIP_PATTERN_SPREAD, FIELD_HEADROOM_SHARE, L1_UNCHANGED_RAMP, LIGHT_SATURATION_RAMP,
    LIGHT_WHITE_RAMP,
};
pub use ym_lens::device::LightMapData;

use super::shader::KernelId;
use super::util::{GROUP, storage_buffer};
use crate::cpu::Placement;
use crate::cpu::tiling::Rect;
use crate::device::{GpuContext, GpuError};
use crate::image::GpuImage;
use crate::pool::PooledBuffer;
use crate::stream::PassList;

/// The uniform block of `lens_light.wgsl` (`LightU`).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LightU {
    /// `g0` of each channel, 0.
    clip: [f32; 4],
    /// The clip test's ramp: from, to, 0, 0.
    ramp: [f32; 4],
    /// Input row stride, plane length, 0, 0.
    src: [u32; 4],
    /// Output row stride, plane length, width, height.
    dst: [u32; 4],
    /// The mode, the scale `k` of the picture, the lens plane's scale `k_L`, 0.
    mode: [u32; 4],
    /// The output's first frame column and row, the input's first frame column and row.
    org: [i32; 4],
    /// L1's output (mode 2): its first frame column and row, 0, 0.
    yin: [i32; 4],
    /// L1's output (mode 2): row stride, plane length.
    ys: [u32; 4],
    /// The map's width and height (0: no light in the photo), 0, 0.
    map: [u32; 4],
    /// The cap's scale, 0, 0, 0.
    mapf: [f32; 4],
}

/// The photo's clipped lights on the device: the EV of every pixel of the lens plane, uploaded once per photo
/// (`ym_lens::LensPlanes::light_map_data`).
#[derive(Clone)]
pub struct LightMapBuffer {
    buf: Arc<PooledBuffer>,
    width: u32,
    height: u32,
    plane_scale: u32,
    scale: f32,
}

impl std::fmt::Debug for LightMapBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LightMapBuffer")
            .field("size", &(self.width, self.height))
            .field("plane_scale", &self.plane_scale)
            .field("bytes", &self.buf.size())
            .finish()
    }
}

impl LightMapBuffer {
    /// Uploads a photo's map (a photo without lights uploads one word and reads as no light).
    ///
    /// # Errors
    ///
    /// [`GpuError::InvalidInput`] for a map whose size does not match its samples; the pool's errors.
    pub fn upload(ctx: &GpuContext, data: &LightMapData) -> Result<Self, GpuError> {
        if !data.ev.is_empty() && data.ev.len() != data.width * data.height {
            return Err(GpuError::InvalidInput(format!(
                "lens light map: {} samples for a {} x {} plane",
                data.ev.len(),
                data.width,
                data.height
            )));
        }
        let (width, height) = if data.ev.is_empty() {
            (0, 0)
        } else {
            (data.width as u32, data.height as u32)
        };
        let empty = [0.0f32];
        let samples: &[f32] = if data.ev.is_empty() { &empty } else { &data.ev };
        let buf = storage_buffer(ctx, "lens light map", bytemuck::cast_slice(samples))?;
        Ok(LightMapBuffer {
            buf: Arc::new(buf),
            width,
            height,
            plane_scale: data.plane_scale as u32,
            scale: data.scale as f32,
        })
    }

    /// Bytes on the device.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.buf.size()
    }
}

/// The largest storage binding (bytes) the pass needs for an image of `width x height` pixels.
#[must_use]
pub fn max_binding_bytes(width: usize, height: usize) -> u64 {
    GpuImage::buffer_bytes(width, height, 3)
}

fn check_three_planes(input: &GpuImage) -> Result<(), GpuError> {
    if input.planes != 3 {
        return Err(GpuError::InvalidInput(format!(
            "lens light: {} planes, not 3",
            input.planes
        )));
    }
    Ok(())
}

fn base_uniform(pass: LightPass) -> LightU {
    let mut u = LightU::zeroed();
    u.clip = [pass.clip[0], pass.clip[1], pass.clip[2], 0.0];
    u.ramp = [pass.ramp.0, pass.ramp.1, 0.0, 0.0];
    u
}

/// Records the pass over `input` (three planes): a new image of the same size and place with every pixel's
/// light in its own colour (mode 0: L1's input).
///
/// # Errors
///
/// [`GpuError::InvalidInput`] for an image that is not three planes; the pool's errors; shader errors.
pub fn encode_light(
    passes: &mut PassList<'_>,
    input: &GpuImage,
    pass: LightPass,
) -> Result<GpuImage, GpuError> {
    check_three_planes(input)?;
    let ctx = passes.context();
    let dst = GpuImage::alloc(
        ctx,
        "lens lit input",
        input.width,
        input.height,
        3,
        input.place,
    )?;
    let mut u = base_uniform(pass);
    u.src = [input.stride as u32, input.plane_len() as u32, 0, 0];
    u.dst = [
        dst.stride as u32,
        dst.plane_len() as u32,
        input.width as u32,
        input.height as u32,
    ];
    let ub = passes.uniform("lens light uniform", &u)?;
    let dummy_a = passes.scratch("lens light unused", 16)?;
    let dummy_b = passes.scratch("lens light unused", 16)?;
    let dummy_c = passes.scratch("lens light unused", 16)?;
    passes.dispatch(
        KernelId::LensLight,
        "lens light",
        &[
            (0, ub.binding()),
            (1, input.buf.binding()),
            (2, dst.buf.binding()),
            (3, dummy_a.binding()),
            (4, dummy_b.binding()),
            (5, dummy_c.binding()),
        ],
        (
            (input.width as u32).div_ceil(GROUP),
            (input.height as u32).div_ceil(GROUP),
            1,
        ),
    )?;
    Ok(dst)
}

/// L2's extra light and the picture it gathers, on the device.
pub struct HeadroomImages {
    /// The light L2 spreads that the picture does not record: the picture with the light's colour and the
    /// headroom of the clipped small lights, less the picture (after L1, where L1 left a pixel as it is).
    pub extra: GpuImage,
    /// The picture L2 gathers, where that is not the raw picture (after L1, which changed some pixels); `None`
    /// when it is the raw picture.
    pub own: Option<GpuImage>,
}

/// Records the headroom pass: L2's extra light from the raw picture `raw` (a window whose first pixel is the
/// frame pixel `raw_at`), and, when L1 reshaped, L1's output `l1` (an image whose first pixel is `l1_at`; the
/// pass runs over it). `k` is the scale of the picture the window belongs to.
///
/// Without `l1` the pass runs over the raw window and gives `extra` alone (mode 1); with it, over L1's output,
/// and gives `extra` and `own` (mode 2).
///
/// # Errors
///
/// [`GpuError::InvalidInput`] for images that are not three planes or an L1 output outside the raw window; the
/// pool's errors; shader errors.
pub fn encode_headroom(
    passes: &mut PassList<'_>,
    raw: &GpuImage,
    raw_at: (usize, usize),
    l1: Option<(&GpuImage, (usize, usize))>,
    pass: LightPass,
    map: &LightMapBuffer,
    k: usize,
) -> Result<HeadroomImages, GpuError> {
    check_three_planes(raw)?;
    let ctx = passes.context();
    let (rect, mode) = match l1 {
        Some((img, at)) => {
            check_three_planes(img)?;
            let rect = Rect::new(at.0, at.1, img.width, img.height);
            if at.0 < raw_at.0
                || at.1 < raw_at.1
                || rect.x1() > raw_at.0 + raw.width
                || rect.y1() > raw_at.1 + raw.height
            {
                return Err(GpuError::InvalidInput(format!(
                    "lens light: L1's output {rect:?} lies outside the raw window at {raw_at:?}"
                )));
            }
            (rect, 2u32)
        }
        None => (Rect::new(raw_at.0, raw_at.1, raw.width, raw.height), 1u32),
    };
    let extra = GpuImage::alloc(
        ctx,
        "lens extra light",
        rect.w,
        rect.h,
        3,
        Placement::whole(rect.w, rect.h),
    )?;
    let own = if l1.is_some() {
        Some(GpuImage::alloc(
            ctx,
            "lens l2 picture",
            rect.w,
            rect.h,
            3,
            Placement::whole(rect.w, rect.h),
        )?)
    } else {
        None
    };
    let mut u = base_uniform(pass);
    u.src = [raw.stride as u32, raw.plane_len() as u32, 0, 0];
    u.dst = [
        extra.stride as u32,
        extra.plane_len() as u32,
        rect.w as u32,
        rect.h as u32,
    ];
    u.mode = [mode, k as u32, map.plane_scale, 0];
    u.org = [
        rect.x as i32,
        rect.y as i32,
        raw_at.0 as i32,
        raw_at.1 as i32,
    ];
    if let Some((img, at)) = l1 {
        u.yin = [at.0 as i32, at.1 as i32, 0, 0];
        u.ys = [img.stride as u32, img.plane_len() as u32, 0, 0];
    }
    u.map = [map.width, map.height, 0, 0];
    u.mapf = [map.scale, 0.0, 0.0, 0.0];
    let ub = passes.uniform("lens headroom uniform", &u)?;
    let dummy_y = passes.scratch("lens light unused", 16)?;
    let dummy_own = passes.scratch("lens light unused", 16)?;
    let y_in = match l1 {
        Some((img, _)) => img.buf.binding(),
        None => dummy_y.binding(),
    };
    let own_out = match &own {
        Some(o) => o.buf.binding(),
        None => dummy_own.binding(),
    };
    passes.dispatch(
        KernelId::LensLight,
        "lens headroom",
        &[
            (0, ub.binding()),
            (1, raw.buf.binding()),
            (2, extra.buf.binding()),
            (3, y_in),
            (4, own_out),
            (5, map.buf.binding()),
        ],
        (
            (rect.w as u32).div_ceil(GROUP),
            (rect.h as u32).div_ceil(GROUP),
            1,
        ),
    )?;
    passes.hold(Arc::clone(&map.buf));
    Ok(HeadroomImages { extra, own })
}

/// The colour pass on the host's behalf: uploads `input` (three planes, `width x height`), runs it and reads the
/// planes back. The GPU counterpart of `ym_lens::lit_input`; for tests.
///
/// # Errors
///
/// As [`encode_light`], plus the upload's and the readback's.
pub fn light(
    ctx: &GpuContext,
    input: &[f32],
    width: usize,
    height: usize,
    pass: LightPass,
) -> Result<Vec<f32>, GpuError> {
    let src = GpuImage::upload(
        ctx,
        "lens light input",
        width,
        height,
        3,
        input,
        Placement::whole(width, height),
    )?;
    let mut passes = PassList::new(ctx, "lens light");
    let img = encode_light(&mut passes, &src, pass)?;
    let read = img.encode_read(&mut passes)?;
    let done = passes.submit()?.finish()?;
    img.take(&done, read)
}

/// The headroom pass on the host's behalf: uploads `raw` (three planes, `size`, whose first pixel is the frame
/// pixel `raw_at`) and, when given, L1's output (`l1`: its samples, size and first frame pixel), runs the pass
/// and reads back `extra` and `own` (`own` empty without L1). The GPU counterpart of `ym_lens::l2_input`; for
/// tests.
///
/// # Errors
///
/// As [`encode_headroom`], plus the uploads' and the readbacks'.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn headroom(
    ctx: &GpuContext,
    raw: &[f32],
    size: (usize, usize),
    raw_at: (usize, usize),
    l1: Option<(&[f32], (usize, usize), (usize, usize))>,
    pass: LightPass,
    map: &LightMapData,
    k: usize,
) -> Result<(Vec<f32>, Vec<f32>), GpuError> {
    let src = GpuImage::upload(
        ctx,
        "lens headroom raw",
        size.0,
        size.1,
        3,
        raw,
        Placement::whole(size.0, size.1),
    )?;
    let l1_img = match l1 {
        Some((data, s, _)) => Some(GpuImage::upload(
            ctx,
            "lens headroom l1",
            s.0,
            s.1,
            3,
            data,
            Placement::whole(s.0, s.1),
        )?),
        None => None,
    };
    let map = LightMapBuffer::upload(ctx, map)?;
    let mut passes = PassList::new(ctx, "lens headroom");
    let l1_arg = match (&l1_img, l1) {
        (Some(img), Some((_, _, at))) => Some((img, at)),
        _ => None,
    };
    let out = encode_headroom(&mut passes, &src, raw_at, l1_arg, pass, &map, k)?;
    let read_extra = out.extra.encode_read(&mut passes)?;
    let read_own = match &out.own {
        Some(o) => Some(o.encode_read(&mut passes)?),
        None => None,
    };
    let done = passes.submit()?.finish()?;
    let extra = out.extra.take(&done, read_extra)?;
    let own = match (&out.own, read_own) {
        (Some(o), Some(id)) => o.take(&done, id)?,
        _ => Vec::new(),
    };
    Ok((extra, own))
}

/// The constants the colour function of `lens_light_colour.wgsl` reads, from the CPU reference's.
pub(super) fn constants() -> String {
    format!(
        "const LL_CLIP_PATTERN: f32 = {:?};\n\
         const LL_SAT_FROM: f32 = {:?};\n\
         const LL_SAT_TO: f32 = {:?};\n\
         const LL_WHITE_RAMP: f32 = {:?};\n",
        CLIP_PATTERN_SPREAD as f32,
        LIGHT_SATURATION_RAMP.0 as f32,
        LIGHT_SATURATION_RAMP.1 as f32,
        LIGHT_WHITE_RAMP as f32,
    )
}

/// The WGSL of the shared colour function: its constants, then its text.
pub(super) fn colour_function() -> String {
    let mut s = constants();
    s.push_str(include_str!("../shaders/lens_light_colour.wgsl"));
    s
}

/// The WGSL of the pass.
pub(super) fn body() -> String {
    let mut s = colour_function();
    s.push_str(&format!(
        "const LL_SHARE: f32 = {:?};\n\
         const LL_L1_FROM: f32 = {:?};\n\
         const LL_L1_TO: f32 = {:?};\n",
        FIELD_HEADROOM_SHARE as f32, L1_UNCHANGED_RAMP.0 as f32, L1_UNCHANGED_RAMP.1 as f32,
    ));
    s.push_str(include_str!("../shaders/lens_light.wgsl"));
    s.push_str(&format!(
        "\n@compute @workgroup_size({GROUP}, {GROUP}, 1)\n\
         fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{ ll_main(gid); }}\n"
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
        let (fields, span) = uniform_layout(KernelId::LensLight, "pu");
        let want = [
            ("clip", offset_of!(LightU, clip)),
            ("ramp", offset_of!(LightU, ramp)),
            ("src", offset_of!(LightU, src)),
            ("dst", offset_of!(LightU, dst)),
            ("mode", offset_of!(LightU, mode)),
            ("org", offset_of!(LightU, org)),
            ("yin", offset_of!(LightU, yin)),
            ("ys", offset_of!(LightU, ys)),
            ("map", offset_of!(LightU, map)),
            ("mapf", offset_of!(LightU, mapf)),
        ];
        assert_eq!(fields.len(), want.len());
        for ((name, offset), (want_name, want_offset)) in fields.iter().zip(want) {
            assert_eq!(name, want_name);
            assert_eq!(*offset as usize, want_offset, "offset of {name}");
        }
        assert_eq!(span as usize, size_of::<LightU>());
    }
}
