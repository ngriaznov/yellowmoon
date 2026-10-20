//! The magnification of a view above 100 % on the device (`docs/17-gpu.md` section 16): the
//! Catmull-Rom magnification of the sharpened 100 % picture to the view, with the paper, and C13 fused
//! (the transfer, the dither at the pixel's place in the view, the paper colour), as one RGBA8 word per
//! pixel. The CPU reference is [`crate::cpu::magnify`] (`ym-core` runs it for a view the device does
//! not make).
//!
//! The host hands the kernel the taps of [`MagnifyMap`] (one per column and one per row, made in f64
//! by the CPU reference's own code), so the kernel computes no position and no weight: its only
//! arithmetic is the sum, in the CPU's order, with every product and partial sum behind the fast-math
//! barrier (`shaders/magnify.wgsl`). The magnified values are the CPU reference's bit for bit on every
//! adapter (`tests/gpu_magnify.rs`); the 8-bit codes then differ from the CPU's only by C13 itself
//! (f32 on the device, f64 on the CPU), the C13 bound of `docs/17-gpu.md` section 12.
//!
//! [`encode_magnify`] records the pass into a [`PassList`]; [`magnify_planes`] and [`magnify_rgba8`]
//! are the thin host wrappers the tests use.

use bytemuck::{Pod, Zeroable};
use ym_color::encode::OutputSpace;

use super::color::dither_table;
use super::encode::C13;
use super::shader::KernelId;
use super::util::{GROUP, STORAGE_IN, STORAGE_OUT};
use crate::cpu::magnify::{AxisTap, MagnifyMap};
use crate::cpu::tiling::Rect;
use crate::cpu::{Placement, RgbImage};
use crate::device::{GpuContext, GpuError};
use crate::image::GpuImage;
use crate::pool::PooledBuffer;
use crate::stream::{PassList, upload_bytes};

/// The uniform block of `magnify.wgsl` (`MagnifyU`): 64 bytes.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
struct MagnifyU {
    view: [u32; 4],
    src: [u32; 4],
    c13: [u32; 4],
    org: [u32; 4],
}

/// One tap as the shader reads it (`Tap`): 48 bytes.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default, Debug, PartialEq)]
struct TapU {
    idx: [u32; 4],
    w: [f32; 4],
    at: u32,
    inside: u32,
    pad: [u32; 2],
}

impl From<&AxisTap> for TapU {
    fn from(t: &AxisTap) -> Self {
        Self {
            idx: t.idx,
            w: t.weights,
            at: t.mask_at,
            inside: u32::from(t.inside),
            pad: [0; 2],
        }
    }
}

/// Where the pass writes the view.
pub enum MagnifyTarget<'a> {
    /// Four f32 planes of a view-sized image: the magnified display-linear RGB (0 on the paper) and
    /// 1.0 on the paper. This followed by the encode kernel is the [`MagnifyTarget::Rgba8`] pass bit
    /// for bit.
    Planes(&'a GpuImage),
    /// RGBA8 words, one a pixel, the view's width a row (`4 w h` bytes from offset 0), C13 fused.
    Rgba8 {
        /// The buffer.
        dst: &'a PooledBuffer,
        /// The transfer, the dither and its origin (the place of the view's pixel (0, 0) in the
        /// picture the dither texels are counted in).
        c13: C13,
        /// The paper colour.
        background: [u8; 3],
    },
}

/// The size of the largest storage binding the viewport's pass over a `view` makes (K4 of
/// `docs/17-gpu.md`): the RGBA8 output, 4 bytes a pixel. The source is the caller's (the sharpened
/// 100 % picture under the view, which the colour pass made and which `ym-core` sizes against the
/// design binding), the taps are 48 bytes a column and a row, and the four-plane output of
/// [`MagnifyTarget::Planes`], 16 bytes a pixel, is for the tests.
#[must_use]
pub fn max_binding_bytes(view: Rect) -> u64 {
    4 * view.w as u64 * view.h as u64
}

/// The WGSL of the kernel: the C13 helpers and the pass.
pub(super) fn body() -> String {
    let mut s = String::from(include_str!("../shaders/c13.wgsl"));
    s.push('\n');
    s.push_str(include_str!("../shaders/magnify.wgsl"));
    s
}

/// The `src` words of the uniform block: the source's stride and plane length, whether it has a mask
/// plane, and the fast-math barrier's zero (`opaque` ORs it into every product and partial sum). The
/// zero must be the bits of `+0.0`: any other word would change every value, and `-0.0` would set
/// every sign bit (`docs/17-gpu.md` K8).
fn source_words(stride: usize, plane_len: usize, planes: usize) -> [u32; 4] {
    [
        stride as u32,
        plane_len as u32,
        u32::from(planes == 4),
        0.0f32.to_bits(),
    ]
}

fn background_word(bg: [u8; 3]) -> u32 {
    u32::from(bg[0]) | (u32::from(bg[1]) << 8) | (u32::from(bg[2]) << 16) | 0xff00_0000
}

/// Records the magnification of `src` (the sharpened 100 % picture under the view: three planes of
/// display-linear RGB, or four with the letterbox mask) through the taps of `map` into `target`.
/// `src` and the target must stay alive until the list has finished. Nothing is read back.
///
/// # Errors
///
/// [`GpuError::InvalidInput`] when `src` is not a three- or four-plane image of the size the taps were
/// made for, the target is too small or not a four-plane image, or the view needs more workgroups than
/// the device allows; the pool's errors for the tap tables.
pub fn encode_magnify(
    passes: &mut PassList<'_>,
    src: &GpuImage,
    map: &MagnifyMap,
    target: MagnifyTarget<'_>,
) -> Result<(), GpuError> {
    let ctx = passes.context();
    let (w, h) = (map.width(), map.height());
    if !matches!(src.planes, 3 | 4) || (src.width, src.height) != map.src {
        return Err(GpuError::InvalidInput(format!(
            "the magnification reads the {} x {} picture its taps were made for, with three or four \
             planes, not a {} x {} image of {} planes",
            map.src.0, map.src.1, src.width, src.height, src.planes
        )));
    }
    if w == 0 || h == 0 {
        return Err(GpuError::InvalidInput(
            "the magnification needs a view of at least one pixel".into(),
        ));
    }
    let limit = ctx.info().limits.max_compute_workgroups_per_dimension as usize;
    if w.div_ceil(GROUP as usize) > limit || h.div_ceil(GROUP as usize) > limit {
        return Err(GpuError::InvalidInput(format!(
            "a {w} x {h} view needs more workgroups than the device allows ({limit})"
        )));
    }
    let mut u = MagnifyU {
        view: [w as u32, h as u32, 0, 0],
        src: source_words(src.stride, src.plane_len(), src.planes),
        ..MagnifyU::default()
    };
    let dst: &PooledBuffer = match &target {
        MagnifyTarget::Planes(image) => {
            if image.planes != 4 || (image.width, image.height) != (w, h) {
                return Err(GpuError::InvalidInput(format!(
                    "the magnified planes are four planes of {w} x {h}, not {} of {} x {}",
                    image.planes, image.width, image.height
                )));
            }
            u.view[2] = image.stride as u32;
            u.view[3] = 0;
            u.c13[3] = image.plane_len() as u32;
            &image.buf
        }
        MagnifyTarget::Rgba8 {
            dst,
            c13,
            background,
        } => {
            let bytes = (4 * w * h) as u64;
            if dst.size() < bytes {
                return Err(GpuError::InvalidInput(format!(
                    "a {} byte buffer does not hold a {w} x {h} RGBA8 view",
                    dst.size()
                )));
            }
            u.view[2] = w as u32;
            u.view[3] = 1;
            u.c13 = [
                u32::from(c13.space == OutputSpace::Rec2020),
                u32::from(c13.dither),
                background_word(*background),
                0,
            ];
            u.org = [c13.origin.0 as u32, c13.origin.1 as u32, 0, 0];
            dst
        }
    };
    let mut table = |label: &str, taps: &[AxisTap]| -> Result<_, GpuError> {
        let words: Vec<TapU> = taps.iter().map(TapU::from).collect();
        let bytes: &[u8] = bytemuck::cast_slice(&words);
        let buf = ctx.pool().acquire(label, bytes.len() as u64, STORAGE_IN)?;
        upload_bytes(ctx, &buf, 0, bytes)?;
        Ok(passes.keep(buf))
    };
    let cols = table("magnify columns", &map.cols)?;
    let rows = table("magnify rows", &map.rows)?;
    let dither = dither_table(ctx)?;
    let ub = passes.uniform("magnify uniform", &u)?;
    passes.dispatch(
        KernelId::Magnify,
        "magnify",
        &[
            (0, ub.binding()),
            (1, src.buf.binding()),
            (2, cols.binding()),
            (3, rows.binding()),
            (4, dither.binding()),
            (5, dst.binding()),
        ],
        (
            w.div_ceil(GROUP as usize) as u32,
            h.div_ceil(GROUP as usize) as u32,
            1,
        ),
    )
}

/// Uploads `image` (planar display-linear RGB) and its optional letterbox `mask` (1 outside the
/// photo) as the source picture.
fn upload_source(
    ctx: &GpuContext,
    image: &RgbImage,
    mask: Option<&[u8]>,
) -> Result<GpuImage, GpuError> {
    let (w, h) = (image.width, image.height);
    match mask {
        Some(m) => {
            if m.len() != w * h {
                return Err(GpuError::InvalidInput(format!(
                    "a mask of {} bytes for a {w} x {h} picture",
                    m.len()
                )));
            }
            let mut data = image.data.clone();
            data.extend(m.iter().map(|&v| if v != 0 { 1.0f32 } else { 0.0 }));
            GpuImage::upload(ctx, "magnify source", w, h, 4, &data, image.place)
        }
        None => GpuImage::upload_rgb(ctx, "magnify source", image),
    }
}

/// Uploads the source picture, magnifies it to planes and reads them back: the magnified RGB,
/// interleaved (`3 w h`, zero on the paper), and the paper mask (`w h`, 1 on the paper), the shape of
/// [`crate::cpu::magnify::magnify`]. The thin wrapper the tests use.
///
/// # Errors
///
/// The errors of [`encode_magnify`] and of the upload and readback.
pub fn magnify_planes(
    ctx: &GpuContext,
    image: &RgbImage,
    mask: Option<&[u8]>,
    map: &MagnifyMap,
) -> Result<(Vec<f32>, Vec<u8>), GpuError> {
    let src = upload_source(ctx, image, mask)?;
    let (w, h) = (map.width(), map.height());
    let out = GpuImage::alloc(ctx, "magnified", w, h, 4, Placement::whole(w, h))?;
    let mut passes = PassList::new(ctx, "magnify planes");
    encode_magnify(&mut passes, &src, map, MagnifyTarget::Planes(&out))?;
    let id = out.encode_read(&mut passes)?;
    let _src = passes.keep(src.buf);
    let done = passes.submit()?.finish()?;
    let planar = out.take(&done, id)?;
    let n = w * h;
    let mut rgb = vec![0.0f32; 3 * n];
    for (i, px) in rgb.chunks_exact_mut(3).enumerate() {
        px[0] = planar[i];
        px[1] = planar[n + i];
        px[2] = planar[2 * n + i];
    }
    let paper = planar[3 * n..]
        .iter()
        .map(|&v| u8::from(v != 0.0))
        .collect();
    Ok((rgb, paper))
}

/// Uploads the source picture, magnifies and encodes it to RGBA8 (`4 w h` bytes) and reads that
/// back. The thin wrapper the tests use.
///
/// # Errors
///
/// The errors of [`encode_magnify`] and of the upload and readback.
pub fn magnify_rgba8(
    ctx: &GpuContext,
    image: &RgbImage,
    mask: Option<&[u8]>,
    map: &MagnifyMap,
    c13: C13,
    background: [u8; 3],
) -> Result<Vec<u8>, GpuError> {
    let src = upload_source(ctx, image, mask)?;
    let bytes = (4 * map.width() * map.height()) as u64;
    let dst = ctx.pool().acquire("magnified rgba8", bytes, STORAGE_OUT)?;
    let mut passes = PassList::new(ctx, "magnify rgba8");
    encode_magnify(
        &mut passes,
        &src,
        map,
        MagnifyTarget::Rgba8 {
            dst: &dst,
            c13,
            background,
        },
    )?;
    let id = passes.read(&dst, 0, bytes)?;
    let _src = passes.keep(src.buf);
    let done = passes.submit()?.finish()?;
    Ok(done.read(id).to_vec())
}

#[cfg(test)]
mod tests {
    use std::mem::{offset_of, size_of};

    use naga::valid::{Capabilities, ValidationFlags, Validator};

    use super::*;
    use crate::device::StorageMode;
    use crate::kernels::shader_source;

    #[test]
    fn the_uniform_block_and_the_tap_have_the_layout_of_the_wgsl_structs() {
        let source = shader_source(StorageMode::F32, KernelId::Magnify);
        let module = naga::front::wgsl::parse_str(&source)
            .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&source)));
        Validator::new(ValidationFlags::all(), Capabilities::empty())
            .validate(&module)
            .expect("valid");
        let find = |name: &str| {
            module
                .types
                .iter()
                .find(|(_, t)| t.name.as_deref() == Some(name))
                .map(|(_, t)| t.inner.clone())
                .unwrap_or_else(|| panic!("the struct {name}"))
        };
        let check = |name: &str, fields: &[(&str, usize)], size: usize| {
            let naga::TypeInner::Struct { members, span } = find(name) else {
                panic!("{name} is not a struct");
            };
            assert_eq!(members.len(), fields.len(), "{name}");
            for (member, (field, offset)) in members.iter().zip(fields) {
                assert_eq!(member.name.as_deref(), Some(*field), "{name}");
                assert_eq!(member.offset as usize, *offset, "offset of {name}.{field}");
            }
            assert_eq!(span as usize, size, "size of {name}");
        };
        check(
            "MagnifyU",
            &[
                ("view", offset_of!(MagnifyU, view)),
                ("src", offset_of!(MagnifyU, src)),
                ("c13", offset_of!(MagnifyU, c13)),
                ("org", offset_of!(MagnifyU, org)),
            ],
            size_of::<MagnifyU>(),
        );
        check(
            "Tap",
            &[
                ("idx", offset_of!(TapU, idx)),
                ("w", offset_of!(TapU, w)),
                ("at", offset_of!(TapU, at)),
                ("inside", offset_of!(TapU, inside)),
                ("pad0", offset_of!(TapU, pad)),
                ("pad1", offset_of!(TapU, pad) + 4),
            ],
            size_of::<TapU>(),
        );
    }

    #[test]
    fn a_tap_goes_up_as_the_shader_reads_it() {
        let t = AxisTap {
            idx: [1, 2, 3, 4],
            weights: [-0.5, 0.25, 1.0, 0.25],
            mask_at: 7,
            inside: true,
        };
        assert_eq!(
            TapU::from(&t),
            TapU {
                idx: [1, 2, 3, 4],
                w: [-0.5, 0.25, 1.0, 0.25],
                at: 7,
                inside: 1,
                pad: [0; 2],
            }
        );
        assert_eq!(background_word([1, 2, 3]), 0xff03_0201);
    }

    #[test]
    fn the_barrier_word_is_plus_zero_with_and_without_a_mask() {
        for planes in [3, 4] {
            let words = source_words(100, 100 * 60, planes);
            assert_eq!(words[3], 0, "+0.0, not -0.0 or anything else");
            assert_eq!(words[2], u32::from(planes == 4));
        }
    }

    #[test]
    fn the_binding_is_the_rgba8_view() {
        let r = Rect::new(0, 0, 5120, 2880);
        assert_eq!(max_binding_bytes(r), 4 * 5120 * 2880);
        assert!(max_binding_bytes(Rect::new(0, 0, 2560, 1440)) < max_binding_bytes(r));
    }
}
