//! C13 on the device: the output transfer function, the quantiser, the blue-noise dither and the
//! letterbox of a display-linear image (`docs/02-colour-pipeline.md`, stage C13; `docs/gpu-plan.md`
//! G-04). It turns the f32 result of the colour chain and of output sharpening into what a consumer
//! reads:
//!
//! * [`Encoding::Rgba8`]: one word per pixel, the viewport's and the thumbnails' format, with the
//!   letterbox painted where the source's mask plane says the pixel is outside the photo;
//! * [`Encoding::Rgb8`]: interleaved bytes `r g b r g b ...`, the rows padded to four pixels
//!   ([`row_words`]), the format of the JPEG, PNG and TIFF 8-bit bands;
//! * [`Encoding::Rgb16`]: interleaved little-endian 16-bit samples with 10 to 16 significant bits,
//!   the rows padded to two pixels, the format of the 16-bit bands.
//!
//! The dither is the CPU's (`ym_color::encode::dither_threshold`) at the absolute position of the
//! pixel in the picture, so a band, a tile and the whole picture give the same bytes. The functions
//! are those of the colour chain's own C13 (`shaders/c13.wgsl`), and the sharpening kernel fuses the
//! same ones into its last pass; `tests/gpu_pipeline.rs` pins all three together.
//!
//! The entry point is [`encode_c13`], which records one pass into a
//! [`PassList`](crate::stream::PassList); [`encode_host`] is the thin host wrapper the tests use.

use bytemuck::{Pod, Zeroable};
use ym_color::encode::OutputSpace;

use super::color::dither_table;
use super::shader::KernelId;
use super::util::{STORAGE_OUT, even};
use crate::cpu::RgbImage;
use crate::cpu::tiling::Rect;
use crate::device::{GpuContext, GpuError};
use crate::image::GpuImage;
use crate::pool::PooledBuffer;
use crate::stream::PassList;

/// The layout the pass writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Encoding {
    /// RGBA8, one word per pixel (red in the low byte, alpha 255); a pixel the source's mask marks
    /// as outside the photo gets `background`.
    Rgba8 {
        /// The letterbox colour.
        background: [u8; 3],
    },
    /// Interleaved RGB bytes (the letterbox, if any, is `background`).
    Rgb8 {
        /// The letterbox colour.
        background: [u8; 3],
    },
    /// Interleaved RGB 16-bit samples with `bits` (1 to 16) significant bits; never dithered. The
    /// letterbox is black.
    Rgb16 {
        /// Significant bits.
        bits: u32,
    },
}

impl Encoding {
    fn kind(self) -> u32 {
        match self {
            Encoding::Rgba8 { .. } => 1,
            Encoding::Rgb8 { .. } => 3,
            Encoding::Rgb16 { .. } => 4,
        }
    }

    /// Pixels one thread writes: 1, 4 or 2.
    fn pixels_per_thread(self) -> usize {
        match self {
            Encoding::Rgba8 { .. } => 1,
            Encoding::Rgb8 { .. } => 4,
            Encoding::Rgb16 { .. } => 2,
        }
    }
}

/// C13's inputs that are not the pixels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct C13 {
    /// The output colour space (its transfer function).
    pub space: OutputSpace,
    /// Whether the 8-bit codes are dithered; otherwise they are rounded. Ignored by
    /// [`Encoding::Rgb16`].
    pub dither: bool,
    /// The absolute position of the window's pixel (0, 0) in the picture, which selects the dither
    /// texel.
    pub origin: (usize, usize),
}

/// Words one destination row of a `width`-pixel window takes: `width` for RGBA8, and three words for
/// every four (RGB8) or two (RGB16) pixels, rounded up.
pub fn row_words(encoding: Encoding, width: usize) -> usize {
    match encoding {
        Encoding::Rgba8 { .. } => width,
        Encoding::Rgb8 { .. } => width.div_ceil(4) * 3,
        Encoding::Rgb16 { .. } => width.div_ceil(2) * 3,
    }
}

/// Bytes of a destination buffer for `rows` rows of a `width`-pixel window.
pub fn band_bytes(encoding: Encoding, width: usize, rows: usize) -> u64 {
    (row_words(encoding, width) * rows * 4) as u64
}

/// The size of the largest storage binding an encode pass over the `window` makes (K4 of
/// `docs/gpu-plan.md`): the four-plane source, 16 bytes a pixel.
pub fn max_binding_bytes(window: Rect) -> u64 {
    16 * even(window.w) as u64 * window.h as u64
}

/// The uniform block of `encode.wgsl` (`EncodeU`): 20 words.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
struct EncodeU {
    width: u32,
    height: u32,
    src_stride: u32,
    src_plane: u32,
    src_x: u32,
    src_y: u32,
    dst_stride: u32,
    dst_x: u32,
    dst_y: u32,
    kind: u32,
    space: u32,
    dither: u32,
    has_mask: u32,
    bg: u32,
    org_x: u32,
    org_y: u32,
    bits: u32,
    pad: [u32; 3],
}

/// The WGSL of the encode kernel: the C13 helpers and the pass.
pub(super) fn body() -> String {
    let mut s = String::from(include_str!("../shaders/c13.wgsl"));
    s.push('\n');
    s.push_str(include_str!("../shaders/encode.wgsl"));
    s
}

fn background_word(bg: [u8; 3]) -> u32 {
    u32::from(bg[0]) | (u32::from(bg[1]) << 8) | (u32::from(bg[2]) << 16) | 0xff00_0000
}

/// Records the C13 pass over the `window` of `src` (a display-linear image of three planes, or four
/// with the letterbox mask) into `dst`: `rows` rows of [`row_words`] words each, the window's first
/// row at row `dst_row` (and, for [`Encoding::Rgba8`], its first pixel at pixel `dst_x`; the
/// interleaved encodings start at the left edge).
///
/// `dst` must stay alive until the list has finished. Nothing is read back.
///
/// # Errors
///
/// [`GpuError::InvalidInput`] when the window is empty or outside `src`, `src` is not a three- or
/// four-plane image, `dst` is too small, the bits of an [`Encoding::Rgb16`] are outside 1 to 16, or
/// the window needs more workgroups than the device allows.
#[allow(clippy::too_many_arguments)] // the natural argument list; a struct would only move them
pub fn encode_c13(
    passes: &mut PassList<'_>,
    src: &GpuImage,
    window: Rect,
    encoding: Encoding,
    c13: C13,
    dst: &PooledBuffer,
    dst_x: usize,
    dst_row: usize,
) -> Result<(), GpuError> {
    if !matches!(src.planes, 3 | 4) {
        return Err(GpuError::InvalidInput(format!(
            "C13 reads a three- or four-plane image, not {} planes",
            src.planes
        )));
    }
    if window.is_empty() || window.x1() > src.width || window.y1() > src.height {
        return Err(GpuError::InvalidInput(format!(
            "the window {window:?} is empty or outside the {} x {} source",
            src.width, src.height
        )));
    }
    if let Encoding::Rgb16 { bits } = encoding
        && !(1..=16).contains(&bits)
    {
        return Err(GpuError::InvalidInput(format!(
            "a 16-bit band needs 1 to 16 bits, got {bits}"
        )));
    }
    if dst_x != 0 && !matches!(encoding, Encoding::Rgba8 { .. }) {
        return Err(GpuError::InvalidInput(
            "an interleaved band starts at the left edge of its rows".into(),
        ));
    }
    let stride_words = if matches!(encoding, Encoding::Rgba8 { .. }) {
        // An RGBA8 destination is as wide as its pixels reach.
        dst_x + window.w
    } else {
        row_words(encoding, window.w)
    };
    let need = ((dst_row + window.h) * stride_words * 4) as u64;
    if dst.size() < need {
        return Err(GpuError::InvalidInput(format!(
            "a {} byte band does not hold {} rows of {stride_words} words",
            dst.size(),
            dst_row + window.h
        )));
    }
    encode_c13_strided(
        passes,
        src,
        window,
        encoding,
        c13,
        dst,
        stride_words,
        dst_x,
        dst_row,
    )
}

/// [`encode_c13`] with an explicit row stride in words (an RGBA8 view wider than the window).
#[allow(clippy::too_many_arguments)] // the natural argument list; a struct would only move them
pub fn encode_c13_strided(
    passes: &mut PassList<'_>,
    src: &GpuImage,
    window: Rect,
    encoding: Encoding,
    c13: C13,
    dst: &PooledBuffer,
    dst_stride_words: usize,
    dst_x: usize,
    dst_row: usize,
) -> Result<(), GpuError> {
    let ctx = passes.context();
    let need = ((dst_row + window.h) * dst_stride_words * 4) as u64;
    if dst.size() < need {
        return Err(GpuError::InvalidInput(format!(
            "a {} byte band does not hold {} rows of {dst_stride_words} words",
            dst.size(),
            dst_row + window.h
        )));
    }
    let limit = ctx.info().limits.max_compute_workgroups_per_dimension as usize;
    let gx = window
        .w
        .div_ceil(encoding.pixels_per_thread())
        .div_ceil(16);
    let gy = window.h.div_ceil(16);
    if gx > limit || gy > limit {
        return Err(GpuError::InvalidInput(format!(
            "a {} x {} window needs more workgroups than the device allows ({limit})",
            window.w, window.h
        )));
    }
    let (bg, bits) = match encoding {
        Encoding::Rgba8 { background } | Encoding::Rgb8 { background } => {
            (background_word(background), 8)
        }
        Encoding::Rgb16 { bits } => (0, bits),
    };
    let u = EncodeU {
        width: window.w as u32,
        height: window.h as u32,
        src_stride: src.stride as u32,
        src_plane: src.plane_len() as u32,
        src_x: window.x as u32,
        src_y: window.y as u32,
        dst_stride: dst_stride_words as u32,
        dst_x: dst_x as u32,
        dst_y: dst_row as u32,
        kind: encoding.kind(),
        space: u32::from(c13.space == OutputSpace::Rec2020),
        dither: u32::from(c13.dither && !matches!(encoding, Encoding::Rgb16 { .. })),
        has_mask: u32::from(src.planes == 4),
        bg,
        org_x: c13.origin.0 as u32,
        org_y: c13.origin.1 as u32,
        bits,
        pad: [0; 3],
    };
    let ub = passes.uniform("encode uniform", &u)?;
    let dither = dither_table(ctx)?;
    passes.dispatch(
        KernelId::Encode,
        "encode c13",
        &[
            (0, ub.binding()),
            (1, src.buf.binding()),
            (2, dither.binding()),
            (3, dst.binding()),
        ],
        (gx.max(1) as u32, gy.max(1) as u32, 1),
    )
}

/// What [`encode_host`] returns, with the row padding removed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HostBand {
    /// Width in pixels.
    pub width: usize,
    /// Height in pixels.
    pub height: usize,
    /// RGBA8: `4 w h` bytes; RGB8: `3 w h` bytes; RGB16: `3 w h` little-endian samples (`6 w h`
    /// bytes), all row-major.
    pub bytes: Vec<u8>,
}

/// Uploads a display-linear `image` (and, if given, a letterbox `mask` with 1 where the pixel is
/// outside the photo), runs [`encode_c13`] over the `window` of it and reads the band back with the
/// row padding removed. The thin wrapper the tests use.
///
/// # Errors
///
/// The errors of [`encode_c13`] and of the upload and readback.
pub fn encode_host(
    ctx: &GpuContext,
    image: &RgbImage,
    mask: Option<&[u8]>,
    window: Rect,
    encoding: Encoding,
    c13: C13,
) -> Result<HostBand, GpuError> {
    let (w, h) = (image.width, image.height);
    let src = match mask {
        Some(m) => {
            if m.len() != w * h {
                return Err(GpuError::InvalidInput(format!(
                    "a mask of {} bytes for a {w} x {h} image",
                    m.len()
                )));
            }
            let mut data = image.data.clone();
            data.extend(m.iter().map(|&v| if v != 0 { 1.0f32 } else { 0.0 }));
            GpuImage::upload(ctx, "encode input", w, h, 4, &data, image.place)?
        }
        None => GpuImage::upload_rgb(ctx, "encode input", image)?,
    };
    let words = row_words(encoding, window.w);
    let bytes = band_bytes(encoding, window.w, window.h);
    let dst = ctx.pool().acquire("encode band", bytes, STORAGE_OUT)?;
    let mut passes = PassList::new(ctx, "encode c13");
    encode_c13(&mut passes, &src, window, encoding, c13, &dst, 0, 0)?;
    let id = passes.read(&dst, 0, bytes)?;
    let _src = passes.keep(src.buf);
    let done = passes.submit()?.finish()?;
    let raw = done.read(id);
    let (bpp, row_bytes) = match encoding {
        Encoding::Rgba8 { .. } => (4, window.w * 4),
        Encoding::Rgb8 { .. } => (3, window.w * 3),
        Encoding::Rgb16 { .. } => (6, window.w * 6),
    };
    let mut out = Vec::with_capacity(window.w * window.h * bpp);
    for y in 0..window.h {
        let start = y * words * 4;
        out.extend_from_slice(&raw[start..start + row_bytes]);
    }
    Ok(HostBand {
        width: window.w,
        height: window.h,
        bytes: out,
    })
}

#[cfg(test)]
mod tests {
    use std::mem::{offset_of, size_of};

    use naga::valid::{Capabilities, ValidationFlags, Validator};

    use super::*;
    use crate::device::StorageMode;
    use crate::kernels::shader_source;

    #[test]
    fn rows_are_padded_to_whole_groups() {
        let rgba = Encoding::Rgba8 {
            background: [0; 3],
        };
        let rgb8 = Encoding::Rgb8 {
            background: [0; 3],
        };
        let rgb16 = Encoding::Rgb16 { bits: 16 };
        assert_eq!(row_words(rgba, 7), 7);
        assert_eq!([1, 4, 5, 8].map(|w| row_words(rgb8, w)), [3, 3, 6, 6]);
        assert_eq!([1, 2, 3, 4].map(|w| row_words(rgb16, w)), [3, 3, 6, 6]);
        assert_eq!(band_bytes(rgb8, 5, 2), 48);
    }

    #[test]
    fn the_uniform_block_has_the_layout_of_the_wgsl_struct() {
        let source = shader_source(StorageMode::F32, KernelId::Encode);
        let module = naga::front::wgsl::parse_str(&source)
            .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&source)));
        Validator::new(ValidationFlags::all(), Capabilities::empty())
            .validate(&module)
            .expect("valid");
        let var = module
            .global_variables
            .iter()
            .map(|(_, v)| v)
            .find(|v| v.name.as_deref() == Some("eu"))
            .expect("the uniform eu");
        let naga::TypeInner::Struct { members, span } = &module.types[var.ty].inner else {
            panic!("eu is not a struct");
        };
        let fields: [(&str, usize); 20] = [
            ("width", offset_of!(EncodeU, width)),
            ("height", offset_of!(EncodeU, height)),
            ("src_stride", offset_of!(EncodeU, src_stride)),
            ("src_plane", offset_of!(EncodeU, src_plane)),
            ("src_x", offset_of!(EncodeU, src_x)),
            ("src_y", offset_of!(EncodeU, src_y)),
            ("dst_stride", offset_of!(EncodeU, dst_stride)),
            ("dst_x", offset_of!(EncodeU, dst_x)),
            ("dst_y", offset_of!(EncodeU, dst_y)),
            ("kind", offset_of!(EncodeU, kind)),
            ("space", offset_of!(EncodeU, space)),
            ("dither", offset_of!(EncodeU, dither)),
            ("has_mask", offset_of!(EncodeU, has_mask)),
            ("bg", offset_of!(EncodeU, bg)),
            ("org_x", offset_of!(EncodeU, org_x)),
            ("org_y", offset_of!(EncodeU, org_y)),
            ("bits", offset_of!(EncodeU, bits)),
            ("pad0", offset_of!(EncodeU, pad)),
            ("pad1", offset_of!(EncodeU, pad) + 4),
            ("pad2", offset_of!(EncodeU, pad) + 8),
        ];
        assert_eq!(members.len(), fields.len());
        for (member, (name, offset)) in members.iter().zip(fields) {
            assert_eq!(member.name.as_deref(), Some(name));
            assert_eq!(member.offset as usize, offset, "offset of {name}");
        }
        assert_eq!(*span as usize, size_of::<EncodeU>());
    }
}
