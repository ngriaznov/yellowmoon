//! R6 bilateral denoise on the GPU (G-06), and the noise-sigma plane of a tile that this stage, the
//! R7 colour filter and Richardson-Lucy read.
//!
//! [`encode_bilateral`] is the device-in, device-out entry point (`docs/gpu-plan.md` section 4.3); the
//! host wrapper [`bilateral`] uploads an image, runs it and reads the core back, for the parity tests.
//! The CPU reference is [`crate::cpu::denoise::bilateral_whole`]: a 5 x 5 bilateral filter in the
//! opponent space `(Y, R - Y, B - Y)` with the spatial weights of `BilateralParams` and a range
//! bandwidth `h = strength * sqrt(6) * sigma` read from the noise sigma of the centre pixel
//! (`docs/05` section R6). The shader is `shaders/bilateral.wgsl`.
//!
//! # The noise sigma of a tile
//!
//! The photo's sigma map exists at half resolution (one value per 2 x 2 cell, `docs/06`) and the CPU
//! chain expands it once to a full-resolution map of 200 MB at 50 MP (`full_sigma_map` in
//! `ym-core`: each cell's sigma times the vignetting gain at the cell's centre, copied to its four
//! photosites). The device does not need that map. [`SigmaHalf`] is the half plane and the cell gains
//! as plain data; [`encode_sigma`] expands the rectangle of a tile into a one-plane [`GpuImage`] with
//! the same rule and the same single multiplication, so the plane holds the CPU map's bits for every
//! cell (`tests/gpu_denoise.rs` compares every cell of rectangles at odd origins, and the last cell of
//! odd-sized frames, whose cell index is clamped). A host plane or one scalar are the other two
//! sources ([`SigmaInput`]).
//!
//! # What runs where
//!
//! One dispatch per stage over the pixels of the core. The window of a pixel is read with the mirror
//! of the input rectangle's edge, as the CPU's padded planes are, so a tile whose rectangle carries
//! the stage's 2 px of real context gives the bits of the whole-frame run (D5). The sum order is the
//! CPU's (window rows top to bottom, each left to right); the sums are `f32` where the CPU's are `f32`
//! too, so the only differences are the `exp` of the device against the CPU's, the division, and a
//! tap whose exponent is within rounding of 9.
//!
//! # Memory
//!
//! Nothing besides the input, the sigma plane and the output: 7 f32 per core pixel at most.

use bytemuck::{Pod, Zeroable};

use super::shader::KernelId;
use super::util::{GROUP, storage_buffer};
use crate::cpu::denoise::BilateralParams;
use crate::cpu::tiling::Rect;
use crate::cpu::{LUMA, Placement, RgbImage};
use crate::device::{GpuContext, GpuError};
use crate::image::GpuImage;
use crate::stream::PassList;

/// The largest window radius the kernel takes (the default filter has radius 2).
pub const MAX_RADIUS_PX: usize = 4;

/// The half-resolution noise sigma of a photo and the vignetting gain at the centre of each of its
/// cells, as plain data (`docs/gpu-plan.md` section 4.3): the pieces `ym-core` already has
/// (`SigmaPlane` and `gain_plane`). Cell `(cx, cy)` covers the photosites `(2 cx, 2 cy)` to
/// `(2 cx + 1, 2 cy + 1)`; the last cell also covers the odd edge of an odd-sized frame.
#[derive(Clone, Copy, Debug)]
pub struct SigmaHalf<'a> {
    /// Width in cells.
    pub width: usize,
    /// Height in cells.
    pub height: usize,
    /// `width * height` sigmas, row-major.
    pub data: &'a [f32],
    /// The gain of every cell (`width * height`, the multiplier R3 applies at the cell's centre), or
    /// `None` for a photo without a vignetting profile (a gain of exactly 1).
    pub gains: Option<&'a [f32]>,
}

impl SigmaHalf<'_> {
    /// Checks the lengths.
    ///
    /// # Errors
    ///
    /// [`GpuError::InvalidInput`] for an empty plane or a data or gain slice of the wrong length.
    pub fn validate(&self) -> Result<(), GpuError> {
        let n = self.width * self.height;
        if self.width == 0 || self.height == 0 {
            return Err(GpuError::InvalidInput(
                "the half-resolution sigma plane is empty".into(),
            ));
        }
        if self.data.len() != n || self.gains.is_some_and(|g| g.len() != n) {
            return Err(GpuError::InvalidInput(format!(
                "a {} x {} sigma plane needs {n} values and as many gains, got {} and {:?}",
                self.width,
                self.height,
                self.data.len(),
                self.gains.map(<[f32]>::len)
            )));
        }
        Ok(())
    }

    /// The sigma of photosite `(x, y)` of the frame by the rule of the CPU's `full_sigma_map`: the
    /// cell `(x / 2, y / 2)` clamped to the plane, times its gain. The reference the device's
    /// expansion is tested against.
    #[must_use]
    pub fn at(&self, x: usize, y: usize) -> f32 {
        let cy = (y / 2).min(self.height - 1);
        let cx = (x / 2).min(self.width - 1);
        let g = self.gains.map_or(1.0, |g| g[cy * self.width + cx]);
        self.data[cy * self.width + cx] * g
    }
}

/// Where the noise sigma of a tile comes from.
#[derive(Clone, Copy, Debug)]
pub enum SigmaInput<'a> {
    /// The same sigma at every pixel (the chain's scalar sigma).
    Scalar(f32),
    /// One sigma per pixel of the tile, row-major, `width * height` of the tile's rectangle (the
    /// tile of a full-resolution map the host already has).
    Plane(&'a [f32]),
    /// The half-resolution plane, expanded on the device (the cells the rectangle reads are cut here).
    Half(&'a SigmaHalf<'a>),
    /// The cells of one rectangle, already cut ([`SigmaTile`]): what a job that moves to the device
    /// thread carries.
    Tile(&'a SigmaTile),
}

/// The cells of the half-resolution sigma plane (and their gains) that one rectangle of the frame
/// reads, owned: the form of [`SigmaHalf`] that a device job carries across threads. A tile of `n`
/// pixels holds about `n / 4` cells (twice that with the gains).
#[derive(Clone, Debug)]
pub struct SigmaTile {
    rect: Rect,
    /// First cell column and row of the cut, its width and height.
    cells: [u32; 4],
    /// Last cell column and row of the whole plane.
    last: [u32; 2],
    data: Vec<f32>,
    gains: Option<Vec<f32>>,
}

impl SigmaTile {
    /// Cuts the cells that the photosites of `rect` (frame coordinates) read.
    ///
    /// # Errors
    ///
    /// [`GpuError::InvalidInput`] for an empty rectangle or a half plane of the wrong length.
    pub fn cut(half: &SigmaHalf<'_>, rect: Rect) -> Result<SigmaTile, GpuError> {
        half.validate()?;
        if rect.is_empty() {
            return Err(GpuError::InvalidInput(
                "the sigma cells of an empty rectangle".into(),
            ));
        }
        let (last_x, last_y) = (half.width - 1, half.height - 1);
        let cx0 = (rect.x / 2).min(last_x);
        let cy0 = (rect.y / 2).min(last_y);
        let cx1 = ((rect.x1() - 1) / 2).min(last_x);
        let cy1 = ((rect.y1() - 1) / 2).min(last_y);
        let (cw, ch) = (cx1 - cx0 + 1, cy1 - cy0 + 1);
        let cut = |src: &[f32]| -> Vec<f32> {
            let mut out = Vec::with_capacity(cw * ch);
            for cy in cy0..=cy1 {
                out.extend_from_slice(&src[cy * half.width + cx0..cy * half.width + cx1 + 1]);
            }
            out
        };
        Ok(SigmaTile {
            rect,
            cells: [cx0 as u32, cy0 as u32, cw as u32, ch as u32],
            last: [last_x as u32, last_y as u32],
            data: cut(half.data),
            gains: half.gains.map(cut),
        })
    }

    /// The rectangle of the frame the cut was made for.
    #[must_use]
    pub fn rect(&self) -> Rect {
        self.rect
    }

    /// Bytes held.
    #[must_use]
    pub fn bytes(&self) -> usize {
        4 * (self.data.len() + self.gains.as_ref().map_or(0, Vec::len))
    }
}

/// The uniform block of `sigma_expand.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SgU {
    /// Frame `x`, `y`, width and height of the rectangle.
    rect: [u32; 4],
    /// First cell column and row of the uploaded cells, their width and height.
    cells: [u32; 4],
    /// Last cell column and row of the whole plane, has gains, mode (0 scalar, 1 cells).
    limit: [u32; 4],
    /// Row stride of the output plane.
    dst: [u32; 4],
    /// The scalar.
    scalar: [f32; 4],
}

/// Records the noise sigma of `rect` (the tile's rectangle in frame coordinates) as a new one-plane
/// image of `rect.w x rect.h`: the plane every sigma-reading stage of the tile binds. A
/// [`SigmaInput::Plane`] is uploaded as it is; a scalar and the half-resolution plane are written by
/// one dispatch.
///
/// # Errors
///
/// [`GpuError::InvalidInput`] for an empty rectangle, a plane of the wrong length or a rectangle
/// outside the frame the half plane covers (the clamped cell index would hide a wrong origin);
/// [`GpuError::OutOfBudget`] and friends from the pool; shader errors.
pub fn encode_sigma(
    passes: &mut PassList<'_>,
    sigma: &SigmaInput<'_>,
    rect: Rect,
) -> Result<GpuImage, GpuError> {
    let ctx = passes.context();
    if rect.is_empty() {
        return Err(GpuError::InvalidInput(
            "the sigma plane of an empty rectangle".into(),
        ));
    }
    let place = Placement::whole(rect.w, rect.h);
    let mut u = SgU {
        rect: [rect.x as u32, rect.y as u32, rect.w as u32, rect.h as u32],
        cells: [0; 4],
        limit: [0; 4],
        dst: [0; 4],
        scalar: [0.0; 4],
    };
    // The cells the rectangle reads, as a rectangle of the half plane, cut here so that a tile uploads
    // a quarter of its own size.
    let (cell_data, cell_gain) = match *sigma {
        SigmaInput::Plane(plane) => {
            if plane.len() != rect.w * rect.h {
                return Err(GpuError::InvalidInput(format!(
                    "a sigma plane of {} values for a {} x {} rectangle",
                    plane.len(),
                    rect.w,
                    rect.h
                )));
            }
            return GpuImage::upload(ctx, "sigma plane", rect.w, rect.h, 1, plane, place);
        }
        SigmaInput::Scalar(v) => {
            u.scalar[0] = v;
            (vec![0.0f32], vec![0.0f32])
        }
        SigmaInput::Half(half) => {
            let tile = SigmaTile::cut(half, rect)?;
            return encode_sigma_tile(passes, &tile, rect);
        }
        SigmaInput::Tile(tile) => return encode_sigma_tile(passes, tile, rect),
    };
    let out = GpuImage::alloc(ctx, "sigma tile", rect.w, rect.h, 1, place)?;
    u.dst = [out.stride as u32, 0, 0, 0];
    let data = passes.keep(storage_buffer(
        ctx,
        "sigma cells",
        bytemuck::cast_slice(&cell_data),
    )?);
    let gains = passes.keep(storage_buffer(
        ctx,
        "sigma cell gains",
        bytemuck::cast_slice(&cell_gain),
    )?);
    let ub = passes.uniform("sigma uniform", &u)?;
    passes.dispatch(
        KernelId::SigmaExpand,
        "sigma expand",
        &[
            (0, ub.binding()),
            (1, data.binding()),
            (2, gains.binding()),
            (3, out.buf.binding()),
        ],
        (
            (rect.w as u32).div_ceil(GROUP),
            (rect.h as u32).div_ceil(GROUP),
            1,
        ),
    )?;
    Ok(out)
}

/// The expansion of a cut ([`SigmaTile`]) into the plane of `rect`.
fn encode_sigma_tile(
    passes: &mut PassList<'_>,
    tile: &SigmaTile,
    rect: Rect,
) -> Result<GpuImage, GpuError> {
    let ctx = passes.context();
    if tile.rect != rect {
        return Err(GpuError::InvalidInput(format!(
            "the sigma cells were cut for {:?}, the plane is for {rect:?}",
            tile.rect
        )));
    }
    let out = GpuImage::alloc(
        ctx,
        "sigma tile",
        rect.w,
        rect.h,
        1,
        Placement::whole(rect.w, rect.h),
    )?;
    let u = SgU {
        rect: [rect.x as u32, rect.y as u32, rect.w as u32, rect.h as u32],
        cells: tile.cells,
        limit: [
            tile.last[0],
            tile.last[1],
            u32::from(tile.gains.is_some()),
            1,
        ],
        dst: [out.stride as u32, 0, 0, 0],
        scalar: [0.0; 4],
    };
    let data = passes.keep(storage_buffer(
        ctx,
        "sigma cells",
        bytemuck::cast_slice(&tile.data),
    )?);
    let gains = passes.keep(storage_buffer(
        ctx,
        "sigma cell gains",
        bytemuck::cast_slice(tile.gains.as_deref().unwrap_or(&[1.0])),
    )?);
    let ub = passes.uniform("sigma uniform", &u)?;
    passes.dispatch(
        KernelId::SigmaExpand,
        "sigma expand",
        &[
            (0, ub.binding()),
            (1, data.binding()),
            (2, gains.binding()),
            (3, out.buf.binding()),
        ],
        (
            (rect.w as u32).div_ceil(GROUP),
            (rect.h as u32).div_ceil(GROUP),
            1,
        ),
    )?;
    Ok(out)
}

/// The expansion of [`encode_sigma`] on the host: upload, one dispatch, read back, dense
/// (`rect.w * rect.h`). For the tests that compare it with the CPU's map cell by cell.
///
/// # Errors
///
/// As [`encode_sigma`], plus the readback's.
pub fn expand_sigma(
    ctx: &GpuContext,
    sigma: &SigmaInput<'_>,
    rect: Rect,
) -> Result<Vec<f32>, GpuError> {
    let mut passes = PassList::new(ctx, "sigma tile");
    let plane = encode_sigma(&mut passes, sigma, rect)?;
    let read = plane.encode_read(&mut passes)?;
    let done = passes.submit()?.finish()?;
    plane.take(&done, read)
}

/// The uniform block of `bilateral.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BilU {
    /// Input width, height, row stride, plane length.
    dims: [u32; 4],
    /// Core `x`, `y`, width and height in the input.
    core: [u32; 4],
    /// Output row stride, output plane length, sigma row stride, radius.
    dst: [u32; 4],
    /// The input's origin in the sigma plane.
    sig: [u32; 4],
    /// `strength * sqrt(6)`, the sigma scale.
    k: [f32; 4],
    /// The luma weights.
    luma: [f32; 4],
}

/// The spatial weights of the window, with the CPU's expression (`exp(-d^2 / (2 s^2))` in `f32`).
fn spatial_weights(p: &BilateralParams) -> Vec<f32> {
    let r = p.radius_px;
    let side = 2 * r + 1;
    let mut spatial = vec![0.0f32; side * side];
    for dy in 0..side {
        for dx in 0..side {
            let d2 = ((dx as isize - r as isize).pow(2) + (dy as isize - r as isize).pow(2)) as f32;
            spatial[dy * side + dx] = (-0.5 * d2 / (p.spatial_sigma_px * p.spatial_sigma_px)).exp();
        }
    }
    spatial
}

/// Checks the filter parameters.
fn check_params(p: &BilateralParams) -> Result<(), GpuError> {
    if p.radius_px > MAX_RADIUS_PX {
        return Err(GpuError::InvalidInput(format!(
            "the bilateral window radius is {} px, the kernel takes at most {MAX_RADIUS_PX}",
            p.radius_px
        )));
    }
    if !(p.spatial_sigma_px.is_finite() && p.spatial_sigma_px > 0.0)
        || !(p.strength.is_finite() && p.strength >= 0.0)
    {
        return Err(GpuError::InvalidInput(format!(
            "bilateral parameters {p:?}: the spatial sigma must be positive and the strength not negative"
        )));
    }
    Ok(())
}

/// The largest storage binding (bytes) a bilateral stage needs for a compute rectangle of `rect`
/// pixels: the three planes of the input (the output and the sigma plane are smaller). A test checks
/// it against the 128 MiB design binding at [`crate::chain::MAX_PHYS_AREA`] (`docs/gpu-plan.md` K4).
#[must_use]
pub fn max_binding_bytes(rect: Rect) -> u64 {
    GpuImage::buffer_bytes(rect.w, rect.h, 3)
}

/// Records the R6 bilateral denoise of the `core` rectangle of `input` (device in, device out) and
/// returns the core as a new three-plane image whose placement is the input's moved to the core.
///
/// `input` is the compute rectangle (its edge mirrors, so a core away from the frame's edge needs
/// `params.radius_px` real pixels around it); `sigma` is the tile's sigma plane ([`encode_sigma`]) and
/// `sigma_origin` the position of the input's pixel `(0, 0)` in it; the sigma used at a pixel is the
/// plane's value times `scale` (the network scale `(L / L_auto)^1.5` of `detail::network_sigma_scale`;
/// the CPU applies it to the plane before the filter). A pixel whose bandwidth is not above 1e-12
/// (a zero sigma, a zero scale) is copied.
///
/// # Errors
///
/// [`GpuError::InvalidInput`] for an input that is not three planes, an empty core or one outside the
/// input, a sigma plane that does not cover the core, a scale that is not finite, or parameters the
/// kernel refuses; the pool's errors; shader errors.
pub fn encode_bilateral(
    passes: &mut PassList<'_>,
    input: &GpuImage,
    core: Rect,
    sigma: &GpuImage,
    sigma_origin: (usize, usize),
    scale: f32,
    params: &BilateralParams,
) -> Result<GpuImage, GpuError> {
    let ctx = passes.context();
    check_params(params)?;
    if input.planes != 3 {
        return Err(GpuError::InvalidInput(format!(
            "bilateral denoise: the input has {} planes, not 3",
            input.planes
        )));
    }
    if core.is_empty() || core.x1() > input.width || core.y1() > input.height {
        return Err(GpuError::InvalidInput(format!(
            "bilateral denoise: the core {core:?} is empty or outside the {} x {} input",
            input.width, input.height
        )));
    }
    if sigma.planes != 1
        || sigma_origin.0 + core.x1() > sigma.width
        || sigma_origin.1 + core.y1() > sigma.height
    {
        return Err(GpuError::InvalidInput(format!(
            "bilateral denoise: the {} x {} sigma plane at {sigma_origin:?} does not cover the core {core:?}",
            sigma.width, sigma.height
        )));
    }
    if !scale.is_finite() {
        return Err(GpuError::InvalidInput(format!(
            "bilateral denoise: the sigma scale is {scale}"
        )));
    }
    let out = GpuImage::alloc(
        ctx,
        "bilateral rgb",
        core.w,
        core.h,
        3,
        input.place.offset(core.x, core.y),
    )?;
    let spatial = passes.keep(storage_buffer(
        ctx,
        "bilateral weights",
        bytemuck::cast_slice(&spatial_weights(params)),
    )?);
    let root6 = 6f32.sqrt();
    let u = BilU {
        dims: [
            input.width as u32,
            input.height as u32,
            input.stride as u32,
            input.plane_len() as u32,
        ],
        core: [core.x as u32, core.y as u32, core.w as u32, core.h as u32],
        dst: [
            out.stride as u32,
            out.plane_len() as u32,
            sigma.stride as u32,
            params.radius_px as u32,
        ],
        sig: [sigma_origin.0 as u32, sigma_origin.1 as u32, 0, 0],
        k: [params.strength * root6, scale, 0.0, 0.0],
        luma: [LUMA[0], LUMA[1], LUMA[2], 0.0],
    };
    let ub = passes.uniform("bilateral uniform", &u)?;
    passes.dispatch(
        KernelId::Bilateral,
        "bilateral denoise",
        &[
            (0, ub.binding()),
            (1, spatial.binding()),
            (2, input.buf.binding()),
            (3, sigma.buf.binding()),
            (4, out.buf.binding()),
        ],
        (
            (core.w as u32).div_ceil(GROUP),
            (core.h as u32).div_ceil(GROUP),
            1,
        ),
    )?;
    Ok(out)
}

/// The bilateral denoise on the host: uploads `img`, expands `sigma` for its rectangle in the frame
/// (`img.place`), runs [`encode_bilateral`] on `core` and reads the core back. `sigma` is indexed like
/// the image (a [`SigmaInput::Plane`] is `img.width * img.height` long), `scale` multiplies it as
/// [`encode_bilateral`] says. The GPU counterpart of
/// [`bilateral_whole`](crate::cpu::denoise::bilateral_whole) when `core` is the whole image.
///
/// # Errors
///
/// As [`encode_bilateral`] and [`encode_sigma`], plus the readback's.
pub fn bilateral(
    ctx: &GpuContext,
    img: &RgbImage,
    core: Rect,
    sigma: &SigmaInput<'_>,
    scale: f32,
    params: &BilateralParams,
) -> Result<RgbImage, GpuError> {
    let input = GpuImage::upload_rgb(ctx, "bilateral input", img)?;
    let mut passes = PassList::new(ctx, "bilateral denoise");
    let rect = Rect::new(img.place.x0, img.place.y0, img.width, img.height);
    let plane = encode_sigma(&mut passes, sigma, rect)?;
    let out = encode_bilateral(&mut passes, &input, core, &plane, (0, 0), scale, params)?;
    let read = out.encode_read(&mut passes)?;
    let done = passes.submit()?.finish()?;
    let data = out.take(&done, read)?;
    Ok(RgbImage {
        width: out.width,
        height: out.height,
        place: out.place,
        data,
    })
}

/// The WGSL of a bilateral-module kernel: the shader file plus its `main`.
pub(super) fn body(id: KernelId) -> String {
    match id {
        KernelId::SigmaExpand => include_str!("../shaders/sigma_expand.wgsl").to_string(),
        _ => {
            let mut s = String::from(include_str!("../shaders/bilateral.wgsl"));
            s.push_str(&format!(
                "\n@compute @workgroup_size({GROUP}, {GROUP}, 1)\n\
                 fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{ bilateral_filter(gid); }}\n"
            ));
            s
        }
    }
}

#[cfg(test)]
mod tests {
    use std::mem::{offset_of, size_of};

    use naga::valid::{Capabilities, ValidationFlags, Validator};

    use super::*;
    use crate::device::StorageMode;
    use crate::kernels::shader_source;

    fn uniform_layout(id: KernelId, name: &str) -> (Vec<(String, u32)>, u32) {
        let source = shader_source(StorageMode::F32, id);
        let module = naga::front::wgsl::parse_str(&source)
            .unwrap_or_else(|e| panic!("{id:?}: {}", e.emit_to_string(&source)));
        Validator::new(ValidationFlags::all(), Capabilities::all())
            .validate(&module)
            .unwrap_or_else(|e| panic!("{id:?}: {e:?}"));
        let var = module
            .global_variables
            .iter()
            .map(|(_, v)| v)
            .find(|v| v.name.as_deref() == Some(name))
            .unwrap_or_else(|| panic!("{id:?}: no uniform `{name}`"));
        let naga::TypeInner::Struct { members, span } = &module.types[var.ty].inner else {
            panic!("`{name}` is not a struct");
        };
        (
            members
                .iter()
                .map(|m| (m.name.clone().unwrap_or_default(), m.offset))
                .collect(),
            *span,
        )
    }

    #[test]
    fn the_uniform_blocks_have_the_layout_of_their_rust_structs() {
        let (fields, span) = uniform_layout(KernelId::Bilateral, "bp");
        let want = [
            ("dims", offset_of!(BilU, dims)),
            ("core", offset_of!(BilU, core)),
            ("dst", offset_of!(BilU, dst)),
            ("sig", offset_of!(BilU, sig)),
            ("k", offset_of!(BilU, k)),
            ("luma", offset_of!(BilU, luma)),
        ];
        assert_eq!(fields.len(), want.len());
        for ((name, offset), (want_name, want_offset)) in fields.iter().zip(want) {
            assert_eq!(name, want_name);
            assert_eq!(*offset as usize, want_offset, "offset of {name}");
        }
        assert_eq!(span as usize, size_of::<BilU>());

        let (fields, span) = uniform_layout(KernelId::SigmaExpand, "sp");
        let want = [
            ("rect", offset_of!(SgU, rect)),
            ("cells", offset_of!(SgU, cells)),
            ("limit", offset_of!(SgU, limit)),
            ("dst", offset_of!(SgU, dst)),
            ("scalar", offset_of!(SgU, scalar)),
        ];
        assert_eq!(fields.len(), want.len());
        for ((name, offset), (want_name, want_offset)) in fields.iter().zip(want) {
            assert_eq!(name, want_name);
            assert_eq!(*offset as usize, want_offset, "offset of {name}");
        }
        assert_eq!(span as usize, size_of::<SgU>());
    }

    #[test]
    fn the_sigma_rule_is_the_cpus_clamped_cell_times_gain() {
        let data: Vec<f32> = (0..6).map(|i| 0.01 * (i + 1) as f32).collect();
        let gains: Vec<f32> = (0..6).map(|i| 1.0 + 0.1 * i as f32).collect();
        let half = SigmaHalf {
            width: 3,
            height: 2,
            data: &data,
            gains: Some(&gains),
        };
        // 3 x 2 cells cover 6 x 4 photosites; an odd frame's extra column/row reads the last cell.
        assert_eq!(half.at(0, 0), data[0] * gains[0]);
        assert_eq!(half.at(3, 3), data[4] * gains[4]);
        assert_eq!(half.at(5, 3), data[5] * gains[5]);
        assert_eq!(half.at(6, 4), data[5] * gains[5], "clamped to the plane");
        let plain = SigmaHalf {
            gains: None,
            ..half
        };
        assert_eq!(plain.at(2, 1), data[1]);
        assert!(
            SigmaHalf {
                width: 3,
                height: 3,
                ..half
            }
            .validate()
            .is_err()
        );
    }
}
