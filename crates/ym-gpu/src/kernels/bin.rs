//! The k x k binning of the fit view on the device, and the sigma it leaves (G-07, `docs/01`: the
//! reduced path).
//!
//! The CPU reference is `ym-core`'s `bin_mosaic` (one pixel per `k x k` block with all three
//! colours: green the mean of the block's greens, red and blue the block's mean of their lattice
//! interpolated linearly, with the lattice weights of [`lattice_weights`]) followed by the
//! vignetting gain at the binned pixels (`gain_plane`), and `binned_sigma`. The device does both in
//! one thread per output pixel, from a **slab** of the normalised mosaic (full width, some rows with
//! two rows of context, [`slab_rows`]) so the frame's normalised mosaic (200 MB at 50 MP) never exists:
//! [`encode_bin`] writes the rows `rows` of the output and a caller makes the slabs one after the
//! other. Indices beyond the frame's edge mirror with the parity kept, as the CPU's do, and every
//! product and sum keeps the CPU's order and rounding (`opaque`).

use std::ops::Range;

use bytemuck::{Pod, Zeroable};

use super::prep::{VigU, VignetteTable};
use super::shader::KernelId;
use super::util::GROUP;
use crate::cpu::tiling::Rect;
use crate::cpu::{Cfa, CfaColor, Mosaic, Placement, RgbImage};
use crate::device::{GpuContext, GpuError};
use crate::image::GpuImage;
use crate::stream::PassList;

/// The largest block size the kernel takes (the weights of a colour fit 12 floats).
pub const MAX_K: usize = 16;

/// The uniform block of `bin.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BinU {
    /// The runtime zero of the barrier (always +0.0 bits).
    zero: [u32; 4],
    /// Slab width, height, row stride, `k`.
    dims: [u32; 4],
    /// Frame width, height, the slab's first frame row, output width.
    frame: [u32; 4],
    /// First output row, output rows, output row stride, output plane length.
    out: [u32; 4],
    /// The plane index at (0, 0), (1, 0), (0, 1), (1, 1) of the frame.
    cfa: [u32; 4],
    /// Red lattice phase (x, y), blue lattice phase (x, y).
    lat: [u32; 4],
    /// Weight counts: red x, red y, blue x, blue y.
    nw: [u32; 4],
    /// Vignetting on, sigma plane width, height, step.
    flags: [u32; 4],
    /// The green count and the noise factor of the binning.
    f: [f32; 4],
    /// The four weight arrays, three vec4s each.
    wts: [[f32; 4]; 12],
    vig: VigU,
}

/// What the binning needs to know about the frame.
#[derive(Clone, Debug)]
pub struct BinParams {
    /// The block size: even, at most [`MAX_K`].
    pub k: usize,
    /// The pattern at the frame's first photosite.
    pub cfa: Cfa,
    /// Frame width in photosites.
    pub frame_w: usize,
    /// Frame height in photosites.
    pub frame_h: usize,
    /// The vignetting profile, multiplied in at the binned pixels, or `None`.
    pub vignette: Option<VignetteTable>,
}

impl BinParams {
    /// Output width.
    #[must_use]
    pub fn out_w(&self) -> usize {
        self.frame_w / self.k
    }

    /// Output height.
    #[must_use]
    pub fn out_h(&self) -> usize {
        self.frame_h / self.k
    }

    pub(crate) fn check(&self) -> Result<(), GpuError> {
        if self.k < 2 || !self.k.is_multiple_of(2) || self.k > MAX_K {
            return Err(GpuError::InvalidInput(format!(
                "bin: k = {} (even, 2 to {MAX_K})",
                self.k
            )));
        }
        if self.out_w() == 0 || self.out_h() == 0 {
            return Err(GpuError::InvalidInput(format!(
                "bin: a {} x {} frame holds no block of {}",
                self.frame_w, self.frame_h, self.k
            )));
        }
        if (self.frame_w as u64).pow(2) + (self.frame_h as u64).pow(2) >= 1 << 32 {
            return Err(GpuError::InvalidInput(
                "bin: the frame is too large for the 32-bit radius of the vignetting gain".into(),
            ));
        }
        Ok(())
    }
}

fn hat_integral(u: f64) -> f64 {
    if u <= -2.0 {
        0.0
    } else if u <= 0.0 {
        (u + 2.0) * (u + 2.0) / 4.0
    } else if u <= 2.0 {
        2.0 - (2.0 - u) * (2.0 - u) / 4.0
    } else {
        2.0
    }
}

/// The weights of the sites of one colour's lattice along one axis for a block of `k` photosites
/// (`ym-core`'s `lattice_weights`, the same expressions): the colour's lattice has a site every two
/// photosites at parity `q`, and a site's weight is the mean over the block of the triangle that
/// interpolates the lattice linearly. They sum to one.
#[must_use]
pub fn lattice_weights(k: usize, q: usize) -> Vec<f32> {
    (-1..=(k / 2) as isize)
        .map(|j| {
            let t = 2.0 * j as f64 + q as f64 + 0.5;
            ((hat_integral(k as f64 - t) - hat_integral(-t)) / k as f64) as f32
        })
        .collect()
}

/// The frame rows a slab needs to bin the output rows `rows`: two rows of context on each side
/// (clamped to the frame), the start made even so that the slab keeps the superpixel grid. The
/// result is `first..end` in frame rows.
#[must_use]
pub fn slab_rows(k: usize, rows: &Range<usize>, frame_h: usize) -> Range<usize> {
    let first = (rows.start * k).saturating_sub(2) & !1;
    let end = (rows.end * k + 2).min(frame_h);
    first..end
}

/// The parity `(x, y)` of the lattice of `colour` in the 2 x 2 cell of `cfa`.
fn lattice_parity(cfa: Cfa, colour: CfaColor) -> Result<(usize, usize), GpuError> {
    (0..2usize)
        .flat_map(|y| (0..2usize).map(move |x| (x, y)))
        .find(|&(x, y)| cfa.color_at(x, y) == colour)
        .ok_or_else(|| {
            GpuError::InvalidInput(format!("bin: the pattern has no {colour:?} photosite"))
        })
}

fn base_uniform(p: &BinParams, noise_factor: f32) -> Result<BinU, GpuError> {
    p.check()?;
    let (rx, ry) = lattice_parity(p.cfa, CfaColor::Red)?;
    let (bx, by) = lattice_parity(p.cfa, CfaColor::Blue)?;
    let mut wts = [[0.0f32; 4]; 12];
    let mut nw = [0u32; 4];
    for (which, (q, slot)) in [(rx, 0usize), (ry, 1), (bx, 2), (by, 3)]
        .into_iter()
        .enumerate()
    {
        let _ = slot;
        let w = lattice_weights(p.k, q);
        nw[which] = w.len() as u32;
        for (i, v) in w.iter().enumerate() {
            wts[which * 3 + i / 4][i % 4] = *v;
        }
    }
    let plane = |dx: usize, dy: usize| p.cfa.color_at(dx, dy).plane() as u32;
    Ok(BinU {
        zero: [0; 4],
        dims: [0, 0, 0, p.k as u32],
        frame: [p.frame_w as u32, p.frame_h as u32, 0, p.out_w() as u32],
        out: [0; 4],
        cfa: [plane(0, 0), plane(1, 0), plane(0, 1), plane(1, 1)],
        lat: [rx as u32, ry as u32, bx as u32, by as u32],
        nw,
        flags: [
            u32::from(p.vignette.is_some()),
            0,
            0,
            (p.k / 2).max(1) as u32,
        ],
        f: [(p.k * p.k / 2) as f32, noise_factor, 0.0, 0.0],
        wts,
        vig: VigU::new(p.vignette.as_ref(), p.frame_w, p.frame_h),
    })
}

/// The largest storage binding (bytes) a bin pass needs for a slab of `rows` rows of a frame
/// `frame_w` wide: the slab's mosaic.
#[must_use]
pub fn max_binding_bytes(frame_w: usize, rows: usize) -> u64 {
    GpuImage::buffer_bytes(frame_w, rows, 1)
}

/// Records the binning of the output rows `rows` (`0..out_h`) from `slab`, a one-plane image of the
/// full frame width that starts at frame row `slab.place.y0` and holds the rows [`slab_rows`] gives,
/// into the three-plane `out` (`out_w x out_h`). Red and blue are multiplied by the vignetting gain
/// at the binned pixel when the parameters have a profile; so is green.
///
/// # Errors
///
/// [`GpuError::InvalidInput`] for a slab that is not one plane of the frame's width, does not hold the
/// rows the output rows need, an output of the wrong size, or parameters the kernel refuses; shader
/// errors.
pub fn encode_bin(
    passes: &mut PassList<'_>,
    slab: &GpuImage,
    params: &BinParams,
    rows: Range<usize>,
    out: &GpuImage,
) -> Result<(), GpuError> {
    let mut u = base_uniform(params, 0.0)?;
    if slab.planes != 1 || slab.width != params.frame_w {
        return Err(GpuError::InvalidInput(format!(
            "bin: the slab is {} plane(s) of {} columns, the frame is {} wide",
            slab.planes, slab.width, params.frame_w
        )));
    }
    let need = slab_rows(params.k, &rows, params.frame_h);
    let have = slab.place.y0..slab.place.y0 + slab.height;
    if rows.is_empty()
        || rows.end > params.out_h()
        || have.start > need.start
        || have.end < need.end
    {
        return Err(GpuError::InvalidInput(format!(
            "bin: the slab holds frame rows {have:?}, the output rows {rows:?} need {need:?}"
        )));
    }
    if (out.width, out.height, out.planes) != (params.out_w(), params.out_h(), 3) {
        return Err(GpuError::InvalidInput(format!(
            "bin: the output is {} x {} x {}, expected {} x {} x 3",
            out.width,
            out.height,
            out.planes,
            params.out_w(),
            params.out_h()
        )));
    }
    u.dims = [
        slab.width as u32,
        slab.height as u32,
        slab.stride as u32,
        params.k as u32,
    ];
    u.frame[2] = slab.place.y0 as u32;
    u.out = [
        rows.start as u32,
        (rows.end - rows.start) as u32,
        out.stride as u32,
        out.plane_len() as u32,
    ];
    // The output rows are written at their own row index: the dispatch starts at `rows.start`.
    let ub = passes.uniform("bin uniform", &u)?;
    passes.dispatch(
        KernelId::Bin,
        "bin",
        &[
            (0, ub.binding()),
            (1, slab.buf.binding()),
            (2, out.buf.binding()),
        ],
        (
            (params.out_w() as u32).div_ceil(GROUP),
            ((rows.end - rows.start) as u32).div_ceil(GROUP),
            1,
        ),
    )?;
    Ok(())
}

/// Records the sigma of the binned image: for every output pixel the root of the mean square of the
/// `step x step` cells of the half-resolution plane `sigma` it covers, times `noise_factor` (the
/// binning's `binning_noise_factor`) and the vignetting gain at the pixel. The result is
/// `out_w x out_h`, one plane, **without** the sigma scale (the denoise applies it).
///
/// # Errors
///
/// [`GpuError::InvalidInput`] for a plane that is not one plane or parameters the kernel refuses;
/// the pool's errors; shader errors.
pub fn encode_bin_sigma(
    passes: &mut PassList<'_>,
    sigma: &GpuImage,
    params: &BinParams,
    noise_factor: f32,
) -> Result<GpuImage, GpuError> {
    let ctx = passes.context();
    let mut u = base_uniform(params, noise_factor)?;
    if sigma.planes != 1 {
        return Err(GpuError::InvalidInput(format!(
            "bin sigma: the plane has {} planes",
            sigma.planes
        )));
    }
    let (ow, oh) = (params.out_w(), params.out_h());
    let out = GpuImage::alloc(ctx, "binned sigma", ow, oh, 1, Placement::whole(ow, oh))?;
    u.dims = [
        sigma.width as u32,
        sigma.height as u32,
        sigma.stride as u32,
        params.k as u32,
    ];
    u.flags[1] = sigma.width as u32;
    u.flags[2] = sigma.height as u32;
    u.out = [0, oh as u32, out.stride as u32, out.plane_len() as u32];
    let ub = passes.uniform("bin sigma uniform", &u)?;
    passes.dispatch(
        KernelId::BinSigma,
        "bin sigma",
        &[
            (0, ub.binding()),
            (1, sigma.buf.binding()),
            (2, out.buf.binding()),
        ],
        ((ow as u32).div_ceil(GROUP), (oh as u32).div_ceil(GROUP), 1),
    )?;
    Ok(out)
}

/// The binning of a whole host mosaic through the device, in slabs of `slab_out_rows` output rows
/// (0: one slab). For the parity tests.
///
/// # Errors
///
/// As [`encode_bin`], plus the readbacks'.
pub fn bin_mosaic(
    ctx: &GpuContext,
    m: &Mosaic,
    k: usize,
    vignette: Option<VignetteTable>,
    slab_out_rows: usize,
) -> Result<RgbImage, GpuError> {
    let params = BinParams {
        k,
        cfa: m.cfa,
        frame_w: m.width,
        frame_h: m.height,
        vignette,
    };
    params.check()?;
    let (ow, oh) = (params.out_w(), params.out_h());
    let out = GpuImage::alloc(ctx, "bin out", ow, oh, 3, Placement::whole(ow, oh))?;
    let step = if slab_out_rows == 0 {
        oh
    } else {
        slab_out_rows
    };
    let mut r = 0;
    while r < oh {
        let rows = r..(r + step).min(oh);
        let need = slab_rows(k, &rows, m.height);
        let sub = m.extract(Rect::new(0, need.start, m.width, need.end - need.start));
        let slab = GpuImage::upload(
            ctx,
            "bin slab",
            sub.width,
            sub.height,
            1,
            &sub.data,
            Placement::whole(m.width, m.height).offset(0, need.start),
        )?;
        let mut passes = PassList::new(ctx, "bin");
        encode_bin(&mut passes, &slab, &params, rows.clone(), &out)?;
        passes.submit()?.finish()?;
        r = rows.end;
    }
    let mut passes = PassList::new(ctx, "bin read");
    let read = out.encode_read(&mut passes)?;
    let done = passes.submit()?.finish()?;
    let data = out.take(&done, read)?;
    Ok(RgbImage {
        width: ow,
        height: oh,
        place: Placement::whole(ow, oh),
        data,
    })
}

/// The WGSL of a bin-module kernel.
pub(super) fn body(id: KernelId) -> String {
    let mut s = String::new();
    s.push_str(include_str!("../shaders/front_opaque.wgsl"));
    s.push_str(include_str!("../shaders/front_exact.wgsl"));
    s.push_str(include_str!("../shaders/front_vig.wgsl"));
    s.push_str(include_str!("../shaders/bin.wgsl"));
    let entry = match id {
        KernelId::BinSigma => "bin_sigma_main",
        _ => "bin_main",
    };
    s.push_str(&format!(
        "\n@compute @workgroup_size({GROUP}, {GROUP}, 1)\n\
         fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{ {entry}(gid); }}\n"
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
        for id in [KernelId::Bin, KernelId::BinSigma] {
            let (fields, span) = uniform_layout(id, "pu");
            let want = [
                ("zero", offset_of!(BinU, zero)),
                ("dims", offset_of!(BinU, dims)),
                ("frame", offset_of!(BinU, frame)),
                ("out", offset_of!(BinU, out)),
                ("cfa", offset_of!(BinU, cfa)),
                ("lat", offset_of!(BinU, lat)),
                ("nw", offset_of!(BinU, nw)),
                ("flags", offset_of!(BinU, flags)),
                ("f", offset_of!(BinU, f)),
                ("wts", offset_of!(BinU, wts)),
                ("vig", offset_of!(BinU, vig)),
            ];
            assert_eq!(fields.len(), want.len(), "{id:?}");
            for ((name, offset), (want_name, want_offset)) in fields.iter().zip(want) {
                assert_eq!(name, want_name);
                assert_eq!(*offset as usize, want_offset, "{id:?}: offset of {name}");
            }
            assert_eq!(span as usize, size_of::<BinU>(), "{id:?}");
        }
    }

    #[test]
    fn the_lattice_weights_sum_to_one_and_fit_the_uniform() {
        for k in [2usize, 4, 8, 16] {
            for q in 0..2 {
                let w = lattice_weights(k, q);
                assert_eq!(w.len(), k / 2 + 2);
                assert!((w.iter().map(|v| f64::from(*v)).sum::<f64>() - 1.0).abs() < 1e-6);
                assert!(w.len() <= 12);
            }
        }
    }

    #[test]
    fn a_slab_has_two_rows_of_context_and_an_even_start() {
        assert_eq!(slab_rows(4, &(3..5), 100), 10..22);
        assert_eq!(slab_rows(4, &(0..2), 100), 0..10);
        assert_eq!(slab_rows(4, &(23..25), 100), 90..100);
        assert_eq!(slab_rows(2, &(1..2), 100), 0..6);
    }
}
