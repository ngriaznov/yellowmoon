//! The export tail (`tail::ExportTail`) against the CPU export of the same picture: the
//! sampler, the colour chain C1 to C12, the Lanczos-3 resize, output sharpening and C13, in the
//! order and with the settings `ym_export` uses.
//!
//! * **Parity.** The codes the tail returns are the CPU export's within [`MAX_CODE_DIFF_8`] (8 bit,
//!   dithered) or [`MAX_CODE_DIFF_16`] (16 bit), and at least [`MIN_EXACT_8`] of the 8-bit codes are
//!   exact, for plain, sharpened, shrunk, enlarged, shrunk and sharpened, straightened (with the
//!   letterbox), wide and wide-gamut exports.
//! * **The cut does not show.** The bytes do not depend on the size of the tiles, and a tile that
//!   reads nothing of the photo may be left out (its pixels are black).
//! * **Order and memory.** The bands come in order and cover the picture once; a tail that has
//!   finished holds nothing, and one that is cancelled gives everything back.
//! * **One tile to the host** (the DNG): the colour stage of a tile read back equals the CPU's.
//! * Poison: every test device fills its buffers with NaN, so a read of what nobody wrote fails.

mod common;
#[path = "export_common/mod.rs"]
mod export_common;

use std::sync::Arc;

use common::TestGpu;
use common::chains;
use export_common::{Scene, ci_note, cpu_codes, diff_codes, exact_scene, rotated_scene};
use ym_color::display::{ColorChain, Frame, Layout, OutputTarget};
use ym_color::encode::OutputSpace;
use ym_gpu::cpu::RgbImage;
use ym_gpu::cpu::output_sharpen::{OutputSharpen, scale_for_edges};
use ym_gpu::device::{GpuError, StoragePreference};
use ym_gpu::kernels::color::{Stop, release_tables};
use ym_gpu::queue::{CancelToken, JobSpec, Priority};
use ym_gpu::tail::{
    Band, BandData, BandFormat, ExportTail, SHARPEN_WINDOW_PIXELS, SLAB_BYTES, TailRoute,
    TailSharpen, TailSpec, colour_tile_to_host,
};

/// The largest difference between a device code and the CPU's, 8 bit. The first measured maximum
/// on Mesa lavapipe over [`configs`] was 1 (the dither flips a code when the f32 colour chain and
/// the f64 one land on two sides of a threshold); the bound is the "one code" of `docs/17-gpu.md`
/// section 12.
const MAX_CODE_DIFF_8: u32 = 1;
/// The share of 8-bit codes that must be exact (`docs/17-gpu.md` section 12: at least 99 %). The
/// first measured minimum over [`configs`] was 0.9999.
const MIN_EXACT_8: f64 = 0.99;
/// The largest difference, 16 bit, of an export that is not sharpened (the first proposal: 8
/// codes). The first measured maximum was 1.
const MAX_CODE_DIFF_16: u32 = 8;
/// Output sharpening amplifies a difference in the colour pass at a few pixels, where its halo
/// limiter decides on a rank of luma values that the f32 and f64 chains order differently (the
/// colour passes differ by at most 4.4e-6 of display light, the sharpening alone by 4.8e-7, the two
/// together by 9.8e-4 at 195 of 14.8 M samples: `docs/algorithms/04` section 10, "the limiter's
/// rank flips"). A sharpened 16-bit export may therefore differ by more than [`MAX_CODE_DIFF_16`]
/// at this share of its codes at most. The first measured share was 9.6e-6 (142 of 14,826,240
/// codes on the wide picture), the bound is twice that, rounded up.
const MAX_FLIP_SHARE_16: f64 = 2e-5;
/// ... and none of them is further off than this (the first measured maximum was 191 codes of
/// 65535, 0.3 %; the bound is twice that).
const MAX_FLIP_DIFF_16: u32 = 382;

struct Config {
    name: &'static str,
    scene: Scene,
    out: (usize, usize),
    sharpen_pct: f64,
    bits16: bool,
    target: OutputTarget,
    tile: usize,
}

fn space_of(target: OutputTarget) -> OutputSpace {
    match target {
        OutputTarget::Rec2020 => OutputSpace::Rec2020,
        _ => OutputSpace::Srgb,
    }
}

fn chain(target: OutputTarget) -> Arc<ColorChain> {
    Arc::new(ColorChain::new(&chains::full(target)).expect("chain"))
}

/// 301 x 1301 is not a multiple of anything: slabs of 512 rows (three of them), tiles that cross
/// slab borders (tile rows of 330 and 256), an odd stride.
fn configs() -> Vec<Config> {
    let exact = |w, h, seed| exact_scene(w, h, 1, 0, seed);
    vec![
        Config {
            name: "plain",
            scene: exact(300, 1300, 1),
            out: (300, 1300),
            sharpen_pct: 0.0,
            bits16: false,
            target: OutputTarget::Srgb,
            tile: 400,
        },
        Config {
            name: "sharpened",
            scene: exact(300, 1300, 2),
            out: (300, 1300),
            sharpen_pct: 100.0,
            bits16: false,
            target: OutputTarget::Srgb,
            tile: 330,
        },
        Config {
            name: "shrunk, 16 bit",
            scene: exact(301, 1301, 3),
            out: (217, 938),
            sharpen_pct: 0.0,
            bits16: true,
            target: OutputTarget::Srgb,
            tile: 512,
        },
        Config {
            name: "shrunk and sharpened",
            scene: exact(301, 1301, 4),
            out: (150, 650),
            sharpen_pct: 100.0,
            bits16: false,
            target: OutputTarget::Srgb,
            tile: 256,
        },
        Config {
            name: "straightened, sharpened",
            scene: rotated_scene(400, 1100, (360, 1000), 6.0, 1.0, 5),
            out: (360, 1000),
            sharpen_pct: 100.0,
            bits16: false,
            target: OutputTarget::Srgb,
            tile: 300,
        },
        Config {
            name: "enlarged, sharpened",
            scene: exact(120, 300, 6),
            out: (260, 640),
            sharpen_pct: 100.0,
            bits16: false,
            target: OutputTarget::Srgb,
            tile: 100,
        },
        Config {
            name: "wide (9504 px), sharpened, 16 bit",
            scene: exact(9504, 520, 7),
            out: (9504, 520),
            sharpen_pct: 100.0,
            bits16: true,
            target: OutputTarget::Srgb,
            tile: 1024,
        },
        Config {
            name: "Rec.2020, strongest sharpening",
            scene: exact(120, 200, 8),
            out: (120, 200),
            sharpen_pct: 200.0,
            bits16: false,
            target: OutputTarget::Rec2020,
            tile: 64,
        },
    ]
}

fn format_of(bits16: bool) -> BandFormat {
    if bits16 {
        BandFormat::Sixteen
    } else {
        BandFormat::Eight { dither: true }
    }
}

fn spec_of(
    sc: &Scene,
    chain: &Arc<ColorChain>,
    out: (usize, usize),
    sharpen_pct: f64,
    space: OutputSpace,
    format: BandFormat,
) -> TailSpec {
    TailSpec {
        rendered: (sc.w, sc.h),
        out,
        chain: Arc::clone(chain),
        grid: sc.grid,
        sharpen: (sharpen_pct > 0.0).then(|| TailSharpen {
            params: OutputSharpen::screen().with_amount_pct(sharpen_pct as f32),
            scale: scale_for_edges(out.0.max(out.1), sc.w.max(sc.h)),
        }),
        space,
        format,
    }
}

/// Feeds the tiles of `sc` in row order, left to right, leaving out the tiles that read nothing of
/// the photo, and declares each tile row complete. `after_row` sees the tail and the bands of each
/// row (to cancel, or to count).
fn feed(
    tail: &mut ExportTail<'_>,
    sc: &Scene,
    tile: usize,
    after_row: impl FnMut(&mut ExportTail<'_>, &[Band]),
) -> Result<Vec<Band>, GpuError> {
    feed_regions(
        tail,
        sc,
        tile,
        |_, _, _, _| (Arc::clone(&sc.camera), (0, 0)),
        after_row,
    )
}

/// [`feed`] with the region of the camera image a tile is given chosen by `region(ox, oy, w, h)`
/// (the render hands the tail the rectangle the tile reads, not the photo).
fn feed_regions(
    tail: &mut ExportTail<'_>,
    sc: &Scene,
    tile: usize,
    region: impl Fn(usize, usize, usize, usize) -> (Arc<RgbImage>, (usize, usize)),
    mut after_row: impl FnMut(&mut ExportTail<'_>, &[Band]),
) -> Result<Vec<Band>, GpuError> {
    let mut bands = Vec::new();
    let mut oy = 0;
    while oy < sc.h {
        let th = tile.min(sc.h - oy);
        let mut ox = 0;
        while ox < sc.w {
            let tw = tile.min(sc.w - ox);
            if !export_common::tile_is_outside(sc, (ox, oy), tw, th) {
                let (image, origin) = region(ox, oy, tw, th);
                tail.add_tile(image, origin, (ox, oy), tw, th)?;
            }
            ox += tw;
        }
        let got = tail.advance(oy + th)?;
        after_row(tail, &got);
        bands.extend(got);
        oy += th;
    }
    bands.extend(tail.finish()?);
    Ok(bands)
}

#[allow(clippy::too_many_arguments)] // one configuration of a run, passed once
fn run_tail(
    dev: &TestGpu,
    sc: &Scene,
    chain: &Arc<ColorChain>,
    out: (usize, usize),
    sharpen_pct: f64,
    format: BandFormat,
    space: OutputSpace,
    tile: usize,
) -> Vec<Band> {
    let route = TailRoute::new(&dev.gpu, JobSpec::new(Priority::Export), CancelToken::new());
    let mut tail =
        ExportTail::new(route, spec_of(sc, chain, out, sharpen_pct, space, format)).expect("tail");
    feed(&mut tail, sc, tile, |_, _| {}).expect("tail run")
}

/// The codes of the bands, in order, checking that they follow each other without a gap.
fn codes(bands: &[Band], out: (usize, usize)) -> (Vec<u32>, bool) {
    let mut next = 0;
    let mut all = Vec::new();
    let mut sixteen = false;
    for b in bands {
        assert_eq!(b.y0, next, "bands follow each other");
        next += b.rows;
        match &b.data {
            BandData::Eight(v) => {
                assert_eq!(v.len(), b.rows * out.0 * 3);
                all.extend(v.iter().map(|&c| u32::from(c)));
            }
            BandData::Sixteen(v) => {
                sixteen = true;
                assert_eq!(v.len(), b.rows * out.0 * 3);
                all.extend(v.iter().map(|&c| u32::from(c)));
            }
        }
    }
    assert_eq!(next, out.1, "the bands cover the picture");
    (all, sixteen)
}

#[test]
fn the_tail_matches_the_cpu_export_in_every_configuration() {
    {
        let Some(dev) = common::gpu() else { return };
        let (mut worst8, mut min_exact8, mut worst16) = (0u32, 1.0f64, 0u32);
        for c in configs() {
            let ch = chain(c.target);
            let space = space_of(c.target);
            let format = format_of(c.bits16);
            let bands = run_tail(
                dev,
                &c.scene,
                &ch,
                c.out,
                c.sharpen_pct,
                format,
                space,
                c.tile,
            );
            let (got, sixteen) = codes(&bands, c.out);
            assert_eq!(sixteen, c.bits16);
            let want = cpu_codes(&c.scene, &ch, c.out, c.sharpen_pct, space, c.bits16);
            let want: Vec<u32> = if c.bits16 {
                want.sixteen
                    .expect("16 bit")
                    .iter()
                    .map(|&v| u32::from(v))
                    .collect()
            } else {
                want.eight
                    .expect("8 bit")
                    .iter()
                    .map(|&v| u32::from(v))
                    .collect()
            };
            let d = diff_codes(&got, &want);
            let over = (0..got.len())
                .filter(|&i| got[i].abs_diff(want[i]) > MAX_CODE_DIFF_16)
                .count();
            // The share over 8 codes is printed for the 16-bit exports too: `MAX_FLIP_SHARE_16`
            // is held by this number, and Metal's value is what a hardware column is made of.
            ci_note(
                "tail parity",
                &format!(
                    "{}: {} codes, {:.4} exact, max {}{} ({})",
                    c.name,
                    d.n,
                    d.exact_share(),
                    d.max,
                    if c.bits16 {
                        format!(
                            ", {over} over {MAX_CODE_DIFF_16} ({:.2e})",
                            over as f64 / d.n as f64
                        )
                    } else {
                        String::new()
                    },
                    dev.gpu.info().name
                ),
            );
            if c.bits16 {
                if c.sharpen_pct > 0.0 {
                    assert!(
                        over as f64 <= MAX_FLIP_SHARE_16 * got.len() as f64
                            && d.max <= MAX_FLIP_DIFF_16,
                        "{}: {over} codes over {MAX_CODE_DIFF_16}, max {}",
                        c.name,
                        d.max
                    );
                } else {
                    assert!(d.max <= MAX_CODE_DIFF_16, "{}: max {}", c.name, d.max);
                }
                worst16 = worst16.max(d.max);
            } else {
                assert!(d.max <= MAX_CODE_DIFF_8, "{}: max {}", c.name, d.max);
                assert!(
                    d.exact_share() >= MIN_EXACT_8,
                    "{}: only {:.4} exact",
                    c.name,
                    d.exact_share()
                );
                worst8 = worst8.max(d.max);
                min_exact8 = min_exact8.min(d.exact_share());
            }
        }
        ci_note(
            "tail parity",
            &format!(
                "worst 8-bit difference {worst8} code, smallest exact share {min_exact8:.4}, worst 16-bit difference {worst16} codes"
            ),
        );
    }
}

/// The export tail runs the colour warp (C9b, `docs/02`) with the rest of the colour pass: an export
/// of a warped chain, shrunk and sharpened, matches the CPU export of the same chain with the bounds
/// of [`the_tail_matches_the_cpu_export_in_every_configuration`], and differs from the export without
/// the warp. The pins sit on the scene's own colours (the C9 chroma of three of its pixels, a third of
/// a turn apart in hue): two moves, one with exposure and split tonal weights, and a lock. First
/// measured on Mesa lavapipe: at most 1 code off, 0.9999 exact; 56 % of the codes differ from the
/// export without the warp.
#[test]
fn the_export_tail_applies_the_colour_warp() {
    use ym_color::warp::{WarpParams, WarpPin, chroma_of};
    let Some(dev) = common::gpu() else { return };
    let sc = exact_scene(240, 360, 1, 0, 23);
    let out = (180, 270);
    let mut params = chains::full(OutputTarget::Srgb);
    let plain = Arc::new(ColorChain::new(&params).expect("chain"));
    let n = sc.camera.width * sc.camera.height;
    let mut colours: Vec<[f64; 2]> = (0..n)
        .map(|i| chroma_of(plain.trace(chains::pixel(&sc.camera, i)).c9_logc4))
        .filter(|c| c[0].hypot(c[1]) > 0.03)
        .collect();
    assert!(
        colours.len() * 20 > n,
        "{} coloured pixels of {n}",
        colours.len()
    );
    colours.sort_by(|a, b| a[1].atan2(a[0]).total_cmp(&b[1].atan2(b[0])));
    let at = |f: f64| colours[(f * (colours.len() - 1) as f64) as usize];
    let (s1, s2, l) = (at(0.2), at(0.53), at(0.86));
    params.warp = WarpParams {
        pins: vec![
            WarpPin {
                range: 0.06,
                ..WarpPin::moving(s1, [s1[0] * 0.4 - s1[1], s1[1] * 0.4 + s1[0]])
            },
            WarpPin {
                range: 0.05,
                shadows: 0.3,
                highlights: 1.0,
                exposure_ev: 0.7,
                ..WarpPin::moving(s2, [s2[0] * 1.6, s2[1] * 1.6])
            },
            WarpPin {
                range: 0.04,
                ..WarpPin::locked_at(l)
            },
        ],
    };
    let warped = Arc::new(ColorChain::new(&params).expect("chain"));
    let (space, format) = (OutputSpace::Srgb, format_of(false));
    let (got, _) = codes(
        &run_tail(dev, &sc, &warped, out, 100.0, format, space, 128),
        out,
    );
    let (without, _) = codes(
        &run_tail(dev, &sc, &plain, out, 100.0, format, space, 128),
        out,
    );
    let want: Vec<u32> = cpu_codes(&sc, &warped, out, 100.0, space, false)
        .eight
        .expect("8 bit")
        .iter()
        .map(|&v| u32::from(v))
        .collect();
    let d = diff_codes(&got, &want);
    let changed = diff_codes(&got, &without);
    ci_note(
        "tail warp",
        &format!(
            "warped export against the CPU: max {}, {:.4} exact; {:.3} of the codes differ from the unwarped export",
            d.max,
            d.exact_share(),
            1.0 - changed.exact_share()
        ),
    );
    assert!(
        1.0 - changed.exact_share() > 0.05,
        "the warp changed only {:.4} of the codes",
        1.0 - changed.exact_share()
    );
    assert!(d.max <= MAX_CODE_DIFF_8, "max {}", d.max);
    assert!(
        d.exact_share() >= MIN_EXACT_8,
        "{:.4} exact",
        d.exact_share()
    );
}

/// The whole picture through the host wrappers of the kernels, one stage after the other and no
/// slabs, chunks or windows: sampler, colour pass, resize, sharpening, C13.
fn device_whole(dev: &TestGpu, c: &Config) -> Vec<u32> {
    use ym_gpu::cpu::RgbImage;
    use ym_gpu::cpu::tiling::Rect;
    use ym_gpu::kernels::PixelData;
    use ym_gpu::kernels::color::{ColorOutput, PreparedChain};
    use ym_gpu::kernels::encode::{C13, Encoding, encode_host};
    use ym_gpu::kernels::output_sharpen::PreparedSharpen;
    use ym_gpu::kernels::resize::resize;
    use ym_gpu::kernels::sampler::sample;

    let camera = Arc::clone(&c.scene.camera);
    let grid = c.scene.grid;
    let (w, h) = (c.scene.w, c.scene.h);
    let out = c.out;
    let ch = chain(c.target);
    let pct = c.sharpen_pct;
    let bits16 = c.bits16;
    let space = space_of(c.target);
    let band = dev
        .gpu
        .run(JobSpec::new(Priority::Export), move |ctx, _| {
            let sampled = sample(ctx, &camera, (0, 0), &grid, (0, 0), w, h)?;
            let prepared = PreparedChain::new(ctx, &ch)?;
            let colour =
                prepared.render(ctx, &sampled.img, (0, 0), ColorOutput::Stage(Stop::C12))?;
            let PixelData::F32Planar(mut planar) = colour.data else {
                return Err(GpuError::InvalidInput("planar f32 expected".into()));
            };
            if let Some(outside) = &sampled.outside {
                let n = w * h;
                for (i, &o) in outside.iter().enumerate() {
                    if o != 0 {
                        for p in 0..3 {
                            planar[p * n + i] = 0.0;
                        }
                    }
                }
            }
            let mut image = RgbImage::from_planar(w, h, planar)?;
            if out != (w, h) {
                image = resize(ctx, &image, out.0, out.1)?;
            }
            if pct > 0.0 {
                let scale = scale_for_edges(out.0.max(out.1), w.max(h));
                let params = OutputSharpen::screen().with_amount_pct(pct as f32);
                image = PreparedSharpen::new(ctx, &params, scale)?.run(ctx, &image)?;
            }
            let encoding = if bits16 {
                Encoding::Rgb16 { bits: 16 }
            } else {
                Encoding::Rgb8 { background: [0; 3] }
            };
            encode_host(
                ctx,
                &image,
                None,
                Rect::whole(out.0, out.1),
                encoding,
                C13 {
                    space,
                    dither: true,
                    origin: (0, 0),
                },
            )
        })
        .expect("whole-frame device chain");
    if bits16 {
        band.bytes
            .chunks_exact(2)
            .map(|b| u32::from(u16::from_le_bytes([b[0], b[1]])))
            .collect()
    } else {
        band.bytes.iter().map(|&b| u32::from(b)).collect()
    }
}

/// The slabs, the windows across their borders, the chunks of the resize and of the sharpening and
/// the order of the passes do not show: the tail's codes are the codes of the same kernels run on
/// the whole picture at once (the sharpening stage is tile-exact, D5 of `docs/17-gpu.md`).
#[test]
fn the_tail_is_the_whole_frame_device_chain_bit_for_bit() {
    let Some(dev) = common::gpu() else { return };
    let mut cases = configs();
    // The wide picture is too slow to run twice; a 3000 px one has two sharpening chunks.
    cases.truncate(6);
    cases.push(Config {
        name: "3000 px, shrunk and sharpened",
        scene: exact_scene(3000, 900, 1, 0, 12),
        out: (2400, 720),
        sharpen_pct: 100.0,
        bits16: true,
        target: OutputTarget::Srgb,
        tile: 1024,
    });
    for c in &cases {
        let ch = chain(c.target);
        let space = space_of(c.target);
        let bands = run_tail(
            dev,
            &c.scene,
            &ch,
            c.out,
            c.sharpen_pct,
            format_of(c.bits16),
            space,
            c.tile,
        );
        let (got, _) = codes(&bands, c.out);
        let want = device_whole(dev, c);
        assert!(
            got == want,
            "{}: the tail differs from the whole-frame chain at {} codes",
            c.name,
            got.iter().zip(&want).filter(|(a, b)| a != b).count()
        );
    }
}

#[test]
fn the_bytes_do_not_depend_on_the_size_of_the_tiles() {
    {
        let Some(dev) = common::gpu() else { return };
        // Shrunk and sharpened, and straightened with a letterbox and sharpening: every stage.
        for c in [&configs()[3], &configs()[4]] {
            let ch = chain(c.target);
            let space = space_of(c.target);
            let format = format_of(c.bits16);
            let reference = run_tail(
                dev,
                &c.scene,
                &ch,
                c.out,
                c.sharpen_pct,
                format,
                space,
                c.tile,
            );
            let (want, _) = codes(&reference, c.out);
            for tile in [97, 200, 513, 4000] {
                let bands = run_tail(
                    dev,
                    &c.scene,
                    &ch,
                    c.out,
                    c.sharpen_pct,
                    format,
                    space,
                    tile,
                );
                let (got, _) = codes(&bands, c.out);
                assert!(
                    got == want,
                    "{}: tiles of {tile} px give other codes than tiles of {}",
                    c.name,
                    c.tile
                );
            }
        }
    }
}

#[test]
fn bands_come_out_as_soon_as_their_rows_and_halo_are_complete() {
    let Some(dev) = common::gpu() else { return };
    // 4000 px wide: a sharpening chunk is 591 core rows, so the 1200 rows come out in two bands,
    // the first as soon as its core and the stage's halo are complete.
    let sc = exact_scene(4000, 1200, 1, 0, 13);
    let ch = chain(OutputTarget::Srgb);
    let route = TailRoute::new(&dev.gpu, JobSpec::new(Priority::Export), CancelToken::new());
    let spec = spec_of(
        &sc,
        &ch,
        (4000, 1200),
        100.0,
        OutputSpace::Srgb,
        format_of(false),
    );
    let mut tail = ExportTail::new(route, spec).expect("tail");
    let mut per_row = Vec::new();
    let mut declared = 0;
    let bands = feed(&mut tail, &sc, 330, |t, got| {
        declared = (declared + 330).min(sc.h);
        // What is encoded is behind what is declared complete (the halo below a chunk is needed),
        // except at the end.
        assert!(
            t.rows_encoded() < declared || declared == sc.h,
            "{} rows encoded after {declared} were declared",
            t.rows_encoded()
        );
        per_row.push(got.iter().map(|b| b.rows).sum::<usize>());
    })
    .expect("run");
    let total: usize = bands.iter().map(|b| b.rows).sum();
    assert_eq!(total, 1200);
    // The first chunk is 591 core rows (a 2.5 Mpx window is 625 rows at this width, less the stage's
    // halo of 17 rows on each side); it needs 608 rows to be complete, which the second tile row
    // brings (660). The second chunk waits for the end.
    assert_eq!(
        per_row,
        [0, 591, 0, 609],
        "rows returned after each tile row"
    );
    assert_eq!(tail.rows_encoded(), 1200);
}

#[test]
fn a_finished_tail_holds_nothing_and_a_cancelled_one_gives_everything_back() {
    {
        // A device of its own: the pool's ledger is the device's, and the other tests of this binary
        // run on the shared one at the same time.
        let Some(gpu) = common::start_unpoisoned(StoragePreference::Auto) else {
            return;
        };
        let dev = &TestGpu {
            mode: gpu.info().storage_mode,
            gpu,
        };
        let c = &configs()[3];
        let ch = chain(c.target);
        let in_use = || {
            dev.gpu
                .run(JobSpec::new(Priority::Export), |ctx, _| {
                    // The colour tables are resident on purpose (they are what makes an unchanged
                    // chain cost nothing); the ledger counts them until they are let go.
                    release_tables(ctx);
                    ctx.pool().trim();
                    Ok(ctx.pool().stats().in_use_bytes)
                })
                .expect("pool")
        };
        let baseline = in_use();
        // A whole run.
        {
            let route =
                TailRoute::new(&dev.gpu, JobSpec::new(Priority::Export), CancelToken::new());
            let spec = spec_of(
                &c.scene,
                &ch,
                c.out,
                c.sharpen_pct,
                space_of(c.target),
                format_of(false),
            );
            let mut tail = ExportTail::new(route, spec).expect("tail");
            let mut peak = 0u64;
            feed(&mut tail, &c.scene, c.tile, |_, _| {
                let in_use = dev
                    .gpu
                    .run(JobSpec::new(Priority::Export), |ctx, _| {
                        Ok(ctx.pool().stats().in_use_bytes)
                    })
                    .expect("pool");
                peak = peak.max(in_use);
            })
            .expect("run");
            let picture = (c.scene.w * c.scene.h * 16) as u64;
            ci_note(
                "tail memory",
                &format!(
                    "{}: {} px, resident at most {} KiB between rows ({}% of the four-plane picture)",
                    c.name,
                    c.scene.w * c.scene.h,
                    (peak - baseline) / 1024,
                    (peak - baseline) * 100 / picture.max(1)
                ),
            );
            drop(tail);
        }
        let after = in_use();
        assert_eq!(after, baseline, "a finished tail leaves nothing in use");
        // A cancelled run: the token is set after the second tile row.
        {
            let token = CancelToken::new();
            let route = TailRoute::new(&dev.gpu, JobSpec::new(Priority::Export), token.clone());
            let spec = spec_of(
                &c.scene,
                &ch,
                c.out,
                c.sharpen_pct,
                space_of(c.target),
                format_of(false),
            );
            let mut tail = ExportTail::new(route, spec).expect("tail");
            let mut rows = 0;
            let result = feed(&mut tail, &c.scene, c.tile, |_, _| {
                rows += 1;
                if rows == 2 {
                    token.cancel();
                }
            });
            assert!(matches!(result, Err(GpuError::Cancelled)), "{result:?}");
            drop(tail);
        }
        let after = in_use();
        assert_eq!(after, baseline, "a cancelled tail leaves nothing in use");
    }
}

#[test]
fn tiles_must_lie_in_the_picture_and_not_behind_the_rows_declared_complete() {
    let Some(dev) = common::gpu() else { return };
    let sc = exact_scene(64, 80, 1, 0, 9);
    let ch = chain(OutputTarget::Srgb);
    let route = TailRoute::new(&dev.gpu, JobSpec::new(Priority::Export), CancelToken::new());
    let spec = spec_of(&sc, &ch, (64, 80), 0.0, OutputSpace::Srgb, format_of(false));
    let mut tail = ExportTail::new(route, spec).expect("tail");
    let refused = |r: Result<(), GpuError>, what: &str| {
        assert!(matches!(r, Err(GpuError::InvalidInput(_))), "{what}: {r:?}");
    };
    let cam = || Arc::clone(&sc.camera);
    refused(
        tail.add_tile(cam(), (0, 0), (60, 0), 8, 8),
        "past the right edge",
    );
    refused(
        tail.add_tile(cam(), (0, 0), (0, 76), 8, 8),
        "past the bottom",
    );
    refused(tail.add_tile(cam(), (0, 0), (0, 0), 0, 8), "no width");
    tail.add_tile(cam(), (0, 0), (0, 0), 64, 40)
        .expect("a tile");
    tail.advance(40).expect("advance");
    refused(
        tail.add_tile(cam(), (0, 0), (0, 10), 64, 8),
        "a tile in rows that were declared complete",
    );
    // An empty picture and a picture too wide for a slab are refused at the start.
    let route = TailRoute::new(&dev.gpu, JobSpec::new(Priority::Export), CancelToken::new());
    let mut empty = spec_of(&sc, &ch, (64, 80), 0.0, OutputSpace::Srgb, format_of(false));
    empty.out = (0, 80);
    assert!(matches!(
        ExportTail::new(route.clone(), empty),
        Err(GpuError::InvalidInput(_))
    ));
    let mut wide = spec_of(&sc, &ch, (64, 80), 0.0, OutputSpace::Srgb, format_of(false));
    wide.rendered = (9_000_000, 80);
    wide.out = (9_000_000, 80);
    assert!(matches!(
        ExportTail::new(route, wide),
        Err(GpuError::BufferTooLarge { .. })
    ));
}

/// The DNG's way: one tile coloured on the device, read back as interleaved f32 and compared with
/// the CPU chain on the same sampled tile.
#[test]
fn one_tile_read_back_is_the_cpu_colour_pass() {
    {
        let Some(dev) = common::gpu() else { return };
        let sc = rotated_scene(300, 260, (300, 260), 8.0, 1.0, 11);
        let ch = chain(OutputTarget::Srgb);
        let route = TailRoute::new(&dev.gpu, JobSpec::new(Priority::Export), CancelToken::new());
        let (off, w, h) = ((0usize, 0usize), 150usize, 130usize);
        let sampled =
            ym_gpu::cpu::sampler::sample_to_camera(&sc.camera, (0, 0), &sc.grid, off, w, h);
        assert!(
            sampled.outside.is_some(),
            "the test tile has letterbox in it"
        );
        for (stop, name) in [(Stop::C12, "C12"), (Stop::DngLinearAwg4, "DNG")] {
            let got = colour_tile_to_host(
                &route,
                Arc::clone(&sc.camera),
                (0, 0),
                sc.grid,
                off,
                w,
                h,
                Arc::clone(&ch),
                stop,
            )
            .expect("tile");
            let frame = Frame::planar(w, h, &sampled.img.data).expect("frame");
            let mut want = vec![0.0f32; 3 * w * h];
            if stop == Stop::C12 {
                ch.render_display_linear_f32(&frame, Layout::Interleaved, &mut want, 1)
                    .expect("cpu");
            } else {
                ch.render_dng_linear_awg4_f32(&frame, Layout::Interleaved, &mut want, 1)
                    .expect("cpu");
            }
            if let Some(outside) = &sampled.outside {
                for (i, &o) in outside.iter().enumerate() {
                    if o != 0 {
                        want[3 * i..3 * i + 3].fill(0.0);
                    }
                }
            }
            assert_eq!(got.len(), want.len());
            let worst = got
                .iter()
                .zip(&want)
                .map(|(&a, &b)| {
                    if a.is_nan() || b.is_nan() {
                        f32::INFINITY
                    } else {
                        (a - b).abs() / b.abs().max(1.0)
                    }
                })
                .fold(0.0f32, f32::max);
            ci_note(
                "tile to host",
                &format!("{name}: worst relative difference {worst:e}"),
            );
            // First measured on Mesa lavapipe: 3.1e-6 (C12) and 3.3e-6 (DNG), relative to the value
            // (at least 1): f32 against the f64 chain. The bound is twice the larger.
            assert!(worst <= 6.7e-6, "{name}: {worst:e}");
        }
    }
}

/// The codes the tail makes of `sc` against the CPU export's, with the bounds of
/// [`the_tail_matches_the_cpu_export_in_every_configuration`]: 8 bit within one code with 99 % exact
/// (a picture of a few pixels: every code but one), 16 bit within [`MAX_CODE_DIFF_16`] unless
/// sharpened (then the limiter's rank flips, at most [`MAX_FLIP_SHARE_16`] of the codes).
fn assert_tail_matches_cpu(
    dev: &TestGpu,
    name: &str,
    sc: &Scene,
    out: (usize, usize),
    pct: f64,
    bits16: bool,
    tile: usize,
) {
    let ch = chain(OutputTarget::Srgb);
    let bands = run_tail(
        dev,
        sc,
        &ch,
        out,
        pct,
        format_of(bits16),
        OutputSpace::Srgb,
        tile,
    );
    let (got, sixteen) = codes(&bands, out);
    assert_eq!(sixteen, bits16, "{name}");
    let want = cpu_codes(sc, &ch, out, pct, OutputSpace::Srgb, bits16);
    let want: Vec<u32> = if bits16 {
        want.sixteen
            .expect("16 bit")
            .iter()
            .map(|&v| u32::from(v))
            .collect()
    } else {
        want.eight
            .expect("8 bit")
            .iter()
            .map(|&v| u32::from(v))
            .collect()
    };
    assert_eq!(got.len(), want.len(), "{name}");
    let d = diff_codes(&got, &want);
    let label = format!(
        "{name}: {}x{} -> {}x{}, {pct} %, {} bit, tile {tile}: max {}, {:.4} exact",
        sc.w,
        sc.h,
        out.0,
        out.1,
        if bits16 { 16 } else { 8 },
        d.max,
        d.exact_share()
    );
    if bits16 {
        let over = (0..got.len())
            .filter(|&i| got[i].abs_diff(want[i]) > MAX_CODE_DIFF_16)
            .count();
        if pct > 0.0 {
            // A handful of codes may flip on a rank. The stage scales R, G and B of a pixel by one
            // ratio, so one flipped pixel moves up to three codes. Software (lavapipe): no small
            // picture flips, floor 2. Hardware (Metal at 6d4d2f4, macos-14 and macos-15 alike):
            // 5 x 1030, 16 bit, 3 of 15,450 codes over 8 (one pixel), at most 32 off, while the
            // sharpening kernel alone matches lavapipe's bits there; floor 6, two pixels.
            let floor = if dev.gpu.info().is_software { 2.0 } else { 6.0 };
            if over > 0 {
                // Printed where a flip happened, so that the next Metal run measures the floor.
                ci_note(
                    "tiny pictures",
                    &format!("{label}: {over} over {MAX_CODE_DIFF_16}, max {}", d.max),
                );
            }
            assert!(
                over as f64 <= (MAX_FLIP_SHARE_16 * got.len() as f64).max(floor)
                    && d.max <= MAX_FLIP_DIFF_16,
                "{label}: {over} over {MAX_CODE_DIFF_16}"
            );
        } else {
            assert!(d.max <= MAX_CODE_DIFF_16, "{label}");
        }
    } else {
        let over = (0..got.len())
            .filter(|&i| got[i].abs_diff(want[i]) > MAX_CODE_DIFF_8)
            .count();
        if pct > 0.0 {
            // A picture this narrow is nothing but borders for the sharpening, and its limiter's
            // rank flips (see MAX_FLIP_SHARE_16) can move a dithered code by more than one. First
            // measured, 301 x 1301 to 7 x 1301: 3 of 27,321 codes over 1 (1.1e-4), at most 2 off.
            // The evidence that it is the stage's sensitivity and not the tail: the same input
            // through the sharpening kernel and the CPU stage differs by 5e-7, the tail is the
            // whole-frame device chain bit for bit (`extreme_resizes_match_the_cpu_export`), and a
            // random relative perturbation of 1e-6 of the CPU's own samples (a quarter of the f32
            // against f64 colour difference) makes 6 codes of the CPU's own export differ by more
            // than 1, up to 3. The bounds are twice the first measured share and the perturbed max.
            assert!(
                d.max <= 3 && over as f64 <= (2.2e-4 * got.len() as f64).max(2.0),
                "{label}: {over} codes over {MAX_CODE_DIFF_8}"
            );
        } else {
            assert!(d.max <= MAX_CODE_DIFF_8, "{label}");
        }
        // On a picture of a handful of pixels one dithered code is more than 1 %.
        let allowed = ((1.0 - MIN_EXACT_8) * got.len() as f64).ceil().max(2.0) as usize;
        assert!(
            d.n - d.exact <= allowed,
            "{label}: {} codes differ",
            d.n - d.exact
        );
    }
}

/// Pictures of a few pixels, one pixel wide or high, odd: the slabs, the chunks, the bands and the
/// sharpening window all degenerate to one piece, and nothing may panic or read what was not written
/// (every test device runs under poison).
#[test]
fn tiny_and_one_pixel_wide_pictures_match_the_cpu_export() {
    let Some(dev) = common::gpu() else { return };
    let sizes = [
        (1, 1),
        (1, 2),
        (2, 1),
        (2, 2),
        (3, 5),
        (1, 37),
        (37, 1),
        (17, 3),
        (64, 1),
        (1, 600),
        (5, 1030),
    ];
    for (i, &(w, h)) in sizes.iter().enumerate() {
        let sc = exact_scene(w, h, 1, 0, 300 + i as u64);
        for pct in [0.0, 100.0] {
            for bits16 in [false, true] {
                assert_tail_matches_cpu(dev, "tiny", &sc, (w, h), pct, bits16, 16);
            }
        }
    }
}

/// The picture shrunk to a single pixel, to a single row or column, and by a large factor in one
/// direction: the taps of the first and last output, a mid image of one column, a vertical chunk
/// that reads the whole picture.
#[test]
fn extreme_resizes_match_the_cpu_export() {
    let Some(dev) = common::gpu() else { return };
    let sc = exact_scene(301, 1301, 1, 0, 400);
    for out in [
        (1, 1),
        (2, 2),
        (1, 650),
        (150, 1),
        (301, 1),
        (1, 1301),
        (7, 1301),
        (300, 1300),
        (301, 1300),
        (302, 1301),
        (400, 5),
    ] {
        // The tail is the whole-frame device chain bit for bit (the cut shows nowhere) ...
        for pct in [0.0, 100.0] {
            let c = Config {
                name: "extreme resize",
                scene: exact_scene(301, 1301, 1, 0, 400),
                out,
                sharpen_pct: pct,
                bits16: false,
                target: OutputTarget::Srgb,
                tile: 300,
            };
            let bands = run_tail(
                dev,
                &c.scene,
                &chain(c.target),
                out,
                pct,
                format_of(false),
                OutputSpace::Srgb,
                c.tile,
            );
            let (got, _) = codes(&bands, out);
            assert!(
                got == device_whole(dev, &c),
                "301x1301 -> {}x{}, {pct} %: the tail differs from the whole-frame chain",
                out.0,
                out.1
            );
        }
        // ... and the CPU export's codes within the bounds above.
        for pct in [0.0, 100.0] {
            assert_tail_matches_cpu(dev, "extreme resize", &sc, out, pct, false, 300);
        }
        assert_tail_matches_cpu(dev, "extreme resize", &sc, out, 0.0, true, 513);
    }
}

/// A straightened picture that is not sharpened and not resized keeps the letterbox plane of the
/// slab in the window C13 reads (the sharpened and the resized pictures read three planes): the
/// corners must come out black, in 8 and 16 bit, and the rest as the CPU's.
#[test]
fn a_straightened_picture_without_sharpening_or_resize_matches_the_cpu_export() {
    let Some(dev) = common::gpu() else { return };
    let sc = rotated_scene(400, 1100, (360, 1000), 6.0, 1.0, 5);
    for bits16 in [false, true] {
        assert_tail_matches_cpu(dev, "letterbox", &sc, (360, 1000), 0.0, bits16, 300);
        assert_tail_matches_cpu(dev, "letterbox", &sc, (360, 1000), 0.0, bits16, 97);
        assert_tail_matches_cpu(dev, "letterbox", &sc, (240, 667), 0.0, bits16, 300);
    }
    // The corner is black.
    let ch = chain(OutputTarget::Srgb);
    let bands = run_tail(
        dev,
        &sc,
        &ch,
        (360, 1000),
        0.0,
        format_of(false),
        OutputSpace::Srgb,
        300,
    );
    let (got, _) = codes(&bands, (360, 1000));
    assert!(
        got[..3].iter().all(|&c| c == 0),
        "the top-left corner is outside the photo"
    );
}

/// The rectangle of `camera` a tile of an identity-geometry picture reads (the tile and a margin of
/// four pixels, which the sampler's bilinear taps stay inside), as the render hands it to the tail.
fn crop_of(
    camera: &Arc<RgbImage>,
) -> impl Fn(usize, usize, usize, usize) -> (Arc<RgbImage>, (usize, usize)) + '_ {
    move |ox, oy, w, h| {
        let (x0, y0) = (ox.saturating_sub(4), oy.saturating_sub(4));
        let (x1, y1) = (
            (ox + w + 4).min(camera.width),
            (oy + h + 4).min(camera.height),
        );
        let (rw, rh) = (x1 - x0, y1 - y0);
        let plane = camera.width * camera.height;
        let mut data = Vec::with_capacity(3 * rw * rh);
        for c in 0..3 {
            for y in y0..y1 {
                let from = c * plane + y * camera.width + x0;
                data.extend_from_slice(&camera.data[from..from + rw]);
            }
        }
        (
            Arc::new(RgbImage::from_planar(rw, rh, data).expect("a region")),
            (x0, y0),
        )
    }
}

/// FNV-1a over the codes of the bands, in order.
fn fingerprint(bands: &[Band]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325_u64;
    let mut mix = |b: u8| {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0100_0000_01b3);
    };
    for b in bands {
        match &b.data {
            BandData::Eight(v) => v.iter().for_each(|&c| mix(c)),
            BandData::Sixteen(v) => v
                .iter()
                .for_each(|&c| c.to_le_bytes().into_iter().for_each(&mut mix)),
        }
    }
    h
}

/// 50 MP (8688 x 5792) through the tail in 1024 px tiles: what the device holds between two tile rows and at its peak,
/// against the 800 MB the four planes of the rendered picture would be, and the same codes twice.
/// Ignored: it makes a 600 MB camera image and runs the colour chain over 50 million pixels (about
/// a minute on lavapipe in a test build); `mac-gpu.yml` runs it on Metal.
#[test]
#[ignore = "50 MP: run with --ignored (mac-gpu.yml runs it on Metal)"]
fn fifty_megapixels_leave_a_few_slabs_on_the_device_and_the_same_bytes_twice() {
    let Some(gpu) = common::start_unpoisoned(StoragePreference::Auto) else {
        return;
    };
    let dev = TestGpu {
        mode: gpu.info().storage_mode,
        gpu,
    };
    let (bw, bh) = (8688, 5792);
    let sc = exact_scene(bw, bh, 1, 0, 50);
    let ch = chain(OutputTarget::Srgb);
    let picture = (bw * bh * 16) as u64;
    for (name, out, pct, bits16) in [
        ("full size, sharpened, 16 bit", (bw, bh), 100.0, true),
        (
            "long edge 4000, sharpened, 8 bit",
            (4000, 2667),
            100.0,
            false,
        ),
    ] {
        let mut prints = Vec::new();
        for pass in 0..2 {
            dev.gpu
                .run(JobSpec::new(Priority::Export), |ctx, _| {
                    ctx.pool().reset_peaks();
                    Ok(())
                })
                .expect("reset");
            let started = std::time::Instant::now();
            let route =
                TailRoute::new(&dev.gpu, JobSpec::new(Priority::Export), CancelToken::new());
            let spec = spec_of(&sc, &ch, out, pct, OutputSpace::Srgb, format_of(bits16));
            let mut tail = ExportTail::new(route, spec).expect("tail");
            let mut between_rows = 0u64;
            let bands = feed_regions(&mut tail, &sc, 1024, crop_of(&sc.camera), |_, _| {
                let in_use = dev
                    .gpu
                    .run(JobSpec::new(Priority::Export), |ctx, _| {
                        Ok(ctx.pool().stats().in_use_bytes)
                    })
                    .expect("pool");
                between_rows = between_rows.max(in_use);
            })
            .expect("tail run");
            let seconds = started.elapsed().as_secs_f64();
            let peak = dev
                .gpu
                .run(JobSpec::new(Priority::Export), |ctx, _| {
                    Ok(ctx.pool().stats().peak_in_use_bytes)
                })
                .expect("pool");
            let rows: usize = bands.iter().map(|b| b.rows).sum();
            assert_eq!(rows, out.1, "{name}: the bands cover the picture");
            ci_note(
                "50 MP tail",
                &format!(
                    "{name}, pass {pass}: {:.1} s; resident between tile rows at most {} MiB, peak {} MiB ({}% of the {} MiB four-plane picture)",
                    seconds,
                    between_rows >> 20,
                    peak >> 20,
                    peak * 100 / picture,
                    picture >> 20
                ),
            );
            // The design: a few slabs of 128 MiB at most, one window of the sharpening's scratch (76
            // bytes a pixel) and the tiles and bands in flight, never the picture.
            let design = 3 * SLAB_BYTES + SHARPEN_WINDOW_PIXELS as u64 * 76 + (64 << 20);
            assert!(
                peak <= design,
                "{name}: peak {peak} against the design's {design}"
            );
            assert!(
                peak < picture,
                "{name}: peak {peak} of a {picture} B picture"
            );
            prints.push(fingerprint(&bands));
        }
        assert_eq!(prints[0], prints[1], "{name}: the same codes twice");
    }
}
