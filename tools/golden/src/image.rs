//! Pictures for the comparison: reading the 16-bit PNG that `ym render` writes, a hash of its pixels, and the
//! CIEDE2000 difference of two pictures on a 1/4-resolution downsample (`docs/13-testing.md`, Golden images:
//! "ΔE00 per pixel on a 1/4-resolution downsample; reports median, p99 and max").
//!
//! The hash is over the decoded samples and the size, not over the file: the PNG carries the application's version in
//! its XMP, which must not change the hash of an unchanged picture.

use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use serde::Serialize;
use sha2::{Digest, Sha256};
use ym_color::consts::{Vec3, linalg};
use ym_color::deltae::{d65_white_xyz, delta_e_2000, percentile, xyz_to_lab};
use ym_color::encode::srgb_decode;
use ym_color::gamut;

/// How much smaller the picture is made before the comparison (`docs/13`: 1/4).
pub const DOWNSAMPLE: usize = 4;

/// A picture of 16-bit sRGB-encoded samples, interleaved R, G, B.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rgb16 {
    /// Width in pixels.
    pub width: usize,
    /// Height in pixels.
    pub height: usize,
    /// `width * height * 3` samples.
    pub data: Vec<u16>,
}

/// Reads a PNG as 16-bit RGB. An 8-bit picture is widened (`v * 257`); a grey or alpha picture is refused (the
/// export is always RGB).
pub fn read_png16(path: &Path) -> Result<Rgb16> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut reader = png::Decoder::new(BufReader::new(file))
        .read_info()
        .with_context(|| format!("reading the header of {}", path.display()))?;
    let size = reader
        .output_buffer_size()
        .with_context(|| format!("{} is too large", path.display()))?;
    let mut buf = vec![0_u8; size];
    let info = reader
        .next_frame(&mut buf)
        .with_context(|| format!("decoding {}", path.display()))?;
    ensure!(
        info.color_type == png::ColorType::Rgb,
        "{} is {:?}, not RGB",
        path.display(),
        info.color_type
    );
    let (width, height) = (info.width as usize, info.height as usize);
    let bytes = &buf[..info.buffer_size()];
    let data: Vec<u16> = match info.bit_depth {
        png::BitDepth::Sixteen => bytes
            .chunks_exact(2)
            .map(|b| u16::from_be_bytes([b[0], b[1]]))
            .collect(),
        png::BitDepth::Eight => bytes.iter().map(|&b| u16::from(b) * 257).collect(),
        other => bail!("{} has {other:?} samples", path.display()),
    };
    ensure!(
        data.len() == width * height * 3,
        "{}: {} samples for {width} x {height}",
        path.display(),
        data.len()
    );
    Ok(Rgb16 {
        width,
        height,
        data,
    })
}

impl Rgb16 {
    /// SHA-256 over the size and the samples (big-endian, as PNG stores them), in hex.
    #[must_use]
    pub fn pixel_sha256(&self) -> String {
        let mut h = Sha256::new();
        h.update(b"rgb16be");
        h.update((self.width as u64).to_le_bytes());
        h.update((self.height as u64).to_le_bytes());
        for chunk in self.data.chunks(1 << 15) {
            let bytes: Vec<u8> = chunk.iter().flat_map(|v| v.to_be_bytes()).collect();
            h.update(&bytes);
        }
        hex::encode(h.finalize())
    }

    /// CIE L*a*b* (D65) of the picture after a `DOWNSAMPLE` x `DOWNSAMPLE` box filter in linear light. Rows and
    /// columns that do not fill a block are dropped. The samples are read as sRGB.
    #[must_use]
    pub fn lab_downsampled(&self) -> LabImage {
        let (w, h) = (self.width / DOWNSAMPLE, self.height / DOWNSAMPLE);
        let to_xyz = gamut::matrices().rec709_to_xyz;
        let white = d65_white_xyz();
        // 65 536 table entries: the sRGB decode of every 16-bit code, so a 24 MP picture is not 72 M `powf` calls.
        let table: Vec<f64> = (0..=u16::MAX)
            .map(|v| srgb_decode(f64::from(v) / 65535.0))
            .collect();
        let mut lab = Vec::with_capacity(w * h);
        let norm = 1.0 / (DOWNSAMPLE * DOWNSAMPLE) as f64;
        for by in 0..h {
            for bx in 0..w {
                let mut sum = [0.0_f64; 3];
                for dy in 0..DOWNSAMPLE {
                    let row = (by * DOWNSAMPLE + dy) * self.width + bx * DOWNSAMPLE;
                    for dx in 0..DOWNSAMPLE {
                        let i = (row + dx) * 3;
                        for c in 0..3 {
                            sum[c] += table[usize::from(self.data[i + c])];
                        }
                    }
                }
                let linear = sum.map(|s| s * norm);
                lab.push(xyz_to_lab(linalg::mul_vec(&to_xyz, linear), white));
            }
        }
        LabImage {
            width: w,
            height: h,
            lab,
        }
    }
}

/// A picture in CIE L*a*b*.
#[derive(Clone, Debug)]
pub struct LabImage {
    /// Width in pixels.
    pub width: usize,
    /// Height in pixels.
    pub height: usize,
    /// `width * height` colours.
    pub lab: Vec<Vec3>,
}

/// Median, 99th percentile, mean and maximum of the per-pixel CIEDE2000 difference of two pictures.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct DeltaStats {
    /// Number of pixels compared.
    pub pixels: usize,
    /// Median ΔE00.
    pub median: f64,
    /// 99th percentile.
    pub p99: f64,
    /// Mean.
    pub mean: f64,
    /// Maximum.
    pub max: f64,
}

/// The ΔE00 between two pictures of the same size, as described at the top of this module.
///
/// # Errors
///
/// When the pictures have different sizes or are smaller than one downsample block.
pub fn delta_e_stats(a: &Rgb16, b: &Rgb16) -> Result<DeltaStats> {
    ensure!(
        (a.width, a.height) == (b.width, b.height),
        "the pictures differ in size: {} x {} against {} x {}",
        a.width,
        a.height,
        b.width,
        b.height
    );
    let (la, lb) = (a.lab_downsampled(), b.lab_downsampled());
    ensure!(
        !la.lab.is_empty(),
        "the picture is smaller than {DOWNSAMPLE} x {DOWNSAMPLE} pixels"
    );
    let diffs: Vec<f64> = la
        .lab
        .iter()
        .zip(&lb.lab)
        .map(|(&x, &y)| delta_e_2000(x, y))
        .collect();
    Ok(stats_of(&diffs))
}

/// How far apart two pictures are at full resolution, in codes of 16 bits (the GPU mode of `golden check` holds the
/// share of samples above a threshold against a gate, `docs/17-gpu.md` section 12).
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct CodeStats {
    /// Samples compared (`3 * width * height`).
    pub samples: usize,
    /// The threshold in codes: a sample counts when it differs by more than this.
    pub threshold: u32,
    /// Samples that differ by more than `threshold` codes.
    pub over: usize,
    /// `over / samples`.
    pub share_over: f64,
    /// The largest difference of any sample, in codes.
    pub max: u32,
}

/// The difference of two pictures of the same size in codes of 16 bits.
///
/// # Errors
///
/// When the pictures have different sizes.
pub fn code_stats(a: &Rgb16, b: &Rgb16, threshold: u32) -> Result<CodeStats> {
    ensure!(
        (a.width, a.height) == (b.width, b.height),
        "the pictures differ in size: {} x {} against {} x {}",
        a.width,
        a.height,
        b.width,
        b.height
    );
    let mut over = 0_usize;
    let mut max = 0_u32;
    for (&x, &y) in a.data.iter().zip(&b.data) {
        let d = u32::from(x.abs_diff(y));
        over += usize::from(d > threshold);
        max = max.max(d);
    }
    let samples = a.data.len();
    Ok(CodeStats {
        samples,
        threshold,
        over,
        share_over: over as f64 / samples.max(1) as f64,
        max,
    })
}

/// Median, 99th percentile, mean and maximum of a non-empty list (a NaN counts as infinitely different).
#[must_use]
pub fn stats_of(values: &[f64]) -> DeltaStats {
    let clean: Vec<f64> = values
        .iter()
        .map(|v| if v.is_nan() { f64::INFINITY } else { *v })
        .collect();
    DeltaStats {
        pixels: clean.len(),
        median: percentile(&clean, 50.0).unwrap_or(f64::NAN),
        p99: percentile(&clean, 99.0).unwrap_or(f64::NAN),
        mean: clean.iter().sum::<f64>() / clean.len().max(1) as f64,
        max: clean.iter().copied().fold(0.0, f64::max),
    }
}

/// Writes a picture as a 16-bit RGB PNG (the reference images, and the pictures of the tests).
///
/// The file appears all at once: it is written under a temporary name in the same folder and renamed, so a run that is
/// killed half way (a CI job that times out) never leaves a truncated reference that a cache would keep.
pub fn write_png16(path: &Path, picture: &Rgb16) -> Result<()> {
    let dir = path.parent().filter(|d| !d.as_os_str().is_empty());
    if let Some(dir) = dir {
        std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    }
    let name = path
        .file_name()
        .map_or_else(|| "picture".into(), |n| n.to_string_lossy());
    let tmp = path.with_file_name(format!(".{name}.{}.tmp", std::process::id()));
    let written = (|| -> Result<()> {
        let file = File::create(&tmp).with_context(|| format!("creating {}", tmp.display()))?;
        let mut encoder = png::Encoder::new(
            std::io::BufWriter::new(file),
            u32::try_from(picture.width).context("the picture is too wide for a PNG")?,
            u32::try_from(picture.height).context("the picture is too tall for a PNG")?,
        );
        encoder.set_color(png::ColorType::Rgb);
        encoder.set_depth(png::BitDepth::Sixteen);
        let mut writer = encoder.write_header()?;
        let bytes: Vec<u8> = picture.data.iter().flat_map(|v| v.to_be_bytes()).collect();
        writer.write_image_data(&bytes)?;
        writer.finish()?;
        Ok(())
    })();
    match written {
        Ok(()) => {
            std::fs::rename(&tmp, path).with_context(|| format!("replacing {}", path.display()))
        }
        Err(e) => {
            let _ = std::fs::remove_file(&tmp);
            Err(e)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat(width: usize, height: usize, rgb: [u16; 3]) -> Rgb16 {
        Rgb16 {
            width,
            height,
            data: (0..width * height).flat_map(|_| rgb).collect(),
        }
    }

    #[test]
    fn identical_pictures_have_no_difference() {
        let a = flat(16, 8, [30000, 20000, 10000]);
        let s = delta_e_stats(&a, &a).unwrap();
        assert_eq!((s.median, s.p99, s.max, s.mean), (0.0, 0.0, 0.0, 0.0));
        assert_eq!(s.pixels, 4 * 2);
    }

    #[test]
    fn one_16_bit_code_is_invisible_and_a_grey_step_is_not() {
        let a = flat(8, 8, [30000, 30000, 30000]);
        let b = flat(8, 8, [30001, 30000, 30000]);
        assert!(delta_e_stats(&a, &b).unwrap().max < 0.01);
        // 8-bit codes 128 and 129 in mid grey: about 0.4 ΔE00
        let c = flat(8, 8, [128 * 257, 128 * 257, 128 * 257]);
        let d = flat(8, 8, [129 * 257, 129 * 257, 129 * 257]);
        let s = delta_e_stats(&c, &d).unwrap();
        assert!((0.2..0.8).contains(&s.max), "{}", s.max);
        // black against white is 100 in L*
        let w = flat(8, 8, [65535; 3]);
        let k = flat(8, 8, [0; 3]);
        assert!((delta_e_stats(&k, &w).unwrap().max - 100.0).abs() < 0.5);
    }

    #[test]
    fn the_box_filter_averages_in_linear_light() {
        // one block of alternating black and white: the linear mean is 0.5, which is code 0.7354 (sRGB), not 0.5
        let mut p = flat(4, 4, [0; 3]);
        for (i, px) in p.data.chunks_exact_mut(3).enumerate() {
            if (i / 4 + i % 4) % 2 == 0 {
                px.copy_from_slice(&[65535; 3]);
            }
        }
        let grey = flat(4, 4, [(0.735_357 * 65535.0) as u16; 3]);
        assert!(delta_e_stats(&p, &grey).unwrap().max < 0.05);
    }

    #[test]
    fn differences_in_size_are_an_error_and_edges_are_dropped() {
        assert!(delta_e_stats(&flat(8, 8, [1; 3]), &flat(8, 12, [1; 3])).is_err());
        assert!(delta_e_stats(&flat(3, 3, [1; 3]), &flat(3, 3, [1; 3])).is_err());
        // 9 x 9 compares the 2 x 2 blocks; the ninth row and column do not count
        let a = flat(9, 9, [100; 3]);
        let mut b = a.clone();
        b.data[(8 * 9 + 8) * 3] = 65535;
        assert_eq!(delta_e_stats(&a, &b).unwrap().max, 0.0);
    }

    #[test]
    fn the_hash_follows_the_samples_and_the_size() {
        let a = flat(4, 2, [5, 6, 7]);
        let b = flat(2, 4, [5, 6, 7]);
        assert_ne!(a.pixel_sha256(), b.pixel_sha256());
        let mut c = a.clone();
        c.data[10] = 8;
        assert_ne!(a.pixel_sha256(), c.pixel_sha256());
        assert_eq!(a.pixel_sha256(), a.clone().pixel_sha256());
        assert_eq!(a.pixel_sha256().len(), 64);
    }

    #[test]
    fn png_round_trip_keeps_every_sample() {
        let dir = tempfile::tempdir().unwrap();
        let mut p = flat(5, 3, [0; 3]);
        for (i, v) in p.data.iter_mut().enumerate() {
            *v = (i as u16).wrapping_mul(2579);
        }
        let path = dir.path().join("sub/x.png");
        write_png16(&path, &p).unwrap();
        assert_eq!(read_png16(&path).unwrap(), p);
    }

    #[test]
    fn a_written_png_leaves_no_temporary_file_and_replaces_an_old_one() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("x.png");
        write_png16(&path, &flat(4, 4, [1, 2, 3])).unwrap();
        write_png16(&path, &flat(4, 4, [4, 5, 6])).unwrap();
        assert_eq!(read_png16(&path).unwrap().data[..3], [4, 5, 6]);
        let names: Vec<String> = std::fs::read_dir(dir.path())
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(names, ["x.png"]);
        // a failure removes its temporary file and leaves no picture
        let bad = Rgb16 {
            width: 4,
            height: 4,
            data: vec![0; 3],
        };
        assert!(write_png16(&dir.path().join("bad.png"), &bad).is_err());
        assert!(!dir.path().join("bad.png").exists());
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 1);
    }

    #[test]
    fn stats_of_counts_nan_as_failure() {
        let s = stats_of(&[0.0, 1.0, f64::NAN]);
        assert!(s.max.is_infinite());
        let s = stats_of(&[1.0, 2.0, 3.0]);
        assert_eq!((s.median, s.max, s.pixels), (2.0, 3.0, 3));
    }
}
