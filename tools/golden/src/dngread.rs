//! A small reader for the DNG files `ym render --format dng` writes: the LinearRaw pixels of the raw sub-IFD and the
//! `BaselineExposure` of IFD0. It reads exactly what `ym-export` writes (uncompressed strips of 16-bit RGB) and refuses
//! anything else with a message; it is not a general DNG reader.
//!
//! The golden runner reads the pixels back to compare Yellowmoon's colour with the ground truth of the synthetic
//! scenes in scene-linear AWG4, before the display transform (`golden truth`).

use anyhow::{Context, Result, bail, ensure};

/// The `NewSubFileType`, `PhotometricInterpretation`, strip and size tags this reader needs.
mod tag {
    pub const NEW_SUBFILE_TYPE: u16 = 254;
    pub const IMAGE_WIDTH: u16 = 256;
    pub const IMAGE_LENGTH: u16 = 257;
    pub const BITS_PER_SAMPLE: u16 = 258;
    pub const COMPRESSION: u16 = 259;
    pub const PHOTOMETRIC: u16 = 262;
    pub const STRIP_OFFSETS: u16 = 273;
    pub const SAMPLES_PER_PIXEL: u16 = 277;
    pub const STRIP_BYTE_COUNTS: u16 = 279;
    pub const SUB_IFDS: u16 = 330;
    pub const ORIENTATION: u16 = 274;
    pub const BASELINE_EXPOSURE: u16 = 50730;
    pub const PROFILE_LOOK_TABLE_DIMS: u16 = 50981;
}

/// `PhotometricInterpretation` of LinearRaw.
const LINEAR_RAW: u32 = 34892;

/// The LinearRaw image of a DNG.
#[derive(Clone, Debug, PartialEq)]
pub struct LinearRaw {
    /// Width in pixels.
    pub width: usize,
    /// Height in pixels.
    pub height: usize,
    /// `width * height * 3` samples, interleaved.
    pub data: Vec<u16>,
    /// `BaselineExposure` of IFD0 in EV (`log2(x_white)` plus the fitted offset), when present.
    pub baseline_exposure_ev: Option<f64>,
    /// The TIFF `Orientation` of IFD0 (1 to 8), when present: the pixels are stored unrotated.
    pub orientation: Option<u16>,
    /// The `ProfileLookTableDims` of IFD0 (hue, saturation and value divisions), when present: the writer's identity
    /// profile has 2 x 2 x 2, a fitted one has more (`dngfit gate` uses it to tell the two apart).
    pub look_table_dims: Option<[u32; 3]>,
}

struct Tiff<'a> {
    bytes: &'a [u8],
    little: bool,
}

struct Field {
    tag: u16,
    kind: u16,
    count: usize,
    /// Offset of the 4-byte value field of the directory entry.
    value_at: usize,
}

impl Tiff<'_> {
    fn u16_at(&self, at: usize) -> Result<u16> {
        let b: [u8; 2] = self
            .bytes
            .get(at..at + 2)
            .and_then(|s| s.try_into().ok())
            .with_context(|| format!("the file ends at byte {at}"))?;
        Ok(if self.little {
            u16::from_le_bytes(b)
        } else {
            u16::from_be_bytes(b)
        })
    }

    fn u32_at(&self, at: usize) -> Result<u32> {
        let b: [u8; 4] = self
            .bytes
            .get(at..at + 4)
            .and_then(|s| s.try_into().ok())
            .with_context(|| format!("the file ends at byte {at}"))?;
        Ok(if self.little {
            u32::from_le_bytes(b)
        } else {
            u32::from_be_bytes(b)
        })
    }

    fn directory(&self, at: usize) -> Result<Vec<Field>> {
        let n = usize::from(self.u16_at(at)?);
        (0..n)
            .map(|i| {
                let e = at + 2 + i * 12;
                Ok(Field {
                    tag: self.u16_at(e)?,
                    kind: self.u16_at(e + 2)?,
                    count: self.u32_at(e + 4)? as usize,
                    value_at: e + 8,
                })
            })
            .collect()
    }

    /// Where the values of a field are: inline when they fit in four bytes, else at the stored offset.
    fn data_at(&self, f: &Field, unit: usize) -> Result<usize> {
        if f.count * unit <= 4 {
            Ok(f.value_at)
        } else {
            Ok(self.u32_at(f.value_at)? as usize)
        }
    }

    /// The integer values of a SHORT, LONG or IFD field.
    fn integers(&self, f: &Field) -> Result<Vec<u32>> {
        match f.kind {
            3 => {
                let at = self.data_at(f, 2)?;
                (0..f.count)
                    .map(|i| self.u16_at(at + 2 * i).map(u32::from))
                    .collect()
            }
            4 | 13 => {
                let at = self.data_at(f, 4)?;
                (0..f.count).map(|i| self.u32_at(at + 4 * i)).collect()
            }
            other => bail!("tag {} has type {other}, not an integer type", f.tag),
        }
    }

    fn single(&self, fields: &[Field], tag: u16) -> Result<u32> {
        let f = fields
            .iter()
            .find(|f| f.tag == tag)
            .with_context(|| format!("tag {tag} is missing"))?;
        self.integers(f)?
            .first()
            .copied()
            .with_context(|| format!("tag {tag} is empty"))
    }

    fn srational(&self, f: &Field) -> Result<f64> {
        ensure!(
            f.kind == 10 && f.count >= 1,
            "tag {} is not an SRATIONAL",
            f.tag
        );
        let at = self.data_at(f, 8)?;
        let n = self.u32_at(at)? as i32;
        let d = self.u32_at(at + 4)? as i32;
        ensure!(d != 0, "tag {} has a zero denominator", f.tag);
        Ok(f64::from(n) / f64::from(d))
    }
}

/// Reads the LinearRaw image of the DNG in `bytes`.
///
/// # Errors
///
/// When the file is not a TIFF, has no LinearRaw sub-IFD, or stores its pixels in a way `ym-export` does not (another
/// bit depth, compression or sample count).
pub fn read_linear_raw(bytes: &[u8]) -> Result<LinearRaw> {
    ensure!(bytes.len() >= 8, "the file is too short to be a TIFF");
    let little = match &bytes[..2] {
        b"II" => true,
        b"MM" => false,
        _ => bail!("not a TIFF file"),
    };
    let t = Tiff { bytes, little };
    ensure!(t.u16_at(2)? == 42, "not a TIFF file (bad magic)");
    let ifd0 = t.directory(t.u32_at(4)? as usize)?;
    let baseline_exposure_ev = ifd0
        .iter()
        .find(|f| f.tag == tag::BASELINE_EXPOSURE)
        .map(|f| t.srational(f))
        .transpose()?;
    let orientation = ifd0
        .iter()
        .find(|f| f.tag == tag::ORIENTATION)
        .and_then(|f| t.integers(f).ok())
        .and_then(|v| v.first().copied())
        .and_then(|v| u16::try_from(v).ok());
    let look_table_dims = ifd0
        .iter()
        .find(|f| f.tag == tag::PROFILE_LOOK_TABLE_DIMS)
        .and_then(|f| t.integers(f).ok())
        .and_then(|v| <[u32; 3]>::try_from(v).ok());
    let subs = ifd0
        .iter()
        .find(|f| f.tag == tag::SUB_IFDS)
        .context("IFD0 has no SubIFDs: this is not a Yellowmoon DNG")?;
    for offset in t.integers(subs)? {
        let fields = t.directory(offset as usize)?;
        let is_raw = t.single(&fields, tag::NEW_SUBFILE_TYPE).ok() == Some(0)
            && t.single(&fields, tag::PHOTOMETRIC).ok() == Some(LINEAR_RAW);
        if !is_raw {
            continue;
        }
        let width = t.single(&fields, tag::IMAGE_WIDTH)? as usize;
        let height = t.single(&fields, tag::IMAGE_LENGTH)? as usize;
        ensure!(
            t.single(&fields, tag::SAMPLES_PER_PIXEL)? == 3,
            "the raw image does not have 3 samples"
        );
        ensure!(
            t.single(&fields, tag::COMPRESSION)? == 1,
            "the raw image is compressed"
        );
        let bits = fields
            .iter()
            .find(|f| f.tag == tag::BITS_PER_SAMPLE)
            .context("BitsPerSample is missing")?;
        ensure!(
            t.integers(bits)?.iter().all(|&b| b == 16),
            "the raw image is not 16-bit"
        );
        let offsets = t.integers(
            fields
                .iter()
                .find(|f| f.tag == tag::STRIP_OFFSETS)
                .context("StripOffsets is missing")?,
        )?;
        let counts = t.integers(
            fields
                .iter()
                .find(|f| f.tag == tag::STRIP_BYTE_COUNTS)
                .context("StripByteCounts is missing")?,
        )?;
        ensure!(
            offsets.len() == counts.len(),
            "strip offsets and counts differ in number"
        );
        // A damaged header can name any size: never allocate or multiply on its word.
        let want = width
            .checked_mul(height)
            .and_then(|n| n.checked_mul(6))
            .filter(|&n| n <= bytes.len())
            .with_context(|| {
                format!(
                    "a {width} x {height} RGB image does not fit in a file of {} bytes",
                    bytes.len()
                )
            })?;
        let mut raw = Vec::with_capacity(want);
        for (&o, &c) in offsets.iter().zip(&counts) {
            let strip = (o as usize)
                .checked_add(c as usize)
                .and_then(|end| bytes.get(o as usize..end))
                .context("a strip lies outside the file")?;
            raw.extend_from_slice(strip);
        }
        ensure!(
            raw.len() == want,
            "the strips hold {} bytes, a {width} x {height} RGB image needs {want}",
            raw.len()
        );
        let data = raw
            .chunks_exact(2)
            .map(|b| {
                if little {
                    u16::from_le_bytes([b[0], b[1]])
                } else {
                    u16::from_be_bytes([b[0], b[1]])
                }
            })
            .collect();
        return Ok(LinearRaw {
            width,
            height,
            data,
            baseline_exposure_ev,
            orientation,
            look_table_dims,
        });
    }
    bail!("no LinearRaw sub-IFD in the file")
}

impl LinearRaw {
    /// The mean of the stored samples in the rectangle `[x, y, width, height]`, as fractions of the 16-bit range.
    ///
    /// # Errors
    ///
    /// When the rectangle is empty or reaches outside the picture.
    pub fn mean_rgb(&self, rect: [u32; 4]) -> Result<[f64; 3]> {
        let [x, y, w, h] = rect.map(|v| v as usize);
        ensure!(w > 0 && h > 0, "an empty rectangle");
        ensure!(
            x + w <= self.width && y + h <= self.height,
            "the rectangle {rect:?} reaches outside the {} x {} picture",
            self.width,
            self.height
        );
        let mut sum = [0.0_f64; 3];
        for row in y..y + h {
            for col in x..x + w {
                let i = (row * self.width + col) * 3;
                for (acc, &v) in sum.iter_mut().zip(&self.data[i..i + 3]) {
                    *acc += f64::from(v);
                }
            }
        }
        let n = (w * h) as f64 * 65535.0;
        Ok(sum.map(|s| s / n))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ym_export::dng::{BaselineExposure, DngProfile};
    use ym_export::{
        ColorSpace, DngInputs, ExportFormat, ExportHooks, ExportRequest, ImageRef, Metadata,
        PixelEncoding, Size, encode_to_vec,
    };

    /// A DNG written by `ym-export` with known pixels.
    pub(crate) fn dng_of(w: u32, h: u32, f: impl Fn(u32, u32) -> [f32; 3]) -> (Vec<u8>, f32) {
        let mut raw = Vec::new();
        for y in 0..h {
            for x in 0..w {
                raw.extend_from_slice(&f(x, y));
            }
        }
        let preview = vec![0.25_f32; (w * h * 3) as usize];
        let meta = Metadata::default();
        let profile = DngProfile::identity();
        let req = ExportRequest {
            pixels: ImageRef::new(w, h, &raw, PixelEncoding::SceneLinearAwg4).unwrap(),
            format: ExportFormat::Dng,
            color_space: ColorSpace::Srgb,
            size: Size::Full,
            metadata: &meta,
            remove_location: false,
            dng: Some(DngInputs {
                preview: ImageRef::new(w, h, &preview, PixelEncoding::DisplayLinear).unwrap(),
                orientation: 1,
                crop: None,
                baseline_exposure: BaselineExposure::default(),
                profile: &profile,
            }),
        };
        let (bytes, summary) = encode_to_vec(&req, &ExportHooks::none()).unwrap();
        (bytes, summary.dng.unwrap().x_white)
    }

    #[test]
    fn the_pixels_and_the_baseline_exposure_come_back() {
        let (bytes, x_white) = dng_of(40, 24, |x, y| [x as f32 / 40.0, y as f32 / 24.0, 0.5]);
        assert_eq!(x_white, 1.0);
        let raw = read_linear_raw(&bytes).unwrap();
        assert_eq!((raw.width, raw.height), (40, 24));
        assert_eq!(raw.baseline_exposure_ev, Some(0.0));
        assert_eq!(raw.orientation, Some(1));
        for (x, y) in [(0usize, 0usize), (39, 23), (7, 11)] {
            let i = (y * 40 + x) * 3;
            let want = [x as f64 / 40.0, y as f64 / 24.0, 0.5];
            for (c, w) in want.iter().enumerate() {
                let got = f64::from(raw.data[i + c]) / 65535.0;
                assert!((got - w).abs() < 1e-4, "{x},{y},{c}: {got} vs {w}");
            }
        }
        let m = raw.mean_rgb([0, 0, 40, 24]).unwrap();
        assert!((m[2] - 0.5).abs() < 1e-4);
    }

    #[test]
    fn a_bright_scene_stores_the_scaled_values_and_says_how_much() {
        let (bytes, x_white) = dng_of(16, 16, |_, _| [4.0, 2.0, 1.0]);
        assert!((x_white - 4.0).abs() < 0.01);
        let raw = read_linear_raw(&bytes).unwrap();
        let m = raw.mean_rgb([0, 0, 16, 16]).unwrap();
        assert!(
            (m[0] - 1.0).abs() < 1e-3 && (m[1] - 0.5).abs() < 1e-3 && (m[2] - 0.25).abs() < 1e-3
        );
        assert!((raw.baseline_exposure_ev.unwrap() - f64::from(x_white).log2()).abs() < 1e-6);
    }

    #[test]
    fn the_look_table_dims_tell_the_identity_profile_from_a_fitted_one() {
        let (bytes, _) = dng_of(8, 8, |_, _| [0.1; 3]);
        // `DngProfile::identity()` writes a 2 x 2 x 2 table
        assert_eq!(
            read_linear_raw(&bytes).unwrap().look_table_dims,
            Some([2, 2, 2])
        );
    }

    #[test]
    fn a_damaged_header_that_names_a_huge_picture_is_an_error_and_not_a_crash() {
        let (bytes, _) = dng_of(8, 8, |_, _| [0.1; 3]);
        // find the value fields of ImageWidth and ImageLength in the raw sub-IFD and overwrite them with 0xFFFF
        let t = Tiff {
            bytes: &bytes,
            little: &bytes[..2] == b"II",
        };
        let ifd0 = t.directory(t.u32_at(4).unwrap() as usize).unwrap();
        let subs = ifd0.iter().find(|f| f.tag == tag::SUB_IFDS).unwrap();
        let mut spots = Vec::new();
        for offset in t.integers(subs).unwrap() {
            let fields = t.directory(offset as usize).unwrap();
            if t.single(&fields, tag::NEW_SUBFILE_TYPE).ok() == Some(0) {
                for f in &fields {
                    if f.tag == tag::IMAGE_WIDTH || f.tag == tag::IMAGE_LENGTH {
                        spots.push((f.value_at, f.kind));
                    }
                }
            }
        }
        assert_eq!(spots.len(), 2);
        let mut broken = bytes.clone();
        for (at, kind) in spots {
            let n = if kind == 3 { 2 } else { 4 };
            broken[at..at + n].fill(0xFF);
        }
        let e = read_linear_raw(&broken).unwrap_err();
        assert!(format!("{e:#}").contains("does not fit"), "{e:#}");
    }

    #[test]
    fn rectangles_outside_the_picture_and_other_files_are_refused() {
        let (bytes, _) = dng_of(8, 8, |_, _| [0.1; 3]);
        let raw = read_linear_raw(&bytes).unwrap();
        assert!(raw.mean_rgb([4, 4, 8, 8]).is_err());
        assert!(raw.mean_rgb([0, 0, 0, 3]).is_err());
        assert!(read_linear_raw(b"not a tiff at all").is_err());
        assert!(read_linear_raw(&bytes[..100]).is_err());
        let mut broken = bytes.clone();
        broken[2] = 43;
        assert!(read_linear_raw(&broken).is_err());
    }
}
