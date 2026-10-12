//! Where the estimator reads its pixels from.
//!
//! The estimator touches about 2 MP of a 50 MP frame, so it asks for rectangles instead of a whole
//! normalised plane. [`U16Mosaic`] applies R1 (`docs/05-raw-processing.md`) to only those
//! rectangles, which keeps estimation off the 50 MP normalisation pass.

use super::EstimateError;

/// Lowest value R1 keeps.
const NEGATIVE_CLAMP: f32 = -0.05;

/// A 2-D plane of normalised values that can be read by rectangle.
///
/// Rectangles handed to [`PlaneSource::read_rect`] always start at an even `x0` and `y0`, so the
/// CFA phase of the origin is preserved.
pub trait PlaneSource: Sync {
    /// Width in photosites.
    fn width(&self) -> usize;
    /// Height in photosites.
    fn height(&self) -> usize;
    /// Fills `out` (`w * h` values, row-major) with the rectangle at `(x0, y0)`.
    fn read_rect(&self, x0: usize, y0: usize, w: usize, h: usize, out: &mut [f32]);
}

/// Checks that a buffer of `len` values holds `height` rows of `width` values that are `stride`
/// apart. The size arithmetic is checked: dimensions that come from a file header must not be able
/// to overflow it (a wrapped product would let a short buffer through and fail later on a read).
fn check_len(len: usize, width: usize, height: usize, stride: usize) -> Result<(), EstimateError> {
    let needed = stride
        .checked_mul(height.saturating_sub(1))
        .and_then(|rows| rows.checked_add(width));
    match needed {
        Some(needed) if width > 0 && height > 0 && stride >= width && len >= needed => Ok(()),
        _ => Err(EstimateError::BadInput(format!(
            "buffer of {len} values does not hold {width} x {height} (stride {stride})"
        ))),
    }
}

/// A normalised `f32` mosaic (R1 already applied), for tests and for callers that normalised early.
pub struct F32Plane<'a> {
    data: &'a [f32],
    width: usize,
    height: usize,
    stride: usize,
}

impl<'a> F32Plane<'a> {
    /// Wraps a tightly packed plane.
    ///
    /// # Errors
    ///
    /// Returns [`EstimateError::BadInput`] when the slice is too short.
    pub fn new(data: &'a [f32], width: usize, height: usize) -> Result<Self, EstimateError> {
        Self::with_stride(data, width, height, width)
    }

    /// Wraps a plane whose rows are `stride` values apart.
    ///
    /// # Errors
    ///
    /// Returns [`EstimateError::BadInput`] when the slice is too short.
    pub fn with_stride(
        data: &'a [f32],
        width: usize,
        height: usize,
        stride: usize,
    ) -> Result<Self, EstimateError> {
        check_len(data.len(), width, height, stride)?;
        Ok(Self {
            data,
            width,
            height,
            stride,
        })
    }
}

impl PlaneSource for F32Plane<'_> {
    fn width(&self) -> usize {
        self.width
    }
    fn height(&self) -> usize {
        self.height
    }
    fn read_rect(&self, x0: usize, y0: usize, w: usize, h: usize, out: &mut [f32]) {
        for r in 0..h {
            let src = (y0 + r) * self.stride + x0;
            out[r * w..(r + 1) * w].copy_from_slice(&self.data[src..src + w]);
        }
    }
}

/// A raw `u16` mosaic normalised on the fly: `(raw - black[p]) / (white[p] - black[p])` per CFA
/// position `p = 2 * (y & 1) + (x & 1)`, floored at -0.05 (R1).
pub struct U16Mosaic<'a> {
    data: &'a [u16],
    width: usize,
    height: usize,
    stride: usize,
    black: [f32; 4],
    scale: [f32; 4],
}

impl<'a> U16Mosaic<'a> {
    /// Wraps a tightly packed mosaic with the per-position black and white levels.
    ///
    /// # Errors
    ///
    /// Returns [`EstimateError::BadInput`] when the slice is too short or a level pair has
    /// `white <= black`.
    pub fn new(
        data: &'a [u16],
        width: usize,
        height: usize,
        black: [f32; 4],
        white: [f32; 4],
    ) -> Result<Self, EstimateError> {
        Self::with_stride(data, width, height, width, black, white)
    }

    /// Like [`U16Mosaic::new`] for rows that are `stride` values apart.
    ///
    /// # Errors
    ///
    /// Returns [`EstimateError::BadInput`] when the slice is too short or a level pair has
    /// `white <= black`.
    pub fn with_stride(
        data: &'a [u16],
        width: usize,
        height: usize,
        stride: usize,
        black: [f32; 4],
        white: [f32; 4],
    ) -> Result<Self, EstimateError> {
        check_len(data.len(), width, height, stride)?;
        let mut scale = [0.0; 4];
        for p in 0..4 {
            if white[p].partial_cmp(&black[p]) != Some(std::cmp::Ordering::Greater) {
                return Err(EstimateError::BadInput(format!(
                    "white level {} is not above black level {} at CFA position {p}",
                    white[p], black[p]
                )));
            }
            scale[p] = 1.0 / (white[p] - black[p]);
        }
        Ok(Self {
            data,
            width,
            height,
            stride,
            black,
            scale,
        })
    }
}

impl PlaneSource for U16Mosaic<'_> {
    fn width(&self) -> usize {
        self.width
    }
    fn height(&self) -> usize {
        self.height
    }
    fn read_rect(&self, x0: usize, y0: usize, w: usize, h: usize, out: &mut [f32]) {
        for r in 0..h {
            let src = &self.data[(y0 + r) * self.stride + x0..][..w];
            let dst = &mut out[r * w..(r + 1) * w];
            let py = ((y0 + r) & 1) << 1;
            for (c, (o, &raw)) in dst.iter_mut().zip(src).enumerate() {
                let p = py | ((x0 + c) & 1);
                *o = ((f32::from(raw) - self.black[p]) * self.scale[p]).max(NEGATIVE_CLAMP);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn u16_mosaic_applies_r1_per_position() {
        let (w, h) = (4, 4);
        let data: Vec<u16> = (0..16).map(|i| 100 + i as u16 * 10).collect();
        let black = [100.0, 110.0, 120.0, 130.0];
        let white = [200.0, 210.0, 220.0, 230.0];
        let m = U16Mosaic::new(&data, w, h, black, white).unwrap();
        let mut out = vec![0.0; 16];
        m.read_rect(0, 0, 4, 4, &mut out);
        for y in 0..4usize {
            for x in 0..4usize {
                let p = ((y & 1) << 1) | (x & 1);
                let raw = f32::from(data[y * 4 + x]);
                let want = ((raw - black[p]) / (white[p] - black[p])).max(-0.05);
                assert!((out[y * 4 + x] - want).abs() < 1e-6, "({x},{y})");
            }
        }
        // A sub-rectangle at an even origin sees the same values.
        let mut sub = vec![0.0; 4];
        m.read_rect(2, 2, 2, 2, &mut sub);
        assert_eq!(sub, vec![out[10], out[11], out[14], out[15]]);
    }

    #[test]
    fn negative_values_are_floored() {
        let data = vec![0u16; 4];
        let m = U16Mosaic::new(&data, 2, 2, [1000.0; 4], [2000.0; 4]).unwrap();
        let mut out = vec![0.0; 4];
        m.read_rect(0, 0, 2, 2, &mut out);
        assert!(out.iter().all(|&v| v == -0.05));
    }

    #[test]
    fn bad_buffers_and_levels_are_rejected() {
        assert!(F32Plane::new(&[0.0; 3], 2, 2).is_err());
        assert!(U16Mosaic::new(&[0; 4], 2, 2, [10.0; 4], [10.0; 4]).is_err());
        assert!(U16Mosaic::new(&[0; 4], 2, 2, [0.0; 4], [f32::NAN; 4]).is_err());
    }

    #[test]
    fn dimensions_that_overflow_the_size_arithmetic_are_rejected_not_wrapped() {
        // Found by fuzzing the constructors with header-sized garbage: `stride * (height - 1)`
        // overflowed (a panic in debug builds, a wrapped, too-small product in release builds).
        let big = usize::MAX / 2 + 3;
        let words = [0u16; 16];
        let floats = [0.0f32; 16];
        assert!(U16Mosaic::with_stride(&words, 4, big, 4, [0.0; 4], [1.0; 4]).is_err());
        assert!(U16Mosaic::with_stride(&words, 4, 4, usize::MAX - 1, [0.0; 4], [1.0; 4]).is_err());
        assert!(F32Plane::with_stride(&floats, 4, big, 4).is_err());
        assert!(F32Plane::with_stride(&floats, 4, 4, usize::MAX - 1).is_err());
        assert!(F32Plane::with_stride(&floats, usize::MAX, 1, usize::MAX).is_err());
        // Zero sizes and a stride below the width are still rejected; exact fits are accepted.
        assert!(F32Plane::new(&floats, 0, 4).is_err());
        assert!(F32Plane::new(&floats, 4, 0).is_err());
        assert!(F32Plane::with_stride(&floats, 4, 4, 3).is_err());
        assert!(F32Plane::new(&floats, 4, 4).is_ok());
        assert!(F32Plane::with_stride(&floats[..14], 4, 4, 4).is_err());
        assert!(F32Plane::with_stride(&floats[..14], 2, 4, 4).is_ok());
    }

    #[test]
    fn f32_plane_reads_rectangles_with_a_stride() {
        let data: Vec<f32> = (0..30).map(|i| i as f32).collect();
        let p = F32Plane::with_stride(&data, 4, 3, 10).unwrap();
        let mut out = vec![0.0; 4];
        p.read_rect(2, 1, 2, 2, &mut out);
        assert_eq!(out, vec![12.0, 13.0, 22.0, 23.0]);
    }
}
