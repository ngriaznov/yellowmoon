//! Planar images placed in the frame, and pixel rectangles.

use crate::error::LensError;

/// A planar RGB image (red plane, then green, then blue; row-major) placed in the frame at scale `k`.
///
/// The frame at scale `k` is the full-resolution frame (after R9's auto crop) reduced `k` times, `frame_width`
/// by `frame_height` pixels as the caller counts them. All lengths of the lens derive from the half-diagonal of
/// that frame, and the optical centre is `(frame_width / 2, frame_height / 2)` in continuous coordinates where
/// pixel `i` covers `[i, i + 1)`.
#[derive(Clone, Copy, Debug)]
pub struct ImageRef<'a> {
    /// Width of the image in pixels.
    pub width: usize,
    /// Height of the image in pixels.
    pub height: usize,
    /// Frame column of the image's first column.
    pub x0: usize,
    /// Frame row of the image's first row.
    pub y0: usize,
    /// Width of the frame at this scale.
    pub frame_width: usize,
    /// Height of the frame at this scale.
    pub frame_height: usize,
    /// `3 * width * height` samples: the red plane, the green plane, the blue plane.
    pub data: &'a [f32],
}

/// A pixel rectangle `[x, x + w) x [y, y + h)` in frame coordinates at the current scale.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PixelRect {
    /// Left edge.
    pub x: usize,
    /// Top edge.
    pub y: usize,
    /// Width in pixels.
    pub w: usize,
    /// Height in pixels.
    pub h: usize,
}

impl PixelRect {
    /// A rectangle from its corner and size.
    #[must_use]
    pub const fn new(x: usize, y: usize, w: usize, h: usize) -> Self {
        PixelRect { x, y, w, h }
    }

    /// One past the last column.
    #[must_use]
    pub const fn x1(&self) -> usize {
        self.x + self.w
    }

    /// One past the last row.
    #[must_use]
    pub const fn y1(&self) -> usize {
        self.y + self.h
    }

    /// True when `inner` lies entirely inside this rectangle.
    #[must_use]
    pub fn contains(&self, inner: &PixelRect) -> bool {
        inner.x >= self.x && inner.y >= self.y && inner.x1() <= self.x1() && inner.y1() <= self.y1()
    }

    /// This rectangle grown by `px` on every side and clamped to a `frame_width` x `frame_height` frame.
    #[must_use]
    pub fn grown(&self, px: usize, frame_width: usize, frame_height: usize) -> PixelRect {
        let x = self.x.saturating_sub(px);
        let y = self.y.saturating_sub(px);
        let x1 = (self.x1().saturating_add(px)).min(frame_width);
        let y1 = (self.y1().saturating_add(px)).min(frame_height);
        PixelRect {
            x,
            y,
            w: x1.saturating_sub(x),
            h: y1.saturating_sub(y),
        }
    }
}

impl std::fmt::Display for PixelRect {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}, {}) x [{}, {})", self.x, self.x1(), self.y, self.y1())
    }
}

impl<'a> ImageRef<'a> {
    /// The rectangle this image holds, in frame coordinates.
    #[must_use]
    pub const fn rect(&self) -> PixelRect {
        PixelRect {
            x: self.x0,
            y: self.y0,
            w: self.width,
            h: self.height,
        }
    }

    /// Plane `c` (0 red, 1 green, 2 blue), `width * height` samples.
    ///
    /// # Panics
    ///
    /// Panics when `c > 2` or the data are shorter than [`ImageRef::validate`] allows.
    #[must_use]
    pub fn plane(&self, c: usize) -> &'a [f32] {
        let n = self.width * self.height;
        &self.data[c * n..(c + 1) * n]
    }

    /// Checks that the data have `3 * width * height` samples and the image lies inside its frame.
    ///
    /// # Errors
    ///
    /// [`LensError::Frame`] when it does not.
    pub fn validate(&self) -> Result<(), LensError> {
        if self.frame_width == 0 || self.frame_height == 0 {
            return Err(LensError::Frame("the frame is empty".into()));
        }
        let len = 3usize
            .checked_mul(self.width)
            .and_then(|n| n.checked_mul(self.height));
        if len != Some(self.data.len()) {
            return Err(LensError::Frame(format!(
                "a {}x{} planar RGB image needs 3 * {} samples, found {}",
                self.width,
                self.height,
                self.width * self.height,
                self.data.len()
            )));
        }
        if self.x0.saturating_add(self.width) > self.frame_width
            || self.y0.saturating_add(self.height) > self.frame_height
        {
            return Err(LensError::Frame(format!(
                "the image at ({}, {}) of {}x{} does not fit the {}x{} frame",
                self.x0, self.y0, self.width, self.height, self.frame_width, self.frame_height
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grown_clamps_to_the_frame() {
        let r = PixelRect::new(2, 3, 10, 10);
        assert_eq!(r.grown(5, 100, 100), PixelRect::new(0, 0, 17, 18));
        assert_eq!(r.grown(5, 14, 15), PixelRect::new(0, 0, 14, 15));
        assert!(r.grown(5, 14, 15).contains(&r));
    }

    #[test]
    fn validate_catches_bad_sizes() {
        let data = vec![0.0f32; 12];
        let ok = ImageRef {
            width: 2,
            height: 2,
            x0: 1,
            y0: 1,
            frame_width: 3,
            frame_height: 3,
            data: &data,
        };
        assert!(ok.validate().is_ok());
        assert!(ImageRef { x0: 2, ..ok }.validate().is_err());
        assert!(ImageRef { width: 3, ..ok }.validate().is_err());
    }
}
