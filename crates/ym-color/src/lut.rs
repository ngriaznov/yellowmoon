//! 3D LUT sampling (tetrahedral) and the 33-cube bake used by built-in looks (`docs/02-colour-pipeline.md` section C10).
//!
//! * [`Lut3d`]: a cube of `size^3` RGB entries, **red varying fastest** (index `r + size * (g + size * b)`),
//!   over a per-channel input domain. [`Lut3d::sample`] clamps the input to the domain and
//!   interpolates **tetrahedrally** (the same six-tetrahedra split as OpenColorIO and most
//!   renderers), never trilinearly. The GPU kernel does the same manually in the shader.
//! * [`Lut1d`]: a per-channel 1D table (the optional shaper in front of a 3D LUT), linear
//!   interpolation, input clamped to its domain.
//! * [`Lut`]: an optional shaper followed by a cube, the form a `.cube` file takes.
//! * [`Lut3d::bake`]: samples any function on the grid, which is how parametric looks become a
//!   65-point cube (see [`crate::look`]).
//! * [`Lut3d::to_cube_string`]: writes a `.cube` file that [`crate::cube::parse`] reads back exactly.

use std::fmt::Write as _;

use crate::consts::Vec3;
use crate::cube::MAX_3D_SIZE;

/// Size of the cube that parametric built-in looks are baked into. 65 points (it was 33 until the
/// richer Signature look of 2026-10-19, whose steeper luma curve a 33-point cube followed only to
/// 1.35 display codes on saturated colours; the error falls with the square of the spacing).
pub const BAKE_SIZE: usize = 65;

fn lerp3(a: Vec3, b: Vec3, t: f64) -> Vec3 {
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    ]
}

/// Position of `x` in a table of `size` entries over `[min, max]`: clamped, normalised, scaled to
/// `0 ..= size - 1`. A NaN input is treated as the domain minimum.
fn grid_position(x: f64, min: f64, max: f64, size: usize) -> f64 {
    let clamped = if x.is_nan() { min } else { x.clamp(min, max) };
    (clamped - min) / (max - min) * (size - 1) as f64
}

/// A 3D LUT: `size^3` entries, red fastest, over `[domain_min, domain_max]` per channel.
#[derive(Debug, Clone, PartialEq)]
pub struct Lut3d {
    size: usize,
    domain_min: Vec3,
    domain_max: Vec3,
    data: Vec<Vec3>,
}

/// Why a LUT could not be built.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum LutError {
    /// The table does not hold `size^3` (3D) or `size` (1D) entries.
    #[error("expected {expected} entries for size {size}, got {found}")]
    WrongEntryCount {
        /// The size given.
        size: usize,
        /// The number of entries required.
        expected: usize,
        /// The number of entries supplied.
        found: usize,
    },
    /// A table needs at least two entries per axis.
    #[error("LUT size {0} is below the minimum of 2")]
    TooSmall(usize),
    /// Domain minimum must be below the maximum on every channel, and both finite.
    #[error("input domain is empty or not finite on channel {channel}")]
    BadDomain {
        /// Channel index 0..3.
        channel: usize,
    },
    /// A table value is NaN or infinite.
    #[error("LUT entry {index} is not finite")]
    NotFinite {
        /// Index of the first offending entry.
        index: usize,
    },
    /// A baked cube may not be larger than a `.cube` file may declare.
    #[error("cannot bake a {size}-point cube, the largest size is {max}")]
    TooLarge {
        /// The size asked for.
        size: usize,
        /// The largest size, [`crate::cube::MAX_3D_SIZE`].
        max: usize,
    },
}

fn check_domain(min: Vec3, max: Vec3) -> Result<(), LutError> {
    for channel in 0..3 {
        // The width must be finite too: a domain such as [-1e308, 1e308] overflows when normalising.
        let ok = min[channel].is_finite()
            && max[channel].is_finite()
            && min[channel] < max[channel]
            && (max[channel] - min[channel]).is_finite();
        if !ok {
            return Err(LutError::BadDomain { channel });
        }
    }
    Ok(())
}

impl Lut3d {
    /// Builds a cube from `data` (red fastest).
    ///
    /// # Errors
    ///
    /// [`LutError`] when `size < 2`, the entry count is not `size^3`, the domain is empty or not
    /// finite, or any entry is not finite.
    pub fn new(
        size: usize,
        domain_min: Vec3,
        domain_max: Vec3,
        data: Vec<Vec3>,
    ) -> Result<Self, LutError> {
        if size < 2 {
            return Err(LutError::TooSmall(size));
        }
        let expected = size.checked_pow(3).unwrap_or(usize::MAX);
        if data.len() != expected {
            return Err(LutError::WrongEntryCount {
                size,
                expected,
                found: data.len(),
            });
        }
        check_domain(domain_min, domain_max)?;
        if let Some(index) = data.iter().position(|v| v.iter().any(|c| !c.is_finite())) {
            return Err(LutError::NotFinite { index });
        }
        Ok(Self {
            size,
            domain_min,
            domain_max,
            data,
        })
    }

    /// The identity cube of the given size over `[0, 1]^3`.
    ///
    /// # Errors
    ///
    /// [`LutError::TooSmall`] when `size < 2`, [`LutError::TooLarge`] above [`MAX_3D_SIZE`].
    pub fn identity(size: usize) -> Result<Self, LutError> {
        Self::bake(size, [0.0; 3], [1.0; 3], |rgb| rgb)
    }

    /// Samples `f` at every grid point of a `size^3` cube over `[domain_min, domain_max]`
    /// (the grid point `i` of an axis is `min + (max - min) * i / (size - 1)`).
    ///
    /// # Errors
    ///
    /// [`LutError`] when `size` is below 2 or above [`crate::cube::MAX_3D_SIZE`] (so the cube is
    /// never larger than a `.cube` file may declare and the allocation stays bounded), the domain
    /// is empty, or `f` returns a non-finite value.
    pub fn bake(
        size: usize,
        domain_min: Vec3,
        domain_max: Vec3,
        mut f: impl FnMut(Vec3) -> Vec3,
    ) -> Result<Self, LutError> {
        if size < 2 {
            return Err(LutError::TooSmall(size));
        }
        if size > MAX_3D_SIZE {
            return Err(LutError::TooLarge {
                size,
                max: MAX_3D_SIZE,
            });
        }
        check_domain(domain_min, domain_max)?;
        let last = (size - 1) as f64;
        let coord = |axis: usize, i: usize| {
            domain_min[axis] + (domain_max[axis] - domain_min[axis]) * (i as f64 / last)
        };
        let mut data = Vec::with_capacity(size * size * size);
        for b in 0..size {
            for g in 0..size {
                for r in 0..size {
                    data.push(f([coord(0, r), coord(1, g), coord(2, b)]));
                }
            }
        }
        Self::new(size, domain_min, domain_max, data)
    }

    /// Entries per axis.
    #[must_use]
    pub fn size(&self) -> usize {
        self.size
    }

    /// Per-channel input minimum.
    #[must_use]
    pub fn domain_min(&self) -> Vec3 {
        self.domain_min
    }

    /// Per-channel input maximum.
    #[must_use]
    pub fn domain_max(&self) -> Vec3 {
        self.domain_max
    }

    /// The entries, red fastest.
    #[must_use]
    pub fn data(&self) -> &[Vec3] {
        &self.data
    }

    /// The entry at grid indices `(r, g, b)`, or `None` when out of range.
    #[must_use]
    pub fn entry(&self, r: usize, g: usize, b: usize) -> Option<Vec3> {
        if r < self.size && g < self.size && b < self.size {
            Some(self.data[r + self.size * (g + self.size * b)])
        } else {
            None
        }
    }

    fn at(&self, r: usize, g: usize, b: usize) -> Vec3 {
        self.data[r + self.size * (g + self.size * b)]
    }

    /// Splits an input into the lower grid corner and the fractional position inside the cell.
    fn locate(&self, rgb: Vec3) -> ([usize; 3], Vec3) {
        let mut idx = [0_usize; 3];
        let mut frac = [0.0_f64; 3];
        for c in 0..3 {
            let pos = grid_position(rgb[c], self.domain_min[c], self.domain_max[c], self.size);
            let i = (pos.floor() as usize).min(self.size - 2);
            idx[c] = i;
            frac[c] = pos - i as f64;
        }
        (idx, frac)
    }

    /// Tetrahedral interpolation. The input is clamped to the domain first.
    ///
    /// The cell is split into six tetrahedra by the ordering of the fractional coordinates
    /// `(fr, fg, fb)`; inside each, the result is the lower corner plus three edge steps weighted
    /// by the sorted fractions, so the interpolation is continuous across cell faces and exact at
    /// grid points.
    #[must_use]
    pub fn sample(&self, rgb: Vec3) -> Vec3 {
        let ([i, j, k], [fr, fg, fb]) = self.locate(rgb);
        let c000 = self.at(i, j, k);
        let c111 = self.at(i + 1, j + 1, k + 1);
        // (path of corners, weights): out = c000 + w0 (p0 - c000) + w1 (p1 - p0) + w2 (p2 - p1) with p2 chain ending at c111.
        let (p1, p2, w0, w1, w2) = if fr > fg {
            if fg > fb {
                // R > G > B: c000 -> c100 -> c110 -> c111
                (self.at(i + 1, j, k), self.at(i + 1, j + 1, k), fr, fg, fb)
            } else if fr > fb {
                // R > B >= G: c000 -> c100 -> c101 -> c111
                (self.at(i + 1, j, k), self.at(i + 1, j, k + 1), fr, fb, fg)
            } else {
                // B >= R > G: c000 -> c001 -> c101 -> c111
                (self.at(i, j, k + 1), self.at(i + 1, j, k + 1), fb, fr, fg)
            }
        } else if fb > fg {
            // B > G >= R: c000 -> c001 -> c011 -> c111
            (self.at(i, j, k + 1), self.at(i, j + 1, k + 1), fb, fg, fr)
        } else if fb > fr {
            // G >= B > R: c000 -> c010 -> c011 -> c111
            (self.at(i, j + 1, k), self.at(i, j + 1, k + 1), fg, fb, fr)
        } else {
            // G >= R >= B: c000 -> c010 -> c110 -> c111
            (self.at(i, j + 1, k), self.at(i + 1, j + 1, k), fg, fr, fb)
        };
        let mut out = [0.0; 3];
        for c in 0..3 {
            out[c] =
                c000[c] + w0 * (p1[c] - c000[c]) + w1 * (p2[c] - p1[c]) + w2 * (c111[c] - p2[c]);
        }
        out
    }

    /// The entries as RGBA f32 with alpha 1, red fastest: the layout uploaded as an f32 storage
    /// buffer (`size^3` `vec4<f32>` nodes; the shader interpolates by hand, no hardware filtering,
    /// and f32 because an f16 texture would round by up to 2.4e-4).
    #[must_use]
    pub fn to_rgba_f32(&self) -> Vec<f32> {
        self.data
            .iter()
            .flat_map(|v| [v[0] as f32, v[1] as f32, v[2] as f32, 1.0])
            .collect()
    }

    /// Trilinear interpolation (for comparison and tests only; looks use [`Lut3d::sample`]).
    #[must_use]
    pub fn sample_trilinear(&self, rgb: Vec3) -> Vec3 {
        let ([i, j, k], [fr, fg, fb]) = self.locate(rgb);
        let x00 = lerp3(self.at(i, j, k), self.at(i + 1, j, k), fr);
        let x10 = lerp3(self.at(i, j + 1, k), self.at(i + 1, j + 1, k), fr);
        let x01 = lerp3(self.at(i, j, k + 1), self.at(i + 1, j, k + 1), fr);
        let x11 = lerp3(self.at(i, j + 1, k + 1), self.at(i + 1, j + 1, k + 1), fr);
        lerp3(lerp3(x00, x10, fg), lerp3(x01, x11, fg), fb)
    }

    /// Writes the cube as a `.cube` file (`TITLE`, `LUT_3D_SIZE`, `DOMAIN_MIN/MAX`, red fastest).
    /// Values are written with Rust's shortest round-trip formatting, so parsing the text gives back
    /// the same numbers bit for bit.
    #[must_use]
    pub fn to_cube_string(&self, title: &str) -> String {
        let mut s = String::with_capacity(self.data.len() * 24 + 128);
        let clean: String = title
            .chars()
            .filter(|c| !c.is_control() && *c != '"')
            .collect();
        let _ = writeln!(s, "TITLE \"{clean}\"");
        let _ = writeln!(s, "LUT_3D_SIZE {}", self.size);
        let _ = writeln!(
            s,
            "DOMAIN_MIN {} {} {}",
            self.domain_min[0], self.domain_min[1], self.domain_min[2]
        );
        let _ = writeln!(
            s,
            "DOMAIN_MAX {} {} {}",
            self.domain_max[0], self.domain_max[1], self.domain_max[2]
        );
        for v in &self.data {
            let _ = writeln!(s, "{} {} {}", v[0], v[1], v[2]);
        }
        s
    }
}

/// A 1D LUT: per-channel table of `size` entries over `[domain_min, domain_max]`.
#[derive(Debug, Clone, PartialEq)]
pub struct Lut1d {
    size: usize,
    domain_min: Vec3,
    domain_max: Vec3,
    data: Vec<Vec3>,
}

impl Lut1d {
    /// Builds a 1D table; column `c` of `data` is the table of channel `c`.
    ///
    /// # Errors
    ///
    /// [`LutError`] when `size < 2`, the entry count differs from `size`, the domain is empty or
    /// not finite, or an entry is not finite.
    pub fn new(
        size: usize,
        domain_min: Vec3,
        domain_max: Vec3,
        data: Vec<Vec3>,
    ) -> Result<Self, LutError> {
        if size < 2 {
            return Err(LutError::TooSmall(size));
        }
        if data.len() != size {
            return Err(LutError::WrongEntryCount {
                size,
                expected: size,
                found: data.len(),
            });
        }
        check_domain(domain_min, domain_max)?;
        if let Some(index) = data.iter().position(|v| v.iter().any(|c| !c.is_finite())) {
            return Err(LutError::NotFinite { index });
        }
        Ok(Self {
            size,
            domain_min,
            domain_max,
            data,
        })
    }

    /// Entries per channel.
    #[must_use]
    pub fn size(&self) -> usize {
        self.size
    }

    /// Per-channel input minimum.
    #[must_use]
    pub fn domain_min(&self) -> Vec3 {
        self.domain_min
    }

    /// Per-channel input maximum.
    #[must_use]
    pub fn domain_max(&self) -> Vec3 {
        self.domain_max
    }

    /// The entries (row `i` holds the R, G and B tables' entry `i`).
    #[must_use]
    pub fn data(&self) -> &[Vec3] {
        &self.data
    }

    /// The entries as RGBA f32 with alpha 1 (`size` `vec4<f32>` entries of a storage buffer).
    #[must_use]
    pub fn to_rgba_f32(&self) -> Vec<f32> {
        self.data
            .iter()
            .flat_map(|v| [v[0] as f32, v[1] as f32, v[2] as f32, 1.0])
            .collect()
    }

    /// Linear interpolation per channel, input clamped to the domain.
    #[must_use]
    pub fn sample(&self, rgb: Vec3) -> Vec3 {
        let mut out = [0.0; 3];
        for c in 0..3 {
            let pos = grid_position(rgb[c], self.domain_min[c], self.domain_max[c], self.size);
            let i = (pos.floor() as usize).min(self.size - 2);
            let f = pos - i as f64;
            let a = self.data[i][c];
            let b = self.data[i + 1][c];
            out[c] = a + (b - a) * f;
        }
        out
    }
}

/// An optional 1D shaper followed by a 3D cube: what a `.cube` file describes.
#[derive(Debug, Clone, PartialEq)]
pub struct Lut {
    /// Title from the file, when present.
    pub title: Option<String>,
    /// Optional 1D shaper applied first (its output is the cube's input).
    pub shaper: Option<Lut1d>,
    /// The 3D cube.
    pub cube: Lut3d,
}

impl Lut {
    /// A LUT with just a cube.
    #[must_use]
    pub fn from_cube(cube: Lut3d) -> Self {
        Self {
            title: None,
            shaper: None,
            cube,
        }
    }

    /// Applies the shaper (if any), then the cube.
    #[must_use]
    pub fn sample(&self, rgb: Vec3) -> Vec3 {
        match &self.shaper {
            Some(shaper) => self.cube.sample(shaper.sample(rgb)),
            None => self.cube.sample(rgb),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_cube_reproduces_its_input() {
        let lut = Lut3d::identity(5).unwrap();
        for rgb in [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.3, 0.7, 0.1],
            [0.5, 0.25, 0.9],
        ] {
            let out = lut.sample(rgb);
            for c in 0..3 {
                assert!((out[c] - rgb[c]).abs() < 1e-15);
            }
        }
    }

    #[test]
    fn red_is_the_fastest_axis() {
        let lut = Lut3d::bake(3, [0.0; 3], [1.0; 3], |rgb| rgb).unwrap();
        assert_eq!(lut.data()[1], [0.5, 0.0, 0.0]);
        assert_eq!(lut.data()[3], [0.0, 0.5, 0.0]);
        assert_eq!(lut.data()[9], [0.0, 0.0, 0.5]);
        assert_eq!(lut.entry(2, 0, 1), Some([1.0, 0.0, 0.5]));
        assert_eq!(lut.entry(3, 0, 0), None);
    }

    #[test]
    fn rejects_bad_tables() {
        assert!(matches!(Lut3d::identity(1), Err(LutError::TooSmall(1))));
        assert!(matches!(
            Lut3d::new(2, [0.0; 3], [1.0; 3], vec![[0.0; 3]; 7]),
            Err(LutError::WrongEntryCount {
                expected: 8,
                found: 7,
                ..
            })
        ));
        assert!(matches!(
            Lut3d::new(2, [0.0; 3], [0.0, 1.0, 1.0], vec![[0.0; 3]; 8]),
            Err(LutError::BadDomain { channel: 0 })
        ));
        let mut data = vec![[0.0; 3]; 8];
        data[5][1] = f64::NAN;
        assert!(matches!(
            Lut3d::new(2, [0.0; 3], [1.0; 3], data),
            Err(LutError::NotFinite { index: 5 })
        ));
    }
}
