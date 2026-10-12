//! Log-chroma histograms of the 1/8-resolution preview (`training/illum/features.py` is the reference).
//!
//! * Channel *intensity*: every unmasked pixel whose three channels are finite and above `min_intensity` has the
//!   log-chroma `u = ln(G/R)`, `v = ln(G/B)` and the weight `sqrt(Y)` with `Y = 0.25 R + 0.5 G + 0.25 B`.
//! * Channel *edge*: the masked local absolute deviation of each colour channel (the mean absolute difference to
//!   the unmasked pixels among the 8 neighbours inside the image) is an image of its own; its log-chroma and
//!   `sqrt(Y)` weight are histogrammed the same way (all three edge values above `min_edge`).
//! * A pixel is splatted bilinearly onto the `n x n` grid whose bin `(i, j)` sits at `u = lo_u + i * bin_size`,
//!   `v = lo_v + j * bin_size`; a pixel outside `0 <= (u - lo_u) / bin_size < n - 1` (same for `v`) is dropped.
//! * Each channel is divided by the sum of its counted weights.
//! * Large previews are subsampled: only the pixels `(y, x)` with `y` and `x` multiples of
//!   `stride = max(1, isqrt(width * height / max_samples))` are histogrammed (edges are still taken between neighbouring
//!   pixels of the full preview). With `max_samples = 16384` the 1080 x 720 preview of a 50 MP frame uses every 6th
//!   pixel in both directions, 21 600 pixels.

use super::{IllumError, RawPreview};

/// The histogram channels a model can use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HistogramChannel {
    /// Weighted log-chroma of the pixels.
    Intensity,
    /// Weighted log-chroma of the local absolute deviation (the edges).
    Edge,
}

/// Geometry and thresholds of a model's histograms.
#[derive(Debug, Clone, PartialEq)]
pub struct HistogramSpec {
    /// Bins per axis (a power of two).
    pub n_bins: usize,
    /// Bin spacing in log-chroma units.
    pub bin_size: f64,
    /// `u` of bin 0.
    pub lo_u: f64,
    /// `v` of bin 0.
    pub lo_v: f64,
    /// A pixel needs all three channels above this to count (white level = 1).
    pub min_intensity: f64,
    /// An edge pixel needs all three edge values above this to count.
    pub min_edge: f64,
    /// The channels, in the order of the filter bank.
    pub channels: Vec<HistogramChannel>,
    /// Upper bound of the number of pixels histogrammed (sets the subsampling stride).
    pub max_samples: usize,
}

impl HistogramSpec {
    /// The pixel step of the subsampling grid for a `width x height` preview.
    #[must_use]
    pub fn stride(&self, width: usize, height: usize) -> usize {
        (width.saturating_mul(height) / self.max_samples.max(1))
            .isqrt()
            .max(1)
    }
}

/// The normalised histograms of one preview and their statistics.
#[derive(Debug, Clone, PartialEq)]
pub struct Histograms {
    /// `channels * n * n` values, channel-major then `i` (`u` bin) then `j` (`v` bin); each channel sums to 1, or is 0 when
    /// it counted nothing.
    pub data: Vec<f64>,
    /// Per channel: pixels that carried weight divided by all pixels.
    pub valid_fraction: Vec<f64>,
    /// Per channel: counted weight divided by the weight of all valid pixels (1 when there were none).
    pub in_range: Vec<f64>,
    /// Per channel: the counted weight before normalisation.
    pub mass: Vec<f64>,
}

#[inline]
fn luminance(r: f64, g: f64, b: f64) -> f64 {
    0.25 * r + 0.5 * g + 0.25 * b
}

/// Splats the pixels `px(i)` (`None` = not counted) of a `count`-pixel image into `hist` and returns
/// `(pixels with weight, counted weight, weight of all valid pixels)`.
fn splat_channel(
    spec: &HistogramSpec,
    (width, height, stride): (usize, usize, usize),
    floor: f64,
    px: impl Fn(usize, usize) -> Option<[f64; 3]>,
    hist: &mut [f64],
) -> (usize, f64, f64) {
    let n = spec.n_bins;
    let limit = (n - 1) as f64;
    let (mut selected, mut counted, mut total) = (0usize, 0.0f64, 0.0f64);
    for (y, x) in (0..height)
        .step_by(stride)
        .flat_map(|y| (0..width).step_by(stride).map(move |x| (y, x)))
    {
        let Some([r, g, b]) = px(x, y) else { continue };
        if !(r.is_finite() && g.is_finite() && b.is_finite() && r > floor && g > floor && b > floor)
        {
            continue;
        }
        selected += 1;
        let u = (g / r).ln();
        let v = (g / b).ln();
        let w = luminance(r, g, b).sqrt();
        total += w;
        let fu = (u - spec.lo_u) / spec.bin_size;
        let fv = (v - spec.lo_v) / spec.bin_size;
        if !(fu >= 0.0 && fu < limit && fv >= 0.0 && fv < limit) {
            continue;
        }
        counted += w;
        let (iu, iv) = (fu.floor(), fv.floor());
        let (tu, tv) = (fu - iu, fv - iv);
        let base = iu as usize * n + iv as usize;
        hist[base] += w * ((1.0 - tu) * (1.0 - tv));
        hist[base + n] += w * (tu * (1.0 - tv));
        hist[base + 1] += w * ((1.0 - tu) * tv);
        hist[base + n + 1] += w * (tu * tv);
    }
    (selected, counted, total)
}

/// The masked local absolute deviation at pixel `(x, y)`, `None` for a masked pixel or one without unmasked neighbours.
///
/// The neighbours are visited row by row, left to right (the order of `training/illum/features.py::edge_image`), so the sums
/// agree with the reference to the last digit.
fn edge_at(preview: &RawPreview<'_>, x: usize, y: usize) -> Option<[f64; 3]> {
    let (w, h) = (preview.width, preview.height);
    let centre_index = y * w + x;
    if preview.is_clipped(centre_index) {
        return None;
    }
    let centre = preview.pixel(centre_index);
    let mut acc = [0.0f64; 3];
    let mut count = 0.0f64;
    for ny in y.saturating_sub(1)..=(y + 1).min(h - 1) {
        for nx in x.saturating_sub(1)..=(x + 1).min(w - 1) {
            let idx = ny * w + nx;
            if idx == centre_index || preview.is_clipped(idx) {
                continue;
            }
            let q = preview.pixel(idx);
            acc[0] += (q[0] - centre[0]).abs();
            acc[1] += (q[1] - centre[1]).abs();
            acc[2] += (q[2] - centre[2]).abs();
            count += 1.0;
        }
    }
    if count == 0.0 {
        return None;
    }
    Some([acc[0] / count, acc[1] / count, acc[2] / count])
}

/// The normalised histograms of `preview` for the model geometry `spec`.
///
/// # Errors
///
/// [`IllumError::PreviewSize`] when the buffers do not match the stated size, [`IllumError::InvalidModel`] for a
/// specification with fewer than two bins per axis (a model file never has one).
pub fn compute(spec: &HistogramSpec, preview: &RawPreview<'_>) -> Result<Histograms, IllumError> {
    preview.validate()?;
    let n = spec.n_bins;
    if n < 2 {
        return Err(IllumError::InvalidModel(format!(
            "a histogram needs at least 2 bins per axis, not {n}"
        )));
    }
    let (w, h) = (preview.width, preview.height);
    let stride = spec.stride(w, h);
    let sampled = h.div_ceil(stride) * w.div_ceil(stride);
    let grid = (w, h, stride);
    let mut data = vec![0.0f64; spec.channels.len() * n * n];
    let mut valid_fraction = Vec::with_capacity(spec.channels.len());
    let mut in_range = Vec::with_capacity(spec.channels.len());
    let mut mass = Vec::with_capacity(spec.channels.len());
    for (c, channel) in spec.channels.iter().enumerate() {
        let hist = &mut data[c * n * n..(c + 1) * n * n];
        let (selected, counted, total) = match channel {
            HistogramChannel::Intensity => splat_channel(
                spec,
                grid,
                spec.min_intensity,
                |x, y| {
                    let i = y * w + x;
                    (!preview.is_clipped(i)).then(|| preview.pixel(i))
                },
                hist,
            ),
            HistogramChannel::Edge => splat_channel(
                spec,
                grid,
                spec.min_edge,
                |x, y| edge_at(preview, x, y),
                hist,
            ),
        };
        valid_fraction.push(if sampled > 0 {
            selected as f64 / sampled as f64
        } else {
            0.0
        });
        in_range.push(if total > 0.0 { counted / total } else { 1.0 });
        mass.push(counted);
        if counted > 0.0 {
            for v in hist.iter_mut() {
                *v /= counted;
            }
        }
    }
    Ok(Histograms {
        data,
        valid_fraction,
        in_range,
        mass,
    })
}
