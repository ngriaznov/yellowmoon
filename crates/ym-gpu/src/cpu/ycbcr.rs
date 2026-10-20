//! Y/Cb/Cr planes and the final blend shared by the single-frame denoisers
//! (`docs/algorithms/05-single-frame-denoise.md`, section 0 and reading G2, G5).
//!
//! BT.709: `Y = 0.2126 R + 0.7152 G + 0.0722 B`, `Cb = (B − Y) / 1.8556`, `Cr = (R − Y) / 1.5748`, with
//! the exact inverse. The planes are `f64`, the precision the methods work in (reading G1).

use super::RgbImage;

/// Luma weights of BT.709 (`R`, `G`, `B`).
pub const KR: f64 = 0.2126;
/// See [`KR`].
pub const KG: f64 = 0.7152;
/// See [`KR`].
pub const KB: f64 = 0.0722;
/// `Cb = (B − Y) / CB_DIV`: `2 (1 − KB)`.
pub const CB_DIV: f64 = 1.8556;
/// `Cr = (R − Y) / CR_DIV`: `2 (1 − KR)`.
pub const CR_DIV: f64 = 1.5748;

/// Three `f64` planes of one size, in the order the method names them (`[R, G, B]` or `[Y, Cb, Cr]`).
pub type Planes = [Vec<f64>; 3];

/// The image's red, green and blue planes as `f64`.
pub fn rgb_planes(img: &RgbImage) -> Planes {
    let n = img.width * img.height;
    let plane = |c: usize| {
        img.data[c * n..(c + 1) * n]
            .iter()
            .map(|&v| f64::from(v))
            .collect()
    };
    [plane(0), plane(1), plane(2)]
}

/// An image of the size and place of `like` from three `f64` planes (`[R, G, B]`).
pub fn image_from_rgb(planes: &Planes, like: &RgbImage) -> RgbImage {
    let mut out = RgbImage {
        width: like.width,
        height: like.height,
        place: like.place,
        data: Vec::with_capacity(like.data.len()),
    };
    for p in planes {
        out.data.extend(p.iter().map(|&v| v as f32));
    }
    out
}

/// `[Y, Cb, Cr]` of the image (BT.709).
pub fn to_ycbcr(img: &RgbImage) -> Planes {
    let [r, g, b] = rgb_planes(img);
    let mut y = Vec::with_capacity(r.len());
    let mut cb = Vec::with_capacity(r.len());
    let mut cr = Vec::with_capacity(r.len());
    for i in 0..r.len() {
        let yy = KR * r[i] + KG * g[i] + KB * b[i];
        y.push(yy);
        cb.push((b[i] - yy) / CB_DIV);
        cr.push((r[i] - yy) / CR_DIV);
    }
    [y, cb, cr]
}

/// The inverse of [`to_ycbcr`]: an image of the size and place of `like`.
pub fn from_ycbcr(planes: &Planes, like: &RgbImage) -> RgbImage {
    let [y, cb, cr] = planes;
    let mut rgb: Planes = [
        Vec::with_capacity(y.len()),
        Vec::with_capacity(y.len()),
        Vec::with_capacity(y.len()),
    ];
    for i in 0..y.len() {
        let r = y[i] + CR_DIV * cr[i];
        let b = y[i] + CB_DIV * cb[i];
        let g = (y[i] - KR * r - KB * b) / KG;
        rgb[0].push(r);
        rgb[1].push(g);
        rgb[2].push(b);
    }
    image_from_rgb(&rgb, like)
}

/// The final blend of every method: `lerp(denoised, input, blend / 100)`; `blend <= 0` returns the
/// denoised image as it is (reading G5).
pub fn blend(denoised: RgbImage, input: &RgbImage, blend: f64) -> RgbImage {
    if blend <= 0.0 {
        return denoised;
    }
    let t = (blend / 100.0).min(1.0);
    let mut out = denoised;
    for (o, &i) in out.data.iter_mut().zip(&input.data) {
        let d = f64::from(*o);
        *o = (d + (f64::from(i) - d) * t) as f32;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn image(w: usize, h: usize) -> RgbImage {
        let n = w * h;
        let data = (0..3 * n)
            .map(|i| ((i * 7919) % 1000) as f32 / 1000.0 - 0.1)
            .collect();
        RgbImage::from_planar(w, h, data).unwrap()
    }

    #[test]
    fn ycbcr_round_trips_to_a_few_ulp_of_f32() {
        let img = image(9, 7);
        let back = from_ycbcr(&to_ycbcr(&img), &img);
        let worst = img
            .data
            .iter()
            .zip(&back.data)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(worst < 1e-6, "{worst}");
    }

    #[test]
    fn grey_has_zero_chroma_and_white_is_one() {
        let img = RgbImage::from_planar(1, 1, vec![1.0, 1.0, 1.0]).unwrap();
        let [y, cb, cr] = to_ycbcr(&img);
        assert!((y[0] - 1.0).abs() < 1e-15);
        assert!(cb[0].abs() < 1e-15 && cr[0].abs() < 1e-15);
    }

    #[test]
    fn blend_zero_is_the_denoised_image_and_100_the_input() {
        let a = image(4, 3);
        let mut b = a.clone();
        b.data.iter_mut().for_each(|v| *v *= 0.5);
        assert_eq!(blend(b.clone(), &a, 0.0), b);
        assert_eq!(blend(b.clone(), &a, 100.0).data, a.data);
        let half = blend(b.clone(), &a, 50.0);
        assert!((half.data[5] - 0.75 * a.data[5]).abs() < 1e-7);
    }
}
