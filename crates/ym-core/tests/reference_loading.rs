//! Reading a reference picture (`docs/34-reference-grade.md` section 6 and the "Reference loading" gate of section 10, RG-05): the
//! formats, the colour (our own ICC reader), EXIF orientation, black bars, the minimum size and the reduction. Every picture here is
//! synthetic and made by the test; no film still and no photograph is in the repository.
//!
//! The ICC conversion against its oracles is `crates/ym-export/tests/icc_reader.rs`; the letterbox rule's own cases are in
//! `crates/ym-color/src/letterbox.rs`. These tests go through the loader: a file in, a [`ReferencePicture`] or the plain error out.

use std::io::Cursor;

use ym_color::encode::srgb_decode;
use ym_color::icc::RgbProfile;
use ym_core::reference::{
    ANALYSIS_LONG_EDGE_PX, MIN_SHORT_SIDE_PX, ReferenceError, ReferenceNote, ReferencePicture,
    ReferenceSource, SCENE_LONG_EDGE_PX,
};
use ym_export::ColorSpace;
use ym_export::icc::{FixtureCurve, FixtureLayout, colorants_d50, fixture_profile, profile};

// ---------------------------------------------------------------------------------------------
// Synthetic pictures and encoders
// ---------------------------------------------------------------------------------------------

/// SplitMix64.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn signed(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 52) as f64 - 1.0
    }
}

/// A colourful picture: the hue runs along x, the lightness along y, with a little noise. RGB, 8 bits.
fn colourful(w: usize, h: usize, seed: u64) -> Vec<u8> {
    let mut rng = Rng(seed);
    let mut d = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        for x in 0..w {
            let t = x as f64 / w as f64;
            let l = 70.0 + 130.0 * (y as f64 / h as f64);
            let rgb = [
                l * (0.6 + 0.4 * (std::f64::consts::TAU * t).sin()),
                l * (0.6 + 0.4 * (std::f64::consts::TAU * (t + 0.33)).sin()),
                l * (0.6 + 0.4 * (std::f64::consts::TAU * (t + 0.67)).sin()),
            ];
            for v in rgb {
                d.push((v + 6.0 * rng.signed()).clamp(0.0, 255.0).round() as u8);
            }
        }
    }
    d
}

/// The same curve for the three channels.
fn three(c: FixtureCurve) -> [FixtureCurve; 3] {
    [c.clone(), c.clone(), c]
}

/// A picture of four coloured quadrants: A top left, B top right, C bottom left, D bottom right (red, green, blue, yellow).
fn quadrants(w: usize, h: usize) -> Vec<u8> {
    let mut d = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        for x in 0..w {
            let c = match (x < w / 2, y < h / 2) {
                (true, true) => [220, 30, 30],
                (false, true) => [30, 200, 40],
                (true, false) => [30, 50, 220],
                (false, false) => [230, 220, 30],
            };
            d.extend_from_slice(&c);
        }
    }
    d
}

/// `picture` (`w x h`) inside black bars.
fn barred(
    picture: &[u8],
    w: usize,
    h: usize,
    top: usize,
    bottom: usize,
    left: usize,
    right: usize,
) -> (Vec<u8>, usize, usize) {
    let (ow, oh) = (w + left + right, h + top + bottom);
    let mut d = vec![0u8; ow * oh * 3];
    for y in 0..h {
        let at = ((y + top) * ow + left) * 3;
        d[at..at + w * 3].copy_from_slice(&picture[y * w * 3..(y + 1) * w * 3]);
    }
    (d, ow, oh)
}

fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &b in data {
        crc ^= u32::from(b);
        for _ in 0..8 {
            crc = if crc & 1 != 0 {
                (crc >> 1) ^ 0xEDB8_8320
            } else {
                crc >> 1
            };
        }
    }
    !crc
}

/// A chunk of a PNG, with its length and CRC.
fn chunk(kind: &[u8; 4], data: &[u8]) -> Vec<u8> {
    let mut c = (data.len() as u32).to_be_bytes().to_vec();
    let mut body = kind.to_vec();
    body.extend_from_slice(data);
    c.extend_from_slice(&body);
    c.extend_from_slice(&crc32(&body).to_be_bytes());
    c
}

/// `png` with these chunks inserted right after IHDR.
fn with_chunks(png: &[u8], chunks: &[([u8; 4], Vec<u8>)]) -> Vec<u8> {
    let ihdr_end = 8 + 12 + 13;
    let mut out = png[..ihdr_end].to_vec();
    for (kind, data) in chunks {
        out.extend_from_slice(&chunk(kind, data));
    }
    out.extend_from_slice(&png[ihdr_end..]);
    out
}

fn zlib_store(data: &[u8]) -> Vec<u8> {
    // A stored (uncompressed) deflate stream with an Adler-32, for the iCCP chunk.
    let mut out = vec![0x78, 0x01];
    let mut chunks = data.chunks(65_535).peekable();
    if data.is_empty() {
        out.extend_from_slice(&[1, 0, 0, 0xFF, 0xFF]);
    }
    while let Some(c) = chunks.next() {
        out.push(u8::from(chunks.peek().is_none()));
        out.extend_from_slice(&(c.len() as u16).to_le_bytes());
        out.extend_from_slice(&(!(c.len() as u16)).to_le_bytes());
        out.extend_from_slice(c);
    }
    let (mut a, mut b) = (1u32, 0u32);
    for &x in data {
        a = (a + u32::from(x)) % 65_521;
        b = (b + a) % 65_521;
    }
    out.extend_from_slice(&((b << 16) | a).to_be_bytes());
    out
}

/// The `iCCP` chunk's data for `profile`.
fn iccp(profile: &[u8]) -> Vec<u8> {
    let mut d = b"profile\0\0".to_vec();
    d.extend_from_slice(&zlib_store(profile));
    d
}

/// A PNG of `w x h` with samples of `bits` (8 or 16: `data` holds the samples of 16 bits big-endian) in `color`.
fn png_bytes(
    w: usize,
    h: usize,
    color: png::ColorType,
    bits: png::BitDepth,
    data: &[u8],
) -> Vec<u8> {
    let mut out = Vec::new();
    let mut enc = png::Encoder::new(&mut out, w as u32, h as u32);
    enc.set_color(color);
    enc.set_depth(bits);
    let mut writer = enc.write_header().expect("header");
    writer.write_image_data(data).expect("data");
    writer.finish().expect("finish");
    out
}

fn png_rgb8(w: usize, h: usize, rgb: &[u8]) -> Vec<u8> {
    png_bytes(w, h, png::ColorType::Rgb, png::BitDepth::Eight, rgb)
}

fn tiff_rgb8(w: usize, h: usize, rgb: &[u8], tags: &[(tiff::tags::Tag, TiffTag)]) -> Vec<u8> {
    let mut out = Cursor::new(Vec::new());
    let mut enc = tiff::encoder::TiffEncoder::new(&mut out).expect("tiff");
    let mut image = enc
        .new_image::<tiff::encoder::colortype::RGB8>(w as u32, h as u32)
        .expect("image");
    for (tag, value) in tags {
        match value {
            TiffTag::Short(v) => image.encoder().write_tag(*tag, *v).expect("tag"),
            TiffTag::Bytes(v) => image.encoder().write_tag(*tag, &v[..]).expect("tag"),
        }
    }
    image.write_data(rgb).expect("data");
    out.into_inner()
}

enum TiffTag {
    Short(u16),
    Bytes(Vec<u8>),
}

fn tiff_rgb16(w: usize, h: usize, rgb: &[u16]) -> Vec<u8> {
    let mut out = Cursor::new(Vec::new());
    let mut enc = tiff::encoder::TiffEncoder::new(&mut out).expect("tiff");
    enc.write_image::<tiff::encoder::colortype::RGB16>(w as u32, h as u32, rgb)
        .expect("image");
    out.into_inner()
}

fn tiff_gray16(w: usize, h: usize, g: &[u16]) -> Vec<u8> {
    let mut out = Cursor::new(Vec::new());
    let mut enc = tiff::encoder::TiffEncoder::new(&mut out).expect("tiff");
    enc.write_image::<tiff::encoder::colortype::Gray16>(w as u32, h as u32, g)
        .expect("image");
    out.into_inner()
}

fn tiff_gray8(w: usize, h: usize, g: &[u8]) -> Vec<u8> {
    let mut out = Cursor::new(Vec::new());
    let mut enc = tiff::encoder::TiffEncoder::new(&mut out).expect("tiff");
    enc.write_image::<tiff::encoder::colortype::Gray8>(w as u32, h as u32, g)
        .expect("image");
    out.into_inner()
}

/// The Exif TIFF structure with one orientation entry (little endian).
fn exif_with_orientation(o: u16) -> Vec<u8> {
    let mut d = b"Exif\0\0II*\0".to_vec();
    d.extend_from_slice(&8u32.to_le_bytes());
    d.extend_from_slice(&1u16.to_le_bytes());
    d.extend_from_slice(&[0x12, 0x01, 3, 0, 1, 0, 0, 0]);
    d.extend_from_slice(&o.to_le_bytes());
    d.extend_from_slice(&[0, 0, 0, 0, 0, 0]);
    d
}

/// A JPEG of `rgb` at `quality`, 4:4:4 or 4:2:0, with an optional profile and orientation.
fn jpeg_bytes(
    w: usize,
    h: usize,
    rgb: &[u8],
    quality: u8,
    subsampled: bool,
    icc: Option<&[u8]>,
    orientation: Option<u16>,
) -> Vec<u8> {
    let mut out = Vec::new();
    let mut enc = jpeg_encoder::Encoder::new(&mut out, quality);
    enc.set_sampling_factor(if subsampled {
        jpeg_encoder::SamplingFactor::F_2_2
    } else {
        jpeg_encoder::SamplingFactor::F_1_1
    });
    if let Some(p) = icc {
        enc.add_icc_profile(p).expect("icc");
    }
    if let Some(o) = orientation {
        enc.add_app_segment(1, exif_with_orientation(o))
            .expect("exif");
    }
    enc.encode(rgb, w as u16, h as u16, jpeg_encoder::ColorType::Rgb)
        .expect("encode");
    out
}

fn jpeg_grey(w: usize, h: usize, g: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    jpeg_encoder::Encoder::new(&mut out, 95)
        .encode(g, w as u16, h as u16, jpeg_encoder::ColorType::Luma)
        .expect("encode");
    out
}

/// The mean of the analysis picture, per channel.
fn mean_linear(p: &ReferencePicture) -> [f64; 3] {
    let a = p.analysis();
    let mut m = [0.0; 3];
    for px in &a.linear {
        for c in 0..3 {
            m[c] += px[c];
        }
    }
    m.map(|v| v / a.linear.len() as f64)
}

/// The mean of the linear sRGB of `rgb`, per channel (the expected mean of an sRGB picture).
fn expected_mean(rgb: &[u8]) -> [f64; 3] {
    let mut m = [0.0; 3];
    for px in rgb.chunks_exact(3) {
        for c in 0..3 {
            m[c] += srgb_decode(f64::from(px[c]) / 255.0);
        }
    }
    m.map(|v| v / (rgb.len() / 3) as f64)
}

fn assert_close(a: [f64; 3], b: [f64; 3], tol: f64, what: &str) {
    for c in 0..3 {
        assert!(
            (a[c] - b[c]).abs() <= tol,
            "{what}: channel {c}: {} against {}",
            a[c],
            b[c]
        );
    }
}

// ---------------------------------------------------------------------------------------------
// Formats
// ---------------------------------------------------------------------------------------------

const W: usize = 400;
const H: usize = 300;

#[test]
fn jpeg_png_and_tiff_read_at_8_and_16_bits_and_untagged_is_srgb() {
    let rgb = colourful(W, H, 1);
    let want = expected_mean(&rgb);
    // PNG, TIFF 8 bits: lossless, the same picture.
    let png8 = ReferencePicture::from_bytes(&png_rgb8(W, H, &rgb)).expect("png 8");
    let tiff8 = ReferencePicture::from_bytes(&tiff_rgb8(W, H, &rgb, &[])).expect("tiff 8");
    assert_eq!(png8.analysis(), tiff8.analysis());
    assert_eq!(png8.source(), ReferenceSource::File);
    assert!(png8.notes().is_empty() && !png8.is_grey());
    assert_eq!((png8.analysis().width, png8.analysis().height), (W, H));
    assert_close(mean_linear(&png8), want, 1e-12, "png 8");
    // 16 bits with the 8-bit codes widened (v * 257): the same linear values.
    let wide: Vec<u16> = rgb.iter().map(|&v| u16::from(v) * 257).collect();
    let tiff16 = ReferencePicture::from_bytes(&tiff_rgb16(W, H, &wide)).expect("tiff 16");
    assert_close(mean_linear(&tiff16), want, 1e-9, "tiff 16");
    let be: Vec<u8> = wide.iter().flat_map(|v| v.to_be_bytes()).collect();
    let png16 = ReferencePicture::from_bytes(&png_bytes(
        W,
        H,
        png::ColorType::Rgb,
        png::BitDepth::Sixteen,
        &be,
    ))
    .expect("png 16");
    assert_eq!(png16.analysis().width, W);
    assert_close(mean_linear(&png16), want, 1e-9, "png 16");
    // A 16-bit picture is read at its precision: codes between the 8-bit ones are not rounded to them.
    let fine: Vec<u16> = (0..W * H * 3)
        .map(|i| 20_000 + (i % 3) as u16 * 100 + (i / 3 % 7) as u16)
        .collect();
    let p = ReferencePicture::from_bytes(&tiff_rgb16(W, H, &fine)).expect("fine");
    let first = p.analysis().linear[0];
    assert!(first[0] != first[1], "the channels differ by 100 of 65535");
    // JPEG at a high quality: close.
    let jpg = ReferencePicture::from_bytes(&jpeg_bytes(W, H, &rgb, 100, false, None, None))
        .expect("jpeg");
    assert_close(mean_linear(&jpg), want, 4e-3, "jpeg");
    // The same file twice is the same bits.
    let again = ReferencePicture::from_bytes(&png_rgb8(W, H, &rgb)).expect("again");
    assert_eq!(again.analysis(), png8.analysis());
    assert_eq!(again.scene(), png8.scene());
    assert_eq!(again.digest(), png8.digest());
}

#[test]
fn grey_pictures_read_and_say_so() {
    let g: Vec<u8> = (0..W * H).map(|i| ((i % W) * 255 / W) as u8).collect();
    let want = {
        let m: f64 = g
            .iter()
            .map(|&v| srgb_decode(f64::from(v) / 255.0))
            .sum::<f64>()
            / g.len() as f64;
        [m; 3]
    };
    let png = ReferencePicture::from_bytes(&png_bytes(
        W,
        H,
        png::ColorType::Grayscale,
        png::BitDepth::Eight,
        &g,
    ))
    .expect("png grey");
    let tif8 = ReferencePicture::from_bytes(&tiff_gray8(W, H, &g)).expect("tiff grey");
    let wide: Vec<u16> = g.iter().map(|&v| u16::from(v) * 257).collect();
    let tif16 = ReferencePicture::from_bytes(&tiff_gray16(W, H, &wide)).expect("tiff grey 16");
    let jpg = ReferencePicture::from_bytes(&jpeg_grey(W, H, &g)).expect("jpeg grey");
    for (name, p) in [
        ("png", &png),
        ("tiff 8", &tif8),
        ("tiff 16", &tif16),
        ("jpeg", &jpg),
    ] {
        assert!(p.is_grey(), "{name}");
        assert_eq!(p.notes(), &[ReferenceNote::NoColour], "{name}");
        for px in &p.analysis().linear {
            assert!(
                px[0] == px[1] && px[1] == px[2],
                "{name}: a grey pixel is neutral"
            );
        }
        assert_close(
            mean_linear(p),
            want,
            if name == "jpeg" { 3e-3 } else { 1e-9 },
            name,
        );
    }
    assert_eq!(
        ReferenceNote::NoColour.text(),
        "This picture has no colour. The photo will be matched in black and white."
    );
}

#[test]
fn an_alpha_channel_is_not_part_of_the_colour() {
    let rgb = colourful(W, H, 2);
    let rgba: Vec<u8> = rgb
        .chunks_exact(3)
        .flat_map(|p| [p[0], p[1], p[2], 17])
        .collect();
    let p = ReferencePicture::from_bytes(&png_bytes(
        W,
        H,
        png::ColorType::Rgba,
        png::BitDepth::Eight,
        &rgba,
    ))
    .expect("rgba");
    assert_close(mean_linear(&p), expected_mean(&rgb), 1e-12, "rgba");
}

#[test]
fn heic_and_other_formats_are_refused_with_the_plain_text() {
    let heic = {
        let mut d = vec![0, 0, 0, 24];
        d.extend_from_slice(b"ftypheic\0\0\0\0mif1heic");
        d.extend_from_slice(&[0; 400]);
        d
    };
    let gif =
        b"GIF89a\x01\0\x01\0\x80\0\0\0\0\0\xff\xff\xff,\0\0\0\0\x01\0\x01\0\0\x02\x02D\x01\0;"
            .to_vec();
    let bmp = {
        let mut d = b"BM".to_vec();
        d.extend_from_slice(&[0; 100]);
        d
    };
    let webp = b"RIFF\x24\0\0\0WEBPVP8 ".to_vec();
    let pdf = b"%PDF-1.7\n".to_vec();
    for (name, bytes) in [
        ("heic", heic),
        ("gif", gif),
        ("bmp", bmp),
        ("webp", webp),
        ("pdf", pdf),
        ("empty", vec![]),
        ("text", b"hello".to_vec()),
    ] {
        assert_eq!(
            ReferencePicture::from_bytes(&bytes).unwrap_err(),
            ReferenceError::Unsupported,
            "{name}"
        );
    }
    assert_eq!(
        ReferenceError::Unsupported.to_string(),
        "Yellowmoon reads JPEG, PNG and TIFF pictures. Save this one as one of those, then choose it again."
    );
}

#[test]
fn a_raw_file_is_refused_by_its_name_and_by_its_dng_tag() {
    let dir = tempfile::tempdir().expect("dir");
    let rgb = colourful(W, H, 3);
    // An ARW is TIFF underneath: the extension says what it is.
    let arw = dir.path().join("DSC00001.ARW");
    std::fs::write(&arw, tiff_rgb8(W, H, &rgb, &[])).expect("write");
    assert_eq!(
        ReferencePicture::from_file(&arw).unwrap_err(),
        ReferenceError::Unsupported
    );
    let dng = dir.path().join("photo.dng");
    std::fs::write(&dng, tiff_rgb8(W, H, &rgb, &[])).expect("write");
    assert_eq!(
        ReferencePicture::from_file(&dng).unwrap_err(),
        ReferenceError::Unsupported
    );
    // A DNG renamed: the DNGVersion tag gives it away.
    let renamed = dir.path().join("photo.tif");
    std::fs::write(
        &renamed,
        tiff_rgb8(
            W,
            H,
            &rgb,
            &[(
                tiff::tags::Tag::Unknown(50706),
                TiffTag::Bytes(vec![1, 4, 0, 0]),
            )],
        ),
    )
    .expect("write");
    assert_eq!(
        ReferencePicture::from_file(&renamed).unwrap_err(),
        ReferenceError::Unsupported
    );
    // A plain TIFF by file is read.
    let tif = dir.path().join("plain.tif");
    std::fs::write(&tif, tiff_rgb8(W, H, &rgb, &[])).expect("write");
    assert!(ReferencePicture::from_file(&tif).is_ok());
}

#[test]
fn a_damaged_or_missing_file_is_unreadable_and_never_a_panic() {
    let rgb = colourful(W, H, 4);
    let png = png_rgb8(W, H, &rgb);
    let jpg = jpeg_bytes(W, H, &rgb, 90, true, None, None);
    let tif = tiff_rgb8(W, H, &rgb, &[]);
    for (name, bytes) in [("png", &png), ("jpeg", &jpg), ("tiff", &tif)] {
        // A decoder that is not strict fills a cut picture's missing part with grey and the fit would measure the grey: a picture that
        // ends early is refused (PNG and TIFF refuse it themselves).
        for cut in [
            8usize,
            20,
            100,
            bytes.len() / 4,
            bytes.len() / 2,
            bytes.len() * 9 / 10,
        ] {
            let r = ReferencePicture::from_bytes(&bytes[..cut]);
            assert!(
                matches!(
                    r,
                    Err(ReferenceError::Unreadable | ReferenceError::Unsupported)
                ),
                "{name} cut at {cut}: {r:?}"
            );
        }
        // Noise after the signature.
        let mut noisy = bytes[..16].to_vec();
        noisy.extend((0..2000).map(|i| (i * 37 % 251) as u8));
        assert!(
            ReferencePicture::from_bytes(&noisy).is_err(),
            "{name} noise"
        );
    }
    let dir = tempfile::tempdir().expect("dir");
    assert_eq!(
        ReferencePicture::from_file(&dir.path().join("missing.jpg")).unwrap_err(),
        ReferenceError::Unreadable
    );
    assert_eq!(
        ReferenceError::Unreadable.to_string(),
        "This file can't be read. It may be damaged. Choose another picture."
    );
}

// ---------------------------------------------------------------------------------------------
// Colour
// ---------------------------------------------------------------------------------------------

/// A picture of saturated colours of Display P3 and what they are in sRGB.
fn p3_chart() -> Vec<u8> {
    let mut d = Vec::new();
    for y in 0..H {
        for x in 0..W {
            let r = (10 + (x * 90 / W)) as u8;
            let g = (255 - (y * 60 / H)) as u8;
            let b = (20 + ((x + y) * 120 / (W + H))) as u8;
            d.extend_from_slice(&[r, g, b]);
        }
    }
    d
}

/// What the reader says each 8-bit code of `rgb` is in linear sRGB under `profile`, averaged.
fn expected_under(profile: &[u8], rgb: &[u8]) -> [f64; 3] {
    let t = RgbProfile::parse(profile).expect("reads").to_linear_srgb();
    let tables = [t.table(0, 256), t.table(1, 256), t.table(2, 256)];
    let mut m = [0.0; 3];
    for px in rgb.chunks_exact(3) {
        let v = t.convert([
            tables[0][usize::from(px[0])],
            tables[1][usize::from(px[1])],
            tables[2][usize::from(px[2])],
        ]);
        for c in 0..3 {
            m[c] += v[c];
        }
    }
    m.map(|v| v / (rgb.len() / 3) as f64)
}

#[test]
fn an_embedded_profile_is_applied_in_png_jpeg_and_tiff() {
    let rgb = p3_chart();
    let as_srgb = expected_mean(&rgb);
    let p3 = profile(ColorSpace::DisplayP3);
    let want = expected_under(p3, &rgb);
    let apart = (0..3)
        .map(|c| (want[c] - as_srgb[c]).abs())
        .fold(0.0f64, f64::max);
    assert!(
        apart > 0.02,
        "the chart must tell P3 from sRGB: {want:?} against {as_srgb:?}"
    );
    let png = with_chunks(&png_rgb8(W, H, &rgb), &[(*b"iCCP", iccp(p3))]);
    let from_png = ReferencePicture::from_bytes(&png).expect("png with a profile");
    assert!(from_png.notes().is_empty());
    assert_close(mean_linear(&from_png), want, 1e-9, "png iCCP");
    let jpg = ReferencePicture::from_bytes(&jpeg_bytes(W, H, &rgb, 100, false, Some(p3), None))
        .expect("jpeg with a profile");
    assert_close(mean_linear(&jpg), want, 5e-3, "jpeg APP2");
    let tif = ReferencePicture::from_bytes(&tiff_rgb8(
        W,
        H,
        &rgb,
        &[(tiff::tags::Tag::IccProfile, TiffTag::Bytes(p3.to_vec()))],
    ))
    .expect("tiff with a profile");
    assert!(tif.notes().is_empty());
    assert_close(mean_linear(&tif), want, 1e-9, "tiff 34675");
    // Untagged is sRGB: the same picture without the profile reads differently.
    let plain = ReferencePicture::from_bytes(&png_rgb8(W, H, &rgb)).expect("plain");
    assert_close(mean_linear(&plain), as_srgb, 1e-12, "untagged");
    let moved = (0..3)
        .map(|c| (mean_linear(&plain)[c] - mean_linear(&from_png)[c]).abs())
        .fold(0.0f64, f64::max);
    assert!(moved > 0.02, "the profile must change the picture: {moved}");
}

#[test]
fn every_fixture_profile_is_read_by_the_loader() {
    let rgb = p3_chart();
    let srgb_like = fixture_profile(
        "sRGB-like",
        FixtureLayout::V4,
        colorants_d50(ColorSpace::Srgb).0,
        three(FixtureCurve::Para(
            3,
            vec![2.4, 1.0 / 1.055, 0.055 / 1.055, 1.0 / 12.92, 0.040_45],
        )),
        &[],
    );
    let display = fixture_profile(
        "Display",
        FixtureLayout::V2,
        colorants_d50(ColorSpace::DisplayP3).0,
        three(FixtureCurve::Para(
            3,
            vec![2.4, 1.0 / 1.055, 0.055 / 1.055, 1.0 / 12.92, 0.040_45],
        )),
        &[],
    );
    let gamma = fixture_profile(
        "gamma",
        FixtureLayout::V2,
        colorants_d50(ColorSpace::Rec2020).0,
        three(FixtureCurve::Gamma(2.2)),
        &[],
    );
    for (name, p) in [
        ("sRGB", profile(ColorSpace::Srgb).to_vec()),
        ("Display P3", profile(ColorSpace::DisplayP3).to_vec()),
        ("Rec.2020", profile(ColorSpace::Rec2020).to_vec()),
        ("sRGB-like para", srgb_like),
        ("screenshot Display", display),
        ("gamma", gamma),
    ] {
        let png = with_chunks(&png_rgb8(W, H, &rgb), &[(*b"iCCP", iccp(&p))]);
        let got = ReferencePicture::from_bytes(&png).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert!(got.notes().is_empty(), "{name}: {:?}", got.notes());
        assert_close(mean_linear(&got), expected_under(&p, &rgb), 1e-9, name);
    }
}

#[test]
fn cicp_and_the_srgb_chunk_are_read_and_the_rest_is_taken_as_srgb_with_the_note() {
    let rgb = p3_chart();
    let base = png_rgb8(W, H, &rgb);
    let as_srgb = expected_mean(&rgb);
    // sRGB chunk: sRGB, no note.
    let tagged = with_chunks(&base, &[(*b"sRGB", vec![0])]);
    let p = ReferencePicture::from_bytes(&tagged).expect("sRGB chunk");
    assert!(p.notes().is_empty());
    assert_close(mean_linear(&p), as_srgb, 1e-12, "sRGB chunk");
    // cICP of our exports: sRGB, Display P3, Rec.2020.
    let p3_icc = expected_under(profile(ColorSpace::DisplayP3), &rgb);
    for (cp, want, name) in [
        ([1u8, 13, 0, 1], as_srgb, "cICP sRGB"),
        ([12, 13, 0, 1], p3_icc, "cICP Display P3"),
        (
            [9, 1, 0, 1],
            expected_under(profile(ColorSpace::Rec2020), &rgb),
            "cICP Rec.2020",
        ),
    ] {
        let p = ReferencePicture::from_bytes(&with_chunks(&base, &[(*b"cICP", cp.to_vec())]))
            .expect(name);
        assert!(p.notes().is_empty(), "{name}");
        assert_close(mean_linear(&p), want, 2e-3, name);
    }
    // The profile wins over cICP when both are there.
    let both = with_chunks(
        &base,
        &[
            (*b"cICP", vec![1, 13, 0, 1]),
            (*b"iCCP", iccp(profile(ColorSpace::DisplayP3))),
        ],
    );
    let p = ReferencePicture::from_bytes(&both).expect("both");
    assert_close(mean_linear(&p), p3_icc, 1e-9, "iCCP before cICP");
    // PQ and HLG, primaries and a curve this reader does not know: not handled, taken as sRGB, and the card says so.
    for cp in [[9u8, 16, 0, 1], [9, 18, 0, 1], [5, 13, 0, 1], [1, 8, 0, 1]] {
        let p = ReferencePicture::from_bytes(&with_chunks(&base, &[(*b"cICP", cp.to_vec())]))
            .expect("unhandled cicp");
        assert_eq!(p.notes(), &[ReferenceNote::ProfileUnreadable], "{cp:?}");
        assert_close(mean_linear(&p), as_srgb, 1e-12, "fallback");
    }
}

#[test]
fn an_unreadable_profile_falls_back_to_srgb_with_the_note() {
    let rgb = p3_chart();
    let as_srgb = expected_mean(&rgb);
    let lut = fixture_profile(
        "with a table",
        FixtureLayout::V4,
        colorants_d50(ColorSpace::DisplayP3).0,
        [
            FixtureCurve::Identity,
            FixtureCurve::Identity,
            FixtureCurve::Identity,
        ],
        &[(*b"A2B0", vec![0; 64])],
    );
    let mut cut = profile(ColorSpace::DisplayP3).to_vec();
    cut.truncate(300);
    let mut grey = profile(ColorSpace::Srgb).to_vec();
    grey[16..20].copy_from_slice(b"GRAY");
    for (name, bytes) in [
        ("a lookup table", lut),
        ("cut short", cut),
        ("a grey profile on a colour picture", grey),
        ("noise", (0..600).map(|i| (i * 13 % 256) as u8).collect()),
    ] {
        let png = with_chunks(&png_rgb8(W, H, &rgb), &[(*b"iCCP", iccp(&bytes))]);
        let p = ReferencePicture::from_bytes(&png).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(p.notes(), &[ReferenceNote::ProfileUnreadable], "{name}");
        assert_close(mean_linear(&p), as_srgb, 1e-12, name);
        // The same through a JPEG and a TIFF.
        let j =
            ReferencePicture::from_bytes(&jpeg_bytes(W, H, &rgb, 100, false, Some(&bytes), None))
                .expect(name);
        assert_eq!(
            j.notes(),
            &[ReferenceNote::ProfileUnreadable],
            "{name} jpeg"
        );
        let t = ReferencePicture::from_bytes(&tiff_rgb8(
            W,
            H,
            &rgb,
            &[(tiff::tags::Tag::IccProfile, TiffTag::Bytes(bytes))],
        ))
        .expect(name);
        assert_eq!(
            t.notes(),
            &[ReferenceNote::ProfileUnreadable],
            "{name} tiff"
        );
    }
    assert_eq!(
        ReferenceNote::ProfileUnreadable.text(),
        "The picture's colour profile couldn't be read, so it is taken as sRGB."
    );
}

// ---------------------------------------------------------------------------------------------
// Orientation
// ---------------------------------------------------------------------------------------------

/// The corner of the displayed picture that is stored corner `A`, `B`, `C` or `D` (EXIF 2.32, table of orientations): displayed top
/// left, top right, bottom left, bottom right.
const DISPLAYED_CORNERS: [[usize; 4]; 8] = [
    [0, 1, 2, 3], // 1: as stored
    [1, 0, 3, 2], // 2: mirrored horizontally
    [3, 2, 1, 0], // 3: turned 180
    [2, 3, 0, 1], // 4: mirrored vertically
    [0, 2, 1, 3], // 5: transposed
    [2, 0, 3, 1], // 6: turned 90 clockwise to display
    [3, 1, 2, 0], // 7: transversed
    [1, 3, 0, 2], // 8: turned 270 clockwise to display
];

fn corner_colours(p: &ReferencePicture) -> [[f64; 3]; 4] {
    let a = p.analysis();
    let at = |x: usize, y: usize| a.linear[y * a.width + x];
    [
        at(a.width / 8, a.height / 8),
        at(a.width - 1 - a.width / 8, a.height / 8),
        at(a.width / 8, a.height - 1 - a.height / 8),
        at(a.width - 1 - a.width / 8, a.height - 1 - a.height / 8),
    ]
}

#[test]
fn exif_orientation_turns_a_tiff_and_a_jpeg_upright() {
    let (w, h) = (480usize, 300usize);
    let stored = quadrants(w, h);
    let corners = |o: u16| -> ([[f64; 3]; 4], (usize, usize)) {
        let t = ReferencePicture::from_bytes(&tiff_rgb8(
            w,
            h,
            &stored,
            &[(tiff::tags::Tag::Orientation, TiffTag::Short(o))],
        ))
        .expect("tiff");
        (
            corner_colours(&t),
            (t.analysis().width, t.analysis().height),
        )
    };
    let (reference, _) = corners(1);
    for o in 1..=8u16 {
        let (got, size) = corners(o);
        let want = DISPLAYED_CORNERS[usize::from(o) - 1];
        for (i, &stored_corner) in want.iter().enumerate() {
            assert_close(
                got[i],
                reference[stored_corner],
                1e-9,
                &format!("tiff orientation {o}, corner {i}"),
            );
        }
        let expect_size = if o >= 5 { (h, w) } else { (w, h) };
        assert_eq!(size, expect_size, "tiff orientation {o}");
        // A JPEG carries it in Exif, and the picture it makes is the same turn (4:4:4, high quality: flat colours stay flat).
        let j = ReferencePicture::from_bytes(&jpeg_bytes(w, h, &stored, 100, false, None, Some(o)))
            .expect("jpeg");
        let jc = corner_colours(&j);
        for (i, &stored_corner) in want.iter().enumerate() {
            assert_close(
                jc[i],
                reference[stored_corner],
                2e-2,
                &format!("jpeg orientation {o}, corner {i}"),
            );
        }
        assert_eq!(
            (j.analysis().width, j.analysis().height),
            expect_size,
            "jpeg orientation {o}"
        );
    }
    // A PNG has no orientation in this reader: it is as stored (its pixels are the picture).
    let png = ReferencePicture::from_bytes(&png_rgb8(w, h, &stored)).expect("png");
    assert_eq!((png.analysis().width, png.analysis().height), (w, h));
}

// ---------------------------------------------------------------------------------------------
// Bars, size and reduction
// ---------------------------------------------------------------------------------------------

#[test]
fn a_letterboxed_picture_is_trimmed_exactly_in_every_format() {
    let (w, h) = (480usize, 270usize);
    let pic = colourful(w, h, 5);
    let (d, ow, oh) = barred(&pic, w, h, 67, 67, 0, 0);
    let want = expected_mean(&pic);
    for (name, bytes) in [
        ("png", png_rgb8(ow, oh, &d)),
        ("tiff", tiff_rgb8(ow, oh, &d, &[])),
        ("jpeg q100", jpeg_bytes(ow, oh, &d, 100, false, None, None)),
    ] {
        let p = ReferencePicture::from_bytes(&bytes).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(
            (p.analysis().width, p.analysis().height),
            (w, h),
            "{name}: the bars are exactly gone"
        );
        assert_eq!(p.notes(), &[ReferenceNote::BlackBarsRemoved], "{name}");
        assert_close(
            mean_linear(&p),
            want,
            if name == "png" || name == "tiff" {
                1e-12
            } else {
                4e-3
            },
            name,
        );
    }
    // Uneven bars, and a pillarbox.
    let (d, ow, oh) = barred(&pic, w, h, 20, 41, 0, 0);
    let p = ReferencePicture::from_bytes(&png_rgb8(ow, oh, &d)).expect("uneven");
    assert_eq!((p.analysis().width, p.analysis().height), (w, h));
    let tall = colourful(300, 400, 6);
    let (d, ow, oh) = barred(&tall, 300, 400, 0, 0, 90, 80);
    let p = ReferencePicture::from_bytes(&png_rgb8(ow, oh, &d)).expect("pillarbox");
    assert_eq!(
        (p.analysis().width, p.analysis().height),
        (300, 400),
        "the pillarbox is exactly gone"
    );
    assert_eq!(p.notes(), &[ReferenceNote::BlackBarsRemoved]);
    // A window box.
    let (d, ow, oh) = barred(&pic, w, h, 30, 30, 50, 50);
    let p = ReferencePicture::from_bytes(&png_rgb8(ow, oh, &d)).expect("window box");
    assert_eq!((p.analysis().width, p.analysis().height), (w, h));
    assert_eq!(
        ReferenceNote::BlackBarsRemoved.text(),
        "Black bars were removed from the reference."
    );
}

#[test]
fn jpeg_ringing_at_the_edge_of_a_bar_is_handled_at_every_quality_and_alignment() {
    let (w, h) = (480usize, 270usize);
    let pic = colourful(w, h, 7);
    for quality in [30u8, 50, 75, 90] {
        for subsampled in [false, true] {
            for bar in [60usize, 61, 63, 67, 70, 75] {
                let (d, ow, oh) = barred(&pic, w, h, bar, bar + 3, 0, 0);
                let bytes = jpeg_bytes(ow, oh, &d, quality, subsampled, None, None);
                let p = ReferencePicture::from_bytes(&bytes).expect("jpeg");
                assert_eq!(
                    (p.analysis().width, p.analysis().height),
                    (w, h),
                    "q{quality} subsampled {subsampled} bar {bar}"
                );
            }
        }
    }
}

/// A night frame: a dark noisy sky that brightens towards the horizon and a lit town at the bottom. Its top 180 rows pass the POC's
/// two thresholds (a deviation under 2.0 and a mean under 16).
fn night_frame(w: usize, h: usize) -> Vec<u8> {
    let mut rng = Rng(99);
    let mut d = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        let t = y as f64 / (h as f64 * 0.8);
        for _ in 0..w {
            for _ in 0..3 {
                let v = if y < (h * 4) / 5 {
                    0.3 + 14.0 * t.min(1.0).powi(3) + (0.3 + 1.6 * t.min(1.0)) * rng.signed() * 1.7
                } else {
                    90.0 + 60.0 * rng.signed()
                };
                d.push(v.clamp(0.0, 255.0).round() as u8);
            }
        }
    }
    d
}

#[test]
fn a_night_frame_with_many_dark_rows_is_not_trimmed() {
    let (w, h) = (480usize, 300usize);
    let night = night_frame(w, h);
    for (name, bytes) in [
        ("png", png_rgb8(w, h, &night)),
        ("jpeg q90", jpeg_bytes(w, h, &night, 90, true, None, None)),
        ("tiff", tiff_rgb8(w, h, &night, &[])),
    ] {
        let p = ReferencePicture::from_bytes(&bytes).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(
            (p.analysis().width, p.analysis().height),
            (w, h),
            "{name}: nothing was cut"
        );
        assert!(p.notes().is_empty(), "{name}: {:?}", p.notes());
    }
    // The same frame rotated to a dark left edge: a dark sky on the side.
    let (d, ow, oh) = {
        let mut rot = vec![0u8; night.len()];
        for y in 0..h {
            for x in 0..w {
                // The frame turned a quarter: columns become rows (the picture is h wide and w tall).
                let (nx, ny) = (y, x);
                rot[(ny * h + nx) * 3..(ny * h + nx) * 3 + 3]
                    .copy_from_slice(&night[(y * w + x) * 3..(y * w + x) * 3 + 3]);
            }
        }
        (rot, h, w)
    };
    let p = ReferencePicture::from_bytes(&png_rgb8(ow, oh, &d)).expect("rotated");
    assert_eq!((p.analysis().width, p.analysis().height), (ow, oh));
    assert!(p.notes().is_empty());
    // A night frame that really has bars: the bar below the lit town goes (the picture steps up there); the bar above a sky as dark
    // and as flat as itself stays, because nothing tells the two apart (`docs/34` section 16.5, the limits).
    let (d, ow, oh) = barred(&night, w, h, 50, 50, 0, 0);
    let p = ReferencePicture::from_bytes(&png_rgb8(ow, oh, &d)).expect("barred night");
    assert_eq!((p.analysis().width, p.analysis().height), (w, h + 50));
}

#[test]
fn a_flat_dark_picture_is_not_trimmed_into_nothing_and_is_refused_as_empty() {
    let flat = vec![0u8; 400 * 300 * 3];
    assert_eq!(
        ReferencePicture::from_bytes(&png_rgb8(400, 300, &flat)).unwrap_err(),
        ReferenceError::Empty
    );
    let mut rng = Rng(5);
    let dim: Vec<u8> = (0..400 * 300 * 3)
        .map(|_| (8.0 + 0.7 * rng.signed()).round() as u8)
        .collect();
    assert_eq!(
        ReferencePicture::from_bytes(&png_rgb8(400, 300, &dim)).unwrap_err(),
        ReferenceError::Empty
    );
    assert_eq!(
        ReferenceError::Empty.to_string(),
        "There's no picture inside the black bars. Choose another picture."
    );
    // A picture that is merely dark, with texture, is a picture.
    let dark: Vec<u8> = (0..400 * 300 * 3)
        .map(|_| (10.0 + 9.0 * rng.signed()).round().max(0.0) as u8)
        .collect();
    let p = ReferencePicture::from_bytes(&png_rgb8(400, 300, &dark)).expect("dark");
    assert_eq!((p.analysis().width, p.analysis().height), (400, 300));
}

#[test]
fn a_picture_that_is_too_small_is_refused_after_its_bars_are_removed() {
    let n = MIN_SHORT_SIDE_PX;
    let ok = colourful(n + 100, n, 8);
    assert!(ReferencePicture::from_bytes(&png_rgb8(n + 100, n, &ok)).is_ok());
    let small = colourful(n + 100, n - 1, 9);
    assert_eq!(
        ReferencePicture::from_bytes(&png_rgb8(n + 100, n - 1, &small)).unwrap_err(),
        ReferenceError::TooSmall
    );
    let narrow = colourful(n - 1, 600, 10);
    assert_eq!(
        ReferencePicture::from_bytes(&png_rgb8(n - 1, 600, &narrow)).unwrap_err(),
        ReferenceError::TooSmall
    );
    // The size is the picture inside the bars: a tall frame with bars whose picture is too small is refused,
    // and one whose picture is big enough is not.
    let (d, ow, oh) = barred(&colourful(500, n - 1, 11), 500, n - 1, 100, 100, 0, 0);
    assert_eq!(
        ReferencePicture::from_bytes(&png_rgb8(ow, oh, &d)).unwrap_err(),
        ReferenceError::TooSmall
    );
    let (d, ow, oh) = barred(&colourful(500, n, 12), 500, n, 100, 100, 0, 0);
    assert!(ReferencePicture::from_bytes(&png_rgb8(ow, oh, &d)).is_ok());
    assert_eq!(
        ReferenceError::TooSmall.to_string(),
        format!(
            "This picture is too small to match. Use one that is at least {n} pixels on its short side."
        )
    );
}

#[test]
fn the_scene_and_analysis_pictures_are_the_area_average_and_are_never_enlarged() {
    // 2048 x 1280 of 2 x 2 blocks of two greys: every output pixel of 512 (a 4 x 4 box) averages the same number of each.
    let (w, h) = (2048usize, 1280usize);
    let mut rgb = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let v = if (x / 2 + y / 2) % 2 == 0 { 40 } else { 200 };
            rgb[(y * w + x) * 3..(y * w + x) * 3 + 3].copy_from_slice(&[v, v, v]);
        }
    }
    let p = ReferencePicture::from_bytes(&png_rgb8(w, h, &rgb)).expect("big");
    assert_eq!(
        (p.scene().width, p.scene().height),
        (SCENE_LONG_EDGE_PX, 640)
    );
    assert_eq!(
        (p.analysis().width, p.analysis().height),
        (ANALYSIS_LONG_EDGE_PX, 320)
    );
    // The average is in linear light: half of each, not the average of the codes.
    let want = 0.5 * (srgb_decode(40.0 / 255.0) + srgb_decode(200.0 / 255.0));
    for px in &p.analysis().linear {
        assert!((px[0] - want).abs() < 1e-12, "{px:?} against {want}");
    }
    // The scene picture (a 2 x 2 box) sees each block whole, and the mean of it all is the same.
    let (lo, hi) = (srgb_decode(40.0 / 255.0), srgb_decode(200.0 / 255.0));
    for px in &p.scene().linear {
        assert!((px[0] - lo).abs() < 1e-12 || (px[0] - hi).abs() < 1e-12);
    }
    let scene_mean =
        p.scene().linear.iter().map(|px| px[0]).sum::<f64>() / p.scene().linear.len() as f64;
    assert!(
        (scene_mean - want).abs() < 1e-9,
        "{scene_mean} against {want}"
    );
    let g = srgb_decode(40.0 / 255.0);
    assert!(
        want > 0.5 * (g + g) + 0.05,
        "linear light is not the mean of the codes"
    );
    // The scene picture as codes for the segmenter: 3 bytes a pixel.
    assert_eq!(p.scene().srgb8().len(), SCENE_LONG_EDGE_PX * 640 * 3);
    // Smaller pictures keep their size.
    let small = colourful(700, 600, 13);
    let p = ReferencePicture::from_bytes(&png_rgb8(700, 600, &small)).expect("small");
    assert_eq!((p.scene().width, p.scene().height), (700, 600));
    assert_eq!(
        (p.analysis().width, p.analysis().height),
        (ANALYSIS_LONG_EDGE_PX, 439)
    );
    // Tall pictures reduce along the long edge too.
    let tall = colourful(600, 1500, 14);
    let p = ReferencePicture::from_bytes(&png_rgb8(600, 1500, &tall)).expect("tall");
    assert_eq!(
        (p.scene().width, p.scene().height),
        (410, SCENE_LONG_EDGE_PX)
    );
    assert_eq!(
        (p.analysis().width, p.analysis().height),
        (205, ANALYSIS_LONG_EDGE_PX)
    );
}

#[test]
fn a_reference_has_no_way_to_be_written_and_its_debug_text_holds_no_pixels() {
    let rgb = colourful(W, H, 15);
    let p = ReferencePicture::from_bytes(&png_rgb8(W, H, &rgb)).expect("png");
    let text = format!("{p:?}");
    assert!(text.len() < 600, "{text}");
    assert!(
        !text.contains("0."),
        "no pixel value is in the debug text: {text}"
    );
}
