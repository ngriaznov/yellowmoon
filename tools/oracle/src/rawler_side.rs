//! rawler (LGPL-2.1) as an in-process oracle.

use std::path::Path;

use rawler::RawImageData;

/// What rawler made of a file, reduced to what the comparison needs.
pub struct Decoded {
    /// Samples per row of the full image.
    pub width: usize,
    /// Rows.
    pub height: usize,
    /// Components per pixel (1 for a mosaic, 3 for camera-processed data).
    pub cpp: usize,
    /// `width * height * cpp` samples.
    pub data: Vec<u16>,
    /// Black levels as rawler lists them (red, green, green, blue for a Sony mosaic).
    pub black: Vec<f32>,
    /// White levels.
    pub white: Vec<u32>,
    /// White-balance coefficients (red, green, blue, unused), green normalised to 1.
    pub wb: [f32; 4],
}

/// Decodes `path` with rawler; panics inside rawler are reported as errors.
pub fn decode(path: &Path) -> Result<Decoded, String> {
    let path = path.to_owned();
    // rawler panics on layouts it was not written for; the panic message is noise here.
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let result = std::panic::catch_unwind(move || rawler::decode_file(&path));
    std::panic::set_hook(previous);
    let image = match result {
        Ok(Ok(image)) => image,
        Ok(Err(e)) => return Err(e.to_string()),
        Err(_) => return Err("rawler panicked".to_owned()),
    };
    let RawImageData::Integer(data) = image.data else {
        return Err("rawler returned floating-point data".to_owned());
    };
    Ok(Decoded {
        width: image.width,
        height: image.height,
        cpp: image.cpp,
        data,
        black: image
            .blacklevel
            .levels
            .iter()
            .map(|r| r.n as f32 / r.d.max(1) as f32)
            .collect(),
        white: image.whitelevel.0.clone(),
        wb: image.wb_coeffs,
    })
}
