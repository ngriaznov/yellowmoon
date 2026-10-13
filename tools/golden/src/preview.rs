//! The scene's JPEG for the ARW's maker-note preview (tag 0x2001).
//!
//! `ym_raw::testgen::ArwBuilder::preview_jpeg` writes the scene's own JPEG into the maker note byte for byte, so this
//! module only checks that the JPEG is the size the sidecar says ([`ensure_size`]) and offers [`jpeg_size`] and
//! [`pad_exact`] to the tests. (Before `preview_jpeg` existed the writer asked the builder for a padded placeholder and
//! swapped the JPEG in after the fact; the files it wrote then carried the JPEG padded with `COM` segments.)

use anyhow::{Result, ensure};
use ym_raw::metadata::jpeg_frame_header;

/// `(width, height)` from the frame header of a JPEG, if it has one.
pub fn jpeg_size(jpeg: &[u8]) -> Option<(u16, u16)> {
    let (_, pos) = jpeg_frame_header(jpeg)?;
    // `pos` is the length field of the frame header, then precision (1 byte), height and width (2 bytes each).
    let h = u16::from_be_bytes([*jpeg.get(pos + 3)?, *jpeg.get(pos + 4)?]);
    let w = u16::from_be_bytes([*jpeg.get(pos + 5)?, *jpeg.get(pos + 6)?]);
    Some((w, h))
}

/// `jpeg` padded with `COM` segments (after its first segment, so it still starts `FF D8 FF DB`) to exactly `target`
/// bytes. Needs `target` to be at least four bytes longer than the JPEG, or equal to it.
pub fn pad_exact(jpeg: &[u8], target: usize) -> Result<Vec<u8>> {
    ensure!(
        jpeg.len() <= target,
        "JPEG of {} bytes does not fit {target}",
        jpeg.len()
    );
    if jpeg.len() == target {
        return Ok(jpeg.to_vec());
    }
    ensure!(
        jpeg.starts_with(&[0xFF, 0xD8, 0xFF, 0xDB]),
        "preview JPEG must start FF D8 FF DB"
    );
    let first_len = usize::from(u16::from_be_bytes([jpeg[4], jpeg[5]]));
    let split = 4 + first_len;
    ensure!(split <= jpeg.len(), "truncated first JPEG segment");
    let mut gap = target - jpeg.len();
    ensure!(
        gap >= 4,
        "a padding gap of {gap} bytes cannot hold a COM segment"
    );
    let mut out = Vec::with_capacity(target);
    out.extend_from_slice(&jpeg[..split]);
    while gap > 0 {
        // A COM segment is FF FE, a 16-bit length that counts itself, then the comment: 4 to 65 537 bytes.
        let mut total = gap.min(65_537);
        if gap - total != 0 && gap - total < 4 {
            total -= 4; // leave at least four bytes for the next segment
        }
        out.extend_from_slice(&[0xFF, 0xFE]);
        out.extend_from_slice(&((total - 2) as u16).to_be_bytes());
        out.extend(std::iter::repeat_n(b' ', total - 4));
        gap -= total;
    }
    out.extend_from_slice(&jpeg[split..]);
    debug_assert_eq!(out.len(), target);
    Ok(out)
}

/// Fails unless `jpeg` has a frame header of `width` x `height` pixels (what the sidecar says).
pub fn ensure_size(jpeg: &[u8], width: u16, height: u16) -> Result<()> {
    ensure!(
        jpeg_size(jpeg) == Some((width, height)),
        "the preview JPEG is {:?}, the sidecar says {width} x {height}",
        jpeg_size(jpeg)
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ym_raw::testgen::tiny_jpeg;

    #[test]
    fn pad_exact_hits_every_gap_size_and_keeps_the_jpeg_valid() {
        let jpeg = tiny_jpeg(48, 32, 7);
        for gap in [
            0usize, 4, 5, 6, 7, 8, 9, 64, 65_537, 65_538, 65_540, 65_541, 65_544, 131_074, 200_001,
        ] {
            let target = jpeg.len() + gap;
            let padded = pad_exact(&jpeg, target).unwrap_or_else(|e| panic!("gap {gap}: {e}"));
            assert_eq!(padded.len(), target, "gap {gap}");
            assert!(padded.starts_with(&[0xFF, 0xD8, 0xFF, 0xDB]), "gap {gap}");
            assert_eq!(jpeg_size(&padded), Some((48, 32)), "gap {gap}");
            assert!(padded.ends_with(&[0xFF, 0xD9]), "gap {gap}");
        }
        for gap in [1usize, 2, 3] {
            assert!(
                pad_exact(&jpeg, jpeg.len() + gap).is_err(),
                "gap {gap} cannot hold a segment"
            );
        }
        assert!(pad_exact(&jpeg, jpeg.len() - 1).is_err());
        assert!(pad_exact(b"not a jpeg", 100).is_err());
    }

    #[test]
    fn jpeg_size_reads_the_frame_header() {
        assert_eq!(jpeg_size(&tiny_jpeg(321, 123, 1)), Some((321, 123)));
        assert_eq!(jpeg_size(b"\xFF\xD8\xFF\xD9"), None);
        assert_eq!(jpeg_size(b""), None);
    }

    #[test]
    fn ensure_size_compares_the_frame_header_with_the_sidecar() {
        let jpeg = tiny_jpeg(96, 64, 5);
        assert!(ensure_size(&jpeg, 96, 64).is_ok());
        assert!(ensure_size(&jpeg, 64, 96).is_err());
        assert!(ensure_size(b"not a jpeg", 96, 64).is_err());
    }
}
