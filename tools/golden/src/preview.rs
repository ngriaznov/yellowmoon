//! Putting the scene's JPEG into the ARW's maker-note preview (tag 0x2001).
//!
//! `ym_raw::testgen::ArwBuilder` always writes its own tiny gradient JPEG there; it has no way to take a JPEG from the
//! caller. This module works around that without changing the container layout: the builder is asked for a preview of
//! the right size padded to a length that fits the real JPEG, the placeholder is found through the probe, and it is
//! replaced byte for byte by the real JPEG padded with `COM` segments to exactly the same length. Nothing moves, so
//! every offset and length in the file stays valid. (A `preview_jpeg` setter on the builder would make this
//! unnecessary; `docs/13-testing.md` lists it as a follow-up.)

use anyhow::{Result, bail, ensure};
use ym_raw::metadata::{jpeg_frame_header, probe_bytes};
use ym_raw::testgen::{pad_jpeg_to, tiny_jpeg};

/// The seed `ArwBuilder` uses for the maker-note preview (`tiny_jpeg(w, h, 2)`).
const BUILDER_PREVIEW_SEED: u64 = 2;

/// What to ask the builder for so that the placeholder is long enough to be replaced by `jpeg`.
pub fn placeholder_min_bytes(jpeg: &[u8], width: u16, height: u16) -> usize {
    jpeg.len()
        .max(tiny_jpeg(width, height, BUILDER_PREVIEW_SEED).len())
        + 64
}

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

/// Replaces the builder's preview placeholder in `file` by `jpeg`. `(width, height)` and `min_bytes` are what the
/// builder was given (`preview(Some((width, height)))`, `preview_min_bytes(min_bytes)`).
pub fn embed(
    file: &mut [u8],
    jpeg: &[u8],
    width: u16,
    height: u16,
    min_bytes: usize,
) -> Result<()> {
    ensure!(
        jpeg_size(jpeg) == Some((width, height)),
        "the preview JPEG is {:?}, the sidecar says {width} x {height}",
        jpeg_size(jpeg)
    );
    let placeholder = pad_jpeg_to(tiny_jpeg(width, height, BUILDER_PREVIEW_SEED), min_bytes);
    let meta = probe_bytes(file).map_err(|e| anyhow::anyhow!("probing the written file: {e}"))?;
    let Some(found) = meta.preview else {
        bail!("the written file has no preview to replace");
    };
    let (offset, length) = (found.offset as usize, found.length as usize);
    ensure!(
        length == placeholder.len() && file.get(offset..offset + length) == Some(&placeholder[..]),
        "the probe's preview ({length} bytes at {offset}) is not the builder's placeholder"
    );
    let padded = pad_exact(jpeg, length)?;
    file[offset..offset + length].copy_from_slice(&padded);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn the_placeholder_is_always_long_enough() {
        let big = vec![0u8; 10];
        assert!(
            placeholder_min_bytes(&big, 96, 64)
                >= tiny_jpeg(96, 64, BUILDER_PREVIEW_SEED).len() + 64
        );
        let huge = vec![0u8; 1_000_000];
        assert!(placeholder_min_bytes(&huge, 96, 64) >= 1_000_064);
    }
}
