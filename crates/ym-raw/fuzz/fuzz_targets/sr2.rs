//! Fuzzes the SR2 parser: the pad cipher, the decrypted-directory parser and the level fitting.
//! The input is `key (4 bytes) | file offset (4 bytes) | block`. Property: no panic.

#![no_main]

use libfuzzer_sys::fuzz_target;
use ym_raw::layout::CfaPattern;
use ym_raw::sr2;

fuzz_target!(|data: &[u8]| {
    let Some((head, block)) = data.split_at_checked(8) else {
        return;
    };
    let key = u32::from_le_bytes([head[0], head[1], head[2], head[3]]);
    let file_offset = u32::from_le_bytes([head[4], head[5], head[6], head[7]]);

    // As if the block were already decrypted.
    let plain = sr2::parse_decrypted(block, file_offset);

    // As if it were the encrypted block of a real file.
    let block_info = sr2::Sr2Block {
        offset: file_offset,
        length: block.len() as u32,
        key,
    };
    let _ = sr2::parse_encrypted_block(block.to_vec(), block_info);

    // The cipher alone: twice is the identity.
    let mut copy = block.to_vec();
    sr2::apply_pad(&mut copy, key);
    sr2::apply_pad(&mut copy, key);
    assert_eq!(copy, block);

    // Whatever was parsed must survive the bit-depth check without panicking.
    if let Some(parsed) = plain {
        let black = parsed.black.unwrap_or([0; 4]);
        let white = parsed.white.unwrap_or([u16::MAX; 4]);
        for bits in [8u16, 12, 14, 16] {
            let _ = sr2::fit_levels(black, white, bits, CfaPattern::RGGB);
        }
    }
});
