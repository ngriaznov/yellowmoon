# ym-raw fuzz targets

`cargo-fuzz` project for the ARW container and the pixel decoders. It is its own workspace (nightly toolchain, not a
member of the root workspace). Property for every target: no panic and no unbounded allocation for
any input.

| Target | Exercises |
|---|---|
| `tiff_walker` | TIFF header, directory chains, SubIFD/EXIF/GPS pointers, every typed value reader, headerless directories with and without a value-offset bias |
| `sr2` | the pad cipher (twice is the identity), the decrypted SR2 parser, the encrypted-block path, level fitting |
| `probe` | the whole metadata probe: container, EXIF, maker note, SR2, raw IFD choice, storage checks, preview reader |
| `craw` | the cRAW block decoder and qstep map: arbitrary payload bytes and tone-curve knots, a short payload must be an error, a decoded block must be inside the curve and the qstep map the size of the image |
| `ljpeg` | the lossless JPEG decoder: headers, Huffman tables, restart intervals, predictors 1 to 7, point transform, subsampled components, the Sony 4:2:0 quirk; bounded by a sample budget |
| `ycbcr` | Lossless M and S: the fuzz input replaces the tile payload of a valid synthetic file (4:2:0, 4:2:2 or 4:4:4), decoded with both chroma modes |
| `decode_any` | `decode::decode_bytes` on arbitrary bytes: container, probe and every pixel decoder; a success has consistent sizes, a failure a typed error |

```
rustup toolchain install nightly --profile minimal
cargo install cargo-fuzz

# seed the corpus with valid synthetic files (the corpus is git-ignored)
cargo run -p ym-raw --features testgen --example gen_seeds -- crates/ym-raw/fuzz/corpus/probe
cp crates/ym-raw/fuzz/corpus/probe/* crates/ym-raw/fuzz/corpus/tiff_walker/
# seeds for every decoder target at once (probe, tiff_walker, decode_any, ljpeg, craw, ycbcr)
cargo run -p ym-raw --features testgen --example gen_seeds -- --fuzz-root crates/ym-raw/fuzz/corpus

cd crates/ym-raw
cargo +nightly fuzz run tiff_walker -- -max_total_time=600 -dict=fuzz/dictionaries/tiff.dict
cargo +nightly fuzz run sr2        -- -max_total_time=600
cargo +nightly fuzz run probe      -- -max_total_time=600 -dict=fuzz/dictionaries/tiff.dict
cargo +nightly fuzz run craw       -- -max_total_time=600
cargo +nightly fuzz run ljpeg      -- -max_total_time=600
cargo +nightly fuzz run ycbcr      -- -max_total_time=600
cargo +nightly fuzz run decode_any -- -max_total_time=600 -dict=fuzz/dictionaries/tiff.dict
```

The nightly CI job (`fuzz` in `.github/workflows/ci.yml`, schedule only) runs each target for 10 minutes, one matrix entry per target, with the flags above plus `-rss_limit_mb=2048 -timeout=20`; it seeds the corpus with `gen_seeds`, continues from the previous night's corpus (cache) and uploads `fuzz/artifacts/<target>/` when a target crashes. `crates/ym-color/fuzz` (`cube_parse`, `look_json`) is in the same job.

 `cargo fuzz build` builds
every target at once; `cargo fuzz run` takes one target at a time. A crash is saved under
`fuzz/artifacts/<target>/`; turn it into a unit test in the crate before fixing it.
