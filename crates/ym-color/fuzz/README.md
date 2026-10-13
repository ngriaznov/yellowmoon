# ym-color fuzz targets

`cargo-fuzz` project for the `.cube` parser and the parametric look loader (`docs/02` Tests, `docs/13` Fuzzing). Like
`crates/ym-raw/fuzz` it is its own workspace (nightly toolchain, not a member of the root workspace). The seed corpus is
committed (`corpus/cube_parse/*.cube`, `corpus/look_json/*.json`); `artifacts/` is git-ignored.

| Target | Properties |
|---|---|
| `cube_parse` | `parse_bytes` never panics or aborts on allocation and returns promptly; an accepted file is self-consistent (entry counts, finite values, ordered domains); sampling it at NaN, infinite and huge inputs stays finite; writing it with `to_cube_string` and parsing it again gives the same cube |
| `look_json` | `ParametricLook::from_json` never panics; an accepted look evaluates without NaN and bakes (or returns a typed error, for example a CDL whose output overflows) |

```
rustup toolchain install nightly --profile minimal
cargo install cargo-fuzz --locked

cd crates/ym-color
cargo +nightly fuzz run cube_parse -- -max_total_time=600 -rss_limit_mb=2048 -timeout=20
cargo +nightly fuzz run look_json  -- -max_total_time=600 -rss_limit_mb=2048 -timeout=20
```

The nightly CI job (`fuzz` in `.github/workflows/ci.yml`) runs both targets for 10 minutes each, next to the seven targets of
`crates/ym-raw/fuzz`. A crash is saved under `fuzz/artifacts/<target>/`; turn it into a unit test in the crate before fixing it.

Runs on 2026-10-12 (AddressSanitizer, 4-vCPU machine, `-rss_limit_mb=2048`), recorded in `docs/02`:

| Target | Time | Runs | Coverage / features | Result |
|---|---|---|---|---|
| `cube_parse` | 3,601 s | 32,511,747 (9,028 per second) | 617 / 2,089 | no crash, timeout or out-of-memory |
| `look_json` | 241 s | 4,169,092 (17,299 per second) | 1,800 / 4,964 | no crash, timeout or out-of-memory |
