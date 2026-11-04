# Provenance check

`shingles.py` compares `crates/ym-film/src/` with spektrafilm (Python, GPL-3.0), spektrafilm-rs (Rust, GPL-3.0) and vkdt's
`filmsim` module, by 8-token shingles of normalised tokens and by a skeleton of keywords and brackets (`docs/24` section 3.1 rule 5,
section 8). It counts and prints normalised token sequences with file names and line numbers; it prints no source text of the
other projects. The clones live outside the repository.

```
S=/path/to/scratch
git clone https://github.com/andreavolpato/spektrafilm.git $S/spektrafilm && git -C $S/spektrafilm checkout 3bb2c2d2801ff68b92019cf1dbcbb133d60832bc
git clone --depth 1 https://github.com/turbasvin/spektrafilm-rs.git $S/spektrafilm-rs
git clone --filter=blob:none --no-checkout https://github.com/hanatos/vkdt.git $S/vkdt
git -C $S/vkdt sparse-checkout set src/pipe/modules/filmsim && git -C $S/vkdt checkout
python3 -I tools/provenance/shingles.py --ours crates/ym-film/src \
  --other spektrafilm=$S/spektrafilm/src/spektrafilm --other spektrafilm-rs=$S/spektrafilm-rs/crates \
  --other vkdt-filmsim=$S/vkdt/src/pipe/modules/filmsim --baseline ym-color=crates/ym-color/src \
  --commit spektrafilm=<sha> --commit spektrafilm-rs=<sha> --commit vkdt=<sha> --out provenance.md
python3 -I -m unittest discover -s tools/provenance
```

The baseline (an unrelated Rust tree of ours) gives the chance level: normalised Rust of one style shares many short shingles.
A sequence is trivial when it is under 24 tokens, uses three or fewer kinds of token, or occurs in three or more of our files.
