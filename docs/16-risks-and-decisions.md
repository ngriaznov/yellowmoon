# 16 · Risks and open decisions

The biggest risk is the ALEXA target itself. Most open decisions are settled by M0 and M1 experiments. Record each decision here with its numbers when it's made.

## Risks

| Risk | Effect | Mitigation |
|---|---|---|
| ALEXA curve estimate is off by more than 5% | emulated colour drifts, mostly in saturated reds and greens | ensemble spread published; validate against the HdM and CIC 2015 curves if obtainable; swap in measured data if released |
| Far-red tail above 660 nm is unconstrained | skin and red fabrics under tungsten may render wrong | ensemble variants for the tail; targeted review of skin under tungsten in M1-H5 |
| Estimate pulled toward the stills-camera prior | real Sony-to-ALEXA differences understated | prior variants without Sony bodies (already within 2.6% RFSE); compare with ARRI-rendered sample footage offline |
| Light class hard to estimate from RGB alone | wrong map under LEDs and fluorescents | blend maps by class probability; light presets in the Colour tab; as-shot WB prior |
| Metal surface under a transparent Tauri webview | flicker, input routing or resize glitches | M0-06 spike; fallback: native child view |
| wgpu's Apple wide-gamut fixes unreleased in Oct 2026 | colour shifts in Display P3 output | pin the wgpu version; test P3 output in M0 against exported files |
| GPU limits on 50 MP frames | crashes or slow previews on smaller GPUs | tiled f16 pipeline; classical fallback; memory budget from device limits |
| burn API churn | rework on upgrades | ONNX models as the stable artefact; ort as fallback runtime |
| No GPU on CI runners | GPU code untested in CI | lavapipe job for correctness; Metal checks on a Mac before each gate |
| ARRI changes download URLs or objects to automated access | wizard breaks | import from file is the primary path; the wizard opens ARRI's page rather than deep links |
| Some Sony variants decode wrongly (older bodies, cRAW quirks, M/S) | wrong colour or artefacts | every corpus file checked against LibRaw in CI; clear message for anything unverified |
| ARW 6.0 codec: undocumented, hard-coded constants, possible Sony patents | A7 V and A7R VI compressed files don't open, or decode wrongly | ship last behind a feature flag; check against LibRaw master and lightcraft; ask Sony; clear message meanwhile |
| Lossless M/S files aren't Bayer | emulation is approximate | reduced path, flagged in the app |
| Noise estimate off on textured or filtered files | over- or under-smoothing | texture fallback, filtered-file check, bias toward over-estimating, σ jitter in training |
| Other apps render our DNGs differently | the look's tone differs outside Lightroom | look baked into pixels; full-size preview; per-app behaviour documented |
| Training compute unavailable | M2 slips | small models (hours on one GPU); Apple MPS fallback; classical path ships regardless |
| Few openly licensed A1 photos | golden set thin on scene and light variety | three licence tiers (`docs/test-photo-sources.md`); A1 II and A7R IV-family files as stand-ins |
| No controlled captures in V1 | illuminant, noise and lens accuracy on the A1 is validated only indirectly (synthetic, other cameras, oracles) | public-data gates per `docs/13`; noise biased toward over-estimating; captures in V2 tighten the A1 numbers |
| Updater signing key lost or leaked | installed copies can't update, or a third party could ship updates | key generated once, kept outside the repository, backed up in Mykyta's password manager; never logged |
| Yellow Moon Post Production (Northern Ireland) in a neighbouring field | naming dispute in the UK | clearance search before launch |
| Patent US 11948279 | claim against the joint demosaic+denoise network | FTO check before 1.0; classical path as a fallback |

## Open decisions

| Decision | Options | Decided by | Status |
|---|---|---|---|
| Native viewport approach | Metal layer behind transparent webview, or native child view; fallback: frames streamed to the webview | M0-06 spike | open: both backends are built (native `NSView` + wgpu surface in Display P3 behind the webview, `src-tauri/src/native.rs`, type-checked for macOS but never run; fallback of PNG frames with a Display P3 profile over `ym-frame://`, tested on Linux). Decided only after the Mac check (M0-H4; checklist in `docs/11-ui.md`). The fallback deviates from "full images never cross the bridge" (`docs/01`): it exists for non-macOS development and as the fallback if the native view fails checks 1 to 6 |
| ML runtime | burn on wgpu, or ort with CoreML | M0-08 benchmark | open: harness ready (`crates/ym-ml/examples/bench.rs`); Linux CPU numbers not decision-grade; needs the Mac run (M0-H4) |
| Illuminant estimator | FFCC, C5 or DMCC style | M1-06 gates | decided: FFCC-style (2026-10-12; log below) |
| Sensor map form | table blend, or tiny MLP | M1-05 (MLP only if ≥ 0.05 ΔE00 better) | table blend, Lab-optimal fit (2026-10-12); MLP not tried (headroom analysis: free/root-poly maps only reach LED-V2 ≈ 0.51) |
| Default display transform | ACES 2 as is, or a tuned variant | M1 visual comparison against ARRI's display LUT on test frames (offline) | ACES 2 by default |
| Tint mapping | our green-gain mapping (±0.5 stop) | M1 review | provisional |
| Curve domain | LogC4 after the look (default), or display-referred | M3 usability review | provisional: LogC4 |
| Before view | LogC4 "camera log" (as mocked), or Neutral without adjustments | M3 review | provisional: as mocked |
| CIE-derived tables | ship fitted coefficients with attribution, or restrict to formula SPDs + own measurements | M1-H4 | open |
| Signature look | its character and name | Mykyta in M3 | open |
| Code signing for 1.0 | unsigned like plugout, or Developer ID + notarisation | Mykyta before 1.0 | open |
| Trademark clearance | own search in EUIPO/INPI/WIPO/UK IPO, or a trademark attorney | Mykyta before announcing | open |
| Repository and domain | `github.com/ngriaznov/yellowmoon` (personal account, like plugout); domain given in the handover message | Mykyta | decided |

## Decision log

Add entries as `YYYY-MM-DD · decision · numbers · who`.

- 2026-10-13 · R6 classical default becomes the **adaptive ratio demosaic** (`Method::AdaptiveRatio`, `docs/algorithms/01`), the owner's choice after a blind look (RA-H4: six real files, 1:1 crops of edges, texture, colour and shadows, labels shuffled per file) and the `docs/05` rule "best PSNR becomes the default": in linear light over 9 cells (Kodak, McMaster, the synthetic set; signal scales 1, 0.1, 0.01) mean 36.05 dB against Menon 34.43 and LMMSE 34.28, 7 cells won; zipper 20 % against 39 % of edge pixels on the synthetic set. On gamma-encoded Kodak it trails (36.0 against 38.9 dB), which does not apply to Yellowmoon's linear-light demosaic. Menon and LMMSE stay selectable. CPU cost about 1.4–2× Menon until the GPU kernel (RA-09), accepted by the owner for the better look; goldens re-baselined with the change. · RA-05 numbers in `docs/05` R6 · Mykyta
- 2026-10-13 · The Look tab gets a "Tone" card under the Look card with Temperature, Contrast and Saturation, mirroring Colour › White balance › Kelvin, Light › Contrast and Colour › Saturation (same settings; nothing removed). Contrast and Saturation default to 0; Temperature shows the photo's Auto Kelvin with the AUTO tag, moving it makes the white balance manual and Reset returns it to Auto, as on the Colour tab; Tint stays on the Colour tab. Saturation becomes subtractive everywhere: per pixel in LogC4, `v′ = M + (v − M)·(1 + s/100)` with `M = max(r, g, b)`, so more saturation makes colours denser and darker (the brightest channel and neutrals are kept) and less makes them paler; replaces the additive `v′ = L + (v − L)·(1 + s/100)`. Renders at 0 are unchanged. · — · Mykyta
- 2026-10-13 · README screenshots show the real app on real photos. Mykyta's own ARWs live on the orphan branch `readme-photos` in Git LFS, never merged and not test data (an exception to the rule that raw files are never committed: clones of other branches fetch only LFS pointers). Screenshots made from them are published in the README; the photos stay © Mykyta Griaznov. · — · Mykyta
- 2026-10-13 · Output sharpening (RA-08, RA-H3) is a per-photo develop setting: `DevelopSettings.output_sharpening`, 0–200 %, default 100 %, slider "Output" in Detail › Sharpening beside the capture-sharpening Amount and Radius. Everything else is spec 04's screen preset (radius 1 px, threshold 1 level, halo suppression 60 %, smooth gain). It applies to every display-referred render at its own size (viewport, 100 % tiles, thumbnails, look tiles, raster exports after the resize), never to DNG. No export switch, no print preset in the UI. · — · Mykyta
- 2026-10-13 · The raw-algorithms package (four specs, C++ references, vectors) is treated as algorithm descriptions (known techniques plus the owner's own combinations and constants, per the prior-art record) and implemented in Rust from the specs; no C++ enters the repository (the references are a development oracle outside it; the generated vectors stay as fixed test data); a prior-art survey records where each piece comes from (`research/raw-algorithms-prior-art.md`). A provenance check found no GPL/LGPL/AGPL code or tables; `docs/14` row. · scan of RawTherapee b2ee504, librtprocess 9a85827, darktable ddc0bc5, LibRaw 7bfffe2, RCD-Demosaicing 0aa9d6e, rawspeed 0c16fee · Mykyta
- 2026-10-13 · The M1-08 auto-exposure gate uses G3: our `EV_auto` against the camera engine's own gain (`log2(0.18/x)`, `x` = the scene-linear level the embedded JPEG renders as mid grey, by quantile matching at the JPEG's size), instead of median luminance in linear light. The old metric mixed in the display transform (ACES 2 puts 0.18 at display 0.10, a camera JPEG near 0.18: about −0.85 EV by design) and the camera's tone curve. Tolerance ±0.5 EV and the 80 % bar are unchanged. · 9 real files (local only): old metric 5 of 9, G3 6 of 9 within ±0.5 EV, 9 of 9 within ±0.75; synthetic tier 37 of 37 · Mykyta
- 2026-10-12 · The command-line tool is called `ym`, not `yellowmoon`: binary, Homebrew cask link, "Install Command-Line Tool…" link, help text, docs and the `ym doctor` request in the issue template. The crate stays `ym-cli`. Inside the app bundle the sidecar is `Contents/MacOS/ym`, which doesn't collide with the `Yellowmoon` main binary on case-insensitive file systems, so the `yellowmoon-cli` workaround in docs/12 goes away. Rename lands with the CLI integration task (M1-14/M3-14). · — · Mykyta
- 2026-10-12 · Illuminant estimator: FFCC-style log-chroma filter with an edge channel, class head, learned confidence and the as-shot prior, one model per profile (13 k parameters). Beats grey world (+prior), grey edge, shades of grey, white patch and the as-shot value alone on held-out synthetic scenes; C5/DMCC need a multi-camera real dataset V1 can't get. · median angular error 1.22–1.30° on all 11 profiles (gate 2°); Cube++ and NUS-8 blocked · agent (M1-06), pending Mykyta's review
- 2026-10-12 · The profile camera locus and Duv tables use the published raw-scale Weta curves; the sensor maps and ALEXA white tables stay on per-channel peak-normalised curves (scale-invariant there). The first build used peak-normalised curves for the locus too, which put real as-shot whites at Duv −0.02…−0.05. · A7 III −0.040 → −0.0065, A7 IV −0.048 → −0.0008 on real files · orchestrator (defect fix)
- 2026-10-12 · Lens corrections follow darktable's embedded-metadata maths, verified with darktable-cli as a separate-process oracle: vignetting divides by G = 2^(0.5 − 2^(v·2^−13 − 1)) (not squared), knots at (i + 0.5)/(n − 1), linear interpolation; the same knot layout for distortion and CA. docs/05 R3/R9 corrected. · distortion and CA within 0.009 px, gain within 0.008 % (gates 0.2 px / 0.5 %) · agent (M1-12 review)
- 2026-10-12 · Default classical demosaic: Menon (DDFAPD). LMMSE leads on gamma-encoded test sets but our implementation is scale-dependent on linear data, which is what the pipeline feeds it. · Kodak linear 36.67/36.17/35.61 dB (LMMSE at scale 1/0.1/0.01) vs Menon 36.43 dB constant · orchestrator
- 2026-10-12 · Licence policy: the IJG licence (jpeg-encoder's forward DCT) is allowed in deny.toml; the acknowledgement "This software is based in part on the work of the Independent JPEG Group." goes in NOTICE, docs/14 and the About window. · — · orchestrator, pending Mykyta's OK
- 2026-10-12 · The updater signing key is not generated by the agent: the build environment is ephemeral and can't set repository secrets. Exact commands for Mykyta in docs/release-signing.md. · — · agent (M0-03)
- 2026-10-12 · Commits are authored as ngriaznov (GitHub noreply address) with no co-author trailers. · — · Mykyta
- 2026-10-12 · Sensor-map fit objective: Lab-optimal white-preserving 3×3 (was Y^(−2/3)-weighted RGB least squares). M1-05 gate for A7R IV with the old objective failed on LED-V2 (0.773 / 1.377) and D 6750 K (0.508); with the Lab objective 8 of 9 lights pass, LED-V2 stays a reported failure (0.759 / 1.515). · gate.py · Mykyta
