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
| Native viewport approach | Metal layer behind transparent webview, or native child view | M0-06 spike | open |
| ML runtime | burn on wgpu, or ort with CoreML | M0-08 benchmark | open |
| Illuminant estimator | FFCC, C5 or DMCC style | M1-06 gates | open |
| Sensor map form | table blend, or tiny MLP | M1-05 (MLP only if ≥ 0.05 ΔE00 better) | open |
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
