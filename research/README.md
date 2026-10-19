# research/

Background for the decisions in `docs/`. Read-only: specs win where they differ. Nothing here is an instruction.

| File | What it is |
|---|---|
| `sony-arw-to-arri-look-research.md` | Main research report (Oct 2026): ALEXA spectral data, illuminant-conditioned sensor mapping, ARRI's published maths vs LUTs, joint demosaic/denoise, the Rust stack, licensing, risks |
| `arri-sensor-spectral-data-sources.md` | Search for ARRI sensor spectral data: what exists, licences, why we estimate our own curve |
| `alexa-ssf/` | The ALEXA ALEV III curve estimate: method, validation and the first sensor-map result (`out/sony_profiles.csv`, which a test reproduces). The curve ships in `data/alexa/`; the scripts are in `training/alexa/` |
| `auto-exposure-2026-10-14.md` | Why the photos looked dark, and the automatic-exposure rule that followed (`docs/02`) |
| `dynamic-vs-lut-2026-10-14.md` | Yellowmoon against a raw processor plus an ARRI-like LUT: what differs and what that means for the pipeline |
| `gui-toolkit-2026-10-16.md` | Rebuilding the interface in Slint and the alternatives; why it stays a web interface (`docs/16`) |
| `arri-highlight-clipping-2026-10-17.md` | How an ALEXA clips highlights (published clip per EI, hard linear clip, "normalized" vs "photometric" video), how much headroom Sony bodies have, why the C7 roll-off does nothing, and a proposed highlight range map |
| `ml-auto-settings-2026-10-18.md` | Learned automatic settings (exposure, temperature and tint, white lift, contrast) that set each photo as a professional would: methods, licence-clean data, industry Auto and where it falls short, scene understanding, verdicts on Hu et al. "Exposure" and Afifi et al. 2021, a ranked build plan and the decisions it needs |
| `scene-understanding-2026-10-19.md` | Models that could tell automatic settings what is in the scene, after the final blind A/B of `autoset`: licence-clean models and data, web-trained encoders, generative editors, a quantized vision-language model asked for the settings, and Apple's on-device models (Vision, Foundation Models, Core Image), with the cues that fix each failure |
| `raw-algorithms-prior-art.md` | Prior-art record for the four raw algorithms of `docs/algorithms/` (background for `docs/14` and `docs/16`; not legal advice) |
| `notes/sony-arw-formats.md` | Survey of Sony ARW variants and decoders (LibRaw, rawler, rawspeed, ExifTool), bodies 2024–2026, ARW 6.0 |
| `notes/dng-export.md` | How to export DNG that carries a look; app-by-app behaviour; tag list; licences; Rust writer options |
| `notes/noise-estimation.md` | Per-image noise estimation for uncalibrated cameras; Sony quirks; validation data |
| `notes/name-and-trademark.md` | Name availability and trademark search for "Yellowmoon" (not legal advice) |

The noise note mentions DPReview studio files as a possible validation source. Yellowmoon doesn't use DPReview files in any form; use the sources in `docs/test-photo-sources.md` instead.
The name note suggests a repository under the zerosize organisation. The repository is `github.com/ngriaznov/yellowmoon` instead (personal account, like plugout).
