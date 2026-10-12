# research/

Background for the decisions in `docs/`. Read-only: specs win where they differ. Nothing here is an instruction.

| File | What it is |
|---|---|
| `sony-arw-to-arri-look-research.md` | Main research report (Oct 2026): ALEXA spectral data, illuminant-conditioned sensor mapping, ARRI's published maths vs LUTs, joint demosaic/denoise, the Rust stack, licensing, risks |
| `arri-sensor-spectral-data-sources.md` | Search for ARRI sensor spectral data: what exists, licences, why we estimate our own curve |
| `alexa-ssf/` | The ALEXA ALEV III curve estimate: README with method and validation, `NOTICE`, scripts (`ssf_core.py`, `estimate_alexa.py`, `validate_loo.py`, `sony_profiles.py`, `plot_curves.py`, `fetch_sources.sh`) and outputs (`out/`: curve, variants, diagnostics, LOO results, Sony profile comparison, chart). `fetch_sources.sh` downloads the inputs; colour-science 0.4.7 |
| `notes/sony-arw-formats.md` | Survey of Sony ARW variants and decoders (LibRaw, rawler, rawspeed, ExifTool), bodies 2024–2026, ARW 6.0 |
| `notes/dng-export.md` | How to export DNG that carries a look; app-by-app behaviour; tag list; licences; Rust writer options |
| `notes/noise-estimation.md` | Per-image noise estimation for uncalibrated cameras; Sony quirks; validation data |
| `notes/name-and-trademark.md` | Name availability and trademark search for "Yellowmoon" (not legal advice) |
| `development-plan-doc-snapshot.md` | Text snapshot of the earlier Claude Docs plan, for reference. `HANDOVER.md` lists where this package supersedes it |

The noise note mentions DPReview studio files as a possible validation source. The plan doesn't use DPReview files in any form; use the sources in `docs/test-photo-sources.md` instead.
The name note suggests a repository under the zerosize organisation. The plan uses `github.com/ngriaznov/yellowmoon` instead (personal account, like plugout).
