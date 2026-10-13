# Yellowmoon

**Cinema colour for Sony raw files.**

Yellowmoon is a free, open-source raw processor for macOS. Drop in ARW files from any Sony camera and every photo comes out developed, with the colour of a cinema camera: exposure, white balance, noise and colour are handled automatically. Pick a look, adjust a little if you want, export. Done.

> Status: in development. Nothing to download yet.

## Why

I like how ARRI looks, but I can't afford an ALEXA. I've used LUTs that smart people make, and they're hit-or-miss: a LUT is static, and it can't know what light a photo was taken in.

So we built a tool that does know. Yellowmoon models how a Sony sensor and an ALEXA sensor each respond to light, estimates the light in every photo with a small learned model, and maps one sensor to the other for that light. A second network, for demosaicing and noise, is built and waiting for its training run; until then classical algorithms do that work. Every model is trained from scratch, on openly licensed and synthetic data only.

It's free, and it's simple: open your files, export, and never look back. Nothing to catalogue, nothing to maintain. It's how I process photos of my family and our trips.

## What it's for

We don't intend to compete with existing raw processors or to replace any of them. Yellowmoon is simply a quick way to get ARRI-like results from Sony cameras, open source and free.

## What it does

- Opens ARW files from every Sony camera: uncompressed, lossless and compressed.
- Detects your camera and uses its colour profile, built from published sensor measurements of 11 Sony bodies (plus the cameras that share their sensors).
- Estimates the light in each photo with a learned model and renders it the way an ALEXA (ALEV III sensor) would have seen it.
- Measures each photo's noise and sets noise reduction and sharpening to match it. Demosaicing and denoising are classical algorithms today, and a learned network takes over once it is trained.
- Built-in looks, and a guided import of ARRI's free Look Library.
- Exports JPEG, TIFF, PNG, and DNG with your look baked in.
- A command-line tool, `ym`, with the same results as the app, for scripts and batch jobs.

## How the colour works

In short: estimate the light, translate what the Sony sensor recorded into what an ALEXA sensor would have recorded under that light, then grade and display it the way a cinema colourist would.

1. **Raw.** The ARW is decoded to linear camera RGB: black level removed, clipped highlights rebuilt, demosaiced, denoised, lens-corrected.
2. **Light.** A small learned model (FFCC-style colour constancy, about 13,000 parameters per camera profile, trained on synthetic scenes) estimates the scene illuminant from the raw data and the camera's own white balance.
3. **Sensor map.** A 3×3 matrix built for that light (blended by light type and colour temperature) turns white-balanced Sony RGB into the RGB an ALEV III sensor would have recorded. It is fitted from Weta's measured Sony spectral sensitivities and our own estimate of the ALEV III curves, which we publish.
4. **ALEXA colour.** From there it follows the ALEXA's own path: ARRI's published camera matrices for 17 colour temperatures (interpolated in mired) into ARRI Wide Gamut 3, then Wide Gamut 4 and exposure. White balance, the sensor map and this step fold into one 3×3 matrix per photo, so colour changes are instant.
5. **Grade and look.** Highlight roll-off, then LogC4 (ARRI's log encoding), where contrast, colour balance, curves and looks work as in a grading suite. A look is a CDL plus a 3D LUT; ARRI's free Look Library imports directly.
6. **Display.** The ACES 2.0 output transform (or ARRI's own display LUT, once imported) renders the result for your screen or the export colour space.

The details, with every constant and gate, are in `docs/02-colour-pipeline.md`.

## Install

Coming with the first release: a `.dmg` from GitHub Releases, or Homebrew:

```
brew tap ngriaznov/yellowmoon https://github.com/ngriaznov/yellowmoon
brew install --cask yellowmoon
```

The app updates itself.

## Command line

```
ym render ~/Pictures/2026-10-04 --look neutral --format jpeg --long-edge 3000
ym info DSC04204.ARW --json
ym looks import ~/Downloads/<ARRI look package>.zip
```

## Licence and credits

Apache-2.0. See `NOTICE` for third-party data and credits, including ARRI's published ALEXA colour matrices (via the Academy's aces-dev, AMPAS licence) and Weta's Sony sensor measurements (rawtoaces-data, Apache-2.0).

This product includes DNG technology under license by Adobe.

Yellowmoon is not affiliated with ARRI or Sony. ARRI and ALEXA are trademarks of ARRI AG; Sony is a trademark of Sony Group Corporation. ARRI's looks are never included with Yellowmoon: you download them from ARRI and the app imports them.
