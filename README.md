# Yellowmoon

**Cinema colour for Sony raw files.**

Yellowmoon is a free, open-source raw processor for macOS. Drop in ARW files from any Sony camera and every photo comes out developed, with the colour of a cinema camera: exposure, white balance, noise and colour are handled automatically. Pick a look, adjust a little if you want, export. Done.

> Status: in development. Nothing to download yet.

## Why

I like how ARRI looks, but I can't afford an ALEXA. I've used LUTs that smart people make, and they're hit-or-miss: a LUT is static, and it can't know what light a photo was taken in.

So we built a tool that does know. Yellowmoon models how a Sony sensor and an ALEXA sensor each respond to light, estimates the light in every photo, and maps one to the other for that light. Small learned models handle the rest of the raw work: demosaicing, noise and detail. We trained them from scratch, on openly licensed data and our own captures.

It's free, and it's simple: open your files, export, and never look back. Nothing to catalogue, nothing to maintain. It's how I process photos of my family and our trips.

## What it's for

We don't intend to compete with existing raw processors or to replace any of them. Yellowmoon is simply a quick way to get ARRI-like results from Sony cameras, open source and free.

## What it does

- Opens ARW files from every Sony camera: uncompressed, lossless and compressed.
- Detects your camera and uses its colour profile, built from published sensor measurements of 11 Sony bodies (plus the cameras that share their sensors).
- Estimates the light in each photo and renders it the way an ALEXA (ALEV III sensor) would have seen it.
- Learned demosaic and noise reduction, tuned for every ISO.
- Built-in looks, and a guided import of ARRI's free Look Library.
- Exports JPEG, TIFF, PNG, and DNG with your look baked in.
- A command-line tool, `yellowmoon`, with the same results as the app, for scripts and batch jobs.

## Install

Coming with the first release: a `.dmg` from GitHub Releases, or Homebrew:

```
brew tap ngriaznov/yellowmoon https://github.com/ngriaznov/yellowmoon
brew install --cask yellowmoon
```

The app updates itself.

## Command line

```
yellowmoon render ~/Pictures/2026-10-04 --look neutral --format jpeg --long-edge 3000
yellowmoon info DSC04204.ARW --json
yellowmoon looks import ~/Downloads/<ARRI look package>.zip
```

## Licence and credits

Apache-2.0. See `NOTICE` for third-party data and credits, including ARRI's published ALEXA colour matrices (via the Academy's aces-dev, AMPAS licence) and Weta's Sony sensor measurements (rawtoaces-data, Apache-2.0).

This product includes DNG technology under license by Adobe.

Yellowmoon is not affiliated with ARRI or Sony. ARRI and ALEXA are trademarks of ARRI AG; Sony is a trademark of Sony Group Corporation. ARRI's looks are never included with Yellowmoon: you download them from ARRI and the app imports them.
