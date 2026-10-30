# Yellowmoon

**Cinema colour for Sony raw files.**

Yellowmoon is a free, open-source raw processor for macOS. Drop in ARW files from any Sony camera and every photo comes out developed, with the colour of a cinema camera: exposure, white balance, noise and colour are handled automatically. Pick a look, adjust a little if you want, export. Done. It's fast: everything after decoding runs on the Mac's graphics card, in the app and in the `ym` command-line tool, exports included. The GPU code is written for [wgpu](https://wgpu.rs), which runs it on Metal, not in Metal directly.

## Screenshots

![A photo developed automatically, Look tab](docs/images/readme/photo-look.png)

| | |
|---|---|
| ![Light tab: exposure, histogram, highlights and shadows](docs/images/readme/photo-light.png) | ![Colour tab: white balance and colour balance](docs/images/readme/photo-colour.png) |
| ![Colour warp: one pin on the grass, moved towards green](docs/images/readme/photo-colour-warp.png) | ![Detail tab: sharpening and noise reduction](docs/images/readme/photo-detail.png) |

![Export panel](docs/images/readme/photo-export.png)

The screenshots are of the development build on Linux (`tools/readme`), in the dark glass interface; on a Mac the photo is drawn by a native view, a Metal layer that wgpu draws into.

## Sony's colour and Yellowmoon's

One raw file from a Sony A1: on the left as the camera renders it, on the right as Yellowmoon develops it with its default look, Signature.

| Sony | Yellowmoon |
|---|---|
| ![The camera's own rendering of the photo](docs/images/readme/compare-sony.jpg) | ![The same raw file developed by Yellowmoon with the Signature look](docs/images/readme/compare-yellowmoon.jpg) |

## Why

I like how ARRI looks, but I can't afford an ALEXA. I've used LUTs that smart people make, and they're hit-or-miss: a LUT is static, and it can't know what light a photo was taken in.

So I built a tool that does know. Yellowmoon models how a Sony sensor and an ALEXA sensor each respond to light, estimates the light in every photo with a small learned model, and maps one sensor to the other for that light. The model is trained from scratch, on openly licensed and synthetic data only.

It's free, and it's simple: open your files, export, and never look back. Nothing to catalogue, nothing to maintain. It's how I process photos of my family and our trips.

## What it's for

I don't intend to compete with existing raw processors or to replace any of them. Yellowmoon is simply a quick way to get ARRI-like results from Sony cameras, open source and free.

## What it does

- Opens ARW files from every Sony camera: uncompressed, lossless and compressed.
- Detects your camera and uses its colour profile, built from published sensor measurements of 11 Sony bodies (plus the cameras that share their sensors).
- Estimates the light in each photo with a learned model and renders it the way an ALEXA (ALEV III sensor) would have seen it.
- Measures each photo's noise and sets noise reduction and sharpening to match it. Demosaicing and sharpening are variations of classic methods: an edge-directed colour-ratio demosaic, Richardson–Lucy deconvolution for capture sharpening and an unsharp mask with halo control for output sharpening.
- Built-in looks, and a guided import of ARRI's free Look Library.
- Exports JPEG, TIFF, PNG, and DNG with your look baked in.
- A command-line tool, `ym`, with the same results as the app, for scripts and batch jobs.
- Auto can ask a grader to fine-tune each photo: Apple's on-device model on macOS 27, or the Claude Code or Antigravity you have installed (below).

## How the colour works

In short: estimate the light, translate what the Sony sensor recorded into what an ALEXA sensor would have recorded under that light, then grade and display it the way a cinema colourist would.

1. **Raw.** The ARW is decoded to linear camera RGB: black level removed, clipped highlights rebuilt, demosaiced, denoised, lens-corrected.
2. **Light.** A small learned model (FFCC-style colour constancy, about 13,000 parameters per camera profile, trained on synthetic scenes) estimates the scene illuminant from the raw data and the camera's own white balance.
3. **Sensor map.** A 3×3 matrix built for that light (blended by light type and colour temperature) turns white-balanced Sony RGB into the RGB an ALEV III sensor would have recorded. It is fitted from Weta's measured Sony spectral sensitivities and my own estimate of the ALEV III curves, which I publish.
4. **ALEXA colour.** From there it follows the ALEXA's own path: ARRI's published camera matrices for 17 colour temperatures (interpolated in mired) into ARRI Wide Gamut 3, then Wide Gamut 4 and exposure. White balance, the sensor map and this step fold into one 3×3 matrix per photo, so colour changes are instant.
5. **Grade and look.** The highlight range (where the camera's clipping point lands, up to an ALEXA's at EI 800), then LogC4 (ARRI's log encoding), where contrast, colour balance, curves and looks work as in a grading suite. A look is a CDL plus a 3D LUT; ARRI's free Look Library imports directly.
6. **Display.** Our own display rendering (a tone curve and a few colour terms, written as plain formulas) renders the result for your screen or the export colour space.

The details, with every constant and gate, are in `docs/02-colour-pipeline.md`.

## Auto grading

Auto sets the exposure and the white of each photo by a fixed rule. That is **Default Auto**: it is the same on every computer and sends nothing anywhere. Beside it, another grader can look at each photo and fine-tune Auto's values: exposure, highlights, shadows, black level, contrast, warmth, tint, saturation and the colours of the shadows and highlights. Choose who grades in **Preferences › Auto grading**:

- **Apple Intelligence** (a Mac with Apple silicon, macOS 27 and Apple Intelligence turned on): Apple's on-device model. It runs on your Mac and sends nothing anywhere. Yellowmoon uses only the on-device model, never Apple's cloud models.
- **Claude Code** and **Antigravity (agy)**: the programs you installed and signed in to yourself. Yellowmoon runs them on your computer and hands them a small copy of each photo (768 pixels on the long side) and your instructions. The program sends them to its own service, Anthropic's or Google's, or the one you set it up to use, with your own account and plan; Yellowmoon makes no connection of its own and never touches your sign-in. The original file, its name and its place on disk stay on your computer. Antigravity is marked experimental.

How it behaves:

- Apple Intelligence is the default where it is available. Claude Code and Antigravity are never chosen for you: the first time you choose one, Yellowmoon says what is sent and where, and asks you to agree. Cancel keeps the previous choice. A row for a program that is installed but not usable is listed greyed out and says why, for example "Not signed in".
- A photo appears at once with Default Auto. When the answer arrives, a second or a few later, its Auto values and the picture update, and the Automatic card on the Look tab says who graded it. Exports wait for the answers. A photo whose answer fails keeps Default Auto's values.
- Only values you leave on Auto change. Anything you set by hand stays as you set it. Your own instructions, in Preferences, go to whichever grader you chose.
- Answers can differ. Apple's model gives the same answer on one Mac and macOS version, but Claude Code and Antigravity have no seed, so the same photo can be graded differently from run to run. Every export records which Auto made it and the model that answered (`ym:Auto` and `ym:AutoAdvisor` in its metadata), and its settings pin the values, so rendering from them gives the same picture.
- A grader that is not there is not shown, or is greyed out, and Auto works as before.
- `ym` follows the same setting and the same consent. `ym render --auto-provider claude` (or `apple`, `agy`, `yellowmoon`) chooses for one run; naming a program is your agreement to send photos to it for that run, and nothing is saved. `ym doctor` says what is available and, if not, why. `--apple-auto on` and `off` still work.

This is a first version: how much each answer moves a photo is still being tuned. The details are in `docs/22-auto-providers.md` and, for Apple's model, `docs/19-apple-auto.md`.

## Install

Download the latest `.dmg` from [GitHub Releases](https://github.com/ngriaznov/yellowmoon/releases/latest), or use Homebrew:

```
brew tap ngriaznov/yellowmoon https://github.com/ngriaznov/yellowmoon
brew install --cask yellowmoon
```

Yellowmoon needs macOS 14 Sonoma or later, on Apple silicon or Intel. Auto with Apple Intelligence needs macOS 27 on Apple silicon, with Apple Intelligence turned on in System Settings.

Open the `.dmg` and drag Yellowmoon to Applications. The app updates itself: it checks for a new version every hour and installs it in the background; click **Update ready** at the top right to restart into it now, or it runs the next time you open Yellowmoon (Yellowmoon › Check for Updates… checks at once). Versions 0.1.x are test releases.

### Allow Yellowmoon to open

Yellowmoon is free and is not signed with an Apple Developer ID, so the first time you open it macOS stops it with "Apple could not verify Yellowmoon is free of malware" (or "Yellowmoon is damaged and can't be opened"). The app is fine; macOS only knows that it was downloaded and not notarised by Apple. Allow it once, in one of these ways:

- **macOS 15 Sequoia and later.** Open Yellowmoon and choose **Done** in the warning. Open **System Settings › Privacy & Security**, scroll down to **Security**, click **Open Anyway** next to "Yellowmoon was blocked to protect your Mac", and confirm with your password or Touch ID. Open Yellowmoon again and choose **Open Anyway**.
- **macOS 14 Sonoma.** In Applications, Control-click (or right-click) Yellowmoon, choose **Open**, then **Open** again in the warning.
- **Terminal, any version** (also when macOS says the app is damaged). Remove the download mark, then open the app as usual:

  ```
  xattr -dr com.apple.quarantine /Applications/Yellowmoon.app
  ```

You only do this once. After that Yellowmoon opens like any other app, and the updates it installs itself don't ask again. A new `.dmg` downloaded by hand needs the same step.

## Command line

```
ym render ~/Pictures/2026-10-04 --look neutral --format jpeg --long-edge 3000
ym info DSC04204.ARW --json
ym render DSC04204.ARW --auto-provider yellowmoon   # Default Auto, the same on every computer
ym render DSC04204.ARW --auto-provider claude       # Claude Code grades it (sends a small copy of the photo to Anthropic)
ym looks import ~/Downloads/<ARRI look package>.zip
```

## Licence and credits

Apache-2.0. Model weights in `data/models/` carry the licence their model card states (a model trained on ShareAlike data is CC BY-SA 4.0, with its own `LICENSE` file). See `NOTICE` for third-party data and credits, including ARRI's published ALEXA colour matrices (via the Academy's aces-dev, AMPAS licence) and Weta's Sony sensor measurements (rawtoaces-data, Apache-2.0).

This product includes DNG technology under license by Adobe.

Yellowmoon is not affiliated with ARRI or Sony. ARRI and ALEXA are trademarks of ARRI AG; Sony is a trademark of Sony Group Corporation. ARRI's looks are never included with Yellowmoon: you download them from ARRI and the app imports them.
