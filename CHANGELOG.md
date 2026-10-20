# Changelog

All notable changes to Yellowmoon are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) (0.x during M0 to M3, 1.0.0 at the end of M4).

A version bump on `main` publishes a release (`docs/12-distribution.md`), so every "Release x.y.z" pull request
moves the entries below from Unreleased into a dated section for that version.

## [Unreleased]

### Added

- **Auto with Apple Intelligence** (macOS 27 on Apple silicon, with Apple Intelligence turned on). Apple's on-device model looks at each photo and grades it, as a colourist would, into the best finished photograph the scene allows while keeping its light (a night stays dark, golden hour stays warm): it sets the exposure, highlights, shadows, black level, contrast and saturation, and makes the white warmer or cooler and greener or more magenta. Its default instructions ask for this base look with no style; your own instructions can also ask it to colour the shadows and the highlights (teal shadows and warm highlights, for example). It runs on the Mac and sends nothing anywhere. It is on by default where it is available and can be turned off in Preferences; on other Macs the setting is not shown and Auto works as before. Only values left on Auto change. A photo appears at once and updates when the answer arrives; exports wait for the answers, and each export records which Auto made it. The Automatic card lists each value Apple's model set, with its short plan for the photo. `ym` follows the same setting, `--apple-auto on|off` overrides it for one run, and `ym doctor` says whether it is available.
- **Instructions for Apple Intelligence** in Preferences, under the Apple Auto switch: the text Apple's model reads before each photo, which you can rewrite (up to 2000 characters) to steer the grade, and **Reset to default** to go back. A change asks every photo again. `ym` uses the same instructions, and `--apple-auto-instructions FILE` uses a file's for one run. Reports and export metadata say whether the default or your own instructions were used.
- Contrast, saturation, highlights, shadows and black, and the shadows and highlights colour wheels, now have an Auto value, like exposure: 0 (the centre for a wheel) unless Apple Intelligence chose one. Re-run on the Automatic card puts them back to Auto only while Apple Intelligence acts on the photo; otherwise it leaves them as you set them. `ym render` takes `--contrast auto`, `--saturation auto`, `--highlights auto`, `--shadows auto` and `--black auto`.
- **Remove from filmstrip.** Right-click a photo in the filmstrip and choose "Remove from filmstrip" (or "Remove picked photos" while some are picked), or press Delete or Backspace to remove the photo on screen. The photo leaves the strip and its edits go with it; the file on disk is never touched, and the notice says so ("Removed 3 photos. The files are still on disk."). If the photo on screen was removed the next one is shown (the previous one when it was the last); with no photo left the start screen shows. The notice has an Undo button, and ⌘Z does the same while nothing was changed after the removal: the photos come back where they were, with their settings. Only the last removal can be undone, until the next removal or until the session is closed. An export that is already running finishes the photos it was started with. Burst frames that stay share the median exposure of the frames that are left.

### Changed

- Choosing a look now changes only the photo on screen; use Apply to › Picked or All to copy it.
- The photo is drawn straight from the graphics card's memory, without a copy through the processor, so panning and zooming are lighter.
- Zoomed in past 100 %, the photo is now enlarged and sharpened on the graphics card too, so panning and zooming there are as light as at the other zooms. The enlarged photo looks the same as before.
- **Contrast** works around mid grey as before but no longer pushes the darkest and brightest tones past black and white: it bends toward them instead, so blacks stay black. Colours keep their hue as contrast changes, and more contrast adds some colour as it adds contrast (less contrast takes some away).
- **Saturation** no longer makes colours darker. More saturation makes colours stronger at the same brightness, about as strongly as before, and less makes them greyer, down to grey at -100. Muted colours gain the most, colours that are already strong gain less, and no colour is pushed past black. The Signature look and other looks are unchanged. Photos with Contrast and Saturation at 0 render exactly as before (pipeline 0.4.3).
- `ym render --json` and `ym info --json`, and the settings an export records in its metadata (`ym:Settings`), now write contrast, saturation, highlights, shadows and black as Auto or a value, for example `{"kind": "manual", "value": 0}`, where they used to write a plain number, and the shadows and highlights colour wheels as `{"kind": "manual", "point": [0, 0]}` where they used to write a point. The JSON's `auto` object also lists these automatic values and where Auto came from. Scripts that read them as numbers or points need updating; older settings files still read as before.
- The Signature look, the default look for new photos, is richer: more contrast (a steeper curve around mid grey, darker shadows, brighter lights) and more colour (saturation 1.45, was 1.20). Mid grey stays where it was. Built-in looks are now prepared at a finer grid of 65 points, so the stronger curve stays smooth (pipeline 0.4.2).
- Output sharpening: the halo limiter no longer switches on tiny differences, so the graphics card and the CPU sharpen hard edges and clipped areas alike. Sharpened pictures change very slightly at a few edge pixels (pipeline 0.4.1).

### Fixed

- The ring around the look in use is no longer cut off. A tile in the first row, the first column or the last column of a section of the Look tab lost the 4 px of its ring that stood outside the grid; the ring is whole now, and so is the focus ring when you Tab to a tile.
- Crop: the handles on the right and bottom edges of the frame were cut off by up to 4 px when the frame is the whole photo, as it is when Crop opens. They are drawn just inside the photo, and the focus rings of the frame and of the handles on the edge are drawn inside them too.
- A look picked on one photo could end up on, or be shown wrongly for, another open photo when you moved to the next photo before the app had finished: the Look tab could keep showing the old look for a photo that had been given the new one, two picks made in turn on different photos could leave every photo with the first look, and the rest of a slider, wheel, curve or colour-warp drag could be sent to the next photo. Each pick and each drag now belongs to the photo it was made on, and the photo you are looking at shows the look it really has.
- Colour warp: Range now limits how far a pin's change reaches, even after a long drag; before, a long drag widened it so far that most of the picture moved. Colours that lie in the way of a long drag still move with it. Short moves look as they did (pipeline 0.4.4).

## [0.1.10] - 2026-10-17

Highlights that end where an ALEXA's would: the camera's clipping point now lands at white, and can be placed anywhere from where it was recorded up to white.

### Changed

- **RAW › Range › Highlights** (the card was Roll-off) now moves where the camera's clipping point lands, from where it was recorded (0) to white (1), as an ALEXA's own video output places its clip at white. Everything up to a stop under the clip is left alone, so mid grey never moves. The default is 1, the clip at white. The old highlight roll-off changed at most a few codes on real photos and is gone.
- Imported ARRI display LUTs and the LogC3-to-Rec.709 looks now receive the full highlight range, up to the clip an ALEXA at EI 800 would have, and apply ARRI's own shoulder to it.
- The white lift (Light › White) now sizes itself after the highlight range: a photo whose highlights the range already took to white gets less lift, and a photo whose brightest areas sit well under the clip gets the lift it had before.
- Every picture's brightest tones change (pipeline 0.4.0).

### Fixed

- A cancelled export gives its graphics memory back before the next picture is drawn. Work of the export that had not started yet used to keep its memory until its own turn in the queue.

## [0.1.9] - 2026-10-17

A colour warp on the Colour tab: pins that move chosen colours across a colour plane, as in DaVinci Resolve.

### Added

- **Colour warp** on the Colour tab, like DaVinci Resolve's Color Warper: drag on the colour plane to move a colour, and nearby colours follow smoothly. Lock holds a colour in place, Pick colour takes a colour from the photo, and each pin has Range, Shadows, Highlights, Pivot and Exposure. Greys never move. The plane shows your photo's own colours as faint dots.

## [0.1.8] - 2026-10-17

A neutral graphite interface, so nothing on screen tints your photo, with centred tabs, smaller slider knobs and a smaller point on the colour wheels.

### Changed

- The interface is now a neutral graphite grey instead of a cool blue. Nothing behind the panes or in them has a hue any more, so the colour on screen is your photo's, the yellow buttons' and the tabs'.
- The tool tabs sit in the middle of their row, sliders have a small round knob that leaves more room to drag, and the point on the colour wheels is smaller.

## [0.1.7] - 2026-10-17

The first release for macOS 14 Sonoma and later; Macs on macOS 12 and 13 stay on 0.1.6.

### Changed

- Yellowmoon now needs macOS 14 Sonoma or later. If your Mac runs macOS 12 or 13, keep version 0.1.6: this update installs by itself but will not open there, so download 0.1.6 again from GitHub Releases if that happens.

## [0.1.6] - 2026-10-16

A new dark interface of glass panes over a still, cool glow, so the photo is the brightest thing on screen.

### Changed

- A new dark glass interface: the same layout with new controls throughout, yellow kept for buttons, the photo count in the filmstrip's caption ("4 of 12 · 3 picked"), and larger thumbnails.
- The new interface needs a recent web engine from macOS (Safari 16.4 or later). On macOS 12 and 13, install the latest system and Safari updates before you update Yellowmoon, or the controls may not draw correctly.

## [0.1.5] - 2026-10-16

Before matches the camera more closely: no lens vignetting correction, and colours for the A1 II and A3000.

### Changed

- Before no longer brightens the corners: like the camera's own picture, it leaves out the lens vignetting correction. The A1 II uses the A1's colours, and the A3000 the generic E-mount colours.

## [0.1.4] - 2026-10-16

Before shows the photo as the camera renders it, photos can be added and settings copied between them, and the app feels like a Mac app: a pan moves only the photo, pinch zooms, text stays put, and it keeps up while you work.

### Added

- An **Add photos** button on the Photos screen, after Pick white. It opens the file dialog and adds what you choose after the photos that are open; the photo you are looking at stays where it is, and the new ones develop in the filmstrip. Files or a folder dropped anywhere on the Photos screen are added the same way, and ⌘O does the same as the button. A notice says what happened: "Added 3 photos. 2 were already open.", or "These photos are already open." when there was nothing new. In a window narrower than 1274 px the render status text beside the buttons is left out to make room.
- **Copy settings and Paste**, to move a photo's settings onto other photos. A row at the foot of the tool panel has the two buttons; the keys ⇧⌘C, ⇧⌘V and ⌥⇧⌘V (when no text box has focus) and the Edit menu's Copy Settings, Paste Settings and Paste Settings to Picked do the same, the last one pasting onto every picked photo in one go. Everything is copied except the crop, the turn and the Pick white point, which belong to the photo itself; settings that are Automatic stay Automatic, so each photo keeps its own exposure, white balance and noise, and a white balance that came from a picked white is pasted as the Kelvin and tint it found. Each photo's paste is one step of that photo's Undo. Photos that are still developing or can't be opened are skipped, and the notice says how many. What you copied stays in the app until you quit, also when you open another folder; it is never saved and never put on the system clipboard.

### Changed

- Before shows the photo as a Sony camera would render it by default: its own white balance, colours and tone, without anything Yellowmoon adds. It used to show the flat log picture.

### Fixed

- The notice bar on the Photos screen no longer covers the tool panel, so the buttons at its foot (Copy settings, Paste, and the Export panel's buttons) stay in reach while a notice shows.
- A two-finger pan over the photo moves only the photo. The whole window no longer moves with it, and nothing else on the screen bounces or scrolls under your fingers: the header, the filmstrip, the dialogs and the ends of the tool panel stay where they are.
- Pinching on the trackpad zooms the photo about your fingers. Before, it did nothing. ⌘ or Ctrl with the scroll still zooms.
- Text in the app is no longer selectable, so dragging over the window, a double click on the photo or ⌘A selects nothing. What you may want to copy still is: fields, the command line in the Export panel and the folder it saves to, the path of the command-line tool, error messages, file names and the About text. Buttons show the arrow, and only links show the pointing hand. The web page's own right-click menu (Reload, Back) no longer opens.
- The app keeps up while you drag a slider or pan the photo. Scrolling, pinching and dragging over the photo are sent once for each frame that is drawn, and the screen redraws only what changed instead of the whole window for every frame, every step of a zoom and every file an export finishes. The histogram and the middle readout on the Light and Curve tabs update up to five times a second while you work, and once more when you stop, instead of after every frame.
- The Look tab stays quick with hundreds of looks. Only the looks you can see are drawn again after an edit, and only once you have stopped moving a slider for a moment; the ones you scrolled away from keep their picture until you come back.

## [0.1.3] - 2026-10-16

ARRI Look Files, a section per look library, and no duplicate looks.

### Added

- ARRI Look Files. Yellowmoon now imports `.alf4` (ALEXA 35 looks), `.alf4c` (the creative look of a custom colour management; its own display transforms are not used, Yellowmoon's display rendering is) and `.aml` (looks for LogC3 cameras, made for Rec.709) next to `.cube` files, from the same places: Looks › Import, a folder, a zip or a single file, and `ym looks import`. A look file says what it expects as input, so it asks no question: a LogC4 look gets its colour correction and LUT, a Rec.709 look takes the place of the display rendering. The two Look Files downloads on ARRI's page (87 LogC4 looks, 261 LogC3 looks in three strengths) are recognised as packages, and ARRI's LogC4 library, which comes both as 3D LUTs and as `.alf4` files, stays one entry per look (the 3D LUT, which has more points, is kept). The LogC3 look files are listed as their own looks, one per strength. The file is kept byte for byte and checked on every load. Files for another colour space, ARRI's older `.xml` look files (ALF-1), damaged files and files with nothing to import are refused with a plain sentence, and the rest of a folder or zip still imports: the files that were left out are named on the Recognise step, and `ym looks import` prints them and exits 1. The look's Intensity setting inside a file is not read (a look opens at 100 %). `docs/08-looks.md` has the formats and what is still open.

### Changed

- The Look tab lists each library in its own section, which folds away, and a name that appears in two libraries says which one ("Film · LogC3"). ARRI's LogC4, LogC3, LogC3 to Rec.709 and ALF2 looks and your own LUTs each have a heading with a count, a search opens the sections that match, and a look's tooltip says what it is ("3D LUT, 65 points"). `ym looks list` has a LIBRARY column, and its JSON gains `library`, `format` and `lut_size`.
- Importing again never duplicates a look. The same file under another name is not added a second time (a renamed copy of your own LUT stays one entry, and a file you imported on its own that is one of ARRI's looks becomes that ARRI look when you import ARRI's download), a folder that holds both ARRI libraries (3D LUTs and look files) gives one entry per look, and an import that adds nothing leaves the list of looks untouched. A single file of ARRI's library gets the same name and id as it has in the whole download, so importing it alone and the download later never lists a look twice. If a file you had made your default look turns out to be one of ARRI's, the default follows it. If the 3D LUT of a look went missing, importing the look files brings the look back.

## [0.1.2] - 2026-10-16

Updates install themselves.

### Changed

- Updates install themselves. The app checks for a new version every hour (was every 6 hours) and, when it finds one, downloads and installs it in the background; the header then says Update ready, which restarts into it at once, and the next launch runs it anyway. Until now the app only offered the update ("Update available") and waited for a click. An update still never interrupts an export, and "Check for updates automatically" in Preferences still switches all of it off.

## [0.1.1] - 2026-10-15

Fixes for what the first test release showed on a Mac.

### Fixed

- The photo shows in the main view on a Mac. In 0.1.0 the picture was drawn on the graphics card behind the page, but the page's own white background covered it, so the photo cell stayed blank while the thumbnails and looks showed. The Photos screen now lets the picture through, and the window behind the page is paper white, so nothing else changes. A check on a Mac in CI opens photos and fails if the page covers the picture again.
- No more "method `backend` is never used" warning when building the app for release.
- The ARRI import wizard recognises ARRI's real 3D-LUT downloads. "ARRI Look Library LogC4 (log-to-log) 3D-LUTs" (87 looks, one folder, files called `1110 Black and White.cube`) was shown as "Not from a package we know" and locked Import: the rules asked for group folders the real zip does not have. Now the zip or folder name, the LUT size and the look codes (the first digit is the group) identify it, in any layout, and the three 3D-LUT packages are known by their file checksums. ARRI's camera look-file zips (`.alf4`, `.aml`) are told apart and the wizard says which 3D-LUT download to take.
- Import is no longer locked when a chosen package is missing but something else can be imported; an unknown set whose input was answered counts for the chosen package with the same input.

## [0.1.0] - 2026-10-15

The first test release, a universal disk image for Apple silicon and Intel Macs on GitHub Releases. It is not signed with an Apple Developer ID: the README says how to allow it to open the first time. It updates itself: each later 0.1.x is published with the signed update files, and the app offers it. These entries cover everything so far: the M0 skeleton, the `ym` command-line tool, the raw algorithms (`docs/algorithms/`) and the GPU path, with the owner's decisions (`docs/16`).

### Changed

- The Signature look is the default (the owner's decisions of 2026-10-14; pipeline version 0.3.0, was 0.2.0, golden images re-baselined). Every photo opens with more contrast and more colour than before: a soft curve on the brightness of the scene (slope 1.25 through the middle, a soft toe and shoulder) and a colour boost of 1.20, with middle grey exactly where it was. Neutral stays in the list as the look that changes nothing. The name may change later. An ARRI look you imported still replaces it, as before.
- Cameras without their own colour data can now read white balance correctly, starting with the A1. The A1 (and the A1 II) borrows the A7R IV's colour profile and used to read a daylight photo as about 4400 K, which made pictures too warm; it now reads about 5200 K, and its Daylight preset (labelled 5500 K) reads within 300 K of its label. The correction is a small factor on the body's red and blue white, taken from the white-balance presets the camera writes into its own files. Cameras with their own colour data are unchanged, and so are the other borrowing bodies until files with presets are available for them (`docs/03`).
- Highlights reach a real white. Each photo gets a small automatic lift of its brightest tones (at most about one stop on the 12 test photos, nothing for a photo whose highlights are not bright; middle grey and everything below one stop over it stay put), sized from the photo's own highlights. The Automatic card shows it as a White row, the Light tab has a White offset slider (0 is the automatic amount; bring the lift to nothing by moving it down), and `ym render --white-offset EV` does the same. The DNG carries the lift. An imported ARRI display LUT and the LogC3-to-Rec.709 looks keep ARRI's own highlights.
- Exposure, brightness and the DNG (the owner's decisions of 2026-10-14; pipeline version 0.2.0, was 0.1.2, golden images re-baselined). Auto uses the camera's own exposure and your exposure compensation in full: a fixed +1.11 EV, the median gain of the camera JPEG engines on twelve Sony photos, with a highlight guard that can only lower it, and the compensation you dialled on the camera reaches the picture as it does in the camera's JPEG. The brightness of the photo no longer moves Auto, so a shot taken too dark stays dark until you move the Exposure slider; this is an interim rule until the intelligent automatic settings arrive. Pictures are 0.85 EV brighter on screen, about as bright as a Sony JPEG, with the ARRI colours unchanged (the lift comes after the look, so a look still sees middle grey where it expects it). ARRI's own display renderings, an imported ARRI display LUT and the LogC3-to-Rec.709 looks, keep ARRI's brightness, so the Intensity slider of such a look also changes brightness. The DNG opens in Adobe's apps as bright as the app and carries every change except sharpening (no capture or output sharpening; the demosaic's shadow step stays). It now works with the LogC3-to-Rec.709 looks and an imported display LUT, approximately: the DNG is close to, not the same as, the picture in the app.
- The classical demosaic (R6) is now the adaptive ratio demosaic (`docs/05`; the owner's decision of 2026-10-13 after a blind look). In linear light its mean PSNR over nine cells is 36.05 dB against 34.43 for Menon, the previous default, and 20 % of edge pixels show zipper artefacts against 39 %. The demosaic stage takes 1.5 to 1.7 times Menon's CPU time, which adds 3 to 13 % to an export. Menon, LMMSE, bilinear and the log-guided demosaic stay available to tests through a switch; the method is not a setting. A photo whose noise model is missing or unusable renders as if it had no noise, instead of failing. Every default render changes, so the pipeline version is 0.1.2 (was 0.1.1) and the golden images are re-baselined.
- Lateral chromatic aberration is corrected on the mosaic before the demosaic (R5b), not in the lens resample after it. On the real files measured, with two independent estimators, it leaves less red/blue misregistration than correcting after the demosaic (`docs/05`, R9, has the numbers). The binned fit view and the thumbnails still correct it in R9 (`docs/05`).
- Saturation is subtractive: more saturation makes colours denser and darker, less makes them paler, and neutrals and the brightest channel stay (`docs/02`, C9). Renders at 0 are unchanged.
- The command-line tool is called `ym` everywhere (decision 2026-10-12): the binary, `--version`
  (`ym <version> (pipeline <v>, <sha>)`), the help text, the sidecar (`Contents/MacOS/ym`, so it cannot
  collide with the app's `Yellowmoon` binary), the Homebrew cask's `binary` stanza, the issue forms
  (`ym doctor`), the README and the docs. The sidecar files are `src-tauri/binaries/ym-<triple>`
  (`tools/dev/prepare-sidecar.sh`, `release.yml`).

### Added

- On a Mac, everything after decoding runs on the graphics card (Metal through wgpu), in the app and in `ym`, exports included: the raw chain, the lens corrections, the fit view, the colour chain, output sharpening, the resize and the export encoding (`docs/17-gpu.md`). The CPU stays the reference and the fallback: a graphics-card error makes the whole photo again on the CPU, with a note, and `ym render --cpu` asks for the CPU. Exports from the app and from `ym` are byte-identical on the same machine. A 50 MP export holds no full picture in memory any more (215 MiB of host memory at its peak, was 791 MiB).
- Output sharpening: the develop setting `output_sharpening` (0 to 200 %, default 100), the Output slider in Detail › Sharpening and `ym render --output-sharpening`. A linear-light luma unsharp mask with halo control, applied to every picture at the size it is shown or written, in JPEG, TIFF and PNG exports after the resize, never in a DNG; 0 switches it off. It makes a quality 92 JPEG 0.4 to 4.3 % larger (three real files, full size and 2048 px).
- The Tone card on the Look tab: Temperature, Contrast and Saturation, the same settings as on the Colour and Light tabs.
- Diffraction deconvolution for capture sharpening (R8), an Airy-pattern PSF from the f-number and the pixel pitch, with sensor widths in the camera registry for 27 of 28 bodies. It is not the default and has no setting; Richardson–Lucy stays until the owner decides.
- A log-guided fast demosaic, internal and not used by any render.
- WGSL kernels with parity tests on software Vulkan for the adaptive ratio demosaic, the log-guided demosaic, diffraction deconvolution and output sharpening. None is wired into a render yet.
- `docs/algorithms/`: the four algorithm specifications of the raw-algorithms package, the integration guide and the test vectors.
- `ym render` with every option of `docs/10-cli.md`: files, folders (`--recursive`) and `-` for paths on
  stdin; look, intensity, exposure (`--exposure`, `--exposure-offset`), white balance and light presets, the
  grade, noise reduction, sharpening, the RAW switches, roll-off, a forced profile and the display rendering;
  JPEG, TIFF, PNG and DNG with their options, `--out`, `--name`, `--skip-existing`/`--overwrite`; `--settings`
  (a file or stdin, one document or one per file name); `--json` (one object per file, as each finishes);
  `--require-gpu` (exit 4); a progress line on a terminal; Ctrl-C stops after the current tile, deletes the
  partial file and exits 130. Exit codes 0 to 5 as documented (M1-14, M3-14).
- `ym info` shows the camera profile and why it was chosen and the automatic estimates (exposure, white
  balance, light, noise model, sharpening, noise reduction), with `--json` in the shape of `render --json`.
- `ym looks list`, `import`, `verify`, `default` and `bake` (a 33-point `.cube` and a contact sheet), and
  `ym doctor` (GPU, render path, model weights with hashes, profiles, looks, folders; `--json` too).
- App and CLI parity test: the same job through the app's export path and through the binary is
  byte-identical for JPEG, TIFF (8 and 16 bit, none, LZW, ZIP), PNG (8 and 16 bit) and DNG, with options and
  settings (M3-14).
- Cargo workspace with the `ym-*` library crates, the `ym-cli` binary (`ym`), the `src-tauri` app crate
  and the test-only tools, with empty documented APIs and an explicit member list (M0-01).
- React, TypeScript and Vite frontend skeleton with Vitest and Playwright, the design tokens and a placeholder
  Start screen (M0-01).
- Continuous integration (`.github/workflows/ci.yml`): frontend tests and build with an npm licence check,
  Playwright end-to-end tests, Rust format, clippy and tests, `cargo deny`, a self-test that proves `deny.toml`
  rejects GPL, LGPL and banned crates, and checks for the fetch scripts.
  Later jobs (GPU on software Vulkan, decoder oracle, golden images, DNG validation, training dry run) are
  present as comments and are switched on as their tasks land (M0-02).
- Licence policy in `deny.toml`: Apache-2.0-compatible licences only; GPL, LGPL and AGPL crates and the crates
  `dng`, `rawler`, `rawloader`, `ljpeg` and `jpegxl-rs` are rejected in shipped code (M0-02).
- Issue forms for bug reports (asks for `ym doctor` output, macOS version, camera and steps) and feature
  requests, and a pull request template (M0-02).
- Release pipeline (`.github/workflows/release.yml`): a version bump on `main` builds the universal macOS
  bundle with a universal command-line sidecar, signs the updater artefacts, publishes the release with
  `latest.json` and updates the Homebrew cask in `Casks/yellowmoon.rb` (M0-03).
- `docs/release-signing.md`: the steps to create the updater signing key and set the repository secrets (M0-03).
- Test data fetching (`tools/fetch/`): `samples.sh` for CC0 Sony ARW files from RawDB and raw.pixls.us and
  `golden.sh` for the golden photo set, both checking SHA-256 against `tools/fetch/manifest.json` (M0-09).
- `docs/validation-data.md`: the public dataset or independent tool behind every gate in `docs/13`, so no
  gate needs a test shoot (M0-10).

[Unreleased]: https://github.com/ngriaznov/yellowmoon/commits/main
