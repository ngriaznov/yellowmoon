# README screenshots

`screenshots.mjs` takes pictures of the real Yellowmoon app showing real photos, for the project README. It runs the debug Tauri shell under Xvfb, opens your photos the way a person does (a drop on the window), waits until every frame, thumbnail and tile on screen is final, and writes one PNG per state, shrunk for the web.

```
npm run build && cargo build -p yellowmoon-app        # once, or pass --build
npm run readme:screenshots -- --photos ~/Pictures/Lisbon --out ~/yellowmoon-shots
```

It reuses the machinery of `npm run e2e:shell` (`e2e/real/host.mjs`: the server that hands the built frontend to the app, the Xvfb screen, the app, the screenshot; `e2e/real/kit.js`: the helpers that run inside the webview). The app starts clean: an empty `YELLOWMOON_HOME`, so no imported looks and default preferences. Nothing is written next to your photos and nothing is exported (the Export panel is only opened). Originals are only read.

## What it writes

| File | State | Photo (default) |
|---|---|---|
| `start-ready.png` | Start screen after the drop, every photo developed (the list with its READY tags) | all |
| `photo-look.png` | a photo open, Look tab (Automatic card, intensity, look tiles) | the first |
| `photo-light.png` | Light tab (exposure, histogram, highlights and shadows); left out for a single photo | the second |
| `photo-colour.png` | Colour tab (white balance card, colour balance wheels) | the most colourful |
| `photo-detail.png` | Detail tab (sharpening, noise reduction) | the first |
| `photo-export.png` | Export panel over a photo | the first |
| `*-1x.png` | the same at half size, written when it is less than 60 % of the 2x file | |
| `manifest.json` | what each file shows, its size, which photos were used | |

The pictures are of the webview only (no window frame, no menu bar), 2880 × 1800 px: a 1440 × 900 CSS pixel window at a device scale of 2, the size of `design/screenshots/`. The page is made 1440 × 900 with `xdotool`, because the window the app opens has a 25 px menu bar inside its 900 px on Linux.

The "most colourful" photo is worked out from the thumbnails of the filmstrip (Hasler and Süsstrunk's colourfulness measure, run in the page); the numbers are printed in the log.

## Requirements

- Linux (the debug app is the Linux build; the pictures are of that build, not of the Mac one), Node 22.
- `Xvfb`, ImageMagick (`import`, `convert`) and `xdotool`: `sudo apt install xvfb imagemagick xdotool`.
- `pngquant` and `optipng` (or `oxipng`) to shrink the PNGs: `sudo apt install pngquant optipng`. Without pngquant ImageMagick's own quantiser is used, which is worse; with neither the PNGs stay lossless and large.
- `dist/` (`npm run build`) and `target/debug/yellowmoon-app` (`cargo build -p yellowmoon-app`), or `--build` to make both first. The tool never builds on its own, so it can't fail halfway through a compile somebody else is editing.
- Port 1420 free: the debug app loads its frontend from there, as `npm run dev`, Playwright and `npm run e2e:shell` do. The tool waits up to ten minutes for a run that is in the way.

## Choosing the photos

```
--photos <folder-or-files...>   one folder (the app names the session after it), or ARW files
--name Lisbon                   the folder name to show; the photos are linked, not copied, into a temporary folder of that name
--pick look=first,light=2,colour=colourful,detail=DSC04204,export=last
--pick DSC04204                 one photo for every state
```

A pick is `first`, `second`, `last`, `colourful`, a number (1 is the first in the filmstrip) or a file name, or part of one. Landscape photos with a lot of sky and colour make the best pictures: a portrait photo is letterboxed on the paper-white cell, as the app draws it.

`--export-to "~/Pictures/Lisbon export"` is the folder the Export panel says it saves to. The default is that sentence with the folder's name, as in the mockup; the real default is next to the photos and would put your own paths on the picture. Nothing is exported either way.

Other options (`npm run readme:screenshots -- --help`): `--states`, `--scale 1`, `--max-kb`, `--min-colours`, `--dither`, `--one-x`, `--no-optimise`, `--keep-raw`, `--dist`, `--binary`, `--data`, `--stable-ms`, `--timeout-min`.

The debug app reads the repository's `data/` folder at run time. When the app was built before a change to `data/` (the app then refuses every photo with an "unknown field" error), or `data/` is being edited, give `--data` a copy: `git archive HEAD data | tar -x -C /tmp/snapshot` and `--data /tmp/snapshot/data`.

## Size

The target is 600 KB for each PNG. The interface is flat colour and compresses to almost nothing; the photo does not: a 2880 × 1800 capture is 1.7 to 3.4 MB, and as a 256-colour PNG still 0.7 to 0.9 MB. The picture is shrunk to an 8-bit PNG with three palettes, because one palette over everything loses what is small (the yellow moon beside a cyan title block came out as mud, the colour wheels as grey discs):

- the interface keeps its 48 most frequent exact colours, mapped without dithering (the page tells the tool where the photos are, so thumbnails and the photo cell are not part of it), plus every colour of `design/tokens.css` that is on the picture at all: a small flat colour is not among the most frequent once the capture is halved for the 1x file (the moon mark is 20 pixels there, behind a thousand shades of grey from the text and the resize), and without this it came out grey or as the colour of the title block beside it;
- the colour wheels, the histogram and the curve plot get 40 colours of their own;
- the photographs (the photo cell, the filmstrip, the look tiles) get what is left, up to 168 colours, searched with `pngquant` for the largest palette that is under the target, not below `--min-colours` (dithering at 0.5, so a sky stays smooth). `optipng` or `oxipng` then repacks the file.

On the test photos the 2x files ended with 96 to 168 photo colours at 580 to 600 KB, except a detailed photo (a brick tower in front of trees), which is 900 KB at the floor of 96. A photo bands visibly below about 96 colours (the default floor): when a picture is still over the target there, the log says so; raise `--max-kb`, use the 1x file (about 320 KB at 256 colours), or lower `--min-colours` and look at the result. A JPEG at quality 92 is about 500 KB for the same capture, if the README ever prefers that to a palette PNG.

## What counts as "final"

A picture is taken only when, on the Photos screen: a frame of the cell's exact size is drawn, no "Still developing" note covers the photo, every thumbnail of the filmstrip is drawn, every look tile on the Look tab shows the open photo (the address of a tile names the photo it was made from), the Light tab's histogram has its curves, the Export panel has its size estimate, and the page has not changed for 1.5 s. After the picture it looks again, and takes it again if anything moved. Before each photo picture the photo is drawn once more (Before on, then off), so the render time in the header is a warm one.

## Updating the README

1. Run the tool on photos you may publish (below), into a folder outside the repository, and look at every PNG at full size.
2. Copy the PNGs you want into `docs/screenshots/` (or wherever the README points).
3. In `README.md`, link them with a fixed width, so the 2x file is crisp on Retina displays and the page stays narrow:

   ```html
   <img src="docs/screenshots/photo-look.png" alt="Yellowmoon with a photo open on the Look tab" width="800">
   ```

4. Add the credit line under the pictures (below) and commit the PNGs with the README change.

## Licence rule

A screenshot of a photo is a copy of the photo. Only use photos you may publish: your own, or openly licensed ones, and credit them. A line like

> Photos: © Your Name, used with permission. / Photo by Name, CC BY 4.0.

goes under the pictures in the README (and `NOTICE` if the licence asks for it).

The files in `testdata/real/` are for local evaluation only (unclear licences). The tool may be tried on them, but the pictures must be written outside the repository; it refuses to write inside the repository when a photo comes from `testdata/`.

## Known differences from the Mac app

- The Linux build draws with the classical CPU path and the fallback viewport (the frame is a PNG shown by the webview), so the header says "Classical · N ms · Display P3" and not "Neural". The Mac app draws with Metal.
- The pictures are of WebKitGTK's rendering: text smoothing is grey-scale (the tool sets it). The frames of the photo are tagged Display P3 and WebKitGTK converts them to sRGB for the screen (checked against `sample_pixel`: a saturated blue reads 51, 83, 115 in P3 and 40, 83, 118 on the screen), so the PNGs are plain sRGB; a P3 display shows the Mac app's photo slightly more saturated than these.
- The Look tab shows the built-in looks only: the app starts with no ARRI looks imported, and none may be added to the repository.
- The Start screen lists the photos with their READY tags; the app does not draw thumbnails there.

## Tests

`npx vitest run tools/readme` (part of `npm test`) tests the command line, the photo and licence rules, shrinking and the shared server without the app. Running the tool is the test of the rest.
