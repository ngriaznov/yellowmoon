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
| `photo-colour-warp.png` | Colour tab scrolled to the Colour warp card: the photo's colours on the chroma plane, and the pins of `--warp` if any (removed again afterwards) | that of `photo-colour` |
| `photo-detail.png` | Detail tab (sharpening, noise reduction) | the first |
| `photo-export.png` | Export panel over a photo | the first |
| `*-1x.png` | the same at half size, written when it is less than 60 % of the 2x file (always, with `--one-x always`) | |
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

A pick is `first`, `second`, `last`, `colourful`, a number (1 is the first in the filmstrip) or a file name, or part of one. `colour-warp` shows the photo of `colour` unless it is given one of its own (`--pick colour-warp=2`). Landscape photos with a lot of sky and colour make the best pictures: a portrait photo is letterboxed on the paper-white cell, as the app draws it.

`--export-to "~/Pictures/Lisbon export"` is the folder the Export panel says it saves to. The default is that sentence with the folder's name, as in the mockup; the real default is next to the photos and would put your own paths on the picture. Nothing is exported either way.

Other options (`npm run readme:screenshots -- --help`): `--states`, `--scale 1`, `--palette`, `--max-kb`, `--min-colours`, `--dither`, `--one-x`, `--no-optimise`, `--keep-raw`, `--dist`, `--binary`, `--data`, `--stable-ms`, `--timeout-min`.

The debug app reads the repository's `data/` folder at run time. When the app was built before a change to `data/` (the app then refuses every photo with an "unknown field" error), or `data/` is being edited, give `--data` a copy: `git archive HEAD data | tar -x -C /tmp/snapshot` and `--data /tmp/snapshot/data`.

## Pins on the Colour warp card

```
--warp "0.35,0.9:-0.014,0"                     one pin: the colour at 35 % across and 90 % down the photo, moved left on the plane
--warp "0.35,0.9:-0.014,0; 0.4,0.43:0.01,0"     two pins
```

Without `--warp` the card shows the photo's colours on the plane and no pins. With it, each pin is added the way a person adds one: Pick colour on, a click on the photo at `x,y` (a share of the photo cell's width and height; the photo fills the cell at Fit for a landscape photo), then the arrow keys on the new pin until it has moved by `da,db` on the chroma plane (right and up; one key press is 0.002, with Shift 0.02). The log prints each pin as the core holds it. After the picture the card's Reset removes the pins, so the states after it show the photo as it was.

The plane is 0.6 across and a photo's colours usually sit within 0.05 of its middle, so moves are small: 0.01 to 0.02 is a visible change. A pin on a colour close to grey moves everything near grey with it (the first try of the README picture turned a dry field pink); try a move with `ym render --settings` (a `color_warp` document) before taking the picture. The README picture is one pin on the grass of A1M01749, moved 0.014 to the left: greener, cooler grass, with the sky and the dry field left as they were.

## Size

A capture is 2880 × 1800 px and 4.5 to 4.8 MB as a PNG. Each one is cut down to an 8-bit PNG (at most 256 colours) with `pngquant`, then `optipng` or `oxipng` repacks it without changing a pixel. `--palette` chooses how the colours are shared out.

**`--palette whole` (the default).** One palette over the entire picture, dithered. `--dither` is 1 (pngquant's own default). The palette starts at 256 colours and is searched down, by bisection, until the file is under `--max-kb` (default 600), but never below `--min-colours` (192 in this mode). The dark glass interface is blurred translucent panes over a gradient, not flat colour: the right-hand panel alone has thousands of shades, and they only look smooth with a fine palette and dithering. The small accent colours (the cyan Auto pill, the yellow Export button and moon mark, the green and coral pills on the other tabs) stay as they are on the screen.

**`--palette regions`.** Three palettes, one for each kind of thing on the picture (the page tells the tool where the photos and plots are). It was the only mode before the glass interface, and it suits an interface of flat colours. `--dither` is 0.5 and `--min-colours` 96 in this mode.

- The interface keeps its 48 most frequent exact colours, mapped without dithering (thumbnails and the photo cell are not part of it), plus every colour of `design/tokens.css` that is on the picture at all. A small flat colour is not among the most frequent once the capture is halved for the 1x file (the moon mark is 20 pixels there, behind a thousand shades of grey from the text and the resize), and without this it came out grey or as the colour of the title block beside it.
- The colour wheels, the histogram and the curve plot get 40 colours of their own.
- The photographs (the photo cell, the filmstrip, the look tiles) get what is left, up to 168 colours, searched with `pngquant` for the largest palette that is under the target, not below `--min-colours`.

On the glass interface this mode comes out wrong, which is why `whole` is the default. The panel's gradient is mapped to its 48 most frequent colours without dithering, so it is blotchy, with dark teal patches. And the accent colours of the tabs are in `src/styles/glass-c2.css`, not in `design/tokens.css`, so nothing keeps them: the Light tab's cyan Auto pill came out white. On a flat interface it is the other way round: one palette over the whole picture loses what is small (the yellow moon beside a cyan title block came out as mud, the colour wheels as grey discs), because the photo's pixels outnumber them a thousand to one.

**The 1x files.** A `-1x.png` is made from the lossless capture, never from the 2x PNG: the capture is scaled to half with ImageMagick's Lanczos filter, then cut down once. It is the file the README uses, so it is the one to look at. It is written when it is under 60 % of the 2x file, or always with `--one-x always`.

**What to expect.** The 1x files come out at 256 colours and 450 to 470 KB, under the target, so the search stops at once. The 2x files cannot get under 600 KB with a dithered palette over the whole picture: they end at the floor of 192 colours at about 1.6 MB, and the log says so ("over 600 KB"). Use the 1x file, or raise `--max-kb`. `--min-colours 256` keeps the best palette when the target cannot be met anyway, and `--dither 0.5` makes a 2x file about 5 % smaller. A JPEG at quality 92 (4:4:4) is about 380 KB for a 1x capture, if the README ever prefers that to a palette PNG.

## What counts as "final"

A picture is taken only when, on the Photos screen: a frame of the cell's exact size is drawn, no "Still developing" note covers the photo, every thumbnail of the filmstrip is drawn, every look tile on the Look tab shows the open photo (the address of a tile names the photo it was made from), the Light tab's histogram has its curves, the Colour warp plane has the photo's colours, the Export panel has its size estimate, and the page has not changed for 1.5 s. After the picture it looks again, and takes it again if anything moved. Before each photo picture the photo is drawn once more (Before on, then off), so the render time in the header is a warm one. After the pins of `--warp` it is drawn twice: the first time can take the late frame of the last key press for its own, and the header then shows that frame's cold time.

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
