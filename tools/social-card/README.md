# Social card

`docs/images/social-card.png` is the picture GitHub shows when the repository is linked: the moon mark, the name and the tagline over the app with a photo open, 1280 × 640 px. It is variant B of `card.html`, drawn by `render.mjs` in Chromium (Playwright) from the README's Look tab screenshot.

GitHub has no API for it: upload the PNG in the repository's **Settings → General → Social preview**. Do it again whenever the file changes.

## Making it again

1. Take the README screenshots with `--keep-raw` (`tools/readme/README.md`). The card uses the lossless capture, not the shrunk PNG:

   ```
   npm run readme:screenshots -- --photos testdata/readme-photos/src --name Photos --pick A1M01749 --keep-raw --out ~/yellowmoon-shots
   ```

2. Render the card:

   ```
   node tools/social-card/render.mjs --variant B --shot ~/yellowmoon-shots/raw/photo-look.png --out docs/images/social-card.png
   ```

The run fails when Figtree did not load, an image is missing, any text is closer than 64 px to an edge (some sites crop the card), or text comes within 24 px of the screenshot. The picture is cut to 256 colours with dithering (`pngquant`, then `optipng`), which keeps it near 330 KB. GitHub refuses files over 1 MB.

## The page

`card.html` is one file with inline CSS and an SVG mark. Its colours come from the dark glass interface (`src/styles/glass-c2.css`): the paper, the still, neutral graphite aurora (grey blobs, no hue, since 0.1.8) and the glass frame. Yellow is only the mark. `render.mjs` puts the font (`design/fonts/figtree-latin-wght-normal.woff2`, OFL) in as a data URI when it renders, because a page opened from a file may not load a font from another file.

Options (`node tools/social-card/render.mjs --help`):

- `--variant A|B|C`: A puts the name on the left and the app on the right; B, the one in use, puts the name above the app; C puts the name over the photo alone.
- `--shot <png>`: the app screenshot.
- `--photo <jpg>`: a developed photo, for C.
- `--photo-pos`, `--crop`: where the photo sits in C.
- `--out <png>`: the output file.
- `--jpeg [q]`: also write a JPEG to compare.
- `--colors N`: the palette size.
- `--no-quant`: keep the PNG lossless.

## Licence

The photo is © Mykyta Griaznov, used with permission (NOTICE item 10), like the README screenshots. Figtree is under the SIL Open Font License (`design/fonts/OFL.txt`).

## Tests

`npx vitest run tools/social-card` (part of `npm test`) tests the command line and the page with the font put in.
