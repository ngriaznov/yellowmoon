# Depth mask mockups (D-05, first part)

Mockups of the Depth mask's interface for the owner's review (D-H1). **`docs/23` §4.8 and `docs/11` (The Depth card) win wherever the two differ**; where they are silent, the copy here is a proposal and `mockup-notes.md` lists it. After approval the spec text is corrected to match these pictures.

**These are mockups, not screenshots of the app.** The interface around the pictures is the app's real DOM and CSS (glasscn components, `design/tokens.css`, lifted from the node mockups in `design/nodes/`), at 1440 x 900 and shot at 2x (2880 x 1800 JPEGs). `index.html` is the walk-through with a caption under each picture.

## What is made up

The photo is the owner's own (`design/nodes/html/assets/photo-graded.jpg`). **The grey depth picture and every pink mask are stand-ins, not the model's output.** `make-pictures.py` draws the depth from the photo's layout (sky and hills far, the field in the middle, the grass near), cuts the mountain out as one flat plane, adds a little of the photo's texture and blurs it, then rank-normalises it and cuts the bands with the rules of `docs/23` §4.8 (the two-decimal Pick depth band a fifth wide, the smoothstep ramp). So the numbers on the cards agree with the pictures, but they say nothing about what Depth Anything will give on this photo. The Subject matte on page 08 is the mountain outline from the Object mockup.

## The pages

| File | Page |
|---|---|
| `01-add-mask-menu.jpg` | Add mask menu with Depth under Background |
| `02-finding.jpg` | A new Depth row while the map is made |
| `03-depth-card.jpg` | The ready Depth card (Focus plane tooltip shown) |
| `04-pick-depth.jpg` | Pick depth on: crosshair and hint pill |
| `04b-pick-depth-click.jpg` | After a click: Near 24, Far 44 |
| `05-no-focus-point.jpg` | Focus plane off, "This photo has no focus point." |
| `05b-focus-centre.jpg` | Focus at the exact centre: "The camera may not have recorded where it focused. Check the result." |
| `06-show-depth.jpg` | Show depth: greyscale, white near, chip "DEPTH" |
| `07-no-model.jpg` | The depth model is missing: no menu item; a row from a file says why |
| `07b-cant-be-made.jpg` | The map failed |
| `08-depth-and-subject.jpg` | Depth (far plane) with a Subject row subtracted |

## Files

- `html/pages/*.html`: one page per picture, built by `make-pages.py` from the node mockups' DOM (03, 07, 08). `html/app.css`, `html/nv.css` are the Film set's; `html/dp.css` is the little this set adds (the spinner, the tooltip bubble).
- `make-pictures.py`: the photo, the stand-in depth and the masks, and `html/assets/numbers.json` (the numbers the cards print).
- `make-pages.py`, `shoot.mjs`: the pages and the shots. Run in this order: `python3 make-pictures.py && python3 make-pages.py && node shoot.mjs` (the shots use the repository's `node_modules/playwright` and the installed Chromium).
- `mockup-notes.md`: what each page shows, the copy, the findings and the questions.
