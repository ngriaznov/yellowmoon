# Depth mockup notes (D-05)

All pages are the Mask tab of a Corrector node ("Haze", node 02) in the node editor's layout. The Radial row on page 01 and 07 is only there to show a list with more than one row.

## What each page shows

- **01 Add mask menu.** Depth is the fifth Find item: glyph (three bars on a floor line, a distance glyph), name "Depth", line "Near or far, from depth" (the spec's copy). The menu is **573 px tall** with ten items; with nine (page 07) it is 527 px, already above the 520 px that `docs/23` §7 gives. Each item adds 46 px. It still fits a 900 px window.
- **02 Finding.** The new row is selected, its eye is on, the subtitle shows a spinner and "Finding…" (spec). The pill over the photo reads "Getting ready…" with a spinner, as the spec gives for a pick that waits for the map. Pick depth is already pressed (spec: on when the row is added). The photo has no tint yet: the row selects nothing until the map is there.
- **03 Ready.** The card in the spec's order: **Pick depth** toggle (eyedropper) and **Focus plane** button on one line, **Near** 0, **Far** 50, **Softness** 50 as plain numbers, **Show depth** toggle, then **Refine edges** 50 %, **Feather** 0, **Grow** 0, **Opacity** 100 %. Pick depth is off here (a person who pressed Esc). The Focus plane tooltip ("Select what the camera focused on") is drawn. The row's subtitle is the band, "0 to 50".
- **04 Pick depth.** The toggle is lit like Pick white; the cursor is the same crosshair; the pill is "Click the part of the photo to select. Esc stops." (spec). The tint is still the old band.
- **04b After a click.** On the field: Near 24, Far 44, the row says "24 to 44", the tint moves to the field. The tool and the pill stay on (spec: one undo step per click, tool stays on).
- **05 No focus point.** The photo's camera line says "Panorama" (a stand-in for a file that has no focus point). Focus plane is dimmed and off, with the line "This photo has no focus point." under it (spec). The reason sits in the card, not in a tooltip, so it can be read without hovering a disabled button.
- **05b Focus at the centre.** The band is made (40 to 60 around the middle of the frame) and the card says "The camera may not have recorded where it focused. Check the result." (spec).
- **06 Show depth.** Toggled by the card's **Show depth** toggle (there is no other control and no key in the spec). The photo is replaced by the grey map, white near, whole and not cut. The chip over the photo says "DEPTH" (spec). The row's eye and the toolbar's Show mask are off (exclusive). The tooltip "Show how near or far each part is. White is near." is drawn. The camera line goes unreadable over a white foreground: this is the existing line, shown as it would be.
- **07 No depth model.** The menu lists nine items and no Depth (hidden, not greyed: spec). A Depth row that a file or a paste brought says "Depth masks need the depth model. Reinstall the app." in its subtitle and has no card (spec). In the 300 px row the sentence takes three lines beside Add and Subtract.
- **07b Can't be made.** The row says "Can't be made" and the card is the spec's sentence with an invented model text ("the model stopped before it finished"). The card has no controls.
- **08 Depth with a Subject row.** The far plane (Depth 55 to 100, Add) with Subject subtracted. The pink tint is the sky and the haze, and the mountain's edge is the Subject matte's. This is the second recipe of `docs/23` §4.8 ("add the Depth row for the far plane and subtract Subject from it"). The first recipe (Subject added, a Depth row with Invert subtracted) would look the same in the list with the modes swapped and Invert pressed on the Depth row. A one-line tip is drawn under the card.

## Copy

Spec copy used as written: "Near or far, from depth"; "Finding…"; "Getting ready…"; "Click the part of the photo to select. Esc stops."; "Select what the camera focused on"; "This photo has no focus point."; "The camera may not have recorded where it focused. Check the result."; "Show how near or far each part is. White is near."; "DEPTH"; "Can't be made"; "The Depth mask can't be made: <the model's text>. Try again, or remove the mask."; "Depth masks need the depth model. Reinstall the app."

**Proposed (the spec gives none):**

| Where | Copy |
|---|---|
| Card title | "Depth" (shown as DEPTH by the style) |
| Toggle names | "Pick depth", "Focus plane", "Show depth" |
| Slider names | "Near", "Far", "Softness" (the spec's names) |
| Pick depth's screen-reader name | "Click the part of the photo to select (D)" (the key D is a proposal; it needs a check against `docs/11` shortcuts) |
| Chip on every other Depth page | "Mask · Depth" (as "Mask · Object") |
| Tip on page 08 only | "Depth is soft at edges. Add a Subject mask and subtract it to keep a clean edge." |

## Findings

1. **The Add mask menu no longer fits 520 px.** 573 px with Depth, as above. It is the spec's own open check (`docs/23` §7). It is not clipped at 900 px height, and the menu scrolls inside Radix's available height on a shorter window.
2. **The long "no model" sentence crowds the row** (page 07): three lines beside the Add/Subtract control. Options: leave it, shorten it ("Needs the depth model. Reinstall the app." in the row, the full sentence nowhere), or hide Add/Subtract on that row. I left the spec's sentence.
3. **The glyph in the menu.** The spec says the Find items get "a sparkle for AI mask". The node mockups' menu has no sparkle on any item (Object and Subject use sparkle-like icons as their kind icons, Background a dashed circle). I gave Depth its own distance glyph and no sparkle. The node's footer already shows the sparkle when it has a Depth row (`docs/23` §7).
4. **The Show depth toggle sits between Softness and the shared sliders** as the spec orders it, so it reads as a part of the band. If it should read as a view control, it could go to the card's end, next to "Show mask" in the toolbar. Not drawn.

## Open questions for the owner

1. **Does a new Depth row say "Finding…" with its eye off?** The spec says the spinner shows "while the eye or Show depth is on", but a new row has Pick depth on and the person's first click waits. I drew the eye on. If the eye starts off, the row would show its band while the map is made and only the pill would say "Getting ready…".
2. **"Try again" has no control.** The failed card says "Try again, or remove the mask." The worker does not start again until the key changes (`docs/23` §4.8), and the card has no button. Either add a **Try again** button (it asks the worker for a new key, no model change), or change the sentence to "Change the photo's exposure to try again, or remove the mask" (not plain), or drop "Try again". I drew the sentence without a button.
3. **Show depth's place and a key.** Is a toggle in the card enough? A small near-to-far key on the photo ("Near" white, "Far" black) would help the first time; it is not drawn.
4. **The tip on page 08.** Do you want a line under the card that points to Subject? It is drawn on page 08 only. It could be shown always, or only when the row has Refine edges at 50 and no Subject or Object row, or not at all.
5. **Pick depth on by default.** The spec has Pick depth on when the row is added, as Object's tool is. Page 02 shows it. If it is off by default, the first thing the person sees is the default band (nearest half) as page 03 shows.
6. **Focus plane's reason in a tooltip too?** Page 05 puts the reason in the card. A disabled button gives no hover tooltip in glasscn here, so it stays in the card only.
7. **The menu's height** (finding 1): accept 573 px, or move the less used items (Brightness range) under a "More" item.

## Stand-ins, again

The depth picture, the tints and the Focus plane / Pick depth bands are drawn from the photo's layout by `make-pictures.py` and are not the model's output. The bands read "24 to 44" and "40 to 60" because the stand-in has those ranks, not because Depth Anything would.

8. **Invert.** The task list named Invert among the card's controls. The spec (`docs/11`) has no second Invert in the card: the row's own Invert icon is it, so none is drawn in the card.
