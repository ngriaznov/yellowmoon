# Colour Compressor mockups: notes and questions (CC-01, 2026-11-18)

What the drawings decided where `docs/32` left room, and the questions for the owner. The eleven questions with their defaults are `docs/32` §10; this file adds what the pictures show.

## What the drawings decided

1. **Three cards, in the order the person works:** Target (pick, then check), Compress (the amounts, the most used), Range (the least used, last). At 1440 × 900 the first two fit; the Range card scrolls a little. The Resolve manual lists Target and then Compress; the Range is ours.
2. **The swatch has a name.** "Orange, a little pale" uses the warp's eight hue names; it is also the swatch's accessible name, with the three numbers. A target is easier to trust when it is named.
3. **The three target fields are the same sliders as everywhere** (glasscn's `Slider` with its value box), with the Hue track drawn as a hue ramp. They carry no Reset of their own (a target has no default); the card's Reset clears the target.
4. **Show affected area is the toolbar's Show mask**, with an eye in the Range card's header (the eye of Face Refinement's cards, in the mask accent). One view state, two switches.
5. **The disabled states say why:** no target, so "Pick a colour to compress toward." under the sliders, "–" in the fields, and the eye's tooltip "Pick a colour first." (`aria-disabled`, as elsewhere in the app).
6. **Range values read as half-widths** ("±30°", "±5", "±1.5 EV"), so the number says how far from the target a colour is still pulled.
7. **Head line.** The name is "Compressor" (so it fits a card and the head), the line says "Colour Compressor"; when bypassed or not connected the line is just "Bypassed" or "Not connected", because the name already takes most of the head.
8. **A pick too grey for a hue** moves the target to Saturation 2 and says so in a note in the Target card (the warp's `sk-note` style), once.
9. **The glyph** of the node is two arrows pointing at a dashed line (the menu's icon and the card's footer glyph).
10. **The tab is "Compress"**, in the Colour tab's green, so the node does not look like a Film node or Face Refinement (yellow).

## Questions for the owner

The eleven of `docs/32` §10, in brief, each with the default the pictures use:

| # | Question | Default |
|---|---|---|
| 1 | The name: "Colour Compressor" (Resolve's), or "Colour Uniformity" or "Colour Gather" | Colour Compressor |
| 2 | A separate shift of the target | No: the target's fields can be typed |
| 3 | Amounts after the first pick | Hue 50 %, Saturation 50 %, Lightness 0 % |
| 4 | A built-in "Skin" starting point | No; the defaults suit skin, and `docs/10` gets an example |
| 5 | Its own Range, or masks only as Resolve | Its own Range, with masks on top |
| 6 | Show mask on the node shows the mask times the range | Yes |
| 7 | Pick colour on when the node is added | Yes |
| 8 | The pick is a 7 × 7 mean | Yes |
| 9 | A Strength or Blend slider | No |
| 10 | The tab colour and the menu's place | Green; after Face Refinement |
| 11 | Pipeline 0.13.0 | Yes |
