# Colour Uniform mockups: notes (CU-01, revised 2026-11-18)

What the drawings decided where `docs/32` left room. The owner's answers of 2026-11-18 (CU-H1) and the one decision of ours that waits for a yes are in `docs/32` §10.

## What the drawings decided

1. **Two cards, in the order the person works:** Target (pick, then check or adjust), Compress (the amounts). With no Range card both fit the panel at 1440 × 900 with room to spare.
2. **The swatch has a name.** "Orange, a little pale" uses the warp's eight hue names; it is also the swatch's accessible name, with the three numbers. The swatch is not a button: glasscn has no colour picker component (`src/components/glass/` has none), so the three sliders are the picker. The Hue track is drawn as a rainbow.
3. **The target's three fields are on with no target.** Typing or dragging them is the second way to choose the colour, so they rest at Hue 0°, Saturation 2, Lightness 0 and show "–" until a colour is set. The Compress sliders are off until there is a target, and say why.
4. **Show mask is the ordinary one.** No eye in a card, no second switch. The picture shows the node's mask: the seeded Colour range row.
5. **The first pick adds a Colour range row** (the decision of ours, `docs/32` §10): the Mask tab shows it with the Colour range's own defaults, Width 39 and Softness 50, and its centre on the picked colour. Delete it and the node acts on the whole photo.
6. **Head.** The name on the card and in the head is "Uniform" (seven letters, not cut); the line says "Colour Uniform", "Bypassed" or "Not connected". The corrector's "· 2 masks" text is not repeated: the line would be cut at 400 px.
7. **A pick too grey for a hue** moves the target to Saturation 2 and says so in a note in the Target card, once.
8. **The glyph** of the node is two arrows pointing at a dashed line (the menu's icon and the card's footer glyph, beside the mask glyph).
9. **The tab is "Compress"**, in the Colour tab's green, so the node does not look like a Film node or Face Refinement (yellow).
10. **The diagram shows the two guards.** The hue pull is full up to 90° from the target's hue and gone by 150° (so the colour opposite is not dragged across the wheel); neutrals are kept. The Colour range mask is drawn as the pink disc.

## The owner's answers (2026-11-18) the pictures follow

Name "Colour Uniform"; first-pick amounts 50 / 50 / 0; masks alone (no Range); no Skin preset; no separate target shift; no Blend slider; pipeline the next minor step; Pick colour on when the node is added, 7 × 7 mean, after Face Refinement in the menu, the Colour tab's green, ordinary Show mask; and the target can be picked and set by its Hue, Saturation and Lightness.
