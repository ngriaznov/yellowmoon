# Node view mockups, round 2: notes

Pictures only. Nothing here is product code; the repository was not touched. Round 1 (layout B, the old numbering) is in `old/`. DCTL was dropped at the owner's request and is not in these screens.

Files: `index.html` (walk-through with a caption under each screen), `sheet.jpg` (contact sheet), the 13 PNGs at 2880 × 1800 and one crop, `07b-mask-rows-invert.png`. Build files are in `work/`.

## Decided (2026-11-01)
- **Layout A.** Viewer on top, node editor below it (45 % of the height under the header), inspector 400 px and full height. Layout B is dropped.
- **Tab split**, "for now it is ok". A corrector has **Light · Colour · Curve · Mask**. The Input has **RAW · Lens · Light · Colour · Detail**. The Output has **Look · Detail** (Look is today's Look card; Detail holds only the Output sharpening row; its head reads "Output · cine rendering, look and export").
- **The look is set once per photo, on the Output**, not in the correctors. The node cards carry no look glyph. The look chip on the photo is unchanged. The simple view's Look tab writes to the Output.
- **Pick white leaves the header** and lives in the Colour tab's White balance card. W opens the Colour tab and starts picking. Like Pick colour, picking ends when the Colour tab is left. In the node view W selects the Input and opens its Colour tab. A corrector's Colour tab (Temperature and tint trim) has no Pick white in this version.
- **Compact node head** (screens 05 to 13): number chip, name at 20 px bold, a quiet line, and Bypass, Reset and More as icon buttons. No 40 px tab title in the node view. The simple view keeps its title blocks.
- **Key ports stay on every node** ("I'm fine with clutter for now").
- **Masks are summed, as in Resolve.** The masks of a node add up. A mask set to Subtract cuts itself out. A colour or brightness range keeps only its part of what the shapes and AI masks gave (as a Resolve qualifier is limited to its windows). A node with only ranges uses the range alone. Invert exists twice: on the Masks card header (the node's whole combined mask) and on each row (that mask alone).
- **If an object search finds nothing, nothing changes.** No point is added and no empty mask is kept. The existing error bar, the one Pick white uses when it refuses a spot, says: "Nothing found there. Click another part of the object, or draw a box."
- **Graph · List** switch in the toolbar, after Delete (an option, see the Resolve 21 section).
- Shared nodes across photos are later. Presets are not planned until the features settle.

## Choices where the brief was silent
- **Mask accent:** orchid pink `#FF6FD2`. Black on it 8.5:1, on the ground 7.7:1, on a glass pane 5.3:1. It also colours the mask tint, the key ports and key links.
- **Pick white placement:** a row of its own in the White balance card, between the light presets and the Kelvin slider: a small toggle with an eyedropper and "Pick white", and "or press W" with a key chip at the right. The card header already holds AUTO and Reset, so nothing is crowded. Pressed, it is yellow, like the header toggles.
- **While picking:** the hint is the "Object found" pill (top centre of the photo) reading "Click something neutral: grey or white. Esc cancels." The cursor is a crosshair drawn white with a dark outline, over a pale cloud of about R 221, G 224, B 229.
- **Simple view notice (04):** "This photo has 4 nodes. These controls change the input, the output and node 1, Main." The link, Open nodes, sits at the end of the text; it takes two lines.
- **Do the three Mask cards fit without scrolling at 1440 × 900? Yes, with two mask rows:** the head is 52 px, the tab row 46, and the cards are 244 + 316 + 114 = 674 px in a 678 px scroll area (4 px to spare). A third mask row (about 89 px) would make the panel scroll a little. The Colour tab still scrolls (Colour balance and Colour warp are tall, as today) and so does the Output's Look tab (768 px of cards).
- **Screen 07:** Object (2 points, Add, selected) and Radial (Feather 45, Subtract). The overlay is the mountain tinted with a soft hole at its right half, the radial's outline, its inner (feather) ring and a centre handle. The Object card shows because that row is selected. Screen 07b shows the Radial row with its Invert pressed.
- **Screen 10, Input · Colour:** only the White balance card belongs to the Input, so the tab is short. Colour balance and Colour warp stay with the correctors. The Input's head keeps only the first sentence of the owner's blurb, "Turns the camera's data into the scene."; the second ("Every node works on what it makes.") did not fit on one line.
- **Screen 11, Output · Look:** today's Look card. Sections in the app's own order: Built in (Neutral, Signature chosen), an imported library with a display rendering ("ARRI looks · LogC3 to Rec.709", 87, folded; the app's own section name, filled from the person's own import), then Your LUTs. I used Your LUTs with invented names so that no real ARRI look name appears. Looks after the first nine carry no key number, as in the app. Apply to reads "This photo, Picked · 3, All · 7".
- **Tooltips of the head's three icon buttons** ("Bypass this node (B)", "Reset this node", "More") are in the markup and not drawn: a tooltip under the head covers the tab labels.
- **Node head line:** "Corrector · whole photo", "Corrector · 2 masks" (the row text says which), "Corrector · Linear".
- **Layout A geometry (1440 × 900)** is as in round 1: photo 666 × 444, divider 10, editor 371 (canvas 327 under a 44 px toolbar). Expand: photo 405 × 270, editor 545. At 1728 × 1117 the same rules give about a 844 × 563 photo and a 1328 × 469 editor (computed, not drawn).
- **List view (13):** rows for Input, 01 to 04 and Output, each with a grip (reorder; absent on Input and Output), a number chip or a lock, a 62 × 32 thumbnail, the name, the kind ("Corrector · Object, Radial (subtract)"), mask glyphs, a Bypass toggle and More. The selected row has the yellow ring. The inspector is as in screen 07.
- **Photos:** 7 of the owner's own photos, in capture order; folder "Provence"; "2 of 7".

## Resolve 21 reference
Sources are reviews and Blackmagic's announcement as quoted by them; their sites were not read.
- Resolve 21 (released 3 June 2026; 21.1 due September 2026) has a new **Photo page** that brings the Color page's node grading to stills, with native raw for Sony and others. Its Inspector has four tabs, **Raw · Photo · Effects · File**. A filmstrip of the album runs along the bottom, with a Lightbox grid mode. Photo page adjustments **appear as the first node** of the Color page's graph (reportedly not editable there). "Shared nodes" apply one look across an album.
- The Color page node editor gets a **layer-list view**: nodes listed as rows by number, easy to label, lock, bypass and remove in large trees.
- **Magic Mask** (click a subject) predates 21. 21 adds "Render in Place", which caches a tracked mask as a matte node. One third-party guide says 21 merged the Person and Object modes; unconfirmed.
- **For us:** our simple view writing to the Input, node 1 and the Output matches "Photo page edits become the first node". Unlike Resolve's, ours stay editable in the node view. A list view of the graph is a good idea (screen 13). Shared nodes across photos are later.

## What did not fit
- The Colour tab (Colour balance and Colour warp) and the Output's Look tab scroll; only the Mask tab was made to fit.
- The Add mask menu is 520 px tall and covers the right half of the graph at 1440 × 900.
- At 100 % four nodes fit in the editor with 63 px between them; any branch needs zoom or Expand (screen 12 needs 75 % and Expand).
- Screen 12 has one link crossing: 02's key link crosses Main's link to 04.
- Tooltips over the head would hide the tab labels (see above).
- The list view (13) does not show key links or branches. In a graph like screen 12 it needs an indent or a "fed from" column.
- With the Input's Colour tab as only one card, the panel is mostly empty.

## Open questions for the owner
1. **A per-node "pick neutral" for later.** A corrector's Colour tab has no Pick white. Should a corrector get one that sets its Temperature and tint trim from a neutral spot?
2. **How soft edges add up.** The plan has to fix the maths. Candidates: a clamped sum, a maximum, or screen, 1 − (1 − a)(1 − b). Subtract would then multiply by 1 − b. Ranges would multiply what is left.
3. **Copying masks.** May a mask be copied to another photo with Copy settings? Linear, Radial and Brush can; Object, People and Background cannot.
4. **Three or more mask rows.** The Mask tab scrolls from the third row. Is that fine, or should the View card go into the head's menu?
5. **The simple view when node 1 is deleted.** The simple view writes to the input, the output and node 1. If node 1 is deleted or reordered, which node does it write to?
6. **Output's Detail tab** holds one row (Output sharpening). Keep two tabs, or show it in the Look tab?
7. **List view.** Keep Graph and List both? How should a list show branches and key links?
8. **Names and numbers.** Do numbers change when a node is deleted or inserted? Mixers have none. Long names are cut at about ten letters.
9. **Header in the node view.** Before compares with the camera default; Crop acts on the photo. Do Export, Copy settings and Paste work on every node?
10. **Show mask** is a toolbar toggle and the View card's control. Keep both?
11. **Photo switcher.** With the filmstrip hidden, picks and Remove from filmstrip are out of sight.
