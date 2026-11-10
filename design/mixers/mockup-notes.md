# Mixer node mockups: notes (X-02)

Pictures only. Nothing here is product code. The brief is `docs/25-mixers.md` §5; the look is today's app (the C2 dark glass: `design/screenshots-c2/photos-nodes-add-node.png`), not the pre-glass pages of `design/nodes/`. Where `design/nodes/` and the app differ I drew the app: the header has its icons (Before, Crop, Nodes), the toolbar has no Graph · List switch and no "Later" tag, and the cards, ports, links, markers, the Add node menu and the node head use the app's own rules (copied from `src/styles/glass-c2.css` into `html/mixers.css`).

## What I drew, and how

- **Zoom 100 %.** Screen 12 of `design/nodes/` is at 75 % (its correctors are 112.5 × 84 px) while its glyph cards are 56 px wide, so that Layer was really 75 px wide at 100 %. Here everything is at 100 %: a corrector is 150 × 112 px (`graphLayout.ts`), a glyph card is **56 px wide**, and its height is **64 px with two ports, 24 px more for each further port**: Splitter and Combiner 88 px (three ports); a Parallel with two inputs and the "+" 88 px; a Layer with three inputs and the "+" 112 px; one input and the "+" 64 px; eight inputs 208 px. 56 px is enough: the width never limits anything, the ports are 24 px apart on its side. Next to a 150 px corrector it reads as a small junction, which is what a mixer is.
- **Mid-line alignment.** A glyph card is as high as its ports need, so I put its **middle on the line where a corrector's picture ports are** (the card's `pos` y times 150, plus 56), not its top. Otherwise a Parallel between two correctors would not sit on the line, and the Output marker (which sits on the line of the node that feeds it) would be off. See question 3.
- **Names.** Above a glyph card, as on a corrector. "Splitter", "Combiner" and "Parallel" are about 52 to 58 px wide at 12 px bold, so the name's box is 24 px wider than the card on each side (a corrector's name box is the card's width). A name is still cut at ten letters.
- **Ports.** Inputs stacked on the left, 24 px apart and centred on the card's middle, in port order from the top; the picture output at the right at mid height. The "+" is the last input port: an empty triangle (15 × 16 px, larger than the 8 × 10 arrowhead so that a plus fits in it), drawn at 85 % strength, yellow and larger while a link is over it. A mixer with eight inputs has none. `states.jpg` shows one input with the "+", two inputs with the "+" and eight inputs.
- **Glyphs.** Parallel and Layer: the menu's icons (ChevronsRight, Layers). **Splitter:** one line in and three out (a fan); **Combiner:** three in and one out (the same drawn the other way). The menu's item "Splitter and combiner" keeps its own icon (lucide Split), as in the app today. The Splitter's glyph sits toward its one input and the Combiner's toward its one output, so the channel letters have the other side.
- **Channel ports (spec §5.2).** Three options are on `states.jpg`: A plain grey ports (the place says the channel, as in Resolve); B tinted ports (red, green and blue squares and arrowheads); **C tinted ports and a letter R, G or B inside the card next to each** (drawn in `05` and `06`). I propose **C**: the tint tells the channels apart at a glance and the letter does not rely on colour (the tints are `#ff7a6e`, `#62d68f` and `#6fadff`). Pictures are grey everywhere else, so the tint also says "this port is one channel of a picture". The panels' rows repeat the letter in a chip.
- **States.** Selected: the 2 px yellow ring (radius 20 px, as screen 12). Bypassed (a mixer): the glyph at half strength and the diagonal slash across the box, as on a corrector. Dead: the whole card at 45 % (box, name, ports), tooltip "Not connected to the output.". A Splitter and a Combiner have no bypassed state.
- **Panel.** The corrector's compact head (`ym-nodehead`) with the kind's glyph in the chip where the number goes. One tab, in **the Look tab's yellow** (as proposed in §5.5); it also colours the Opacity slider's fill. No histogram. The toolbar's **Show mask** is off on all four (`aria-disabled`; tooltip "This node has no mask."); **Bypass** is on for Parallel and Layer and off for Splitter and Combiner; Reset is off on Parallel, Splitter and Combiner (a Layer's is on).
- **Parallel › Inputs.** Rows are in port order, 1 to N (the order changes nothing). Each row: thumbnail, the input's number chip, the feeding node's "number name". No handle and no words.
- **Layer › Blend.** The Blend card as screen 12 draws it, and the Inputs card from the top of the stack: "On top" on the first row and "Below" on the last, nothing on the middle one.
- **Mode menu.** The same glass list as the app's menus, seven names, a tick at the current mode. The spec gives no line per mode, so there is none (question 9).
- **Link drag (07).** Drawn from Sky's output (Sky is not selected; pressing an output does not select). Lit: every picture input that is not upstream of Sky (the Parallel's two inputs and "+", the Combiner's three, Skin's and Rest's, the Output's). Dim: Main's and Sky's own inputs (a link there would be a cycle) and every key port. The "+" under the pointer is yellow. Corrector inputs light too, because §5.3 says "every picture input that may take it".
- **Thumbnails.** The Red, Green and Blue correctors show the channel of Main's thumbnail as a grey picture (made for this folder, `html/assets/node-*-channel.jpg`): hand-made, like all thumbnails in the node mockups.
- **Not drawn.** A Layer row being dragged (`docs/25` §10 lists it for X-02; this round's brief did not), and a Parallel of three branches (§10 says three; the brief and screens 02 and 07 have two, and the "+" shows where a third goes). A mixer fed by a mixer or by a Splitter output (the Inputs row would show a glyph in place of the thumbnail).

## New strings

Every line of §5.1, §5.5 and §5.8 is used as the spec gives it. These are **not in the spec** (drafts for X-H1):

| String | Where |
|---|---|
| "Inputs", "Blend", "Channels" as the titles of the cards (the tabs' names are the spec's) | the panels |
| "R", "G", "B" in a chip before the rows "Red", "Green", "Blue" | Splitter and Combiner panels |
| "Reset" on the Blend card (as screen 12) | Layer panel |
| the base line ("They start from node 1, Main.") as its own paragraph under the sentence | Parallel panel |

Everything else on `states.jpg` (captions such as "2 inputs and the +") belongs to the sheet, not to the product.

## Questions for the owner at X-H1

**Answered, 2026-11-10:** "Approve as drawn"; item 1 **"Add a corrector too, as Resolve"** (`docs/25` §5.4); item 2 **option C** (tinted ports and R/G/B letters); item 9 **no lines** in the mode menu. The rest stay as drawn.

1. **Item 36 (spec §9): should Add node › Parallel also add a corrector beside the selected node, as Resolve's Add Parallel does?** Drawn with the mixer alone (the lead's brief). Resolve's reading adds a second corrector already linked to input 2, so there is something to grade at once; it is one more node and one more thing to delete. Needs an answer before X-06.
2. **Channel ports: A, B or C?** (above). I propose C.
3. **What does a glyph card's `pos` mean?** A corrector's top left is its `pos`. A glyph card's height varies from 64 to 208 px, so a top-left `pos` puts its ports on a different line from the corrector it follows. I used the **middle line** (`pos.y` is the row of the picture line for every kind). That makes the trio of §5.4 (rows y − 1, y, y + 1) line up. Say if you prefer top-left.
4. **Glyph card width.** 56 px as the spec says. Alternatives: 64 px (the name "Combiner" then fits without the wider name box) or 72 px. I think 56 is right.
5. **Bypassed mixer.** Slash across the box and a half-strength glyph, as drawn; or the slash through the glyph alone. Cosmetic.
6. **"node 3" in the text, "03" on the chip.** The spec writes "To node 3, Red" and "They start from node 1, Main."; the cards show "03" and "01". Drawn as the spec writes it. Keep?
7. **Row order.** The Parallel lists its inputs 1 to N from the top (what the ports show); the Layer lists from the top of the stack (N first), which reads upside down against the ports. Keep both as the spec says?
8. **Tab colour.** Yellow, as proposed. Any other tab colour is one CSS variable.
9. **Mode menu lines.** None drawn. A quiet second line per mode ("Add: adds the layer's light to what is below") would help people who do not know blend modes, and makes the menu taller (seven rows of about 52 px). If you want them I will write the seven lines.
10. **The 64-node tooltip on Splitter and combiner.** The item is off from 60 nodes (it adds five) with the tooltip "A photo can have up to 64 nodes.". A person with 61 nodes reads "up to 64" and is puzzled. A line of its own would be clearer: "This adds five nodes, and a photo can have up to 64." Your call.

## Found while drawing (spec §5 and the code it touches)

1. **"Nine ports: 232 px" (§5.2, risk 6) cannot happen.** A mixer with eight inputs has no "+", so the tallest card has eight ports, **208 px**. The "+" appears up to seven inputs (eight ports). A Splitter and a Combiner are 88 px.
2. **Links and ports in the code do not know the new port names.** `LinkEdge.tsx` takes `data-kind` and the arrowhead (`ARROW[kind]`) from `link.from.port`, which is "picture" or "key" today; `red`, `green`, `blue` and `picture.N` need a "type of this port" function there and in `Port.tsx` (`boxOf` knows picture and key, at fixed y). `graphLayout.ts` (`layoutGraph`, `portPoint`, `cardsOf`) gives every card 150 × 112 px with ports at y = 56 and 97; glyph cards need their own box and port places, and the Output marker (placed on the feeder's `PICTURE_PORT_Y`) needs the mid height of a glyph card. §5.2's "Code" paragraph covers the handles but not these; X-06 should list them.
3. **§5.3 does not say which inputs dim.** I dimmed the inputs that would close a cycle (the dragged output's own node and everything upstream) as well as key ports. The spec says only "never a key port".
4. **A tall card spills out of its grid row.** A 208 px card centred on a row reaches 104 px above and below the line; the grid unit is 150 px, so a corrector one row away in the same column is 10 px clear. Mixers usually sit in a column of their own; the fit-to-view must use the real box of a glyph card.
5. **The "+" is bigger than an arrowhead** (15 × 16 px against 8 × 10): a plus does not fit in a small triangle. It stays 24 px from the last real input, so the app's hit area (about 9 px around the glyph) does not overlap its neighbour.
6. **Show mask and the toolbar's Bypass** are off for the new kinds as well as the panel's buttons; the tooltips need the same texts in both places. Drawn off; no tooltip is open on the screens (a tooltip under the head hides the tab, as the node mockup notes say).
7. **`docs/25` §10 X-02** lists a Layer row being dragged and a Parallel of three branches; neither is drawn (see "Not drawn").
