# Film node mockups: notes

Pictures only. The repository's product code was not touched. Method as `design/nodes/mockup-notes.md`: the app's captured DOM and CSS, edited by a script, opened at 1440 × 900, shot at 2×. The base page was `design/nodes/html/pages/05.html` (Main alone, Light tab) and `06.html` (the Add node menu); the Mask tab reuses `07.html`'s Masks and View cards and the Output tab reuses `11.html`.

## Geometry (1440 × 900, as the node view, layout A)
- Photo 666 × 444, divider 10, editor 371 with a 44 px toolbar, canvas 327; panel 400 wide; head 52, tab row 46.
- Graph: Input at 12, Main at 121.4, Film at 334.8 (the app's 213.4 px pitch), Output at 976; node 150 × 112, thumbnail 81. The Film → Output link is long, as Main → Output is on an untouched photo.
- The toolbar has no Graph · List switch (docs/23 §7.3).
- The tab row has five tabs (Film, Print, Halation, Grain, Mask), 4 in option B; each is flex 1 and fits at 400 px.
- Pickers are popovers in the glass menu surface: the list (372 × 570, scrolls; the last rows fade out) under the Stock field; the tiles (784 × about 640, six columns, no scroll) from the panel's right edge over the photo.
- The colour pad is 168 × 168 with two number fields beside it.

## Choices where the brief was silent
- **Photo.** The viewer shows the hand-made film picture of `photo-graded.jpg` with the Signature chip, as the viewer does with a look on. On the Output screen the chip stays; the note tells the person to choose Neutral.
- **Stock info line.** A quiet line under the Stock field gives the kind and ISO ("Colour negative · ISO 400"); under Paper it names the matching paper.
- **Credit line.** At the foot of the Film tab, outside the cards, as quiet text; "Andrea Volpato (spektrafilm)" is the link. About repeats it with the CC BY-SA 4.0 link and the trademark line.
- **Slide stock.** The Print tab is dimmed but selectable. Its page opens with the sentence, then Paper ("None") and Lights greyed.
- **Option tag.** A pill under the header at the top left of the viewer: "Option A" in the yellow, then what it is about.
- **More.** A quiet disclosure row (chevron and the word), the same on the Lights card of both options.
- **About.** The app's About is not drawn elsewhere in this repository, so the dialog is a plain glass dialog over the dimmed screen; its layout is a proposal. The version line is left out.
- **Mask tab.** One Radial row set to Add (the film look applies inside the shape), the View card with its two segments.

## What did not fit or could not be drawn
- The tile picker covers the Output marker and the toolbar's right half.
- Tiles for the pushed Portra 800 stocks reuse the Portra 800 tile, and all tiles are small variations of one scene, so they differ less than real renders would.
- No loading state for the 20 tile renders is drawn.
- The Halation tab shows no visible glow on the photo at fit size; the effect is subtle at this scale by design.
- Tooltips are not drawn.

## For the spec
See `docs/24` §4 (revised) and the small list of corrections in the report that came with this round.
