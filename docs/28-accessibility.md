# 28 · Accessibility (M4-04)

This document is the audit and the plan for task **M4-04 · Accessibility** of `docs/15-milestones.md`: *VoiceOver labels, focus order, keyboard-only use, contrast check of tab colours. Done when: the checklist passes.* The checklist did not exist; section 6 is it.

Date of the audit: 2026-11-13. Branch `m4-04-audit`, made from `main` at b53da3db. The audit changed no product code and added no file to the repository except this one. It reads the code and the specs, and it measures what a Linux machine can measure. Sections: 1 scope, 2 the audit (numbers), 3 gaps, 4 findings with their fixes, 5 tasks, 6 the checklist, 7 checks on a Mac, 8 questions for the owner.

## 1. Scope

**In scope.** The webview of the app (every screen and state of `e2e/helpers/states.ts`), as a VoiceOver, keyboard, Voice Control and low-vision user meets it; the three macOS display settings that matter to glass (Reduce Motion, Reduce Transparency, Increase Contrast); the contrast of text, the focus ring and the tab colours; the native Metal viewport as far as it touches the above.

**The target** is WCAG 2.2 level AA, read for a native-feel macOS app: 1.1.1, 1.3.1, 1.4.3, 1.4.11, 2.1.1, 2.1.2, 2.1.4, 2.4.3, 2.4.7, 2.5.8, 4.1.2, 4.1.3. Where Apple's own behaviour is the better guide (focus returns, Full Keyboard Access, system display settings) it wins, and the section says so.

**Out of scope.** The CLI (`ym` prints plain text; a screen reader reads a terminal). Light appearance (the app is dark only, `docs/11` *Dark only*). Text size: the window is fixed-density and macOS gives a web view no text-size setting, so 1.4.4 and 1.4.10 are not tested. Touch targets for fingers. Forced-colours mode: WebKit has none; the rules in `glass.css:1058` stay as they are. The glass look stays: nothing below proposes to drop it (the owner's rule for this audit). The Reduce Transparency answer is a fallback (A-06), not a different design.

## 2. The audit

### How the numbers were made

- **Machine and browser.** Linux x86_64 VM, 4 vCPU, Chromium 1243 headless (`/opt/pw-browsers/chromium`), Playwright 1.56.1 from the repository, 1440 × 900 at scale 1, against the Vite dev server and the mock API. This is not WKWebView; every Chromium result that depends on WebKit's behaviour is repeated as a Mac check in section 7.
- **States.** The 79 states of `STATES` in `e2e/helpers/states.ts` (numbers 01 to 84; the list in the file says "42 states" in a comment, it holds 79 now), opened with `openState`, plus Start to Photos with `mockPhotos=500` and `mockPhotos=2000`.
- **axe.** `axe-core` 4.10.2 installed in a scratch directory (not in `package.json`), injected into the page, tags `wcag2a`, `wcag2aa`, `wcag21a`, `wcag21aa`, `wcag22aa` and `best-practice`, violations and "incomplete" results kept.
- **Focus order.** `Tab` pressed from the start of the document, 26 states and the 500-photo Photos screen; for each stop the element, its role and name, the computed outline, and whether the ring is drawn. Plus an inventory of every visible interactive element with its `tabindex`.
- **Contrast.** axe cannot decide contrast on translucent glass (it returned "incomplete" for 79 of 79 states, 357 nodes), so contrast is measured on pixels: a screenshot, the text box of every visible text element, the background as the commonest colour inside the box (the glyph pixels are a minority), the foreground as the computed colour at its own alpha and opacity over that background. 1,060 text elements in 25 states on the app's own aurora; the same on pure black; and the elements that lie over the photo cell with the cell painted white, mid grey and black. The method ignores `text-shadow`, and the pixels include the glass's sheen; the first makes the camera line's number a little pessimistic, the second makes the numbers true to what is drawn.
- **Media features.** Chromium's `Emulation.setEmulatedMedia` for `prefers-reduced-motion`, `prefers-reduced-transparency`, `prefers-contrast` and `forced-colors`, reading the computed style of the panes, buttons, dialog and chips.
- **Not measured, and said so:** VoiceOver itself, WKWebView, the native viewport, Full Keyboard Access, the real system settings, Voice Control, a real photograph behind a menu. The scripts are scratch files, not committed (Appendix).

### 2.1 What is already good

The app starts well. Measured:

- axe finds **no violation in 36 of the 79 states**, among them every state of the list the task names: Start empty, the Photos screen with one photo open, **each of the seven tabs**, Crop, Pick white, the pixel readout, Before, **Preferences** (three states), About, the close-session dialog, the **look-import wizard** (all five steps), the **error bar**, the Export panel while it runs with the strip, Colour warp, and the Photos screen with **500 and with 2000 photos** (the filmstrip is windowed to 16 thumbnails).
- Every control that `e2e/a11y.spec.ts` can find has a role and a name; the spec walks the accessibility tree of Start, each tab, Crop, Pick white, the Export panel in four formats, the wizard, the dialogs, the node view, the node panels and the mixers (`e2e/a11y.spec.ts:28-42` lists the roles that need a name). Sliders, wheels, curve points, colour-warp pins, crop handles, mask handles and node cards are named with values (`Wheel.tsx:158-165`, `CurvePlot.tsx:321-327`, `WarpPlane.tsx:361-367`, `CropOverlay.tsx:148`, `MaskHandles.tsx:186`, `NodeCard.tsx:174-177`). The histogram is a named image (`Histogram.tsx:50`). The graph's ports, links and the "+" ports are `aria-hidden` and the keyboard path is the More menu (`docs/23` §7.4).
- Every one of the 26 states walked has a visible ring on every Tab stop but four (below): a 3 px outline with a 2 px offset, in the tab's accent (`rgb(255 194 14)` on Look, `rgb(123 227 95)` on Colour) or `--focus` (`#0db7e8`) on look tiles; the tabs draw it inset (`-3px`).
- No keyboard trap in any state: Tab leaves every screen; the three dialogs keep Tab inside; Esc closes them.
- Single-letter shortcuts do **not** fire in a text box. Tested with 18 keys (P, Z, C, W, N, F, B, M, A, `\`, 1, Delete, Backspace, ←, →, Esc, Return, Space) in the Look search box and in a slider's value box: 0 shortcut fired, the text went into the box (`shortcuts.ts:152-250`, `isTextTarget`).
- Reduce Motion is honoured: transitions go to `none` and animations to 1 µs (`global.css:184`, `glass.css:988`), the glass's own press effects to 0 ms; the dialog's view transition checks the media query (`dialog.tsx:15-16`). Nothing in the page animates in JavaScript (`requestAnimationFrame` is used for measuring, not for motion). The native viewport draws frames on request and runs no animation.
- Colour is never the only carrier: each tab has a name, each accent only paints what belongs to that tab.
- A filmstrip context menu opens from the keyboard (the Menu key and ⇧F10 both, measured) and Esc returns to the thumbnail.

### 2.2 axe results, per rule and per screen

79 states, one run each (the 500-photo state failed in the first harness for a waiting reason and was run again by hand: 0 violations at 500 and at 2000 photos).

| Rule | Impact | Nodes | States | What it is |
|---|---|---|---|---|
| `region` | moderate | 36 | 36 | 28 node-view states: the divider `role="separator"` stands outside any landmark (`Divider.tsx:59-73`, mounted by `Photos.tsx:255` beside, not inside, the graph's `section`); 1 state: Radix's select viewport (Preferences with providers); 7 states with a menu open: the portal's wrapper |
| `aria-hidden-focus` | serious | 28 | 7 | An open Radix menu (modal) puts `aria-hidden` on the header, the graph and the divider, which hold focusable content: the header's buttons, the divider (`tabindex=0`), the graph's selected card. Radix keeps Tab and the pointer out of them while the menu is open, so a user cannot reach them. A known Radix pattern |
| `scrollable-region-focusable` | serious | 10 | 10 | 7 real: `.ym-develop-list` on Start (`ExportQueueList.tsx:148`, states 02 and 03) and `.ym-filmstrip__scroll` in the Export panel (states 25 to 29), neither with a focusable child; 3 inside an open menu (`role="menu"` with overflow) |
| `landmark-one-main`, `page-has-heading-one` | moderate | 7 + 7 | 7 | The same seven open-menu states: `aria-hidden` on `main` makes the document look empty. An artefact of Radix's `hideOthers` |
| `color-contrast` | n/a | 357 | 79 "incomplete" | axe cannot decide through `backdrop-filter`; see 2.4 |
| `aria-prohibited-attr` | serious | 4 | 4 "incomplete" | `<code aria-label="Command line">` in the Export panel (`ExportOptionsForm.tsx:340`): `aria-label` is not allowed on `code` |
| `aria-valid-attr-value` | critical | 1 | 1 "incomplete" | The Stock field's `aria-controls` names a popover that is not in the page yet (`.ym-stock-field`, Film, state 68); Radix creates it a frame later |
| `bypass` | serious | 7 | 7 "incomplete" | The same seven open-menu states |

By the screens the task names:

| Screen | States | Violations |
|---|---|---|
| Start, empty | 01, 04 | 0 |
| Start, developing / ready | 02, 03 | 1 each (the unfocusable scroller) |
| One photo open, each tab (Look, Light, Colour, Detail, Curve, RAW, Lens), Lens chosen and missing | 10 to 15, 37 to 39 | 0 |
| Crop, Crop with Auto, Pick white, Pick colour, Before, pixel readout | 16, 17, 18, 19, 32, 36, 50 | 0 |
| Filmstrip with many photos | 500 and 2000 photos | 0 |
| The node view (graph, panels, masks, Film, mixers), no menu open | 42, 51 to 58, 60 to 65, 67 to 75, 78 to 79, 81 to 84 | `region` 1 each (the divider) |
| A menu open in the node view | 43, 52, 59, 66, 70, 77, 80 | the Radix pattern above (4 to 6 rules each) |
| Export dialog (panel), four formats and running | 25 to 29 | `scrollable-region-focusable` 1 each; `aria-prohibited-attr` incomplete |
| Preferences, About, consent, close session, import .cube | 20 to 23, 34, 35, 41 | 0 |
| Preferences with providers (Select open) | 40 | `region` 1 |
| Look-import wizard, five steps | 05 to 09 | 0 |
| Error bar | 24 | 0 |

**Reading.** Of the 29 non-menu `region` hits, 28 are one element (the divider) and 1 is a Radix portal. Of the real findings, there are four: the divider's landmark, the two scrollers, the `code` label. The seven menu states are one pattern and are allowlisted, with a reason, in the test (A-01).

### 2.3 Names for VoiceOver

Roles and states: Radix gives `role`, `aria-expanded`, `aria-checked`, `aria-selected`, `aria-pressed` and the labels; our wrappers keep them. Checked where a wrapper could lose them: `Button` (the tooltip tree keeps the `<button>` and its description, `docs/11` *Tooltips*; tested by `toHaveAccessibleDescription`), `Slider` (`aria-labelledby` and the value box named "{label} value"), `Switch`, `Segmented` (a `radiogroup`), `Tabs` (`TabBar.tsx:24-31`), `Dialog` (names itself from its title), `Menu`. None loses a name. What is lost is something else:

| # | Element | What a VoiceOver user gets | Evidence |
|---|---|---|---|
| N1 | The photo cell's chips: "Neutral · 100%", "Before · camera default", the camera line, the ⌥ pixel readout | The picture is `role="img"` and its children are presentational (ARIA: an `img` has no accessible children), so WebKit does not expose the four texts. The image's label names the photo and the look ("Photo DSC04204, Neutral · 100%") but not "Before", the camera line or the readout. Chromium's tree prints them as text inside the image; VoiceOver in WKWebView follows the ARIA rule (Mac check A-MC-1) | `PhotoCell.tsx:224-247`; `tree-results` of the audit: `img "Photo DSC04204, Neutral · 100%": Neutral · 100% Sony A1 · …` |
| N2 | A photo in the Export panel's strip | `role="img"` with a name; correct for a picture, but the strip is a scroller with no focusable child, so a keyboard user cannot scroll it (see F-4) | `Filmstrip.tsx:199-211` |
| N3 | The 2000-photo filmstrip | A photo that is not in the page cannot be reached by VoiceOver until it is scrolled near; the items carry no `aria-setsize` or `aria-posinset`, so "photo 812 of 2000" is not said. Measured at 500 and 2000: 16 thumbnails in the page, 0 `aria-setsize`. Already noted in `docs/11:771` as M4-04 work | `Filmstrip.tsx`, `windowed.tsx` |
| N4 | The Export panel's command line | `aria-label` on `code`, which is not allowed; a screen reader may ignore it | `ExportOptionsForm.tsx:340` |
| N5 | The node view's divider | A named, valued separator (good) outside every landmark, so "Node editor" and "Tools" do not contain it; it is read after the header with nothing to say where it belongs | `Divider.tsx:59-73` |
| N6 | What a shortcut did | A key press that changes the look (1 to 9), Before, zoom, Crop, Nodes, a pick, or a photo step changes the picture and, in some cases, a label; nothing is announced. Only full screen says anything (`Photos.tsx:1007-1010`, the one `aria-live` region of the screen). The pressed state of a focused toggle is read; a key pressed while focus is elsewhere says nothing | `shortcuts.ts`, `Photos.tsx:1007` |

The live regions that exist (`role="status"`, `role="alert"`) cover the error bar, the export, the wizard, the picking hints, the open progress and the update pill; they were checked by `e2e/a11y.spec.ts` and are not a gap.

**The native viewport.** `native.rs` adds a plain `NSView` behind the webview (`native.rs:183-186`); it calls no accessibility API (`grep` for `Accessibility`, `isAccessibilityElement` in `src-tauri/src/` finds nothing). A plain `NSView` is not an accessibility element, so VoiceOver should see only the webview, whose `role="img"` names the photo. The webview's cell is transparent and holds the label; this is the design, and it needs a Mac to confirm (A-MC-2). Also unmeasured: where the first responder is after a click on the photo (the shortcuts are a `window` `keydown` listener in the webview).

### 2.4 Focus order and keyboard-only use

Tab order, from the start of the document, Photos screen, Look tab (43 stops, no stop without a ring, none off screen): Before, Crop, Nodes, Add photos, Fit, Update, Export; one stop for the filmstrip (the current photo; ← and → move, roving); the tab bar (one stop on the selected tab); then the panel's controls from top to bottom; Copy settings and Paste at the foot. The tab bar and the filmstrip are each one stop by design. The filmstrip comes before the panel; the panel stands to the right of the photo and the strip under it (`design/screenshots-c2/photos-look.png`), so the order is header, photo, strip, panel: reading order, kept (no question).

Stops per state (real stops, rings, the Tab order wrapping out of the document at the end):

| State | Stops | Without a visible ring |
|---|---|---|
| Start empty / ready | 8 / 3 | 0 / 1 |
| Photos Look / Light / Colour / Detail / Curve / RAW / Lens | 43 / 21 / 20 / 17 / 10 / 11 / 10 | 0 |
| Crop / Pick white | 23 / 23 | 0 / 1 |
| Colour warp | 7 | 0 |
| Preferences / About / import .cube / close session | 9 / 9 / 4 / 3 | 0 |
| Error bar | 10 | 0 |
| Export panel / running | 15 / 5 | 1 / 1 |
| Node view (default / Light / masks / Film / Parallel) | 40 / 27 / 21 / 16 / 11 | 0 |
| Wizard step 1 / step 5 | 10 / 10 | 0 |
| 500 photos | 41 | 0 |

**Findings.**

| # | Finding | Evidence |
|---|---|---|
| F1 | **An invisible Tab stop in Pick white.** While Pick white is on, the tab list itself is a Tab stop (`div role="tablist"`, `outline-style: none`, `:focus-visible` true): Radix's roving group is focusable when it has no tabbable item, and the tab list is drawn with no ring. A keyboard user presses Tab and nothing seems to happen. Measured in state 17 | probe: stop 1 of 23 |
| F2 | **Two scrollers are Tab stops with Chromium's thin default outline (1 px, `#eee`)** (Start's list; the Export panel's strip). In Chromium a scroller with no focusable child takes keyboard focus; WebKit does not, so on a Mac the user cannot scroll them from the keyboard at all. Either way it is the axe rule above | states 03, 25, 29 |
| F3 | **Dialog focus returns, except from About.** Preferences opened with ⌘, from the Fit button returns focus to Fit (good, `Dialog.tsx:28-53` remembers the opener). About is opened from inside Preferences and replaces it (`Overlays.tsx:21`): the remembered opener is the About button, which is gone, so Esc leaves the focus on `body`. The consent dialog returns to the Auto grading select (good), Preferences after it to Fit (good) | `dialogs-results` |
| F4 | **The Export panel does not take or give focus on ⌘E.** With focus on Fit, ⌘E opens the panel and the focus is on `body` (the header's controls are replaced); Esc closes it and the focus is on `body` again. The header's Export button path is fine (focus stays on it). `ExportPanel.tsx:69-78` moves focus to `[data-autofocus]` when the view changes inside the panel, but not when the panel opens or closes | `dialogs-results` |
| F5 | **No keyboard path to pan** a zoomed photo. ← and → step photos, ⌘+ and ⌘− zoom, Z toggles Fit and 100 %, but nothing moves the picture except the pointer (drag, wheel) (`docs/11:797-829`) | shortcut table |
| F6 | **The pixel readout is pointer-only** (⌥ held over the photo): the values of one pixel. Pick white's crosshair (`PickLayer`, arrows and Return) shows a keyboard path exists for the same job | `PhotoCell.tsx:195-210` |
| F7 | **Typing is safe, but 21 single-key shortcuts are on whenever focus is not in a text box** (P, Z, C, W, N, F, B, M, A, `\`, 1 to 9, Delete). WCAG 2.1.4 (level A) wants such a shortcut to be switched off, remapped, or active only when its control has focus. A Voice Control or dictation user who says "n" or "c" can fire one by accident. The menu bar has an item for each (`docs/11` *Menus*), so a switch costs nothing | `shortcuts.ts:226-250` |

**What cannot be done without a pointer** (and whether a keyboard path exists):

| Task | Keyboard path today | Needed? |
|---|---|---|
| Crop frame and its eight handles | Yes: focusable, named, arrows move them (`CropOverlay.tsx:125-150`); the frame is `role="group"` with `tabindex=0`, where `application` or a slider role would say "arrows work here" | Role only (A-P8) |
| Mask handles (Linear, Radial) | Yes: nudged by the arrows (`docs/23` §7.6) | No |
| Brush mask (paint strokes) | **No.** `BrushLayer` is `aria-hidden` and not a Tab stop (`docs/23` §7.6, N-14) | Decision (Q2) |
| Object mask (SAM points) | **No** ("no keyboard path for placing points", `docs/23:1458`) | Yes (A-09) |
| Subject, Background, Depth, Colour range, Brightness range masks | Yes: Subject and Background are one add; Depth and Colour range have sliders; a range is picked on the crosshair layer, which has arrows and Return | No |
| White-balance picking and colour picking | Yes: `PickLayer`, `role="application"`, arrows move the crosshair, Return picks (`PickLayer.tsx:76-113`) | No |
| The colour wheels | Yes: `role="slider"`, arrows, Delete centres (`Wheel.tsx:117-165`) | No |
| Curve points | Yes: sliders with arrows and Delete; Return on the plot adds a point (`CurvePlot.tsx:215-290`) | No |
| Colour warp pins | Yes: sliders, Delete (`WarpPlane.tsx:285-300`) | No |
| The node graph: select, add, delete, bypass, rename, reorder | Yes: one Tab stop, arrows move, Return selects, B bypasses, Delete deletes, ⌥← ⌥→ reorder, Add node menu | No |
| The graph's links: make, move, remove | Yes: More › Link picture to, Link key to, Remove link (`docs/23` §7.4) | No |
| Resize the divider | Yes: ↑ and ↓, ⇧ for 50 (`Divider.tsx:47`) | No |
| Before / after | A toggle (`\` and the Before button); there is no split | No |
| Pan a zoomed photo | **No** | Yes (A-10) |
| Pixel readout | **No** | Yes, as part of A-10 |
| Remove a photo, pick, pick all | Yes (Delete, P, ⌘A, the filmstrip's menu by ⇧F10) | No |
| Scroll the Start list and the Export strip | **No in WebKit** | Yes (A-02) |

### 2.5 Contrast

WCAG 2.2: 4.5:1 for text (3:1 at 24 px, or at 18.66 px bold), 3:1 for the parts of a control and for focus indicators (1.4.11).

**The tab colours** (computed from the tokens of `src/styles/glass-c2.css:46-52` and `tokens.css`; the pane is `#303030`, the lighter and so the worse of the two glass estimates in `contrast.node.test.ts`, the ground `#0f0f10`):

| Tab | Colour | Black text on it (4.5) | On the pane (3) | On the ground (3) | On a white photo |
|---|---|---|---|---|---|
| Look | `#ffc20e` | 12.98 | 8.15 | 11.84 | 1.62 |
| Light | `#2ec4f4` | 10.32 | 6.48 | 9.41 | 2.04 |
| Colour | `#7be35f` | 13.00 | 8.17 | 11.86 | 1.62 |
| Detail | `#f4b9ac` | 12.38 | 7.78 | 11.30 | 1.70 |
| Curve | `#ff7340` | 7.78 | 4.89 | 7.10 | 2.70 |
| RAW | `#5b9bff` | 7.58 | 4.76 | 6.91 | 2.77 |
| Lens | `#bd8cff` | 8.36 | 5.25 | 7.63 | 2.51 |
| Mask | `#ff6fd2` | 8.46 | 5.31 | 7.71 | 2.48 |

All eight pass the two limits that apply to them (the task's "contrast check of tab colours"): the ink on the colour, and the colour on the pane and the ground. The last column is why the accent is never a lone signal over the photo: the crop frame and the mask handles use the brand yellow and the mask pink over the picture, where on white they fall to 1.6 and 2.5. The handles carry a dark fill and a 2 px border (`glass-c2-pages.css:674-679`), so they read by shape; the frame's 2 px line is the same yellow (`glass-c2-pages.css:657`) with a dimming outside it. That is an acceptable 1.4.11 case (the frame is also the edge of the dimmed area), and is listed in the Mac checks.

The existing test `src/styles/contrast.node.test.ts` checks the ink on each tab colour, each tab colour on the pane and the ground, the Lens violet at 7:1, and the mask accent; it reads hex tokens only. It does not read the glass's text tokens (`--glass-label*`), the disabled text, the placeholder, the focus ring against the glass, or anything rendered. A-05 adds those.

**Text, rendered** (1,060 elements in 25 states; the dark UI is the app's aurora and, as a stress case, black):

| Text | Colour (computed) | Worst measured | Where | Verdict |
|---|---|---|---|---|
| Body and labels | `#fafafa` | 5.67 | "Add", "Subtract", "Overlay" on the segmented control's fill (`#646464`) | pass |
| Secondary (`--glass-label-secondary`, `oklch(0.88 0 0)`) | about `#d7d7d7` | 4.98 | Preferences and About descriptions, over the dialog's `#54585c` | pass, close |
| Tertiary (`--glass-label-tertiary`, `oklch(0.62 0 0)`) | about `#868686` | 4.68 | the 12 px "or press" hint under Pick white (`.ym-pick-row__keys`, `aria-hidden`) | pass, closest |
| Placeholder ("Search looks…") | tertiary | above 9.4 | Look tab | pass |
| Large step numbers (48 px) | tertiary | 4.05 | Start's "1 2 3" | pass (needs 3) |
| **Disabled** | tertiary at reduced opacity | **1.47** | the `⌘E` hint in a disabled Export button; 3.64 "← Back" (wizard step 1); 4.05 Export; 4.17 Show mask; 4.36 Nodes while cropping; 4.48 Paste, Delete, Delete pin | exempt (1.4.3: inactive components); listed because several are under 4.5 and a low-vision user may still need to read the key hint |
| Chips on the photo ("Neutral · 100%", "Before", the readout) | `#fff` on a dark chip | 8.59 on a white photo, 14.7 on grey | over the cell | pass |
| **The camera line** (`.ym-photo-cell__camera`) | `#fff`, no chip, `text-shadow: 0 1px 3px rgb(0 0 0 / .45)` | **1.03 on a white photo, 4.17 on mid grey**, 21 on black | bottom right of the photo | **fails** on a bright photo (`components.css:242-251`) |
| **A menu over the photo** (the Fit menu: "Fit", "50 %", "100 %", "200 %") | `#fafafa` on `glass-liquid` (40 % tint) | **2.39 to 2.91 on a white photo**, 5.67 to 6.82 on grey, 11.2 on black | opens over the cell | **fails** on a bright photo |
| Panes (header, tool panel, Export panel, dialogs) | | 0 failures of non-disabled text on the aurora and on black | | pass |

On a white backdrop under the panes, 24 to 44 of 32 to 70 elements per state fail; that is not a case the app has, because the aurora (not the photo) is behind the panes and the photo cell is cut out of it (`docs/11` *Ground*). The cases that are real are the ones that float over the photo: the camera line, menus, popovers, tooltips, the Pick white and crop hints. The dialogs are near-opaque (rendered over white and black the same, 4.98 and up) and pass.

**Focus ring.** Computed outline: 3 px, 2 px off. Colours: the tab accent on controls (the ring's contrast with the pane is the accent's, 4.76 to 8.17); `--focus` `#0db7e8` on look tiles (5.63 on the pane, 8.17 on the ground); the tab triggers draw it inset in a lightened accent (`rgb(253 224 149)`). All pass 3:1 on the glass. On a white photo the yellow ring is 1.62 (the crop frame's focus ring is drawn inside the frame in `--ring`, `glass-c2-pages.css:664-670`); the ring of a control over the photo has nothing to contrast with there, which A-P5 covers.

**Target size (2.5.8, 24 px).** Interactive elements below 24 px: the tab triggers (15 px high), the crop handles (14 px), the mask handles (14 and 17 px), the slider knobs (16 px), the divider (10 px high). Each passes by the spacing exception (a 24 px circle on its centre touches no other target) except the divider, which sits 10 px from the cards above and the tabs below; it has a keyboard path. Recorded, not a task.

### 2.6 Reduce Motion, Reduce Transparency, Increase Contrast

Measured in Chromium on the Look tab and on Preferences:

| Setting | CSS today | Measured |
|---|---|---|
| Reduce Motion | `global.css:184-191`, `glass.css:988-1011` | Transitions `none`, animations 1 µs, `--glass-duration` 0 ms. No gap. JavaScript reads the query once in `dialog.tsx:15` and `use-glass-pointer.ts:22-24`; `canMorph` reads it on each call, fine |
| Reduce Transparency | `glass.css:1013-1028` | Chromium matches the query: pane tint 45 % to 90 %, blur 28 to 40 px, sheen off. The panes stay translucent (90 %) and still blur; the aurora is still drawn; the chip and the filmstrip thumbnail are unchanged; **menus, popovers and tooltips go to 90 %, never to solid**. A user who set Reduce Transparency asked for no see-through surfaces |
| Increase Contrast | `glass.css:1030-1056` | Pane tint 94 %, the rim 1.5 px and white, the secondary label lightened to `oklch(0.93)`, sheen off. Good. The tertiary label and the disabled text are not raised |
| Forced colours | `glass.css:1058-1085` | Not available in WebKit; left alone |

**The open question is WebKit.** The three rules are written as media queries. As far as the audit could establish, WKWebView honours `prefers-reduced-motion` and `prefers-contrast`; support for `prefers-reduced-transparency` is not something this audit could confirm, and Chromium's emulation proves nothing about it. The native side does not help: nothing in `src-tauri/src/` reads `NSWorkspace.accessibilityDisplayShouldReduceTransparency`, `…ReduceMotion` or `…IncreaseContrast` (`grep` finds a debug line, `app.rs:408`, that logs the motion query only). If WebKit does not match the query, Reduce Transparency does nothing in the app. That is A-MC-4, and A-06 builds the fix that does not depend on the answer: the shell reads the three settings, sets `data-reduce-transparency`, `data-increase-contrast` and `data-reduce-motion` on the root, follows the change notification, and the CSS answers both the query and the attribute.

The native viewport has no setting to honour: it draws a still frame. What Reduce Transparency changes there is the window behind it, which is already opaque `#0f0f10` (`native.rs`, `docs/11` *Native viewport*).

## 3. Gaps

| # | Gap | Where measured |
|---|---|---|
| A-G1 | Two scrollers have no keyboard route (Start's list, the Export panel's strip); in WebKit they cannot be scrolled from the keyboard | 2.2 |
| A-G2 | The node view's divider stands outside every landmark | 2.2, N5 |
| A-G3 | `aria-label` on `code`; the Stock field's `aria-controls` | 2.2, N4 |
| A-G4 | The photo cell's chips, camera line and readout are inside `role="img"` | N1 |
| A-G5 | The camera line is unreadable on a bright photo (1.03:1); menus and popovers over a bright photo are 2.4:1 to 2.9:1 | 2.5 |
| A-G6 | A Tab stop with no ring (the tab list in Pick white) | F1 |
| A-G7 | Focus is lost to `body` after About and after the Export panel opens or closes by ⌘E | F3, F4 |
| A-G8 | Nothing announces what a shortcut did | N6 |
| A-G9 | The windowed lists give no position (`aria-setsize`, `aria-posinset`) | N3 |
| A-G10 | Brush strokes and Object points have no keyboard path; a zoomed photo cannot be panned from the keyboard; the pixel readout is pointer-only | 2.4 |
| A-G11 | Single-key shortcuts cannot be switched off (WCAG 2.1.4, level A) | F7 |
| A-G12 | Reduce Transparency and Increase Contrast rest on media queries that WebKit may not match; nothing reads the system setting; Reduce Transparency leaves floating layers 90 % and blurred | 2.6 |
| A-G13 | Floating layers (menus, popovers, tooltips) are 40 % glass, readable over the aurora and not over a bright photo | 2.5 |
| A-G14 | No test catches a regression in any of this: axe is not run, the focus order has no golden list, and contrast is checked on hex tokens, not on what is drawn | 2.5 |
| A-G15 | The crop frame is `role="group"` with `tabindex="0"`: a screen reader says "group", not that the arrows move it (the label does say so) | 2.4 |
| A-G16 | Nothing has been tried with VoiceOver, Full Keyboard Access, Voice Control or the real system settings | section 7 |

## 4. Findings

Each finding has a fix and the test that proves it.

**A-P1 (A-G1) Scrollers get a keyboard route.** `.ym-develop-list` and the Export strip's `.ym-filmstrip__scroll` are scroll containers with nothing to focus. Fix: give each `tabindex="0"`, `role="region"` and a name ("Photos to develop", "Photos to export"), and the same ring as any control (`outline: 3px solid var(--ring)`, offset `-3px` so it is not clipped). The arrow keys then scroll it in WebKit and Chromium. The Export strip keeps its photos as images. *Proof:* axe `scrollable-region-focusable` is 0 in states 02, 03, 25 to 29; the focus-order test lists the region with a ring.

**A-P2 (A-G2, A-G3) Landmarks and attributes.** Put the divider inside the node view's landmark: render it as the first child of `section aria-label="Node editor"` or give the group around the divider and the graph `role="region"`. Replace `<code aria-label="Command line">` with `<code>` inside a `<div role="group" aria-label="Command line">`. Stock field: render `aria-controls` only while the popover is mounted. For the Radix select viewport, add the allowlist (Radix's portal, a dialog child). *Proof:* axe `region` 0 in the 29 non-menu node states; `aria-prohibited-attr` 0.

**A-P3 (A-G4) The photo cell's texts leave the image.** Make the picture `role="img"` a sibling of the chips, not their parent: `<div class="ym-photo-cell__view" role="img" aria-label=…>` holds only the `<img>`; the chip, the Before badge, the camera line and the readout go in a `<div role="group" aria-label="Photo information">` next to it, still positioned over the cell. The image label keeps the photo and the look. The readout becomes a `role="status"` that is polite but updated at most once a second. *Proof:* an a11y test reads the tree (`toMatchAriaSnapshot`): the chip, "Before · camera default" and the camera line are text under a group, not inside `img`.

**A-P4 (A-G5, A-G13) Legible over any photo.** Two changes. (1) The camera line gets the chip's material: a dark glass pill (`rgba(14,14,18,0.74)`, the chip's own value, `components.css:229`), measured 8.6:1 on white. A text-shadow cannot do it: no shadow rescues white on white. (2) Menus, popovers, select lists and tooltips are drawn as `glass-strong` with a floor of 88 % tint, whatever the setting, so a menu over a white photo reads at 6:1 or more. The panes stay as they are. *Proof:* a rendered-contrast e2e test (A-05) paints the cell white, grey and black behind the camera line, the chips, the Fit menu, a tooltip and the Pick white hint, and asserts 4.5:1 for each.

**A-P5 (A-G5) The focus ring over the photo.** The ring of a control that lies over the picture (the crop frame and handles, the mask handles, the crosshair layer) is a double ring: the accent with a 1 px dark line outside it (`box-shadow: 0 0 0 1px rgb(0 0 0 / .6)`), so it reads on white (the yellow alone is 1.6:1). *Proof:* the rendered test measures the ring against a white photo at 3:1 for the union of both lines.

**A-P6 (A-G6, A-G7) Focus order and focus return.**
1. The tab list is not a Tab stop: set `tabIndex={-1}` on `TabsList`'s root (`TabBar.tsx:25`), or keep one tab tabbable while Pick white locks the others.
2. About opened from Preferences returns focus to the element that opened Preferences: `Overlays` keeps a stack of openers instead of one.
3. ⌘E moves the focus into the Export panel (`[data-autofocus]`, the main button) when it opens, and back to the control that had it, else to the header's Export button, when it closes by Esc or Done. The same for the node view's N.
*Proof:* a Playwright test with a golden Tab list per screen (26 states of 2.4; each stop's role and name, in order, and `outline-style` not `none`); a test of the three returns.

**A-P7 (A-G8, A-G9) Say what a shortcut did, and where a photo is.** One polite live region in the Photos screen (`Photos.tsx:1007`, which exists) says the result of each shortcut in a short plain sentence: "Look: Film warm.", "Before.", "After.", "Zoom 100 %.", "Picked. 4 picked.", "Crop. Press Return to finish.", "Node view.", "Photo 5 of 12." A key held down says it once, at the end. The windowed filmstrip sets `aria-setsize` and `aria-posinset` on each drawn thumbnail (the `Show …` buttons are `button`s, so the pair goes on a wrapping `role="listitem"` inside a `role="list"`, or the strip becomes a `listbox` with `option`s; the first is the smaller change). The Start list and the Export queue do the same. *Proof:* Vitest on the announce function for each shortcut; a Playwright test that presses 3, `\`, Z, P and reads the region; the 2000-photo test reads `aria-setsize="2000"` and `aria-posinset="812"` on the 812th photo once it is scrolled into view.

**A-P8 (A-G15) The crop frame's role.** `role="group"` with `tabindex="0"` becomes `role="application"` with the same label, or the frame becomes a slider-like `role="group"` plus an `aria-describedby` that says the arrows move it and ⇧ moves faster. The eight handles stay buttons. *Proof:* the tree test.

**A-P9 (A-G10) Object points from the keyboard.** The Object mask's points are placed on the crosshair layer already built for Pick white (`PickLayer.tsx:76-113`): with the Object row selected, Return starts "Place points", the arrows move the crosshair (⇧ for ten steps), Return adds a positive point, ⇧Return a negative one, Backspace removes the last, Esc ends. The layer's name says it, as Pick white's does. *Proof:* a Playwright test that places two points by keys and reads the same `points` the mouse test does (`ObjectTool.test.tsx` is the model).

**A-P10 (A-G10) Pan and read a pixel from the keyboard.** With the photo zoomed and the cell focused (the cell becomes focusable when zoomed, a Tab stop named "Photo, zoomed. The arrow keys move it."), the arrows pan by 5 % of the view, ⇧ by 25 %, Home returns to the centre. ← and → keep stepping photos when the filmstrip, not the cell, has the focus; the cell's own arrows are used up there (the rule `usesHorizontalArrows` already exists, `shortcuts.ts`). A second key, `I` while the cell is focused, reads the pixel under the cross of the view's centre into the polite region ("R 127 G 166 B 199"). *Proof:* a Vitest of the pan arithmetic (the clamps are in `zoom.ts`); a Playwright test that zooms to 200 %, presses →, and sees the offset change.

**A-P11 (A-G11) A switch for single-key shortcuts.** Preferences gets "Single-key shortcuts" (a `Switch`, on by default, one row, with the line "Turn off if you use speech input or a screen reader. Every action stays in the menu bar."). Off: P, Z, C, W, N, F, B, M, A, `\`, 1 to 9 and Delete do nothing outside a text box (⌘-keys, the arrows and Esc stay). The menu bar's items show the letter in their name for those that have one; a menu item runs the same code (`state/menu.ts`). The value is one of the few preferences (`docs/01` *Persistence*). *Proof:* `shortcuts.test.ts` gets a context flag; a Playwright test turns it off and checks that each of the 21 does nothing and that the menu item does it.

**A-P12 (A-G12) The shell reads the display settings.** `src-tauri/src/` gains `a11y.rs` (macOS only; a no-op elsewhere): on start and on `NSWorkspaceAccessibilityDisplayOptionsDidChangeNotification` it reads `accessibilityDisplayShouldReduceTransparency`, `…ReduceMotion` and `…IncreaseContrast` through `objc2-app-kit` (already a dependency, `native.rs:68`) and emits `display_options { reduce_transparency, reduce_motion, increase_contrast }`; the front end sets `data-reduce-transparency`, `data-reduce-motion` and `data-increase-contrast` on `<html>`. The CSS answers both: `@media (prefers-reduced-transparency: reduce), :root[data-reduce-transparency]`. Under Reduce Transparency: panes `--glass-opacity: 100 %`, no `backdrop-filter`, the aurora is still drawn but flat (it is the pane behind that matters, and a solid pane hides it), menus and tooltips solid. The look (shape, rim, type, accents) is unchanged. Under Increase Contrast: the tertiary label and the disabled text go to the secondary's value, and the focus ring to 4 px. *Proof:* Vitest of the event handler; a Playwright test that sets the three attributes and asserts computed opacity 100 %, `backdrop-filter: none`, and the rendered contrast test passes again; `native-viewport.node.test.ts` style checks that the rules exist. On a Mac the three settings are toggled live (A-MC-4).

**A-P13 (A-G14) The tests.** (a) `e2e/axe.spec.ts`: axe over the 79 states with a named allowlist (the seven open-menu states for `aria-hidden-focus`, `landmark-one-main`, `page-has-heading-one`, `region`, `bypass`; `color-contrast`, which A-05 replaces). (b) `e2e/focus-order.spec.ts`: the golden Tab lists. (c) `src/styles/contrast.node.test.ts` is extended (A-05). (d) `e2e/a11y-tree.spec.ts` keeps what `a11y.spec.ts` does and adds the A-P3 snapshot.

**A-P14 Dependencies.** `@axe-core/playwright` and `axe-core` are MPL-2.0. Under `docs/14` they are treated as `lightningcss` is (`docs/14:57`): **a dev dependency, a build and test tool, not shipped.** No file of axe is in the app, in the repository or in the bundle; we do not modify it; MPL-2.0 file-level terms reach only a file we change and distribute. They stay out of the `--production` npm licence check and out of `THIRD-PARTY.md`; the row goes into the `docs/14` table with that reason. If axe ever ran inside the shipped app, the position would change; it does not. The allowance is recorded as a question (Q5) only because it adds a dependency.

## 5. Tasks

One branch and one pull request each (`m4-04-a-03-…`), the usual gate (fmt, clippy, tests, `npm test`, `npm run build`, `npm run e2e`). In the order that gives most value first: the harness that stops regressions, the two things a keyboard or VoiceOver user trips over most, the contrast that fails today, then the larger work.

**A-01 · The axe harness** (A-P13a, A-P14). `@axe-core/playwright` as a dev dependency, `e2e/axe.spec.ts` over every state of `STATES` and the 500-photo Photos screen, the allowlist with a reason on each entry, `docs/14` row. The baseline of this audit is the starting allowlist: the seven open-menu states; the four real findings listed as "known, fixed by A-02" so the test is green on day one and each fix removes a line. *Done when:* the spec runs in `npm run e2e` in under three minutes, fails when a new violation is added to any state, and the allowlist holds only the entries of 2.2 with their reasons; `docs/14` lists axe as a dev dependency under the `lightningcss` rule; `npm ci`, the licence checks and `cargo deny` pass.

**As built (A-01).** `@axe-core/playwright` 4.10.2 is a dev dependency, pinned exactly (it brings `axe-core` 4.10.3; the audit ran 4.10.2). `e2e/axe.spec.ts` is one test per state of `STATES` (79 states) and one for the 500-photo Photos screen, `many-photos-500`; each opens its own page through `openState`, and the describe is `parallel`, so the two workers CI has split them: the run took 3.2 minutes on the 4-CPU agent (6.4 minutes one after the other; opening a state costs 1.5 to 4.3 s and axe about 1 s, so the bound of three minutes holds on a quiet machine and is a little over here). The rule sets are the six of the audit. A finding is a violation, or an incomplete result, except two kinds of incomplete that are not counted and are named in the file with their reason: `color-contrast` (switched off for every state, A-05 replaces it) and `aria-hidden-focus` and `bypass` (a Radix modal's `hideOthers`: axe asks a person to check what is hidden, and the dialogs and menus of Radix keep Tab out of it; they were incomplete, not violations, in the audit). `ALLOWED` is a list of `{rule, states, reason}`, and the test fails on a finding that no entry covers **and** on an entry whose rule axe no longer reports in one of its states, so a fix has to remove its line. On day one it holds: the seven open-menu states (`aria-hidden-focus`, `landmark-one-main`, `page-has-heading-one`, `region`, and `scrollable-region-focusable` in 43, 52 and 59, where the menu's list scrolls); `region` in 40 (the Select's viewport, a Radix portal); and the four findings of 2.2, marked "Known, fixed by A-02": `scrollable-region-focusable` in 02, 03 and 25 to 29, `region` in the 28 node-view states, `aria-prohibited-attr` in 25 to 28 and `aria-valid-attr-value` in 68. The `docs/14` row is in (MPL-2.0, test tool only, as `lightningcss`); the CI licence line (`--production`) lists no axe package and exits 0, and `Cargo.lock` is untouched, so `cargo deny` has nothing new to read. Checklist line 1 is *built, with the four A-02 entries still allowed*.

**A-02 · The four axe findings** (A-P1, A-P2). Focusable, named scrollers; the divider in a landmark; the command line's group; the Stock field's `aria-controls`. *Done when:* the allowlist of A-01 has no entry left for them; axe reports 0 `scrollable-region-focusable` outside open menus and 0 `region` outside the Radix portals; the focus-order test of A-04 shows the two regions with a ring.

**A-03 · The photo cell** (A-P3, A-P8). Chips, camera line and readout out of the image; the readout as a throttled status; the crop frame's role. *Done when:* the tree test shows the four texts in a group beside the image, not inside it; the crop frame is announced with its arrows hint; the screenshots of `design/screenshots-c2/` still match (nothing moves).

**A-04 · Focus order and focus return** (A-P6, A-P13b). The tab list out of the Tab order, About's opener, ⌘E and N focus in and out; the golden Tab lists. *Done when:* the golden test passes for the 26 states and fails if a stop is added, removed, reordered or left without a ring; the three focus-return tests pass; no state has a stop with `outline-style: none`.

**A-05 · Contrast, measured and kept** (A-P13c). Extend `contrast.node.test.ts` to read `--glass-label`, `-secondary`, `-tertiary`, the focus colours and the pane estimate, and to assert: body 7:1, secondary 4.5:1, tertiary 4.5:1 on the pane and the ground, the ring 3:1 on both. A rendered-contrast Playwright test (`e2e/contrast.spec.ts`, the method of 2.5, with the cell painted white, grey and black) over 25 states. Disabled text is listed, not asserted, except that a disabled button's key hint is at least 3:1. *Done when:* both tests pass; the table of 2.5 is reproduced by the test's output; the camera line, the Fit menu and the hints over the photo fail the new test until A-07 lands, and are listed in it as `expected failures` with the task's name, so the test is green and the fix cannot be forgotten.

**A-06 · Legible over any photo** (A-P4, A-P5). The camera line's pill, the 88 % floor for menus, popovers, selects and tooltips, the double ring over the picture. *Done when:* the rendered test of A-05 has no expected failure left: every text over the cell is 4.5:1 and every ring 3:1 on white, grey and black; the design screenshots that show a menu or the camera line are re-taken and reviewed by the owner ([HUMAN] look check; the pill is the only visible change).

**A-07 · Reduce Transparency and Increase Contrast from the system** (A-P12). `a11y.rs`, the event, the three attributes, the CSS fallback. *Done when:* the unit and Playwright tests pass; with `data-reduce-transparency` set, the panes are opaque, no `backdrop-filter` is computed anywhere, and the rendered test passes; the Mac check A-MC-4 is listed in `docs/STATUS.md`.

**A-08 · Announcements and positions** (A-P7). The live region's sentences, `aria-setsize` and `aria-posinset` in the filmstrip, the Start list and the Export queue. *Done when:* the tests of A-P7 pass at 12, 500 and 2000 photos, and the windowed lists keep their latency (`e2e/perf/latency.perf.ts` with `YM_PERF_PHOTOS=2000`, the 1.25 ratio of H-09 holds).

**A-09 · The single-key switch** (A-P11). Preferences row, the context flag, the menu names, the persistence. *Done when:* the Playwright test turns it off and none of the 21 keys acts, while each menu item still does; the preference survives a restart (`docs/01` *Persistence* lists it); `docs/11` says it.

**A-10 · Pan and pixel readout from the keyboard** (A-P10). *Done when:* a zoomed photo pans with the arrows and Home; `I` reads the centre pixel into the live region; the filmstrip's arrows still step photos; `docs/11`'s shortcut table lists both.

**A-11 · Object points from the keyboard** (A-P9). *Done when:* two points are placed by keys and give the same `points` as the mouse; the Masks card's hint says how; `docs/23`'s "no keyboard path for placing points" line is replaced by what the code does. The Brush stays pointer-only unless the owner decides otherwise (Q2).

**A-12 · Close the books** (STATUS, `docs/11`, `docs/15`, `docs/14`, the user guide's accessibility page for M4-03). *Done when:* each spec says what the code now does, the checklist of section 6 is run, its numbers are in `docs/STATUS.md`, and A-MC-1 to A-MC-10 are under *Waiting on the owner*.

**Suggested order:** A-01, A-02, A-04 and A-05 can start at once. A-03 after A-01. A-06 after A-05. A-07 and A-08 are independent. A-09, A-10, A-11 last; each is a feature and each is a decision the owner can drop (section 8). A-12 last.

## 6. The checklist (M4-04 done-when)

M4-04 is done when every line below is true. Each line is a CI check or a named Mac check; CI lines are run by `npm test` or `npm run e2e`, so the gate stays one command.

**Names (VoiceOver labels)**
1. axe, rule sets `wcag2a`, `wcag2aa`, `wcag21a`, `wcag21aa`, `wcag22aa`, `best-practice`: **0 violations in each of the 79 states and in the 500-photo state**, with only the allowlisted Radix open-menu entries of 2.2 (`e2e/axe.spec.ts`, A-01: built; the four A-02 entries are still on the list).
2. Every control has a role and a name in every state (`e2e/a11y.spec.ts`, which exists and stays green).
3. The photo's chip, the Before badge, the camera line and the readout are not inside `role="img"` (`e2e/a11y-tree.spec.ts`, A-03).
4. The windowed filmstrip, the Start list and the Export queue carry `aria-setsize` and `aria-posinset` at 2000 photos (A-08).
5. VoiceOver reads Start, Photos, each tab, the node view, the Export panel, Preferences and the wizard with a sensible name, role and value for each control: **A-MC-1.**

**Focus order and keyboard**
6. The golden Tab list of each of the 26 states matches (`e2e/focus-order.spec.ts`, A-04); no stop has `outline-style: none`; no keyboard trap (Tab leaves every screen; a dialog keeps Tab inside).
7. Focus returns: after Esc from Preferences, About, the consent dialog, any menu and the Export panel, the focus is on the control that opened it (A-04 tests).
8. Single-letter keys do nothing in a text box (the 18-key test, kept) and nothing at all with "Single-key shortcuts" off (A-09).
9. Every task of 2.4 has a keyboard path, except Brush strokes if the owner keeps Q2's default: crop, mask handles, Object points (A-11), picking, wheels, curves, warp pins, graph, links, divider, scrolling of every list, pan and pixel readout (A-10). A Playwright test per path.
10. Full Keyboard Access on a Mac reaches every control and no stop is invisible: **A-MC-3.**

**Contrast**
11. `contrast.node.test.ts`: the eight tab colours (ink on colour 4.5:1; colour on pane and ground 3:1), body 7:1, secondary and tertiary 4.5:1 on the pane and the ground, the ring 3:1 on both (A-05).
12. `e2e/contrast.spec.ts`: over the 25 states, **0 non-disabled text under 4.5:1 (3:1 when large) on the aurora and on black**, and 0 under 4.5:1 for the camera line, the chips, the menus, the tooltips and the hints over a white, a grey and a black photo; the focus ring 3:1 over each (A-05, A-06).
13. The disabled key hint is at least 3:1; the other disabled texts are listed in the test output (A-05).

**System settings**
14. With the three root attributes set, the rendered test passes with every pane opaque and no `backdrop-filter` (A-07); with `prefers-reduced-motion`, nothing animates (the existing tests).
15. The real settings change the app live on a Mac: **A-MC-4.**

**Mac checks**
16. A-MC-1 to A-MC-10 done by the owner, results in `docs/STATUS.md`.

Until the owner has run the Mac checks, M4-04 reads *built and unproven*, as M4-02 did.

## 7. Checks on a Mac

None of these can be done on Linux. Each has a place in `docs/STATUS.md` under *Waiting on the owner* when its task lands (M0-H4 style).

| ID | Check | For |
|---|---|---|
| A-MC-1 | **VoiceOver on WKWebView.** Turn VoiceOver on (⌘F5) and go through Start, Photos (Look, Light, Colour, Curve, Lens), Crop, Pick white, the node view, the Export panel, Preferences, About and the wizard with the VO keys only. For each control: is it announced with its name, role and value ("Exposure, slider, 0.30")? In particular: does VoiceOver say the look name, "Before" and the camera line (N1; after A-03, as a group)? Does it announce the photo as an image with its label and stop there? Are the look tiles, the filmstrip thumbnails ("Show DSC04204, picked, current") and the colour wheels sensible? Does the web rotor list headings and landmarks (header, Tools, Photos, Node editor)? | A-02, A-03, A-08 |
| A-MC-2 | **The native viewport and NSAccessibility.** With VoiceOver on, move the VO cursor over the photo: it must land on one element, the image labelled "Photo …", and not on an unnamed group or on a second element for the Metal view. Click the photo, then press P, ←, →, 1 and ⌘E: the shortcuts still work (the click must not make the Metal `NSView` the first responder and swallow keys). If the view shows up, set it not accessible (`setAccessibilityElement(false)`) in `native.rs` | A-03 |
| A-MC-3 | **Full Keyboard Access and Tab.** With Full Keyboard Access off, then on (System Settings › Keyboard › Keyboard navigation): does Tab reach every button, tab, slider and switch, or only text fields and links? (WebKit's default skips buttons unless the system setting says otherwise; a Tauri window may or may not follow it.) Is the ring visible on each stop on the P3 display over the glass and the aurora? Does Space scroll the Start list when it has the focus? If Tab skips buttons with the setting off, a task is added to turn it on in the web view's configuration | A-04, A-02 |
| A-MC-4 | **Reduce Motion, Reduce Transparency, Increase Contrast**, each turned on while the app is open (System Settings › Accessibility › Display). Does the app follow at once, with no restart? Are the panes, menus and tooltips solid under Reduce Transparency? Is Increase Contrast visible (rims, secondary text)? Does Reduce Motion stop the glass's press and morph effects and the slide of the tab pill? Report which of the three media queries WebKit matches (open the web inspector and run `matchMedia('(prefers-reduced-transparency: reduce)').matches`) | A-07 |
| A-MC-5 | **Voice Control.** Say "Show numbers" and "Show names": every button, tab, slider, tile and thumbnail has a name that can be spoken ("Click Export", "Click Crop"); say "Click Crop", "Click Light". Does the app react to single letters spoken as dictation (F7)? With A-09 on, are they off? | A-09 |
| A-MC-6 | **Quick Nav and shortcut collisions.** With VoiceOver's Quick Nav on (← and → together), do the arrow keys, ⌥← and ⌥→ still reach the filmstrip and the graph? Does anything clash with the VO keys (⌃⌥) or with VoiceOver's single-key commands? | A-04 |
| A-MC-7 | **Contrast on the real display.** On the built-in Liquid Retina XDR display at the default brightness, and on an external sRGB display, look at the camera line, the Fit menu and a tooltip over a bright photo (a snow scene, an overexposed sky) before and after A-06; check that the tab colours and the focus ring are distinct in a dim room and in daylight | A-06 |
| A-MC-8 | **Focus returns and the Export panel with VoiceOver.** ⌘E: is the panel announced when it opens, does the focus start on its main button, and does Esc return to where it was? Preferences › About › Esc | A-04 |
| A-MC-9 | **The announcements.** With VoiceOver on, press 3, `\`, Z, P, C, N on the photo: is each result said once, politely, and not so often that it talks over the person? Does a held key say it once? Is the sentence for stepping photos useful ("Photo 5 of 12") or noise? | A-08 |
| A-MC-10 | **Pointer-free day.** Open a folder, develop, pick, apply a look, crop, set a white point, make an Object mask, adjust a curve, export — with the keyboard alone, with and without VoiceOver. Note every place the keyboard path is longer than 8 key presses for something the pointer does in one gesture | A-09 to A-11 |

## 8. Questions for the owner

Only the real decisions; each has the default the plan takes.

1. **Single-key shortcuts.** WCAG 2.1.4 (level A) asks for a way to switch single-key shortcuts off. Add "Single-key shortcuts" to Preferences, **on by default** (A-09)? *Recommended: yes.* Default on keeps every habit and the mockups; off is for speech input and screen-reader users. The alternative, requiring a modifier on every shortcut, would break the way the app is used and is not proposed.
2. **Brush strokes without a pointer.** A brush stroke has no keyboard path, and designing one (a crosshair with a pen-down key, a size and a stroke list) is a feature of its own. *Recommended:* keep the Brush pointer-only in 1.0, say so in the user guide, and note that Subject, Background, Depth, Linear, Radial, Colour range and Brightness range masks, and (after A-11) Object points, do the same work from the keyboard. If you want it, it is a separate task after 1.0.
3. **Floating layers are near-opaque.** Menus, popovers, selects and tooltips float over the photo, where 40 % glass is unreadable on a bright picture (2.4:1). *Recommended:* make them at least 88 % tint always (A-06), and keep the panes translucent. Same glass, the same rim, less see-through for what pops over the photo. The alternative, raising them only under Reduce Transparency, leaves the default broken for everyone.
4. **The camera line gets a pill.** The line under the photo is bare white text and unreadable on a bright sky. *Recommended:* give it the chip's dark glass, so it reads like "Neutral · 100%" at the other corner. This is the one visible design change of the plan; the screenshots of `design/screenshots-c2/` that show it are re-taken. A stronger shadow cannot fix white on white.
5. **axe as a dev dependency.** `@axe-core/playwright` and `axe-core` are MPL-2.0 and are test tools that never ship, the case `docs/14` already allows for `lightningcss`. *Recommended:* yes (A-01). Say no and the checklist's lines 1 to 3 go to a hand-run check.

## Appendix · How to repeat the measurements

None of the scripts is in the repository. They live in the audit's scratch directory and are short.

```bash
# axe: a scratch install, never added to package.json
mkdir -p $SCRATCH/m404 && cd $SCRATCH/m404 && npm init -y && npm i axe-core@4.10.2 pngjs@7.0.0
# a config that points testDir at the scratch specs and launches Chromium with
#   launchOptions: { executablePath: '/opt/pw-browsers/chromium' }, trace: 'off'
npm run dev &                      # the mock on :1420
cd $REPO && node_modules/.bin/playwright test -c $SCRATCH/m404/playwright.config.ts --trace off axe

# axe spec: for each [name] of STATES (e2e/helpers/states.ts):
#   const { page } = await openState(browser, name); await page.evaluate(AXE_SOURCE);
#   axe.run(document, { runOnly: { type: 'tag', values: [...] }, resultTypes: ['violations', 'incomplete'] })

# focus order: openState, blur, press Tab up to 140 times, after each press describe document.activeElement
#   (role, name, outline-style and width, bounding box); stop on the second visit of the same box or on body.

# contrast: openState, collect visible elements with a direct text node (colour resolved through a 1x1 canvas drawn over black
#   and over white to recover its alpha), screenshot, take the commonest colour in each box as the background.
#   Repeat with `.ym-aurora { background: #000 }`, and with `.ym-photo-cell { background: #fff | #808080 | #000 }` and the
#   photo hidden, keeping the elements that overlap the cell. The Fit menu: click "Fit", wait for role=menu, then the same.

# media features: context.newCDPSession(page).send('Emulation.setEmulatedMedia', { features: [{ name: 'prefers-reduced-transparency', value: 'reduce' }] })
#   and read getComputedStyle of .ym-header, .ym-tools, [role=dialog], button.glass.
```
