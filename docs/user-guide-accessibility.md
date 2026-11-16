# Accessibility

> Draft for the user guide (`docs/15` M4-03, which will take this page into `docs/user/`). It is written from what the code does (`docs/28`). The sentences about VoiceOver and the system settings are confirmed on a Mac by the owner's checks A-MC-1 to A-MC-11 (`docs/STATUS.md`, *Waiting on the owner*); change anything those checks contradict.

Yellowmoon works with the keyboard alone, with VoiceOver, and with the display settings that macOS offers. This page says what to expect.

## VoiceOver

Turn VoiceOver on with ⌘F5. Every button, tab, slider, switch and menu has a name, and sliders say their value, for example "Exposure, slider, 0.30".

- **The photo** is read as one image, with the photo's name and the look it wears: "Photo DSC04204, Neutral · 100%". Next to it is a group called "Photo information". It holds the look name (or "Before · camera default" while you compare), the camera line, and the pixel readout when you ask for one.
- **The filmstrip** reads each photo by name, and says whether it is picked and whether it is the one on screen: "Show DSC04204, picked, current". In a long folder it also says where the photo is in the list, for example "812 of 2000". The list on the Start screen and the queue in the Export panel do the same.
- **Tabs, wheels, curves and pins.** The seven tabs are a tab bar. The colour wheels, the curve points and the Colour warp pins are sliders: they say what they change and their value.
- **Crop.** The crop frame says that the arrow keys move it and that Shift moves it faster. Each of its eight handles is a button with its own name.
- **The node view.** The graph is one stop in the Tab order. Each card says its number, its name, what feeds it, and whether it has masks.
- **What a key did.** When you press a key on the photo, Yellowmoon says the result in a short sentence: "Look: Film warm.", "Before.", "Zoom 100 %.", "Picked. 4 picked.", "Photo 5 of 12.", "Node view.". If you hold a key down, it says the first step and then the last, not every step in between. Nothing is said for an action that opens a dialog or moves the focus; those announce themselves.
- **Errors and progress** are read when they appear: the error bar, the export, the look-import steps and the hints while you pick a white point.

## Moving around with the keyboard

Press Tab to move forward and Shift-Tab to move back. The control that has the focus has a ring around it, 3 pixels wide, in the colour of its tab. The tab bar is a single stop: press ← and → to change tab. The filmstrip is a single stop too: press ← and → to change photo.

| To do this | Press |
|---|---|
| Open files / open a folder | ⌘O / ⇧⌘O |
| Export | ⌘E (start the export with ⌘↩) |
| Preferences | ⌘, |
| Close the session | ⌘W |
| Go to the next / previous photo | → / ← |
| Pick or unpick the photo, pick all | P, ⌘A |
| Choose a look from the Look tab | 1 to 9 |
| Switch tab | ⌘1 to ⌘7 |
| Before and after | \ |
| Fit and 100 % | Z |
| Zoom in and out | ⌘+ and ⌘− |
| Crop (Return finishes) | C |
| Pick white | W (Esc ends it) |
| Node view on and off | N |
| Full screen | F |
| Undo / redo | ⌘Z / ⇧⌘Z |
| Copy settings / paste | ⇧⌘C / ⇧⌘V |
| Move the crop frame | Arrow keys, Shift for bigger steps |
| Move the crosshair when picking a colour or a white point | Arrow keys; Return picks |
| Move a mask's handles (Linear and Radial) | Arrow keys, 1 pixel at a time; Shift for 10 |
| Move around a zoomed photo | Tab to the photo, then arrow keys (5 % of the view; Shift 25 %); Home centres it |
| Read one pixel of a zoomed photo | I, with the photo focused: it says "R 127 G 166 B 199." |
| Place points for an Object mask | Return on the Object row, arrows to move the crosshair, Return for a point, Shift-Return to take a part away, Backspace to remove the last point, Esc to finish |
| Colour wheels, curve points and Colour warp pins | Arrow keys move them; Delete takes a point or pin away; Return on a curve adds a point |
| The node view | Arrow keys move between cards, Return selects, B bypasses, Delete removes, ⌥← and ⌥→ reorder; the More menu links nodes |
| Resize the node graph | Tab to the divider, then ↑ and ↓ (Shift for bigger steps) |
| Open the menu of a photo in the filmstrip | The Menu key, or Shift-F10 |

Every action that has a single-letter key also has an item in the menu bar, so you can find it by name.

If a list or a strip is longer than its box (the Start list, the Export strip), Tab to it and use the arrow keys to scroll.

## Single-key shortcuts

Letters such as P, C, W, N, F, B, M, A, Z, the numbers 1 to 9, Delete and Backspace work as shortcuts whenever you are not typing in a box. If you use speech input or a screen reader, a spoken letter can trigger one by accident.

Open Preferences (⌘,) and turn off **Single-key shortcuts**. Those keys, and I on a zoomed photo, then do nothing outside a text box. The shortcuts with ⌘, the arrow keys, Esc and the Return of Crop keep working. Every action stays in the menu bar: the Photo menu has Crop, Auto Crop, Pick White, Remove Photo, Bypass Node and Show or Hide Node Mask, and Looks has Apply Look 1 to 9. The setting is remembered.

## Reduce Transparency, Increase Contrast and Reduce Motion

Yellowmoon follows System Settings › Accessibility › Display, and changes at once when you change a setting, with no restart.

- **Reduce Transparency.** Every panel, menu, dialog and tooltip becomes solid, and nothing blurs what is behind it. The colours, the type and the shapes stay as they are.
- **Increase Contrast.** Rims become white and thicker, the grey text is lightened, and the focus ring grows from 3 to 4 pixels.
- **Reduce Motion.** Transitions and animations stop: the panels, the tab highlight and the press effects change without moving.

Menus, tooltips and dialogs are always nearly solid, so they read over a bright photo. The camera line under the photo sits on a dark pill for the same reason.

## What needs a pointer

**The Brush mask is pointer-only.** Painting strokes needs a mouse or trackpad. The other kinds of mask do the same work from the keyboard:

- **Subject** and **Background** are added with one action.
- **Depth**, **Colour range** and **Brightness range** have sliders; a range is picked on the photo with the crosshair and the arrow keys.
- **Linear** and **Radial** have handles that the arrow keys move.
- **Object** takes points from the keyboard, as in the table above.

The Object mask's Box tool also needs a pointer (it takes two corners). Use points instead.
