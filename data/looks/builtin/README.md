# data/looks/builtin/

Built-in looks as JSON parameters (`yellowmoon.parametric-look/1`, see `docs/08-looks.md`):

- `neutral.json`: the identity (M1-10). Neutral also exists in code, so the app works without this folder. It is the look of "no look" and stays in the list, but it is no longer the default.
- `signature.json`: the look new photos get out of the box (owner's decision of 2026-10-14; the name may change later, which changes the file's `name` and nothing else). The library lists exactly the files that are here, so a folder without it has Neutral only and new photos then get Neutral.

Any other `*.json` here is listed as `builtin:<file name>`. No ARRI data ever goes here.

## Signature

A LogC4 tone curve on the luminance of the pixel, then ASC CDL saturation 1.20:

- `luma_curve`: 15 points `{ "x", "y" }`, the curve of candidate B of the look preview. It is applied as `v += T(L) - L` on all three channels, `L` the LogC4 code of the AWG4 luminance (the way the Curve tab's luma curve works, `docs/02` C11), through the monotone cubic of `ym_color::curve::Curve` (Fritsch-Carlson tangents). Mid grey (LogC4 0.2784, scene 0.18) stays where it is. Against Neutral it darkens the shadows (about -0.4 stop from -3 to -6 stops), lifts the lights (+0.22 stop at +1, +0.42 at +2, +0.52 at +3) and eases the highest lights back.
- `cdl`: slope 1, offset 0, power 1 and saturation 1.20 (Rec.709 luma weights on the LogC4 codes).

The constants were chosen by eye on twelve photos and nothing was fitted to any look file or LUT. The loader bakes the look into a 33-point cube like every parametric look, so the CPU and the GPU read it as an ordinary LUT. To change it, edit the points and the saturation here; `ym looks bake signature.json --preview <ARW>...` renders a contact sheet against Neutral. A displayed neutral ramp at the app's +0.85 EV placement lands at about 42 / 120 / 204 (8-bit codes at -2 stops / mid grey / +2 stops) against 52 / 120 / 192 for Neutral.
