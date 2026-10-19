# data/looks/builtin/

Built-in looks as JSON parameters (`yellowmoon.parametric-look/1`, see `docs/08-looks.md`):

- `neutral.json`: the identity (M1-10). Neutral also exists in code, so the app works without this folder. It is the look of "no look" and stays in the list, but it is no longer the default.
- `signature.json`: the look new photos get out of the box (owner's decision of 2026-10-14, made richer on 2026-10-19; the name may change later, which changes the file's `name` and nothing else). The library lists exactly the files that are here, so a folder without it has Neutral only and new photos then get Neutral.

Any other `*.json` here is listed as `builtin:<file name>`. No ARRI data ever goes here.

## Signature

A LogC4 tone curve on the luminance of the pixel, then ASC CDL saturation 1.45:

- `luma_curve`: 15 points `{ "x", "y" }`, strength B of the richer look of 2026-10-19 (the first version's curve with its departure from the identity scaled by 1.6, slope about 1.40 at mid grey). It is applied as `v += T(L) - L` on all three channels, `L` the LogC4 code of the AWG4 luminance (the way the Curve tab's luma curve works, `docs/02` C11), through the monotone cubic of `ym_color::curve::Curve` (Fritsch-Carlson tangents). Mid grey (LogC4 0.2784, scene 0.18) stays where it is. Against Neutral it darkens the shadows (about -0.75 stop from -3 to -6 stops), lifts the lights (+0.36 stop at +1, +0.66 at +2, +0.83 at +3) and eases the highest lights back.
- `cdl`: slope 1, offset 0, power 1 and saturation 1.45 (Rec.709 luma weights on the LogC4 codes).

The constants were chosen by eye (twelve photos on 2026-10-14, three strengths on 49 photos on 2026-10-19) and nothing was fitted to any look file or LUT. The loader bakes the look into a 65-point cube like every parametric look (33 points until 2026-10-19), so the CPU and the GPU read it as an ordinary LUT. To change it, edit the points and the saturation here; `ym looks bake signature.json --preview <ARW>...` renders a contact sheet against Neutral.
