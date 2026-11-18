# Colour compressor: what Resolve's tool does, and what comparable tools do (2026-11-18)

Background for `docs/32-color-compressor.md`. The owner's request, 2026-11-18: "plan 'color compressor' node, similar to Resolve color compressor, that will also go into 0.2.4". This is research text: it describes the world and holds no instructions (`AGENTS.md`). **We copy no code and no proprietary formula.** The maths in `docs/32` is our own, designed from the behaviour described here. Where a source is unclear this file says so.

## 1. DaVinci Resolve: Color Compressor (ResolveFX Color)

**Source.** DaVinci Resolve 18 Reference Manual, chapter "ResolveFX Color" (the page 3348 of the PDF's own numbering), section "Color Compressor (Studio Version Only)". Read from a community mirror of Blackmagic's PDF: <https://ltbits.github.io/davinci-resolve-manuals/DR18/DR18-RM12-ResolveFXOverview.pdf> (the text was extracted with `pdftotext`; the same text is on Blackmagic's own manual pages, which the search tool did not return). The manual's HTML copy is also at <https://www.steakunderwater.com/VFXPedia/__man/Resolve18-6/DaVinciResolve18_Manual_files/part3167.htm> (listed in the search results; its text was not read here).

**What the manual says (quoted from the PDF text, shortened only where marked).**

- "This filter lets you compress a range of colors toward a single target color. It works best when applied to similarly colored regions that have been isolated with a secondary qualifier or window. For example, if you're adjusting an irregularly lit product such as a soda can or a dress that must have a very specific hue, you can isolate that item and use this filter to push the range of Hue, Saturation, and Luminance closer to the ideal color that's identified using the color picker."
- **Target Color**: "A color control with an eyedropper button that you choose or sample a single set of Hue/Saturation/Luminance values you want to push all other colors toward."
- **Compress Hue, Compress Saturation, and Compress Luminance**: "These sliders let you individually compress the range of colors you're adjusting in the image to more and more closely match the image [sic: the target]. At 0, no compression is applied, at 0.500 the original range of colors in the image have been adjusted to be halfway between their original values and the values of the Target Color, and at 1.000 the original range of colors has been set to be identical to the Target Color."
- The figure caption: "(Before/After) Using the Color Compressor plugin to compress original range of hues of the jacket into a narrow 'target' range within a secondary operation."
- Every ResolveFX filter also has a **Blend** slider (the Resolve 16 manual, "The Blend slider found in the parameters of every ResolveFX filter"; <https://ltbits.github.io/davinci-resolve-manuals/DR16/DR16-RM-11–ResolveFX.pdf>). It is a general mix with the unprocessed picture, not part of this tool.
- The filter is **Studio only**.

**What that gives us, and what it leaves open.**

| Question | What the manual says | Unclear |
|---|---|---|
| Controls | One colour control (Target Color, with an eyedropper) and three sliders (Compress Hue, Saturation, Luminance), plus Blend | The slider range is "0 … 1.000" in the text; the interface's step and default are not given |
| How the target is chosen | An eyedropper on the picture, or the colour control, as H/S/L values | Whether the eyedropper reads one pixel or an average; in which space |
| Range and softness of the selection | **None.** The filter has no range, softness or selection of its own. It acts on whatever reaches it, so the colourist isolates the region first with a qualifier or window (a "secondary operation") | Nothing: this is explicit, and it is the main difference from Capture One |
| What "compress" does | A linear pull on each of H, S and L toward the target's value: 0 nothing, 0.5 halfway, 1 identical to the target | Whether hue is pulled by the shortest way round the wheel (it must be, but the manual does not say); the colour model the H, S and L belong to (Resolve's HSL, in the timeline's colour space, is the likely reading); how pixels near grey, where hue is undefined, behave |
| Display or mask output | None described | Whether the plug-in has an overlay: none is mentioned |
| Typical use | Irregularly lit products, a dress that must have one hue | Skin, foliage and sky are not mentioned in the manual |

**Other sources.** Search found no Blackmagic forum thread or tutorial that explains the tool beyond the manual; the pages returned were about the HSL qualifier, Color Warper and curves in general, not about this filter. One of them (Frame.io, <https://blog.frame.io/2024/05/15/davinci-resolve-qualifier-color-spaces/>) notes that saturation in logarithmic spaces is compressed, which makes HSL qualifiers harder to use on log footage. That is general background for working in a log space, not a statement about this filter. A paid third-party DCTL, "QT Color Compressor" (<https://qazistoolkit.squarespace.com/qt-color-compressor>), is sold as a tool that pulls nearby colours toward a target hue and saturation with compress controls at 1.0; that is a vendor's description and says nothing about Resolve's maths. **Skin, foliage and sky as uses are therefore our reading of the tool's purpose, not Blackmagic's statement.** Colourists use skin uniformity as the standard case for a qualifier plus a pull toward one colour; that is general practice (and what Capture One's Skin Tone tool, below, is built for), not something these sources show for the Resolve filter.

## 2. Capture One: Skin Tone (Color Editor)

**Sources.** Capture One's own pages (the support article <https://support.captureone.com/hc/en-us/articles/360002596077> and the blog <https://www.captureone.com/blog/achieving-perfect-skin-tones-using-capture-one>; the support article returned HTTP 403 to the fetch tool, so its text was seen only through a search summary and the blog's text); a trainer's page, Mastin Labs, <https://mastinlabs.com/blogs/photoism/how-to-edit-skin-tone-in-capture-one>; the Fstoppers article <https://fstoppers.com/capture-one/achieving-perfect-skin-tones-and-color-using-capture-one-68502>. The blog covers Capture One 11, "an older version".

**What they say.**

- The tool "has two groups of sliders; Amount and Uniformity" (blog). **Amount** has Hue, Saturation and Lightness sliders that "selectively change the skin tones to a preferred look" (blog); Hue shifts skin toward green or red (Mastin Labs). **Uniformity** has Hue, Saturation and Lightness sliders that "determine how 'blended' the adjusted values become. The further to the right you go, the more uniform the blend" (Mastin Labs); "The Uniformity tools are used to even out different variables" (blog). The three Uniformity sliders replaced one in Capture One 9 (a trainer's remark, unverified).
- **Selection.** The person picks a skin colour with an eyedropper, "in the midtones" (Mastin Labs); a colour wheel then shows the selection, which can be adjusted by hand. A **Smoothness** slider "dictates how far the selection gradient extends beyond the specific color selected" (Mastin Labs). "View selected color range" shows the selection with everything else in black and white (Mastin Labs). A local mask can keep the effect off other colours (blog).
- **Behaviour.** Dragging Uniformity to the right pulls the selected range's hue, saturation or lightness toward the picked colour (the search summary of Capture One's support text; unverified because the page was not readable). The ranges' maths, the space and the maximum are not published. Practitioners warn that Lightness Uniformity looks artificial when pushed (a trainer quoted in the search summary).

**What it gives us.** Capture One is the model for a photographer's tool: it has its **own selection** (picked colour plus Smoothness) and a **view of the selection**, where Resolve has neither because Resolve has qualifiers. Its Uniformity is the same idea as Resolve's Compress: a 0 to 100 pull toward the picked colour, per axis. Its Amount sliders (a shift of the target) have no counterpart in Resolve's tool.

## 3. Others, briefly

- **Lightroom Classic and Camera Raw.** The Color Mixer / HSL panel shifts hue, saturation and luminance of eight colour bands by an offset; it does not pull colours toward one value, so it cannot make colours alike (general knowledge of the tool; no source fetched). Camera Raw's point colour and range mask (2022 and later) can sample a colour and adjust hue, saturation and luminance of a range, with a Range slider and a Show Color Range View; its Uniformity-like behaviour is not part of it (from memory; not verified here).
- **Photoshop.** Hue/Saturation with an eyedropper-set range (four-handle range with falloff) and Selective Color shift colours by amounts; neither compresses. Not sourced here.
- **Baselight and others.** Not researched: nothing was read about a Baselight or other equivalent, and none is claimed.

## 4. What this means for Yellowmoon (the design input)

1. The **law** is simple and the same in both tools: on each of hue, saturation and lightness, move a colour toward the target's value by a share, 0 to 100 %. Our own maths: a convex pull with a smooth, monotone weight (`docs/32` §3).
2. Resolve has no selection because it has qualifiers and windows; Yellowmoon has **masks, including Colour range**, in every corrector. A pull whose strength is a smooth selection weight W is the same as a pull applied through a mask of the same shape (`x + W·κ·(t − x)`). So a **built-in Range** (hue, saturation, lightness, softness, with Show affected area as Capture One's "View selected color range") is the Resolve workflow with the qualifier folded in, and nothing is lost when a person adds a mask on top.
3. A photographer will start from a click on skin, so the target is **picked on the photo** (Resolve and Capture One both have an eyedropper) and can then be **edited as numbers** (Resolve's colour control shows H, S and L; Capture One's wheel and sliders).
4. Open in the sources, so decided by us and written as decisions in `docs/32` §10: the colour space of the three axes; the treatment of grey (hue undefined); an average rather than one pixel for the pick; the Resolve "Blend" slider, which our node does not need (masks and the amounts cover it).
