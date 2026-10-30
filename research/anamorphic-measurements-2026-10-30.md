# Real anamorphic lenses: squeeze at close focus is published, distortion numbers are not

Research date 2026-10-30. Scope: the owner's question of the same day, "Are there any measurements of squeeze and unsqueeze we can use to cover the gap?" The gap is that the anamorphic emulation of `docs/21-lens-character.md` (L1 oval bokeh with `q` up to `Q_max` 2, L4 bend `x' = x(1 + mu x^2)`, `y' = y(1 - beta x^2)`, `mu = mu0 (q - 1)`, `beta = beta0 (q - 1)`) rests on one patent design, US 3,002,427 (Isco, 1961, about 1.9x), traced by Yellowmoon (`data/lenses/anamorphic.json`: `mu0` -0.0768375, `beta0` 0.0702585; `data/lenses/README.md`), and not on measurements of real lenses. `research/lens-character-2026-10-23.md` had found no published measurements of distortion or bokeh shape for the lens sets it studied.

Method: makers' data sheets, VFX documentation, papers, patents, forum threads and reviews were searched and read. `WebFetch` could not resolve several hosts in this session, so pages and PDFs were fetched with `curl` through the session proxy and converted to text; the sizes were checked first and only small files were read. A few numbers below are computed here from those files and from the repo's JSON; they are marked "computed here". Anything that could not be opened is marked **(unverified)** and listed again in section 7.

---

## 0. Short answer

1. **Distortion.** No file of measured distortion of a real anamorphic lens with a licence that allows use was found. The makers who measure (Cooke, ZEISS) keep the per-lens data inside paid or gated tools. Data sheets give words ("low distortion", "straight lines stay straight") and no numbers. Two free collections of lens-grid maps for anamorphic lenses circulate (83 and 19 anamorphic sets), with no licence stated. One open-access paper (CC BY 4.0) prints calibrated numbers for a lab rig with a 1.34x attachment, which is not a cinema lens.
2. **Squeeze against focus distance.** One maker publishes it as numbers: ARRI's data sheets for nine 2x primes give the horizontal and vertical magnification at close focus. Their ratio is 1.97 to 2.01, so the squeeze of those lenses is within about 1.5 % of 2 at the nearest focus. Other statements say older lenses and adapters lose squeeze up close (a forum anecdote of 1.85x at 1.5 m on a Kowa, with a centre squeeze lower than the edges; "2.0 may really be 1.9 or 1.85" in SynthEyes' documentation). Cooke records a "dynamic squeeze ratio" per frame. A published optical model (anamorphic distance) explains the mechanism and has one measured value, for the lab rig.
3. **Bokeh shape.** A paraxial derivation (a forum post that works the paraxial model) gives front anamorphics ovals of 4:1 on the sensor and 2:1 after de-squeezing for a 2x lens, and circles after de-squeezing for a rear anamorphic. That matches `Q_max = 2`. No measurement of the ratio against distance or frame position was found.
4. **Mumps.** Described in several sources and explained as a squeeze that falls at close focus (early lenses) or is held at the subject and over-squeezed behind it (Panavision's fix). Wikipedia adds that the squeeze is not uniform across the field in any anamorphic system. No numeric model or measurement was found. The L4 bend is a centre-to-edge non-uniformity of the horizontal scale; the focus-dependent part is not in the model.
5. **Field of view.** On a 36 x 24 mm frame a 2x lens gives 3:1 after de-squeezing, so a 3:2 photo has half the width that such a lens would frame on that sensor. That matters for which lens-field position a photo edge stands for (section 3.4).
6. **Flares and streaks.** No measurement of streak length or colour for real anamorphics was found.

---

## 1. Distortion of real anamorphic lenses

### 1.1 What the makers publish

- **ARRI/ZEISS Master Anamorphic data sheets** (nine 2x primes, 28 to 180 mm, dated April 2016). Each sheet has the magnification ratio "at the close focus setting; horizontal (H) and vertical (V)", the angle of view on a Super 35 Cinemascope aperture, and the entrance pupil distance. Distortion appears only as claims: "Low distortion for uncompromised Cinemascope" with the sub-points "straight lines stay straight", "no distortion compensation in post necessary" and "low distortion even at close focus"; also "Virtually no image breathing" and "No anamorphic mumps (fat face effect)" ([data sheets, copy hosted by Lens Rentals](https://www.lensrentals.com/product-assets/78548f60-aafb-44d7-a3b7-56885aea23f3/master-anamorhic-data-sheets-data.pdf); [ARRI leaflet](https://www.Arri.com/resource/blob/178218/438cc2a1de8e350a92253e452107d790/master-anamorphic-product-leaflet-2015-data.pdf)). No distortion figure, curve or grid is in them.
- **Vantage Hawk V-Lite 1.3x** technical data holds focal lengths, close focus distances, angles of view and sizes, with no distortion or squeeze figures ([technical data](https://www.vantagefilm.com/file/edee/downloads/optics/1.3x-hawk-anamorphic-lenses/hawk-v-lite-13x-technical-data.pdf)). The product page names "barrel distortion, flares, and shallow depth of field" as desirable anamorphic qualities and gives no number ([Vantage](https://www.vantagefilm.com/en/products/hawk-anamorphic/hawk-1-3x)).
- **Sirui** (1.6x Saturn and Venus): the maker's own blog says "similar pincushion distortion strongest at 35mm" and no percentage; it also says Saturn "held a constant 1.6x squeeze from close focus to infinity, while Venus's squeeze shifted slightly at close focus", unquantified and promotional ([Sirui blog](https://store.sirui.com/blogs/guides/sirui-saturn-vs-venus-anamorphic-lens-comparison)).
- **Atlas, Laowa, Angenieux, Cooke Anamorphic/i, Kowa**: retailer and rental listings and press pages carry qualitative words ("very low distortion", "improved edge distortion control") and no percentages. No datasheet with distortion numbers was found for them (search results only, **(unverified)** beyond that).

### 1.2 What the makers measure and keep

- **Cooke Lens System for Nuke.** "Each lens is measured via Cooke patented technology to generate lens specific dynamic values", with "distortion removal and recovery ... at every frame", "full support for both spherical and anamorphic lenses", "dynamic squeeze ratio for anamorphic lenses" per frame, and "entrance pupil offsets and centre shifts" ([Cooke Digital](https://cookeoptics.com/cooke-digital/)). It is software delivered to selected productions ("The Lighthouse projects get access to the software"). No sample data are published.
- **Cooke /i3** metadata carries, according to a third-party interface document, "Anamorphic squeeze factor", "Shading data" and "Distortion data" ([Factory Optic](https://factoryoptic.netlify.app/docs/serialinterface/cookei)). Cooke's leaflet says "The /i3 shading and distortion data is also available from the Cooke Optics website" ([leaflet](https://cookeoptics.com/wp-content/uploads/2021/11/Cooke-Leaflet1-rawfiles.pdf)). A public per-lens file was not found; the Cooke "Part III" document on shading and spherical distortion data returned a 404 and is **(unverified)**.
- **ZEISS eXtended Data and CinCraft Mapper** give per-lens, per-setting distortion and shading maps for ZEISS lenses, as a paid service with a free trial budget (search results; the page itself, which was read, shows the CinCraft overview) ([ZEISS CinCraft](https://www.zeiss.com/photonics-and-optics/en/cinematography/cincraft/mapper.html)). Anamorphic lenses are not named on that page. **ZEISS CinCraft LensCore** (a Nuke plugin with ray-traced "digital lenses", sold by subscription) ships profiles of ZEISS cinema lenses; anamorphic support is not confirmed **(unverified, search results only)** ([ZEISS LensCore](https://www.zeiss.com/photonics-and-optics/en/cinematography/cincraft/lenscore.html)).
- **Cooke patent US 12,548,129 B2** (inventor Ian Sheret, Cooke Optics Ltd, priority 2 September 2021, granted 10 February 2026 per Google Patents, whose status line is not a legal conclusion) describes a distortion model for anamorphic lenses (the Cooke Lens System page does not say it uses this one): `x' = x + sum_{i=0..7} sum_{j=0..7} Dx_ij x^i y^j` and the same for `y'`, fitted from grids shot at several distances, plus an "anamorphic entrance pupil model" with three shift coefficients (`S_x3`, `S_y1`, `S_y3`). Claim 1 requires non-zero coefficients for `i + j = 2` "to characterise a decentre of the lens". The patent prints no example coefficients ([Google Patents](https://patents.google.com/patent/US12548129B2/en)).

### 1.3 Models in VFX tools (forms, no values)

- **3DEqualizer anamorphic models** (Science-D-Visions): "3DE4 Anamorphic Standard Degree 4" has the coefficients `Cx02, Cy02, Cx22, Cy22, Cx04, Cy04, Cx24, Cy24, Cx44, Cy44`, squeeze, and a lens rotation for "'human-touched' mounted anamorphic lenses"; a degree-6 model adds six more coefficients (`Cx06, Cy06, Cx26, Cy26, Cx46, Cy46`) ([Natron LensDistortion](https://natron-docs.readthedocs.io/en/latest/plugins/net.sf.openfx.LensDistortion.html)). Epic's Unreal documentation reads, in a search excerpt, `Cx02` as the X coefficient of the `r^2` term, `Cx22` of the `r^2 cos(2 phi)` term, `Cx04` of `r^4`, `Cx24` of `r^4 cos(2 phi)`, `Cx44` of `r^4 cos(4 phi)`, and refers to the model PDF in 3DEqualizer's plug-in kit for the exact equation **(unverified; page not opened)** ([Epic](https://dev.epicgames.com/documentation/unreal-engine/API/Plugins/CameraCalibrationCore/FAnamorphicDistortionParameters)).
- **Nuke LensDistortion** has a lens type "Anamorphic" with `anamorphicSqueeze` ("used to rescale the x coefficients"), `anamorphicTwist` ("lens twist, measured in degrees") and `anamorphicScale` ("Compensates for lens breathing, the slight changes in focal length when changing the focus") ([Foundry](https://learn.foundry.com/nuke/Content/reference_guide/transform_nodes/lensdistortion.html)).
- **SynthEyes** uses "a polynomial model widely used in the VFX industry, with nine separate parameters for X and nine for Y, plus five additional parameters", and says its set is "still only a small portion of what the actual complexity is like ... there can still be distortions that won't be modeled at all" ([models](https://borisfx.com/documentation/syntheyes/anamorphicsguide_files/the_anamorphic_lens_models/); [calibration](https://borisfx.com/documentation/syntheyes/cameracalibration_files/anamorphic_lenses/)). Its tip for a typical anamorphic start: "If the shot has barrel distortion across the top/bottom (bulging up/down), and pincushion distortion left/right (drawn in), set the eccentricity to -1.0 to start with."
- **OpenTrackIO / OpenLensIO** (SMPTE RIS OSVP) carries `anamorphicSqueeze`, defined as the "Nominal ratio of height to width of the image of an axis-aligned square captured by the camera sensor ... not however an exact number over the entire captured area due to a lens' intrinsic analog nature", and a Brown-Conrady distortion list for spherical distortion; it has no anamorphic distortion model ([OpenTrackIO](https://www.opentrackio.org/); licence of the spec text not read).
- **Lens Distortion Encoding System 1.0** (Fober, arXiv 2411.16946) builds synthetic anamorphic view maps from a fisheye-family projection with a squeeze factor `s`; it holds no measured lens ([arXiv](https://arxiv.org/abs/2411.16946)). Licence: CC BY-NC-ND 3.0.
- **Lensfun** has no anamorphic lens in its list (one spherical "Samyang 35mm T1.5 Cine" appears) and its models are radial ([lens list](https://lensfun.github.io/lenslist/)). Database CC BY-SA 3.0; the libraries are LGPL 3 and the applications GPL 3 ([README](https://raw.githubusercontent.com/lensfun/lensfun/master/README.md)).

### 1.4 Papers

- Chen, Zhang and Xi, "3D Metrology Using One Camera with Rotating Anamorphic Lenses", Sensors 22(21):8407, 2022, **CC BY 4.0** ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC9656777/)). The rig is a front attachment of three cylindrical elements on a 16 mm Computar lens with a 3.45 um pixel camera (2448 x 2448). Table 2 gives the calibrated intrinsics: `fx` 12.0520 mm, `fy` 16.1026 mm (squeeze `fy/fx` = 1.336, computed here), principal point (1279, 1002.3) px, anamorphic distance `ad` 26.5516 mm, anamorphic angle -0.5789 degrees, and distortion coefficients (`k1` 0.0304, `n1` 0.0183 and others for a 19-coefficient model with radial, third-order and second-order parts). The units of the image coordinates in the distortion equation are not stated in the text read, so the coefficient values cannot be read as strengths. The paper does not report distortion in percent.
- Chen and Zhang, "High-precision anamorphic lens calibration with 3D and 2D calibration targets", Applied Optics 61(20):6062, 2022: a polynomial model for anamorphic lenses; "Two anamorphic lenses are calibrated"; abstract only read ([PubMed](https://pubmed.ncbi.nlm.nih.gov/36255848/)). Optica copyright; full text not opened.
- Zhang et al., "Camera calibration for anamorphic lenses with three-dimensional targets", Applied Optics 59(2):324, 2020: the anamorphic imaging model with two extra intrinsic parameters, "the distance between the two optical centers is the anamorphic distance"; the calibrated distance and angle "are very stable for different positions of the calibration target"; abstract only read ([PubMed](https://pubmed.ncbi.nlm.nih.gov/32225310/)).
- Yuan and Sasian, "Aberrations of anamorphic optical systems. II", Applied Optics 48:2836, 2009: primary (third-order) aberration theory for cylindrical systems, with distortion among the aberration types; abstract level only **(unverified)**.
- Dodoc, "Anamorphic prime and zoom lenses", SPIE 11106, 2019, and Lippman, Teverovsky and Bentley, Optical Engineering 60(5):051203, 2021: design-side papers on power layouts and first-order design; no distortion measurements in the excerpts seen **(unverified)**.

### 1.5 Reviews

Neil Oseman's lens tests (Alexa XT, 4:3 mode) compare wide vintage and modern 2x lenses by eye: "A bulging effect is very obvious on all of these lenses, due to the focal lengths being quite wide for anamorphic", and the Cooke's corners "appear to be stretched away from the centre relative to the other two" ([Oseman](https://neiloseman.com/?p=9515)). ShareGrid's 2017 anamorphic test shot thirteen sets under one setup with a side-by-side viewer and no measured values ([No Film School](https://nofilmschool.com/2017/08/sharegrid-anamorphic-lens-test)). Searches for DXOMARK, Lensfun or CineD lab measurements of an anamorphic lens found nothing with numbers.

### 1.6 The source patent itself

US 3,002,427 states focal lengths and radii and no distortion value ([Google Patents](https://patents.google.com/patent/US3002427A/en)), so the traced `mu0` and `beta0` cannot be compared with a figure from the patent.

---

## 2. Table of sources

"Fact or file" says whether what would be used is a number that describes a lens (a fact) or a file/text with terms.

| Source | Lens, squeeze | What it gives | Numbers or form | Licence or terms | Fact or file |
|---|---|---|---|---|---|
| ARRI Master Anamorphic data sheets | nine 2x primes, 28 to 180 mm | H and V magnification at close focus; angle of view; entrance pupil; qualitative distortion | squeeze at close focus 1.970 to 2.013 (section 3.1); entrance pupil -188 mm (28 mm) | "© ARRI/2016", "Technical data subject to change without further notice"; no other terms printed | numbers are facts; the sheet is a copyrighted document |
| Hawk V-Lite 1.3x technical data | 1.3x primes | sizes and angles; no distortion | none for distortion | Vantage, no terms read | file |
| Cooke Lens System for Nuke | Cooke spherical and anamorphic | per-lens measured profile, dynamic squeeze, entrance pupil offsets | not public | proprietary software; Cooke terms | file, gated |
| Cooke /i3 metadata | Cooke /i3 lenses | squeeze factor, shading, distortion per lens and setting | not public | Cooke; "Part III" document marked confidential in search results **(unverified)** | gated |
| Cooke patent US 12,548,129 B2 | any anamorphic | model form: 8 x 8 bivariate polynomial plus entrance pupil shift | form only | active patent per Google Patents | form |
| ZEISS XD / CinCraft Mapper / LensCore | ZEISS lenses | distortion and shading maps; ray-traced digital lenses | maps | paid service or licence | gated; anamorphic not confirmed |
| Chen, Zhang, Xi 2022 (Sensors) | lab rig, 1.336x | calibrated intrinsics, anamorphic distance, 19-coefficient distortion | `ad` 26.55 mm; `fy/fx` 1.336; coefficients without stated units | CC BY 4.0 | text under CC BY; numbers are facts |
| Chen and Zhang 2022; Zhang et al. 2020 (Applied Optics) | lab rigs | anamorphic distance model; polynomial distortion model | abstracts only read | Optica copyright | facts only |
| 3DEqualizer, Nuke, SynthEyes, Epic models | any | forms and parameter names | no values | vendor documentation | form |
| OpenTrackIO | any | nominal squeeze field; spherical distortion list | the "nominal, not exact" definition | not read | form |
| LDES 1.0 (arXiv) | synthetic | anamorphic view maps from a projection model | none measured | CC BY-NC-ND 3.0 | file, not usable |
| Lensfun | none anamorphic | none | none | database CC BY-SA 3.0 | none |
| SynthEyes documentation | any | what squeeze does with focus; typical distortion pattern | "2.0 may really be 1.9 or 1.85"; "barrel top/bottom, drawn in left/right" | Boris FX documentation | text |
| Wikipedia, Anamorphic format | any | mumps, field of view, Panavision's fixes | "horizontal view of a 25mm spherical lens" for 50 mm at 2x | CC BY-SA (4.0 is the site's usual licence; not re-read here) | text |
| Oseman, ShareGrid, forums | various | qualitative comparisons, anecdotes | one anecdotal 1.85x at 1.5 m on a Kowa | not stated | text |
| Brian Caldwell post, EOSHD 2016 | front and rear anamorphic | paraxial bokeh ratio derivation | 4:1 on sensor, 2:1 de-squeezed (front 2x); 2:1 on sensor, 1:1 de-squeezed (rear 2x) | forum post | argument we can reproduce |
| US 10,126,533 | small-squeeze hybrid | design aimed at oval bokeh beyond the squeeze | none | patent text | form |
| US 3,002,427 | Isco 1961 | prescription; no distortion value | none | expired patent | prescription |

---

## 3. Squeeze and unsqueeze in practice

### 3.1 Does the effective squeeze change with focus distance, and by how much

**ARRI's close-focus numbers.** The sheets define the magnification ratio as "the size of an object on the film/sensor plane (first number) to the size of that object in real life (second number) at the close focus setting; horizontal (H) and vertical (V)". Ratio of the second numbers, H over V, is the squeeze at the nearest focus (computed here; the "range" allows for the printed figures being rounded to a tenth, plus or minus 0.05 on each):

| Nominal focal length | Close focus | H | V | Squeeze at close focus | Range from rounding |
|---|---|---|---|---|---|
| 28 mm | 0.65 m | 1:32.3 | 1:16.4 | 1.970 | 1.960 to 1.979 |
| 35 mm | 0.75 m | 1:32.3 | 1:16.1 | 2.006 | 1.997 to 2.016 |
| 40 mm | 0.70 m | 1:25.6 | 1:12.8 | 2.000 | 1.988 to 2.012 |
| 50 mm | 0.75 m | 1:22.2 | 1:11.1 | 2.000 | 1.987 to 2.014 |
| 60 mm | 0.90 m | 1:24.3 | 1:12.2 | 1.992 | 1.980 to 2.004 |
| 75 mm | 0.90 m | 1:19.6 | 1:9.8 | 2.000 | 1.985 to 2.015 |
| 100 mm | 0.95 m | 1:14.7 | 1:7.4 | 1.986 | 1.966 to 2.007 |
| 135 mm | 1.20 m | 1:15.6 | 1:7.8 | 2.000 | 1.981 to 2.019 |
| 180 mm | 1.50 m | 1:15.9 | 1:7.9 | 2.013 | 1.994 to 2.032 |

Eight of nine fall within rounding of 2.000; the 28 mm is about 1.5 % below. The figures describe the subject plane at the nearest focus, from the maker's sheet (whether from design or from measured units is not stated). They say nothing about out-of-focus planes or about positions in the frame.

**Other statements.**
- "The anamorphic element is likely not exactly its stated value, ie 2.0 may really be 1.9 or 1.85 and may vary with focus distance (an argument for secondary 'rangefinder' lenses)", and "a little experimentation does seem to show substantial differences in the anamorphic ratio compared to the specification, with a dependency on the focal distance" ([SynthEyes](https://borisfx.com/documentation/syntheyes/cameracalibration_files/anamorphic_lenses/)). No lens or distance is named.
- A reply to a Kowa B&H 2x owner's question (EOSHD, 24 September 2015): "above 10m can be treated as 2x. Closer focus will reduce the ratio. Closest 5f/1.5m is about 1.85x, IMO. Also, center part is less ratio then left/right edge" ([EOSHD](https://www.eoshd.com/comments/topic/9477-aspect-ratio-help/)). A reply in another thread says "at closer range, you loose the x2 squeeze ratio" ([EOSHD](https://www.eoshd.com/comments/topic/27172-mumps-with-the-kowa-16h8zbh/)). Anecdotes, no method, one lens.
- Cooke records a squeeze factor per frame in /i3 and a "dynamic squeeze ratio" in its Nuke system (section 1.2), which implies that the value moves with focus, and gives no values.
- Nuke's `anamorphicScale` exists "to compensate for lens breathing" ([Foundry](https://learn.foundry.com/nuke/Content/reference_guide/transform_nodes/lensdistortion.html)).
- OpenTrackIO's definition of squeeze calls it nominal and "not ... exact ... over the entire captured area" ([OpenTrackIO](https://www.opentrackio.org/)).
- ARRI says 2.0x is used on "90% of all anamorphic lenses out there" ([ARRI white paper](https://100-assets.arri.de/arri_assets/lf_assets/downloads/en/ALEXA_LF_&_Anamorphic_Lenses_-_White_Paper.pdf)); other squeezes in the same paper are 1.25x, 1.3x and 1.5x; Sirui's lenses are 1.6x ([Sirui blog](https://store.sirui.com/blogs/guides/sirui-saturn-vs-venus-anamorphic-lens-comparison)).

**A mechanism with one measured value.** In the anamorphic imaging model of Zhang et al. and Chen et al. the two axes have different optical centres a distance `ad` apart. Equation (1) of the Sensors paper has the horizontal image coordinate `fx X/Z` and the vertical one `fy Y/(Z + ad)`. A square of side `s` at distance `Z` therefore images with width proportional to `fx s/Z` and height proportional to `fy s/(Z + ad)`, and the height-to-width ratio (OpenTrackIO's definition of squeeze) is `(fy/fx) Z/(Z + ad)`. With the rig's `fy/fx` = 1.336 and `ad` = 26.55 mm (derived here from the paper's equation and Table 2, sign as in the paper):

| Object distance | Z/(Z + ad) | Effective squeeze |
|---|---|---|
| 0.3 m | 0.919 | 1.227 |
| 0.5 m | 0.950 | 1.269 |
| 1 m | 0.974 | 1.302 |
| 2 m | 0.987 | 1.319 |
| 5 m | 0.995 | 1.329 |
| 10 m | 0.997 | 1.333 |

This is one lab rig. ARRI's sheets list one entrance pupil number per lens (-188 mm for the 28 mm), not one per axis, so `ad` for cinema lenses is not published.

### 3.2 Mumps

- Description: "the anamorphic squeeze is not uniform across the image field in any anamorphic system ... faces look somewhat like they have the mumps" in the centre, "skinny-looking" at the edges, "Panavision was the first company to produce an anti-mumps system in the late 1950s" (a second lens linked to focus, later counter-rotating elements) ([Wikipedia](https://en.wikipedia.org/wiki/Anamorphic_format)).
- David Mullen, quoted in SynthEyes' guide: in early CinemaScope "as you focused closer and closer, the compression ratio changed, not squeezing the in-focus object enough (less than 2X)"; Panavision "solved this by making the compression error occur in what's OUT of focus ... so the out of focus background gets compressed MORE than 2X". SynthEyes' conclusion: "Mumps aren't fixed, they are simply hiding" ([SynthEyes optics and symptoms](https://borisfx.com/documentation/syntheyes/anamorphicsguide_files/optics_and_symptoms/)).
- Mechanism in the same guide: horizontal and vertical axes "may have different entrance pupil locations", so "the distance from any object to the nodal point(s) is different for each axis", and as the elements move for focus "the relative proportions and aspect change".
- ARRI states "No anamorphic mumps" for its set; the numbers in section 3.1 are consistent with a subject-plane squeeze held near 2.
- No model of mumps as a function of distance and frame position, and no measurement of it, was found beyond these.

### 3.3 Bokeh aspect ratio

- A paraxial derivation by Brian Caldwell (EOSHD, 1 July 2016): a front anamorphic "preserves the f/#" of the lens it is attached to in both axes. For a 2:1 attachment on a 100 mm f/2 lens (50 mm and 100 mm focal lengths, f/2 in both), "a de-focused point source evaluated at the image plane is an ellipse with an aspect ratio of 4:1", and after the 2x de-squeeze "you are left with de-focused ellipses with an aspect ratio of 2:1". A rear anamorphic does not preserve the f-number (50 mm f/2 becomes 100 mm f/4 in the powered axis), gives 2:1 on the sensor and a circle after de-squeezing ([EOSHD](https://www.eoshd.com/comments/topic/20171-ellipticaloval-bokeh-explained/)). The derivation treats defocus in object space, paraxially, with the two entrance pupils coincident; it is an argument, not a measurement. It agrees with the one-line version in Wikipedia ("vertical ovals rather than circles ... in proportion to the squeeze factor") and with the spec's `Q_max` = 2 for a 2x lens.
- US 10,126,533 B1 describes a small-squeeze design built to keep "elliptically-shaped defocused point images" and differential depth of field, which means the oval ratio is a design property and not always the squeeze ratio ([Google Patents](https://patents.google.com/patent/US10126533)). No oval ratio is published in the data sheets read for lenses of 1.3x to 1.8x.
- Against distance and frame position: the derivation gives a ratio independent of how far the point is from focus. With non-coincident pupils (`ad` not zero) a distance dependence is expected at second order; the field dependence from mechanical vignetting (cat's-eye clipping) is not quantified anywhere found. No measurement was found.

### 3.4 The field of a 2x lens on a 3:2 or 16:9 frame

- A 50 mm 2x lens has "the horizontal view of a 25mm spherical lens, while maintaining the vertical view ... of a 50mm" ([Wikipedia](https://en.wikipedia.org/wiki/Anamorphic_format)). ARRI's sheet for the 50 mm gives 48 degrees by 21 degrees on a 22.5 x 18.7 mm aperture; these angles correspond to focal lengths of 25.3 mm horizontally and 50.4 mm vertically (computed here), as the nominal number refers to the vertical focal length.
- De-squeezed aspect on common frames: 4:3 sensor mode with 2x gives 2.66:1 ([Oseman](https://neiloseman.com/?p=9515)); a 16:9 area with 2x gives 3.554:1 ([SynthEyes](https://borisfx.com/documentation/syntheyes/cameracalibration_files/anamorphic_lenses/)); 1.6x gives 2.8:1 on 16:9 and 2.4:1 on 3:2 ([Wikipedia](https://en.wikipedia.org/wiki/Anamorphic_format)); a 2880 x 2880 sensor area with 2x gives 2:1 ([ARRI white paper](https://100-assets.arri.de/arri_assets/lf_assets/downloads/en/ALEXA_LF_&_Anamorphic_Lenses_-_White_Paper.pdf)). A 36 x 24 mm frame with 2x gives 3:1 (computed here).
- A spherical photo is never squeezed, so it holds a narrower horizontal field than a 2x lens puts on the same width: a 50 mm spherical lens on a 36 mm wide frame sees a half-angle of 19.8 degrees, a 50 mm 2x lens 35.8 degrees on that width (computed here). The distortion of the cylindrical attachment grows with field angle, and a sensor position maps to a de-squeezed position twice as far from the centre. Which lens-field position the edge of a 3:2 photo stands for (the full squeezed frame, or the central part of a 3:1 frame) therefore changes the edge distortion strongly. In the L4 formulas `x` is in half-diagonals of the photo's own canvas, and the trace behind `mu0` and `beta0` ran "over the full 36 x 24 mm frame" (`data/lenses/README.md`). Computed here from `anamorphic.json` at `q` = 2, for a 36 mm wide frame:

| Frame | Photo edge stands for | Side edge drawn in | Local horizontal scale at the side edge (centre = 1) | Vertical scale lost at the side edge | Bow of the top edge, share of picture height |
|---|---|---|---|---|---|
| 3:2 | the whole traced frame | 5.3 % | 0.840 | 4.9 % | 2.43 % |
| 3:2 | the central half of the width of a wider frame | 1.3 % | 0.960 | 1.2 % | 0.61 % |
| 16:9 | the whole traced frame | 5.8 % | 0.825 | 5.3 % | 2.67 % |
| 16:9 | the central half of the width of a wider frame | 1.5 % | 0.956 | 1.3 % | 0.67 % |

(The second row of each pair applies the same `mu` and `beta` with `x` halved, which is the factor-of-four sensitivity of an `x^2` term; it is an illustration, not a model of the central crop.) For comparison, ARRI states "straight lines stay straight" for its set, and the Hawk page lists barrel distortion as a desired trait.

---

## 4. Flares and streaks

No measurement of streak length or colour for real anamorphic lenses was found. What exists is explanation: reflections between glass and air "can happen both before and after the anamorphic lens components", so the ones before are stretched and the ones after are not ([Wronski](https://bartwronski.com/2015/03/09/anamorphic-lens-flares-and-visual-effects/)); coating colour is described qualitatively by vendors and forum threads; Oseman's test shows a "horizontal line across most of the frame" for one lens and "doesn't flare much at all" for another, where the anamorphic element sits in the middle of the lens ([Oseman](https://neiloseman.com/?p=9515)). The flare dataset of Wu et al. (CC BY 4.0), cited in `research/lens-character-2026-10-23.md`, has its lab captures through a spherical smartphone lens, so it is not a source for streaks.

---

## 5. Mapping to the model

### 5.1 What each source could set or check

| Parameter | Source | How |
|---|---|---|
| `Q_max` (bokeh oval ratio) | Caldwell derivation; Wikipedia; ARRI white paper (2.0x on most lenses) | the de-squeezed oval ratio of a front anamorphic equals its squeeze (2 for 2x; 1.8, 1.6, 1.5, 1.3 for the others); rear designs give 1 |
| `q` against distance | ARRI close-focus ratios; SynthEyes; the `ad` model | one maker's modern 2x primes: constant within about 1.5 % at the subject; an adapter-era lens: 1.85 near 1.5 m (anecdote); the form `q Z/(Z + ad)` |
| `mu0`, `beta0` | grids of real lenses (a Cooke or ZEISS profile); no public numbers exist | fit `x' = x(1 + mu x^2)` and `y' = y(1 - beta x^2)` to the displacement field after taking out the round (spherical) part; the residual shows what two terms miss |
| signs | SynthEyes tip; Wikipedia; Oseman | edges drawn in horizontally and barrel across top and bottom is the usual pattern; `mu0` < 0 and `beta0` > 0 follow it |
| magnitude range | data sheets (words); Oseman | words only: "straight lines stay straight" at the low end, "a bulging effect ... very obvious" on wide vintage glass at the high end; the traced attachment gives a 2.4 % top-edge bow on a 3:2 frame at `q` = 2 (section 3.4) |
| the field position a photo edge stands for | section 3.4 | a choice, not measurable |

### 5.2 What the two-term model represents and what it leaves out

Represents:
- a horizontal scale that falls from the centre to the sides as `1 + 3 mu x^2` (centre wider, edges thinner, the direction of the mumps description);
- a vertical scale that falls with `|x|` as `1 - beta x^2` (horizontal lines bow, barrel across top and bottom);
- verticals that stay vertical (`x'` depends on `x` only), unevenly spaced.

If the `r^2` and `r^2 cos(2 phi)` reading of the 3DEqualizer coefficients is right, `x' = x(1 + mu x^2)` is the case `Cx02 = Cx22 = mu/2` and `y' = y(1 - beta x^2)` the case `Cy02 = Cy22 = -beta/2` (derived here from the excerpt in section 1.3, **unverified** against the plug-in kit PDF).

Leaves out:
- bowing of vertical lines (a `y^2` term in `x'`, the difference `Cx02 - Cx22` above), which SynthEyes' "pincushion left/right" wording and the 3DE degree-4 model allow;
- terms of order `r^4` and higher, and so a mustache shape (a sign change between the `r^2` and `r^4` terms). The lens's own radial `delta1, delta2` terms are round and cannot make an axis-dependent mustache. Degree 4 is the working standard in the tools above, Cooke fits degree 7, and the Applied Optics model has second, third and radial orders;
- decentring, axis rotation ("twist"; "the rotation angle is very setup-dependent") and tilt;
- dependence on focus and distance (`q` against distance, breathing scale, entrance pupil shift);
- different strength per axis independent of `q` (the 3DE and SynthEyes models have separate X and Y coefficient sets; the lab rig's `k1` and `n1` differ).

---

## 6. What this means for Yellowmoon

**Data that could check or replace the traced numbers.**

1. **ARRI's close-focus magnifications** check `q` against distance and support holding `q` constant for a modern front anamorphic. They do not touch `mu0` or `beta0`. Licence question for the owner: the numbers are facts printed in a document marked "© ARRI/2016"; whether citing or storing derived ratios sits inside `AGENTS.md` rule 2 (it names the Look Library and display LUTs, not data sheets) and how any table would name lenses under rule 3 is the owner's call.
2. **The Chen, Zhang and Xi paper (CC BY 4.0)** is licensed for reuse with attribution: `ad` = 26.55 mm and squeeze 1.336 for one rig give the form and size of the distance dependence. It is a lab attachment and not a cinema lens, and its distortion coefficients lack stated units. Question: whether a metrology rig is acceptable as a reference for `q` against distance.
3. **Cooke and ZEISS per-lens data** would be the best source and are proprietary and gated. Licence questions: terms of Cooke's Nuke system and ZEISS's services; a written permission or a data agreement would be needed for any number to enter the repository; and Cooke's US 12,548,129 B2 (active per Google Patents; claim 1 requires degree-2 decentre coefficients, which L4 has none of) would be part of any decision to move to a general polynomial model with cross terms. Not a legal reading.
4. **More expired patents** (the ones `research/lens-character-2026-10-23.md` lists: US 2,832,262, US 3,217,597, US 6,310,731, US 4,017,160) traced the same way would show whether the Isco attachment's `mu0` and `beta0` are typical of designs of different squeeze. They are design data and not measurements, and they stay inside the V1 data rule. The Isco patent itself prints no distortion value to check against.
6. **Formulas and facts from the models and Wikipedia** (3DEqualizer, Nuke, SynthEyes, OpenTrackIO forms; the mumps description; the field-of-view arithmetic) are forms and arithmetic that carry no file terms of their own (the vendor pages themselves are copyrighted); Wikipedia text is CC BY-SA if it were ever copied rather than restated; LDES is CC BY-NC-ND and not usable.

**What the available data say about the current numbers.** The sign pattern of the traced bend (sides drawn in, barrel across top and bottom) matches the pattern the VFX documentation gives as typical. The magnitude is unchecked: at `q` = 2 on a 3:2 frame the trace implies a 5.3 % draw-in of the side edge and a 2.4 % bow of the top edge (section 3.4). ARRI's "straight lines stay straight" carries no figure and Oseman's "bulging" is a visual remark, so neither gives a number to compare these with. `Q_max` = 2 matches the geometry of a 2x front anamorphic.

**Open design question the data cannot settle.** Which lens-field position the edge of a spherical photo stands for (section 3.4) changes the edge distortion by a factor of about four for the same `mu` and `beta`; measurements of a lens give distortion against position in the squeezed frame and against field angle, and do not say which of these the emulation should map the photo onto.

**Unmeasured anywhere found.**
- Numeric distortion of any named cinema anamorphic lens under a usable licence;
- how distortion scales with the squeeze (the `(q - 1)` factor in `mu` and `beta` is a design choice with no published support found);
- squeeze against focus distance beyond the nearest-focus ratio of one maker's set, and the shape of the curve between near and far;
- squeeze non-uniformity across the field in numbers (the size of the centre-to-edge mumps);
- bokeh oval ratio against distance and position in the frame;
- streak length and colour;
- the per-axis entrance pupil separation `ad` of any cinema lens.

---

## 7. Not opened, or opened only partly

- Epic's `FAnamorphicDistortionParameters` page and the 3DEqualizer plug-in kit model PDF (exact anamorphic equations): not opened; the reading of the coefficient names is from a search excerpt.
- Optica full texts (Applied Optics 61:6062, 59:324, 48:2836): abstracts only. The Optica host did not resolve in this session.
- Cooke /i3 "Part III" PDF: the link returned a 404; its description is from search results.
- The ASC article "Practical Optics 3: Introduction to Anamorphic" returned 403 and was not read; US 12,276,780 ("Anamorphic photography and squeeze ratios for digital imagers") returned a CAPTCHA and was not retried.
- Dodoc 2019 and Lippman 2021: search results only.
- ZEISS LensCore lens list and anamorphic support: search results only.
- OpenTrackIO licence text: not read.
- Which of the ARRI close-focus figures come from design values and which from measured units: not stated in the sheets.
