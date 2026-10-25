# Lens data

**Synthetic, computed from public lens designs, not measured.**

These files describe how seven lenses and one anamorphic attachment treat light, as the numbers and tables that `ym-lens` reads. Nothing here was measured: no lens, no photo and no test chart was used. Each lens comes from a prescription (radii, thicknesses, glasses) that Yellowmoon keyed in by hand from an expired patent. Real rays were traced through that prescription, scaled to the focal length of the lens it stands for, over the full 36 x 24 mm frame, and the results were fitted into the formulas and tables below.

The numbers are computed offline by a tool that is not part of the repository or the app, so they cannot be regenerated from the repository alone. Only the results are kept here.

## Files

| File | What it holds |
|---|---|
| `index.json` | the seven lenses in menu order and the name of the attachment file |
| `<id>/lens.json` | one lens: name, line, starting amounts, provenance, the field curves (spread-function width, relative illumination, distortion, swirl, astigmatism, cat's-eye), the bokeh rim profile, and the flare numbers (glow, veil, aperture, ghosts, starburst) |
| `<id>/tables.safetensors` | `psf` `[9, 8, 33, 33]` (field radius, wavelength, y, x) and `spikes` `[8, 129, 129]`, f32 |
| `anamorphic.json` | the attachment: bend, oval field, side softness, fringe, ghost stretch, streak, oval ghosts |
| `SHA256SUMS` | SHA-256 of every other file, in `sha256sum` format |

All lengths are in half-diagonal units of the emulated frame (half-diagonal 21.633 mm, at the design's focal length). Spectra use the 31 wavelengths 400, 410, ... 700 nm. The eight spread-function wavelengths are 420, 460, 500, 540, 580, 620, 660 and 700 nm. Each table slice is a set of cell masses that sums to 1: cell `k` of `n` covers `[-w + 2wk/n, -w + 2w(k+1)/n]`, with x radial (outwards) and y tangential for `psf`, and x to the right and y down for `spikes`.

## The designs

| Id | Source | Year | Design type | Traced as | Focal length checked against the patent |
|---|---|---|---|---|---|
| default | US 2,532,751 (J. G. Baker, Perkin-Elmer), Example 3 | 1950 | double Gauss, six elements in four components | 50 mm, f/2.4 | stated: 1.000; computed: 1.0008 |
| classic | US 2,019,985 (H. W. Lee, Kapella Ltd) | 1935 | double Gauss with split rear component, seven elements | 50 mm, f/2 | stated: 1.000 inch; computed: 1.036 (see note 1) |
| vintage | US 1,786,916 (W. Merte, Carl Zeiss), Fig. 2 | 1930 | double Gauss, six elements, with a single biconvex rear element | 58 mm, f/2 | stated: 100; computed: 99.91 |
| petzval | US 2,500,046 (W. Schade, Eastman Kodak) | 1950 | Petzval portrait objective: cemented front doublet, air-spaced rear doublet | 85 mm, f/2.8 | stated: 100 mm; back focus 56.5 mm; computed: 99.98 mm; back focus 56.45 mm |
| triplet | US 568,052 (H. D. Taylor), Series I | 1896 | triplet, three single elements (positive, negative, positive) | 50 mm, f/4 | stated: 1.000; computed: 1.0031 |
| sonnar | US 2,084,309 (L. Bertele, Zeiss Ikon) | 1937 | triplet derivative with cemented groups: single front element, cemented triplet, cemented doublet, rear meniscus | 50 mm, f/2 | stated: not stated (the table scales to about 100); computed: 100.24 |
| detuned | the Default's patent, decentred and tilted (below) | 1950 | as Default | 50 mm, f/2.4 | as Default |
| anamorphic | US 3,002,427 (P. Schaefter and K. Kirchhoff, Isco Optische Werke) | 1961 | four-member cylindrical afocal front attachment | used in front of the Default; attachment scaled by 2 | stated focal lengths -64.27, +58.83, -28.36 mm and squeeze 1.925; computed -64.27, +58.83, -28.36 mm and squeeze 1.920 |

Notes on the designs:

1. Classic: the patent's radii are printed to two or three figures (for example 0.9 and 6.1 inch), which is enough to move the computed focal length by 3.6 % from the stated one inch. The data are scaled from the computed focal length, so the focal length of the lens it stands for is exact.
2. Default: the patent gives three examples, all f/2 and focal length 1.000; Example 3 is used. The lens is traced at f/2.4, about half a stop below its largest opening, because at f/2 its zonal spherical aberration and coma (RMS spot 16 um on the axis and 45 um at half field, against 7 and 10 um at f/2.4) do not fit "clean and modern".
3. Openings: Classic's patent is an f/1.4 design (it quotes its f/2 predecessor); it is traced stopped down to f/2. Vintage (an f/1.4 design as well) is traced stopped down to f/2, the Petzval at f/2.8 (its patent draws the aberration plot to f/2.0 and f/2.8), the Sonnar at f/2 (the patent says only "a great aperture").
4. Triplet: the patent gives finished diameters, used as the clear apertures; its diaphragm for f/4 (0.177 of the focal length across) is reproduced by the traced stop to 0.1%. The lens is designed to cover about 14 degrees; at the corner of a 50 mm frame (23 degrees) it passes almost no light (T 0.04) and its spread function there is wider than the runtime draws (see "Spread-function width"), so the numbers describe soft, dark corners. At rho = 1 the traced chief ray grazes total internal reflection, so the spread functions of that radius are those of rho = 0.875 made 20 % wider before the cut.
5. Sonnar: the patent describes a front lens, a cemented triplet, a cemented pair and a rear meniscus (a triplet derivative with cemented groups).
6. Glass: every patent gives the refractive index and the Abbe number at the d line (the Triplet's patent also gives the C-to-F dispersion). The index at other wavelengths is a three-term Cauchy formula through the C, d and F lines, with the F-line value taken from the normal partial-dispersion line `P(F,d) = 0.7183 - 0.00039 V`.
7. Attachment: the patent prints no Abbe numbers. They are assumed within the usual range of each index so that the attachment is afocal and achromatic to first order: T1 and T2 (n 1.52) V = 64, T3 (n 1.65) V = 41.8, the front lens of T4 (n 1.73) V = 26.0, its rear lens (n 1.56) V = 62.0. The patent's stated focal length of the doublet T4 (167.66 mm) is not reproduced (149.8 mm computed), while the afocal ratio and the other stated focal lengths are.

## Spread-function width

`field.psf_half_width` is the width that holds 99 % of the traced rays at every wavelength, but at most 0.012 half-diagonals, which is the widest spread function the runtime draws. Where the traced spread is wider, the `psf` table keeps only the central 0.012 half-diagonals (about 260 um on the sensor): the rays outside that square are dropped (their light is the faint coma and flare tail, which the Flares glow stands for) and each slice is normalised to 1 again. At the other radii the table leaves out about 1 % of the rays by the 99 % rule. The share of rays left out where the cap applied (mean over the eight wavelengths, worst wavelength in brackets):

| Lens | Field radius | `psf_half_width` | share left out |
|---|---|---|---|
| classic | 0.375 | 0.012 (traced 0.0174) | 2.8 % (3.6 %) |
| classic | 0.5 | 0.012 (traced 0.0234) | 6.8 % (7.2 %) |
| classic | 0.625 | 0.012 (traced 0.0292) | 10.4 % (11.9 %) |
| classic | 0.75 | 0.012 (traced 0.0294) | 14.2 % (16.7 %) |
| classic | 0.875 | 0.012 (traced 0.0255) | 14.2 % (17.2 %) |
| classic | 1 | 0.012 (traced 0.0255) | 14.9 % (18.8 %) |
| vintage | 0.5 | 0.012 (traced 0.0122) | 0.1 % (0.3 %) |
| vintage | 0.625 | 0.012 (traced 0.016) | 2.7 % (3.6 %) |
| vintage | 0.75 | 0.012 (traced 0.0158) | 3.0 % (3.9 %) |
| vintage | 0.875 | 0.012 (traced 0.0121) | 0.0 % (0.1 %) |
| vintage | 1 | 0.012 (traced 0.0129) | 0.4 % (0.9 %) |
| petzval | 1 | 0.012 (traced 0.0148) | 0.7 % (4.3 %) |
| triplet | 0.625 | 0.012 (traced 0.0181) | 1.9 % (5.4 %) |
| triplet | 0.75 | 0.012 (traced 0.0253) | 10.3 % (21.5 %) |
| triplet | 0.875 | 0.012 (traced 0.038) | 71.5 % (74.0 %) |
| triplet | 1 | 0.012 (traced 0.0465) | 77.1 % (79.1 %) |
| sonnar | 0.625 | 0.012 (traced 0.0276) | 3.8 % (4.2 %) |
| sonnar | 0.75 | 0.012 (traced 0.0406) | 8.7 % (9.1 %) |
| sonnar | 0.875 | 0.012 (traced 0.0378) | 10.9 % (11.5 %) |
| sonnar | 1 | 0.012 (traced 0.0227) | 2.8 % (3.5 %) |

## Coatings (assumed, an artistic choice)

- default: assumed broadband multi-layer coating, 0.3 % per surface at 545 nm, rising to about 0.8 % at both ends.
- classic: assumed single quarter-wave MgF2 layer centred on 450 nm (reflects more red and yellow).
- vintage: assumed single quarter-wave MgF2 layer centred on 550 nm.
- petzval: uncoated glass (Fresnel reflection only).
- triplet: uncoated glass (Fresnel reflection only).
- sonnar: assumed single quarter-wave MgF2 layer centred on 520 nm.
- detuned: as Default (assumed broadband multi-layer coating).
- anamorphic: single quarter-wave MgF2 layer centred on 640 nm on the cylinder surfaces (reflects more blue), so the streak is blue.

Ghost fractions are the product of the two surface reflectances at 550 nm and the share of the beam that survives the apertures; the 31-sample `coating` of a ghost is the product of the two reflectance spectra, scaled to 1 at 550 nm.

## What is traced and what is tuned

Traced from the prescription: the spread-function tables (real rays, one common best-focus plane per lens, diffraction at the lens's f-number added, the central 0.012 half-diagonals at most), relative illumination, distortion, the cat's-eye offset, the bokeh rim profile, and the ghost positions, sizes, clips, fractions and spectra.

Tuned by hand (an artistic choice, not computed):

- the clear apertures of every lens except the Triplet, set so that the elements pass the stated share of the geometrically possible beam at the frame corner (Default 75 %, Classic 62 %, Vintage 55 %, Sonnar 60 %, Petzval 45 %, Detuned 75 %), which fixes relative illumination T(rho);
- the swirl `sigma(rho)` and the astigmatism `a(rho)`. At Edges 100 the tangential gather's half-length (the half-length of its biweight line, `docs/21` section 4, L1, *As built (field swirl)*) is `k sigma(rho) b + a(rho) H`, with `b` the blur radius of the pixel, `H` the half-diagonal and `k` = sqrt(7/3): `sigma` is a number of blur radii (the half-length of the box line with the same spread), so a defocused light becomes an oval along the tangent of the aspect `sqrt(1 + 4 sigma^2 / 3)`, and `a` is a length that does not depend on the blur (the tangential and the sagittal focal surfaces part with the field radius, whatever the defocus), in half-diagonals, so that even a mildly blurred background swirls and a small light of the outer field becomes a lemon along its circle. The **Petzval's** are rules, not the trace, so that its swirl shows: `sigma(rho) = 2 t^1.3` and `a(rho) = 0.03 t^1.64` with `t = (rho - 0.2) / 0.8` from `rho` 0.2 outwards and 0 inside it (continuous with a continuous first derivative, rising monotonically), stored at the nine radii: `sigma` 0.28 at `rho` 0.375, 0.56 at 0.5, 1.23 at 0.75 and 2.0 at the corner (a disc becomes an oval 2.5 times as long as wide there); `a` 0.0025 at 0.375, 0.006 at 0.5 (5 px on a 1500 px frame, 22 px on 24 MP), 0.016 at 0.75 and 0.03 at the corner (27 px on a 1500 px frame, 108 px on 24 MP). The **Vintage's** are a third of the Petzval's (`sigma` 0.67 and `a` 0.01 at the corner). Every other lens keeps the traced shape of `sigma`, the traced tangential-to-radial spread of the pupil at the stop minus 1, multiplied by a gain per lens (default 0.12, classic 0.35, triplet 0.24, sonnar 0.45, detuned 0.35); without the gain the traced Default would swirl as much as the traced Petzval, because its small elements cut the pupil at the corner. (The traced Petzval reached 0.94 at the corner and 0.09 at `rho` 0.5, the traced Vintage with its gain of 0.6 0.54: too little to see.) Their `a` is 0: the tangential minus the radial rms spread of the traced spread functions at best focus (the mean over the eight wavelengths, in half-diagonals, at `rho` 0.5, 0.75 and 1) is Default -0.0001, +0.0003 and +0.0009, Classic -0.0012, -0.0026 and -0.0000, Vintage -0.0024, -0.0015 and +0.0009, Triplet -0.0008, -0.0043 and -0.0015, Sonnar -0.0004, -0.0000 and +0.0010, Detuned -0.0008, -0.0005 and +0.0006 (the Petzval's own +0.0004, +0.0006 and +0.0001), under 1 px on a 1500 px frame where positive (0.9 px at the largest) and mostly negative (the traced spreads are longer along the radius: coma), and the field optics (L2) already draw those spreads over the whole frame, defocused or not, so adding them to the bokeh gather would count them twice;
- the whole `flare.glow`, `flare.veil` and `flare.aperture` blocks and the starburst `fraction`, `radius` and `f_number_full` (the starburst radius is a deliberate exaggeration of how diffraction spikes read in overexposed photos; the table itself is the diffraction pattern of the aperture outline on a fixed extent of 18 lambda/D at 550 nm, scaled with wavelength);
- where the attachment puts the sides of the frame on the relative illumination curve: 0.93 of the tables' last radius (`ym_lens::consts::SIDE_TABLE_RADIUS`) and, at most, 0.6 of the way past it to where the end slope would reach 0 (`T_OVERSHOOT`), constants of the field law (*How the runtime reads the field curves*, below), not numbers in these files;
- the starting amounts `start`;
- the roundness of each aperture outline, and the blade counts (round for the Petzval);
- the share of the traced lateral colour that is drawn: 0.3 (`ym_lens::consts::LATERAL_COLOR_SCALE`, a constant applied when the spread functions are projected on the camera, not a number in these files). At load, each camera channel's table is moved so that its centroid's offset from the green channel's is 0.3 of the traced one. The traced offsets, widened to the spread the runtime draws, put red and blue 0.35 % (Vintage) to 0.39 % (Classic) of a half-diagonal apart at the sides of the frame at Edges 100 with Anamorphic 100 (3.1 to 3.5 px on a 1500 px frame, where a strong real lens shows 0.05 to 0.1 % at the corner); with 0.3 the worst lens (Classic) has 0.12 %, Vintage 0.10 %, the Default 0.01 %. One number for every lens, so each keeps its lateral colour relative to the others.

## How the runtime reads the field curves

`field.psf_half_width` and `field.relative_illumination` give nine values each. The runtime reads `psf_half_width` between the radii with the monotone cubic of Fritsch and Butland (continuous, with a continuous first derivative, and monotone wherever the values are) and, beyond `rho` = 1, where the attachment's oval field radius reaches over the outer part of the long axis (up to 1.3 times the half-width of the frame at Anamorphic 100), along a line with the cubic's end slope. It is made non-decreasing first, by pooling neighbouring values that fall; only Detuned has any (0.0081, 0.0074, 0.0076 at `rho` 0.5, 0.625 and 0.75 are read as 0.0077 each). The kernel's width, `omega = e' (1 + S) w`, is bent towards the widest spread the runtime draws (0.012 half-diagonals) from 0.009 on, not cut at it.

`relative_illumination` is read as a **concave** curve (the falloff of a flat field never bends back, so no band shows on snow, fog or sky): the nine values are pooled so that the slope between neighbours never rises (a sequence that already has that shape is used as it is; of the shipped lenses only the Triplet, whose last three intervals rise, changes, by at most 0.014, and the Petzval by 0.0003), joined by a curve of falling slope that passes through each value (within 0.002 of the monotone cubic for the others), and beyond `rho` = 1 carried straight on at its end slope. The gain reads it at the oval radius divided by a scale: 1 without the attachment, and with it the scale that puts the middle of the long edge of the frame at 0.93 of the tables' last radius (`SIDE_TABLE_RADIUS`, tuned: at 1.0 the sky of the darkest test photo kept 0.47 to 0.49 of its brightness at the outer 20 px of Vintage 100 / 100 / 100 against the 0.55 asked for) and keeps the frame's corner no farther past the last radius than 0.6 of the way to where the end slope would take `T` to 0 (`T_OVERSHOOT`, tuned: on a 3:2 frame only the Triplet, whose corner is nearly dark, reaches it). The data files themselves are unchanged.

## Flares as they are drawn

The files above hold the traced numbers. A few rules of the flare presentation are applied when the lenses are loaded, by `ym-lens` (`consts.rs`, `l3/items.rs`), so that the ghosts read as light and not as stickers; they change what is drawn, not what is stored:

- **Ghost coatings are pale.** The tint of a coating (the product of two reflectance spectra, projected on the camera's channels and scaled to a mean of 1) differs between the channels by a factor of 1.8 for Default and Detuned, 3.3 to 4.5 for Classic, 2.0 to 2.8 for Vintage, 1.2 to 2.1 for Sonnar and 1.9 to 4.5 for the attachment's ghosts (the spectra themselves span 6 to 26 times); Petzval and Triplet are uncoated and differ by 1.06. A ghost's tint is brought to at most 1.2 between the largest and the smallest channel (`GHOST_TINT_MAX_RATIO`) by raising it to a power below 1 and scaling it to a mean of 1: the hue and the order of the channels stay, so a violet coating makes a pale violet ghost and an orange one a pale amber. Every coating here has its reflectance minimum in the green, so left alone every ghost of a lens would be the same magenta: the coating of the lens's ghost `i` is moved along the wavelengths by `-70 + 190 frac(0.618034 (i + 1))` nm (`GHOST_COATING_SHIFT_NM`; surfaces are not coated alike and a ray meets a coating at an angle, which moves its reflectance minimum toward blue) before it is projected, so a lens's twelve ghosts are amber, magenta, blue and cyan in turn, each pale. The attachment's ghosts keep their coating.
- **A ghost's edge is a ramp, and its inside is not flat.** The `soft` of a ghost (some 0.03 for half of them: a hard disc) is at least 0.10 (`GHOST_SOFT_MIN`) when drawn: the 10 to 90 % width of the edge is then 26 % of the ghost's radius. The inside is shaded across by 35 % (`GHOST_SHADE`): brighter on the side that faces the optical axis, fainter on the other, with the same light in all.
- **A ghost has a brightest value.** Ghosts of one light never pass 0.004 of the clip level at Flares 100 (`GHOST_PEAK_MAX`; the same ghost of the cells of a large light counts together), and ghosts of apothem under 0.1 half-diagonals pass less, in proportion down to a quarter (`GHOST_FULL_RADIUS`, `GHOST_SMALL_SHARE`). A small clipped light is assumed thousands of times brighter than the clip level, and a ghost of a few pixels (Classic's 0.006, Sonnar's 0.016) came out many times the clip level.
- **Ghost fractions are the data's, at most twice.** A lens whose twelve fractions add up to less than 0.0008 (`GHOST_FRACTION_FLOOR`: Default and Detuned 1.2e-4, Vintage 3.4e-4, Sonnar 8.6e-4) has all of them scaled by `floor / sum`, at most 2 (`GHOST_GAIN_MAX`): the lens as if its coatings were somewhat worse, with the proportions between its ghosts as traced. Classic, Triplet and Petzval are as stored. The glow is raised to at least 0.02 and the starburst to 0.02 (`GLOW_FRACTION_FLOOR`, `STARBURST_FRACTION_FLOOR`).
- **The veil is a neutral haze.** The veil (`flare.veil`) is the light of the whole picture spread flat over the frame; its colour is the mean of its three channels (`HAZE_NEUTRAL`), not the picture's average colour. The Vintage's `flare.glow` and `flare.veil` are tuned (data version 3: veil 0.003, glow 0.03 with a reach of 0.25 half-diagonals, was 0.03, 0.05 and 1): with its power law of 2.1 a glow that reaches the whole frame sends more than half of its light farther than 0.1 half-diagonals, and the lens lifted the blacks of a bright photo by 20 to 60 codes at Flares 45 to 100.
- **Streaks.** `STREAK_*` constants (`docs/21`, section 4, L3, *Streaks*): a streak's peak saturates with the light's energy, is divided down for a wide light and is nothing for a surface (a light spread over more than 0.055 half-diagonals); lights on neighbouring rows are one streak; the strongest of a frame is drawn first and each after it at 0.75 of the one before, five at most and none under 0.03 clip levels; its reach grows with the logarithm of the light's energy; across the axis it is a Gaussian and a wider halo; its colour is the light's own at the light and the tail's (the streak tint at half its chroma, `STREAK_TAIL_CHROMA`) further out.

## Detuned

Detuned is the Default's prescription with its four element groups (the front element, the cemented pair before the stop, the cemented pair after it, the rear element) decentred and tilted by seeded random amounts (seed 22, largest decentre 0.32 mm, largest tilt 0.4 degrees; the values drawn are listed below). The field tables are the average over eight azimuths, because the table format is radially symmetric. The distortion, ghosts (with their edges softened by a factor of 1.6) and relative illumination follow the Default; the glow, veil and spikes are set by hand, and the spike table is the diffraction pattern of the aperture with a seeded random phase screen (rms 1.2 rad at 550 nm, correlation length 5 % of the pupil diameter), which gives its smeared flares.

| Group (surfaces) | decentre x, y (mm) | tilt about x, y (degrees) |
|---|---|---|
| [0, 1] | -0.111, +0.124 | +0.269, +0.167 |
| [2, 3, 4] | -0.307, +0.080 | +0.215, -0.290 |
| [6, 7, 8] | +0.222, +0.075 | -0.134, -0.108 |
| [9, 10] | -0.189, +0.027 | +0.073, +0.344 |

## Anamorphic attachment

Traced in front of the Default lens at f/2.4 with the patent's millimetre prescription scaled by 2, so its front element is about 96 mm across. The measured squeeze is 1.920; `bend` was fitted at that squeeze and stored per unit of `q - 1`. The apertures of the attachment are not in the patent: they are tuned (round clear apertures that pass 80 % of the possible beam at the frame corner). `kappa` is fitted to the extra falloff the attachment adds along the long side, against the Default's own at the same sensor point; `side_softness` to the growth of the spot size along the long side. Tuned: `kappa` (0.30) and `side_softness` (0.60), because the traced attachment is a well-corrected design that adds almost nothing along the long side (traced values 0.005 and 0.000); the streak's `fraction` (0.004; the traced sum over the 26 line-shaped ghosts is 0.00195: at Flares 100 and Anamorphic 100 it sets the peak of a streak, 100 times the fraction, 0.4 of the clip level, `STREAK_PEAK_PER_FRACTION`), `reach` (0.6; traced 95 % radius 1.01), `core` (0.04), `power` (1.5) and `tint` (the traced reflectance times the glass's transmission, below). The streak's core and power were 0.001 and 1.5 until data version 2, then 0.008 and 1, 0.05 and 1.2 in version 3 and are 0.04 and 1.5 since version 4, with the reach 0.6 (it was 1): a streak is drawn as a line with a peak (its brightness at the light, below the light's own) that falls along its length to nothing at its reach, with a zero slope, and the first review of the third round found the streak of the moon crossing 70 % of the frame as a band; with these numbers the brightest streak adds 2 codes or more over about half the frame width at most (Anamorphic 100) and its profile is 0.5 of its peak at 12 px and 0.14 at 100 px of a 1500 px frame; the light's energy sets the peak and the length of each light's streak (`docs/21`, section 4, L3, *As built (streaks as light)*). The tint was the traced coating (a single quarter-wave MgF2 layer centred on 640 nm, whose reflectance rises to 13 times its mean toward 380 nm) until version 3, which the camera reads as 0.51 : 0.74 : 1.75, a violet-blue (hue 295 degrees in CIELAB on the sRGB frame); the glass train absorbs the violet, so the stored tint is the traced reflectance times `0.02 + 0.98 smoothstep((nm - 400) / 60)`, which the camera reads as 0.53 : 0.87 : 1.60 and the tail (half its chroma) as 0.76 : 0.94 : 1.30, a blue (hue 262 degrees); the oval ghosts' `s` (the geometric mean of the traced horizontal and vertical radii), `soft` (0.15) and their clip (none: `n` 0, `clip_radius` 4); `fringe` (1.0: the traced horizontal colour of the attachment, 35 um between 460 and 620 nm at the sensor edge, is far larger than the Default's own (0.6 um) and cannot be written as a multiple of it); and the assumed Abbe numbers.

## Licence

The prescriptions are facts of old designs, transcribed by hand from public patents and checked against the numbers the patents state. None comes from a lens database, a lens library or another project's files. The tables and numbers here are Yellowmoon's own computed results, under the repository's licence.
