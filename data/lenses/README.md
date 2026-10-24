# Lens data

**Synthetic, computed from public lens designs, not measured.**

These files describe how seven lenses and one anamorphic attachment treat light, as the numbers and tables that `ym-lens` reads. Nothing here was measured: no lens, no photo and no test chart was used. Each lens comes from a prescription (radii, thicknesses, glasses) that Yellowmoon keyed in by hand from an expired patent. Real rays were traced through that prescription, scaled to the focal length of the lens it stands for, over the full 36 x 24 mm frame, and the results were fitted into the formulas and tables below.

The numbers are computed offline by a tool that is not part of the repository or the app, so they cannot be regenerated from the repository alone. Only the results are kept here.

## Files

| File | What it holds |
|---|---|
| `index.json` | the seven lenses in menu order and the name of the attachment file |
| `<id>/lens.json` | one lens: name, line, starting amounts, provenance, the field curves (spread-function width, relative illumination, distortion, swirl, cat's-eye), the bokeh rim profile, and the flare numbers (glow, veil, aperture, ghosts, starburst) |
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
- the swirl `sigma(rho)`: the shape is the traced tangential-to-radial spread of the pupil at the stop minus 1, multiplied by a gain per lens (default 0.12, classic 0.35, vintage 0.6, petzval 1, triplet 0.24, sonnar 0.45, detuned 0.35); without the gain the traced Default would swirl as much as the Petzval, because its small elements cut the pupil at the corner;
- the whole `flare.glow`, `flare.veil` and `flare.aperture` blocks and the starburst `fraction`, `radius` and `f_number_full` (the starburst radius is a deliberate exaggeration of how diffraction spikes read in overexposed photos; the table itself is the diffraction pattern of the aperture outline on a fixed extent of 18 lambda/D at 550 nm, scaled with wavelength);
- the starting amounts `start`;
- the roundness of each aperture outline, and the blade counts (round for the Petzval).

## Detuned

Detuned is the Default's prescription with its four element groups (the front element, the cemented pair before the stop, the cemented pair after it, the rear element) decentred and tilted by seeded random amounts (seed 22, largest decentre 0.32 mm, largest tilt 0.4 degrees; the values drawn are listed below). The field tables are the average over eight azimuths, because the table format is radially symmetric. The distortion, ghosts (with their edges softened by a factor of 1.6) and relative illumination follow the Default; the glow, veil and spikes are set by hand, and the spike table is the diffraction pattern of the aperture with a seeded random phase screen (rms 1.2 rad at 550 nm, correlation length 5 % of the pupil diameter), which gives its smeared flares.

| Group (surfaces) | decentre x, y (mm) | tilt about x, y (degrees) |
|---|---|---|
| [0, 1] | -0.111, +0.124 | +0.269, +0.167 |
| [2, 3, 4] | -0.307, +0.080 | +0.215, -0.290 |
| [6, 7, 8] | +0.222, +0.075 | -0.134, -0.108 |
| [9, 10] | -0.189, +0.027 | +0.073, +0.344 |

## Anamorphic attachment

Traced in front of the Default lens at f/2.4 with the patent's millimetre prescription scaled by 2, so its front element is about 96 mm across. The measured squeeze is 1.920; `bend` was fitted at that squeeze and stored per unit of `q - 1`. The apertures of the attachment are not in the patent: they are tuned (round clear apertures that pass 80 % of the possible beam at the frame corner). `kappa` is fitted to the extra falloff the attachment adds along the long side, against the Default's own at the same sensor point; `side_softness` to the growth of the spot size along the long side. Tuned: `kappa` (0.30) and `side_softness` (0.60), because the traced attachment is a well-corrected design that adds almost nothing along the long side (traced values 0.005 and 0.000); the streak's `fraction` (0.004; the traced sum over the 26 line-shaped ghosts is 0.00195), `reach` (1; traced 95 % radius 1.01), `core` (0.008) and `power` (1). The streak's core and power were 0.001 and 1.5 until data version 2 (a core about the traced line thickness): with a core of a pixel and a steep tail only 7 % of the streak's light lay beyond a tenth of a half-diagonal from the light, so the streak that cylinder-lens flares draw across the frame, a line that fades slowly and is still there at half the frame's width, could not be seen; with a core of 0.008 and a tail that falls as `1 / x` (cut over the last fifth of the reach) 45 % of it does, 23 % beyond 0.3 half-diagonals; a core of 0.008 half-diagonals and a tail that falls as `1 / x` carry a tenth of the light beyond a tenth of a half-diagonal; the oval ghosts' `s` (the geometric mean of the traced horizontal and vertical radii), `soft` (0.15) and their clip (none: `n` 0, `clip_radius` 4); `fringe` (1.0: the traced horizontal colour of the attachment, 35 um between 460 and 620 nm at the sensor edge, is far larger than the Default's own (0.6 um) and cannot be written as a multiple of it); and the assumed Abbe numbers.

## Licence

The prescriptions are facts of old designs, transcribed by hand from public patents and checked against the numbers the patents state. None comes from a lens database, a lens library or another project's files. The tables and numbers here are Yellowmoon's own computed results, under the repository's licence.
