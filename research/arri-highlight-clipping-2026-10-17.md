# ALEXA highlight clipping on Sony raw files: what ARRI publishes, what the Sony has, what we can build

For Mykyta, 2026-10-17. Research only. Written by a research agent; no other file in the repository was changed and nothing was committed. Numbers marked *computed* are my own arithmetic on the cited formulas (short Python scripts in the session scratchpad, not committed). The ACES 2 display codes come from the ACES 2 SDR 100-nit tonescale, re-implemented for this note and checked against `docs/02` (scene 0.18 gives display 0.1000; placed grey gives 0.1893, sRGB code 120.4).

## 0. Short answer

1. **The ALEXA's sensor clip is published exactly.** ARRI gives it per EI in three places: the LogC3 table of "clipping level" code values, the LogC3 formula, and ARRI's own ACES input transform for ARRIRAW [S1, S4]. For ALEV III at EI 800 the sensor clips 7.64 stops above 18 % grey (computed from the formula). ARRI's data sheets say 7.4 [S7, S8]. For the ALEXA 35 the clip is 9.35 stops above grey at EI 800 [S2, S3].
2. **The ALEXA clip is a hard clip of a linear signal.** ARRI's formulas carry the sensor signal linearly up to 1.0 and stop there [S1, S2, S4]. No sensor roll-off is published, so there is none to copy. ARRI's only "shoulders" are an invertible encoding shoulder in LogC3 above EI 1600 [S1, S4] and the display rendering [S5, S6].
3. **A Sony file always clips first.** At the camera's own exposure (our +1.11 EV gain), a Sony body clips 3.58 stops above grey on all 12 local photos (`docs/02`). DXOMARK's saturation-based ISO gives 3.40 to 3.54 stops for 13 of 14 bodies, and 3.68 for the A7S III [S19, S20, S21]. That is 4.05 stops less than an ALEXA at EI 800 and 1.7 stops less than at EI 160, ARRI's lowest EI. In ALEXA terms, a Sony photo at Auto behaves like an ALEXA at about EI 50.
4. **So "apply the ALEXA's clip" changes nothing.** Below the Sony clip, Yellowmoon already holds the values that an ALEXA exposed for the same grey would record. The ALEXA's clip sits 4 stops above anything the Sony recorded. It only comes into play when Exposure is pushed by about +4 EV.
5. **What an ALEXA shows and a Sony cannot:** highlights that keep separating above +3.6 stops and approach white smoothly. Through our ACES 2 chain (no look, no lift), the Sony clip lands at sRGB code 229 and an ALEXA EI 800 clip at code 254. That 25-code band is what is missing. Data cannot fill it; only a choice of ours can.
6. **Why the Highlights slider does nothing.** The C7 shoulder pushes values down towards 1.5 × clip, and its knee is at most 2.5 stops below the clip. On a neutral, it lowers the clip by 0.02 to 0.14 stop and moves nothing below +2 stops by more than 0.003 stop. Through ACES 2 that is at most 2 codes. It is also not something an ALEXA does.
7. **ARRI's display rendering publishes one usable rule.** The ALEXA's in-camera video output is "normalized": the sensor clip goes to 100 % video at every EI [S5, S6]. The shapes of ARRI's shoulders (K1S1, REVEAL) exist only as LUTs, which we may not use.
8. **Recommendation.**
   - Retire the C7 shoulder (P1).
   - Replace it with a "highlight range" stage (P2). It maps the Sony's recorded range onto the ALEXA's: no change up to a knee, the Sony clip placed on the ALEXA EI 800 clip (ARRI's number), and a hard cap at that clip.
   - Its one control runs from ARRI's "photometric" conversion (0: today's picture) to ARRI's "normalized" conversion (1: the clip at white).
   - Choose its constants by testing on public raw files that we clip synthetically, so the unclipped file is the truth (P4), then by a blind A/B.
9. **What public data cannot give us:**
   - how ARRI treats colour channels at and near the clip;
   - the order in which the ALEXA's channels clip under each light;
   - the shape of ARRI's highlight roll-off.

   These are unpublished, inside the closed Image SDK, or in all-rights-reserved LUTs. The EULA of ARRI's Reference Tool (ART) forbids reverse engineering.

## 1. How an ALEXA clips and renders highlights

### 1.1 The sensor clip per EI (ALEV III)

Published facts:

- LogC3 is a set of curves, one per EI. Each curve maps 18 % grey to LogC 0.391 (400/1023). "The maximum value of the Log C curve depends on the EI value" [S1].
- ARRI lists the LogC3 value for "sensor saturation (a sensor signal of 1.0)": EI 160 0.8128, 200 0.8341, 250 0.8549, 320 0.8773, 400 0.8968, 500 0.9158, 640 0.9362, 800 0.9539, 1000 0.9711, 1280 0.9895, 1600 and above 1.0000 [S1].
- The 2011 white paper says: "With an exposure of EI 400, for example, the ALEXA has headroom for 6 ½ stops. Exposing at EI 800 increases the headroom by one stop", and "At EI 400, the maximum code value is 917/1023 and at EI 800, the maximum is 976" [S5].
- ARRI's ACES input transform for ARRIRAW was written by ARRI's `v3_IDT_maker.py` v0.09 on 22 December 2016. It reads `black = 256.0 / 65535.0`, `exp_factor = 0.18 / (0.01 * (400.0/EI))` and `r_lin = (rIn - black) * exp_factor`, on "white-balanced, black-subtracted linear values", with no clamp [S4]. So grey sits 0.01 × 400/EI above black on the 16-bit sensor scale.
- Data sheet (ALEXA Plus): "EI 160 (+5.0 / -9.0), EI 200 (+5.3 / -8.7), EI 400 (+6.3 / -7.7), EI 800 (+7.4 / -6.6), EI 1600 (+8.4 / -5.6), EI3200 (+9.4 / -4.6) Values behind the exposure index are the number of stops above and below 18% grey. These values are for Log C." It also gives "14 stops for all sensitivity settings from EI 160 to EI 3200, as measured with the ARRI Dynamic Range Test Chart (DRTC)" [S8]. The VFX FAQ repeats 7.4 stops above and 6.6 below at EI 800 [S7].

Computed from [S1] and [S4]: scene-linear clip (grey = 0.18) = (1 − 256/65535) × 0.045 × EI, and stops above grey = log2(0.996 × EI / 4).

| EI | data sheet, stops above grey [S8] | formula, stops above grey | scene-linear at clip | LogC3 at clip [S1] | LogC4 of that level (computed) |
|---|---|---|---|---|---|
| 160 | 5.0 | 5.32 | 7.17 | 0.8128 | 0.609 |
| 200 | 5.3 | 5.64 | 8.96 | 0.8341 | 0.630 |
| 400 | 6.3 | 6.64 | 17.9 | 0.8968 | 0.695 |
| 800 | 7.4 | 7.64 | 35.9 | 0.9539 | 0.760 |
| 1600 | 8.4 | 8.64 | 71.7 | 1.0 (encoding shoulder) | 0.824 |
| 3200 | 9.4 | 9.64 | 143 | 1.0 (encoding shoulder) | 0.889 |

- Decoding ARRI's clip table with ARRI's own exposure-value parameters gives the same levels to 0.01 stop at EI 160 to 800. At EI 1600 the compact formula gives 8.56 stops; the document warns that it is approximate there [S1].
- The data sheet sits 0.24 to 0.34 stop below the formula at every EI. ARRI does not say why. Possible reasons are the DRTC measurement or a safety margin; I found no statement either way.
- An independent test sides with the formula. CineD found "about 7.8 stops" between 18 % grey and the point where a face clipped, on a Mini LF in ARRIRAW at EI 800 [S16].

### 1.2 ALEXA 35 (ALEV 4), for contrast

Published facts:

- The LogC4 hardware curve is `E' = min((log2(a_h · E_sensor + 64) − 6) / 14 · b + c, 1.0)` with `a_h = (2^18 − 16) · EI / 800` [S2].
- "117.45 = 0.18/(400/260991)" [S2]: at EI 800, grey sits at 400 of a maximum sensor signal of 260991.
- In the false-colour specification, the red zone ("1/3 stop below clipping") starts at E_sensor 207148.69, which is 260991 / 2^(1/3). The grey band at EI 400 is 720 to 880 [S3].
- The same document says: "For HEI ≥ 3200, the top two zone's bounds (Red, Yellow) are held at a constant EI 3200 … This results in the visualization of earlier signal clipping than what is present in the sensor linear values or RAW file" [S3].
- The VFX FAQ gives 9.3 stops above grey and 7.7 below at EI 800 [S7].
- The product page claims 17 stops, "1.5 stops in the highlights and a stop in the shadows over previous ALEXA cameras", and a "naturalistic, film-like highlight roll-off" [S9].

Computed: the clip is 117.45 × EI/800 in scene-linear. That is 7.03 stops above grey at EI 160, 9.35 at EI 800, 10.35 at EI 1600 and 11.35 at EI 3200 (LogC4 1.0). At EI 6400 LogC4 stops at 11.35 while the raw holds 12.35. In LogC4 code, the EI 800 clip is 0.870. The difference from ALEV III at EI 800 is 9.35 − 7.64 = 1.7 stops; ARRI says 1.5.

### 1.3 Linear up to a hard clip

- ARRI's LogC3 document normalises the sensor signal from 0 to 1 and calls 1.0 "sensor saturation". Its curve is linear in the sensor signal [S1].
- ARRI's input transform is linear up to 1.0 and has no clamp and no shoulder in scene-linear [S4].
- The LogC4 hardware curve is a log of the linear sensor signal with a hard limit at 1.0 [S2].
- "After an exposed image is read from the sensor, the camera applies a white balance and then encodes the sensor-linear to a Log C image" [S12].
- The only encoding roll-off is in LogC3. "The curves for EI values above 1600 … have an additional shoulder or roll-off built in to keep the maximum Log C code values within the available output range" [S1]. The 2011 paper lists a "Soft-shoulder for exposure index settings greater than EI 1600" [S5].
- ARRI's IDT maker implements that shoulder as a Hermite spline on code values above 0.8 whenever the clip would land above 1.0, and inverts it on decode [S4]. Computed: that starts at EI 1600, where the clip would land at 1.005, and grows to 1.051 at EI 3200. LogC4 has no such shoulder, only the hard 1.0 limit above EI 3200.
- **Inference:** ARRI's "film-like highlight roll-off" [S9] describes the rendered picture, not the sensor data. Nothing published describes a non-linear sensor response near full well.

### 1.4 Per channel, and white balance

Published facts:

- The tables give one clip value per EI, for the normalised sensor signal of a photosite [S1, S3].
- The false-colour zones are computed on an achromatic, Rec.709-weighted LogC signal, not per channel [S3].
- White balance is applied after readout and before LogC [S12].
- "For a green screen shot neither white point setting has an effect on the green channel gain." And: "the blue gain used with a 3200K tungsten light is substantially higher than the blue gain used with a 5600K daylight" [S7]. So ALEXA white balance is a set of per-channel gains: green is fixed, and red and blue change with the colour temperature.
- ARRI's input transform expects values that are already white-balanced. ARRI does not publish the gains (`research/sony-arw-to-arri-look-research.md`).

Inference:

- As on any Bayer sensor, every photosite clips at the same sensor signal. After the gains, the three channels therefore clip at different scene levels. Which one clips first depends on the light and the white balance.
- The published "stops above grey" figures are presumably for the fixed-gain channel. Under a light where red or blue needs a gain below 1, a neutral clips earlier than the table says.

Not published: the gains, the clip order, and the size of that effect. Our ALEXA curve estimate cannot fill the gap. Its channel scale "is not identifiable from white-balanced matrices, so each channel is normalised to peak 1" (`research/alexa-ssf/README.md`).

### 1.5 Colour near the clip

What is published is qualitative only:

- About SUP 3.x, ARRI says: "a better handling of those out-of-gamut colors. Their mapping into the target color space is more gentle and smoother … With increasing exposure the color patches in those images maintain more of their color … The transition into over-exposure happens in a smoother way" [S5]. This is about the Rec 709 output, not the sensor.
- The Rec 709 conversion is a tone-map curve "applied to the Log C data" that "compresses the shadows and highlights" [S5]. A "tone-mapped" ALEXA Wide Gamut to Rec 709 matrix follows. It "generates less saturated RGB values … to compensate for the increase of saturation produced by the tone-map curve" [S1].
- The Knee look parameter "is applied to all channels equally (master control). It has no effect on the mid gray level" [S12].
- About REVEAL, ARRI says only: "Greatly improved bright saturated colors" [S13].

Not published:

- whether ARRI clamps clipped photosites to a neutral;
- any highlight desaturation in the ADA-7 debayer or in ACE4;
- the ACE4 matrices.

The Image SDK is partner-only, and ART's EULA (section 4.2) forbids reverse engineering (`docs/14`).

**Inference:** [S1] and [S12] describe one tone curve applied to the three LogC channels. Such a curve pulls bright saturated colours towards white as each channel reaches the shoulder; the less saturated matrix is ARRI's published counterweight. The exact hue path exists only in the LUT.

### 1.6 The display rendering's shoulder

Published facts:

- **The in-camera video output is normalised.** The Color FAQ says: "In a video image, however, one wants to have the clipping point of the sensor represented by the maximum signal (100%) regardless if this clipping point is 6, 7, or 8 stops above mid gray. Therefore, the LUTs for conversion from Log C to Video normalize this range … This normalization is done always in the camera" [S6].
- The white paper says: "We call this conversion normalized because the maximum exposure value is always represented by the maximum video code value (1023 in 10 bit). When the lookup tables are generated one can decide to omit this scaling. We call this photometric conversion" [S5].
- For a timeline with mixed EIs, ARRI advises one LUT: "use the LUT computed for EI 800" [S6]. The FAQ also says the normalisation has a side effect: "you get lower blacks" [S6].
- The ARRI Color Tool raises a "3D-LUT/look warning" when a look's 3D LUT does not map "max. white (65535, 65535, 65535)" to 65535 (by more than 65 code values). ARRI calls it a technical warning, not a creative one [S12]. ARRI expects the top of LogC to reach the top of the display.
- The default ARRI 709 rendering maps "Middle gray … to 40% video level" [S12].
- **LogC4 / REVEAL has a fixed display rendering.** There is one set of LogC4 display LUTs for all EIs; the file names carry no EI [S11].
- ARRI's own HDR page shows an ALEXA 35 Rec 709 picture in which "the highlights in the red emergency light look clipped", while the LogC4 recording holds them [S10]. So with REVEAL the display reaches white at a fixed scene level, below the EI 800 sensor clip.
- The K1S1 ("Classic") and REVEAL display transforms are published only as `.cube` LUTs [S11]. Those LUTs are all rights reserved (`docs/14`).

### 1.7 Summary: sensor clip, encoding shoulder, display shoulder

| | Where | ALEV III | ALEV 4 (ALEXA 35) | Published? | What we can use |
|---|---|---|---|---|---|
| Sensor clip | photosite, linear signal | 5.3 to 9.6 stops above grey, by EI | 7.0 to 12.4 stops | exactly (formulas, tables) | the numbers |
| Encoding shoulder | LogC3 above EI 1600 | Hermite spline, invertible | none (a hard 1.0 limit above EI 3200) | exactly (IDT maker) | nothing: we decode it away |
| Display shoulder | display transform | camera output normalised: clip at 100 % (LogC3 era) | one fixed rendering for all EIs (LogC4) | the principle only | the principle; not the shape |

## 2. What data we may and may not use

| Source | What it gives | Licence or terms | Use under the V1 rule |
|---|---|---|---|
| ARRI LogC3 VFX document [S1], LogC4 specification [S2], false-colour specification [S3], 2011 white paper [S5], Color by Numbers [S12], Color and VFX FAQs [S6, S7], data sheets [S8] | clip per EI; the encodings; the order of white balance and encoding; normalised conversion; knee | no licence text; functional facts (`docs/14`) | yes: numbers and principles, implemented from the specification |
| ARRI's IDT maker and CTL in aces-dev [S4] | grey and clip on the sensor scale; no clamp; the LogC3 shoulder above EI 1600 | AMPAS licence (with notice), like the ARRI matrices we already ship | yes |
| ARRI display LUTs, Look Library | shoulder shape, hue near white | all rights reserved | runtime import only. Never fitted, used in tests or shipped (`docs/14` rule 4) |
| ARRI sample footage, "Helen & John" [S14] | real ALEXA highlights | "created for workflow evaluation purposes"; no grant | offline viewing at most (`docs/14`); no fitting |
| ARRI Reference Tool, ART CMD [S15] | REVEAL processing of ARRIRAW | EULA section 4.2 forbids reverse engineering; use limited to the documented purposes (`docs/14`) | no highlight study. Its manual lists EI, CCT and tint overrides and no highlight control |
| ARRI Image SDK | ARRIRAW processing | Partner Program, closed | no |
| HdM ALEXA SXT curves | possibly the absolute channel scale | no licence | validation only (`docs/14`). Whether their raw channel scale is preserved is unverified |
| HDR4EU deliverable D2.5 [S23] | cited by [S2] for the LogC4 design | not in the project's public deliverables list | not available |
| CineD lab tests [S16, S17, S18] | dynamic range and latitude, ALEXA and Sony | articles, © CineD | context only; nothing ships |
| DXOMARK measured ISO and dynamic range [S19, S20] | Sony headroom at base ISO | website, © DXOMARK | numbers for this research; nothing needs to ship |
| PhotonsToPhotos PDR [S22] | Sony dynamic range | website | context only |
| Public Sony raws (RawDB CC0, raw.pixls.us CC0, Play Raw CC BY / BY-SA) | scenes with real headroom | as in `docs/test-photo-sources.md` | the test in P4 |

None of the proposals below needs a third-party number in shipped code:

- The Sony clip comes from each file's white levels and the chain.
- The ALEXA clip comes from ARRI's published formula.
- The DXOMARK and CineD figures only support this analysis.

## 3. The Sony side

### 3.1 Headroom above grey at base ISO

**Our own figure.** `docs/02` reports that the clip is a constant +3.58 stops above grey on all 12 local photos at the camera gain of +1.11 EV. Green clips first, and the clip is 2.16 in AWG4. The gain is the median of the cameras' own JPEG engines (std 0.13 EV).

**DXOMARK.** DXOMARK's "measured ISO" is the saturation-based ISO 12232 speed [S20]: `S_sat = 78 / H_sat`. When the measured ISO equals the setting, this puts 18 % grey at 12.7 % of saturation, half a stop above a 100 % white [S21]. Grey then sits log2(1/0.127) ≈ 2.96 stops below the clip, and every stop by which the measured ISO falls below the setting adds one stop. Computed from DXOMARK's numbers at ISO 100 [S19]:

| Body | Measured ISO at ISO 100 | Headroom above grey (computed) | DR, EV (print, 8 MP) |
|---|---|---|---|
| A1 | 70 | 3.48 | 14.47 |
| A7R V | 74 | 3.40 | 14.75 |
| A7R IV | 72 | 3.44 | 14.77 |
| A7R III | 70 | 3.48 | 14.70 |
| A7R II | 74 | 3.40 | 13.89 |
| A7 IV | 68 | 3.52 | 14.73 |
| A7 III | 74 | 3.40 | 14.61 |
| A7 II | 73 | 3.42 | 13.62 |
| A7C | 67 | 3.54 | 14.65 |
| A7S III | 61 | 3.68 | 13.91 |
| A7S II | 74 | 3.40 | 13.29 |
| A9 | 74 | 3.40 | 13.27 |
| A6400 | 73 | 3.42 | 13.60 |
| A6300 | 72 | 3.44 | 13.73 |

- Each body's headroom is the same at ISO 200, 400 and 800 to within 0.06 stop.
- The DXOMARK route assumes that the camera meters grey to the ISO 12232 standard exposure. Our 12-file figure (3.58) uses the grey of the camera's own JPEG and comes out 0.04 to 0.18 stop higher.
- Two independent routes give the same answer: about 3.5 stops.

### 3.2 Total range

- **DXOMARK** (print-normalised, SNR 1): Sony full-frame bodies have 13.3 to 14.8 EV at base ISO [S19].
- **PhotonsToPhotos PDR at ISO 100:** A1 11.50, A7R IV 11.62, A7R III 11.64, A7 III 11.60, A7 IV 11.57, A7R V 11.69 [S22]. This is a different, stricter metric, and no ALEXA has been measured on that scale.
- **ALEXA:** ARRI quotes 14 stops on its DRTC chart [S8].
  - CineD measured the Mini LF at 13.5 stops (SNR 2) and 14.7 (SNR 1) in ProRes, and 13.4 (SNR 2) in ARRIRAW. The ALEXA Classic measured 13.8 and 14.8 [S16].
  - CineD measured the ALEXA 35 in ARRIRAW at EI 800 at 15.1 and 16.3 [S18].
- **Sony in CineD's video tests** (S-Log3, ISO 800, internal recording): the A1 measured 11.9 / 13.2 in 8K and 12.7 / 13.8 in UHD [S17].
  - Exposure latitude was 8 stops for the A1 (3 over, 5 under), against 10 for the Mini LF and 12 for the ALEXA 35 [S16, S17, S18].
  - CineD: "the benchmark so far is the ARRI ALEA [sic] Mini LF which exhibits about 2 stops more in the dynamic range and latitude discipline" [S17].

Inference:

- Normalised to a similar size, a modern Sony still sensor at base ISO has about the same total range as ALEV III: about 14.5 EV at SNR 1, against 14.7.
- What differs for highlights is where grey is placed. The Sony stills meter puts grey 3.5 stops below the clip; an ALEXA at EI 800 puts it 7.4 to 7.6 stops below.
- Sony's own S-Log3 exposure lies in between. In CineD's A1 test, the face at base exposure sat about +2.5 stops above grey (60 % in S-Log3; computed from the S-Log3 curve [S24]). It clipped 4 stops above base exposure, so roughly 5.5 to 6.5 stops above grey were held.
- The metrics differ: DXOMARK's print normalisation, ARRI's DRTC chart, and CineD's SNR measured on video files. Treat the comparison as approximate.

### 3.3 What this means for a Sony raw file

**The gap.**

- The gap is 7.64 − 3.58 = 4.05 stops at EI 800 (3.8 with ARRI's rounded 7.4), and 1.7 stops at EI 160.
- The equivalent EI of a Sony render at Auto is 4 × 2^3.58 ≈ 48 by the formula, or 800 × 2^(3.58 − 7.4) ≈ 57 by the data sheet.
- Raising Exposure in Yellowmoon raises the headroom above rendered grey stop for stop, just as raising the EI does on an ALEXA. At +2 EV it is about EI 190.

**Below the Sony clip, nothing differs.** The same scene-linear values give the same LogC4 and the same rendering. Computed for a neutral, ACES 2 with the +0.85 EV placement, no look and no lift, in sRGB codes:

| stops above grey | 1.0 | 2.0 | 2.58 | 3.08 | 3.58 (Sony clip) | 5.32 (ALEXA EI 160 clip) | 7.64 (ALEXA EI 800 clip) |
|---|---|---|---|---|---|---|---|
| sRGB code | 158 | 192 | 208 | 220 | 229 | 247 | 254 |

- For everything the Sony recorded, today's render is already the ALEXA's render.
- The only difference is above +3.58 stops. An ALEXA keeps rising to code 254; the Sony stops at 229.
- R5 can lift partly clipped areas a little. A source at 2.6 × clip comes back at about 1.3 × (`docs/05`), which is +0.4 stop, code 234.

**In LogC4 terms** (computed): the Sony clip sits at 0.498, the ALEV III EI 800 clip at 0.760 and the ALEXA 35 EI 800 clip at 0.870. ARRI's looks and display LUTs were made for footage that reaches 0.76 to 0.87. Sony data stops at 0.50, so the top third of their range is never used.

**Why C7 does nothing** (computed, neutral, same chain):

| where | r = 0 | r = 0.5 (default) | r = 1 |
|---|---|---|---|
| at the clip | −0.018 stop (code 228.5) | −0.090 stop (227.4) | −0.139 stop (226.5) |
| 1 stop below the clip (+2.58) | 0 | −0.002 | −0.018 |
| +2.0 stops above grey | 0 | 0 | −0.003 |

The shoulder compresses exactly where ACES 2 already compresses. On a neutral it moves the picture by at most 2 codes. The 14 codes in your measurement must come from saturated colours, whose largest channel reaches the knee at a lower luminance (inference).

### 3.4 What "ALEXA highlight clipping" can mean for a Sony file

| Meaning | What it would take | Verdict |
|---|---|---|
| a. Put the clip where the ALEXA's is | the published number; trivial to apply | No effect: the Sony clip is lower. It binds only after a push of about +4 EV. Worth keeping as a cap (P2) |
| b. Clip to white the way ARRI does | a hard clip, plus a rendering in which the clip reaches white | The hard clip is already there: R5 levels fully clipped quads to a neutral (`docs/05`). The white at the clip is P2 at r = 1 |
| c. ARRI's channel-saturation behaviour and near-clip colour | ARRI's processing near the clip | Not possible: unpublished, not identifiable from our curve estimate, and locked in LUTs, the SDK and ART. The closest legal route is to give the user's own imported ARRI display LUT ALEXA-range values (P2 sits before LogC4) |
| d. ALEXA latitude through exposure placement | expose about 4 stops below the camera's meter, then raise Exposure in Yellowmoon | Real, but it happens at capture. With DXOMARK's 14.5 EV it leaves about 7 stops below grey at SNR 1, against ARRI's 6.6 at EI 800 (inference, mixed metrics). Advice and a readout (P3), not a pipeline stage |
| e. Emulate the latitude by stretching | map the Sony's 3.6 stops onto the ALEXA's 7.6 | Invents the band from code 229 to 254. ARRI's normalised video does the same when an ALEXA runs at a low EI: the clip goes to white whatever the EI. The principle is ARRI's; the curve would be ours (P2) |

## 4. Proposals

### P1 · Retire the C7 highlight shoulder (small)

- **Stage.** C7. For highlights, `f(n) = n`. The low-light toe stays.
- **Data.** ARRI's linear-to-clip formulas [S1, S2, S4]: there is no sensor roll-off to emulate. Your measurement and section 3.3: no visible effect. Nothing is estimated.
- **Effect.** Values above the clip from R5 pass unchanged into LogC4. That is fine: LogC4 holds up to 469.8, and ACES 2 compresses.
- **The slider.** Remove it, or give it to P2.
- **Risk.** Nothing visible. Golden images move by a few codes, so they need a re-baseline and a `PIPELINE_VERSION` bump.
- **Tests.** An identity property (`f(n) = n` bit for bit for the highlight part); the toe tests stay; CPU/GPU parity is unchanged.

### P2 · C7 becomes a highlight range map, from the Sony's range onto the ALEXA's (medium)

**Stage.** C7, in scene-linear AWG4 before the LogC4 encode. Looks, the curve, ACES 2, an imported ARRI display LUT and the DNG all see its output.

**Maths.** Work in stops, `s = log2(Y / 0.18)`, where `Y` is the pixel's Rec.709 luma (the same luma the white lift uses). The gain multiplies all three channels.

```
h_S  = log2(c / 0.18)                     Sony headroom. c = FusedChain::clip() at the current exposure (3.58 stops at Auto)
h_A  = log2((1 - 256/65535) * EI_A / 4)   ALEXA clip. EI_A = 800 gives 7.64   [S1, S4]
T    = h_S + r * max(h_A - h_S, 0)        target for the Sony clip; r is the control, 0..1
s_k  = max(1, h_S - d)                    knee; d = depth of the knee below the clip (taste; 1 stop proposed)
u    = (s - s_k) / (h_S - s_k)
s'   = s                                  for s <= s_k
s'   = s + (T - h_S) * phi(u)             phi(u) = u^2 for u <= 1; 1 + 2(u - 1) for u > 1 (R5 values above the clip)
s'   = min(s', h_A)                       the ALEXA's hard clip at EI_A
gain = 2^(s' - s)
```

- When `h_S ≥ h_A` (Exposure pushed by about +4 EV or more), set `T = h_A`. The stage is then the identity plus the ALEXA's hard clip at `h_A`. This is the only case where an "ALEXA sensor clip" actually acts.
- **Properties.**
  - No change below the knee, so mid grey never moves.
  - Smooth (C1) at the knee.
  - Monotone.
  - The Sony clip lands exactly on `T`.
  - Channel ratios are kept.
  - `r = 0` is the identity, which is today's picture minus P1.

**Display result** (computed; neutral; ACES 2 with the placement; no look, no lift; sRGB codes):

| stops above grey | 1.0 | 2.0 | 2.58 | 3.08 | 3.33 | 3.58 (clip) | 3.96 (R5) |
|---|---|---|---|---|---|---|---|
| today (= what an ALEXA records) | 158 | 192 | 208 | 220 | 225 | 229 | 234 |
| r = 0.5, d = 1 | 158 | 192 | 208 | 229 | 240 | 248 | 254 |
| r = 1, d = 1 | 158 | 192 | 208 | 236 | 248 | 254 | 254 |
| r = 0.5, d = 2.58 (knee at +1 stop) | 158 | 201 | 225 | 239 | 245 | 248 | 252 |
| r = 1, d = 2.58 | 158 | 209 | 236 | 249 | 252 | 254 | 254 |

**Parameters and where they come from.**

- `c`: from each file (white levels, white balance, the chain). Published by the file itself.
- `EI_A = 800` and the formula: ARRI's [S1, S4]. Published. ARRI's rounded 7.4 [S7, S8] would put the target 0.24 stop lower; either is defensible. The formula reproduces ARRI's own LogC3 clip code, 0.9539.
- `r`: the user's control.
  - 0 is ARRI's "photometric" conversion: the clip renders where its scene level puts it.
  - 1 is ARRI's "normalized" conversion: the clip goes to white [S5, S6].
  - The principle is published; the curve between the two ends is ours.
- `d` and `phi`: taste, not ARRI. A deep knee spreads the stretch over more stops. A shallow knee keeps more of the recorded range true.
- **The norm:** Rec.709 luma, as in the white lift, whose proof that every channel stays monotone carries over. Today's C7 uses the largest channel. For an expansion that would lift saturated LED colours by up to 4 stops.

**What the control would mean.** "Highlight white", from "as recorded" (0) to "the clip at white, as an ALEXA's own video output does" (1). The label must not use ARRI names (`docs/14` rule 6).

**Relation to the white lift (C12 step 1b).** The two aim at the same missing white. Options:

- (a) P2 replaces the lift's estimator. The lift then follows from the exposure, deterministically, instead of from the photo's highlights.
- (b) Keep both, and re-tune the lift's constants with P2 in place.
- (c) Keep only the lift, and do P1.

Do not stack them unchecked. The lift (up to 2^3) on top of P2's placed 64.6 would pass ACES 2's roof at 128 × white, near the float32 reference glitch noted in `docs/02` C12.

**Imported ARRI display LUTs and LogC3 looks.** Today they keep ARRI's own highlights (`docs/02`). With P2 at C7, they receive values up to LogC4 0.76 (LogC3 0.954), the range ARRI made them for, and apply ARRI's own shoulder to it. I think that is closer to ARRI than today, but it changes a documented rule, so it is your decision.

**Exposure.** `c` follows the Exposure slider, so P2 adapts the way the EI does on an ALEXA. In normalised mode (r = 1) the clip stays at white at every exposure, so darkening a photo does not darken a clipped sky. That is ARRI's in-camera behaviour. It is also why r should not default to 1 before a blind A/B.

**Risks.**

- Taste. At r = 1 it brightens recorded highlights in the knee band by 16 to 29 codes at +3.1 stops.
- Every golden image moves.
- The hue of near-white colours follows ACES 2's path to white over a wider input range.
- Noise near the clip is stretched in scene terms. ACES 2 compresses it again: in the top stop, the display contrast ends up about twice today's.
- Posterisation is not expected for uncompressed and lossless raws: the top stop of a 14-bit raw holds about 8,000 levels. Lossy cRAW files have coarser steps near the clip (the R1 qstep); check them in P4.

**Determinism and GPU.**

- P2 is a per-pixel function of four uniforms (`h_S`, `s_k`, `T`, `h_A`) and the luma weights, built from `log2`, a clamp and `exp2`, like `white_lift_gain`.
- It needs a CPU reference and a parity test on lavapipe.
- Tiles agree because nothing is spatial.

**Tests without measurements.**

- Property tests:
  - identity below `s_k`, bit for bit;
  - every channel monotone;
  - `f(c) = T` to 1e-12;
  - the cap at `h_A`;
  - `r = 0` is the identity.
- The truth test of P4.
- A blind A/B on at least 12 photos (extend DH-3).

### P3 · Tell the photographer where the clip is (small)

- **A readout** in the Automatic card or the Light tab: "Highlights clip 3.6 stops above mid grey". As a factual second line, the equivalent EI: about EI 50 at Auto, about EI 190 at +2 EV.
- **A docs line:** "For more highlight room, expose under and raise Exposure. Each stop under gains one stop above grey and costs one stop of shadow noise" (inference from section 3.2).
- **Data:** the file and ARRI's formula; no third-party numbers.
- **Side fix: the highlight guard.** `docs/02` says the auto-exposure highlight guard's level of 8.0 "stands for ALEXA's headroom". In fact 8.0 is 5.47 stops above grey, which matches an ALEXA at about EI 180, not EI 800 (35.9). It cannot fire on Sony frames either way (STATUS question 23). Options:
  - correct the comment;
  - set the level to 35.9 (the guard stays dormant);
  - remove the guard.

### P4 · A truth test from public files: synthetic clipping (medium, tooling)

- **Idea.** A public Sony raw exposed well below its clip holds real data above +3.58 stops once rendered at a pushed exposure.
  - Clip a copy of its mosaic where a normally exposed frame would clip: multiply by 2^k, then take the minimum with the white level.
  - Run R5 and P2 on the copy.
  - Render the unclipped original through the same chain with the ALEXA's EI 800 clip, and compare.
  - For that scene, the original stands in for "the same photo with ALEXA headroom".
- **Metrics.**
  - display ΔE00 and code error in the clipped region and in the knee band;
  - the share of pixels within 4 codes;
  - the profile across clip boundaries, to catch fringes.

  Sweep `r`, `d` and R5 variants.
- **Data.** RawDB and raw.pixls.us (CC0); Play Raw files under CC BY or CC BY-SA showing high-contrast scenes: lamps at night, windows, sunsets exposed to protect the highlights. The 12 local files may be used locally only. This complies with the V1 rule: public files, no captures.
- **What it can decide.** The knee depth `d`; a default for `r` (to put before the blind A/B); whether R5's global chroma offset helps or hurts beyond 1.3 × clip.
- **Limits.**
  - The truth has the Sony's colour, not ALEXA colour. That is fine: below the clip, colour is the sensor map's job.
  - Pushed files carry more noise.
  - It shows what the true values would have looked like, not what you prefer. Preference stays a blind A/B.
- **Bonus.** The same synthetic clipping makes a better test for R5. The `docs/05` gate ("within ΔE00 5 of the surrounding unclipped area") could then be checked against the truth instead of the surround.

### P5 · Better reconstruction of partly clipped colour (R5, medium): the one lever backed by real data

- An ALEXA records the 4 stops that the Sony clips. Where only one or two Sony channels clipped, R5 can recover part of that.
- Today a source at 2.6 × clip comes back at about 1.3 × with a paler hue (`docs/05`).
- A more hue-preserving extrapolation would get ALEXA-like latitude from data the Sony did record. For example, carry the unclipped chromaticity in from the boundary rather than applying one global offset per channel.
- Test it with P4. Keep the clean-room rule of `docs/05`.

## 5. What cannot be done, and why

- **How ARRI treats channels at and near the clip** (a neutral clamp or not; desaturation in ADA-7 or ACE4). Unpublished. The SDK is closed, ART's EULA forbids reverse engineering (section 4.2), and the sample footage is for workflow evaluation only.
- **The ALEXA's per-channel clip order and levels under each light.** The white-balance gains are unpublished. Our estimate's channel scale is not identifiable (each channel is normalised to peak 1). The only public facts are that the green gain is fixed and that the blue gain is "substantially higher" at 3200 K [S7]. HdM's measured SXT curves might carry the raw channel scale, but they are validation-only and unverified.
- **The shape of ARRI's highlight roll-off and its hue near white** (K1S1, REVEAL). These exist only as LUTs, all rights reserved. The published fragments are qualitative:
  - a tone curve applied to the LogC channels;
  - a less saturated matrix after it;
  - a knee that is a master control applied equally to all channels;
  - gentler out-of-gamut mapping since SUP 3.

  The legal route is the user's own LUT, imported at runtime.
- **How ARRI's normalised conversion reshapes the curve between EIs.** Only the principle (the clip at 100 %) and one side effect ("lower blacks") are published [S6].
- **Why ARRI's data sheet says 7.4 stops when its own formula gives 7.64.**
- **Data the Sony did not record.** Fully clipped areas carry no information. No model and no ARRI data can say how bright they were. P2 chooses a level; it does not recover one.
- **The LogC4 design report.** HDR4EU D2.5, cited by the LogC4 specification, is not public [S23].

## 6. Questions for Mykyta

1. P1: remove the Highlights roll-off slider, or keep the control for P2?
2. P2: should highlights follow ARRI's in-camera "normalized" behaviour (the clip at white), ARRI's "photometric" behaviour (as recorded), or a default in between chosen by blind A/B?
3. P2 and the white lift: should P2 replace the lift's estimator, should both stay (re-tuned), or should only the lift stay?
4. Should imported ARRI display LUTs and LogC3-to-Rec 709 looks receive P2's range? Putting P2 at C7 would change the rule that they keep ARRI's own highlights.
5. P2's target EI: fixed at EI 800 (ARRI's base EI, and ARRI's advice when one LUT serves every EI), or a user choice from EI 160 to 1600?
6. The highlight guard: keep 8.0 with a corrected comment, set it to the EI 800 clip, or remove it?
7. P3: show the clip readout and the equivalent EI?

## Sources

All web pages were retrieved on 2026-10-17. "Used" names what this note takes from each source.

- [S1] ARRI (Harald Brendel), "ALEXA Log C Curve – Usage in VFX", version of 9 March 2017. https://www.arri.com/resource/blob/31918/66f56e6abb6e5b6553929edf9aa7483e/2017-03-alexa-logc-curve-in-vfx-data.pdf. Used: the clipping-level table per EI; parameter tables for sensor signal and exposure value; the shoulder above EI 1600; black at 256/65535; the tone map on LogC and the less saturated tone-mapped matrix.
- [S2] ARRI (Sean Cooper, Harald Brendel), "ARRI LogC4 Logarithmic Color Space – Specification", version of 23 January 2025. https://www.arri.com/resource/blob/278790/bea879ac0d041a925bed27a096ab3ec2/2022-05-arri-logc4-specification-data.pdf. Used: the hardware encoding curve with `a_h = (2^18 − 16) · EI/800` and its 1.0 limit; `117.45 = 0.18/(400/260991)`; the EI-independent curve; the HDR4EU footnote.
- [S3] ARRI (Sean Cooper), "ARRI LogC False Color – Exposure Zones and Key – Specification", 4 February 2025. https://www.arri.com/resource/blob/390448/2d469ae1e98110562fe347cea50a284b/arri-logc-false-color-specification-data.pdf. Used: the ALEXA 35 clip at E_sensor 260991 (the red zone from 207148.69); the grey band at EI 400; achromatic zones; bounds held above EI 3200; the LogC3 bounds per EI.
- [S4] ARRI, via the Academy's aces-dev repository (tag v1.3.1), `transforms/ctl/idt/vendorSupplied/arri/alexa/v3_IDT_maker.py` (IDT maker v0.09) and `v3/EI800/IDT.ARRI.Alexa-v3-raw-EI800-CCT6500.ctl` ("Written by v3_IDT_maker.py v0.09 on Thursday 22 December 2016"). https://github.com/ampas/aces-dev/blob/v1.3.1/transforms/ctl/idt/vendorSupplied/arri/alexa/v3_IDT_maker.py. AMPAS licence. Used: grey at 0.01 × 400/EI on the sensor scale; linear decoding without a clamp; white-balanced input; the Hermite shoulder above code value 0.8.
- [S5] ARRI, "ALEXA Color Processing White Paper", 1 March 2011 (mirror at Samy's). https://www.samys.com/images/pdf/ALEXA-Color-Processing-White-Paper.pdf. Used: normalised and photometric conversion; headroom of 6½ stops at EI 400 plus one stop at EI 800; maximum codes 917 and 976; the soft shoulder above EI 1600; the SUP 3 out-of-gamut handling and the smoother transition into over-exposure.
- [S6] ARRI, "Color FAQ" (web page). https://www.arri.com/en/learn/camera-systems/image-science/color-faq. Used: "Normalized & Photometric conversion" (the clip at 100 % in the camera; the EI 800 LUT for a mixed timeline; lower blacks).
- [S7] ARRI, "Visual Effects FAQ" (web page). https://www.arri.com/en/learn-help/learn-help-camera-system/image-science/vfx-faq. Used: 7.4 (ALEV III) and 9.3 (ALEV 4) stops above grey and 6.6 and 7.7 below at EI 800; the green gain unchanged by white balance; the higher blue gain at 3200 K.
- [S8] ARRI, ALEXA Plus technical data (web page on arri.ae). https://arri.ae/alexa/alexa-plus-2/technical-data/. Used: the EI table from +5.0/−9.0 to +9.4/−4.6; 14 stops on the DRTC. The ALEXA Mini LF page on arri.com (https://www.arri.com/en/cine-systems/cine-cameras/alexa-mini-lf) gives "14+ stops over the entire sensitivity range from EI 160 to EI 3200" and "EI 800 base sensitivity".
- [S9] ARRI, ALEXA 35 product page (web page). https://www.arri.com/en/cine-systems/cine-cameras/legacy-cine-cameras/alexa-35. Used: 17 stops; "1.5 stops in the highlights and a stop in the shadows over previous ALEXA cameras"; "film-like highlight roll-off"; EI 160 to 6400.
- [S10] ARRI, "ALEXA 35 and ALEXA 35 Xtreme High Dynamic Range" (web page). https://arri.com/en/camera-systems/cameras/alexa-35-xtreme/alexa-35-and-alexa-35-xtreme-high-dynamic-range. Used: Rec 709 highlights that "look clipped" while LogC4 holds them.
- [S11] ARRI (Simon Duschl), "LUT Naming Convention – LogC4 and LogC3 – Guideline", 10 July 2025. https://www.arri.com/resource/blob/294602/99cf69084be7f6638181660483d3e39d/arri-lut-naming-convention-logc4-and-logc3-guideline-en-data.pdf. Used: one set of LogC4 display LUTs with no EI; the four in-camera display transforms; the Classic and v1 LogC3 Rec 709 LUTs.
- [S12] ARRI, "AMIRA & ALEXA Mini Color by Numbers – White Paper", 3 May 2018. https://www.arri.com/resource/blob/31924/a987b5462abe89c58f3f7903d4123891/2018-03-arri-amira-alexamini-color-by-numbers-data.pdf. Used: white balance before the LogC encode; Knee as a master control applied to all channels, with no effect on mid grey; mid grey at 40 % video; the ARRI Color Tool's max-white warning for 3D LUTs.
- [S13] ARRI, "REVEAL Color Science" (web page). https://www.arri.com/en/learn/camera-systems/image-science/reveal-color-science. Used: ADA-7 and ACE4 as named blocks; "Greatly improved bright saturated colors"; REVEAL for older ARRIRAW through the SDK.
- [S14] ARRI, "Camera Sample Footage & Reference Image" (web page). https://www.arri.com/en/learn/camera-systems/camera-sample-footage-reference-image. Used: "created for workflow evaluation purposes".
- [S15] ARRI, "ARRI Reference Tools Command-Line" user manual v1.0.0, 12 January 2026. https://www.arri.com/resource/blob/402608/20ac446709713dfa4271aa47f633c439/arri-user-manual-art-cmd-v1-0-0-data.pdf. Used: the processing options (EI, CCT and tint overrides; no highlight control). EULA position as recorded in `docs/14`.
- [S16] CineD, "ARRI Alexa Classic & Mini LF Lab Test: Rolling Shutter, Dynamic Range and Latitude", 21 April 2021. https://www.cined.com/arri-alexa-classic-mini-lf-lab-test-rolling-shutter-dynamic-range-and-latitude/. Used: 13.8/14.8 and 13.5/14.7 stops (13.4 in ARRIRAW); 10 stops of latitude; about 7.8 stops from grey until a face clips.
- [S17] CineD, "Sony A1 Lab Test – Rolling Shutter, Dynamic Range and Latitude", 2 July 2021. https://www.cined.com/sony-a1-lab-test-rolling-shutter-dynamic-range-and-latitude/. Used: 11.9/13.2 (8K) and 12.7/13.8 (UHD) stops; 8 stops of latitude (3 over, 5 under); the forehead clipping 4 stops over base; "about 2 stops more" for the Mini LF.
- [S18] CineD, "ARRI ALEXA 35 Lab Test: Rolling Shutter, Dynamic Range and Latitude – plus Video!", 17 August 2022. https://www.cined.com/arri-alexa-35-lab-test-rolling-shutter-dynamic-range-and-latitude-plus-video/. Used: 15.1/16.3 stops; 12 stops of latitude; EI only redistributes code values.
- [S19] DXOMARK, sensor measurements (Measured ISO; Dynamic Range, print) for the Sony A1, A7R V, A7R IV, A7R III, A7R II, A7 IV, A7 III, A7 II, A7C, A7S III, A7S II, A9, A6400 and A6300. Pages such as https://www.dxomark.com/Cameras/Sony/A7R-IV---Measurements; the values come from the pages' data endpoints `https://www.dxomark.com/dakdata/graph/sensor/<id>/iso` and `…/dynamicRange/print`. Used: measured ISO and dynamic range at ISO 100 to 800.
- [S20] DXOMARK, glossary "ISO speed" (web page). https://www.dxomark.com/glossary/iso-speed/. Used: measured ISO is the saturation-based ISO 12232 speed; manufacturers keep "headroom" to recover blown highlights.
- [S21] Wikipedia, "Film speed" (web page). https://en.wikipedia.org/wiki/Film_speed. Used: the factor 78 puts 18 % grey at 12.7 % of saturation, half a stop of headroom above 100 % white. Corroborated by Bill Claff, "DxOMark ISO Sensitivities", https://photonstophotos.net/Investigations/DxOMark_ISO_Sensitivities.htm (`S_sat = 78/H_sat`; a typical shift of −0.33 EV).
- [S22] Bill Claff, PhotonsToPhotos, "Photographic Dynamic Range versus ISO Setting" chart (web page). https://www.photonstophotos.net/Charts/PDR.htm. Used: PDR at ISO 100 (axis x = log2(ISO / 3.125)) for six Sony bodies.
- [S23] CORDIS, HDR4EU project 761544, results page. https://cordis.europa.eu/project/id/761544/results. Used: D2.5 is not among the listed public deliverables.
- [S24] colour-science (BSD-3), S-Log3 transfer function, citing Sony's "TechnicalSummary_for_S-Gamut3Cine_S-Gamut3_S-Log3_V1_00.pdf". https://colour.readthedocs.io/en/v0.3.13/_modules/colour/models/rgb/transfer_functions/sony_slog.html. Used: S-Log3 puts 18 % grey at 420/1023 with 261.5 codes per decade, for the estimate of +2.5 stops at 60 %.

Repository sources:

- `docs/02-colour-pipeline.md`: C7; C12 (placement, white lift, ACES 2 roof and glitch); the Sony clip of +3.58 stops (2.16 in AWG4); the camera gain; the highlight guard.
- `docs/05-raw-processing.md`: R1 clip mask; R5 and its 1.3 × result.
- `docs/14-licensing.md`: ART EULA, LUT and footage rules, rule 8.
- `docs/test-photo-sources.md`: public raw tiers.
- `research/alexa-ssf/README.md`: channel scale not identifiable.
- `research/dynamic-vs-lut-2026-10-14.md`: earlier highlight numbers.
- `research/sony-arw-to-arri-look-research.md`: unpublished gains and highlight treatment.
- Your measurement of the slider (task brief): 11 photos, at most 14 of 255 codes, no photo above 0.01 % of pixels changing by more than 4 codes.
