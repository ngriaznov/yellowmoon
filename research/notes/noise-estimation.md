## Yellowmoon: per-image raw noise estimation for Sony ARW

### Recommendation

1. **Noise model, per image, in normalised units.** Use var(μ) = a·(μ − δ_c) + b + σ_row² + q²/12.
   - Share **a** across the four CFA planes. Shot-noise gain in DN does not depend on colour. The per-channel spread in darktable's profiles looks like white-balance scaling (my inference). Fit a per channel only as a QA check.
   - Estimate small per-channel offsets **δ_c**. ELD found this "colour bias" was the largest single gain in its ablation (+0.9 to +2.2 dB) [11].
2. **Estimator.** Reimplement it in Rust from the papers. Do not port code, because the licences forbid it.
   - Start with a Foi-style pass on each half-resolution plane: wavelet detail coefficients, a smoothness mask, intensity bins, a robust MAD variance per bin, and a clipped-data likelihood near white [1].
   - Use a Chen-style PCA eigenvalue test per bin as the fallback for heavily textured scenes [4].
   - Add one YOND-style refinement: denoise once, then re-fit using residual variance minus the variance of the denoised image. YOND reports the error falling from up to 29% to about 2% [6].
   - Do a MAP fit with priors.
3. **Priors.**
   - The A1 calibration gives log a = α·log ISO + β and b = σ_pre²g² + σ_post², fitted separately for each gain regime.
   - For other bodies, use a wide log-normal prior on a (about ±1 stop around a ∝ ISO). Take regime breaks from a table that Yellowmoon generates itself, and let the data override it.
   - Map extended low ISOs to base gain.
   - Pool estimates across files with the same model, ISO and bit depth within the current batch. This keeps the app stateless.
4. **Row noise.** Take row medians of the residual inside smooth regions, take a second difference across rows, and subtract the expected pixel-noise contribution. Expect it to be small. ELD's released Sony A7S2 parameters give σ_row ≈ 0.16 × σ_read at K ≈ 1.5 (my calculation from their file).
5. **Detect pre-filtered raws.** Compare the variance of 2×2-binned residuals with the per-pixel variance; white noise gives a ratio of 1/4. Where filtering shows up, inflate σ. Over-estimating σ is the safer error: CBDNet trains with an asymmetric loss to suppress under-estimation [13].
6. **Sigma map and training.**
   - Compute σ = √(a·x̂ + b_eff) from a smoothed x̂, not from the noisy value.
   - Train on "virtual cameras" sampled the ELD way: log K uniform, log σ linear in log K plus scatter, Tukey tails, row noise.
   - Jitter the sigma map by about ±25% during training.
   - ELD found that borrowing another camera's parameters cost only 0.01–0.36 dB [11]. Li et al. found under 0.1 dB of sensitivity to the K range [7].
7. **DNG NoiseProfile** (tag 51041, DOUBLE, count 2 or 2×ColorPlanes) [21].
   - The model is N(x) = √(S·x + O), with x a linear signal in [0,1], S > 0 and O ≥ 0. It describes the data after linearisation and before demosaic. Planes follow CFAPlaneColor order.
   - For a linear (demosaiced) DNG, it describes the noise "immediately following the demosaic step".
   - For a mosaic export, write S = a/(W−B) and O = (b + σ_row² + q²/12)/(W−B)², clamping O ≥ 0.
   - For a denoised LinearRaw export, either re-estimate the residual noise of the output or omit the tag.
   - When the input is a DNG, read NoiseProfile as a prior.

### Methods

| Method | Estimates | Accuracy | Robustness | Cost | Licence |
|---|---|---|---|---|---|
| Foi 2008 [1,2] | a, b, clipping; pedestal allows b < 0 | Synthetic: a within ~0.5–1% | Thin edges and texture: +41% error, +11% with MAD | O(N); no timing given | TUT non-commercial only |
| Liu–Tanaka–Okutomi 2013 / 2014 [3] | 2013: Gaussian σ. 2014: 3-parameter NLF (Poisson-Gaussian) | Real Nikon frames: bias 3.35 (from Chen's test) | Needs weak-texture patches | ~4.2 s per image (MATLAB) | MATLAB File Exchange; licence file not checked |
| Chen 2015 [4] | Homoscedastic σ only; apply per bin or after a VST | Bias 0.12, √MSE 0.25 | Designed for texture | ~0.58 s (MATLAB) | No licence stated |
| Zou & Fu 2022 / TPAMI 2025 [5] | K, Gaussian read σ, μ_c, σ_row (no Tukey) | No numeric parameter error reported | Claims unseen cameras, shown only qualitatively | ResNet | No official code found (unverified) |
| YOND 2025 [6] | α, σ² | Fine stage ~2% on synthetic data incl. A7S2; Foi up to 34%, Zou 62% in worst rows | Fails with few flat regions; clipping not discussed | 11 M parameters; VST under 1 s per 12 MP | Code "to be released" |
| Li et al. CVPR 2025 [7] | Dark-frame sampling plus K = QE×AG | Robust to K choice | Needs dark frames | – | Repo MIT |
| LED 2023 [8] | Virtual cameras plus 2 pairs | – | Needs pairs | – | CC BY-NC 4.0 (unusable here) |
| Cao CVPR 2023 / LLD [9] | Flow model: FPN, black-level error, row, quantisation | 43.36 dB on SID | Needs pairs | – | No licence stated |
| Mohammadi 2026 [10] | Black-level error (colour bias) network | – | Calibration-free | – | No code listed |

### ELD (Wei et al. CVPR 2020 / TPAMI 2021) [11]

- **Components:** Poisson shot noise, Tukey-λ read noise, colour bias, row noise and uniform quantisation. There is no separate fixed-pattern term.
- **Ablation:** colour bias matters most, then true Poisson, then Tukey tails (+0.4 to +0.9 dB). Row and quantisation noise each add under 0.13 dB.
- **ISO scaling:** log σ_TL and log σ_r are each linear in log K.
- **Cameras:** Sony A7S2, Nikon D850, Canon 70D and 700D, plus 5D4 in the code.
- **Licences:** the ELD code is MIT, and the dataset is "evaluation purposes only". SID code is MIT; no dataset licence is stated [12].
- **Reuse for validation: yes.** The ELD repo ships `camera_params/release/SonyA7S2_params.npy`. Values: K from 0.094 to 6.01; log σ_read = 0.541·log K + 1.218; log σ_row = 0.528·log K − 0.592; Tukey λ from −0.14 to 0.13 by ISO; per-channel bias of ±1–4 DN.

### Sony-specific findings

- **No usable optical black.**
  - rawspeed declares BlackAreas for only 1 of its 110 Sony models (DSC-RX1R, a 14-px strip) [14].
  - In modern ARW, the right-edge columns are duplicated pixels or zero padding [15].
  - Read noise therefore has to come from image content, or from dark frames (A1 only).
- **Black and white level:** 4-value per-CFA tag 0x7310 and white level 0x787F in the encrypted SR2 sub-IFD. Typical values are 512/16383 [14].
- **No noise metadata found in ARW** (ExifTool not fully checked, so unverified). Use maker-note tag 0x2029 RAWFileType to choose the quantisation model.
- **Bit depth changes with mode.** The A7R IV records 12-bit in continuous drive with compressed raw, and compressed raw never exceeds 13-bit [18]. Detect bit depth from the spacing of codes in the histogram.
- **In-camera noise reduction:**
  - High ISO NR does not apply to RAW on the ILCE-1 [16].
  - Long Exposure NR (dark-frame subtraction) runs at 1 s or longer, mechanical shutter only, and not in continuous drive [16]. That it reaches the raw and raises b is my inference.
  - "Star eater" raw spatial filtering affects timed exposures of 3.2–4 s and longer plus Bulb on the A7R II, A7S II, A7R III and A7 III, and every timed exposure on the A9 [17]. The A1, A7 IV and A7R V are undocumented.
- **ISO regimes inferred from darktable community profiles [20] (unverified):**
  - Extended ISO 50–80 shows a at about 2× the trend, i.e. base gain pulled. EXIF ISO is not the gain there.
  - Above roughly ISO 16000 on the A1, A7R V and A9 III, ≥64000 on the A7 III and A7 IV, and ≥128000 on the A7S III, a drops to about 0.5–0.65 of the trend. That suggests raw processing; PhotonsToPhotos notes that "Hi" ISOs behave strangely [19].
- **Dual-gain switch ISO** (where b drops in darktable profiles):
  - A1: 500. A1 II: 500 (a forum claims 640).
  - A7 III: 640. A7 IV: ~400. A7 V: 1000.
  - A7R V: 320 (forum, citing PhotonsToPhotos: 320, and the same for the A7R IV).
  - A7S III: 1600.
  - A7C: 640. A7C II: 400.
  - A9: 500. A9 II: 640. A9 III: 800, with base ISO 250 [19].
- **Is EXIF ISO enough to pick the regime?** For stills, ISO plus model is enough once that table exists, with these caveats:
  - extended ISOs;
  - the top-ISO processed regime;
  - bit depth and shooting mode;
  - ISOSpeedRatings is 16-bit, so it reads 65535 for higher ISOs, and rawspeed reads only that tag;
  - video log bases differ.

### Validation data

- **Your A1:** calibrated truth. Test the estimator against it on textured, clipped and high-ISO shots. This is the primary benchmark.
- **A7S2 via ELD, SID and LLD [9,11,12]:** a second Sony body with physics truth. Terms are evaluation-only or unstated.
- **DND [24]:** Sony A7R and RX100 IV, with NLFs from Tobit regression that agree with Foi to 0.3%. Licensed for non-commercial use. Whether the NLFs are published is unverified.
- **RawNIND [23]:** ILCE-7C, 148 scenes, noisy/clean pairs, so you can derive an empirical NLF. Licence not verified.
- **darktable `noiseprofiles.json` [20]:** 73 Sony models with (a, b) per ISO. It is GPL-3.0, so use it offline as a reference only and don't ship it.
- **raw.pixls.us [22]:** CC0, but it explicitly does not want ISO series. Use it for decoder coverage and a "noise-on-noise" test: inject a known Poisson-Gaussian increment and check you recover it.
- **DPReview / Imaging Resource studio ISO series:** the grey patches give flat-field truth. They are copyrighted, so use them for private validation only (terms unverified).

### Sources

1. Foi et al. 2008: https://webpages.tuni.fi/foi/papers/Foi-PoissonianGaussianClippedRaw-2007-IEEE_TIP.pdf
2. Foi software and licence: https://webpages.tuni.fi/foi/sensornoise.html
3. Liu–Tanaka–Okutomi 2014: https://search.isc.ac/inventory/49/23796.htm ; code: https://www.mathworks.com/matlabcentral/fileexchange/43224
4. Chen et al. 2015: https://openaccess.thecvf.com/content_iccv_2015/papers/Chen_An_Efficient_Statistical_ICCV_2015_paper.pdf
5. Zou & Fu: https://arxiv.org/abs/2204.01716 ; TPAMI 2025: https://pure.bit.edu.cn/en/publications/calibration-free-raw-image-denoising-via-fine-grained-noise-estim/
6. YOND: https://arxiv.org/abs/2506.03645
7. Li et al. 2025: https://arxiv.org/abs/2505.00045 ; code: https://github.com/SonyResearch/raw_image_denoising
8. LED: https://github.com/Srameo/LED
9. Cao et al. 2023: https://openaccess.thecvf.com/content/CVPR2023/papers/Cao_Physics-Guided_ISO-Dependent_Sensor_Noise_Modeling_for_Extreme_Low-Light_Photography_CVPR_2023_paper.pdf ; LLD: https://github.com/happycaoyue/LLD
10. Mohammadi et al. 2026: https://arxiv.org/abs/2607.11090
11. ELD: https://arxiv.org/abs/2003.12751 ; https://arxiv.org/abs/2108.02158 ; code: https://github.com/Vandermode/ELD
12. SID: https://github.com/cchen156/Learning-to-See-in-the-Dark
13. CBDNet: https://arxiv.org/abs/1807.04686
14. rawspeed (ArwDecoder.cpp, TiffTag.h, cameras.xml): https://github.com/darktable-org/rawspeed
15. pixls.us masked-pixels thread: https://discuss.pixls.us/t/confusion-about-cropping-of-masked-pixels-in-raw-images/47199
16. Sony ILCE-1 help guide: https://helpguide.sony.net/ilc/2040/v1/en/contents/TP1000409229.html ; https://helpguide.sony.net/ilc/2040/v1/en/contents/TP1000425464.html
17. Star eater: https://www.lonelyspeck.com/sony-star-eater-and-how-to-fix-it/ ; https://blog.kasson.com/a7riii/sony-a7riii-raw-spatial-processing/
18. Kasson, A7R IV continuous-drive bit depth: https://blog.kasson.com/a7riv/sony-a7riv-continuous-drive-mode-precision
19. PhotonsToPhotos: https://photonstophotos.net/Charts/RN_ADU.htm ; forum: https://alphashooters.com/community/threads/native-iso-help-a1-a7rv-a7r1v.9973/
20. darktable noise profiles: https://github.com/darktable-org/darktable/blob/master/data/noiseprofiles.json
21. DNG SDK noise profile notes: https://lxr.kde.org/source/graphics/digikam/core/libs/dngwriter/extra/dng_sdk/dng_negative.h ; tag: https://awaresystems.be/imaging/tiff/tifftags/noiseprofile.html
22. raw.pixls.us: https://raw.pixls.us/
23. RawNIND: https://arxiv.org/abs/2501.08924
24. DND: https://noise.visinf.tu-darmstadt.de/ ; https://arxiv.org/abs/1707.01313

**Worth saving to memory (Yellowmoon):**
- Noise plan: per-image Foi/PCA estimator with A1 priors.
- Modern Sony ARW has no usable optical black.
- darktable noise profiles are GPL, so they are for reference only.
- raw.pixls.us has no ISO series.