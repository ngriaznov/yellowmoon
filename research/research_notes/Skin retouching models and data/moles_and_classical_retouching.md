# Beauty-mark detection and classical (non-ML) blemish removal for Yellowmoon's skin retouch

Scope: (a) finding moles and beauty marks so blemish removal leaves them alone (`BeautyMarkDetector`), and (b) classical detection and removal methods that could implement `BlemishProposer` without a trained network, judged against the gain-map design of `docs/27-skin-retouch.md`. Research date 2026-11-11. Local spec references are to `docs/27-skin-retouch.md` in this worktree ([docs/27](../../../docs/27-skin-retouch.md)).

Assumption used for every pixel figure below (not sourced, flagged in Gaps): the 450 px side of the blemish map (docs/27 §6.1 step 1) spans about 15 cm of face, so **1 px ≈ 0.33 mm**. Where a face is smaller than 450 px in the source, the map is not upscaled (docs/27 §6.1 step 1), so pixel sizes shrink in proportion.

---

## Q1. Learned mole / nevus detectors: datasets, licences, and why dermatology data does not fit portraits

### Takeaway
No licence-clean dataset of moles in ordinary face photographs was found. The few dermatology sets with shippable licences (ISIC 2016/2017 CC-0, ISIC 2024 SLICE-3D "Permissive" CC-BY, PAD-UFES-20 CC BY 4.0) are close-ups of single lesions on any body site, at a scale and lighting far from a 450 px portrait face, and none labels acne. The big, well-known sets (HAM10000, ISIC 2019/2020, SLICE-3D main, Fitzpatrick17k, ACNE04) are non-commercial or academic-only, so they may serve local evaluation at most. A learned `BeautyMarkDetector` is not realistic for round 2; a classical one is.

### Cited Findings

**Dataset licences**
- ISIC challenge data licences as listed by the archive: 2016 (Parts 1–3B) "CC-0"; 2017 (Parts 1–3) "CC-0"; 2018 shows "CC-0" after Task 1 and "CC-BY-NC" after Task 3 (Task 2 not labelled separately); 2019 "CC-BY-NC"; 2020 "CC-BY-NC"; 2024 SLICE-3D training "CC-BY-NC"; "2024 SLICE-3D Permissive" "CC-BY" — [ISIC challenge data page](https://challenge.isic-archive.com/data/)
- HAM10000 (dermatoscopic images; Harvard Dataverse DOI 10.7910/DVN/DBW86T) is listed as CC BY-NC 4.0 by the ISIC API, Academic Torrents and Dataset Ninja; a UCF mirror says it is "for non-commercial use only" and the Dataverse terms must be accepted; one mirror (hyper.ai) lists "Other", and the Dataverse page itself was not retrieved — [Academic Torrents](https://academictorrents.com/details/dc3188ee1ce7e2d2254113111b406c484101ba65); [Dataset Ninja](https://cdn.datasetninja.com/skin-cancer-ham10000); [UCF mirror](https://complexity.cecs.ucf.edu/ham10000/); [ISIC API collection](https://api.isic-archive.com/collections/212/?page=383)
- Fitzpatrick17k: the official repository states a Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported licence; 16,577 clinical images from two atlases (DermaAmin, Atlas Dermatologico); the repo ships annotations and scripts, the images are downloaded from their original sources — [github.com/mattgroh/fitzpatrick17k](https://www.github.com/mattgroh/fitzpatrick17k); [paper, arXiv 2104.09957](https://arxiv.org/abs/2104.09957)
- PAD-UFES-20: Mendeley Data page states **CC BY 4.0**; 2,298 images, 1,641 lesions, 1,373 patients; images "collected using different smartphone devices", varying sizes, PNG; six classes: BCC, SCC (incl. Bowen's), actinic keratosis, seborrheic keratosis, melanoma, **nevus**; no per-class counts or body sites on the page — [Mendeley Data zr7vgbcyr2/1](https://data.mendeley.com/datasets/zr7vgbcyr2/1); [arXiv 2007.00478](https://arxiv.org/abs/2007.00478)
- ACNE04 (Wu et al., ICCV 2019, acne severity and lesion counts): "free for academic usage"; other uses need the first author's permission; no formal open licence found — [github.com/xpwu95/ldl](https://github.com/xpwu95/ldl); [HF mirror README](https://huggingface.co/datasets/ManuelHettich/acne04/blob/main/README.md); [paper PDF](https://orca.cardiff.ac.uk/125008/1/AcneLDL_ICCV2019.pdf)
- FFHQ (the obvious face source for a trained proposer) is CC BY-NC-SA; docs/27 already records that no licence-clean training data for a proposer is known — [docs/27 §1](../../../docs/27-skin-retouch.md)

**Face-photo mole data and detectors**
- A search for a public, mole-labelled face-photo dataset found none; what exists is commercial (Nexdata "Human Facial Skin Defects" lists acne, stains, wrinkles but not moles; Unidata's facial skin condition set mentions pigmentation, not moles), a 2025 Mendeley release on congenital facial nevi (licence not checked), and a 2024 arXiv CNN for facial blemishes/lesions built from 2,300 patient images — [Nexdata card](https://huggingface.co/datasets/Nexdata/Human_Facial_Skin_Defects_Data); [Unidata](https://unidata.pro/datasets/facial-skin-condition-image-dataset/); [Mendeley cb3gsmc2mz](https://data.mendeley.com/datasets/cb3gsmc2mz); [arXiv 2402.08592](https://arxiv.org/pdf/2402.08592)
- HAUT.AI (commercial SaaS) offers a facial "Moles" metric: counts for the whole face and for nose, forehead, perioral, left/right cheek, plus a vector mask; works with 1 or 3 face images; describes moles only as "pigmented spots that vary in size, shape, and color"; publishes no size/colour/shape criteria and does not explain how moles are told from freckles or acne — [HAUT.AI docs](https://docs.saas.haut.ai/haut.ai/haut.ai-features/algorithms/face-skin-analysis-3-0/pigmentation/moles)
- A 2020 J. Craniofacial Surgery study segmented nevi in whole-face photos with logistic regression plus OpenCV edge tracing and reported agreement with dermatologists on a small sample — [catalogue record](https://katalog.lib.cas.cz/KNAV/EdsRecord/edsair,edsair.doi.dedup.....11c292e286d158d4ebd9c1b6deae52bd)

**Dermoscopy vs portraits**
- Dermoscopic/close-up imaging works at a very different scale: a L'Oréal pore study measured cheeks with a polarised dermatoscope and used a 250 µm threshold for "visible" pores — [Flament et al. 2015, CCID](https://pmc.ncbi.nlm.nih.gov/articles/PMC4337418)
- HAM10000 is a dermatoscopic collection (the ISIC 2018 Task 3 data) — [ISIC challenge data page](https://challenge.isic-archive.com/data/)

### Inferences
- **Licence-clean, but the wrong domain.** ISIC 2016/2017 (CC-0) and PAD-UFES-20 (CC BY 4.0) could legally train a patch classifier, but one lesion fills the frame (hundreds of pixels across), with dermatoscope or close-up phone lighting, often polarised, with no face context, no makeup, no pores at portrait scale. In a portrait at 450 px a 3 mm mole is about 9 px across. A network trained on close-ups and run on 9 px blobs is guessing; downscaling the close-ups to 9 px would leave almost no signal beyond "dark round blob of some colour", which a classical rule already captures.
- **No acne negatives.** None of the shippable sets contains acne, post-inflammatory marks or freckles as labelled negatives, which are exactly the confusions a beauty-mark detector must resolve. PAD-UFES-20's seborrhoeic keratosis and nevus classes are the closest.
- **Evaluation only.** HAM10000, ISIC 2019/2020, SLICE-3D (main), Fitzpatrick17k are fine for local, non-shipping evaluation under the constraint given; ACNE04 is "academic usage" only, which is narrower than "non-commercial" (Yellowmoon is a free open-source project, not academic); ask the author or the owner before even local use.
- **Weights trained on any CC BY-NC data cannot ship** (docs/14 via docs/27 §1), so the learned route for moles in V1/round 2 is effectively closed unless the owner approves new data work (and V1 bans own measurements, AGENTS.md "V1 data rule").

### Gaps
- SD-198 and Derm7pt licences were not checked in this session.
- PAD-UFES-20 body-site distribution (how many lesions are on the face) is not given on the Mendeley page; the paper's metadata would answer it.
- The SLICE-3D "Permissive" subset's composition (image size, crop scale, how many nevi) was not checked.
- The HAM10000 licence was not read on the Harvard Dataverse page itself (secondary sources agree on CC BY-NC 4.0; one mirror says "Other").
- The 2025 Mendeley congenital-facial-nevi release and the 2024 arXiv facial-blemish CNN data licences were not checked.

---

## Q2. Classical mole detection on face photos, and how commercial tools decide what to keep

### Takeaway
The face-photo literature that finds moles classically (Pierrard & Vetter 2007, Cho/Freeman/Tsao 2007, Park & Jain 2010, NessAiver 2013) converges on one recipe: restrict to skin, find dark compact blobs with a scale-space LoG/DoG, then prune with contrast, size, shape and isolation, and (in the learned variants) a small classifier. Its known failure is pores and hair roots being taken for moles at high resolution, fixed by a minimum size. Commercial tools treat moles as a separate class with their own control (Adobe "Fade", Retouch4me "Moles" slider, Evoto "Face Mole" button plus a restore pen); none documents its criteria. Industry etiquette and Adobe's default keep moles at least partly, which matches docs/27's "boxes punch holes in the blemish mask".

### Cited Findings

**Classical / semi-classical mole detection on skin and faces**
- Cho, Freeman & Tsao, "A reliable skin mole localization scheme" (MIT CSAIL / Harvard Dermatology, MMBIA workshop ~2007): skin detected non-parametrically in CbCr (Neyman–Pearson test, median filter to clean the mask); hair removed with steerable filters plus a graphical model; "Moles are modeled as circular regions of lower intensity than the surround: scale-space difference-of-Gaussian (DoG) filters are used to locate possible mole candidates of varying sizes"; DoG applied to R, G, B separately and the union of maxima taken, candidates within a radius of another removed; patches of width 2√2 × the DoG radius classified by an RBF SVM on a gist-like steerable-pyramid descriptor; on 28 test images sensitivity 84.7 % and diagnostic accuracy 79.3 % (79.4 %/76.5 % without hair removal) — [paper PDF](https://people.csail.mit.edu/billf/papers/MMBIA_FinalV7.pdf)
- Same paper, failure modes: "When the image is at high enough resolution to reliably visualize pores on skin, pores can be recognized as moles at certain lighting conditions"; hair roots likewise; the fix is a user-set minimum candidate radius; blurred moles are missed — [paper PDF](https://people.csail.mit.edu/billf/papers/MMBIA_FinalV7.pdf)
- Pierrard & Vetter, "Skin Detail Analysis for Face Recognition" (CVPR 2007): localise prominent facial skin irregularities, "in particular nevi (moles, birthmarks)"; LoG filter for candidates, normalised cross-correlation and a saliency measure to accept them; texture-based skin segmentation that needs no colour or training; positions registered to a 3D morphable model. Cho et al. note it "register[s] only salient skin irregularities that are isolated from other irregularities" — [MLAnthology entry](https://mlanthology.org/cvpr/2007/pierrard2007cvpr-skin); [Cho et al. PDF](https://people.csail.mit.edu/billf/papers/MMBIA_FinalV7.pdf); [Pierrard dissertation record](https://edoc.unibas.ch/795)
- Park & Jain, "Face Matching and Retrieval Using Soft Biometrics" (IEEE TIFS 5(3) 2010): facial-mark detector = active appearance model to locate and exclude facial features + LoG blob detection + morphological operators; evaluated on FERET and two mugshot databases — [NIJ abstract](https://nij.ojp.gov/library/publications/face-matching-and-retrieval-using-soft-biometrics)
- Zhang, Tulyakov & Govindaraju (ICB 2009) combined facial skin marks with eigenfaces, building on Pierrard & Vetter — [paper PDF](https://www.acsu.buffalo.edu/~tulyakov/papers/ZhiZhang_SkinMarkCombinations_ICB2009.pdf) (fetch failed, 503; described via search snippet)
- NessAiver (UMD MSc thesis 2013, advisor Chellappa), "Edge-Based Automated Facial Blemish Removal": histogram equalisation and illumination compensation; edge map + geometric priors to mask facial features; colour and reflectance thresholds for a skin map; "A Laplacian of Gaussian blob detector is used to identify potential candidates", then thresholding, dilation and morphological filtering — [UMD DRUM](https://drum.lib.umd.edu/handle/1903/14774)

**What moles look like (the rules' physical basis)**
- DermNet: naevi "vary in colour from pink or flesh tones to dark brown, steel blue, or black"; lighter skin tends to lighter naevi, darker skin to dark brown/black; "mostly round or oval"; "may be flat or protruding"; sizes "from a couple of millimetres to several centimetres"; ABCD flags diameter > 6 mm; "Miescher naevus is a dome-shaped smooth dermal naevus often found on the face" — [DermNet, melanocytic naevus](https://dermnetnz.org/topics/melanocytic-naevus)
- Common acquired naevi are usually < 6 mm (Osmosis), < 8 mm in one review, "usually small – 2–3 mm in diameter – and uniformly brown" (GPnotebook) — [Osmosis](https://www.osmosis.org/answers/melanocytic-nevus); [GPnotebook](https://gpnotebook.com/en-GB/pages/general-information/clinical-classification/banal-or-acquired-naevi); [Consultant360](https://www.consultant360.com/photo-essay/atlas-lumps-and-bumps-part-42-acquired-melanocytic-nevi)
- Acne papules are ≤ 5 mm, nodules > 5 mm (acne convention; general dermatology uses 1 cm) — [Healthline](https://www.healthline.com/health/papules-acne); [BMC table](https://biomeddermatol.biomedcentral.com/articles/10.1186/s41702-018-0036-5/tables/1)
- Pores visible to the naked eye are about 250–500 µm (0.05–0.2 mm²) — [Flament et al. 2015](https://pmc.ncbi.nlm.nih.gov/articles/PMC4337418)
- Tsumura et al. (SIGGRAPH 2003, ACM TOG 22(3) 770–779) separate a single camera image of skin into **melanin** and **haemoglobin** components by ICA in the optical-density domain, after removing shading by colour-vector analysis; the premise is that the two pigments are spatially independent — [project page](https://www.mi.tj.chiba-u.jp/~tsumura/Tsumura/papers/siggraph2003.htm); [PDF](https://www.mi.tj.chiba-u.jp/~tsumura/Tsumura/papers/TsumuraSig03.pdf)

**How commercial tools decide what to keep**
- Adobe Lightroom AI blemish removal (Learn page, 18 March 2026): Amount — lower values remove only the "bigger blemishes" first, 100 % removes smaller ones too; a contextual **Fade** slider "appears ... when freckles, moles, or beauty marks are detected": "The Fade slider controls the visibility of items like freckles and moles"; plus Skin Smoothing and face choice — [Adobe Learn](https://www.adobe.com/learn/lightroom-cc/web/ai-blemish-removal)
- Adobe Camera Raw help (via search snippet; the page returned 403 when fetched): Fade "adjusts the visibility of permanent marks such as moles and freckles. The default value of 50 partially fades all permanent marks, and dragging it to 100 fades them completely" — [Camera Raw help](https://helpx.adobe.com/id_id/camera-raw/desktop/ai-and-generative-features/remove-blemishes-from-photos.html)
- Retouch4me Heal: fixes "blemishes, pimples, post-acne marks and enlarged pores", automatically detecting problem areas; the Apex 1.211 update added a **Moles** slider ("At the maximum setting, detected moles are completely removed"; lower values leave them more visible) and **Keep Bindi** ("Heal automatically detects the bindi and excludes it from retouching"); no criteria for mole vs blemish published — [Retouch4me Apex 1.211 blog](https://retouch4.me/blog/retouch4me-apex-1-211-update-more-control-in-heal-with-the-new-moles-slider-and-keep-bindi); [Heal page](https://retouch4.me/heal)
- Evoto: "Face Mole" is a one-click removal with no slider; "If there are moles you wish to keep, use the Manual Tuning Pen ... to restore them"; Freckle and Acne have separate sliders under a main toggle; a support article on over-removal advises lowering blemish removal and combining the freckle slider and pen — [Evoto support, Blemish Removal](https://support.evoto.ai/blemish-removal/); [Evoto: bring back freckles/moles](https://support.evoto.ai/?p=12744)
- Retouching etiquette: "permanent features like moles or freckles should not be removed unless the client specifically asks for it" (one guide, via search) — [iStock guide](https://tool.istock.tw/en/blog/a-guide-to-basic-portrait-retouching-smoothing-skin-and-removing-blemishes)

### Inferences

**Scale in Yellowmoon's map (assumption 1 px ≈ 0.33 mm at 450 px):**

| Feature | Physical size (sources above) | Pixels in the 450 px map | % of face size D |
|---|---|---|---|
| Visible pore | 0.25–0.5 mm | 0.75–1.5 | 0.17–0.33 % |
| Acne papule / pustule | ≤ 5 mm (typ. 1–5) | 3–15 | 0.7–3.3 % |
| Acne nodule | > 5 mm | > 15 | > 3.3 % |
| Common naevus | 2–6 mm (2–3 typical) | 6–18 | 1.3–4 % |
| Brief's blemish scale 0.3–1.5 % of D | — | 1.35–6.75 | — (0.45–2.25 mm) |

- The brief's 0.3–1.5 % band starts inside the pore range and stops below the larger papules. A better band for spot detection is roughly **0.7–3.5 % of D (diameter)**, with anything below ~0.6 % (≈ 3 px) treated as texture. Moles (1.3–4 %) overlap the acne band almost entirely, so **size cannot separate moles from acne; colour and contrast must.**
- At 450 px, pores are at or below one pixel after area downscaling (docs/27 §6.1 step 1), so the Cho et al. pore failure mode mostly disappears in the map; it reappears only if detection runs on full-resolution crops. Run the detector on the same ≤ 450 px area-downscaled crop as the map.

**A classical `BeautyMarkDetector` (proposed rules; thresholds are hypotheses to calibrate, none is from a source):**
1. Work on the map's own encoding (sqrt, mean-normalised; docs/27 §6.1 step 2), in logs: `y = log((R+2G+B)/4)`, and per-channel `log R, log G, log B`. Logs make every threshold a ratio, independent of exposure and the mean normalisation.
2. Local skin reference per pixel: a robust background, e.g. a grey **closing** of `y` at radius ≈ 2.5 % D (removes dark things up to ≈ 5 % D across) or a masked Gaussian over a ring 3–5× the blob radius. Contrast `c = y − y_bg ≤ 0`.
3. Candidates: scale-space minima of LoG/DoG on `y` with σ chosen so blob diameters 1.2–4.5 % D are covered (for a 2-D LoG the peak response for a disc of radius r is at σ ≈ r/√2 — standard Lindeberg scale selection, not re-verified here), or MSER dark regions; keep a candidate only if it persists over ≥ 2 adjacent scales.
4. **Brown vs red**: melanin darkens all three channels with blue most (its absorption falls with wavelength); haemoglobin darkens green and blue and leaves red almost unchanged (Tsumura's two-pigment separation is the principled version). A cheap proxy: `q = Δlog R / Δlog G` inside the blob vs background: q near 0 (or negative) → red (acne, erythema) → blemish; q well above 0 with a clear luminance drop → brown → mole/freckle/post-acne mark candidate. Calibrate q and c on public photos; a full Tsumura ICA per face is possible later but sensitive to shading and AWG4→camera assumptions.
5. Shape and sharpness: circularity `4πA/P² ≳ 0.7`, aspect ≲ 1.6, and a steeper edge than inflamed spots (acne has a diffuse red halo; naevi are "well-circumscribed").
6. Saliency and isolation (Pierrard & Vetter): keep the most contrasted, isolated marks; cap the count (e.g. ≤ 8–12 per face) so a freckled face does not become a mask full of holes. Freckles that slip through are protected too, which matches what Adobe's Fade groups with moles.
7. Box: the detected dark extent plus ~30 % margin. docs/27 §5.3 sets the mask to 0 only inside the ellipse **inscribed** in the box and ramps to the √2 ellipse; a box drawn tight to the mole lets the ramp heal the mole's rim ("shrinking mole").
- **Known misses and false keeps:** flesh-coloured dome naevi (Miescher, common on the face) are not dark, so the rules miss them; but a dark/red-spot proposer barely touches them either, so the miss is cheap. Brown post-acne marks (which Retouch4me Heal explicitly removes) look like small flat moles and will often be kept; brown seborrhoeic keratoses likewise. Dark-skinned faces (Fitzpatrick V–VI) lower redness contrast and raise post-inflammatory pigmentation; the log-ratio rules are scale-free but untested there.
- **Policy fit:** the commercial pattern is "detect moles as their own class, keep them by default or partially, let the user change it". docs/27 keeps them fully (hole in the mask) and relies on the Brush for misses; a later "Moles" control would map onto the existing hole (scale the hole depth) without new maths.

### Gaps
- No numeric thresholds (contrast, hue ratio, circularity) for mole vs acne in face photos were found in any source; Pierrard & Vetter's and Park & Jain's parameters were not read (full texts not retrieved).
- PortraitPro, Capture One and Retouch4me Dodge & Burn mole handling: no documentation found in this session. The Fstoppers comparison of Photoshop, Capture One and Evoto blemish handling was found but not read.
- How Adobe, Retouch4me and Evoto detect moles (learned, almost certainly, but undocumented).
- The physical-to-pixel assumption (15 cm per 450 px) depends on what `Face.scale` measures with the chosen face finder (docs/27 §3); it should be checked against Apple Vision / MediaPipe boxes.

---

## Q3. Classical blemish detection: redness, small spots, and pores vs spots

### Takeaway
Classical acne detection uses the CIELAB a* channel (or an R–G excess) to separate inflamed lesions from skin, then thresholds or blob-detects (LoG/DoG, adaptive threshold); dark small spots are found with LoG/DoG or the morphological black top-hat. These are fragile when used as global absolute thresholds (the CNN literature says so), but in Yellowmoon's setting, local log-ratios inside a known face at a known scale, with the gain stage's soft strength threshold behind them, they are workable. Pores are separated from spots by scale (a minimum area of ≈ 3×3 px at 450 px) and by contrast, not by colour.

### Cited Findings
- Maroni et al. (Univ. Bergamo): mobile-photo acne pipeline of body-part detection, skin segmentation, heat-mapping and blob detection; uses CIELab a* "to boost contrast between inflamed zones and healthy skin"; lesions extracted with adaptive thresholding and Laplacian-of-Gaussian filtering — [Univ. Bergamo publication](https://cal.unibg.it/publications/automated-detection-extraction-and-counting-of-acne-lesions-for-automatic-evaluation-and-tracking-of-acne-severity); [arXiv 1711.04481](https://arxiv.org/pdf/1711.04481.pdf)
- A Korean study detects acne by thresholding a normalised a* image, since a* is positive for red pixels — [KCI/UTB record](https://vufind.katalog.k.utb.cz/SummonRecord/FETCH-LOGICAL-n647-4f6ec66e4b12de88f30b9aadbf66b18d3b14f3dd2ea54827e21072cad453e513/Details)
- Shen et al. (Sci. Rep. 2018, VGG16-based acne classifier) describe earlier threshold-based methods as "highly dependent on threshold values" and lacking generalisation — [Sci. Rep. PDF](https://www.nature.com/articles/s41598-018-24204-6.pdf)
- A 2025 arXiv review converts RGB to Lab, enhances contrast, smooths and segments acne with k-means — [arXiv 2503.02835](https://arxiv.org/html/2503.02835v1)
- Batool & Chellappa (IEEE TIP 2014) detect wrinkles and imperfections with Gabor filter responses and texture orientation fields, separating skin from imperfections with a GMM fitted by EM — [University of Galway record](https://research.universityofgalway.ie/en/publications/detection-and-inpainting-of-facial-wrinkles-using-texture-orienta-3/)
- NessAiver 2013: LoG blob candidates → thresholding → dilation → morphological filtering, with a facial-feature mask so features are not taken for blemishes — [UMD DRUM](https://drum.lib.umd.edu/handle/1903/14774)
- Cho et al.: at high resolution pores and hair roots are false positives; a minimum-radius cut removes them — [paper PDF](https://people.csail.mit.edu/billf/papers/MMBIA_FinalV7.pdf)
- Frequency-separation practice puts the crossover radius "at which very fine details like skin pores and eyelashes blur" (Creative Bloq); Fstoppers found 2 px too small and 7.8 px already pulling tonal transitions into the high band on its test image; SLR Lounge: the radius "varies with the resolution of your camera, the resolution of the image, and how large the subject is in the frame", and an over-large radius causes the flat, unnatural look — [Creative Bloq](https://www.creativebloq.com/photography/retouch-images-frequency-separation-5132640); [Fstoppers](https://fstoppers.com/the-ultimate-guide-to-the-frequency-separation-technique); [SLR Lounge](https://www.slrlounge.com/the-many-faces-of-frequency-separation/)
- Retouch4me Heal treats "enlarged pores" as a defect to fix, alongside pimples and post-acne marks — [Retouch4me Heal](https://retouch4.me/heal)

### Inferences
- **Redness without CIELAB.** The input is linear AWG4 (docs/27 §1), white-balanced by `rgb_gain`. Converting to CIELAB needs AWG4→XYZ and a white point; the cheaper, exposure-invariant equivalent is a log-chromaticity `ρ = log(R/G)` compared with its local skin median/Gaussian. Red excess `Δρ > t_red` marks inflamed spots; a matching `Δlog G < 0` (haemoglobin absorbs green) separates real lesions from warm shadows, where all channels drop together.
- **Dark small spots: black top-hat in log space.** `T = log(closing_r(I)) − log(I)` per channel (or on G and on Y), r ≈ 2 % D. T is ≥ 0 and large only for dark features narrower than the disc. On G it fires for red spots too (they are dark in green), on all channels for brown spots. Thresholding T at a ratio (e.g. ≥ 6–10 % linear), then dropping components with area below ≈ 9 px (pores, noise) or above the disc, and elongated ones (fine lines, stray hairs), gives a spot mask with few parameters. DoG at σ matched to 0.7–3.5 % D is the alternative and gives a graded response.
- **Pores vs spots.** At 450 px, visible pores (0.25–0.5 mm) are ≤ 1.5 px and are averaged down by the area downscale; a minimum component area of ~3×3 px and a contrast threshold keep them. Pores therefore need no special model in the map; and because the final image is `original × upsampled smooth gain` (docs/27 §6.2), full-resolution pore texture survives everywhere the gain is 1 and is modulated, not erased, where it is not.
- **Threshold fragility** (Shen et al.) is real for global thresholds across photos. Here thresholds act after mean normalisation, in log ratios relative to the local skin of the same face, and the amount slider re-thresholds strength downstream, so a mis-set detection threshold degrades gracefully (weak spots fall below the slider's strength cut) rather than flipping results.
- **Face features.** The proposer will also "clean" nostrils, lash lines, lip corners and brow hairs (dark, compact). That is harmless only because docs/27 applies the blemish mask at step 5 (`g = 1` where `m == 0`); the skin mask must exclude them with margin.

### Gaps
- No source gives a validated red-excess or top-hat threshold for portrait photos; values above are starting points.
- No paper found that evaluates classical acne detection on ordinary (non-clinical) portrait photos with makeup and studio lighting.
- Behaviour on Fitzpatrick V–VI skin was not found in the sources read.

---

## Q4. Classical removal producing a "clean" image: methods, quality, artefacts, determinism, Rust fit

### Takeaway
For Yellowmoon the decisive fact is that the proposer's output is not shown: it becomes a per-channel gain `(OUT/IN)²`, clamped to 0.5..2.5, computed at ≤ 450 px and applied bilinearly to the full-resolution image (docs/27 §6.1–6.2). Texture therefore always comes from the original, and what the proposer must supply is a correct **low-frequency tone and colour** inside each spot and **exactly IN everywhere else**. That favours detect-then-fill-the-low-pass methods (frequency separation with a harmonic/membrane fill, or per-channel grey closing gated by a detection mask) over texture-synthesis inpainting (Criminisi, PatchMatch), whose copied texture would turn into a 450 px-scale texture pattern in the gain and ghost over the full-resolution texture. Edge-aware smoothers (bilateral, guided, domain transform) by design keep high-contrast spots, so they are skin smoothers, not blemish removers. All candidates can be deterministic; the best fits are also the simplest in Rust and on the GPU.

### Cited Findings
- **Exemplar inpainting** — Criminisi, Pérez & Toyama, "Region Filling and Object Removal by Exemplar-Based Image Inpainting", IEEE TIP 13(9) 1200–1212, 2004: best-first fill order where confidence propagates "in a manner similar to the propagation of information in inpainting", colour values from exemplar-based synthesis, block-based sampling for efficiency — [Microsoft Research PDF](https://microsoft.com/en-us/research/wp-content/uploads/2016/02/criminisi_tip2004.pdf); [MSR page](https://www.microsoft.com/en-us/research/?p=150053)
- **PatchMatch** — Barnes, Shechtman, Finkelstein, Goldman, ACM TOG (SIGGRAPH) 28(3) 2009: randomised correspondence; "some good patch matches can be found via random sampling"; 20–100× faster than prior art; basis for retargeting, completion and reshuffling — [Princeton project page](https://gfx.cs.princeton.edu/pubs/Barnes_2009_PAR/)
- **Fast marching inpainting** — Telea, "An Image Inpainting Technique Based on the Fast Marching Method", J. Graphics Tools 9(1) 2004 (DOI 10.1080/10867651.2004.10487596): treats the hole as a level set filled by FMM; "very simple to implement, fast, and produces nearly identical results to more complex, and usually slower, known methods"; exposed in OpenCV as `INPAINT_TELEA` — [paper PDF](https://webspace.science.uu.nl/~telea001/uploads/PAPERS/JGT04/paper.pdf); [OpenCV tutorial](https://docs.opencv.org/4.3.0/df/d3d/tutorial_py_inpainting.html)
- **Poisson editing** — Pérez, Gangnet & Blake, "Poisson Image Editing", ACM TOG 22(3) 2003 (SIGGRAPH), seamless cloning by solving a Poisson equation with Dirichlet boundary — [DOI 10.1145/882262.882269](https://doi.org/10.1145/882262.882269) (bibliographic; not fetched in this session)
- **Guided filter** — He, Sun & Tang (ECCV 2010; TPAMI 2013): "a fast and non-approximate linear time algorithm, regardless of the kernel size and the intensity range"; "better behaviors near edges" than the bilateral filter; "much faster and sometimes (though not always) works better" — [project page](https://people.csail.mit.edu/kaiming/eccv10/index.html)
- **Domain transform** — Gastal & Oliveira (SIGGRAPH 2011): "the first edge-preserving filter to work on color images at arbitrary scales in real time", cost "not affected by the choice of the filter parameters"; noisy inputs need robust derivative estimates (low-pass pre-filtering suggested) — [project page](https://www.inf.ufrgs.br/~eslgastal/DomainTransform/)
- **Facial blemish practice** — NessAiver 2013 compared four inpainting techniques for detected facial blemishes and chose one that "fills in pixels based on finding a patch in the nearby image region", with fill priority to strong edges (a Criminisi-type method) — [UMD DRUM](https://drum.lib.umd.edu/handle/1903/14774)
- Batool & Chellappa 2014 inpaint removed wrinkles and imperfections with exemplar-based constrained texture synthesis — [University of Galway record](https://research.universityofgalway.ie/en/publications/detection-and-inpainting-of-facial-wrinkles-using-texture-orienta-3/)
- A 2021 IEEE TETCI survey on wrinkle detection and inpainting tested flood-fill, Coherence Sensitive Hashing and exemplar inpainting and found flood-fill "preserved realism" (no significant difference in age prediction) — [unpaywall record](https://unpaywall.org/10.1109%2FTETCI.2021.3075723) (via search snippet)
- **darktable retouch** (GPL, read-only for us per AGENTS.md rule 1): tools clone, heal, fill, blur; heal: "the color and luma of the sample is blended to fit better with the surroundings"; wavelet decompose into detail scales plus residual, so one can "deal with skin blotches and blemishes on a coarse layer of detail" and keep fine texture; blur on a coarse scale evens tone — [darktable manual, retouch](https://docs.darktable.org/usermanual/development/en/module-reference/processing-modules/retouch/)
- **Frequency separation** (practice): split low (blur) and high (detail) at the radius where pores blur; heal tone/colour on the low layer, keep texture on the high; too large a radius causes the flat look — [Creative Bloq](https://www.creativebloq.com/photography/retouch-images-frequency-separation-5132640); [SLR Lounge](https://www.slrlounge.com/the-many-faces-of-frequency-separation/); [Fstoppers](https://fstoppers.com/the-ultimate-guide-to-the-frequency-separation-technique)
- docs/27 §1's own suggestion for the classical proposer: "a morphological closing and an edge-aware smoothing at blemish scale" — [docs/27 §1](../../../docs/27-skin-retouch.md)

**What already exists in this repository (checked by grep in the worktree):**
- f32 O(1)-per-pixel box mean, crop-bit-identical: `crates/ym-gpu/src/cpu/filters.rs` `box_mean`; f64 box mean and guided filter: `crates/ym-color/src/mask_raster.rs` `box_mean`, `guided`; const-generic guided filter (chroma): `crates/ym-gpu/src/cpu/detail.rs` `guided`, `chroma_guided_filter`; bilateral CPU `crates/ym-gpu/src/cpu/denoise.rs` `bilateral_whole`, `bilateral_opponent` and GPU `crates/ym-gpu/src/kernels/bilateral.rs`, `kernels/guided.rs`; u8 disc dilation and Gaussian (private) in `crates/ym-retouch/src/feather.rs` `dilate_disc`, `gaussian`.
- Not present: grey-level (f32) erosion/dilation/closing, image median filter, LoG/DoG pyramid, connected components, any inpainting (Telea, Poisson/harmonic, exemplar).

### Inferences

**Method-by-method, judged for docs/27 (quality on blemishes; artefacts; determinism; Rust/GPU):**

| Method | Removes blemishes? | Typical artefacts | Fit with the gain map | Determinism; Rust/GPU |
|---|---|---|---|---|
| Grey **closing** per channel, disc r ≈ 2 % D | Yes: any dark-in-a-channel feature narrower than the disc; red spots (dark in G/B) and brown spots (dark in all) alike | Flat plateaus and disc-shaped edges; brightens **all** dark texture smaller than the disc (pores, fine lines, stubble) unless gated; does nothing for bright spots | Good **if gated** by a spot mask, otherwise it writes a pore-scale brightening into the gain everywhere | Exact, order-free; trivial in Rust; separable/van Herk forms for square or octagon SEs; GPU-friendly |
| **Area / attribute closing** (max-tree) | Yes, removes dark components below an area regardless of shape, keeps long lines | Less plateau shape artefact than disc closing; still brightens pores unless a contrast (dynamics) floor is added | Good with a contrast floor | Deterministic; union-find max-tree is sequential (CPU only), fine on a 270 k-pixel map |
| **Median** filter, r ≈ blemish radius | Removes spots smaller than ~half the window | Rounds corners, removes thin lines and pore texture, cartoonish at large r | Acceptable on a low-pass layer only | Deterministic; cost grows with r² unless a histogram method on quantised data is used |
| **Frequency separation** + heal the low pass | Yes for tone/colour; leaves the blemish's own sharp detail (pustule tip, mole edge) in the high band | Residual ring if the crossover radius is too small; flat look if too large | **Best match**: with multiplicative separation `OUT = L′ · (IN/L)`, `OUT/IN = L′/L`, so the gain is by construction the low-frequency correction | Deterministic; Gaussian + per-pixel ops; GPU-friendly |
| **Harmonic / membrane fill** (Poisson with zero guidance) of L over the spot mask | Yes: smooth fill from the ring around the spot | No texture inside (irrelevant here: texture comes from the original via the gain); a dark neighbour on the boundary (protected mole, nostril) bleeds in | Excellent for L′ above; small sparse systems per spot | Deterministic with fixed iteration count and red–black order; GPU: Jacobi/multigrid |
| **Poisson seamless clone** (guidance from a source patch) | Yes | Needs a source patch choice; colour bleeding at boundary if source has different texture | Brings 450 px texture into the gain (see exemplar row) | Deterministic; Rust-easy |
| **Telea FMM** | Yes for small holes | Smearing/streaks along the fill direction for holes ≳ 10 px; blur | Fine for L′ on small spots; harmonic fill is simpler and smoother | Deterministic with heap tie-breaking; inherently sequential (CPU) |
| **Criminisi exemplar** | Yes, with texture | Wrong-source copying (another spot, a nostril rim, a pore cluster), seams, greedy error propagation | **Poor**: copied texture at 450 px differs from the original's, so `(OUT/IN)²` carries a 450 px texture pattern that upsamples onto the full-resolution texture (ghost/double texture), and the strength `|log2 g|` would rank texture mismatch as "strong blemish" | Deterministic with fixed scan and tie rules; sequential; slow |
| **PatchMatch** completion | Yes, with texture | As Criminisi, plus randomness | Same problem as Criminisi | Deterministic only with a seeded RNG and fixed propagation order (parallel variants change results); complex |
| **Bilateral / guided / domain transform** at blemish scale | Only low-contrast ones; a spot whose contrast exceeds σ_r (bilateral) or whose local variance exceeds ε (guided) is preserved as an "edge" | Plastic skin, texture loss; halos/gradient reversal for bilateral | Writes a global smoothing into the gain (that is the job of Even skin, docs/27 §7, not of Blemish) | Deterministic; already in the repo; GPU-ready |

- **Gate everything.** At amounts above 0.75, docs/27 §6.2 lerps `s` toward 1 for every pixel, and at `a ≈ 1` `s = clamp(alpha/0.01)`; so any tiny correction the proposer makes away from spots (pore smoothing, closing's brightening) is applied in full and upsampled into a soft, blotchy veil at full resolution. A classical proposer should output `OUT = IN` bit-exactly outside its detected spots.
- **Frequency-separation + harmonic fill is what darktable's "heal on a coarse wavelet scale" and the retouchers' "heal the low layer" do**, expressed in a form whose ratio is directly the gain docs/27 wants. Reading darktable to understand this is allowed; copying is not (GPL).
- **The blemish's own high band.** A whitehead tip or the crisp rim of a dark spot sits in `IN/L`; after the fill it remains as a faint ring or dot. Options: (a) choose the crossover σ ≈ 1–1.5 px at 450 (pore scale, below spot scale) so most spot energy is in L; (b) inside the spot core, compress `H = IN/L` toward 1 (e.g. clamp `log H` to ± a few %). Option (b) touches texture only inside spots.
- **Per-channel closing handles redness without a colour model** (red spots are dark in G and B), and the chroma guard (Q5) then normalises the result's colour to the neighbourhood, so a per-channel tone fill is enough; an explicit hue model is needed only for the beauty-mark decision, not for the fill.
- **OpenCV's Telea and Navier–Stokes inpainting** are permissively licensed in recent OpenCV (Apache-2.0 from 4.5, recalled, not re-verified here), so they may be read; GIMP's heal tool and G'MIC are GPL/CeCILL-type and darktable is GPL, read-only.

### Gaps
- No quantitative, published comparison of these methods on facial blemishes in portraits was found (NessAiver's thesis abstract names only the winner; the full PDF was not read).
- Pérez 2003 and the Photoshop Healing Brush literature were not fetched; descriptions above are standard but unverified in this session.
- OpenCV's current licence for the inpainting module was not re-checked in this session.

---

## Q5. Runtime at a 450×600 face crop on Apple silicon, and interaction with the gain-map and chroma-guard stages

### Takeaway
At 450×600 (270 k pixels, 810 k samples) every viable classical proposer is cheap: the recommended detect + frequency-separation + harmonic-fill design is on the order of **10–30 ms single-threaded, a few ms with rayon**, against docs/27's map being cached per face and independent of the amount. Bilateral at blemish radius and exemplar inpainting are the slow ones (≈ 0.1–1 s). The chroma guard turns any hue change larger than ~11 % (linear) into the neighbourhood's chroma, so a per-channel proposer need only get luminance right; the 0.5..2.5 clamp caps corrections at about +1.3 / −1 stop; and the strength ordering the amount slider relies on comes for free if the proposer corrects each spot fully.

### Cited Findings
- Guided filter and domain transform costs are independent of kernel size / filter parameters (linear time) — [guided filter page](https://people.csail.mit.edu/kaiming/eccv10/index.html); [domain transform page](https://www.inf.ufrgs.br/~eslgastal/DomainTransform/)
- PatchMatch's only published performance figure on its page is "20–100x" over prior art — [Princeton project page](https://gfx.cs.princeton.edu/pubs/Barnes_2009_PAR/)
- Telea's method is described as "fast" and simpler than comparable methods — [paper PDF](https://webspace.science.uu.nl/~telea001/uploads/PAPERS/JGT04/paper.pdf)
- docs/27 constants this interacts with: map size `k = 450 / max(1, face_px)`, never upscaled (§6.1 step 1); sqrt encoding and `0.5 / mean` normalisation (step 2); tiles of `output_size_px()` with `border_px()` edge-replicated context (step 3); chroma guard with `radius = max(w,h)/80`, `conf = clamp(1.1 − 10·(1 − min(r)/max(r)), 1e-5, 1)` on `r_c = ((OUT_c+1e-5)/(IN_c+1e-5))²`, `OUT_c = lerp(B_c·k, OUT_c, conf²)` with `B = box_blur(OUT·conf)` and `k` a luma ratio (step 4); gain `clamp((OUT/IN)², 0.5, 2.5)`, masked and border-faded (step 5); strength `max|log2 g|`, max filter r 3, box blur r 3 (step 6); apply `s = clamp(alpha / max(0.01, 1 − a), 0, 1)` with low/high-amount ramps (§6.2) — [docs/27](../../../docs/27-skin-retouch.md)
- The `BlemishProposer` trait passes only the planar tile (`run(&self, input: &[f32])`), no mask and no face size (§4) — [docs/27](../../../docs/27-skin-retouch.md)

### Inferences

**Order-of-magnitude runtimes** (my op-count estimates, not measurements; assume ~1–3 G simple f32 ops/s per Apple-silicon performance core in scalar Rust, more with auto-vectorisation; "‖" = rayon over ~8 cores; 3 channels unless noted):

| Stage | Estimate (1 core) | ‖ |
|---|---|---|
| Area downscale + sqrt + mean (docs/27 steps 1–2) | ~1–3 ms | < 1 ms |
| Box blur (running sums, existing `box_mean`) | ~2–5 ms | < 1 ms |
| Separable Gaussian σ ≈ 1.5 px | ~5–10 ms | ~1–2 ms |
| Grey closing, disc r ≈ 9 px, brute force (~250 taps × 2) | ~150–300 ms | ~20–40 ms |
| Grey closing, octagon/square via van Herk–Gil-Werman lines | ~5–10 ms | ~1–2 ms |
| Black top-hat + threshold + connected components (one channel max) | ~5–10 ms | (CC sequential) |
| LoG/DoG, ~6 scales on luminance | ~15–30 ms | ~3–5 ms |
| Median 11×11 by selection | ~150–300 ms | ~25–50 ms |
| Guided filter, grey guide / colour guide | ~5–10 / ~20–40 ms | ~1–2 / ~4–8 ms |
| Brute-force bilateral r ≈ 6–9 px | ~0.5–1.5 s | ~100–200 ms |
| Domain transform (RF, 3 iterations) | ~5–15 ms | ~2–4 ms |
| Harmonic fill of all spots (≈ 1–5 % of pixels, ~200 red–black Gauss–Seidel sweeps over spots only) | ~2–10 ms | ~1–3 ms |
| Whole-image Poisson (CG) / multigrid | ~0.3–1 s / ~20–50 ms | — |
| Telea FMM on spot pixels only | ~1–5 ms | sequential |
| Criminisi, local search window ~61×61, patch 9×9, spots ≈ 2 % | ~0.1–0.5 s | partly |
| PatchMatch completion on spots, multi-scale | ~50–200 ms | non-trivial to keep deterministic |
| docs/27 steps 4–7 (guard, gain, strength) | ~5–10 ms | ~1–2 ms |

GPU (later round): every per-pixel stage, closing (separable), Gaussian, top-hat, LoG and Jacobi harmonic fill map directly to kernels; connected components and FMM do not (run them on the CPU on the small map, or replace CC with an iterative label-propagation kernel with a fixed iteration count).

**Interaction with docs/27 stages:**
- **Gain clamp 0.5..2.5 (linear).** A spot whose linear ratio to skin is below 0.4 (e.g. a very dark scab) cannot be fully lifted, leaving a lighter residual; acne usually needs 10–40 % in G, well inside the clamp. Moles darker than 0.4 would also only half-heal, a second reason to protect them.
- **Chroma guard.** Since `r_c` is the squared sqrt ratio, i.e. the linear ratio, `conf = 1` up to ~1 % channel spread, ~0.6 at 5 %, ~0.1 at 10 %, floor at ≥ 11 %. A red-spot correction (G and B lifted more than R) therefore lands at the floor, and the guard replaces the spot's chroma with the conf-weighted box average of `OUT` over `radius ≈ 600/80 = 7 px`, keeping `OUT`'s luma. With `OUT = IN` outside spots, that average is the surrounding skin's chroma: the desired result. Consequences: (1) the proposer's own colour accuracy matters little; its **luminance** must be right. (2) A purely achromatic proposer (`OUT = IN · scalar`) passes the guard untouched (`conf = 1`) and cannot remove redness. (3) For spots wider than ~2 × 7 px, interior box windows hold only floor-weighted pixels, so `B·k` falls back to the proposer's own chroma; for large nodules the per-channel fill's colour matters again. (4) The box also averages non-skin `OUT` pixels (lips, brows) near the mask edge, because the guard runs before the mask (step 4 before step 5); keep spot detection off those regions with margin, or the guard pulls lip red into adjacent skin spots.
- **Strength and the amount slider.** `alpha ≈ max_c |log2 g_c|` per spot, so a classical proposer that fully corrects each detected spot yields the right ordering: deep spots (`|log2 g| ≈ 0.5–1`) survive low amounts, faint ones (`≈ 0.05–0.15`) appear only at high amounts. A proposer that corrects only partially (e.g. a closing truncated by a small disc) compresses that range and makes the slider feel abrupt. Optionally scale each spot's correction by a smooth confidence (detection margin), which feeds into `alpha` and lets dubious spots appear only at high amounts.
- **Beauty-mark holes.** They act at step 5 only; the proposer still "cleans" the mole in `OUT`, and the hole's √2 ramp applies part of that correction to the mole's rim. Hence the generous box margin (Q2), and the proposer should not use a protected mole as fill boundary for an adjacent spot (it cannot see the boxes: the trait has no mask input). Passing the protection to the proposer would need a trait change.
- **Tiles and scale.** The classical proposer can declare `border_px` = its largest operator radius (e.g. 32 like the trained net, or 2 × closing radius) and an `output_size_px` such as 256; the map is ≤ 450×~600 so it is one to six tiles. Detection near tile seams is identical if every operator only reads within the border (fixed-order sums keep determinism). The proposer needs the face size in map pixels (`min(450, face_px)`) to set radii; the trait does not pass it, so construct the classical proposer per face with that value (the alternative, inferring scale from the tile, is unreliable).

### Gaps
- No measured Rust timings at this size exist in the sources; the table is an op-count estimate and should be replaced by a `criterion` bench once the proposer exists (the repository already has CPU bench scaffolding in `crates/ym-gpu/benches/cpu_stages.rs`).
- The chroma-guard numbers assume `OUT` and `IN` near the mean-normalised 0.5 level; very dark skin regions (where `1e-5` and `1/256` floors matter) were not analysed.

---

## Q6. Recommendation for each part

### Takeaway
**(a) Beauty marks:** ship a deterministic classical `BeautyMarkDetector` (dark, compact, brown, isolated blobs at 1.2–4.5 % of face size, ranked by contrast, capped in count, boxes padded ~30 %), keep the Brush as the escape hatch, and do not train a learned detector: there is no licence-clean face data. **(b) Proposer:** ship a classical `BlemishProposer` that *detects* spots (per-channel black top-hat in log space plus a red-excess test, pores removed by a minimum area) and *fills only the low-frequency layer* inside them (multiplicative frequency separation with a harmonic fill), returning `OUT = IN` exactly elsewhere. Avoid texture-synthesis inpainting and edge-aware smoothing as the removal step.

### Cited Findings
- Commercial practice treats moles as their own class with a dedicated keep/remove control and keeps them at least partly by default (Adobe Fade default 50; Retouch4me Moles slider; Evoto pen restore) — [Adobe Learn](https://www.adobe.com/learn/lightroom-cc/web/ai-blemish-removal); [Camera Raw help](https://helpx.adobe.com/id_id/camera-raw/desktop/ai-and-generative-features/remove-blemishes-from-photos.html); [Retouch4me](https://retouch4.me/blog/retouch4me-apex-1-211-update-more-control-in-heal-with-the-new-moles-slider-and-keep-bindi); [Evoto](https://support.evoto.ai/blemish-removal/)
- Classical face-mole detectors in the literature are LoG/DoG candidate + pruning pipelines — [Cho et al.](https://people.csail.mit.edu/billf/papers/MMBIA_FinalV7.pdf); [Park & Jain](https://nij.ojp.gov/library/publications/face-matching-and-retrieval-using-soft-biometrics); [NessAiver](https://drum.lib.umd.edu/handle/1903/14774)
- Retouchers heal blemishes on the low-frequency layer and keep texture on the high one; darktable heals on a coarse wavelet scale — [Creative Bloq](https://www.creativebloq.com/photography/retouch-images-frequency-separation-5132640); [darktable manual](https://docs.darktable.org/usermanual/development/en/module-reference/processing-modules/retouch/)
- docs/27 already sketches "a morphological closing and an edge-aware smoothing at blemish scale" as the stand-in — [docs/27 §1](../../../docs/27-skin-retouch.md)

### Inferences

**(a) `ClassicalBeautyMarkDetector` (implements `BeautyMarkDetector`)**
1. Area-downscale the face crop to the map size (≤ 450 px), sqrt-encode and mean-normalise exactly as docs/27 §6.1 steps 1–2 (share the code), work in logs.
2. Restrict to the skin mask eroded by ~1.5 % D (eyes, brows, nostrils, lips, hairline out).
3. Candidates: DoG minima on log-luminance over 4–6 scales covering diameters 1.2–4.5 % D, persistent over ≥ 2 scales.
4. Accept if: contrast vs local background `≤ −c_min` (start ~ −15 % linear); brown, not red (`Δlog R / Δlog G` above a calibrated q, start ~0.4); circularity ≳ 0.7; aspect ≲ 1.6.
5. Rank by contrast × isolation; keep at most N (start 10); output boxes padded 30 % around the dark extent, scaled back to full-resolution image coordinates.
6. Deterministic: fixed scale list, raster-order tie-breaking, fixed-order reductions; no randomness. Cost ~20–40 ms single-threaded.
7. Tests: synthetic faces (the crate's mocks) with drawn brown discs, red discs, 1-px dot noise ("pores") and lines; brown discs found, red discs and pores not, lines not.
8. Calibrate thresholds on public portrait photos only, per the V1 data rule (`docs/test-photo-sources.md`); record the limitation in `docs/STATUS.md`.
- Expected behaviour to document: dark brown moles and most freckles kept; flesh-coloured dome naevi not detected (and mostly untouched by the proposer anyway); brown post-acne marks often kept; dense freckles capped. The Brush covers the rest.

**(b) `ClassicalBlemishProposer` (implements `BlemishProposer`), built per face with `face_map_px = min(450, face_px)`**
1. `L = Gaussian(IN, σ ≈ 0.25 % D ≈ 1–1.5 px)`, `H = IN / L` (per channel; floor the denominator).
2. Spot mask `M`: per-channel black top-hat in log space `T_c = log closing_r(L_c) − log L_c`, r ≈ 2 % D (octagon via van Herk–Gil-Werman); `M₀ = max_c T_c ≥ t_dark` (start ~0.08) **or** red excess `Δ log(R/G) ≥ t_red` with `Δ log G < 0`; drop components smaller than a disc of diameter ≈ 0.7 % D (≈ 3 px, i.e. pores and noise) or larger than the closing disc, and elongated ones (lines, hairs); dilate by 1–2 px; soften to a 0..1 weight by a smoothstep on the margin over threshold.
3. `L′` = harmonic fill of `L` over `M` from its boundary ring (red–black Gauss–Seidel, fixed sweep count, or a direct solve per small component), per channel.
4. Optional: inside the spot core, compress `log H` toward 0 (removes pustule tips and dark rims).
5. `OUT = IN · (1 + w · (L′/L − 1))` with `w` the soft mask; `OUT = IN` bit-exactly where `w = 0`.
6. `border_px = 2r + ~4` (covers closing, Gaussian and fill context), `output_size_px = 256`. Cost ~10–30 ms single-threaded, a few ms with rayon; all stages except connected components are per-pixel or small stencils, ready for GPU kernels.
- Why not the literal docs/27 sketch (closing + edge-aware smoothing, ungated): closing alone brightens all sub-disc dark texture, and edge-aware smoothing preserves exactly the high-contrast spots and smooths the low-contrast texture; ungated, both write a pore-scale veil into the gain that the high-amount ramp applies everywhere. Kept as an inner step (closing for detection) it is fine.
- Why not Criminisi/PatchMatch/clone: the gain stage keeps the original texture already; copied 450 px texture turns into a ghost texture in `(OUT/IN)²` and inflates strength.
- What a trained network would add later (round 2+, owner's yes needed): better recall on low-contrast and large lesions, fewer false fills on facial structure, learned colour; the classical proposer stays as the deterministic fallback and as the test oracle for the trait contract.
- Suggested spec edits for docs/27 (for the owner/maintainer): (1) widen the stated blemish scale from 0.3–1.5 % to ~0.7–3.5 % of D; (2) state that a proposer must return `OUT = IN` away from blemishes; (3) note that the classical proposer is constructed per face with the face size in map pixels, since the trait does not pass it.

### Gaps
- None of the proposed thresholds has been validated on real portraits; they need calibration on public photos and the owner's blind tests ([HUMAN] review).
- How well the harmonic low-pass fill hides large (> 4 % D) nodules or clustered acne is unknown; a clustered region may need a coarser L and a larger fill, which the gain clamp may cap.
- Whether the owner wants an explicit "Moles" control (as Retouch4me/Adobe offer) is a product decision not covered by the brief.
