# Face parsing and skin masks for the `FaceParser` trait (skin without ears, eyes, neck skin)

Scope: licence-clean ways to make the three soft masks of `docs/27-skin-retouch.md` §4 (`FacePart::SkinWithoutEars`, `Eyes`, `NeckSkin`, full image coordinates, faces up to ~4000 px), for weights that must pass `docs/14-licensing.md` rule 5 ("Shipped weights only from CC0, CC BY, CC BY-SA, Apache or our own data"; third-party weights are judged by their own licence, as SAM 2.1 and Depth Anything V2-Small were). Researched 2026-11-11. Run-time inference must be Rust (`ym-ml`) or Apple frameworks; Python only for export and evaluation.

What the repo already has (read in this session, no web source): `ym-ml` runs SAM 2.1 small and Depth Anything V2-Small in Rust; `ym-apple` calls `VNGenerateForegroundInstanceMaskRequest` (Subject) and `VNGeneratePersonInstanceMaskRequest` (People), macOS 14+, with a `FakeVision` for Linux tests; `docs/algorithms/06-masks.md` step 5 is a guided filter ("Refine edges", He, Sun and Tang 2010) with the Input's luma as guide, `r = round(ρ · 0.008 · u_px)`, `ε = 10^(−2 − 3ρ)`; `docs/27` §1 already proposes "masks from the mesh (the face oval minus eyes, brows and lips) and, for the neck, the People mask below the jaw" and notes CelebAMask-HQ rules out the common face parsers.

**A local experiment was run in this session** (Python, evaluation only, scratchpad, nothing committed): MediaPipe's `selfie_multiclass_256x256.tflite` (float32, 16,371,837 bytes, from `storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/`) and `face_landmarker.task` (float16, 3,758,596 bytes) through the `mediapipe` 1.2.2 Python package, on MediaPipe's own two test portraits (`storage.googleapis.com/mediapipe-assets/portrait.jpg`, 820×1024, face ≈192 px wide; `business-person.png`, 958×1358, face ≈251 px wide), each run on the full image and on a face crop. Mesh polygons (face oval, eyes, brows, lips) were rasterised and the multiclass labels counted inside them. Results are quoted below as "local experiment". Two images at small face sizes test what each class *contains*, not accuracy.

---

## 1. Face-parsing models and datasets: exact licences, what could ship, what we could train

### Takeaway
Every widely used face-parsing checkpoint (BiSeNet face-parsing.PyTorch, SegFormer face-parsing, SegFace, FaRL, RTNet/iBugMask, Sapiens) is either trained on non-commercial data (CelebAMask-HQ, LaPa, FaceSynthetics) or carries a non-commercial licence itself, so none can ship; at most they are local evaluation oracles. The one face-parsing dataset found that permits commercial use is **EasyPortrait** (40k consented crowd photos, a non-CC "attribution, conditions reserved" licence reworked from CC BY-SA 4.0), whose skin class explicitly excludes ears, nostrils and the second chin but has no neck class; a network we train on it would carry ShareAlike terms. Synthetic data works for face parsing (FaceSynthetics matched real-data training on Helen), but FaceSynthetics itself is non-commercial and its renderer is not released; Microsoft's permissive SynthHuman (CDLA-Permissive-2.0) has no part labels.

### Cited Findings

**Datasets**
- CelebAMask-HQ: 30,000 images, 512×512 masks, 19 classes including skin, eyes, brows, ears, neck; "The use of this software is RESTRICTED to non-commercial research and educational purposes" and "The CelebAMask-HQ dataset is available for non-commercial research purposes only." — [CelebAMask-HQ README](https://github.com/switchablenorms/CelebAMask-HQ)
- LaPa: >22,000 images, 11-category label maps; "made freely available to academic and non-academic entities for non-commercial purposes" — [LaPa README](https://github.com/JDAI-CV/lapa-dataset)
- FaceSynthetics (Microsoft, "Fake It Till You Make It"): 100,000 synthetic images at 512×512, 19 classes + ignore (background, skin, nose, left/right eye, left/right brow, left/right ear, mouth interior, top/bottom lip, **neck**, hair, **beard**, clothing, glasses, headwear, facewear); "The Face Synthetics dataset can be used for non-commercial research." No generator or assets are offered, only the images. — [FaceSynthetics README](https://github.com/microsoft/FaceSynthetics)
- FaceSynthetics results: a UNet with ResNet-18 encoder trained on synthetic data only scored Helen F1 skin 95.1 (same as the same network trained on real Helen, 95.1) and overall 92.0 (real-data training: 91.6); a separate "label adaptation" UNet, trained only on pairs of predicted and ground-truth label maps (never on real images), corrects systematic label differences such as the jawline; "training with synthetic data alone produces comparable results to training with real data." — [Wood et al. 2021, arXiv:2109.15102](https://arxiv.org/abs/2109.15102)
- A 2023 study attributes over 50 % of the synthetic-to-real gap in face parsing to the distribution gap (not photorealism) and says that after addressing distribution and label gaps a synthetic-trained model matches one trained on a similar amount of real data. — [Friedman et al., arXiv:2303.15219](https://arxiv.org/abs/2303.15219v1)
- **EasyPortrait**: v2.0 (2023-11-13) has 40,000 images (≈38.3K FullHD; 30,000 train / 4,000 val / 6,000 test, 91.78 GB, 13,705 unique persons); v1 (2023-02-23) 20,000 images. Classes: background, person, skin, left brow, right brow, left eye, right eye, lips, teeth (no ears, no neck, no hair). README: "licensed under a variant of Creative Commons Attribution-ShareAlike 4.0 International License." No licence is stated for the code or the released weights (MMSegmentation 0.30.0 based). Best face parsing: SegFormer-B0 at 1024×1024, 85.42 mIoU (README lists 14.9 M parameters); FPN + ResNet50 at 1024, 85.37. — [EasyPortrait README](https://github.com/hukenovs/easyportrait)
- EasyPortrait licence text (`license/en_us.pdf`, read in full): titled "Public license with attribution and conditions reserved"; footnote: "This license is not a Creative Commons license. The text of this license is a reworking of a Creative Commons Corporation (Attribution-ShareAlike 4.0) license … under the terms of the CC0". It grants a worldwide, royalty-free licence to reproduce and make available the material and to create Adapted Material; Adapted Material must carry "a license with the same License Elements, this version or a later version, or a BY-SA-compatible license"; no non-commercial clause; publicity/privacy rights "are not granted"; database rights are defined under "the Civil Code of the Russian Federation"; termination on breach with 30-day cure. — [EasyPortrait licence PDF](https://github.com/hukenovs/easyportrait/blob/main/license/en_us.pdf)
- EasyPortrait paper (VISAPP 2025): images collected and annotated through Toloka and ABC Elementary crowdsourcing; "all crowd workers signed a consent form allowing us to process and publish their photos"; 96 % FullHD, minimum 720×720; annotation rule: "ears, second chin, and nostrils are not included in the skin class", "the rare bristle also considered skin", "beard with a strict border are considered occlusion", "the skin class should affect only skin without hair, eyes, and other face attributes"; limitations: "designed for one concrete domain — video conferencing", bias towards Caucasian subjects and Russia; the authors urge users to share work "under the same license as the original" and plan to add "neck", "beard", "hair", etc. The paper itself is published CC BY-NC-ND 4.0 (this is the article's licence, not the dataset's; one search summary confused the two). — [EasyPortrait, VISAPP 2025 PDF](https://www.scitepress.org/Papers/2025/131353/131353.pdf)
- iBugMask / RTNet (Lin et al. 2021): 11-class map where "skin" includes face **and scalp**, 14-class map adds left/right ear and glasses; code listed MIT; no licence or terms stated for the iBugMask dataset or the weights. — [hhj1897/face_parsing](https://github.com/hhj1897/face_parsing)
- Microsoft SynthHuman (DAViD, ICCV 2025): 298,008 synthetic samples (98,040 face, 99,976 full body, 99,992 upper body) with foreground alpha, absolute depth, normals and intrinsics, **no face-part or skin labels**; "The SynthHuman dataset is licensed under the CDLA-2.0"; "DAViD models and runtime code are licensed under the MIT License"; models released as ONNX (soft foreground segmentation, depth, normals). — [microsoft/DAViD](https://github.com/microsoft/DAViD); Hugging Face mirror labels it "CDLA - Permissive - 2.0" — [Voxel51 card](https://docs.voxel51.com/dataset_zoo/datasets_hf/synthhuman.html)
- SynthForge (controllable 3D-generated face dataset) is CC BY-NC-SA 4.0. — [search summary of SynthForge, arXiv:2406.07840](https://arxiv.org/html/2406.07840) (licence seen only in a search summary; not opened)
- An extended search found no 2025–2026 face-parsing dataset under CC BY/CC0 with skin, ear and neck classes. — (negative result of the search; see Gaps)

**Models / weights**
- face-parsing.PyTorch (BiSeNet): code MIT; the README's training section says "Train the model using CelebAMask-HQ dataset"; the released checkpoint has no licence statement. — [zllrunning/face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch)
- `jonathandinu/face-parsing` (SegFormer fine-tuned from `nvidia/mit-b5`, 84.6 M parameters, 19 CelebAMask-HQ labels): card restricts use to "non-commercial research and educational purposes". — [Hugging Face card](https://huggingface.co/jonathandinu/face-parsing)
- SegFace (AAAI 2025): code MIT; results and checkpoints on CelebAMask-HQ, LaPa (and a Helen checkpoint `swinb_helen_512`); backbones ConvNeXt, EfficientNet, MobileNet, ResNet100, Swin-B, SwinV2-B; mean F1 on CelebAMask-HQ 87.50–89.22 at 512; MobileNet 87.91 at ~96 FPS; weights licence not stated. — [Kartik-3004/SegFace](https://github.com/Kartik-3004/SegFace)
- FaRL: code MIT; pretrained on LAION-Face ("the human face subset of LAION-400M", 20 M of ~50 M pairs used); parsing configs for CelebAMask-HQ and LaPa at 448×448; no weights licence stated. — [FacePerceiver/FaRL](https://github.com/FacePerceiver/FaRL)
- Sapiens (Meta): licence file is "Creative Commons Attribution-NonCommercial 4.0 International Public License"; 0.3B–2B models at 1024×1024; its face-segmentation fine-tuning recipe uses the FaceSynthetics 19 classes. — [Sapiens LICENSE](https://raw.githubusercontent.com/facebookresearch/sapiens/main/LICENSE); [Sapiens SEG_README](https://github.com/facebookresearch/sapiens/blob/main/docs/finetune/SEG_README.md)
- Human parsing (LIP, CIHP; SCHP): no upstream licence found in this session; a third-party Space quotes SCHP as "only open for Non-commercial Use" and SCHP's code as MIT (Peike Li, 2020). LIP's 19 classes have "Face" but no neck. — [search results incl. LIP paper arXiv:1703.05446](https://arxiv.org/pdf/1703.05446) (licence unverified)

### Inferences
- **Shippable today: no learned face parser.** Everything with face-part classes traces to CelebAMask-HQ, LaPa, Helen/iBugMask, FaceSynthetics, LAION-Face or a CC BY-NC licence. These may be used only as local evaluation oracles (e.g. to score our masks), never bundled, never used to label training data for shipped weights (pseudo-labels from a non-commercial model would launder the data; `docs/14` rule 5 lists the allowed sources).
- **Trainable by us, cleanly: EasyPortrait** is the only real face-parsing set found with commercial use allowed, consent documented, and a skin class that already matches `SkinWithoutEars` (ears, nostrils, second chin out; eyes, brows, lips, teeth separate). Weights trained on it would be "Adapted Material" under a BY-SA-style licence, so under rule 5's spirit they ship as CC BY-SA 4.0 data files in `data/models/<name>/` with their own `LICENSE`. Because the licence is *not* a CC licence (a Russian-law reworking), rule 5's text ("CC BY-SA") does not literally cover it: **an owner decision**. Missing: neck (only "person"), beard (labelled occlusion), and the domain is webcam selfies, not 24–61 MP raw portraits.
- **Effort to train on EasyPortrait (estimate, not sourced):** a SegFormer-B0-class network (the README's best face-parsing model) at 512–1024 px on 30k images is a single-GPU job of the order of a day; plus the Rust port of a SegFormer (simpler than SAM's Hiera trunk) and parity tests; plus the 92 GB download through `tools/fetch/`. It needs the owner's training budget (`docs/27` §1: "training on Modal needs the owner's yes").
- **Synthetic route:** proven in principle (Helen F1 parity), but the only face-parsing synthetic set (FaceSynthetics) is non-commercial and its asset library is not released; building our own renderer (head models, hair, clothing, labels per pixel) is months of work and outside V1's "no measurements, public data" scope. SynthHuman (CDLA-Permissive-2.0) could at most train a person/foreground matte, which Vision and the multiclass model already give.

### Gaps
- Helen dataset terms (Flickr images, Le et al. 2012) and iBugMask terms were not found.
- DML-CSR and EHANet: licences and training sets were not checked in this session; by their papers' benchmarks they are trained on CelebAMask-HQ/LaPa/Helen, so they fall in the same non-commercial bucket (unverified).
- EasyPortrait weights published in the repository have no stated licence; whether they count as Adapted Material of the dataset is unclear. Training our own avoids the question.
- Whether EasyPortrait v2.0 added classes beyond the nine was not confirmed (the README lists nine).
- No permissive (CC BY/CC0/Apache) face-parsing dataset from 2025–2026 was found; absence of evidence, not proof.

---

## 2. MediaPipe selfie multiclass and hair segmenter; Apple Vision

### Takeaway
MediaPipe's **Selfie Multiclass** (Apache-2.0, 256×256, classes background, hair, body-skin, face-skin, clothes, others) gives a usable face-skin region and, separately, **neck as body-skin**, but its face-skin class *includes* the eyes, brows, lips, teeth **and ears** (local experiment), so it must be combined with mesh polygons. The **Hair Segmenter** (Apache-2.0, 512×512) is a selfie-only model. Apple Vision has no face-skin, neck or face-part segmentation for arbitrary images: its person/foreground masks are whole-person, its face landmarks trace only cheek-to-chin, and the skin/hair/teeth mattes exist only as auxiliary data captured by Apple cameras.

### Cited Findings
- MediaPipe Image Segmenter models: SelfieSegmenter 256×256 / 144×256 (background, person); **HairSegmenter 512×512** (background, hair); **SelfieMulticlass 256×256**: "0 background 1 hair 2 body-skin 3 face-skin 4 clothes 5 others (accessories)"; DeepLab-V3 257×257. The guide page states no licence. — [MediaPipe Image segmentation guide](https://developers.google.com/edge/mediapipe/solutions/vision/image_segmenter)
- Multiclass model card: "LICENSED UNDER Apache License, Version 2.0"; Vision Transformer "with customized bottleneck and decoder"; inputs "[256, 256, 3] for smaller model and [512, 512, 3] for larger model"; output a [256×256×6] (or 512×512×6) probability tensor; model date May 10, 2023; "may not provide pixel perfect masks"; out of scope: "applications that require pixel perfect masks"; mean IoU 81.10 (512 model) and 77.23 (256 model) on 1,902 instances; worst skin-tone group (Monk 9–10, 31 instances) 71.86 / 68.25; training data described only as "thousands of samples with diverse identities … various blendshape combinations". — [Multiclass model card PDF](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20Multiclass%20Segmentation.pdf)
- Hair segmentation card: "Licensed under Apache License, Version 2.0", 2019 model, encoder-decoder; "not suitable for people too far away from the camera (e.g. further than 5 feet/1.5 meters) or multiple people"; thin long hairstyles and headwear "may not be reliably segmented"; regional IoU ≈ 80–86 %. — [Hair Segmentation model card PDF](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20-%20Hair%20Segmentation.pdf)
- **Local experiment** (models from the URLs above, images `portrait.jpg` and `business-person.png` from `storage.googleapis.com/mediapipe-assets/`): inside the mesh's eye, brow and outer-lip polygons the multiclass label was **face-skin 100 %** in all four runs (two images × full/face crop); teeth of an open smile were face-skin; in the two ear zones just outside the face oval the label was face-skin 73–89 % where an ear was visible (portrait: 83 %/89 %; business person 73 % on the visible ear); a box under the chin was **body-skin 83–87 %** where the neck was visible (business person) and clothes 85–89 % where a collar covered it (portrait); face-skin ran up to the hairline and excluded hair; 79–82 % of face-skin pixels lay inside the mesh face oval, 9–11 % above the oval's top landmark (10), the rest at the ears and temples. Hands were body-skin too. — [model](https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite), [test image 1](https://storage.googleapis.com/mediapipe-assets/portrait.jpg), [test image 2](https://storage.googleapis.com/mediapipe-assets/business-person.png)
- A third-party project builds face-skin masks from the multiclass output and subtracts eye, lip and brow masks made from Face Mesh landmarks (an example, not an official recipe). — [HardikUppal/ColorPalette segmentation_utils.py](https://huggingface.co/spaces/HardikUppal/ColorPalette/blob/main/src/segmentation_utils.py)
- Apple `VNGeneratePersonSegmentationRequest` (macOS 12+): "produces a matte image for a person it finds", with a `qualityLevel` setting; no part classes. — [Apple docs](https://developer.apple.com/documentation/vision/vngeneratepersonsegmentationrequest)
- Apple `VNFaceLandmarks2D` (macOS 10.13+): regions faceContour, leftEye, rightEye, leftEyebrow, rightEyebrow, nose, noseCrest, medianLine, outerLips, innerLips, leftPupil, rightPupil; **faceContour traces "from the left cheek, over the chin, to the right cheek"** (no forehead). — [Apple docs](https://developer.apple.com/documentation/vision/vnfacelandmarks2d)
- `AVSemanticSegmentationMatte` (hair, skin, teeth, later glasses): "The image file contains the semantic segmentation matte as an auxiliary image, accessible using the ImageIO framework's CGImageSourceCopyAuxiliaryDataInfoAtIndex"; its initialiser is "from auxiliary image information in an image file"; matte types are those "you can capture along with the primary image". — [Apple docs](https://developer.apple.com/documentation/avfoundation/avsemanticsegmentationmatte); introduced at WWDC 2019 session 260 for captured photos — [WWDC19 260](https://developer.apple.com/videos/play/wwdc2019/260/)

### Inferences
- **Face-skin class alone is not `SkinWithoutEars`:** it must be cut by mesh polygons (eyes, brows, lips, nostrils) and clipped laterally to drop the ears. It *does* supply what the mesh lacks: the forehead up to the real hairline and a hair boundary.
- **Body-skin is a good neck source** across platforms (Rust port), but it also includes hands, arms and chest: keep only the connected component touching the jawline and clip it to a band below the chin.
- Resolution: 256×256 over a face crop means one model pixel spans ~16–24 full-resolution pixels for a 4000 px face (arithmetic); the mask is a coarse prior that the guided filter of `docs/algorithms/06` must pull to edges.
- Only the 256 model appears on the download page; the card's 512 model was not found as a download (see Gaps).
- **Apple Vision cannot be the `FaceParser`:** no skin/neck/part masks for an ARW-derived image; `AVSemanticSegmentationMatte` is capture-only auxiliary data (ARW files carry none). Vision's landmarks could give eye/lip polygons on a Mac but not the forehead, and a Vision-only route would make Linux `ym` output differ (hard rule 6 already tolerates this for People, but the retouch masks would be better cross-platform).
- MediaPipe weights are Apache-2.0 by their cards; training data is Google's own and undisclosed, the same situation as BlazeFace noted in `research/scene-understanding-2026-10-19.md` and the SAM 2.1 / Depth Anything precedent (weights judged by their own licence, owner accepted 2026-11-07). Worth stating in the model card.

### Gaps
- Whether the 512×512 multiclass model was ever published (the card mentions it; the guide lists only 256×256).
- How the multiclass model labels beards, stubble and moustaches (hair vs face-skin) was not tested (no bearded test image used).
- The multiclass training data is not described beyond "thousands of samples … blendshape combinations"; whether synthetic or real is unknown.
- Whether macOS 26-era Vision added any face-part segmentation was not checked; Apple documentation found shows none.

---

## 3. Masks from a dense face mesh (MediaPipe 478 points), refined at full resolution

### Takeaway
MediaPipe Face Landmarker (Apache-2.0, 478 points from a 256×256 crop) gives exact *topology* for eyes, brows, lips, nose and the face oval, so eye/brow/lip/nostril cut-outs and the `Eyes` mask are best made from it; but its oval stops mid-forehead and runs inside the temples (≈20 % of real face skin lies outside it), it cannot see beards, bangs or the neck, and its single-image error (~3.2 % of interocular distance) is tens of pixels on a 4000 px face, so the polygons must be softened and snapped to image edges, and the outer boundary should come from a segmenter, not the oval.

### Cited Findings
- Face Mesh V2 card: input "Image of cropped face with 25% margin on each side, and resized to 256x256"; output "478 3D landmarks"; MobileNetV2-like CNN; "LICENSED UNDER Apache License, Version 2.0"; model date September 15, 2022; not suitable for faces "looking away from the camera (more than 80°)", "inclined from the vertical orientation (more than 8°)", "only partially visible (less than 50% of the face)", or too small to rescale to 192×192 "without quality degradation"; training images "captured on a diverse set of smartphone cameras … via an AR application". — [Face Mesh V2 model card PDF](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20MediaPipe%20Face%20Mesh%20V2.pdf)
- Same card: human-annotator discrepancy 2.56 % IOD MAE; reacquisition mode (a single image with a detector, our case) 3.24–3.25 % IOD MAE by gender, 3.09–3.75 % by skin tone, 2.67–3.85 % by region; tracking mode 2.06–3.27 %. — [Face Mesh V2 model card PDF](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20MediaPipe%20Face%20Mesh%20V2.pdf)
- The MediaPipe Python API (1.2.2, this session) exposes contour sets `FACE_LANDMARKS_FACE_OVAL`, `LEFT_EYE`, `RIGHT_EYE`, `LEFT_EYEBROW`, `RIGHT_EYEBROW`, `LEFT_IRIS`, `RIGHT_IRIS`, `LIPS`, `NOSE`, `CONTOURS`, `TESSELATION`; the eye contours are two lid paths sharing the corner vertices (e.g. 33 and 133). — local experiment with [mediapipe on PyPI](https://pypi.org/project/mediapipe/)
- **Local experiment**: the oval's top vertex (landmark 10) sits mid-forehead, well below the hairline, in both test images; 9–11 % of the multiclass face-skin lay above it and a further ~10 % at temples and ears outside the oval; the oval's sides (landmarks 234/454) follow the cheek–ear junction, so ears fall outside it. — [model](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task), [test images](https://storage.googleapis.com/mediapipe-assets/portrait.jpg)
- Vision's own contour omits the forehead entirely ("from the left cheek, over the chin, to the right cheek"). — [Apple VNFaceLandmarks2D](https://developer.apple.com/documentation/vision/vnfacelandmarks2d)
- The guided filter can be subsampled for an O(N/s²) cost, "a speedup of >10x with almost no visible degradation". — [He & Sun, Fast Guided Filter, arXiv:1505.00996](https://arxiv.org/abs/1505.00996)
- `docs/27` §5 builds every derived mask with feathering proportional to `F` (face size) and integer `u8` arithmetic, and `docs/27` §1 already plans "the face oval minus eyes, brows and lips" and "the People mask below the jaw". — repo file `docs/27-skin-retouch.md` (read this session)

### Inferences
- **Error scale at full resolution:** interocular distance is roughly 0.4–0.45 of face width, so 3.2 % IOD is ~50–60 px on a 4000 px-wide face (arithmetic on the card's figure; contour points are likely worse than the average). Polygons should therefore be (a) built on the 2D projection at full resolution with sub-pixel vertices, (b) dilated for cut-outs (eyes, brows, lips) by a few % of `F` so lashes and lip edges stay out of the skin, and (c) feathered with `docs/27` §5.1 and pulled to edges with the guided filter (Refine edges) at the face's scale.
- **Hairline:** the oval misses the upper forehead (and the scalp on bald heads, which arguably should be retouched too); bangs below landmark 10 are included as "skin". A segmenter's face-skin (MediaPipe multiclass) or hair mask fixes both; a per-face skin-colour model (section 4) is the fallback.
- **Beard and stubble:** the mesh covers bearded areas as face; a hair/beard-aware class or a colour/texture test is needed. FaceSynthetics shows that a "beard" class is learnable, but no clean dataset labels it (EasyPortrait calls strict-bordered beards "occlusion").
- **Ears:** handled well by the oval (they lie outside it); a segmenter's face-skin must be clipped by the oval's lateral extent below the brow line to drop ears while keeping the forehead.
- **Eyes mask:** the mesh's eye contours (plus iris points) are the most reliable source of `FacePart::Eyes`; `docs/27` §5.4 only feathers it outward by 0.03 F to protect the eye from the dark-circle lift.
- **Neck:** the mesh gives only the jaw line (oval landmarks 132→152→361), which is the natural upper boundary for a neck region taken from body-skin, the People mask or a colour model grown downward.
- **Roll and profile:** the card's 8° roll limit means the crop must be rotated to level the eyes (from the detector's eye keypoints) before the landmark model; beyond ~80° yaw the mesh is unreliable and the tools should skip the face.

### Gaps
- No published accuracy of the MediaPipe face-oval contour against a parsing ground truth was found.
- Landmark accuracy on very large (4000 px) faces after downscaling to 256 px was not measured.

---

## 4. Classical skin-colour segmentation and skin datasets

### Takeaway
Global colour rules (YCbCr/HSV thresholds, a Bayes histogram) are clearly weaker than CNNs on generic skin datasets (average F1 0.63 vs 0.71–0.89) and fail on skin-coloured backgrounds, coloured light, makeup and dark or aged skin; none of the public skin datasets states a licence, and two were scraped from Google Images, so none can train shipped weights. Used narrowly, as a **per-face model fitted on skin sampled from the mesh**, colour is still a useful soft multiplier at mask edges (hairline, beard, glasses), not a mask on its own.

### Cited Findings
- Survey table of skin datasets (sizes, ground truth, availability): Compaq 4,675 (not available), ECU 4,000 precise (link marked unavailable), Schmugge 845, Pratheepan 78 precise, VDM 285, SFA 1,118 medium precision, HGR 1,558 precise, SDD 21,000 (not available), ABD 1,400 precise (GitHub); "Pratheepan images were randomly downloaded from Google" and ABD images were collected via Google image search; the article gives no licence terms for any dataset (its CC BY 4.0 covers the article only). — [Nanni et al., "A Standardized Approach for Skin Detection", J. Imaging 2023 (PMC9965146)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9965146/)
- Same survey, failure modes: colour clustering works only where ethnicity and background are known; skin tone ranges and ageing; lighting changes and shadows ("highly challenging"); tattoos and makeup; skin-coloured backgrounds; ground-truth errors on lips and mouths. — [PMC9965146](https://pmc.ncbi.nlm.nih.gov/articles/PMC9965146/)
- Same survey, F1-style scores averaged over eight datasets: colour Bayes (2002) 0.630; spatial SA3 (2014) 0.652; U-Net 0.706; SegNet 0.734; 2020 method 0.781; 2021 method 0.883; proposed ensemble 0.892; on HGR the colour Bayes (0.871) beats U-Net (0.836); on VMD Bayes scores 0.252. — [PMC9965146](https://pmc.ncbi.nlm.nih.gov/articles/PMC9965146/)
- Earlier fair comparison listing Pratheepan, ECU, SFA, HGR with sizes and hosting. — [Lumini & Nanni, arXiv:1802.02531](https://arxiv.org/pdf/1802.02531)

### Inferences
- **Per-face adaptation removes the main weakness.** The failure modes above come from one global model across people, cameras and lights. Sampling the face's own skin from interior mesh triangles (cheeks, forehead below landmark 10, chin; away from eyes, lips, nostrils and the oval's edge) and fitting a 2–3 component Gaussian mixture in a chroma space gives a likelihood that adapts to tone, white balance and the scene. In Yellowmoon's scene-linear AWG4, log-chroma (log R/G, log B/G) is intensity-invariant, so shading and the curve do not move it (inference, not tested).
- **Where it still fails:** blond or red hair and light beards on fair skin (similar chroma), lips (that is why lips come from the mesh), skin-coloured clothing and wood or sand behind the head, strong coloured light split across the face, specular highlights (chroma collapses to the illuminant), heavy foundation that differs from the neck. So: use it as a soft multiplier inside the segmenter/mesh region and for growing the neck downward, never alone.
- **Datasets:** with no licence stated and Google-scraped sources, the skin datasets are evaluation-only at best; a per-face model needs no training data at all, which is its main attraction here.
- Determinism: an EM fit must be seeded and summed in a fixed order (`AGENTS.md` rule 8); a closed-form single Gaussian (mean and covariance of sampled pixels) avoids EM entirely.

### Gaps
- The YCbCr/HSV threshold literature and GMM numbers (e.g. Jones & Rehg 2002, Kakumanu 2007 survey) were not opened in this session; the survey above gives no GMM figures.
- No source evaluated per-face mesh-sampled skin models against parsing ground truth; that would be our own evaluation.
- Exact licences of Pratheepan, ECU, SFA, HGR, ABD were not found on their own pages (not visited).

---

## 5. SAM 2.1 (already shipped) prompted by mesh points for face skin and neck

### Takeaway
SAM with point prompts is poor at face parts: with one centre point on CelebAMask-HQ, SAM ViT-H scored skin 71.4 IoU, neck 33.9, brows ~23–25, upper lip 8.6, mean 35.1 over 18 parts, because it "predicts the most probable contour containing the prompt" and facial parts share skin's colour and texture. SAM 2.1 small is unlikely to do better; it is not a primary route for skin or neck, though a box-plus-points prompt might be tried as an optional refinement.

### Cited Findings
- Facial-part segmentation with frozen SAM (ViT-H), single foreground point at each mask's centre, highest-confidence mask, CelebA(Mask)-HQ, 18 parts: skin 71.37, nose 12.62, left/right eye 50.18/55.83, left/right brow 24.82/22.71, left/right ear 44.93/47.97, upper lip 8.57, lower lip 20.35, mouth 55.85, hair 20.35, neck 33.91, mean 35.05; a prompt-learning adaptation trained on the dataset reached mean 71.67 (skin 84.95, neck 51.59). Explanation: parts such as nose and ears have "similar color and texture to the 'skin'", and SAM "predicts the most probable contour containing the prompt, rather than aligning with a specific user intention." — [Task-Specific Adaptation of Segmentation Foundation Model via Prompt Learning, arXiv:2403.09199](https://arxiv.org/html/2403.09199v2)
- SAM predicts multiple masks per ambiguous prompt (whole, part, subpart). — [Kirillov et al., Segment Anything, arXiv:2304.02643](https://arxiv.org/pdf/2304.02643)
- SAM 2.1 small weights are Apache-2.0 and already ship in Yellowmoon (`ym_ml::sam2`, NOTICE item 10). — repo `docs/14-licensing.md` (read this session)

### Inferences
- **Face skin:** a centre point on the cheek most likely returns the whole face *including* eyes, brows and lips (the "part" level) or the whole head/person; the mesh cut-outs would still be needed, and the result would be no better than the multiclass face-skin at a far higher cost (SAM's 1024 px trunk per face).
- **Neck:** 33.9 IoU with a single point is too weak. A box from the jaw to ~one face-height below, positive points under the chin and negative points on the collar and jaw might do better, but no source measured it; would need our own evaluation set.
- **Possible role:** as an optional "Refine" for the person silhouette around the neck where Vision's People mask is unavailable (Linux), or to separate a hand touching the face; not as the `FaceParser`.
- The adapted model's gains (35 → 72 mean) came from training on CelebA-HQ, which we cannot use; the same adaptation on EasyPortrait is conceivable but is the same training-budget question as section 1.

### Gaps
- No study of SAM 2 / 2.1 (as opposed to SAM 1 ViT-H) on face parts was found; SAM 2.1 small was not tested here.
- Multi-point or box prompts for skin/neck were not evaluated in any source found.

---

## 6. Recommended route, and why

### Takeaway
Round 2 should build the `FaceParser` from **two Apache-2.0 MediaPipe models run in Rust**, the Face Landmarker mesh (eyes, brows, lips, nostrils, oval, jaw line) and the Selfie Multiclass segmenter on a face crop (face-skin for the forehead and hairline, body-skin for the neck), combined with integer mask arithmetic, a per-face skin-colour multiplier, and the existing guided filter at full resolution. No training is needed and every weight is Apache-2.0. A learned face parser trained by us on EasyPortrait is the upgrade path, gated on the owner's training budget and on accepting its ShareAlike-style, non-CC licence.

### Cited Findings
- Multiclass face-skin contains eyes, brows, lips, teeth and ears; body-skin holds the neck (local experiment, section 2). — [model](https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite)
- Both MediaPipe cards state Apache-2.0. — [Multiclass card](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20Multiclass%20Segmentation.pdf); [Face Mesh V2 card](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20MediaPipe%20Face%20Mesh%20V2.pdf)
- EasyPortrait's skin class excludes ears, nostrils and second chin and its licence permits commercial use with ShareAlike-style terms. — [EasyPortrait paper](https://www.scitepress.org/Papers/2025/131353/131353.pdf); [licence PDF](https://github.com/hukenovs/easyportrait/blob/main/license/en_us.pdf)
- All common face-parsing checkpoints rest on non-commercial data or licences (section 1). — [CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ); [LaPa](https://github.com/JDAI-CV/lapa-dataset); [FaceSynthetics](https://github.com/microsoft/FaceSynthetics); [Sapiens LICENSE](https://raw.githubusercontent.com/facebookresearch/sapiens/main/LICENSE)

### Inferences

**Comparison (dense)**

| Route | SkinWithoutEars | Eyes | NeckSkin | Licence of what ships | Resolution / quality | Cost |
|---|---|---|---|---|---|---|
| CelebAMask-HQ / LaPa / FaceSynthetics-trained parsers (BiSeNet, SegFormer-B5, SegFace, FaRL, RTNet, Sapiens) | yes (ears, neck separate classes) | yes | yes (CelebAMask-HQ, FaceSynthetics) | **cannot ship** (NC data or NC licence) | 448–1024 px, best quality | eval oracle only |
| Our parser trained on **EasyPortrait** | yes (ears, nostrils, second chin already out) | yes | no class (person only) | ShareAlike-style, non-CC; owner decision | up to 1024 px, 85.4 mIoU (SegFormer-B0) | training budget + SegFormer port |
| **MediaPipe Selfie Multiclass** (256) | partial: includes eyes, brows, lips, ears | no | yes, as body-skin (with hands, chest) | Apache-2.0 | 256 px, IoU 77 (card) | ViT port to `ym-ml` |
| **MediaPipe Face Landmarker** mesh | partial: oval misses upper forehead, temples; includes beard, bangs | **yes** (lid contours, iris) | no (jaw line only) | Apache-2.0 | 256 px crop, ~3.2 % IOD error | CNN port + BlazeFace or Vision face rects |
| Apple Vision (People/Subject, landmarks) | no (landmarks: cheek-to-chin only) | yes (Mac only) | People mask below jaw, includes clothes | OS framework | Mac only; Linux `ym` differs | already wired for People |
| Per-face skin-colour model from mesh samples | soft trim only | no | grows neck from jaw | our code | full resolution | small |
| SAM 2.1 small with mesh-point prompts | weak (skin IoU 71, SAM 1 data) | weak | weak (neck 34) | Apache-2.0, already shipped | 1024 px trunk | already shipped; poor fit |

**Proposed construction (round 2), per face, in full image coordinates**
1. **Face rect and roll:** Vision face rectangles on Macs or BlazeFace (in the `.task` bundle) in Rust; rotate the crop so the eyes are level (card: ≤8° roll).
2. **Mesh:** Face Landmarker on a 256×256 crop with 25 % margin; map the 478 points back to full resolution.
3. **Segmenter:** Multiclass on a square crop of ~2× face size (face plus neck), 256×256; upsample its soft face-skin and body-skin probabilities to the crop at full resolution.
4. **SkinWithoutEars** = face-skin × (oval ∪ forehead band) − dilated polygons of eyes, brows, lips and nostrils (from the `NOSE` contour), where the forehead band is the region above the brow line between the oval's temple vertices: this keeps the forehead up to the segmenter's hairline and drops the ears, which lie outside the oval below the brows. Optionally × the per-face skin-colour likelihood near the boundary (beard, bangs, glasses).
5. **Eyes** = union of the two lid polygons, dilated slightly (docs/27 §5.4 feathers it again).
6. **NeckSkin** = the body-skin component connected to the jaw line (oval vertices from 132 through 152 to 361), clipped to a band from the jaw to ~0.8 face heights below and ~1.3 oval widths across, minus the face mask; on Macs it may be intersected with the People mask; optionally grown or trimmed by the skin-colour model where the segmenter is unsure.
7. **Full-resolution edges:** feather per `docs/27` §5.1, then the guided filter of `docs/algorithms/06` step 5 with luma guide, radius scaled to the face (not the image), so the 256 px priors land on the real jaw, hairline and collar edges.
8. **Evaluation only:** score against a CelebAMask-HQ-trained parser (SegFace or BiSeNet) on public CC0 portraits locally; never ship or distil from it.

**Why this route**
- It is the only one where **every shipped weight is Apache-2.0 today**, with no training, no new data and no owner budget; it fits `docs/14` precedent (third-party weights judged by their own licence) and `docs/27` §1's plan.
- The two models are complementary in exactly the places each fails: the mesh gives topology (eyes, brows, lips, nostrils, jaw, the ear boundary) that the segmenter merges into face-skin; the segmenter gives the hairline and the neck that the mesh cannot see.
- It runs the same in Rust on Linux and macOS (deterministic, rule 8; CLI parity, rule 6), unlike a Vision-only route.
- Small: the multiclass model is 16.4 MB in float32 (~4 M parameters, ~8 MB in f16) and the landmarker bundle 3.8 MB (float16), against SAM 2.1 small's 77 MB.
- Its known weak spots (beards and stubble, bangs, light hair on light skin, 256 px priors) are soft-edge problems that the colour multiplier and the guided filter address, and the Brush protection row of `docs/27` §1 covers the rest by hand.
- **Upgrade path:** train a SegFormer-B0-class parser on EasyPortrait (skin already excludes ears and nostrils) and keep the mesh for eyes and the multiclass body-skin for the neck; ship its weights as CC BY-SA 4.0 data files under rule 5, once the owner accepts the non-CC licence and the training cost.

**For the owner (decisions this research surfaces)**
- Accept MediaPipe Face Landmarker and Selfie Multiclass weights (Apache-2.0, Google's undisclosed own training data) as shipped weights, like SAM 2.1 and Depth Anything.
- Whether EasyPortrait's licence ("not a Creative Commons license", BY-SA rework under Russian law) counts as CC BY-SA for rule 5, if a trained parser is wanted later.

### Gaps
- The proposed combination has not been evaluated on real high-resolution portraits; quality at the hairline, on beards and on dark skin (the multiclass card's weakest group, IoU 68.25 for Monk 9–10 at 256 px) needs our own test set built from CC0/public-domain portraits.
- The Rust port effort for the multiclass ViT and the landmarker CNN, and the TFLite-to-safetensors export path, were not scoped here.
- No measurement of how the multiclass model behaves on a face crop of a 24–61 MP image downscaled to 256 px versus its intended selfie framing.
