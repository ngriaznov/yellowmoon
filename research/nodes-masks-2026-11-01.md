# Nodes and masks: licences, Apple Vision and the Resolve 21 Photo page (2026-11-01)

Background for `docs/23-nodes.md` and `docs/algorithms/06-masks.md`. It records what was read, where, and what could not be read. It describes the world; it holds no instructions. Where a source is a summary by someone else it says so. Dates are the days the pages were read (2026-11-01).

## 1. SAM 2 and SAM 2.1: the licence

**Source: the repository's README** (`https://raw.githubusercontent.com/facebookresearch/sam2/main/README.md`, read 2026-11-01). The licence sentences, verbatim:

> "The SAM 2 model checkpoints, SAM 2 demo code (front-end and back-end), and SAM 2 training code are licensed under Apache 2.0, however the Inter Font and Noto Color Emoji used in the SAM 2 demo code are made available under the SIL Open Font License, version 1.1."

(The sentence is one line in the README; the fonts belong to the demo, which Yellowmoon does not use.)

**Source: the repository's `LICENSE`** (`https://raw.githubusercontent.com/facebookresearch/sam2/main/LICENSE`): the Apache License, Version 2.0, January 2004, the standard text. It has no copyright holder line other than the appendix's template (`Copyright [yyyy] [name of copyright owner]`).

**The 2.1 checkpoints** (README, "The table below shows the improved SAM 2.1 checkpoints released on September 29, 2024." Speed "measured on an A100 with `torch 2.5.1, cuda 12.4`"):

| Checkpoint | Size (M) | Speed (FPS) |
|---|---|---|
| `sam2.1_hiera_tiny` | 38.9 | 91.2 |
| `sam2.1_hiera_small` | 46 | 84.8 |
| `sam2.1_hiera_base_plus` | 80.8 | 64.1 |
| `sam2.1_hiera_large` | 224.4 | 39.5 |

The file names are `sam2.1_hiera_tiny.pt`, `sam2.1_hiera_small.pt`, `sam2.1_hiera_base_plus.pt` and `sam2.1_hiera_large.pt`. The README does not say that "Size (M)" counts parameters, and does not give file sizes in bytes. The speed column is video frames per second on an A100, not our image-only CPU path.

**The small configuration** (`sam2/configs/sam2.1/sam2.1_hiera_s.yaml`, read the same day): trunk `Hiera` with `embed_dim: 96`, `num_heads: 1`, `stages: [1, 2, 11, 2]`, `global_att_blocks: [7, 10, 13]`, `window_pos_embed_bkg_spatial_size: [7, 7]`; neck `FpnNeck` with `d_model: 256`, `backbone_channel_list: [768, 384, 192, 96]`, `fpn_top_down_levels: [2, 3]`, `fpn_interp_model: nearest`, `scalp: 1`; `image_size: 1024`, `use_high_res_features_in_sam: true`, `multimask_output_in_sam`, `pred_obj_scores: true`, `directly_add_no_mem_embed: true`. The page the reader returned did not show the `sam_mask_decoder_extra_args` (the dynamic multimask stability values 0.05 and 0.98 used in `docs/algorithms/06` §6.1 come from memory of Meta's configurations and are to be checked against the checkpoint's configuration by N-17).

**Preprocessing and prompts** (`sam2/utils/transforms.py`): `ToTensor`, then `Resize((1024, 1024))` with torchvision's defaults (no interpolation or antialias argument is passed), then `Normalize` with mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`; point coordinates are divided by the image's width and height when asked to and multiplied by the resolution. **Image predictor** (`sam2/sam2_image_predictor.py`): `set_image` adds the learned `no_mem_embed` to the lowest-resolution feature map when `directly_add_no_mem_embed` is set; the feature sizes are `(256, 256), (128, 128), (64, 64)`, the last being the image embedding and the first two the high-resolution features for the decoder; a box is two points labelled 2 and 3; the decoder returns low-resolution logits clamped to ±32 and the predicted IoUs; the mask threshold is 0.0. The torchvision version that the default `Resize` antialiasing depends on was not pinned by these files.

**The paper:** N. Ravi et al., "SAM 2: Segment Anything in Images and Videos", arXiv:2408.00714 (2024). Not opened in this session (arXiv could not be reached); cited from memory.

## 2. SAM 3: the licence

SAM 3 (`github.com/facebookresearch/sam3`) is **not Apache-2.0**.

**Primary source:** the repository's `LICENSE` (`https://raw.githubusercontent.com/facebookresearch/sam3/main/LICENSE`), titled "SAM License", "Last Updated: November 19, 2025", defining the "Agreement" and the "SAM Materials" (Documentation, models, software and algorithms). Its main terms as the reader summarised them (the full text was not reproduced here):

- use restrictions: compliance with laws, privacy rules and sanctions; no ITAR-controlled activities or prohibited end uses such as military or warfare, nuclear, espionage or weapons;
- redistribution: anyone receiving the materials or derivatives gets them under the agreement, with a copy of it; published research that uses them must acknowledge it;
- no reverse engineering, and no encouraging others to;
- termination by Meta on breach (delete and stop using), and termination of the licence if the licensee sues Meta claiming the materials infringe its IP; the licensee defends and indemnifies Meta against third-party claims from its use or distribution;
- Meta may modify the agreement, changes take effect at once and continued use is acceptance;
- warranty and liability disclaimers; California law.

The repository's README says: "This project is licensed under the SAM License - see the LICENSE file for details." and gives the model size as 848 million parameters.

**Secondary sources** (search results, 2026-11-01, not the licence text): Roboflow's "Segment Anything 3 Model Licensing and Pricing" summary (it says it is a summary, not legal advice) and the ScanCode LicenseDB entry for `sam-2025-11-19` describe the same terms: royalty-free, commercial use allowed, with the conditions above. The licence text on Hugging Face (`huggingface.co/facebook/sam3`) was not opened (the site is not reachable from this session).

**Reading for Yellowmoon:** the licence restricts uses and can be changed by its author, so it is not compatible with the rule that shipped code and weights are Apache-2.0 compatible (`docs/14` rule 5). The owner decided on 2026-11-01 that SAM 3 is out and SAM 2.1 is the model.

## 3. Apple Vision: the requests the plan uses

Read from Apple's documentation data (`https://developer.apple.com/tutorials/data/documentation/vision/<name>.md`, 2026-11-01); the pages' availability metadata and text, verbatim where quoted:

| Symbol | Availability in the page's metadata | Apple's one-line description |
|---|---|---|
| `VNGenerateForegroundInstanceMaskRequest` | iOS 17.0, iPadOS 17.0, Mac Catalyst 17.0, **macOS 14.0**, tvOS 17.0, visionOS 1.0 | "A request that generates an instance mask of noticable objects to separate from the background." (sic) |
| `VNGeneratePersonInstanceMaskRequest` | the same, **macOS 14.0** | "An object that produces a mask of individual people it finds in the input image." |
| `VNInstanceMaskObservation` (the result type of both; `results: [VNInstanceMaskObservation]?`) | the same, **macOS 14.0** | "An observation that contains an instance mask that labels instances in the mask." |
| `VNGeneratePersonSegmentationRequest` | iOS 15.0, **macOS 12.0** | "An object that produces a matte image for a person it finds in the input image." (one mask for the people together; `qualityLevel`, `outputPixelFormat`) |

The observation's members: `allInstances: IndexSet` ("The collection that contains all instances, excluding the background."), `instanceMask: CVPixelBuffer` ("The resulting mask that represents all instances."), `generateMask(forInstances:)` ("Creates a low-resolution mask from the instances you specify."), `generateMaskedImage(ofInstances:from:croppedToInstancesExtent:)` ("Creates a high-resolution image where everything becomes transparent black, except for the instances you specify.") and `generateScaledMaskForImage(forInstances:from:)` ("Creates a high-resolution mask where everything becomes transparent black, except for the instances you specify."). These are the macOS 14 symbols; the plan's "Subject" is the foreground instance request and "People" the person instance request.

**Not read or not found:** a third-party reference says the person instance request gives a separate mask for up to four people; Apple's page does not say how many. The pixel format of the scaled mask buffer is not stated on these pages. Apple's pages do not describe an API that says whether a Mac can run a request without running it; the spec's probe therefore runs each request once on a small picture (`docs/23` §3.10). Whether the foreground instance request runs on Intel Macs is not stated on these pages. Apple's page data is copyright 2026 and read as published; nothing of it is copied into the product.

## 4. `@xyflow/react` (React Flow): the licence

Read from GitHub on 2026-11-01. The repository `github.com/xyflow/xyflow`: "React Flow and Svelte Flow are MIT licensed." Its `LICENSE` starts "MIT License", "Copyright (c) 2019-2025 webkid GmbH". `packages/react/package.json` on the main branch: name `@xyflow/react`, version `12.12.0`, `"license": "MIT"`, dependencies `@xyflow/system` (a workspace package of the same repository), `classcat` `^5.0.3` and `zustand` `^4.4.0`; peer dependencies `react`, `react-dom` and their types, `>=17`. The dependencies' own repositories: `classcat` 5.0.5 `"license": "MIT"`, `zustand` 5.0.16 `"license": "MIT"` (main branches; the range `^4.4.0` that `@xyflow/react` asks for was not looked up separately). The npm registry was not read, so the published version and its tarball's licence file were not checked here.

## 5. DaVinci Resolve 21's Photo page

What was and was not read. **The sources named for this note were not opened:** Blackmagic's New Features Guide (the site is not reachable from this session), Newsshooter's two articles, PetaPixel's article and ProVideo Coalition's article (their hosts do not resolve from this session). The facts below come from **search-result summaries of those pages and other outlets** (which quote them), and from the notes of the mockup round of 2026-11-01 (`design/nodes/mockup-notes.md`, which say that their sources "are reviews and Blackmagic's announcement as quoted by them; their sites were not read"). They are second-hand.

- **Release.** Resolve 21 was announced in April 2026 (NAB) with a new **Photo page**, and one result reports it out of beta on June 3, 2026; the notes say 21.1 was due in September 2026.
- **What the Photo page is.** It brings the Color page's node-based grading to still photos, with native raw editing for Sony, Canon and other cameras, tethered shooting with Sony and Canon cameras, an album with a filmstrip along the bottom, and masking, Magic Mask among it.
- **Its Inspector** (ProVideo Coalition, as quoted by a search summary): four tabs, **Raw**, **Photo** ("cropping, transforms, and additional color correction"), **Effects** and **File** ("file metadata, adding comments, color-coding"). "Shared nodes" can apply one look across an album.
- **How it meets the Color page's nodes** (ProVideo Coalition, as quoted): "adjustments and effects added in the Photo page will show up as the first node", and those adjustments cannot be altered in the Color page; the same outlet says corrections made in the Color page show on the Photo page but cannot be tweaked there. One walkthrough describes sending an image from the Photo page to the Color page to build a full node tree with serial nodes, parallel nodes, qualifiers and power windows. **The sources disagree** on whether the Photo-page edits stay editable on the Color page.
- **The Color page's node editor** gets a layer-list view of the nodes (from the mockup notes; not re-read).
- **Magic Mask** works on stills and can be rendered in place as a matte (summaries); one source puts Magic Mask in the Studio edition and another says most Photo page features are in the free version, with Magic Mask and Film Look Creator in Studio.
- **For Yellowmoon** (a reading, not a source fact): the simple view writing the Input, node 1 and the Output matches "Photo page adjustments appear as the first node", and unlike Resolve's those stay editable in the node view. The list view was dropped by the owner. Shared nodes across photos and presets are not planned.

## 6. Measurements made while writing the spec

From the repository at `bd51a9d` (the code and data the spec's choices rest on):

- **`white_pick`'s frame.** `Renderer::pick_point_at` (`crates/ym-core/src/pipeline/frame.rs`) maps a click through the geometry to a point "as fractions of the photo as the camera oriented it, over the image area **before the lens crop**" (it applies the lens's `source_position` back to the uncorrected sensor image). That is not "after the lens geometry", which is how the plan's words describe the frame of masks.
- **The locus tables.** The `locus.xy` tables of the 11 shipped camera profiles (`data/cameras/profiles/*/profile.json`) are identical (the same SHA-256 of the JSON), and so are the `alexa_white` tables. The daylight part of `locus.xy` differs from the CIE daylight polynomial (`x_D = −4.6070e9/T³ + 2.9678e6/T² + 0.09911e3/T + 0.244063` up to 7000 K, `−2.0064e9/T³ + 1.9018e6/T² + 0.24748e3/T + 0.237040` above, `y_D = −3x_D² + 2.870x_D − 0.275`) by at most 1.7e-4 in `x` and `y` over 4907 K to 9634 K (checked at eight temperatures; the table's value at 6504 K is (0.312652, 0.328962), the polynomial's (0.312714, 0.329119), D65 (0.3127, 0.3290)).
- **GPU bindings today.** `crates/ym-gpu/src/kernels/color.rs` and `shaders/color.wgsl`: group 0 has one uniform (1664 bytes) and five storage buffers (the dither tables, the curves, the look LUTs, the picture in and out); `docs/17` K3 allows 8 storage buffers and 1 uniform.
- **The shipped models** run as own pure-Rust CPU runtimes (`ym_color::illum`, `ym_ml::crop_net`, `ym_ml::bokeh_net`); `bokeh_net`'s determinism notes (f32, fixed order, no fused multiply-add, `libm`, thread-count independent) are the pattern the SAM runtime follows.
- **Papers cited from memory** (not opened): K. He, J. Sun and X. Tang, "Guided Image Filtering", ECCV 2010 (IEEE TPAMI 35(6), 2013); K. He and J. Sun, "Fast Guided Filter", arXiv:1505.00996 (2015).
