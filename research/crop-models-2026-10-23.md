# Lightweight learned croppers (2026-10-23)

The owner, after the held-out test of Auto crop's classical proposer (`docs/STATUS.md`): "I see crop composition is pretty bad, cutting objects, no thirds rule. Stop this approach and search for a lightweight ML model we can use for composition and cropping."

## Answer

Good lightweight croppers exist, and one has exactly the shape Yellowmoon needs. **None can ship as is**, because all were trained on crop datasets that carry no licence (GAICD, CPC, FCDB) and fail `docs/14` rule 5. The way through is the one the owner already chose for scene understanding on 2026-10-18 (`research/scene-understanding-2026-10-19.md`): train our own small model on clean images, with labels from an Apache-2.0 model run offline. Here that means a FocalNet-sized cropper of about 1–5M parameters. Its labels come from FrameLite, an Apache-2.0 cropper run offline, and from GenCrop-style self-supervision on Wikimedia Commons' well-composed photos.

## Candidates

| Model | What it is | Size | Licence of weights | Training data | Verdict |
|---|---|---|---|---|---|
| **Appwrite FocalNet** ([card](https://huggingface.co/appwrite/focalnet), [code](https://github.com/appwrite/focalnet)) | RepViT-M0.9 encoder predicting a 64×64 importance map, plus a 7,069-parameter composition head that ranks up to 125 candidate crops (5 positions × 5 zooms) for a requested aspect ratio. Among crops within 0.05 of the best retained importance, it picks the highest preference score. | 4,757,694 parameters, 19.45 MiB FP32 ONNX, 256×256 input; INT8 failed its quality gate | MIT (model files and training code) | Importance labels on 500k Open Images V7 photos, made by U²-Net plus YuNet teachers. Ranking head pretrained on CPC, fine-tuned on GAICD. The card says "CPC and GAICD archives do not state explicit image or annotation licenses". GAICD test: Spearman 0.765, top-5 43.7 % (test split inspected during development). | **The right shape, the wrong data.** YuNet was trained on WIDER FACE (CC BY-NC-ND); the backbone is timm's ImageNet RepViT. Evaluation only; its MIT training code is reusable. |
| **FrameLite** ([card](https://huggingface.co/zlab-sjtu/FrameLite), [paper](https://arxiv.org/abs/2610.17639)) | Qwen3.5-0.8B fine-tuned to answer one crop directly (greedy decoding, no candidate search) | 224 MB LoRA and vision merger, plus the 0.8B base; the README asks for a CUDA BF16 GPU | Apache-2.0 (base Qwen3.5-0.8B also Apache-2.0) | GAIC, CPC and FCDB for crops; RefCOCO/RefCOCO+ for localisation | **Best accuracy found.** GAICv2 Acc₁/₅ at 0.85: 0.914; FLMS IoU 0.868; FCDB IoU 0.712, ahead of 7B VLM croppers. Too large and non-deterministic to ship, but an Apache-2.0 model, so an **offline teacher** under the owner's 2026-10-18 rule. |
| ProCrop ([card](https://huggingface.co/BWGZK/ProCrop)) | Retrieval-augmented Conditional DETR | 44.8M trainable parameters, 512 MB | Apache-2.0 | CPC, and AVA images for retrieval | Not lightweight; data not clean. |
| GAIC ([code](https://github.com/HuiZeng/Grid-Anchor-based-Image-Cropping-Pytorch), [paper](https://www4.comp.polyu.edu.hk/~cslzhang/paper/CVPR19-GAIC.pdf)) | Grid-anchor candidate scorer, MobileNetV2 or ShuffleNetV2 | under 2.5M parameters; 200 FPS on a GTX 1080 Ti, 12 FPS on an i7 | MIT code; weights on Google Drive | GAICD (no licence), ImageNet backbone | Architecture reference only. |
| Twitter saliency cropper ([code](https://github.com/twitter-research/image-crop-analysis)) | Fast DeepGaze-style gaze predictor | — | Apache-2.0 analysis code | — | No weights found in the repository's listing. |
| GenCrop ([paper](https://arxiv.org/abs/2312.12080)) | Weakly supervised: outpaint professional photos, then learn to crop back to the original frame. CACNet-style model. | small CNN plus transformer encoder | — | Unsplash photos, Stable Diffusion outpainting | **A recipe, not a model.** Its authors call it competitive with supervised croppers using no crop annotations. A clean version is possible (below). |

## The clean route

- **Images.** Open Images V7: images "listed as having a CC BY 2.0 license", annotations CC BY 4.0, with author fields for attribution (scene-understanding research). Wikimedia Commons: 469,984 Quality images and 22,613 Featured pictures (Commons category counts, 2026-10-23), reviewed for composition, under free licences (CC BY, CC BY-SA, CC0). Any CC BY-SA image makes the weights CC BY-SA 4.0, kept in `data/models/<name>/` with their own `LICENSE`, as rule 5 sets out.
- **Labels, two sources.**
  1. FrameLite run offline on the images for each aspect (Original, 1:1, 4:5, 16:9). This works like the owner's 2026-10-18 teacher rule, but FrameLite was itself fine-tuned on the unlicensed crop sets, which the general VLMs that rule had in mind were not. **The owner decides whether the rule extends to it.**
  2. GenCrop-style pairs: outpaint Commons Featured and Quality pictures with Qwen-Image-Edit (Apache-2.0), and the photographer's own frame becomes the target crop. Nothing trained on research data enters this chain.
- **Student.** FocalNet's layout (importance map plus a tiny ranking head over candidate crops), started from scratch or from formula-driven pretraining, never from ImageNet weights. About 1–5M parameters, safetensors in `data/models/`. It runs in `ym-ml`'s CPU runtime on the 768 px analysis picture, so the app and `ym` give the same answer. Composition, the thirds included, is learned from the teacher's and the photographers' frames rather than written as a rule. A hard constraint like FocalNet's retained-importance tolerance keeps it from cutting subjects.
- **Cost (estimates, not measured).** FrameLite labelling on a GPU: about a second per image per aspect, so tens of GPU-hours for 100–200k images. Outpainting with a 20B editor is the expensive part, roughly 10–20 s per image, so 150–300 GPU-hours for 50k images. Training the student: a few GPU-hours. A GPU is needed; the container has none.
- **Gate.** Blind pairs on the 150 held-out Play Raw photos (`testdata/cropeval/heldout/`), against the centred frame and against the photographer's framing, before anything ships.

## Evaluation on 150 unseen photos (2026-10-23, evaluation only)

The two candidates ran on the 150 held-out Play Raw photos (`testdata/cropeval/heldout/`, the same previews the classical proposer was tested on), in the scratchpad: nothing was installed in the repository and no weights were kept. Blind pairs, judged by the agent, then unblinded; a tie counts half.

| | Compared with | Pairs that differ | Won | Lost | Tied | Share | Speed (CPU, this container) |
|---|---|---|---|---|---|---|---|
| **FocalNet**, Square and 16:9 | the largest centred frame of the aspect | 162 of 300 | 39 (25 clearly) | 24 (10 clearly) | 99 | 55 % | 67 ms a crop, 4 threads, FP32 ONNX |
| **FrameLite**, its own free crop | the whole photo | 134 of 150 | 50 (13 clearly) | 4 (none clearly) | 80 | 67 % | 8 s a photo, 4 threads, FP32 |
| Classical proposer v3, Square and 16:9 (for scale) | the largest centred frame | 59 of 300 | 21 | 18 | 20 | 52 % | 20 ms |

- **FocalNet** moves the frame three times as often as the classical proposer and still wins more than it loses. Its wins are the classical proposer's failures: it holds what the centred frame cuts (a car's nose and wheel, a chef's head, a moon the centred 16:9 loses). Its losses are a subject pushed against the edge (a dog's head), and twice framing sky and moon instead of the scene below (a road at dusk).
- **FrameLite** almost never makes a photo worse. It tightens loose frames (median crop area 75 % of the photo), puts a rider on a third, and turns a night skyline into a panorama without the empty foreground water. It kept the whole photo, or nearly, on 16. Its four losses were all slight. Caveat: these pairs are only partly blind, because the uncropped photo usually keeps the camera's 3:2 shape.
- Neither is shippable (above). But on photos neither had seen, both beat the classical proposer, and FrameLite's taste is good enough to be the teacher for our own model. 4:5 was run and not judged.

## Next step

The learned route is worth paying for. Distil a FocalNet-sized student (about 5M parameters, about 70 ms a crop on a CPU) from FrameLite's crops and from GenCrop-style outpainted pairs, on Open Images and Commons photos, and gate it on these held-out photos against FrameLite itself and against the photographer's framing. This waits on the owner's questions (a) and (b) in `docs/STATUS.md` (22).

## Pilot: a student from FrameLite's labels (2026-10-23)

The owner answered (a) and (b): FrameLite may label our images offline, and the GPU is Modal, with a hard $30 limit. The pipeline is `training/crop/`, started from GitHub Actions.

- **Labels.** FrameLite's free crop of 12,000 Open Images photos (CC BY 2.0; 11,999 usable). It took 2.82 s a photo on an L4, against the 1 s assumed above, so labels cost about $0.95 per 1,000 photos, and 100–200k photos would take 80–160 GPU-hours, more than guessed above. The median label keeps 84 % of the photo's area.
- **Student.** A MobileNetV2-style network of 1.1M parameters, trained from scratch to regress the crop box from a 256 px letterboxed photo. One run: 10,827 photos to train, 1,172 to validate, 40 epochs in 10.2 minutes, about $0.3.
- **Agreement with FrameLite (IoU).** Validation: student 0.861, whole photo 0.830, mean crop 0.796. It reached 0.856 by epoch 5; the training loss kept falling while validation stayed flat. On the 150 held-out Play Raw photos: student 0.822, whole photo 0.751, mean crop 0.755.
- **Blind pairs** on the 150 held-out photos against the whole photo, as for FrameLite above: the student moved the frame on 100, won 31 (1 clearly), lost 4 (none clearly), tied 65. That is 64 %, against FrameLite's 67 %. Its median crop keeps 83 % of the photo, FrameLite's 75 %.
- **Reading.** A small clean model does learn the direction of FrameLite's taste: it trims loose edges and never makes a photo clearly worse. It does not learn FrameLite's bolder reframing, so most of its moves are too small to judge. The single learning-curve point leaves open whether more labels or a different student fix that. Regressing one box may average towards a gentle crop; FocalNet's layout (an importance map and a score per candidate crop) would not, and trains on the same labels.
- **Cost.** About $16.5 of the $30, of which about $5 was wasted when the orchestrator restarted and labelled some chunks twice (`docs/STATUS.md`). Next choices: `docs/STATUS.md`, question 23.

## Also considered

Apple's Vision framework (attention saliency, subject masks, face, human and animal detection, horizon) needs no shipped weights, but only on Macs. Per the owner it waits until it is needed.
