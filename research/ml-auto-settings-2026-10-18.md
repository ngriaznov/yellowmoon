# Predict a professional's settings, not pixels

Yellowmoon can beat industry Auto on everyday photos by teaching a small model to do what a professional photographer does when opening a file: recognise what the photo is, then choose a handful of settings for it. Those settings are **exposure, temperature and tint, white lift and a bounded contrast**, and Yellowmoon's fixed pipeline renders them. No published model can ship for this under Yellowmoon's licence rules, because nearly every useful one was trained on MIT-Adobe FiveK, MSEC or PPR10K, which are research-only or non-commercial. So Yellowmoon trains its own value predictor, `autoset`, an MLP of 30–70k parameters on scene-linear brightness quantiles, clip statistics, `illum`'s light estimate and class, EXIF and faces. Scene understanding, the owner's north star, comes from **SceneNet**, a small encoder of our own: 1–4M parameters, 2–8 MB in f16, a few milliseconds on the CPU. It is trained from scratch on CC BY images, with scene labels from Apache-2.0 vision-language teachers run offline. It hands `autoset` an interpretable vector of 16 scene probabilities (snow, sunset, backlit portrait, tungsten interior and so on), which shifts the settings from professional priors per scene. The outputs are corrections to today's rules (the +1.11 EV camera gain, `illum`'s white, the current lift, the look's contrast). Each carries an uncertainty that pulls it back to today's behaviour when the model is unsure. The labels come from fitting Yellowmoon's own settings to licence-clean references: Sony camera JPEGs in CC0 ARWs, Google's HDR+ finals, Cube++'s neutral and camera-AWB pairs, and Play Raw. Exact known-offset self-supervision adds more. One finding needs saying plainly: semantic input is proven for faces and local effects, but no study shows it improves *global* exposure on raw files. SceneNet therefore ships only if a per-scene ablation shows it helps and no scene category gets worse. "Punchy and professional" is open ground. Lightroom Auto's documented failures are a flat HDR-like look, lifted shadows and oversaturation, and Adobe's newer Adaptive profile deliberately avoids "opinionated" contrast. Lower error against one expert proves nothing, because FiveK's five experts already disagree by about **0.2–0.3 EV**. So the proof must be blind wins against Lightroom Auto and the camera JPEG on everyday scenes, with guards against overdoing it.

**Hu et al.'s Exposure has the right shape.** Because the owner likes its results, its unpaired critic becomes stage 2, trained through a differentiable copy of our pipeline. Its checkpoint, TF1 code, RL agent and random dropout at inference are left out. **Afifi et al.'s network is rejected**: it is an 8-bit sRGB pixel model under CC BY-NC-SA. Its trick of rendering raws at known EV offsets is rebuilt on licence-clean raws. Mykyta decides the rescue policy, the gate redesign, whether contrast becomes an Auto setting, personalisation, the face input, the A/B time and the INTEL-TAU licence question. He also decides three scene questions: whether labels from Apache-2.0 VLM teachers, themselves trained on web data, may train shipped weights; whether a frozen web- or ImageNet-trained backbone is ever acceptable; and whether the encoder may be about 8 MB or must stay at 4 MB or less.

## Industry Auto flattens everyday photos, which leaves punch unclaimed

The tools Yellowmoon must beat come in three designs: some predict display-referred slider values, some apply metering heuristics, and one predicts a local tone map.

- **Lightroom and Camera Raw Auto** (2017, Adobe Sensei) predicts exactly eight sliders: Exposure, Contrast, Highlights, Shadows, Whites, Blacks, Saturation and Vibrance ([Adobe Lightroom API](https://developer.adobe.com/firefly-services/docs/lightroom/guides/auto-tone/)). White balance is a separate Auto.
- **Adobe Adaptive** (2024) is the strongest commercial design. It predicts a grid of **12,288 local tone curves** and three colour tables (global, sky, main subject), trained on "thousands of hand-edited photos". A team reviewed the edits for one style that "looks appealing and natural, avoiding opinionated renderings with extreme contrast" ([Adobe Blog](https://blog.adobe.com/en/publish/2024/10/14/the-adobe-adaptive-profile)).
- **Capture One** meters on an 18 % grey average. Its Smart Adjustments copy exposure and white from a reference image with face priority ([Capture One support](https://support.captureone.com/hc/en-us/articles/7129920998045-Smart-Adjustments)).
- **DxO's Smart Lighting** weights exposure towards detected faces ([DxO user guide](https://userguides.dxo.com/photolab/en/the-customize-tab/)).
- **Apple's Core Image** auto-enhance analyses the histogram, faces and metadata, then configures a short chain of filters ([Apple](https://developer.apple.com/library/archive/documentation/GraphicsImaging/Conceptual/CoreImaging/ci_autoadjustment/ci_autoadjustmentSAVE.html)).
- **Resolve's Auto Color** was built for Rec.709 at gamma 2.4. Blackmagic calls it a neutral starting point that is "hit-or-miss" ([DaVinci Resolve Club](https://davinciresolveclub.com/davinci-resolve-auto-color/)). Resolve 21's Photo page adds an ML "Auto Balance" over Temp, Tint, Lift, Gamma, Gain, Contrast, Shadows, Highlights and Saturation ([Blackmagic](https://documents.blackmagicdesign.com/SupportNotes/DaVinci_Resolve_21_New_Features_Guide.pdf?_v=1776322810000)).
- **Colourlab.ai** sells one-click neutral balance of exposure, contrast and white ([Colourlab](https://3.colourlab.ai/creator/)).
- **Google Photos'** automatic edit is the best-scoring industry baseline found. In MonetGPT's comparison on 400 FiveK photos it reached **25.86 dB, ahead of every research method** in the table, including MonetGPT's 23.75 ([Dutt 2025](https://arxiv.org/abs/2505.06176)).

Google's Pixel work is not a model for this goal. HDR+ serves phone capture and Night Sight serves low light. Yellowmoon borrows two published techniques from them:

- **Exposure by example with face weighting** (HDR+): match 4 × 64 brightness quantiles against about 5,000 hand-tuned scenes, "strongly boost" face regions, and keep candidates within a factor of 8 in luminance ([Hasinoff et al. 2016](https://www.hdrplusdata.org/hdrplus.pdf)).
- **White balance trained on photographers' preferred whites**, tagged while viewing the final rendering (Night Sight) ([Liba et al. 2019](https://arxiv.org/abs/1910.11336)).

| Tool | What its Auto decides | Documented weakness |
|---|---|---|
| Lightroom / ACR Auto | 8 display sliders, from a learned model | Shadows opened (+59 in one example) and contrast lowered, an "HDR'd" look; exposure left low, +0.45 where the editor wanted +0.85 ([Kelby](https://lightroomkillertips.com/make-lightrooms-improved-auto-button-even-better/)). "A bit heavy handed with saturation" ([Friedl](https://regex.info/blog/2018-04-20/2860)). Dark evening scenes turned to daylight ([Adobe Community](https://community.adobe.com/questions-680/auto-tone-results-in-overexposed-photos-915527)) |
| Lightroom Auto WB | neutral white | Sunsets "turn everything blue" ([Adobe](https://www.adobe.com/creativecloud/photography/technique/white-balance.html)); magenta in forests ([Fstoppers](https://fstoppers.com/lightroom/how-perfect-white-balance-lightroom-690136)); drifts from frame to frame in a series ([Shotkit](https://shotkit.com/auto-temp-white-balance-lightroom/)) |
| Adobe Adaptive | local tone grid plus 3 colour tables | Adobe's own list: skin slightly off, too bright or too dark overall ([Adobe Blog](https://blog.adobe.com/en/publish/2024/10/14/the-adobe-adaptive-profile)). Removes fog and mood, "overly bright and generic"; reddish skin ([Fstoppers](https://fstoppers.com/lightroom/are-lightrooms-adaptive-profiles-better-presets-699596)). Shadows too light ([Benz](https://gregbenzphotography.com/lightroom-acr/acr-17-ai-adobe-adaptive-profiles-non-destructive-denoise-generative-expand/)) |
| Capture One | grey-card metering, face priority | Reviewers find it has "a little more 'punch'" than Lightroom, which "flattens" ([Life After Photoshop](https://lifeafterphotoshop.com/dxo-vs-lightroom-vs-capture-one-pro-best/), [Photography Life](https://photographylife.com/lightroom-vs-capture-one)); consistency of Smart Adjustments needed fixing ([Capture One](https://support.captureone.com/hc/en-us/community/posts/10101121924765-Improved-Smart-white-balance)) |
| DxO PhotoLab | face-weighted Smart Lighting | Face mode silently reports "No faces detected" when a face is partly hidden ([DxO](https://userguides.dxo.com/photolab/en/the-customize-tab/)); "can look a little flat" ([Life After Photoshop](https://lifeafterphotoshop.com/dxo-vs-lightroom-vs-capture-one-pro-best/)) |
| Apple Photos / Core Image | histogram, face and metadata analysis, then filters | Method and quality undocumented beyond the archived guide |
| Google Photos | not published | No failures documented in these notes; top fidelity score above |
| Resolve Auto Color / Auto Balance | neutral balance; ML Temp, Tint, Lift… | "No idea which warmth belongs to a sunset" ([DaVinci Resolve Club](https://davinciresolveclub.com/davinci-resolve-auto-color/)); "a reasonable first pass" ([XDA](https://www.xda-developers.com/tried-davinci-resolve-new-photo-editor-not-what-i-expected/)) |
| Colourlab.ai | neutral exposure, contrast, white | Targets a neutral starting point for a colourist; no evaluation published |

The weaknesses follow a pattern, and it is the opposite of "punchy and professional". Lightroom opens shadows, lowers contrast and saturates. Adaptive brightens towards the generic. Neutral auto whites remove warmth the scene was meant to have. Resolve and Colourlab deliberately stop at a neutral starting point for a colourist to finish. Kelby's corrections of Lightroom Auto point the other way: nearly double the exposure, halve the shadow lift, bring contrast back to zero or above ([Kelby](https://lightroomkillertips.com/make-lightrooms-improved-auto-button-even-better/)). The few numbers show the generic Auto is beatable. In Park et al.'s study, Lightroom's auto white balance plus auto tone scored **2.74 against 2.11 for the unedited input and 3.37 for the expert**, closing about half the gap ([Park 2018](https://arxiv.org/abs/1804.04450)). A 2019 research method was preferred over Auto-Tone **67 % to 21 %** in an anonymised pairwise test ([Zhang et al. 2019](https://arxiv.org/abs/1910.13688)). No blind comparison of today's modes across products was found. No product exposes a confidence for its auto settings, although FFCC, the method behind Google's white balance, produces a full posterior ([Barron & Tsai 2017](https://arxiv.org/abs/1611.07596)). No open-source raw processor found ships learned auto exposure; darktable adds a fixed +0.7 EV ([darktable manual](https://docs.darktable.org/usermanual/development/en/preferences-settings/processing/)), the same idea as our +1.11 EV.

Human disagreement sets a low ceiling on accuracy. Bychkovsky et al. found FiveK's retouchers differ by **3.3 L\* on average and up to 23.5**, where 2.3 L\* is one just-noticeable difference ([Bychkovsky 2011](https://people.csail.mit.edu/vladb/photoadjust/db_imageadjust.pdf)). One EV is about 15 L\* at mid grey, so that is roughly 0.22 EV. Yang et al. report that **75 % of FiveK images vary by more than 0.255 EV across the five experts**, and their best learned model reaches 0.23–0.29 EV mean absolute error per expert ([Yang 2018](https://arxiv.org/abs/1803.02269)). Beating Lightroom by a few hundredths of an EV against expert C would only fit one person's taste. "Clearly better" has to be counted in blind wins and gross failures, as defined in the evaluation section below.

Yellowmoon starts with three structural advantages over every tool in the table. It decides in scene-linear light before the look, knowing the exact Sony clip (3.58 stops over grey at Auto) ([docs/02](../docs/02-colour-pipeline.md)). Its light estimate is physical and spectral, and the user's white is already a separate quantity. Its renderer is fixed, known and deterministic. A few values also carry most of what "auto" means. In Zeng et al.'s ablation, one per-image, mostly-brightness scalar delivers **about 2.8 of the 4.8 dB** that a full image-adaptive LUT model gains ([Zeng 2020](https://arxiv.org/abs/2009.14468)).

## Punch is a joint decision over five settings, not exposure alone

A photo that looks finished on open is not just correctly exposed. Its mid-tones sit where a professional would put them, its whites reach white, its contrast has bite without crushing, and its colour is warm or cool on purpose. Lightroom Auto acknowledges this by setting eight sliders at once. Its complaints show which of them break a "professional" look when driven per photo: the shadow and contrast sliders produce the HDR-flat look, and saturation produces the over-vivid one. The decision space is also low-dimensional: in FiveK, **the first principal component of expert tone curves explains 90 % of their variance and the first three 99 %** ([Bychkovsky 2011](https://people.csail.mit.edu/vladb/photoadjust/db_imageadjust.pdf)). A handful of well-chosen outputs is enough.

Yellowmoon's controls map onto Lightroom's sliders almost one to one. The C9 primary grade in LogC4 has Contrast (`v = g + (v − g)·2^(0.6c/100)`, pivoting on grey), Brightness, Shadows, Highlights, Black, White, three-zone Colour balance and a subtractive Saturation ([docs/02, C9](../docs/02-colour-pipeline.md)). The Look tab's Tone card mirrors Temperature, Contrast and Saturation ([docs/11](../docs/11-ui.md)). The question is which of these the model should set per photo and which the look should own.

| Setting | Yellowmoon control | Lightroom Auto analogue | Per photo or look | Reason |
|---|---|---|---|---|
| Exposure | `EV = +1.11 + ΔEV` (C1–C6 gain) | Exposure | **Predicted** | The dominant single value (Zeng); Kelby's main fix to Lightroom Auto |
| Temperature and tint | user white in C4–C5 (Tone card Temperature; tint on the Colour tab) | separate Auto WB | **Predicted** as a preferred offset over `illum` | Pros keep sunset and tungsten warmth; Adaptive learned "a cooler white balance than the ambient illumination" for food ([Adobe Blog](https://blog.adobe.com/en/publish/2024/10/14/the-adobe-adaptive-profile)) |
| White lift | amount A at C12 step 1b | Whites, partly | **Predicted** (already per photo) | ACES 2's flat shoulder leaves whites grey; the C9 White and Highlights sliders would change 40–83 % of pixels to do the same ([docs/02](../docs/02-colour-pipeline.md)) |
| Contrast | C9 Contrast, Tone card | Contrast | **Predicted, bounded** offset around the look's own contrast | "Punch" is mostly mid-tone contrast; Lightroom lowers it and is criticised for it |
| Saturation | C9 Saturation (subtractive) | Saturation, Vibrance | **Look default at first**; small bounded head later if the A/B asks | Oversaturation and reddish skin are the documented failures; subtractive saturation also darkens colours |
| Shadows, Highlights, Black, White zones, Brightness, Colour balance | C9 rows | Shadows, Highlights, Blacks, Whites | **Left at 0** | Per-photo shadow lift is the source of Lightroom's HDR look; EV, lift and C7 already cover the ends |
| Highlight range | C7 `r`, default 1 | none | **Owner default**, not predicted | Owner's decision of 2026-10-17 ([docs/02](../docs/02-colour-pipeline.md)) |
| The look | Signature (LogC4 curve plus saturation) | profile | **Fixed** | Defines the ALEXA identity |

The recommendation is to predict four values jointly: EV, the white offset, the lift and a bounded contrast. These interact on screen. Contrast pivots on grey, so it changes how bright a mis-placed exposure looks. The lift can fight the exposure. Warmth reads differently at different brightness, which is why Night Sight's taggers needed the final rendering ([Liba 2019](https://arxiv.org/abs/1910.11336)). A per-photo contrast does change one thing the owner set on purpose, the look's fixed contrast. The design therefore makes it an offset that shrinks to zero when the model is unsure, keeps its range tight and asymmetric until the A/B says otherwise, and asks Mykyta whether the Tone card's AUTO button, which today "speaks for the temperature only", should also cover contrast ([docs/11](../docs/11-ui.md)). Saturation starts at the look's default, because oversaturation is exactly Lightroom Auto's documented failure ([Friedl](https://regex.info/blog/2018-04-20/2860), [Life After Photoshop](https://lifeafterphotoshop.com/dxo-vs-lightroom-vs-capture-one-pro-best/)).

## Scene understanding: a small encoder of our own carries the professional's judgment

The owner's north star is a model that "understands the scene and sets auto settings accordingly, how a professional photographer would". Professionals and camera makers already decide by scene:

- Sony's Intelligent Auto recognises "Portrait, Night Portrait, Night Scene, Backlight Portrait, Backlight, Landscape, Macro, Spotlight, and Low Light" ([Sony Help Guide](https://helpguide.sony.net/ilc/2210/v1/en/contents/TP1000844775.html)).
- Nikon's matrix metering "accesses a database of over 30,000 actual images to determine the best exposure for the scene" ([Nikon](https://www.nikonusa.com/learn-and-explore/photography-glossary/)).
- Adobe's Adaptive learned from its hand-edited set "to lighten subjects slightly, especially if they're in the foreground or in shadow, … to make food brighter, more colorful, and with a cooler white balance than the ambient illumination" ([Adobe Blog](https://blog.adobe.com/en/publish/2024/10/14/the-adobe-adaptive-profile)).

The recommendation is to give `autoset` an explicit, interpretable reading of what the photo is, from an encoder Yellowmoon trains itself, and to prove, scene category by scene category, that it helps.

### Semantics are proven for faces and local effects, not yet for global exposure on raw

The evidence for semantic input is real, but it sits mostly in faces and local processing:

- **Content-aware enhancement.** Kaufman et al. detect faces, skies and underexposed salient regions and correct each in its own way. Their method was preferred **64 % to 24 %** over Picasa, Office Picture Manager and Photoshop Elements on 100 FiveK images. The margin was **70 % to 17 %** on images with faces and sky, and 55 % to 25 % on images with neither ([Kaufman 2012](https://pages.cs.huji.ac.il/danix-lab/cglab/projects/cape/cape-cgf-2012.pdf)).
- **Local effects.** Semantic context cut Yan et al.'s error on local, stylised effects by 23–25 %, for example Foreground Pop-Out from 9.27 to 7.08 (CIELab L2) ([Yan 2016](https://arxiv.org/abs/1412.7725)).
- **People.** PPR10K's human-region weighting added about **0.3–0.4 dB** for each of its three experts ([Liang 2021](https://arxiv.org/abs/2105.09180)). HDR+ "strongly boost[s]" face regions in production ([Hasinoff 2016](https://www.hdrplusdata.org/hdrplus.pdf)), and Capture One and DxO weight exposure towards faces.
- **White balance and metering.** A semantic mask cut a white-balance CNN's per-pixel error by more than 40 %, though on synthetic sRGB casts with ground-truth masks ([Afifi 2018](https://arxiv.org/abs/1802.00153)). Semantic features cut FFCC's worst-25 % error on Gehler–Shi from 4.65° to 4.23° ([FFCC](https://arxiv.org/abs/1611.07596), [CLCC table](https://arxiv.org/abs/2106.04989)). A learned metering importance map "significantly reduces the regression error" of an exposure network ([Yang 2018](https://arxiv.org/abs/1803.02269)).

For **global** exposure on raw files, the record is thin. The only direct test, Bychkovsky 2011, found that "local histograms, color distributions, and scene descriptors ... did not improve the results" ([Bychkovsky 2011](https://people.csail.mit.edu/vladb/photoadjust/db_imageadjust.pdf)). No paper was found that isolates scene features for global exposure or white-balance *value* prediction on raw data against global statistics alone (the scene notes of this study (working notes, not kept)). Users in a 2023 study of hard lighting preferred "a simple saliency method" to more complex strategies ([Tedla 2023](https://arxiv.org/abs/2309.04542)).

The 2025–2026 VLM retouching agents show that choosing sliders from semantics is feasible, but none reports EV or kelvin error against experts:

- **MonetGPT** fine-tunes Qwen-VL-7B on puzzles built from PPR10K expert edits. Users preferred it in a 50-image study, at about **25 s per image** on an RTX 4090 ([Dutt 2025](https://arxiv.org/abs/2505.06176)).
- **JarvisArt** is Qwen2.5-VL-7B trained on 50k chain-of-thought samples plus reinforcement, under a non-commercial licence covering weights and inference code ([Lin 2025](https://arxiv.org/abs/2506.17612), [repo](https://github.com/LYL1015/JarvisArt)).
- **PhotoArtAgent** plans Lightroom edits with GPT-4o-class models; 76.2 % of images needed several revise rounds ([Chen 2025](https://arxiv.org/abs/2505.23130)).
- **RetouchIQ** (CVPR 2026) outputs executable Lightroom adjustments with an RL-tuned "generalist reward"; only its listing was read ([Wu 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Wu_RetouchIQ_MLLM_Agents_for_Instruction-Based_Image_Retouching_with_Generalist_Reward_CVPR_2026_paper.html)).

Q-Bench finds that MLLMs' low-level visual skills are "still unstable and relatively imprecise" ([Q-Bench](https://arxiv.org/abs/2309.14181)). VLMs are reliable for *what* is in the frame, not for *how bright* it should be.

The honest reading: semantics should matter most where global statistics confuse intent with error.

- Snow and beaches are correct when bright.
- Sunsets are correct when warm and darker than a foreground meter suggests.
- Backlit faces should be exposed for the face, not the sky.
- A low-key interior under tungsten should not be lifted to mid grey.

Elsewhere the gain on global EV will be modest. The per-scene ablation in the evaluation section is the proof that the encoder helps, and Bychkovsky's negative result is the warning.

### Professional rules per scene become priors, not labels

Practitioner guidance is consistent, though no structured dataset of it exists:

- **Snow.** Snow fools meters into grey. "+1 stop is usually about right, but if it takes +2 stops, use whatever looks best" ([Rockwell](https://kenrockwell.com/tech/snow.htm)); Canon suggests "+1/3 to +2 stops" (from a search summary) ([Canon USA](https://www.usa.canon.com/learning/training-articles/training-articles-list/photographing-snow)).
- **Sunsets.** Meter the sky beside the sun and start around −1 stop, because exposing for the ground "will almost certainly give you a washed-out sky". Auto white balance "will tend to over compensate for the deep reds and oranges" (low-authority how-to sources) ([PictureCorrect](https://www.picturecorrect.com/how-to-get-stunning-colors-in-your-sunset-photography/), [Light Stalking](https://www.lightstalking.com/wp-content/uploads/Sunset-Photography-cheat-sheet.pdf)). Night Sight's photographers kept "some of the color of the illuminant" in sunsets and clubs ([Liba 2019](https://arxiv.org/abs/1910.11336)).
- **Night stays night.** HDR+ keeps candidates within a factor of 8 of the scene's luminance ([Hasinoff 2016](https://www.hdrplusdata.org/hdrplus.pdf)).
- **Faces first.** HDR+ and the PPR10K experts both prioritise people.

The scene notes turn this guidance into a prior table. Each EV prior is an offset from a mid-grey meter on the statistics, the metering value Yellowmoon still reports but no longer applies ([docs/STATUS.md](../docs/STATUS.md)); the head converts it into a correction over the camera gain. These are starting values for Mykyta to tune, not measured truths. The low-key, food and architecture rows are the notes' own inference.

| Scene class | EV prior | White prior | Other |
|---|---|---|---|
| Snow or beach (high-key) | +1.0 to +1.5 | neutral to slightly cool | protect highlight detail (C7 range) |
| Sunset or sunrise | −0.3 to −1.0, relative to the foreground | keep warm: correct only a fraction of the cast | saturation unchanged or slightly up |
| Golden hour | 0 | keep warm | — |
| Night city or stage | keep the Bv placement (factor-of-8 rule) | keep part of the light's colour | — |
| Backlit portrait | raise for the face region | — | the highlight guard decides how much sky is lost |
| Low-key interior or mood | do not lift to mid grey | keep part of the tungsten | white lift off |
| Food | small positive | neutral to slightly warm; correct a warm ambient more than in other interiors, as Adaptive learned | — |
| Architecture, daylight landscape | from the statistics | neutral | — |

The priors enter training in three ways:

- as a regulariser that pulls predictions towards the prior where no label exists, weighted by scene confidence;
- as evaluation assertions, for example "the median snow EV is at least +0.7 over the mid-grey meter";
- as the initial per-scene offsets of the head.

They partly restate what the camera's own exposure already did, since the Sony JPEG carries its scene decision. They matter most where the camera's exposure was wrong, or where the raw was exposed to protect highlights.

### No public encoder passes rule 5 as shipped weights

All CPU timings below are for ONNX Runtime 1.29 on its CPU provider, fp32, on one x86 machine, measured for the scene notes (the scene notes of this study (working notes, not kept)).

| Encoder | Image params | Size (fp16) | CPU, 1 / 4 threads | Weights licence | Training data | Verdict |
|---|---|---|---|---|---|---|
| MobileCLIP-S0 | 11.4M | 22.9 MB | 50.3 / 20.1 ms | Apple research-only terms that exclude "product development" ([LICENSE_MODELS](https://github.com/apple/ml-mobileclip/blob/main/LICENSE_MODELS)) | DataCompDR-1B | **Excluded, even as a teacher** |
| DINOv3 ViT-S/16 | about 21M | — | — | custom DINOv3 License; derivatives only under it, and Meta may change it ([LICENSE.md](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md)) | LVD-1689M | Excluded |
| DINOv2 ViT-S/14 | 21M | 44.4 MB | 150.4 / 81.8 ms | Apache-2.0 ([model card](https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md)) | LVD-142M (undisclosed web crawl) | Feature teacher; frozen in the app only by your decision, with a first-run download |
| SigLIP 2 B/16 | about 93M | vision tower about 186 MB | not measured | Apache-2.0 ([card](https://huggingface.co/google/siglip2-base-patch16-224)) | WebLI (Google-internal) | Offline zero-shot teacher |
| OpenCLIP ViT-B/32 | about 88M | — | — | MIT, but "any deployed use case … is currently out of scope" ([card](https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K)) | LAION-2B | Avoid |
| TinyCLIP ViT-8M/16 | 8M | about 16 MB | not measured | MIT ([repo](https://github.com/wkcn/TinyCLIP)) | YFCC-15M | Web data, no advantage over our own |
| MobileNetV4-Conv-S | 3.8M | 7.6 MB | 3.5 / 1.4 ms | "apache-2.0" tag ([timm card](https://huggingface.co/timm/mobilenetv4_conv_small.e2400_r224_in1k)) | ImageNet-1k | **Architecture template**; its frozen weights only by your decision |

Industry practice and recent case law treat permissively licensed, web-trained weights as normal. Google ships SigLIP 2 under Apache-2.0 trained on WebLI, and Meta ships DINOv2 under Apache-2.0 trained on LVD-142M. The courts have so far been favourable:

- A UK court held that Stable Diffusion's weights "did not store any Getty images" ([SCL](https://www.scl.org/high-court-issues-eagerly-awaited-judgment-in-getty-v-stability-ai-case/)).
- Hamburg's Higher Regional Court let LAION rely on the scientific text-and-data-mining exception on 10 December 2025 ([DLA Piper](https://www.dlapiper.com/en/insights/blogs/mse-today/2025/robert-kneschke-v-laion)).
- Two US courts found training to be fair use on the facts, while warning that this "does not mean any use of copyrighted material to train LLMs is lawful" ([Jackson Walker](https://www.jw.com/news/insights-kadrey-meta-bartz-anthropic-ai-copyright/)).

Yellowmoon's rule 5 is a data-provenance policy stricter than the law: "shipped weights only from CC0, CC BY, CC BY-SA, Apache or our own data" ([docs/14](../docs/14-licensing.md)). ImageNet's terms are non-commercial research ([timm PR 660](https://github.com/huggingface/pytorch-image-models/pull/660/files)), WebLI is unreleased and LVD-142M is undisclosed. Three routes follow, in rising licence risk:

1. **Our own encoder, trained from scratch on CC BY, CC0 or CC BY-SA images.** Complies with rule 5 as written.
2. **Our own encoder on clean images, with labels or soft targets from web-trained teachers run offline.** The student's data is clean, and the teachers' outputs are information rather than training images. Whether rule 5 "looks through" to a teacher's training data is a policy question the rule text does not settle, and no court has ruled on such a chain.
3. **A frozen DINOv2-S or MobileNetV4 in the app.** Common practice, but it breaks rule 5 as written and adds 8–44 MB, so it needs a recorded decision in `docs/16`.

The recommendation is route 2, with route 3 logged as a question for Mykyta.

Determinism holds on one machine but not automatically across machines. All three measured encoders gave byte-identical outputs on repeated runs and between 1 and 4 threads (the scene notes of this study (working notes, not kept)). A 2026 study, however, found that switching only the execution provider made fp32 outputs "disagree on every CIFAR-10 image", and that x86 kernel dispatch splits the outputs of one network into many classes across hardware ([arXiv 2607.23227](https://arxiv.org/abs/2607.23227)). Quantisation is no shortcut either. The public dynamic-int8 exports were slower than fp32 (MobileNetV4 37.6 ms against 3.5 ms), and MobileCLIP's int8 export was broken, with a cosine of 0.148 to its fp32 output. The encoder therefore runs on the CPU provider only, with a fixed thread count and quantised outputs, and never as int8 without a parity test.

### Apache-2.0 VLM teachers label clean images offline

Several current open VLMs are Apache-2.0 with no restriction on using their outputs to train other models:

- Qwen3-VL ([card](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct));
- Qwen2.5-VL-7B ([card](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct));
- InternVL3.5-8B ([card](https://huggingface.co/OpenGVLab/InternVL3_5-8B));
- Gemma 4, Apache-2.0 per its card; the full licence page was not read ([card](https://huggingface.co/google/gemma-4-12B-it));
- SigLIP 2 as a zero-shot scorer.

Others must not teach shipped weights:

- **Gemma 3.** Its terms make a model trained on Gemma's outputs a "Model Derivative" that must carry Gemma's use restrictions ([Gemma Terms](https://ai.google.dev/gemma/terms)).
- **Llama 3.2.** A model trained on its outputs must begin its name with "Llama" ([Llama 3.2 LICENSE](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/LICENSE)).
- **Qwen2.5-VL-3B and 72B.** The 3B licence is non-commercial, and the 72B licence requires "Built with Qwen" ([3B](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/blob/main/LICENSE), [72B](https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct/blob/main/LICENSE)).
- **MobileCLIP.** Research only.
- **Molmo.** Apache-2.0, but its card says it is "intended for research and educational use", which makes it a weaker choice ([card](https://huggingface.co/allenai/Molmo-7B-D-0924)).

The labelling run happens offline, at training time only:

1. **Image pool.** Open Images V7 has about 9 M images. Its annotations are CC BY 4.0, and its images are "listed as having a CC BY 2.0 license", though Google disclaims per-image accuracy ([Open Images V7](https://storage.googleapis.com/openimages/web/factsfigures_v7.html)). Keep only images whose own licence field says CC BY, add the cleared raws rendered through Yellowmoon, and keep a per-image licence manifest.
2. **Questions.** Teachers answer a fixed JSON taxonomy with categorical fields only:
   - scene: portrait, group, landscape, sunset or sunrise, snow or beach, night city, interior, stage, food, architecture, macro or nature, sport;
   - subject: face count, and whether a person, food, building or sky is present;
   - light: direct sun, overcast or shade, golden hour, blue hour, tungsten, fluorescent or LED, mixed, flash, stage colour;
   - intent: high-key, normal, low-key, intended silhouette, warm or cool mood;
   - backlit: yes or no.

   Teachers are never asked for EV numbers. To judge settings, a teacher picks the best of a bracket of Yellowmoon renders; comparative selection suits aesthetic judgment better than scalar scores ([Visual Aesthetic Benchmark](https://icml.cc/virtual/2026/68543)). Those picks are weak labels, checked against the owner's A/B.
3. **Determinism.** Greedy decoding, pinned model revisions with their commit SHAs in the model card, and stored raw outputs.
4. **Fusion.** Labels where at least two of three teachers agree become hard labels; agreement sets their weight; disagreements become soft labels at lower weight.
5. **Audit [HUMAN].** Mykyta labels about 300 random images, so teacher accuracy per class is known and reported in `docs/STATUS.md` before training.

No published work distils a VLM's retouching judgment into a tiny model. Small predictors trained on CLIP embeddings, and CSKD's distillation of CLIP into a small aesthetic model, show the pattern works for coarse judgments ([improved-aesthetic-predictor](https://github.com/christophschuhmann/improved-aesthetic-predictor), [CSKD](https://arxiv.org/abs/2307.15640)). Apple Vision's classifier and aesthetics score need no shipped weights. They do not run on the Linux CI and may change with macOS updates, so they serve evaluation only ([Apple docs](https://developer.apple.com/documentation/vision/classifyimagerequest)).

### SceneNet conditions the settings head through an interpretable scene vector

```
preview (scene-linear → fixed default Yellowmoon render, 224 px)
   ├─► stats (HDR+ quantiles, clip stats, EXIF Bv/ISO/EC …)       [ours, no weights]
   ├─► YuNet faces → count, area, face-region quantiles          [MIT, ~0.1 MB]
   └─► SceneNet (MobileNetV4-Conv-S-like, width ×0.5–1.0)        [ours, 2–8 MB f16]
           → s = 16 scene/intent probabilities + 4 cues
             (sky fraction, backlit score, warm-light score, people area)
head:  h  = MLP(stats ⊕ faces)          (30–70k params)
       h' = γ(s) ⊙ h + β(s)             (FiLM)
       Δ  = W·h' + Σ_k s_k · o_k        (o_k = per-scene offsets, initialised from the prior table)
outputs: ΔEV, Δmired, ΔDuv, white lift, contrast (saturation later),
         each clamped, rounded (EV 0.01, mired 1) and shrunk by its uncertainty
```

- **A 16-way vector, not a 512-d embedding.** With about 8,700 labelled scenes, a 512-d input invites overfitting. A small probability vector is inspectable (the debug output can say "Snow, high-key"), lets the prior table act directly, makes evaluation by category natural, and matches the way Sony exposes its scene logic.
- **Exposure invariance is essential.** SceneNet trains with ±3 EV, ±30 mired and ±0.01 Duv photometric augmentation plus a consistency loss, so it cannot mistake an underexposed frame for "night". Brightness information reaches the head only through the statistics and Bv.
- **Size against the budget.** `docs/12` allows model weights of "a few MB in total" and moves anything beyond about 30 MB to a first-run download ([docs/12](../docs/12-distribution.md)). There are three options:
  - (a) MobileNetV4-Conv-S at full width: 3.8M parameters, **7.6 MB** in f16;
  - (b) a ×0.5–0.75 width or MobileNetV3-Small-class network: 1–2.5M parameters, **2–5 MB**, which fits "a few MB";
  - (c) a ViT-S student at 44 MB, which is not worth it for 16 classes.
- **Runtime.** ONNX Runtime's CPU provider (or burn's ndarray backend) runs with a fixed thread count in both the app and `ym`, never the CoreML provider. At 1–4 ms on the CPU, the GPU and Neural Engine are unnecessary. Golden tests check 20 fixed previews: the scene vector within 1e-4 of PyTorch, and the final settings exact after rounding.
- **Licence.** With a CC0/CC BY image pool, SceneNet's weights are Apache-2.0. If CC BY-SA images (Play Raw or HDR+ renders) join the pool, they become CC BY-SA 4.0 (`docs/14` rule 5).

## Neither suggested reference ships as it stands

### Exposure (Hu et al., TOG 2018): keep the formulation and the critic, drop the artefact

Exposure is the closest published precedent to what the owner asked for. On a **64 × 64 thumbnail of linear raw-derived data**, a reinforcement-learning agent picks up to five global "filters" and their parameters. The filters are exposure (tanh-bounded to ±3.5 EV), gamma (1/3 to 3), white balance, saturation, 8-step tone and colour curves, contrast and a black-and-white blend, applied at full resolution ([Hu 2018](https://arxiv.org/abs/1709.09602), [config_example.py](https://github.com/yuanming-hu/exposure/blob/master/config_example.py)). The white-balance filter holds red fixed, scales green and blue by exp(±0.5) and renormalises by luminance. It is a brightness-neutral, two-degree-of-freedom white, equivalent to temperature plus tint and independent of the EV head ([filters.py](https://github.com/yuanming-hu/exposure/blob/master/filters.py)). Training needs no pairs. The inputs were 2,000 FiveK raws, the targets expert C's retouches of *other* photos or a 500px artist's portfolio, and the reward a Wasserstein GAN critic; it trained in about 100 minutes on a GTX 1080 Ti ([README](https://github.com/yuanming-hu/exposure)). The results the owner likes are real. Mechanical Turk users rated it **3.43 out of 5, against 3.37 for Pix2pix** (which needs paired data), 3.30 for human novices using the same filters, and 3.66 for expert C ([Hu 2018](https://arxiv.org/abs/1709.09602)). Fidelity scores against one expert are harsh: trained on PPR10K expert A and tested on FiveK, it reaches **15.12 dB against MonetGPT's 23.75** ([Dutt 2025](https://arxiv.org/abs/2505.06176)). That is the expected weakness of a method that learns a style rather than a person's specific edits.

The artefact cannot ship, for four reasons:

- **Licence.** The code is MIT ([LICENSE](https://github.com/yuanming-hu/exposure/blob/master/LICENSE)), but the checkpoint sits in a submodule with no licence file and was trained on research-only FiveK ([exposure_models](https://github.com/yuanming-hu/exposure_models), [FiveK licence](https://data.csail.mit.edu/graphics/fivek/legal/LicenseAdobe.txt)).
- **Framework.** It is TensorFlow 1, last changed in 2019.
- **Determinism.** It is **random at inference by design**: "the system is learning a one-to-many mapping … achieved using (random) dropout … you may get slightly different results every time" ([README FAQ](https://github.com/yuanming-hu/exposure)). That breaks AGENTS rule 8.
- **Colour space.** It was trained on linearised ProPhoto, not our camera-to-AWG4 path.

The RL agent also solves a problem Yellowmoon does not have. It chooses which filter comes next, while our stages and their order are fixed, so a single-step actor that outputs all settings at once is enough.

**Verdict: adopt the formulation, make the unpaired critic stage 2, and re-implement from the paper and the MIT code.** The formulation is a thumbnail-in, values-out actor with tanh-bounded heads, zero-initialised biases ("start at no change") and a 2-DOF brightness-neutral white. The checkpoint, the RL agent and the dropout stay out. The section "A Hu-style critic …" below gives the route its full design.

### Learning Multi-Scale Photo Exposure Correction (Afifi et al., CVPR 2021): reject the network, rebuild the dataset recipe

Afifi's model is a coarse-to-fine Laplacian-pyramid CNN of **about 7M parameters**. It maps an 8-bit sRGB image to a corrected 8-bit sRGB image at 512 px, then upsamples through a bilateral grid, taking about 4.5 s for a 16 MP image ([Afifi 2021](https://arxiv.org/abs/2003.11596)). The code is MATLAB; the 28.2 MB model is **CC BY-NC-SA 4.0**, and the README says it "CANNOT be used for commercial purposes" ([repo](https://github.com/mahmoudnafifi/Exposure_Correction)). It outputs pixels, not settings, and it works on display-referred data, where "the ISP's nonlinear rendering" makes errors hard to undo ([Afifi & Brown 2020](https://arxiv.org/abs/2004.01354)). Its average of **19.48 dB** over five experts has since been raised to **21.82 dB** by CoTF and others. All of these are still pixel models, under non-commercial terms or without licence files ([LoopExpose 2025](https://arxiv.org/abs/2511.06066), [UEC code](https://github.com/BeyondHeaven/uec_code)).

The valuable part is the dataset, MSEC. Each FiveK raw was rendered in Adobe Camera Raw at **relative EVs of −1.5, −1, 0, +1 and +1.5**, giving 24,330 images whose exposure offset is exact by construction ([Afifi 2021](https://arxiv.org/abs/2003.11596)). MSEC follows FiveK's research licence and its targets are expert C's retouches, so it is evaluation-only. As an evaluation set it is useful: FiveK's Lightroom catalogue holds each expert's slider values, so an absolute label is expert C's Exposure minus the offset ([FiveK](https://data.csail.mit.edu/graphics/fivek/)). The recipe itself transfers cleanly. Scale a CC0 or CC BY raw by 2^k in scene-linear light and adjust its EXIF exposure time to match, and the correct correction is known exactly. UEC did this with an emulated ISP and a 19,388-parameter model ([Cui 2024](https://arxiv.org/abs/2507.17252)). This teaches *consistency* (how far this frame is from our reference) but not *taste* (what a professional's reference is).

**Verdict: reject the network and its weights; use MSEC for evaluation only; rebuild the known-offset recipe on licence-clean raws through Yellowmoon's own renderer.**

| Reference | Use as is | Fine-tune | Re-implement | Verdict |
|---|---|---|---|---|
| Hu et al. 2018, Exposure | No: unlicensed FiveK weights, TF1, random at inference | No: inherits FiveK | Yes: bounded value heads, 2-DOF white, unpaired critic (MIT code, read and rewritten) | **Adopt the shape; critic as stage 2** |
| Afifi et al. 2021, MSEC network | No: CC BY-NC-SA, sRGB pixel output, 7M parameters | No | Dataset recipe only | **Reject the model; rebuild known-offset data on clean raws; MSEC for evaluation** |

### Nothing else pretrained passes the licence filter

Several codebases are permissive enough to read and re-implement: Exposure, FFCC, C5, NeurOp, DeepLPF, image-adaptive 3D LUT, AdaInt, and the PPR10K and MonetGPT code. Their weights, however, are trained on FiveK, PPR10K or data with unclear terms, and most of the models output pixels or LUTs rather than our settings ([NeurOp](https://github.com/amberwangyili/neurop), [3D-LUT](https://github.com/HuiZeng/Image-Adaptive-3DLUT)). Fine-tuning any of them would put research-licence data into shipped weights, which `docs/07` forbids ([docs/07](../docs/07-ml-models.md)).

| Candidate | Licence (code / weights / data) | Output | Verdict |
|---|---|---|---|
| FFCC (our `illum`) | Apache-2.0 ([LICENSE](https://raw.githubusercontent.com/google/ffcc/master/LICENSE)) / ours / synthetic | illuminant | **Keep; retrain as `illum` v2** |
| C5 | Apache-2.0 ([LICENSE](https://raw.githubusercontent.com/mahmoudnafifi/C5/master/LICENSE)) / unclear | illuminant | Re-implement its raw-to-raw augmentation |
| CCMNet | CC BY-NC ([README](https://raw.githubusercontent.com/DY112/CCMNet/main/README.md)) | illuminant | Idea only (camera-locus fingerprint), from the paper |
| Zhao–Afifi–Brown preferred-white map | no release found ([paper](https://arxiv.org/abs/2507.01342)) | preferred white | Re-implement the idea (~500 parameters) |
| Exposure, NeurOp, DeepLPF, CURL, 3D-LUT, AdaInt | MIT / BSD / Apache code; FiveK or PPR10K weights | values, curves, LUTs | Read for design; ship no weights |
| MSEC, UEC, LCDPNet, CoTF, LoopExpose | NC, CC BY-NC-SA or no licence file | pixels | Evaluation diagnostics at most |
| RSFNet | GPL-3.0 ([repo](https://github.com/Vicky0522/RSFNet)) | region values | Do not read for code |
| MonetGPT, JarvisArt, PhotoArtAgent, RetouchIQ | see the scene section | slider values | Not in the product, and not as teachers (PPR10K-derived or non-commercial weights); general Apache-2.0 VLMs teach instead |
| General image encoders (MobileCLIP, DINOv2/v3, SigLIP 2, OpenCLIP, MobileNetV4) | see the scene section | features | None passes rule 5 as shipped weights; MobileCLIP is excluded even as a teacher |
| NIMA (idealo) | Apache-2.0 code, AVA-trained weights ([repo](https://github.com/idealo/image-quality-assessment)) | aesthetic score | Avoid as a reward: learned aesthetic rewards "were not stable enough" ([Park 2018](https://arxiv.org/abs/1804.04450)) |
| YuNet face detector | MIT ([OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)); WIDER FACE terms unchecked | face boxes | **The one as-is candidate**, after a provenance check |
| Ultralytics YOLO | AGPL-3.0 ([LICENSE](https://raw.githubusercontent.com/ultralytics/ultralytics/main/LICENSE)) | detections | Excluded |

What Yellowmoon trains is therefore four small models of its own:

- **`autoset`**, the joint value predictor;
- **SceneNet**, a scene encoder of 1–4M parameters trained from scratch on CC BY images, which tells `autoset` what the photo is;
- **`autoset-wb`**, a preferred-white map of about 600 parameters that can ship first or become a sub-head of `autoset`;
- **`illum` v2**, the physical light estimator retrained on real data.

The evidence favours this split over one large model. HDR+ shipped a nearest-neighbour blend on brightness quantiles. Bychkovsky's regressors on about 50 hand-built features (LASSO 4.9 L\*, GPR 4.7 L\*) beat metric learning (11.5 L\*) ([Bychkovsky 2011](https://people.csail.mit.edu/vladb/photoadjust/db_imageadjust.pdf)). And MonetGPT's direct-regression baseline "overfits and predicts the same values" when data is scarce ([Dutt 2025](https://arxiv.org/abs/2505.06176)), a warning to keep inputs compact and data plentiful.

## A residual value predictor sits on the camera gain and on `illum`

**Inputs** are statistics Yellowmoon already computes, or nearly computes, on the 1/8 preview used for metering. That is about 400–600 numbers in six groups:

- **Brightness quantiles** in HDR+'s layout: 64 log2-luminance quantiles at two scales for both mean and max of RGB, centre-weighted. The max channel accounts for colour clipping ([Hasinoff 2016](https://www.hdrplusdata.org/hdrplus.pdf)).
- **Clip statistics**: the share and position of clipped blocks, and the distance from the 99.5th percentile to the Sony clip that C7 already knows.
- **`illum`'s outputs**: the (u, v) estimate, its covariance, six class probabilities, the confidence, and the distance of the as-shot white from the estimate.
- **EXIF**: scene brightness Bv, ISO, shutter, aperture, exposure compensation and focal length, all already parsed ([metadata.rs](../crates/ym-raw/src/metadata.rs)). DRO, Creative Style, metering mode, flash and scene mode come once the maker notes are parsed; today only the evaluation harness reads them ([tools/eval/autoexp.py](../tools/eval/autoexp.py)).
- **Faces**: count, area, position and face-region quantiles.
- **The scene vector** from SceneNet: 16 scene and intent probabilities plus four cues, which also condition the head through FiLM and per-scene offsets (see the scene section).

Bv works as a professional's sense of the light. It is HDR+'s factor-of-8 rule in feature form: snow stays bright, a dim interior stays intimate, and the model does not normalise every scene to one grey. That was the error the 2026-10-14 study measured in the earlier metered rule ([research/auto-exposure-2026-10-14.md](../research/auto-exposure-2026-10-14.md)).

**Outputs** are residuals over today's rules, so zero means "behave exactly as now":

- `EV = CAMERA_GAIN_EV + ΔEV`, clamped to ±3, with the highlight guard and the burst median unchanged. "The intelligent automatic settings … replace step 4, not the rest" ([docs/02](../docs/02-colour-pipeline.md)).
- A preferred-white offset (Δmired, ΔDuv) on the **user** white in C4–C5, never on `illum`'s scene light, which drives the spectral sensor map and must stay physical.
- The white-lift amount A, or a multiplier on the current estimator's target, still clamped to 0…2 stops.
- A bounded contrast offset around the look's contrast.
- A log-variance for each output.

The guard is dormant on Sony files anyway: a fully clipped pixel sits at about 2.2 scene-linear after +1.11 EV, under its 8.0 level ([docs/STATUS.md](../docs/STATUS.md)). The clip features take over its job.

**Preferred white separates physics from taste, as Samsung and Google do.** Samsung learned **a 539-parameter mapping from a neutral estimate to the preferred white** in CIE XYZ. It transfers to unseen cameras, runs in 0.024 ms on a phone CPU, and cuts C5's error on an unseen camera from 2.77° to 1.34° mean against the preferred target ([Zhao, Afifi, Brown 2025](https://arxiv.org/abs/2507.01342)). In a separate Samsung study, observers chose the photographer's preferred white over the chart-neutral one in **71.95 % of trials** ([Afifi 2025](https://arxiv.org/abs/2504.05623)). Sony's own "Priority Set in AWB: Ambience" exists to "produce a warm atmosphere" under incandescent light ([Sony Help Guide](https://helpguide.sony.net/ilc/2210/v1/en/contents/TP1000876546.html)). The evidence is not one-sided: one study judged neutral 6500 K portraits the most authentic and professional ([Suk et al. 2020](https://library.imaging.org/jist/articles/64/2/jist0681)). The strength of the preference is therefore a taste parameter, and it should depend on the scene, which is another job for scene understanding.

A principled rule needs no data: a partial adaptation from the estimated light towards about 6000 K on the locus, Zhai and Luo's surface-colour neutral ([Zhai & Luo 2017](https://library.imaging.org/cic/articles/25/1/art00016)). The degree of adaptation D falls at low CCT, high chroma and low adapting luminance. CIECAM02 gives D = F·(1 − (1/3.6)·exp((−L_A − 42)/92)), with L_A from Bv ([luxpy, CIE formula](https://github.com/ksmet1977/luxpy/blob/master/luxpy/color/cat/chromaticadaptation.py); read only, it is GPL). With two or three constants set by Mykyta in a blind test, this gives a shippable warm-keeping behaviour on day one. The learned map then improves on it.

**`illum` v2 improves the physical half.** Today's model was trained on synthetic scenes only. Its synthetic Bv range stops at 13 while sunlit files read 14.3–14.6, and the planned Cube++ evaluation never ran ([docs/02](../docs/02-colour-pipeline.md)). The cheapest documented win is metadata. On Simple Cube++, a model of about 5k parameters given ISO, exposure time and noise statistics cuts the worst-25 % error from **4.05° (FFCC) to 2.68°**, and the maximum from 45.2° to 10.9° ([Afifi 2025](https://arxiv.org/abs/2504.05623)). Three further changes follow the literature:

- **Real training data.** Train on Cube++ (CC BY 4.0, 4,890 Canon raws), converted into each Sony profile's raw space. It is reachable on Zenodo ([Zenodo](https://zenodo.org/records/4153431)).
- **One cross-body model.** Replace the per-profile filters with one model and a camera-locus embedding computed from Weta's curves. This is CCMNet's idea, re-implemented from the paper because its code is CC BY-NC ([CCMNet](https://arxiv.org/abs/2504.07959)).
- **Worst-case focus.** Add brightness augmentation, which cut error 5.04 % across six models ([BRE 2025](https://arxiv.org/abs/2502.12418)), and gate on the worst 25 % as the 2020 challenge did; its winner reached 4.08° ([IEC#2](https://arxiv.org/abs/2012.15779)).

Trained on Cube++ alone, `illum` v2 stays Apache-2.0.

**Uncertainty makes the model safe.** A five-member seeded ensemble with Gaussian negative-log-likelihood heads gives both epistemic and aleatoric variance ([Lakshminarayanan 2017](https://arxiv.org/abs/1612.01474), [Kendall & Gal 2017](https://arxiv.org/abs/1703.04977)). The applied correction is `Δ_used = Δ · σ_prior² / (σ_prior² + σ_pred²)`, a continuous shrink towards today's rule with no thresholds that flip. When the model is unsure, the photo comes out as it does now. The same σ is a confidence the UI can show, which no competitor offers.

**Determinism and cost fit the rules.** The model runs once per photo on the CPU in hand-written, fixed-order f64 Rust, like `illum`, whose port matches Python to 2.8e-13 ([docs/02](../docs/02-colour-pipeline.md)). ONNX Runtime stays off the head's path: its determinism flag targets GPU kernels, and its CPU kernels dispatch per instruction set ([ORT options](https://www.sidefx.com/docs/hdk/struct_ort_1_1_session_options.html), [MLAS dispatch](https://app.semanticdiff.com/gh/microsoft/onnxruntime/commit/566406617cae56528f38436cacdeea3926475ad5)). Measured in this study on a loaded 2.1 GHz Xeon, single-threaded, with outputs bit-identical across runs (a throwaway Rust benchmark, not kept):

- a 70,344-parameter feature MLP takes **0.040 ms**;
- the 634-parameter preference MLP takes 0.4 µs;
- an optional 55,592-parameter CNN on a 128 px thumbnail takes 22–30 ms.

SceneNet adds 1–4 ms on ONNX Runtime's CPU provider with a fixed thread count. Its scene vector and the final settings are rounded (EV 0.01, mired 1), so x86 CI and Apple-silicon results agree within a stated tolerance (the scene notes of this study (working notes, not kept)). Feature extraction costs more: 156 ms with a full sort, so it must use `select_nth`, as the white-lift estimator does. Weights go in `data/models/autoset/` with a model card, SHA-256 and exported test vectors. `PIPELINE_VERSION` is bumped and the model version is written into export XMP, so the app and `ym` stay bit-identical (rules 6 and 8).

## Labels come from fitting Yellowmoon's own render to licence-clean references

No dataset combines Sony raws, human exposure or white edits and a licence that allows shipped weights (the datasets notes of this study (working notes, not kept)). The training set is assembled from references that are each someone else's taste. Each is converted into **Yellowmoon settings** by fitting, and the loss stays in settings space (EV, mired, contrast units). A display-space loss against a Sony or Google JPEG would teach the model to imitate those engines' tone curves rather than choose settings for ours.

| Source | Licence and weight consequence | What it labels | Role |
|---|---|---|---|
| raw.pixls.us and RawDB CC0 Sony ARWs | CC0 ([raw.pixls.us](https://raw.pixls.us/json/getrepository.php?set=all), [RawDB](https://rawdb.dnglab.org/api/sets)) | Sony's mid-grey placement and as-shot white, from the embedded JPEG | Sony prior and non-regression. **196 and 229 files**, mostly overlapping; daylight, low ISO, no people by site policy ([raw.pixls.us](https://raw.pixls.us/)) |
| Google HDR+ bursts | CC BY-SA 4.0 ([HDR+ dataset](https://hdrplusdata.org/dataset.html)); weights become CC BY-SA 4.0 | Google's finished exposure and white on 3,640 everyday phone scenes (`final.jpg`); mid-tones and white only, because its tone mapping is local | Breadth of everyday scenes, including backlit and high-contrast ones |
| Cube++ | CC BY 4.0 ([README](https://github.com/Visillect/CubePlusPlus)); weights stay Apache-2.0 | Two chart-neutral lights plus Canon's own AWB (`cam_estimation.csv`) on 4,890 raws; scene labels | The only clean (neutral, camera-preferred) white pairs; `illum` v2; scene-head targets |
| Play Raw (pixls.us) | per post; CC0, BY or BY-SA usable; edits carry their editors' own licences ([Play Raw rules](https://discuss.pixls.us/t/playraw-stuff-to-keep-in-mind/4781)) | Sony raws of real subjects; several human edits per raw where licensed | Split by source: part training, part the held-out A/B set |
| Open Images V7 | annotations CC BY 4.0; images "listed as having a CC BY 2.0 license", per-image accuracy disclaimed ([Open Images V7](https://storage.googleapis.com/openimages/web/factsfigures_v7.html)); CC BY images only, so SceneNet stays Apache-2.0 | scene, subject, light and intent labels from Apache-2.0 VLM teachers | SceneNet's training pool |
| Known-offset copies of the above | inherits the source | exact relative EV | Unlimited equivariance labels |
| `training/synth` scenes | ours | physics sanity | Regression tests |
| INTEL-TAU | **CC BY-SA 4.0 per the Finnish registry** ([research.fi](https://tiedejatutkimus.fi/fi/results/dataset/f0570a3f-3d77-4f44-9ef1-99ab4878f17c)), but evaluation-only in `docs/14` ([docs/14](../docs/14-licensing.md)) | 7,022 illuminant ground truths, three cameras | Evaluation until the discrepancy is resolved |
| FiveK, MSEC, PPR10K, LSMI, SID, NUS-8, Gehler–Shi | research-only, non-commercial or unstated ([FiveK](https://data.csail.mit.edu/graphics/fivek/), [PPR10K](https://github.com/csjliang/PPR10K), [LSMI](https://github.com/DY112/LSMI-dataset)) | experts' sliders and whites | **Evaluation only** (owner, 2026-10-18) |

Fitting is cheap because every stage before C7 is linear. **EV** reuses the existing G3 estimator, which finds by quantile matching the scene level the reference renders as mid grey. It is exact to 0.02 EV on synthetic charts with known gain and good to about 0.1 EV on real photos ([docs/02](../docs/02-colour-pipeline.md)). It should be restricted to a mid-tone band of about −2 to +1 stops, so the reference's toe and shoulder do not leak in. **White** uses the reference's own white where one exists: camera as-shot gains, Cube++ estimates, or an XMP Temperature/Tint mapped through the camera matrices. Otherwise it is fitted on unclipped near-neutral pixels. **Contrast and lift** are fitted jointly by a Nelder–Mead search over a few preview renders, or later through the differentiable proxy. Regularise them towards the current lift estimator and zero contrast, because they trade off against EV. Each photo's fit residual becomes its label variance, so doubtful labels count less.

Fitted labels are unreliable in six known situations:

- DRO, HDR+ or Lightroom local tone maps move quantiles locally, so their labels are "effective" EV.
- Highlights at code 255 carry no level.
- Saturated colours differ between colour sciences, so white is fitted on neutrals only.
- Low-key and high-key intent cannot be told from error.
- Sunsets and stage light have no neutrals.
- Contrast fitted to an engine with a different tone curve partly measures that curve.

The last two cases are where a professional's judgment matters most. They need human signal: the owner's choices or the unpaired critic below.

Known-offset self-supervision covers what references cannot. Scaling a raw by 2^k, with EXIF adjusted to match, must move the prediction by exactly −k. Equivariance is therefore a loss term with exact labels and unlimited samples. Yang et al. showed that pretraining on synthetic ±2 EV shifts beat ImageNet initialisation for every FiveK expert ([Yang 2018](https://arxiv.org/abs/1803.02269)). One policy question sits inside this. Today exposure compensation dialled on the camera reaches the picture in full, and "a mis-exposed shot is not rescued" ([docs/STATUS.md](../docs/STATUS.md)). A model trained only for equivariance would undo deliberate EC. The design gives the model EC as an input and keeps EC in the target, so dialled-in intent survives while metering errors are corrected. Whether Auto should rescue at all is Mykyta's call.

The stage-1 objective is therefore the Gaussian NLL of each residual against fitted labels, plus three terms:

- an equivariance term;
- burst consistency, so frames of a burst share a prediction;
- an L2 pull towards zero, so the camera's choice holds unless the evidence is strong.

The licence consequence is settled. Because HDR+ and most Play Raw raws are CC BY-SA, **`autoset`'s weights are CC BY-SA 4.0**. They are separate data files in `data/models/autoset/` with their own `LICENSE`, listed in `NOTICE` and `THIRD-PARTY.md`, while the code stays Apache-2.0 ([docs/14 rule 5](../docs/14-licensing.md), [docs/16](../docs/16-risks-and-decisions.md)). A companion `autoset-clean`, trained only on CC0, CC BY and synthetic data, costs one extra training run. The A/B then shows what the ShareAlike data actually buys.

## A Hu-style critic can carry the owner's taste where labels run out

The owner's preference for Exposure's results points at the one route that needs no paired labels, which are V1's scarcest ingredient. It suits "punchy and professional" particularly well, because that is a holistic judgment of a finished picture. A professional recognises it without being able to write it down as EV and contrast targets per photo.

In this route, an **actor** reads the thumbnail features and outputs Yellowmoon's own settings: EV, temperature and tint offset, white lift and the bounded contrast. The settings drive a **differentiable copy of Yellowmoon's render**. A **critic** compares those renders with a collection of finished photos whose look we want, and the actor is trained until its renders are indistinguishable from that collection. No raw needs a matching edit, so any unlabelled raw becomes training input, including the sunsets, interiors and portraits where fitted labels fail. Hu's own result is the strongest argument: trained on only 2,000 unpaired inputs, the actor matched a paired pixel model and beat human novices in user ratings ([Hu 2018](https://arxiv.org/abs/1709.09602)).

**The render must be ours, not Hu's filters.** Yellowmoon's stages are analytic or reference-defined ([docs/02](../docs/02-colour-pipeline.md)):

- C1–C6 are fused 3 × 3 matrices, and exposure is 2^EV;
- C7 is a closed-form monotone gain;
- LogC4 is closed form;
- C9 Contrast is a pivot about grey;
- the Signature look is a luma curve plus saturation;
- the lift is a smoothstep;
- the display placement is 2^0.85.

These port directly to JAX or PyTorch. ACES 2's gamut compression involves iterative solves, so it is baked into a dense 3D LUT (for example 65³ over a log shaper) with differentiable interpolation. A parity test against `ym render`, with a stated maximum code error, keeps the proxy honest, in the style of the existing CPU/GPU parity tests. The proxy is exact rather than learned. That is easier than Tseng et al.'s black-box ISP proxies, which already beat months of expert tuning ([Tseng 2019](https://light.princeton.edu/proxy_opt/)). With a differentiable render and a fixed stage order, Hu's RL agent disappears: the actor is a single-step regressor, and the critic's gradient reaches its weights directly.

**Determinism holds where it matters.** Hu's one-to-many dropout is removed. The shipped actor is the same f64 MLP as `autoset`, run without dropout or sampling, and the critic is discarded after training. Training is seeded per rule 8. GAN runs differ between seeds more than regression runs do, which is a reason to pick a seed by the A/B; it does not make the shipped artefact any less deterministic.

**The target set decides the licence and the style.** Licence-clean candidates exist, but each brings a foreign look:

- HDR+ `final.jpg` (CC BY-SA, Google's local tone mapping);
- embedded Sony JPEGs in CC0 ARWs (CC0, Sony's curve, only about 200 files);
- Cube++ camera JPEGs (CC BY, Canon's curve);
- Play Raw edits, only where each editor licensed them CC0, BY or BY-SA ([Play Raw](https://discuss.pixls.us/t/why-do-we-need-to-add-a-license-to-play-raw-images/46116));
- the CC BY and BY-SA subset of KonIQ-10k, whose per-image licences are mixed ([KonIQ](https://arxiv.org/pdf/1803.08489), [VILA attribution table](https://arxiv.org/pdf/2303.14302)).

Wikimedia Commons' freely licensed JPEGs were not examined in these notes ([Commons:File types](https://commons.wikimedia.org/wiki/Commons:File_types)). A critic trained on foreign renders tells them apart from ours by tone-curve shape and local contrast, which the actor cannot reproduce, and then pushes EV, lift and contrast to compensate. The result is style drift towards Google's or Sony's look and away from the ALEXA one.

The stronger target set is **renders through Yellowmoon itself that the owner approved as punchy and professional**. Public raws are rendered at their stage-1 settings and shown to Mykyta as contact sheets; every render he accepts joins the target distribution. Inside our renderer the critic can only learn what the settings control: brightness placement, warmth, highlight lift and contrast. Owner taste steers through two levers. Curation decides what "finished" means, and the scene mix of the accepted set decides which kinds of photo the critic knows. If the targets are renders of CC0 and CC BY raws only, the actor's weights can stay Apache-2.0; any CC BY-SA raw or image in the target set makes them CC BY-SA 4.0. Whether Mykyta's own existing finished photos may join the target set is a separate decision. `docs/test-photo-sources.md` allows them only for testing and evaluation ([docs/test-photo-sources.md](../docs/test-photo-sources.md)).

**It is weaker than supervised regression in three ways.**

1. **Instability.** Hu's authors say RL and WGAN training need "much human labor and expertise … to properly set the hyper-parameters" ([Hu 2018](https://arxiv.org/abs/1709.09602)). Distort-and-Recover dropped learned aesthetic rewards because "the results were not stable enough" ([Park 2018](https://arxiv.org/abs/1804.04450)).
2. **Style drift.** An unpaired critic reproduces the *marginal* look of its targets, so any content gap between the training raws and the targets becomes a push on the settings. More sunsets in the targets warms everything; more bright travel photos brightens interiors. That is the same "generic" failure reviewers report for Adaptive. The mitigations:
   - condition the critic on Bv, `illum`'s class and the scene vector;
   - stratify the target set by scene type;
   - keep HDR+'s factor-of-8 luminance bound;
   - keep the tanh limits and the shrinkage.
3. **Evaluation.** Critic loss is not a quality measure, and there is no ground truth to score against. Hu reported histogram intersections and Mechanical Turk ratings, and fidelity scores against one expert later ranked it last ([Dutt 2025](https://arxiv.org/abs/2505.06176)). The blind A/B is the only gate.

**So the critic ranks as stage 2 and builds on the plan rather than replacing it.** The regression `autoset` ships first and becomes the actor's initialisation. Stage 2 fine-tunes it on a combined loss: NLL anchor + λ_adv·critic + λ_BT·Bradley–Terry utility, the utility fitted on Mykyta's pairwise A/B choices through the same proxy. The anchor keeps the actor near verified labels, the critic supplies holistic taste on unlabelled scenes, and the pairwise term ties both to the owner's judgments. The A/B decides whether λ_adv earns its place.

## Blind wins on everyday scenes define "punchy and professional"

The operational definition proposed here is this: **a photo is punchy and professional on open when, shown blind, the owner would deliver it as it is, and prefers it to the alternatives.** It is measured three ways, all on 100–150 public Sony raws (CC0 and CC BY or BY-SA Play Raw), held out by source from all training and fitting. The raws are stratified into about ten everyday scene categories. Mykyta tags them himself, not the teachers, so the evaluation is not circular:

- portrait, including backlit portrait;
- group;
- daylight landscape;
- architecture;
- street and travel;
- food and close-up;
- golden hour and sunset;
- snow or beach;
- interior under tungsten or mixed light;
- low-key or mood.

A small night city or stage stratum serves only as a guard that night stays night; low light is not the goal.

**Preference.** The blind web page planned in `docs/13` runs on a calibrated P3 display, with random order and side, ties allowed and each pair shown twice ([docs/13](../docs/13-testing.md)). The prompt is "Which would you deliver as it is?". **Track A** compares whole pictures: new Yellowmoon, current Yellowmoon, Lightroom Classic Auto (tone and white, Adobe default profile) and the camera JPEG, plus Adaptive, Capture One or DxO auto if Mykyta has them. **Track B** isolates the settings. It renders through the same Yellowmoon pipeline the model's settings, settings fitted to Lightroom Auto's output, settings fitted to the camera JPEG, and today's rule. A Track-B win supports the clean claim "our automatic settings beat Lightroom's", free of colour-science differences. The proposed gates are:

- at least **60 % of non-tie wins against Lightroom Auto and against the camera JPEG**, pooled over the everyday strata, with the Wilson 95 % lower bound above 50 %;
- at least 45 % in every stratum (non-inferiority);
- reported per stratum, with Bradley–Terry scores and bootstrap intervals.

This is stricter than `docs/13`'s whole-app bar of "ours preferred or tied in ≥ 50 % of pairs", because the owner asked for better, not equal. Detecting 60 % against 50 % needs about 194 independent non-tie judgments; with one observer that means 100+ photos × 2 presentations × 2–3 comparisons (the design notes of this study (working notes, not kept)).

**Scene ablation.** Scene understanding must earn its place category by category. The same Track-B set is rendered from four models, each one step up a ladder:

1. statistics only;
2. statistics plus faces;
3. statistics, faces and the scene vector;
4. the same plus the per-scene priors.

A step ships only if it wins where it claims to help: faces on the portrait strata; the scene vector on backlit portraits, snow or beach, sunsets, warm or mixed interiors and low-key scenes. **No category may get worse.** In every stratum the new step must hold at least 45 % of non-tie judgments against the step before (the non-inferiority margin), and its gross-failure rate must not rise (the scene notes of this study (working notes, not kept)). This is the test Bychkovsky's negative result calls for. SceneNet's own checks run alongside:

- a confusion matrix against Mykyta's tags, with per-class precision, recall and calibration (ECE);
- invariance: the top-class probability moves by less than 0.05 under ±2 EV and ±20 mired input shifts;
- prior assertions: the median snow EV is at least +0.7 over the mid-grey meter, sunsets are not neutralised beyond a set fraction of their cast, and night stays within the factor-of-8 band.

**Ready on open.** Mykyta corrects each condition's render to taste, blind to its source. The report gives:

- the share he left untouched;
- the median and 90th-percentile |ΔEV|, |Δmired| and |Δcontrast| he applied;
- the **gross-failure rate**, more than 0.5 EV or about 10 mired of correction.

The gate proposal is that `autoset`'s gross-failure rate is at most half of today's rule and below Lightroom Auto's, fitted through our renderer.

**Not overdone.** Punch fails as crushed shadows, clipped highlights, hot skin and colours that tire the eye. The industry evidence shows the risk is real: Lightroom's saturation is "heavy handed" ([Friedl](https://regex.info/blog/2018-04-20/2860)), DxO's ClearView Plus is heavy-handed at its default ([CaptureLandscapes](https://www.capturelandscapes.com/dxo-photolab/)), and Adobe chose a "non-opinionated" style precisely to avoid extreme contrast ([Adobe Blog](https://blog.adobe.com/en/publish/2024/10/14/the-adobe-adaptive-profile)). Four guards keep "punchy" honest:

- **Direction of corrections.** In the ready-on-open session, if Mykyta lowers contrast on more photos than he raises it, the model overshoots.
- **Clipping and crushing.** The share of pixels at display codes ≥ 250 and ≤ 5 is reported against the camera JPEG.
- **Skin.** Skin hue in face boxes is reported against the camera JPEG.
- **Overdone sentinel.** `autoset` with its contrast pushed well past the model's choice appears in a few pairs. If the sentinel beats `autoset`, the side-by-side test is rewarding vividness rather than quality, and the presentation should switch to single images with a "would you change this?" question.

Weak labels alone support a credible *non-regression* gate:

- G3 within ±0.5 EV on ≥ 80 % of "normal" held-out files;
- equivariance error ≤ 0.05 EV;
- burst spread near zero;
- the `illum` gates unchanged;
- fitted-label error on held-out HDR+ and Cube++.

They cannot support "better than industry", because every weak label is another engine's taste.

## Ranked build plan

The ranking orders work by expected gain against effort and licence risk. Each rank's gate must pass before the next changes defaults, and effort figures are rough estimates, not measurements. Ranks 2–4 do not depend on each other and can run in parallel once rank 1 exists. Rank 1 also supplies the statistics-only and statistics-plus-faces baselines that the scene ablation needs.

| Rank | Deliverable | Data | Model | Gates | Effort (estimate) | Weights licence |
|---|---|---|---|---|---|---|
| 0 | Foundations: maker-note parsing in `ym-raw` (DRO, Creative Style, metering mode, exposure program, flash, scene mode); `training/autoset/labels.py` (mid-tone G3 EV fit, as-shot white, joint lift and contrast fit, residual as variance); `tools/fetch` for HDR+ and Cube++ with SHA-256; evaluation harness fitting our settings to FiveK, MSEC and PPR10K experts (evaluation only); Track-B mode, ready-on-open session and sentinel in `tools/abtest` | all sources above | none | fetch checksums; label fit reproduces G3 to 0.02 EV on synthetic charts | about 1–2 weeks | none |
| 1 | **`autoset` v1**: ΔEV, (Δmired, ΔDuv), lift A, bounded contrast, four log-variances; shrinkage to today's rule; burst sharing extended to all outputs; faces as input once YuNet's provenance is cleared | CC0 Sony, HDR+, Cube++, Play Raw training part, known offsets, synth | MLP of about 30–70k parameters, 5-seed ensemble, f64 Rust | non-regression gates above; Rust–Python parity ≤ 1e-9; FiveK expert-band ≥ 90 % (reported); **[HUMAN] blind A/B: ≥ 60 % vs Lightroom Auto and vs camera JPEG pooled, ≥ 45 % per stratum, overdone guards clean** | about 2–3 weeks plus A/B | CC BY-SA 4.0 (`autoset-clean` sibling Apache-2.0 for comparison) |
| 2 | **Preferred white**: the analytic partial-adaptation rule (2–3 constants), then the learned `autoset-wb` map (12 inputs → 16 → 16 → 8 → 2, 634 parameters); card copy says "preferred white", not "measured light" | Cube++ (neutral, camera-AWB) pairs mapped to CCT/Duv; Sony as-shot whites against `illum`'s histogram-only estimate | rule, then tiny MLP | Track-B wins on sunset, interior and portrait strata; \|Δmired\| edit distance | about 1 week | Apache-2.0 (Cube++ is CC BY) |
| 3 | **`illum` v2**: real Cube++ training, Bv and noise metadata, cross-body locus embedding with imaginary-camera augmentation, brightness augmentation, Bv prior redrawn to 14–16 for sun | Cube++, synthetic; INTEL-TAU only if its licence is confirmed | FFCC core, 5–50k parameters | mean, median, Tri, B25, **W25** and W5 on held-out Cube++-converted and synthetic sets; sensor-map gate re-run | about 2 weeks | Apache-2.0 (CC BY-SA if INTEL-TAU joins) |
| 4 | **SceneNet and the scene-aware head** (the north-star step): an encoder trained from scratch giving 16 scene probabilities plus 4 cues; FiLM conditioning and per-scene offsets initialised from the prior table. Teacher labelling and the 300-image [HUMAN] audit can start during rank 1 | Open Images V7 CC BY images plus cleared raws rendered through Yellowmoon; labels from Qwen3-VL, InternVL3.5 and Gemma 4 (greedy decoding, pinned revisions, two-of-three agreement) and SigLIP 2 zero-shot; settings head on rank-1 labels | 1–2.5M parameters (2–5 MB f16) or 3.8M (7.6 MB); ONNX Runtime CPU provider, fixed threads; head as rank 1 | teacher accuracy per class from the audit; confusion matrix and ECE against Mykyta's tags; invariance < 0.05 under ±2 EV and ±20 mired; golden scene vector within 1e-4; **ablation ladder: wins on its strata, no category worse (≥ 45 % per stratum, no rise in gross failures)**; prior assertions | about 3–4 weeks plus the audit | Apache-2.0 with a CC BY pool; CC BY-SA 4.0 if BY-SA images join |
| 5 | **Stage 2 through the renderer**: differentiable proxy (closed-form C1–C11, lift, LogC4, look, placement; ACES 2 baked to a 65³ LUT) with a parity gate; Bradley–Terry utility on owner pairs; **Hu-style critic** on owner-approved Yellowmoon renders; optional bounded saturation head | owner A/B pairs and accepted contact sheets; all unlabelled raws | rank-1 actor fine-tuned; critic discarded | proxy parity at a stated code error; Track-B wins against rank 1 and against Lightroom decisions re-rendered in Yellowmoon | about 4–6 weeks plus several owner sessions | Apache-2.0 if the targets are renders of CC0/CC BY raws only; else CC BY-SA 4.0 |
| 6 | Later: a 128 px CNN branch (about 56k parameters) if global features miss content; a personal offset file if allowed; per-tile local tone (V2 pipeline work) | as above | small additions | stratum and per-user A/B | about 1–2 weeks each | as the parent model |

Not recommended at any rank:

- FiveK-, MSEC- or PPR10K-trained weights, including distillation from them;
- a frozen web- or ImageNet-trained backbone in the app, unless you decide otherwise;
- MobileCLIP and DINOv3 at any stage, and Gemma 3, Llama 3.2 or Qwen2.5-VL-3B/72B as teachers;
- AGPL detectors;
- ONNX Runtime for the settings head, and the CoreML or GPU providers anywhere on the auto path;
- VLM agents inside the product;
- pixel-output exposure networks.

## Decisions only Mykyta can make

| Decision | Options | Recommendation | Blocks |
|---|---|---|---|
| **Rescue policy** | (a) keep "a mis-exposed shot is not rescued"; (b) correct metering errors but keep dialled EC as intent; (c) full rescue | (b): EC as input and kept in the target; reverses part of the 2026-10-14 decision | rank 1 targets |
| **Gate redesign** | keep G3 on all files, or restrict G3 to "normal" scenes and add the everyday-strata A/B (≥ 60 % pooled vs Lightroom Auto and camera JPEG, Wilson lower bound > 50 %, ≥ 45 % per stratum), gross-failure, ready-on-open and overdone guards | restrict and add; AGENTS lets thresholds change only with your agreement | rank 1 acceptance |
| **Contrast as an Auto setting** | keep the look's fixed contrast, or a bounded per-photo offset; whether the Tone card's AUTO covers contrast too | bounded offset in v1, shrinking to 0 when unsure, range set by you after the first A/B | rank 1 outputs, `docs/11` copy |
| **Saturation** | look default only, or a small bounded head | look default until the A/B asks for more | rank 5 |
| **Preferred-white strength** | neutral, Sony-like "Ambience", or stronger, by scene | set the rule's constants by eye in a blind test; card copy changes to "preferred white" | rank 2 |
| **VLM-teacher labels for shipped weights** | does rule 5 allow labels and soft targets from Apache-2.0 VLMs (Qwen3-VL, InternVL3.5, Gemma 4) that were themselves trained on web data? | yes, as route 2: the student's images stay CC BY, the teachers' licences put no clause on outputs, and pinned revisions are recorded in the model card; record the decision in `docs/16` | rank 4 labels |
| **Frozen web- or ImageNet-trained backbone** | never; or a frozen DINOv2-S or MobileNetV4 in the app | never in the app; DINOv2-S and SigLIP 2 only as offline teachers | rank 4 route |
| **Scene encoder size** | about 8 MB (MobileNetV4-Conv-S, 3.8M parameters) or 4 MB or less (width ×0.5–0.75, 1–2.5M) | start at 4 MB or less; grow to about 8 MB only if the ablation shows the smaller encoder losing on some stratum | rank 4 size, `docs/12` budget |
| **Teacher audit** | about 300 random images labelled by you before training | yes; it measures each teacher's accuracy per class | rank 4 training |
| **Face detection** | no face input; YuNet (MIT code, WIDER FACE terms unchecked); BlazeFace (Apache-2.0 per mirrors only) | check YuNet's training-data terms first; faces are the best-supported semantic input | rank 1 inputs |
| **Personalisation** | none; a small versioned offset file (a few coefficients) read by both the app and `ym render --auto-profile`, recorded in XMP | allow only as a preference file if rule 4 ("a few preferences") stretches to it; few-shot offsets help with as few as 2–3 photos ([Caicedo 2011](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/12/CVPR-2011a.pdf), [Bychkovsky 2011](https://people.csail.mit.edu/vladb/photoadjust/db_imageadjust.pdf)) | rank 6 |
| **A/B sessions** | about 100–150 photos × 2 presentations × 2–3 comparisons, a ready-on-open pass, and Lightroom Auto (optionally Adaptive, Capture One, DxO) renders on a Mac | yes; Adobe's terms on publishing benchmark comparisons are unknown, so keep the results internal until checked | every acceptance gate |
| **Critic target set** | owner-approved Yellowmoon renders; foreign finished photos (HDR+, Play Raw edits); your own existing finished photos | owner-approved renders of CC0/CC BY raws; your own photos only if you extend `docs/test-photo-sources.md` to training | rank 5 |
| **INTEL-TAU licence** | the Finnish registry says CC BY-SA 4.0; `docs/14` says evaluation-only | confirm on the Etsin landing page (`urn:nbn:fi:att:f8b62270-d471-4036-b427-f21bce32b965`), then update `docs/14` and `STATUS.md` | rank 3 data |
| **Play Raw inventory** | the forum blocks automated listing (HTTP 503) | a manual pass, or access from a host the forum allows, to count Sony ARWs and their licences | size of the held-out A/B set |

## Conclusion

The industry's gap is one of judgment. Every tool surveyed either aims at a neutral starting point for someone else to finish, or, like Lightroom Auto, finishes with lifted shadows, low contrast and strong saturation, a style that reviewers and Adobe's own newer design both move away from. "Like a professional, on open" therefore means confident, scene-dependent decisions about mid-tone placement, contrast and warmth, with a doubt that falls back to the camera's choice. A 0.04 ms network on good features, plus a scene encoder of a few MB that tells it what the photo is, is enough to make them. Whether the encoder earns its place is decided category by category. The literature proves semantics for faces and local effects but not yet for global exposure on raw, so the ablation decides, not the ambition. The scarce ingredient is licence-clean evidence of taste on real subjects, so an hour of the owner's blind A/B or contact-sheet approval is worth more than any public dataset. It is also the only evidence that can back the claim "better than Lightroom". The Hu-style critic is the way to make that hour scale, provided its targets are Yellowmoon's own approved renders.

Global settings also have a ceiling. Adobe Adaptive, HDR+ and ProRAW all moved to spatially varying tone because one global curve cannot both protect a sky and open a face ([Adobe Blog](https://blog.adobe.com/en/publish/2024/10/14/the-adobe-adaptive-profile), [Lux](https://www.lux.camera/understanding-proraw/)). Once `autoset` has drawn everything it can from EV, white, lift and contrast, the next gain comes from a coarse per-tile gain grid predicted the same way. That is a pipeline change for V2.
