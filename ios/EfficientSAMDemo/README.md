# EfficientSAM iOS Demo

Pick a photo and either segment everything at once, or tap an object to segment
just that one — both on device.

| mode | what it does | models |
|---|---|---|
| **Everything** | one pass over an N×N grid of prompts, returns every object | `*_encoder` / `*_decoder` |
| **Tap to Segment** | encode once, then one mask per tap | `*_prompt_encoder` / `*_prompt_decoder` |

## Build

The Core ML models must exist before generating the project. Both notebooks are
required — they produce **different** models, not two settings of one:

```bash
# 1. Export the models (from the repo root)
jupyter nbconvert --to notebook --execute --inplace \
  notebooks/EfficientSAM_coreml_export.ipynb
jupyter nbconvert --to notebook --execute --inplace \
  notebooks/EfficientSAM_prompt_coreml_export.ipynb

# 2. Generate the Xcode project and copy the models in
cd ios/EfficientSAMDemo
python3 generate_project.py

# 3. Open
open EfficientSAMDemo.xcodeproj
```

`generate_project.py` copies `weights/coreml/*.mlpackage` into the app target,
so re-run it after re-exporting. Set your own team in Signing & Capabilities to
run on a physical device; the simulator needs no signing.

## Layout

| file | role |
|---|---|
| `Segmenter.swift` | everything mode — encode, batched decode, filter, NMS, render |
| `PromptSegmenter.swift` | tap mode — encode once, one decoder call per tap |
| `ContentView.swift` | landing screen, mode picker, progress, results |
| `generate_project.py` | writes `project.pbxproj`, `Info.plist`, scheme; copies models |

## The two model families are not interchangeable

This is the thing most likely to waste your afternoon. The exports differ in
their **input contract**, and mixing them produces masks that look plausible and
are wrong:

| | segment-everything | prompt |
|---|---|---|
| resize to 1024 | letterbox, pad bottom-right | **stretch**, aspect not preserved |
| normalization | *not applied* (see below) | ImageNet mean/std, folded into graph |
| prompts per call | 8 queries × 1 point | 1 query × 2 points |
| labels used | `1` | `1`, or `2`/`3` for box corners |
| candidate sorting | host picks argmax IoU | sorted inside the graph, take index 0 |

Because the prompt models stretch, prompt coordinates scale by `1024/width` and
`1024/height` **independently**. A single uniform scale factor is only correct
for square images.

> **Known gap:** the segment-everything encoder skips `EfficientSam.preprocess`
> normalization — it feeds raw `[0,1]` to `patch_embed`. It is self-consistent
> end to end and produces sane masks, but it is off-distribution from training.
> The prompt export folds normalization into the graph and asserts against
> `get_image_embeddings` (the stock path), so it cannot regress the same way.
> Fixing the everything models means re-exporting and re-running
> `generate_project.py`; mask output will shift.

## How it works

The models are exported as a **split encoder / decoder** pair, which is what
makes this viable on device:

| stage | cost | frequency |
|---|---|---|
| encoder | ~0.15s (Mac), ~2.5s (simulator) | **once per image** |
| decoder | ~5ms per point | once per prompt batch |

`Segmenter` encodes the image once, caches the `(1, 256, 64, 64)` embedding, and
feeds it to every decoder call. Running the encoder per batch instead — as the
original Python notebook did — is what makes a naive port unusably slow.

Peak memory is `decoderBatch * 3 * 256 * 256` floats per call and **does not
scale with grid size**. A 32×32 grid costs the same peak as 8×8, just more
iterations. Masks are filtered by IoU and stability *before* anything is
upsampled, so the accumulator never holds full-resolution data.

## Constraints baked into the exported models

These are fixed at export time. Changing any of them means re-running the
notebook with new values and regenerating the project:

- **`encoderSize = 1024`** — the encoder input is exactly 1024×1024. Images are
  letterboxed to the **top-left**, padding bottom and right.
- **`decoderBatch = 8`** — the decoder's prompt shape is static. Partial final
  batches must be padded (`Segmenter` pads coords with zeros and labels with
  `-1`).
- **`outputSize = 256`** — decoder logit resolution. Survivors are upsampled to
  full resolution; everything filtered out never costs full-res memory.

## Two things that will silently produce wrong masks

Both of these were live bugs during development. Neither crashes — you get
plausible-looking masks in the wrong places.

**Point coordinates must be in 1024-space.** The grid is built against the
letterboxed size, not the original image dimensions. Passing raw image pixel
coordinates yields masks that look reasonable but address the wrong regions.

**The input tensor must not be row-flipped.** Core Graphics has a bottom-left
origin, so `ctx.draw(cg, in: CGRect(x: 0, y: size - fitH, ...))` already places
the image in the *top* rows of the buffer — matching the export. Copying rows in
reverse "to undo the CG flip" shifts the image down by the padding height, and
every grid point then lands on the wrong content.

## Notes

- Requires iOS 17. `MLModel.prediction(from:options:)` is async-only on iOS 17+;
  the synchronous `prediction(fromFeatures:)` is unavailable on iOS.
- The simulator logs `E5RT ... MpsGraph backend validation on incompatible OS`
  and falls back to CPU. Harmless, but simulator timings are far slower than
  real hardware — profile on device.
- `computeUnits = .all` lets Core ML schedule across ANE/GPU/CPU. Compare
  against `.cpuAndNeuralEngine` in Instruments before assuming ANE residency.
- Detail picker: Fast = 8×8 (64 prompts), Balanced = 16×16 (256), Thorough =
  32×32 (1024).
- Tap mode offers ViT-Tiny (10M) and ViT-Small (26M). Switching variants
  discards the cached embedding — it was produced by a different encoder and
  means nothing to the other decoder.

## One tap is a weak prompt

Measured on `figs/examples/dogs.jpg` at (580, 350), ViT-Tiny:

| prompt | predicted IoU | coverage |
|---|---|---|
| one point, second slot ignored | 0.564 | 2.0% |
| two points on the same dog | 0.904 | 11.8% |
| box around both dogs | 0.924 | 11.7% |

A single click returns a *part* of the object (here, the tan dog's chest) rather
than the whole animal. That is normal SAM behaviour for an ambiguous click, not
a coordinate bug — the returned mask is centred exactly on the tap.

Duplicating the tap into both prompt slots raises predicted IoU to 0.703 while
coverage stays at 1.9% — the same partial mask, just scored more confidently.
It is not an improvement, so `segment(atPoint:)` keeps the honest single-point
encoding (`label = -1` on the unused slot). Use `segmentPair(a:b:)` or
`segment(box:)` when a stronger prompt is available.
