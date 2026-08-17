# EfficientSAM iOS Demo

Pick a photo, run EfficientSAM's segment-everything on device, see the masks.

## Build

The Core ML models must exist before generating the project:

```bash
# 1. Export the models (from the repo root)
jupyter nbconvert --to notebook --execute --inplace \
  notebooks/EfficientSAM_coreml_export.ipynb

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
| `Segmenter.swift` | the whole pipeline — encode, batched decode, filter, NMS, render |
| `ContentView.swift` | landing screen, picker, progress, results |
| `generate_project.py` | writes `project.pbxproj`, `Info.plist`, scheme; copies models |

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
