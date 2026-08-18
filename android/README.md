# EfficientSAM Android Demo

Tap-to-segment demo for the TFLite/LiteRT export, mirroring `ios/EfficientSAMDemo`.
Pick a photo, tap an object, get a mask.

## Models

The `.tflite` files are not committed. Generate them, then copy into assets:

```bash
jupyter nbconvert --execute --to notebook notebooks/EfficientSAM_tflite_export.ipynb
cp weights/tflite/*.tflite android/app/src/main/assets/
```

Four files are expected:

| asset | size |
| --- | --- |
| `efficient_sam_vitt_prompt_encoder.tflite` | 27.8 MB |
| `efficient_sam_vitt_prompt_decoder.tflite` | 20.6 MB |
| `efficient_sam_vits_prompt_encoder.tflite` | 95.5 MB |
| `efficient_sam_vits_prompt_decoder.tflite` | 20.6 MB |

Shipping only `vitt` halves the APK; drop `VITS` from `PromptSegmenter.Variant`
if you do, otherwise picking ViT-Small throws `ModelMissingException`.

## Build

```bash
cd android
./gradlew :app:assembleDebug     # or open in Android Studio
```

Requires JDK 17 and `sdk.dir` in `local.properties`. `minSdk` is 26.

## How it works

Two graphs, run at different rates:

- **Encoder** — `(1,3,1024,1024)` → `(1,256,64,64)`. Once per image, ~540 ms
  (vitt) / ~1130 ms (vits).
- **Decoder** — embedding + 2 points → `(1,1,3,256,256)` masks + `(1,1,3)` IoUs.
  Once per tap, ~28 ms.

Caching the embedding is what makes tapping feel instant.

### Two things the port must get right

**The input is a stretch, not a letterbox.** The encoder resizes to 1024x1024
without preserving aspect ratio, so prompt coordinates are scaled *per axis*.
A single scale factor is only correct for square images.

**The decoder output is unsorted.** The Core ML export sorts its three
candidates by predicted IoU in-graph; this one cannot — `torch.argsort` lowers
to `STABLEHLO_SORT`, which LiteRT has no kernel for. The host takes the argmax
instead. Candidate 0 is frequently *not* the best: on `figs/examples/dogs.jpg`
both test taps select candidate 2. Skipping the argmax silently returns worse
masks.

Prompt labels: `1` foreground, `-1` unused slot, `2`/`3` box top-left and
bottom-right. A box is one two-point prompt, not two prompts —
`segmentBox` covers it.

## Notes

- Assets need `noCompress += "tflite"`; the interpreter mmaps them out of the
  APK and a compressed asset has no mappable offset.
- The GPU delegate is best-effort. These graphs contain ops it may not support,
  so a delegate failure falls back to CPU instead of taking the interpreter down.
