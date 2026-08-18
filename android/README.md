# EfficientSAM Android Demo

Tap-to-segment demo for the TFLite/LiteRT export, mirroring `ios/EfficientSAMDemo`.
Pick a photo, tap an object, get a mask.

## Models

The `.tflite` files are not committed. Generate them, then copy into assets:

```bash
# float32 (required)
jupyter nbconvert --execute --to notebook notebooks/EfficientSAM_tflite_export.ipynb
# fp16 + int8 (optional)
python notebooks/export_quantized_tflite.py

cp weights/tflite/*.tflite android/app/src/main/assets/
```

Encoder sizes per precision; the decoder is 20.6 MB (12.6 MB fp16):

| encoder | fp32 | fp16 | int8 |
| --- | --- | --- | --- |
| vitt | 27.8 MB | 15.6 MB | 9.8 MB |
| vits | 95.5 MB | 51.1 MB | 29.4 MB |

`selfTest` only reports variants whose assets are actually bundled, so shipping
a subset is fine — but `PromptSegmenter.Variant` still lists them all, and
*selecting* an absent one throws `ModelMissingException`. Bundling everything
makes for a ~335 MB APK, which is fine for sideloading and not for Play.

### Precision notes

**fp16 does not make inference faster.** XNNPACK dequantizes to float32 to
compute, so fp16 buys download size, nothing else — measured encode times are
identical to fp32. **int8 is the one that actually helps**: ~22% (vitt) to ~35%
(vits) faster encode, and lower peak memory.

**int8 quantizes the encoder only.** Quantizing the decoder yields a graph that
the desktop LiteRT runtime accepts and Android then refuses to allocate:

```
transpose_conv.cc:312 weights->type != input->type (INT8 != FLOAT32)
```

The mask upsampler's `TRANSPOSE_CONV` needs weights matching its float
activations, so `_int8` pairs an int8 encoder with a float32 decoder. Little is
lost: the encoder holds nearly all the weight.

Accuracy against fp32 on `figs/examples/dogs.jpg` — fp16 is effectively
lossless (100% mask agreement), int8 stays ≥99.9% agreement / ≥98.3% mask-IoU,
and every variant picks the same candidate index.

## Build

```bash
cd android
./gradlew :app:assembleDebug     # or open in Android Studio
```

Requires JDK 17 and `sdk.dir` in `local.properties`. `minSdk` is 26.

## How it works

Two graphs, run at different rates:

- **Encoder** — `(1,3,1024,1024)` → `(1,256,64,64)`. Once per image.
- **Decoder** — embedding + 2 points → `(1,1,3,256,256)` masks + `(1,1,3)` IoUs.
  Once per tap.

Caching the embedding is what makes tapping feel instant: the encoder is
seconds, the decoder is well under one.

Measured with the in-app self test (fp32, CPU/XNNPACK):

| device | vitt encode | vits encode | decode |
| --- | --- | --- | --- |
| Galaxy S20 FE (Snapdragon 865) | ~7.5 s | ~18.9 s | ~0.3 s |
| Galaxy M12 (Exynos 850) | ~22.4 s | ~55.3 s | ~0.9 s |

Peak memory is substantial — 0.9-1.9 GB depending on variant — hence
`android:largeHeap="true"`.

### Why it runs on the CPU

`PromptSegmenter.useGpu` defaults to **false**. Not caution — measurement. The
delegate loads, then claims only 59 of the encoder's 636 nodes and builds
*zero* kernels for them:

```
SLICE: Max version supported: 2. Requested version 5.
RESHAPE / TRANSPOSE: OP is supported, but tensor type/shape isn't compatible.
TfLiteGpuDelegate Init: Batch size mismatch, expected 1 but got 3
```

This is not a dynamic-shape problem — both graphs are fully static (zero
tensors with a `-1` in `shape_signature`). It is op coverage in the ViT
attention blocks. Attempting the delegate costs init plus a failed interpreter
construction before falling back, which measurably *slowed* encode, so it is
off by default and left switchable for other hardware or a re-export.

Note also that the GPU dependency is `org.tensorflow:tensorflow-lite-gpu`, not
`com.google.ai.edge.litert:litert-gpu:1.0.1`. Every `GpuDelegate` constructor
in the latter — including the no-arg one — touches `GpuDelegateFactory$Options`,
a class that ships in none of the litert 1.0.1 artifacts, so it dies with
`NoClassDefFoundError`. The two families cannot be mixed: both publish
`org.tensorflow.lite.*` and the build fails on duplicate classes.

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
