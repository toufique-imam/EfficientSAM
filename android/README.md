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

Measured with the in-app **Benchmark Grid** (encoder, at each device's best
thread count — 4 on the S20 FE, 8 on the M12):

| variant | S20 FE (SD 865) | M12 (Exynos 850) | peak PSS |
| --- | --- | --- | --- |
| ViT-Tiny fp32 | 5.2 s | 19.6 s | ~1.0 GB |
| ViT-Tiny fp16 | 5.4 s | 20.4 s | ~1.1 GB |
| ViT-Tiny int8 | 5.1 s | 18.4 s | ~1.0 GB |
| ViT-Small fp32 | 12.7 s | 50.4 s | ~1.7 GB |
| ViT-Small fp16 | 14.7 s | 56.3 s | ~2.0 GB |
| ViT-Small int8 | **10.9 s** | **31.1 s** | ~1.6 GB |

Decode is ~0.2 s (S20 FE) / ~0.9 s (M12), flat across variant and thread count.

Peak memory is substantial — 1.0-2.0 GB depending on variant — hence
`android:largeHeap="true"`.

### Threads: more is not better

`PromptSegmenter.threads` defaults to the size of the **big cluster**, not the
core count: cores whose `cpuinfo_max_freq` is within 15% of the maximum. That
gives 4 on the S20 FE and 8 on the M12, and both are the measured optimum.

On the S20 FE (4× A55 + 4× A77) 4 threads beats 8 in *every* variant — TFLite
splits work evenly, so the little cores finish late and the big cores wait on
stragglers. On the single-tier M12 (8× A55) 8 beats 4, but only by ~7%.

Counting cores at exactly the top frequency would be wrong: the 865 clocks one
prime core to 2841 MHz and its three siblings to 2419 MHz, so an exact match
returns 1 and idles the rest of the big cluster.

fp16 is worth calling out separately: it is **not faster** on either device and
uses more memory, because the weights are stored fp16 and dequantized to fp32
to compute. int8 is the only variant that meaningfully wins, and mostly for
ViT-Small.

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
