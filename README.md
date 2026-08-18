# EfficientSAM — mobile ports

Read the main README here:
https://github.com/yformer/EfficientSAM/blob/main/README.md

This fork adds on-device ports of EfficientSAM plus the export tooling and
benchmarks behind them. Nothing in the model code changed; the work is in
getting the graphs onto phones and measuring what they cost there.

| | iOS | Android |
|---|---|---|
| runtime | Core ML | LiteRT / TF Lite |
| app | [`ios/EfficientSAMDemo`](ios/EfficientSAMDemo/) ([README](ios/EfficientSAMDemo/README.md)) | [`android/`](android/) ([README](android/README.md)) |
| modes | segment-everything + tap-to-segment | tap-to-segment |
| export | [`EfficientSAM_coreml_export.ipynb`](notebooks/EfficientSAM_coreml_export.ipynb), [`EfficientSAM_prompt_coreml_export.ipynb`](notebooks/EfficientSAM_prompt_coreml_export.ipynb) | [`EfficientSAM_tflite_export.ipynb`](notebooks/EfficientSAM_tflite_export.ipynb), [`export_quantized_tflite.py`](notebooks/export_quantized_tflite.py) |

Both apps use the same split-export trick: run the encoder **once per image**,
cache the `(1, 256, 64, 64)` embedding, and pay only a decoder call per tap.
Re-encoding per prompt is what makes a naive port unusable.

## Export contract

The prompt models stretch the input to 1024×1024 **without preserving aspect
ratio**, so prompt coordinates scale by `1024/width` and `1024/height`
*independently*. A single uniform scale is only correct for square images.

Two exports of the same model differ in ways that silently produce wrong masks
if mixed up — see the [iOS README](ios/EfficientSAMDemo/README.md#the-two-model-families-are-not-interchangeable)
for the full table.

The TFLite decoder additionally returns its three mask candidates **unsorted**.
Core ML sorts them in-graph; TFLite cannot, because `torch.argsort` lowers to
`STABLEHLO_SORT` and LiteRT has no kernel for it. The host must take the argmax
over the IoU vector — on `figs/examples/dogs.jpg`, candidate 0 is not the best
one for either test prompt.

## Quantization

`notebooks/export_quantized_tflite.py` reuses the fp32 notebook's wrappers to
emit fp16 and int8 builds. Encoder sizes:

| encoder | fp32 | fp16 | int8 |
|---|---|---|---|
| vitt | 27.8 MB | 15.6 MB | 9.8 MB |
| vits | 95.5 MB | 51.1 MB | 29.4 MB |

Accuracy against fp32 on `dogs.jpg`: fp16 is effectively lossless (100% mask
agreement), int8 stays ≥99.9% agreement and ≥98.3% mask-IoU, and every variant
selects the same candidate index.

**int8 quantizes the encoder only.** Quantizing the decoder yields a graph the
desktop LiteRT runtime accepts and Android then refuses to allocate
(`transpose_conv.cc:312 weights->type != input->type (INT8 != FLOAT32)`) — the
mask upsampler's `TRANSPOSE_CONV` needs weights matching its float activations.
This reproduces only on device.

## Benchmarks

Measured in-app (`Benchmark Grid`), encoder time on `assets/test_image.jpg`,
across two devices with very different CPUs:

**Galaxy S20 FE** — Snapdragon 865, 4× A55 @1.8 GHz + 4× A77 @2.4–2.8 GHz:

| variant | 1 thread | 4 threads | 8 threads | peak PSS |
|---|---|---|---|---|
| ViT-Tiny fp32 | 7.9 s | **5.2 s** | 5.4 s | 1034 MB |
| ViT-Tiny fp16 | 7.7 s | **5.4 s** | 5.7 s | 1147 MB |
| ViT-Tiny int8 | 9.2 s | **5.1 s** | 5.5 s | 989 MB |
| ViT-Small fp32 | 17.8 s | **12.7 s** | 13.6 s | 1752 MB |
| ViT-Small fp16 | 20.9 s | **14.7 s** | 16.0 s | 1959 MB |
| ViT-Small int8 | 21.9 s | **10.9 s** | 11.3 s | 1614 MB |

**Galaxy M12** — Exynos 850, 8× Cortex-A55 @2.0 GHz (single tier):

| variant | 1 thread | 4 threads | 8 threads | peak PSS |
|---|---|---|---|---|
| ViT-Tiny fp32 | 36.0 s | 21.2 s | **19.6 s** | 980 MB |
| ViT-Tiny fp16 | 35.6 s | 21.4 s | **20.4 s** | 1098 MB |
| ViT-Tiny int8 | 42.1 s | 19.8 s | **18.4 s** | 948 MB |
| ViT-Small fp32 | 83.3 s | 53.5 s | **50.4 s** | 1700 MB |
| ViT-Small fp16 | 83.5 s | 54.0 s | **56.3 s** | 1993 MB |
| ViT-Small int8 | 87.7 s | 35.2 s | **31.1 s** | 1542 MB |

Decode is ~0.2 s on the S20 FE and ~0.9–1.2 s on the M12, flat across variant
and thread count, so only encode is worth optimizing.

Four results worth carrying forward:

- **More threads is not better.** On the S20 FE, 4 threads beats 8 in *every*
  variant: TFLite splits work evenly, so the four little A55s finish late and
  the big cores wait on stragglers. On the single-tier M12, 8 beats 4 — but only
  by ~7%. `PromptSegmenter.threads` therefore defaults to the size of the big
  cluster (cores within 15% of the top `cpuinfo_max_freq`), which yields 4 on
  the S20 FE and 8 on the M12.
- **fp16 does not speed anything up.** It matches or trails fp32 on both devices
  and uses *more* memory — weights are stored fp16 and dequantized to fp32 to
  compute, so they exist twice. It is a download-size win only.
- **int8 is the only real speedup, and only for ViT-Small.** At its best thread
  count it cuts vits from 12.7 s to 10.9 s (S20 FE) and 50.4 s to 31.1 s (M12),
  with the lowest memory of the three. For ViT-Tiny it roughly ties fp32. It is
  also *slower* than fp32 at 1 thread on both devices.
- **Threading saturates early.** A third of the encoder's 637 ops are
  `RESHAPE`/`TRANSPOSE`, which move data without doing arithmetic, so the graph
  is closer to memory-bound than compute-bound.

### The GPU delegate does not work for these graphs

Measured, not assumed. The delegate loads, claims 59 of the encoder's 636 nodes,
then builds **zero** kernels:

```
SLICE: Max version supported: 2. Requested version 5.
RESHAPE / TRANSPOSE: OP is supported, but tensor type/shape isn't compatible.
TfLiteGpuDelegate Init: Batch size mismatch, expected 1 but got 3
```

This is not a dynamic-shape problem — both graphs are fully static. It is op
coverage in the ViT attention blocks. Attempting the delegate costs init plus a
failed interpreter construction before falling back, which measurably *slowed*
encoding, so `PromptSegmenter.useGpu` defaults to `false`.

### Where the time actually goes

Encoder cost is dominated by attention over 4096 tokens (1024 px ÷ 16 patch),
not by data movement: one 4096-token score matrix costs ~10 ms against ~40 ms
for all 38 transposes combined. Lowering the encoder resolution therefore
attacks the real cost, but it trades accuracy on small objects — measured in
PyTorch across six prompts:

| resolution | speedup | mean mask-IoU vs 1024 | worst |
|---|---|---|---|
| 768 px | ~3.5× | 93% | 68% |
| 512 px | 9–13× | 73–88% | 6.6% |

512 px looks nearly lossless on a single large-object prompt and then collapses
on small ones, so it is not a safe default. Either resolution requires a
re-export: `get_abs_pos` interpolates the position embeddings, and the decoder's
dense positional encoding must be rebuilt for the new grid.

## Layout

```
android/                          Compose tap-to-segment app + benchmarks
ios/EfficientSAMDemo/             SwiftUI app, both modes
notebooks/EfficientSAM_tflite_export.ipynb    fp32 TFLite export
notebooks/export_quantized_tflite.py          fp16 + int8 export
notebooks/EfficientSAM_*coreml_export.ipynb   Core ML exports
```

Model weights are not committed — see each app's README for the export and copy
steps.
