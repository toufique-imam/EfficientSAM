package com.example.efficientsam

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import androidx.compose.runtime.Immutable
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

/**
 * Runs the interactive EfficientSAM path on LiteRT: tap a point, get one mask.
 *
 * This is the Android counterpart of the iOS `PromptSegmenter`, and it honours
 * the same two contracts baked into the export:
 *
 * - **Resize.** The encoder input is a *stretch* to 1024x1024, not an
 *   aspect-preserving fit. Every prompt coordinate is therefore scaled
 *   per-axis; a single scale factor would only be right for square images.
 * - **Prompt shape.** One query of two points. A box is *one* prompt whose
 *   corners are labelled 2 (top-left) and 3 (bottom-right), not two prompts.
 *
 * One thing differs from the Core ML build and it matters: the Core ML graph
 * sorts its three candidate masks by predicted IoU, but the TFLite export does
 * not -- `torch.argsort` lowers to STABLEHLO_SORT, which the LiteRT runtime has
 * no kernel for. Candidate 0 here is *not* the best one, so [bestIndex] picks
 * the argmax. Dropping that step silently returns a worse mask.
 *
 * The encoder/decoder split earns more here than in segment-everything: the
 * encoder runs once per image (~540 ms vitt, ~1130 ms vits on a mid device) and
 * each later tap costs only a decoder call (~28 ms).
 */
class PromptSegmenter(private val context: Context) {

    /** Which exported checkpoint to run. */
    enum class Variant(val id: String, val displayName: String, val blurb: String) {
        VITT("vitt", "ViT-Tiny", "10M params · faster to encode"),
        VITS("vits", "ViT-Small", "26M params · higher predicted IoU");

        val encoderAsset get() = "efficient_sam_${id}_prompt_encoder.tflite"
        val decoderAsset get() = "efficient_sam_${id}_prompt_decoder.tflite"
    }

    class ModelMissingException(asset: String) : Exception(
        "Model $asset not found in assets. Run notebooks/EfficientSAM_tflite_export.ipynb, " +
            "then copy weights/tflite/*.tflite into android/app/src/main/assets/."
    )

    class NotEncodedException : Exception("Image has not been encoded yet.")

    @Immutable
    data class Result(
        /** The original image with the mask tinted over it. */
        val overlay: Bitmap,
        /** Model-predicted IoU for the returned mask, in [0, 1]. */
        val iou: Float,
        /** Fraction of the image the mask covers. */
        val coverage: Float,
        val decodeMillis: Long,
    )

    private companion object {
        // Must match the values baked into the exported models. Changing any of
        // these means re-exporting from EfficientSAM_tflite_export.ipynb.
        const val ENCODER_SIZE = 1024
        const val OUTPUT_SIZE = 256
        const val NUM_POINTS = 2
        const val EMBED_C = 256
        const val EMBED_HW = 64
        const val NUM_CANDIDATES = 3

        const val MASK_COLOR = 0x7328C463.toInt() // green @ ~45% alpha
    }

    // One interpreter pair at a time; guarded because taps can overlap with a
    // variant switch coming from the UI thread.
    private val lock = Mutex()

    private var variant: Variant = Variant.VITT
    private var encoder: Interpreter? = null
    private var decoder: Interpreter? = null
    private var gpuDelegate: GpuDelegate? = null

    /** Cached per-image state, so taps only pay for the decoder. */
    private var embedding: ByteBuffer? = null
    private var sourceImage: Bitmap? = null
    private var origW = 0
    private var origH = 0

    var encodeMillis: Long = 0
        private set

    val isReady: Boolean get() = embedding != null && decoder != null

    // MARK: - Model loading

    private fun loadModels(variant: Variant) {
        // Switching variants invalidates the cached embedding: it came from a
        // different encoder and means nothing to this decoder.
        if (variant != this.variant) {
            this.variant = variant
            close()
        }
        if (encoder != null && decoder != null) return

        val options = Interpreter.Options().apply {
            numThreads = Runtime.getRuntime().availableProcessors().coerceAtMost(4)
        }

        // The GPU delegate is a best-effort speedup. These graphs contain ops
        // it may not support, and a delegate that fails at init takes the whole
        // interpreter down with it -- so a failure falls back to CPU rather
        // than propagating.
        if (CompatibilityList().isDelegateSupportedOnThisDevice) {
            runCatching {
                GpuDelegate().also {
                    gpuDelegate = it
                    options.addDelegate(it)
                }
            }
        }

        encoder = runCatching { Interpreter(mapAsset(variant.encoderAsset), options) }
            .getOrElse { cpuOnlyFallback(variant.encoderAsset) }
        decoder = runCatching { Interpreter(mapAsset(variant.decoderAsset), options) }
            .getOrElse { cpuOnlyFallback(variant.decoderAsset) }
    }

    private fun cpuOnlyFallback(asset: String): Interpreter {
        gpuDelegate?.close()
        gpuDelegate = null
        return Interpreter(
            mapAsset(asset),
            Interpreter.Options().apply { numThreads = 4 },
        )
    }

    /**
     * Maps the model directly out of the APK. Requires `noCompress += "tflite"`
     * in the Gradle config -- a compressed asset has no mappable file offset
     * and [android.content.res.AssetManager.openFd] throws for it.
     */
    private fun mapAsset(name: String): MappedByteBuffer {
        val fd = try {
            context.assets.openFd(name)
        } catch (e: Exception) {
            throw ModelMissingException(name)
        }
        return fd.use {
            FileInputStream(it.fileDescriptor).use { stream ->
                stream.channel.map(
                    FileChannel.MapMode.READ_ONLY,
                    it.startOffset,
                    it.declaredLength,
                )
            }
        }
    }

    // MARK: - Encoding

    /** Runs the encoder once and caches the embedding for subsequent taps. */
    suspend fun encode(image: Bitmap, variant: Variant) = withContext(Dispatchers.Default) {
        lock.withLock {
            loadModels(variant)
            val enc = encoder ?: throw ModelMissingException(variant.encoderAsset)

            sourceImage = image
            origW = image.width
            origH = image.height

            val input = makeImageInput(image)
            val output = ByteBuffer
                .allocateDirect(EMBED_C * EMBED_HW * EMBED_HW * 4)
                .order(ByteOrder.nativeOrder())

            val start = System.nanoTime()
            enc.run(input, output)
            encodeMillis = (System.nanoTime() - start) / 1_000_000

            output.rewind()
            embedding = output
        }
    }

    // MARK: - Prompting

    /**
     * Segments whatever is under ([x], [y]), given in **original image pixels**.
     *
     * The second prompt slot is marked unused (label -1) rather than filled
     * with a duplicate: the graph pads to its max point count with -1
     * internally, so an ignored slot is exactly what it expects.
     */
    suspend fun segmentAtPoint(x: Float, y: Float): Result =
        segment(floatArrayOf(x, 0f), floatArrayOf(y, 0f), floatArrayOf(1f, -1f))

    /**
     * Segments using two foreground points, both in original image pixels.
     * Two points on the same object usually beat one: the model gets a hint
     * about extent, not just location.
     */
    suspend fun segmentPair(ax: Float, ay: Float, bx: Float, by: Float): Result =
        segment(floatArrayOf(ax, bx), floatArrayOf(ay, by), floatArrayOf(1f, 1f))

    /**
     * Segments the contents of a box given by two opposite corners in original
     * image pixels. Labels 2/3 are the box top-left / bottom-right embeddings
     * the prompt encoder was trained with.
     */
    suspend fun segmentBox(left: Float, top: Float, right: Float, bottom: Float): Result =
        segment(floatArrayOf(left, right), floatArrayOf(top, bottom), floatArrayOf(2f, 3f))

    private suspend fun segment(
        xs: FloatArray,
        ys: FloatArray,
        labels: FloatArray,
    ): Result = withContext(Dispatchers.Default) {
        lock.withLock {
            val dec = decoder ?: throw NotEncodedException()
            val embed = embedding ?: throw NotEncodedException()
            val base = sourceImage ?: throw NotEncodedException()

            val coords = ByteBuffer.allocateDirect(NUM_POINTS * 2 * 4)
                .order(ByteOrder.nativeOrder())
            val labelBuf = ByteBuffer.allocateDirect(NUM_POINTS * 4)
                .order(ByteOrder.nativeOrder())

            for (i in 0 until NUM_POINTS) {
                // Per-axis scale, because the encoder input is a stretch. A
                // uniform factor would only be correct for square images.
                coords.putFloat(xs[i] * ENCODER_SIZE / origW)
                coords.putFloat(ys[i] * ENCODER_SIZE / origH)
                labelBuf.putFloat(labels[i])
            }
            coords.rewind()
            labelBuf.rewind()
            embed.rewind()

            // Inputs are bound positionally: the converter names them args_0..2
            // rather than anything meaningful, so index order is the contract.
            val inputs = arrayOf<Any>(embed, coords, labelBuf)
            val masks = ByteBuffer
                .allocateDirect(NUM_CANDIDATES * OUTPUT_SIZE * OUTPUT_SIZE * 4)
                .order(ByteOrder.nativeOrder())
            val ious = ByteBuffer
                .allocateDirect(NUM_CANDIDATES * 4)
                .order(ByteOrder.nativeOrder())
            val outputs = mapOf<Int, Any>(0 to masks, 1 to ious)

            val start = System.nanoTime()
            dec.runForMultipleInputsOutputs(inputs, outputs)
            val decodeMillis = (System.nanoTime() - start) / 1_000_000

            masks.rewind()
            ious.rewind()

            // The one line the Core ML build does not need: this graph returns
            // its candidates unsorted, so rank them here.
            val iouValues = FloatArray(NUM_CANDIDATES) { ious.getFloat(it * 4) }
            val best = bestIndex(iouValues)

            val plane = OUTPUT_SIZE * OUTPUT_SIZE
            val offset = best * plane * 4
            val mask = BooleanArray(plane)
            var area = 0
            for (i in 0 until plane) {
                // Logits, not probabilities: >= 0 is the foreground threshold.
                if (masks.getFloat(offset + i * 4) >= 0f) {
                    mask[i] = true
                    area++
                }
            }

            Result(
                overlay = renderOverlay(base, mask, xs, ys, labels),
                iou = iouValues[best],
                coverage = area.toFloat() / plane,
                decodeMillis = decodeMillis,
            )
        }
    }

    private fun bestIndex(ious: FloatArray): Int {
        var best = 0
        for (i in ious.indices) if (ious[i] > ious[best]) best = i
        return best
    }

    // MARK: - Input preparation

    /**
     * Stretches the image to fill 1024x1024 and returns it as a CHW float
     * buffer in [0, 1].
     *
     * Deliberately **not** aspect-preserving -- see the class doc. ImageNet
     * mean/std normalization is folded into the exported graph, so these values
     * stay in [0, 1] and must not be normalized again here.
     */
    private fun makeImageInput(image: Bitmap): ByteBuffer {
        val scaled = Bitmap.createScaledBitmap(image, ENCODER_SIZE, ENCODER_SIZE, true)
        val pixels = IntArray(ENCODER_SIZE * ENCODER_SIZE)
        scaled.getPixels(pixels, 0, ENCODER_SIZE, 0, 0, ENCODER_SIZE, ENCODER_SIZE)
        if (scaled != image) scaled.recycle()

        val plane = ENCODER_SIZE * ENCODER_SIZE
        val buffer = ByteBuffer.allocateDirect(3 * plane * 4).order(ByteOrder.nativeOrder())
        val floats = buffer.asFloatBuffer()

        // CHW, so each channel is written as a contiguous plane rather than
        // interleaved per pixel.
        for (i in 0 until plane) {
            floats.put(i, ((pixels[i] shr 16) and 0xFF) / 255f)
            floats.put(plane + i, ((pixels[i] shr 8) and 0xFF) / 255f)
            floats.put(plane * 2 + i, (pixels[i] and 0xFF) / 255f)
        }
        return buffer
    }

    // MARK: - Rendering

    private fun renderOverlay(
        base: Bitmap,
        mask: BooleanArray,
        xs: FloatArray,
        ys: FloatArray,
        labels: FloatArray,
    ): Bitmap {
        val out = base.copy(Bitmap.Config.ARGB_8888, true)
        val w = out.width
        val h = out.height
        val row = IntArray(w)

        val mr = Color.red(MASK_COLOR)
        val mg = Color.green(MASK_COLOR)
        val mb = Color.blue(MASK_COLOR)
        val alpha = Color.alpha(MASK_COLOR) / 255f

        // The mask is 256x256 regardless of image size, so it is sampled
        // nearest-neighbour per destination row. Blending per pixel avoids the
        // 1px striping a rect-per-run fill produces at fractional scales.
        for (y in 0 until h) {
            val my = (y.toLong() * OUTPUT_SIZE / h).toInt().coerceIn(0, OUTPUT_SIZE - 1)
            out.getPixels(row, 0, w, 0, y, w, 1)
            var touched = false
            for (x in 0 until w) {
                val mx = (x.toLong() * OUTPUT_SIZE / w).toInt().coerceIn(0, OUTPUT_SIZE - 1)
                if (!mask[my * OUTPUT_SIZE + mx]) continue
                val p = row[x]
                row[x] = Color.argb(
                    255,
                    (Color.red(p) * (1 - alpha) + mr * alpha).toInt(),
                    (Color.green(p) * (1 - alpha) + mg * alpha).toInt(),
                    (Color.blue(p) * (1 - alpha) + mb * alpha).toInt(),
                )
                touched = true
            }
            if (touched) out.setPixels(row, 0, w, 0, y, w, 1)
        }

        // Mark the prompt so it is obvious what was asked for.
        val radius = (maxOf(w, h) / 90f).coerceAtLeast(4f)
        val canvas = android.graphics.Canvas(out)
        val paint = android.graphics.Paint(android.graphics.Paint.ANTI_ALIAS_FLAG)
        for (i in xs.indices) {
            if (labels[i] < 0f) continue
            paint.color = Color.WHITE
            canvas.drawCircle(xs[i], ys[i], radius, paint)
            paint.color = Color.rgb(52, 199, 89)
            canvas.drawCircle(xs[i], ys[i], radius * 0.65f, paint)
        }
        return out
    }

    fun close() {
        encoder?.close()
        decoder?.close()
        gpuDelegate?.close()
        encoder = null
        decoder = null
        gpuDelegate = null
        embedding = null
    }
}
