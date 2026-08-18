package com.example.efficientsam

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Debug
import android.util.Log
import androidx.compose.runtime.Immutable
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.exp

/**
 * Samsung's `interactiveseg.tflite`, benchmarked next to EfficientSAM as a
 * point of comparison.
 *
 * It solves the same user-facing problem -- click an object, get its mask --
 * with a completely different architecture, which is the interesting part:
 *
 * |                | EfficientSAM        | interactiveseg     |
 * |----------------|---------------------|--------------------|
 * | shape          | split encoder/decoder | single stage     |
 * | input          | NCHW 1024x1024      | NHWC 400x400 + guidance |
 * | graph          | ViT, attention-heavy | CNN, 315 CONV_2D  |
 * | per-tap cost   | decoder only (embedding cached) | full model |
 *
 * Because it is one stage there is nothing to cache: every click re-runs the
 * whole network. That would normally be the expensive design, but the network
 * is small enough that a full run still costs less than EfficientSAM's decoder
 * alone -- see the benchmark output.
 *
 * Contract (verified against the reference playground and re-probed here):
 * two inputs, image `(1,400,400,3)` RGB in [0,1] and guidance
 * `(1,400,400,5)`; output `(1,400,400,1)` of **raw logits**, so a sigmoid is
 * required. Guidance channel 0 marks positive clicks and channel 4 negative
 * ones, each drawn as a filled disk of radius 10 px. The other three channels
 * exist but are not part of the documented interface.
 */
class InteractiveSeg(private val context: Context) {

    private companion object {
        const val TAG = "EfficientSAM"
        const val ASSET = "interactiveseg.tflite"
        const val SIZE = 400
        const val GUIDANCE_CHANNELS = 5
        const val POSITIVE_CHANNEL = 0
        const val CLICK_RADIUS_PX = 10

        /** Same photo and prompt the EfficientSAM benchmark uses, so the numbers line up. */
        const val TEST_IMAGE = "test_image.jpg"
        const val TEST_PROMPT_X = 0.30f
        const val TEST_PROMPT_Y = 0.55f
    }

    @Immutable
    data class Result(
        val threads: Int,
        val inferenceMillis: Long,
        /** Fraction of the frame above 0.5 probability. */
        val coverage: Float,
        val peakMemoryMb: Int,
        val modelMemoryMb: Int,
    )

    val isBundled: Boolean
        get() = context.assets.list("")?.contains(ASSET) == true

    // Kept between taps. There is no embedding to cache -- every click reruns
    // the whole network -- but the interpreter and the source bitmap are worth
    // holding onto.
    private var interpreter: Interpreter? = null
    private var sourceImage: Bitmap? = null
    private var imageInput: ByteBuffer? = null

    var threads: Int = Runtime.getRuntime().availableProcessors()

    /**
     * Prepares for tapping: builds the interpreter and pre-resizes the image.
     *
     * Named to mirror [PromptSegmenter.encode] so the UI can treat the two
     * models the same way, but it does far less -- there is no encoder pass
     * here, just setup, so it returns in milliseconds.
     */
    suspend fun prepare(image: Bitmap) = withContext(Dispatchers.Default) {
        close()
        sourceImage = image
        imageInput = makeImageInput(image)
        interpreter = Interpreter(
            mapAsset(),
            Interpreter.Options().apply { numThreads = threads },
        )
    }

    val isReady: Boolean get() = interpreter != null && imageInput != null

    /**
     * Segments whatever is under ([x], [y]), given in original image pixels.
     *
     * Returns the same shape as [PromptSegmenter.Result] so the tap UI does not
     * need to branch. `iou` is reported as the mean probability inside the
     * mask: this model has no IoU head, and leaving the field at zero would
     * read as a confident-but-terrible prediction rather than "not available".
     */
    suspend fun segment(x: Float, y: Float): PromptSegmenter.Result =
        withContext(Dispatchers.Default) {
            val itp = interpreter ?: throw PromptSegmenter.NotEncodedException()
            val image = imageInput ?: throw PromptSegmenter.NotEncodedException()
            val base = sourceImage ?: throw PromptSegmenter.NotEncodedException()

            val guidanceStart = System.nanoTime()
            val guidance = makeGuidance(x / base.width, y / base.height)
            val logits = ByteBuffer
                .allocateDirect(SIZE * SIZE * 4)
                .order(ByteOrder.nativeOrder())

            image.rewind()
            val inputs = if (itp.getInputTensor(0).shape().last() == 3) {
                arrayOf<Any>(image, guidance)
            } else {
                arrayOf<Any>(guidance, image)
            }
            val start = System.nanoTime()
            itp.runForMultipleInputsOutputs(inputs, mapOf<Int, Any>(0 to logits))
            val ms = (System.nanoTime() - start) / 1_000_000
            logits.rewind()
            val guidanceMs = (start - guidanceStart) / 1_000_000

            val plane = SIZE * SIZE
            val mask = BooleanArray(plane)
            var area = 0
            var probSum = 0f
            for (i in 0 until plane) {
                val p = 1f / (1f + exp(-logits.getFloat(i * 4)))
                if (p >= 0.5f) {
                    mask[i] = true
                    area++
                    probSum += p
                }
            }

            val postMs = (System.nanoTime() - start) / 1_000_000 - ms
            val overlayStart = System.nanoTime()
            val overlay = renderOverlay(base, mask, x, y)
            Log.i(
                TAG,
                "interactiveseg tap: guidance ${guidanceMs}ms invoke ${ms}ms " +
                    "sigmoid ${postMs}ms overlay ${(System.nanoTime() - overlayStart) / 1_000_000}ms",
            )

            PromptSegmenter.Result(
                overlay = overlay,
                iou = if (area > 0) probSum / area else 0f,
                coverage = area.toFloat() / plane,
                decodeMillis = ms,
            )
        }

    /** Tints the mask over the source image, matching the EfficientSAM overlay. */
    private fun renderOverlay(base: Bitmap, mask: BooleanArray, px: Float, py: Float): Bitmap {
        val out = base.copy(Bitmap.Config.ARGB_8888, true)
        val w = out.width
        val h = out.height
        val row = IntArray(w)
        val color = 0x7328C463
        val mr = android.graphics.Color.red(color)
        val mg = android.graphics.Color.green(color)
        val mb = android.graphics.Color.blue(color)
        val alpha = android.graphics.Color.alpha(color) / 255f

        // The mask is 400x400 regardless of image size, sampled nearest per row.
        for (y in 0 until h) {
            val my = (y.toLong() * SIZE / h).toInt().coerceIn(0, SIZE - 1)
            out.getPixels(row, 0, w, 0, y, w, 1)
            var touched = false
            for (x in 0 until w) {
                val mx = (x.toLong() * SIZE / w).toInt().coerceIn(0, SIZE - 1)
                if (!mask[my * SIZE + mx]) continue
                val p = row[x]
                row[x] = android.graphics.Color.argb(
                    255,
                    (android.graphics.Color.red(p) * (1 - alpha) + mr * alpha).toInt(),
                    (android.graphics.Color.green(p) * (1 - alpha) + mg * alpha).toInt(),
                    (android.graphics.Color.blue(p) * (1 - alpha) + mb * alpha).toInt(),
                )
                touched = true
            }
            if (touched) out.setPixels(row, 0, w, 0, y, w, 1)
        }

        val radius = (maxOf(w, h) / 90f).coerceAtLeast(4f)
        val canvas = android.graphics.Canvas(out)
        val paint = android.graphics.Paint(android.graphics.Paint.ANTI_ALIAS_FLAG)
        paint.color = android.graphics.Color.WHITE
        canvas.drawCircle(px, py, radius, paint)
        paint.color = android.graphics.Color.rgb(52, 199, 89)
        canvas.drawCircle(px, py, radius * 0.65f, paint)
        return out
    }

    fun close() {
        interpreter?.close()
        interpreter = null
        imageInput = null
        sourceImage = null
    }

    private fun mapAsset(): MappedByteBuffer {
        val fd = try {
            context.assets.openFd(ASSET)
        } catch (e: Exception) {
            throw PromptSegmenter.ModelMissingException(ASSET)
        }
        return fd.use {
            FileInputStream(it.fileDescriptor).use { stream ->
                stream.channel.map(FileChannel.MapMode.READ_ONLY, it.startOffset, it.declaredLength)
            }
        }
    }

    /** Times one inference per thread count, mirroring [PromptSegmenter.benchmarkGrid]. */
    suspend fun benchmark(
        counts: List<Int> = listOf(1, 4, Runtime.getRuntime().availableProcessors()),
    ): List<Result> = withContext(Dispatchers.Default) {
        val bitmap = context.assets.open(TEST_IMAGE).use { stream ->
            BitmapFactory.decodeStream(
                stream, null,
                BitmapFactory.Options().apply { inPreferredConfig = Bitmap.Config.ARGB_8888 },
            )
        } ?: return@withContext emptyList()

        val image = makeImageInput(bitmap)
        val guidance = makeGuidance(TEST_PROMPT_X, TEST_PROMPT_Y)
        bitmap.recycle()

        counts.distinct().sorted().mapNotNull { n ->
            var interpreter: Interpreter? = null
            try {
                System.gc()
                val baseline = Debug.getNativeHeapAllocatedSize()

                interpreter = Interpreter(
                    mapAsset(),
                    Interpreter.Options().apply { numThreads = n },
                )

                // Inputs are bound by index. The two are distinguished by their
                // channel count rather than by name or order, because the
                // converter's ordering is not part of the contract.
                val inputs = interpreter.let { itp ->
                    val a = itp.getInputTensor(0).shape().last()
                    if (a == 3) arrayOf<Any>(image, guidance) else arrayOf<Any>(guidance, image)
                }
                val logits = ByteBuffer
                    .allocateDirect(SIZE * SIZE * 4)
                    .order(ByteOrder.nativeOrder())
                val outputs = mapOf<Int, Any>(0 to logits)

                image.rewind(); guidance.rewind()
                val start = System.nanoTime()
                interpreter.runForMultipleInputsOutputs(inputs, outputs)
                val ms = (System.nanoTime() - start) / 1_000_000
                logits.rewind()

                val peakNative = Debug.getNativeHeapAllocatedSize()
                val memInfo = Debug.MemoryInfo()
                Debug.getMemoryInfo(memInfo)

                // Raw logits: sigmoid, then threshold at 0.5. Equivalent to
                // logit >= 0, but kept explicit to match the reference.
                var above = 0
                for (i in 0 until SIZE * SIZE) {
                    val p = 1f / (1f + exp(-logits.getFloat(i * 4)))
                    if (p >= 0.5f) above++
                }
                val coverage = above.toFloat() / (SIZE * SIZE)
                val modelMb = ((peakNative - baseline) / (1024 * 1024)).toInt().coerceAtLeast(0)
                val peakMb = memInfo.totalPss / 1024

                Log.i(
                    TAG,
                    "interactiveseg threads=$n infer ${ms}ms cov ${"%.1f".format(coverage * 100)}% " +
                        "peakPss ${peakMb}MB model ${modelMb}MB",
                )
                Result(n, ms, coverage, peakMb, modelMb)
            } catch (e: Exception) {
                Log.w(TAG, "interactiveseg threads=$n failed: ${e.message}", e)
                null
            } finally {
                interpreter?.close()
            }
        }
    }

    /** Bilinear resize to 400x400, NHWC float32 in [0,1]. */
    private fun makeImageInput(image: Bitmap): ByteBuffer {
        val scaled = Bitmap.createScaledBitmap(image, SIZE, SIZE, true)
        val pixels = IntArray(SIZE * SIZE)
        scaled.getPixels(pixels, 0, SIZE, 0, 0, SIZE, SIZE)
        if (scaled != image) scaled.recycle()

        val buffer = ByteBuffer.allocateDirect(SIZE * SIZE * 3 * 4).order(ByteOrder.nativeOrder())
        val floats = buffer.asFloatBuffer()
        // NHWC, so channels are interleaved per pixel -- the opposite of the
        // planar layout EfficientSAM's NCHW encoder wants.
        for (i in pixels.indices) {
            val p = pixels[i]
            floats.put(i * 3, ((p shr 16) and 0xFF) / 255f)
            floats.put(i * 3 + 1, ((p shr 8) and 0xFF) / 255f)
            floats.put(i * 3 + 2, (p and 0xFF) / 255f)
        }
        return buffer
    }

    /** One positive click as a filled disk in guidance channel 0. */
    private fun makeGuidance(fx: Float, fy: Float): ByteBuffer {
        val buffer = ByteBuffer
            .allocateDirect(SIZE * SIZE * GUIDANCE_CHANNELS * 4)
            .order(ByteOrder.nativeOrder())
        val floats = buffer.asFloatBuffer()
        val cx = fx * SIZE
        val cy = fy * SIZE
        val r2 = CLICK_RADIUS_PX * CLICK_RADIUS_PX
        for (y in 0 until SIZE) {
            val dy = y - cy
            for (x in 0 until SIZE) {
                val dx = x - cx
                if (dx * dx + dy * dy <= r2) {
                    floats.put((y * SIZE + x) * GUIDANCE_CHANNELS + POSITIVE_CHANNEL, 1f)
                }
            }
        }
        return buffer
    }
}
