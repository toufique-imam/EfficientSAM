package com.example.efficientsam

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.layout.onSizeChanged
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.IntSize
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlin.math.min

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            MaterialTheme(colorScheme = darkColorScheme()) {
                Surface(modifier = Modifier.fillMaxSize()) {
                    TapToSegmentScreen()
                }
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun TapToSegmentScreen() {
    val context = androidx.compose.ui.platform.LocalContext.current
    val scope = rememberCoroutineScope()

    val segmenter = remember { PromptSegmenter(context) }
    DisposableEffect(Unit) { onDispose { segmenter.close() } }

    var variant by remember { mutableStateOf(PromptSegmenter.Variant.VITT) }
    var sourceImage by remember { mutableStateOf<Bitmap?>(null) }
    var result by remember { mutableStateOf<PromptSegmenter.Result?>(null) }
    var isEncoding by remember { mutableStateOf(false) }
    var isDecoding by remember { mutableStateOf(false) }
    var encodeMillis by remember { mutableLongStateOf(0L) }
    var error by remember { mutableStateOf<String?>(null) }
    var selfTestResults by remember { mutableStateOf<List<PromptSegmenter.VariantCheck>?>(null) }
    var isSelfTesting by remember { mutableStateOf(false) }

    fun encode(bitmap: Bitmap) {
        scope.launch {
            isEncoding = true
            result = null
            try {
                segmenter.encode(bitmap, variant)
                encodeMillis = segmenter.encodeMillis
            } catch (e: Exception) {
                error = e.message ?: "Encoding failed."
                sourceImage = null
            } finally {
                isEncoding = false
            }
        }
    }

    val picker = rememberLauncherForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        if (uri == null) return@rememberLauncherForActivityResult
        scope.launch {
            try {
                val bitmap = withContext(Dispatchers.IO) { loadBitmap(context, uri) }
                sourceImage = bitmap
                encode(bitmap)
            } catch (e: Exception) {
                error = "Could not read that image."
            }
        }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("EfficientSAM") },
                actions = {
                    if (sourceImage != null) {
                        TextButton(onClick = {
                            sourceImage = null
                            result = null
                            encodeMillis = 0
                        }) { Text("New Image") }
                    }
                },
            )
        }
    ) { padding ->
        Box(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
        ) {
            val image = sourceImage
            if (image == null) {
                LandingView(
                    variant = variant,
                    onVariantChange = { variant = it },
                    onPick = { picker.launch("image/*") },
                    isSelfTesting = isSelfTesting,
                    selfTestResults = selfTestResults,
                    onSelfTest = {
                        scope.launch {
                            isSelfTesting = true
                            selfTestResults = null
                            try {
                                selfTestResults = segmenter.selfTest()
                            } catch (e: Exception) {
                                error = e.message ?: "Self test failed."
                            } finally {
                                isSelfTesting = false
                            }
                        }
                    },
                )
            } else {
                TapView(
                    image = image,
                    result = result,
                    variant = variant,
                    isEncoding = isEncoding,
                    isDecoding = isDecoding,
                    encodeMillis = encodeMillis,
                    onTap = { px, py ->
                        scope.launch {
                            isDecoding = true
                            try {
                                result = segmenter.segmentAtPoint(px, py)
                            } catch (e: Exception) {
                                error = e.message ?: "Segmentation failed."
                            } finally {
                                isDecoding = false
                            }
                        }
                    },
                )
            }
        }
    }

    error?.let { message ->
        AlertDialog(
            onDismissRequest = { error = null },
            title = { Text("Something went wrong") },
            text = { Text(message) },
            confirmButton = { TextButton(onClick = { error = null }) { Text("OK") } },
        )
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun LandingView(
    variant: PromptSegmenter.Variant,
    onVariantChange: (PromptSegmenter.Variant) -> Unit,
    onPick: () -> Unit,
    isSelfTesting: Boolean,
    selfTestResults: List<PromptSegmenter.VariantCheck>?,
    onSelfTest: () -> Unit,
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(horizontal = 32.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center,
    ) {
        Text(
            "Tap to Segment",
            style = MaterialTheme.typography.headlineSmall,
            fontWeight = FontWeight.SemiBold,
        )
        Spacer(Modifier.height(8.dp))
        Text(
            "Tap anything in the photo to segment just that object.",
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            textAlign = TextAlign.Center,
        )

        Spacer(Modifier.height(32.dp))

        // Split across two rows rather than one: the variant list is the cross
        // product of size and precision, and six segments in a row is
        // unreadable on a phone.
        Text("Model", style = MaterialTheme.typography.labelMedium)
        Spacer(Modifier.height(6.dp))
        val sizes = listOf("vitt" to "ViT-Tiny", "vits" to "ViT-Small")
        SingleChoiceSegmentedButtonRow {
            sizes.forEachIndexed { index, (id, label) ->
                SegmentedButton(
                    selected = variant.id == id,
                    onClick = {
                        // Keep the current precision if that combination was
                        // bundled, else fall back to whatever this size has.
                        val want = PromptSegmenter.Variant.entries.firstOrNull {
                            it.id == id && it.precision == variant.precision
                        }
                        onVariantChange(want ?: PromptSegmenter.Variant.entries.first { it.id == id })
                    },
                    shape = SegmentedButtonDefaults.itemShape(index = index, count = sizes.size),
                ) { Text(label) }
            }
        }

        Spacer(Modifier.height(10.dp))
        Text("Precision", style = MaterialTheme.typography.labelMedium)
        Spacer(Modifier.height(6.dp))
        SingleChoiceSegmentedButtonRow {
            PromptSegmenter.Precision.entries.forEachIndexed { index, p ->
                SegmentedButton(
                    selected = variant.precision == p,
                    onClick = {
                        PromptSegmenter.Variant.entries
                            .firstOrNull { it.id == variant.id && it.precision == p }
                            ?.let(onVariantChange)
                    },
                    shape = SegmentedButtonDefaults.itemShape(
                        index = index,
                        count = PromptSegmenter.Precision.entries.size,
                    ),
                ) { Text(p.label) }
            }
        }

        Spacer(Modifier.height(6.dp))
        Text(
            variant.blurb,
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )

        Spacer(Modifier.height(32.dp))
        Button(
            onClick = onPick,
            enabled = !isSelfTesting,
            modifier = Modifier.fillMaxWidth(),
        ) {
            Text("Pick an Image", fontWeight = FontWeight.SemiBold)
        }

        Spacer(Modifier.height(12.dp))

        OutlinedButton(
            onClick = onSelfTest,
            enabled = !isSelfTesting,
            modifier = Modifier.fillMaxWidth(),
        ) {
            if (isSelfTesting) {
                CircularProgressIndicator(
                    modifier = Modifier.size(16.dp),
                    strokeWidth = 2.dp,
                )
                Spacer(Modifier.width(10.dp))
                // Both encoders run here, so this is seconds, not milliseconds.
                Text("Testing both models…")
            } else {
                Text("Run Self Test")
            }
        }

        if (selfTestResults != null) {
            Spacer(Modifier.height(16.dp))
            SelfTestReport(selfTestResults)
        }
    }
}

/**
 * Per-variant pass/fail from [PromptSegmenter.selfTest]. Failures show the
 * reason rather than just a cross -- the useful cases (missing asset, shape
 * mismatch) are all distinguishable from the message.
 */
@Composable
private fun SelfTestReport(results: List<PromptSegmenter.VariantCheck>) {
    Surface(
        shape = MaterialTheme.shapes.medium,
        color = MaterialTheme.colorScheme.surfaceVariant,
        modifier = Modifier.fillMaxWidth(),
    ) {
        Column(modifier = Modifier.padding(14.dp)) {
            results.forEachIndexed { index, check ->
                if (index > 0) Spacer(Modifier.height(10.dp))
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(
                        if (check.passed) "PASS" else "FAIL",
                        style = MaterialTheme.typography.labelMedium,
                        fontFamily = FontFamily.Monospace,
                        color = if (check.passed) Color(0xFF34C759) else MaterialTheme.colorScheme.error,
                    )
                    Spacer(Modifier.width(10.dp))
                    Text(
                        check.variant.displayName,
                        style = MaterialTheme.typography.bodyMedium,
                        fontWeight = FontWeight.SemiBold,
                    )
                    if (check.passed) {
                        Spacer(Modifier.weight(1f))
                        Text(
                            if (check.usedGpu) "GPU" else "CPU",
                            style = MaterialTheme.typography.labelSmall,
                            fontFamily = FontFamily.Monospace,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                }
                Text(
                    check.detail,
                    style = MaterialTheme.typography.bodySmall,
                    color = if (check.passed) MaterialTheme.colorScheme.onSurfaceVariant
                            else MaterialTheme.colorScheme.error,
                )
                if (check.passed) {
                    Text(
                        "${check.encodeMillis}ms enc · ${check.decodeMillis}ms dec · " +
                            "${check.peakMemoryMb}MB peak · ${check.modelMemoryMb}MB model",
                        style = MaterialTheme.typography.labelSmall,
                        fontFamily = FontFamily.Monospace,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }
        }
    }
}

@Composable
private fun TapView(
    image: Bitmap,
    result: PromptSegmenter.Result?,
    variant: PromptSegmenter.Variant,
    isEncoding: Boolean,
    isDecoding: Boolean,
    encodeMillis: Long,
    onTap: (Float, Float) -> Unit,
) {
    Column(
        modifier = Modifier.fillMaxSize(),
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        // The overlay replaces the source image in place, so the tap target
        // geometry stays identical between states.
        val shown = result?.overlay ?: image
        var containerSize by remember { mutableStateOf(IntSize.Zero) }

        Box(
            modifier = Modifier
                .weight(1f)
                .fillMaxWidth()
                .padding(horizontal = 16.dp)
                .onSizeChanged { containerSize = it }
                .pointerInput(image, isEncoding, isDecoding) {
                    detectTapGestures { offset ->
                        if (isEncoding || isDecoding) return@detectTapGestures
                        val rect = fittedRect(
                            image.width.toFloat(), image.height.toFloat(),
                            containerSize.width.toFloat(), containerSize.height.toFloat(),
                        ) ?: return@detectTapGestures
                        // Ignore taps in the letterbox bars around the image.
                        if (offset.x < rect[0] || offset.x > rect[0] + rect[2] ||
                            offset.y < rect[1] || offset.y > rect[1] + rect[3]
                        ) return@detectTapGestures
                        val px = (offset.x - rect[0]) / rect[2] * image.width
                        val py = (offset.y - rect[1]) / rect[3] * image.height
                        onTap(px, py)
                    }
                },
            contentAlignment = Alignment.Center,
        ) {
            Image(
                bitmap = shown.asImageBitmap(),
                contentDescription = "Segmentation target",
                modifier = Modifier.fillMaxSize(),
                contentScale = ContentScale.Fit,
            )

            if (isEncoding || isDecoding) {
                // Loading sits over the image rather than replacing it, so the
                // photo stays visible and it is clear what is being worked on.
                Box(
                    modifier = Modifier
                        .fillMaxSize()
                        .background(Color.Black.copy(alpha = 0.35f)),
                    contentAlignment = Alignment.Center,
                ) {
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        CircularProgressIndicator(color = Color.White)
                        Spacer(Modifier.height(10.dp))
                        Text(
                            if (isEncoding) "Preparing image…" else "Segmenting…",
                            style = MaterialTheme.typography.bodySmall,
                            color = Color.White,
                        )
                    }
                }
            }
        }

        if (result != null) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 16.dp),
                horizontalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                Stat("%.2f".format(result.iou), "IoU", Modifier.weight(1f))
                Stat("%.0f%%".format(result.coverage * 100), "coverage", Modifier.weight(1f))
                Stat("${result.decodeMillis}ms", "decode", Modifier.weight(1f))
            }
        } else if (!isEncoding) {
            Text(
                "Tap anything in the photo",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }

        Spacer(Modifier.height(8.dp))
        Text(
            "${variant.displayName} · encoded once in ${"%.2f".format(encodeMillis / 1000f)}s",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        Spacer(Modifier.height(12.dp))
    }
}

@Composable
private fun Stat(value: String, label: String, modifier: Modifier = Modifier) {
    Surface(
        modifier = modifier,
        shape = MaterialTheme.shapes.medium,
        color = MaterialTheme.colorScheme.surfaceVariant,
    ) {
        Column(
            modifier = Modifier.padding(vertical = 10.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            Text(
                value,
                style = MaterialTheme.typography.titleMedium,
                fontFamily = FontFamily.Monospace,
            )
            Text(
                label,
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}

/**
 * Where a [ContentScale.Fit] image actually lands inside its container, as
 * [x, y, width, height]. Needed to turn a tap into image pixels -- using the
 * container directly would skew every coordinate by the letterbox bars.
 */
private fun fittedRect(
    imageW: Float,
    imageH: Float,
    containerW: Float,
    containerH: Float,
): FloatArray? {
    if (imageW <= 0f || imageH <= 0f || containerW <= 0f || containerH <= 0f) return null
    val scale = min(containerW / imageW, containerH / imageH)
    val w = imageW * scale
    val h = imageH * scale
    return floatArrayOf((containerW - w) / 2f, (containerH - h) / 2f, w, h)
}

/**
 * Decodes the picked image, downsampling anything huge. The encoder stretches
 * to 1024 regardless, so a 12 MP original costs memory without buying detail --
 * and the overlay is a full-size ARGB copy of whatever is kept here.
 */
private fun loadBitmap(context: android.content.Context, uri: Uri): Bitmap {
    val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
    context.contentResolver.openInputStream(uri)?.use {
        BitmapFactory.decodeStream(it, null, bounds)
    }

    var sample = 1
    val maxDim = 2048
    while (bounds.outWidth / sample > maxDim || bounds.outHeight / sample > maxDim) {
        sample *= 2
    }

    val options = BitmapFactory.Options().apply {
        inSampleSize = sample
        inPreferredConfig = Bitmap.Config.ARGB_8888
    }
    val bitmap = context.contentResolver.openInputStream(uri)?.use {
        BitmapFactory.decodeStream(it, null, options)
    } ?: throw IllegalStateException("decode failed")

    // getPixels/copy below assume ARGB_8888; a hardware or 565 bitmap would
    // either throw or quietly lose the channels the model reads.
    return if (bitmap.config == Bitmap.Config.ARGB_8888) bitmap
    else bitmap.copy(Bitmap.Config.ARGB_8888, false)
}
