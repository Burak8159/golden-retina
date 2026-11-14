package org.goldenhorn8159.goldenretina

import android.os.Bundle
import android.content.Context
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import androidx.compose.foundation.text.KeyboardOptions
import androidx.activity.ComponentActivity
import androidx.activity.compose.BackHandler
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.animation.AnimatedContent
import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.togetherWith
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.Home
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.filled.Lock
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Search
import androidx.compose.material.icons.filled.Security
import androidx.compose.material.icons.filled.Sensors
import androidx.compose.material.icons.filled.Visibility
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CenterAlignedTopAppBar
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.dynamicDarkColorScheme
import androidx.compose.material3.dynamicLightColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.rotate
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.drawscope.rotate
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.semantics.heading
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import kotlinx.coroutines.delay
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.coroutines.launch
import org.json.JSONObject
import java.io.File
import java.net.HttpURLConnection
import java.net.URL
import java.security.MessageDigest

private val GoldenGreen = Color(0xFF176B55)
private val GoldenGreenDark = Color(0xFFA0D9C2)
private val Pine = Color(0xFF103A34)
private val SoftMint = Color(0xFFDDF2E8)

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent { GoldenRetinaTheme { GoldenRetinaApp() } }
    }
}

@Composable
private fun GoldenRetinaTheme(content: @Composable () -> Unit) {
    val dark = isSystemInDarkTheme()
    val context = LocalContext.current
    val scheme = if (android.os.Build.VERSION.SDK_INT >= 31) {
        if (dark) dynamicDarkColorScheme(context) else dynamicLightColorScheme(context)
    } else if (dark) {
        darkColorScheme(primary = GoldenGreenDark, onPrimary = Color(0xFF00382B), secondary = SoftMint)
    } else {
        lightColorScheme(primary = GoldenGreen, onPrimary = Color.White, secondary = Pine, secondaryContainer = SoftMint)
    }
    MaterialTheme(colorScheme = scheme, content = content)
}

private enum class Destination { HOME, ABOUT, DEMO }
private enum class DemoPhase { WELCOME, SEARCHING, CONNECTED, RIGHT_SCAN, RIGHT_DONE, LEFT_SCAN, LEFT_DONE, ANALYZING, RESULTS }

@Composable
private fun GoldenRetinaApp() {
    var destination by rememberSaveable { mutableStateOf(Destination.HOME.name) }
    BackHandler(enabled = destination == Destination.DEMO.name) { destination = Destination.HOME.name }

    when (Destination.valueOf(destination)) {
        Destination.HOME, Destination.ABOUT -> MainTabs(
            selected = Destination.valueOf(destination),
            onSelect = { destination = it.name },
            onStartDemo = { destination = Destination.DEMO.name },
        )
        Destination.DEMO -> DemoExperience(onExit = { destination = Destination.HOME.name })
    }
}

@Composable
private fun MainTabs(selected: Destination, onSelect: (Destination) -> Unit, onStartDemo: () -> Unit) {
    Scaffold(
        bottomBar = {
            NavigationBar {
                NavigationBarItem(
                    selected = selected == Destination.HOME,
                    onClick = { onSelect(Destination.HOME) },
                    icon = { Icon(Icons.Filled.Home, contentDescription = null) },
                    label = { Text("Home") },
                )
                NavigationBarItem(
                    selected = selected == Destination.ABOUT,
                    onClick = { onSelect(Destination.ABOUT) },
                    icon = { Icon(Icons.Filled.Info, contentDescription = null) },
                    label = { Text("About") },
                )
            }
        },
    ) { insets ->
        when (selected) {
            Destination.HOME -> HomeScreen(Modifier.padding(insets), onStartDemo)
            Destination.ABOUT -> AboutScreen(Modifier.padding(insets))
            Destination.DEMO -> Unit
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun HomeScreen(modifier: Modifier = Modifier, onStartDemo: () -> Unit) {
    var showHardwareInfo by rememberSaveable { mutableStateOf(false) }
    Scaffold(
        modifier = modifier,
        topBar = {
            CenterAlignedTopAppBar(
                title = {
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        Text("GOLDEN HORN 8159", style = MaterialTheme.typography.labelSmall, letterSpacing = 1.3.sp, color = MaterialTheme.colorScheme.primary)
                        Text("Golden Retina", style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.Bold)
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(containerColor = MaterialTheme.colorScheme.background),
            )
        },
    ) { inner ->
        BoxWithConstraints(modifier = Modifier.fillMaxSize().padding(inner)) {
            Column(
                modifier = Modifier.align(Alignment.TopCenter).widthIn(max = 680.dp).fillMaxSize().verticalScroll(rememberScrollState()).padding(horizontal = 20.dp, vertical = 12.dp),
                verticalArrangement = Arrangement.spacedBy(20.dp),
            ) {
                HomeHero()
                Column(verticalArrangement = Arrangement.spacedBy(5.dp)) {
                    Text("Choose how to continue", style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.Bold, modifier = Modifier.semantics { heading() })
                    Text("Try the guided demo, or learn about the goggles connection planned for later.", style = MaterialTheme.typography.bodyLarge, color = MaterialTheme.colorScheme.onSurfaceVariant)
                }
                Card(
                    onClick = onStartDemo,
                    shape = RoundedCornerShape(24.dp),
                    colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceContainerLow),
                    modifier = Modifier.fillMaxWidth(),
                ) {
                    Row(Modifier.fillMaxWidth().padding(18.dp), verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(14.dp)) {
                        Surface(shape = CircleShape, color = MaterialTheme.colorScheme.primaryContainer, modifier = Modifier.size(52.dp)) {
                            Box(contentAlignment = Alignment.Center) { Icon(Icons.Filled.PlayArrow, contentDescription = null, tint = MaterialTheme.colorScheme.onPrimaryContainer, modifier = Modifier.size(30.dp)) }
                        }
                        Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(5.dp)) {
                            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                                Text("Explore the demo", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
                                DemoPill()
                            }
                            Text("Device setup, both eye scans, and example results", style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
                        }
                    }
                }
                Card(
                    onClick = { showHardwareInfo = true },
                    shape = RoundedCornerShape(24.dp),
                    colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceContainerLow),
                    modifier = Modifier.fillMaxWidth(),
                ) {
                    Row(Modifier.fillMaxWidth().padding(18.dp), verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(14.dp)) {
                        Icon(Icons.Filled.Visibility, contentDescription = null, tint = MaterialTheme.colorScheme.onSurfaceVariant, modifier = Modifier.size(30.dp))
                        Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                            Text("Connect goggles", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
                            Text("Hardware pairing is unavailable", style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
                        }
                        Surface(shape = CircleShape, color = MaterialTheme.colorScheme.surfaceContainerHighest) {
                            Row(Modifier.padding(horizontal = 10.dp, vertical = 7.dp), verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(5.dp)) {
                                Icon(Icons.Filled.Lock, contentDescription = null, modifier = Modifier.size(15.dp), tint = MaterialTheme.colorScheme.onSurfaceVariant)
                                Text("Locked", style = MaterialTheme.typography.labelMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
                            }
                        }
                    }
                }
                Row(horizontalArrangement = Arrangement.spacedBy(10.dp), verticalAlignment = Alignment.Top, modifier = Modifier.padding(bottom = 10.dp)) {
                    Icon(Icons.Filled.Security, contentDescription = null, tint = MaterialTheme.colorScheme.primary)
                    Text("Demo only · No image is captured, no AI runs, and no health result is produced.", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                }
            }
        }
    }
    if (showHardwareInfo) {
        androidx.compose.material3.AlertDialog(
            onDismissRequest = { showHardwareInfo = false },
            icon = { Icon(Icons.Filled.Lock, contentDescription = null, tint = MaterialTheme.colorScheme.primary) },
            title = { Text("Goggles are still in development") },
            text = { Text("There is no finished goggles hardware or Bluetooth protocol to pair with yet. The real connection will be added after the device and its communication method are ready.") },
            confirmButton = { TextButton(onClick = { showHardwareInfo = false }) { Text("Got it") } },
        )
    }
}

@Composable
private fun HomeHero() {
    Row(
        modifier = Modifier.fillMaxWidth().clip(RoundedCornerShape(28.dp)).background(Pine).padding(20.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(6.dp),
    ) {
        Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(10.dp)) {
            Text("A clearer view of eye screening", style = MaterialTheme.typography.headlineSmall, color = Color.White, fontWeight = FontWeight.Bold)
            Text("Follow a simple, step-by-step product demonstration.", style = MaterialTheme.typography.bodyMedium, color = Color.White.copy(alpha = .84f))
            Text("GUIDED EXPERIENCE", style = MaterialTheme.typography.labelSmall, color = SoftMint, letterSpacing = 1.sp)
        }
        RetinaArtwork(Modifier.size(112.dp).clip(CircleShape))
    }
}

@Composable
private fun DemoPill() {
    Surface(shape = CircleShape, color = MaterialTheme.colorScheme.primaryContainer) {
        Text("DEMO", modifier = Modifier.padding(horizontal = 9.dp, vertical = 5.dp), style = MaterialTheme.typography.labelSmall, fontWeight = FontWeight.Bold, color = MaterialTheme.colorScheme.onPrimaryContainer, letterSpacing = .7.sp)
    }
}

@Composable
private fun AboutScreen(modifier: Modifier = Modifier) {
    val context = LocalContext.current
    val prefs = remember { context.getSharedPreferences("model_registry", Context.MODE_PRIVATE) }
    var registryUrl by rememberSaveable { mutableStateOf(prefs.getString("url", "") ?: "") }
    var status by rememberSaveable { mutableStateOf("No model registry configured. The app continues in demo-only mode.") }
    var busy by remember { mutableStateOf(false) }
    val initialModelStates = remember {
        mapOf("retinal_classifier" to "Not installed", "ir_eye_tracker" to "Not installed") +
            prefs.all.keys.filter { it.startsWith("model:") }.associate { key ->
                key.removePrefix("model:") to (prefs.getString(key, "Not installed") ?: "Not installed")
            }
    }
    var modelStates by remember { mutableStateOf(initialModelStates) }
    val scope = rememberCoroutineScope()
    BoxWithConstraints(modifier.fillMaxSize()) {
        Column(Modifier.align(Alignment.TopCenter).widthIn(max = 680.dp).fillMaxWidth().verticalScroll(rememberScrollState()).padding(horizontal = 20.dp, vertical = 16.dp).padding(bottom = 20.dp), verticalArrangement = Arrangement.spacedBy(16.dp)) {
            Text("About", style = MaterialTheme.typography.headlineMedium, fontWeight = FontWeight.Bold, modifier = Modifier.semantics { heading() })
            Text("An early software concept from Golden Horn 8159.", style = MaterialTheme.typography.bodyLarge, color = MaterialTheme.colorScheme.onSurfaceVariant)
            Card(shape = RoundedCornerShape(24.dp), colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceContainerLow)) {
                Column(Modifier.padding(18.dp), verticalArrangement = Arrangement.spacedBy(14.dp)) {
                    Text("About this demo", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
                    AboutFact("Every scan and result is simulated", Icons.Filled.PlayArrow)
                    AboutFact("No camera images are collected", Icons.Filled.Visibility)
                    AboutFact("No goggles or Bluetooth are active", Icons.Filled.Lock)
                    AboutFact("Not a medical device or diagnosis", Icons.Filled.Security)
                }
            }
            Card(shape = RoundedCornerShape(24.dp), colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceContainerLow)) {
                Column(Modifier.padding(18.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
                    Text("On-device models", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
                    Text("The design keeps inference on this phone. The registry only distributes model files; it does not receive eye images or run inference. This demo does not yet run a disease or tracking model.", style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
                    OutlinedTextField(
                        value = registryUrl,
                        onValueChange = { registryUrl = it },
                        label = { Text("HTTPS registry base URL") },
                        keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Uri),
                        singleLine = true,
                        modifier = Modifier.fillMaxWidth(),
                    )
                    Button(enabled = !busy, onClick = {
                        prefs.edit().putString("url", registryUrl.trim()).apply()
                        busy = true
                        status = "Checking model registry…"
                        // The request only fetches the manifest and model artifacts. No image is uploaded.
                        scope.launch {
                            val result = runCatching { PhoneModelRegistry.refresh(context, registryUrl.trim()) }
                            status = result.fold({ reply ->
                                modelStates = reply
                                "Registry checked. Model files are local and are not yet activated for scan inference."
                            }, { "Registry check failed: ${it.message ?: "network or manifest error"}" })
                            busy = false
                        }
                    }) {
                        Icon(Icons.Filled.Refresh, contentDescription = null)
                        Text("  Check for model updates")
                    }
                    if (busy) LinearProgressIndicator(Modifier.fillMaxWidth())
                    Text(status, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                    modelStates.toSortedMap().forEach { (id, state) ->
                        HorizontalDivider()
                        Column(verticalArrangement = Arrangement.spacedBy(3.dp)) {
                            Text(if (id == "ir_eye_tracker") "IR eye tracker" else if (id == "retinal_classifier") "Retinal disease head · one task per release" else id.replace('_', ' '), style = MaterialTheme.typography.labelLarge, fontWeight = FontWeight.SemiBold)
                            Text(state, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                        }
                    }
                }
            }
            Text("The current checkpoints are research prototypes and are not yet qualified mobile inference bundles. The guided scan remains simulated and does not assess a person’s eyes or support health decisions.", style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
    }
}

private object PhoneModelRegistry {
    private var loadedModelId: String? = null
    private var loadedSession: OrtSession? = null

    fun openSession(context: Context, id: String): OrtSession? {
        require(id.matches(Regex("[A-Za-z0-9_-]+"))) { "Invalid model identifier." }
        if (loadedModelId == id) return loadedSession
        loadedSession?.close()
        loadedSession = null
        loadedModelId = null
        val prefs = context.getSharedPreferences("model_registry", Context.MODE_PRIVATE)
        val version = prefs.getString("model-version:$id", null) ?: return null
        val modelFile = File(File(context.filesDir, "golden-retina-models"), "$id-$version.onnx")
        if (!modelFile.isFile) return null
        val options = OrtSession.SessionOptions()
        loadedSession = try { OrtEnvironment.getEnvironment().createSession(modelFile.absolutePath, options) } finally { options.close() }
        loadedModelId = id
        return loadedSession
    }

    fun closeSession() { loadedSession?.close(); loadedSession = null; loadedModelId = null }

    suspend fun refresh(context: Context, base: String): Map<String, String> = withContext(Dispatchers.IO) {
        val root = URL(base.trimEnd('/') + "/v1/manifest.json")
        require(root.protocol == "https") { "Use an HTTPS registry URL." }
        val manifestText = (root.openConnection() as HttpURLConnection).run {
            connectTimeout = 15_000
            readTimeout = 30_000
            instanceFollowRedirects = false
            require(responseCode == HttpURLConnection.HTTP_OK) { "Manifest host returned HTTP $responseCode." }
            getInputStream().bufferedReader().use { it.readText() }
        }
        val manifest = JSONObject(manifestText)
        require(manifest.optInt("schema_version", -1) == 1) { "Unsupported manifest schema." }
        val models = manifest.getJSONObject("models")
        val directory = File(context.filesDir, "golden-retina-models").apply { mkdirs() }
        val versions = mutableMapOf<String, String>()
        models.keys().asSequence().toList().forEach { id ->
            require(id.matches(Regex("[A-Za-z0-9_-]+"))) { "Invalid model identifier in manifest." }
            val item = models.optJSONObject(id) ?: return@forEach
            val version = item.optString("version").takeIf { it.isNotBlank() && it != "null" }
            val fileUrl = item.optString("artifact_url")
            val expected = item.optString("sha256").lowercase()
            if (item.optString("status") != "ready" || item.optString("format") != "onnx" || version == null) {
                versions[id] = "Awaiting mobile-ready ONNX release"
                return@forEach
            }
            require(version.matches(Regex("[A-Za-z0-9._-]{1,64}"))) { "$id has an invalid model version." }
            require(fileUrl.startsWith("https://")) { "$id artifact URL must use HTTPS." }
            require(expected.matches(Regex("[0-9a-f]{64}"))) { "$id has an invalid SHA-256." }
            val target = File(directory, "$id-$version.onnx")
            if (!target.isFile || sha256(target) != expected) {
                if (loadedModelId == id) closeSession()
                val staging = File(directory, ".$id-$version.part")
                try {
                    URL(fileUrl).openConnection().run {
                        connectTimeout = 15_000
                        readTimeout = 120_000
                        if (this is HttpURLConnection) {
                            instanceFollowRedirects = false
                            require(responseCode == HttpURLConnection.HTTP_OK) { "Model host returned HTTP $responseCode." }
                        }
                        getInputStream().use { input -> staging.outputStream().use { output -> input.copyTo(output) } }
                    }
                    require(sha256(staging) == expected) { "$id download failed checksum verification." }
                    if (target.exists()) target.delete()
                    check(staging.renameTo(target)) { "Could not install $id bundle." }
                } finally { staging.delete() }
            }
            validateOnnx(target)
            directory.listFiles()?.filter { it.name.startsWith("$id-") && it.name.endsWith(".onnx") && it != target }?.forEach { it.delete() }
            versions[id] = "Ready locally · $version · SHA-256 verified · ONNX session opened"
            context.getSharedPreferences("model_registry", Context.MODE_PRIVATE).edit().putString("model-version:$id", version).apply()
        }
        versions.forEach { (id, value) -> context.getSharedPreferences("model_registry", Context.MODE_PRIVATE).edit().putString("model:$id", value).apply() }
        versions
    }

    private fun validateOnnx(path: File) {
        val options = OrtSession.SessionOptions()
        try { OrtEnvironment.getEnvironment().createSession(path.absolutePath, options).close() } finally { options.close() }
    }

    private fun sha256(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        file.inputStream().buffered().use { input ->
            val buffer = ByteArray(1024 * 1024)
            while (true) { val count = input.read(buffer); if (count < 0) break; digest.update(buffer, 0, count) }
        }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }
}

@Composable
private fun AboutFact(text: String, icon: androidx.compose.ui.graphics.vector.ImageVector) {
    Row(horizontalArrangement = Arrangement.spacedBy(12.dp), verticalAlignment = Alignment.CenterVertically) {
        Icon(icon, contentDescription = null, tint = MaterialTheme.colorScheme.primary, modifier = Modifier.size(22.dp))
        Text(text, style = MaterialTheme.typography.bodyLarge)
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun DemoExperience(onExit: () -> Unit) {
    var phaseName by rememberSaveable { mutableStateOf(DemoPhase.WELCOME.name) }
    val phase = DemoPhase.valueOf(phaseName)
    val scanning = phase == DemoPhase.RIGHT_SCAN || phase == DemoPhase.LEFT_SCAN
    val progress by animateFloatAsState(if (scanning) 1f else 0f, animationSpec = tween(3500, easing = LinearEasing), label = "sample scan progress")

    LaunchedEffect(phase) {
        when (phase) {
            DemoPhase.SEARCHING -> { delay(2200); phaseName = DemoPhase.CONNECTED.name }
            DemoPhase.CONNECTED -> { delay(1400); phaseName = DemoPhase.RIGHT_SCAN.name }
            DemoPhase.RIGHT_SCAN -> { delay(3600); phaseName = DemoPhase.RIGHT_DONE.name }
            DemoPhase.LEFT_SCAN -> { delay(3600); phaseName = DemoPhase.LEFT_DONE.name }
            DemoPhase.ANALYZING -> { delay(3000); phaseName = DemoPhase.RESULTS.name }
            else -> Unit
        }
    }

    Scaffold(
        topBar = {
            CenterAlignedTopAppBar(
                navigationIcon = {
                    IconButton(onClick = onExit) { Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = "Back to Golden Retina home") }
                },
                title = { DemoPill() },
            )
        },
    ) { inner ->
        Column(Modifier.fillMaxSize().padding(inner).padding(horizontal = 22.dp), horizontalAlignment = Alignment.CenterHorizontally) {
            if (phase != DemoPhase.WELCOME && phase != DemoPhase.RESULTS) {
                LinearProgressIndicator(progress = { journeyProgress(phase, progress) }, modifier = Modifier.fillMaxWidth().height(4.dp).clip(CircleShape), color = MaterialTheme.colorScheme.primary, trackColor = MaterialTheme.colorScheme.surfaceContainerHighest)
            }
            AnimatedContent(
                targetState = phase,
                modifier = Modifier.fillMaxSize(),
                transitionSpec = { fadeIn(tween(180)) togetherWith fadeOut(tween(130)) },
                label = "demo phase",
            ) { target ->
                when (target) {
                    DemoPhase.WELCOME -> DemoWelcome { phaseName = DemoPhase.SEARCHING.name }
                    DemoPhase.SEARCHING -> DeviceSearch()
                    DemoPhase.CONNECTED -> DeviceConnected()
                    DemoPhase.RIGHT_SCAN -> EyeScan("Right", progress)
                    DemoPhase.RIGHT_DONE -> EyeCaptured("Right", "Continue to left eye") { phaseName = DemoPhase.LEFT_SCAN.name }
                    DemoPhase.LEFT_SCAN -> EyeScan("Left", progress)
                    DemoPhase.LEFT_DONE -> EyeCaptured("Left", "Show example summary") { phaseName = DemoPhase.ANALYZING.name }
                    DemoPhase.ANALYZING -> DemoAnalysis()
                    DemoPhase.RESULTS -> DemoResults(onReplay = { phaseName = DemoPhase.WELCOME.name }, onDone = onExit)
                }
            }
        }
    }
}

private fun journeyProgress(phase: DemoPhase, scanProgress: Float): Float = when (phase) {
    DemoPhase.WELCOME, DemoPhase.SEARCHING -> .12f
    DemoPhase.CONNECTED -> .22f
    DemoPhase.RIGHT_SCAN -> .40f + scanProgress * .20f
    DemoPhase.RIGHT_DONE -> .60f
    DemoPhase.LEFT_SCAN -> .62f + scanProgress * .20f
    DemoPhase.LEFT_DONE -> .82f
    DemoPhase.ANALYZING -> .91f
    DemoPhase.RESULTS -> 1f
}

@Composable
private fun DemoWelcome(onStart: () -> Unit) {
    Column(Modifier.fillMaxSize().padding(vertical = 18.dp), horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.SpaceBetween) {
        Spacer(Modifier.height(8.dp))
        Column(horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.spacedBy(18.dp)) {
            RetinaArtwork(Modifier.size(190.dp).clip(CircleShape))
            Text("See how a scan could work", style = MaterialTheme.typography.headlineMedium, textAlign = TextAlign.Center, fontWeight = FontWeight.Bold)
            Text("This guided simulation shows device setup, a scan of each eye, and an example summary.", style = MaterialTheme.typography.bodyLarge, textAlign = TextAlign.Center, color = MaterialTheme.colorScheme.onSurfaceVariant)
            AboutFact("No camera or goggles are used", Icons.Filled.Visibility)
        }
        Button(onClick = onStart, modifier = Modifier.fillMaxWidth().heightIn(min = 54.dp), shape = RoundedCornerShape(18.dp)) { Text("Start the demo", style = MaterialTheme.typography.titleMedium) }
    }
}

@Composable
private fun DeviceSearch() {
    val transition = rememberInfiniteTransition(label = "device radar")
    val rotation by transition.animateFloat(0f, 360f, infiniteRepeatable(tween(2600, easing = LinearEasing), RepeatMode.Restart), label = "radar sweep")
    Column(Modifier.fillMaxSize().padding(vertical = 24.dp), horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.Center) {
        Box(Modifier.size(250.dp), contentAlignment = Alignment.Center) {
            Canvas(Modifier.fillMaxSize()) {
                val center = Offset(size.width / 2, size.height / 2)
                listOf(.92f, .68f, .42f).forEach { factor -> drawCircle(GoldenGreen.copy(alpha = .14f), radius = size.minDimension * factor / 2, center = center, style = Stroke(width = 1.3.dp.toPx())) }
                rotate(rotation, center) {
                    drawArc(brush = Brush.sweepGradient(listOf(Color.Transparent, GoldenGreen.copy(alpha = .42f), Color.Transparent), center), startAngle = -42f, sweepAngle = 84f, useCenter = true, topLeft = Offset(size.width * .04f, size.height * .04f), size = Size(size.width * .92f, size.height * .92f))
                }
            }
            Surface(Modifier.size(92.dp), shape = CircleShape, color = MaterialTheme.colorScheme.primaryContainer) {
                Box(contentAlignment = Alignment.Center) { Icon(Icons.Filled.Sensors, contentDescription = null, tint = MaterialTheme.colorScheme.onPrimaryContainer, modifier = Modifier.size(42.dp)) }
            }
        }
        Spacer(Modifier.height(24.dp))
        Text("Looking for the demo device…", style = MaterialTheme.typography.headlineSmall, textAlign = TextAlign.Center, fontWeight = FontWeight.Bold)
        Spacer(Modifier.height(8.dp))
        Text("Simulated search · no Bluetooth scan is running", style = MaterialTheme.typography.bodyMedium, textAlign = TextAlign.Center, color = MaterialTheme.colorScheme.onSurfaceVariant)
    }
}

@Composable
private fun DeviceConnected() {
    Column(Modifier.fillMaxSize(), horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.Center) {
        Icon(Icons.Filled.CheckCircle, contentDescription = null, tint = MaterialTheme.colorScheme.primary, modifier = Modifier.size(72.dp))
        Spacer(Modifier.height(20.dp))
        Text("Demo goggles connected", style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.Bold, textAlign = TextAlign.Center)
        Spacer(Modifier.height(8.dp))
        Text("This connection is simulated. No real hardware was found or paired.", style = MaterialTheme.typography.bodyLarge, textAlign = TextAlign.Center, color = MaterialTheme.colorScheme.onSurfaceVariant)
    }
}

@Composable
private fun EyeScan(eye: String, progress: Float) {
    Column(Modifier.fillMaxSize().padding(vertical = 18.dp), horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.spacedBy(18.dp)) {
        Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
            Text("Scanning", style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.Bold)
            Spacer(Modifier.weight(1f))
            Surface(shape = CircleShape, color = MaterialTheme.colorScheme.secondaryContainer) {
                Row(Modifier.padding(horizontal = 12.dp, vertical = 8.dp), verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                    Icon(Icons.Filled.Visibility, contentDescription = null, modifier = Modifier.size(18.dp), tint = MaterialTheme.colorScheme.onSecondaryContainer)
                    Text("$eye eye", style = MaterialTheme.typography.labelLarge, color = MaterialTheme.colorScheme.onSecondaryContainer)
                }
            }
        }
        Box(Modifier.size(250.dp), contentAlignment = Alignment.Center) {
            RetinaArtwork(Modifier.size(236.dp).clip(CircleShape))
            Canvas(Modifier.size(236.dp)) {
                drawCircle(Color.White.copy(alpha = .55f), style = Stroke(1.dp.toPx()))
                val y = size.height * progress.coerceIn(0f, 1f)
                drawLine(Color.White, Offset(size.width * .13f, y), Offset(size.width * .87f, y), strokeWidth = 2.dp.toPx(), cap = StrokeCap.Round)
                drawLine(SoftMint.copy(alpha = .42f), Offset(size.width * .13f, y + 4.dp.toPx()), Offset(size.width * .87f, y + 4.dp.toPx()), strokeWidth = 5.dp.toPx(), cap = StrokeCap.Round)
            }
        }
        Text(if (progress < .98f) "Capturing the ${eye.lowercase()} eye…" else "$eye eye captured", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
        LinearProgressIndicator(progress = { progress }, modifier = Modifier.fillMaxWidth().height(5.dp).clip(CircleShape))
        Text("DEMO IMAGE · NOT A REAL EYE", style = MaterialTheme.typography.labelSmall, letterSpacing = 1.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
        Spacer(Modifier.weight(1f))
        Text("Keep still and look toward the center", style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
    }
}

@Composable
private fun EyeCaptured(eye: String, action: String, onContinue: () -> Unit) {
    Column(Modifier.fillMaxSize().padding(vertical = 22.dp), horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.SpaceBetween) {
        Spacer(Modifier.height(12.dp))
        Column(horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.spacedBy(14.dp)) {
            Icon(Icons.Filled.CheckCircle, contentDescription = null, tint = MaterialTheme.colorScheme.primary, modifier = Modifier.size(68.dp))
            Text("$eye eye scan complete", style = MaterialTheme.typography.headlineSmall, textAlign = TextAlign.Center, fontWeight = FontWeight.Bold)
            Text("Sample image captured in the demo", style = MaterialTheme.typography.bodyLarge, color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
        Button(onClick = onContinue, modifier = Modifier.fillMaxWidth().heightIn(min = 54.dp), shape = RoundedCornerShape(18.dp)) { Text(action, style = MaterialTheme.typography.titleMedium) }
    }
}

@Composable
private fun DemoAnalysis() {
    val transition = rememberInfiniteTransition(label = "demo analysis")
    val rotation by transition.animateFloat(0f, 360f, infiniteRepeatable(tween(2200, easing = LinearEasing)), label = "analysis rotation")
    Column(Modifier.fillMaxSize(), horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.Center) {
        Box(Modifier.size(160.dp), contentAlignment = Alignment.Center) {
            Canvas(Modifier.fillMaxSize().rotate(rotation)) {
                drawArc(GoldenGreen, -40f, 265f, false, style = Stroke(5.dp.toPx(), cap = StrokeCap.Round))
            }
            Icon(Icons.Filled.Search, contentDescription = null, tint = MaterialTheme.colorScheme.primary, modifier = Modifier.size(46.dp))
        }
        Spacer(Modifier.height(22.dp))
        Text("Reviewing both sample images", style = MaterialTheme.typography.headlineSmall, textAlign = TextAlign.Center, fontWeight = FontWeight.Bold)
        Spacer(Modifier.height(8.dp))
        Text("This is a timed animation. No AI model is running.", style = MaterialTheme.typography.bodyLarge, textAlign = TextAlign.Center, color = MaterialTheme.colorScheme.onSurfaceVariant)
    }
}

@Composable
private fun DemoResults(onReplay: () -> Unit, onDone: () -> Unit) {
    BoxWithConstraints(Modifier.fillMaxSize()) {
        Column(Modifier.align(Alignment.TopCenter).widthIn(max = 680.dp).fillMaxWidth().fillMaxHeight().verticalScroll(rememberScrollState()).padding(vertical = 16.dp), verticalArrangement = Arrangement.spacedBy(16.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                DemoPill()
                Spacer(Modifier.weight(1f))
                Text("EXAMPLE ONLY", style = MaterialTheme.typography.labelSmall, letterSpacing = .8.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
            Text("Your demo is complete", style = MaterialTheme.typography.headlineMedium, fontWeight = FontWeight.Bold, modifier = Modifier.semantics { heading() })
            Text("Here is how a future screening summary could be organized.", style = MaterialTheme.typography.bodyLarge, color = MaterialTheme.colorScheme.onSurfaceVariant)
            BoxWithConstraints(Modifier.fillMaxWidth()) {
                val horizontal = maxWidth >= 460.dp
                if (horizontal) {
                    Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                        SampleEyeCard("Right eye", "Example summary A", "0.18", Modifier.weight(1f))
                        SampleEyeCard("Left eye", "Example summary B", "0.64", Modifier.weight(1f))
                    }
                } else {
                    Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                        SampleEyeCard("Right eye", "Example summary A", "0.18", Modifier.fillMaxWidth())
                        SampleEyeCard("Left eye", "Example summary B", "0.64", Modifier.fillMaxWidth())
                    }
                }
            }
            Card(shape = RoundedCornerShape(22.dp), colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceContainerLow)) {
                Column(Modifier.padding(18.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                    Text("Illustrative sample output", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
                    Text("A future report could summarize each eye and explain whether a clinician should review the images.", style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
                    HorizontalDivider()
                    Text("The numbers are invented demo values. They are not predictions, probabilities for you, or medical advice.", style = MaterialTheme.typography.bodyMedium, fontWeight = FontWeight.Medium)
                }
            }
            AboutFact("No image from you was captured or analyzed", Icons.Filled.Security)
            Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                OutlinedAction("Replay demo", Icons.Filled.Refresh, onReplay, Modifier.weight(1f))
                Button(onClick = onDone, modifier = Modifier.weight(1f).heightIn(min = 52.dp), shape = RoundedCornerShape(16.dp)) { Text("Done") }
            }
        }
    }
}

@Composable
private fun OutlinedAction(label: String, icon: androidx.compose.ui.graphics.vector.ImageVector, onClick: () -> Unit, modifier: Modifier = Modifier) {
    androidx.compose.material3.OutlinedButton(onClick = onClick, modifier = modifier.heightIn(min = 52.dp), shape = RoundedCornerShape(16.dp)) {
        Icon(icon, contentDescription = null)
        Spacer(Modifier.width(7.dp))
        Text(label)
    }
}

@Composable
private fun SampleEyeCard(eye: String, summary: String, value: String, modifier: Modifier = Modifier) {
    Card(modifier, shape = RoundedCornerShape(22.dp), colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceContainerLow)) {
        Column(Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            RetinaArtwork(Modifier.fillMaxWidth().height(108.dp).clip(RoundedCornerShape(15.dp)))
            Text(eye, style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.Bold)
            Text(summary, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
            Row(verticalAlignment = Alignment.Bottom, horizontalArrangement = Arrangement.spacedBy(5.dp)) {
                Text(value, style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.Bold, color = MaterialTheme.colorScheme.primary)
                Text("sample", style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant, modifier = Modifier.padding(bottom = 3.dp))
            }
        }
    }
}

@Composable
private fun RetinaArtwork(modifier: Modifier = Modifier) {
    Canvas(modifier.background(Pine)) {
        val center = Offset(size.width * .52f, size.height * .5f)
        val radius = size.minDimension * .46f
        drawCircle(brush = Brush.radialGradient(listOf(Color(0xFF4FC494), Color(0xFF168269), Color(0xFF0C4A43)), center, radius), radius = radius, center = center)
        drawCircle(Color.White.copy(alpha = .24f), radius = radius * .91f, center = center, style = Stroke(width = size.minDimension * .008f))
        val optic = Offset(center.x - radius * .39f, center.y + radius * .12f)
        val angles = listOf(-2.85f, -2.45f, -2.1f, -1.75f, -1.4f, -1.04f, -.72f, -.36f, 0f, .38f, .76f, 1.1f, 1.45f, 1.83f, 2.2f, 2.6f)
        angles.forEachIndexed { index, angle ->
            val length = radius * if (index % 3 == 0) 1.62f else 1.3f
            val end = Offset(optic.x + kotlin.math.cos(angle) * length, optic.y + kotlin.math.sin(angle) * length)
            val p1 = Offset(optic.x + kotlin.math.cos(angle + .14f) * length * .42f, optic.y + kotlin.math.sin(angle + .14f) * length * .42f)
            val p2 = Offset(optic.x + kotlin.math.cos(angle - .1f) * length * .72f, optic.y + kotlin.math.sin(angle - .1f) * length * .72f)
            val path = Path().apply { moveTo(optic.x, optic.y); cubicTo(p1.x, p1.y, p2.x, p2.y, end.x, end.y) }
            drawPath(path, Color.White.copy(alpha = .62f), style = Stroke(width = size.minDimension * if (index % 3 == 0) .012f else .007f, cap = StrokeCap.Round))
        }
        drawCircle(Color(0xFFFFD68A), radius * .095f, Offset(center.x + radius * .25f, center.y - radius * .13f))
        drawCircle(Color(0xFFFFE6B5).copy(alpha = .45f), radius * .15f, optic)
    }
}
