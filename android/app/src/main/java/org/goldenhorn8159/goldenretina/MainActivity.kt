package org.goldenhorn8159.goldenretina

import android.os.Bundle
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.media.ExifInterface
import android.graphics.Color as AndroidColor
import android.net.Uri
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.OnnxTensor
import androidx.compose.foundation.text.KeyboardOptions
import androidx.activity.ComponentActivity
import androidx.activity.compose.BackHandler
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
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
import androidx.compose.foundation.Image
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
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CenterAlignedTopAppBar
import androidx.compose.ui.window.Dialog
import androidx.compose.ui.window.DialogProperties
import androidx.compose.material3.CircularProgressIndicator
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
import androidx.compose.material3.Tab
import androidx.compose.material3.PrimaryTabRow
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
import androidx.compose.ui.graphics.asImageBitmap
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
        setContent {
            val prefs = getSharedPreferences("golden_retina_settings", Context.MODE_PRIVATE)
            var colorChoice by rememberSaveable { mutableStateOf(prefs.getString("color", "system") ?: "system") }
            GoldenRetinaTheme(colorChoice) {
                GoldenRetinaApp(colorChoice, onColorChoice = {
                    colorChoice = it
                    prefs.edit().putString("color", it).apply()
                })
            }
        }
    }
}

@Composable
private fun GoldenRetinaTheme(colorChoice: String, content: @Composable () -> Unit) {
    val dark = when (colorChoice) { "dark" -> true; "light" -> false; else -> isSystemInDarkTheme() }
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

private enum class Destination { HOME, SCREENING, DEMO, ANALYZE, HISTORY, DEVICE, GUIDE, SETTINGS }
private enum class DemoPhase { WELCOME, SEARCHING, CONNECTED, RIGHT_SCAN, RIGHT_DONE, LEFT_SCAN, LEFT_DONE, ANALYZING, RESULTS }
private data class ConditionItem(val name: String, val rightScore: Int, val leftScore: Int)
private data class ConditionGroup(val name: String, val conditions: List<ConditionItem>) {
    val rightMax: Int get() = conditions.maxOfOrNull { it.rightScore } ?: 0
    val leftMax: Int get() = conditions.maxOfOrNull { it.leftScore } ?: 0
}
private const val MODEL_MANIFEST_URL = "https://raw.githubusercontent.com/Burak8159/golden-retina/main/mobile_models/onnx/manifest.json"
private const val BUNDLED_MODEL_VERSION = "20261114-r3"

@Composable
private fun GoldenRetinaApp(colorChoice: String, onColorChoice: (String) -> Unit) {
    var destination by rememberSaveable { mutableStateOf(Destination.HOME.name) }
    BackHandler(enabled = destination == Destination.DEMO.name || destination == Destination.GUIDE.name || destination == Destination.ANALYZE.name) {
        destination = when (destination) {
            Destination.GUIDE.name -> Destination.SETTINGS.name
            Destination.ANALYZE.name, Destination.DEMO.name -> Destination.SCREENING.name
            else -> Destination.HOME.name
        }
    }

    when (Destination.valueOf(destination)) {
        Destination.HOME, Destination.SCREENING, Destination.DEMO, Destination.ANALYZE, Destination.HISTORY, Destination.DEVICE, Destination.GUIDE, Destination.SETTINGS -> MainTabs(
            selected = Destination.valueOf(destination),
            onSelect = { destination = it.name },
            onStartDemo = { destination = Destination.DEMO.name },
            onOpenResearch = { destination = Destination.ANALYZE.name },
            onOpenGuide = { destination = Destination.GUIDE.name },
            colorChoice = colorChoice,
            onColorChoice = onColorChoice,
        )
    }
}

@Composable
private fun MainTabs(selected: Destination, onSelect: (Destination) -> Unit, onStartDemo: () -> Unit, onOpenResearch: () -> Unit, onOpenGuide: () -> Unit, colorChoice: String, onColorChoice: (String) -> Unit) {
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
                    selected = selected in setOf(Destination.SCREENING, Destination.DEMO, Destination.ANALYZE),
                    onClick = { onSelect(Destination.SCREENING) },
                    icon = { Icon(Icons.Filled.Search, contentDescription = null) },
                    label = { Text("Screening") },
                )
                NavigationBarItem(
                    selected = selected == Destination.HISTORY,
                    onClick = { onSelect(Destination.HISTORY) },
                    icon = { Icon(Icons.Filled.Refresh, contentDescription = null) },
                    label = { Text("History") },
                )
                NavigationBarItem(
                    selected = selected == Destination.DEVICE,
                    onClick = { onSelect(Destination.DEVICE) },
                    icon = { Icon(Icons.Filled.Lock, contentDescription = null) },
                    label = { Text("Device") },
                )
                NavigationBarItem(
                    selected = selected == Destination.SETTINGS || selected == Destination.GUIDE,
                    onClick = { onSelect(Destination.SETTINGS) },
                    icon = { Icon(Icons.Filled.Settings, contentDescription = null) },
                    label = { Text("Settings") },
                )
            }
        },
    ) { insets ->
        when (selected) {
            Destination.HOME -> HomeScreen(Modifier.padding(insets), onStartDemo)
            Destination.SCREENING -> ScreeningHubScreen(Modifier.padding(insets), onStartDemo, onOpenResearch)
            Destination.DEMO -> DemoExperience(modifier = Modifier.padding(insets), onExit = { onSelect(Destination.SCREENING) }, onOpenResearch = onOpenResearch)
            Destination.ANALYZE -> OnDeviceInferenceScreen(onBack = { onSelect(Destination.SCREENING) })
            Destination.GUIDE -> HowToUseScreen(Modifier.padding(insets))
            Destination.HISTORY -> ScreeningHistoryScreen(Modifier.padding(insets), onStartDemo)
            Destination.DEVICE -> DeviceStatusScreen(Modifier.padding(insets), onOpenGuide)
            Destination.SETTINGS -> SettingsScreen(Modifier.padding(insets), colorChoice, onColorChoice, onOpenGuide)
        }
    }
}

@Composable
private fun ScreeningHubScreen(modifier: Modifier, onStartDemo: () -> Unit, onOpenResearch: () -> Unit) {
    Column(modifier.fillMaxSize().verticalScroll(rememberScrollState()).widthIn(max = 680.dp).padding(horizontal = 20.dp, vertical = 16.dp), verticalArrangement = Arrangement.spacedBy(16.dp)) {
        Text("Screening", style = MaterialTheme.typography.headlineMedium, fontWeight = FontWeight.Bold, modifier = Modifier.semantics { heading() })
        Text("Choose a clearly labeled software demonstration or analyze a selected image with an installed on-device research model.", style = MaterialTheme.typography.bodyLarge, color = MaterialTheme.colorScheme.onSurfaceVariant)
        Card(shape = RoundedCornerShape(22.dp), colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.secondaryContainer)) {
            Column(Modifier.padding(18.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Text("Research / demonstration", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
                Text("The headset is unavailable. Demo sample values are illustrative; research model scores are not calibrated disease risks or diagnoses.", style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSecondaryContainer)
            }
        }
        Button(onClick = onStartDemo, modifier = Modifier.fillMaxWidth().heightIn(min = 52.dp)) {
            Icon(Icons.Filled.PlayArrow, contentDescription = null)
            Spacer(Modifier.width(8.dp))
            Text("Start guided demo")
        }
        androidx.compose.material3.OutlinedButton(onClick = onOpenResearch, modifier = Modifier.fillMaxWidth().heightIn(min = 52.dp)) {
            Icon(Icons.Filled.Search, contentDescription = null)
            Spacer(Modifier.width(8.dp))
            Text("Analyze a fundus image")
        }
        Text("Future headset screening requires compatible hardware, its documented interface, and validated task-specific capabilities.", style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
        Text("Unavailable future capabilities", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold, modifier = Modifier.semantics { heading() })
        FutureCapabilityCard("Functional vision tests", "Requires calibrated headset displays, a documented response method, and validation for each test.")
        FutureCapabilityCard("Retinal image review", "This build has no headset-captured retinal images or image metadata to review.")
        FutureCapabilityCard("Reports and sharing", "There are no real examination records or report provenance to export from this build.")
    }
}

@Composable
private fun FutureCapabilityCard(title: String, requirement: String) {
    Card(shape = RoundedCornerShape(20.dp), colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceContainerLow)) {
        Column(Modifier.fillMaxWidth().padding(16.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalAlignment = Alignment.CenterVertically) {
                Icon(Icons.Filled.Lock, contentDescription = null, tint = MaterialTheme.colorScheme.onSurfaceVariant)
                Text(title, style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
            }
            Text("Unavailable · $requirement", style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
    }
}

@Composable
private fun ScreeningHistoryScreen(modifier: Modifier, onStartDemo: () -> Unit) {
    Column(modifier.fillMaxSize().verticalScroll(rememberScrollState()).widthIn(max = 680.dp).padding(horizontal = 24.dp, vertical = 28.dp), verticalArrangement = Arrangement.spacedBy(14.dp), horizontalAlignment = Alignment.Start) {
        Icon(Icons.Filled.Refresh, contentDescription = null, tint = MaterialTheme.colorScheme.primary, modifier = Modifier.size(40.dp))
        Text("No examination history", style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.Bold, modifier = Modifier.semantics { heading() })
        Text("Real headset examinations are not available yet. Demo sessions are not saved as health records.", style = MaterialTheme.typography.bodyLarge, color = MaterialTheme.colorScheme.onSurfaceVariant)
        TextButton(onClick = onStartDemo) { Text("Explore a demo") }
    }
}

@Composable
private fun DeviceStatusScreen(modifier: Modifier, onOpenGuide: () -> Unit) {
    Column(modifier.fillMaxSize().verticalScroll(rememberScrollState()).widthIn(max = 680.dp).padding(horizontal = 20.dp, vertical = 16.dp), verticalArrangement = Arrangement.spacedBy(16.dp)) {
        Text("Device", style = MaterialTheme.typography.headlineMedium, fontWeight = FontWeight.Bold, modifier = Modifier.semantics { heading() })
        Card(shape = RoundedCornerShape(22.dp), colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceContainerLow)) {
            Column(Modifier.padding(18.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                Text("Headset unavailable", style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.Bold)
                Text("This build cannot discover or connect to a headset. No discovery scan or permission request is started.", style = MaterialTheme.typography.bodyLarge, color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
        }
        Text("Unavailable in this build", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
        Text("Battery and signal telemetry, firmware, calibration, imaging, and device diagnostics need a versioned hardware interface before they can be shown.", style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
        Button(onClick = onOpenGuide, modifier = Modifier.fillMaxWidth().heightIn(min = 52.dp)) {
            Icon(Icons.Filled.Info, contentDescription = null)
            Spacer(Modifier.width(8.dp))
            Text("Open setup guide")
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
                        Icon(Icons.Filled.Search, contentDescription = null, tint = MaterialTheme.colorScheme.onSurfaceVariant, modifier = Modifier.size(30.dp))
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
                    Icon(Icons.Filled.Lock, contentDescription = null, tint = MaterialTheme.colorScheme.primary)
                    Text("Demo only · Example values are illustrative; no real image or medical result is produced.", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                }
            }
        }
    }
    if (showHardwareInfo) {
        androidx.compose.material3.AlertDialog(
            onDismissRequest = { showHardwareInfo = false },
            icon = { Icon(Icons.Filled.Lock, contentDescription = null, tint = MaterialTheme.colorScheme.primary) },
            title = { Text("Goggles are still in development") },
            text = { Text("There is no finished headset hardware or documented connection method yet. Device setup will be added after the hardware interface is defined.") },
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
private fun HowToUseScreen(modifier: Modifier = Modifier) {
    BoxWithConstraints(modifier.fillMaxSize()) {
        Column(Modifier.align(Alignment.TopCenter).widthIn(max = 680.dp).fillMaxWidth().verticalScroll(rememberScrollState()).padding(horizontal = 20.dp, vertical = 18.dp), verticalArrangement = Arrangement.spacedBy(16.dp)) {
            Text("How to use", style = MaterialTheme.typography.headlineMedium, fontWeight = FontWeight.Bold, modifier = Modifier.semantics { heading() })
            Text("A quick guide to the intended Golden Retina flow. This app currently provides a simulation only.", style = MaterialTheme.typography.bodyLarge, color = MaterialTheme.colorScheme.onSurfaceVariant)
            Card(shape = RoundedCornerShape(24.dp), colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceContainerLow)) {
                Column(Modifier.padding(18.dp), verticalArrangement = Arrangement.spacedBy(15.dp)) {
                    GuideStep("1", "Prepare goggles", "Connect them from Device after compatible hardware and its connection method are documented.")
                    GuideStep("2", "Press the goggles button", "In the intended product, the physical button starts acquisition. In the demo, tap the on-screen control to simulate it.")
                    GuideStep("3", "Position and hold still", "The planned flow guides positioning. The animated eye is an illustration; it does not track your gaze.")
                    GuideStep("4", "Scan both eyes", "The demo proceeds from the right eye to the left eye automatically.")
                    GuideStep("5", "Review the sample summary", "Expand condition groups to see illustrative right- and left-eye percentages. These fixed examples are not produced by an image or disease model.")
                }
            }
            Card(shape = RoundedCornerShape(22.dp), colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceContainerHighest)) {
                Column(Modifier.fillMaxWidth().heightIn(min = 170.dp).padding(22.dp), horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.Center) {
                    Icon(Icons.Filled.PlayArrow, contentDescription = null, tint = MaterialTheme.colorScheme.primary, modifier = Modifier.size(40.dp))
                    Spacer(Modifier.height(8.dp))
                    Text("Instructional video coming soon", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
                    Text("Video placeholder · no playback available", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                }
            }
            Text("No device discovery, camera capture, eye tracking, or disease inference is active in this simulated flow.", style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
    }
}

@Composable
private fun GuideStep(number: String, title: String, description: String) {
    Row(horizontalArrangement = Arrangement.spacedBy(12.dp), verticalAlignment = Alignment.Top) {
        Surface(shape = CircleShape, color = MaterialTheme.colorScheme.primaryContainer, modifier = Modifier.size(30.dp)) {
            Box(contentAlignment = Alignment.Center) { Text(number, color = MaterialTheme.colorScheme.onPrimaryContainer, fontWeight = FontWeight.Bold) }
        }
        Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
            Text(title, style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.SemiBold)
            Text(description, style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
    }
}

@Composable
private fun SettingsScreen(modifier: Modifier = Modifier, colorChoice: String, onColorChoice: (String) -> Unit, onOpenGuide: () -> Unit = {}) {
    val context = LocalContext.current
    val prefs = remember { context.getSharedPreferences("model_registry", Context.MODE_PRIVATE) }
    var registryUrl by rememberSaveable { mutableStateOf(prefs.getString("url", MODEL_MANIFEST_URL) ?: MODEL_MANIFEST_URL) }
    var status by rememberSaveable { mutableStateOf("Eye localization is bundled. Disease models install individually from the verified registry.") }
    var busy by remember { mutableStateOf(false) }
    val initialModelStates = remember {
        (listOf("ir_eye_tracker") + MobileResearchInference.diseaseIds).associateWith { id ->
            val version = prefs.getString("model-version:$id", null)
            val installedFile = version?.let { File(File(context.filesDir, "golden-retina-models"), "$id-$it.onnx") }
            when {
                installedFile?.isFile == true -> "Current update installed · $version · SHA-256 verified"
                id == "ir_eye_tracker" -> "Bundled on-device synthetic research model · v$BUNDLED_MODEL_VERSION"
                else -> "Not installed · check for model updates"
            }
        }
    }
    var modelStates by remember { mutableStateOf(initialModelStates) }
    var showDeviceInfo by rememberSaveable { mutableStateOf(false) }
    var showModelLab by rememberSaveable { mutableStateOf(false) }
    val scope = rememberCoroutineScope()
    if (showModelLab) {
        Dialog(onDismissRequest = { showModelLab = false }, properties = DialogProperties(usePlatformDefaultWidth = false)) {
            Surface(Modifier.fillMaxSize(), color = MaterialTheme.colorScheme.background) {
                OnDeviceInferenceScreen(onBack = { showModelLab = false })
            }
        }
    }
    BoxWithConstraints(modifier.fillMaxSize()) {
        Column(Modifier.align(Alignment.TopCenter).widthIn(max = 680.dp).fillMaxWidth().verticalScroll(rememberScrollState()).padding(horizontal = 20.dp, vertical = 16.dp).padding(bottom = 20.dp), verticalArrangement = Arrangement.spacedBy(16.dp)) {
            Text("Settings", style = MaterialTheme.typography.headlineMedium, fontWeight = FontWeight.Bold, modifier = Modifier.semantics { heading() })
            Card(shape = RoundedCornerShape(24.dp), colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceContainerLow)) {
                Column(Modifier.padding(18.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
                    Text("Color appearance", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.fillMaxWidth()) {
                        listOf("system" to "System", "light" to "Light", "dark" to "Dark").forEach { (value, label) ->
                            if (colorChoice == value) Button(onClick = { onColorChoice(value) }, modifier = Modifier.weight(1f).heightIn(min = 48.dp)) { Text(label) }
                            else androidx.compose.material3.OutlinedButton(onClick = { onColorChoice(value) }, modifier = Modifier.weight(1f).heightIn(min = 48.dp)) { Text(label) }
                        }
                    }
                }
            }
            Card(shape = RoundedCornerShape(24.dp), colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceContainerLow)) {
                Column(Modifier.padding(18.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                    Text("Privacy and accessibility", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
                    Text("Research image analysis runs on this phone. This build has no account, cloud health-record service, report-sharing service, or real examination history.", style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
                    Text("Use Android font scaling and TalkBack. English interface; Turkish medical terminology requires human review before release.", style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
                }
            }
            Card(shape = RoundedCornerShape(24.dp), colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceContainerLow)) {
                Column(Modifier.padding(18.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                    Text("Goggles", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
                    Button(onClick = { showDeviceInfo = true }, modifier = Modifier.fillMaxWidth().heightIn(min = 52.dp)) {
                        Icon(Icons.Filled.Lock, contentDescription = null)
                        Spacer(Modifier.width(8.dp))
                    Text("Connect goggles")
                    }
                    Text("Unavailable · hardware and connection method are not ready", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                }
            }
            if (showDeviceInfo) {
                Card(shape = RoundedCornerShape(18.dp), colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.errorContainer)) {
                    Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                        Text("Device connection unavailable", fontWeight = FontWeight.Bold, color = MaterialTheme.colorScheme.onErrorContainer)
                        Text("No device discovery or pairing is implemented because compatible goggles and their connection method are not documented.", color = MaterialTheme.colorScheme.onErrorContainer)
                        TextButton(onClick = { showDeviceInfo = false }) { Text("Got it") }
                    }
                }
            }
            Text("About Golden Retina", style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.Bold, modifier = Modifier.semantics { heading() })
            Text("An early software concept from Golden Horn 8159.", style = MaterialTheme.typography.bodyLarge, color = MaterialTheme.colorScheme.onSurfaceVariant)
            androidx.compose.material3.OutlinedButton(onClick = onOpenGuide, modifier = Modifier.fillMaxWidth().heightIn(min = 50.dp)) {
                Icon(Icons.Filled.Info, contentDescription = null)
                Spacer(Modifier.width(8.dp))
                Text("How to use and safety guide")
            }
            Card(shape = RoundedCornerShape(24.dp), colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceContainerLow)) {
                Column(Modifier.padding(18.dp), verticalArrangement = Arrangement.spacedBy(14.dp)) {
                    Text("About this demo", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
                    AboutFact("Every scan and result is simulated", Icons.Filled.PlayArrow)
                    AboutFact("No camera images are collected", Icons.Filled.Lock)
                    AboutFact("No headset connection is active", Icons.Filled.Lock)
                    AboutFact("Not a medical device or diagnosis", Icons.Filled.Info)
                }
            }
            Card(shape = RoundedCornerShape(24.dp), colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceContainerLow)) {
                Column(Modifier.padding(18.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
                    Text("On-device models", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
                    Text("Models run locally on this phone. Disease models download one at a time from the verified HTTPS registry; selected images are not uploaded. The guided demo remains simulated.", style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
                    Button(onClick = { showModelLab = true }, modifier = Modifier.fillMaxWidth().heightIn(min = 50.dp)) {
                        Icon(Icons.Filled.Search, contentDescription = null)
                        Spacer(Modifier.width(8.dp))
                        Text("Run on-device research models")
                    }
                    OutlinedTextField(
                        value = registryUrl,
                        onValueChange = { registryUrl = it },
                        label = { Text("HTTPS model manifest URL") },
                        keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Uri),
                        singleLine = true,
                        modifier = Modifier.fillMaxWidth(),
                    )
                    Button(enabled = !busy, onClick = {
                        prefs.edit().putString("url", registryUrl.trim()).apply()
                        busy = true
                        status = "Checking model registry…"
                        // Only the small manifest is fetched here. Model files download individually on request.
                        scope.launch {
                            val result = runCatching { PhoneModelRegistry.refresh(context, registryUrl.trim()) }
                            status = result.fold({ reply ->
                                modelStates = modelStates + reply
                                "Registry checked. Model files download only when you choose an update; inference stays on-device."
                            }, { "Registry check failed: ${it.message ?: "network or manifest error"}" })
                            busy = false
                        }
                    }) {
                        Icon(Icons.Filled.Refresh, contentDescription = null)
                        Text("  Check for model updates")
                    }
                    Text("For safety, updates are accepted only from the Golden Retina GitHub model registry. New model releases can be published there without updating this app.", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                    if (busy) LinearProgressIndicator(Modifier.fillMaxWidth())
                    Text(status, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                    modelStates.toSortedMap().forEach { (id, state) ->
                        HorizontalDivider()
                        Column(verticalArrangement = Arrangement.spacedBy(3.dp)) {
                            Text(modelDisplayName(id), style = MaterialTheme.typography.labelLarge, fontWeight = FontWeight.SemiBold)
                            Text(state, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                            if (state.startsWith("Update available ·")) {
                                val bytes = runCatching {
                                    JSONObject(prefs.getString("manifest-json", "{}") ?: "{}").getJSONObject("models").getJSONObject(id).optLong("bytes")
                                }.getOrDefault(0L)
                                TextButton(enabled = !busy, onClick = {
                                    busy = true
                                    status = "Downloading and checking ${modelDisplayName(id)}…"
                                    scope.launch {
                                        val result = runCatching { PhoneModelRegistry.install(context, id) }
                                        status = result.fold({ installed ->
                                            modelStates = modelStates + (id to installed)
                                            "Model downloaded, checksum-verified and opened by the phone runtime."
                                        }, { "Model update failed: ${it.message ?: "download or model check error"}" })
                                        busy = false
                                    }
                                }) { Text("Download model update · ${formatModelSize(bytes)}") }
                            }
                        }
                    }
                }
            }
            Text("Disease heads are retrospective research models with patient independence unverified. The eye-localization model was trained on synthetic renders only. Neither is a clinical diagnosis or validated goggle tracker.", style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
    }
}

private fun modelDisplayName(id: String): String = when (id) {
    "ir_eye_tracker" -> "Eye localization · synthetic research model"
    "rfmid_dr" -> "Diabetic retinopathy · RFMiD"
    "rfmid_armd" -> "Age-related macular degeneration · RFMiD"
    "rfmid_dn" -> "Drusen · RFMiD"
    "rfmid_mya" -> "Myopia · RFMiD"
    "rfmid_brvo" -> "Branch retinal vein occlusion · RFMiD"
    "rfmid_odc" -> "Optic-disc cupping finding · RFMiD"
    "rfmid_odp" -> "Optic-disc pallor finding · RFMiD"
    "rfmid_ode" -> "Optic-disc edema finding · RFMiD"
    "rfmid_tsln" -> "Tessellated fundus finding · RFMiD"
    "rfmid_ls" -> "Laser-scar finding · RFMiD"
    "rfmid_csr" -> "Central serous retinopathy · RFMiD"
    "rfmid_rs" -> "Retinal scar finding · RFMiD"
    "rfmid_crs" -> "Chorioretinitis · RFMiD"
        "rfmid_crvo" -> "Central retinal vein occlusion · RFMiD label"
        "rfmid_erm" -> "Epiretinal membrane finding · RFMiD"
        "rfmid_ms" -> "Macular scar finding · RFMiD"
        "rfmid_rt" -> "Retinal traction finding · RFMiD"
    else -> id.replace('_', ' ')
}

private fun formatModelSize(bytes: Long): String = when {
    bytes <= 0L -> "size unknown"
    bytes >= 1_048_576L -> "${"%.1f".format(bytes / 1_048_576.0)} MB"
    else -> "${"%.0f".format(bytes / 1_024.0)} KB"
}

private data class MobileDiseaseScore(val id: String, val value: Float, val version: String)
private data class MobileEyeResult(val pupilPixels: Int, val centerX: Double?, val centerY: Double?, val score: Float?)

private data class MobileDiseaseGroup(val name: String, val ids: List<String>, val isFinding: Boolean = false)

private object MobileDiseaseCatalog {
    val coreIds = listOf("rfmid_dr", "rfmid_armd", "rfmid_mya", "rfmid_brvo")
    val diseaseIds = listOf("rfmid_dr", "rfmid_armd", "rfmid_mya", "rfmid_brvo", "rfmid_csr", "rfmid_crs", "rfmid_crvo")
    val findingIds = listOf("rfmid_dn", "rfmid_odc", "rfmid_odp", "rfmid_ode", "rfmid_tsln", "rfmid_ls", "rfmid_rs", "rfmid_ms", "rfmid_rt", "rfmid_erm")
    val allIds = diseaseIds + findingIds
    val groups = listOf(
        MobileDiseaseGroup("Diabetic retinopathy", listOf("rfmid_dr")),
        MobileDiseaseGroup("Age-related macular degeneration", listOf("rfmid_armd")),
        MobileDiseaseGroup("Myopia", listOf("rfmid_mya")),
        MobileDiseaseGroup("Branch retinal vein occlusion", listOf("rfmid_brvo")),
        MobileDiseaseGroup("Central retinal vein occlusion", listOf("rfmid_crvo")),
        MobileDiseaseGroup("Central serous retinopathy", listOf("rfmid_csr")),
        MobileDiseaseGroup("Chorioretinitis", listOf("rfmid_crs")),
        MobileDiseaseGroup("Drusen", listOf("rfmid_dn"), isFinding = true),
        MobileDiseaseGroup("Optic-disc findings", listOf("rfmid_odc", "rfmid_odp", "rfmid_ode"), isFinding = true),
        MobileDiseaseGroup("Tessellation", listOf("rfmid_tsln"), isFinding = true),
        MobileDiseaseGroup("Treatment and scar findings", listOf("rfmid_ls", "rfmid_rs", "rfmid_ms"), isFinding = true),
        MobileDiseaseGroup("Epiretinal membrane finding", listOf("rfmid_erm"), isFinding = true),
        MobileDiseaseGroup("Retinal traction", listOf("rfmid_rt"), isFinding = true),
    )
}

@Composable
private fun OnDeviceInferenceScreen(onBack: () -> Unit) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    var selectedEye by remember { mutableStateOf(0) }
    var rightImage by remember { mutableStateOf<Bitmap?>(null) }
    var leftImage by remember { mutableStateOf<Bitmap?>(null) }
    val selectedImage = if (selectedEye == 0) rightImage else leftImage
    var rightDiseaseResults by remember { mutableStateOf<List<MobileDiseaseScore>>(emptyList()) }
    var leftDiseaseResults by remember { mutableStateOf<List<MobileDiseaseScore>>(emptyList()) }
    var imageTargetEye by remember { mutableStateOf(0) }
    val diseaseResults = if (selectedEye == 0) rightDiseaseResults else leftDiseaseResults
    var expandedScoreGroups by remember { mutableStateOf(setOf("Diabetic retinopathy")) }
    var eyeResult by remember { mutableStateOf<MobileEyeResult?>(null) }
    var running by remember { mutableStateOf(false) }
    var runningDiseaseModels by remember { mutableStateOf(false) }
    var loadingImage by remember { mutableStateOf(false) }
    var diseaseModelsCompleted by remember { mutableStateOf(0) }
    var status by remember { mutableStateOf("Choose a color fundus photograph or an eye/face frame, then run the matching local model.") }
    val imagePicker = rememberLauncherForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        if (uri != null) {
            scope.launch {
                val targetEye = imageTargetEye
                loadingImage = true
                if (targetEye == 0) { rightImage = null; rightDiseaseResults = emptyList() } else { leftImage = null; leftDiseaseResults = emptyList() }
                eyeResult = null
                status = "Preparing the selected image locally…"
                val decoded = runCatching { withContext(Dispatchers.IO) { decodeSelectedImage(context, uri) } }.getOrNull()
                if (targetEye == 0) { rightImage = decoded; rightDiseaseResults = emptyList() } else { leftImage = decoded; leftDiseaseResults = emptyList() }
                status = if (decoded == null) "Could not open that image. Choose a readable JPEG, PNG, or WebP image." else "${if (targetEye == 0) "Right" else "Left"} eye image loaded locally. Run its research models."
                loadingImage = false
            }
        }
    }
    val installedDiseaseIds = MobileResearchInference.availableDiseaseIds(context)
    Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(20.dp), verticalArrangement = Arrangement.spacedBy(14.dp)) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            IconButton(onClick = onBack) { Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = "Back") }
            Text("On-device model lab", style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.Bold)
        }
            Text("17 optional RFMiD research heads: 7 dataset condition labels and 10 retinal image findings. Four condition heads are the post-hoc core shortlist; other outputs are exploratory. Install only the models you need. No glaucoma classifier is installed.", style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
        Text("Images stay on this phone. Add one color fundus photo for each eye; each eye is analyzed separately.", style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
        PrimaryTabRow(selectedTabIndex = if (selectedEye == 0) 1 else 0) {
            Tab(selected = selectedEye == 1, enabled = !running && !loadingImage, onClick = { selectedEye = 1 }, text = { Text("Left · ${leftDiseaseResults.maxOfOrNull { it.value }?.let { "%.1f%%".format(it * 100) } ?: "—"}") })
            Tab(selected = selectedEye == 0, enabled = !running && !loadingImage, onClick = { selectedEye = 0 }, text = { Text("Right · ${rightDiseaseResults.maxOfOrNull { it.value }?.let { "%.1f%%".format(it * 100) } ?: "—"}") })
        }
        Button(enabled = !running && !loadingImage, onClick = { imageTargetEye = selectedEye; imagePicker.launch("image/*") }, modifier = Modifier.fillMaxWidth().heightIn(min = 50.dp)) {
            Icon(Icons.Filled.Search, contentDescription = null)
            Spacer(Modifier.width(8.dp))
            Text("Choose ${if (selectedEye == 0) "right" else "left"}-eye fundus photo")
        }
        selectedImage?.let { bitmap ->
            Image(bitmap.asImageBitmap(), contentDescription = "Selected image", modifier = Modifier.fillMaxWidth().heightIn(max = 280.dp).clip(RoundedCornerShape(16.dp)))
        }
        Button(enabled = selectedImage != null && installedDiseaseIds.isNotEmpty() && !running && !loadingImage, onClick = {
            val bitmap = selectedImage ?: return@Button
            if (selectedEye == 0) rightDiseaseResults = emptyList() else leftDiseaseResults = emptyList()
            eyeResult = null
            running = true
            runningDiseaseModels = true
            diseaseModelsCompleted = 0
            status = "Running ${installedDiseaseIds.size} installed disease-specific ONNX models on-device…"
            scope.launch {
                val result = runCatching {
                    withContext(Dispatchers.IO) {
                        MobileResearchInference.classify(context, bitmap) { completed, id ->
                            scope.launch {
                                diseaseModelsCompleted = completed
                                status = "Completed $completed/${installedDiseaseIds.size}: ${modelDisplayName(id)}"
                            }
                        }
                    }
                }
                result.onSuccess { if (selectedEye == 0) rightDiseaseResults = it else leftDiseaseResults = it; status = "Completed locally for the ${if (selectedEye == 0) "right" else "left"} eye. Research scores are not diagnoses." }
                    .onFailure { if (selectedEye == 0) rightDiseaseResults = emptyList() else leftDiseaseResults = emptyList(); status = "Disease inference failed: ${it.message ?: "model/runtime error"}" }
                running = false
                runningDiseaseModels = false
            }
        }, modifier = Modifier.fillMaxWidth().heightIn(min = 50.dp)) { Text("Run installed research models") }
        if (installedDiseaseIds.isEmpty()) {
            Text("No disease models are installed yet. Open Settings, check for updates, and download only the condition models you need.", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
        if (diseaseResults.isNotEmpty()) {
            Text("${if (selectedEye == 0) "Right" else "Left"} eye · per-disease model scores", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
            Text("Only installed models are scored; a missing model is not assessed, not 0%. The 0.50 cutoff is for research only. Percentages are raw model scores, not calibrated probabilities, patient-risk estimates, or diagnoses. Group values show the highest score for this eye.", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
        MobileDiseaseCatalog.groups.filterNot { it.isFinding }.takeIf { groups -> groups.any { group -> diseaseResults.any { it.id in group.ids } } }?.let { diseaseGroups ->
            Text("Dataset condition labels", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
            diseaseGroups.forEach { group ->
            val groupName = group.name
            val rows = diseaseResults.filter { it.id in group.ids }
            val highest = rows.maxOfOrNull { it.value }
            if (highest != null) {
                Card(
                    onClick = { expandedScoreGroups = if (groupName in expandedScoreGroups) expandedScoreGroups - groupName else expandedScoreGroups + groupName },
                    colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceContainerLow),
                ) {
                    Column(Modifier.fillMaxWidth().padding(14.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Column(Modifier.weight(1f)) {
                                Text(groupName, fontWeight = FontWeight.Bold)
                                Text("Highest raw score", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                            }
                            Text("${"%.1f".format(highest * 100)}%", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
                        }
                        if (groupName in expandedScoreGroups) {
                            HorizontalDivider()
                            rows.forEach { item ->
                                Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                                    Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(3.dp)) {
                                        Text(modelDisplayName(item.id), style = MaterialTheme.typography.bodyMedium, fontWeight = FontWeight.SemiBold)
                                        Text("RFMiD · v${item.version} · ${if (item.id in MobileDiseaseCatalog.coreIds) "core shortlist" else "exploratory"} · ${if (item.value >= 0.5f) "above" else "below"} research cutoff", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                                    }
                                    Text("${"%.1f".format(item.value * 100)}%", style = MaterialTheme.typography.bodyMedium, fontWeight = FontWeight.SemiBold)
                                }
                            }
                        }
                    }
                }
            }
            }
        }
        MobileDiseaseCatalog.groups.filter { it.isFinding }.takeIf { groups -> groups.any { group -> diseaseResults.any { it.id in group.ids } } }?.let { findingGroups ->
            Text("Retinal image findings", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
            findingGroups.forEach { group ->
                val groupName = group.name
                val rows = diseaseResults.filter { it.id in group.ids }
                val highest = rows.maxOfOrNull { it.value }
                if (highest != null) Card(
                    onClick = { expandedScoreGroups = if (groupName in expandedScoreGroups) expandedScoreGroups - groupName else expandedScoreGroups + groupName },
                    colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceContainerLow),
                ) {
                    Column(Modifier.fillMaxWidth().padding(14.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Column(Modifier.weight(1f)) {
                                Text(groupName, fontWeight = FontWeight.Bold)
                                Text("Highest raw score", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                            }
                            Text("${"%.1f".format(highest * 100)}%", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
                        }
                        if (groupName in expandedScoreGroups) {
                            HorizontalDivider()
                            rows.forEach { item ->
                                Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                                    Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(2.dp)) {
                                        Text(modelDisplayName(item.id), style = MaterialTheme.typography.bodyMedium)
                                        Text("RFMiD · model v${item.version}", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                                    }
                                    Text("${"%.1f".format(item.value * 100)}%", style = MaterialTheme.typography.bodyMedium, fontWeight = FontWeight.SemiBold)
                                }
                            }
                        }
                    }
                }
            }
        }
        Button(enabled = selectedImage != null && !running && !loadingImage, onClick = {
            val bitmap = selectedImage ?: return@Button
            eyeResult = null
            running = true
            runningDiseaseModels = false
            status = "Running synthetic-domain eye localization on-device…"
            scope.launch {
                val result = runCatching { withContext(Dispatchers.IO) { MobileResearchInference.locate(context, bitmap) } }
                result.onSuccess { eyeResult = it; status = "Completed locally. This is not validated real-eye tracking." }
                    .onFailure { eyeResult = null; status = "Eye-model inference failed: ${it.message ?: "model/runtime error"}" }
                running = false
            }
        }, modifier = Modifier.fillMaxWidth().heightIn(min = 50.dp)) { Text("Run eye localization model") }
        eyeResult?.let { result ->
            val description = if (result.centerX != null && result.centerY != null) {
                "Synthetic-domain pupil candidate · x ${"%.0f".format(result.centerX)} px, y ${"%.0f".format(result.centerY)} px · ${result.pupilPixels} pixels · score ${"%.3f".format(result.score ?: 0f)}"
            } else "No pupil candidate above the minimum-pixel rule · ${result.pupilPixels} pixels found"
            Text(description, style = MaterialTheme.typography.bodyMedium)
        }
        if (loadingImage) LinearProgressIndicator(Modifier.fillMaxWidth())
        if (runningDiseaseModels) {
            LinearProgressIndicator(
                progress = { diseaseModelsCompleted.toFloat() / installedDiseaseIds.size.coerceAtLeast(1) },
                modifier = Modifier.fillMaxWidth(),
            )
            Text("Completed $diseaseModelsCompleted of ${installedDiseaseIds.size} installed models", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
        } else if (running) LinearProgressIndicator(Modifier.fillMaxWidth())
        Text(status, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
        Text("Glaucoma is not classified here. RFMiD optic-disc cupping is a finding, not a glaucoma diagnosis. These disease heads were evaluated on RFMiD image splits with patient independence unverified; scores are not calibrated diagnoses. Eye localization was trained on synthetic renders only, not people, IR images, or goggles. Do not use for patient care or motor control.", style = MaterialTheme.typography.bodySmall, fontWeight = FontWeight.Medium, color = MaterialTheme.colorScheme.onSurfaceVariant)
    }
}

private fun decodeSelectedImage(context: Context, uri: Uri): Bitmap? {
    val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
    context.contentResolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it, null, bounds) }
    if (bounds.outWidth <= 0 || bounds.outHeight <= 0) return null
    var sample = 1
    while (bounds.outWidth / sample > 2048 || bounds.outHeight / sample > 2048) sample *= 2
    val options = BitmapFactory.Options().apply { inSampleSize = sample; inPreferredConfig = Bitmap.Config.ARGB_8888 }
    val decoded = context.contentResolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it, null, options) } ?: return null
    val orientation = context.contentResolver.openInputStream(uri)?.use { stream ->
        ExifInterface(stream).getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL)
    } ?: ExifInterface.ORIENTATION_NORMAL
    val transform = Matrix()
    when (orientation) {
        ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> transform.setScale(-1f, 1f)
        ExifInterface.ORIENTATION_ROTATE_180 -> transform.setRotate(180f)
        ExifInterface.ORIENTATION_FLIP_VERTICAL -> transform.setScale(1f, -1f)
        ExifInterface.ORIENTATION_TRANSPOSE -> { transform.setRotate(90f); transform.postScale(-1f, 1f) }
        ExifInterface.ORIENTATION_ROTATE_90 -> transform.setRotate(90f)
        ExifInterface.ORIENTATION_TRANSVERSE -> { transform.setRotate(-90f); transform.postScale(-1f, 1f) }
        ExifInterface.ORIENTATION_ROTATE_270 -> transform.setRotate(-90f)
        else -> return decoded
    }
    val upright = Bitmap.createBitmap(decoded, 0, 0, decoded.width, decoded.height, transform, true)
    if (upright !== decoded) decoded.recycle()
    return upright
}

private object MobileResearchInference {
    val diseaseIds = MobileDiseaseCatalog.allIds

    fun availableDiseaseIds(context: Context): List<String> = diseaseIds.filter { id ->
        val version = context.getSharedPreferences("model_registry", Context.MODE_PRIVATE).getString("model-version:$id", null)
        version?.let { File(File(context.filesDir, "golden-retina-models"), "$id-$it.onnx").isFile } == true
    }

    fun classify(context: Context, source: Bitmap, onProgress: (Int, String) -> Unit): List<MobileDiseaseScore> =
        buildList {
            val available = availableDiseaseIds(context)
            require(available.isNotEmpty()) { "No disease model is installed. Check for model updates in Settings, then download the conditions you need." }
            available.forEachIndexed { index, id ->
                val session = openSession(context, id)
                val score = try {
                    require(session.inputNames.size == 1 && session.outputNames.size == 1) { "Unexpected ONNX input/output count for $id." }
                    val inputName = session.inputNames.first()
                    val outputName = session.outputNames.first()
                    val input = inputTensor(source, 320, true)
                    try {
                        session.run(mapOf(inputName to input)).use { result ->
                            val output = result.get(outputName).orElseThrow { IllegalStateException("Missing model output: $outputName") } as OnnxTensor
                            val tensorInfo = output.info
                            require(tensorInfo.shape.contentEquals(longArrayOf(1L, 1L))) { "Unexpected output shape from $id." }
                            val values = output.floatBuffer.duplicate()
                            require(values.remaining() == 1) { "Expected one scalar output from $id." }
                            val rawScore = values.get()
                            require(rawScore.isFinite() && rawScore in 0f..1f) { "Invalid score range from $id." }
                            val version = context.getSharedPreferences("model_registry", Context.MODE_PRIVATE)
                                .getString("model-version:$id", null)
                                ?: throw IllegalStateException("Missing installed version metadata for $id.")
                            MobileDiseaseScore(id, rawScore, version)
                        }
                    } finally { input.close() }
                } finally { session.close() }
                add(score)
                onProgress(index + 1, id)
            }
        }

    fun locate(context: Context, source: Bitmap): MobileEyeResult {
        val session = openSession(context, "ir_eye_tracker")
        try {
            require(session.inputNames.size == 1 && session.outputNames.size == 1) { "Unexpected ONNX input/output count for eye localization." }
            val inputName = session.inputNames.first()
            val outputName = session.outputNames.first()
            val input = inputTensor(source, 384, false)
            try {
                session.run(mapOf(inputName to input)).use { result ->
                    val output = result.get(outputName).orElseThrow { IllegalStateException("Missing model output: $outputName") } as OnnxTensor
                    val buffer = output.floatBuffer
                    val values = FloatArray(buffer.remaining())
                    buffer.get(values)
                    val plane = 384 * 384
                    require(values.size == plane * 3) { "Eye model output shape mismatch." }
                    var count = 0
                    var sumX = 0.0
                    var sumY = 0.0
                    var sumScore = 0.0
                    for (pixel in 0 until plane) {
                        val score = values[2 * plane + pixel]
                        if (score > values[pixel] && score > values[plane + pixel]) {
                            count++
                            sumX += pixel % 384
                            sumY += pixel / 384
                            sumScore += score
                        }
                    }
                    if (count < 5) return MobileEyeResult(count, null, null, null)
                    return MobileEyeResult(count, sumX / count * source.width / 384.0, sumY / count * source.height / 384.0, (sumScore / count).toFloat())
                }
            } finally { input.close() }
        } finally { session.close() }
    }

    private fun openSession(context: Context, id: String): OrtSession {
        require(id.matches(Regex("[A-Za-z0-9_-]+")))
        val preferences = context.getSharedPreferences("model_registry", Context.MODE_PRIVATE)
        val version = preferences.getString("model-version:$id", null)
        val installed = version?.let { File(File(context.filesDir, "golden-retina-models"), "$id-$it.onnx") }
        val modelFile = if (installed?.isFile == true) installed else {
            require(id == "ir_eye_tracker") { "Download ${modelDisplayName(id)} from Settings before running it." }
            val bundled = File(context.cacheDir, "golden-retina-bundled/$BUNDLED_MODEL_VERSION/$id.onnx")
            if (!bundled.isFile) {
                bundled.parentFile?.mkdirs()
                context.assets.open("$id.onnx").use { input -> bundled.outputStream().use { output -> input.copyTo(output) } }
            }
            bundled
        }
        val options = OrtSession.SessionOptions()
        return try {
            // Keep the ONNX graph at basic optimization: extended fusion turns
            // the FP16 ConvNeXt GELU into a custom float16 kernel that Android
            // CPU EP does not provide.
            options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.BASIC_OPT)
            OrtEnvironment.getEnvironment().createSession(modelFile.absolutePath, options)
        } finally { options.close() }
    }

    private fun inputTensor(source: Bitmap, size: Int, normalize: Boolean): OnnxTensor {
        val resized = Bitmap.createScaledBitmap(source, size, size, true)
        val pixels = IntArray(size * size)
        resized.getPixels(pixels, 0, size, 0, 0, size, size)
        val values = FloatArray(3 * size * size)
        val means = if (normalize) floatArrayOf(.485f, .456f, .406f) else floatArrayOf(0f, 0f, 0f)
        val stds = if (normalize) floatArrayOf(.229f, .224f, .225f) else floatArrayOf(1f, 1f, 1f)
        val plane = size * size
        for (i in pixels.indices) {
            values[i] = (AndroidColor.red(pixels[i]) / 255f - means[0]) / stds[0]
            values[plane + i] = (AndroidColor.green(pixels[i]) / 255f - means[1]) / stds[1]
            values[2 * plane + i] = (AndroidColor.blue(pixels[i]) / 255f - means[2]) / stds[2]
        }
        if (resized !== source) resized.recycle()
        return OnnxTensor.createTensor(OrtEnvironment.getEnvironment(), java.nio.FloatBuffer.wrap(values), longArrayOf(1, 3, size.toLong(), size.toLong()))
    }
}

private object PhoneModelRegistry {
    private const val bundledVersion = BUNDLED_MODEL_VERSION

    suspend fun refresh(context: Context, base: String): Map<String, String> = withContext(Dispatchers.IO) {
        val inputUrl = URL(base.trim())
        val root = if (inputUrl.path.endsWith("manifest.json")) inputUrl else URL(base.trimEnd('/') + "/v1/manifest.json")
        require(isTrustedRegistry(root)) { "Use the official Golden Retina registry on raw.githubusercontent.com/Burak8159/golden-retina." }
        val manifestText = (root.openConnection() as HttpURLConnection).run {
            connectTimeout = 15_000
            readTimeout = 30_000
            instanceFollowRedirects = false
            require(responseCode == HttpURLConnection.HTTP_OK) { "Manifest host returned HTTP $responseCode." }
            require(url.protocol == "https" && isTrustedRegistry(url)) { "Registry redirect was not trusted." }
            require(contentLengthLong in 1..1_048_576) { "Model registry is empty or unexpectedly large." }
            getInputStream().bufferedReader().use { it.readText() }
        }
        val manifest = JSONObject(manifestText)
        require(manifest.optInt("schema_version", -1) == 1) { "Unsupported manifest schema." }
        val models = manifest.getJSONObject("models")
        require(models.length() > 0) { "Model manifest contains no models." }
        val allowedIDs = (MobileResearchInference.diseaseIds + "ir_eye_tracker").toSet()
        val modelIds = models.keys().asSequence().toSet()
        require(modelIds.isNotEmpty() && allowedIDs.containsAll(modelIds)) { "Registry contains an unsupported model identifier." }
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
            val artifact = URL(fileUrl)
            require(isTrustedArtifact(artifact, id)) { "$id artifact URL is not in the trusted Golden Retina model folder." }
            val bytes = item.optLong("bytes", -1)
            require(bytes in 1..MAX_MODEL_BYTES) { "$id has an invalid or excessive model size." }
            require(expected.matches(Regex("[0-9a-f]{64}"))) { "$id has an invalid SHA-256." }
            val installedVersion = context.getSharedPreferences("model_registry", Context.MODE_PRIVATE).getString("model-version:$id", null)
            val installedFile = installedVersion?.let { File(File(context.filesDir, "golden-retina-models"), "$id-$it.onnx") }
            versions[id] = when {
                installedVersion == version && installedFile?.isFile == true -> "Current update installed · $version · SHA-256 verified"
                id == "ir_eye_tracker" && version == bundledVersion -> "Bundled on device · $version"
                else -> "Update available · $version"
            }
        }
        context.getSharedPreferences("model_registry", Context.MODE_PRIVATE).edit().putString("manifest-json", manifestText).apply()
        versions.forEach { (id, value) -> context.getSharedPreferences("model_registry", Context.MODE_PRIVATE).edit().putString("model:$id", value).apply() }
        versions
    }

    suspend fun install(context: Context, id: String): String = withContext(Dispatchers.IO) {
        require(id.matches(Regex("[A-Za-z0-9_-]+"))) { "Invalid model identifier." }
        val prefs = context.getSharedPreferences("model_registry", Context.MODE_PRIVATE)
        val manifestText = prefs.getString("manifest-json", null) ?: error("Check the model manifest first.")
        val item = JSONObject(manifestText).getJSONObject("models").getJSONObject(id)
        val version = item.optString("version")
        require(item.optString("status") == "ready" && item.optString("format") == "onnx") { "No downloadable ONNX release exists for this model." }
        require(version.matches(Regex("[A-Za-z0-9._-]{1,64}"))) { "Invalid model version." }
        val fileUrl = URL(item.getString("artifact_url"))
        require(isTrustedArtifact(fileUrl, id)) { "Model download is outside the trusted Golden Retina repository." }
        val expectedBytes = item.optLong("bytes", -1)
        require(expectedBytes in 1..MAX_MODEL_BYTES) { "Model size is invalid or exceeds the safe download limit." }
        val expected = item.getString("sha256").lowercase()
        require(expected.matches(Regex("[0-9a-f]{64}"))) { "Invalid model checksum." }
        val directory = File(context.filesDir, "golden-retina-models").apply { mkdirs() }
        val target = File(directory, "$id-$version.onnx")
        val staging = File(directory, ".$id-$version.part")
        try {
            val connection = fileUrl.openConnection() as HttpURLConnection
            connection.connectTimeout = 15_000
            connection.readTimeout = 120_000
            connection.instanceFollowRedirects = true
            require(connection.responseCode == HttpURLConnection.HTTP_OK && connection.url.protocol == "https") {
                "Model host did not return a secure successful download."
            }
            require(isTrustedArtifact(connection.url, id)) { "Model download redirected outside the trusted repository." }
            require(connection.contentLengthLong == expectedBytes) { "Model size does not match the registry." }
            connection.inputStream.use { input -> staging.outputStream().use { output ->
                val buffer = ByteArray(1024 * 1024)
                var total = 0L
                while (true) {
                    val count = input.read(buffer)
                    if (count < 0) break
                    total += count
                    require(total <= expectedBytes) { "Model download exceeded its declared size." }
                    output.write(buffer, 0, count)
                }
                require(total == expectedBytes) { "Model download ended before the declared size." }
            } }
            require(sha256(staging) == expected) { "Downloaded model failed SHA-256 verification." }
            validateOnnx(staging)
            val backup = File(directory, ".$id-$version.previous")
            if (backup.exists()) check(backup.delete()) { "Could not clear the previous staging backup." }
            val hadPrevious = target.exists()
            if (hadPrevious) check(target.renameTo(backup)) { "Could not preserve the installed model during update." }
            if (!staging.renameTo(target)) {
                if (hadPrevious) backup.renameTo(target)
                error("Could not install the verified model.")
            }
            if (backup.exists()) backup.delete()
            directory.listFiles()?.filter { it.name.startsWith("$id-") && it.name.endsWith(".onnx") && it != target }?.forEach { it.delete() }
            prefs.edit().putString("model-version:$id", version).putString("model:$id", "Current update installed · $version · SHA-256 verified").apply()
            "Current update installed · $version · SHA-256 verified"
        } finally { staging.delete() }
    }

    private fun validateOnnx(path: File) {
        val options = OrtSession.SessionOptions()
        try {
            options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.BASIC_OPT)
            OrtEnvironment.getEnvironment().createSession(path.absolutePath, options).close()
        } finally { options.close() }
    }

    private fun isTrustedRegistry(url: URL): Boolean =
        url.protocol == "https" && url.host.equals("raw.githubusercontent.com", ignoreCase = true) &&
            url.path.startsWith("/Burak8159/golden-retina/")

    private fun isTrustedArtifact(url: URL, id: String): Boolean =
        isTrustedRegistry(url) && url.path.startsWith("/Burak8159/golden-retina/mobile_models/onnx/") &&
            url.path.substringAfterLast('/') == "$id.onnx"

    private const val MAX_MODEL_BYTES = 128L * 1024L * 1024L

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
private fun DemoExperience(modifier: Modifier = Modifier, onExit: () -> Unit, onOpenResearch: () -> Unit) {
    var phaseName by rememberSaveable { mutableStateOf(DemoPhase.WELCOME.name) }
    val phase = DemoPhase.valueOf(phaseName)
    val scanning = phase == DemoPhase.RIGHT_SCAN || phase == DemoPhase.LEFT_SCAN
    val progress by animateFloatAsState(if (scanning) 1f else 0f, animationSpec = tween(3500, easing = LinearEasing), label = "sample scan progress")

    LaunchedEffect(phase) {
        when (phase) {
            DemoPhase.SEARCHING -> { delay(2200); phaseName = DemoPhase.CONNECTED.name }
            DemoPhase.RIGHT_SCAN -> { delay(3600); phaseName = DemoPhase.RIGHT_DONE.name }
            DemoPhase.RIGHT_DONE -> { delay(1100); phaseName = DemoPhase.LEFT_SCAN.name }
            DemoPhase.LEFT_SCAN -> { delay(3600); phaseName = DemoPhase.LEFT_DONE.name }
            DemoPhase.LEFT_DONE -> { delay(1100); phaseName = DemoPhase.ANALYZING.name }
            DemoPhase.ANALYZING -> { delay(3000); phaseName = DemoPhase.RESULTS.name }
            else -> Unit
        }
    }

    Scaffold(
        modifier = modifier,
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
                    DemoPhase.CONNECTED -> DeviceConnected { phaseName = DemoPhase.RIGHT_SCAN.name }
                    DemoPhase.RIGHT_SCAN -> EyeScan("Right", progress)
                    DemoPhase.RIGHT_DONE -> EyeCaptured("Right", "Starting the left-eye scan automatically…")
                    DemoPhase.LEFT_SCAN -> EyeScan("Left", progress)
                    DemoPhase.LEFT_DONE -> EyeCaptured("Left", "Preparing the example summary automatically…")
                    DemoPhase.ANALYZING -> DemoAnalysis()
                    DemoPhase.RESULTS -> DemoResults(onReplay = { phaseName = DemoPhase.WELCOME.name }, onDone = onExit, onAnalyze = onOpenResearch)
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
            Text("This guided simulation shows device setup, a goggles-button press, both eye scans, and an example summary.", style = MaterialTheme.typography.bodyLarge, textAlign = TextAlign.Center, color = MaterialTheme.colorScheme.onSurfaceVariant)
            AboutFact("No camera or goggles are used", Icons.Filled.Info)
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
                Box(contentAlignment = Alignment.Center) { Icon(Icons.Filled.Search, contentDescription = null, tint = MaterialTheme.colorScheme.onPrimaryContainer, modifier = Modifier.size(42.dp)) }
            }
        }
        Spacer(Modifier.height(24.dp))
        Text("Looking for the demo device…", style = MaterialTheme.typography.headlineSmall, textAlign = TextAlign.Center, fontWeight = FontWeight.Bold)
        Spacer(Modifier.height(8.dp))
        Text("Simulated search · no device discovery is running", style = MaterialTheme.typography.bodyMedium, textAlign = TextAlign.Center, color = MaterialTheme.colorScheme.onSurfaceVariant)
    }
}

@Composable
private fun DeviceConnected(onButtonPress: () -> Unit) {
    Column(Modifier.fillMaxSize().padding(vertical = 24.dp), horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.Center) {
        Icon(Icons.Filled.CheckCircle, contentDescription = null, tint = MaterialTheme.colorScheme.primary, modifier = Modifier.size(72.dp))
        Spacer(Modifier.height(20.dp))
        Text("Demo goggles connected", style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.Bold, textAlign = TextAlign.Center)
        Spacer(Modifier.height(8.dp))
        Text("In the real product, press the button on the goggles to begin. Tap below to simulate that press.", style = MaterialTheme.typography.bodyLarge, textAlign = TextAlign.Center, color = MaterialTheme.colorScheme.onSurfaceVariant)
        Spacer(Modifier.height(28.dp))
        Button(onClick = onButtonPress, modifier = Modifier.fillMaxWidth().heightIn(min = 56.dp), shape = RoundedCornerShape(18.dp)) {
            Icon(Icons.Filled.PlayArrow, contentDescription = null)
            Spacer(Modifier.width(8.dp))
            Text("Simulate goggles button press", style = MaterialTheme.typography.titleMedium)
        }
        Spacer(Modifier.height(10.dp))
        Text("Demo interaction · no goggles are connected", style = MaterialTheme.typography.bodySmall, textAlign = TextAlign.Center, color = MaterialTheme.colorScheme.onSurfaceVariant)
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
                    Icon(Icons.Filled.Search, contentDescription = null, modifier = Modifier.size(18.dp), tint = MaterialTheme.colorScheme.onSecondaryContainer)
                    Text("$eye eye", style = MaterialTheme.typography.labelLarge, color = MaterialTheme.colorScheme.onSecondaryContainer)
                }
            }
        }
        Box(Modifier.size(250.dp), contentAlignment = Alignment.Center) {
            EyeTrackerGraphic(progress, Modifier.fillMaxWidth().height(190.dp))
        }
        Text(
            when {
                progress >= 1f -> "$eye eye demo step complete · no image captured"
                progress < .34f -> "Finding the ${eye.lowercase()} eye…"
                progress < .68f -> "Adjusting position and focus…"
                else -> "Capturing the ${eye.lowercase()} eye…"
            },
            style = MaterialTheme.typography.titleMedium,
            fontWeight = FontWeight.SemiBold,
            textAlign = TextAlign.Center,
        )
        LinearProgressIndicator(progress = { progress }, modifier = Modifier.fillMaxWidth().height(5.dp).clip(CircleShape))
        Text("DEMO IMAGE · NOT A REAL EYE", style = MaterialTheme.typography.labelSmall, letterSpacing = 1.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
        Text("Simulated eye tracking · no gaze sensor is active", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
        Spacer(Modifier.weight(1f))
        Text("Keep still and look toward the center", style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
    }
}

@Composable
private fun EyeTrackerGraphic(progress: Float, modifier: Modifier = Modifier) {
    Canvas(modifier) {
        val left = size.width * .08f
        val right = size.width * .92f
        val top = size.height * .22f
        val bottom = size.height * .78f
        val midY = size.height * .5f
        val eye = Path().apply {
            moveTo(left, midY)
            cubicTo(size.width * .30f, top, size.width * .70f, top, right, midY)
            cubicTo(size.width * .70f, bottom, size.width * .30f, bottom, left, midY)
            close()
        }
        drawPath(eye, Color(0xFFE8F2EB))
        drawPath(eye, GoldenGreenDark, style = Stroke(width = 3.dp.toPx()))
        val irisX = left + (right - left) * (.42f + .16f * progress.coerceIn(0f, 1f))
        val irisRadius = size.height * .22f
        drawCircle(Color(0xFF329A78), irisRadius, Offset(irisX, midY))
        drawCircle(Color(0xFF082B27), irisRadius * .55f, Offset(irisX, midY))
        drawCircle(Color.White, 5.dp.toPx(), Offset(irisX, midY))
        drawCircle(Color.White.copy(alpha = .8f), 14.dp.toPx(), Offset(irisX, midY), style = Stroke(width = 1.5.dp.toPx()))
        val scanY = size.height * (.12f + .76f * progress.coerceIn(0f, 1f))
        drawLine(Color.White.copy(alpha = .9f), Offset(size.width * .04f, scanY), Offset(size.width * .96f, scanY), 2.dp.toPx(), cap = StrokeCap.Round)
    }
}

@Composable
private fun EyeCaptured(eye: String, nextStep: String) {
    Column(Modifier.fillMaxSize().padding(vertical = 22.dp), horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.SpaceBetween) {
        Spacer(Modifier.height(12.dp))
        Column(horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.spacedBy(14.dp)) {
            Icon(Icons.Filled.CheckCircle, contentDescription = null, tint = MaterialTheme.colorScheme.primary, modifier = Modifier.size(68.dp))
            Text("$eye eye demo step complete", style = MaterialTheme.typography.headlineSmall, textAlign = TextAlign.Center, fontWeight = FontWeight.Bold)
            Text("No eye image was captured", style = MaterialTheme.typography.bodyLarge, color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
        Row(Modifier.fillMaxWidth().heightIn(min = 54.dp), horizontalArrangement = Arrangement.Center, verticalAlignment = Alignment.CenterVertically) {
            CircularProgressIndicator(modifier = Modifier.size(20.dp), strokeWidth = 2.dp)
            Spacer(Modifier.width(10.dp))
            Text(nextStep, style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
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
private fun DemoResults(onReplay: () -> Unit, onDone: () -> Unit, onAnalyze: () -> Unit) {
    var selectedEye by remember { mutableStateOf(1) }
    val leftScores = listOf(
        "rfmid_dr" to .627f, "rfmid_armd" to .149f, "rfmid_mya" to .085f, "rfmid_brvo" to .051f,
        "rfmid_dn" to .093f, "rfmid_odc" to .184f, "rfmid_odp" to .061f, "rfmid_ode" to .032f,
        "rfmid_tsln" to .214f, "rfmid_ls" to .044f, "rfmid_csr" to .072f, "rfmid_rs" to .038f,
        "rfmid_crs" to .026f, "rfmid_crvo" to .019f, "rfmid_erm" to .058f, "rfmid_ms" to .029f,
        "rfmid_rt" to .041f,
    )
    val rightScores = listOf(
        "rfmid_dr" to .124f, "rfmid_armd" to .087f, "rfmid_mya" to .231f, "rfmid_brvo" to .042f,
        "rfmid_dn" to .051f, "rfmid_odc" to .103f, "rfmid_odp" to .027f, "rfmid_ode" to .018f,
        "rfmid_tsln" to .164f, "rfmid_ls" to .022f, "rfmid_csr" to .039f, "rfmid_rs" to .015f,
        "rfmid_crs" to .011f, "rfmid_crvo" to .008f, "rfmid_erm" to .025f, "rfmid_ms" to .014f,
        "rfmid_rt" to .020f,
    )
    val scores = if (selectedEye == 1) leftScores else rightScores
    val eyeName = if (selectedEye == 1) "Left" else "Right"
    Column(
        Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(24.dp),
        verticalArrangement = Arrangement.spacedBy(18.dp),
    ) {
        DemoPill()
        Text("Sample analysis", style = MaterialTheme.typography.headlineMedium, fontWeight = FontWeight.Bold)
        Text("A sample report preview", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
        Text(
            "These fixed example scores demonstrate the results layout. No eye images were captured or analyzed, and these are not predictions for you.",
            style = MaterialTheme.typography.bodyLarge,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )

        PrimaryTabRow(selectedTabIndex = if (selectedEye == 1) 0 else 1) {
            Tab(selected = selectedEye == 1, onClick = { selectedEye = 1 }, text = { Text("Left · ${"%.1f".format(leftScores.maxOf { it.second } * 100)}%") })
            Tab(selected = selectedEye == 0, onClick = { selectedEye = 0 }, text = { Text("Right · ${"%.1f".format(rightScores.maxOf { it.second } * 100)}%") })
        }
        Text(
            "$eyeName eye · sample model scores",
            style = MaterialTheme.typography.titleMedium,
            fontWeight = FontWeight.Bold,
        )
        Text("Dataset condition labels", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
        MobileDiseaseCatalog.groups.filterNot { it.isFinding }.forEach { group ->
            val groupScores = scores.filter { (id, _) -> id in group.ids }
            var expanded by remember(group.name, selectedEye) { mutableStateOf(group.name == "Diabetic retinopathy") }
            Card(onClick = { expanded = !expanded }, colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceContainerLow)) {
                Column(Modifier.fillMaxWidth().padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Column(Modifier.weight(1f)) {
                            Text(group.name, fontWeight = FontWeight.SemiBold)
                            Text("Highest sample score", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                        }
                        Text("${"%.1f".format(groupScores.maxOf { it.second } * 100)}%", style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.Bold)
                    }
                    if (expanded) {
                        HorizontalDivider()
                        groupScores.forEach { (id, score) ->
                            Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                                Row(verticalAlignment = Alignment.CenterVertically) {
                                    Text(modelDisplayName(id), Modifier.weight(1f), style = MaterialTheme.typography.bodyMedium, fontWeight = FontWeight.SemiBold)
                                    Text("${"%.1f".format(score * 100)}%", style = MaterialTheme.typography.bodyMedium, fontWeight = FontWeight.Bold)
                                }
                                LinearProgressIndicator(progress = { score }, modifier = Modifier.fillMaxWidth().height(5.dp))
                                Text("${if (id in MobileDiseaseCatalog.coreIds) "Core shortlist" else "Exploratory head"} · fixed sample value", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                            }
                        }
                    }
                }
            }
        }
        Text("Retinal image findings", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
        MobileDiseaseCatalog.groups.filter { it.isFinding }.forEach { group ->
            val groupScores = scores.filter { (id, _) -> id in group.ids }
            var expanded by remember(group.name, selectedEye) { mutableStateOf(false) }
            Card(onClick = { expanded = !expanded }, colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceContainerLow)) {
                Column(Modifier.fillMaxWidth().padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Column(Modifier.weight(1f)) {
                            Text(group.name, fontWeight = FontWeight.SemiBold)
                            Text("Highest sample score", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                        }
                        Text("${"%.1f".format(groupScores.maxOf { it.second } * 100)}%", style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.Bold)
                    }
                    if (expanded) {
                        HorizontalDivider()
                        groupScores.forEach { (id, score) ->
                            Text("${modelDisplayName(id)} · ${"%.1f".format(score * 100)}%", style = MaterialTheme.typography.bodyMedium)
                        }
                    }
                }
            }
        }
        Text(
            "Sample values only · no images were captured and no model produced these values. They are not your predictions, disease risks, or a diagnosis. Select Analyze your own image to run installed models locally.",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        Button(onClick = onAnalyze, modifier = Modifier.fillMaxWidth().heightIn(min = 52.dp)) {
            Icon(Icons.Filled.Search, contentDescription = null)
            Spacer(Modifier.width(8.dp))
            Text("Analyze your own fundus image")
        }
        Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            OutlinedAction("Replay demo", Icons.Filled.Refresh, onReplay, Modifier.weight(1f))
            Button(onClick = onDone, modifier = Modifier.weight(1f).heightIn(min = 52.dp), shape = RoundedCornerShape(16.dp)) { Text("Done") }
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
