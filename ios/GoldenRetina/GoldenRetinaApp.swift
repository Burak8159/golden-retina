import SwiftUI
import CryptoKit
import Foundation
import ImageIO
import OnnxRuntimeBindings
import PhotosUI
import UIKit

@main
struct GoldenRetinaApp: App {
    var body: some Scene {
        WindowGroup {
            RootView()
                .tint(BrandColor.green)
        }
    }
}

enum BrandColor {
    static let green = Color(red: 0.08, green: 0.43, blue: 0.36)
    static let mint = Color(red: 0.68, green: 0.91, blue: 0.81)
    static let ink = Color.primary
}

enum AppTab: Hashable {
    case home
    case demo
    case research
    case guide
    case settings
}

struct RootView: View {
    @State private var selection: AppTab = .home
    @AppStorage("goldenRetinaColorScheme") private var colorChoice = "system"

    private var preferredScheme: ColorScheme? {
        switch colorChoice { case "light": .light; case "dark": .dark; default: nil }
    }

    var body: some View {
        TabView(selection: $selection) {
            NavigationStack {
                HomeView(onOpenResearch: { selection = .research })
            }
            .tabItem { Label("Home", systemImage: "house.fill") }
            .tag(AppTab.home)

            NavigationStack {
                DemoFlowView(onExit: { selection = .home }, onOpenResearch: { selection = .research })
            }
            .tabItem { Label("Demo", systemImage: "eye") }
            .tag(AppTab.demo)

            NavigationStack { OnDeviceInferenceView() }
                .tabItem { Label("Analyze", systemImage: "waveform.path.ecg") }
                .tag(AppTab.research)

            NavigationStack { HowToUseView() }
                .tabItem { Label("How to use", systemImage: "questionmark.circle") }
                .tag(AppTab.guide)

            NavigationStack { SettingsView() }
                .tabItem { Label("Settings", systemImage: "gearshape") }
                .tag(AppTab.settings)
        }
        .preferredColorScheme(preferredScheme)
    }
}

struct HomeView: View {
    var onOpenResearch: () -> Void = {}
    @State private var showDemo = false
    @State private var showGogglesInfo = false

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 24) {
                HStack {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("GOLDEN HORN 8159")
                            .font(.caption.weight(.bold))
                            .tracking(1.6)
                            .foregroundStyle(BrandColor.green)
                        Text("Golden Retina")
                            .font(.largeTitle.bold())
                    }
                    Spacer()
                    Image(systemName: "eye.circle.fill")
                        .font(.system(size: 42))
                        .symbolRenderingMode(.palette)
                        .foregroundStyle(.white, BrandColor.green)
                        .accessibilityLabel("Golden Retina")
                }
                .padding(.top, 12)

                hero

                VStack(alignment: .leading, spacing: 12) {
                    Text("Choose how to continue")
                        .font(.title3.bold())
                    Text("Try a guided demo now, or connect compatible goggles when the hardware is ready.")
                        .font(.body)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }

                Button { showDemo = true } label: {
                    HStack(spacing: 16) {
                        Image(systemName: "play.circle.fill")
                            .font(.system(size: 34))
                            .foregroundStyle(BrandColor.green)
                        VStack(alignment: .leading, spacing: 4) {
                            HStack(spacing: 8) {
                                Text("Explore the demo")
                                    .font(.headline)
                                DemoBadge()
                            }
                            Text("See simulated device setup and the automatic two-eye scan flow")
                                .font(.subheadline)
                                .foregroundStyle(.secondary)
                                .multilineTextAlignment(.leading)
                        }
                        Spacer(minLength: 0)
                        Image(systemName: "chevron.right")
                            .font(.footnote.weight(.bold))
                            .foregroundStyle(.tertiary)
                    }
                    .padding(18)
                    .background(.background, in: RoundedRectangle(cornerRadius: 22, style: .continuous))
                    .overlay(RoundedRectangle(cornerRadius: 22, style: .continuous).stroke(.quaternary, lineWidth: 1))
                    .contentShape(RoundedRectangle(cornerRadius: 22, style: .continuous))
                }
                .buttonStyle(.plain)
                .accessibilityHint("Opens a fully simulated experience. No eye images are captured.")

                Button { showGogglesInfo = true } label: {
                    HStack(spacing: 16) {
                        Image(systemName: "eyeglasses")
                            .font(.system(size: 30))
                            .foregroundStyle(.secondary)
                            .frame(width: 38)
                        VStack(alignment: .leading, spacing: 4) {
                            Text("Connect goggles")
                                .font(.headline)
                                .foregroundStyle(.primary)
                            Text("Hardware connection is not available yet")
                                .font(.subheadline)
                                .foregroundStyle(.secondary)
                        }
                        Spacer(minLength: 0)
                        Label("Locked", systemImage: "lock.fill")
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(.secondary)
                            .padding(.horizontal, 10)
                            .padding(.vertical, 7)
                            .background(.quaternary, in: Capsule())
                    }
                    .padding(18)
                    .background(.background, in: RoundedRectangle(cornerRadius: 22, style: .continuous))
                    .overlay(RoundedRectangle(cornerRadius: 22, style: .continuous).stroke(.quaternary, lineWidth: 1))
                    .contentShape(RoundedRectangle(cornerRadius: 22, style: .continuous))
                }
                .buttonStyle(.plain)
                .accessibilityHint("Explains why real goggles cannot be connected yet.")

                Label("Demo only · Example values are illustrative; no real image or medical result is produced", systemImage: "checkmark.shield")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                    .padding(.bottom, 12)
            }
            .padding(.horizontal, 20)
            .frame(maxWidth: 600)
            .frame(maxWidth: .infinity)
        }
        .safeAreaPadding(.bottom, 24)
        .background(Color(uiColor: .systemGroupedBackground))
        .navigationBarHidden(true)
        .fullScreenCover(isPresented: $showDemo) { DemoFlowView(onOpenResearch: { showDemo = false; DispatchQueue.main.async(execute: onOpenResearch) }) }
        .sheet(isPresented: $showGogglesInfo) {
            GogglesUnavailableSheet()
                .presentationDetents([.medium])
                .presentationDragIndicator(.visible)
        }
    }

    private var hero: some View {
        GeometryReader { geometry in
            let artworkSize = min(geometry.size.width * 0.36, 154)
            ZStack(alignment: .leading) {
                RoundedRectangle(cornerRadius: 28, style: .continuous)
                    .fill(Color(red: 0.08, green: 0.28, blue: 0.25))
                RetinaArtwork()
                    .frame(width: artworkSize, height: artworkSize)
                    .opacity(0.9)
                    .position(
                        x: geometry.size.width - artworkSize / 2,
                        y: geometry.size.height * 0.52
                    )
                    .accessibilityHidden(true)
                VStack(alignment: .leading, spacing: 9) {
                    Text("A clearer view of eye screening")
                        .font(.title3.bold())
                        .foregroundStyle(.white)
                        .fixedSize(horizontal: false, vertical: true)
                    Text("Follow a simple, step-by-step product demonstration.")
                        .font(.subheadline)
                        .foregroundStyle(.white.opacity(0.84))
                        .fixedSize(horizontal: false, vertical: true)
                    Label("Guided experience", systemImage: "sparkles")
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(BrandColor.mint)
                        .padding(.top, 3)
                }
                .frame(maxWidth: min(geometry.size.width * 0.56, 205), alignment: .leading)
                .padding(22)
            }
            .clipShape(RoundedRectangle(cornerRadius: 28, style: .continuous))
        }
        .frame(height: 230)
    }
}

struct DemoBadge: View {
    var body: some View {
        Text("DEMO")
            .font(.caption2.weight(.heavy))
            .tracking(0.8)
            .foregroundStyle(BrandColor.green)
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .background(BrandColor.green.opacity(0.11), in: Capsule())
            .accessibilityLabel("Simulation")
    }
}

struct GogglesUnavailableSheet: View {
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            Image(systemName: "eyeglasses")
                .font(.system(size: 36))
                .foregroundStyle(BrandColor.green)
            Text("Goggles are still in development")
                .font(.title2.bold())
            Text("There is no finished goggle hardware or Bluetooth connection to pair with yet. This option will become available after the device and its connection protocol are ready.")
                .font(.body)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            Button("Got it") { dismiss() }
                .buttonStyle(.borderedProminent)
                .tint(BrandColor.green)
                .frame(maxWidth: .infinity, alignment: .trailing)
        }
        .padding(24)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
    }
}

struct HowToUseView: View {
    var body: some View {
        List {
            Section {
                Label("Guided walkthrough", systemImage: "list.number")
                    .font(.headline)
                Text("The demo is a simulation. It does not connect to goggles, capture images, track your gaze, or run a disease model.")
                    .foregroundStyle(.secondary)
            }
            Section("How a real session is intended to work") {
                manualStep("1", title: "Prepare the goggles", body: "Turn on the goggles and pair them from Settings once compatible hardware and its Bluetooth protocol are available.")
                manualStep("2", title: "Start with the goggles button", body: "In the intended product flow, pressing the physical goggles button begins acquisition. The demo uses an on-screen button to represent that action.")
                manualStep("3", title: "Hold still and look toward the guide", body: "The planned interface guides positioning and image capture. The current eye graphic is illustrative only; it does not track the user.")
                manualStep("4", title: "Capture both eyes", body: "The flow scans the right eye and then switches automatically to the left eye.")
                manualStep("5", title: "Review the sample summary", body: "The demo shows expandable condition groups with illustrative right- and left-eye percentages. These fixed examples are not produced by an image or disease model.")
            }
            Section("Instructional video") {
                ZStack {
                    RoundedRectangle(cornerRadius: 16).fill(Color.secondary.opacity(0.12))
                    VStack(spacing: 10) {
                        Image(systemName: "play.rectangle").font(.system(size: 34)).foregroundStyle(BrandColor.green)
                        Text("Instructional video coming soon").font(.headline)
                        Text("Video placeholder · no playback available").font(.caption).foregroundStyle(.secondary)
                    }
                    .padding(24)
                }
                .frame(height: 170)
                .accessibilityElement(children: .combine)
            }
            Section("Safety and current limitations") {
                Label("No real scans or clinical results are produced", systemImage: "cross.case")
                Label("Bluetooth goggles pairing is not implemented", systemImage: "eyeglasses")
            }
        }
        .navigationTitle("How to use")
        .navigationBarTitleDisplayMode(.inline)
    }

    private func manualStep(_ number: String, title: String, body: String) -> some View {
        HStack(alignment: .top, spacing: 12) {
            Text(number).font(.headline).foregroundStyle(.white)
                .frame(width: 30, height: 30).background(BrandColor.green, in: Circle())
            VStack(alignment: .leading, spacing: 5) {
                Text(title).font(.subheadline.weight(.semibold))
                Text(body).font(.footnote).foregroundStyle(.secondary)
            }
        }
        .padding(.vertical, 4)
    }
}

struct SettingsView: View {
    @StateObject private var registry = OnDeviceModelRegistry()
    @State private var registryURL = UserDefaults.standard.string(forKey: "goldenRetinaModelRegistryURL") ?? MobileModelRegistryURL.manifest
    @AppStorage("goldenRetinaColorScheme") private var colorChoice = "system"
    @State private var showBluetoothInfo = false

    var body: some View {
        List {
            Section("Appearance") {
                Picker("Color", selection: $colorChoice) {
                    Text("System").tag("system")
                    Text("Light").tag("light")
                    Text("Dark").tag("dark")
                }
                .pickerStyle(.segmented)
                .accessibilityLabel("App color appearance")
            }
            Section("Goggles") {
                Button { showBluetoothInfo = true } label: {
                    Label("Connect goggles via Bluetooth", systemImage: "eyeglasses")
                }
                .foregroundStyle(.primary)
                Label("Unavailable · hardware and Bluetooth protocol are not ready", systemImage: "lock.fill")
                    .font(.footnote).foregroundStyle(.secondary)
            }
            Section {
                Label("Golden Retina", systemImage: "eye.circle.fill")
                    .font(.title3.bold())
                    .foregroundStyle(BrandColor.green)
                Text("An early software concept from Golden Horn 8159.")
                    .foregroundStyle(.secondary)
            }
            Section("About this demo") {
                Label("Every scan and result is simulated", systemImage: "play.rectangle")
                Label("No camera images are collected", systemImage: "camera.fill")
                Label("No goggles or Bluetooth connection are active", systemImage: "eyeglasses")
                Label("Not a medical device or diagnosis", systemImage: "cross.case")
            }
            Section("On-device models") {
                Text("Models run on this phone with ONNX Runtime. The small eye-localization model is bundled; disease models download individually from the verified HTTPS registry. Selected images stay on-device; the guided demo remains simulated.")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                NavigationLink {
                    OnDeviceInferenceView()
                } label: {
                    Label("Run on-device research models", systemImage: "cpu")
                }
                TextField("HTTPS model manifest URL", text: $registryURL)
                    .textInputAutocapitalization(.never)
                    .keyboardType(.URL)
                    .autocorrectionDisabled()
                Text("For safety, model updates must come from the Golden Retina GitHub repository. New approved model versions can be published there without updating this app.")
                    .font(.caption).foregroundStyle(.secondary)
                Button {
                    UserDefaults.standard.set(registryURL, forKey: "goldenRetinaModelRegistryURL")
                    Task { await registry.refresh(registryURL: registryURL) }
                } label: {
                    Label("Check model updates", systemImage: "arrow.clockwise")
                }
                .disabled(registry.isBusy)
                if registry.isBusy { ProgressView("Checking registry…") }
                if !registry.message.isEmpty {
                    Text(registry.message).font(.footnote).foregroundStyle(.secondary)
                }
                let bundledIDs = ["ir_eye_tracker"]
                let modelIDs = Array(Set(bundledIDs).union(registry.installedVersions.keys).union(registry.manifest?.models.keys ?? Dictionary<String, MobileModelRelease>().keys)).sorted()
                ForEach(modelIDs, id: \.self) { id in
                    let release = registry.manifest?.models[id]
                    let bundled = bundledIDs.contains(id) ? "20261114-r3" : nil
                    modelRow(release?.displayName ?? modelDisplayName(id), release: release, installed: registry.installedVersions[id] ?? bundled, id: id)
                }
            }
            Section {
                Text("RFMiD models are research classifiers, not clinical diagnoses. The eye-localization model was trained on synthetic renders and is not validated on people, infrared cameras, or goggles. Do not use outputs for patient care or motor control.")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            } footer: {
                Text("Golden Retina · Early concept")
            }
        }
        .navigationTitle("Settings")
        .navigationBarTitleDisplayMode(.inline)
        .alert("Goggles connection unavailable", isPresented: $showBluetoothInfo) {
            Button("OK", role: .cancel) { }
        } message: {
            Text("Bluetooth pairing is not active because compatible goggles hardware and its connection protocol are not available yet.")
        }
    }

    @ViewBuilder
    private func modelRow(_ name: String, release: MobileModelRelease?, installed: String?, id: String) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text(name).font(.subheadline.weight(.semibold))
                Spacer()
                if installed != nil {
                    Label("On device · v\(installed!)", systemImage: "checkmark.circle.fill").font(.caption).foregroundStyle(BrandColor.green)
                } else {
                    Text(release?.status.replacingOccurrences(of: "_", with: " ").capitalized ?? "Not checked")
                        .font(.caption).foregroundStyle(.secondary)
                }
            }
            if let release, release.status == "ready", let version = release.version,
               installed != version {
                Button("Download v\(release.version ?? "model") · \(formatModelSize(release.bytes))") {
                    Task { await registry.install(id: id) }
                }
                .font(.caption.weight(.semibold))
                .disabled(registry.isBusy)
            }
        }
    }
}

private func formatModelSize(_ bytes: Int?) -> String {
    guard let bytes, bytes > 0 else { return "size unknown" }
    return String(format: "%.1f MB", Double(bytes) / 1_048_576)
}

private struct DiseaseScore: Identifiable {
    let id: String
    let score: Float
}

private func prepareModelImage(_ data: Data) -> UIImage? {
    guard let source = CGImageSourceCreateWithData(data as CFData, nil),
          let thumbnail = CGImageSourceCreateThumbnailAtIndex(
            source,
            0,
            [
                kCGImageSourceCreateThumbnailFromImageAlways: true,
                kCGImageSourceCreateThumbnailWithTransform: true,
                kCGImageSourceThumbnailMaxPixelSize: 2048
            ] as CFDictionary
          ) else { return nil }
    return UIImage(cgImage: thumbnail, scale: 1, orientation: .up)
}

private struct DiseaseScoreGroup: Identifiable {
    let id: String
    let modelIDs: [String]
}

private struct EyeLocalization {
    let pupilCount: Int
    let centerX: Double?
    let centerY: Double?
    let meanScore: Float?
}

private enum BundledResearchInference {
    static let diseaseIDs = [
        "rfmid_dr", "rfmid_armd", "rfmid_mya", "rfmid_brvo"
    ]

    static var installedDiseaseIDs: [String] { diseaseIDs.filter { modelURL(id: $0) != nil } }

    static func diseaseScores(image: UIImage, onProgress: (Int, String) -> Void) throws -> [DiseaseScore] {
        var scores: [DiseaseScore] = []
        let available = installedDiseaseIDs
        guard !available.isEmpty else { throw RegistryFailure.noDiseaseModelsInstalled }
        for (index, id) in available.enumerated() {
            let values = try run(image: image, id: id, size: 320, imagenetNormalize: true)
            guard values.count == 1, let rawScore = values.first, rawScore.isFinite,
                  (0...1).contains(rawScore) else { throw RegistryFailure.badResponse }
            scores.append(DiseaseScore(id: id, score: min(max(rawScore, 0), 1)))
            onProgress(index + 1, id)
        }
        return scores
    }

    static func locateEyes(image: UIImage) throws -> EyeLocalization {
        let values = try run(image: image, id: "ir_eye_tracker", size: 384, imagenetNormalize: false)
        let plane = 384 * 384
        guard values.count == 3 * plane else { throw RegistryFailure.badResponse }
        var count = 0
        var sumX = 0.0
        var sumY = 0.0
        var sumScore: Double = 0
        for pixel in 0..<plane {
            let pupil = values[2 * plane + pixel]
            if pupil > values[pixel], pupil > values[plane + pixel] {
                let x = pixel % 384
                let y = pixel / 384
                count += 1
                sumX += Double(x)
                sumY += Double(y)
                sumScore += Double(pupil)
            }
        }
        guard count >= 5, let cg = image.cgImage else {
            return EyeLocalization(pupilCount: count, centerX: nil, centerY: nil, meanScore: nil)
        }
        return EyeLocalization(
            pupilCount: count,
            centerX: (sumX / Double(count)) * Double(cg.width) / 384,
            centerY: (sumY / Double(count)) * Double(cg.height) / 384,
            meanScore: Float(sumScore / Double(count))
        )
    }

    private static func run(image: UIImage, id: String, size: Int, imagenetNormalize: Bool) throws -> [Float] {
        guard let modelURL = modelURL(id: id) else { throw RegistryFailure.badResponse }
        let env = try ORTEnv(loggingLevel: ORTLoggingLevel.warning)
        let options = try ORTSessionOptions()
        try options.setGraphOptimizationLevel(ORTGraphOptimizationLevel.basic)
        let session = try ORTSession(env: env, modelPath: modelURL.path, sessionOptions: options)
        let inputNames = try session.inputNames()
        let outputNames = try session.outputNames()
        guard inputNames.count == 1, outputNames.count == 1,
              let inputName = inputNames.first,
              let outputName = outputNames.first,
              let pixels = resizedRGB(image, size: size) else { throw RegistryFailure.badResponse }
        let means: [Float] = imagenetNormalize ? [0.485, 0.456, 0.406] : [0, 0, 0]
        let stds: [Float] = imagenetNormalize ? [0.229, 0.224, 0.225] : [1, 1, 1]
        var channels = [Float](repeating: 0, count: 3 * size * size)
        for pixel in 0..<(size * size) {
            for channel in 0..<3 {
                let value = Float(pixels[pixel * 4 + channel]) / 255
                channels[channel * size * size + pixel] = (value - means[channel]) / stds[channel]
            }
        }
        let data = channels.withUnsafeBytes { Data($0) }
        let input = try ORTValue(
            tensorData: NSMutableData(data: data),
            elementType: ORTTensorElementDataType.float,
            shape: [NSNumber(value: 1), NSNumber(value: 3), NSNumber(value: size), NSNumber(value: size)]
        )
        let outputs = try session.run(withInputs: [inputName: input], outputNames: [outputName], runOptions: nil)
        guard let output = outputs[outputName] else { throw RegistryFailure.badResponse }
        let outputData = try output.tensorData() as Data
        return outputData.withUnsafeBytes { raw in
            Array(raw.bindMemory(to: Float.self))
        }
    }

    private static func modelURL(id: String) -> URL? {
        if let versions = UserDefaults.standard.dictionary(forKey: "goldenRetinaInstalledModelVersions") as? [String: String],
           let version = versions[id],
           let support = try? FileManager.default.url(for: .applicationSupportDirectory, in: .userDomainMask, appropriateFor: nil, create: false) {
            let installed = support.appendingPathComponent("GoldenRetinaModels", isDirectory: true)
                .appendingPathComponent("\(id)-\(version).onnx")
            if FileManager.default.fileExists(atPath: installed.path) { return installed }
        }
        return Bundle.main.url(forResource: id, withExtension: "onnx")
    }

    private static func resizedRGB(_ image: UIImage, size: Int) -> [UInt8]? {
        guard let cgImage = image.cgImage else { return nil }
        var bytes = [UInt8](repeating: 0, count: size * size * 4)
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        bytes.withUnsafeMutableBytes { buffer in
            guard let context = CGContext(
                data: buffer.baseAddress, width: size, height: size,
                bitsPerComponent: 8, bytesPerRow: size * 4, space: colorSpace,
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
            ) else { return }
            context.interpolationQuality = .high
            context.draw(cgImage, in: CGRect(x: 0, y: 0, width: size, height: size))
        }
        return bytes
    }
}

private struct OnDeviceInferenceView: View {
    private let scoreGroups = [
        DiseaseScoreGroup(id: "Retina and macula", modelIDs: ["rfmid_dr", "rfmid_armd", "rfmid_mya", "rfmid_brvo"])
    ]
    private enum EyeSide: String { case right = "Right", left = "Left" }
    @State private var selectedEye: EyeSide = .right
    @State private var selectedItem: PhotosPickerItem?
    @State private var rightImage: UIImage?
    @State private var leftImage: UIImage?
    @State private var rightDiseaseResults: [DiseaseScore] = []
    @State private var leftDiseaseResults: [DiseaseScore] = []
    private var selectedImage: UIImage? { selectedEye == .right ? rightImage : leftImage }
    private var diseaseResults: [DiseaseScore] { selectedEye == .right ? rightDiseaseResults : leftDiseaseResults }
    @State private var eyeResult: EyeLocalization?
    @State private var status = "Choose a color fundus photograph or an eye/face frame to run the matching local model."
    @State private var isRunning = false
    @State private var isLoadingImage = false
    @State private var isRunningDiseaseModels = false
    @State private var diseaseModelsCompleted = 0

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                Text("On-device model lab").font(.largeTitle.bold())
                Text("Inference runs locally on this iPhone. Images are not uploaded or saved by this screen.")
                    .foregroundStyle(.secondary)
                Picker("Eye", selection: $selectedEye) {
                    Text("Right · \(rightDiseaseResults.map(\.score).max().map { String(format: "%.1f%%", $0 * 100) } ?? "—")").tag(EyeSide.right)
                    Text("Left · \(leftDiseaseResults.map(\.score).max().map { String(format: "%.1f%%", $0 * 100) } ?? "—")").tag(EyeSide.left)
                }
                .pickerStyle(.segmented)
                .disabled(isRunning || isLoadingImage)
                PhotosPicker(selection: $selectedItem, matching: .images) {
                    Label(rightImage == nil && selectedEye == .right || leftImage == nil && selectedEye == .left ? "Choose \(selectedEye.rawValue.lowercased())-eye fundus photo" : "Replace \(selectedEye.rawValue.lowercased())-eye photo", systemImage: "photo.on.rectangle.angled")
                        .frame(maxWidth: .infinity, minHeight: 52)
                }
                .buttonStyle(.borderedProminent)
                .tint(BrandColor.green)
                .disabled(isRunning || isLoadingImage)
                Text("Select Right and Left separately, add one color fundus photograph for each eye, then run the disease models on each selected image. Scores are shown under the eye selected here.")
                    .font(.footnote).foregroundStyle(.secondary)
                if let selectedImage {
                    Image(uiImage: selectedImage).resizable().scaledToFit().frame(maxHeight: 260)
                        .clipShape(RoundedRectangle(cornerRadius: 16))
                }
                Button {
                    runDiseaseModels()
                } label: {
                    Label(isRunning ? "Running locally…" : "Run disease classifiers", systemImage: "cross.case")
                        .frame(maxWidth: .infinity, minHeight: 48)
                }
                .buttonStyle(.bordered)
                .disabled(selectedImage == nil || isRunning || isLoadingImage || BundledResearchInference.installedDiseaseIDs.isEmpty)
                if BundledResearchInference.installedDiseaseIDs.isEmpty {
                    Text("No disease models are installed yet. Check for updates in Settings and download only the condition models you need.")
                        .font(.footnote).foregroundStyle(.secondary)
                }
                if !diseaseResults.isEmpty {
                    Text("\(selectedEye.rawValue) eye · per-disease model scores")
                        .font(.headline)
                    Text("These percentages are raw model scores for the selected image, not calibrated probabilities, patient-risk estimates, or diagnoses.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }
                if isRunningDiseaseModels {
                    ProgressView(value: Double(diseaseModelsCompleted), total: Double(max(BundledResearchInference.installedDiseaseIDs.count, 1)))
                        .tint(BrandColor.green)
                        .accessibilityLabel("Disease model progress")
                    Text("Completed \(diseaseModelsCompleted) of \(BundledResearchInference.installedDiseaseIDs.count) installed models")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }
                ForEach(scoreGroups) { group in
                    let rows = diseaseResults.filter { group.modelIDs.contains($0.id) }
                    if let highest = rows.map(\.score).max() {
                        DisclosureGroup {
                            ForEach(rows) { result in
                                HStack {
                                    Text(modelDisplayName(result.id))
                                    Spacer()
                                    Text(String(format: "%.1f%%", result.score * 100)).monospacedDigit().fontWeight(.semibold)
                                }
                                .padding(12)
                                .background(.quaternary.opacity(0.5), in: RoundedRectangle(cornerRadius: 12))
                            }
                        } label: {
                            HStack {
                                VStack(alignment: .leading, spacing: 3) {
                                    Text(group.id).font(.subheadline.weight(.semibold))
                                    Text("Highest model score (%)").font(.caption).foregroundStyle(.secondary)
                                }
                                Spacer()
                                Text(String(format: "%.1f%%", highest * 100)).monospacedDigit().fontWeight(.bold)
                            }
                        }
                    }
                }
                Button {
                    runEyeModel()
                } label: {
                    Label(isRunning ? "Running locally…" : "Run eye localization model", systemImage: "eye")
                        .frame(maxWidth: .infinity, minHeight: 48)
                }
                .buttonStyle(.bordered)
                .disabled(selectedImage == nil || isRunning || isLoadingImage)
                if let eyeResult {
                    if let x = eyeResult.centerX, let y = eyeResult.centerY {
                        Text(String(format: "Synthetic-domain pupil candidate: x %.0f, y %.0f px · %d pixels · score %.3f", x, y, eyeResult.pupilCount, eyeResult.meanScore ?? 0))
                    } else {
                        Text("No pupil candidate above the minimum-pixel rule (\(eyeResult.pupilCount) pixels found).")
                    }
                }
                Text(status).font(.footnote).foregroundStyle(.secondary)
                Text("Limits: the RFMiD heads are image-split research models with patient independence unverified. Scores are not calibrated diagnoses. Eye localization was trained/evaluated on synthetic renders only, not real people, IR frames, or goggles. Do not use for patient care or motor control.")
                    .font(.footnote.weight(.medium)).foregroundStyle(.secondary)
            }
            .padding(20)
        }
        .navigationTitle("Research inference")
        .navigationBarTitleDisplayMode(.inline)
        .onChange(of: selectedItem) { _, item in
            guard let item else { return }
            let targetEye = selectedEye
            if targetEye == .right { rightImage = nil; rightDiseaseResults = [] } else { leftImage = nil; leftDiseaseResults = [] }
            eyeResult = nil
            isLoadingImage = true
            status = "Preparing the selected image locally…"
            Task {
                defer { isLoadingImage = false }
                if let data = try? await item.loadTransferable(type: Data.self), let image = prepareModelImage(data) {
                    if targetEye == .right { rightImage = image; rightDiseaseResults = [] } else { leftImage = image; leftDiseaseResults = [] }
                    eyeResult = nil
                    status = "\(targetEye.rawValue) eye image is ready. Run its local disease models."
                } else {
                    status = "Could not load that image. Choose a JPEG, PNG, or HEIC image."
                }
            }
        }
    }

    private func runDiseaseModels() {
        guard let selectedImage else { return }
        let targetEye = selectedEye
        if targetEye == .right { rightDiseaseResults = [] } else { leftDiseaseResults = [] }
        eyeResult = nil
        isRunning = true
        isRunningDiseaseModels = true
        diseaseModelsCompleted = 0
        let installedIDs = BundledResearchInference.installedDiseaseIDs
        guard !installedIDs.isEmpty else { isRunning = false; isRunningDiseaseModels = false; status = "Download at least one disease model from Settings first."; return }
        status = "Running \(installedIDs.count) installed disease-specific ONNX models on this phone…"
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                let values = try BundledResearchInference.diseaseScores(image: selectedImage) { completed, id in
                    DispatchQueue.main.async {
                        diseaseModelsCompleted = completed
                        status = "Completed \(completed)/\(installedIDs.count): \(modelDisplayName(id))"
                    }
                }
                DispatchQueue.main.async {
                    if targetEye == .right { rightDiseaseResults = values } else { leftDiseaseResults = values }
                    status = "Completed locally for the \(targetEye.rawValue.lowercased()) eye. These are research scores, not diagnostic findings."
                    isRunning = false
                    isRunningDiseaseModels = false
                }
            } catch {
                DispatchQueue.main.async { status = "Inference failed: \(error.localizedDescription)"; isRunning = false; isRunningDiseaseModels = false }
            }
        }
    }

    private func runEyeModel() {
        guard let selectedImage else { return }
        eyeResult = nil
        isRunning = true
        isRunningDiseaseModels = false
        status = "Running synthetic-domain eye localization on this phone…"
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                let value = try BundledResearchInference.locateEyes(image: selectedImage)
                DispatchQueue.main.async {
                    eyeResult = value
                    status = "Completed locally. This result is not validated eye tracking."
                    isRunning = false
                }
            } catch {
                DispatchQueue.main.async { status = "Inference failed: \(error.localizedDescription)"; isRunning = false }
            }
        }
    }
}

private func modelDisplayName(_ id: String) -> String {
    switch id {
    case "ir_eye_tracker": return "Eye localization · synthetic research model"
    case "rfmid_dr": return "Diabetic retinopathy · RFMiD"
    case "rfmid_armd": return "Age-related macular degeneration · RFMiD"
    case "rfmid_dn": return "Drusen · RFMiD"
    case "rfmid_mya": return "Myopia · RFMiD"
    case "rfmid_brvo": return "Branch retinal vein occlusion · RFMiD"
    case "rfmid_odc": return "Optic-disc cupping · RFMiD"
    case "rfmid_odp": return "Optic-disc pallor · RFMiD"
    case "rfmid_ode": return "Optic-disc edema · RFMiD"
    case "rfmid_tsln": return "Tessellated fundus finding · RFMiD"
    case "rfmid_ls": return "Laser-scar finding · RFMiD"
    case "rfmid_csr": return "Central serous retinopathy · RFMiD"
    case "rfmid_rs": return "Retinal scar finding · RFMiD"
    case "rfmid_crs": return "Chorioretinitis · RFMiD"
    case "rfmid_crvo": return "Central retinal vein occlusion · RFMiD · low support"
    case "rfmid_erm": return "Epiretinal membrane · RFMiD · low support"
    case "rfmid_ms": return "Macular scar · RFMiD · low support"
    case "rfmid_rt": return "Retinal traction · RFMiD · low support"
    default: return id.replacingOccurrences(of: "_", with: " ")
    }
}

private struct MobileRegistryManifest: Decodable {
    let schemaVersion: Int
    let models: [String: MobileModelRelease]
    enum CodingKeys: String, CodingKey { case schemaVersion = "schema_version", models }
}

private enum MobileModelRegistryURL {
    static let manifest = "https://raw.githubusercontent.com/Burak8159/golden-retina/main/mobile_models/onnx/manifest.json"
}

private struct MobileModelRelease: Decodable {
    let displayName: String
    let status: String
    let version: String?
    let format: String?
    let artifactURL: URL?
    let sha256: String?
    let bytes: Int?
    enum CodingKeys: String, CodingKey { case displayName = "display_name", status, version, format, artifactURL = "artifact_url", sha256, bytes }
}

@MainActor
private final class OnDeviceModelRegistry: ObservableObject {
    @Published var manifest: MobileRegistryManifest?
    @Published var installedVersions: [String: String] = [:]
    @Published var isBusy = false
    @Published var message = ""

    init() { installedVersions = Self.readInstalledVersions() }

    func refresh(registryURL: String) async {
        guard let input = URL(string: registryURL), input.scheme?.lowercased() == "https", input.host != nil else {
            message = "Enter the HTTPS URL of the model registry. No registry is configured yet."
            manifest = nil
            return
        }
        guard input.host?.lowercased() == "raw.githubusercontent.com",
              input.path.hasPrefix("/Burak8159/golden-retina/") else {
            message = "For safety, use the official Golden Retina model registry on raw.githubusercontent.com/Burak8159/golden-retina."
            manifest = nil
            return
        }
        let url = input.lastPathComponent == "manifest.json"
            ? input
            : input.appendingPathComponent("v1").appendingPathComponent("manifest.json")
        isBusy = true
        defer { isBusy = false }
        do {
            var request = URLRequest(url: url)
            request.cachePolicy = .reloadIgnoringLocalCacheData
            request.timeoutInterval = 20
            let (data, response) = try await URLSession.shared.data(for: request)
            guard data.count <= 1_048_576 else { throw RegistryFailure.badResponse }
            guard (response as? HTTPURLResponse)?.statusCode == 200 else { throw RegistryFailure.badResponse }
            guard response.url?.scheme?.lowercased() == "https",
                  response.url?.host?.lowercased() == "raw.githubusercontent.com",
                  response.url?.path.hasPrefix("/Burak8159/golden-retina/") == true else { throw RegistryFailure.insecureRedirect }
            let decoded = try JSONDecoder().decode(MobileRegistryManifest.self, from: data)
            guard decoded.schemaVersion == 1 else { throw RegistryFailure.unsupportedSchema }
            let allowedIDs = Set(BundledResearchInference.diseaseIDs + ["ir_eye_tracker"])
            guard !decoded.models.isEmpty, Set(decoded.models.keys).isSubset(of: allowedIDs) else { throw RegistryFailure.badResponse }
            for (id, release) in decoded.models where release.status == "ready" {
                guard let bytes = release.bytes, (1...(128 * 1_048_576)).contains(bytes),
                      let artifact = release.artifactURL,
                      artifact.scheme?.lowercased() == "https",
                      artifact.host?.lowercased() == "raw.githubusercontent.com",
                      artifact.path.hasPrefix("/Burak8159/golden-retina/mobile_models/onnx/"),
                      artifact.lastPathComponent == "\(id).onnx" else { throw RegistryFailure.badResponse }
            }
            manifest = decoded
            message = "Registry checked. Download only the disease models you need; all inference stays on this iPhone."
        } catch {
            manifest = nil
            message = "Could not check registry: \(error.localizedDescription)"
        }
    }

    func install(id: String) async {
        guard id.range(of: "^[A-Za-z0-9_-]+$", options: .regularExpression) != nil,
              let release = manifest?.models[id], release.status == "ready",
              release.format?.lowercased() == "onnx", let artifactURL = release.artifactURL,
              artifactURL.scheme == "https", let expected = release.sha256?.lowercased(),
              expected.count == 64, let version = release.version,
              let expectedBytes = release.bytes, (1...(128 * 1_048_576)).contains(expectedBytes),
              artifactURL.host?.lowercased() == "raw.githubusercontent.com",
              artifactURL.path.hasPrefix("/Burak8159/golden-retina/mobile_models/onnx/"),
              artifactURL.lastPathComponent == "\(id).onnx",
              version.range(of: "^[A-Za-z0-9._-]{1,64}$", options: .regularExpression) != nil else {
            message = "This model has no compatible, downloadable ONNX release yet."
            return
        }
        isBusy = true
        defer { isBusy = false }
        do {
            var request = URLRequest(url: artifactURL)
            request.timeoutInterval = 120
            let (temporaryURL, response) = try await URLSession.shared.download(for: request)
            guard (response as? HTTPURLResponse)?.statusCode == 200 else { throw RegistryFailure.badResponse }
            guard response.url?.scheme?.lowercased() == "https",
                  response.url?.host?.lowercased() == "raw.githubusercontent.com",
                  response.url?.path.hasPrefix("/Burak8159/golden-retina/mobile_models/onnx/") == true,
                  response.url?.lastPathComponent == "\(id).onnx" else { throw RegistryFailure.insecureRedirect }
            let actualBytes = (try FileManager.default.attributesOfItem(atPath: temporaryURL.path)[.size] as? NSNumber)?.intValue
            guard actualBytes == expectedBytes else { throw RegistryFailure.badResponse }
            let actual = try Self.sha256(file: temporaryURL)
            guard actual == expected else { throw RegistryFailure.hashMismatch }
            let directory = try Self.modelDirectory()
            let target = directory.appendingPathComponent("\(id)-\(version).onnx")
            let staging = directory.appendingPathComponent(".\(id)-\(version).download")
            if FileManager.default.fileExists(atPath: staging.path) { try FileManager.default.removeItem(at: staging) }
            try FileManager.default.moveItem(at: temporaryURL, to: staging)
            let environment = try ORTEnv(loggingLevel: ORTLoggingLevel.warning)
            let options = try ORTSessionOptions()
            try options.setGraphOptimizationLevel(ORTGraphOptimizationLevel.basic)
            _ = try ORTSession(env: environment, modelPath: staging.path, sessionOptions: options)
            let backup = directory.appendingPathComponent(".\(id)-\(version).previous")
            let hadPrevious = FileManager.default.fileExists(atPath: target.path)
            if FileManager.default.fileExists(atPath: backup.path) { try? FileManager.default.removeItem(at: backup) }
            if hadPrevious { try FileManager.default.moveItem(at: target, to: backup) }
            do {
                try FileManager.default.moveItem(at: staging, to: target)
            } catch {
                if hadPrevious, FileManager.default.fileExists(atPath: backup.path) {
                    try? FileManager.default.moveItem(at: backup, to: target)
                }
                throw error
            }
            if FileManager.default.fileExists(atPath: backup.path) { try? FileManager.default.removeItem(at: backup) }
            let oldVersions = try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
                .filter { $0.lastPathComponent.hasPrefix("\(id)-") && $0.pathExtension == "onnx" && $0 != target }
            for oldVersion in oldVersions { try FileManager.default.removeItem(at: oldVersion) }
            var versions = Self.readInstalledVersions()
            versions[id] = version
            UserDefaults.standard.set(versions, forKey: "goldenRetinaInstalledModelVersions")
            installedVersions = versions
            message = "\(releaseName(id)) \(version) downloaded, checksum verified, and opened by the on-device runtime."
        } catch {
            message = "Model update failed: \(error.localizedDescription)"
        }
    }

    private func releaseName(_ id: String) -> String { id == "ir_eye_tracker" ? "Synthetic eye localization" : "Retinal classifier" }
    private static func modelDirectory() throws -> URL {
        let root = try FileManager.default.url(for: .applicationSupportDirectory, in: .userDomainMask, appropriateFor: nil, create: true)
        let directory = root.appendingPathComponent("GoldenRetinaModels", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory
    }
    private static func readInstalledVersions() -> [String: String] {
        guard let saved = UserDefaults.standard.dictionary(forKey: "goldenRetinaInstalledModelVersions") as? [String: String],
              let support = try? FileManager.default.url(for: .applicationSupportDirectory, in: .userDomainMask, appropriateFor: nil, create: false) else { return [:] }
        let directory = support.appendingPathComponent("GoldenRetinaModels", isDirectory: true)
        return saved.filter { id, version in
            id.range(of: "^[A-Za-z0-9_-]+$", options: .regularExpression) != nil &&
            version.range(of: "^[A-Za-z0-9._-]{1,64}$", options: .regularExpression) != nil &&
            FileManager.default.fileExists(atPath: directory.appendingPathComponent("\(id)-\(version).onnx").path)
        }
    }
    private static func sha256(file: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: file)
        defer { try? handle.close() }
        var hasher = SHA256()
        while let chunk = try handle.read(upToCount: 1024 * 1024), !chunk.isEmpty { hasher.update(data: chunk) }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }
}

private enum RegistryFailure: LocalizedError {
    case badResponse, unsupportedSchema, hashMismatch, insecureRedirect, noDiseaseModelsInstalled
    var errorDescription: String? {
        switch self {
        case .badResponse: return "The registry returned an unsuccessful response."
        case .unsupportedSchema: return "This app does not support that registry schema version."
        case .hashMismatch: return "The downloaded model failed its SHA-256 integrity check."
        case .insecureRedirect: return "The registry redirected to a non-HTTPS URL."
        case .noDiseaseModelsInstalled: return "No disease models are installed. Check for model updates in Settings and download the conditions you need."
        }
    }
}

struct RetinaArtwork: View {
    var body: some View {
        Canvas { context, size in
            let center = CGPoint(x: size.width * 0.57, y: size.height * 0.48)
            let radius = min(size.width, size.height) * 0.43
            let disc = CGRect(x: center.x - radius, y: center.y - radius, width: radius * 2, height: radius * 2)
            context.fill(Path(ellipseIn: disc), with: .linearGradient(
                Gradient(colors: [Color(red: 0.24, green: 0.72, blue: 0.55), Color(red: 0.05, green: 0.36, blue: 0.34)]),
                startPoint: CGPoint(x: disc.minX, y: disc.minY), endPoint: CGPoint(x: disc.maxX, y: disc.maxY)))
            context.stroke(Path(ellipseIn: disc.insetBy(dx: 8, dy: 8)), with: .color(.white.opacity(0.28)), lineWidth: 1)
            let optic = CGPoint(x: center.x - radius * 0.43, y: center.y + radius * 0.14)
            let angles: [CGFloat] = [-1.35, -1.0, -0.68, -0.35, 0.0, 0.42, 0.75, 1.08, 1.38, 1.72, 2.1, 2.45]
            context.clip(to: Path(ellipseIn: disc))
            for (index, angle) in angles.enumerated() {
                var path = Path()
                path.move(to: optic)
                let length = radius * (index.isMultiple(of: 3) ? 1.63 : 1.32)
                let end = CGPoint(x: optic.x + cos(angle) * length, y: optic.y + sin(angle) * length)
                let control = CGPoint(x: optic.x + cos(angle + 0.16) * length * 0.56, y: optic.y + sin(angle + 0.16) * length * 0.56)
                path.addQuadCurve(to: end, control: control)
                context.stroke(path, with: .color(.white.opacity(0.64)), lineWidth: index.isMultiple(of: 3) ? 2.1 : 1.25)
            }
            let fovea = CGRect(x: center.x + radius * 0.18, y: center.y - radius * 0.18, width: radius * 0.20, height: radius * 0.20)
            context.fill(Path(ellipseIn: fovea), with: .color(Color(red: 0.98, green: 0.81, blue: 0.49).opacity(0.9)))
        }
        .accessibilityHidden(true)
    }
}

enum DemoPhase: Equatable {
    case welcome, searching, connected, rightScan, rightDone, leftScan, leftDone, analyzing, results
}

struct DemoFlowView: View {
    @Environment(\.dismiss) private var dismiss
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    var onExit: (() -> Void)? = nil
    var onOpenResearch: (() -> Void)? = nil
    @State private var phase: DemoPhase = .welcome
    @State private var scanProgress = 0.0
    @State private var scanKey = UUID()

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                HStack {
                    DemoBadge()
                    Spacer()
                    Button("Exit demo", action: exitDemo)
                        .font(.subheadline.weight(.semibold))
                        .frame(minHeight: 44)
                }
                .padding(.horizontal, 20)
                .padding(.top, 8)

                if phase != .welcome && phase != .results {
                    ProgressView(value: journeyProgress)
                        .tint(BrandColor.green)
                        .padding(.horizontal, 20)
                        .padding(.top, 8)
                        .accessibilityLabel("Demo progress")
                }

                Group {
                    switch phase {
                    case .welcome: DemoWelcomeView(onStart: { phase = .searching })
                    case .searching: DeviceSearchView()
                    case .connected: ConnectedView(onButtonPress: { beginScan(.rightScan) })
                    case .rightScan: EyeScanView(eye: "Right", progress: scanProgress)
                    case .rightDone: CapturedView(eye: "Right", nextStep: "Starting the left-eye scan automatically…")
                    case .leftScan: EyeScanView(eye: "Left", progress: scanProgress)
                    case .leftDone: CapturedView(eye: "Left", nextStep: "Preparing the example summary automatically…")
                    case .analyzing: AnalysisView()
                    case .results: DemoResultsView(onReplay: reset, onDone: exitDemo, onAnalyze: { if let onOpenResearch { onOpenResearch() } else { exitDemo() } })
                    }
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .animation(reduceMotion ? nil : .easeInOut(duration: 0.25), value: phase)
            }
            .background(Color(uiColor: .systemGroupedBackground))
            .toolbar(.hidden, for: .navigationBar)
            .task(id: phase) { await advanceTimedPhases() }
            .onChange(of: phase) { _, newPhase in
                if newPhase == .rightScan || newPhase == .leftScan {
                    scanProgress = 0
                    scanKey = UUID()
                    withAnimation(.linear(duration: 3.4)) { scanProgress = 1 }
                }
            }
        }
        .interactiveDismissDisabled(false)
    }

    private var journeyProgress: Double {
        switch phase {
        case .welcome, .searching: 0.12
        case .connected: 0.22
        case .rightScan, .rightDone: 0.40 + scanProgress * 0.20
        case .leftScan, .leftDone: 0.62 + scanProgress * 0.20
        case .analyzing: 0.91
        case .results: 1
        }
    }

    @MainActor private func advanceTimedPhases() async {
        do {
            switch phase {
            case .searching:
                try await Task.sleep(for: .seconds(2.2))
                phase = .connected
            case .connected:
                // Wait for the simulated goggles button press.
                break
            case .rightScan:
                try await Task.sleep(for: .seconds(3.6))
                phase = .rightDone
            case .rightDone:
                try await Task.sleep(for: .seconds(1.1))
                beginScan(.leftScan)
            case .leftScan:
                try await Task.sleep(for: .seconds(3.6))
                phase = .leftDone
            case .leftDone:
                try await Task.sleep(for: .seconds(1.1))
                phase = .analyzing
            case .analyzing:
                try await Task.sleep(for: .seconds(3.1))
                phase = .results
            default: break
            }
        } catch is CancellationError {
            // A phase change cancels this simulated step.
        } catch {
            phase = .results
        }
    }

    private func beginScan(_ next: DemoPhase) {
        scanProgress = 0
        scanKey = UUID()
        phase = next
    }

    private func reset() {
        scanProgress = 0
        phase = .welcome
    }

    private func exitDemo() {
        if let onExit { onExit() } else { dismiss() }
    }
}

struct DemoWelcomeView: View {
    let onStart: () -> Void
    var body: some View {
        VStack(spacing: 24) {
            Spacer()
            RetinaArtwork()
                .frame(width: 210, height: 210)
                .accessibilityHidden(true)
            VStack(spacing: 10) {
                Text("See how a scan could work")
                    .font(.largeTitle.bold())
                    .multilineTextAlignment(.center)
                Text("This guided simulation shows device setup, a goggles-button press, both eye scans, and an example result.")
                    .font(.body)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
            }
            .padding(.horizontal, 24)
            Label("No camera or goggles are used", systemImage: "camera.macro")
                .font(.subheadline.weight(.medium))
                .foregroundStyle(.secondary)
            Spacer()
            Button(action: onStart) {
                Text("Start the demo")
                    .font(.headline)
                    .frame(maxWidth: .infinity, minHeight: 54)
            }
            .buttonStyle(.borderedProminent)
            .tint(BrandColor.green)
            .padding(.horizontal, 24)
            .padding(.bottom, 20)
        }
    }
}

struct DeviceSearchView: View {
    @State private var animate = false
    var body: some View {
        VStack(spacing: 24) {
            Spacer()
            ZStack {
                Circle().stroke(BrandColor.green.opacity(0.10), lineWidth: 1).frame(width: 220, height: 220)
                Circle().stroke(BrandColor.green.opacity(0.18), lineWidth: 1).frame(width: 164, height: 164)
                Circle().fill(BrandColor.green.opacity(0.08)).frame(width: 106, height: 106)
                Image(systemName: "eyeglasses").font(.system(size: 42)).foregroundStyle(BrandColor.green)
                Circle().fill(BrandColor.green).frame(width: 12, height: 12).offset(x: animate ? 83 : -83, y: animate ? -35 : 35)
            }
            .accessibilityHidden(true)
            .task {
                withAnimation(.easeInOut(duration: 1.1).repeatForever(autoreverses: true)) { animate = true }
            }
            VStack(spacing: 8) {
                Text("Looking for the demo device…")
                    .font(.title2.bold())
                    .multilineTextAlignment(.center)
                Text("Simulated search · no Bluetooth scan is running")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
            }
            Spacer()
        }
        .padding(24)
        .accessibilityElement(children: .combine)
        .accessibilityLabel("Demo simulation. Looking for a sample goggles device. No Bluetooth scan is running.")
    }
}

struct ConnectedView: View {
    let onButtonPress: () -> Void
    var body: some View {
        VStack(spacing: 20) {
            Spacer()
            Image(systemName: "checkmark.circle.fill")
                .font(.system(size: 68))
                .foregroundStyle(BrandColor.green)
            VStack(spacing: 8) {
                Text("Demo goggles connected")
                    .font(.title2.bold())
                Text("In the real product, press the button on the goggles to begin. Tap below to simulate that press.")
                    .font(.body)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
            }
            .padding(.horizontal, 24)
            Spacer()
            Button(action: onButtonPress) {
                Label("Simulate goggles button press", systemImage: "button.programmable")
                    .font(.headline)
                    .frame(maxWidth: .infinity, minHeight: 56)
            }
            .buttonStyle(.borderedProminent)
            .tint(BrandColor.green)
            .padding(.horizontal, 20)
            Text("Demo interaction · no goggles are connected")
                .font(.caption)
                .foregroundStyle(.secondary)
            Spacer()
        }
        .padding(24)
    }
}

struct EyeScanView: View {
    let eye: String
    let progress: Double
    var body: some View {
        VStack(spacing: 22) {
            HStack {
                Text("Scanning")
                    .font(.title2.bold())
                Spacer()
                Text("\(eye.uppercased()) EYE")
                    .font(.caption.weight(.bold))
                    .tracking(1)
                    .foregroundStyle(BrandColor.green)
                    .padding(.horizontal, 11)
                    .padding(.vertical, 8)
                    .background(BrandColor.green.opacity(0.10), in: Capsule())
            }
            .padding(.top, 16)
            ZStack {
                RoundedRectangle(cornerRadius: 30).fill(Color(red: 0.07, green: 0.22, blue: 0.20))
                EyeTrackingArtwork(progress: progress)
                    .padding(20)
                GeometryReader { proxy in
                    Capsule().fill(.white).frame(width: proxy.size.width * 0.78, height: 2)
                        .shadow(color: BrandColor.mint, radius: 7)
                        .position(x: proxy.size.width / 2, y: proxy.size.height * (0.12 + 0.76 * progress))
                }
                .padding(20)
            }
            .frame(width: 300, height: 220)
            .clipShape(RoundedRectangle(cornerRadius: 30))
            .accessibilityHidden(true)
            VStack(spacing: 10) {
                Text(scanStatus)
                    .font(.headline)
                    .contentTransition(.opacity)
                ProgressView(value: progress)
                    .tint(BrandColor.green)
                    .accessibilityLabel("\(eye) eye demo scan progress")
                Text("DEMO IMAGE · NOT A REAL EYE")
                    .font(.caption2.weight(.bold))
                    .tracking(1.1)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            Text("Keep still and look toward the center")
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .padding(.bottom, 18)
        }
        .padding(.horizontal, 24)
        .padding(.top, 12)
    }

    private var scanStatus: String {
        if progress >= 1 { return "\(eye) eye sample captured" }
        if progress < 0.34 { return "Finding the \(eye.lowercased()) eye…" }
        if progress < 0.68 { return "Adjusting position and focus…" }
        return "Capturing the \(eye.lowercased()) eye…"
    }
}

struct CapturedView: View {
    let eye: String
    let nextStep: String
    var body: some View {
        VStack(spacing: 22) {
            Spacer()
            Image(systemName: "checkmark.circle.fill")
                .font(.system(size: 60))
                .foregroundStyle(BrandColor.green)
            VStack(spacing: 8) {
                Text("\(eye) eye scan complete")
                    .font(.title2.bold())
                Text("Sample image captured in the demo")
                    .font(.body)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            HStack(spacing: 10) {
                ProgressView().tint(BrandColor.green)
                Text(nextStep).font(.subheadline.weight(.medium)).foregroundStyle(.secondary)
            }
            .frame(maxWidth: .infinity, minHeight: 54)
            .padding(.bottom, 20)
        }
        .padding(.horizontal, 24)
    }
}

struct AnalysisView: View {
    @State private var turn = false
    var body: some View {
        VStack(spacing: 22) {
            Spacer()
            ZStack {
                Circle().stroke(BrandColor.green.opacity(0.16), lineWidth: 1).frame(width: 170, height: 170)
                Circle().trim(from: 0.08, to: 0.72).stroke(BrandColor.green, style: StrokeStyle(lineWidth: 5, lineCap: .round))
                    .frame(width: 145, height: 145)
                    .rotationEffect(.degrees(turn ? 360 : 0))
                Image(systemName: "sparkle.magnifyingglass")
                    .font(.system(size: 42))
                    .foregroundStyle(BrandColor.green)
            }
            .accessibilityHidden(true)
            .task {
                withAnimation(.linear(duration: 2).repeatForever(autoreverses: false)) { turn = true }
            }
            VStack(spacing: 8) {
                Text("Reviewing both sample images")
                    .font(.title2.bold())
                    .multilineTextAlignment(.center)
                Text("The app is showing a timed animation. No AI model is running.")
                    .font(.body)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
                Label("Simulated eye tracking · demo only", systemImage: "scope")
                    .font(.caption.weight(.medium)).foregroundStyle(.secondary)
            }
            Spacer()
        }
        .padding(24)
    }
}

private struct EyeTrackingArtwork: View {
    let progress: Double
    var body: some View {
        Canvas { context, size in
            let eye = CGRect(x: size.width * 0.08, y: size.height * 0.22, width: size.width * 0.84, height: size.height * 0.56)
            var outline = Path()
            outline.move(to: CGPoint(x: eye.minX, y: eye.midY))
            outline.addQuadCurve(to: CGPoint(x: eye.maxX, y: eye.midY), control: CGPoint(x: eye.midX, y: eye.minY))
            outline.addQuadCurve(to: CGPoint(x: eye.minX, y: eye.midY), control: CGPoint(x: eye.midX, y: eye.maxY))
            context.fill(outline, with: .color(Color(red: 0.91, green: 0.96, blue: 0.92)))
            context.stroke(outline, with: .color(BrandColor.mint), lineWidth: 3)
            let irisCenter = CGPoint(x: eye.minX + eye.width * (0.42 + 0.16 * progress), y: eye.midY)
            let irisRect = CGRect(x: irisCenter.x - eye.height * 0.25, y: irisCenter.y - eye.height * 0.25, width: eye.height * 0.5, height: eye.height * 0.5)
            context.fill(Path(ellipseIn: irisRect), with: .color(Color(red: 0.18, green: 0.56, blue: 0.47)))
            let pupilRect = irisRect.insetBy(dx: irisRect.width * 0.28, dy: irisRect.height * 0.28)
            context.fill(Path(ellipseIn: pupilRect), with: .color(Color(red: 0.04, green: 0.16, blue: 0.15)))
            let gaze = CGRect(x: irisCenter.x - 5, y: irisCenter.y - 5, width: 10, height: 10)
            context.stroke(Path(ellipseIn: gaze.insetBy(dx: -9, dy: -9)), with: .color(.white), lineWidth: 1.5)
            context.fill(Path(ellipseIn: gaze), with: .color(.white))
        }
    }
}

struct DemoResultsView: View {
    let onReplay: () -> Void
    let onDone: () -> Void
    let onAnalyze: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            DemoBadge()
            Text("Demo complete").font(.largeTitle.bold())
            Text("The guided scan sequence was simulated. No eye images were captured or analyzed, so this demo does not produce disease classifications or risk percentages.")
                .foregroundStyle(.secondary)
            Button(action: onAnalyze) {
                Label("Analyze a fundus image", systemImage: "waveform.path.ecg")
                    .frame(maxWidth: .infinity, minHeight: 52)
            }
            .buttonStyle(.borderedProminent)
            .tint(BrandColor.green)
            Text("The on-device model lab runs the installed condition-specific research models on an image you select. It displays raw model scores, not calibrated disease risks or a medical diagnosis.")
                .font(.footnote).foregroundStyle(.secondary)
            Spacer()
            HStack {
                Button("Replay demo", action: onReplay).buttonStyle(.bordered)
                Spacer()
                Button("Done", action: onDone).buttonStyle(.bordered)
            }
        }
        .padding(24)
        .frame(maxWidth: 600, maxHeight: .infinity, alignment: .topLeading)
        .background(Color(uiColor: .systemGroupedBackground))
    }
}
