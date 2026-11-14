import SwiftUI

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
    case about
}

struct RootView: View {
    @State private var selection: AppTab = .home

    var body: some View {
        TabView(selection: $selection) {
            NavigationStack {
                HomeView()
            }
            .tabItem { Label("Home", systemImage: "house.fill") }
            .tag(AppTab.home)

            NavigationStack {
                AboutView()
            }
            .tabItem { Label("About", systemImage: "info.circle.fill") }
            .tag(AppTab.about)
        }
    }
}

struct HomeView: View {
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
                            Text("See device setup, both eye scans, and sample results")
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

                Label("Demo only · No camera, Bluetooth, or medical result is used", systemImage: "checkmark.shield")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                    .padding(.bottom, 12)
            }
            .padding(.horizontal, 20)
            .frame(maxWidth: 600)
            .frame(maxWidth: .infinity)
        }
        .background(Color(uiColor: .systemGroupedBackground))
        .navigationBarHidden(true)
        .fullScreenCover(isPresented: $showDemo) { DemoFlowView() }
        .sheet(isPresented: $showGogglesInfo) {
            GogglesUnavailableSheet()
                .presentationDetents([.medium])
                .presentationDragIndicator(.visible)
        }
    }

    private var hero: some View {
        ZStack(alignment: .bottomLeading) {
            RoundedRectangle(cornerRadius: 28, style: .continuous)
                .fill(Color(red: 0.08, green: 0.28, blue: 0.25))
            RetinaArtwork()
                .frame(width: 190, height: 190)
                .opacity(0.95)
                .frame(maxWidth: .infinity, alignment: .trailing)
                .padding(.trailing, 4)
                .padding(.top, 8)
                .accessibilityHidden(true)
            VStack(alignment: .leading, spacing: 10) {
                Text("A clearer view of eye screening")
                    .font(.title2.bold())
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
            .padding(22)
            .frame(maxWidth: 245, alignment: .leading)
        }
        .frame(height: 230)
        .clipShape(RoundedRectangle(cornerRadius: 28, style: .continuous))
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

struct AboutView: View {
    var body: some View {
        List {
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
            Section {
                Text("The current research models are development prototypes. This app does not connect to them or assess a person’s eyes. Do not use the demo to make health decisions.")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            } footer: {
                Text("Golden Retina · Early concept")
            }
        }
        .navigationTitle("About")
        .navigationBarTitleDisplayMode(.inline)
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
    @State private var phase: DemoPhase = .welcome
    @State private var scanProgress = 0.0
    @State private var scanKey = UUID()

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                HStack {
                    DemoBadge()
                    Spacer()
                    Button("Exit demo") { dismiss() }
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
                    case .connected: ConnectedView()
                    case .rightScan: EyeScanView(eye: "Right", progress: scanProgress)
                    case .rightDone: CapturedView(eye: "Right", onContinue: { beginScan(.leftScan) })
                    case .leftScan: EyeScanView(eye: "Left", progress: scanProgress)
                    case .leftDone: CapturedView(eye: "Left", onContinue: { phase = .analyzing })
                    case .analyzing: AnalysisView()
                    case .results: DemoResultsView(onReplay: reset, onDone: { dismiss() })
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
                try await Task.sleep(for: .seconds(1.4))
                beginScan(.rightScan)
            case .rightScan:
                try await Task.sleep(for: .seconds(3.6))
                phase = .rightDone
            case .leftScan:
                try await Task.sleep(for: .seconds(3.6))
                phase = .leftDone
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
                Text("This guided simulation shows device setup, a scan of each eye, and an example result.")
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
    var body: some View {
        VStack(spacing: 20) {
            Spacer()
            Image(systemName: "checkmark.circle.fill")
                .font(.system(size: 68))
                .foregroundStyle(BrandColor.green)
            VStack(spacing: 8) {
                Text("Demo goggles connected")
                    .font(.title2.bold())
                Text("This connection is part of the simulation. No hardware was found or paired.")
                    .font(.body)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
            }
            .padding(.horizontal, 24)
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
                Circle().fill(Color(red: 0.07, green: 0.22, blue: 0.20))
                RetinaArtwork().padding(18)
                Circle().stroke(.white.opacity(0.75), lineWidth: 2)
                    .frame(width: 220, height: 220)
                    .overlay(alignment: .top) {
                        Capsule().fill(.white).frame(width: 220, height: 2).shadow(color: BrandColor.mint, radius: 7)
                            .offset(y: 110)
                            .offset(y: -220 * progress)
                    }
                    .mask(Circle().frame(width: 220, height: 220))
            }
            .frame(width: 250, height: 250)
            .clipShape(Circle())
            .accessibilityHidden(true)
            VStack(spacing: 10) {
                Text(progress < 1 ? "Capturing the \(eye.lowercased()) eye…" : "\(eye) eye captured")
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
}

struct CapturedView: View {
    let eye: String
    let onContinue: () -> Void
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
            Button(action: onContinue) {
                Text(eye == "Right" ? "Continue to left eye" : "Show example analysis")
                    .font(.headline)
                    .frame(maxWidth: .infinity, minHeight: 54)
            }
            .buttonStyle(.borderedProminent)
            .tint(BrandColor.green)
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
            }
            Spacer()
        }
        .padding(24)
    }
}

struct DemoResultsView: View {
    let onReplay: () -> Void
    let onDone: () -> Void
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                HStack {
                    DemoBadge()
                    Spacer()
                    Text("EXAMPLE ONLY")
                        .font(.caption.weight(.bold))
                        .tracking(0.8)
                        .foregroundStyle(.secondary)
                }
                VStack(alignment: .leading, spacing: 8) {
                    Text("Your demo is complete")
                        .font(.largeTitle.bold())
                    Text("Here is how a future screening summary might be organized.")
                        .font(.body)
                        .foregroundStyle(.secondary)
                }
                HStack(spacing: 12) {
                    MiniEyeCard(eye: "Right eye", status: "Sample reviewed", score: "0.18")
                    MiniEyeCard(eye: "Left eye", status: "Sample for review", score: "0.64")
                }
                VStack(alignment: .leading, spacing: 12) {
                    Label("Illustrative sample output", systemImage: "doc.text.magnifyingglass")
                        .font(.headline)
                    Text("A future report could summarize each eye and explain whether a clinician should review the images.")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                    Divider()
                    Text("The numbers above are invented demo values. They are not predictions, probabilities for you, or medical advice.")
                        .font(.footnote.weight(.medium))
                        .foregroundStyle(.primary)
                }
                .padding(18)
                .background(.background, in: RoundedRectangle(cornerRadius: 20, style: .continuous))
                .overlay(RoundedRectangle(cornerRadius: 20, style: .continuous).stroke(.quaternary, lineWidth: 1))

                Label("No image from you was captured or analyzed", systemImage: "camera.metering.unknown")
                    .font(.subheadline.weight(.semibold))
                    .foregroundStyle(BrandColor.green)
                    .fixedSize(horizontal: false, vertical: true)

                HStack(spacing: 12) {
                    Button(action: onReplay) {
                        Label("Replay demo", systemImage: "arrow.counterclockwise")
                            .frame(maxWidth: .infinity, minHeight: 50)
                    }
                    .buttonStyle(.bordered)
                    Button(action: onDone) {
                        Text("Done").frame(maxWidth: .infinity, minHeight: 50)
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(BrandColor.green)
                }
            }
            .padding(22)
            .frame(maxWidth: 600)
            .frame(maxWidth: .infinity)
        }
        .background(Color(uiColor: .systemGroupedBackground))
    }
}

struct MiniEyeCard: View {
    let eye: String
    let status: String
    let score: String
    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            RetinaArtwork().frame(height: 86).clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
                .accessibilityHidden(true)
            Text(eye).font(.subheadline.weight(.bold))
            Text(status).font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            HStack(alignment: .firstTextBaseline, spacing: 4) {
                Text(score).font(.title2.bold())
                Text("demo").font(.caption2.weight(.semibold)).foregroundStyle(.secondary)
            }
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.background, in: RoundedRectangle(cornerRadius: 18, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 18, style: .continuous).stroke(.quaternary, lineWidth: 1))
        .accessibilityElement(children: .combine)
    }
}
