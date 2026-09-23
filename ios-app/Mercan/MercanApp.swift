import SwiftUI
import Foundation

@main
struct MercanApp: App {
    init() {
        #if targetEnvironment(simulator)
        setenv("GGML_METAL_NO_RESIDENCY", "1", 1)
        #endif
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
                .tint(MercanTheme.coral)
        }
    }
}
