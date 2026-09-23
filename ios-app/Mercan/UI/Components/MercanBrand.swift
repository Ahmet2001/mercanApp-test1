import SwiftUI

enum MercanTheme {
    static let coral = Color(red: 175.0 / 255.0, green: 55.0 / 255.0, blue: 46.0 / 255.0)
    static let warmBackground = Color(red: 250.0 / 255.0, green: 248.0 / 255.0, blue: 244.0 / 255.0)
}

struct MercanMark: View {
    var size: CGFloat = 32

    var body: some View {
        Image("MercanMark")
            .resizable()
            .scaledToFit()
            .frame(width: size, height: size)
            .accessibilityLabel("Mercan")
    }
}

struct MercanWordmark: View {
    var compact = false

    var body: some View {
        HStack(spacing: compact ? 6 : 10) {
            MercanMark(size: compact ? 26 : 36)
            Text("Mercan")
                .font(.system(size: compact ? 20 : 30, weight: .semibold, design: .serif))
                .foregroundStyle(.primary)
        }
    }
}
