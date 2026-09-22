import SwiftUI

private struct OnboardingPage: Identifiable {
    let id: Int
    let title: String
    let detail: String
    let symbol: String
}

struct OnboardingView: View {
    let onFinish: () -> Void
    @State private var page = 0

    private let pages = [
        OnboardingPage(
            id: 0,
            title: "Private by design",
            detail: "Your model and conversations stay on this device. Internet access is only needed when you choose to download a model.",
            symbol: "lock.shield"
        ),
        OnboardingPage(
            id: 1,
            title: "Mercan models",
            detail: "Mercan uses the libmercan runtime and validates model format, architecture and tokenizer compatibility before inference.",
            symbol: "cube.transparent"
        ),
        OnboardingPage(
            id: 2,
            title: "Ready for local AI",
            detail: "Chat, attach documents, tune generation settings and inspect performance without sending prompts to a remote inference API.",
            symbol: "iphone.gen3"
        )
    ]

    var body: some View {
        ZStack {
            MercanTheme.warmBackground.ignoresSafeArea()

            VStack(spacing: 24) {
                Spacer(minLength: 36)
                MercanWordmark()

                TabView(selection: $page) {
                    ForEach(pages) { item in
                        VStack(spacing: 22) {
                            Image(systemName: item.symbol)
                                .font(.system(size: 46, weight: .medium))
                                .foregroundStyle(MercanTheme.coral)
                                .frame(height: 64)

                            Text(item.title)
                                .font(.system(size: 28, weight: .semibold, design: .serif))
                                .multilineTextAlignment(.center)

                            Text(item.detail)
                                .font(.body)
                                .foregroundStyle(.secondary)
                                .multilineTextAlignment(.center)
                                .padding(.horizontal, 28)
                        }
                        .tag(item.id)
                    }
                }
                .tabViewStyle(.page(indexDisplayMode: .always))

                Button {
                    if page < pages.count - 1 {
                        withAnimation { page += 1 }
                    } else {
                        onFinish()
                    }
                } label: {
                    Text(page == pages.count - 1 ? "Start using Mercan" : "Continue")
                        .fontWeight(.semibold)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 14)
                        .background(MercanTheme.coral)
                        .foregroundStyle(.white)
                        .clipShape(RoundedRectangle(cornerRadius: 16))
                }
                .padding(.horizontal, 24)

                Button("Skip") { onFinish() }
                    .foregroundStyle(.secondary)
                    .opacity(page == pages.count - 1 ? 0 : 1)
                    .disabled(page == pages.count - 1)

                Spacer(minLength: 24)
            }
        }
    }
}
