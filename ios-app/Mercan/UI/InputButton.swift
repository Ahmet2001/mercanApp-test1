import SwiftUI

struct InputButton: View {
    @ObservedObject var llamaState: LlamaState
    @State private var inputLink = ""

    private static func extractModelInfo(from link: String) -> (modelName: String, filename: String)? {
        guard let url = URL(string: link),
              !url.lastPathComponent.isEmpty else {
            return nil
        }

        let filename = url.lastPathComponent.removingPercentEncoding ?? url.lastPathComponent
        guard filename.lowercased().hasSuffix(".mercan") else {
            return nil
        }

        let stem = (filename as NSString).deletingPathExtension
        let modelName = stem.replacingOccurrences(of: "-", with: " ")
        return (modelName: modelName, filename: filename)
    }

    private var modelInfo: (modelName: String, filename: String)? {
        Self.extractModelInfo(from: inputLink.trimmingCharacters(in: .whitespacesAndNewlines))
    }

    private func fileURL(_ filename: String) -> URL {
        llamaState.getDocumentsDirectory().appendingPathComponent(filename)
    }

    var body: some View {
        HStack(alignment: .center, spacing: 8) {
            TextField("Mercan model URL (.mercan)", text: $inputLink)
                .textFieldStyle(.plain)
                .textInputAutocapitalization(.never)
                .autocorrectionDisabled()

            if let info = modelInfo {
                let progress = llamaState.activeDownloads[info.filename]
                let isDownloaded = FileManager.default.fileExists(atPath: fileURL(info.filename).path)

                if let progress {
                    Button {
                        llamaState.cancelDownload(filename: info.filename)
                    } label: {
                        ZStack {
                            Circle()
                                .stroke(Color(.systemGray4), lineWidth: 2)
                            Circle()
                                .trim(from: 0, to: progress)
                                .stroke(Color.primary, lineWidth: 2)
                                .rotationEffect(.degrees(-90))
                            Text("\(Int(progress * 100))")
                                .font(.system(size: 10, weight: .semibold))
                                .foregroundColor(.secondary)
                        }
                        .frame(width: 28, height: 28)
                    }
                } else if isDownloaded {
                    Button("Load") {
                        Task {
                            try? await llamaState.loadModel(modelUrl: fileURL(info.filename))
                        }
                    }
                    .font(.subheadline)
                    .fontWeight(.medium)
                } else {
                    Button {
                        llamaState.startDownload(
                            modelName: info.modelName,
                            modelUrl: inputLink.trimmingCharacters(in: .whitespacesAndNewlines),
                            filename: info.filename
                        )
                    } label: {
                        Image(systemName: "arrow.down")
                            .font(.system(size: 14, weight: .semibold))
                            .foregroundColor(.black)
                            .frame(width: 28, height: 28)
                            .background(Color.white)
                            .clipShape(Circle())
                    }
                }
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(Color(.systemGray5))
        .cornerRadius(12)
    }
}
