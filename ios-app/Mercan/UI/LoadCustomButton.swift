import SwiftUI
import UniformTypeIdentifiers

struct LoadCustomButton: View {
    @ObservedObject var llamaState: LlamaState
    @State private var showFileImporter = false
    @State private var importError: String?

    private static let mercanType = UTType(filenameExtension: "mercan", conformingTo: .data)!

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Button {
                showFileImporter = true
            } label: {
                Label("Import Mercan Model", systemImage: "square.and.arrow.down")
            }

            if let importError {
                Text(importError)
                    .font(.caption)
                    .foregroundStyle(.red)
            }
        }
        .fileImporter(
            isPresented: $showFileImporter,
            allowedContentTypes: [Self.mercanType],
            allowsMultipleSelection: false
        ) { result in
            switch result {
            case .success(let files):
                guard let source = files.first else { return }
                importError = nil

                Task {
                    do {
                        let destination = llamaState.getDocumentsDirectory()
                            .appendingPathComponent(source.lastPathComponent)

                        let copiedURL = try await Task.detached(priority: .userInitiated) {
                            let gotAccess = source.startAccessingSecurityScopedResource()
                            defer {
                                if gotAccess { source.stopAccessingSecurityScopedResource() }
                            }

                            if FileManager.default.fileExists(atPath: destination.path) {
                                try FileManager.default.removeItem(at: destination)
                            }
                            try FileManager.default.copyItem(at: source, to: destination)
                            return destination
                        }.value

                        try await llamaState.loadModel(modelUrl: copiedURL)
                    } catch {
                        importError = error.localizedDescription
                    }
                }

            case .failure(let error):
                importError = error.localizedDescription
            }
        }
    }
}
