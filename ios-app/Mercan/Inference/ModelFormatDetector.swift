import Foundation

enum ModelFormat {
    case mercan
    case unknown
}

class ModelFormatDetector {
    static func detectFormat(url: URL) -> ModelFormat {
        url.pathExtension.lowercased() == "mercan" ? .mercan : .unknown
    }

    static func detectFormat(path: String) -> ModelFormat {
        detectFormat(url: URL(fileURLWithPath: path))
    }
}
