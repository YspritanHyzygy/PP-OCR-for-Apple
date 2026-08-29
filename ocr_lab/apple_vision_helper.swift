import CoreGraphics
import Foundation
import ImageIO
import Vision

struct OutputPoint: Codable {
    let x: Double
    let y: Double
}

struct OutputRegion: Codable {
    let text: String
    let confidence: Float
    let quad: [OutputPoint]
}

enum HelperError: Error {
    case missingPath
    case imageLoadFailed
}

func pixelPoint(_ point: CGPoint, width: Int, height: Int) -> OutputPoint {
    OutputPoint(
        x: Double(point.x) * Double(width),
        y: (1 - Double(point.y)) * Double(height)
    )
}

func recognize(path: String) throws -> [OutputRegion] {
    let url = URL(fileURLWithPath: path)
    guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
          let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
        throw HelperError.imageLoadFailed
    }

    let request = VNRecognizeTextRequest()
    request.recognitionLevel = .accurate
    request.usesLanguageCorrection = true
    request.automaticallyDetectsLanguage = true
    try VNImageRequestHandler(cgImage: image, orientation: .up).perform([request])

    return (request.results ?? []).compactMap { observation -> OutputRegion? in
        guard let candidate = observation.topCandidates(1).first else { return nil }
        let text = candidate.string.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return nil }
        return OutputRegion(
            text: text,
            confidence: candidate.confidence,
            quad: [
                pixelPoint(observation.topLeft, width: image.width, height: image.height),
                pixelPoint(observation.topRight, width: image.width, height: image.height),
                pixelPoint(observation.bottomRight, width: image.width, height: image.height),
                pixelPoint(observation.bottomLeft, width: image.width, height: image.height),
            ]
        )
    }.sorted { lhs, rhs in
        let lhsTop = lhs.quad.map(\.y).min() ?? 0
        let rhsTop = rhs.quad.map(\.y).min() ?? 0
        if lhsTop != rhsTop { return lhsTop < rhsTop }
        return (lhs.quad.map(\.x).min() ?? 0) < (rhs.quad.map(\.x).min() ?? 0)
    }
}

do {
    let paths = Array(CommandLine.arguments.dropFirst())
    guard !paths.isEmpty else { throw HelperError.missingPath }
    let batches = try paths.map(recognize(path:))

    let data = try JSONEncoder().encode(batches)
    FileHandle.standardOutput.write(data)
} catch {
    let message = "Apple Vision helper failed: \(error)\n"
    FileHandle.standardError.write(message.data(using: .utf8)!)
    exit(1)
}
