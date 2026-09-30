import Foundation
import AVFoundation
import Vision
import CoreImage

// Read caption candidates from the lower part of an existing local video.
let args = CommandLine.arguments
guard args.count == 5 || args.count == 6 else {
    fatalError("usage: read_video_lyrics VIDEO LANGUAGE STEP_SECONDS_OR_@TIMES OUTPUT_JSON [enhance]")
}
let enhance = args.count == 6 && args[5] == "enhance"
let imageContext = CIContext()
let asset = AVURLAsset(url: URL(fileURLWithPath: args[1]))
let generator = AVAssetImageGenerator(asset: asset)
generator.appliesPreferredTrackTransform = true
generator.maximumSize = CGSize(width: 1280, height: 960)
generator.requestedTimeToleranceBefore = .zero
generator.requestedTimeToleranceAfter = .zero
let duration = CMTimeGetSeconds(asset.duration)
let step = Double(args[3]) ?? 0
guard duration.isFinite && duration > 0 && (step > 0 || args[3].hasPrefix("@")) else {
    fatalError("Video is not readable by AVFoundation; use a supported MP4 input")
}
let times: [Double]
if args[3].hasPrefix("@") {
    let data = try Data(contentsOf: URL(fileURLWithPath: String(args[3].dropFirst())))
    times = try JSONDecoder().decode([Double].self, from: data)
} else {
    times = Array(stride(from: 0.0, to: duration, by: step))
}
let request = VNRecognizeTextRequest()
request.recognitionLevel = .accurate
request.recognitionLanguages = [args[2]]
request.usesLanguageCorrection = true
request.minimumTextHeight = 0.012
request.regionOfInterest = CGRect(x: 0, y: 0, width: 1, height: 0.25)
var records: [[String: Any]] = []
let started = Date()
for time in times {
    try autoreleasepool {
        var actual = CMTime.zero
        var image = try generator.copyCGImage(at: CMTime(seconds: time, preferredTimescale: 60000), actualTime: &actual)
        if enhance {
            let adjusted = CIImage(cgImage: image)
                .applyingFilter("CIColorControls", parameters: [kCIInputSaturationKey: 0.0, kCIInputContrastKey: 2.0])
                .applyingFilter("CIUnsharpMask", parameters: [kCIInputRadiusKey: 1.5, kCIInputIntensityKey: 1.0])
            image = imageContext.createCGImage(adjusted, from: adjusted.extent)!
        }
        try VNImageRequestHandler(cgImage: image).perform([request])
        let texts: [[String: Any]] = (request.results ?? []).compactMap { observation in
            guard let candidate = observation.topCandidates(1).first else { return nil }
            return ["text": candidate.string, "confidence": candidate.confidence,
                    "x": observation.boundingBox.minX, "y": observation.boundingBox.minY,
                    "width": observation.boundingBox.width, "height": observation.boundingBox.height]
        }
        records.append(["requested_time": time, "time": CMTimeGetSeconds(actual), "texts": texts])
    }
    if records.count % 50 == 0 {
        print("OCR \(records.count) frames; \(String(format: "%.1f", time)) / \(String(format: "%.1f", duration)) s")
        fflush(stdout)
    }
}
let result: [String: Any] = ["video": args[1], "language": args[2], "step_seconds": step,
                           "enhanced": enhance,
                           "duration_seconds": duration, "seconds": Date().timeIntervalSince(started),
                           "region_bottom_fraction": 0.25, "frames": records]
let data = try JSONSerialization.data(withJSONObject: result, options: [.prettyPrinted, .sortedKeys])
try data.write(to: URL(fileURLWithPath: args[4]), options: .atomic)
print("Saved \(records.count) frames in \(Date().timeIntervalSince(started)) seconds")
