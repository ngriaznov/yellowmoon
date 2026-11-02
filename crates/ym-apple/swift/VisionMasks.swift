// VisionMasks.swift: Subject and People masks from Apple's Vision framework (docs/23-nodes.md sections 3.10 and
// 4.5, docs/algorithms/06-masks.md section 5.3).
//
// Two C functions, declared in ym_apple.h and called from crates/ym-apple/src/vision_bridge.rs:
//
// * ym_apple_mask_probe(kind): runs the request on a 64 x 64 picture and says whether this Mac can run it.
// * ym_apple_mask_make(kind, rgb, width, height, out, ...): runs the request on a picture and writes the union of
//   its instances (`allInstances`), at the picture's size, into the caller's f32 buffer.
//
// kind 0 is Subject (VNGenerateForegroundInstanceMaskRequest) and kind 1 is People
// (VNGeneratePersonInstanceMaskRequest); both need macOS 14. No Swift error crosses the C interface: every failure is
// a code and, for ym_apple_mask_make, Apple's message in the caller's buffer.
//
// build.rs compiles this file for macOS 14 and later (Swift 5 language mode) for both architectures of the universal
// binary, whenever the SDK has the requests (macOS 14 SDK or later), separately from Apple Auto's bridge
// (AppleAuto.swift): Vision exists on every macOS the app supports, so it is linked normally. Every use of the
// requests sits behind `#available(macOS 14.0, *)`.
//
// Vision is Apple's on-device framework: it sends nothing anywhere (hard rule 7). The same picture on the same Mac
// gives the same mask; another macOS version may answer differently. A test in crates/ym-apple/src/source_tests.rs
// reads this file and checks it against docs/23 and the Rust side.

import CoreGraphics
import CoreVideo
import Foundation
import Vision

/// The requests, as ym_apple_mask_probe and ym_apple_mask_make take them (docs/23 section 4.5).
private enum MaskKindCode {
    static let subject: Int32 = 0
    static let people: Int32 = 1
}

/// The codes of ym_apple_mask_probe.
private enum ProbeCode {
    /// The request ran on the probe picture.
    static let supported: Int32 = 0
    /// macOS is older than 14.
    static let macosTooOld: Int32 = 1
    /// The request failed on the probe picture (a virtual Mac, a Mac that cannot run it), or the kind is unknown.
    static let notSupported: Int32 = 2
}

/// The codes of ym_apple_mask_make.
private enum MakeCode {
    /// The union mask is in `out`.
    static let made: Int32 = 0
    /// Vision found no instance: `out` is all 0.
    static let nothingFound: Int32 = 1
    /// macOS is older than 14.
    static let macosTooOld: Int32 = 2
    /// The request could not run.
    static let notSupported: Int32 = 3
    /// The arguments are wrong (a missing buffer, an empty picture, an unknown kind, a buffer too small).
    static let badInput: Int32 = 4
    /// The request ran and its mask could not be read.
    static let failed: Int32 = 5
}

/// The side of the probe picture in pixels (docs/23 section 3.10: "64 x 64").
private let probeSize = 64

// MARK: - The C functions

/// Whether this Mac can run a request: 0 supported, 1 macOS too old, 2 not supported.
@_cdecl("ym_apple_mask_probe")
public func ymAppleMaskProbe(_ kind: Int32) -> Int32 {
    guard kind == MaskKindCode.subject || kind == MaskKindCode.people else {
        return ProbeCode.notSupported
    }
    guard #available(macOS 14.0, *) else {
        return ProbeCode.macosTooOld
    }
    return autoreleasepool { () -> Int32 in
        guard let image = probePicture() else {
            return ProbeCode.notSupported
        }
        let (code, _) = runRequest(kind: kind, image: image, width: probeSize, height: probeSize, into: nil)
        if code == MakeCode.made || code == MakeCode.nothingFound {
            return ProbeCode.supported
        }
        return ProbeCode.notSupported
    }
}

/// Runs a request on a picture (`width` x `height` pixels of 8-bit sRGB, 3 bytes per pixel, rows top to bottom, no
/// padding) and writes the union of its instances into `out` (`outLength` floats, at least `width` x `height`), in
/// [0, 1], rows top to bottom. Returns a code of MakeCode; `message` (`messageLength` bytes, may be NULL) gets Apple's
/// text for a failure, and "" otherwise.
@_cdecl("ym_apple_mask_make")
public func ymAppleMaskMake(
    _ kind: Int32,
    _ rgb: UnsafePointer<UInt8>?,
    _ width: UInt32,
    _ height: UInt32,
    _ out: UnsafeMutablePointer<Float>?,
    _ outLength: Int,
    _ message: UnsafeMutablePointer<CChar>?,
    _ messageLength: Int
) -> Int32 {
    let messageBuffer = UnsafeMutableRawPointer(message)
    writeVisionMessage("", to: messageBuffer, capacity: messageLength)
    guard kind == MaskKindCode.subject || kind == MaskKindCode.people else {
        writeVisionMessage("The kind of mask is not known.", to: messageBuffer, capacity: messageLength)
        return MakeCode.badInput
    }
    guard let rgb, let out, width > 0, height > 0 else {
        writeVisionMessage("The picture or the mask is missing.", to: messageBuffer, capacity: messageLength)
        return MakeCode.badInput
    }
    let (pixels, overflow) = Int(width).multipliedReportingOverflow(by: Int(height))
    guard !overflow, pixels <= Int.max / 4, outLength >= pixels else {
        writeVisionMessage("The picture is too large, or the mask buffer too small.", to: messageBuffer, capacity: messageLength)
        return MakeCode.badInput
    }
    guard #available(macOS 14.0, *) else {
        writeVisionMessage("Subject and People masks need macOS 14 or later.", to: messageBuffer, capacity: messageLength)
        return MakeCode.macosTooOld
    }
    return autoreleasepool { () -> Int32 in
        // Copied now: the caller's bytes are valid only during this call. RGB becomes RGBX (32 bits per pixel, the
        // fourth byte unused), a layout Core Graphics supports.
        var rgbx = [UInt8](repeating: 255, count: pixels * 4)
        for index in 0..<pixels {
            rgbx[4 * index] = rgb[3 * index]
            rgbx[4 * index + 1] = rgb[3 * index + 1]
            rgbx[4 * index + 2] = rgb[3 * index + 2]
        }
        guard let image = makeVisionImage(rgbx: rgbx, width: Int(width), height: Int(height)) else {
            writeVisionMessage("The picture could not be made.", to: messageBuffer, capacity: messageLength)
            return MakeCode.failed
        }
        let (code, text) = runRequest(kind: kind, image: image, width: Int(width), height: Int(height), into: out)
        if code == MakeCode.nothingFound {
            for index in 0..<pixels {
                out[index] = 0
            }
        }
        writeVisionMessage(text, to: messageBuffer, capacity: messageLength)
        return code
    }
}

// MARK: - The request

/// Runs the request of `kind` on `image` and, when `out` is not NULL, writes the union of the instances into it at
/// `width` x `height`. Returns a code of MakeCode and Apple's text for a failure.
@available(macOS 14.0, *)
private func runRequest(
    kind: Int32,
    image: CGImage,
    width: Int,
    height: Int,
    into out: UnsafeMutablePointer<Float>?
) -> (Int32, String) {
    let handler = VNImageRequestHandler(cgImage: image, options: [:])
    let result: VNInstanceMaskObservation?
    do {
        if kind == MaskKindCode.people {
            let request = VNGeneratePersonInstanceMaskRequest()
            try handler.perform([request])
            result = request.results?.first
        } else {
            let request = VNGenerateForegroundInstanceMaskRequest()
            try handler.perform([request])
            result = request.results?.first
        }
    } catch {
        return (MakeCode.notSupported, describeVisionError(error))
    }
    // No instance at all: the photo has no subject, or no person.
    guard let observation = result, !observation.allInstances.isEmpty else {
        return (MakeCode.nothingFound, "")
    }
    guard let out else {
        // The probe: the request ran, and there is nothing to copy.
        return (MakeCode.made, "")
    }
    let buffer: CVPixelBuffer
    do {
        // The union of all the instances, scaled to the picture's size.
        buffer = try observation.generateScaledMaskForImage(
            forInstances: observation.allInstances, from: handler)
    } catch {
        return (MakeCode.failed, describeVisionError(error))
    }
    return copyMask(buffer, width: width, height: height, into: out)
}

/// Copies Vision's mask (one 32-bit float or 8-bit component per pixel) into `out` as floats in [0, 1] at
/// `width` x `height`. The scaled mask is the picture's size, so this is a plain copy; a mask of another size is
/// resampled bilinearly (half-pixel centres, edges held) rather than refused.
private func copyMask(
    _ buffer: CVPixelBuffer,
    width: Int,
    height: Int,
    into out: UnsafeMutablePointer<Float>
) -> (Int32, String) {
    guard CVPixelBufferLockBaseAddress(buffer, .readOnly) == kCVReturnSuccess else {
        return (MakeCode.failed, "The mask could not be read.")
    }
    defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
    let sourceWidth = CVPixelBufferGetWidth(buffer)
    let sourceHeight = CVPixelBufferGetHeight(buffer)
    let rowBytes = CVPixelBufferGetBytesPerRow(buffer)
    let format = UInt32(CVPixelBufferGetPixelFormatType(buffer))
    let isFloat = format == UInt32(kCVPixelFormatType_OneComponent32Float)
    let isByte = format == UInt32(kCVPixelFormatType_OneComponent8)
    guard let base = CVPixelBufferGetBaseAddress(buffer), sourceWidth > 0, sourceHeight > 0, isFloat || isByte else {
        return (MakeCode.failed, "The mask is in a format this version cannot read (\(format)).")
    }

    /// The mask's value at a pixel, in [0, 1].
    func value(_ x: Int, _ y: Int) -> Float {
        let row = base.advanced(by: y * rowBytes)
        let raw: Float
        if isFloat {
            raw = row.assumingMemoryBound(to: Float.self)[x]
        } else {
            raw = Float(row.assumingMemoryBound(to: UInt8.self)[x]) / 255
        }
        return raw.isNaN ? 0 : min(1, max(0, raw))
    }

    if sourceWidth == width && sourceHeight == height {
        for y in 0..<height {
            for x in 0..<width {
                out[y * width + x] = value(x, y)
            }
        }
        return (MakeCode.made, "")
    }
    let scaleX = Float(sourceWidth) / Float(width)
    let scaleY = Float(sourceHeight) / Float(height)
    for y in 0..<height {
        let fy = (Float(y) + 0.5) * scaleY - 0.5
        let y0 = max(0, min(sourceHeight - 1, Int(fy.rounded(.down))))
        let y1 = min(sourceHeight - 1, y0 + 1)
        let ty = max(0, min(1, fy - Float(y0)))
        for x in 0..<width {
            let fx = (Float(x) + 0.5) * scaleX - 0.5
            let x0 = max(0, min(sourceWidth - 1, Int(fx.rounded(.down))))
            let x1 = min(sourceWidth - 1, x0 + 1)
            let tx = max(0, min(1, fx - Float(x0)))
            let top = value(x0, y0) * (1 - tx) + value(x1, y0) * tx
            let bottom = value(x0, y1) * (1 - tx) + value(x1, y1) * tx
            out[y * width + x] = top * (1 - ty) + bottom * ty
        }
    }
    return (MakeCode.made, "")
}

// MARK: - Helpers

/// The probe picture: a bright disc on a dark ground, `probeSize` pixels square, so that a request has something it
/// may find. Finding nothing in it is an answer too: the probe asks whether the request runs.
private func probePicture() -> CGImage? {
    var rgbx = [UInt8](repeating: 255, count: probeSize * probeSize * 4)
    let centre = Double(probeSize - 1) / 2
    for y in 0..<probeSize {
        for x in 0..<probeSize {
            let dx = Double(x) - centre
            let dy = Double(y) - centre
            let inside = (dx * dx + dy * dy).squareRoot() < 18
            let index = 4 * (y * probeSize + x)
            rgbx[index] = inside ? 230 : 25
            rgbx[index + 1] = inside ? 200 : 30
            rgbx[index + 2] = inside ? 170 : 40
        }
    }
    return makeVisionImage(rgbx: rgbx, width: probeSize, height: probeSize)
}

/// An sRGB picture from RGBX bytes (8 bits per component, the fourth byte skipped).
private func makeVisionImage(rgbx: [UInt8], width: Int, height: Int) -> CGImage? {
    guard let space = CGColorSpace(name: CGColorSpace.sRGB),
          let provider = CGDataProvider(data: Data(rgbx) as CFData)
    else {
        return nil
    }
    return CGImage(
        width: width,
        height: height,
        bitsPerComponent: 8,
        bitsPerPixel: 32,
        bytesPerRow: width * 4,
        space: space,
        bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
        provider: provider,
        decode: nil,
        shouldInterpolate: false,
        intent: .defaultIntent)
}

/// Apple's message for an error.
private func describeVisionError(_ error: Error) -> String {
    let text = error.localizedDescription
    return text.isEmpty ? String(describing: error) : text
}

/// Writes `text` as UTF-8 and a NUL into `capacity` bytes at `buffer`, cut at a character boundary when it does not
/// fit. 0 on success, 1 when there is no buffer or no room for the NUL.
@discardableResult
private func writeVisionMessage(_ text: String, to buffer: UnsafeMutableRawPointer?, capacity: Int) -> Int32 {
    guard let buffer, capacity > 0 else {
        return 1
    }
    let bytes = Array(text.utf8)
    var count = min(bytes.count, capacity - 1)
    if count < bytes.count {
        // Back up over UTF-8 continuation bytes (0b10xxxxxx) so no character is cut in half.
        while count > 0 && (bytes[count] & 0xC0) == 0x80 {
            count -= 1
        }
    }
    for index in 0..<count {
        buffer.storeBytes(of: bytes[index], toByteOffset: index, as: UInt8.self)
    }
    buffer.storeBytes(of: UInt8(0), toByteOffset: count, as: UInt8.self)
    return 0
}
