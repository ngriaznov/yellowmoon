// AppleAuto.swift: Apple Auto's bridge to Apple's on-device model (docs/19-apple-auto.md,
// sections 1, 4 and 11).
//
// Four C functions, declared in ym_apple.h and called from crates/ym-apple/src/bridge.rs:
// ym_apple_auto_availability, ym_apple_auto_model, ym_apple_auto_os and ym_apple_auto_ask.
//
// build.rs compiles this file only for Apple silicon with the macOS 27 SDK or later, for macOS 14
// and later (Swift 5 language mode). Every use of Foundation Models sits behind
// `@available(macOS 27.0, *)` or `#available(macOS 27.0, *)`, and the binaries link the framework
// weakly, so they still start on macOS 14 to 26.
//
// The one model this file uses is SystemLanguageModel.default, which runs on this Mac: no other
// model, no tools, nothing that reaches the network (hard rule 7). A test in
// crates/ym-apple/src/source_tests.rs reads this file and checks it against docs/19: the
// instructions, the prompt, the answer's fields, guides and cases, the steps and the codes.

import CoreGraphics
import Darwin
import Dispatch
import Foundation
import FoundationModels
import ImageIO

/// The version of the instructions and the answer's schema (docs/19 section 4). It must equal
/// `ym_core::apple_auto::PROMPT_VERSION`; a test (src/source_tests.rs) compares them. Change both whenever the
/// instructions, the prompt, the options, or the answer's types, fields, guides or cases change.
let promptVersion = 1

/// The instructions of every session (docs/19 section 4, prompt version 1).
let instructionsText = "You help develop raw photos. You see a photo developed with neutral automatic settings. Suggest the small corrections a careful photographer would make so that it looks its best as a natural, finished picture. Keep what the photographer clearly meant: a night scene stays dark, a bright snowy scene stays bright, a silhouette stays a silhouette, warm candlelight stays warm. Choose keep when the photo already looks right."

/// The words of the prompt, followed by the picture.
let promptText = "Look at this photo and choose the corrections."

/// The codes of ym_apple_auto_availability (docs/19 section 11).
private enum AvailabilityCode {
    static let available: Int32 = 0
    static let macosTooOld: Int32 = 3
    static let deviceNotEligible: Int32 = 4
    static let appleIntelligenceOff: Int32 = 5
    static let modelNotReady: Int32 = 6
    static let unavailable: Int32 = 7
    static let noImageInput: Int32 = 8
}

/// The codes of ym_apple_auto_ask (docs/19 section 11).
private enum AskCode {
    static let answered: Int32 = 0
    static let unavailable: Int32 = 1
    static let rateLimited: Int32 = 2
    static let timeout: Int32 = 3
    static let refused: Int32 = 4
    static let guardrail: Int32 = 5
    static let unsupportedLanguage: Int32 = 6
    static let contextTooLarge: Int32 = 7
    static let assetsUnavailable: Int32 = 8
    static let concurrentRequests: Int32 = 9
    static let badAnswer: Int32 = 10
    static let other: Int32 = 11
}

/// Where ym_apple_auto_ask writes the fields of YmAppleAnswer (ym_apple.h).
private enum AnswerLayout {
    static let exposure = 0
    static let white = 4
    static let contrast = 8
    static let saturation = 12
    static let scene = 16
    static let sceneBytes = 128
}

// MARK: - The answer (docs/19 section 4: fields generated in this order)

@available(macOS 27.0, *)
@Generable
struct Answer {
    @Guide(description: "What the photo shows and how it is lit, in at most eight words.")
    var scene: String

    @Guide(description: "Brightness of the main subject. Change it only when the subject or the whole photo is clearly too dark or too bright.")
    var exposure: Exposure

    @Guide(description: "Colour of the light. Warmer to remove a cold, bluish cast or to bring back golden light; cooler to remove an unwanted yellow or orange cast.")
    var white: White

    @Guide(description: "Difference between light and dark tones. More for a flat or hazy photo, less for harsh light with deep shadows.")
    var contrast: Contrast

    @Guide(description: "Strength of the colours. More for dull colours, less for garish colours or skin that looks too red.")
    var saturation: Saturation
}

@available(macOS 27.0, *)
@Generable
enum Exposure {
    case muchDarker, darker, slightlyDarker, keep, slightlyBrighter, brighter, muchBrighter
}

@available(macOS 27.0, *)
@Generable
enum White {
    case muchCooler, cooler, keep, warmer, muchWarmer
}

@available(macOS 27.0, *)
@Generable
enum Contrast {
    case muchLess, less, keep, more, muchMore
}

@available(macOS 27.0, *)
@Generable
enum Saturation {
    case muchLess, less, keep, more, muchMore
}

@available(macOS 27.0, *)
extension Exposure {
    /// The step: muchDarker -3 ... muchBrighter +3.
    var step: Int32 {
        switch self {
        case .muchDarker: return -3
        case .darker: return -2
        case .slightlyDarker: return -1
        case .keep: return 0
        case .slightlyBrighter: return 1
        case .brighter: return 2
        case .muchBrighter: return 3
        }
    }
}

@available(macOS 27.0, *)
extension White {
    /// The step: muchCooler -2 ... muchWarmer +2.
    var step: Int32 {
        switch self {
        case .muchCooler: return -2
        case .cooler: return -1
        case .keep: return 0
        case .warmer: return 1
        case .muchWarmer: return 2
        }
    }
}

@available(macOS 27.0, *)
extension Contrast {
    /// The step: muchLess -2 ... muchMore +2.
    var step: Int32 {
        switch self {
        case .muchLess: return -2
        case .less: return -1
        case .keep: return 0
        case .more: return 1
        case .muchMore: return 2
        }
    }
}

@available(macOS 27.0, *)
extension Saturation {
    /// The step: muchLess -2 ... muchMore +2.
    var step: Int32 {
        switch self {
        case .muchLess: return -2
        case .less: return -1
        case .keep: return 0
        case .more: return 1
        case .muchMore: return 2
        }
    }
}

// MARK: - The C functions

/// 0 when Apple Auto can run, else the reason code (docs/19 section 1, conditions 3 to 5).
@_cdecl("ym_apple_auto_availability")
public func ymAppleAutoAvailability() -> Int32 {
    guard #available(macOS 27.0, *) else { return AvailabilityCode.macosTooOld }
    return currentAvailability()
}

/// The model variant's display name, for reports and `ym:AppleAuto`.
@_cdecl("ym_apple_auto_model")
public func ymAppleAutoModel(_ buffer: UnsafeMutablePointer<CChar>?, _ length: Int) -> Int32 {
    let raw = UnsafeMutableRawPointer(buffer)
    guard #available(macOS 27.0, *) else {
        writeCString("", to: raw, capacity: length)
        return 1
    }
    return writeCString(SystemLanguageModel.default.variant.displayName, to: raw, capacity: length)
}

/// The macOS version and build, e.g. `27.0.1 (27A404)`.
@_cdecl("ym_apple_auto_os")
public func ymAppleAutoOs(_ buffer: UnsafeMutablePointer<CChar>?, _ length: Int) -> Int32 {
    writeCString(operatingSystem(), to: UnsafeMutableRawPointer(buffer), capacity: length)
}

/// Asks the model about one picture and waits for the answer, an error or the timeout.
@_cdecl("ym_apple_auto_ask")
public func ymAppleAutoAsk(
    _ rgb: UnsafePointer<UInt8>?,
    _ width: UInt32,
    _ height: UInt32,
    _ timeoutMs: UInt32,
    _ out: UnsafeMutableRawPointer?,
    _ message: UnsafeMutablePointer<CChar>?,
    _ messageLength: Int
) -> Int32 {
    let messageBuffer = UnsafeMutableRawPointer(message)
    writeCString("", to: messageBuffer, capacity: messageLength)
    guard let rgb, let out, width > 0, height > 0 else {
        writeCString("The picture or the answer is missing.", to: messageBuffer, capacity: messageLength)
        return AskCode.other
    }
    clearAnswer(out)
    let (pixels, overflow) = Int(width).multipliedReportingOverflow(by: Int(height))
    guard !overflow, pixels <= Int.max / 4 else {
        writeCString("The picture is too large.", to: messageBuffer, capacity: messageLength)
        return AskCode.other
    }
    guard #available(macOS 27.0, *) else {
        writeCString("Apple Auto needs macOS 27 or later.", to: messageBuffer, capacity: messageLength)
        return AskCode.unavailable
    }
    let availability = currentAvailability()
    guard availability == AvailabilityCode.available else {
        writeCString(
            "Apple Intelligence is not available (availability code \(availability)).",
            to: messageBuffer, capacity: messageLength)
        return AskCode.unavailable
    }

    // Copied now: the caller's bytes are valid only during this call. RGB becomes RGBX (32 bits per
    // pixel, the fourth byte unused), a layout Core Graphics supports.
    var rgbx = [UInt8](repeating: 255, count: pixels * 4)
    for index in 0..<pixels {
        rgbx[4 * index] = rgb[3 * index]
        rgbx[4 * index + 1] = rgb[3 * index + 1]
        rgbx[4 * index + 2] = rgb[3 * index + 2]
    }

    switch askWithTimeout(rgbx: rgbx, width: Int(width), height: Int(height), timeoutMs: timeoutMs) {
    case let .answer(exposure, white, contrast, saturation, scene):
        out.storeBytes(of: exposure, toByteOffset: AnswerLayout.exposure, as: Int32.self)
        out.storeBytes(of: white, toByteOffset: AnswerLayout.white, as: Int32.self)
        out.storeBytes(of: contrast, toByteOffset: AnswerLayout.contrast, as: Int32.self)
        out.storeBytes(of: saturation, toByteOffset: AnswerLayout.saturation, as: Int32.self)
        writeCString(scene, to: out + AnswerLayout.scene, capacity: AnswerLayout.sceneBytes)
        return AskCode.answered
    case let .failure(code, text):
        writeCString(text, to: messageBuffer, capacity: messageLength)
        return code
    }
}

// MARK: - Availability

/// Conditions 4 and 5 of docs/19 section 1, read from the model's properties (no model call).
@available(macOS 27.0, *)
private func currentAvailability() -> Int32 {
    let model = SystemLanguageModel.default
    switch model.availability {
    case .available:
        return model.capabilities.contains(.vision)
            ? AvailabilityCode.available : AvailabilityCode.noImageInput
    case .unavailable(.deviceNotEligible):
        return AvailabilityCode.deviceNotEligible
    case .unavailable(.appleIntelligenceNotEnabled):
        return AvailabilityCode.appleIntelligenceOff
    case .unavailable(.modelNotReady):
        return AvailabilityCode.modelNotReady
    default:
        return AvailabilityCode.unavailable
    }
}

// MARK: - The request

/// What a request ended with. Plain values only, so it can cross from the task to the caller.
private enum Outcome {
    case answer(Int32, Int32, Int32, Int32, String)
    case failure(Int32, String)
}

/// The outcome of one request, shared by the task that sets it and the thread that waits for it.
private final class OutcomeBox: @unchecked Sendable {
    private let lock = NSLock()
    private var outcome: Outcome?

    /// Keeps the first outcome.
    func set(_ value: Outcome) {
        lock.lock()
        if outcome == nil {
            outcome = value
        }
        lock.unlock()
    }

    func get() -> Outcome? {
        lock.lock()
        defer { lock.unlock() }
        return outcome
    }
}

/// Runs the request in a detached task and blocks this thread until it ends or the time is up;
/// then the task is cancelled. Only this thread writes the caller's buffers, so nothing is written
/// to them after ym_apple_auto_ask returns.
@available(macOS 27.0, *)
private func askWithTimeout(rgbx: [UInt8], width: Int, height: Int, timeoutMs: UInt32) -> Outcome {
    let box = OutcomeBox()
    let done = DispatchSemaphore(value: 0)
    let task = Task.detached {
        let outcome = await askModel(rgbx: rgbx, width: width, height: height)
        box.set(outcome)
        done.signal()
    }
    let deadline: DispatchTime
    if timeoutMs == 0 {
        deadline = DispatchTime.distantFuture
    } else {
        deadline = DispatchTime.now() + DispatchTimeInterval.milliseconds(Int(timeoutMs))
    }
    if done.wait(timeout: deadline) == .timedOut {
        task.cancel()
        return .failure(AskCode.timeout, "No answer within \(timeoutMs) ms.")
    }
    return box.get() ?? .failure(AskCode.other, "The request ended without an outcome.")
}

/// One request: a new session with the instructions, the prompt with the picture, greedy sampling.
@available(macOS 27.0, *)
private func askModel(rgbx: [UInt8], width: Int, height: Int) async -> Outcome {
    guard let image = makeImage(rgbx: rgbx, width: width, height: height) else {
        return .failure(AskCode.other, "Core Graphics could not make the picture.")
    }
    // No model argument: this initialiser's default model is `SystemLanguageModel.default`. With
    // an explicit one, the macOS 27 SDK finds the call ambiguous with its generic initialiser for
    // any language model, which has no default model and so is never chosen here.
    let session = LanguageModelSession(tools: [], instructions: { instructionsText })
    let options = GenerationOptions(samplingMode: .greedy, temperature: nil, maximumResponseTokens: 160)
    do {
        let response = try await session.respond(
            generating: Answer.self, includeSchemaInPrompt: true, options: options
        ) {
            promptText
            Attachment(image, orientation: .up)
        }
        let answer = response.content
        return .answer(
            answer.exposure.step, answer.white.step, answer.contrast.step,
            answer.saturation.step, answer.scene)
    } catch {
        return .failure(askCode(for: error), describe(error))
    }
}

/// The ask code of an error (docs/19 section 11).
@available(macOS 27.0, *)
private func askCode(for error: Error) -> Int32 {
    if error is CancellationError {
        return AskCode.timeout
    }
    if let error = error as? LanguageModelError {
        switch error {
        case .rateLimited: return AskCode.rateLimited
        case .timeout: return AskCode.timeout
        case .refusal: return AskCode.refused
        case .guardrailViolation: return AskCode.guardrail
        case .unsupportedLanguageOrLocale: return AskCode.unsupportedLanguage
        case .contextSizeExceeded: return AskCode.contextTooLarge
        case .unsupportedGenerationGuide: return AskCode.badAnswer
        case .unsupportedCapability: return AskCode.unavailable
        default: return AskCode.other
        }
    }
    if let error = error as? SystemLanguageModel.Error {
        switch error {
        case .assetsUnavailable: return AskCode.assetsUnavailable
        default: return AskCode.other
        }
    }
    if let error = error as? LanguageModelSession.Error {
        switch error {
        case .concurrentRequests: return AskCode.concurrentRequests
        default: return AskCode.other
        }
    }
    // The errors of macOS 26, deprecated in 27 but possibly still thrown.
    if let error = error as? LanguageModelSession.GenerationError {
        switch error {
        case .assetsUnavailable: return AskCode.assetsUnavailable
        case .decodingFailure: return AskCode.badAnswer
        case .exceededContextWindowSize: return AskCode.contextTooLarge
        case .guardrailViolation: return AskCode.guardrail
        case .rateLimited: return AskCode.rateLimited
        case .refusal: return AskCode.refused
        case .concurrentRequests: return AskCode.concurrentRequests
        case .unsupportedGuide: return AskCode.badAnswer
        case .unsupportedLanguageOrLocale: return AskCode.unsupportedLanguage
        default: return AskCode.other
        }
    }
    return AskCode.other
}

/// Apple's message for an error.
private func describe(_ error: Error) -> String {
    let localized = error.localizedDescription
    let detail = String(describing: error)
    if localized.isEmpty {
        return detail
    }
    return localized == detail ? localized : "\(localized) (\(detail))"
}

// MARK: - Helpers

/// An sRGB picture from RGBX bytes (8 bits per component, the fourth byte skipped).
private func makeImage(rgbx: [UInt8], width: Int, height: Int) -> CGImage? {
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

/// Sets the answer's steps to 0 and its scene to "".
private func clearAnswer(_ out: UnsafeMutableRawPointer) {
    for offset in [AnswerLayout.exposure, AnswerLayout.white, AnswerLayout.contrast, AnswerLayout.saturation] {
        out.storeBytes(of: Int32(0), toByteOffset: offset, as: Int32.self)
    }
    writeCString("", to: out + AnswerLayout.scene, capacity: AnswerLayout.sceneBytes)
}

/// The macOS version as major.minor (plus .patch when it is not 0) and the build, e.g.
/// `27.0.1 (27A404)`.
private func operatingSystem() -> String {
    let version = ProcessInfo.processInfo.operatingSystemVersion
    var text = "\(version.majorVersion).\(version.minorVersion)"
    if version.patchVersion > 0 {
        text += ".\(version.patchVersion)"
    }
    if let build = osBuild() {
        text += " (\(build))"
    }
    return text
}

/// The macOS build (`kern.osversion`), e.g. `27A404`.
private func osBuild() -> String? {
    var size = 0
    guard sysctlbyname("kern.osversion", nil, &size, nil, 0) == 0, size > 0 else {
        return nil
    }
    var bytes = [UInt8](repeating: 0, count: size)
    guard sysctlbyname("kern.osversion", &bytes, &size, nil, 0) == 0 else {
        return nil
    }
    let build = String(decoding: bytes.prefix { $0 != 0 }, as: UTF8.self)
    return build.isEmpty ? nil : build
}

/// Writes `text` as UTF-8 and a NUL into `capacity` bytes at `buffer`, cut at a character boundary
/// when it does not fit. 0 on success, 1 when there is no buffer or no room for the NUL.
@discardableResult
private func writeCString(_ text: String, to buffer: UnsafeMutableRawPointer?, capacity: Int) -> Int32 {
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
