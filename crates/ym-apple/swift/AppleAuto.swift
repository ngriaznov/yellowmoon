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
// crates/ym-apple/src/source_tests.rs reads this file and checks it against docs/19: the prompt,
// the answer's fields, guides and cases, the codes and the layout. The instructions are not here:
// the core passes them on every call (the default, ym_core::apple_auto::DEFAULT_INSTRUCTIONS, or
// the person's own from Preferences, docs/19 section 2).

import CoreGraphics
import Darwin
import Dispatch
import Foundation
import FoundationModels
import ImageIO

/// The version of the prompt and the answer's schema (docs/19 section 4). It must equal
/// `ym_core::apple_auto::PROMPT_VERSION`; a test (src/source_tests.rs) compares them. Change both whenever the
/// prompt, the options, the answer's types, fields, guides or cases, or the core's default instructions change.
let promptVersion = 3

/// The words of the prompt, followed by the picture (docs/19 section 4, prompt version 3).
let promptText = "Grade this photo."

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

/// Where ym_apple_auto_ask writes the fields of YmAppleAnswer (ym_apple.h), and the code of a field
/// the model left empty (YM_APPLE_UNSET).
private enum AnswerLayout {
    static let exposure = 0
    static let highlights = 4
    static let shadows = 8
    static let blacks = 12
    static let contrast = 16
    static let white = 20
    static let tint = 24
    static let saturation = 28
    static let shadowColor = 32
    static let shadowColorStrength = 36
    static let highlightColor = 40
    static let highlightColorStrength = 44
    static let scene = 48
    static let sceneBytes = 128
    static let plan = 176
    static let planBytes = 256
    static let unset = Int32.min
}

// MARK: - The answer (docs/19 section 4: fields generated in this order)

// Every value is optional: a field the model leaves empty sets nothing (docs/19 section 5).

@available(macOS 27.0, *)
@Generable
struct Answer {
    @Guide(description: "What the photo shows and how it is lit, in at most eight words.")
    var scene: String

    @Guide(description: "The look you will give this photo and how you will get it, in at most fifteen words.")
    var plan: String

    @Guide(description: "Brightness of the main subject.")
    var exposure: Exposure?

    @Guide(description: "The brightest tones: darker to bring back sky and bright detail, brighter for an airy, luminous look.")
    var highlights: Highlights?

    @Guide(description: "The dark tones: brighter to open up detail in the shadows, darker for depth and mood.")
    var shadows: Shadows?

    @Guide(description: "The black level: deeper for rich, solid blacks, lifted for a soft, faded, filmic black.")
    var blacks: Blacks?

    @Guide(description: "Overall contrast: more for punch and depth, less for a soft, gentle look.")
    var contrast: Contrast?

    @Guide(description: "Colour of the light: warmer for golden, cosy or sunny light, cooler for blue hour, night or a cold mood; also to remove an unwanted cast.")
    var warmth: Warmth?

    @Guide(description: "Green against magenta: more magenta to remove a green cast from fluorescent light or foliage, greener to remove a magenta cast.")
    var tint: Tint?

    @Guide(description: "Strength of the colours: more for vivid, rich colour, less for a muted, restrained or moody look; keep skin natural.")
    var saturation: Saturation?

    @Guide(description: "A colour for the shadows, as colourists do: teal or blue for a cinematic or cool mood, amber for warmth, none for a neutral grade.")
    var shadowColour: ShadowColour?

    @Guide(description: "How strongly the shadows are coloured.")
    var shadowColourStrength: Strength?

    @Guide(description: "A colour for the highlights: warm or gold for sunlit, golden or nostalgic light (with teal shadows this gives the classic cinematic look), cool for a cold, clean feel, none for neutral.")
    var highlightColour: HighlightColour?

    @Guide(description: "How strongly the highlights are coloured.")
    var highlightColourStrength: Strength?
}

@available(macOS 27.0, *)
@Generable
enum Exposure {
    case muchDarker, darker, slightlyDarker, keep, slightlyBrighter, brighter, muchBrighter
}

@available(macOS 27.0, *)
@Generable
enum Highlights {
    case muchDarker, darker, keep, brighter, muchBrighter
}

@available(macOS 27.0, *)
@Generable
enum Shadows {
    case muchDarker, darker, keep, brighter, muchBrighter
}

@available(macOS 27.0, *)
@Generable
enum Blacks {
    case muchDeeper, deeper, keep, lifted, muchLifted
}

@available(macOS 27.0, *)
@Generable
enum Contrast {
    case muchLess, less, keep, more, muchMore
}

@available(macOS 27.0, *)
@Generable
enum Warmth {
    case muchCooler, cooler, keep, warmer, muchWarmer
}

@available(macOS 27.0, *)
@Generable
enum Tint {
    case muchGreener, greener, keep, moreMagenta, muchMoreMagenta
}

@available(macOS 27.0, *)
@Generable
enum Saturation {
    case muchLess, less, keep, more, muchMore
}

@available(macOS 27.0, *)
@Generable
enum ShadowColour {
    case none, teal, blue, green, amber, magenta
}

@available(macOS 27.0, *)
@Generable
enum HighlightColour {
    case none, warm, gold, cool, pink
}

@available(macOS 27.0, *)
@Generable
enum Strength {
    case subtle, medium, strong
}

/// The code a case is written as (docs/19 section 4: the steps, the colours and the strengths).
@available(macOS 27.0, *)
private protocol AnswerCode {
    var code: Int32 { get }
}

@available(macOS 27.0, *)
extension Exposure: AnswerCode {
    /// The step: muchDarker -3 ... muchBrighter +3.
    var code: Int32 {
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
extension Highlights: AnswerCode {
    /// The step: muchDarker -2 ... muchBrighter +2.
    var code: Int32 {
        switch self {
        case .muchDarker: return -2
        case .darker: return -1
        case .keep: return 0
        case .brighter: return 1
        case .muchBrighter: return 2
        }
    }
}

@available(macOS 27.0, *)
extension Shadows: AnswerCode {
    /// The step: muchDarker -2 ... muchBrighter +2.
    var code: Int32 {
        switch self {
        case .muchDarker: return -2
        case .darker: return -1
        case .keep: return 0
        case .brighter: return 1
        case .muchBrighter: return 2
        }
    }
}

@available(macOS 27.0, *)
extension Blacks: AnswerCode {
    /// The step: muchDeeper -2 ... muchLifted +2.
    var code: Int32 {
        switch self {
        case .muchDeeper: return -2
        case .deeper: return -1
        case .keep: return 0
        case .lifted: return 1
        case .muchLifted: return 2
        }
    }
}

@available(macOS 27.0, *)
extension Contrast: AnswerCode {
    /// The step: muchLess -2 ... muchMore +2.
    var code: Int32 {
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
extension Warmth: AnswerCode {
    /// The step: muchCooler -2 ... muchWarmer +2.
    var code: Int32 {
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
extension Tint: AnswerCode {
    /// The step: muchGreener -2 ... muchMoreMagenta +2.
    var code: Int32 {
        switch self {
        case .muchGreener: return -2
        case .greener: return -1
        case .keep: return 0
        case .moreMagenta: return 1
        case .muchMoreMagenta: return 2
        }
    }
}

@available(macOS 27.0, *)
extension Saturation: AnswerCode {
    /// The step: muchLess -2 ... muchMore +2.
    var code: Int32 {
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
extension ShadowColour: AnswerCode {
    /// The colour: none 0, teal 1, blue 2, green 3, amber 4, magenta 5.
    var code: Int32 {
        switch self {
        case .none: return 0
        case .teal: return 1
        case .blue: return 2
        case .green: return 3
        case .amber: return 4
        case .magenta: return 5
        }
    }
}

@available(macOS 27.0, *)
extension HighlightColour: AnswerCode {
    /// The colour: none 0, warm 1, gold 2, cool 3, pink 4.
    var code: Int32 {
        switch self {
        case .none: return 0
        case .warm: return 1
        case .gold: return 2
        case .cool: return 3
        case .pink: return 4
        }
    }
}

@available(macOS 27.0, *)
extension Strength: AnswerCode {
    /// The strength: subtle 1, medium 2, strong 3.
    var code: Int32 {
        switch self {
        case .subtle: return 1
        case .medium: return 2
        case .strong: return 3
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

/// Asks the model about one picture with the given instructions (UTF-8, NUL-terminated) and waits
/// for the answer, an error or the timeout.
@_cdecl("ym_apple_auto_ask")
public func ymAppleAutoAsk(
    _ rgb: UnsafePointer<UInt8>?,
    _ width: UInt32,
    _ height: UInt32,
    _ instructions: UnsafePointer<CChar>?,
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
    // Copied now, like the picture: the caller's text is valid only during this call.
    let instructionsText = instructions.map { String(cString: $0) } ?? ""
    guard !instructionsText.isEmpty else {
        writeCString("The instructions are missing.", to: messageBuffer, capacity: messageLength)
        return AskCode.other
    }
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

    let outcome = askWithTimeout(
        rgbx: rgbx, width: Int(width), height: Int(height), instructions: instructionsText,
        timeoutMs: timeoutMs)
    switch outcome {
    case let .answer(filled):
        writeAnswer(filled, to: out)
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

/// The fields of an answer as their codes, `AnswerLayout.unset` for a field the model left empty,
/// in the order of YmAppleAnswer. Plain values only, so it can cross from the task to the caller.
private struct Filled {
    var exposure = AnswerLayout.unset
    var highlights = AnswerLayout.unset
    var shadows = AnswerLayout.unset
    var blacks = AnswerLayout.unset
    var contrast = AnswerLayout.unset
    var white = AnswerLayout.unset
    var tint = AnswerLayout.unset
    var saturation = AnswerLayout.unset
    var shadowColor = AnswerLayout.unset
    var shadowColorStrength = AnswerLayout.unset
    var highlightColor = AnswerLayout.unset
    var highlightColorStrength = AnswerLayout.unset
    var scene = ""
    var plan = ""

    /// Whether the model filled at least one value (the scene and the plan are words, not values).
    var anyValue: Bool {
        [
            exposure, highlights, shadows, blacks, contrast, white, tint, saturation, shadowColor,
            shadowColorStrength, highlightColor, highlightColorStrength,
        ].contains { $0 != AnswerLayout.unset }
    }
}

/// What a request ended with.
private enum Outcome {
    case answer(Filled)
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
/// to them after ym_apple_auto_ask returns. A request that runs out of time is a timeout even when
/// the model had filled some fields: where it stopped depends on the time, and the same photo must
/// give the same answer (docs/19 section 7).
@available(macOS 27.0, *)
private func askWithTimeout(
    rgbx: [UInt8], width: Int, height: Int, instructions: String, timeoutMs: UInt32
) -> Outcome {
    let box = OutcomeBox()
    let done = DispatchSemaphore(value: 0)
    let task = Task.detached {
        let outcome = await askModel(rgbx: rgbx, width: width, height: height, instructions: instructions)
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
/// The answer is streamed and the last snapshot kept, so that when the generation stops part way
/// (the token limit, or another error after some fields were produced) the fields already filled
/// are used and nothing is set for the rest (docs/19 section 4). An answer that filled no value is
/// a failure.
@available(macOS 27.0, *)
private func askModel(rgbx: [UInt8], width: Int, height: Int, instructions: String) async -> Outcome {
    guard let image = makeImage(rgbx: rgbx, width: width, height: height) else {
        return .failure(AskCode.other, "Core Graphics could not make the picture.")
    }
    // No model argument: this initialiser's default model is `SystemLanguageModel.default`. With
    // an explicit one, the macOS 27 SDK finds the call ambiguous with its generic initialiser for
    // any language model, which has no default model and so is never chosen here.
    let session = LanguageModelSession(tools: [], instructions: { instructions })
    let options = GenerationOptions(samplingMode: .greedy, temperature: nil, maximumResponseTokens: 320)
    var last: GeneratedContent?
    do {
        let stream = session.streamResponse(
            generating: Answer.self, includeSchemaInPrompt: true, options: options
        ) {
            promptText
            Attachment(image, orientation: .up)
        }
        for try await snapshot in stream {
            last = snapshot.rawContent
        }
    } catch {
        if !(error is CancellationError), let last {
            let filled = filledFields(of: last)
            if filled.anyValue {
                return .answer(filled)
            }
        }
        return .failure(askCode(for: error), describe(error))
    }
    guard let last else {
        return .failure(AskCode.badAnswer, "The model gave no answer.")
    }
    let filled = filledFields(of: last)
    guard filled.anyValue else {
        return .failure(AskCode.badAnswer, "The model filled no value.")
    }
    return .answer(filled)
}

/// The fields of an answer, complete or cut short: each value that is there and is one of its
/// cases, `AnswerLayout.unset` for the others (absent, empty, or cut in the middle of a word).
@available(macOS 27.0, *)
private func filledFields(of content: GeneratedContent) -> Filled {
    var filled = Filled()
    filled.scene = (try? content.value(String.self, forProperty: "scene")) ?? ""
    filled.plan = (try? content.value(String.self, forProperty: "plan")) ?? ""
    filled.exposure = code(of: Exposure.self, "exposure", in: content)
    filled.highlights = code(of: Highlights.self, "highlights", in: content)
    filled.shadows = code(of: Shadows.self, "shadows", in: content)
    filled.blacks = code(of: Blacks.self, "blacks", in: content)
    filled.contrast = code(of: Contrast.self, "contrast", in: content)
    filled.white = code(of: Warmth.self, "warmth", in: content)
    filled.tint = code(of: Tint.self, "tint", in: content)
    filled.saturation = code(of: Saturation.self, "saturation", in: content)
    filled.shadowColor = code(of: ShadowColour.self, "shadowColour", in: content)
    filled.shadowColorStrength = code(of: Strength.self, "shadowColourStrength", in: content)
    filled.highlightColor = code(of: HighlightColour.self, "highlightColour", in: content)
    filled.highlightColorStrength = code(of: Strength.self, "highlightColourStrength", in: content)
    return filled
}

/// The code of one field of an answer, or `AnswerLayout.unset` when it is not one of its cases.
@available(macOS 27.0, *)
private func code<Value: Generable & AnswerCode>(
    of type: Value.Type, _ property: String, in content: GeneratedContent
) -> Int32 {
    guard let value = try? content.value(type, forProperty: property) else {
        return AnswerLayout.unset
    }
    return value.code
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

/// Sets every field of the answer to `AnswerLayout.unset` and its scene and plan to "".
private func clearAnswer(_ out: UnsafeMutableRawPointer) {
    writeAnswer(Filled(), to: out)
}

/// Writes an answer's fields at their offsets in YmAppleAnswer.
private func writeAnswer(_ filled: Filled, to out: UnsafeMutableRawPointer) {
    let fields: [(Int, Int32)] = [
        (AnswerLayout.exposure, filled.exposure),
        (AnswerLayout.highlights, filled.highlights),
        (AnswerLayout.shadows, filled.shadows),
        (AnswerLayout.blacks, filled.blacks),
        (AnswerLayout.contrast, filled.contrast),
        (AnswerLayout.white, filled.white),
        (AnswerLayout.tint, filled.tint),
        (AnswerLayout.saturation, filled.saturation),
        (AnswerLayout.shadowColor, filled.shadowColor),
        (AnswerLayout.shadowColorStrength, filled.shadowColorStrength),
        (AnswerLayout.highlightColor, filled.highlightColor),
        (AnswerLayout.highlightColorStrength, filled.highlightColorStrength),
    ]
    for (offset, value) in fields {
        out.storeBytes(of: value, toByteOffset: offset, as: Int32.self)
    }
    writeCString(filled.scene, to: out + AnswerLayout.scene, capacity: AnswerLayout.sceneBytes)
    writeCString(filled.plan, to: out + AnswerLayout.plan, capacity: AnswerLayout.planBytes)
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
