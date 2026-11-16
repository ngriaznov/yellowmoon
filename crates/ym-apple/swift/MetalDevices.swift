// MetalDevices.swift: the Metal devices of this Mac and what Metal recommends each may use (docs/26-hardening.md H-17,
// finding P5-2; docs/17-gpu.md section 8).
//
// One C function, declared in ym_apple.h and called from crates/ym-apple/src/metal_bridge.rs:
//
// * ym_apple_metal_devices(out, capacity, count): lists the Metal devices (MTLCopyAllDevices(); on an Intel Mac with an
//   AMD GPU that is more than the default device) and writes, for each, its name, its recommendedMaxWorkingSetSize and
//   three flags (hasUnifiedMemory, isLowPower, isRemovable) into the caller's array of YmAppleMetalDevice.
//
// wgpu does not expose recommendedMaxWorkingSetSize; the core matches the device wgpu opened to one of these by name and
// caps the GPU budget with its limit (crates/ym-core/src/app/metal.rs).
//
// build.rs compiles this file for macOS 14 and later (Swift 5 language mode) for both architectures of the universal
// binary, like VisionMasks.swift: Metal is on every Mac the app runs on, so it is a normal link. Nothing here needs
// `#available`: every property is in macOS 10.15 or earlier. Nothing is sent anywhere (hard rule 7) and no Swift error
// crosses the C interface. A test in crates/ym-apple/src/source_tests.rs reads this file and checks it against the
// header and the Rust side.

import Foundation
import Metal

/// The size of one YmAppleMetalDevice in bytes (ym_apple.h asserts the same).
private let metalDeviceStride = 152

/// The offsets of its fields.
private enum MetalDeviceOffset {
    static let recommendedMaxWorkingSetSize = 0
    static let hasUnifiedMemory = 8
    static let isLowPower = 12
    static let isRemovable = 16
    static let name = 24
}

/// The bytes of the name, the NUL included (YM_APPLE_METAL_NAME_BYTES).
private let metalNameBytes = 128

/// Lists the Metal devices into `out` (`capacity` entries of YmAppleMetalDevice) and sets `*count` to how many were
/// written (the devices Metal lists, at most `capacity`). 0 on success, also with no device; 1 when an argument is
/// missing.
@_cdecl("ym_apple_metal_devices")
public func ymAppleMetalDevices(
    _ out: UnsafeMutableRawPointer?,
    _ capacity: Int,
    _ count: UnsafeMutablePointer<Int>?
) -> Int32 {
    guard let count else {
        return 1
    }
    count.pointee = 0
    guard capacity == 0 || out != nil else {
        return 1
    }
    return autoreleasepool { () -> Int32 in
        let devices = MTLCopyAllDevices()
        let written = min(devices.count, capacity)
        guard written > 0, let out else {
            return 0
        }
        for index in 0..<written {
            let device = devices[index]
            let entry = out.advanced(by: index * metalDeviceStride)
            entry.storeBytes(
                of: UInt64(device.recommendedMaxWorkingSetSize),
                toByteOffset: MetalDeviceOffset.recommendedMaxWorkingSetSize,
                as: UInt64.self
            )
            entry.storeBytes(of: flag(device.hasUnifiedMemory), toByteOffset: MetalDeviceOffset.hasUnifiedMemory, as: UInt32.self)
            entry.storeBytes(of: flag(device.isLowPower), toByteOffset: MetalDeviceOffset.isLowPower, as: UInt32.self)
            entry.storeBytes(of: flag(device.isRemovable), toByteOffset: MetalDeviceOffset.isRemovable, as: UInt32.self)
            entry.storeBytes(of: UInt32(0), toByteOffset: 20, as: UInt32.self)
            writeMetalName(device.name, to: entry.advanced(by: MetalDeviceOffset.name), capacity: metalNameBytes)
        }
        count.pointee = written
        return 0
    }
}

/// 1 for true, 0 for false.
private func flag(_ value: Bool) -> UInt32 {
    return value ? 1 : 0
}

/// Writes `text` as UTF-8 and a NUL into `capacity` bytes at `buffer`, cut at a character boundary when it does not
/// fit.
private func writeMetalName(_ text: String, to buffer: UnsafeMutableRawPointer, capacity: Int) {
    let bytes = Array(text.utf8)
    var length = min(bytes.count, capacity - 1)
    if length < bytes.count {
        // Back up over UTF-8 continuation bytes (0b10xxxxxx) so no character is cut in half.
        while length > 0 && (bytes[length] & 0xC0) == 0x80 {
            length -= 1
        }
    }
    for index in 0..<length {
        buffer.storeBytes(of: bytes[index], toByteOffset: index, as: UInt8.self)
    }
    buffer.storeBytes(of: UInt8(0), toByteOffset: length, as: UInt8.self)
}
