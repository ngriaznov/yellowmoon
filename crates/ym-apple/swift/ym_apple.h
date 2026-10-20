/*
 * ym_apple.h: the C interface of Apple Auto's Swift bridge (docs/19-apple-auto.md, section 11).
 *
 * Implemented in AppleAuto.swift with @_cdecl and declared again, with the same types, in Rust
 * (crates/ym-apple/src/bridge.rs, the struct in src/abi.rs). A test reads this file and the Rust
 * declarations to keep the names in step. The Swift side takes `out` as a `void *` and writes the
 * fields at the offsets of the struct below (0, 4, ... 44, the scene at 48 and the plan at 176).
 *
 * Strings are UTF-8 and always NUL-terminated; a string longer than its buffer is cut at a
 * character boundary. A buffer length of 0, or a NULL buffer, makes a function return non-zero.
 *
 * Threads: every function may be called from any thread. ym_apple_auto_ask blocks the calling
 * thread (the session's Apple Auto worker, never the main thread) until the answer, an error or the
 * timeout; it reads `rgb` and writes `out` and `message` only before it returns.
 */

#ifndef YM_APPLE_H
#define YM_APPLE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* The bytes of YmAppleAnswer.scene, the NUL included. */
#define YM_APPLE_SCENE_BYTES 128

/* The bytes of YmAppleAnswer.plan, the NUL included. */
#define YM_APPLE_PLAN_BYTES 256

/* The code of a field the model left empty: nothing is set for that control (docs/19 section 5). */
#define YM_APPLE_UNSET INT32_MIN

/*
 * One answer (docs/19 section 4): the codes of the twelve values, each YM_APPLE_UNSET when the
 * model left it empty, and the model's few words about the scene and its plan.
 * exposure: muchDarker -3 ... keep 0 ... muchBrighter +3.
 * highlights, shadows: muchDarker -2 ... muchBrighter +2. blacks: muchDeeper -2 ... muchLifted +2.
 * contrast, saturation: muchLess -2 ... muchMore +2. white (the model's warmth): muchCooler -2 ...
 * muchWarmer +2. tint: muchGreener -2 ... muchMoreMagenta +2.
 * shadow_color: none 0, teal 1, blue 2, green 3, amber 4, magenta 5.
 * highlight_color: none 0, warm 1, gold 2, cool 3, pink 4.
 * shadow_color_strength, highlight_color_strength: subtle 1, medium 2, strong 3.
 */
typedef struct YmAppleAnswer {
    int32_t exposure;
    int32_t highlights;
    int32_t shadows;
    int32_t blacks;
    int32_t contrast;
    int32_t white;
    int32_t tint;
    int32_t saturation;
    int32_t shadow_color;
    int32_t shadow_color_strength;
    int32_t highlight_color;
    int32_t highlight_color_strength;
    char scene[YM_APPLE_SCENE_BYTES];
    char plan[YM_APPLE_PLAN_BYTES];
} YmAppleAnswer;

#ifndef __cplusplus
_Static_assert(sizeof(YmAppleAnswer) == 432, "YmAppleAnswer is 432 bytes");
_Static_assert(offsetof(YmAppleAnswer, scene) == 48, "the scene starts at byte 48");
_Static_assert(offsetof(YmAppleAnswer, plan) == 176, "the plan starts at byte 176");
#endif

/*
 * Whether Apple Auto can run (docs/19 section 1, conditions 3 to 5). Cheap: property reads, no
 * model call. 0 available; else the reason: 3 macos_too_old, 4 device_not_eligible,
 * 5 apple_intelligence_off, 6 model_not_ready, 7 unavailable (any other reason Apple gives),
 * 8 no_image_input.
 */
int32_t ym_apple_auto_availability(void);

/*
 * The model variant's display name ("AFM 3 Core") into `buf` (`len` bytes). 0 on success; non-zero
 * before macOS 27 (then `buf` holds "").
 */
int32_t ym_apple_auto_model(char *buf, size_t len);

/* The macOS version and build ("27.0.1 (27A404)") into `buf` (`len` bytes). 0 on success. */
int32_t ym_apple_auto_os(char *buf, size_t len);

/*
 * Asks Apple's on-device model about one picture: `width` x `height` pixels of 8-bit sRGB, 3 bytes
 * per pixel, rows top to bottom, no padding (docs/19 section 3), with `instructions` (UTF-8,
 * NUL-terminated, not empty: the default or the person's own, docs/19 sections 2 and 4). One new
 * session, no tools, greedy sampling (docs/19 section 4). Blocks until the answer, an error or
 * `timeout_ms` milliseconds (0: no limit); on a timeout the request is cancelled.
 *
 * 0 on success, with the answer in `*out` (fields the model left empty are YM_APPLE_UNSET; when the
 * generation stopped part way, the fields filled until then). Else an error code, with Apple's
 * message in `message` (`message_len` bytes; may be NULL): 1 unavailable, 2 rate limited,
 * 3 timeout, 4 refused, 5 guardrail, 6 unsupported language, 7 context too large, 8 assets
 * unavailable, 9 concurrent requests, 10 bad answer, 11 other. No retries: the caller retries.
 */
int32_t ym_apple_auto_ask(const uint8_t *rgb, uint32_t width, uint32_t height,
                          const char *instructions, uint32_t timeout_ms, YmAppleAnswer *out,
                          char *message, size_t message_len);

#ifdef __cplusplus
}
#endif

#endif /* YM_APPLE_H */
