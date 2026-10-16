import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

// The real-shell smoke driver (`npm run e2e:shell`, `e2e/real/driver.js`) checks that the filmstrip's thumbnails are addresses on the
// `ym-frame` scheme. The shell builds those addresses (`src-tauri/src/viewport.rs`: `scheme_url`, `new_launch_id`) and puts the launch
// id between the host and the file, `ym-frame://localhost/<launch>/thumb-<photo>-<version>.png`, with `<launch>` the start time in hex
// milliseconds, a hyphen and a count. The driver runs in a webview and cannot be unit tested, so this reads its pattern out of its source
// and holds it to the shell's address shapes. (The Rust test `frame_addresses_use_the_scheme_and_the_launch_id` pins the other side.)

const driver = readFileSync(resolve(import.meta.dirname, "../../e2e/real/driver.js"), "utf8");

/** The driver's `THUMBNAIL_ADDRESS` pattern, as a RegExp. */
function thumbnailAddress(): RegExp {
  const literal = /const THUMBNAIL_ADDRESS = \/(.+)\/;/.exec(driver);
  if (!literal) throw new Error("e2e/real/driver.js has no `const THUMBNAIL_ADDRESS = /…/;`");
  return new RegExp(literal[1] as string);
}

describe("the thumbnail address the real-shell smoke driver accepts", () => {
  const pattern = thumbnailAddress();

  it.each([
    ["macOS and Linux", "ym-frame://localhost/18f3a1b2c3d-0/thumb-3-1.png"],
    ["Windows", "http://ym-frame.localhost/18f3a1b2c3d-a/thumb-3-1.png"],
    ["a later photo, version and launch of the same run", "ym-frame://localhost/18f3a1b2c3d-1f/thumb-120-34.png"],
  ])("takes what the shell serves (%s)", (_name, address) => {
    expect(pattern.test(address)).toBe(true);
  });

  it.each([
    ["no launch id (the shell answers 404 to it)", "ym-frame://localhost/thumb-3-1.png"],
    ["a launch id with a path in it", "ym-frame://localhost/18f3a1b2c3d-0/extra/thumb-3-1.png"],
    ["a launch id that is not a hex time and a count", "ym-frame://localhost/launch/thumb-3-1.png"],
    ["a frame, not a thumbnail", "ym-frame://localhost/18f3a1b2c3d-0/frame-12.png"],
    ["a look tile, not a thumbnail", "ym-frame://localhost/18f3a1b2c3d-0/look-3-4.png"],
    ["another scheme", "https://localhost/18f3a1b2c3d-0/thumb-3-1.png"],
    ["bytes of the event", "data:image/png;base64,iVBORw0KGgo="],
  ])("refuses %s", (_name, address) => {
    expect(pattern.test(address)).toBe(false);
  });
});
