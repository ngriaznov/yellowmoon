import { describe, expect, it } from "vitest";
import { shortcutFor } from "./shortcuts";
import type { KeyLike, ShortcutContext } from "./shortcuts";

const key = (k: string, mods: Partial<KeyLike> = {}, target: EventTarget | null = document.body): KeyLike => ({
  key: k,
  metaKey: false,
  ctrlKey: false,
  altKey: false,
  shiftKey: false,
  target,
  ...mods,
});
const cmd = (k: string, mods: Partial<KeyLike> = {}) => key(k, { metaKey: true, ...mods });
const act = (event: KeyLike, context?: ShortcutContext) => shortcutFor(event, context)?.action ?? null;

describe("the shortcut table of docs/11", () => {
  it("⌘O and ⇧⌘O open files and a folder", () => {
    expect(act(cmd("o"))).toBe("open-files");
    expect(act(cmd("O", { shiftKey: true }))).toBe("open-folder");
    expect(act(cmd("o", { shiftKey: true }))).toBe("open-folder");
  });

  it("⌘E exports and ⌘W closes the session", () => {
    expect(act(cmd("e"))).toBe("export");
    expect(act(cmd("w"))).toBe("close-session");
  });

  it("⌘Z undoes and ⇧⌘Z redoes", () => {
    expect(act(cmd("z"))).toBe("undo");
    expect(act(cmd("Z", { shiftKey: true }))).toBe("redo");
  });

  it("leaves ⌘Z to a text box that has focus", () => {
    const input = document.createElement("input");
    input.type = "text";
    expect(act(cmd("z", { target: input }))).toBeNull();
  });

  it("⇧⌘C copies the settings, ⇧⌘V pastes them, ⌥⇧⌘V pastes them onto the picked photos", () => {
    expect(act(cmd("C", { shiftKey: true }))).toBe("copy-settings");
    expect(act(cmd("c", { shiftKey: true }))).toBe("copy-settings");
    expect(act(cmd("V", { shiftKey: true }))).toBe("paste-settings");
    expect(act(cmd("v", { shiftKey: true }))).toBe("paste-settings");
    expect(act(cmd("V", { shiftKey: true, altKey: true }))).toBe("paste-settings-picked");
    // ⌥ changes what the key reports (√, ◊): the key's place on the keyboard still names it.
    expect(act(cmd("◊", { shiftKey: true, altKey: true, code: "KeyV" }))).toBe("paste-settings-picked");
    expect(act(cmd("√", { altKey: true, shiftKey: true, code: "KeyV" }))).toBe("paste-settings-picked");
  });

  it("leaves plain ⌘C and ⌘V to the system's Copy and Paste", () => {
    expect(act(cmd("c"))).toBeNull();
    expect(act(cmd("v"))).toBeNull();
    expect(act(cmd("v", { altKey: true }))).toBeNull();
    expect(act(cmd("c", { altKey: true, shiftKey: true }))).toBeNull();
    expect(act(key("c", { ctrlKey: true, shiftKey: true }))).toBeNull();
  });

  it("leaves ⇧⌘C, ⇧⌘V and ⌥⇧⌘V to a text box that has focus", () => {
    const input = document.createElement("input");
    input.type = "text";
    const area = document.createElement("textarea");
    for (const target of [input, area]) {
      expect(act(cmd("C", { shiftKey: true, target }))).toBeNull();
      expect(act(cmd("V", { shiftKey: true, target }))).toBeNull();
      expect(act(cmd("V", { shiftKey: true, altKey: true, code: "KeyV", target }))).toBeNull();
    }
    // A slider is not a text box.
    const range = document.createElement("input");
    range.type = "range";
    expect(act(cmd("C", { shiftKey: true, target: range }))).toBe("copy-settings");
  });

  it("⌘A picks all, ⌘+ and ⌘− zoom, ⌘, opens preferences", () => {
    expect(act(cmd("a"))).toBe("pick-all");
    expect(act(cmd("+", { shiftKey: true }))).toBe("zoom-in");
    expect(act(cmd("="))).toBe("zoom-in");
    expect(act(cmd("-"))).toBe("zoom-out");
    expect(act(cmd(","))).toBe("preferences");
  });

  it("the arrows move between photos", () => {
    expect(act(key("ArrowLeft"))).toBe("previous-photo");
    expect(act(key("ArrowRight"))).toBe("next-photo");
  });

  it("the arrows stay with a slider, a text box, a thumbnail and the crop frame", () => {
    const range = document.createElement("input");
    range.type = "range";
    expect(act(key("ArrowRight", {}, range))).toBeNull();
    const text = document.createElement("input");
    expect(act(key("ArrowRight", {}, text))).toBeNull();
    const thumb = document.createElement("button");
    thumb.dataset.photoId = "3";
    expect(act(key("ArrowRight", {}, thumb))).toBeNull();
    const radio = document.createElement("button");
    radio.setAttribute("role", "radio");
    expect(act(key("ArrowRight", {}, radio))).toBeNull();
    const crop = document.createElement("div");
    crop.className = "ym-crop";
    const handle = document.createElement("button");
    crop.append(handle);
    expect(act(key("ArrowRight", {}, handle))).toBeNull();
  });

  it("P picks, backslash flips Before, Z toggles zoom, C crops, W picks white", () => {
    expect(act(key("p"))).toBe("toggle-pick");
    expect(act(key("P", { shiftKey: true }))).toBe("toggle-pick");
    expect(act(key("\\"))).toBe("before");
    expect(act(key("z"))).toBe("toggle-zoom");
    expect(act(key("c"))).toBe("crop");
    expect(act(key("w"))).toBe("pick-white");
  });

  it("types normally in a text box", () => {
    const input = document.createElement("input");
    input.type = "search";
    for (const letter of ["p", "z", "c", "w", "1", "\\"]) expect(act(key(letter, {}, input), { tab: "look" })).toBeNull();
  });

  it("1 to 9 pick looks only on the Look tab", () => {
    expect(shortcutFor(key("3"), { tab: "look" })).toEqual({ action: "pick-look", look: 3 });
    expect(shortcutFor(key("9"), { tab: "look" })).toEqual({ action: "pick-look", look: 9 });
    expect(act(key("3"), { tab: "light" })).toBeNull();
    expect(act(key("3"))).toBeNull();
    expect(act(key("0"), { tab: "look" })).toBeNull();
  });

  it("does not pick looks while cropping or picking white", () => {
    expect(act(key("3"), { tab: "look", mode: "crop" })).toBeNull();
    expect(act(key("3"), { tab: "look", mode: "pick-white" })).toBeNull();
  });

  it("Escape always counts, Return confirms only in the crop panel", () => {
    expect(act(key("Escape"))).toBe("escape");
    expect(act(key("Enter"))).toBeNull();
    expect(act(key("Enter"), { mode: "crop" })).toBe("confirm");
  });

  it("Return on a button or a radio presses it instead of confirming, except on the crop frame", () => {
    const button = document.createElement("button");
    expect(act(key("Enter", {}, button), { mode: "crop" })).toBeNull();
    const radio = document.createElement("div");
    radio.setAttribute("role", "radio");
    expect(act(key("Enter", {}, radio), { mode: "crop" })).toBeNull();
    const text = document.createElement("input");
    expect(act(key("Enter", {}, text), { mode: "crop" })).toBeNull();
    const crop = document.createElement("div");
    crop.className = "ym-crop";
    const handle = document.createElement("button");
    crop.append(handle);
    expect(act(key("Enter", {}, handle), { mode: "crop" })).toBe("confirm");
  });

  it("Delete and Backspace remove the photo from the filmstrip", () => {
    expect(act(key("Delete"))).toBe("remove-photo");
    expect(act(key("Backspace"))).toBe("remove-photo");
    // Focus on a button or on a filmstrip thumbnail still counts: neither uses the keys.
    const thumb = document.createElement("button");
    thumb.dataset.photoId = "3";
    expect(act(key("Delete", {}, thumb))).toBe("remove-photo");
  });

  it("leaves Delete and Backspace to a text box, a slider, a menu, a dialog and the crop frame", () => {
    for (const k of ["Delete", "Backspace"]) {
      for (const type of ["text", "search", "number", "email"]) {
        const input = document.createElement("input");
        input.type = type;
        expect(act(key(k, {}, input)), `${k} in an input of type ${type}`).toBeNull();
      }
      expect(act(key(k, {}, document.createElement("textarea")))).toBeNull();
      expect(act(key(k, {}, document.createElement("select")))).toBeNull();
      const editable = document.createElement("div");
      editable.contentEditable = "true";
      // jsdom does not compute it.
      Object.defineProperty(editable, "isContentEditable", { value: true });
      expect(act(key(k, {}, editable))).toBeNull();
      const range = document.createElement("input");
      range.type = "range";
      expect(act(key(k, {}, range))).toBeNull();
      // Radix's slider knob, a colour-warp pin, a curve point and a wheel are role=slider.
      const knob = document.createElement("span");
      knob.setAttribute("role", "slider");
      expect(act(key(k, {}, knob))).toBeNull();
      const item = document.createElement("div");
      item.setAttribute("role", "menuitem");
      const menu = document.createElement("div");
      menu.setAttribute("role", "menu");
      menu.append(item);
      expect(act(key(k, {}, item))).toBeNull();
      const field = document.createElement("div");
      const dialog = document.createElement("div");
      dialog.setAttribute("role", "dialog");
      dialog.append(field);
      expect(act(key(k, {}, field))).toBeNull();
      const crop = document.createElement("div");
      crop.className = "ym-crop";
      const handle = document.createElement("button");
      crop.append(handle);
      expect(act(key(k, {}, handle))).toBeNull();
    }
  });

  it("does not remove a photo while cropping or picking, with a modifier, or after a control used the key", () => {
    expect(act(key("Delete"), { mode: "crop" })).toBeNull();
    expect(act(key("Delete"), { mode: "pick-white" })).toBeNull();
    expect(act(key("Backspace"), { mode: "pick-color" })).toBeNull();
    expect(act(key("Delete", { shiftKey: true }))).toBeNull();
    expect(act(cmd("Backspace"))).toBeNull();
    expect(act(key("Delete", { altKey: true }))).toBeNull();
    expect(act(key("Backspace", { ctrlKey: true }))).toBeNull();
    expect(act({ ...key("Delete"), defaultPrevented: true })).toBeNull();
  });

  it("removes one photo when Delete is held down, not one after another", () => {
    expect(act({ ...key("Delete"), repeat: false })).toBe("remove-photo");
    expect(act({ ...key("Delete"), repeat: true })).toBeNull();
    expect(act({ ...key("Backspace"), repeat: true })).toBeNull();
  });

  it("ignores a key a control already used", () => {
    expect(act({ ...key("ArrowLeft"), defaultPrevented: true })).toBeNull();
  });

  it("ignores Control and Option combinations and unknown ⌘ keys", () => {
    expect(act(key("p", { ctrlKey: true }))).toBeNull();
    expect(act(key("p", { altKey: true }))).toBeNull();
    expect(act(cmd("q"))).toBeNull();
    expect(act(cmd("o", { altKey: true }))).toBeNull();
  });
});
