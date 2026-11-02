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

  it("W is Pick white on any tab and in any mode (a second W ends it), and ⌘W stays Close session", () => {
    expect(act(key("W", { shiftKey: true }))).toBe("pick-white");
    for (const tab of ["look", "lens", "light", "color", "detail", "curve", "raw"] as const) expect(act(key("w"), { tab }), tab).toBe("pick-white");
    expect(act(key("w"), { mode: "pick-white" })).toBe("pick-white");
    expect(act(key("w"), { mode: "pick-color" })).toBe("pick-white");
    expect(act(key("w"), { mode: "crop" })).toBe("pick-white");
    expect(act(cmd("w"))).toBe("close-session");
    expect(act(key("w", { ctrlKey: true }))).toBeNull();
    expect(act(key("w", { altKey: true }))).toBeNull();
  });

  it("W is one press, one change, and not a letter typed into a menu, a list or a dialog", () => {
    expect(act({ ...key("w"), repeat: false })).toBe("pick-white");
    expect(act({ ...key("w"), repeat: true })).toBeNull();
    for (const role of ["menu", "listbox", "dialog"]) {
      const box = document.createElement("div");
      box.setAttribute("role", role);
      const item = document.createElement("div");
      item.setAttribute("role", role === "menu" ? "menuitem" : "option");
      box.append(item);
      expect(act(key("w", {}, item)), role).toBeNull();
    }
    // A button with focus (the toggle itself) or a slider does not use W.
    expect(act(key("w", {}, document.createElement("button")))).toBe("pick-white");
    const slider = document.createElement("input");
    slider.type = "range";
    expect(act(key("w", {}, slider))).toBe("pick-white");
    expect(act({ ...key("w"), defaultPrevented: true })).toBeNull();
  });

  it("types normally in a text box", () => {
    const input = document.createElement("input");
    input.type = "search";
    for (const letter of ["p", "z", "c", "w", "f", "1", "\\"]) expect(act(key(letter, {}, input), { tab: "look" })).toBeNull();
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

  it("F toggles full screen, with or without Shift (or Caps Lock), as the other letters do", () => {
    expect(act(key("f"))).toBe("full-screen");
    expect(act(key("F", { shiftKey: true }))).toBe("full-screen");
    expect(act(key("F"))).toBe("full-screen");
    // On the Look tab and while picking white or a colour it is still F: the mode stays on in full screen.
    expect(act(key("f"), { tab: "look" })).toBe("full-screen");
    expect(act(key("f"), { mode: "pick-white" })).toBe("full-screen");
    expect(act(key("f"), { mode: "pick-color" })).toBe("full-screen");
  });

  it("F is not full screen with ⌘, ⌃ or ⌥ (⌃⌘F is the system's), in a text box, on a key held down, while cropping, or in a menu", () => {
    expect(act(cmd("f"))).toBeNull();
    expect(act(key("f", { metaKey: true, ctrlKey: true }))).toBeNull();
    expect(act(key("f", { ctrlKey: true }))).toBeNull();
    expect(act(key("ƒ", { altKey: true, code: "KeyF" }))).toBeNull();
    for (const type of ["text", "search"]) {
      const input = document.createElement("input");
      input.type = type;
      expect(act(key("f", {}, input))).toBeNull();
    }
    const area = document.createElement("textarea");
    expect(act(key("f", {}, area))).toBeNull();
    expect(act({ ...key("f"), repeat: true })).toBeNull();
    expect(act({ ...key("f"), repeat: false })).toBe("full-screen");
    expect(act(key("f"), { mode: "crop" })).toBeNull();
    expect(act({ ...key("f"), defaultPrevented: true })).toBeNull();
    // A letter typed in an open menu or list is its type-ahead.
    for (const role of ["menu", "listbox", "dialog"]) {
      const box = document.createElement("div");
      box.setAttribute("role", role);
      const item = document.createElement("div");
      item.setAttribute("role", role === "menu" ? "menuitem" : "option");
      box.append(item);
      expect(act(key("f", {}, item))).toBeNull();
    }
    // A slider or a button with focus does not use F: it is still full screen.
    const slider = document.createElement("input");
    slider.type = "range";
    expect(act(key("f", {}, slider))).toBe("full-screen");
    expect(act(key("f", {}, document.createElement("button")))).toBe("full-screen");
  });

  it("N switches between the simple view and the node view, with or without Shift, on any tab and in any mode but crop", () => {
    expect(act(key("n"))).toBe("nodes");
    expect(act(key("N", { shiftKey: true }))).toBe("nodes");
    expect(act(key("N"))).toBe("nodes");
    for (const tab of ["look", "lens", "light", "color", "detail", "curve", "raw"] as const) expect(act(key("n"), { tab }), tab).toBe("nodes");
    expect(act(key("n"), { mode: "pick-white" })).toBe("nodes");
    expect(act(key("n"), { mode: "pick-color" })).toBe("nodes");
    expect(act(key("n"), { nodeView: true })).toBe("nodes");
    // Crop needs its panel, as it does for F.
    expect(act(key("n"), { mode: "crop" })).toBeNull();
  });

  it("N is not the node view with ⌘, ⌃ or ⌥, in a text box, on a key held down, or in a menu, a list or a dialog", () => {
    expect(act(cmd("n"))).toBeNull();
    expect(act(key("n", { ctrlKey: true }))).toBeNull();
    expect(act(key("n", { altKey: true }))).toBeNull();
    for (const type of ["text", "search"]) {
      const input = document.createElement("input");
      input.type = type;
      expect(act(key("n", {}, input))).toBeNull();
    }
    expect(act(key("n", {}, document.createElement("textarea")))).toBeNull();
    expect(act({ ...key("n"), repeat: true })).toBeNull();
    expect(act({ ...key("n"), repeat: false })).toBe("nodes");
    expect(act({ ...key("n"), defaultPrevented: true })).toBeNull();
    for (const role of ["menu", "listbox", "dialog"]) {
      const box = document.createElement("div");
      box.setAttribute("role", role);
      const item = document.createElement("div");
      item.setAttribute("role", role === "menu" ? "menuitem" : "option");
      box.append(item);
      expect(act(key("n", {}, item)), role).toBeNull();
    }
    // A slider or a button with focus does not use N: it is still the node view.
    const slider = document.createElement("input");
    slider.type = "range";
    expect(act(key("n", {}, slider))).toBe("nodes");
    expect(act(key("n", {}, document.createElement("button")))).toBe("nodes");
  });

  it("Delete and Backspace never remove a photo in the node view (the photos are out of sight there): they are the graph's", () => {
    expect(act(key("Delete"), { nodeView: false })).toBe("remove-photo");
    expect(act(key("Delete"), { nodeView: true })).toBe("delete-node");
    expect(act(key("Backspace"), { nodeView: true })).toBe("delete-node");
    // Not held down, not with Shift, and not where the key is the control's own.
    expect(act(key("Delete", { repeat: true }), { nodeView: true })).toBeNull();
    expect(act(key("Delete", { shiftKey: true }), { nodeView: true })).toBeNull();
    const field = document.createElement("input");
    field.type = "text";
    expect(act(key("Delete", {}, field), { nodeView: true })).toBeNull();
    const slider = document.createElement("input");
    slider.type = "range";
    expect(act(key("Backspace", {}, slider), { nodeView: true })).toBeNull();
    const menu = document.createElement("div");
    menu.setAttribute("role", "menu");
    const item = document.createElement("div");
    item.setAttribute("role", "menuitem");
    menu.append(item);
    expect(act(key("Delete", {}, item), { nodeView: true })).toBeNull();
    // Not while a mode (crop, Pick white) has its own use of the key.
    expect(act(key("Delete"), { nodeView: true, mode: "pick-white" })).toBeNull();
    // Outside the photos' own strip, the photo-level keys are as before: P picks, and the arrows switch photos away from the graph.
    expect(act(key("ArrowRight"), { nodeView: true })).toBe("next-photo");
    expect(act(key("ArrowLeft"), { nodeView: true })).toBe("previous-photo");
    expect(act(key("p"), { nodeView: true })).toBe("toggle-pick");
  });

  describe("the graph's keys (docs/23 §7.7)", () => {
    /** An element inside the graph's canvas, a card. */
    const card = (): HTMLElement => {
      const canvas = document.createElement("div");
      canvas.setAttribute("data-graph", "");
      const el = document.createElement("div");
      el.setAttribute("role", "button");
      canvas.append(el);
      return el;
    };

    it("B bypasses the selected node: one press, in the node view only, and not typed into a menu, a list or a text box", () => {
      expect(act(key("b"), { nodeView: true })).toBe("bypass-node");
      expect(act(key("B", { shiftKey: true }), { nodeView: true })).toBe("bypass-node");
      expect(act(key("b"), { nodeView: false })).toBeNull();
      expect(act(key("b"))).toBeNull();
      expect(act(key("b", { repeat: true }), { nodeView: true })).toBeNull();
      expect(act(key("b"), { nodeView: true, mode: "crop" })).toBeNull();
      const field = document.createElement("input");
      field.type = "text";
      expect(act(key("b", {}, field), { nodeView: true })).toBeNull();
      const menu = document.createElement("div");
      menu.setAttribute("role", "menu");
      const item = document.createElement("div");
      menu.append(item);
      expect(act(key("b", {}, item), { nodeView: true })).toBeNull();
      expect(act(cmd("b"), { nodeView: true })).toBeNull();
    });

    it("M shows or hides the selected node's mask: one press, in the node view only, and not typed into a menu, a list or a text box", () => {
      expect(act(key("m"), { nodeView: true })).toBe("mask-view");
      expect(act(key("M", { shiftKey: true }), { nodeView: true })).toBe("mask-view");
      expect(act(key("m"), { nodeView: false })).toBeNull();
      expect(act(key("m"))).toBeNull();
      expect(act(key("m", { repeat: true }), { nodeView: true })).toBeNull();
      expect(act(key("m"), { nodeView: true, mode: "crop" })).toBeNull();
      const field = document.createElement("input");
      field.type = "text";
      expect(act(key("m", {}, field), { nodeView: true })).toBeNull();
      const menu = document.createElement("div");
      menu.setAttribute("role", "menu");
      const item = document.createElement("div");
      menu.append(item);
      expect(act(key("m", {}, item), { nodeView: true })).toBeNull();
      // ⌘M is Minimise.
      expect(act(cmd("m"), { nodeView: true })).toBeNull();
    });

    it("⌥← and ⌥→ move the focused node earlier or later, with a card of the graph focused", () => {
      expect(act(key("ArrowLeft", { altKey: true }, card()), { nodeView: true })).toBe("move-node-earlier");
      expect(act(key("ArrowRight", { altKey: true }, card()), { nodeView: true })).toBe("move-node-later");
      // Not outside the graph, not in the simple view, not in a text box, not with another modifier or held down.
      expect(act(key("ArrowLeft", { altKey: true }), { nodeView: true })).toBeNull();
      expect(act(key("ArrowLeft", { altKey: true }, card()), { nodeView: false })).toBeNull();
      const field = document.createElement("input");
      field.type = "text";
      expect(act(key("ArrowRight", { altKey: true }, field), { nodeView: true })).toBeNull();
      expect(act(key("ArrowRight", { altKey: true, shiftKey: true }, card()), { nodeView: true })).toBeNull();
      expect(act(key("ArrowRight", { altKey: true, metaKey: true }, card()), { nodeView: true })).toBeNull();
      expect(act(key("ArrowRight", { altKey: true, repeat: true }, card()), { nodeView: true })).toBeNull();
    });

    it("the arrows belong to the graph while a card has the focus: they switch no photo there, and switch them everywhere else", () => {
      for (const arrow of ["ArrowLeft", "ArrowRight"]) {
        expect(act(key(arrow, {}, card()), { nodeView: true })).toBeNull();
        expect(act(key(arrow), { nodeView: true })).not.toBeNull();
      }
      expect(act(key("ArrowUp", {}, card()), { nodeView: true })).toBeNull();
      expect(act(key("ArrowDown", {}, card()), { nodeView: true })).toBeNull();
    });

    it("Delete on a card of the graph is the graph's", () => {
      expect(act(key("Delete", {}, card()), { nodeView: true })).toBe("delete-node");
    });
  });

  it("A toggles Auto crop while the Crop panel is open, and is nothing anywhere else", () => {
    expect(act(key("a"), { mode: "crop" })).toBe("auto-crop");
    expect(act(key("A", { shiftKey: true }), { mode: "crop" })).toBe("auto-crop");
    // A slider or a button of the panel with focus does not use A.
    const slider = document.createElement("input");
    slider.type = "range";
    expect(act(key("a", {}, slider), { mode: "crop" })).toBe("auto-crop");
    expect(act(key("a", {}, document.createElement("button")), { mode: "crop" })).toBe("auto-crop");
    // Outside the Crop panel A does nothing (⌘A picks all), and nor does a key held down, a text box or a modifier.
    expect(act(key("a"))).toBeNull();
    expect(act(key("a"), { tab: "look" })).toBeNull();
    expect(act(key("a"), { mode: "pick-white" })).toBeNull();
    expect(act(cmd("a"), { mode: "crop" })).toBe("pick-all");
    expect(act({ ...key("a"), repeat: true }, { mode: "crop" })).toBeNull();
    expect(act(key("a", { altKey: true }), { mode: "crop" })).toBeNull();
    expect(act(key("a", {}, document.createElement("textarea")), { mode: "crop" })).toBeNull();
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
