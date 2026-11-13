import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { announceFacts, announcementFor, CLEAR_AFTER_MS, createAnnouncer, HELD_QUIET_MS } from "./announce";
import type { Announced, AnnounceFacts } from "./announce";
import { shortcutFor } from "./shortcuts";
import type { ShortcutAction } from "./shortcuts";

const facts = (over: Partial<AnnounceFacts> = {}): AnnounceFacts => ({
  look: "Film warm",
  before: false,
  zoom: { kind: "fit" },
  whole: true,
  picked: true,
  pickedCount: 4,
  position: 5,
  count: 12,
  on: true,
  ...over,
});
const say = (action: Announced, over: Partial<AnnounceFacts> = {}) => announcementFor(action, facts(over));

describe("the sentence for each shortcut (docs/11 §Announcements)", () => {
  it("1–9 say the look as the chip names it", () => {
    expect(say("pick-look")).toBe("Look: Film warm.");
    expect(say("pick-look", { look: null })).toBeNull();
  });

  it("\\ says Before or After", () => {
    expect(say("before", { before: true })).toBe("Before.");
    expect(say("before", { before: false })).toBe("After.");
  });

  it("Z and ⌘+ ⌘− say the zoom in percent, or Fit", () => {
    for (const action of ["toggle-zoom", "zoom-in", "zoom-out"] as const) {
      expect(say(action, { zoom: { kind: "percent", value: 100 }, whole: false })).toBe("Zoom 100 %.");
      expect(say(action, { zoom: { kind: "percent", value: 137.4 }, whole: false })).toBe("Zoom 137 %.");
      expect(say(action, { zoom: { kind: "fit" }, whole: true })).toBe("Zoom to fit.");
      // A step no larger than Fit shows the whole photo: it says so, not "50 %".
      expect(say(action, { zoom: { kind: "percent", value: 50 }, whole: true })).toBe("Zoom to fit.");
    }
  });

  it("P says whether the photo is picked and how many are", () => {
    expect(say("toggle-pick", { picked: true, pickedCount: 4 })).toBe("Picked. 4 picked.");
    expect(say("toggle-pick", { picked: false, pickedCount: 3 })).toBe("Not picked. 3 picked.");
    expect(say("toggle-pick", { picked: false, pickedCount: 0 })).toBe("Not picked. None picked.");
  });

  it("⌘A and Clear Picks say the count", () => {
    expect(say("pick-all", { pickedCount: 12 })).toBe("Picked all. 12 picked.");
    expect(say("pick-all", { pickedCount: 0 })).toBeNull();
    expect(say("clear-picks")).toBe("Picks cleared.");
  });

  it("← and → say where the photo is", () => {
    expect(say("next-photo")).toBe("Photo 5 of 12.");
    expect(say("previous-photo", { position: 1 })).toBe("Photo 1 of 12.");
    expect(say("next-photo", { position: null })).toBeNull();
  });

  it("⌘Z and ⇧⌘Z say so only when there was a step to take", () => {
    expect(say("undo")).toBe("Undone.");
    expect(say("undo", { on: false })).toBeNull();
    expect(say("redo")).toBe("Redone.");
    expect(say("redo", { on: false })).toBeNull();
  });

  it("C, Return and A in Crop", () => {
    expect(say("crop")).toBe("Crop. Press Return to finish.");
    expect(say("confirm")).toBe("Crop finished.");
    expect(say("auto-crop", { on: true })).toBe("Auto crop on.");
    expect(say("auto-crop", { on: false })).toBe("Auto crop off.");
  });

  it("N, B and M in the node view", () => {
    expect(say("nodes", { on: true })).toBe("Node view.");
    expect(say("nodes", { on: false })).toBe("Simple view.");
    expect(say("bypass-node", { on: true })).toBe("Node bypassed.");
    expect(say("bypass-node", { on: false })).toBe("Node on.");
    expect(say("mask-view", { on: true })).toBe("Mask shown.");
    expect(say("mask-view", { on: false })).toBe("Mask hidden.");
  });

  it("the shortcuts whose result moves the focus or opens something announced have no sentence", () => {
    const silent: ShortcutAction[] = [
      "open-files", "open-folder", "export", "close-session", "remove-photo", "pick-white", "delete-node", "move-node-earlier", "move-node-later",
      "full-screen", "copy-settings", "paste-settings", "paste-settings-picked", "preferences", "escape",
    ];
    for (const action of silent) expect(say(action), action).toBeNull();
  });

  it("every sentence is plain: British spelling, no exclamation mark, ends in a full stop", () => {
    const all: Announced[] = ["pick-look", "before", "toggle-zoom", "toggle-pick", "pick-all", "clear-picks", "next-photo", "undo", "redo", "crop", "confirm", "auto-crop", "nodes", "bypass-node", "mask-view"];
    for (const action of all) {
      const text = say(action) ?? "";
      expect(text, action).toMatch(/\.$/);
      expect(text, action).not.toMatch(/!|color|gray/i);
    }
  });
});

describe("single letters typed in a text box have no shortcut, so no sentence", () => {
  it("shortcutFor returns null in a text box for every key that has a sentence", () => {
    const input = document.createElement("input");
    for (const k of ["p", "z", "\\", "c", "n", "b", "m", "a", "1", "5", "9", "ArrowLeft", "ArrowRight"]) {
      const event = { key: k, metaKey: false, ctrlKey: false, altKey: false, shiftKey: false, target: input };
      expect(shortcutFor(event, { tab: "look", nodeView: true, mode: k === "a" ? "crop" : null }), k).toBeNull();
    }
  });
});

describe("the facts of the store", () => {
  it("name the look as the chip does, the photo's place and the pick count", () => {
    const state = {
      session: {
        current_index: 1,
        picked: [10, 11],
        photos: [
          { id: 10, picked: true },
          { id: 11, picked: true },
          { id: 12, picked: false },
        ],
      },
      settings: { 11: { values: { output: { look: "film-warm" } } } },
      looks: [{ id: "film-warm", name: "Film warm", library: "yellowmoon" }],
      before: false,
      zoom: { kind: "percent", value: 100 },
    } as never;
    const f = announceFacts(state, 24, true);
    expect(f).toMatchObject({ look: "Film warm", picked: true, pickedCount: 2, position: 2, count: 3, whole: false, before: false });
  });
});

describe("when a sentence is said", () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  it("a key pressed says its sentence at once", () => {
    const said: string[] = [];
    const a = createAnnouncer((t) => said.push(t));
    a.say("Before.");
    expect(said).toEqual(["Before."]);
  });

  it("the same sentence twice in a row is said twice", () => {
    const said: string[] = [];
    const a = createAnnouncer((t) => said.push(t));
    a.say("Before.");
    a.say("Before.");
    expect(said).toEqual(["Before.", "Before."]);
  });

  it("ten repeats of a held key give one sentence, at the end", () => {
    const said: string[] = [];
    const a = createAnnouncer((t) => said.push(t));
    for (let n = 1; n <= 10; n += 1) {
      a.say(`Photo ${n} of 12.`, true);
      vi.advanceTimersByTime(50);
    }
    expect(said).toEqual([]);
    vi.advanceTimersByTime(HELD_QUIET_MS);
    expect(said).toEqual(["Photo 10 of 12."]);
    vi.advanceTimersByTime(HELD_QUIET_MS * 5);
    expect(said).toEqual(["Photo 10 of 12."]);
  });

  it("a key pressed after a held one says its own sentence and drops the waiting one", () => {
    const said: string[] = [];
    const a = createAnnouncer((t) => said.push(t));
    a.say("Photo 3 of 12.", true);
    a.say("Picked. 1 picked.");
    vi.advanceTimersByTime(HELD_QUIET_MS * 2);
    expect(said).toEqual(["Picked. 1 picked."]);
  });

  it("a null sentence says nothing", () => {
    const said: string[] = [];
    const a = createAnnouncer((t) => said.push(t));
    a.say(null);
    a.say(null, true);
    vi.advanceTimersByTime(HELD_QUIET_MS * 2);
    expect(said).toEqual([]);
  });

  it("the region is emptied after a while, and by clear", () => {
    const said: string[] = [];
    const a = createAnnouncer((t) => said.push(t));
    a.say("After.");
    vi.advanceTimersByTime(CLEAR_AFTER_MS);
    expect(said).toEqual(["After.", ""]);
    a.say("Before.", true);
    a.clear();
    vi.advanceTimersByTime(HELD_QUIET_MS * 2);
    expect(said).toEqual(["After.", "", ""]);
  });
});
