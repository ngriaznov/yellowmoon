import { describe, expect, it } from "vitest";
import { clickKind, installTimeline, RING, Timeline } from "./perfTimeline";
import type { TimelineHost } from "./perfTimeline";

/** A host with a clock and animation frames the test drives. */
function host(gesture = false) {
  let now = 1_000;
  const frames: Array<() => void> = [];
  const sent: string[] = [];
  const h: TimelineHost = {
    now: () => now,
    frame: (callback) => void frames.push(callback),
    send: (json) => void sent.push(json),
    gesture: () => gesture,
  };
  return {
    h,
    sent,
    advance: (ms: number) => void (now += ms),
    /** Runs the frames that are waiting, each at `ms` later. */
    frameAfter: (ms: number) => {
      now += ms;
      const waiting = frames.splice(0);
      for (const callback of waiting) callback();
    },
    reports: () => sent.map((json) => JSON.parse(json) as { page: string; start?: boolean; pointer_moves: number; ticks: Array<Record<string, unknown>>; clicks: Array<Record<string, unknown>> }),
  };
}

describe("the timeline of a slider tick", () => {
  it("stamps a patch from the hand to the second frame after its reply", () => {
    const t = host();
    const timeline = new Timeline(t.h);
    timeline.start();
    timeline.noteInput(990, true);
    timeline.noteInput(995, true);
    const tick = timeline.tick(false);
    t.advance(6);
    tick.replied();
    t.frameAfter(8);
    t.frameAfter(8);
    t.advance(1000);
    timeline.flush();
    const [first, second] = t.reports();
    expect(first?.start).toBe(true);
    expect(first?.ticks).toEqual([]);
    expect(second?.start).toBeUndefined();
    expect(second?.ticks).toEqual([{ n: 1, final: false, t_input: 990, t_sent: 1000, t_reply: 1006, t_painted: 1022 }]);
    timeline.stop();
  });

  it("numbers the patches in the order they are sent, and leaves a patch no hand made without an input", () => {
    const t = host();
    const timeline = new Timeline(t.h);
    timeline.start();
    timeline.tick(false).replied();
    t.advance(500);
    timeline.noteInput(t.h.now() - 10);
    timeline.tick(true).replied();
    t.frameAfter(5);
    t.frameAfter(5);
    t.advance(1000);
    timeline.flush();
    const ticks = t.reports()[1]?.ticks ?? [];
    expect(ticks.map((x) => [x.n, x.final, x.t_input === null])).toEqual([
      [1, false, true],
      [2, true, false],
    ]);
  });

  it("does not take a pointer move from long ago as the input of a patch", () => {
    const t = host();
    const timeline = new Timeline(t.h);
    timeline.noteInput(100, true);
    t.advance(10_000);
    timeline.tick(false);
    t.advance(1000);
    timeline.flush();
    expect(t.reports()[0]?.ticks).toEqual([]);
  });

  it("counts the pointer moves of a held gesture only", () => {
    const held = host(true);
    const a = new Timeline(held.h);
    a.noteInput(1, true);
    a.noteInput(2, true);
    a.flush();
    expect(held.reports()[0]?.pointer_moves).toBe(2);
    const idle = host(false);
    const b = new Timeline(idle.h);
    b.noteInput(1, true);
    b.flush();
    expect(idle.reports()[0]?.pointer_moves).toBe(0);
  });

  it("keeps a patch that is not painted yet for the next report, and sends nothing when nothing happened", () => {
    const t = host();
    const timeline = new Timeline(t.h);
    timeline.start();
    const tick = timeline.tick(false);
    t.advance(1000);
    timeline.flush();
    expect(t.reports()[1]?.ticks).toEqual([]);
    tick.replied();
    t.frameAfter(8);
    t.frameAfter(8);
    t.advance(1000);
    timeline.flush();
    expect(t.reports()[2]?.ticks).toHaveLength(1);
    t.advance(1000);
    timeline.flush();
    expect(t.sent).toHaveLength(3);
  });

  it("keeps at most a ring of patches", () => {
    const t = host();
    const timeline = new Timeline(t.h);
    for (let i = 0; i < RING + 10; i++) timeline.tick(false).replied(false);
    t.advance(1000);
    timeline.flush();
    const ticks = t.reports()[0]?.ticks ?? [];
    expect(ticks).toHaveLength(RING);
    expect(ticks[0]?.n).toBe(11);
  });
});

describe("the timeline of a click", () => {
  it("ends at the second frame after the last change the click made", () => {
    const t = host();
    const timeline = new Timeline(t.h);
    timeline.pointerDown("tab", "Colour", 1000);
    timeline.mutated();
    t.frameAfter(8);
    // Another change while the frames run: the later frames end it.
    timeline.mutated();
    t.frameAfter(8);
    t.frameAfter(8);
    t.advance(1000);
    timeline.flush();
    expect(t.reports()[0]?.clicks).toEqual([{ kind: "tab", label: "Colour", t_input: 1000, t_painted: 1024 }]);
  });

  it("does not record a click that changed nothing, or a pointer that is not a tab, the Nodes button or a menu", () => {
    const t = host();
    const timeline = new Timeline(t.h);
    timeline.pointerDown(null, "", 1);
    timeline.mutated();
    t.frameAfter(8);
    t.frameAfter(8);
    t.advance(1000);
    timeline.flush();
    expect(t.reports()[0]?.clicks).toEqual([]);
  });

  it("names the clicks: tabs, the Nodes button, menus", () => {
    document.body.innerHTML = `
      <div role="tablist"><button role="tab">Colour</button></div>
      <button><span>Nodes</span></button>
      <button aria-haspopup="menu" aria-label="Fit 62%">Fit</button>
      <button>Before</button>`;
    const q = (selector: string) => document.querySelector(selector);
    expect(clickKind(q("[role=tab]"))).toEqual({ kind: "tab", label: "Colour" });
    expect(clickKind(q("button span"))).toEqual({ kind: "node-open", label: "Nodes" });
    expect(clickKind(q("[aria-haspopup]"))).toEqual({ kind: "menu", label: "Fit 62%" });
    expect(clickKind(document.querySelectorAll("button")[3] ?? null)).toBeNull();
    expect(clickKind(null)).toBeNull();
  });
});

describe("installing", () => {
  it("does nothing when the shell did not ask for it", () => {
    expect(installTimeline(false, () => undefined)).toBeNull();
  });

  it("reports at once that the page began", () => {
    const sent: string[] = [];
    const stop = installTimeline(true, (json) => void sent.push(json));
    expect(stop).not.toBeNull();
    expect(JSON.parse(sent[0] ?? "{}")).toMatchObject({ start: true, ticks: [] });
    stop?.();
  });
});
