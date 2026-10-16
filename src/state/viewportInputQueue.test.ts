import { describe, expect, it } from "vitest";
import { ViewportInputQueue } from "./viewportInputQueue";
import type { StreamInput } from "./viewportInputQueue";

/** A queue whose frames, wire and failures the test drives. `hold` keeps each send in flight until the test lets it go. */
function rig({ hold = false }: { hold?: boolean } = {}) {
  const frames: Array<() => void> = [];
  const sent: StreamInput[] = [];
  const mirrored: number[] = [];
  const errors: unknown[] = [];
  const gates: Array<() => void> = [];
  const q = new ViewportInputQueue({
    schedule: (cb) => frames.push(cb),
    mirrorZoom: (f) => mirrored.push(f),
    onError: (e) => errors.push(e),
    send: (e) => {
      sent.push(structuredClone(e));
      return hold ? new Promise<void>((resolve) => gates.push(resolve)) : Promise.resolve();
    },
  });
  /** One animation frame: runs what was scheduled, then lets the promises settle. */
  const frame = async (): Promise<void> => {
    for (const cb of frames.splice(0)) cb();
    await Promise.resolve();
    await Promise.resolve();
  };
  return { q, frame, sent, mirrored, errors, gates, frames };
}

describe("the viewport input queue", () => {
  it("sends a burst of pans inside one frame as one message with the sum, and asks for one frame however many events arrive", async () => {
    const r = rig();
    for (let i = 0; i < 100; i++) r.q.push({ kind: "pan", dx: 1, dy: -2 });
    expect(r.frames).toHaveLength(1);
    expect(r.sent).toEqual([]);
    await r.frame();
    expect(r.sent).toEqual([{ kind: "pan", dx: 100, dy: -200 }]);
  });

  it("keeps the order of pan, zoom and pan, and merges zooms about the same point (within a pixel)", async () => {
    const r = rig();
    r.q.push({ kind: "pan", dx: 5, dy: 0 });
    r.q.push({ kind: "zoom", factor: 1.1, x: 10, y: 10 });
    r.q.push({ kind: "zoom", factor: 1.1, x: 10.4, y: 10 });
    r.q.push({ kind: "pan", dx: 3, dy: 0 });
    await r.frame();
    expect(r.sent).toHaveLength(3);
    expect(r.sent[0]).toEqual({ kind: "pan", dx: 5, dy: 0 });
    expect(r.sent[1]?.kind).toBe("zoom");
    expect(r.sent[1]?.kind === "zoom" && r.sent[1].factor).toBeCloseTo(1.21, 12);
    expect(r.sent[2]).toEqual({ kind: "pan", dx: 3, dy: 0 });
    // The mirror follows once per closed zoom, with the factor the core is asked for.
    expect(r.mirrored).toHaveLength(1);
    expect(r.mirrored[0]).toBeCloseTo(1.21, 12);
  });

  it("takes a zoom about another point for another zoom", async () => {
    const r = rig();
    r.q.push({ kind: "zoom", factor: 2, x: 0, y: 0 });
    r.q.push({ kind: "zoom", factor: 2, x: 50, y: 0 });
    await r.frame();
    expect(r.sent).toHaveLength(2);
    expect(r.mirrored).toEqual([2, 2]);
  });

  it("has one command in flight: later frames pile into one waiting message", async () => {
    const r = rig({ hold: true });
    r.q.push({ kind: "pan", dx: 1, dy: 1 });
    await r.frame();
    expect(r.sent).toHaveLength(1);
    for (let f = 0; f < 10; f++) {
      r.q.push({ kind: "pan", dx: 1, dy: 1 });
      await r.frame();
    }
    expect(r.sent, "nothing else is sent until the first is answered").toHaveLength(1);
    r.gates.shift()?.();
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(r.sent).toHaveLength(2);
    expect(r.sent[1]).toEqual({ kind: "pan", dx: 10, dy: 10 });
    r.gates.shift()?.();
  });

  it("sends at once on flush() (a drag ended) and resolves when the wire is quiet", async () => {
    const r = rig();
    r.q.push({ kind: "pan", dx: 7, dy: 0 });
    expect(r.q.busy).toBe(true);
    await r.q.flush();
    expect(r.sent).toEqual([{ kind: "pan", dx: 7, dy: 0 }]);
    expect(r.q.busy).toBe(false);
  });

  it("resolves flush() only after what is in flight has been answered", async () => {
    const r = rig({ hold: true });
    r.q.push({ kind: "pan", dx: 1, dy: 0 });
    let done = false;
    const flushed = r.q.flush().then(() => {
      done = true;
    });
    await Promise.resolve();
    expect(done).toBe(false);
    r.gates.shift()?.();
    await flushed;
    expect(done).toBe(true);
  });

  it("drops what has not been sent on clear() (another photo is on screen)", async () => {
    const r = rig();
    r.q.push({ kind: "pan", dx: 7, dy: 0 });
    r.q.clear();
    await r.frame();
    expect(r.sent).toEqual([]);
    expect(r.q.busy).toBe(false);
  });

  it("reports a refused send once and drops what was waiting behind it", async () => {
    const errors: unknown[] = [];
    const sent: StreamInput[] = [];
    const q = new ViewportInputQueue({
      schedule: (cb) => cb(),
      mirrorZoom: () => undefined,
      onError: (e) => errors.push(e),
      send: (e) => {
        sent.push(e);
        return Promise.reject(new Error("core says no"));
      },
    });
    q.push({ kind: "pan", dx: 1, dy: 1 });
    q.push({ kind: "zoom", factor: 2, x: 5, y: 5 });
    await q.flush();
    expect(errors).toHaveLength(1);
    expect(sent).toHaveLength(1);
    expect(q.busy).toBe(false);
  });
});
