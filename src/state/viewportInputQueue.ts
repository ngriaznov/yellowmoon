import type { ViewportInput } from "../api/types";

/** The two inputs that arrive faster than the screen draws: a pan (wheel, drag) and a zoom (⌘ wheel, pinch). */
export type StreamInput = Extract<ViewportInput, { kind: "pan" | "zoom" }>;

/** A zoom whose anchor moved by no more than this many CSS pixels is the same zoom. */
const ANCHOR_TOLERANCE_PX = 1;

/** What the queue sends to, and what it asks of, the store. */
export interface ViewportInputSink {
  /** One `viewport_input` command. */
  send: (event: StreamInput) => Promise<void>;
  /** Called once per closed zoom, in order, with its factor: the store's zoom mirror follows what was asked for. */
  mirrorZoom: (factor: number) => void;
  /** A send was refused; what was waiting has been dropped. */
  onError: (error: unknown) => void;
  /** Runs `callback` before the next frame is drawn (`requestAnimationFrame`). */
  schedule: (callback: () => void) => void;
}

/**
 * Wheel, pinch and drag input for the viewport: accumulated while the frame is built and sent once per animation frame, never more
 * than one command in flight, in the order it came (a zoom is about the picture as the pans before it left it). Pans add up, zooms
 * about the same point multiply. Nothing here touches React state.
 *
 * Why: a trackpad sends 60 to 120 wheel events a second and a drag one per pointer move, each one a command to the core. The commands
 * are answered on a pool of threads, so two of them can overtake each other, and the core redraws for each one it gets. One message
 * per frame says the same thing, in order, with a fraction of the traffic (`docs/11` §Photos input).
 */
export class ViewportInputQueue {
  private pending: StreamInput | null = null;
  private readonly outbox: StreamInput[] = [];
  private scheduled = false;
  private sending = false;
  private readonly idle: Array<() => void> = [];

  constructor(private readonly sink: ViewportInputSink) {}

  /** True while anything is collected, waiting or on its way: a click-type input must wait for it (`flush`). */
  get busy(): boolean {
    return this.pending !== null || this.outbox.length > 0 || this.sending;
  }

  /** Collects one event; it goes out with the frame. */
  push(event: StreamInput): void {
    const open = this.pending;
    if (open && sameStream(open, event)) merge(open, event);
    else {
      if (open) this.close(open);
      this.pending = { ...event };
    }
    if (this.scheduled) return;
    this.scheduled = true;
    this.sink.schedule(() => {
      this.scheduled = false;
      void this.flush();
    });
  }

  /** Drops what has not been sent: another photo is on screen now. */
  clear(): void {
    this.pending = null;
    this.outbox.length = 0;
  }

  /** Sends what has been collected, now (a drag ended, a click is about to follow). Resolves when nothing is in flight. */
  flush(): Promise<void> {
    if (this.pending) {
      this.close(this.pending);
      this.pending = null;
    }
    void this.drain();
    return this.sending ? new Promise<void>((resolve) => this.idle.push(resolve)) : Promise.resolve();
  }

  private close(entry: StreamInput): void {
    if (entry.kind === "zoom") this.sink.mirrorZoom(entry.factor);
    const last = this.outbox[this.outbox.length - 1];
    if (last && sameStream(last, entry)) merge(last, entry);
    else this.outbox.push(entry);
  }

  private async drain(): Promise<void> {
    if (this.sending) return;
    this.sending = true;
    try {
      for (let next = this.outbox.shift(); next; next = this.outbox.shift()) await this.sink.send(next);
    } catch (error) {
      this.clear();
      this.sink.onError(error);
    } finally {
      this.sending = false;
      for (const wake of this.idle.splice(0)) wake();
    }
  }
}

function sameStream(a: StreamInput, b: StreamInput): boolean {
  if (a.kind === "zoom" && b.kind === "zoom") return Math.abs(a.x - b.x) <= ANCHOR_TOLERANCE_PX && Math.abs(a.y - b.y) <= ANCHOR_TOLERANCE_PX;
  return a.kind === b.kind;
}

function merge(into: StreamInput, next: StreamInput): void {
  if (into.kind === "pan" && next.kind === "pan") {
    into.dx += next.dx;
    into.dy += next.dy;
  } else if (into.kind === "zoom" && next.kind === "zoom") {
    into.factor *= next.factor;
    into.x = next.x;
    into.y = next.y;
  }
}
