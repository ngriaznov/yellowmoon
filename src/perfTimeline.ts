/**
 * The timeline of a slider tick, page side (U-02; the shell's side and the log format are in `src-tauri/src/timeline.rs` and
 * `docs/01-architecture.md` §Perf log). With the shell started with `YM_PERF=1` the page stamps
 *
 * - each settings patch it sends: `t_input` (the first pointer move or key the patch answers, from the event's own `timeStamp`),
 *   `t_sent`, `t_reply` and `t_painted` (the second animation frame after the reply was drawn into the page);
 * - each click on a tab, the Nodes button or a menu: `t_input` (the `pointerdown`) and `t_painted` (the second frame after the last
 *   change of the page it made).
 *
 * The records sit in rings of 512 and go to the shell once a second in one fire-and-forget `perf_report` command. All times are
 * milliseconds since the Unix epoch (`performance.timeOrigin` plus the page's clock), which is the clock the shell stamps with, so the
 * two sides compare. The `n` of a tick is the position of its `update_settings` call in the page's life; the shell counts the same calls.
 * Nothing leaves the machine. Off (no `YM_PERF`, a browser, a test) nothing is installed and every call does nothing.
 */
import { gestureHeld } from "./state/gestureHold";

/** How many ticks and how many clicks are kept when the shell cannot be reached. */
export const RING = 512;

/** An input older than this when a patch is sent does not belong to it (a mouse that moved a while ago, a key long since). */
const INPUT_FRESH_MS = 250;

/** How long a click may wait for the page to change before it is taken to have changed nothing. */
const CLICK_QUIET_MS = 400;

/** How often the records go to the shell. */
export const REPORT_EVERY_MS = 1000;

/** One `update_settings` call. Times are epoch milliseconds. */
export interface TickRecord {
  n: number;
  final: boolean;
  t_input: number | null;
  t_sent: number;
  t_reply: number | null;
  t_painted: number | null;
}

/** One click of a tab, the Nodes button or a menu. */
export interface ClickRecord {
  kind: "tab" | "node-open" | "menu";
  label: string;
  t_input: number;
  t_painted: number | null;
}

/** What the page can be asked to do and told about by the timeline's tests. */
export interface TimelineHost {
  /** Epoch milliseconds now. */
  now: () => number;
  /** Runs `callback` at the next animation frame. */
  frame: (callback: () => void) => void;
  /** Sends the report to the shell (fire and forget). */
  send: (json: string) => void;
  /** Whether a drag of a knob, a wheel or a pin is held. */
  gesture: () => boolean;
}

/** What a tick's caller does as the command goes and comes back. */
export interface TickHandle {
  /** The reply came and was drawn into the page (or the command failed: `ok` false). */
  replied: (ok?: boolean) => void;
}

const NO_TICK: TickHandle = { replied: () => undefined };

export class Timeline {
  private n = 0;
  private ticks: TickRecord[] = [];
  private clicks: ClickRecord[] = [];
  private input: number | null = null;
  private pointerMoves = 0;
  private started = false;
  private readonly page = `${Math.random().toString(36).slice(2, 10)}`;
  private pending: { kind: ClickRecord["kind"]; label: string; t_input: number; epoch: number; record: ClickRecord } | null = null;
  private timer: ReturnType<typeof setInterval> | null = null;

  constructor(private readonly host: TimelineHost) {}

  /** The latest hand input: a pointer move or a key at `at` (epoch ms). */
  noteInput(at: number, pointerMove = false): void {
    if (this.input === null || at - this.input > INPUT_FRESH_MS) this.input = at;
    if (pointerMove && this.host.gesture()) this.pointerMoves += 1;
  }

  /** A patch is being sent. Call `replied` when the reply was drawn. */
  tick(final: boolean): TickHandle {
    const t_sent = this.host.now();
    const input = this.input !== null && t_sent - this.input <= INPUT_FRESH_MS ? this.input : null;
    this.input = null;
    const record: TickRecord = { n: ++this.n, final, t_input: input, t_sent, t_reply: null, t_painted: null };
    this.push(this.ticks, record);
    return {
      replied: (ok = true) => {
        record.t_reply = this.host.now();
        if (!ok) return;
        // The page drew the reply (the commit is synchronous): the second frame after it is when the person can see it.
        this.host.frame(() => this.host.frame(() => (record.t_painted = this.host.now())));
      },
    };
  }

  /** A pointer went down on something: if it is a tab, the Nodes button or a menu, the click is timed until the page stops changing. */
  pointerDown(kind: ClickRecord["kind"] | null, label: string, at: number): void {
    if (kind === null) return;
    const record: ClickRecord = { kind, label, t_input: at, t_painted: null };
    this.pending = { kind, label, t_input: at, epoch: 0, record };
    const mine = this.pending;
    // A click that changes nothing within the quiet time is not a record.
    setTimeout(() => {
      if (this.pending === mine && mine.epoch === 0) this.pending = null;
    }, CLICK_QUIET_MS);
  }

  /** The page changed (a mutation of its tree). */
  mutated(): void {
    const click = this.pending;
    if (!click) return;
    const epoch = ++click.epoch;
    this.host.frame(() =>
      this.host.frame(() => {
        // Another change since means the page was not done: the later frame ends it.
        if (this.pending !== click || click.epoch !== epoch) return;
        click.record.t_painted = this.host.now();
        this.push(this.clicks, click.record);
        this.pending = null;
      }),
    );
  }

  /** Sends what has been recorded since the last time. The first report says the page began (the shell's count of calls is its zero). */
  flush(): void {
    const first = !this.started;
    if (!first && this.ticks.length === 0 && this.clicks.length === 0 && this.pointerMoves === 0) return;
    this.started = true;
    // A tick is complete once it was painted (or its reply is old enough that it never will be: a failed command). A tick still waiting
    // stays for the next report, and one that waits too long is dropped.
    const now = this.host.now();
    const ready = (t: TickRecord): boolean => t.t_painted !== null || (t.t_reply !== null && now - t.t_reply > 500);
    const done = this.ticks.filter(ready);
    const waiting = this.ticks.filter((t) => !ready(t) && now - t.t_sent < 5000);
    const report = { page: this.page, start: first || undefined, pointer_moves: this.pointerMoves, ticks: done, clicks: this.clicks };
    this.ticks = waiting;
    this.clicks = [];
    this.pointerMoves = 0;
    this.host.send(JSON.stringify(report));
  }

  /** Starts the once-a-second report. */
  start(): void {
    this.flush();
    this.timer ??= setInterval(() => this.flush(), REPORT_EVERY_MS);
  }

  stop(): void {
    if (this.timer !== null) clearInterval(this.timer);
    this.timer = null;
  }

  private push<T>(ring: T[], item: T): void {
    ring.push(item);
    if (ring.length > RING) ring.shift();
  }
}

const NOOP: Pick<Timeline, "tick"> = { tick: () => NO_TICK };

/** What the click at `target` is called, or null for a click that is not timed. */
export function clickKind(target: Element | null): { kind: ClickRecord["kind"]; label: string } | null {
  const button = target?.closest("button, [role='tab'], [role='menuitem'], [role='option'], [aria-haspopup]");
  if (!button) return null;
  const label = (button.getAttribute("aria-label") ?? button.textContent ?? "").trim().replace(/\s+/g, " ").slice(0, 40);
  if (button.matches("[role='tab']")) return { kind: "tab", label };
  if (/^Nodes\b/.test(label)) return { kind: "node-open", label };
  if (button.matches("[role='menuitem'], [role='option'], [aria-haspopup]")) return { kind: "menu", label };
  return null;
}

let shared: Pick<Timeline, "tick"> = NOOP;

/** Stamps a patch being sent; a function that does nothing while the timeline is off. */
export function stampTick(final: boolean): TickHandle {
  return shared.tick(final);
}

/**
 * Installs the timeline on the page: listeners for the hand's input and for clicks, a mutation observer for the page's changes and the
 * once-a-second report. Does nothing unless `enabled` (the shell was started with `YM_PERF`).
 */
export function installTimeline(enabled: boolean, send: (json: string) => void, win: Window = window): (() => void) | null {
  if (!enabled) return null;
  const epoch = (): number => win.performance.timeOrigin + win.performance.now();
  const timeline = new Timeline({
    now: epoch,
    frame: (callback) => void win.requestAnimationFrame(() => callback()),
    send,
    gesture: gestureHeld,
  });
  shared = timeline;
  const move = (e: Event): void => timeline.noteInput(win.performance.timeOrigin + e.timeStamp, true);
  const key = (e: Event): void => timeline.noteInput(win.performance.timeOrigin + e.timeStamp);
  const down = (e: Event): void => {
    const hit = clickKind(e.target instanceof Element ? e.target : null);
    timeline.pointerDown(hit?.kind ?? null, hit?.label ?? "", win.performance.timeOrigin + e.timeStamp);
  };
  win.addEventListener("pointermove", move, { capture: true, passive: true });
  win.addEventListener("keydown", key, { capture: true, passive: true });
  win.addEventListener("pointerdown", down, { capture: true, passive: true });
  const observer = new win.MutationObserver(() => timeline.mutated());
  observer.observe(win.document.documentElement, { subtree: true, childList: true, attributes: true, characterData: true });
  timeline.start();
  return () => {
    timeline.stop();
    observer.disconnect();
    win.removeEventListener("pointermove", move, { capture: true });
    win.removeEventListener("keydown", key, { capture: true });
    win.removeEventListener("pointerdown", down, { capture: true });
    shared = NOOP;
  };
}
