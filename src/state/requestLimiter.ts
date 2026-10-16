/**
 * What a request that was dropped before it started is rejected with. It is not a failure: nobody should show it, count it against a
 * retry allowance or ask again for it (`useLookThumbnail` ignores it); whoever still wants the picture asks for the new one.
 */
export class Superseded extends Error {
  readonly superseded = true;

  constructor() {
    super("This request was replaced before it started.");
    this.name = "Superseded";
  }
}

/** True for the rejection of a request that was {@link RequestLimiter.supersede}d. */
export function isSuperseded(error: unknown): boolean {
  return error instanceof Superseded || (typeof error === "object" && error !== null && (error as { superseded?: unknown }).superseded === true);
}

interface Waiting<Tag> {
  tag: Tag;
  /** The group of requests made in the same task (one commit of the screen): the newest group goes first. */
  batch: number;
  seq: number;
  start: () => Promise<unknown>;
  resolve: (value: unknown) => void;
  reject: (reason: unknown) => void;
}

/**
 * Lets at most `limit` requests run at once and keeps the rest in a queue the caller can thin out. The core has one foreground worker
 * that a job is never taken off, so a hundred look tiles asked for at once are a hundred jobs the next frame of the viewport may have to
 * wait behind; two at a time means it waits behind one.
 *
 * Order: the requests made most recently go first (what has just scrolled into view is what the person is looking at), and the
 * ones made together, in the same task, go in the order they were made (a row of tiles fills from the left). Each request carries a
 * `tag` so a group of them can be dropped (`supersede`) when the thing they were for has changed.
 */
export class RequestLimiter<Tag> {
  private running = 0;
  private batch = 0;
  private seq = 0;
  private batchOpen = false;
  private readonly waiting: Array<Waiting<Tag>> = [];

  constructor(private readonly limit: number) {}

  /** Requests that have started and not finished. */
  get inFlight(): number {
    return this.running;
  }

  /** Requests that are waiting for a place. */
  get queued(): number {
    return this.waiting.length;
  }

  /** Runs `start` when a place is free; resolves and rejects as it does, or rejects with {@link Superseded} if it was dropped first. */
  run<T>(tag: Tag, start: () => Promise<T>): Promise<T> {
    if (!this.batchOpen) {
      this.batchOpen = true;
      this.batch += 1;
      queueMicrotask(() => {
        this.batchOpen = false;
      });
    }
    return new Promise<T>((resolve, reject) => {
      this.waiting.push({ tag, batch: this.batch, seq: (this.seq += 1), start, resolve: resolve as (value: unknown) => void, reject });
      this.pump();
    });
  }

  /** Drops every waiting request whose tag matches; each rejects with {@link Superseded}. A request that has started is not touched. */
  supersede(match: (tag: Tag) => boolean): void {
    const dropped = this.waiting.filter((w) => match(w.tag));
    if (dropped.length === 0) return;
    const kept = this.waiting.filter((w) => !match(w.tag));
    this.waiting.length = 0;
    this.waiting.push(...kept);
    for (const w of dropped) w.reject(new Superseded());
  }

  private take(): Waiting<Tag> | undefined {
    let best = -1;
    for (let i = 0; i < this.waiting.length; i += 1) {
      const w = this.waiting[i] as Waiting<Tag>;
      const b = best < 0 ? undefined : (this.waiting[best] as Waiting<Tag>);
      if (!b || w.batch > b.batch || (w.batch === b.batch && w.seq < b.seq)) best = i;
    }
    return best < 0 ? undefined : this.waiting.splice(best, 1)[0];
  }

  private pump(): void {
    while (this.running < this.limit) {
      const next = this.take();
      if (!next) return;
      this.running += 1;
      let started: Promise<unknown>;
      try {
        started = next.start();
      } catch (error) {
        started = Promise.reject(error);
      }
      void started.then(next.resolve, next.reject).finally(() => {
        this.running -= 1;
        this.pump();
      });
    }
  }
}
