import { memo, useRef } from "react";
import { Item } from "@/components/glass/item";
import { cx } from "./cx";
import { Tag } from "./Tag";
import type { TagVariant } from "./Tag";
import { useWindowedRange } from "./windowed";

/** Where one file of an export is. */
export type QueueState = "waiting" | "running" | "done" | "failed" | "skipped" | "not_tried";

/** One file in the export queue. */
export interface ExportQueueItem {
  id: number | string;
  /** Output file name: "DSC04204_neutral.jpg". */
  file: string;
  state: QueueState;
  /** 0–100 while `running`: the tag reads "45%". */
  percent?: number;
  /** Second line. Defaults: "Queued", "Rendering at full quality", "Saved"; for `failed` give the reason. */
  note?: string;
}

const DEFAULT_NOTES: Readonly<Record<QueueState, string>> = {
  waiting: "Queued",
  running: "Rendering at full quality",
  done: "Saved",
  failed: "Could not be saved",
  skipped: "Skipped, the file already exists",
  not_tried: "Not exported, the export stopped.",
};

const TAGS: Readonly<Record<QueueState, TagVariant>> = {
  waiting: "waiting",
  running: "active",
  done: "done",
  failed: "failed",
  skipped: "skipped",
  not_tried: "skipped",
};

export interface ExportQueueListProps {
  items: readonly ExportQueueItem[];
  /** Accessible name of the list. Default "Export queue". */
  label?: string;
  className?: string;
}

/** One file of the queue, drawn again only when something it shows has changed. */
const QueueRow = memo(
  function QueueRow({ item }: { item: ExportQueueItem }) {
    return (
      <Item asChild size="sm">
        <li className="ym-queue__row">
          <span className="ym-queue__text">
            <span className="ym-queue__file" title={item.file}>
              {item.file}
            </span>
            <span className="ym-queue__note">{item.note ?? DEFAULT_NOTES[item.state]}</span>
          </span>
          <Tag variant={TAGS[item.state]}>
            {item.state === "running" ? `${Math.round(item.percent ?? 0)}%` : item.state === "not_tried" ? "Not exported" : undefined}
          </Tag>
        </li>
      </Item>
    );
  },
  (a, b) =>
    a.item === b.item ||
    (a.item.id === b.item.id && a.item.file === b.item.file && a.item.state === b.item.state && a.item.percent === b.item.percent && a.item.note === b.item.note),
);

/**
 * The per-file list of the Export panel while it runs: a glasscn `Item` row per file with its name, a
 * note and a tag (WAITING, 45% in yellow, DONE in ink, SKIPPED in orange for a file that already
 * existed, NOT EXPORTED in orange for a file the job never got to because it stopped, or FAILED with the reason as the note). A long queue (2000 photos) draws only the rows in view and a margin
 * (`useWindowedRange`); a row is drawn again only when its own file, state, percentage or note changes.
 */
export function ExportQueueList({ items, label = "Export queue", className }: ExportQueueListProps) {
  const list = useRef<HTMLUListElement>(null);
  const win = useWindowedRange({ count: items.length, axis: "y", list, itemSelector: ".ym-queue__row", estimate: 53 });
  return (
    <ul
      ref={list}
      aria-label={label}
      className={cx("ym-queue", className)}
      style={win.windowed ? { paddingTop: win.before, paddingBottom: win.after } : undefined}
    >
      {items.slice(win.start, win.end).map((item) => (
        <QueueRow key={item.id} item={item} />
      ))}
    </ul>
  );
}

/** Where one photo is on the Start screen's list. */
export type DevelopState = "waiting" | "developing" | "ready" | "skipped";

/** One photo being developed. */
export interface DevelopItem {
  id: number | string;
  /** File name without extension: "DSC04201". */
  name: string;
  state: DevelopState;
  /** Second column. Defaults: nothing, "Decoding and demosaicing", "Auto exposure, white balance and noise"; for `skipped` give the reason. */
  note?: string;
}

const DEVELOP_NOTES: Readonly<Record<DevelopState, string>> = {
  waiting: "",
  developing: "Decoding and demosaicing",
  ready: "Auto exposure, white balance and noise",
  skipped: "",
};

export interface DevelopListProps {
  items: readonly DevelopItem[];
  label?: string;
  className?: string;
}

/** One photo of the Start list, drawn again only when something it shows has changed. */
const DevelopRow = memo(
  function DevelopRow({ item }: { item: DevelopItem }) {
    return (
      <Item asChild size="sm">
        <li className="ym-queue__row">
          <span className="ym-queue__file" title={item.name}>
            {item.name}
          </span>
          <span className="ym-queue__note">{item.note ?? DEVELOP_NOTES[item.state]}</span>
          <Tag variant={item.state} />
        </li>
      </Item>
    );
  },
  (a, b) => a.item === b.item || (a.item.id === b.item.id && a.item.name === b.item.name && a.item.state === b.item.state && a.item.note === b.item.note),
);

/**
 * The Start screen's list of photos being developed: a glasscn `Item` row per file with its name, what
 * is happening and a tag (WAITING, DEVELOPING in yellow, READY in ink, SKIPPED in orange with
 * the reason). It scrolls when it is taller than its box, and a long list (2000 photos) draws only the rows in view and a margin.
 */
export function DevelopList({ items, label = "Photos being developed", className }: DevelopListProps) {
  const list = useRef<HTMLUListElement>(null);
  const win = useWindowedRange({ count: items.length, axis: "y", list, itemSelector: ".ym-queue__row", estimate: 41 });
  return (
    // The list scrolls, so it is a keyboard stop (WebKit does not focus a scroller that has nothing to focus, and Chromium shows it with no
    // name) and a region with a name: the arrow keys, Space and Page keys then scroll it (docs/28 A-02; axe `scrollable-region-focusable`).
    <div className={cx("ym-develop-list", className)} role="region" aria-label="Photos to develop" tabIndex={0}>
      <ul
        ref={list}
        aria-label={label}
        className="ym-queue"
        style={win.windowed ? { paddingTop: win.before, paddingBottom: win.after } : undefined}
      >
        {items.slice(win.start, win.end).map((item) => (
          <DevelopRow key={item.id} item={item} />
        ))}
      </ul>
    </div>
  );
}
