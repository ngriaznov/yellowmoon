import { Item } from "@/components/glass/item";
import { cx } from "./cx";
import { Tag } from "./Tag";
import type { TagVariant } from "./Tag";

/** Where one file of an export is. */
export type QueueState = "waiting" | "running" | "done" | "failed" | "skipped";

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
};

const TAGS: Readonly<Record<QueueState, TagVariant>> = {
  waiting: "waiting",
  running: "active",
  done: "done",
  failed: "failed",
  skipped: "skipped",
};

export interface ExportQueueListProps {
  items: readonly ExportQueueItem[];
  /** Accessible name of the list. Default "Export queue". */
  label?: string;
  className?: string;
}

/**
 * The per-file list of the Export panel while it runs: a glasscn `Item` row per file with its name, a
 * note and a tag (WAITING, 45% in yellow, DONE in ink, SKIPPED in orange for a file that already
 * existed, or FAILED with the reason as the note).
 */
export function ExportQueueList({ items, label = "Export queue", className }: ExportQueueListProps) {
  return (
    <ul aria-label={label} className={cx("ym-queue", className)}>
      {items.map((item) => (
        <Item key={item.id} asChild size="sm">
          <li className="ym-queue__row">
            <span className="ym-queue__text">
              <span className="ym-queue__file" title={item.file}>
                {item.file}
              </span>
              <span className="ym-queue__note">{item.note ?? DEFAULT_NOTES[item.state]}</span>
            </span>
            <Tag variant={TAGS[item.state]}>{item.state === "running" ? `${Math.round(item.percent ?? 0)}%` : undefined}</Tag>
          </li>
        </Item>
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

/**
 * The Start screen's list of photos being developed: a glasscn `Item` row per file with its name, what
 * is happening and a tag (WAITING, DEVELOPING in yellow, READY in ink, SKIPPED in orange with
 * the reason). It scrolls when it is taller than its box.
 */
export function DevelopList({ items, label = "Photos being developed", className }: DevelopListProps) {
  return (
    <div className={cx("ym-develop-list", className)}>
      <ul aria-label={label} className="ym-queue">
        {items.map((item) => (
          <Item key={item.id} asChild size="sm">
            <li className="ym-queue__row">
              <span className="ym-queue__file" title={item.name}>
                {item.name}
              </span>
              <span className="ym-queue__note">{item.note ?? DEVELOP_NOTES[item.state]}</span>
              <Tag variant={item.state} />
            </li>
          </Item>
        ))}
      </ul>
    </div>
  );
}
