/**
 * Undo and redo for one photo's settings (`docs/11-ui.md` §Keyboard shortcuts: "in memory,
 * 100 steps"). An entry holds only the settings that one change touched, as two patches: the
 * values before and the values after. Undo sends `before`, redo sends `after`. Entries that
 * touch different settings don't disturb each other, and a change made by "Apply to" on another
 * photo never needs an entry of its own.
 */
import type { DevelopSettings, SettingsPatch } from "../api/types";

/** Steps kept per photo. */
export const HISTORY_LIMIT = 100;

/** One undoable change. */
export interface HistoryEntry {
  before: SettingsPatch;
  after: SettingsPatch;
}

const clone = <T>(value: T): T => JSON.parse(JSON.stringify(value)) as T;
const sameValue = (a: unknown, b: unknown): boolean => JSON.stringify(a) === JSON.stringify(b);

/**
 * What changed between two settings, as an entry; null when nothing did. Compares whole top-level
 * fields, so `white_balance` going from Auto to Manual is one change and a colour wheel moving is
 * one `color_balance` change.
 */
export function diffSettings(before: DevelopSettings, after: DevelopSettings): HistoryEntry | null {
  const entry: HistoryEntry = { before: {}, after: {} };
  let changed = false;
  for (const key of Object.keys(after) as Array<keyof DevelopSettings>) {
    if (sameValue(before[key], after[key])) continue;
    (entry.before as Record<string, unknown>)[key] = clone(before[key]);
    (entry.after as Record<string, unknown>)[key] = clone(after[key]);
    changed = true;
  }
  return changed ? entry : null;
}

/** The two stacks of one photo. */
export class PhotoHistory {
  private undoStack: HistoryEntry[] = [];
  private redoStack: HistoryEntry[] = [];

  /** Records a finished change. A new change ends the redo chain. */
  push(entry: HistoryEntry): void {
    this.undoStack.push(entry);
    if (this.undoStack.length > HISTORY_LIMIT) this.undoStack.shift();
    this.redoStack = [];
  }

  /** Takes the newest change off the undo stack and puts it on the redo stack; null when there is none. */
  undo(): HistoryEntry | null {
    const entry = this.undoStack.pop();
    if (!entry) return null;
    this.redoStack.push(entry);
    return entry;
  }

  /** Takes the newest undone change back; null when there is none. */
  redo(): HistoryEntry | null {
    const entry = this.redoStack.pop();
    if (!entry) return null;
    this.undoStack.push(entry);
    return entry;
  }

  get canUndo(): boolean {
    return this.undoStack.length > 0;
  }

  get canRedo(): boolean {
    return this.redoStack.length > 0;
  }

  /** Steps that can be undone. */
  get undoDepth(): number {
    return this.undoStack.length;
  }

  /** Steps that can be redone. */
  get redoDepth(): number {
    return this.redoStack.length;
  }
}
