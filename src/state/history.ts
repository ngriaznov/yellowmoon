/**
 * Undo and redo for one photo's settings (`docs/11-ui.md` §Keyboard shortcuts: "in memory,
 * 100 steps"). An entry holds only what one change touched, as two node patches: the values
 * before and the values after (`docs/23` section 3.7). Undo sends `before`, redo sends `after`.
 * Entries that touch different nodes or fields don't disturb each other, and a change made by
 * "Apply to" on another photo never needs an entry of its own.
 */
import type { DevelopSettings, SettingsPatch } from "../api/types";
import { diffGraph } from "./settingsGraph";

/** Steps kept per photo. */
export const HISTORY_LIMIT = 100;

/** One undoable change. */
export interface HistoryEntry {
  before: SettingsPatch;
  after: SettingsPatch;
}

/**
 * What changed between two settings, as an entry; null when nothing did. It diffs per node and for the links: an entry has the nodes
 * a change touched and, in each, the fields that differ (a field whole, so `white_balance` going from Auto to Manual is one change and
 * a colour wheel moving is one `color_balance` change; `grade` and `trim` go one field deeper, so a contrast change is a contrast), a
 * node added whole with its `kind` or taken out in `remove`, and `links` when they differ. See {@link diffGraph}.
 */
export function diffSettings(before: DevelopSettings, after: DevelopSettings): HistoryEntry | null {
  return diffGraph(before, after);
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
