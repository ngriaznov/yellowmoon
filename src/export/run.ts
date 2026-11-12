/**
 * The export queue as the panel shows it (`docs/09-export.md` §Queue, `design/mockups/export.html` running and
 * finished states). The core reports `export_progress` events; `applyProgress` folds each one into a run:
 * a row per file with Waiting, a percentage, Done, Skipped or the reason it failed. Pure functions, so the
 * rules are tested without a screen.
 *
 * What an event means (`ExportProgress`): `done` files are finished and the next one, `file`, is
 * under way at `file_percent`. An event with `error` or `skipped` reports the file that just finished
 * (index `done - 1`) and says how. The last event has `done === total`, or `stopped`: a failure of the destination (the disk is
 * full, the folder is read-only or gone) ends the job at that file, and the files after it were never tried (`docs/26` P2-2).
 */
import type { ExportJobId, ExportProgress } from "../api/types";
import type { ExportQueueItem } from "../components";
import { folderLabel, photoCount } from "./format";

/** Where one file of a run is. */
export type RowState = "waiting" | "running" | "done" | "failed" | "skipped" | "not_tried";

/** One file of a run. */
export interface RunRow {
  /** Output file name; the core's name replaces the predicted one as soon as the file is reached. */
  name: string;
  state: RowState;
  /** 0–100 while running. */
  percent: number;
  /** The reason, for a failed file. */
  note?: string;
}

/** The note of a file the job never got to. */
export const NOT_TRIED_NOTE = "Not exported, the export stopped.";

/** A running or finished export. */
export interface ExportRun {
  jobId: ExportJobId;
  phase: "running" | "finished";
  rows: RunRow[];
  /** The folder the files go to, as the panel showed it. */
  destination: string;
  /** Why the job stopped (the core's sentence), when a failure of the destination ended it: the files after the failed one were not tried. */
  stopped?: string;
}

/** A run that has just been queued: every file is waiting. `names` are the predicted file names. */
export function startRun(jobId: ExportJobId, names: readonly string[], destination: string): ExportRun {
  return {
    jobId,
    phase: names.length === 0 ? "finished" : "running",
    rows: names.map((name) => ({ name, state: "waiting", percent: 0 })),
    destination,
  };
}

/** The run after one `export_progress` event (events of other jobs, and of a finished run, change nothing). */
export function applyProgress(run: ExportRun, e: ExportProgress): ExportRun {
  if (e.job_id !== run.jobId || run.phase === "finished") return run;
  // The core knows how many files the job has; the rows were predicted when it was queued (a photo may have been unpicked,
  // or may have failed to develop, in between). The core wins.
  const total = e.total > 0 ? e.total : run.rows.length;
  const done = Math.max(0, Math.min(total, e.done));
  // Copy on write: one copy of the array of rows, and a new object only for a row that changes (a 2000-photo run folds 40,000 events,
  // and most of them change one row), so a row that did not change is the same object as before and its element is not drawn again.
  const rows = run.rows.slice(0, total);
  while (rows.length < total) rows.push({ name: `File ${rows.length + 1}`, state: "waiting", percent: 0 });
  const settled = (state: RowState): boolean => state === "failed" || state === "skipped";
  const change = (i: number, patch: Partial<RunRow>): void => {
    const row = rows[i]!;
    const keys = Object.keys(patch) as (keyof RunRow)[];
    if (keys.every((k) => row[k] === patch[k])) return;
    rows[i] = { ...row, ...patch };
  };

  // Files before `done` are finished; a file that failed or was skipped keeps saying so.
  for (let i = 0; i < done; i++) {
    const row = rows[i]!;
    if (!settled(row.state)) change(i, { state: "done", percent: 100 });
  }
  if ((e.error || e.skipped) && done > 0) {
    const row = rows[done - 1]!;
    const name = e.file || row.name;
    if (e.error) change(done - 1, { name, percent: 100, state: "failed", note: e.error });
    else change(done - 1, { name, percent: 100, state: "skipped" });
  } else if (done < total) {
    const row = rows[done]!;
    change(done, { name: e.file || row.name, state: "running", percent: Math.max(0, Math.min(100, Math.round(e.file_percent ?? 0))) });
  }
  if (e.stopped && e.error && done > 0) {
    // The job ended at this file: the rest were not tried, and nothing more will come.
    for (let i = done; i < total; i++) change(i, { state: "not_tried", percent: 0, note: NOT_TRIED_NOTE });
    return { ...run, rows, phase: "finished", stopped: e.error };
  }
  return { ...run, rows, phase: done >= total ? "finished" : "running" };
}

/** How a run stands. */
export interface RunSummary {
  total: number;
  /** Files finished, however they ended. */
  finished: number;
  /** Files written. */
  saved: number;
  failed: number;
  skipped: number;
  /** Files the job never got to because it stopped. */
  notTried: number;
  /** Overall progress, 0–100, counting the file under way by its own percentage. */
  percent: number;
}

/** Counts and overall progress of a run. */
export function summarise(run: ExportRun): RunSummary {
  const total = run.rows.length;
  let saved = 0;
  let failed = 0;
  let skipped = 0;
  let notTried = 0;
  let partial = 0;
  for (const row of run.rows) {
    if (row.state === "done") saved++;
    else if (row.state === "failed") failed++;
    else if (row.state === "skipped") skipped++;
    else if (row.state === "not_tried") notTried++;
    else if (row.state === "running") partial += row.percent / 100;
  }
  const finished = saved + failed + skipped;
  const percent = total === 0 ? 0 : Math.round(((finished + partial) / total) * 100);
  return { total, finished, saved, failed, skipped, notTried, percent: Math.min(100, percent) };
}

/** Title of the panel: "Exporting", "Exported", "Not exported" when no file could be written, or "Stopped" when the job stopped after saving some. */
export function runTitle(run: ExportRun): string {
  if (run.phase === "running") return "Exporting";
  const s = summarise(run);
  if (run.stopped !== undefined) return s.saved === 0 ? "Not exported" : "Stopped";
  return s.saved === 0 && s.failed > 0 ? "Not exported" : "Exported";
}

/**
 * The line under the title: "1 of 3 done." while running; "3 photos saved to Lisbon export." when finished, with
 * what failed or was skipped added in plain words.
 */
export function runBlurb(run: ExportRun): string {
  const s = summarise(run);
  if (run.phase === "running") return `${s.finished} of ${s.total} done.`;
  const folder = folderLabel(run.destination);
  if (run.stopped !== undefined) {
    const saved = s.saved > 0 ? `${photoCount(s.saved)} saved to ${folder}.` : "Nothing was saved.";
    return `${saved} The export stopped: ${run.stopped}`;
  }
  const parts: string[] = [];
  if (s.saved > 0 || (s.failed === 0 && s.skipped === 0)) parts.push(`${photoCount(s.saved)} saved to ${folder}.`);
  if (s.failed > 0) parts.push(`${photoCount(s.failed)} could not be saved.`);
  if (s.skipped > 0) parts.push(`${photoCount(s.skipped)} skipped because the file already exists.`);
  return parts.join(" ");
}

/** The note under the list: what the mockup says while running and when finished. */
export function runNote(run: ExportRun): string {
  return run.phase === "finished"
    ? "Originals are never changed. Open the folder or go back to your photos."
    : "Originals are never changed. You can keep working while this runs; an update waits until the export is finished.";
}

/** The queue item made for a row, so that a row that did not change gives the list the same item again (the list's rows compare by it). */
const itemOfRow = new WeakMap<RunRow, ExportQueueItem>();

/** The rows as `ExportQueueList` draws them. A row that has not changed (same object, same place) yields the same item as last time. */
export function queueItems(run: ExportRun): ExportQueueItem[] {
  return run.rows.map((row, id) => {
    const known = itemOfRow.get(row);
    if (known && known.id === id) return known;
    const item: ExportQueueItem = {
      id,
      file: row.name,
      state: row.state,
      percent: row.percent,
      note: row.state === "failed" || row.state === "not_tried" ? row.note : undefined,
    };
    itemOfRow.set(row, item);
    return item;
  });
}

/** The text in the Export cell of the header while a run exists: "Exporting 1 of 3", or "Exported" when it has finished ("Export stopped", or "Not exported" when nothing was saved, after a stop). */
export function cellLabel(run: ExportRun): string {
  if (run.phase === "finished") {
    if (run.stopped === undefined) return "Exported";
    return summarise(run).saved === 0 ? "Not exported" : "Export stopped";
  }
  const s = summarise(run);
  return `Exporting ${s.finished} of ${s.total}`;
}
