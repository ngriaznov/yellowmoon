/**
 * The export queue as the panel shows it (`docs/09-export.md` §Queue, `design/mockups/export.html` running and
 * finished states). The core reports `export_progress` events; `applyProgress` folds each one into a run:
 * a row per file with Waiting, a percentage, Done, Skipped or the reason it failed. Pure functions, so the
 * rules are tested without a screen.
 *
 * What an event means (`ExportProgress`): `done` files are finished and the next one, `file`, is
 * under way at `file_percent`. An event with `error` or `skipped` reports the file that just finished
 * (index `done - 1`) and says how. The last event has `done === total`.
 */
import type { ExportJobId, ExportProgress } from "../api/types";
import type { ExportQueueItem } from "../components";
import { folderLabel, photoCount } from "./format";

/** Where one file of a run is. */
export type RowState = "waiting" | "running" | "done" | "failed" | "skipped";

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

/** A running or finished export. */
export interface ExportRun {
  jobId: ExportJobId;
  phase: "running" | "finished";
  rows: RunRow[];
  /** The folder the files go to, as the panel showed it. */
  destination: string;
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
  const rows = run.rows.slice(0, total).map((row) => ({ ...row }));
  while (rows.length < total) rows.push({ name: `File ${rows.length + 1}`, state: "waiting", percent: 0 });
  const settled = (state: RowState): boolean => state === "failed" || state === "skipped";

  // Files before `done` are finished; a file that failed or was skipped keeps saying so.
  for (let i = 0; i < done; i++) {
    const row = rows[i]!;
    if (!settled(row.state)) {
      row.state = "done";
      row.percent = 100;
    }
  }
  if ((e.error || e.skipped) && done > 0) {
    const row = rows[done - 1]!;
    row.name = e.file || row.name;
    row.percent = 100;
    if (e.error) {
      row.state = "failed";
      row.note = e.error;
    } else {
      row.state = "skipped";
    }
  } else if (done < total) {
    const row = rows[done]!;
    if (e.file) row.name = e.file;
    row.state = "running";
    row.percent = Math.max(0, Math.min(100, Math.round(e.file_percent ?? 0)));
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
  /** Overall progress, 0–100, counting the file under way by its own percentage. */
  percent: number;
}

/** Counts and overall progress of a run. */
export function summarise(run: ExportRun): RunSummary {
  const total = run.rows.length;
  let saved = 0;
  let failed = 0;
  let skipped = 0;
  let partial = 0;
  for (const row of run.rows) {
    if (row.state === "done") saved++;
    else if (row.state === "failed") failed++;
    else if (row.state === "skipped") skipped++;
    else if (row.state === "running") partial += row.percent / 100;
  }
  const finished = saved + failed + skipped;
  const percent = total === 0 ? 0 : Math.round(((finished + partial) / total) * 100);
  return { total, finished, saved, failed, skipped, percent: Math.min(100, percent) };
}

/** Title of the panel: "Exporting", "Exported", or "Not exported" when no file could be written. */
export function runTitle(run: ExportRun): string {
  if (run.phase === "running") return "Exporting";
  const s = summarise(run);
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

/** The rows as `ExportQueueList` draws them. */
export function queueItems(run: ExportRun): ExportQueueItem[] {
  return run.rows.map((row, id) => ({
    id,
    file: row.name,
    state: row.state,
    percent: row.percent,
    note: row.state === "failed" ? row.note : undefined,
  }));
}

/** The text in the Export cell of the header while a run exists: "Exporting 1 of 3", or "Exported" when it has finished. */
export function cellLabel(run: ExportRun): string {
  if (run.phase === "finished") return "Exported";
  const s = summarise(run);
  return `Exporting ${s.finished} of ${s.total}`;
}
