import { describe, expect, it } from "vitest";
import type { ExportProgress } from "../api/types";
import { applyProgress, cellLabel, queueItems, runBlurb, runNote, runTitle, startRun, summarise } from "./run";
import type { ExportRun } from "./run";

const NAMES = ["DSC04203_neutral.jpg", "DSC04204_neutral.jpg", "DSC04205_neutral.jpg"];
const event = (patch: Partial<ExportProgress>): ExportProgress => ({
  job_id: 1,
  file: "DSC04203_neutral.jpg",
  done: 0,
  total: 3,
  file_percent: 0,
  error: null,
  skipped: false,
  ...patch,
});
const fresh = (): ExportRun => startRun(1, NAMES, "~/Pictures/Lisbon export");

describe("startRun", () => {
  it("lists every file as waiting", () => {
    const run = fresh();
    expect(run.phase).toBe("running");
    expect(run.rows.map((r) => r.state)).toEqual(["waiting", "waiting", "waiting"]);
    expect(summarise(run)).toMatchObject({ total: 3, finished: 0, percent: 0 });
  });

  it("is finished at once when there is nothing to do", () => {
    expect(startRun(2, [], "~/x").phase).toBe("finished");
  });
});

describe("applyProgress", () => {
  it("puts the file under way at its percentage and keeps the rest waiting", () => {
    const run = applyProgress(fresh(), event({ done: 1, file: "DSC04204_neutral.jpg", file_percent: 45 }));
    expect(run.rows.map((r) => r.state)).toEqual(["done", "running", "waiting"]);
    expect(run.rows[1]).toMatchObject({ percent: 45, name: "DSC04204_neutral.jpg" });
    expect(summarise(run).percent).toBe(48); // (1 + 0.45) / 3
    expect(runBlurb(run)).toBe("1 of 3 done.");
    expect(runTitle(run)).toBe("Exporting");
  });

  it("takes the core's name for a file once it is reached (Keep both adds a number)", () => {
    const run = applyProgress(fresh(), event({ file: "DSC04203_neutral 2.jpg" }));
    expect(run.rows[0]?.name).toBe("DSC04203_neutral 2.jpg");
    expect(queueItems(run)[0]?.file).toBe("DSC04203_neutral 2.jpg");
  });

  it("finishes with the last event", () => {
    let run = applyProgress(fresh(), event({ done: 1, file: "DSC04204_neutral.jpg" }));
    run = applyProgress(run, event({ done: 3, file: "DSC04205_neutral.jpg", file_percent: 100 }));
    expect(run.phase).toBe("finished");
    expect(run.rows.map((r) => r.state)).toEqual(["done", "done", "done"]);
    expect(runTitle(run)).toBe("Exported");
    expect(runBlurb(run)).toBe("3 photos saved to Lisbon export.");
    expect(runNote(run)).toBe("Originals are never changed. Open the folder or go back to your photos.");
    expect(cellLabel(run)).toBe("Exported");
  });

  it("follows the core's count when it differs from the files that were predicted", () => {
    const fewer = applyProgress(fresh(), event({ total: 2, done: 1, file: "DSC04204_neutral.jpg" }));
    expect(fewer.rows).toHaveLength(2);
    expect(fewer.rows.map((r) => r.state)).toEqual(["done", "running"]);
    const more = applyProgress(fresh(), event({ total: 4, done: 0 }));
    expect(more.rows.map((r) => r.name)).toEqual([...NAMES.slice(0, 3), "File 4"]);
    expect(summarise(more).total).toBe(4);
    expect(applyProgress(fresh(), event({ total: 3, done: 3, file_percent: 100 })).phase).toBe("finished");
  });

  it("ignores events of another job and events after the end", () => {
    const run = fresh();
    expect(applyProgress(run, event({ job_id: 9, done: 2 }))).toBe(run);
    const finished = applyProgress(run, event({ done: 3, file_percent: 100 }));
    expect(applyProgress(finished, event({ done: 1 }))).toBe(finished);
  });

  it("marks a file that failed with its reason, and goes on with the next one", () => {
    let run = applyProgress(fresh(), event({ done: 1, file: "DSC04203_neutral.jpg", file_percent: 100, error: "The disk is full. Free some space, then export again." }));
    expect(run.rows[0]).toMatchObject({ state: "failed", note: "The disk is full. Free some space, then export again." });
    run = applyProgress(run, event({ done: 1, file: "DSC04204_neutral.jpg", file_percent: 20 }));
    expect(run.rows.map((r) => r.state)).toEqual(["failed", "running", "waiting"]);
    run = applyProgress(run, event({ done: 3, file: "DSC04205_neutral.jpg", file_percent: 100 }));
    expect(run.rows.map((r) => r.state)).toEqual(["failed", "done", "done"]);
    expect(queueItems(run)[0]).toMatchObject({ state: "failed", note: "The disk is full. Free some space, then export again." });
    expect(runBlurb(run)).toBe("2 photos saved to Lisbon export. 1 photo could not be saved.");
    expect(runTitle(run)).toBe("Exported");
  });

  it("does not let the final event turn a failed last file into a saved one", () => {
    let run = applyProgress(fresh(), event({ done: 2, file: "DSC04204_neutral.jpg" }));
    run = applyProgress(run, event({ done: 3, file: "DSC04205_neutral.jpg", file_percent: 100, error: "The folder can't be written to." }));
    run = applyProgress(run, event({ done: 3, file: "DSC04205_neutral.jpg", file_percent: 100 }));
    expect(run.rows[2]?.state).toBe("failed");
    expect(run.phase).toBe("finished");
  });

  it("says Not exported when nothing could be saved", () => {
    let run = fresh();
    for (let i = 1; i <= 3; i++) run = applyProgress(run, event({ done: i, file: NAMES[i - 1]!, file_percent: 100, error: "No space." }));
    expect(runTitle(run)).toBe("Not exported");
    expect(runBlurb(run)).toBe("3 photos could not be saved.");
  });

  it("marks a skipped file and says why in the blurb", () => {
    let run = applyProgress(fresh(), event({ done: 1, file: "DSC04203_neutral.jpg", file_percent: 100, skipped: true }));
    run = applyProgress(run, event({ done: 3, file: "DSC04205_neutral.jpg", file_percent: 100 }));
    expect(run.rows.map((r) => r.state)).toEqual(["skipped", "done", "done"]);
    expect(runBlurb(run)).toBe("2 photos saved to Lisbon export. 1 photo skipped because the file already exists.");
    expect(queueItems(run)[0]).toMatchObject({ state: "skipped" });
  });
});

describe("the cell in the header", () => {
  it("counts files while running", () => {
    const run = applyProgress(fresh(), event({ done: 1, file: "DSC04204_neutral.jpg", file_percent: 10 }));
    expect(cellLabel(run)).toBe("Exporting 1 of 3");
  });

  it("keeps the running note while running", () => {
    expect(runNote(fresh())).toBe("Originals are never changed. You can keep working while this runs; an update waits until the export is finished.");
  });
});
