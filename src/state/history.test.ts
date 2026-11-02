import { describe, expect, it } from "vitest";
import { applyPatch, defaultSettings, inputPatch, mainGradePatch, mainOf } from "./settingsGraph";
import { diffSettings, HISTORY_LIMIT, PhotoHistory } from "./history";

/** A manual contrast or saturation (`docs/19` section 5: Auto or a value). */
const manual = (value: number) => ({ kind: "manual", value }) as const;

describe("diffSettings", () => {
  it("is null when nothing changed", () => {
    expect(diffSettings(defaultSettings(), defaultSettings())).toBeNull();
  });

  it("lists only the nodes and fields that changed, with their old and new values", () => {
    const before = defaultSettings();
    const after = applyPatch(
      applyPatch(before, mainGradePatch({ contrast: manual(12) })),
      inputPatch({ exposure: { kind: "manual" as const, ev: 0.7 } }),
    );
    expect(diffSettings(before, after)).toEqual({
      before: { nodes: { input: { exposure: { kind: "auto" } }, "1": { grade: { contrast: { kind: "auto" } } } } },
      after: { nodes: { input: { exposure: { kind: "manual", ev: 0.7 } }, "1": { grade: { contrast: manual(12) } } } },
    });
  });

  it("compares nested values as a whole, so a wheel moving is one change", () => {
    const before = defaultSettings();
    const after = applyPatch(before, mainGradePatch({ color_balance: { ...mainOf(before).grade.color_balance, midtones: [0.2, -0.1] } }));
    const entry = diffSettings(before, after);
    expect(Object.keys(entry?.after.nodes?.["1"]?.grade ?? {})).toEqual(["color_balance"]);
  });

  it("copies what it keeps, so later changes to the settings don't rewrite history", () => {
    const before = defaultSettings();
    const after = applyPatch(before, mainGradePatch({ color_balance: { ...mainOf(before).grade.color_balance, midtones: [0.2, 0] } }));
    const entry = diffSettings(before, after);
    mainOf(after).grade.color_balance.midtones[0] = 0.9;
    expect(entry?.after.nodes?.["1"]?.grade?.color_balance?.midtones).toEqual([0.2, 0]);
  });
});

describe("PhotoHistory", () => {
  const entry = (n: number) => ({ before: mainGradePatch({ brightness: n - 1 }), after: mainGradePatch({ brightness: n }) });

  it("undoes and redoes in order", () => {
    const history = new PhotoHistory();
    history.push(entry(1));
    history.push(entry(2));
    expect(history.undo()).toEqual(entry(2));
    expect(history.undo()).toEqual(entry(1));
    expect(history.undo()).toBeNull();
    expect(history.redo()).toEqual(entry(1));
    expect(history.redo()).toEqual(entry(2));
    expect(history.redo()).toBeNull();
  });

  it("starts empty and reports what is available", () => {
    const history = new PhotoHistory();
    expect([history.canUndo, history.canRedo]).toEqual([false, false]);
    history.push(entry(1));
    expect([history.canUndo, history.canRedo]).toEqual([true, false]);
    history.undo();
    expect([history.canUndo, history.canRedo]).toEqual([false, true]);
  });

  it("a new change ends the redo chain", () => {
    const history = new PhotoHistory();
    history.push(entry(1));
    history.undo();
    history.push(entry(5));
    expect(history.canRedo).toBe(false);
    expect(history.redoDepth).toBe(0);
  });

  it("keeps 100 steps and forgets the oldest", () => {
    expect(HISTORY_LIMIT).toBe(100);
    const history = new PhotoHistory();
    for (let i = 1; i <= 130; i++) history.push(entry(i));
    expect(history.undoDepth).toBe(100);
    let oldest = null;
    for (let step = history.undo(); step; step = history.undo()) oldest = step;
    expect(oldest).toEqual(entry(31));
  });
});
