/**
 * What the page keeps of the core's reply to a patch (U-02): a reply that says what the page already shows changes nothing and tells nobody,
 * and one that differs keeps the parts that did not.
 */
import { afterEach, describe, expect, it, vi } from "vitest";
import { createMockApi } from "../api.mock";
import { mainGradePatch } from "./settingsGraph";
import { createStore } from "./store";

const stops: Array<() => void> = [];
afterEach(() => {
  for (const stop of stops.splice(0)) stop();
});

async function opened() {
  const api = createMockApi({ speed: Number.POSITIVE_INFINITY, current: 3 });
  const store = createStore(api, 0);
  stops.push(store.connect());
  await store.load();
  await store.openDialog("files");
  api.flush();
  await vi.waitFor(() => expect(store.getState().session?.photos).toHaveLength(12));
  await vi.waitFor(() => expect(store.currentSettings()).not.toBeNull());
  return { api, store };
}

describe("the reply to a patch", () => {
  it("is not stored when it says what the page already shows: one change of the state for a patch, not two", async () => {
    const { store } = await opened();
    // Settled: the first reads of the photo's settings are in.
    await vi.waitFor(() => expect(store.currentSettings()).not.toBeNull());
    // How many times the settings of the photo became another object (the store tells its listeners for other reasons as well: the undo flags).
    let was = store.currentSettings();
    let changes = 0;
    const stop = store.subscribe(() => {
      const now = store.currentSettings();
      if (now !== was) changes += 1;
      was = now;
    });
    await store.update(mainGradePatch({ brightness: 12 }));
    stop();
    expect(changes, "the optimistic copy is one change; the reply that equals it is none").toBe(1);
    expect(store.currentSettings()?.values.main.grade.brightness).toBe(12);
  });

  it("keeps the parts of the settings that the reply did not change, so a control that reads one of them does not draw", async () => {
    const { store } = await opened();
    const before = store.currentSettings();
    await store.update(mainGradePatch({ brightness: 7 }));
    const after = store.currentSettings();
    expect(after).not.toBe(before);
    expect(after?.values.output).toBe(before?.values.output);
    expect(after?.values.input).toBe(before?.values.input);
    expect(after?.settings.nodes.links).toBe(before?.settings.nodes.links);
  });
});
