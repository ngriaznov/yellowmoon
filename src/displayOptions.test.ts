import { beforeEach, describe, expect, it } from "vitest";
import { createMockApi } from "./api.mock";
import { applyDisplayOptions, watchDisplayOptions } from "./displayOptions";

const root = () => document.documentElement;
const has = (name: string) => root().hasAttribute(name);
const off = { reduce_transparency: false, reduce_motion: false, increase_contrast: false };

beforeEach(() => {
  for (const name of ["data-reduce-transparency", "data-reduce-motion", "data-increase-contrast"]) root().removeAttribute(name);
});

describe("applyDisplayOptions", () => {
  it("sets the attribute of each option that is on and removes the others", () => {
    applyDisplayOptions({ reduce_transparency: true, reduce_motion: false, increase_contrast: true });
    expect([has("data-reduce-transparency"), has("data-reduce-motion"), has("data-increase-contrast")]).toEqual([true, false, true]);
    applyDisplayOptions({ reduce_transparency: false, reduce_motion: true, increase_contrast: false });
    expect([has("data-reduce-transparency"), has("data-reduce-motion"), has("data-increase-contrast")]).toEqual([false, true, false]);
  });
});

describe("watchDisplayOptions", () => {
  it("follows the answer of the command at start", async () => {
    const api = createMockApi({ displayOptions: { ...off, reduce_transparency: true, increase_contrast: true } });
    const stop = watchDisplayOptions(api);
    await api.displayOptions();
    await Promise.resolve();
    expect([has("data-reduce-transparency"), has("data-reduce-motion"), has("data-increase-contrast")]).toEqual([true, false, true]);
    stop();
  });

  it("follows a later event, and removes an attribute when its value goes back to false", async () => {
    const api = createMockApi();
    const stop = watchDisplayOptions(api);
    await api.displayOptions();
    await Promise.resolve();
    expect(has("data-reduce-motion")).toBe(false);
    api.emitEvent("display_options", { ...off, reduce_motion: true });
    await api.flush();
    expect(has("data-reduce-motion")).toBe(true);
    api.emitEvent("display_options", { ...off, reduce_transparency: true });
    await api.flush();
    expect([has("data-reduce-transparency"), has("data-reduce-motion")]).toEqual([true, false]);
    api.emitEvent("display_options", off);
    await api.flush();
    expect(has("data-reduce-transparency")).toBe(false);
    stop();
  });

  it("keeps the event that arrives before the answer, and stops when asked", async () => {
    const api = createMockApi({ displayOptions: { ...off, reduce_motion: true } });
    const stop = watchDisplayOptions(api);
    api.emitEvent("display_options", { ...off, increase_contrast: true });
    await api.flush();
    await Promise.resolve();
    expect([has("data-reduce-motion"), has("data-increase-contrast")]).toEqual([false, true]);
    stop();
    api.emitEvent("display_options", off);
    await api.flush();
    expect(has("data-increase-contrast")).toBe(true);
  });

  it("leaves the page alone when the command is missing", async () => {
    const api = createMockApi();
    api.failNext("displayOptions", { scope: "app", message: "x" });
    const stop = watchDisplayOptions(api);
    await Promise.resolve();
    await Promise.resolve();
    expect(has("data-reduce-transparency")).toBe(false);
    stop();
  });
});
