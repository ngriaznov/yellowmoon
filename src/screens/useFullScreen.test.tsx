/**
 * The window side of full screen (`useFullScreen`, `windowFollower`): the app's window goes into macOS full screen with the view, and
 * back only when the view put it there. A fake window stands in for Tauri's (`getCurrentWindow()`), which only exists in the app.
 */
import { act, renderHook, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { useFullScreen, windowFollower } from "./useFullScreen";
import type { FullScreenWindow } from "./useFullScreen";

/** A window that records what it is asked and answers like Tauri's. */
function fakeWindow(startsFullScreen = false) {
  let fullScreen = startsFullScreen;
  const calls: string[] = [];
  const win: FullScreenWindow = {
    isFullscreen: vi.fn(async () => {
      calls.push("is");
      return fullScreen;
    }),
    setFullscreen: vi.fn(async (on: boolean) => {
      calls.push(on ? "enter" : "exit");
      fullScreen = on;
    }),
  };
  return { win, calls, get fullScreen() { return fullScreen; }, getWindow: () => Promise.resolve(win) };
}

/** Waits for the window calls queued so far. */
const settle = (): Promise<void> => act(async () => {
  await new Promise((resolve) => setTimeout(resolve, 0));
});

describe("windowFollower", () => {
  it("puts the window in full screen, and back", async () => {
    const w = fakeWindow();
    const follow = windowFollower(w.getWindow);
    await follow.enter();
    expect(w.fullScreen).toBe(true);
    await follow.leave();
    expect(w.fullScreen).toBe(false);
    expect(w.calls).toEqual(["is", "enter", "exit"]);
  });

  it("leaves a window that was already in full screen as it is, going in and coming out", async () => {
    const w = fakeWindow(true);
    const follow = windowFollower(w.getWindow);
    await follow.enter();
    await follow.leave();
    expect(w.fullScreen).toBe(true);
    expect(w.win.setFullscreen).not.toHaveBeenCalled();
  });

  it("runs the calls in order when F is pressed twice quickly", async () => {
    const w = fakeWindow();
    const follow = windowFollower(w.getWindow);
    void follow.enter();
    void follow.leave();
    void follow.enter();
    await follow.leave();
    expect(w.calls).toEqual(["is", "enter", "exit", "is", "enter", "exit"]);
    expect(w.fullScreen).toBe(false);
  });

  it("does nothing outside the app, where there is no window to ask", async () => {
    const follow = windowFollower(() => Promise.resolve(null));
    await expect(follow.enter()).resolves.toBeUndefined();
    await expect(follow.leave()).resolves.toBeUndefined();
  });

  it("swallows a window call that fails, and the next one still runs", async () => {
    const w = fakeWindow();
    vi.mocked(w.win.setFullscreen).mockRejectedValueOnce(new Error("not allowed"));
    const follow = windowFollower(w.getWindow);
    await expect(follow.enter()).resolves.toBeUndefined();
    expect(w.fullScreen).toBe(false);
    // It never got in, so leaving asks nothing; the next F tries again.
    await follow.leave();
    await follow.enter();
    expect(w.fullScreen).toBe(true);
  });
});

describe("useFullScreen", () => {
  it("toggles the view and the window with it", async () => {
    const w = fakeWindow();
    const { result } = renderHook(() => useFullScreen(true, w.getWindow));
    expect(result.current.on).toBe(false);
    act(() => result.current.toggle());
    expect(result.current.on).toBe(true);
    await waitFor(() => expect(w.fullScreen).toBe(true));
    act(() => result.current.toggle());
    expect(result.current.on).toBe(false);
    await waitFor(() => expect(w.fullScreen).toBe(false));
  });

  it("puts the window back when the screen goes (⌘W) while in full screen", async () => {
    const w = fakeWindow();
    const { result, unmount } = renderHook(() => useFullScreen(true, w.getWindow));
    act(() => result.current.toggle());
    await waitFor(() => expect(w.fullScreen).toBe(true));
    unmount();
    await waitFor(() => expect(w.fullScreen).toBe(false));
  });

  it("does not take the window out of a full screen the person chose", async () => {
    const w = fakeWindow(true);
    const { result } = renderHook(() => useFullScreen(true, w.getWindow));
    act(() => result.current.toggle());
    await settle();
    act(() => result.current.leave());
    await settle();
    expect(w.fullScreen).toBe(true);
    expect(w.win.setFullscreen).not.toHaveBeenCalled();
  });

  it("is off while not allowed (Export, Crop), ends when they open, and does not come back when they close", async () => {
    const w = fakeWindow();
    const { result, rerender } = renderHook(({ allowed }) => useFullScreen(allowed, w.getWindow), { initialProps: { allowed: false } });
    act(() => result.current.toggle());
    expect(result.current.on).toBe(false);
    rerender({ allowed: true });
    act(() => result.current.toggle());
    expect(result.current.on).toBe(true);
    await waitFor(() => expect(w.fullScreen).toBe(true));
    rerender({ allowed: false });
    expect(result.current.on).toBe(false);
    await waitFor(() => expect(w.fullScreen).toBe(false));
    rerender({ allowed: true });
    await settle();
    expect(result.current.on).toBe(false);
    expect(w.fullScreen).toBe(false);
  });

  it("takes the focus from a control that is hidden now, but not from the photo's own layers", () => {
    const button = document.createElement("button");
    const cell = document.createElement("main");
    cell.setAttribute("data-viewport", "");
    const layer = document.createElement("div");
    layer.tabIndex = 0;
    cell.append(layer);
    document.body.append(button, cell);
    try {
      const outside = renderHook(() => useFullScreen(true, () => Promise.resolve(null)));
      button.focus();
      act(() => outside.result.current.toggle());
      expect(document.activeElement).toBe(document.body);
      outside.unmount();

      const inside = renderHook(() => useFullScreen(true, () => Promise.resolve(null)));
      layer.focus();
      act(() => inside.result.current.toggle());
      expect(document.activeElement).toBe(layer);
      inside.unmount();
    } finally {
      button.remove();
      cell.remove();
    }
  });
});
