import { act, fireEvent, render, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { installManualRaf } from "../test/raf";
import type { ManualRaf } from "../test/raf";
import { useDraft, useFrameEmitter, usePointerDrag } from "./hooks";

let raf: ManualRaf;
beforeEach(() => {
  raf = installManualRaf();
});
afterEach(() => {
  raf.restore();
  vi.useRealTimers();
});

describe("useFrameEmitter", () => {
  it("emits at most once per animation frame, with the newest value", () => {
    const emit = vi.fn();
    const { result } = renderHook(() => useFrameEmitter<number>(emit));
    result.current.push(1);
    result.current.push(2);
    result.current.push(3);
    expect(emit).not.toHaveBeenCalled();
    expect(raf.pending).toBe(1);
    raf.flush();
    expect(emit).toHaveBeenCalledTimes(1);
    expect(emit).toHaveBeenCalledWith(3);
    result.current.push(4);
    raf.flush();
    expect(emit).toHaveBeenLastCalledWith(4);
    expect(emit).toHaveBeenCalledTimes(2);
  });

  it("drops a waiting value on cancel and on unmount", () => {
    const emit = vi.fn();
    const { result, unmount } = renderHook(() => useFrameEmitter<number>(emit));
    result.current.push(1);
    result.current.cancel();
    raf.flush();
    expect(emit).not.toHaveBeenCalled();
    result.current.push(2);
    unmount();
    raf.flush();
    expect(emit).not.toHaveBeenCalled();
  });

  it("calls the latest emit callback", () => {
    const first = vi.fn();
    const second = vi.fn();
    const { result, rerender } = renderHook(({ emit }) => useFrameEmitter<number>(emit), { initialProps: { emit: first } });
    result.current.push(1);
    rerender({ emit: second });
    raf.flush();
    expect(first).not.toHaveBeenCalled();
    expect(second).toHaveBeenCalledWith(1);
  });
});

describe("useDraft", () => {
  const same = (a: number, b: number): boolean => a === b;

  it("shows the controlled value until a draft is held", () => {
    const { result } = renderHook(() => useDraft<number>(5, same));
    expect(result.current.shown).toBe(5);
    act(() => result.current.drag(8));
    expect(result.current.shown).toBe(8);
    expect(result.current.dragging).toBe(true);
  });

  it("holds a finished change until the controlled value catches up", () => {
    const { result, rerender } = renderHook(({ value }) => useDraft<number>(value, same), { initialProps: { value: 5 } });
    act(() => result.current.settle(9));
    expect(result.current.shown).toBe(9);
    rerender({ value: 7 }); // an older reply arrives: keep showing 9
    expect(result.current.shown).toBe(9);
    rerender({ value: 9 });
    expect(result.current.shown).toBe(9);
    rerender({ value: 4 }); // later, someone else changes it
    expect(result.current.shown).toBe(4);
  });

  it("gives up when the value never arrives", () => {
    vi.useFakeTimers();
    const { result } = renderHook(() => useDraft<number>(5, same));
    act(() => result.current.settle(9));
    expect(result.current.shown).toBe(9);
    act(() => {
      vi.advanceTimersByTime(1100);
    });
    expect(result.current.shown).toBe(5);
  });

  it("drops the draft at once when it already equals the controlled value", () => {
    const { result } = renderHook(() => useDraft<number>(5, same));
    act(() => result.current.drag(6));
    act(() => result.current.settle(5));
    expect(result.current.shown).toBe(5);
  });
});

describe("usePointerDrag", () => {
  function Probe({ events }: { events: string[] }) {
    const onPointerDown = usePointerDrag({
      start: () => {
        events.push("start");
      },
      move: (e) => {
        events.push(`move ${e.clientX}`);
      },
      end: (_e, cancelled) => {
        events.push(cancelled ? "cancel" : "end");
      },
    });
    return <div data-testid="el" onPointerDown={onPointerDown} />;
  }

  it("follows the pointer on the window and ends on release", () => {
    const events: string[] = [];
    const { getByTestId } = render(<Probe events={events} />);
    fireEvent.pointerDown(getByTestId("el"), { button: 0, clientX: 1 });
    fireEvent.pointerMove(window, { clientX: 5 });
    fireEvent.pointerMove(window, { clientX: 9 });
    fireEvent.pointerUp(window, { clientX: 9 });
    fireEvent.pointerMove(window, { clientX: 12 });
    expect(events).toEqual(["start", "move 5", "move 9", "end"]);
  });

  it("reports a cancelled drag", () => {
    const events: string[] = [];
    const { getByTestId } = render(<Probe events={events} />);
    fireEvent.pointerDown(getByTestId("el"), { button: 0 });
    fireEvent.pointerCancel(window);
    expect(events).toEqual(["start", "cancel"]);
  });

  it("ignores other buttons", () => {
    const events: string[] = [];
    const { getByTestId } = render(<Probe events={events} />);
    fireEvent.pointerDown(getByTestId("el"), { button: 2 });
    fireEvent.pointerMove(window, { clientX: 5 });
    expect(events).toEqual([]);
  });

  it("stops listening when unmounted mid-drag", () => {
    const events: string[] = [];
    const { getByTestId, unmount } = render(<Probe events={events} />);
    fireEvent.pointerDown(getByTestId("el"), { button: 0 });
    unmount();
    fireEvent.pointerMove(window, { clientX: 5 });
    expect(events).toEqual(["start"]);
  });
});
