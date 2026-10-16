import { act, fireEvent, render, renderHook, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { DropZone } from "./DropZone";
import { domDropSource, filterArwPaths, isArwPath, pathsFromDataTransfer, useWindowDrop } from "./useWindowDrop";
import type { DropHandlers, DropSource } from "./useWindowDrop";

const files = (...names: string[]) => ({
  types: ["Files"],
  files: names.map((name) => ({ name })),
  dropEffect: "none",
});

describe("DropZone", () => {
  it("is a glasscn Card, with an Item for its notice", () => {
    render(<DropZone title="Drop files or a folder" notice="No ARW files in what you dropped." />);
    expect(screen.getByRole("group", { name: "Drop files or a folder" })).toHaveAttribute("data-slot", "card");
    expect(screen.getByRole("alert")).toHaveAttribute("data-slot", "item");
  });

  it("is a named group with a title, a hint and actions", () => {
    render(
      <DropZone title="Drop files or a folder" hint="ARW only, as many as you like">
        <button type="button">Open files…</button>
      </DropZone>,
    );
    expect(screen.getByRole("group", { name: "Drop files or a folder" })).toBeInTheDocument();
    expect(screen.getByText("ARW only, as many as you like")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Open files…" })).toBeInTheDocument();
  });

  it("draws the large icon, but not in the compact variant", () => {
    const { container, rerender } = render(<DropZone title="Drop" />);
    expect(container.querySelector(".ym-drop-zone__icon")).not.toBeNull();
    rerender(<DropZone title="Or drop the zip here" variant="compact" />);
    expect(container.querySelector(".ym-drop-zone__icon")).toBeNull();
    expect(screen.getByRole("group")).toHaveAttribute("data-variant", "compact");
  });

  it("highlights while active", () => {
    const { rerender } = render(<DropZone title="Drop" />);
    expect(screen.getByRole("group")).not.toHaveAttribute("data-active");
    rerender(<DropZone title="Drop" active />);
    expect(screen.getByRole("group")).toHaveAttribute("data-active", "true");
  });

  it("announces a notice about the last drop", () => {
    render(<DropZone title="Drop" notice="No ARW files in what you dropped." />);
    expect(screen.getByRole("alert")).toHaveTextContent("No ARW files in what you dropped.");
  });

  it("accepts files dropped on it and reports their names", () => {
    const onDropPaths = vi.fn();
    render(<DropZone title="Drop" onDropPaths={onDropPaths} />);
    const zone = screen.getByRole("group");
    fireEvent.dragEnter(zone, { dataTransfer: files("DSC04201.ARW") });
    expect(zone).toHaveAttribute("data-active", "true");
    fireEvent.dragOver(zone, { dataTransfer: files("DSC04201.ARW") });
    fireEvent.drop(zone, { dataTransfer: files("DSC04201.ARW", "DSC04202.ARW") });
    expect(onDropPaths).toHaveBeenCalledWith(["DSC04201.ARW", "DSC04202.ARW"]);
    expect(zone).not.toHaveAttribute("data-active");
  });

  it("ignores drags that carry no files, such as selected text", () => {
    const onDropPaths = vi.fn();
    render(<DropZone title="Drop" onDropPaths={onDropPaths} />);
    const zone = screen.getByRole("group");
    fireEvent.dragEnter(zone, { dataTransfer: { types: ["text/plain"], files: [] } });
    fireEvent.drop(zone, { dataTransfer: { types: ["text/plain"], files: [] } });
    expect(onDropPaths).not.toHaveBeenCalled();
    expect(zone).not.toHaveAttribute("data-active");
  });

  it("stays highlighted while the drag moves between its children, and clears when it leaves", () => {
    render(
      <DropZone title="Drop" onDropPaths={() => undefined}>
        <button type="button">Child</button>
      </DropZone>,
    );
    const zone = screen.getByRole("group");
    const child = screen.getByRole("button");
    const data = { dataTransfer: files("a.ARW") };
    fireEvent.dragEnter(zone, data);
    fireEvent.dragEnter(child, data);
    fireEvent.dragLeave(zone, data);
    expect(zone).toHaveAttribute("data-active", "true");
    fireEvent.dragLeave(child, data);
    expect(zone).not.toHaveAttribute("data-active");
  });
});

describe("paths from a drop", () => {
  it("uses the real path when the platform gives one, else the name", () => {
    expect(pathsFromDataTransfer({ files: [{ name: "a.ARW", path: "/Users/me/a.ARW" }, { name: "b.ARW" }] } as unknown as DataTransfer)).toEqual(["/Users/me/a.ARW", "b.ARW"]);
    expect(pathsFromDataTransfer(null)).toEqual([]);
  });

  it("recognises Sony raw files in any case", () => {
    expect(isArwPath("DSC04201.ARW")).toBe(true);
    expect(isArwPath("/x/dsc.arw")).toBe(true);
    expect(isArwPath("photo.jpg")).toBe(false);
    expect(isArwPath("DSC.ARW.zip")).toBe(false);
  });

  it("keeps ARW files and possible folders and ignores everything else", () => {
    expect(filterArwPaths(["a.ARW", "b.jpg", "Lisbon", "/Users/me/Lisbon trip", "c.dng", "d.arw"])).toEqual({
      accepted: ["a.ARW", "Lisbon", "/Users/me/Lisbon trip", "d.arw"],
      ignored: ["b.jpg", "c.dng"],
    });
    expect(filterArwPaths(["a.jpg", "b.png"]).accepted).toEqual([]);
  });

  it("takes a folder whose name has a dot in it for a folder, not for a file", () => {
    const { accepted, ignored } = filterArwPaths(["/Users/me/Lisbon 2026.10", "/Users/me/Trip. Day 2/", "/Users/me/v1.0", "C:\\Photos\\Spain.2026", "/Users/me/notes.txt", "/Users/me/clip.mp4", "/Users/me/a.7z"]);
    expect(accepted).toEqual(["/Users/me/Lisbon 2026.10", "/Users/me/Trip. Day 2/", "/Users/me/v1.0", "C:\\Photos\\Spain.2026"]);
    expect(ignored).toEqual(["/Users/me/notes.txt", "/Users/me/clip.mp4", "/Users/me/a.7z"]);
  });
});

describe("useWindowDrop", () => {
  function fakeSource() {
    let handlers: DropHandlers | null = null;
    const unsubscribe = vi.fn();
    const source: DropSource = {
      subscribe(h) {
        handlers = h;
        return unsubscribe;
      },
    };
    return { source, unsubscribe, get handlers() {
      return handlers as DropHandlers;
    } };
  }

  it("tracks a drag over the window and hands over dropped paths", () => {
    const fake = fakeSource();
    const onDrop = vi.fn();
    const { result } = renderHook(() => useWindowDrop({ onDrop, source: fake.source }));
    expect(result.current.dragging).toBe(false);
    act(() => fake.handlers.onEnter());
    expect(result.current.dragging).toBe(true);
    act(() => fake.handlers.onLeave());
    expect(result.current.dragging).toBe(false);
    act(() => fake.handlers.onDrop(["/a/DSC04201.ARW"]));
    expect(onDrop).toHaveBeenCalledWith(["/a/DSC04201.ARW"]);
  });

  it("unsubscribes when unmounted or disabled", () => {
    const fake = fakeSource();
    const { rerender, unmount } = renderHook(({ enabled }) => useWindowDrop({ onDrop: () => undefined, enabled, source: fake.source }), { initialProps: { enabled: true } });
    rerender({ enabled: false });
    expect(fake.unsubscribe).toHaveBeenCalledTimes(1);
    rerender({ enabled: true });
    unmount();
    expect(fake.unsubscribe).toHaveBeenCalledTimes(2);
  });

  it("works on the window's own DOM events", () => {
    const onDrop = vi.fn();
    const { result } = renderHook(() => useWindowDrop({ onDrop, source: domDropSource(window) }));
    act(() => {
      fireEvent.dragEnter(window, { dataTransfer: files("a.ARW") });
    });
    expect(result.current.dragging).toBe(true);
    act(() => {
      fireEvent.drop(window, { dataTransfer: files("a.ARW", "b.jpg") });
    });
    expect(result.current.dragging).toBe(false);
    expect(onDrop).toHaveBeenCalledWith(["a.ARW", "b.jpg"]);
  });

  it("allows the drop by cancelling dragover for file drags", () => {
    renderHook(() => useWindowDrop({ onDrop: () => undefined, source: domDropSource(window) }));
    const event = new Event("dragover", { cancelable: true, bubbles: true });
    Object.defineProperty(event, "dataTransfer", { value: files("a.ARW") });
    window.dispatchEvent(event);
    expect(event.defaultPrevented).toBe(true);
  });
});
