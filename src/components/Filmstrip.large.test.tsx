import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { useState } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { DevelopList, ExportQueueList } from "./ExportQueueList";
import type { DevelopItem } from "./ExportQueueList";
import { Filmstrip } from "./Filmstrip";
import type { FilmstripPhoto } from "./Filmstrip";
import { useWindowedRange } from "./windowed";

// A folder of 2000 photos (docs/26 H-09): the strip, the Start list and the export queue draw only what is in view and a margin,
// and everything that worked on a thumbnail (arrows, P, the context menu, the reveal of the photo on screen) still works on one that
// was off screen. jsdom has no layout, so the scroller gets a width and a working scrollLeft/scrollTop here, and the items a position.

const N = 2000;
const CELL = 120;

const photos = (n = N): FilmstripPhoto[] => Array.from({ length: n }, (_, i) => ({ id: i + 1, name: `DSC${String(i + 1).padStart(5, "0")}`, src: `t${i}.png` }));

let scrollLeft = 0;
let scrollTop = 0;
const proto = HTMLElement.prototype;
const saved = new Map<string, PropertyDescriptor | undefined>();
function patch(name: string, descriptor: PropertyDescriptor): void {
  saved.set(name, Object.getOwnPropertyDescriptor(proto, name));
  Object.defineProperty(proto, name, { configurable: true, ...descriptor });
}

beforeEach(() => {
  scrollLeft = 0;
  scrollTop = 0;
  patch("clientWidth", { get: () => 1000 });
  patch("clientHeight", { get: () => 400 });
  patch("scrollLeft", { get: () => scrollLeft, set: (v: number) => void (scrollLeft = Math.max(0, v)) });
  patch("scrollTop", { get: () => scrollTop, set: (v: number) => void (scrollTop = Math.max(0, v)) });
  patch("getClientRects", { value: () => [{}] });
});

afterEach(() => {
  for (const [name, d] of saved) {
    if (d) Object.defineProperty(proto, name, d);
    else delete (proto as unknown as Record<string, unknown>)[name];
  }
  saved.clear();
  vi.restoreAllMocks();
});

/** Gives every thumbnail a box by its place in the list, as a browser would lay them out (index from the photo id). */
function layOut(): void {
  vi.spyOn(Element.prototype, "getBoundingClientRect").mockImplementation(function (this: Element) {
    const id = (this as HTMLElement).dataset?.photoId;
    if (id !== undefined) {
      const left = 24 + (Number(id) - 1) * CELL - scrollLeft;
      return { left, right: left + 112, top: 0, bottom: 80, width: 112, height: 80, x: left, y: 0, toJSON: () => ({}) };
    }
    return { left: 0, right: 1000, top: 0, bottom: 400, width: 1000, height: 400, x: 0, y: 0, toJSON: () => ({}) };
  });
}

function Harness({ start = 1, onPick, onRemove }: { start?: number; onPick?: (id: number | string) => void; onRemove?: (id: number | string) => void }) {
  const [current, setCurrent] = useState<number | string>(start);
  return <Filmstrip photos={photos()} currentId={current} onSelect={setCurrent} onTogglePick={onPick} onRemove={onRemove} />;
}

describe("Filmstrip with 2000 photos", () => {
  it("draws a few dozen thumbnails, not 2000, and says all 2000 in its caption", () => {
    layOut();
    render(<Harness />);
    const drawn = screen.getAllByRole("button");
    expect(drawn.length).toBeGreaterThan(5);
    expect(drawn.length).toBeLessThan(100);
    expect(screen.getByText("1 of 2000 · 0 picked")).toBeInTheDocument();
  });

  it("tells the shell which photos it draws (docs/26 H-11), and only a windowed strip does", () => {
    layOut();
    const onInView = vi.fn();
    const { unmount } = render(<Filmstrip photos={photos()} currentId={1} onInView={onInView} />);
    const ids = onInView.mock.calls.at(-1)![0] as number[];
    expect(ids[0]).toBe(1);
    expect(ids.length).toBeGreaterThan(5);
    expect(ids.length).toBeLessThan(100);
    unmount();
    const few = vi.fn();
    render(<Filmstrip photos={photos(20)} currentId={1} onInView={few} />);
    expect(few).not.toHaveBeenCalled();
  });

  it("makes the scroll bar true: the space of the thumbnails that are not drawn is kept", () => {
    layOut();
    const { container } = render(<Harness />);
    const last = [...container.querySelectorAll<HTMLElement>(".ym-filmstrip__thumb")].at(-1)!;
    // Estimated pitch until a browser has measured: the rest of the 2000 thumbnails stand in as the last one's right margin.
    const drawn = container.querySelectorAll(".ym-filmstrip__thumb").length;
    expect(parseFloat(last.style.marginRight)).toBeCloseTo((N - drawn) * 120, 0);
  });

  it("brings a current photo that starts far off into view, and draws it", () => {
    layOut();
    render(<Harness start={1500} />);
    expect(screen.getByRole("button", { name: /^Show DSC01500/ })).toHaveAttribute("aria-current", "true");
    expect(screen.getAllByRole("button").length).toBeLessThan(100);
    expect(scrollLeft).toBeGreaterThan(1000);
  });

  it("moves with the arrow keys through the edge of what is drawn, and keeps the focus on the photo on screen", () => {
    layOut();
    render(<Harness start={30} />);
    for (let i = 0; i < 40; i++) {
      fireEvent.keyDown(document.activeElement && document.activeElement !== document.body ? document.activeElement : screen.getByRole("button", { name: /^Show DSC00030/ }), { key: "ArrowRight" });
    }
    const now = screen.getByRole("button", { name: /^Show DSC00070/ });
    expect(now).toHaveAttribute("aria-current", "true");
    expect(now).toHaveFocus();
    expect(screen.getAllByRole("button").length).toBeLessThan(100);
  });

  it("toggles the pick of a photo that was off screen when it was made current, with P", () => {
    layOut();
    const onPick = vi.fn();
    render(<Harness start={1800} onPick={onPick} />);
    fireEvent.keyDown(screen.getByRole("button", { name: /^Show DSC01800/ }), { key: "p" });
    expect(onPick).toHaveBeenCalledWith(1800);
  });

  it("gives the strip one tab stop even when the photo on screen is not drawn", async () => {
    layOut();
    render(<Harness />);
    const stops = screen.getAllByRole("button").filter((b) => b.tabIndex === 0);
    expect(stops).toHaveLength(1);
    // The person scrolls the strip far away: the photo on screen is no longer drawn, and the first thumbnail drawn is the tab stop.
    act(() => {
      scrollLeft = 90_000;
      fireEvent.scroll(document.querySelector(".ym-filmstrip__scroll")!);
    });
    return waitFor(() => {
      expect(screen.queryByRole("button", { name: /^Show DSC00001/ })).toBeNull();
      const now = screen.getAllByRole("button").filter((b) => b.tabIndex === 0);
      expect(now).toHaveLength(1);
      expect(now[0]).toBe(screen.getAllByRole("button")[0]);
    });
  });

  it("opens the context menu on a thumbnail far from the first", async () => {
    layOut();
    const onRemove = vi.fn();
    render(<Harness start={900} onRemove={onRemove} />);
    fireEvent.contextMenu(screen.getByRole("button", { name: /^Show DSC00900/ }), { clientX: 10, clientY: 10 });
    const item = await screen.findByRole("menuitem", { name: "Remove from filmstrip" });
    fireEvent.click(item);
    expect(onRemove).toHaveBeenCalledWith(900);
  });

  it("draws every thumbnail of a short folder, as before", () => {
    render(<Filmstrip photos={photos(60)} currentId={1} />);
    expect(screen.getAllByRole("button")).toHaveLength(60);
  });

  it("has lazy thumbnails", () => {
    const { container } = render(<Harness />);
    const img = container.querySelector("img")!;
    expect(img).toHaveAttribute("loading", "lazy");
    expect(img).toHaveAttribute("decoding", "async");
  });
});

describe("the Start list and the export queue with 2000 rows", () => {
  const develop: DevelopItem[] = Array.from({ length: N }, (_, i) => ({ id: i, name: `DSC${i}`, state: "ready" }));

  it("draws the rows in view and a margin, with the space of the others as padding", () => {
    render(<DevelopList items={develop} />);
    const rows = screen.getAllByRole("listitem");
    expect(rows.length).toBeGreaterThan(5);
    expect(rows.length).toBeLessThan(100);
    const list = screen.getByRole("list", { name: "Photos being developed" });
    expect(parseFloat(list.style.paddingBottom)).toBeGreaterThan(50_000);
  });

  it("draws the whole queue of a short run", () => {
    const items = Array.from({ length: 30 }, (_, i) => ({ id: i, file: `f${i}.jpg`, state: "waiting" as const }));
    render(<ExportQueueList items={items} />);
    expect(screen.getAllByRole("listitem")).toHaveLength(30);
  });

  it("draws a window of a long queue and follows the scroll of the element that scrolls", async () => {
    const items = Array.from({ length: N }, (_, i) => ({ id: i, file: `f${i}.jpg`, state: "waiting" as const }));
    render(
      <div style={{ overflowY: "auto" }} data-testid="scroller">
        <ExportQueueList items={items} />
      </div>,
    );
    expect(screen.getAllByRole("listitem").length).toBeLessThan(100);
    expect(screen.queryByText("f1000.jpg")).toBeNull();
    act(() => {
      scrollTop = 53 * 1000;
      fireEvent.scroll(screen.getByTestId("scroller"));
    });
    await waitFor(() => expect(screen.getByText("f1000.jpg")).toBeInTheDocument());
    expect(screen.queryByText("f0.jpg")).toBeNull();
    expect(screen.getAllByRole("listitem").length).toBeLessThan(100);
  });
});

describe("useWindowedRange", () => {
  function Probe({ count }: { count: number }) {
    const win = useWindowedRange({ count, axis: "y", list: { current: null }, itemSelector: "li", estimate: 50 });
    return <output>{`${win.start}-${win.end}-${win.windowed}`}</output>;
  }

  it("draws a short list whole and a long one in part", () => {
    const { rerender } = render(<Probe count={40} />);
    expect(screen.getByRole("status")).toHaveTextContent("0-40-false");
    rerender(<Probe count={5000} />);
    const text = screen.getByRole("status").textContent ?? "";
    const [start, end, windowed] = text.split("-");
    expect(start).toBe("0");
    expect(Number(end)).toBeLessThan(100);
    expect(windowed).toBe("true");
  });
});
