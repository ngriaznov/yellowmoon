import { act, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { PhotoCell } from "./PhotoCell";

class FakeResizeObserver {
  static instances: FakeResizeObserver[] = [];
  constructor(private callback: () => void) {
    FakeResizeObserver.instances.push(this);
  }
  observe = vi.fn();
  disconnect = vi.fn();
  fire(): void {
    this.callback();
  }
}

const box = (left: number, top: number, width: number, height: number) =>
  ({ left, top, width, height, right: left + width, bottom: top + height, x: left, y: top, toJSON: () => ({}) }) as DOMRect;

beforeEach(() => {
  FakeResizeObserver.instances = [];
  vi.stubGlobal("ResizeObserver", FakeResizeObserver);
});
afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("PhotoCell", () => {
  it("is the photo section, a data-viewport cell with an image role", () => {
    const { container } = render(<PhotoCell imageLabel="Photo DSC04204, Neutral · 100%" native={false} />);
    expect(screen.getByRole("main", { name: "Photo" })).toHaveAttribute("data-viewport");
    expect(container.querySelector("[data-viewport]")).toHaveClass("ym-photo-cell");
    expect(screen.getByRole("img", { name: "Photo DSC04204, Neutral · 100%" })).toBeInTheDocument();
  });

  it("draws the photo as an image in the browser and the mock", () => {
    const { container } = render(<PhotoCell imageSrc="photo.svg" native={false} />);
    expect(container.querySelector("img")).toHaveAttribute("src", "photo.svg");
    expect(container.firstChild).not.toHaveAttribute("data-native");
  });

  it("draws nothing in the app: the native layer shows behind a transparent cell", () => {
    const { container } = render(<PhotoCell imageSrc="photo.svg" native />);
    expect(container.querySelector("img")).toBeNull();
    expect(container.firstChild).toHaveAttribute("data-native", "true");
  });

  it("shows the look chip and the camera line", () => {
    render(<PhotoCell lookChip="Neutral · 100%" cameraLine="Sony A1 · DSC04204 · ISO 400 · 1/125 · f/2.8" native={false} />);
    expect(screen.getByText("Neutral · 100%")).toHaveClass("ym-photo-cell__chip");
    expect(screen.getByText("Sony A1 · DSC04204 · ISO 400 · 1/125 · f/2.8")).toHaveClass("ym-photo-cell__camera");
  });

  it("shows the Before label and hides the look chip in Before mode", () => {
    render(<PhotoCell lookChip="Neutral · 100%" before native={false} />);
    expect(screen.getByText("Before · camera log")).toHaveClass("ym-photo-cell__before");
    expect(screen.queryByText("Neutral · 100%")).toBeNull();
  });

  it("shows the pixel readout when given one", () => {
    const { rerender } = render(<PhotoCell pixelReadout="R 182 G 164 B 140" native={false} />);
    expect(screen.getByText("R 182 G 164 B 140")).toHaveClass("ym-photo-cell__readout");
    rerender(<PhotoCell pixelReadout={null} native={false} />);
    expect(screen.queryByText("R 182 G 164 B 140")).toBeNull();
  });

  it("lays extra layers over the photo", () => {
    render(
      <PhotoCell native={false}>
        <div data-testid="crop" />
      </PhotoCell>,
    );
    expect(screen.getByTestId("crop").parentElement).toHaveClass("ym-photo-cell__overlay");
  });

  it("passes other attributes through to the section", () => {
    render(<PhotoCell native={false} data-testid="cell" style={{ gridRow: 2 }} />);
    expect(screen.getByTestId("cell").style.gridRow).toBe("2");
  });
});

describe("PhotoCell rectangle", () => {
  it("reports its rectangle and the device scale on mount", () => {
    vi.spyOn(Element.prototype, "getBoundingClientRect").mockReturnValue(box(10, 75, 1030, 686.67));
    vi.stubGlobal("devicePixelRatio", 2);
    const onRect = vi.fn();
    render(<PhotoCell onRect={onRect} native={false} />);
    expect(onRect).toHaveBeenCalledTimes(1);
    expect(onRect).toHaveBeenCalledWith({ x: 10, y: 75, width: 1030, height: 686.67, scale: 2 });
  });

  it("reports again when the cell resizes, and not when nothing changed", () => {
    const spy = vi.spyOn(Element.prototype, "getBoundingClientRect").mockReturnValue(box(0, 75, 900, 600));
    const onRect = vi.fn();
    render(<PhotoCell onRect={onRect} native={false} />);
    expect(FakeResizeObserver.instances).toHaveLength(1);
    expect(FakeResizeObserver.instances[0]?.observe).toHaveBeenCalled();
    act(() => FakeResizeObserver.instances[0]?.fire());
    expect(onRect).toHaveBeenCalledTimes(1);
    spy.mockReturnValue(box(0, 75, 1000, 666.67));
    act(() => FakeResizeObserver.instances[0]?.fire());
    expect(onRect).toHaveBeenCalledTimes(2);
    expect(onRect).toHaveBeenLastCalledWith(expect.objectContaining({ width: 1000, height: 666.67 }));
  });

  it("reports when the cell moves without resizing (the window gets wider)", () => {
    const spy = vi.spyOn(Element.prototype, "getBoundingClientRect").mockReturnValue(box(0, 75, 1000, 666));
    const onRect = vi.fn();
    render(<PhotoCell onRect={onRect} native={false} />);
    spy.mockReturnValue(box(120, 75, 1000, 666));
    fireEvent(window, new Event("resize"));
    expect(onRect).toHaveBeenCalledTimes(2);
    expect(onRect).toHaveBeenLastCalledWith(expect.objectContaining({ x: 120 }));
  });

  it("stops observing when it unmounts", () => {
    const { unmount } = render(<PhotoCell onRect={() => undefined} native={false} />);
    unmount();
    expect(FakeResizeObserver.instances[0]?.disconnect).toHaveBeenCalled();
  });

  it("still reports once where there is no ResizeObserver", () => {
    vi.stubGlobal("ResizeObserver", undefined);
    const onRect = vi.fn();
    render(<PhotoCell onRect={onRect} native={false} />);
    expect(onRect).toHaveBeenCalledTimes(1);
  });
});

describe("PhotoCell pixel probe", () => {
  function setup() {
    const onPixelProbe = vi.fn();
    render(<PhotoCell onPixelProbe={onPixelProbe} native={false} data-testid="cell" />);
    const cell = screen.getByTestId("cell");
    cell.getBoundingClientRect = () => box(100, 50, 600, 400);
    return { cell, onPixelProbe };
  }

  it("reports the position (0–1) while ⌥ is held over the cell", () => {
    const { cell, onPixelProbe } = setup();
    fireEvent.pointerMove(cell, { clientX: 400, clientY: 250, altKey: true });
    expect(onPixelProbe).toHaveBeenLastCalledWith({ x: 0.5, y: 0.5 });
  });

  it("does not report without ⌥", () => {
    const { cell, onPixelProbe } = setup();
    fireEvent.pointerMove(cell, { clientX: 400, clientY: 250 });
    expect(onPixelProbe).not.toHaveBeenCalled();
  });

  it("stops when ⌥ is released, and starts when it is pressed over the cell", () => {
    const { cell, onPixelProbe } = setup();
    fireEvent.pointerMove(cell, { clientX: 400, clientY: 250 });
    fireEvent.keyDown(window, { key: "Alt" });
    expect(onPixelProbe).toHaveBeenLastCalledWith({ x: 0.5, y: 0.5 });
    fireEvent.keyUp(window, { key: "Alt" });
    expect(onPixelProbe).toHaveBeenLastCalledWith(null);
  });

  it("stops when the pointer leaves", () => {
    const { cell, onPixelProbe } = setup();
    fireEvent.pointerMove(cell, { clientX: 400, clientY: 250, altKey: true });
    fireEvent.pointerLeave(cell);
    expect(onPixelProbe).toHaveBeenLastCalledWith(null);
  });

  it("does not report ⌥ pressed while the pointer is elsewhere", () => {
    const { onPixelProbe } = setup();
    fireEvent.keyDown(window, { key: "Alt" });
    expect(onPixelProbe).not.toHaveBeenCalled();
  });
});
