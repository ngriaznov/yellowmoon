import { act, fireEvent, render, screen, within } from "@testing-library/react";
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

  it("draws the shell's picture of the cell one picture pixel to one device pixel (the fallback viewport)", () => {
    vi.stubGlobal("devicePixelRatio", 2);
    const { container } = render(<PhotoCell frame={{ src: "ym-frame://localhost/frame-4.png", widthPx: 2040, heightPx: 1360 }} native={false} />);
    const img = container.querySelector("img") as HTMLImageElement;
    expect(img).toHaveAttribute("src", "ym-frame://localhost/frame-4.png");
    expect(img).toHaveClass("ym-photo-cell__frame");
    expect(img.style.width).toBe("1020px");
    expect(img.style.height).toBe("680px");
    // The cell is paper, not transparent: nothing draws behind a webview that shows the picture itself.
    expect(container.firstChild).not.toHaveAttribute("data-native");
    expect(container.querySelectorAll("img")).toHaveLength(1);
  });

  it("prefers the shell's picture to the mock's, and draws neither natively", () => {
    const { container, rerender } = render(<PhotoCell imageSrc="photo.svg" frame={{ src: "frame.png", widthPx: 100, heightPx: 50 }} native={false} />);
    expect(container.querySelector("img")).toHaveAttribute("src", "frame.png");
    rerender(<PhotoCell imageSrc="photo.svg" frame={{ src: "frame.png", widthPx: 100, heightPx: 50 }} native />);
    expect(container.querySelector("img")).toBeNull();
    rerender(<PhotoCell imageSrc="photo.svg" native={false} />);
    expect(container.querySelector("img")).toHaveAttribute("src", "photo.svg");
  });

  it("shows the look chip and the camera line", () => {
    render(<PhotoCell lookChip="Neutral · 100%" cameraLine="Sony A1 · DSC04204 · ISO 400 · 1/125 · f/2.8" native={false} />);
    expect(screen.getByText("Neutral · 100%")).toHaveClass("ym-photo-cell__chip");
    expect(screen.getByText("Sony A1 · DSC04204 · ISO 400 · 1/125 · f/2.8")).toHaveClass("ym-photo-cell__camera");
  });

  it("shows the Before label and hides the look chip in Before mode", () => {
    render(<PhotoCell lookChip="Neutral · 100%" before native={false} />);
    expect(screen.getByText("Before · camera default")).toHaveClass("ym-photo-cell__before");
    expect(screen.queryByText("Neutral · 100%")).toBeNull();
  });

  it("shows the pixel readout when given one", () => {
    const { rerender } = render(<PhotoCell pixelReadout="R 182 G 164 B 140" native={false} />);
    expect(screen.getByText("R 182 G 164 B 140")).toHaveClass("ym-photo-cell__readout");
    rerender(<PhotoCell pixelReadout={null} native={false} />);
    expect(screen.queryByText("R 182 G 164 B 140")).toBeNull();
  });

  it("keeps the chips, the camera line and the readout out of the image, in a group beside it", () => {
    render(
      <PhotoCell imageLabel="Photo DSC04204, Neutral · 100%" lookChip="Neutral · 100%" cameraLine="Sony A1 · DSC04204" pixelReadout="R 1 G 2 B 3" native={false} />,
    );
    const image = screen.getByRole("img", { name: "Photo DSC04204, Neutral · 100%" });
    const group = screen.getByRole("group", { name: "Photo information" });
    expect(image).toBeEmptyDOMElement();
    expect(image.contains(group)).toBe(false);
    expect(group.parentElement).toBe(image.parentElement);
    expect(within(group).getByText("Neutral · 100%")).toBeInTheDocument();
    expect(within(group).getByText("Sony A1 · DSC04204")).toBeInTheDocument();
  });

  it("puts the Before label in the group too", () => {
    render(<PhotoCell before native={false} />);
    expect(within(screen.getByRole("group", { name: "Photo information" })).getByText("Before · camera default")).toBeInTheDocument();
  });

  describe("the pixel readout as a status", () => {
    beforeEach(() => vi.useFakeTimers());
    afterEach(() => vi.useRealTimers());

    it("draws every value at once but announces at most one a second, the last", () => {
      const { rerender } = render(<PhotoCell pixelReadout="R 0 G 0 B 0" native={false} />);
      const status = screen.getByRole("status");
      expect(status).toBeEmptyDOMElement();
      for (let i = 1; i <= 10; i++) {
        rerender(<PhotoCell pixelReadout={`R ${i} G ${i} B ${i}`} native={false} />);
        // The drawn text is live and hidden from a screen reader.
        expect(document.querySelector(".ym-photo-cell__readout")).toHaveTextContent(`R ${i} G ${i} B ${i}`);
        expect(document.querySelector(".ym-photo-cell__readout")).toHaveAttribute("aria-hidden", "true");
        act(() => void vi.advanceTimersByTime(90));
      }
      // Ten moves in 900 ms: nothing was announced yet.
      expect(status).toBeEmptyDOMElement();
      act(() => void vi.advanceTimersByTime(100));
      expect(status).toHaveTextContent("R 10 G 10 B 10");
    });

    it("announces one update for a burst, then the next burst a second later", () => {
      const { rerender } = render(<PhotoCell pixelReadout="R 1 G 1 B 1" native={false} />);
      const status = screen.getByRole("status");
      rerender(<PhotoCell pixelReadout="R 2 G 2 B 2" native={false} />);
      act(() => void vi.advanceTimersByTime(1000));
      expect(status).toHaveTextContent("R 2 G 2 B 2");
      rerender(<PhotoCell pixelReadout="R 3 G 3 B 3" native={false} />);
      act(() => void vi.advanceTimersByTime(999));
      expect(status).toHaveTextContent("R 2 G 2 B 2");
      act(() => void vi.advanceTimersByTime(1));
      expect(status).toHaveTextContent("R 3 G 3 B 3");
    });

    it("clears at once when the readout goes", () => {
      const { rerender } = render(<PhotoCell pixelReadout="R 1 G 1 B 1" native={false} />);
      act(() => void vi.advanceTimersByTime(1000));
      expect(screen.getByRole("status")).toHaveTextContent("R 1 G 1 B 1");
      rerender(<PhotoCell pixelReadout={null} native={false} />);
      expect(screen.queryByRole("status")).toBeNull();
      // A value that arrives later is not shown by a timer of the old one.
      rerender(<PhotoCell pixelReadout="R 9 G 9 B 9" native={false} />);
      act(() => void vi.advanceTimersByTime(500));
      expect(screen.getByRole("status")).toBeEmptyDOMElement();
      act(() => void vi.advanceTimersByTime(500));
      expect(screen.getByRole("status")).toHaveTextContent("R 9 G 9 B 9");
    });
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

  describe("when the window is dragged to a display of another scale", () => {
    // The cell keeps its size in CSS pixels, so no ResizeObserver or `resize` event is certain; the media query for the scale is.
    class FakeQuery {
      static all: FakeQuery[] = [];
      listeners = new Set<() => void>();
      constructor(readonly media: string) {
        FakeQuery.all.push(this);
      }
      addEventListener = (_type: string, listener: () => void): void => void this.listeners.add(listener);
      removeEventListener = (_type: string, listener: () => void): void => void this.listeners.delete(listener);
      fire(): void {
        for (const listener of [...this.listeners]) listener();
      }
    }
    beforeEach(() => {
      FakeQuery.all = [];
      vi.stubGlobal("matchMedia", (media: string) => new FakeQuery(media));
    });

    it("reports the new scale, and watches for the next change", () => {
      vi.spyOn(Element.prototype, "getBoundingClientRect").mockReturnValue(box(0, 75, 1000, 666));
      vi.stubGlobal("devicePixelRatio", 2);
      const onRect = vi.fn();
      render(<PhotoCell onRect={onRect} native={false} />);
      expect(onRect).toHaveBeenLastCalledWith(expect.objectContaining({ scale: 2 }));
      expect(FakeQuery.all.at(-1)?.media).toBe("(resolution: 2dppx)");
      vi.stubGlobal("devicePixelRatio", 1);
      act(() => FakeQuery.all.at(-1)?.fire());
      expect(onRect).toHaveBeenCalledTimes(2);
      expect(onRect).toHaveBeenLastCalledWith({ x: 0, y: 75, width: 1000, height: 666, scale: 1 });
      // The query is now the one for 1×, and the old one is no longer listened to.
      expect(FakeQuery.all.at(-1)?.media).toBe("(resolution: 1dppx)");
      expect(FakeQuery.all[0]?.listeners.size).toBe(0);
      vi.stubGlobal("devicePixelRatio", 2);
      act(() => FakeQuery.all.at(-1)?.fire());
      expect(onRect).toHaveBeenLastCalledWith(expect.objectContaining({ scale: 2 }));
    });

    it("stops listening when it unmounts", () => {
      const { unmount } = render(<PhotoCell onRect={() => undefined} native={false} />);
      const query = FakeQuery.all.at(-1);
      expect(query?.listeners.size).toBe(1);
      unmount();
      expect(query?.listeners.size).toBe(0);
    });
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

  describe("the keys of a zoomed photo (docs/28 A-10)", () => {
    const keys = () => ({ pan: vi.fn(), centre: vi.fn(), readPixel: vi.fn() });

    it("is not a Tab stop without keys, and keeps the photo's own name", () => {
      render(<PhotoCell imageLabel="Photo DSC04204, Neutral · 100%" native={false} />);
      const view = screen.getByRole("img", { name: "Photo DSC04204, Neutral · 100%" });
      expect(view).not.toHaveAttribute("tabindex");
      expect(view).not.toHaveAttribute("data-pan-view");
    });

    it("is a Tab stop named for what the arrows do, and still says which photo it is", () => {
      render(<PhotoCell imageLabel="Photo DSC04204, Neutral · 100%" native={false} keys={keys()} />);
      const view = screen.getByRole("img", { name: "Photo, zoomed. The arrow keys move it." });
      expect(view).toHaveAttribute("tabindex", "0");
      expect(view).toHaveAttribute("data-pan-view");
      expect(view).toHaveAccessibleDescription("Photo DSC04204, Neutral · 100%. Home centres it. I reads the pixel at the centre.");
      // The picture still holds nothing but the picture.
      expect(view.textContent).toBe("");
    });

    it("the arrows pan by 5 % of the cell, with Shift by 25 %, and the key is not left to the page", () => {
      const h = keys();
      render(<PhotoCell native={false} keys={h} />);
      vi.spyOn(HTMLElement.prototype, "getBoundingClientRect").mockReturnValue(box(0, 0, 1000, 600));
      const view = screen.getByRole("img");
      const cancelled = fireEvent.keyDown(view, { key: "ArrowRight" });
      expect(cancelled).toBe(false);
      expect(h.pan).toHaveBeenLastCalledWith(-50, 0);
      fireEvent.keyDown(view, { key: "ArrowDown", shiftKey: true });
      expect(h.pan).toHaveBeenLastCalledWith(0, -150);
    });

    it("Home centres, I reads the pixel once, and a key held down does not repeat either", () => {
      const h = keys();
      render(<PhotoCell native={false} keys={h} />);
      const view = screen.getByRole("img");
      fireEvent.keyDown(view, { key: "Home" });
      fireEvent.keyDown(view, { key: "Home", repeat: true });
      fireEvent.keyDown(view, { key: "i" });
      fireEvent.keyDown(view, { key: "I", repeat: true });
      expect(h.centre).toHaveBeenCalledTimes(1);
      expect(h.readPixel).toHaveBeenCalledTimes(1);
      expect(h.pan).not.toHaveBeenCalled();
    });

    it("leaves a key with a modifier, and a key that came up from inside the picture, alone", () => {
      const h = keys();
      const { container } = render(
        <PhotoCell native={false} keys={h}>
          <button type="button">inside</button>
        </PhotoCell>,
      );
      const view = screen.getByRole("img");
      fireEvent.keyDown(view, { key: "ArrowRight", metaKey: true });
      fireEvent.keyDown(view, { key: "ArrowRight", altKey: true });
      // A control over the picture (the crop frame lives in the overlay) is not the picture: its keys are its own.
      fireEvent.keyDown(container.querySelector("button") as HTMLElement, { key: "ArrowRight" });
      expect(h.pan).not.toHaveBeenCalled();
    });

    it("a click on the picture does not take the focus, so the arrows keep stepping photos for a mouse user", () => {
      render(
        <>
          <button type="button">elsewhere</button>
          <PhotoCell native={false} keys={keys()} />
        </>,
      );
      const other = screen.getByRole("button", { name: "elsewhere" });
      other.focus();
      const proceeded = fireEvent.mouseDown(screen.getByRole("img"));
      expect(proceeded).toBe(false);
      expect(document.activeElement).not.toBe(screen.getByRole("img"));
    });
  });
});
