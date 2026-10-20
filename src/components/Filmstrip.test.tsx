import { fireEvent, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { describe, expect, it, vi } from "vitest";
import { Filmstrip, filmstripCaption, scrollOffsetToReveal } from "./Filmstrip";
import type { FilmstripPhoto } from "./Filmstrip";

const PHOTOS: FilmstripPhoto[] = [
  { id: 1, name: "DSC04201", src: "a.png" },
  { id: 2, name: "DSC04202", src: "b.png", picked: true },
  { id: 3, name: "DSC04203", src: "c.png" },
  { id: 4, name: "DSC04204", status: "failed" },
  { id: 5, name: "DSC04205", status: "developing" },
];

function Harness({ onSelect, onTogglePick }: { onSelect?: (id: number | string) => void; onTogglePick?: (id: number | string) => void }) {
  const [current, setCurrent] = useState<number | string>(2);
  return (
    <Filmstrip
      photos={PHOTOS}
      currentId={current}
      onSelect={(id) => {
        setCurrent(id);
        onSelect?.(id);
      }}
      onTogglePick={onTogglePick}
    />
  );
}

describe("Filmstrip", () => {
  it("is a nav of one button per photo with the current one marked", () => {
    render(<Harness />);
    expect(screen.getByRole("navigation", { name: "Photos" })).toBeInTheDocument();
    expect(screen.getAllByRole("button")).toHaveLength(5);
    expect(screen.getByRole("button", { name: "Show DSC04202, picked" })).toHaveAttribute("aria-current", "true");
    expect(screen.getByRole("button", { name: "Show DSC04201" })).toHaveAttribute("aria-current", "false");
  });

  it("marks failed and developing files in their names, and shows the CAN'T OPEN tag", () => {
    render(<Harness />);
    const failed = screen.getByRole("button", { name: "Show DSC04204, can't open" });
    expect(within(failed).getByText("Can't open")).toBeInTheDocument();
    expect(failed.querySelector("img")).toBeNull();
    expect(screen.getByRole("button", { name: "Show DSC04205, developing" })).toHaveAttribute("data-status", "developing");
  });

  it("draws a picked marker", () => {
    const { container } = render(<Harness />);
    expect(container.querySelectorAll(".ym-filmstrip__picked")).toHaveLength(1);
  });

  it("shows one caption line: position, and number picked", () => {
    render(<Harness />);
    expect(screen.getByText("2 of 5 · 1 picked")).toBeInTheDocument();
  });

  it("has no counter cell, and describes the strip by its caption", () => {
    const { container } = render(<Harness />);
    expect(container.querySelector(".ym-filmstrip__counter")).toBeNull();
    expect(screen.getByRole("navigation", { name: "Photos" })).toHaveAccessibleDescription("2 of 5 · 1 picked");
  });

  it("selects on click", async () => {
    const onSelect = vi.fn();
    render(<Harness onSelect={onSelect} />);
    await userEvent.click(screen.getByRole("button", { name: "Show DSC04203" }));
    expect(onSelect).toHaveBeenCalledWith(3);
    expect(screen.getByText("3 of 5 · 1 picked")).toBeInTheDocument();
  });

  it("toggles the pick on ⌘-click without selecting", () => {
    const onSelect = vi.fn();
    const onTogglePick = vi.fn();
    render(<Harness onSelect={onSelect} onTogglePick={onTogglePick} />);
    fireEvent.click(screen.getByRole("button", { name: "Show DSC04203" }), { metaKey: true });
    expect(onTogglePick).toHaveBeenCalledWith(3);
    expect(onSelect).not.toHaveBeenCalled();
  });

  it("moves the selection and the focus with ← and →", async () => {
    const user = userEvent.setup();
    const onSelect = vi.fn();
    render(<Harness onSelect={onSelect} />);
    await user.tab();
    expect(screen.getByRole("button", { name: "Show DSC04202, picked" })).toHaveFocus();
    await user.keyboard("{ArrowRight}");
    expect(onSelect).toHaveBeenLastCalledWith(3);
    expect(screen.getByRole("button", { name: "Show DSC04203" })).toHaveFocus();
    await user.keyboard("{ArrowLeft}{ArrowLeft}");
    expect(onSelect).toHaveBeenLastCalledWith(1);
    expect(screen.getByRole("button", { name: "Show DSC04201" })).toHaveFocus();
  });

  it("stops at the ends", async () => {
    const user = userEvent.setup();
    const onSelect = vi.fn();
    render(<Harness onSelect={onSelect} />);
    await user.tab();
    await user.keyboard("{ArrowLeft}{ArrowLeft}{ArrowLeft}");
    expect(onSelect).toHaveBeenCalledTimes(1);
    expect(onSelect).toHaveBeenCalledWith(1);
  });

  it("toggles the pick with P", async () => {
    const user = userEvent.setup();
    const onTogglePick = vi.fn();
    render(<Harness onTogglePick={onTogglePick} />);
    await user.tab();
    await user.keyboard("p");
    expect(onTogglePick).toHaveBeenCalledWith(2);
  });

  it("has one thumbnail in the tab order", () => {
    render(<Harness />);
    const tabbable = screen.getAllByRole("button").filter((b) => b.getAttribute("tabindex") === "0");
    expect(tabbable).toHaveLength(1);
    expect(tabbable[0]).toHaveAccessibleName("Show DSC04202, picked");
  });

  it("scrolls the strip, not the page, to bring the current photo into view", () => {
    const scrollIntoView = vi.fn();
    Element.prototype.scrollIntoView = scrollIntoView;
    try {
      const rect = (left: number, right: number) => ({ left, right, top: 0, bottom: 40, width: right - left, height: 40, x: left, y: 0, toJSON: () => ({}) }) as DOMRect;
      const original = Element.prototype.getBoundingClientRect;
      Element.prototype.getBoundingClientRect = function (this: Element) {
        if (this.classList.contains("ym-filmstrip__scroll")) return rect(0, 500);
        if (this.getAttribute("data-photo-id") === "3") return rect(560, 650);
        return rect(100, 160);
      };
      try {
        const { container, rerender } = render(<Filmstrip photos={PHOTOS} currentId={1} />);
        const strip = container.querySelector(".ym-filmstrip__scroll") as HTMLElement;
        expect(strip.scrollLeft).toBe(0);
        rerender(<Filmstrip photos={PHOTOS} currentId={3} />);
        expect(strip.scrollLeft).toBe(650 - (500 - 24));
        expect(scrollIntoView).not.toHaveBeenCalled();
      } finally {
        Element.prototype.getBoundingClientRect = original;
      }
    } finally {
      // @ts-expect-error jsdom has no scrollIntoView; put it back the way it was
      delete Element.prototype.scrollIntoView;
    }
  });
});

describe("scrollOffsetToReveal", () => {
  const strip = { left: 0, right: 500 };
  it("is 0 for an item inside the strip", () => {
    expect(scrollOffsetToReveal(strip, { left: 100, right: 160 })).toBe(0);
  });
  it("scrolls left to reveal an item cut off at the left, with a margin", () => {
    expect(scrollOffsetToReveal(strip, { left: -30, right: 30 })).toBe(-54);
  });
  it("scrolls right to reveal an item cut off at the right, with a margin", () => {
    expect(scrollOffsetToReveal(strip, { left: 450, right: 540 })).toBe(64);
  });
});

describe("Filmstrip in export mode", () => {
  const exportPhotos: FilmstripPhoto[] = PHOTOS.map((p) => ({ ...p, included: p.id === 2 || p.id === 3 }));

  it("is not interactive and counts what will be exported", () => {
    render(<Filmstrip mode="export" photos={exportPhotos} currentId={2} />);
    expect(screen.getByRole("navigation", { name: "Photos to export" })).toBeInTheDocument();
    expect(screen.queryAllByRole("button")).toHaveLength(0);
    expect(screen.getByText("2 to export")).toBeInTheDocument();
  });

  it("dims what will not be exported and names what will", () => {
    render(<Filmstrip mode="export" photos={exportPhotos} currentId={2} />);
    expect(screen.getByRole("img", { name: "DSC04202, picked, will be exported" })).toHaveAttribute("data-in-export", "true");
    expect(screen.getByRole("img", { name: "DSC04201" })).toHaveAttribute("data-in-export", "false");
  });

  it("takes a custom caption", () => {
    render(<Filmstrip mode="export" photos={exportPhotos} currentId={2} caption="12 photos" />);
    expect(screen.getByText("12 photos")).toBeInTheDocument();
  });

  it("tints by tab through data-tab", () => {
    render(<Filmstrip photos={PHOTOS} currentId={2} tab="light" />);
    expect(screen.getByRole("navigation", { name: "Photos" })).toHaveAttribute("data-tab", "light");
  });
});

describe("filmstripCaption", () => {
  it("says where you are and how many are picked", () => {
    expect(filmstripCaption(PHOTOS, 2)).toBe("2 of 5 · 1 picked");
    expect(filmstripCaption(PHOTOS, null)).toBe("0 of 5 · 1 picked");
  });

  it("counts what will be exported in the Export panel", () => {
    expect(filmstripCaption([{ id: 1, name: "a", included: true }, { id: 2, name: "b" }], 1, "export")).toBe("1 to export");
  });
});

describe("Filmstrip remove menu", () => {
  /** A strip with the menu on, over `photos` (the harness keeps the current photo of its own). */
  function Removable({
    photos = PHOTOS,
    onRemove,
    onRemovePicked,
    initial = 2,
  }: {
    photos?: FilmstripPhoto[];
    onRemove?: (id: number | string) => void;
    onRemovePicked?: () => void;
    initial?: number | string;
  }) {
    const [current, setCurrent] = useState<number | string>(initial);
    return <Filmstrip photos={photos} currentId={current} onSelect={setCurrent} onRemove={onRemove} onRemovePicked={onRemovePicked} />;
  }
  const rightClick = (name: string) => fireEvent.contextMenu(screen.getByRole("button", { name }), { clientX: 40, clientY: 40 });

  it("opens glasscn's context menu on the thumbnail that was right-clicked, and removes that photo and not the current one", async () => {
    const onRemove = vi.fn();
    render(<Removable onRemove={onRemove} />);
    expect(screen.queryByRole("menu")).toBeNull();
    rightClick("Show DSC04203");
    const menu = await screen.findByRole("menu", { name: "DSC04203 actions" });
    expect(menu.getAttribute("data-slot")).toBe("context-menu-content");
    // The thumbnail the menu is about says so; the current photo (DSC04202) is not the one.
    // (Radix hides the page behind an open menu from the accessibility tree.)
    expect(screen.getByRole("button", { name: "Show DSC04203", hidden: true })).toHaveAttribute("data-menu", "open");
    await userEvent.click(within(menu).getByRole("menuitem", { name: "Remove from filmstrip" }));
    expect(onRemove).toHaveBeenCalledTimes(1);
    expect(onRemove).toHaveBeenCalledWith(3);
    expect(screen.queryByRole("menu")).toBeNull();
    // The right click did not choose the photo.
    expect(screen.getByRole("button", { name: "Show DSC04202, picked" })).toHaveAttribute("aria-current", "true");
  });

  it("offers Remove picked photos, with the count, only while photos are picked", async () => {
    const onRemovePicked = vi.fn();
    const picked = PHOTOS.map((p) => ({ ...p, picked: p.id === 1 || p.id === 2 || p.id === 3 }));
    const { unmount } = render(<Removable photos={picked} onRemove={vi.fn()} onRemovePicked={onRemovePicked} />);
    rightClick("Show DSC04204, can't open");
    const menu = await screen.findByRole("menu");
    const items = within(menu).getAllByRole("menuitem");
    expect(items.map((i) => i.textContent)).toEqual(["Remove from filmstrip", "Remove picked photos3"]);
    await userEvent.click(within(menu).getByRole("menuitem", { name: /Remove picked photos/ }));
    expect(onRemovePicked).toHaveBeenCalledTimes(1);
    unmount();

    render(<Removable photos={PHOTOS.map((p) => ({ ...p, picked: false }))} onRemove={vi.fn()} onRemovePicked={onRemovePicked} />);
    rightClick("Show DSC04201");
    const bare = await screen.findByRole("menu");
    expect(within(bare).getAllByRole("menuitem")).toHaveLength(1);
    expect(within(bare).queryByRole("menuitem", { name: /picked/ })).toBeNull();
  });

  it("can be opened from the keyboard with Shift+F10 or the Menu key, and closed with Escape", async () => {
    render(<Removable onRemove={vi.fn()} />);
    screen.getByRole("button", { name: "Show DSC04202, picked" }).focus();
    fireEvent.contextMenu(document.activeElement as HTMLElement, { clientX: 0, clientY: 0 });
    const menu = await screen.findByRole("menu");
    expect(within(menu).getByRole("menuitem", { name: "Remove from filmstrip" })).toBeInTheDocument();
    await userEvent.keyboard("{Escape}");
    expect(screen.queryByRole("menu")).toBeNull();
  });

  it("opens no menu on a right click between the thumbnails, and the menu is about the photo of the last right click", async () => {
    const onRemove = vi.fn();
    const { container } = render(<Removable onRemove={onRemove} />);
    fireEvent.contextMenu(container.querySelector(".ym-filmstrip__scroll") as HTMLElement, { clientX: 3, clientY: 3 });
    expect(screen.queryByRole("menu")).toBeNull();
    rightClick("Show DSC04201");
    await userEvent.keyboard("{Escape}");
    expect(screen.queryByRole("menu")).toBeNull();
    rightClick("Show DSC04205, developing");
    await userEvent.click(within(await screen.findByRole("menu", { name: "DSC04205 actions" })).getByRole("menuitem", { name: "Remove from filmstrip" }));
    expect(onRemove).toHaveBeenCalledTimes(1);
    expect(onRemove).toHaveBeenCalledWith(5);
    // The ring is gone with the menu.
    expect(container.querySelector('[data-menu="open"]')).toBeNull();
  });

  it("has no menu without onRemove, and none in the Export panel's strip", () => {
    const { unmount } = render(<Harness />);
    rightClick("Show DSC04203");
    expect(screen.queryByRole("menu")).toBeNull();
    unmount();
    render(<Filmstrip mode="export" photos={PHOTOS} currentId={2} onRemove={vi.fn()} />);
    fireEvent.contextMenu(screen.getByRole("img", { name: /DSC04203/ }), { clientX: 1, clientY: 1 });
    expect(screen.queryByRole("menu")).toBeNull();
  });

  it("moves the keyboard focus to the photo on screen when the focused thumbnail is removed", () => {
    const { rerender } = render(<Filmstrip photos={PHOTOS} currentId={3} onRemove={vi.fn()} />);
    const focused = screen.getByRole("button", { name: "Show DSC04203" });
    focused.focus();
    expect(focused).toHaveFocus();
    // The photo is removed: its thumbnail is gone and the next photo is on screen.
    rerender(<Filmstrip photos={PHOTOS.filter((p) => p.id !== 3)} currentId={4} onRemove={vi.fn()} />);
    expect(screen.queryByRole("button", { name: "Show DSC04203" })).toBeNull();
    expect(screen.getByRole("button", { name: "Show DSC04204, can't open" })).toHaveFocus();
  });

  it("leaves the focus alone when it was somewhere else", () => {
    const { rerender } = render(
      <>
        <button type="button">Elsewhere</button>
        <Filmstrip photos={PHOTOS} currentId={3} onRemove={vi.fn()} />
      </>,
    );
    screen.getByRole("button", { name: "Show DSC04203" }).focus();
    screen.getByRole("button", { name: "Elsewhere" }).focus();
    rerender(
      <>
        <button type="button">Elsewhere</button>
        <Filmstrip photos={PHOTOS.filter((p) => p.id !== 3)} currentId={4} onRemove={vi.fn()} />
      </>,
    );
    expect(screen.getByRole("button", { name: "Elsewhere" })).toHaveFocus();
  });
});
