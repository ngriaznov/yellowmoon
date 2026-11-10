import { fireEvent, render, renderHook, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { describe, expect, it, vi } from "vitest";
import { TabBar } from "./TabBar";
import { FIRST_NODE_TAB, MASK_TAB, NODE_FOR_SIMPLE_TAB, NODE_TABS, tabInfo, tabPanelProps, TAB_IDS } from "./tabs";
import type { TabId } from "./tabs";
import { tabForShortcut, useTabShortcuts } from "./useTabShortcuts";

function Harness({ initial = "look" as TabId, onChange }: { initial?: TabId; onChange?: (t: TabId) => void }) {
  const [tab, setTab] = useState<TabId>(initial);
  return (
    <TabBar
      value={tab}
      onChange={(t) => {
        setTab(t);
        onChange?.(t);
      }}
    />
  );
}

describe("TabBar", () => {
  it("is a tablist of seven tabs with the active one selected", () => {
    render(<Harness initial="curve" />);
    expect(screen.getByRole("tablist", { name: "Tool tabs" })).toBeInTheDocument();
    const tabs = screen.getAllByRole("tab");
    expect(tabs.map((t) => t.textContent)).toEqual(["Look", "Lens", "Light", "Colour", "Detail", "Curve", "RAW"]);
    expect(screen.getByRole("tab", { name: "Curve" })).toHaveAttribute("aria-selected", "true");
    expect(screen.getByRole("tab", { name: "Look" })).toHaveAttribute("aria-selected", "false");
  });

  it("links every tab to its panel", () => {
    render(<Harness />);
    for (const id of TAB_IDS) {
      const tab = document.getElementById(`ym-tab-${id}`);
      expect(tab).toHaveAttribute("aria-controls", `ym-tabpanel-${id}`);
    }
    expect(tabPanelProps("light")).toEqual({ role: "tabpanel", id: "ym-tabpanel-light", "aria-labelledby": "ym-tab-light", "data-tab": "light" });
  });

  it("selects on click", async () => {
    const onChange = vi.fn();
    render(<Harness onChange={onChange} />);
    await userEvent.click(screen.getByRole("tab", { name: "Detail" }));
    expect(onChange).toHaveBeenCalledWith("detail");
    expect(screen.getByRole("tab", { name: "Detail" })).toHaveAttribute("aria-selected", "true");
  });

  it("has one tab in the tab order", async () => {
    const user = userEvent.setup();
    render(<Harness initial="color" />);
    // Radix's roving focus: the list takes the Tab and hands the focus to the selected tab; the other six are out of the order.
    await user.tab();
    expect(screen.getByRole("tab", { name: "Colour" })).toHaveFocus();
    expect(screen.getByRole("tab", { name: "Look" })).toHaveAttribute("tabindex", "-1");
    await user.tab();
    expect(screen.getAllByRole("tab")).not.toContain(document.activeElement);
  });

  it("moves with ← and → inside the bar, and wraps", async () => {
    const user = userEvent.setup();
    render(<Harness initial="raw" />);
    await user.tab();
    expect(screen.getByRole("tab", { name: "RAW" })).toHaveFocus();
    await user.keyboard("{ArrowRight}");
    expect(screen.getByRole("tab", { name: "Look" })).toHaveAttribute("aria-selected", "true");
    expect(screen.getByRole("tab", { name: "Look" })).toHaveFocus();
    await user.keyboard("{ArrowLeft}");
    expect(screen.getByRole("tab", { name: "RAW" })).toHaveFocus();
    await user.keyboard("{ArrowLeft}{ArrowLeft}");
    expect(screen.getByRole("tab", { name: "Curve" })).toHaveAttribute("aria-selected", "false");
    expect(screen.getByRole("tab", { name: "Detail" })).toHaveAttribute("aria-selected", "true");
  });

  it("jumps with Home and End", async () => {
    const user = userEvent.setup();
    render(<Harness initial="color" />);
    await user.tab();
    await user.keyboard("{End}");
    expect(screen.getByRole("tab", { name: "RAW" })).toHaveFocus();
    await user.keyboard("{Home}");
    expect(screen.getByRole("tab", { name: "Look" })).toHaveFocus();
  });

  it("leaves modified arrow keys alone", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    render(<Harness onChange={onChange} />);
    await user.tab();
    await user.keyboard("{Meta>}{ArrowRight}{/Meta}");
    expect(onChange).not.toHaveBeenCalled();
  });
});

describe("⌘1–⌘7", () => {
  it("maps ⌘1 to ⌘7 onto the tabs in order", () => {
    TAB_IDS.forEach((id, i) => {
      expect(tabForShortcut({ metaKey: true, shiftKey: false, altKey: false, ctrlKey: false, code: `Digit${i + 1}`, key: String(i + 1) })).toBe(id);
    });
  });

  it("follows the bar: Look, Lens, Light, Colour, Detail, Curve, RAW", () => {
    expect(TAB_IDS).toEqual(["look", "lens", "light", "color", "detail", "curve", "raw"]);
    const press = (digit: number) => tabForShortcut({ metaKey: true, shiftKey: false, altKey: false, ctrlKey: false, code: `Digit${digit}`, key: String(digit) });
    expect([1, 2, 3, 4, 5, 6, 7].map(press)).toEqual(TAB_IDS);
  });

  it("ignores other digits and other modifiers", () => {
    const base = { metaKey: true, shiftKey: false, altKey: false, ctrlKey: false, code: "Digit8", key: "8" };
    expect(tabForShortcut(base)).toBeNull();
    expect(tabForShortcut({ ...base, code: "Digit1", key: "1", metaKey: false })).toBeNull();
    expect(tabForShortcut({ ...base, code: "Digit1", key: "1", shiftKey: true })).toBeNull();
    expect(tabForShortcut({ ...base, code: "Digit1", key: "1", altKey: true })).toBeNull();
    expect(tabForShortcut({ ...base, code: "Digit1", key: "1", ctrlKey: true })).toBeNull();
  });

  it("switches tabs from anywhere in the window, even from a text field", () => {
    const onSelect = vi.fn();
    renderHook(() => useTabShortcuts(onSelect));
    const input = document.createElement("input");
    document.body.append(input);
    input.focus();
    fireEvent.keyDown(input, { key: "4", code: "Digit4", metaKey: true });
    expect(onSelect).toHaveBeenCalledWith("color");
    fireEvent.keyDown(window, { key: "7", code: "Digit7", metaKey: true });
    expect(onSelect).toHaveBeenLastCalledWith("raw");
    input.remove();
  });

  it("prevents the browser's own handling of the shortcut", () => {
    renderHook(() => useTabShortcuts(() => undefined));
    const event = new KeyboardEvent("keydown", { key: "2", code: "Digit2", metaKey: true, cancelable: true });
    window.dispatchEvent(event);
    expect(event.defaultPrevented).toBe(true);
  });

  it("does nothing when disabled or unmounted", () => {
    const onSelect = vi.fn();
    const { rerender, unmount } = renderHook(({ enabled }) => useTabShortcuts(onSelect, enabled), { initialProps: { enabled: false } });
    fireEvent.keyDown(window, { key: "1", code: "Digit1", metaKey: true });
    expect(onSelect).not.toHaveBeenCalled();
    rerender({ enabled: true });
    fireEvent.keyDown(window, { key: "1", code: "Digit1", metaKey: true });
    expect(onSelect).toHaveBeenCalledTimes(1);
    unmount();
    fireEvent.keyDown(window, { key: "2", code: "Digit2", metaKey: true });
    expect(onSelect).toHaveBeenCalledTimes(1);
  });
});


describe("the tabs of the node view (docs/23 §7.5)", () => {
  it("are RAW · Lens · Light · Colour · Detail on the Input, Light · Colour · Curve · Mask on a corrector and Look · Detail on the Output", () => {
    expect(NODE_TABS.input).toEqual(["raw", "lens", "light", "color", "detail"]);
    expect(NODE_TABS.corrector).toEqual(["light", "color", "curve", "mask"]);
    expect(NODE_TABS.output).toEqual(["look", "detail"]);
    expect(Object.values(FIRST_NODE_TAB)).toEqual(["raw", "light", "film", "look", "inputs", "blend", "channels", "channels"]);
    // The nodes of `docs/25` have one tab each (§5.5): Inputs, Blend and Channels twice.
    expect([NODE_TABS.parallel, NODE_TABS.layer, NODE_TABS.splitter, NODE_TABS.combiner]).toEqual([["inputs"], ["blend"], ["channels"], ["channels"]]);
    expect(tabInfo("blend").label).toBe("Blend");
    expect(NODE_TABS.film).toEqual(["film", "print", "halation", "grain", "mask"]);
    expect(tabInfo("mask")).toBe(MASK_TAB);
    // The simple view's seven stay seven: Mask is the node view's.
    expect(TAB_IDS).toHaveLength(7);
    expect(TAB_IDS).not.toContain("mask");
  });

  it("have a tab bar of their own, with the same ids and the labels of the tabs", async () => {
    const onChange = vi.fn();
    render(<TabBar value="mask" onChange={onChange} tabs={NODE_TABS.corrector} />);
    expect(screen.getAllByRole("tab").map((t) => t.textContent)).toEqual(["Light", "Colour", "Curve", "Mask"]);
    expect(screen.getByRole("tab", { name: "Mask" })).toHaveAttribute("aria-selected", "true");
    expect(document.getElementById("ym-tab-mask")).toHaveAttribute("aria-controls", "ym-tabpanel-mask");
    await userEvent.click(screen.getByRole("tab", { name: "Curve" }));
    expect(onChange).toHaveBeenCalledWith("curve");
  });

  it("are chosen by ⌘1 to ⌘5 in the order of the node's row, and the digits past the last tab choose nothing", () => {
    const press = (digit: number, tabs: readonly TabId[]) => tabForShortcut({ metaKey: true, shiftKey: false, altKey: false, ctrlKey: false, code: `Digit${digit}`, key: String(digit) }, tabs);
    expect([1, 2, 3, 4, 5].map((d) => press(d, NODE_TABS.input))).toEqual(["raw", "lens", "light", "color", "detail"]);
    expect([1, 2, 3, 4, 5, 6, 7].map((d) => press(d, NODE_TABS.corrector))).toEqual(["light", "color", "curve", "mask", null, null, null]);
    expect([1, 2, 3].map((d) => press(d, NODE_TABS.output))).toEqual(["look", "detail", null]);
    expect(press(7, TAB_IDS)).toBe("raw");
  });

  it("make the shortcut hook follow the node's row, and let the keys it has no tab for through", () => {
    const onSelect = vi.fn();
    const { rerender } = renderHook(({ tabs }) => useTabShortcuts(onSelect, true, tabs), { initialProps: { tabs: NODE_TABS.output as readonly TabId[] } });
    const key = (digit: number): boolean => !fireEvent.keyDown(window, { key: String(digit), code: `Digit${digit}`, metaKey: true, cancelable: true });
    expect(key(2)).toBe(true);
    expect(onSelect).toHaveBeenLastCalledWith("detail");
    onSelect.mockClear();
    expect(key(3)).toBe(false);
    expect(onSelect).not.toHaveBeenCalled();
    rerender({ tabs: NODE_TABS.corrector });
    expect(key(3)).toBe(true);
    expect(onSelect).toHaveBeenLastCalledWith("curve");
  });

  it("map each tab of the simple view onto a node and a tab", () => {
    expect(NODE_FOR_SIMPLE_TAB).toEqual({
      look: { kind: "output", tab: "look" },
      lens: { kind: "input", tab: "lens" },
      light: { kind: "corrector", tab: "light" },
      color: { kind: "corrector", tab: "color" },
      detail: { kind: "input", tab: "detail" },
      curve: { kind: "corrector", tab: "curve" },
      raw: { kind: "input", tab: "raw" },
    });
    // Every target is a tab its node has.
    for (const { kind, tab } of Object.values(NODE_FOR_SIMPLE_TAB)) expect(NODE_TABS[kind]).toContain(tab);
  });
});
