/**
 * The More menu (`docs/23` §7.2 *The keyboard path*, §7.5): what it lists for each kind of node, what the lists hold (the inputs the same
 * rules light up while a link is dragged), and what each item asks for.
 */
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import type { DevelopSettings, NodeId, NodeLink, NodePatch } from "../../api/types";
import { sortedLinks } from "../../state/graphOps";
import { applyPatch, defaultSettings } from "../../state/settingsGraph";
import { NodeMoreMenu } from "./NodeMoreMenu";

const pic = (from: NodeId, to: NodeId): NodeLink => ({ from: { node: from, port: "picture" }, to: { node: to, port: "picture" } });
const key = (from: number, to: number): NodeLink => ({ from: { node: from, port: "key" }, to: { node: to, port: "key" } });

/** Main → Mountain → Sky → Warmth, with Mountain's key feeding Warmth. */
function graph(): DevelopSettings {
  const nodes: Record<string, NodePatch> = {
    "2": { kind: "corrector", name: "Mountain", pos: [1, 0] },
    "3": { kind: "corrector", name: "Sky", pos: [2, 0] },
    "4": { kind: "corrector", name: "Warmth", pos: [3, 0] },
  };
  return applyPatch(defaultSettings(), { nodes, links: sortedLinks([pic("input", 1), pic(1, 2), pic(2, 3), pic(3, 4), pic(4, "output"), key(2, 4)]) });
}

async function open(node: NodeId, settings = graph()) {
  const handlers = { onRename: vi.fn(), onMoveEarlier: vi.fn(), onMoveLater: vi.fn(), onLink: vi.fn(), onRemoveLink: vi.fn(), onDelete: vi.fn() };
  render(<NodeMoreMenu settings={settings} node={node} {...handlers} />);
  const user = userEvent.setup();
  await user.click(screen.getByRole("button", { name: "More" }));
  const menu = await screen.findByRole("menu", { name: "More" });
  return { user, menu, ...handlers };
}

/** Chooses a menu item with the keyboard: it has the focus, and Return chooses it (a submenu's items need no pointer to be reached). */
async function choose(user: ReturnType<typeof userEvent.setup>, item: HTMLElement): Promise<void> {
  item.focus();
  await user.keyboard("{Enter}");
}

const labels = (menu: HTMLElement): string[] => within(menu).getAllByRole("menuitem").map((i) => i.textContent?.trim() ?? "");

describe("the More menu", () => {
  it("lists, for a corrector: Rename, Move earlier, Move later, Link picture to, Link key to, Remove link, Delete node", async () => {
    const { menu } = await open(3);
    expect(labels(menu)).toEqual(["Rename", "Move earlier", "Move later", "Link picture to", "Link key to", "Remove link", "Delete node"]);
  });

  it("is a button named More that opens upward over the photo", async () => {
    const { menu } = await open(3);
    expect(menu).toHaveAttribute("data-side", "top");
  });

  it("has no Move items for Main, and Delete node is off with the sentence that says why", async () => {
    const { menu } = await open(1);
    expect(labels(menu)).toEqual(["Rename", "Link picture to", "Link key to", "Remove link", "Delete node"]);
    const del = within(menu).getByRole("menuitem", { name: "Delete node" });
    expect(del).toHaveAttribute("aria-disabled", "true");
    expect(del).toHaveAccessibleDescription("Main can't be deleted. Bypass it instead.");
  });

  it("has Link picture to and Remove link for the Input, and for the Output Remove link alone, with nothing to rename, move or delete", async () => {
    const input = await open("input");
    expect(labels(input.menu)).toEqual(["Link picture to", "Remove link"]);
  });

  it("greys Move earlier for the node after Main and Move later for the last one", async () => {
    const first = await open(2);
    expect(within(first.menu).getByRole("menuitem", { name: "Move earlier" })).toHaveAttribute("aria-disabled", "true");
    expect(within(first.menu).getByRole("menuitem", { name: "Move later" })).not.toHaveAttribute("aria-disabled");
  });

  it("lists the inputs a picture can be linked to, by number and name, and the Output", async () => {
    const { user, menu, onLink } = await open(2);
    await user.click(within(menu).getByRole("menuitem", { name: "Link picture to" }));
    const items = (await screen.findAllByRole("menuitem")).map((i) => i.textContent?.trim());
    // Not Main (its picture comes from the input), not itself, not Sky (already linked): Warmth and the Output.
    expect(items.filter((t) => /^(Node|Output)/.test(t ?? ""))).toEqual(["Node 4, Warmth", "Output"]);
    await choose(user, screen.getByRole("menuitem", { name: "Output" }));
    expect(onLink).toHaveBeenCalledWith({ node: 2, port: "picture" }, { node: "output", port: "picture" });
  });

  it("lists the key inputs downstream of the node, and links the one chosen", async () => {
    const { user, menu, onLink } = await open(3);
    await user.click(within(menu).getByRole("menuitem", { name: "Link key to" }));
    expect((await screen.findAllByRole("menuitem", { name: /^Node/ })).map((i) => i.textContent?.trim())).toEqual(["Node 4, Warmth"]);
    await choose(user, screen.getByRole("menuitem", { name: "Node 4, Warmth" }));
    expect(onLink).toHaveBeenCalledWith({ node: 3, port: "key" }, { node: 4, port: "key" });
  });

  it("turns Link key to off when there is nowhere to link it, as the last node's", async () => {
    const { menu } = await open(4);
    expect(within(menu).getByRole("menuitem", { name: "Link key to" })).toHaveAttribute("aria-disabled", "true");
  });

  it("lists the node's links, in and out, picture and key, and removes the one chosen", async () => {
    const { user, menu, onRemoveLink } = await open(2);
    await user.click(within(menu).getByRole("menuitem", { name: "Remove link" }));
    const items = (await screen.findAllByRole("menuitem")).map((i) => i.textContent?.trim()).filter((t) => /^(Picture|Key) (from|to)/.test(t ?? ""));
    expect(items).toEqual(["Picture from Node 1, Main", "Picture to Node 3, Sky", "Key to Node 4, Warmth"]);
    await choose(user, screen.getByRole("menuitem", { name: "Key to Node 4, Warmth" }));
    expect(onRemoveLink).toHaveBeenCalledWith(key(2, 4));
  });

  it.each([
    ["Move earlier", "onMoveEarlier"],
    ["Move later", "onMoveLater"],
    ["Delete node", "onDelete"],
  ] as const)("asks for what %s says", async (name, handler) => {
    const opened = await open(3);
    await opened.user.click(within(opened.menu).getByRole("menuitem", { name }));
    expect(opened[handler]).toHaveBeenCalledTimes(1);
  });

  it("asks for Rename once the menu has closed, so that the text box keeps the focus", async () => {
    const { user, menu, onRename } = await open(3);
    await user.click(within(menu).getByRole("menuitem", { name: "Rename" }));
    await waitFor(() => expect(onRename).toHaveBeenCalledTimes(1));
    expect(screen.queryByRole("menu")).toBeNull();
  });
});
