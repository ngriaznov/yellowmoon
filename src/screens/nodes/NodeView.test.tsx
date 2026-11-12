/**
 * The node view's shell on the Photos screen (`docs/23` §6 and §7.1, `docs/11` §Node view): the Nodes button and N, layout A with its
 * divider and Expand, the toolbar with an empty canvas, the photo switcher, and the notice and glyph of the simple view. The screen runs
 * on the mock, as in a browser; the geometry in pixels is `layout.test.ts` and the browser's (`e2e/nodes.spec.ts`).
 */
import { act, fireEvent, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import type { MockOptions } from "../../api.mock";
import type { NodeLink, SettingsPatch } from "../../api/types";
import { mainPatch } from "../../state/settingsGraph";
import { openPhotos, openTab, renderApp, tools } from "../../test/harness";
import type { Harness } from "../../test/harness";

let h: Harness | undefined;
let windowHeight = window.innerHeight;
beforeEach(() => {
  windowHeight = window.innerHeight;
  // The screens of the references are 1440 × 900; jsdom's window is 768 high.
  Object.defineProperty(window, "innerHeight", { value: 900, configurable: true, writable: true });
});
afterEach(() => {
  h?.unmount();
  h = undefined;
  Object.defineProperty(window, "innerHeight", { value: windowHeight, configurable: true, writable: true });
});

const picture = (from: number | "input", to: number | "output"): NodeLink => ({ from: { node: from, port: "picture" }, to: { node: to, port: "picture" } });

/** Node 2 after Main. */
const TWO: SettingsPatch = {
  nodes: { "2": { kind: "corrector", name: "Warmth", pos: [1, 0] } },
  links: [picture("input", 1), picture(1, 2), picture(2, "output")],
};
/** Nodes 2, 3 and 4 after Main: four correctors, the photo of the mockup (`design/nodes/04-simple-photo-has-nodes.jpg`). */
const FOUR: SettingsPatch = {
  nodes: {
    "2": { kind: "corrector", name: "Mountain", pos: [1, 0] },
    "3": { kind: "corrector", name: "Sky", pos: [2, 0] },
    "4": { kind: "corrector", name: "Warmth", pos: [3, 0] },
  },
  links: [picture("input", 1), picture(1, 2), picture(2, 3), picture(3, 4), picture(4, "output")],
};

/** The mockup's state: photo 4 (DSC04204) current, photos 3 to 5 picked; `settings` is what DSC04204 opens with. */
async function setup(settings?: SettingsPatch, options: MockOptions = {}): Promise<Harness> {
  h = renderApp({ current: 3, picked: [2, 3, 4], ...(settings ? { initialSettings: { DSC04204: settings } } : {}), ...options });
  await openPhotos(h);
  return h;
}

const screenEl = (): HTMLElement => h!.container.querySelector<HTMLElement>('.ym-screen[data-screen="photos"]')!;
const header = (): HTMLElement => screen.getByRole("navigation", { name: "Actions" });
const nodesButton = (): HTMLElement => within(header()).getByRole("button", { name: "Nodes" });
const graph = (): HTMLElement => screen.getByRole("region", { name: "Node editor" });
const divider = (): HTMLElement => screen.getByRole("separator", { name: "Resize the graph" });
const switcher = (): HTMLElement => screen.getByRole("group", { name: "Photos" });
const current = (): string | undefined => h!.store.currentPhoto()?.file_name;
const picked = (): number[] => [...(h!.store.getState().session?.picked ?? [])];
const inNodeView = (): boolean => screenEl().getAttribute("data-nodes") === "true";

async function openNodeView(): Promise<void> {
  await h!.user.click(nodesButton());
  await waitFor(() => expect(inNodeView()).toBe(true));
}

describe("the Nodes button", () => {
  it("comes after Crop and before Add photos, a toggle with the tooltip 'Nodes (N)'", async () => {
    await setup();
    const labels = within(header()).getAllByRole("button").map((b) => b.textContent);
    expect(labels.slice(0, 5)).toEqual(["Before", "Crop", "Nodes", "Add photos", expect.stringMatching(/^Fit/)]);
    expect(nodesButton()).toHaveAttribute("aria-pressed", "false");
    expect(nodesButton()).toHaveAccessibleDescription("Nodes (N)");
    expect(nodesButton().closest("[data-slot]")).toHaveAttribute("data-slot", "toggle");
  });

  it("is pressed in the node view, and a click goes there and back", async () => {
    await setup();
    expect(inNodeView()).toBe(false);
    await h!.user.click(nodesButton());
    await waitFor(() => expect(nodesButton()).toHaveAttribute("aria-pressed", "true"));
    expect(inNodeView()).toBe(true);
    await h!.user.click(nodesButton());
    await waitFor(() => expect(nodesButton()).toHaveAttribute("aria-pressed", "false"));
    expect(inNodeView()).toBe(false);
  });

  it("is N too, from any tab", async () => {
    await setup();
    await openTab(h!, "Colour");
    await h!.user.keyboard("n");
    expect(inNodeView()).toBe(true);
    expect(nodesButton()).toHaveAttribute("aria-pressed", "true");
    await h!.user.keyboard("N");
    expect(inNodeView()).toBe(false);
  });

  it("does not take N typed into a text box, a key held down, a menu or a dialog", async () => {
    await setup();
    // A text box: the Look tab's search field.
    const search = screen.getByRole("searchbox", { name: /search/i });
    await h!.user.click(search);
    await h!.user.keyboard("n");
    expect(inNodeView()).toBe(false);
    expect(search).toHaveValue("n");
    // A key held down: the first press only.
    (document.activeElement as HTMLElement | null)?.blur();
    fireEvent.keyDown(document.body, { key: "n", repeat: true });
    expect(inNodeView()).toBe(false);
    fireEvent.keyDown(document.body, { key: "n" });
    expect(inNodeView()).toBe(true);
    fireEvent.keyDown(document.body, { key: "n", repeat: true });
    expect(inNodeView()).toBe(true);
    // A menu: the zoom menu's type-ahead.
    await h!.user.click(within(header()).getByRole("button", { name: /^Fit/ }));
    const menu = await screen.findByRole("menu");
    fireEvent.keyDown(within(menu).getAllByRole("menuitemradio")[0]!, { key: "n" });
    expect(inNodeView()).toBe(true);
    await h!.user.keyboard("{Escape}");
    await waitFor(() => expect(screen.queryByRole("menu")).toBeNull());
    // A dialog: Preferences.
    await h!.user.keyboard("{Meta>},{/Meta}");
    const dialog = await screen.findByRole("dialog");
    fireEvent.keyDown(dialog, { key: "n" });
    expect(inNodeView()).toBe(true);
  });

  it("is not Esc: Esc ends a pick first and never leaves the node view", async () => {
    await setup();
    await openNodeView();
    await h!.user.keyboard("{Escape}");
    expect(inNodeView()).toBe(true);
    // Pick white: Esc ends it, and the node view stays.
    await h!.user.keyboard("w");
    await screen.findByText("Click something neutral: grey or white. Esc cancels.");
    await h!.user.keyboard("{Escape}");
    await waitFor(() => expect(screen.queryByText("Click something neutral: grey or white. Esc cancels.")).toBeNull());
    expect(inNodeView()).toBe(true);
  });

  it("ends Pick white and Pick colour, as every way of leaving the Colour tab does", async () => {
    await setup();
    await h!.user.keyboard("w");
    await screen.findByText("Click something neutral: grey or white. Esc cancels.");
    await h!.user.click(nodesButton());
    await waitFor(() => expect(screen.queryByText("Click something neutral: grey or white. Esc cancels.")).toBeNull());
    expect(inNodeView()).toBe(true);
    // The node view opens on Main › Colour (the simple view stood on Colour), whose Pick colour is off; Pick white is the Input's.
    expect(screen.queryByRole("button", { name: "Pick white" })).toBeNull();
    expect(screen.getByRole("button", { name: "Pick colour" })).toHaveAttribute("aria-pressed", "false");
  });

  it("is off while cropping, and N does nothing there; the Export panel has no Nodes button, and N does nothing there either", async () => {
    await setup();
    await openNodeView();
    await h!.user.click(screen.getByRole("button", { name: "Crop" }));
    await screen.findByRole("complementary", { name: "Crop" });
    expect(nodesButton()).toHaveAttribute("aria-disabled", "true");
    await h!.user.click(nodesButton());
    await h!.user.keyboard("n");
    expect(inNodeView()).toBe(true);
    await h!.user.keyboard("{Escape}");
    await waitFor(() => expect(screen.queryByRole("complementary", { name: "Crop" })).toBeNull());
    expect(nodesButton()).not.toHaveAttribute("aria-disabled");
    // Export.
    await h!.user.keyboard("{Meta>}e{/Meta}");
    await screen.findByRole("complementary", { name: "Export" });
    expect(within(header()).queryByRole("button", { name: "Nodes" })).toBeNull();
    await h!.user.keyboard("n");
    expect(inNodeView()).toBe(true);
  });

  it("gives way to full screen: the photo alone, and the node view back with F", async () => {
    await setup();
    await openNodeView();
    await h!.user.keyboard("f");
    await waitFor(() => expect(screenEl()).toHaveAttribute("data-fullscreen", "true"));
    expect(inNodeView()).toBe(false);
    expect(screen.queryByRole("separator", { name: "Resize the graph" })).toBeNull();
    expect(screen.queryByRole("region", { name: "Node editor" })).toBeNull();
    await h!.user.keyboard("f");
    await waitFor(() => expect(screenEl()).not.toHaveAttribute("data-fullscreen"));
    expect(inNodeView()).toBe(true);
    expect(nodesButton()).toHaveAttribute("aria-pressed", "true");
  });
});

describe("layout A", () => {
  it("replaces the filmstrip with the divider and the graph, the panel staying where it is", async () => {
    await setup();
    expect(screen.getByRole("navigation", { name: "Photos" })).toBeInTheDocument();
    expect(screen.queryByRole("separator")).toBeNull();
    await openNodeView();
    expect(screen.queryByRole("navigation", { name: "Photos" })).toBeNull();
    expect(divider()).toBeInTheDocument();
    expect(graph()).toBeInTheDocument();
    expect(screen.getByRole("complementary", { name: "Tools" })).toBeInTheDocument();
    expect(screen.getByRole("main", { name: "Photo" })).toBeInTheDocument();
    // In the grid's order: the header, the photo, the divider, the graph and the panel.
    const order = [...screenEl().children].map((el) => el.className.toString().split(" ").find((c) => /^ym-(header|viewer|divider|graph|tools|filmstrip)$/.test(c)));
    expect(order.filter(Boolean)).toEqual(["ym-header", "ym-viewer", "ym-divider", "ym-graph", "ym-filmstrip", "ym-tools"]);
    expect(screenEl().querySelector(".ym-filmstrip")).not.toBeVisible();
    // And back.
    await h!.user.click(nodesButton());
    expect(screen.getByRole("navigation", { name: "Photos" })).toBeInTheDocument();
    expect(screen.queryByRole("separator")).toBeNull();
    expect(screen.queryByRole("region", { name: "Node editor" })).toBeNull();
  });

  it("gives the screen its sizes: the editor at 45 % of the height under the header, the photo row the rest", async () => {
    await setup();
    expect(screenEl().style.getPropertyValue("--editor-h")).toBe("");
    await openNodeView();
    // 1440 × 900: an editor of 371 px and a photo row of 444 px (a cell of 666 × 444).
    expect(screenEl().style.getPropertyValue("--editor-h")).toBe("371px");
    expect(screenEl().style.getPropertyValue("--viewer-h")).toBe("444px");
  });

  it("follows the divider and Expand, and keeps both while the node view is closed and opened again", async () => {
    await setup();
    await openNodeView();
    divider().focus();
    await h!.user.keyboard("{ArrowUp}");
    expect(screenEl().style.getPropertyValue("--editor-h")).toBe("381px");
    expect(screenEl().style.getPropertyValue("--viewer-h")).toBe("434px");
    await h!.user.click(screen.getByRole("button", { name: "Expand" }));
    expect(screen.getByRole("button", { name: "Expand" })).toHaveAttribute("aria-pressed", "true");
    expect(screenEl().style.getPropertyValue("--editor-h")).toBe("545px");
    expect(screenEl().style.getPropertyValue("--viewer-h")).toBe("270px");
    await h!.user.click(nodesButton());
    await h!.user.click(nodesButton());
    expect(screen.getByRole("button", { name: "Expand" })).toHaveAttribute("aria-pressed", "true");
    await h!.user.click(screen.getByRole("button", { name: "Expand" }));
    expect(screenEl().style.getPropertyValue("--editor-h")).toBe("381px");
  });

  it("has the toolbar of docs/23 §7.3 over the graph, and the photo switcher in it", async () => {
    await setup();
    // The node view opens on the node of the tab the simple view stands on: Light is Main's.
    await openTab(h!, "Light");
    await openNodeView();
    const bar = within(graph()).getByRole("group", { name: "Graph tools" });
    const labels = within(bar).getAllByRole("button").map((b) => b.getAttribute("aria-label") ?? b.textContent?.trim());
    // More is the third icon of the node's head since N-09, not a button of the toolbar.
    expect(labels).toEqual(["Add node", "Bypass", "Delete", "Previous photo", "Next photo", "Show mask", "Expand"]);
    // With Main selected: Bypass is on (and not pressed), Delete says why it is off, Show mask is on (a corrector has a mask) and not pressed.
    expect(within(bar).getByRole("button", { name: "Bypass" })).not.toHaveAttribute("aria-disabled");
    expect(within(bar).getByRole("button", { name: "Bypass" })).toHaveAttribute("aria-pressed", "false");
    expect(within(bar).getByRole("button", { name: "Bypass" })).toHaveAccessibleDescription("Bypass this node (B)");
    expect(within(bar).getByRole("button", { name: "Delete" })).toHaveAttribute("aria-disabled", "true");
    expect(within(bar).getByRole("button", { name: "Delete" })).toHaveAccessibleDescription("Main can't be deleted. Bypass it instead.");
    expect(within(bar).getByRole("button", { name: "Show mask" })).not.toHaveAttribute("aria-disabled");
    expect(within(bar).getByRole("button", { name: "Show mask" })).toHaveAccessibleDescription("Show mask (M)");
    expect(within(bar).getByRole("button", { name: "Show mask" })).toHaveAttribute("aria-pressed", "false");
    expect(within(bar).getByRole("button", { name: "Expand" })).toHaveAttribute("aria-pressed", "false");
    expect(within(bar).getByRole("button", { name: "Expand" })).not.toHaveAttribute("aria-disabled");
    // Main is selected, as the panel's first node, and the canvas draws the graph.
    expect(graph().querySelector(".ym-graph__canvas")).toHaveAttribute("data-selected-node", "1");
    expect(graph().querySelector(".ym-graph__canvas")).not.toBeEmptyDOMElement();
  });

  it("dims the graph and makes it read only while Crop or Export shows", async () => {
    await setup();
    await openNodeView();
    expect(graph()).not.toHaveAttribute("data-dimmed");
    expect(graph()).not.toHaveAttribute("inert");
    await h!.user.click(screen.getByRole("button", { name: "Crop" }));
    await screen.findByRole("complementary", { name: "Crop" });
    expect(graph()).toHaveAttribute("data-dimmed", "true");
    expect(graph()).toHaveAttribute("inert");
    await h!.user.keyboard("{Escape}");
    await waitFor(() => expect(graph()).not.toHaveAttribute("data-dimmed"));
    await h!.user.keyboard("{Meta>}e{/Meta}");
    await screen.findByRole("complementary", { name: "Export" });
    expect(graph()).toHaveAttribute("data-dimmed", "true");
    expect(graph()).toHaveAttribute("inert");
    await h!.user.click(screen.getByRole("button", { name: "← Back to photos" }));
    await waitFor(() => expect(graph()).not.toHaveAttribute("inert"));
  });
});

describe("the Add node menu", () => {
  it("opens upward from the button: Corrector, Film, Face Refinement, then Parallel, Layer and Splitter and combiner, all on, each with its line", async () => {
    await setup();
    await openNodeView();
    await h!.user.click(within(graph()).getByRole("button", { name: "Add node" }));
    const menu = await screen.findByRole("menu", { name: "Add node" });
    expect(menu).toHaveAttribute("data-side", "top");
    const items = within(menu).getAllByRole("menuitem");
    expect(items.map((i) => i.querySelector(".ym-menu-line__name")?.textContent)).toEqual(["Corrector", "Film", "Face Refinement", "Parallel", "Layer", "Splitter and combiner"]);
    expect(within(items[0]!).getByText("Grades all or part of the photo.")).toBeInTheDocument();
    expect(within(items[1]!).getByText("Models a colour negative, a print and a scan.")).toBeInTheDocument();
    expect(items[1]).not.toHaveAttribute("aria-disabled");
    expect(within(items[2]!).getByText("Refines each face: skin, under-eyes and teeth.")).toBeInTheDocument();
    // The three items of `docs/25` are on, with their lines, and no "Later" tag.
    expect(within(items[3]!).getByText("Grade branches side by side.")).toBeInTheDocument();
    expect(within(items[4]!).getByText("Blend one branch over another.")).toBeInTheDocument();
    expect(within(items[5]!).getByText("Split a picture, then join it again.")).toBeInTheDocument();
    for (const item of items) {
      expect(item).not.toHaveAttribute("aria-disabled");
      expect(within(item).queryByText("Later")).toBeNull();
    }
    await h!.user.keyboard("{Escape}");
    await waitFor(() => expect(screen.queryByRole("menu")).toBeNull());
  });
});

describe("the photo switcher", () => {
  it("reads '4 of 12' and moves to the next and the previous photo, as the filmstrip's keys do", async () => {
    await setup();
    await openNodeView();
    expect(within(switcher()).getByText("4 of 12")).toBeInTheDocument();
    await h!.user.click(within(switcher()).getByRole("button", { name: "Next photo" }));
    await waitFor(() => expect(current()).toBe("DSC04205.ARW"));
    expect(within(switcher()).getByText("5 of 12")).toBeInTheDocument();
    await h!.user.click(within(switcher()).getByRole("button", { name: "Previous photo" }));
    await h!.user.click(within(switcher()).getByRole("button", { name: "Previous photo" }));
    await waitFor(() => expect(current()).toBe("DSC04203.ARW"));
    expect(within(switcher()).getByText("3 of 12")).toBeInTheDocument();
    // Still the node view.
    expect(inNodeView()).toBe(true);
  });

  it("is off at the ends of the folder, and keeps the focus (aria-disabled, not disabled)", async () => {
    await setup(undefined, { current: 0, picked: [] });
    await openNodeView();
    expect(within(switcher()).getByText("1 of 12")).toBeInTheDocument();
    const previous = within(switcher()).getByRole("button", { name: "Previous photo" });
    expect(previous).toHaveAttribute("aria-disabled", "true");
    expect(previous).not.toBeDisabled();
    await h!.user.click(previous);
    expect(current()).toBe("DSC04201.ARW");
    expect(within(switcher()).getByRole("button", { name: "Next photo" })).not.toHaveAttribute("aria-disabled");
    h!.unmount();
    h = undefined;
    await setup(undefined, { current: 11, picked: [] });
    await openNodeView();
    expect(within(switcher()).getByText("12 of 12")).toBeInTheDocument();
    expect(within(switcher()).getByRole("button", { name: "Next photo" })).toHaveAttribute("aria-disabled", "true");
  });

  it("leaves ← and → switching photos, P picking, and Delete and Backspace taking no photo out", async () => {
    await setup();
    await openNodeView();
    await h!.user.keyboard("{ArrowRight}");
    await waitFor(() => expect(current()).toBe("DSC04205.ARW"));
    expect(within(switcher()).getByText("5 of 12")).toBeInTheDocument();
    await h!.user.keyboard("{ArrowLeft}{ArrowLeft}");
    await waitFor(() => expect(current()).toBe("DSC04203.ARW"));
    // Focus on a button of the switcher does not stop the arrows either.
    within(switcher()).getByRole("button", { name: "Next photo" }).focus();
    await h!.user.keyboard("{ArrowRight}");
    await waitFor(() => expect(current()).toBe("DSC04204.ARW"));
    // P picks the photo on screen (it was picked: now it is not).
    expect(picked()).toEqual([3, 4, 5]);
    await h!.user.keyboard("p");
    await waitFor(() => expect(picked()).toEqual([3, 5]));
    // Delete never removes a photo here.
    await h!.user.keyboard("{Delete}{Backspace}");
    expect(h!.store.getState().session?.photos).toHaveLength(12);
    expect(current()).toBe("DSC04204.ARW");
    expect(screen.queryByText(/Removed/)).toBeNull();
    // In the simple view it still does.
    await h!.user.click(nodesButton());
    await waitFor(() => expect(inNodeView()).toBe(false));
    await h!.user.keyboard("{Delete}");
    await waitFor(() => expect(h!.store.getState().session?.photos).toHaveLength(11));
  });
});

describe("the notice of the simple view", () => {
  const notice = (text: string | RegExp): HTMLElement => screen.getByText(text).closest<HTMLElement>("[data-slot='item']")!;

  it("is not there for a photo whose graph the simple view shows all of", async () => {
    await setup();
    expect(within(tools()).queryByRole("button", { name: "Open nodes" })).toBeNull();
    expect(screen.queryByText(/This photo has/)).toBeNull();
    expect(screenEl().querySelector(".ym-filmstrip__nodes")).toBeNull();
  });

  it("with four nodes says so under the tab row of every tab, with Open nodes at the end of the text", async () => {
    await setup(FOUR);
    for (const name of ["Look", "Light", "Colour", "RAW"]) {
      await openTab(h!, name);
      const text = "This photo has 4 nodes. These controls change the input, the output and node 1, Main.";
      const item = notice(/This photo has 4 nodes\./);
      expect(item, name).toHaveAttribute("role", "status");
      expect(item).toHaveTextContent(`${text} Open nodes`);
      expect(within(item).getByRole("button", { name: "Open nodes" })).toBeInTheDocument();
      // Under the tab row and above the cards, in the panel but outside what scrolls.
      const panel = tools();
      const kids = [...panel.children];
      expect(kids.indexOf(item.parentElement!), name).toBe(1);
      expect(kids[0]).toHaveAttribute("data-slot", "tabs");
      expect(kids[2]).toHaveClass("ym-tools__scroll");
      expect(within(kids[2] as HTMLElement).queryByText(/This photo has/)).toBeNull();
    }
  });

  it("counts the correctors: two nodes", async () => {
    await setup(TWO);
    expect(notice(/This photo has 2 nodes\./)).toHaveTextContent("This photo has 2 nodes. These controls change the input, the output and node 1, Main. Open nodes");
  });

  it("for Main alone says what Main has: a trim, masks or a bypass", async () => {
    const text = "Node 1, Main, has a trim, masks or a bypass that these controls don't show.";
    await setup(mainPatch({ trim: { exposure_ev: 0.5 } }));
    expect(notice(text)).toHaveTextContent(`${text} Open nodes`);
    h!.unmount();
    await setup(mainPatch({ bypass: true }));
    expect(notice(text)).toBeInTheDocument();
    h!.unmount();
    await setup(mainPatch({ masks: { invert: false, items: [{ kind: "linear", mode: "add", invert: false, enabled: true, opacity: 1, center: [0.5, 0.5], angle_deg: 0, feather: 0.2 }] } }));
    expect(notice(text)).toBeInTheDocument();
  });

  it("comes and goes with the photo, and with a change the person makes", async () => {
    await setup(FOUR);
    expect(screen.getByText(/This photo has 4 nodes/)).toBeInTheDocument();
    await h!.user.keyboard("{ArrowRight}");
    await waitFor(() => expect(current()).toBe("DSC04205.ARW"));
    await waitFor(() => expect(screen.queryByText(/This photo has/)).toBeNull());
    await h!.user.keyboard("{ArrowLeft}");
    await screen.findByText(/This photo has 4 nodes/);
    // A photo that has none gets the notice when its Main is trimmed.
    await h!.user.keyboard("{ArrowRight}");
    await waitFor(() => expect(current()).toBe("DSC04205.ARW"));
    await act(async () => {
      await h!.store.update(mainPatch({ trim: { temperature: 20 } }));
    });
    await screen.findByText(/Node 1, Main, has a trim/);
  });

  it("Open nodes goes to the node view with Main selected, where there is no notice", async () => {
    await setup(FOUR);
    await openTab(h!, "Light");
    await h!.user.click(within(tools()).getByRole("button", { name: "Open nodes" }));
    await waitFor(() => expect(inNodeView()).toBe(true));
    expect(nodesButton()).toHaveAttribute("aria-pressed", "true");
    expect(graph().querySelector(".ym-graph__canvas")).toHaveAttribute("data-selected-node", "1");
    expect(screen.queryByText(/This photo has/)).toBeNull();
    await h!.user.click(nodesButton());
    await screen.findByText(/This photo has 4 nodes/);
  });
});

describe("the glyph on the filmstrip", () => {
  it("marks the thumbnail of a photo with nodes, with '4 nodes' as its words, and no other", async () => {
    await setup(FOUR);
    const thumb = screen.getByRole("button", { name: "Show DSC04204, picked, 4 nodes" });
    expect(thumb.querySelector(".ym-filmstrip__nodes")).toHaveAttribute("data-nodes", "4 nodes");
    expect(screenEl().querySelectorAll(".ym-filmstrip__nodes")).toHaveLength(1);
    expect(screen.getByRole("button", { name: "Show DSC04205, picked" })).toBeInTheDocument();
  });

  it("says 'Node 1 has masks or a trim' for Main alone with one, and takes the glyph away with the cause", async () => {
    await setup(mainPatch({ trim: { exposure_ev: 0.5 } }));
    expect(screen.getByRole("button", { name: "Show DSC04204, picked, node 1 has masks or a trim" })).toBeInTheDocument();
    expect(screenEl().querySelector(".ym-filmstrip__nodes")).toHaveAttribute("data-nodes", "Node 1 has masks or a trim");
    await act(async () => {
      await h!.store.update(mainPatch({ trim: { exposure_ev: 0 } }));
    });
    await waitFor(() => expect(screenEl().querySelector(".ym-filmstrip__nodes")).toBeNull());
  });
});


describe("Show mask (N-11)", () => {
  const showMask = (): HTMLElement => within(within(graph()).getByRole("group", { name: "Graph tools" })).getByRole("button", { name: "Show mask" });
  /** A click on a card of the graph. */
  const selectCard = async (name: RegExp): Promise<void> => {
    const card = within(graph()).getByRole("button", { name });
    fireEvent.click(card);
    await waitFor(() => expect(card).toHaveAttribute("aria-pressed", "true"));
  };

  /** The node view opens on the Output (the Look tab stands behind it); Main is the corrector these tests start on. */
  const openOnMain = async (): Promise<void> => {
    await openNodeView();
    await selectCard(/^Node 1, Main/);
  };

  it("is a toggle: a click turns the view on for the selected corrector, with the Overlay style, and another turns it off", async () => {
    await setup(FOUR);
    await openOnMain();
    expect(h!.api.viewport.maskView).toBeNull();
    await h!.user.click(showMask());
    await waitFor(() => expect(showMask()).toHaveAttribute("aria-pressed", "true"));
    expect(h!.api.viewport.maskView).toEqual({ node: 1, style: "overlay" });
    await h!.user.click(showMask());
    await waitFor(() => expect(showMask()).toHaveAttribute("aria-pressed", "false"));
    expect(h!.api.viewport.maskView).toBeNull();
  });

  it("follows the selected corrector, and is off, with its sentence, on the Input and the Output; the choice comes back with a corrector", async () => {
    await setup(FOUR);
    await openOnMain();
    await h!.user.click(showMask());
    await selectCard(/^Node 3, Sky/);
    await waitFor(() => expect(h!.api.viewport.maskView).toEqual({ node: 3, style: "overlay" }));
    await selectCard(/^Input/);
    await waitFor(() => expect(h!.api.viewport.maskView).toBeNull());
    expect(showMask()).toHaveAttribute("aria-disabled", "true");
    expect(showMask()).toHaveAccessibleDescription("The input and the output have no mask.");
    expect(showMask()).toHaveAttribute("aria-pressed", "false");
    // A click on it does nothing.
    await h!.user.click(showMask());
    expect(h!.store.getState().maskView.on).toBe(true);
    await selectCard(/^Output/);
    expect(showMask()).toHaveAttribute("aria-disabled", "true");
    await selectCard(/^Node 1, Main/);
    await waitFor(() => expect(h!.api.viewport.maskView).toEqual({ node: 1, style: "overlay" }));
    expect(showMask()).toHaveAttribute("aria-pressed", "true");
  });

  it("is M in the node view, with the last style; M does nothing on the Input", async () => {
    await setup();
    await openOnMain();
    h!.store.setMaskView({ style: "mask_only", on: false });
    await h!.user.keyboard("m");
    await waitFor(() => expect(h!.api.viewport.maskView).toEqual({ node: 1, style: "mask_only" }));
    await h!.user.keyboard("m");
    await waitFor(() => expect(h!.api.viewport.maskView).toBeNull());
    await selectCard(/^Input/);
    await h!.user.keyboard("m");
    expect(h!.store.getState().maskView.on).toBe(false);
  });

  it("ends when the node view is left, Crop or Export opens, and comes back as it was", async () => {
    await setup();
    await openOnMain();
    h!.store.setMaskView({ style: "mask_only" });
    await waitFor(() => expect(h!.api.viewport.maskView).toEqual({ node: 1, style: "mask_only" }));
    // The simple view has no mask view.
    await h!.user.click(nodesButton());
    await waitFor(() => expect(inNodeView()).toBe(false));
    await waitFor(() => expect(h!.api.viewport.maskView).toBeNull());
    await openOnMain();
    await waitFor(() => expect(h!.api.viewport.maskView).toEqual({ node: 1, style: "mask_only" }));
    // Crop.
    await h!.user.click(within(header()).getByRole("button", { name: "Crop" }));
    await waitFor(() => expect(h!.api.viewport.maskView).toBeNull());
    await h!.user.click(within(header()).getByRole("button", { name: "Crop" }));
    await waitFor(() => expect(h!.api.viewport.maskView).toEqual({ node: 1, style: "mask_only" }));
    // Export.
    act(() => h!.store.openExport());
    await waitFor(() => expect(h!.api.viewport.maskView).toBeNull());
    act(() => h!.store.closeExport());
    await waitFor(() => expect(h!.api.viewport.maskView).toEqual({ node: 1, style: "mask_only" }));
  });
});
