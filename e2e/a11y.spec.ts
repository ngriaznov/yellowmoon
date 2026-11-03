import { expect, test } from "@playwright/test";
import type { Page } from "@playwright/test";
import { configureMock, openApp, openPhotos, setValue } from "./helpers/app";
import { FOUR_NODES, GRADED_NODES } from "./helpers/states";

// Accessibility basics (docs/11, a preview of M4-04) in a real browser: every control has a role and a name, every screen has one
// main landmark, the keyboard alone gets through the main flow, and every stop of the Tab order shows the 3 px focus ring.
// Contrast of the tab colours is in src/styles/contrast.node.test.ts.

const MOCKUP = { mockCurrent: 3, mockPicked: "2,3,4" } as const;

/** Roles that name something to press, type into or choose: without a name a screen reader says only "button". */
const NEEDS_A_NAME = new Set([
  "button",
  "checkbox",
  "combobox",
  "dialog",
  "link",
  "menuitem",
  "menuitemradio",
  "option",
  "radio",
  "radiogroup",
  "searchbox",
  "slider",
  "spinbutton",
  "switch",
  "tab",
  "tablist",
  "textbox",
]);

/** The accessibility tree of the page as Playwright prints it. */
const tree = (page: Page): Promise<string> => page.locator("body").ariaSnapshot();

/** Lines of the tree that are controls without a name. */
function unnamed(snapshot: string): string[] {
  const found: string[] = [];
  for (const line of snapshot.split("\n")) {
    const match = /^\s*- ([a-z]+)((?: "(?:[^"\\]|\\.)*")?)((?: \[[^\]]*\])*)(?::.*)?$/.exec(line);
    if (match && NEEDS_A_NAME.has(match[1] as string) && !match[2]) found.push(line.trim());
  }
  return found;
}

const count = (snapshot: string, role: string): number => (snapshot.match(new RegExp(`^\\s*- ${role}\\b`, "gm")) ?? []).length;

async function expectNamed(page: Page, where: string): Promise<string> {
  const snapshot = await tree(page);
  expect(unnamed(snapshot), `${where}: controls without a name`).toEqual([]);
  return snapshot;
}

async function toWizard(page: Page): Promise<void> {
  await openApp(page, { mockLooks: 0 });
  await page.getByRole("button", { name: "Import →" }).click();
  await expect(page.getByRole("heading", { level: 1, name: "Choose what to import" })).toBeVisible();
}

test.describe("the node view", () => {
  // docs/23 §7.8: the divider, the toolbar's toggles, the switcher and the menu items have names; since N-08 the cards, the markers, the zoom
  // control and the More menu have too (the node panel's tabs come with N-09 to N-12, and this file walks them then).
  test("every control is named, there is one main and one banner, the divider is a separator with a name and a value, the Add node menu's items are named and the later ones disabled", async ({ page }) => {
    await openApp(page, MOCKUP);
    await openPhotos(page);
    // The node view opens on the node of the simple view's tab (docs/23 §7.5): Light is Main's.
    await page.getByRole("tab", { name: "Light", exact: true }).click();
    await page.getByRole("button", { name: "Nodes", exact: true }).click();
    await expect(page.getByRole("separator", { name: "Resize the graph" })).toBeVisible();
    let snapshot = await expectNamed(page, "the node view");
    expect(count(snapshot, "main")).toBe(1);
    expect(count(snapshot, "banner")).toBe(1);
    expect(snapshot).toMatch(/- separator "Resize the graph"/);
    expect(snapshot).toMatch(/- region "Node editor"/);
    expect(snapshot).toMatch(/- group "Graph tools"/);
    expect(snapshot).toMatch(/- group "Photos"/);
    for (const name of ["Add node", "Bypass", "Delete", "Previous photo", "Next photo", "Show mask", "Expand"]) expect(snapshot, name).toMatch(new RegExp(`- button "${name}"`));
    // The node's head (N-09): a named group with Bypass, Reset and More, the tabs of a corrector, and the panel's own cards.
    expect(snapshot).toMatch(/- group "Node":/);
    for (const name of ["Bypass this node", "Reset this node", "More"]) expect(snapshot, name).toMatch(new RegExp(`- button "${name}"`));
    expect(snapshot).toMatch(/- heading "Main" \[level=2\]/);
    for (const name of ["Light", "Colour", "Curve", "Mask"]) expect(snapshot, name).toMatch(new RegExp(`- tab "${name}"`));
    // The graph: Main between the Input and the Output, each one button with its name and whether it is the selected one; the zoom control.
    expect(snapshot).toMatch(/- button "Input"/);
    expect(snapshot).toMatch(/- button "Node 1, Main" \[pressed\]/);
    expect(snapshot).toMatch(/- button "Output"/);
    expect(snapshot).toMatch(/- application "Graph":/);
    expect(snapshot).toMatch(/- group "Zoom":/);
    for (const name of ["Zoom out", "Zoom in", "Fit the graph"]) expect(snapshot, name).toMatch(new RegExp(`- button "${name}"`));
    // The ports and the links are drawn for the pointer: a screen reader lists none of them.
    expect(snapshot).not.toMatch(/Edge from/);
    expect(snapshot).not.toMatch(/^\s*- img$/m);
    // Toggles say whether they are pressed; the Nodes button too.
    expect(snapshot).toMatch(/- button "Nodes" \[pressed\]/);
    expect(snapshot).toMatch(/- button "Expand"\n/);
    await page.getByRole("button", { name: "Expand", exact: true }).click();
    snapshot = await expectNamed(page, "the node view, expanded");
    expect(snapshot).toMatch(/- button "Expand" \[pressed\]/);
    // The menu.
    await page.getByRole("button", { name: "Add node" }).click();
    await expect(page.getByRole("menu", { name: "Add node" })).toBeVisible();
    snapshot = await expectNamed(page, "the Add node menu");
    for (const name of ["Corrector", "Parallel", "Layer", "Splitter and combiner", "Outside"]) expect(snapshot, name).toContain(`- menuitem "${name}`);
    for (const name of ["Parallel", "Layer", "Splitter and combiner", "Outside"]) {
      await expect(page.getByRole("menuitem", { name: new RegExp(`^${name}`) })).toHaveAttribute("aria-disabled", "true");
    }
  });

  test("the graph with nodes: every card is named with its number, name, masks and what feeds it; the More menu and its lists are named; a link's menu too", async ({ page }) => {
    await openApp(page, MOCKUP);
    await configureMock(page, { initialSettings: { DSC04204: { nodes: GRADED_NODES.nodes, links: GRADED_NODES.links } } as never });
    await openPhotos(page);
    await page.getByRole("tab", { name: "Light", exact: true }).click();
    await page.getByRole("button", { name: "Nodes", exact: true }).click();
    await expect(page.locator('[data-node-card="4"]')).toBeVisible();
    let snapshot = await expectNamed(page, "the graph with nodes");
    expect(snapshot).toContain('- button "Node 1, Main" [pressed]');
    expect(snapshot).toContain('- button "Node 2, Mountain, masked by Object and Radial, picture from node 1"');
    expect(snapshot).toContain('- button "Node 3, Sky, masked by Linear, picture from node 2"');
    expect(snapshot).toContain('- button "Node 4, Warmth, picture from node 3"');
    // A bypassed card and a card that is not connected say so; selection moves the pressed state.
    await page.locator('[data-node-card="3"]').click();
    await page.keyboard.press("b");
    snapshot = await expectNamed(page, "a bypassed card");
    expect(snapshot).toContain('- button "Node 3, Sky, masked by Linear, bypassed, picture from node 2" [pressed]');
    expect(snapshot).toMatch(/- button "Bypass" \[pressed\]/);
    // More: the items and the three lists.
    await page.getByRole("button", { name: "More" }).click();
    snapshot = await expectNamed(page, "the More menu");
    for (const name of ["Rename", "Move earlier", "Move later", "Link picture to", "Link key to", "Remove link", "Delete node"]) expect(snapshot, name).toContain(`- menuitem "${name}"`);
    for (const name of ["Link picture to", "Link key to", "Remove link"]) {
      await page.getByRole("menuitem", { name }).click();
      snapshot = await expectNamed(page, `the ${name} list`);
      await page.keyboard.press("ArrowLeft");
    }
    await page.keyboard.press("Escape");
    // The Delete link menu of a link.
    const a = (await page.locator('.ym-port__hit[data-port-node="1"][data-port-name="picture"][data-port-side="out"]').boundingBox())!;
    const b = (await page.locator('.ym-port__hit[data-port-node="2"][data-port-name="picture"][data-port-side="in"]').boundingBox())!;
    await page.mouse.click((a.x + a.width / 2 + b.x + b.width / 2) / 2, a.y + a.height / 2, { button: "right" });
    snapshot = await expectNamed(page, "a link's menu");
    expect(snapshot).toContain('- menuitem "Delete link"');
    await page.keyboard.press("Escape");
    // An error says why in a named alert: a refused link.
    const out = (await page.locator('.ym-port__hit[data-port-node="2"][data-port-name="picture"][data-port-side="out"]').boundingBox())!;
    const main = (await page.locator('.ym-port__hit[data-port-node="1"][data-port-name="picture"][data-port-side="in"]').boundingBox())!;
    await page.mouse.move(out.x + out.width / 2, out.y + out.height / 2);
    await page.mouse.down();
    await page.mouse.move(main.x + main.width / 2, main.y + main.height / 2, { steps: 8 });
    await page.mouse.up();
    await expect(page.getByRole("alert")).toHaveText(/Main always takes its picture from the input\./);
    await expectNamed(page, "a refusal");
  });

  test("the graph is one Tab stop: the selected card, then the zoom control; the arrows move between the nodes, Return selects", async ({ page }) => {
    await openApp(page, MOCKUP);
    await configureMock(page, { initialSettings: { DSC04204: { nodes: GRADED_NODES.nodes, links: GRADED_NODES.links } } as never });
    await openPhotos(page);
    await page.getByRole("tab", { name: "Light", exact: true }).click();
    await page.getByRole("button", { name: "Nodes", exact: true }).click();
    await expect(page.locator('[data-node-card="4"]')).toBeVisible();
    // Tab through the canvas: the stops that are in it, from the toolbar's last button until the focus leaves.
    await page.getByRole("button", { name: "Expand", exact: true }).focus();
    const inside: string[] = [];
    for (let i = 0; i < 8; i++) {
      await page.keyboard.press("Tab");
      const here = await page.evaluate(() => {
        const el = document.activeElement as HTMLElement | null;
        return el?.closest(".ym-graph__canvas") ? (el.getAttribute("aria-label") ?? el.textContent ?? "").trim() : null;
      });
      if (here === null) break;
      inside.push(here);
    }
    // One card (the selected one, Main), and the three buttons of the zoom control.
    expect(inside).toEqual(["Node 1, Main", "Zoom out", "Zoom in", "Fit the graph"]);
    // Shift+Tab comes back to the card, and the arrows go from node to node with the selection.
    for (let i = 0; i < 4; i++) await page.keyboard.press("Shift+Tab");
    await expect(page.locator('[data-node-card="1"]')).toBeFocused();
    await page.keyboard.press("ArrowRight");
    await expect(page.locator('[data-node-card="2"]')).toBeFocused();
    await expect(page.locator('[data-node-card="2"]')).toHaveAttribute("aria-pressed", "true");
    // Tab now lands on the node the arrows chose.
    await page.getByRole("button", { name: "Expand", exact: true }).focus();
    await page.keyboard.press("Tab");
    await expect(page.locator('[data-node-card="2"]')).toBeFocused();
    // Return and Space select the focused card (a card's own button), and only that one is pressed.
    await page.keyboard.press("ArrowRight");
    await page.keyboard.press("ArrowRight");
    await page.keyboard.press("Enter");
    await expect(page.locator('[data-node-card="4"]')).toHaveAttribute("aria-pressed", "true");
    await expect(page.locator('[data-node-card][aria-pressed="true"]')).toHaveCount(1);
  });

  test("a photo that has nodes: the notice is a named status with a link, the thumbnail's name says it has nodes", async ({ page }) => {
    await openApp(page, MOCKUP);
    await configureMock(page, { initialSettings: { DSC04204: { nodes: FOUR_NODES.nodes, links: FOUR_NODES.links } } as never });
    await openPhotos(page);
    const snapshot = await expectNamed(page, "a photo that has nodes");
    expect(snapshot).toMatch(/- status:\n?\s*- text: .*This photo has 4 nodes\./s);
    expect(snapshot).toMatch(/- button "Open nodes"/);
    expect(snapshot).toMatch(/- button "Show DSC04204, picked, 4 nodes"/);
    // A status is a polite live region by its role: the notice is read once, when it appears.
    await expect(page.getByRole("status").filter({ hasText: "This photo has 4 nodes." })).toBeVisible();
  });
});

test.describe("names and landmarks", () => {
  test("Start: every control is named, one main landmark, one h1", async ({ page }) => {
    await openApp(page, { mockLooks: 0 });
    let snapshot = await expectNamed(page, "Start");
    expect(count(snapshot, "main")).toBe(1);
    expect(count(snapshot, "banner")).toBe(1);
    expect(snapshot).toMatch(/heading "Drop Sony ARW files\." \[level=1\]/);
    await page.getByRole("button", { name: "Open files…" }).click();
    await expect(page.getByRole("heading", { level: 2, name: "12 photos ready" })).toBeVisible();
    snapshot = await expectNamed(page, "Start, developed");
    expect(count(snapshot, "main")).toBe(1);
    expect(count(snapshot, "heading")).toBeGreaterThanOrEqual(2);
  });

  test("Photos: every tab, Crop, Pick white and the zoom menu", async ({ page }) => {
    await openApp(page, MOCKUP);
    await openPhotos(page);
    let snapshot = await expectNamed(page, "Photos");
    expect(count(snapshot, "main")).toBe(1);
    expect(count(snapshot, "banner")).toBe(1);
    // The folder name is the screen's heading.
    expect(snapshot).toMatch(/heading "Lisbon" \[level=1\]/);
    for (const tab of ["Look", "Lens", "Light", "Colour", "Detail", "Curve", "RAW"]) {
      await page.getByRole("tab", { name: tab }).click();
      await expect(page.getByRole("tab", { name: tab })).toHaveAttribute("aria-selected", "true");
      await expectNamed(page, `the ${tab} tab`);
    }
    await page.getByRole("button", { name: "Crop" }).click();
    await expect(page.getByRole("button", { name: "Done" })).toBeVisible();
    snapshot = await expectNamed(page, "Crop");
    expect(count(snapshot, "main")).toBe(1);
    await page.keyboard.press("Escape");
    // Pick white is a row of the Colour tab's White balance card: the toggle, its description, the pill and the layer all have names.
    await page.getByRole("tab", { name: "Colour" }).click();
    const white = page.getByRole("button", { name: "Pick white" });
    await expect(white).toHaveAccessibleDescription("Click something neutral on the photo (W)");
    await white.click();
    await expect(white).toHaveAttribute("aria-pressed", "true");
    snapshot = await expectNamed(page, "Pick white");
    // The pill is a polite status, the layer an application that says how to use it from the keyboard, and the key chip is hidden.
    const pill = page.getByRole("status").filter({ hasText: "Click something neutral: grey or white. Esc cancels." });
    await expect(pill).toHaveAttribute("aria-live", "polite");
    await expect(page.getByRole("application", { name: "Pick white. Click something neutral, or move the crosshair with the arrow keys and press Return." })).toBeVisible();
    await expect(page.locator(".ym-pick-row__keys")).toHaveAttribute("aria-hidden", "true");
    expect(snapshot).not.toContain("or press");
    await page.keyboard.press("Escape");
    await page.getByRole("button", { name: /^Fit/ }).click();
    await expect(page.getByRole("menu")).toBeVisible();
    await expectNamed(page, "the zoom menu");
  });

  test("Look tab sections: every header is a named disclosure button, folded or searching", async ({ page }) => {
    await openApp(page, { ...MOCKUP, mockLibraries: "all" });
    await openPhotos(page);
    let snapshot = await expectNamed(page, "the Look tab with every library");
    for (const name of [
      "Built in, 2 looks, collapse",
      "ARRI looks · LogC4, 7 looks, collapse",
      "ARRI looks · LogC3, 5 looks, collapse",
      "ARRI looks · LogC3 to Rec.709, 4 looks, collapse",
      "ARRI looks · LogC3 to Rec.709 · ALF2, 6 looks, collapse",
      "Your LUTs, 5 looks, collapse",
    ]) {
      expect(snapshot, name).toMatch(new RegExp(`- button "${name.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}" \\[expanded\\]`));
    }
    // Folded: no [expanded], the name says expand, and the section holding the current look says so.
    await page.getByRole("button", { name: "Built in, 2 looks, collapse" }).click();
    snapshot = await expectNamed(page, "a folded section");
    expect(snapshot).toContain('- button "Built in, 2 looks, holds the current look, expand"');
    expect(snapshot).not.toContain('"Built in, 2 looks, holds the current look, expand" [expanded]');
    // Searching: every header with matches is open and still named.
    await page.getByRole("searchbox", { name: "Search looks" }).fill("film");
    snapshot = await expectNamed(page, "a search over folded sections");
    expect(snapshot).toContain('- button "ARRI looks · LogC4, 2 of 7 looks" [disabled] [expanded]');
  });

  test("Export panel, in each format, and while it runs", async ({ page }) => {
    await openApp(page, { ...MOCKUP, mockExportHold: 1.45 });
    await openPhotos(page);
    await page.keyboard.press("Meta+e");
    await expect(page.getByRole("complementary", { name: "Export" })).toBeVisible();
    for (const format of ["JPEG", "TIFF", "PNG", "DNG"]) {
      await page.getByRole("radiogroup", { name: "Format" }).getByRole("radio", { name: format }).click();
      const snapshot = await expectNamed(page, `Export, ${format}`);
      expect(count(snapshot, "main")).toBe(1);
    }
    await page.getByRole("button", { name: /^Export 3 photos/ }).click();
    await expect(page.getByText("1 of 3 done.")).toBeVisible();
    await expectNamed(page, "Export, running");
  });

  test("the wizard and the dialogs", async ({ page }) => {
    await toWizard(page);
    let snapshot = await expectNamed(page, "wizard, step 1");
    expect(count(snapshot, "main")).toBe(1);
    await page.getByRole("button", { name: /^Next/ }).click();
    await expect(page.getByRole("heading", { level: 1, name: "Download from ARRI" })).toBeVisible();
    await expectNamed(page, "wizard, step 2");
    await page.getByRole("button", { name: "Cancel" }).click();
    await page.keyboard.press("Meta+,");
    await expect(page.getByRole("dialog", { name: "Preferences" })).toBeVisible();
    await expectNamed(page, "Preferences");
    await page.getByRole("button", { name: /About/ }).click();
    await expect(page.getByRole("dialog", { name: /About/ })).toBeVisible();
    await expectNamed(page, "About");
  });
});

test.describe("Apple Auto (docs/19)", () => {
  test("the Automatic card's rows and line and the Preferences select are named", async ({ page }) => {
    await openApp(page, { ...MOCKUP, mockAppleAuto: "answered" });
    await openPhotos(page);
    let snapshot = await expectNamed(page, "Photos with Apple Auto");
    expect(snapshot).toContain('button "Contrast +15, Auto. Open Tone"');
    expect(snapshot).toContain('button "Saturation −15, Auto. Open Tone"');
    expect(snapshot).toContain('button "Highlights −20, Auto. Open Light"');
    expect(snapshot).toContain('button "Shadows colour Teal · medium, Auto. Open Colour"');
    expect(snapshot).toContain("Set with Apple Intelligence.");
    expect(snapshot).toContain("Teal shadows and warm highlights for a cinematic late-sun street");
    await page.keyboard.press("Meta+,");
    await expect(page.getByRole("combobox", { name: "Auto grading" })).toBeVisible();
    snapshot = await expectNamed(page, "Preferences with Apple Auto");
    expect(snapshot).toMatch(/combobox "Auto grading"/);
    // The instructions for Apple's model: a named text box and its Reset to default, disabled while the text is the default.
    expect(snapshot).toMatch(/textbox "Instructions for Apple Intelligence"/);
    expect(snapshot).toMatch(/button "Reset to default" \[disabled\]/);
  });

  test("the select opens from the keyboard, and the rows are Tab stops with the ring", async ({ page }) => {
    await openApp(page, { ...MOCKUP, mockAppleAuto: "answered" });
    await openPhotos(page);
    await tabTo(page, (el) => el.getAttribute("aria-label") === "Contrast +15, Auto. Open Tone");
    await expect(page.locator(":focus")).toHaveCSS("outline-width", "3px");
    await page.keyboard.press("Meta+,");
    const select = page.getByRole("combobox", { name: "Auto grading" });
    await select.focus();
    await page.keyboard.press("Enter");
    await expect(page.getByRole("option", { name: "Default Auto" })).toBeVisible();
    await page.keyboard.press("ArrowUp");
    // Radix moves the focus a tick after the key.
    await expect(page.getByRole("option", { name: "Default Auto" })).toBeFocused();
    await page.keyboard.press("Enter");
    await expect(select).toHaveText("Default Auto");
  });
});

test.describe("Auto providers (docs/22 section 12)", () => {
  /** Preferences with every provider listed: Apple Intelligence and Claude Code available, Antigravity signed out. */
  async function preferences(page: Page, options: Record<string, string | number> = {}): Promise<void> {
    await openApp(page, { ...MOCKUP, mockProviders: "apple,claude,agy:signed_out", mockAppleAuto: "answered", ...options });
    await openPhotos(page);
    await page.keyboard.press("Meta+,");
    await expect(page.getByRole("combobox", { name: "Auto grading" })).toBeVisible();
  }

  test("the select, its entries (a program that cannot be used says why), the notes and the instructions are named", async ({ page }) => {
    await preferences(page);
    let snapshot = await expectNamed(page, "Preferences with every provider");
    expect(snapshot).toMatch(/combobox "Auto grading"/);
    expect(snapshot).toContain("Antigravity is installed but not signed in. Open Terminal, run agy and sign in.");
    expect(snapshot).toMatch(/textbox "Instructions for Apple Intelligence"/);
    await page.getByRole("combobox", { name: "Auto grading" }).click();
    await expect(page.getByRole("option", { name: "Claude Code" })).toBeVisible();
    snapshot = await expectNamed(page, "the open list");
    expect(snapshot).toMatch(/option "Default Auto"/);
    expect(snapshot).toMatch(/option "Apple Intelligence" \[selected\]/);
    expect(snapshot).toMatch(/option "Claude Code"/);
    // The signed-out program is listed, disabled, and its name says why.
    expect(snapshot).toMatch(/option "Antigravity \(agy\), experimental, not signed in" \[disabled\]/);
  });

  test("the consent dialog is named, its buttons too, focus starts on Cancel, Esc and Tab stay in it", async ({ page }) => {
    await preferences(page);
    await page.getByRole("combobox", { name: "Auto grading" }).click();
    await page.getByRole("option", { name: "Claude Code" }).click();
    const dialog = page.getByRole("dialog", { name: "Send photos to Claude Code?" });
    await expect(dialog).toBeVisible();
    const snapshot = await expectNamed(page, "the consent dialog");
    expect(snapshot).toMatch(/dialog "Send photos to Claude Code\?"/);
    expect(snapshot).toMatch(/button "Cancel"/);
    expect(snapshot).toMatch(/button "Use Claude Code"/);
    await expect(dialog.getByRole("button", { name: "Cancel" })).toBeFocused();
    // Tab goes to the other button and back, never behind the dialog.
    await page.keyboard.press("Tab");
    await expect(dialog.getByRole("button", { name: "Use Claude Code" })).toBeFocused();
    await expect(page.locator(":focus")).toHaveCSS("outline-width", "3px");
    await page.keyboard.press("Tab");
    await expect(dialog.getByRole("button", { name: "Cancel" })).toBeFocused();
    // Esc closes it, Preferences stays, and the focus goes back to the select.
    await page.keyboard.press("Escape");
    await expect(dialog).toHaveCount(0);
    await expect(page.getByRole("dialog", { name: "Preferences" })).toBeVisible();
    await expect(page.getByRole("combobox", { name: "Auto grading" })).toBeFocused();
  });

  test("the Automatic card's line names the program", async ({ page }) => {
    await openApp(page, { ...MOCKUP, mockProviders: "claude", mockAutoProvider: "claude", mockConsent: "claude", mockAppleAuto: "answered" });
    await openPhotos(page);
    const snapshot = await expectNamed(page, "Photos with Claude Code");
    expect(snapshot).toContain("Set with Claude Code.");
  });
});

test.describe("Auto crop (docs/20)", () => {
  test("the Crop panel's Auto is named Auto crop, its note is a status, and A, Space and Return work it, with the ring", async ({ page }) => {
    await openApp(page, MOCKUP);
    await openPhotos(page);
    await page.getByRole("button", { name: "Crop", exact: true }).click();
    const panel = page.getByRole("complementary", { name: "Crop" });
    const auto = panel.getByRole("button", { name: "Auto crop" });
    await expect(auto).toHaveAttribute("aria-pressed", "false");
    // A, while the panel is open.
    await page.keyboard.press("a");
    await expect(auto).toHaveAttribute("aria-pressed", "true");
    await expect(panel.getByRole("status")).toContainText("Levelled by 1.4°.");
    let snapshot = await expectNamed(page, "Crop with Auto on");
    expect(snapshot).toMatch(/button "Auto crop" \[pressed\]/);
    expect(snapshot).toContain("Cropped to frame the subject.");
    // The toggle is a Tab stop of the card's header, before Reset, with the ring; Space and Return press it.
    await tabTo(page, (el) => el.getAttribute("aria-label") === "Auto crop");
    await expect(page.locator(":focus")).toHaveCSS("outline-width", "3px");
    await page.keyboard.press("Tab");
    await expect(page.locator(":focus")).toHaveText("Reset");
    await auto.focus();
    await page.keyboard.press("Space");
    await expect(auto).toHaveAttribute("aria-pressed", "false");
    await page.keyboard.press("Enter");
    await expect(auto).toHaveAttribute("aria-pressed", "true");
    // Return pressed the toggle and did not finish the crop.
    await expect(panel).toBeVisible();
    snapshot = await expectNamed(page, "Crop with Auto on again");
    expect(count(snapshot, "main")).toBe(1);
  });
});

/** Presses Tab until the focused element satisfies `matches` (it runs in the page, so it can't use anything outside itself), at most `limit` times. */
async function tabTo(page: Page, matches: (el: Element) => boolean, limit = 80): Promise<void> {
  for (let i = 0; i < limit; i++) {
    await page.keyboard.press("Tab");
    if (await page.evaluate(`(${matches.toString()})(document.activeElement)`)) return;
  }
  throw new Error(`Tab did not reach the element within ${limit} presses`);
}

test.describe("keyboard only", () => {
  test("Start → open → Photos → Light tab → Exposure → Export, without the mouse", async ({ page }) => {
    await openApp(page, MOCKUP);
    await tabTo(page, (el) => (el.textContent ?? "").trim() === "Open files…");
    await page.keyboard.press("Enter");
    await expect(page.getByRole("heading", { level: 2, name: "12 photos ready" })).toBeVisible();
    await tabTo(page, (el) => (el.textContent ?? "").trim().startsWith("Open photos"));
    await page.keyboard.press("Enter");
    await expect(page.getByRole("region", { name: "Automatic" })).toBeVisible();

    // The tab bar is one stop; the arrows move between the tabs.
    await tabTo(page, (el) => el.getAttribute("role") === "tab");
    await expect(page.getByRole("tab", { name: "Look" })).toBeFocused();
    await page.keyboard.press("ArrowRight");
    await expect(page.getByRole("tab", { name: "Lens" })).toHaveAttribute("aria-selected", "true");
    await page.keyboard.press("ArrowRight");
    await expect(page.getByRole("tab", { name: "Light" })).toHaveAttribute("aria-selected", "true");
    await expect(page.getByRole("complementary", { name: "Tools" }).getByRole("heading", { level: 2, name: "Light" })).toBeVisible();

    // A slider takes the arrows; Shift takes ten steps; Home goes to the end of the range.
    await tabTo(page, (el) => el.getAttribute("role") === "slider" && document.getElementById(el.getAttribute("aria-labelledby") ?? "")?.textContent === "Exposure");
    const exposure = page.getByRole("slider", { name: "Exposure" });
    await expect(exposure).toBeFocused();
    const value = async (): Promise<number> => Number(await exposure.getAttribute("aria-valuenow"));
    const before = await value();
    await page.keyboard.press("ArrowRight");
    expect(await value()).toBeGreaterThan(before);
    await page.keyboard.press("Shift+ArrowRight");
    expect(await value()).toBeGreaterThan(before + 0.05);
    await page.keyboard.press("Home");
    expect(await value()).toBe(-4);
    await expect(page.getByRole("textbox", { name: "Exposure value" })).toHaveValue("−4.00");
    // ⌘Z takes the keys back.
    await page.keyboard.press("Meta+z");

    // Export: ⌘E opens the panel, Tab reaches the button, Return starts it, Tab reaches Done.
    await page.keyboard.press("Meta+e");
    await expect(page.getByRole("complementary", { name: "Export" })).toBeVisible();
    await expect(page.getByRole("button", { name: /^Export 3 photos/ })).toContainText("≈");
    await tabTo(page, (el) => (el.textContent ?? "").trim().startsWith("Export 3 photos"));
    await page.keyboard.press("Enter");
    await expect(page.getByRole("heading", { level: 2, name: "Exported" })).toBeVisible();
    await tabTo(page, (el) => (el.textContent ?? "").trim() === "Done");
    await page.keyboard.press("Enter");
    await expect(page.getByRole("complementary", { name: "Tools" })).toBeVisible();
  });

  test("a look tile, a segment, a switch and the search box work from the keyboard", async ({ page }) => {
    await openApp(page, MOCKUP);
    await openPhotos(page);
    await tabTo(page, (el) => el.getAttribute("aria-label") === "Search looks");
    await page.keyboard.type("film");
    await expect(page.getByText("2 of 7")).toBeVisible();
    // The field's Clear button (there while the field holds text) is not a Tab stop: the first tile is the next one.
    await page.keyboard.press("Tab");
    await expect(page.getByRole("button", { name: "Film warm look, key 3" })).toBeFocused();
    await page.keyboard.press("Enter");
    await expect(page.getByRole("button", { name: "Film warm look, key 3" })).toHaveAttribute("aria-pressed", "true");
    await expect(page.locator(".ym-photo-cell__chip")).toHaveText("Film warm · 100%");

    // Apply to is a radio group: the arrows move and choose.
    await page.getByRole("radiogroup", { name: "Apply look to" }).getByRole("radio", { name: /^This photo/ }).focus();
    await page.keyboard.press("ArrowRight");
    await expect(page.getByRole("radiogroup", { name: "Apply look to" }).getByRole("radio", { name: /^Picked/ })).toHaveAttribute("aria-checked", "true");

    // A switch toggles with Space.
    await page.keyboard.press("Meta+7");
    const lens = page.getByRole("switch", { name: /Lens corrections/ });
    await lens.focus();
    const was = await lens.getAttribute("aria-checked");
    await page.keyboard.press("Space");
    await expect(lens).not.toHaveAttribute("aria-checked", was ?? "");
  });

  test("a typed value takes Return, and Esc gives it up", async ({ page }) => {
    await openApp(page, MOCKUP);
    await openPhotos(page);
    await page.keyboard.press("Meta+3");
    await setValue(page, "Contrast", "25");
    await expect(page.getByRole("slider", { name: "Contrast" })).toHaveAttribute("aria-valuenow", "25");
    const box = page.getByRole("textbox", { name: "Contrast value" });
    await box.click();
    await box.fill("80");
    await box.press("Escape");
    await expect(box).toHaveValue("+25");
  });

  test("the wizard: Tab reaches Next, Return moves on, Back returns", async ({ page }) => {
    await toWizard(page);
    await tabTo(page, (el) => (el.textContent ?? "").trim().startsWith("Next"));
    await page.keyboard.press("Enter");
    await expect(page.getByRole("heading", { level: 1, name: "Download from ARRI" })).toBeVisible();
    await tabTo(page, (el) => el.tagName === "BUTTON" && (el.textContent ?? "").includes("Back"));
    await page.keyboard.press("Enter");
    await expect(page.getByRole("heading", { level: 1, name: "Choose what to import" })).toBeVisible();
  });

  test("a dialog traps focus, Esc closes it, and focus goes back", async ({ page }) => {
    await openApp(page, MOCKUP);
    await openPhotos(page);
    const fit = page.getByRole("button", { name: /^Fit/ });
    await fit.focus();
    await page.keyboard.press("Meta+,");
    const dialog = page.getByRole("dialog", { name: "Preferences" });
    await expect(dialog).toBeVisible();
    for (let i = 0; i < 25; i++) {
      await page.keyboard.press("Tab");
      expect(await page.evaluate(() => !!document.activeElement?.closest("[role=dialog]"))).toBe(true);
    }
    await page.keyboard.press("Escape");
    await expect(dialog).toHaveCount(0);
    await expect(page.locator(":focus")).toHaveCount(1);
  });
});

test.describe("focus ring", () => {
  /** Walks the Tab order and returns each stop's label and whether its focus ring is the 3 px outline with a 2 px offset. */
  async function walk(page: Page, stops: number): Promise<Array<{ label: string; ring: boolean }>> {
    const seen: Array<{ label: string; ring: boolean }> = [];
    for (let i = 0; i < stops; i++) {
      await page.keyboard.press("Tab");
      seen.push(
        await page.evaluate(async () => {
          // The slider sets its ring in a state update; let it draw first.
          await new Promise<void>((resolve) => requestAnimationFrame(() => requestAnimationFrame(() => resolve())));
          const el = document.activeElement as HTMLElement | null;
          if (!el || el === document.body) return { label: "BODY", ring: true };
          // A text input inside a capsule (the search field) draws its ring on the capsule around it (focus-within); every other stop
          // draws its own.
          const rings = (node: Element | null): boolean => {
            if (!node) return false;
            const style = getComputedStyle(node);
            return style.outlineStyle !== "none" && parseFloat(style.outlineWidth) >= 3;
          };
          const own = rings(el) || (el instanceof HTMLInputElement && el.type !== "checkbox" && rings(el.parentElement));
          const label = (el.getAttribute("aria-label") ?? el.textContent ?? el.tagName).trim().slice(0, 40);
          return { label, ring: own };
        }),
      );
    }
    return seen;
  }

  test("every stop of Start's Tab order has the ring", async ({ page }) => {
    await openApp(page, { mockLooks: 0 });
    const seen = await walk(page, 7);
    expect(seen.filter((s) => !s.ring)).toEqual([]);
    expect(seen.map((s) => s.label)).toContain("Open files…");
  });

  test("every stop of the Photos Look tab has the ring", async ({ page }) => {
    await openApp(page, MOCKUP);
    await openPhotos(page);
    const seen = await walk(page, 30);
    expect(seen.filter((s) => !s.ring)).toEqual([]);
    expect(seen.length).toBe(30);
  });

  test("every stop of the node view's Tab order has the ring: the header, the divider, the toolbar's buttons, the graph, then the panel's head", async ({ page }) => {
    await openApp(page, MOCKUP);
    await openPhotos(page);
    await page.getByRole("tab", { name: "Light", exact: true }).click();
    await page.getByRole("button", { name: "Nodes", exact: true }).click();
    await expect(page.getByRole("separator", { name: "Resize the graph" })).toBeVisible();
    await page.getByRole("button", { name: "Nodes", exact: true }).focus();
    const seen = await walk(page, 26);
    expect(seen.filter((s) => !s.ring)).toEqual([]);
    const labels = seen.map((s) => s.label);
    // The divider comes after the header and before the toolbar; the arrows of the switcher are stops with names.
    expect(labels.indexOf("Resize the graph")).toBeGreaterThan(-1);
    expect(labels.indexOf("Resize the graph")).toBeLessThan(labels.indexOf("Add node"));
    for (const name of ["Add node", "Previous photo", "Next photo", "Expand"]) expect(labels, name).toContain(name);
    // The disabled ones stay in the order (aria-disabled, not disabled) and say so.
    for (const name of ["Bypass", "Delete", "Show mask"]) expect(labels, name).toContain(name);
    // The graph is one stop (the selected card, Main) and the zoom control three, between the toolbar and the panel, each with the ring.
    expect(labels.slice(labels.indexOf("Expand") + 1, labels.indexOf("Expand") + 5)).toEqual(["Node 1, Main", "Zoom out", "Zoom in", "Fit the graph"]);
    // The panel follows with the node's head: Bypass, Reset and More, then the tab row (one stop, on the selected tab).
    expect(labels.slice(labels.indexOf("Fit the graph") + 1, labels.indexOf("Fit the graph") + 5)).toEqual(["Bypass this node", "Reset this node", "More", "Light"]);
  });

  test("every stop of the Lens tab has the ring: the list is one stop, then Reset and the three sliders with their boxes", async ({ page }) => {
    await openApp(page, MOCKUP);
    await openPhotos(page);
    await page.keyboard.press("Meta+2");
    await expect(page.getByRole("tab", { name: "Lens" })).toHaveAttribute("aria-selected", "true");
    // From the tab bar, the list takes one Tab (on the chosen row: a radio), then Reset and the sliders of the Amounts card.
    await page.getByRole("tab", { name: "Lens" }).focus();
    const first = await walk(page, 1);
    expect(first).toEqual([{ label: "", ring: true }]);
    await expect(page.getByRole("radio", { name: /^Default / })).toBeFocused();
    const seen = await walk(page, 7);
    expect(seen.filter((s) => !s.ring)).toEqual([]);
    expect(seen.map((s) => s.label)).toEqual(["Reset", "", "Flares value", "", "Edges value", "", "Anamorphic value"]);
    await expect(page.getByRole("region", { name: "Amounts" }).getByRole("textbox", { name: "Anamorphic value" })).toBeFocused();
  });

  test("a focused tab shows the 3 px ring, drawn inside its cell", async ({ page }) => {
    await openApp(page, MOCKUP);
    await openPhotos(page);
    await page.keyboard.press("Meta+7");
    await tabTo(page, (el) => el.getAttribute("role") === "tab");
    const raw = page.getByRole("tab", { name: "RAW" });
    await expect(raw).toBeFocused();
    await expect(raw).toHaveCSS("outline-width", "3px");
    // glasscn's tabs draw the ring inset (the tab row is a pane of its own, there is no room outside it).
    await expect(raw).toHaveCSS("outline-offset", "-3px");
  });
});

test.describe("the node panel", () => {
  // docs/23 §7.8 and N-09: the panel of the selected node has a head with named buttons, the tabs of its kind and cards whose controls are named,
  // on every tab of every kind of node. The Tab order of each state of the panel is counted: each stop has the ring.
  const graded = async (page: Page, simple: string): Promise<void> => {
    await openApp(page, MOCKUP);
    await configureMock(page, { initialSettings: { DSC04204: { nodes: GRADED_NODES.nodes, links: GRADED_NODES.links } } as never });
    await openPhotos(page);
    await page.getByRole("tab", { name: simple, exact: true }).click();
    await page.getByRole("button", { name: "Nodes", exact: true }).click();
    await expect(page.locator('[data-node-card="4"]')).toBeVisible();
  };
  test("every control on every tab of every kind of node has a name, and the head says which node it is", async ({ page }) => {
    await graded(page, "Light");
    const tabs: Array<[string | number, string[]]> = [
      ["input", ["RAW", "Lens", "Light", "Colour", "Detail"]],
      [2, ["Light", "Colour", "Curve", "Mask"]],
      ["output", ["Look", "Detail"]],
    ];
    for (const [node, names] of tabs) {
      await page.locator(`[data-node-card="${node}"]`).click();
      await expect(page.locator(`[data-node-card="${node}"]`)).toHaveAttribute("aria-pressed", "true");
      for (const name of names) {
        await page.getByRole("tab", { name, exact: true }).click();
        await expect(page.getByRole("tab", { name, exact: true })).toHaveAttribute("aria-selected", "true");
        await page.waitForTimeout(150);
        const snapshot = await expectNamed(page, `${node} › ${name}`);
        expect(snapshot, `${node} › ${name}`).toMatch(/- group "Node":/);
        expect(snapshot, `${node} › ${name}`).toMatch(/- tabpanel/);
      }
    }
    // The head names the node: a heading, and the buttons that act on it.
    await page.locator('[data-node-card="3"]').click();
    const snapshot = await expectNamed(page, "Sky");
    expect(snapshot).toMatch(/- heading "Sky" \[level=2\]/);
    expect(snapshot).toMatch(/- group "Node actions":/);
  });

  test("the Mask tab's rows are named like the cards: kind, 'mask' and what they say, and each row's controls are named", async ({ page }) => {
    await graded(page, "Light");
    await page.locator('[data-node-card="2"]').click();
    await page.getByRole("tab", { name: "Mask", exact: true }).click();
    const snapshot = await expectNamed(page, "the Mask tab");
    expect(snapshot).toContain('- button "Object mask, 2 points"');
    expect(snapshot).toContain('- button "Radial mask, Feather 45"');
    expect(snapshot).toContain('- radiogroup "Object mask, add or subtract"');
    expect(snapshot).toContain('- radiogroup "Radial mask, add or subtract"');
    expect(snapshot.match(/- button "Invert this mask"/g)).toHaveLength(2);
    expect(snapshot.match(/- button "Show this mask"/g)).toHaveLength(2);
    expect(snapshot.match(/- switch "Turn this mask off"/g)).toHaveLength(2);
    expect(snapshot).toMatch(/- button "Add mask"/);
    expect(snapshot).toMatch(/- button "Invert"/);
    await expect(page.getByRole("button", { name: "Add mask" })).not.toHaveAttribute("aria-disabled", "true");
    // The selected row is the pressed one.
    await page.getByRole("button", { name: "Radial mask, Feather 45" }).click();
    await expect(page.getByRole("button", { name: "Radial mask, Feather 45" })).toHaveAttribute("aria-pressed", "true");
    await expect(page.getByRole("button", { name: "Object mask, 2 points" })).toHaveAttribute("aria-pressed", "false");
  });

  test("the Add mask menu, the selected mask's card, the handles and the pick are named", async ({ page }) => {
    await openApp(page, { ...MOCKUP, mockMasks: "all" });
    await configureMock(page, { initialSettings: { DSC04204: { nodes: GRADED_NODES.nodes, links: GRADED_NODES.links } } as never });
    await openPhotos(page);
    await page.getByRole("tab", { name: "Light", exact: true }).click();
    await page.getByRole("button", { name: "Nodes", exact: true }).click();
    await expect(page.locator('[data-node-card="4"]')).toBeVisible();
    await page.locator('[data-node-card="3"]').click();
    await page.getByRole("tab", { name: "Mask", exact: true }).click();
    // The menu: its groups and every item have names.
    await page.getByRole("button", { name: "Add mask" }).click();
    let snapshot = await expectNamed(page, "the Add mask menu");
    expect(snapshot).toMatch(/- menu "Add mask":/);
    expect(count(snapshot, "menuitem")).toBe(9);
    await page.keyboard.press("Escape");
    // A Linear's card and its handles.
    await page.getByRole("button", { name: /^Linear mask/ }).click();
    snapshot = await expectNamed(page, "a Linear's card with its handles");
    expect(snapshot).toMatch(/- region "Linear"/);
    expect(snapshot).toMatch(/- slider "Angle"/);
    for (const name of ["Move the Linear mask", "Linear mask feather, start line", "Linear mask feather, end line", "Rotate the Linear mask"]) {
      expect(snapshot).toContain(`- button "${name}. The arrow keys move it."`);
    }
    // A Colour range being picked.
    await page.locator('[data-node-card="4"]').click();
    await page.getByRole("button", { name: "Add mask" }).click();
    await page.getByRole("menuitem", { name: /^Colour range/ }).click();
    snapshot = await expectNamed(page, "a Colour range being picked");
    expect(snapshot).toMatch(/- application "Pick colour for the Colour range mask\./);
    expect(snapshot).toMatch(/- button "Pick colour" \[pressed\]/);
  });

  test("the handles of a Linear are Tab stops, one each, with the ring", async ({ page }) => {
    await graded(page, "Light");
    await page.locator('[data-node-card="3"]').click();
    await page.getByRole("tab", { name: "Mask", exact: true }).click();
    await page.getByRole("button", { name: /^Linear mask/ }).click();
    await expect(page.locator("[data-mask-handles] button")).toHaveCount(4);
    // From the zoom button (the header's last stop) the next Tab is the first handle: the keyboard's focus has the ring.
    await page.getByRole("button", { name: /^Fit \d+%/ }).focus();
    for (let i = 0; i < 6; i++) {
      await page.keyboard.press("Tab");
      if (await page.evaluate(() => document.activeElement?.closest("[data-mask-handles]") !== null)) break;
    }
    const ring = await page.evaluate(() => {
      const st = getComputedStyle(document.activeElement as HTMLElement);
      return { style: st.outlineStyle, width: parseFloat(st.outlineWidth) };
    });
    expect(ring.style).not.toBe("none");
    expect(ring.width).toBeGreaterThanOrEqual(3);
    // Four stops inside the layer, then the focus leaves it.
    const seen: string[] = [];
    for (let i = 0; i < 4; i++) {
      seen.push((await page.evaluate(() => document.activeElement?.getAttribute("aria-label") ?? "")).replace(/\. The arrow keys move it\.$/, ""));
      await page.keyboard.press("Tab");
    }
    expect(seen).toEqual(["Move the Linear mask", "Linear mask feather, start line", "Linear mask feather, end line", "Rotate the Linear mask"]);
    expect(await page.evaluate(() => document.activeElement?.closest("[data-mask-handles]") !== null)).toBe(false);
  });

  test("the Brush card is named, and the drawing surface is a surface, not a control: no name, no role, no Tab stop", async ({ page }) => {
    await graded(page, "Light");
    await page.locator('[data-node-card="4"]').click();
    await page.getByRole("tab", { name: "Mask", exact: true }).click();
    await page.getByRole("button", { name: "Add mask" }).click();
    await page.getByRole("menuitem", { name: /^Brush/ }).click();
    const snapshot = await expectNamed(page, "a Brush's card");
    expect(snapshot).toMatch(/- region "Brush"/);
    expect(snapshot).toContain('- radiogroup "Paint or erase"');
    expect(snapshot).toMatch(/- radio "Paint" \[checked\]/);
    expect(snapshot).toMatch(/- radio "Erase"/);
    for (const name of ["Size", "Hardness", "Refine edges", "Feather", "Grow", "Opacity"]) expect(snapshot, name).toMatch(new RegExp(`- slider "${name}"`));
    expect(snapshot).toContain('- button "Brush mask, Paint on the photo" [pressed]');
    // The layer over the photo is not in the accessibility tree at all.
    await expect(page.locator("[data-brush-layer]")).toHaveAttribute("aria-hidden", "true");
    expect(snapshot).not.toMatch(/brush-layer|Brush layer/i);
  });

  test("the Object card is named, its row says 'Getting ready…' while the encoder runs, and the points on the photo are a surface, not controls", async ({ page }) => {
    await graded(page, "Light");
    await page.locator('[data-node-card="4"]').click();
    await page.getByRole("tab", { name: "Mask", exact: true }).click();
    await page.getByRole("button", { name: "Add mask" }).click();
    await page.getByRole("menuitem", { name: /^Object/ }).click();
    await expect(page.getByRole("region", { name: "Object" })).toBeVisible();
    const snapshot = await expectNamed(page, "an Object's card");
    expect(snapshot).toMatch(/- region "Object"/);
    expect(snapshot).toContain('- radiogroup "Object tool"');
    expect(snapshot).toMatch(/- radio "Add point" \[checked\]/);
    expect(snapshot).toMatch(/- radio "Remove point"/);
    expect(snapshot).toMatch(/- radio "Box"/);
    for (const name of ["Refine edges", "Feather", "Grow", "Opacity"]) expect(snapshot, name).toMatch(new RegExp(`- slider "${name}"`));
    // The encoder is getting the photo ready (the mock: 900 ms), and the row's name says so; then its own words.
    expect(snapshot).toMatch(/- button "Object mask, (Getting ready…|Click the object on this photo)" \[pressed\]/);
    await expect(page.getByRole("button", { name: "Object mask, Click the object on this photo" })).toBeVisible();
    // The layer over the photo is not in the accessibility tree, and neither are the points drawn in it.
    await expect(page.locator("[data-object-layer]")).toHaveAttribute("aria-hidden", "true");
    expect(snapshot).not.toMatch(/object-layer|Object layer|object-point/i);
  });

  test("the Background card is named, its subtitle reads 'Finding…' while the map is made, and nothing on the photo is a control", async ({ page }) => {
    await graded(page, "Light");
    await page.locator('[data-node-card="4"]').click();
    await page.getByRole("tab", { name: "Mask", exact: true }).click();
    await page.getByRole("button", { name: "Add mask" }).click();
    await page.getByRole("menuitem", { name: /^Background/ }).click();
    await expect(page.getByRole("region", { name: "Background" })).toBeVisible();
    const finding = await expectNamed(page, "a Background row finding");
    // The first frame finds the photo's map (the mock: 400 ms): the row's name says so, without the spinner (it is decoration).
    expect(finding).toMatch(/- button "Background mask, (Finding…|Level 50)" \[pressed\]/);
    await expect(page.getByRole("button", { name: "Background mask, Level 50" })).toBeVisible();
    const snapshot = await expectNamed(page, "a Background's card");
    expect(snapshot).toMatch(/- region "Background"/);
    for (const name of ["Level", "Softness", "Refine edges", "Feather", "Grow", "Opacity"]) expect(snapshot, name).toMatch(new RegExp(`- slider "${name}"`));
    // No handles, no layer: the map is the photo's own.
    await expect(page.locator("[data-mask-handles]")).toHaveCount(0);
    await expect(page.locator("[data-brush-layer]")).toHaveCount(0);
  });

  // The stops of each state, counted: the head's buttons, the tab row (two: glasscn's tab list, a container that draws no ring and hands the
  // arrows to the tabs, then the selected tab), the cards' controls (a slider is two: its thumb and its value box), and the foot's two buttons.
  type Stops = { state: string; simple: string; node: number | "input" | "output"; tab: string; stops: number; before?: (page: Page) => Promise<void> };
  const STOPS: Stops[] = [
    { state: "the Input's Colour tab", simple: "Colour", node: "input", tab: "Colour", stops: 14 },
    { state: "a corrector's Light tab", simple: "Light", node: 2, tab: "Light", stops: 25 },
    { state: "a corrector's Colour tab", simple: "Colour", node: 2, tab: "Colour", stops: 21 },
    { state: "the Output's Look tab", simple: "Look", node: "output", tab: "Look", stops: 23 },
    // The head's buttons (3), the tab row (2), the Masks card's Add mask, Invert and Reset (3), the two rows' four stops each (the row's button, the chosen
    // segment of Add | Subtract, Invert and the eye: 8), the View card's chosen segment, and the foot's two buttons.
    { state: "a corrector's Mask tab with two rows and the View card", simple: "Light", node: 2, tab: "Mask", stops: 19 },
    // Sky's one row, selected: its card has Reset and three sliders (Angle, Feather, Opacity; a slider is two stops, and Reset in a row's end is a stop only
    // while the slider differs from its default, which Sky's Linear does at an angle of 4 and a feather of 25). Counted below, with the ring on every stop.
    {
      state: "a Linear's Mask tab with its card",
      simple: "Light",
      node: 3,
      tab: "Mask",
      stops: 24,
      before: (page) => page.getByRole("button", { name: /^Linear mask/ }).click(),
    },
    // A Brush row just added, on Warmth (node 4): the head's 3, the tab row's 2, the Masks card's 3, the row's 4, then the BRUSH card: its Reset, the chosen segment of
    // Paint · Erase, six sliders of two stops each (none of them differs from its default, so none has a Reset stop), the View card's chosen segment and the foot's two
    // buttons. The drawing surface adds none.
    {
      state: "a Brush's Mask tab with its card",
      simple: "Light",
      node: 4,
      tab: "Mask",
      stops: 29,
      before: async (page) => {
        await page.getByRole("button", { name: "Add mask" }).click();
        await page.getByRole("menuitem", { name: /^Brush/ }).click();
        await expect(page.getByRole("region", { name: "Brush" })).toBeVisible();
        await expect(page.locator("[data-brush-layer]")).toBeVisible();
      },
    },
    // An Object row just added, on Warmth (node 4): the head's 3, the tab row's 2, the Masks card's 3, the row's 4, then the OBJECT card: its Reset, the chosen segment of
    // Add point · Remove point · Box, four sliders of two stops each (Refine edges, Feather, Grow, Opacity; none differs from its default), the View card's chosen segment
    // and the foot's two buttons: 25. The surface over the photo adds none.
    {
      state: "an Object's Mask tab with its card",
      simple: "Light",
      node: 4,
      tab: "Mask",
      stops: 25,
      before: async (page) => {
        await page.getByRole("button", { name: "Add mask" }).click();
        await page.getByRole("menuitem", { name: /^Object/ }).click();
        await expect(page.getByRole("region", { name: "Object" })).toBeVisible();
        await expect(page.locator("[data-object-layer]")).toBeVisible();
        await expect(page.getByRole("button", { name: "Object mask, Click the object on this photo" })).toBeVisible();
      },
    },
    // A Background row just added, on Warmth (node 4): the head's 3, the tab row's 2, the Masks card's 3, the row's 4, then the BACKGROUND card: its Reset, six sliders of two
    // stops each (Level, Softness, Refine edges, Feather, Grow, Opacity; none differs from its default, so none has a Reset stop of its own), the View card's chosen segment
    // and the foot's two buttons: 28. Nothing is drawn on the photo.
    {
      state: "a Background's Mask tab with its card",
      simple: "Light",
      node: 4,
      tab: "Mask",
      stops: 28,
      before: async (page) => {
        await page.getByRole("button", { name: "Add mask" }).click();
        await page.getByRole("menuitem", { name: /^Background/ }).click();
        await expect(page.getByRole("region", { name: "Background" })).toBeVisible();
        await expect(page.getByRole("button", { name: "Background mask, Level 50" })).toBeVisible();
      },
    },
    // A Subject row just added, on Warmth (node 4): the head's 3, the tab row's 2, the Masks card's 3, the row's 4, then the SUBJECT card: its Reset, four sliders of two stops
    // each (Refine edges, Feather, Grow, Opacity; none differs from its default, so none has a Reset stop of its own), the View card's chosen segment and the foot's two
    // buttons: 3 + 2 + 3 + 4 + 9 + 3 = 24. Nothing is drawn on the photo, and no tool: Vision's mask has nothing of its own to set.
    {
      state: "a Subject's Mask tab with its card",
      simple: "Light",
      node: 4,
      tab: "Mask",
      stops: 24,
      before: async (page) => {
        await page.getByRole("button", { name: "Add mask" }).click();
        await page.getByRole("menuitem", { name: /^Subject/ }).click();
        await expect(page.getByRole("region", { name: "Subject", exact: true })).toBeVisible();
        await expect(page.getByRole("button", { name: /^Subject mask, (Finding…|Foreground)/ })).toBeVisible();
      },
    },
  ];
  for (const { state, simple, node, tab, stops, before } of STOPS) {
    test(`${state}: every stop of the panel's Tab order has the ring, and the stops are counted`, async ({ page }) => {
      await graded(page, simple);
      await page.locator(`[data-node-card="${node}"]`).click();
      await page.getByRole("tab", { name: tab, exact: true }).click();
      await expect(page.getByRole("tab", { name: tab, exact: true })).toHaveAttribute("aria-selected", "true");
      await before?.(page);
      await page.waitForTimeout(300);
      // Walk the panel from the graph's last stop until the focus leaves it, and count what it stopped on.
      await page.getByRole("button", { name: "Fit the graph" }).focus();
      const seen: Array<{ label: string; ring: boolean }> = [];
      for (let i = 0; i < 60; i++) {
        await page.keyboard.press("Tab");
        const here = await page.evaluate(async () => {
          await new Promise<void>((resolve) => requestAnimationFrame(() => requestAnimationFrame(() => resolve())));
          const el = document.activeElement as HTMLElement | null;
          if (!el || !el.closest('aside[aria-label="Tools"]')) return null;
          const style = getComputedStyle(el);
          const parent = el.parentElement ? getComputedStyle(el.parentElement) : style;
          const ring = (st: CSSStyleDeclaration): boolean => st.outlineStyle !== "none" && parseFloat(st.outlineWidth) >= 3;
          return { label: (el.getAttribute("aria-label") ?? el.textContent ?? el.tagName).trim().slice(0, 40), ring: ring(style) || (el instanceof HTMLInputElement && ring(parent)) };
        });
        if (here === null) break;
        seen.push(here);
      }
      // Every stop has the ring but the tab list, which is the container of the tabs and draws none (as in the simple view).
      expect(seen.filter((s) => !s.ring).map((s) => s.label)).toEqual(["Tool tabs"]);
      expect(seen.map((s) => s.label)[0]).toMatch(/^(Bypass this node|Reset (input|output))$/);
      expect(seen.map((s) => s.label).slice(-2)).toEqual(["Copy settings", "Paste"]);
      expect(seen, `the Tab stops of ${state}: ${seen.map((s) => s.label).join(" | ")}`).toHaveLength(stops);
    });
  }
});
