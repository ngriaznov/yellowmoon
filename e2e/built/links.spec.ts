import { expect, test } from "@playwright/test";
import type { Page } from "@playwright/test";
import { configureMock, openApp, openPhotos } from "../helpers/app";
import { FOUR_NODES } from "../helpers/states";

// The links of the node view in the BUILT app (`npm run e2e:built`: `vite build`, then `vite preview`, config `playwright.built.config.ts`).
// The dev server hid two bugs of the built app: xyflow's stylesheet winning over ours (grey selected links), and the links vanishing after
// some dragging (a rebuilt node object lost the position of its ports, and the second measurement did not always come). Both depend on how
// the bundle is split and ordered and on the timing of a fast build, so the check is made on the files that ship.

const LINKS = FOUR_NODES.links.length;

/** The links as the page draws them: each one's path must have a real `d`, a box, and a stroke that shows. */
async function drawn(page: Page): Promise<{ count: number; bad: string[] }> {
  return page.evaluate(() => {
    const bad: string[] = [];
    const edges = [...document.querySelectorAll<SVGGElement>(".react-flow__edge")];
    for (const edge of edges) {
      const path = edge.querySelector<SVGPathElement>(".react-flow__edge-path");
      const d = path?.getAttribute("d") ?? "";
      const box = path?.getBoundingClientRect();
      const style = path ? getComputedStyle(path) : null;
      if (!/^M-?[\d.]+,-?[\d.]+ C/.test(d) || /NaN/.test(d)) bad.push(`d=${d}`);
      else if (!box || box.width <= 0) bad.push("no box");
      else if (!style || style.stroke === "none" || style.opacity === "0" || style.visibility === "hidden" || style.display === "none") bad.push(`hidden ${style?.stroke}`);
    }
    return { count: edges.length, bad };
  });
}

async function expectLinks(page: Page, when: string): Promise<void> {
  await expect.poll(() => drawn(page), { message: when, timeout: 2_000 }).toEqual({ count: LINKS, bad: [] });
}

async function drag(page: Page, id: number, dx: number, dy: number, steps = 2): Promise<void> {
  const box = await page.locator(`[data-node-card="${id}"]`).boundingBox();
  if (!box) throw new Error(`card ${id} is not on screen`);
  const x = box.x + box.width / 2;
  const y = box.y + box.height / 2;
  await page.mouse.move(x, y);
  await page.mouse.down();
  await page.mouse.move(x + dx, y + dy, { steps });
  await page.mouse.up();
}

test("the links stay drawn through many drags, pans and selections", async ({ page }) => {
  test.setTimeout(240_000);
  await openApp(page, { mockCurrent: 3, mockPicked: "2,3,4" });
  await configureMock(page, { initialSettings: { DSC04204: { nodes: FOUR_NODES.nodes, links: FOUR_NODES.links } } as never });
  await openPhotos(page);
  await page.getByRole("button", { name: "Nodes", exact: true }).click();
  await expectLinks(page, "when the graph opens");

  // Quick drops, one after the other with no pause: the case that lost the links (it showed in 1 run of 3 within 40 drops).
  for (let i = 0; i < 60; i++) {
    await drag(page, 1 + (i % 4), i < 30 ? 10 : -10, i % 2 ? 30 : -30);
    await expectLinks(page, `after drop ${i + 1}`);
  }

  // A drop outside the canvas, a pan, a zoom, a selection and a deselection.
  const canvas = (await page.locator(".ym-flow").boundingBox())!;
  await drag(page, 2, 0, -(canvas.y + 40), 4);
  await expectLinks(page, "after a drop above the canvas");
  await page.mouse.move(canvas.x + canvas.width / 2, canvas.y + canvas.height - 10);
  await page.mouse.wheel(60, 40);
  await expectLinks(page, "after a pan");
  await page.keyboard.down("Meta");
  await page.mouse.wheel(0, -200);
  await page.keyboard.up("Meta");
  await expectLinks(page, "after a zoom");
  await page.locator('[data-node-card="3"]').click();
  await expectLinks(page, "after a selection");
  await page.mouse.click(canvas.x + canvas.width / 2, canvas.y + canvas.height - 6);
  await expectLinks(page, "after a click on the canvas");
});

test("xyflow's stylesheet does not win over ours in the built app", async ({ page }) => {
  await openApp(page, { mockCurrent: 3, mockPicked: "2,3,4" });
  await configureMock(page, { initialSettings: { DSC04204: { nodes: FOUR_NODES.nodes, links: FOUR_NODES.links } } as never });
  await openPhotos(page);
  await page.getByRole("button", { name: "Nodes", exact: true }).click();
  await expectLinks(page, "when the graph opens");
  // A card is an arrow, not xyflow's grab hand (`.react-flow__node.draggable`).
  await expect(page.locator(".react-flow__node").first()).toHaveCSS("cursor", "default");
  // A picked link is thick and in the accent, not xyflow's grey #555.
  const line = (await page.locator(".react-flow__edge-path").nth(1).boundingBox())!;
  await page.mouse.click(line.x + line.width / 2, line.y + line.height / 2);
  await page.mouse.move(5, 5);
  const path = page.locator(".react-flow__edge:has(.ym-link[data-selected]) .react-flow__edge-path");
  await expect(path).toHaveCSS("stroke-width", "3px");
  const stroke = await path.evaluate((el) => getComputedStyle(el).stroke);
  expect(stroke).not.toMatch(/85, 85, 85|#555/);
});
