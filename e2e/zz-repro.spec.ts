import { test } from "@playwright/test";
import type { Page } from "@playwright/test";
import { configureMock, openApp, openPhotos } from "./helpers/app";
import { FOUR_NODES } from "./helpers/states";

async function probe(page: Page, label: string) {
  const r = await page.evaluate(() => {
    const edges = [...document.querySelectorAll(".react-flow__edge")].map((e) => {
      const p = e.querySelector<SVGPathElement>(".react-flow__edge-path");
      const bb = p?.getBoundingClientRect();
      const cs = p ? getComputedStyle(p) : null;
      return { d: (p?.getAttribute("d") ?? "").slice(0, 30), w: bb?.width, stroke: cs?.stroke, op: cs?.opacity, vis: cs?.visibility, disp: cs?.display };
    });
    return { n: edges.length, edges };
  });
  console.log(label, JSON.stringify(r));
}

test("repro", async ({ page, baseURL }) => {
  await openApp(page, { mockCurrent: 3, mockPicked: "2,3,4" });
  await configureMock(page, { initialSettings: { DSC04204: { nodes: FOUR_NODES.nodes, links: FOUR_NODES.links } } as never });
  await openPhotos(page);
  await page.getByRole("button", { name: "Nodes", exact: true }).click();
  await page.waitForTimeout(500);
  await probe(page, "initial");
  for (let i = 0; i < 12; i++) {
    for (const id of [1, 2, 3, 4]) {
      const c = page.locator(`[data-node-card="${id}"]`);
      const b = (await c.boundingBox())!;
      await page.mouse.move(b.x + b.width / 2, b.y + b.height / 2);
      await page.mouse.down();
      await page.mouse.move(b.x + b.width / 2 + 40 * (i % 3), b.y + b.height / 2 + 60 + i * 5, { steps: 5 });
      await page.mouse.up();
      await page.waitForTimeout(80);
    }
    await probe(page, "round " + i);
  }
});
