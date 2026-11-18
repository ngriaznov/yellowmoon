import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { AddNodeMenu } from "./AddNodeMenu";

async function open(count: number, onAddColorUniform?: () => void) {
  const user = userEvent.setup();
  render(<AddNodeMenu count={count} onAddCorrector={() => undefined} onAddColorUniform={onAddColorUniform} onAddParallel={() => undefined} />);
  await user.click(screen.getByRole("button", { name: "Add node" }));
  return { user, menu: await screen.findByRole("menu", { name: "Add node" }) };
}

describe("the Colour Uniform item of Add node (docs/32 section 4)", () => {
  it("adds one node: on at 63 nodes, off at 64 with the sentence, and the rule comes after the fourth item", async () => {
    const add = vi.fn();
    const { user, menu } = await open(63, add);
    const item = within(menu).getByRole("menuitem", { name: /^Colour Uniform/ });
    expect(item).not.toHaveAttribute("aria-disabled", "true");
    expect(within(menu).getAllByRole("menuitem").map((i) => i.querySelector(".ym-menu-line__name")?.textContent)).toEqual(["Corrector", "Film", "Face Refinement", "Colour Uniform", "Parallel", "Layer", "Splitter and combiner"]);
    const rule = menu.querySelector('[role="separator"]')!;
    expect(rule.previousElementSibling?.textContent).toMatch(/^Colour Uniform/);
    await user.click(item);
    expect(add).toHaveBeenCalledTimes(1);
  });

  it("is off at 64 nodes", async () => {
    const { menu } = await open(64, () => undefined);
    expect(within(menu).getByRole("menuitem", { name: /^Colour Uniform/ })).toHaveAttribute("aria-disabled", "true");
  });

  it("is off without a handler", async () => {
    const { menu } = await open(2);
    expect(within(menu).getByRole("menuitem", { name: /^Colour Uniform/ })).toHaveAttribute("aria-disabled", "true");
  });
});
