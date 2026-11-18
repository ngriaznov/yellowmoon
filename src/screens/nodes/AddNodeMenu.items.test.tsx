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
  it("adds one node: on at 63 nodes, off at 64 with the sentence, and the rule comes after the fifth item", async () => {
    const add = vi.fn();
    const { user, menu } = await open(63, add);
    const item = within(menu).getByRole("menuitem", { name: /^Colour Uniform/ });
    expect(item).not.toHaveAttribute("aria-disabled", "true");
    expect(within(menu).getAllByRole("menuitem").map((i) => i.querySelector(".ym-menu-line__name")?.textContent)).toEqual(["Corrector", "Film", "Face Refinement", "Colour Uniform", "Blur", "Parallel", "Layer", "Splitter and combiner"]);
    const rule = menu.querySelector('[role="separator"]')!;
    expect(rule.previousElementSibling?.textContent).toMatch(/^Blur/);
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

describe("the Blur item of Add node (docs/33 section 4)", () => {
  async function openBlur(count: number, onAddBlur?: () => void) {
    const user = userEvent.setup();
    render(<AddNodeMenu count={count} onAddCorrector={() => undefined} onAddBlur={onAddBlur} />);
    await user.click(screen.getByRole("button", { name: "Add node" }));
    return { user, menu: await screen.findByRole("menu", { name: "Add node" }) };
  }

  it("comes after Colour Uniform, says its line, and adds one node", async () => {
    const add = vi.fn();
    const { user, menu } = await openBlur(2, add);
    const item = within(menu).getByRole("menuitem", { name: /^Blur/ });
    expect(item).toHaveTextContent("Soft, lens, motion or zoom blur.");
    expect(item.previousElementSibling?.textContent).toMatch(/^Colour Uniform/);
    await user.click(item);
    expect(add).toHaveBeenCalledTimes(1);
  });

  it("is on at 63 nodes", async () => {
    const on = await openBlur(63, () => undefined);
    expect(within(on.menu).getByRole("menuitem", { name: /^Blur/ })).not.toHaveAttribute("aria-disabled", "true");
  });

  it("is off at 64 nodes and without a handler", async () => {
    const { menu } = await openBlur(64, () => undefined);
    const item = within(menu).getByRole("menuitem", { name: /^Blur/ });
    expect(item).toHaveAttribute("aria-disabled", "true");
    expect(item).toHaveAttribute("aria-description", "A photo can have up to 64 nodes.");
  });
});
