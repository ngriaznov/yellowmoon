import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { TooltipTrigger } from "@/components/glass/tooltip";
import { Button } from "./Button";
import { OptionalTooltip } from "./OptionalTooltip";

describe("OptionalTooltip around a pressed button", () => {
  it("keeps the pressed look: Radix's trigger state does not overwrite the toggle's data-state", () => {
    const { rerender } = render(
      <OptionalTooltip text="Click the photo">
        <TooltipTrigger asChild>
          <Button pressed>Pick colour</Button>
        </TooltipTrigger>
      </OptionalTooltip>,
    );
    const button = screen.getByRole("button", { name: "Pick colour" });
    expect(button).toHaveAttribute("aria-pressed", "true");
    expect(button).toHaveAttribute("data-state", "on");
    rerender(
      <OptionalTooltip text="Click the photo">
        <TooltipTrigger asChild>
          <Button pressed={false}>Pick colour</Button>
        </TooltipTrigger>
      </OptionalTooltip>,
    );
    expect(screen.getByRole("button", { name: "Pick colour" })).toHaveAttribute("data-state", "off");
  });
});
