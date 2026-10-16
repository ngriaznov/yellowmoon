import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { RESTART_AFTER_EXPORT } from "../updater";
import { UpdatePill } from "./UpdatePill";

describe("UpdatePill", () => {
  it("draws nothing while idle", () => {
    const { container } = render(<UpdatePill state={{ phase: "idle" }} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("offers a download when an update is available", async () => {
    const onActivate = vi.fn();
    render(<UpdatePill state={{ phase: "available", version: "0.2.0" }} onActivate={onActivate} />);
    const pill = screen.getByRole("button", { name: "Update available" });
    expect(pill).toHaveAttribute("data-phase", "available");
    await userEvent.click(pill);
    expect(onActivate).toHaveBeenCalledTimes(1);
  });

  it("is a status while downloading, with the percentage", () => {
    const { rerender } = render(<UpdatePill state={{ phase: "downloading", version: "0.2.0", percent: 42 }} />);
    expect(screen.getByRole("status")).toHaveTextContent("Downloading 42%");
    expect(screen.queryByRole("button")).toBeNull();
    rerender(<UpdatePill state={{ phase: "downloading", version: "0.2.0" }} />);
    expect(screen.getByRole("status")).toHaveTextContent("Downloading…");
  });

  it("offers a restart when the update is ready", async () => {
    const onActivate = vi.fn();
    render(<UpdatePill state={{ phase: "ready", version: "0.2.0" }} onActivate={onActivate} />);
    const pill = screen.getByRole("button", { name: "Update ready" });
    expect(pill).not.toHaveAttribute("aria-disabled");
    await userEvent.click(pill);
    expect(onActivate).toHaveBeenCalledTimes(1);
  });

  it("is disabled while an export runs, and says why", async () => {
    const onActivate = vi.fn();
    render(<UpdatePill state={{ phase: "ready", version: "0.2.0" }} exportRunning onActivate={onActivate} />);
    const pill = screen.getByRole("button", { name: "Update ready" });
    expect(pill).toHaveAttribute("aria-disabled", "true");
    expect(pill).toHaveAccessibleDescription(RESTART_AFTER_EXPORT);
    expect(RESTART_AFTER_EXPORT).toBe("Restarts after the export finishes");
    await userEvent.click(pill);
    expect(onActivate).not.toHaveBeenCalled();
  });

  it("shows why it is locked as a tooltip on hover (a locked glass button takes no pointer events, so a native title would never show)", async () => {
    render(<UpdatePill state={{ phase: "ready", version: "0.2.0" }} exportRunning />);
    const pill = screen.getByRole("button", { name: "Update ready" });
    // The reason is not a native title on the dead button; it is on the tooltip of the span around it, which does take the hover.
    expect(pill).not.toHaveAttribute("title");
    const trigger = pill.closest('[data-slot="tooltip-trigger"]') as HTMLElement;
    expect(trigger).not.toBeNull();
    expect(screen.queryByRole("tooltip")).toBeNull();
    await userEvent.hover(trigger);
    expect(await screen.findByRole("tooltip")).toHaveTextContent(RESTART_AFTER_EXPORT);
  });

  it("shows a failure with the reason on hover", () => {
    render(<UpdatePill state={{ phase: "error", message: "Could not reach the update server." }} />);
    const pill = screen.getByRole("status");
    expect(pill).toHaveTextContent("Update failed");
    expect(pill).toHaveAttribute("title", "Could not reach the update server.");
    expect(pill).toHaveAttribute("data-phase", "error");
  });

  it("hides its square from assistive technology", () => {
    const { container } = render(<UpdatePill state={{ phase: "ready", version: "0.2.0" }} />);
    expect(container.querySelector(".ym-update-pill__square")).toHaveAttribute("aria-hidden", "true");
  });
});
