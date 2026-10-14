import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { TitleBlock } from "./TitleBlock";
import { TABS } from "./tabs";

describe("TitleBlock", () => {
  it("takes the title and the blurb of its tab", () => {
    render(<TitleBlock tab="color" />);
    expect(screen.getByRole("heading", { level: 2, name: "Colour" })).toBeInTheDocument();
    expect(screen.getByText("White balance comes from the light in the scene. Set it by hand when you want a different mood.")).toBeInTheDocument();
  });

  it("carries the tab for the colour", () => {
    const { container } = render(<TitleBlock tab="raw" />);
    expect(container.firstChild).toHaveAttribute("data-tab", "raw");
  });

  it("has the six titles and blurbs of the mockup", () => {
    expect(TABS.map((t) => t.label)).toEqual(["Look", "Light", "Colour", "Detail", "Curve", "RAW"]);
    expect(TABS.find((t) => t.id === "curve")?.blurb).toBe("Fine contrast, for all channels or one at a time.");
    expect(TABS.find((t) => t.id === "light")?.blurb).toBe("Exposure starts at your camera's own. Adjust it from there.");
  });

  it("can be the Export panel's black block", () => {
    const { container } = render(<TitleBlock tone="ink" title="Export" blurb="3 photos from Lisbon. Nothing is saved besides the exported files." />);
    expect(container.firstChild).toHaveAttribute("data-tone", "ink");
    expect(screen.getByRole("heading", { name: "Export" })).toBeInTheDocument();
  });

  it("can announce its blurb as a status, for a queue that changes it (\"1 of 3 done.\")", () => {
    const { rerender } = render(<TitleBlock tone="ink" title="Exporting" blurb="0 of 3 done." live />);
    expect(screen.getByRole("status")).toHaveTextContent("0 of 3 done.");
    expect(screen.getByRole("status")).toHaveAttribute("aria-live", "polite");
    rerender(<TitleBlock tone="ink" title="Exporting" blurb="1 of 3 done." />);
    expect(screen.queryByRole("status")).toBeNull();
  });

  it("omits the blurb when there is none", () => {
    const { container } = render(<TitleBlock title="Crop" tone="ink" />);
    expect(container.querySelector("p")).toBeNull();
  });
});
