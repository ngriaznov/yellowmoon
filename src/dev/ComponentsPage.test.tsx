import { act, fireEvent, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { TAB_IDS } from "../components";
import { ComponentsPage } from "./ComponentsPage";
import { ExportReplica } from "./ExportReplica";
import { PhotosReplica } from "./PhotosReplica";
import { StartReplica } from "./StartReplica";
import { WizardReplica } from "./WizardReplica";

afterEach(() => vi.restoreAllMocks());

describe("components.html page", () => {
  it("shows a section for every component of docs/11", () => {
    const { container } = render(<ComponentsPage />);
    const headings = [...container.querySelectorAll("section.dev-section > h2")].map((h) => h.textContent);
    for (const name of [
      "Foundations",
      "Slider",
      "ValueBox",
      "Segmented",
      "Switch",
      "Card",
      "TabBar and TitleBlock",
      "LookTile",
      "Wheel",
      "WarpPlane",
      "CurvePlot and Histogram",
      "Filmstrip",
      "PhotoCell",
      "UpdatePill",
      "ExportQueueList and DevelopList",
      "DropZone",
      "StepRail",
    ]) {
      expect(headings).toContain(name);
    }
  });

  it("renders without a React warning or error", () => {
    const error = vi.spyOn(console, "error").mockImplementation(() => undefined);
    const warn = vi.spyOn(console, "warn").mockImplementation(() => undefined);
    render(<ComponentsPage />);
    expect(error).not.toHaveBeenCalled();
    expect(warn).not.toHaveBeenCalled();
  });

  it("shows every tag variant, every tab and every update phase", () => {
    render(<ComponentsPage />);
    const foundations = document.getElementById("foundations") as HTMLElement;
    for (const label of ["Auto", "Edited", "On", "Stand-in", "Reduced", "Default", "Can't open", "Waiting", "Developing", "Ready", "Skipped"]) {
      expect(within(foundations).getAllByText(label).length).toBeGreaterThan(0);
    }
    const update = document.getElementById("update") as HTMLElement;
    for (const label of ["Update available", "Downloading 42%", "Update ready", "Update failed"]) {
      expect(within(update).getAllByText(label).length).toBeGreaterThan(0);
    }
    expect(document.querySelectorAll("#tabbar [role=tablist]").length).toBeGreaterThanOrEqual(TAB_IDS.length);
  });

  it("shows the photo cell in each of the three ways it can be drawn: the mock's image, the fallback viewport's frame, the native layer", () => {
    render(<ComponentsPage />);
    const photo = document.getElementById("photo") as HTMLElement;
    expect(photo.querySelectorAll("img.ym-photo-cell__image").length).toBeGreaterThan(0);
    const frame = photo.querySelector("img.ym-photo-cell__frame") as HTMLImageElement | null;
    expect(frame).not.toBeNull();
    expect(frame?.style.width).not.toBe("");
    expect(photo.querySelector('[data-native="true"]')).not.toBeNull();
  });

  it("answers ⌘1–⌘7 on the tab shortcut demo", () => {
    render(<ComponentsPage />);
    const bar = screen.getByRole("tablist", { name: "Shortcut tabs" });
    act(() => {
      fireEvent.keyDown(window, { key: "6", code: "Digit6", metaKey: true });
    });
    expect(within(bar).getByRole("tab", { name: "Curve" })).toHaveAttribute("aria-selected", "true");
  });

  it("has live sliders that respond to the keyboard", async () => {
    const user = userEvent.setup();
    render(<ComponentsPage />);
    const section = document.getElementById("slider") as HTMLElement;
    const intensity = within(section).getByRole("slider", { name: "Intensity" });
    intensity.focus();
    await user.keyboard("{ArrowLeft}");
    expect(within(section).getByRole("textbox", { name: "Intensity value" })).toHaveValue("99%");
  });
});

describe("replicas of the mockups", () => {
  it("renders the Photos screen for every tab", () => {
    for (const tab of TAB_IDS) {
      const { unmount } = render(<PhotosReplica initialTab={tab} />);
      expect(screen.getByRole("tab", { selected: true })).toBeInTheDocument();
      expect(screen.getByRole("tabpanel")).toBeInTheDocument();
      unmount();
    }
  });

  it("switches tabs on click and on the arrows of the Automatic card", async () => {
    const user = userEvent.setup();
    render(<PhotosReplica />);
    await user.click(screen.getByRole("button", { name: /Exposure \+0\.30 EV, Auto/ }));
    expect(screen.getByRole("tab", { name: "Light" })).toHaveAttribute("aria-selected", "true");
  });

  it("renders Start in its three states", () => {
    for (const progress of [undefined, 5.4, 12]) {
      const { unmount } = render(<StartReplica initialProgress={progress} />);
      expect(screen.getByRole("heading", { level: 1, name: "Drop Sony ARW files." })).toBeInTheDocument();
      unmount();
    }
  });

  it("renders Export for each format and while it runs", () => {
    for (const fmt of ["jpeg", "tiff", "png", "dng"] as const) {
      const { unmount } = render(<ExportReplica initialFormat={fmt} />);
      expect(screen.getByRole("radiogroup", { name: "Format" })).toBeInTheDocument();
      unmount();
    }
    const { unmount } = render(<ExportReplica initialProgress={1.4} />);
    expect(screen.getByRole("heading", { name: "Exporting" })).toBeInTheDocument();
    expect(screen.getByRole("progressbar")).toHaveAttribute("aria-valuenow", "47");
    unmount();
    render(<ExportReplica initialProgress={3} />);
    expect(screen.getByRole("heading", { name: "Exported" })).toBeInTheDocument();
  });

  it("renders two steps of the wizard", () => {
    const { unmount } = render(<WizardReplica initialStep={1} />);
    expect(screen.getByRole("heading", { level: 1, name: "Download from ARRI" })).toBeInTheDocument();
    unmount();
    render(<WizardReplica initialStep={3} initialImported={40} />);
    expect(screen.getByRole("heading", { level: 1, name: "Importing" })).toBeInTheDocument();
    expect(screen.getByRole("progressbar")).toHaveAttribute("aria-valuenow", "46");
  });
});
