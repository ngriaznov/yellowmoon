import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { DevelopList, ExportQueueList } from "./ExportQueueList";
import type { DevelopItem, ExportQueueItem } from "./ExportQueueList";

describe("ExportQueueList", () => {
  const items: ExportQueueItem[] = [
    { id: 1, file: "DSC04203_neutral.jpg", state: "done" },
    { id: 2, file: "DSC04204_neutral.jpg", state: "running", percent: 45.4 },
    { id: 3, file: "DSC04205_neutral.jpg", state: "waiting" },
    { id: 4, file: "DSC04206_neutral.jpg", state: "failed", note: "The folder is full." },
  ];

  it("lists one row per file in a named list", () => {
    render(<ExportQueueList items={items} />);
    const list = screen.getByRole("list", { name: "Export queue" });
    expect(within(list).getAllByRole("listitem")).toHaveLength(4);
  });

  it("shows the tag and the default note of each state", () => {
    render(<ExportQueueList items={items} />);
    const rows = screen.getAllByRole("listitem");
    expect(within(rows[0] as HTMLElement).getByText("Done")).toHaveAttribute("data-variant", "done");
    expect(within(rows[0] as HTMLElement).getByText("Saved")).toBeInTheDocument();
    expect(within(rows[1] as HTMLElement).getByText("45%")).toHaveAttribute("data-variant", "active");
    expect(within(rows[1] as HTMLElement).getByText("Rendering at full quality")).toBeInTheDocument();
    expect(within(rows[2] as HTMLElement).getByText("Waiting")).toHaveAttribute("data-variant", "waiting");
    expect(within(rows[2] as HTMLElement).getByText("Queued")).toBeInTheDocument();
  });

  it("says why a file failed", () => {
    render(<ExportQueueList items={items} />);
    const row = screen.getAllByRole("listitem")[3] as HTMLElement;
    expect(within(row).getByText("Failed")).toHaveAttribute("data-variant", "failed");
    expect(within(row).getByText("The folder is full.")).toBeInTheDocument();
  });

  it("shows SKIPPED in orange for a file that already existed, with its own note", () => {
    render(<ExportQueueList items={[{ id: 1, file: "a.jpg", state: "skipped" }]} />);
    const row = screen.getByText("a.jpg").closest("li") as HTMLElement;
    expect(row).toHaveTextContent("Skipped, the file already exists");
    expect(row.querySelector(".ym-tag")).toHaveAttribute("data-variant", "skipped");
    expect(row.querySelector(".ym-tag")).toHaveTextContent("Skipped");
  });

  it("shows 0% for a running file with no percentage yet", () => {
    render(<ExportQueueList items={[{ id: 1, file: "a.jpg", state: "running" }]} />);
    expect(screen.getByText("0%")).toBeInTheDocument();
  });
});

describe("DevelopList", () => {
  const items: DevelopItem[] = [
    { id: 1, name: "DSC04201", state: "ready" },
    { id: 2, name: "DSC04202", state: "developing" },
    { id: 3, name: "DSC04203", state: "waiting" },
    { id: 4, name: "DSC04204", state: "skipped", note: "ARW 6.0 compressed: not supported yet." },
  ];

  it("lists the photos with their tags", () => {
    render(<DevelopList items={items} />);
    const rows = screen.getAllByRole("listitem");
    expect(rows).toHaveLength(4);
    expect(within(rows[0] as HTMLElement).getByText("Ready")).toHaveAttribute("data-variant", "ready");
    expect(within(rows[1] as HTMLElement).getByText("Developing")).toHaveAttribute("data-variant", "developing");
    expect(within(rows[2] as HTMLElement).getByText("Waiting")).toHaveAttribute("data-variant", "waiting");
    expect(within(rows[3] as HTMLElement).getByText("Skipped")).toHaveAttribute("data-variant", "skipped");
  });

  it("explains what is happening, with the copy of the mockup", () => {
    render(<DevelopList items={items} />);
    expect(screen.getByText("Auto exposure, white balance and noise")).toBeInTheDocument();
    expect(screen.getByText("Decoding and demosaicing")).toBeInTheDocument();
    expect(screen.getByText("ARW 6.0 compressed: not supported yet.")).toBeInTheDocument();
  });

  it("is a named list", () => {
    render(<DevelopList items={items} label="Photos being developed" />);
    expect(screen.getByRole("list", { name: "Photos being developed" })).toBeInTheDocument();
  });
});
