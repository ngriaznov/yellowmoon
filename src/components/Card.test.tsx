import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { AutoRow, Card, Field, InfoRow, RowNote, Rows, SwitchRow } from "./Card";
import { ResetLink } from "./ResetLink";
import { AutoToggle } from "./Tag";

describe("Card", () => {
  it("is a section named by its title", () => {
    render(
      <Card title="High dynamic range">
        <p>body</p>
      </Card>,
    );
    const section = screen.getByRole("region", { name: "High dynamic range" });
    expect(within(section).getByRole("heading", { level: 3, name: "High dynamic range" })).toBeInTheDocument();
    expect(within(section).getByText("body")).toBeInTheDocument();
  });

  it("puts the tag and the action beside the title", () => {
    render(
      <Card title="Exposure" tag={<AutoToggle isAuto onAuto={() => undefined} />} action={<ResetLink size="card" />}>
        x
      </Card>,
    );
    const section = screen.getByRole("region", { name: "Exposure" });
    expect(within(section).getByRole("button", { name: "Auto" })).toBeInTheDocument();
    expect(within(section).getByRole("button", { name: "Reset" })).toBeInTheDocument();
  });

  it("gives its tab colour to everything inside", () => {
    render(
      <Card title="Exposure" tab="light">
        x
      </Card>,
    );
    expect(screen.getByRole("region")).toHaveAttribute("data-tab", "light");
  });

  it("has an option density for the Export panel and can drop its last rule", () => {
    render(
      <Card title="Save to" density="option" last>
        x
      </Card>,
    );
    const section = screen.getByRole("region");
    expect(section).toHaveAttribute("data-density", "option");
    expect(section).toHaveAttribute("data-last", "true");
  });
});

describe("Rows and the row types", () => {
  it("shows an info row with its label, value and tag", () => {
    render(
      <Rows>
        <InfoRow label="Detected" value="Sony A1 (ILCE-1)" />
        <InfoRow label="Profile" value="A7R IV" tag="stand-in" />
      </Rows>,
    );
    expect(screen.getByText("Detected")).toBeInTheDocument();
    expect(screen.getByText("Sony A1 (ILCE-1)")).toBeInTheDocument();
    expect(screen.getByText("Stand-in")).toHaveAttribute("data-variant", "stand-in");
  });

  it("describes an automatic row in full and opens its tab", async () => {
    const onOpen = vi.fn();
    render(<AutoRow label="Exposure" value="+0.30 EV" isAuto tabLabel="Light" onOpen={onOpen} />);
    const row = screen.getByRole("button", { name: "Exposure +0.30 EV, Auto. Open Light" });
    expect(within(row).getByText("Auto")).toHaveAttribute("data-variant", "auto");
    await userEvent.click(row);
    expect(onOpen).toHaveBeenCalledTimes(1);
  });

  it("marks an edited row", () => {
    render(<AutoRow label="White balance" value="6200 K · tint 0" isAuto={false} tabLabel="Colour" onOpen={() => undefined} />);
    const row = screen.getByRole("button", { name: /White balance 6200 K · tint 0, Edited/ });
    expect(within(row).getByText("Edited")).toHaveAttribute("data-variant", "edited");
  });

  it("toggles from a switch row", async () => {
    const onChange = vi.fn();
    render(<SwitchRow label="Highlight recovery" checked onChange={onChange} />);
    await userEvent.click(screen.getByRole("switch", { name: "Highlight recovery" }));
    expect(onChange).toHaveBeenCalledWith(false);
  });

  it("spaces the rows when asked", () => {
    const { container } = render(<Rows spaced>x</Rows>);
    expect(container.firstChild).toHaveAttribute("data-spaced", "true");
  });

  it("shows a note and a field", () => {
    render(
      <>
        <RowNote>Best for sharing and the web.</RowNote>
        <Field label="Apply to">
          <span>control</span>
        </Field>
      </>,
    );
    expect(screen.getByText("Best for sharing and the web.")).toBeInTheDocument();
    expect(screen.getByText("Apply to")).toHaveClass("ym-field__label");
    expect(screen.getByText("control")).toBeInTheDocument();
  });
});
