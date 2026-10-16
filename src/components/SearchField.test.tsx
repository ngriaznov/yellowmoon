import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { describe, expect, it } from "vitest";
import { SearchField } from "./SearchField";

function Harness() {
  const [q, setQ] = useState("");
  return <SearchField label="Search looks" value={q} onChange={setQ} />;
}

describe("SearchField", () => {
  it("is a search box named by its label with an ellipsis placeholder", () => {
    render(<SearchField label="Search looks" value="" onChange={() => undefined} />);
    const box = screen.getByRole("searchbox", { name: "Search looks" });
    expect(box).toHaveAttribute("placeholder", "Search looks…");
  });

  it("reports what is typed", async () => {
    render(<Harness />);
    await userEvent.type(screen.getByRole("searchbox"), "film");
    expect(screen.getByRole("searchbox")).toHaveValue("film");
  });

  it("has a Clear button while there is text, that clears and is not a Tab stop (Tab goes to what the field filters)", async () => {
    render(
      <>
        <Harness />
        <button type="button">First tile</button>
      </>,
    );
    const box = screen.getByRole("searchbox");
    expect(screen.queryByRole("button", { name: "Clear" })).toBeNull();
    await userEvent.type(box, "film");
    const clear = screen.getByRole("button", { name: "Clear" });
    expect(clear).toHaveAttribute("tabindex", "-1");
    await userEvent.tab();
    expect(screen.getByRole("button", { name: "First tile" })).toHaveFocus();
    await userEvent.click(clear);
    expect(box).toHaveValue("");
    expect(box).toHaveFocus();
  });

  it("uses an Esc that clears text, and lets an Esc in an empty field through to the page", async () => {
    const seen: boolean[] = [];
    function Page() {
      const [q, setQ] = useState("");
      return (
        // Bubbling after the field's own handler, as the page's shortcuts do: the field has had its say when this runs.
        <div onKeyDown={(event) => seen.push(event.defaultPrevented)}>
          <SearchField label="Search looks" value={q} onChange={setQ} />
        </div>
      );
    }
    render(<Page />);
    const box = screen.getByRole("searchbox");
    await userEvent.type(box, "film");
    seen.length = 0;
    await userEvent.keyboard("{Escape}");
    expect(box).toHaveValue("");
    expect(seen).toEqual([true]);
    await userEvent.keyboard("{Escape}");
    expect(seen).toEqual([true, false]);
  });

  it("hides its magnifier from assistive technology", () => {
    const { container } = render(<SearchField label="Search looks" value="" onChange={() => undefined} />);
    expect(container.querySelector("svg")).toHaveAttribute("aria-hidden", "true");
  });
});
