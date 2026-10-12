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

  it("hides its magnifier from assistive technology", () => {
    const { container } = render(<SearchField label="Search looks" value="" onChange={() => undefined} />);
    expect(container.querySelector("svg")).toHaveAttribute("aria-hidden", "true");
  });
});
