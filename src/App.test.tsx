import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import App, { SCREENS } from "./App";

describe("App", () => {
  it("renders the Start placeholder first", () => {
    const { container } = render(<App />);
    expect(container.querySelector("[data-screen]")).toHaveAttribute("data-screen", "start");
    expect(screen.getByRole("heading", { level: 1, name: "Yellowmoon" })).toBeInTheDocument();
    expect(screen.getByText(/drop arw files to start/i)).toBeInTheDocument();
  });

  it("says plainly that the screen is a placeholder", () => {
    render(<App />);
    expect(screen.getByRole("note")).toHaveTextContent(/placeholder/i);
  });

  it("shows the moon mark next to the name, hidden from assistive technology", () => {
    const { container } = render(<App />);
    const title = screen.getByRole("heading", { level: 1 });
    const moon = title.querySelector(".ym-moon");
    expect(moon).not.toBeNull();
    expect(moon).toHaveAttribute("aria-hidden", "true");
    expect(container.querySelectorAll(".ym-moon")).toHaveLength(1);
  });

  it("routes between the three screens", async () => {
    expect(SCREENS).toEqual(["start", "photos", "import-looks"]);
    const user = userEvent.setup();

    const { container, unmount } = render(<App initialScreen="photos" />);
    expect(container.querySelector("[data-screen]")).toHaveAttribute("data-screen", "photos");
    expect(screen.getByText(/M3-06/)).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Back to Start" }));
    expect(container.querySelector("[data-screen]")).toHaveAttribute("data-screen", "start");
    unmount();

    const second = render(<App initialScreen="import-looks" />);
    expect(second.container.querySelector("[data-screen]")).toHaveAttribute("data-screen", "import-looks");
    expect(screen.getByRole("heading", { level: 1, name: "Import ARRI looks" })).toBeInTheDocument();
  });
});
