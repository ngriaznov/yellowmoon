/**
 * The Look tab's sections (owner's decision of 2026-10-16, `docs/11-ui.md` §Look tab): one section per look library, each a
 * disclosure button that folds away, with the same titles told apart by the section they sit in.
 */
import { act, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { MockOptions } from "../../api.mock";
import { openPhotos, renderApp } from "../../test/harness";
import type { Harness } from "../../test/harness";

let h: Harness | undefined;
afterEach(() => {
  h?.unmount();
  h = undefined;
});

const ALL = { current: 3, picked: [2, 3, 4], allLibraries: true } satisfies MockOptions;

async function setup(options: MockOptions = ALL): Promise<Harness> {
  h = renderApp(options);
  await openPhotos(h);
  return h;
}

const card = () => within(screen.getByRole("region", { name: "Look" }));
const lookOf = (): string | undefined => h?.store.currentSettings()?.values.look;
const search = (): HTMLElement => screen.getByRole("searchbox", { name: "Search looks" });

/** The header buttons of the sections, in the order they are drawn. */
const headers = (): HTMLElement[] =>
  card()
    .getAllByRole("button")
    .filter((b) => b.hasAttribute("aria-expanded"));

/** One section's header button, by the start of its accessible name ("ARRI looks · LogC3, "). */
const header = (start: string): HTMLElement => {
  const found = headers().find((b) => (b.getAttribute("aria-label") ?? "").startsWith(start));
  if (!found) throw new Error(`No section header starts with "${start}": ${headers().map((b) => b.getAttribute("aria-label")).join(" | ")}`);
  return found;
};

/** The part of a section that holds its tiles. */
const bodyOf = (button: HTMLElement): HTMLElement => {
  const id = button.getAttribute("aria-controls") ?? "";
  const body = document.getElementById(id);
  if (!body) throw new Error(`No element has the id ${id}`);
  return body;
};

describe("the sections of the Look tab", () => {
  it("are one per library, in order, each with its label and count", async () => {
    await setup();
    expect(headers().map((b) => b.getAttribute("aria-label"))).toEqual([
      "Built in, 2 looks, collapse",
      "ARRI looks · LogC4, 7 looks, collapse",
      "ARRI looks · LogC3, 5 looks, collapse",
      "ARRI looks · LogC3 to Rec.709, 4 looks, collapse",
      "ARRI looks · LogC3 to Rec.709 · ALF2, 6 looks, collapse",
      "Your LUTs, 5 looks, collapse",
    ]);
    expect(headers().map((b) => b.textContent)).toEqual([
      "Built in2",
      "ARRI looks · LogC47",
      "ARRI looks · LogC35",
      "ARRI looks · LogC3 to Rec.7094",
      "ARRI looks · LogC3 to Rec.709 · ALF26",
      "Your LUTs5",
    ]);
    for (const button of headers()) expect(button).toHaveAttribute("aria-expanded", "true");
  });

  it("make every header a button with a name and a body it controls", async () => {
    await setup();
    for (const button of headers()) {
      expect(button).toHaveAccessibleName();
      expect(bodyOf(button)).toBeInTheDocument();
    }
    expect(new Set(headers().map((b) => b.getAttribute("aria-controls"))).size).toBe(6);
  });

  it("show only the libraries that have looks", async () => {
    await setup({ current: 3, looksImported: true });
    expect(headers().map((b) => b.getAttribute("aria-label"))).toEqual(["Built in, 2 looks, collapse", "ARRI looks · LogC4, 7 looks, collapse"]);
    expect(screen.queryByText("Your LUTs")).toBeNull();
  });

  it("draw the tiles of a section inside it, and the same title in several of them", async () => {
    await setup();
    const logc4 = within(bodyOf(header("ARRI looks · LogC4, ")));
    const logc3 = within(bodyOf(header("ARRI looks · LogC3, ")));
    const mine = within(bodyOf(header("Your LUTs, ")));
    expect(logc4.getAllByRole("button")).toHaveLength(7);
    expect(logc3.getAllByRole("button")).toHaveLength(5);
    expect(logc4.getByRole("button", { name: "Film cool look, key 4" })).toBeInTheDocument();
    expect(logc3.getByRole("button", { name: "Film cool look" })).toBeInTheDocument();
    expect(mine.getByRole("button", { name: "Film cool look" })).toBeInTheDocument();
    // Inside a section a tile keeps the plain name: the header says the library.
    expect(screen.queryByText(/Film cool · /)).toBeNull();
  });

  it("keep the call to action while no ARRI looks are imported, whatever else is", async () => {
    await setup({ current: 3, looksImported: false });
    expect(headers().map((b) => b.getAttribute("aria-label"))).toEqual(["Built in, 2 looks, collapse"]);
    expect(screen.getByText("The free ARRI Look Library is not imported yet.")).toBeInTheDocument();
    expect(screen.getByText("ARRI looks")).toBeInTheDocument();
    expect(card().getByRole("button", { name: "Import ARRI looks →" })).toBeInTheDocument();
  });

  it("drop the call to action as soon as any ARRI library has a look", async () => {
    await setup({ current: 3, looksImported: false, allLibraries: true });
    expect(screen.queryByText("The free ARRI Look Library is not imported yet.")).toBeNull();
    expect(headers().map((b) => b.getAttribute("aria-label")?.split(",")[0])).toEqual([
      "Built in",
      "ARRI looks · LogC3",
      "ARRI looks · LogC3 to Rec.709",
      "ARRI looks · LogC3 to Rec.709 · ALF2",
      "Your LUTs",
    ]);
  });
});

describe("folding a section", () => {
  it("collapses and expands on a click, and says which in its name", async () => {
    await setup();
    const logc3 = header("ARRI looks · LogC3, ");
    expect(within(bodyOf(logc3)).getAllByRole("button")).toHaveLength(5);

    await h!.user.click(logc3);
    expect(logc3).toHaveAttribute("aria-expanded", "false");
    expect(logc3).toHaveAccessibleName("ARRI looks · LogC3, 5 looks, expand");
    expect(within(bodyOf(logc3)).queryAllByRole("button")).toHaveLength(0);
    expect(bodyOf(logc3)).not.toBeVisible();
    // The others are untouched.
    expect(header("ARRI looks · LogC4, ")).toHaveAttribute("aria-expanded", "true");
    expect(within(bodyOf(header("ARRI looks · LogC4, "))).getAllByRole("button")).toHaveLength(7);

    await h!.user.click(logc3);
    expect(logc3).toHaveAttribute("aria-expanded", "true");
    expect(logc3).toHaveAccessibleName("ARRI looks · LogC3, 5 looks, collapse");
    expect(within(bodyOf(logc3)).getAllByRole("button")).toHaveLength(5);
  });

  it("works from the keyboard", async () => {
    await setup();
    const mine = header("Your LUTs, ");
    mine.focus();
    await h!.user.keyboard("{Enter}");
    expect(mine).toHaveAttribute("aria-expanded", "false");
    await h!.user.keyboard(" ");
    expect(mine).toHaveAttribute("aria-expanded", "true");
  });

  it("is remembered in the store for the running app, and nowhere else", async () => {
    await setup();
    expect(h!.store.getState().collapsedLookLibraries).toEqual([]);
    await h!.user.click(header("ARRI looks · LogC3, "));
    await h!.user.click(header("Your LUTs, "));
    expect(h!.store.getState().collapsedLookLibraries).toEqual(["arri_logc3", "user"]);
    // Nothing is written to disk: no storage of the page holds it, and the mock saves nothing without `mockPersist`.
    expect(JSON.stringify({ ...localStorage })).not.toMatch(/collapsed|arri_logc3/);
    expect(JSON.stringify({ ...sessionStorage })).not.toMatch(/collapsed|arri_logc3/);
    // It survives the tab being left and drawn again, and starting a new session.
    await h!.user.click(screen.getByRole("tab", { name: "Light" }));
    await h!.user.click(screen.getByRole("tab", { name: "Look" }));
    expect(header("ARRI looks · LogC3, ")).toHaveAttribute("aria-expanded", "false");
    await act(async () => {
      await h!.store.closeSession();
    });
    expect(h!.store.getState().collapsedLookLibraries).toEqual(["arri_logc3", "user"]);
  });

  it("starts with every section open in a new app", async () => {
    await setup();
    await h!.user.click(header("ARRI looks · LogC4, "));
    h!.unmount();
    await setup();
    for (const button of headers()) expect(button).toHaveAttribute("aria-expanded", "true");
  });

  it("does not ask for the thumbnails of a folded section", async () => {
    h = renderApp(ALL);
    h.store.toggleLookLibrary("arri_alf2");
    const ask = vi.spyOn(h.api, "lookThumbnail");
    await openPhotos(h);
    // 29 looks in all, 6 of them in the folded ALF2 section.
    await waitFor(() => expect(ask).toHaveBeenCalledTimes(23));
    const asked = ask.mock.calls.map(([, look]) => look);
    expect(asked).toContain("arri:3000:film-warm-logc3");
    expect(asked.filter((id) => /-(low|medium|high)$/.test(id))).toEqual([]);
    // Opening the section asks for its tiles.
    await h.user.click(header("ARRI looks · LogC3 to Rec.709 · ALF2, "));
    await waitFor(() => expect(ask.mock.calls.map(([, look]) => look)).toContain("arri:3000:film-warm-low"));
  });
});

describe("the search", () => {
  it("opens every section that has matches, whatever the person folded, and shows 'N of M'", async () => {
    await setup();
    await h!.user.click(header("ARRI looks · LogC3, "));
    await h!.user.click(header("Your LUTs, "));
    expect(header("ARRI looks · LogC3, ")).toHaveAttribute("aria-expanded", "false");

    await h!.user.type(search(), "film");
    // Film warm, Film cool: LogC4 2 of 7, LogC3 2 of 5, LogC3 to Rec.709 2 of 4, ALF2 3 of 6 ("Film warm"), Your LUTs 2 of 5 ("Warm Film", "Film cool").
    expect(headers().map((b) => b.getAttribute("aria-label"))).toEqual([
      "ARRI looks · LogC4, 2 of 7 looks",
      "ARRI looks · LogC3, 2 of 5 looks",
      "ARRI looks · LogC3 to Rec.709, 2 of 4 looks",
      "ARRI looks · LogC3 to Rec.709 · ALF2, 3 of 6 looks",
      "Your LUTs, 2 of 5 looks",
    ]);
    expect(headers().map((b) => b.querySelector(".ym-look-group__count")?.textContent)).toEqual(["2 of 7", "2 of 5", "2 of 4", "3 of 6", "2 of 5"]);
    for (const button of headers()) expect(button).toHaveAttribute("aria-expanded", "true");
    // The folded sections show their matches.
    expect(within(bodyOf(header("ARRI looks · LogC3, "))).getAllByRole("button")).toHaveLength(2);
    expect(within(bodyOf(header("Your LUTs, "))).getAllByRole("button")).toHaveLength(2);
    // A section with no match is left out, and the search does not fold or unfold anything in the store.
    expect(screen.queryByText("Built in")).toBeNull();
    expect(h!.store.getState().collapsedLookLibraries).toEqual(["arri_logc3", "user"]);

    // Clearing the search puts every section back as the person left it.
    await h!.user.clear(search());
    expect(header("ARRI looks · LogC3, ")).toHaveAttribute("aria-expanded", "false");
    expect(header("Your LUTs, ")).toHaveAttribute("aria-expanded", "false");
    expect(header("ARRI looks · LogC4, ")).toHaveAttribute("aria-expanded", "true");
  });

  it("does not fold a section while it holds the open state: the header says so and ignores the click", async () => {
    await setup();
    await h!.user.type(search(), "film warm");
    const logc4 = header("ARRI looks · LogC4, ");
    expect(logc4).toHaveAttribute("aria-disabled", "true");
    // It does nothing, so Tab skips it: from the search box the next stop is the first tile.
    expect(logc4).toHaveAttribute("tabindex", "-1");
    await h!.user.click(search());
    await h!.user.tab();
    expect(card().getAllByRole("button", { name: / look, key \d/ })[0]).toHaveFocus();
    await h!.user.click(logc4);
    expect(logc4).toHaveAttribute("aria-expanded", "true");
    expect(h!.store.getState().collapsedLookLibraries).toEqual([]);
    await h!.user.clear(search());
    expect(header("ARRI looks · LogC4, ")).not.toHaveAttribute("aria-disabled");
    expect(header("ARRI looks · LogC4, ")).not.toHaveAttribute("tabindex");
  });

  it("says when nothing matches, and keeps the tile numbers of the whole list while it narrows it", async () => {
    await setup();
    await h!.user.type(search(), "bleach");
    // Bleach bypass is the 5th look of the whole list (Neutral, Signature, Film warm, Film cool, Bleach bypass) in LogC4, and keeps its key.
    expect(within(bodyOf(header("ARRI looks · LogC4, "))).getByRole("button", { name: "Bleach bypass look, key 5" })).toBeInTheDocument();
    expect(within(bodyOf(header("ARRI looks · LogC3, "))).getByRole("button", { name: "Bleach bypass look" })).toBeInTheDocument();
    await h!.user.clear(search());
    await h!.user.type(search(), "zzz");
    expect(screen.getByText("No looks match “zzz”.")).toBeInTheDocument();
    expect(headers()).toHaveLength(0);
  });
});

describe("the tile numbers and the keys 1 to 9", () => {
  it("follow the display order of every look across the sections, folded or not", async () => {
    await setup();
    // Neutral 1, Signature 2, then the LogC4 section: Film warm 3 … Monochrome 9; the sections after it have no numbers.
    const keys = card()
      .getAllByRole("button", { name: / look, key \d/ })
      .map((b) => b.getAttribute("aria-label"));
    expect(keys).toEqual([
      "Neutral look, key 1",
      "Signature look, key 2",
      "Film warm look, key 3",
      "Film cool look, key 4",
      "Bleach bypass look, key 5",
      "Teal and orange look, key 6",
      "Soft print look, key 7",
      "Night blue look, key 8",
      "Monochrome look, key 9",
    ]);
    // A section that is folded keeps the numbers of its looks: the key still picks one that is out of sight.
    await h!.user.click(header("ARRI looks · LogC4, "));
    await h!.user.keyboard("3");
    await waitFor(() => expect(lookOf()).toBe("arri:3000:film-warm"));
    await h!.user.keyboard("2");
    await waitFor(() => expect(lookOf()).toBe("builtin:signature"));
  });

  it("number the first section that has looks after Built in, when LogC4 has none", async () => {
    await setup({ current: 3, looksImported: false, allLibraries: true });
    const keys = card()
      .getAllByRole("button", { name: / look, key \d/ })
      .map((b) => b.getAttribute("aria-label"));
    expect(keys.slice(0, 4)).toEqual(["Neutral look, key 1", "Signature look, key 2", "Film warm look, key 3", "Film cool look, key 4"]);
    expect(within(bodyOf(header("ARRI looks · LogC3, "))).getByRole("button", { name: "Film warm look, key 3" })).toBeInTheDocument();
  });
});

describe("the tooltip of a tile", () => {
  const titleIn = (start: string, name: RegExp | string): string | null => within(bodyOf(header(start))).getByRole("button", { name }).getAttribute("title");

  it("names the section and what the look is", async () => {
    await setup();
    expect(titleIn("ARRI looks · LogC4, ", /^Film warm look/)).toBe("ARRI looks · LogC4 · 3D LUT, 65 points");
    expect(titleIn("ARRI looks · LogC3, ", /^Film warm look/)).toBe("ARRI looks · LogC3 · 3D LUT, 33 points");
    expect(titleIn("ARRI looks · LogC3 to Rec.709, ", /^Film warm look/)).toBe("ARRI looks · LogC3 to Rec.709 · 3D LUT, 33 points");
    expect(titleIn("ARRI looks · LogC3 to Rec.709 · ALF2, ", /^Film warm · Low look/)).toBe(
      "ARRI looks · LogC3 to Rec.709 · ALF2 · ALF2 look file, 33 points",
    );
    expect(titleIn("Your LUTs, ", /^Warm Film look/)).toBe("Your LUTs · 3D LUT, 33 points");
    expect(titleIn("Your LUTs, ", /^Warm Pull look/)).toBe("Your LUTs · ALF4 look file, 33 points");
    expect(titleIn("Your LUTs, ", /^Custom Look look/)).toBe("Your LUTs · ALF4C look file, 33 points");
    expect(titleIn("Your LUTs, ", /^Night look/)).toBe("Your LUTs · ALF2 look file, 33 points");
  });

  it("is left off a built-in look", async () => {
    await setup();
    expect(titleIn("Built in, ", /^Neutral look/)).toBeNull();
  });

  it("is the reason, for a look whose file is missing", async () => {
    await setup({ ...ALL, missingLooks: ["film-warm"] });
    const tile = within(bodyOf(header("ARRI looks · LogC4, "))).getByRole("button", { name: /^Film warm look, key 3, file missing/ });
    expect(tile).toHaveAttribute("title", "File missing or changed. Import again.");
  });
});

describe("the section that holds the current look", () => {
  const marker = (button: HTMLElement): Element | null => button.querySelector(".ym-look-group__marker");

  it("shows a small square in its header while it is folded, and says so in its name", async () => {
    await setup();
    // Pick the LogC3 "Film cool": its tile has no key, so it is clicked inside the section.
    await h!.user.click(within(bodyOf(header("ARRI looks · LogC3, "))).getByRole("button", { name: "Film cool look" }));
    await waitFor(() => expect(lookOf()).toBe("arri:3000:film-cool-logc3"));
    const logc3 = header("ARRI looks · LogC3, ");
    expect(marker(logc3)).toBeNull();

    await h!.user.click(logc3);
    expect(marker(logc3)).not.toBeNull();
    expect(marker(logc3)).toHaveAttribute("aria-hidden", "true");
    expect(logc3).toHaveAccessibleName("ARRI looks · LogC3, 5 looks, holds the current look, expand");
    // No other section carries it.
    for (const other of headers().filter((b) => b !== logc3)) expect(marker(other)).toBeNull();

    await h!.user.click(logc3);
    expect(marker(logc3)).toBeNull();
    expect(logc3).toHaveAccessibleName("ARRI looks · LogC3, 5 looks, collapse");
  });

  it("moves with the look: folding the section of the old look shows nothing once another is picked", async () => {
    await setup();
    await h!.user.click(header("Built in, "));
    // Neutral is the current look: the folded Built in section holds it.
    expect(marker(header("Built in, "))).not.toBeNull();
    await h!.user.keyboard("3");
    await waitFor(() => expect(lookOf()).toBe("arri:3000:film-warm"));
    expect(marker(header("Built in, "))).toBeNull();
    expect(header("Built in, ")).toHaveAccessibleName("Built in, 2 looks, expand");
  });
});

describe("a name that two libraries share", () => {
  it("says which one on the look chip of the photo", async () => {
    await setup();
    await h!.user.click(within(bodyOf(header("ARRI looks · LogC3, "))).getByRole("button", { name: "Film warm look" }));
    await waitFor(() => expect(lookOf()).toBe("arri:3000:film-warm-logc3"));
    const photo = within(screen.getByRole("main", { name: "Photo" }));
    expect(photo.getByText("Film warm · LogC3 · 100%")).toBeInTheDocument();
    expect(photo.getByRole("img", { name: "Photo DSC04204, Film warm · LogC3 · 100%" })).toBeInTheDocument();
  });

  it("says Your LUTs for the person's own, and leaves a built-in name plain", async () => {
    await setup();
    await h!.user.click(within(bodyOf(header("Your LUTs, "))).getByRole("button", { name: "Film cool look" }));
    await waitFor(() => expect(lookOf()).toBe("user:film-cool"));
    expect(within(screen.getByRole("main", { name: "Photo" })).getByText("Film cool · Your LUTs · 100%")).toBeInTheDocument();
    await h!.user.keyboard("2");
    await waitFor(() => expect(lookOf()).toBe("builtin:signature"));
    expect(within(screen.getByRole("main", { name: "Photo" })).getByText("Signature · 100%")).toBeInTheDocument();
  });

  it("keeps the plain name when the title is only in one library", async () => {
    await setup();
    await h!.user.keyboard("9");
    await waitFor(() => expect(lookOf()).toBe("arri:3000:monochrome"));
    expect(within(screen.getByRole("main", { name: "Photo" })).getByText("Monochrome · 100%")).toBeInTheDocument();
  });
});
