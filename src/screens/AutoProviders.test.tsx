/**
 * The Auto providers in the app (`docs/22-auto-providers.md` section 12): the Preferences row "Auto grading" with its dropdown, hints and
 * notes, the consent dialog, the instructions' label and hint, and the Automatic card's lines, for Apple Intelligence, Claude Code and
 * Antigravity. The mock plays the computer (`MockOptions.providers`, `autoProvider`, `consent`, `os`). Apple Intelligence's own flows (the
 * instructions' saving, the sliders) are in `AppleAuto.test.tsx`; the Start screen's lines are in `Start.test.tsx`.
 */
import { screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { MockOptions } from "../api.mock";
import { openPhotos, renderApp } from "../test/harness";
import type { Harness } from "../test/harness";
import {
  AUTO_GRADING_TITLE,
  CONSENT_COPY,
  consentBody,
  EXPERIMENTAL_NOTE,
  INSTRUCTIONS_COPY,
  PROVIDER_HINTS,
  WINDOWS_CONSENT_SENTENCE,
} from "../state/autoProviders";

let h: Harness | undefined;
afterEach(() => {
  h?.unmount();
  h = undefined;
});

/** Opens Preferences over Start on a mock computer with the given providers. */
async function openPreferences(options: MockOptions = {}): Promise<HTMLElement> {
  h = renderApp(options);
  await screen.findByRole("heading", { level: 1, name: "Drop Sony ARW files." });
  await h.user.keyboard("{Meta>},{/Meta}");
  const dialog = await screen.findByRole("dialog", { name: "Preferences" });
  await waitFor(() => expect(within(dialog).queryByText("Loading your preferences…")).toBeNull());
  await waitFor(() => expect(h!.store.getState().auto).not.toBeNull());
  await waitFor(() => expect(h!.store.getState().os).not.toBeNull());
  return dialog;
}

/** The dropdown's trigger; `hidden` because Radix hides what is behind an open list or a dialog from assistive technology. */
const select = (dialog: HTMLElement): HTMLElement => within(dialog).getByRole("combobox", { name: AUTO_GRADING_TITLE, hidden: true });

/** Opens the dropdown and reads its entries. */
async function entries(dialog: HTMLElement): Promise<HTMLElement[]> {
  await h!.user.click(select(dialog));
  return screen.findAllByRole("option");
}

const ALL = { apple: "available", claude: "available", agy: "available" } as const;

describe("the Auto grading row is shown while a provider is there", () => {
  it("is hidden with no provider (the browser build) and with only Apple Intelligence switched off or a program still checked", async () => {
    for (const providers of [{}, { apple: "unavailable" }, { claude: "checking" }, { agy: "checking", apple: "checking" }] as const) {
      const dialog = await openPreferences({ providers });
      expect(within(dialog).getByRole("group", { name: "Display rendering" })).toBeInTheDocument();
      expect(within(dialog).queryByRole("group", { name: AUTO_GRADING_TITLE })).toBeNull();
      expect(within(dialog).queryByRole("combobox", { name: AUTO_GRADING_TITLE })).toBeNull();
      expect(within(dialog).queryByRole("textbox")).toBeNull();
      h!.unmount();
      h = undefined;
    }
  });

  it("is shown with any one provider, even a program that is only installed", async () => {
    for (const providers of [{ apple: "available" }, { claude: "available" }, { agy: "available" }, { claude: "signed_out" }, { agy: "too_old" }, { claude: "unavailable" }] as const) {
      const dialog = await openPreferences({ providers });
      expect(within(dialog).getByRole("group", { name: AUTO_GRADING_TITLE })).toBeInTheDocument();
      h!.unmount();
      h = undefined;
    }
  });

  it("sits after Display rendering, where the Apple row was, and before Export defaults", async () => {
    const dialog = await openPreferences({ providers: ALL });
    const groups = within(dialog)
      .getAllByRole("group")
      .map((g) => g.getAttribute("aria-label"));
    expect(groups.indexOf(AUTO_GRADING_TITLE)).toBe(groups.indexOf("Display rendering") + 1);
    expect(groups.indexOf("Export defaults")).toBeGreaterThan(groups.indexOf(AUTO_GRADING_TITLE));
  });
});

describe("the dropdown", () => {
  it("lists Yellowmoon's own Auto, Apple Intelligence, Claude Code and Antigravity (agy) in that order", async () => {
    const dialog = await openPreferences({ providers: ALL });
    const options = await entries(dialog);
    expect(options.map((o) => o.textContent)).toEqual(["Yellowmoon's own Auto", "Apple Intelligence", "Claude Code", "Antigravity (agy) Experimental"]);
    expect(select(dialog)).toHaveTextContent("Apple Intelligence");
  });

  it("lists Apple Intelligence only while it is available, and always Yellowmoon's own Auto", async () => {
    const dialog = await openPreferences({ providers: { claude: "available", apple: "unavailable" } });
    expect((await entries(dialog)).map((o) => o.textContent)).toEqual(["Yellowmoon's own Auto", "Claude Code"]);
  });

  it("does not list a program that is not installed or still being checked", async () => {
    const dialog = await openPreferences({ providers: { apple: "available", agy: "checking" } });
    expect((await entries(dialog)).map((o) => o.textContent)).toEqual(["Yellowmoon's own Auto", "Apple Intelligence"]);
  });

  it("lists a program that cannot be used disabled, with a muted tag after its name", async () => {
    const dialog = await openPreferences({ providers: { claude: "signed_out", agy: "too_old", apple: "available" } });
    const options = await entries(dialog);
    expect(options.map((o) => o.textContent)).toEqual(["Yellowmoon's own Auto", "Apple Intelligence", "Claude Code, not signed in Not signed in", "Antigravity (agy) Experimental, update needed Update needed"]);
    const claude = screen.getByRole("option", { name: "Claude Code, not signed in" });
    expect(claude).toHaveAttribute("aria-disabled", "true");
    expect(within(claude).getByText("Not signed in")).toHaveAttribute("data-slot", "badge");
    const agy = screen.getByRole("option", { name: /^Antigravity \(agy\)/ });
    expect(agy).toHaveAttribute("aria-disabled", "true");
    expect(within(agy).getByText("Update needed")).toHaveAttribute("data-slot", "badge");
    // Not working: a program that was found but did not answer its check.
    expect(screen.getByRole("option", { name: "Yellowmoon's own Auto" })).not.toHaveAttribute("aria-disabled");
  });

  it("says Not working for a program that did not answer its check", async () => {
    const dialog = await openPreferences({ providers: { claude: "unavailable" } });
    await entries(dialog);
    const claude = screen.getByRole("option", { name: "Claude Code, not working" });
    expect(claude).toHaveAttribute("aria-disabled", "true");
    expect(within(claude).getByText("Not working")).toBeInTheDocument();
  });

  it("does not choose a disabled entry", async () => {
    const dialog = await openPreferences({ providers: { claude: "signed_out", apple: "available" } });
    await entries(dialog);
    const spy = vi.spyOn(h!.api, "preferencesSet");
    await h!.user.click(screen.getByRole("option", { name: "Claude Code, not signed in" }));
    expect(spy).not.toHaveBeenCalled();
    expect(h!.store.getState().consentFor).toBeNull();
    expect(select(dialog)).toHaveTextContent("Apple Intelligence");
  });

  it("marks Antigravity Experimental, in the list and in the trigger", async () => {
    const dialog = await openPreferences({ providers: { agy: "available" }, autoProvider: "agy", consent: { claude: false, agy: true } });
    // The trigger shows the choice with the badge.
    expect(select(dialog)).toHaveTextContent("Antigravity (agy) Experimental");
    await entries(dialog);
    const option = screen.getByRole("option", { name: /^Antigravity \(agy\)/ });
    expect(within(option).getByText("Experimental")).toHaveAttribute("data-slot", "badge");
  });
});

describe("the hint under the title follows the provider that grades", () => {
  it("says what each provider does and what it sends, word for word", async () => {
    expect(PROVIDER_HINTS.yellowmoon).toBe(
      "Yellowmoon's own Auto sets the exposure and the white by a fixed rule and leaves every other value at 0. It is the same on every computer and sends nothing anywhere.",
    );
    expect(PROVIDER_HINTS.claude).toBe(
      "Claude Code looks at a small copy of each photo and grades it the same way. Yellowmoon runs the Claude Code installed on this computer and hands it the copy, 768 pixels on the long side, and your instructions. Claude Code sends them to Anthropic, or to the service it is set up to use, with your own sign-in. Yellowmoon itself makes no connection. Answers can differ from run to run.",
    );
    expect(PROVIDER_HINTS.agy).toBe(
      "Antigravity looks at a small copy of each photo and grades it the same way. Yellowmoon runs the Antigravity (agy) installed on this computer and hands it the copy, 768 pixels on the long side, and your instructions. Antigravity sends them to Google with your own sign-in. Google may keep and review them unless you opt out in Antigravity's own settings, and Google's Antigravity terms apply. Yellowmoon itself makes no connection. Answers can differ from run to run.",
    );
    const consent = { claude: true, agy: true };
    for (const [choice, hint] of [
      ["yellowmoon", PROVIDER_HINTS.yellowmoon],
      ["apple", PROVIDER_HINTS.apple],
      ["claude", PROVIDER_HINTS.claude],
      ["agy", PROVIDER_HINTS.agy],
    ] as const) {
      const dialog = await openPreferences({ providers: ALL, autoProvider: choice, consent });
      const row = within(dialog).getByRole("group", { name: AUTO_GRADING_TITLE });
      expect(row.querySelector(".ym-pref__hint")).toHaveTextContent(hint);
      // The whole paragraph shows: it is not cut after two lines like the others.
      expect(row.querySelector(".ym-pref__hint")).toHaveClass("line-clamp-none");
      h!.unmount();
      h = undefined;
    }
  });

  it("changes with the choice", async () => {
    const dialog = await openPreferences({ providers: ALL, consent: { claude: true, agy: true } });
    const row = within(dialog).getByRole("group", { name: AUTO_GRADING_TITLE });
    expect(row.querySelector(".ym-pref__hint")).toHaveTextContent(PROVIDER_HINTS.apple);
    await entries(dialog);
    await h!.user.click(screen.getByRole("option", { name: "Claude Code" }));
    await waitFor(() => expect(row.querySelector(".ym-pref__hint")).toHaveTextContent(PROVIDER_HINTS.claude));
  });

  it("says Yellowmoon's own Auto sends nothing while a program is installed but not chosen", async () => {
    const dialog = await openPreferences({ providers: { claude: "available" }, autoProvider: "yellowmoon" });
    const row = within(dialog).getByRole("group", { name: AUTO_GRADING_TITLE });
    expect(row.querySelector(".ym-pref__hint")).toHaveTextContent(PROVIDER_HINTS.yellowmoon);
    expect(select(dialog)).toHaveTextContent("Yellowmoon's own Auto");
  });
});

describe("Antigravity's Experimental note", () => {
  it("shows under the hint, with a badge, only while Antigravity grades", async () => {
    const dialog = await openPreferences({ providers: ALL, autoProvider: "agy", consent: { claude: true, agy: true } });
    const row = within(dialog).getByRole("group", { name: AUTO_GRADING_TITLE });
    const note = within(row).getByText(EXPERIMENTAL_NOTE);
    expect(note).toHaveClass("ym-row-note");
    expect(note.previousElementSibling).toHaveAttribute("data-slot", "badge");
    expect(note.previousElementSibling).toHaveTextContent("Experimental");
    expect(EXPERIMENTAL_NOTE).toBe(
      "Experimental. Antigravity has no way to take a picture on its input, so its agent opens the copy as a file. That is slower and less predictable than Claude Code, and the agent may open other tools. Yellowmoon runs it in a sandbox, in an empty folder, and never skips its permission checks.",
    );
    // The note sits under the hint in the row's text column.
    const hint = row.querySelector(".ym-pref__hint")!;
    expect(hint.compareDocumentPosition(note) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
  });

  it("is not there for the other providers", async () => {
    for (const choice of ["yellowmoon", "apple", "claude"] as const) {
      const dialog = await openPreferences({ providers: ALL, autoProvider: choice, consent: { claude: true, agy: true } });
      expect(within(dialog).queryByText(EXPERIMENTAL_NOTE)).toBeNull();
      h!.unmount();
      h = undefined;
    }
  });
});

describe("the notes under the row", () => {
  it("say what to do for each program that is installed but cannot be used, in the dropdown's order", async () => {
    const dialog = await openPreferences({ providers: { apple: "available", claude: "signed_out", agy: "signed_out" } });
    const notes = within(dialog).getAllByRole("note");
    expect(notes.map((n) => n.textContent)).toEqual([
      "Claude Code is installed but not signed in. Open Terminal, run claude and sign in.",
      "Antigravity is installed but not signed in. Open Terminal, run agy and sign in.",
    ]);
    // Muted lines, as the Camera card's note.
    expect(notes[0]).toHaveClass("ym-row-note");
  });

  it("fill in the version of a program that is too old", async () => {
    const dialog = await openPreferences({ providers: { claude: "too_old", agy: "too_old" } });
    expect(within(dialog).getAllByRole("note").map((n) => n.textContent)).toEqual([
      "Claude Code 2.1.100 is too old. Update it: open Terminal and run claude update.",
      "Antigravity 1.0.9 is too old. Update it: open Terminal and run agy update.",
    ]);
  });

  it("say that a program did not answer its check", async () => {
    const dialog = await openPreferences({ providers: { claude: "unavailable", agy: "unavailable" } });
    expect(within(dialog).getAllByRole("note").map((n) => n.textContent)).toEqual([
      "Claude Code didn't answer its check. Open Terminal, run claude to see if it starts, then open Preferences again.",
      "Antigravity didn't answer its check. Open Terminal, run agy to see if it starts, then open Preferences again.",
    ]);
  });

  it("say 'a terminal window' on Windows", async () => {
    const dialog = await openPreferences({ providers: { claude: "signed_out", agy: "too_old" }, os: "windows" });
    expect(within(dialog).getAllByRole("note").map((n) => n.textContent)).toEqual([
      "Claude Code is installed but not signed in. Open a terminal window, run claude and sign in.",
      "Antigravity 1.0.9 is too old. Update it: open a terminal window and run agy update.",
    ]);
  });

  it("have none for a program that is available, not installed or still being checked", async () => {
    const dialog = await openPreferences({ providers: { apple: "available", claude: "available", agy: "checking" } });
    expect(within(dialog).queryAllByRole("note")).toEqual([]);
  });

  it("follow a program that signs in while Preferences is open (auto_status_changed)", async () => {
    const dialog = await openPreferences({ providers: { claude: "signed_out" } });
    expect(within(dialog).getAllByRole("note")).toHaveLength(1);
    h!.api.setProviders({ claude: "available" });
    await waitFor(() => expect(within(dialog).queryAllByRole("note")).toEqual([]));
    await entries(dialog);
    expect(screen.getByRole("option", { name: "Claude Code" })).not.toHaveAttribute("aria-disabled");
  });
});

describe("the saved choice's note", () => {
  it("says the saved program is not available now and Yellowmoon's own Auto is used until it is", async () => {
    const dialog = await openPreferences({ providers: { apple: "available", claude: "signed_out" }, autoProvider: "claude", consent: { claude: true, agy: false } });
    expect(select(dialog)).toHaveTextContent("Yellowmoon's own Auto");
    expect(within(dialog).getByText("Your saved choice, Claude Code, isn't available now, so Auto uses Yellowmoon's own settings until it is.")).toHaveAttribute("role", "note");
    // Under Yellowmoon's own hint, with no instructions field: nothing grades.
    expect(within(dialog).queryByRole("textbox")).toBeNull();
  });

  it("is there for Antigravity too, and for a program that was not agreed to", async () => {
    let dialog = await openPreferences({ providers: { agy: "too_old" }, autoProvider: "agy" });
    expect(within(dialog).getByText("Your saved choice, Antigravity, isn't available now, so Auto uses Yellowmoon's own settings until it is.")).toBeInTheDocument();
    h!.unmount();
    h = undefined;
    // Available but not agreed to (a file edited by hand): it acts as Yellowmoon's own Auto, and the note says so.
    dialog = await openPreferences({ providers: { claude: "available" }, autoProvider: "claude" });
    expect(within(dialog).getByText("Your saved choice, Claude Code, isn't available now, so Auto uses Yellowmoon's own settings until it is.")).toBeInTheDocument();
  });

  it("is not there when the saved choice is Apple Intelligence, Yellowmoon's own Auto or a program that grades", async () => {
    for (const options of [
      { providers: { claude: "available" }, autoProvider: "apple" },
      { providers: { claude: "available" }, autoProvider: "yellowmoon" },
      { providers: { claude: "available" }, autoProvider: "claude", consent: { claude: true, agy: false } },
    ] as const) {
      const dialog = await openPreferences(options as MockOptions);
      expect(within(dialog).queryByText(/Your saved choice/)).toBeNull();
      h!.unmount();
      h = undefined;
    }
  });

  it("goes when the program is back", async () => {
    const dialog = await openPreferences({ providers: { claude: "signed_out" }, autoProvider: "claude", consent: { claude: true, agy: false } });
    expect(within(dialog).getByText(/Your saved choice/)).toBeInTheDocument();
    h!.api.setProviders({ claude: "available" });
    await waitFor(() => expect(within(dialog).queryByText(/Your saved choice/)).toBeNull());
    expect(select(dialog)).toHaveTextContent("Claude Code");
  });
});

describe("the consent dialog", () => {
  /** Chooses `name` in the dropdown. */
  async function choose(dialog: HTMLElement, name: string | RegExp): Promise<void> {
    await entries(dialog);
    await h!.user.click(screen.getByRole("option", { name }));
  }

  it("opens when Claude Code is chosen for the first time, with its text, and changes nothing until Use Claude Code", async () => {
    const dialog = await openPreferences({ providers: ALL });
    const spy = vi.spyOn(h!.api, "preferencesSet");
    await choose(dialog, "Claude Code");
    const consent = await screen.findByRole("dialog", { name: "Send photos to Claude Code?" });
    expect(within(consent).getByText(CONSENT_COPY.claude.body)).toBeInTheDocument();
    expect(CONSENT_COPY.claude.body).toBe(
      "To grade a photo, Yellowmoon runs Claude Code on this computer and hands it a copy of the photo, 768 pixels on the long side, and your instructions. Claude Code sends them to Anthropic, or to the service it is set up to use, with your own account. The original file, its name and its place on disk stay here. Whatever Anthropic keeps depends on your account's privacy settings. Yellowmoon itself makes no connection.",
    );
    // Cancel is first and takes the focus; the primary button says what it does.
    expect(within(consent).getByRole("button", { name: "Cancel" })).toHaveFocus();
    expect(within(consent).getByRole("button", { name: "Use Claude Code" })).toBeInTheDocument();
    // Nothing changed: the dropdown still shows Apple Intelligence.
    expect(spy).not.toHaveBeenCalled();
    expect(select(dialog)).toHaveTextContent("Apple Intelligence");
    expect((await h!.api.preferencesGet()).auto_provider).toBe("apple");
  });

  it("opens over Preferences: Preferences stays under it", async () => {
    const dialog = await openPreferences({ providers: ALL });
    await choose(dialog, "Claude Code");
    await screen.findByRole("dialog", { name: "Send photos to Claude Code?" });
    expect(screen.getAllByRole("dialog", { hidden: true }).map((d) => d.getAttribute("aria-label") ?? d.querySelector("h2")?.textContent)).toContain("Preferences");
  });

  it("Use Claude Code sends one patch with the choice and the consent, and closes the dialog", async () => {
    const dialog = await openPreferences({ providers: ALL });
    const spy = vi.spyOn(h!.api, "preferencesSet");
    await choose(dialog, "Claude Code");
    const consent = await screen.findByRole("dialog", { name: "Send photos to Claude Code?" });
    await h!.user.click(within(consent).getByRole("button", { name: "Use Claude Code" }));
    await waitFor(() => expect(screen.queryByRole("dialog", { name: "Send photos to Claude Code?" })).toBeNull());
    expect(spy).toHaveBeenCalledTimes(1);
    expect(spy).toHaveBeenCalledWith({ auto_provider: "claude", auto_consent: { claude: true, agy: false } });
    await waitFor(async () => expect(await h!.api.preferencesGet()).toMatchObject({ auto_provider: "claude", auto_consent: { claude: true, agy: false } }));
    await waitFor(() => expect(select(dialog)).toHaveTextContent("Claude Code"));
    // Preferences is still open and has its focus back.
    expect(screen.getByRole("dialog", { name: "Preferences" })).toBeInTheDocument();
  });

  it("asks once per program: the second time Claude Code is chosen it is saved at once", async () => {
    const dialog = await openPreferences({ providers: ALL });
    await choose(dialog, "Claude Code");
    await h!.user.click(await screen.findByRole("button", { name: "Use Claude Code" }));
    await waitFor(() => expect(select(dialog)).toHaveTextContent("Claude Code"));
    await choose(dialog, "Apple Intelligence");
    await waitFor(() => expect(select(dialog)).toHaveTextContent("Apple Intelligence"));
    await choose(dialog, "Claude Code");
    await waitFor(() => expect(select(dialog)).toHaveTextContent("Claude Code"));
    expect(screen.queryByRole("dialog", { name: "Send photos to Claude Code?" })).toBeNull();
  });

  it("Antigravity's dialog says it is experimental, and what the agent may do", async () => {
    const dialog = await openPreferences({ providers: ALL });
    const spy = vi.spyOn(h!.api, "preferencesSet");
    await choose(dialog, /^Antigravity \(agy\)/);
    const consent = await screen.findByRole("dialog", { name: "Send photos to Antigravity?" });
    expect(within(consent).getByText(CONSENT_COPY.agy.body)).toBeInTheDocument();
    expect(CONSENT_COPY.agy.body).toBe(
      "To grade a photo, Yellowmoon runs Antigravity (agy) on this computer and hands it a copy of the photo, 768 pixels on the long side, and your instructions. Antigravity sends them to Google with your own account. Google may keep and review them unless you opt out in Antigravity's own settings, and Google's Antigravity terms apply. The original file, its name and its place on disk stay here. Antigravity is an agent that can open other tools, so Yellowmoon runs it in a sandbox, in an empty folder, and never skips its permission checks. This is experimental. Yellowmoon itself makes no connection.",
    );
    expect(within(consent).getByRole("button", { name: "Cancel" })).toHaveFocus();
    await h!.user.click(within(consent).getByRole("button", { name: "Use Antigravity" }));
    await waitFor(() => expect(spy).toHaveBeenCalledWith({ auto_provider: "agy", auto_consent: { claude: false, agy: true } }));
    await waitFor(() => expect(select(dialog)).toHaveTextContent("Antigravity (agy) Experimental"));
  });

  it("Cancel closes it and keeps the previous choice", async () => {
    const dialog = await openPreferences({ providers: ALL });
    const spy = vi.spyOn(h!.api, "preferencesSet");
    await choose(dialog, "Claude Code");
    const consent = await screen.findByRole("dialog", { name: "Send photos to Claude Code?" });
    await h!.user.click(within(consent).getByRole("button", { name: "Cancel" }));
    await waitFor(() => expect(screen.queryByRole("dialog", { name: "Send photos to Claude Code?" })).toBeNull());
    expect(spy).not.toHaveBeenCalled();
    expect(select(dialog)).toHaveTextContent("Apple Intelligence");
    expect(await h!.api.preferencesGet()).toMatchObject({ auto_provider: "apple", auto_consent: { claude: false, agy: false } });
    // Preferences stays open.
    expect(screen.getByRole("dialog", { name: "Preferences" })).toBeInTheDocument();
  });

  it("Esc closes only the consent dialog and keeps the previous choice", async () => {
    const dialog = await openPreferences({ providers: ALL });
    await choose(dialog, "Claude Code");
    await screen.findByRole("dialog", { name: "Send photos to Claude Code?" });
    await h!.user.keyboard("{Escape}");
    await waitFor(() => expect(screen.queryByRole("dialog", { name: "Send photos to Claude Code?" })).toBeNull());
    expect(screen.getByRole("dialog", { name: "Preferences" })).toBeInTheDocument();
    expect(h!.store.getState().overlay).toBe("preferences");
    expect(select(dialog)).toHaveTextContent("Apple Intelligence");
    expect((await h!.api.preferencesGet()).auto_provider).toBe("apple");
    // A second Esc closes Preferences.
    await h!.user.keyboard("{Escape}");
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
  });

  it("adds the Windows sentence on Windows, and only there", async () => {
    expect(consentBody("claude", "macos")).toBe(CONSENT_COPY.claude.body);
    expect(consentBody("agy", "linux")).toBe(CONSENT_COPY.agy.body);
    expect(consentBody("claude", "windows")).toBe(`${CONSENT_COPY.claude.body} ${WINDOWS_CONSENT_SENTENCE}`);
    expect(WINDOWS_CONSENT_SENTENCE).toBe("The temporary folder's path includes your Windows user name, and the program may send it along.");
    const dialog = await openPreferences({ providers: ALL, os: "windows" });
    await choose(dialog, "Claude Code");
    const consent = await screen.findByRole("dialog", { name: "Send photos to Claude Code?" });
    expect(within(consent).getByText(`${CONSENT_COPY.claude.body} ${WINDOWS_CONSENT_SENTENCE}`)).toBeInTheDocument();
  });

  it("is never asked for Yellowmoon's own Auto or Apple Intelligence", async () => {
    const dialog = await openPreferences({ providers: ALL });
    await choose(dialog, "Yellowmoon's own Auto");
    await waitFor(() => expect(select(dialog)).toHaveTextContent("Yellowmoon's own Auto"));
    await choose(dialog, "Apple Intelligence");
    await waitFor(() => expect(select(dialog)).toHaveTextContent("Apple Intelligence"));
    expect(screen.queryByRole("dialog", { name: /^Send photos/ })).toBeNull();
  });

  it("Preferences, closed while the question is open, takes the question with it", async () => {
    const dialog = await openPreferences({ providers: ALL });
    await choose(dialog, "Claude Code");
    await screen.findByRole("dialog", { name: "Send photos to Claude Code?" });
    h!.store.closeOverlay();
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
    expect(h!.store.getState().consentFor).toBeNull();
  });
});

describe("the instructions follow the provider that grades", () => {
  it("are labelled for the provider, with its hint, and keep one text for all of them", async () => {
    const consent = { claude: true, agy: true };
    const dialog = await openPreferences({ providers: ALL, consent });
    const apple = within(dialog).getByRole("textbox", { name: INSTRUCTIONS_COPY.apple.label });
    expect(INSTRUCTIONS_COPY.apple).toEqual({
      label: "Instructions for Apple Intelligence",
      hint: "Apple's model reads these before each photo. The answer form stays the same.",
    });
    expect(INSTRUCTIONS_COPY.claude).toEqual({
      label: "Instructions for Claude Code",
      hint: "Claude reads these before each photo. The answer form stays the same.",
    });
    expect(INSTRUCTIONS_COPY.agy).toEqual({
      label: "Instructions for Antigravity",
      hint: "Antigravity reads these before each photo. The answer form stays the same.",
    });
    await h!.user.clear(apple);
    await h!.user.type(apple, "Grade it like a quiet documentary.");
    await h!.user.tab();
    await waitFor(async () => expect((await h!.api.preferencesGet()).apple_auto_instructions).toBe("Grade it like a quiet documentary."));
    for (const [name, copy] of [
      ["Claude Code", INSTRUCTIONS_COPY.claude],
      [/^Antigravity \(agy\)/, INSTRUCTIONS_COPY.agy],
      ["Apple Intelligence", INSTRUCTIONS_COPY.apple],
    ] as const) {
      await entries(dialog);
      await h!.user.click(screen.getByRole("option", { name }));
      const field = await within(dialog).findByRole("textbox", { name: copy.label });
      await waitFor(() => expect(within(dialog).getByRole("group", { name: copy.label })).toHaveTextContent(copy.hint));
      // The text, and the saved instructions, stay.
      expect(field).toHaveValue("Grade it like a quiet documentary.");
      expect(within(dialog).getByText(/\d+ of 2000 characters\./)).toBeInTheDocument();
      expect((await h!.api.preferencesGet()).apple_auto_instructions).toBe("Grade it like a quiet documentary.");
    }
  });

  it("are not shown while Yellowmoon's own Auto grades, whatever is installed", async () => {
    const dialog = await openPreferences({ providers: ALL, autoProvider: "yellowmoon" });
    expect(within(dialog).queryByRole("textbox")).toBeNull();
    expect(within(dialog).queryByRole("button", { name: "Reset to default" })).toBeNull();
  });

  it("show when a program is chosen once it is agreed to, and not before", async () => {
    const dialog = await openPreferences({ providers: { claude: "available" }, autoProvider: "yellowmoon" });
    await entries(dialog);
    await h!.user.click(screen.getByRole("option", { name: "Claude Code" }));
    await screen.findByRole("dialog", { name: "Send photos to Claude Code?" });
    expect(within(dialog).queryByRole("textbox")).toBeNull();
    await h!.user.click(screen.getByRole("button", { name: "Use Claude Code" }));
    expect(await within(dialog).findByRole("textbox", { name: "Instructions for Claude Code" })).toBeInTheDocument();
  });
});

describe("the Automatic card names the provider", () => {
  const automatic = () => within(screen.getByRole("region", { name: "Automatic" }));
  const lines = {
    claude: ["Claude Code", "Claude Code is looking at this photo…", "Set with Claude Code.", "Claude Code couldn't look at this photo, so Auto uses Yellowmoon's own settings."],
    agy: ["Antigravity", "Antigravity is looking at this photo…", "Set with Antigravity.", "Antigravity couldn't look at this photo, so Auto uses Yellowmoon's own settings."],
  } as const;

  for (const provider of ["claude", "agy"] as const) {
    const [name, pending, answered, failed] = lines[provider];
    it(`${name}: pending, answered and failed say so with its name`, async () => {
      h = renderApp({ current: 3, appleAuto: "pending", providers: { [provider]: "available" }, autoProvider: provider, consent: { claude: true, agy: true } });
      await openPhotos(h);
      expect(automatic().getByText(pending)).toBeInTheDocument();
      h.api.setAppleAuto("answered");
      await waitFor(() => expect(automatic().getByText(answered)).toBeInTheDocument());
      expect(screen.queryByText(pending)).toBeNull();
      // The rows of the answer and the plan are the same as with Apple Intelligence.
      expect(screen.getByRole("button", { name: "Contrast +15, Auto. Open Tone" })).toBeInTheDocument();
      expect(automatic().getByText("“Teal shadows and warm highlights for a cinematic late-sun street”")).toBeInTheDocument();
      h.api.setAppleAuto("failed");
      await waitFor(() => expect(automatic().getByText(failed)).toBeInTheDocument());
    });
  }

  it("says a program that was chosen is not available right now, and Auto is Yellowmoon's own", async () => {
    const consent = { claude: true, agy: true };
    for (const [provider, state, name] of [
      ["claude", "signed_out", "Claude Code"],
      ["claude", "too_old", "Claude Code"],
      ["agy", "unavailable", "Antigravity"],
    ] as const) {
      h = renderApp({ current: 3, providers: { [provider]: state }, autoProvider: provider, consent });
      await openPhotos(h);
      expect(automatic().getByText(`${name} isn't available right now, so Auto uses Yellowmoon's own settings.`)).toBeInTheDocument();
      const labels = automatic().getAllByRole("button").map((b) => b.querySelector(".ym-auto-row__label")?.textContent ?? "").filter(Boolean);
      expect(labels).toEqual(["Exposure", "White", "White balance", "Noise", "Camera"]);
      h.unmount();
      h = undefined;
    }
  });

  it("says nothing for a program that is still being checked or not agreed to, nor for Apple Intelligence, nor with Yellowmoon's own Auto", async () => {
    for (const options of [
      { providers: { claude: "checking" }, autoProvider: "claude", consent: { claude: true, agy: false } },
      { providers: { claude: "available" }, autoProvider: "claude" },
      { providers: { apple: "unavailable" }, autoProvider: "apple" },
      { providers: { claude: "available" }, autoProvider: "yellowmoon" },
    ] as const) {
      h = renderApp({ current: 3, ...(options as MockOptions) });
      await openPhotos(h);
      expect(automatic().queryByText(/isn't available right now|looking at this photo|Set with|couldn't look/)).toBeNull();
      h.unmount();
      h = undefined;
    }
  });

  it("says a program that was chosen and is gone (uninstalled since) is not available right now", async () => {
    h = renderApp({ current: 3, providers: { apple: "available" }, autoProvider: "agy", consent: { claude: false, agy: true } });
    await openPhotos(h);
    expect(automatic().getByText("Antigravity isn't available right now, so Auto uses Yellowmoon's own settings.")).toBeInTheDocument();
  });

  it("follows a program that signs out: the line appears, and goes when it is back", async () => {
    h = renderApp({ current: 3, appleAuto: "answered", providers: { claude: "available" }, autoProvider: "claude", consent: { claude: true, agy: false } });
    await openPhotos(h);
    expect(automatic().getByText("Set with Claude Code.")).toBeInTheDocument();
    h.api.setProviders({ claude: "signed_out" });
    await waitFor(() => expect(automatic().getByText("Claude Code isn't available right now, so Auto uses Yellowmoon's own settings.")).toBeInTheDocument());
    h.api.setProviders({ claude: "available" });
    await waitFor(() => expect(automatic().getByText("Set with Claude Code.")).toBeInTheDocument());
  });
});
