import { act, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { MockOptions } from "../../api.mock";
import { FAILURE_ZIPS, PACKAGE_ZIPS } from "../../api.mock";
import { openPhotos, renderApp } from "../../test/harness";
import { updater } from "../../updater";
import type { Harness } from "../../test/harness";

let h: Harness | undefined;
afterEach(() => {
  h?.unmount();
  h = undefined;
});

async function start(options: MockOptions = {}): Promise<Harness> {
  h = renderApp(options);
  await screen.findByRole("heading", { level: 1, name: "Drop Sony ARW files." });
  return h;
}

async function openPreferences(options: MockOptions = {}): Promise<HTMLElement> {
  await start(options);
  await h!.user.keyboard("{Meta>},{/Meta}");
  const dialog = await screen.findByRole("dialog", { name: "Preferences" });
  await waitFor(() => expect(within(dialog).queryByText("Loading your preferences…")).toBeNull());
  return dialog;
}

const sw = (dialog: HTMLElement, name: string): HTMLElement => within(dialog).getByRole("switch", { name });

describe("Preferences (⌘,)", () => {
  it("opens from Start with one row for each preference of docs/11", async () => {
    const dialog = await openPreferences();
    for (const row of ["Default look", "Display rendering", "Export defaults", "Remove location from exports", "Watch Downloads during import", "Check for updates automatically", "Command-line tool"]) {
      expect(within(dialog).getByRole("group", { name: row })).toBeInTheDocument();
    }
    expect(within(dialog).getByRole("heading", { level: 2, name: "Preferences" })).toBeInTheDocument();
    expect(within(dialog).getByRole("combobox", { name: "Default look" })).toHaveValue("builtin:neutral");
    expect(sw(dialog, "Remove location from exports")).toHaveAttribute("aria-checked", "false");
    expect(sw(dialog, "Watch Downloads during import")).toHaveAttribute("aria-checked", "false");
    expect(sw(dialog, "Check for updates automatically")).toHaveAttribute("aria-checked", "true");
    expect(within(dialog).getByText("Not installed. It lets you run ym from Terminal.")).toBeInTheDocument();
  });

  it("also opens over the photos, and Esc closes it and gives the keyboard back", async () => {
    h = renderApp({ current: 3, picked: [2, 3, 4] });
    await openPhotos(h);
    await h.user.keyboard("{Meta>},{/Meta}");
    await screen.findByRole("dialog", { name: "Preferences" });
    // The shortcuts of the screen below do nothing while the dialog is open.
    await h.user.keyboard("p");
    expect(h.store.currentPhoto()?.picked).toBe(true);
    await h.user.keyboard("{Escape}");
    expect(screen.queryByRole("dialog")).toBeNull();
    await h.user.keyboard("p");
    await waitFor(() => expect(h!.store.currentPhoto()?.picked).toBe(false));
  });

  it("leaves the panel behind alone when ⌘1 to ⌘6 are pressed over it", async () => {
    h = renderApp({ current: 3 });
    await openPhotos(h);
    await h.user.keyboard("{Meta>}e{/Meta}");
    expect(await screen.findByRole("complementary", { name: "Export" })).toBeInTheDocument();
    await h.user.keyboard("{Meta>},{/Meta}");
    await screen.findByRole("dialog", { name: "Preferences" });
    await h.user.keyboard("{Meta>}3{/Meta}");
    expect(h.store.getState().panel).toBe("export");
    expect(screen.getByRole("complementary", { name: "Export" })).toBeInTheDocument();
    // Once the dialog is closed the shortcut works again.
    await h.user.keyboard("{Escape}");
    await h.user.keyboard("{Meta>}3{/Meta}");
    await waitFor(() => expect(h!.store.getState().panel).toBe("tools"));
    expect(screen.getByRole("tab", { name: "Colour", selected: true })).toBeInTheDocument();
  });

  it("keeps the keyboard inside while it is open, and gives it back on Done", async () => {
    await start();
    const opener = screen.getByRole("button", { name: "Open files ⌘O" });
    opener.focus();
    await h!.user.keyboard("{Meta>},{/Meta}");
    const dialog = await screen.findByRole("dialog", { name: "Preferences" });
    await waitFor(() => expect(within(dialog).queryByText("Loading your preferences…")).toBeNull());
    expect(dialog).toHaveAttribute("aria-modal", "true");
    expect(dialog.contains(document.activeElement)).toBe(true);
    for (let i = 0; i < 25; i++) {
      await h!.user.tab();
      expect(dialog.contains(document.activeElement)).toBe(true);
    }
    for (let i = 0; i < 25; i++) {
      await h!.user.tab({ shift: true });
      expect(dialog.contains(document.activeElement)).toBe(true);
    }
    await h!.user.click(within(dialog).getByRole("button", { name: "Done" }));
    expect(screen.queryByRole("dialog")).toBeNull();
    expect(opener).toHaveFocus();
  });

  it("closes on a click outside it", async () => {
    await openPreferences();
    await h!.user.click(document.querySelector(".ym-dialog-backdrop") as HTMLElement);
    expect(screen.queryByRole("dialog")).toBeNull();
  });

  it("saves each switch as soon as it is flipped, and shows it again next time", async () => {
    const dialog = await openPreferences();
    await h!.user.click(sw(dialog, "Remove location from exports"));
    await h!.user.click(sw(dialog, "Watch Downloads during import"));
    await h!.user.click(sw(dialog, "Check for updates automatically"));
    const prefs = await h!.api.preferencesGet();
    expect(prefs).toMatchObject({ remove_location_from_exports: true, watch_downloads_during_import: true, check_for_updates_automatically: false });
    await h!.user.click(within(dialog).getByRole("button", { name: "Done" }));
    await h!.user.keyboard("{Meta>},{/Meta}");
    const again = await screen.findByRole("dialog", { name: "Preferences" });
    expect(sw(again, "Remove location from exports")).toHaveAttribute("aria-checked", "true");
    expect(sw(again, "Watch Downloads during import")).toHaveAttribute("aria-checked", "true");
    expect(sw(again, "Check for updates automatically")).toHaveAttribute("aria-checked", "false");
  });

  it("puts a switch back and says why when the core refuses", async () => {
    const dialog = await openPreferences();
    h!.api.failNext("preferencesSet", { scope: "preferences", message: "The preferences file can't be written. Check its permissions." });
    await h!.user.click(sw(dialog, "Remove location from exports"));
    expect(await screen.findByRole("alert")).toHaveTextContent("The preferences file can't be written. Check its permissions.");
    expect(sw(dialog, "Remove location from exports")).toHaveAttribute("aria-checked", "false");
  });

  it("changes the default look", async () => {
    const dialog = await openPreferences();
    const select = within(dialog).getByRole("combobox", { name: "Default look" });
    expect(within(select).getAllByRole("option").map((o) => o.textContent)).toContain("Film warm");
    await h!.user.selectOptions(select, "Signature");
    await waitFor(async () => expect((await h!.api.preferencesGet()).default_look).toBe("builtin:signature"));
    await waitFor(() => expect(within(dialog).getByRole("combobox", { name: "Default look" })).toHaveValue("builtin:signature"));
  });

  it("offers ARRI's display rendering only once display LUTs are imported", async () => {
    const dialog = await openPreferences({ looksImported: false });
    const group = within(dialog).getByRole("radiogroup", { name: "Display rendering" });
    expect(within(group).getByRole("radio", { name: "ARRI's display rendering, not imported" })).toHaveAttribute("aria-disabled", "true");
    expect(within(dialog).getByText("Import ARRI's display LUTs to use them here.")).toBeInTheDocument();
    await act(async () => {
      await h!.api.looksImport(`~/Downloads/${PACKAGE_ZIPS.display_luts_logc4}`);
    });
    await h!.store.refreshDisplayLuts();
    const arri = await within(dialog).findByRole("radio", { name: "ARRI's display rendering" });
    expect(arri).not.toHaveAttribute("aria-disabled");
    await h!.user.click(arri);
    await waitFor(async () => expect((await h!.api.preferencesGet()).display_rendering).toEqual({ kind: "arri_display", id: "arri-display:logc4:rec709-g24" }));
    await h!.user.click(within(dialog).getByRole("radio", { name: "Ours · ACES 2" }));
    await waitFor(async () => expect((await h!.api.preferencesGet()).display_rendering).toEqual({ kind: "aces2" }));
  }, 20_000);

  it("edits the export defaults with the options of the Export panel, without photos or a command line", async () => {
    const dialog = await openPreferences();
    await h!.user.click(within(dialog).getByRole("button", { name: "Edit…" }));
    expect(within(dialog).queryByRole("region", { name: "Photos" })).toBeNull();
    expect(within(dialog).queryByRole("region", { name: "Same thing from the command line" })).toBeNull();
    for (const card of ["Format", "Colour space", "Size", "Save to"]) expect(within(dialog).getByRole("region", { name: card })).toBeInTheDocument();
    expect(within(dialog).getByRole("radio", { name: "Full" })).toHaveAttribute("aria-checked", "true");
    await h!.user.click(within(dialog).getByRole("radio", { name: "TIFF" }));
    await h!.user.click(within(dialog).getByRole("radio", { name: "Rec.2020" }));
    await h!.user.click(within(dialog).getByRole("radio", { name: "Long edge" }));
    await waitFor(async () => expect((await h!.api.preferencesGet()).export_defaults).toMatchObject({ format: "tiff", color_space: "rec2020", size: { kind: "long_edge", px: 3000 } }));
    expect(within(dialog).getByText("Next to each source folder")).toBeInTheDocument();
    h!.api.queueChoice("folder", "~/Exports");
    await h!.user.click(within(dialog).getByRole("button", { name: "Choose…" }));
    await waitFor(async () => expect((await h!.api.preferencesGet()).export_defaults.destination).toBe("~/Exports"));
    await h!.user.click(within(dialog).getByRole("button", { name: "Hide" }));
    expect(within(dialog).queryByRole("region", { name: "Format" })).toBeNull();
  });

  it("installs the command-line tool and says where", async () => {
    const dialog = await openPreferences();
    await h!.user.click(within(dialog).getByRole("button", { name: "Install…" }));
    expect(await within(dialog).findByText("Installed at /usr/local/bin/ym")).toBeInTheDocument();
    expect(within(dialog).getByRole("button", { name: "Reinstall…" })).toBeInTheDocument();
    expect(await screen.findByText("The command-line tool is installed at /usr/local/bin/ym.")).toBeInTheDocument();
  });

  it("links to About", async () => {
    const dialog = await openPreferences();
    await h!.user.click(within(dialog).getByRole("button", { name: "About Yellowmoon" }));
    expect(await screen.findByRole("dialog", { name: "About Yellowmoon" })).toBeInTheDocument();
    expect(screen.queryByRole("dialog", { name: "Preferences" })).toBeNull();
  });
});

describe("About", () => {
  async function openAbout(): Promise<HTMLElement> {
    await start();
    act(() => h!.api.emitEvent("menu_action", { action: "about" }));
    return screen.findByRole("dialog", { name: "About Yellowmoon" });
  }

  it("shows the version, the licence and what it is built on", async () => {
    const dialog = await openAbout();
    expect(await within(dialog).findByText(/^Version \d+\.\d+\.\d+ · pipeline 1\.0\.0$/)).toBeInTheDocument();
    expect(within(dialog).getByText("Yellowmoon")).toBeInTheDocument();
    expect(within(dialog).getByText("Apache License 2.0. Copyright 2026 Mykyta Hriaznov and Yellowmoon contributors.")).toBeInTheDocument();
    expect(within(dialog).getByRole("region", { name: "Built with" })).toHaveTextContent("Figtree");
    expect(within(dialog).getByRole("region", { name: "Built with" })).toHaveTextContent("rawtoaces-data");
    expect(within(dialog).getByRole("region", { name: "Built with" })).toHaveTextContent("AMPAS");
  });

  it("carries the Adobe DNG notice, the Independent JPEG Group's acknowledgement and says it is not affiliated with ARRI or Sony", async () => {
    const dialog = await openAbout();
    expect(within(dialog).getByText("This product includes DNG technology under license by Adobe.")).toBeInTheDocument();
    // The same sentence as the system's About panel (`src-tauri/src/about.rs`).
    expect(within(dialog).getByText("This software is based in part on the work of the Independent JPEG Group.")).toBeInTheDocument();
    expect(
      within(dialog).getByText("Not affiliated with ARRI or Sony. ARRI and ALEXA are trademarks of ARRI AG; Sony is a trademark of Sony Group Corporation."),
    ).toBeInTheDocument();
  });

  it("opens the project's page in the browser and closes with Done", async () => {
    const dialog = await openAbout();
    await h!.user.click(within(dialog).getByRole("button", { name: "Yellowmoon on GitHub" }));
    expect(h!.api.system.opened).toEqual(["https://github.com/ngriaznov/yellowmoon"]);
    await h!.user.click(within(dialog).getByRole("button", { name: "Done" }));
    expect(screen.queryByRole("dialog")).toBeNull();
  });
});

describe("the menu bar", () => {
  async function menu(action: string): Promise<void> {
    act(() => h!.api.emitEvent("menu_action", { action } as never));
  }

  it("Preferences… opens Preferences", async () => {
    await start();
    await menu("preferences");
    expect(await screen.findByRole("dialog", { name: "Preferences" })).toBeInTheDocument();
  });

  it("Check for Updates… says what it found", async () => {
    await start();
    await menu("check_for_updates");
    expect(await screen.findByText("Yellowmoon is up to date.")).toBeInTheDocument();
    await h!.user.click(screen.getByRole("button", { name: "Dismiss" }));
    expect(screen.queryByText("Yellowmoon is up to date.")).toBeNull();
  });

  describe("when an update is already on its way", () => {
    afterEach(() => {
      updater.applyExternal({ phase: "idle" });
    });

    it("says it is ready, not that 'Update available' downloads it", async () => {
      await start();
      act(() => updater.applyExternal({ phase: "ready", version: "0.1.1" }));
      await menu("check_for_updates");
      expect(await screen.findByText("Version 0.1.1 is ready. Press Update ready to restart.")).toBeInTheDocument();
      expect(screen.queryByText(/press Update available/)).toBeNull();
    });

    it("says it is downloading", async () => {
      await start();
      act(() => updater.applyExternal({ phase: "downloading", version: "0.1.1", percent: 30 }));
      await menu("check_for_updates");
      expect(await screen.findByText("Version 0.1.1 is downloading.")).toBeInTheDocument();
    });
  });

  it("Install Command-Line Tool… installs it", async () => {
    await start();
    await menu("install_cli");
    expect(await screen.findByText("The command-line tool is installed at /usr/local/bin/ym.")).toBeInTheDocument();
    expect(await h!.api.cliLinkStatus()).toEqual({ installed: true, path: "/usr/local/bin/ym" });
  });

  it("Import ARRI Looks… opens the wizard", async () => {
    await start();
    await menu("import_arri_looks");
    expect(await screen.findByRole("heading", { level: 1, name: "Choose what to import" })).toBeInTheDocument();
  });

  it("Open Files… opens the twelve photos", async () => {
    await start();
    await menu("open_files");
    await screen.findByRole("heading", { level: 2, name: /photos? ready|Developing/ });
    expect(h!.store.getState().session?.photos).toHaveLength(12);
  });

  it("Export…, Undo, Redo, Pick, Next Photo and Before/After work on the photos", async () => {
    h = renderApp({ current: 3, picked: [] });
    await openPhotos(h);
    await menu("export");
    expect(await screen.findByRole("complementary", { name: "Export" })).toBeInTheDocument();
    await menu("pick");
    await waitFor(() => expect(h!.store.currentPhoto()?.picked).toBe(true));
    await menu("next_photo");
    await waitFor(() => expect(h!.store.currentPhoto()?.file_name).toBe("DSC04205.ARW"));
    await menu("previous_photo");
    await waitFor(() => expect(h!.store.currentPhoto()?.file_name).toBe("DSC04204.ARW"));
    await menu("before_after");
    await waitFor(() => expect(h!.store.getState().before).toBe(true));
    await menu("zoom_100");
    await waitFor(() => expect(h!.store.getState().zoom).toEqual({ kind: "percent", value: 100 }));
    await menu("zoom_in");
    await waitFor(() => expect(h!.store.getState().zoom).toEqual({ kind: "percent", value: 200 }));
    await menu("zoom_out");
    await waitFor(() => expect(h!.store.getState().zoom).toEqual({ kind: "percent", value: 100 }));
    await menu("fit");
    await waitFor(() => expect(h!.store.getState().zoom).toEqual({ kind: "fit" }));
    await menu("pick_all");
    await waitFor(() => expect(h!.store.getState().session?.picked).toHaveLength(12));
    await menu("clear_picks");
    await waitFor(() => expect(h!.store.getState().session?.picked).toHaveLength(0));
    await menu("close_session");
    expect(await screen.findByRole("heading", { level: 1, name: "Drop Sony ARW files." })).toBeInTheDocument();
  });

  it("Make Current Look Default saves the look of the photo on screen", async () => {
    h = renderApp({ current: 3 });
    await openPhotos(h);
    await h.store.chooseLook("arri:3000:film-cool");
    await menu("make_default_look");
    await waitFor(async () => expect((await h!.api.preferencesGet()).default_look).toBe("arri:3000:film-cool"));
  });
});

describe("Import .cube…", () => {
  async function chooseCube(...files: string[]): Promise<HTMLElement> {
    await start({ looksImported: false });
    h!.api.queueChoice("files", files.map((f) => `~/LUTs/${f}`));
    act(() => h!.api.emitEvent("menu_action", { action: "import_cube" }));
    return screen.findByRole("dialog", { name: "Import LUTs" });
  }

  it("asks what the LUT expects as input, and imports it into Your LUTs", async () => {
    const dialog = await chooseCube(FAILURE_ZIPS.ownCube);
    expect(within(dialog).getByText("Warm_Film.cube · 1 MB")).toBeInTheDocument();
    const group = within(dialog).getByRole("radiogroup", { name: "Input of Warm_Film.cube" });
    expect(within(group).getAllByRole("radio").map((r) => r.textContent)).toEqual(["LogC4", "LogC3", "LogC3 · Rec.709"]);
    const importButton = within(dialog).getByRole("button", { name: "Import" });
    expect(importButton).toHaveAttribute("aria-disabled", "true");
    expect(importButton).toHaveAttribute("title", "Say what each LUT expects as input.");
    await h!.user.click(within(group).getByRole("radio", { name: "LogC4" }));
    await h!.user.click(importButton);
    expect(await screen.findByText("1 LUT added to Your LUTs.")).toBeInTheDocument();
    expect(screen.queryByRole("dialog")).toBeNull();
    expect((await h!.api.looksList()).find((l) => l.id === "user:warm-film")).toMatchObject({ name: "Warm Film", input: "logc4", group: { kind: "user" } });
  });

  it("keeps the keyboard inside while the question is unanswered, and Esc closes it from anywhere", async () => {
    // jsdom has no layout: every element reports no offsetParent, which the focus trap reads as "not shown".
    const layout = vi.spyOn(HTMLElement.prototype, "offsetParent", "get").mockImplementation(function (this: HTMLElement) {
      return this.parentElement;
    });
    try {
      const dialog = await chooseCube(FAILURE_ZIPS.ownCube);
      const importButton = within(dialog).getByRole("button", { name: "Import" });
      expect(importButton).toHaveAttribute("aria-disabled", "true");
      // Focus starts on the question, not on the button that is locked until it is answered.
      expect(dialog.contains(document.activeElement)).toBe(true);
      expect(document.activeElement).not.toBe(importButton);
      expect(within(dialog).getByRole("radiogroup", { name: "Input of Warm_Film.cube" }).contains(document.activeElement)).toBe(true);
      const visited = new Set<Element | null>();
      for (let i = 0; i < 12; i++) {
        await h!.user.tab();
        expect(dialog.contains(document.activeElement)).toBe(true);
        visited.add(document.activeElement);
      }
      expect(visited.has(importButton)).toBe(true);
      for (let i = 0; i < 12; i++) {
        await h!.user.tab({ shift: true });
        expect(dialog.contains(document.activeElement)).toBe(true);
      }
      // A locked Import still takes the Tab from the last place and starts again at the first.
      importButton.focus();
      await h!.user.tab();
      expect(within(dialog).getByRole("radiogroup", { name: "Input of Warm_Film.cube" }).contains(document.activeElement)).toBe(true);
      // With the focus on the page behind (a click on the dimmed window, say) Tab comes back in and Esc still closes the dialog.
      (document.activeElement as HTMLElement).blur();
      await h!.user.tab();
      expect(dialog.contains(document.activeElement)).toBe(true);
      (document.activeElement as HTMLElement).blur();
      await h!.user.keyboard("{Escape}");
      expect(screen.queryByRole("dialog")).toBeNull();
    } finally {
      layout.mockRestore();
    }
  });

  it("starts on the Import button once there is nothing left to answer", async () => {
    const dialog = await chooseCube(PACKAGE_ZIPS.display_luts_logc4);
    expect(within(dialog).getByRole("button", { name: "Import" })).toHaveFocus();
  });

  it("imports an ARRI package without asking, and lists a file that can't be used with the reason", async () => {
    const dialog = await chooseCube(PACKAGE_ZIPS.display_luts_logc4, FAILURE_ZIPS.alf4Only);
    expect(within(dialog).getByText(/Display LUTs · LogC4 · 22 \.cube\. It is imported as it comes\./)).toBeInTheDocument();
    expect(within(dialog).getByText("This zip has ARRI Look Files (.alf4) but no .cube files. Download the 'LUT (.cube)' version from ARRI's page.")).toBeInTheDocument();
    expect(within(dialog).queryByRole("radiogroup")).toBeNull();
    await h!.user.click(within(dialog).getByRole("button", { name: "Import" }));
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull(), { timeout: 8000 });
    expect(await h!.api.looksDisplayLuts()).toHaveLength(22);
  }, 20_000);

  it("changes nothing when the chooser is cancelled, or the dialog is", async () => {
    await start({ looksImported: false });
    h!.api.queueChoice("files", null);
    act(() => h!.api.emitEvent("menu_action", { action: "import_cube" }));
    await waitFor(() => expect(h!.api.system.opened).toEqual([]));
    expect(screen.queryByRole("dialog")).toBeNull();
    h!.api.queueChoice("files", [`~/LUTs/${FAILURE_ZIPS.ownCube}`]);
    act(() => h!.api.emitEvent("menu_action", { action: "import_cube" }));
    const dialog = await screen.findByRole("dialog", { name: "Import LUTs" });
    await h!.user.click(within(dialog).getByRole("button", { name: "Cancel" }));
    expect(screen.queryByRole("dialog")).toBeNull();
    expect((await h!.api.looksList()).filter((l) => l.group.kind === "user")).toHaveLength(0);
  });

  it("is offered under the look grid in the Look tab", async () => {
    h = renderApp({ current: 3 });
    await openPhotos(h);
    h.api.queueChoice("files", [`~/LUTs/${FAILURE_ZIPS.ownCube}`]);
    await h.user.click(screen.getByRole("button", { name: "Import .cube…" }));
    expect(await screen.findByRole("dialog", { name: "Import LUTs" })).toBeInTheDocument();
  });
});
