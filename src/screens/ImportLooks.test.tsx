import { act, fireEvent, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { MockOptions } from "../api.mock";
import { FAILURE_ZIPS, LOOK_FILE_ZIPS, PACKAGE_ZIPS } from "../api.mock";
import { openPhotos, renderApp } from "../test/harness";
import type { Harness } from "../test/harness";

let h: Harness | undefined;
afterEach(() => {
  h?.unmount();
  h = undefined;
});

const LL4 = PACKAGE_ZIPS.look_library_logc4;
const DL4 = PACKAGE_ZIPS.display_luts_logc4;
const LL3 = PACKAGE_ZIPS.look_library_logc3;

const drop = (...names: string[]): void => {
  const data = { types: ["Files"], files: names.map((name) => ({ name })), dropEffect: "none" };
  act(() => {
    fireEvent.dragEnter(window, { dataTransfer: data });
    fireEvent.drop(window, { dataTransfer: data });
  });
};

/** Start with nothing imported, then into the wizard. */
async function wizard(options: MockOptions = { looksImported: false }): Promise<Harness> {
  h = renderApp(options);
  if (options.looksImported === false) {
    await screen.findByText("Not imported yet");
    await h.user.click(screen.getByRole("button", { name: "Import →" }));
  } else {
    await screen.findByText("7 looks imported");
    act(() => h!.store.go("import-looks"));
  }
  await screen.findByRole("heading", { level: 1, name: "Choose what to import" });
  return h;
}

/** The Next button of the footer (its name changes with the step: Next, Import, Done). */
const next = (): HTMLElement => within(document.querySelector(".ym-wizard__footer") as HTMLElement).getByRole("button", { name: /^(Next|Import|Done)/ });
const rail = (title: string) => screen.getByRole("button", { name: new RegExp(`^${title}[,.]`) });
const step = async (): Promise<string> => (await screen.findByText(/^Import · step \d of 5$/)).textContent ?? "";

/** Choose, then drop the zips, then Recognise. */
async function toRecognise(...names: string[]): Promise<void> {
  await h!.user.click(next());
  await screen.findByRole("heading", { level: 1, name: "Download from ARRI" });
  drop(...names);
  await waitFor(() => expect(next()).not.toHaveAttribute("aria-disabled"));
  await h!.user.click(next());
  await screen.findByRole("table", { name: "What arrived" });
  await waitFor(() => expect(screen.queryByText("Checking")).toBeNull());
}

describe("step 1 · Choose", () => {
  it("shows the mockup's copy, with the recommended package ticked", async () => {
    await wizard();
    expect(screen.getByText("ARRI looks")).toBeInTheDocument();
    expect(await step()).toBe("Import · step 1 of 5");
    expect(screen.getByText("Free from ARRI. Imported to this Mac only, never shared or uploaded.")).toBeInTheDocument();
    expect(screen.getByText("ARRI publishes these looks for free. We can't include them in the app, so you download them from ARRI and the app imports them.")).toBeInTheDocument();
    const ll4 = screen.getByRole("checkbox", { name: "Look Library · LogC4" });
    expect(ll4).toHaveAttribute("aria-checked", "true");
    expect(ll4).toHaveAccessibleDescription(/87 creative looks, log to log.*87 looks · 65-point · 181 MB/);
    expect(within(ll4).getByText("Recommended")).toBeInTheDocument();
    expect(screen.getByRole("checkbox", { name: "Display LUTs · LogC4" })).toHaveAttribute("aria-checked", "false");
    expect(screen.queryByRole("checkbox", { name: "Look Library · LogC3" })).toBeNull();
    expect(screen.getByText("Download total: 181 MB")).toBeInTheDocument();
  });

  it("shows the three older packages when asked, and sums the download", async () => {
    await wizard();
    await h!.user.click(screen.getByRole("checkbox", { name: "Display LUTs · LogC4" }));
    expect(screen.getByText("Download total: 213 MB")).toBeInTheDocument();
    await h!.user.click(screen.getByRole("button", { name: "Show 3 older LogC3 packages" }));
    for (const name of ["Look Library · LogC3", "Look Library · LogC3 to Rec.709", "Display LUTs · LogC3"]) {
      expect(screen.getByRole("checkbox", { name })).toHaveAttribute("aria-checked", "false");
    }
    await h!.user.click(screen.getByRole("checkbox", { name: "Look Library · LogC3" }));
    expect(screen.getByText("Download total: 243 MB")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Hide the older LogC3 packages" })).toBeInTheDocument();
  });

  it("locks Next until a package is chosen, and says so", async () => {
    await wizard();
    await h!.user.click(screen.getByRole("checkbox", { name: "Look Library · LogC4" }));
    expect(screen.getByText("Choose at least one package")).toBeInTheDocument();
    expect(next()).toHaveAttribute("aria-disabled", "true");
    await h!.user.click(next());
    expect(await step()).toBe("Import · step 1 of 5");
  });

  it("locks the later steps in the rail, and Back at the first step", async () => {
    await wizard();
    expect(rail("Choose")).toHaveAttribute("aria-current", "step");
    for (const title of ["Get the files", "Recognise", "Import", "Finish"]) expect(rail(title)).toHaveAttribute("aria-disabled", "true");
    await h!.user.click(rail("Recognise"));
    expect(await step()).toBe("Import · step 1 of 5");
    expect(screen.getByRole("button", { name: "← Back" })).toHaveAttribute("aria-disabled", "true");
  });

  it("Cancel goes back to where the wizard was opened", async () => {
    await wizard();
    await h!.user.click(screen.getByRole("button", { name: "Cancel" }));
    expect(await screen.findByRole("heading", { level: 1, name: "Drop Sony ARW files." })).toBeInTheDocument();
  });
});

describe("step 2 · Get the files", () => {
  async function toGet(options?: MockOptions): Promise<void> {
    await wizard(options);
    await h!.user.click(next());
    await screen.findByRole("heading", { level: 1, name: "Download from ARRI" });
  }

  it("opens ARRI's page in the browser", async () => {
    await toGet();
    expect(screen.getByText("Opens in your browser. Download Look Library · LogC4 there.")).toBeInTheDocument();
    await h!.user.click(screen.getByRole("button", { name: /Open ARRI's page/ }));
    expect(h!.api.system.opened).toEqual(["https://www.arri.com/en/learn/camera-systems/image-science/look-files"]);
  });

  it("waits for the zip, and says what for", async () => {
    await toGet();
    expect(screen.getByText("Waiting for Look Library · LogC4…")).toBeInTheDocument();
    expect(screen.getByText("Next unlocks when the zip arrives.")).toBeInTheDocument();
    expect(next()).toHaveAttribute("aria-disabled", "true");
    expect(screen.getByRole("group", { name: "Or drop the zip here" })).toHaveTextContent("A zip or an unpacked folder both work.");
    expect(rail("Choose")).not.toHaveAttribute("aria-disabled");
    expect(rail("Recognise")).toHaveAttribute("aria-disabled", "true");
  });

  it("watches Downloads when the switch is on, and remembers the choice as a preference", async () => {
    await toGet();
    const watch = screen.getByRole("switch", { name: "Watch my Downloads folder" });
    expect(watch).toHaveAttribute("aria-checked", "false");
    await h!.user.click(watch);
    await waitFor(() => expect(screen.getByText("Watching Downloads for Look Library · LogC4…")).toBeInTheDocument());
    expect((await h!.api.preferencesGet()).watch_downloads_during_import).toBe(true);
    expect(screen.getByRole("switch", { name: "Watch my Downloads folder" })).toHaveAttribute("aria-checked", "true");
    // A zip lands in Downloads.
    act(() => void h!.api.simulateDownload(LL4));
    expect(await screen.findByRole("list", { name: "Files that arrived" })).toHaveTextContent(LL4);
    expect(screen.getByText("Everything has arrived. Press Next to check what each file is.")).toBeInTheDocument();
    await waitFor(() => expect(next()).not.toHaveAttribute("aria-disabled"));
    // Switching off stops watching.
    await h!.user.click(screen.getByRole("switch", { name: "Watch my Downloads folder" }));
    expect(h!.api.simulateDownload(DL4)).toBe(false);
  });

  it("starts with the switch on when the preference says so, and stops watching when the step is left", async () => {
    await toGet();
    await h!.api.preferencesSet({ watch_downloads_during_import: true });
    await h!.store.refreshPreferences();
    await waitFor(() => expect(h!.api.simulateDownload("nothing.zip")).toBe(true));
    await h!.user.click(screen.getByRole("button", { name: "← Back" }));
    await waitFor(() => expect(h!.api.simulateDownload("nothing.zip")).toBe(false));
  });

  it("says why macOS refused, and puts the switch back", async () => {
    await toGet({ looksImported: false, downloadsDenied: true });
    await h!.user.click(screen.getByRole("switch", { name: "Watch my Downloads folder" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Yellowmoon can't open your Downloads folder. Allow it in System Settings, under Privacy & Security, or choose the file yourself.");
    await waitFor(() => expect(screen.getByRole("switch", { name: "Watch my Downloads folder" })).toHaveAttribute("aria-checked", "false"));
    expect(screen.getByText("Waiting for Look Library · LogC4…")).toBeInTheDocument();
  });

  it("takes a zip dropped anywhere in the window", async () => {
    await toGet();
    drop(LL4);
    const list = await screen.findByRole("list", { name: "Files that arrived" });
    await waitFor(() => expect(list).toHaveTextContent("Look Library · LogC4 · 181 MB"));
    await waitFor(() => expect(next()).not.toHaveAttribute("aria-disabled"));
    expect(screen.getByText("Everything has arrived.")).toBeInTheDocument();
  });

  it("takes a file from Choose file…, and a cancelled chooser changes nothing", async () => {
    await toGet();
    h!.api.queueChoice("files", null);
    await h!.user.click(screen.getByRole("button", { name: "Choose file…" }));
    expect(screen.queryByRole("list", { name: "Files that arrived" })).toBeNull();
    await h!.user.click(screen.getByRole("button", { name: "Choose file…" }));
    expect(await screen.findByRole("list", { name: "Files that arrived" })).toHaveTextContent(LL4);
  });

  it("lets a file be removed again", async () => {
    await toGet();
    drop(LL4);
    await h!.user.click(await screen.findByRole("button", { name: `Remove ${LL4}` }));
    expect(screen.queryByRole("list", { name: "Files that arrived" })).toBeNull();
    expect(next()).toHaveAttribute("aria-disabled", "true");
  });

  it("needs a file for each chosen package", async () => {
    await wizard();
    await h!.user.click(screen.getByRole("checkbox", { name: "Display LUTs · LogC4" }));
    await h!.user.click(next());
    expect(screen.getByText("Opens in your browser. Download Look Library · LogC4 and Display LUTs · LogC4 there.")).toBeInTheDocument();
    expect(screen.getByText("Next unlocks when all 2 files have arrived.")).toBeInTheDocument();
    drop(LL4);
    await screen.findByRole("list", { name: "Files that arrived" });
    expect(next()).toHaveAttribute("aria-disabled", "true");
    drop(DL4);
    await waitFor(() => expect(next()).not.toHaveAttribute("aria-disabled"));
  });
});

describe("step 3 · Recognise", () => {
  it("lists what arrived with a Match and the contents", async () => {
    await wizard();
    await toRecognise(LL4);
    expect(await step()).toBe("Import · step 3 of 5");
    expect(screen.getByRole("heading", { level: 1, name: "Found it" })).toBeInTheDocument();
    const row = screen.getAllByRole("row")[1] as HTMLElement;
    expect(row).toHaveTextContent(LL4);
    expect(row).toHaveTextContent("181 MB");
    expect(row).toHaveTextContent("Look Library · LogC4 · log to log");
    expect(row).toHaveTextContent("87 .cube · 65-point");
    expect(within(row).getByText("Match")).toBeInTheDocument();
    expect(screen.getByText("1 package matches what you chose")).toBeInTheDocument();
    expect(screen.getByText(/Nothing is copied until you continue\./)).toBeInTheDocument();
    expect(next()).toHaveTextContent("Import");
    expect(rail("Choose")).not.toHaveAttribute("aria-disabled");
    expect(rail("Get the files")).not.toHaveAttribute("aria-disabled");
    expect(h!.api.system.trashed).toEqual([]);
  });

  it("says which package it is when it is not the one that was chosen", async () => {
    await wizard();
    await toRecognise(LL3);
    expect(screen.getByRole("heading", { level: 1, name: "Check what arrived" })).toBeInTheDocument();
    const row = screen.getAllByRole("row")[1] as HTMLElement;
    expect(within(row).getByText("Not chosen")).toBeInTheDocument();
    expect(row).toHaveTextContent("This is Look Library · LogC3, not Look Library · LogC4, which you chose. Download 'Look Library for LogC4 (3D-LUTs)' from ARRI's page, or go back and choose this one.");
    expect(next()).toHaveAttribute("aria-disabled", "true");
    expect(screen.getByText("Look Library · LogC4 is missing. Go back and add it.")).toBeInTheDocument();
  });

  it("explains a damaged look file in plain words", async () => {
    await wizard();
    await toRecognise(FAILURE_ZIPS.damagedLookFile);
    const row = screen.getAllByRole("row")[1] as HTMLElement;
    expect(within(row).getByText("Can't use")).toBeInTheDocument();
    expect(row).toHaveTextContent("Damaged_Look.alf4 cannot be used. Its checksum does not match, so the file is damaged. Download it again.");
    expect(next()).toHaveAttribute("aria-disabled", "true");
  });

  it("takes ARRI's Look Files version of the chosen library as the library: Match, with the look files counted", async () => {
    await wizard();
    await toRecognise(LOOK_FILE_ZIPS.alf4Library);
    const row = screen.getAllByRole("row")[1] as HTMLElement;
    expect(within(row).getByText("Match")).toBeInTheDocument();
    expect(row).toHaveTextContent("Look Library · LogC4 · log to log");
    expect(row).toHaveTextContent("87 .alf4 · 33-point");
    expect(screen.getByRole("heading", { level: 1, name: "Found it" })).toBeInTheDocument();
    expect(next()).not.toHaveAttribute("aria-disabled");
    // Imported as the library: 87 looks, the ids of the 3D-LUT version.
    await h!.user.click(next());
    await screen.findByRole("heading", { level: 1, name: "Imported" }, { timeout: 10_000 });
    expect(screen.getByText("of 87 files")).toBeInTheDocument();
    // The copy says what is copied: look files, not .cube files.
    expect(screen.getByText("Only the look files, into the app's looks folder")).toBeInTheDocument();
    expect(screen.queryByText(/Only the \.cube files/)).toBeNull();
    expect((await h!.api.looksList()).filter((l) => l.group.kind === "arri")).toHaveLength(87);
  });

  it("reads a person's own look files without a question: Ready, with the input they say", async () => {
    await wizard();
    await toRecognise(FAILURE_ZIPS.ownLookFolder);
    const row = screen.getAllByRole("row")[1] as HTMLElement;
    expect(within(row).getByText("Ready")).toBeInTheDocument();
    expect(row).toHaveTextContent("ARRI Look Files · LogC4");
    expect(row).toHaveTextContent("3 .alf4 · 33-point");
    expect(within(row).queryByRole("radiogroup")).toBeNull();
    expect(row).toHaveTextContent("Counts as Look Library · LogC4. Imported as your own looks, with the input read from each file.");
    expect(screen.getByText("1 package matches what you chose")).toBeInTheDocument();
    expect(screen.queryByText("Say what the other LUTs expect as input.")).toBeNull();
    expect(next()).not.toHaveAttribute("aria-disabled");
    await h!.user.click(next());
    await screen.findByRole("heading", { level: 1, name: "Imported" }, { timeout: 10_000 });
    expect(screen.getByText("Only the look files, into the app's looks folder")).toBeInTheDocument();
    const mine = (await h!.api.looksList()).filter((l) => l.group.kind === "user");
    expect(mine).toHaveLength(3);
    expect(mine.every((l) => l.input === "logc4")).toBe(true);
  });

  it("says an .alf4c's display transforms are not used, and one look file chosen by itself does not stand in for a package", async () => {
    await wizard();
    await toRecognise(FAILURE_ZIPS.ownAlf4c);
    const row = screen.getAllByRole("row")[1] as HTMLElement;
    expect(row).toHaveTextContent("Custom_Look.alf4c: its own display transforms are not used; Yellowmoon's display rendering is.");
    expect(row).not.toHaveTextContent("Counts as");
    // It can be imported, and the chosen package is said not to be here.
    expect(screen.getByText("Look Library · LogC4 isn't here. You can import the rest now and add it later.")).toBeInTheDocument();
    expect(next()).not.toHaveAttribute("aria-disabled");
  });

  it("explains a package with files missing", async () => {
    await wizard();
    await toRecognise(FAILURE_ZIPS.incomplete);
    expect(screen.getAllByRole("row")[1]).toHaveTextContent("should have 87 files, but this zip has 79. The download may be incomplete. Download it again from ARRI's page.");
  });

  it("says a file is missing when it went away after it arrived, and Back and Remove put it right", async () => {
    await wizard();
    await h!.user.click(next());
    drop(LL4);
    await waitFor(() => expect(next()).not.toHaveAttribute("aria-disabled"));
    h!.api.removeSource(LL4);
    await h!.user.click(next());
    const row = (await screen.findAllByRole("row"))[1] as HTMLElement;
    await waitFor(() => expect(within(row).getByText("Missing")).toBeInTheDocument());
    expect(row).toHaveTextContent("That file is not there any more. Download it again, or choose it again.");
    expect(next()).toHaveAttribute("aria-disabled", "true");
    await h!.user.click(within(row).getByRole("button", { name: `Remove ${LL4}` }));
    await h!.user.click(screen.getByRole("button", { name: "← Back" }));
    await screen.findByRole("heading", { level: 1, name: "Download from ARRI" });
    h!.api.addSource(LL4, { source: "zip", size_bytes: 181_000_000, recognition: { kind: "package", package_id: "look_library_logc4", title: "Look Library · LogC4", what: "Look Library · LogC4 · log to log", contents: "87 .cube · 65-point", file_count: 87, note: null } });
    drop(LL4);
    await waitFor(() => expect(next()).not.toHaveAttribute("aria-disabled"));
  });

  it("imports the chosen package even when another file that can't be used was dropped with it", async () => {
    await wizard();
    await toRecognise(LL4, FAILURE_ZIPS.noLuts);
    expect(screen.getAllByRole("row")).toHaveLength(3);
    expect(next()).not.toHaveAttribute("aria-disabled");
    await h!.user.click(screen.getByRole("button", { name: `Remove ${FAILURE_ZIPS.noLuts}` }));
    expect(screen.getAllByRole("row")).toHaveLength(2);
  });

  it("asks what a .cube file of unknown origin expects as input, and waits for the answer", async () => {
    await wizard();
    await toRecognise(LL4, FAILURE_ZIPS.ownCube);
    const row = screen.getAllByRole("row")[2] as HTMLElement;
    expect(within(row).getByText("Question")).toBeInTheDocument();
    expect(within(row).getByText("What does this LUT expect as input?")).toBeInTheDocument();
    const group = within(row).getByRole("radiogroup", { name: `Input of ${FAILURE_ZIPS.ownCube}` });
    expect(within(group).getAllByRole("radio").map((r) => r.textContent)).toEqual(["LogC4", "LogC3", "LogC3 · Rec.709"]);
    expect(next()).toHaveAttribute("aria-disabled", "true");
    expect(screen.getByText("Say what the other LUTs expect as input.")).toBeInTheDocument();
    await h!.user.click(within(group).getByRole("radio", { name: "LogC3 · Rec.709" }));
    expect(within(row).getByText("LogC3 in, Rec.709 out. It replaces our display rendering.")).toBeInTheDocument();
    expect(within(row).getByText("Ready")).toBeInTheDocument();
    expect(next()).not.toHaveAttribute("aria-disabled");
  });

  it("lets Import go ahead when a chosen package did not arrive but something else can be imported", async () => {
    await wizard();
    await h!.user.click(screen.getByRole("checkbox", { name: "Display LUTs · LogC4" }));
    await toRecognise(LL4, FAILURE_ZIPS.noLuts);
    expect(within(screen.getAllByRole("row")[1] as HTMLElement).getByText("Match")).toBeInTheDocument();
    expect(screen.getByText("Display LUTs · LogC4 isn't here. You can import the rest now and add it later.")).toBeInTheDocument();
    expect(screen.queryByText(/is missing\. Go back and add it\./)).toBeNull();
    expect(screen.getByRole("heading", { level: 1, name: "Check what arrived" })).toBeInTheDocument();
    expect(next()).not.toHaveAttribute("aria-disabled");
    await h!.user.click(next());
    expect(await screen.findByRole("heading", { level: 1, name: "Imported" }, { timeout: 10_000 })).toBeInTheDocument();
  });

  it("counts an unknown zip whose input was answered for the chosen package, so nothing is missing", async () => {
    await wizard();
    await toRecognise(FAILURE_ZIPS.otherCamera);
    const row = screen.getAllByRole("row")[1] as HTMLElement;
    expect(within(row).getByText("Question")).toBeInTheDocument();
    // Not answered yet: the question locks Import, and it comes first.
    expect(next()).toHaveAttribute("aria-disabled", "true");
    expect(screen.getByText("Say what the other LUTs expect as input.")).toBeInTheDocument();
    const group = within(row).getByRole("radiogroup", { name: `Input of ${FAILURE_ZIPS.otherCamera}` });
    await h!.user.click(within(group).getByRole("radio", { name: "LogC4" }));
    expect(within(row).getByText("Ready")).toBeInTheDocument();
    expect(within(row).getByText(/Counts as Look Library · LogC4\./)).toBeInTheDocument();
    expect(screen.queryByText(/Look Library · LogC4 is missing/)).toBeNull();
    expect(screen.queryByText(/isn't here/)).toBeNull();
    expect(screen.getByText("1 package matches what you chose")).toBeInTheDocument();
    expect(next()).not.toHaveAttribute("aria-disabled");
  });

  it("lets an unknown zip answered as another input be imported, and says the chosen package is not here", async () => {
    await wizard();
    await toRecognise(FAILURE_ZIPS.otherCamera);
    const row = screen.getAllByRole("row")[1] as HTMLElement;
    const group = within(row).getByRole("radiogroup", { name: `Input of ${FAILURE_ZIPS.otherCamera}` });
    await h!.user.click(within(group).getByRole("radio", { name: "LogC3" }));
    expect(screen.getByText("Look Library · LogC4 isn't here. You can import the rest now and add it later.")).toBeInTheDocument();
    expect(next()).not.toHaveAttribute("aria-disabled");
  });

  it("goes back to an earlier step from the rail and returns with the files still there", async () => {
    await wizard();
    await toRecognise(LL4);
    await h!.user.click(rail("Choose"));
    expect(await step()).toBe("Import · step 1 of 5");
    expect(rail("Recognise")).toHaveAttribute("aria-disabled", "true");
    await h!.user.click(next());
    expect(screen.getByRole("list", { name: "Files that arrived" })).toHaveTextContent(LL4);
    await h!.user.click(next());
    expect(await screen.findByRole("table", { name: "What arrived" })).toBeInTheDocument();
  });
});

describe("steps 4 and 5 · Import and Finish", () => {
  it("copies the files, counts them, and unlocks Next when they are done", async () => {
    await wizard({ looksImported: false, importHoldAt: 40 });
    await toRecognise(LL4);
    await h!.user.click(next());
    await screen.findByRole("heading", { level: 1, name: "Importing" });
    await waitFor(() => expect(screen.getByText("of 87 files")).toBeInTheDocument());
    await waitFor(() => expect(screen.getByRole("progressbar", { name: "Importing looks" })).toHaveAttribute("aria-valuenow", "46"));
    expect(screen.getByText("40")).toBeInTheDocument();
    expect(screen.getByText("This takes a few seconds")).toBeInTheDocument();
    expect(next()).toHaveAttribute("aria-disabled", "true");
    // Nothing can be left or stepped back from while files are copied.
    expect(screen.getByRole("button", { name: "Cancel" })).toHaveAttribute("aria-disabled", "true");
    expect(screen.getByRole("button", { name: "← Back" })).toHaveAttribute("aria-disabled", "true");
    await h!.user.click(rail("Recognise"));
    expect(await step()).toBe("Import · step 4 of 5");
    for (const fact of ["Copied", "Grouped", "Recorded per file"]) expect(screen.getByText(fact)).toBeInTheDocument();
    act(() => h!.api.flush());
    await screen.findByRole("heading", { level: 1, name: "Imported" });
    expect(screen.getByText("87")).toBeInTheDocument();
    expect(screen.getByText("All files copied and checked")).toBeInTheDocument();
    expect(next()).not.toHaveAttribute("aria-disabled");
    expect((await h!.api.looksList()).filter((l) => l.group.kind === "arri")).toHaveLength(87);
  });

  it("finishes: the looks on the current photo, a default, and the zip to the Trash", async () => {
    h = renderApp({ looksImported: false, current: 3 });
    await openPhotos(h);
    await h.user.click(screen.getByRole("button", { name: "Import ARRI looks →" }));
    await screen.findByRole("heading", { level: 1, name: "Choose what to import" });
    await toRecognise(LL4);
    await h.user.click(next());
    await screen.findByRole("heading", { level: 1, name: "Imported" }, { timeout: 5000 });
    await h.user.click(next());
    await screen.findByRole("heading", { level: 1, name: "87 looks ready" });
    expect(screen.getByText("Here they are on your current photo. Pick a default and you are done.")).toBeInTheDocument();
    const tiles = within(document.querySelector(".ym-wizard__tiles") as HTMLElement);
    for (const name of ["Film warm", "Film cool", "Bleach bypass", "Teal and orange", "Soft print", "Night blue", "Monochrome"]) expect(tiles.getByText(name)).toBeInTheDocument();
    expect(screen.getByText("+80")).toBeInTheDocument();
    expect(screen.getByText("And 80 more")).toBeInTheDocument();
    await waitFor(() => expect(document.querySelectorAll(".ym-wizard__tile-picture img")).toHaveLength(7));
    // Default look: Neutral, Signature and the first imported look; Neutral is current.
    const defaultGroup = screen.getByRole("radiogroup", { name: "Default look" });
    expect(within(defaultGroup).getAllByRole("radio").map((r) => r.textContent)).toEqual(["Neutral", "Signature", "Film warm"]);
    expect(within(defaultGroup).getByRole("radio", { name: "Neutral" })).toHaveAttribute("aria-checked", "true");
    expect(screen.queryByRole("radiogroup", { name: "Display rendering" })).toBeNull();
    expect(within(screen.getByRole("radiogroup", { name: "Downloaded zip" })).getByRole("radio", { name: "Move to Trash" })).toHaveAttribute("aria-checked", "true");
    expect(screen.getByText("You can change the default look any time")).toBeInTheDocument();
    await h.user.click(within(defaultGroup).getByRole("radio", { name: "Film warm" }));
    expect(next()).toHaveTextContent("Done");
    await h.user.click(next());
    // Back in the photos, the default look is saved and the zip is in the Trash.
    expect(await screen.findByRole("tab", { name: "Look" })).toBeInTheDocument();
    expect((await h.api.preferencesGet()).default_look).toBe("arri:3000:film-warm");
    expect(h.api.system.trashed).toEqual([LL4]);
  }, 30_000);

  it("asks for another picture when one of the Finish tiles does not load (the shell dropped it)", async () => {
    h = renderApp({ looksImported: false, current: 3 });
    await openPhotos(h);
    await h.user.click(screen.getByRole("button", { name: "Import ARRI looks →" }));
    await screen.findByRole("heading", { level: 1, name: "Choose what to import" });
    await toRecognise(LL4);
    await h.user.click(next());
    await screen.findByRole("heading", { level: 1, name: "Imported" }, { timeout: 5000 });
    await h.user.click(next());
    await screen.findByRole("heading", { level: 1, name: "87 looks ready" });
    const pictures = (): HTMLImageElement[] => [...document.querySelectorAll<HTMLImageElement>(".ym-wizard__tile-picture img")];
    await waitFor(() => expect(pictures()).toHaveLength(7));
    const ask = vi.spyOn(h.api, "lookThumbnail");
    const first = pictures()[0] as HTMLImageElement;
    const before = first.getAttribute("src");
    fireEvent.error(first);
    await waitFor(() => expect(ask).toHaveBeenCalledTimes(1));
    await waitFor(() => {
      const now = pictures()[0]?.getAttribute("src");
      expect(now).toMatch(/^(data:image\/png|blob:)/);
      expect(now).not.toBe(before);
    });
    // The other tiles were left alone.
    expect(ask).toHaveBeenCalledTimes(1);
  }, 30_000);

  it("keeps the zip when asked to", async () => {
    await wizard({ looksImported: false });
    await toRecognise(LL4);
    await h!.user.click(next());
    await screen.findByRole("heading", { level: 1, name: "Imported" }, { timeout: 5000 });
    await h!.user.click(next());
    await screen.findByRole("heading", { level: 1, name: "87 looks ready" });
    await h!.user.click(within(screen.getByRole("radiogroup", { name: "Downloaded zip" })).getByRole("radio", { name: "Keep it" }));
    await h!.user.click(next());
    expect(await screen.findByRole("heading", { level: 1, name: "Drop Sony ARW files." })).toBeInTheDocument();
    expect(h!.api.system.trashed).toEqual([]);
    expect(screen.getByText("87 looks imported")).toBeInTheDocument();
  }, 30_000);

  it("offers ARRI's display rendering when display LUTs were imported, and saves the choice", async () => {
    await wizard({ looksImported: false });
    await h!.user.click(screen.getByRole("checkbox", { name: "Display LUTs · LogC4" }));
    await toRecognise(LL4, DL4);
    expect(screen.getAllByRole("row")).toHaveLength(3);
    await h!.user.click(next());
    await screen.findByRole("heading", { level: 1, name: "Imported" }, { timeout: 8000 });
    await h!.user.click(next());
    await screen.findByRole("heading", { level: 1, name: "87 looks ready" });
    const display = screen.getByRole("radiogroup", { name: "Display rendering" });
    expect(within(display).getByRole("radio", { name: "Ours · ACES 2" })).toHaveAttribute("aria-checked", "true");
    await h!.user.click(within(display).getByRole("radio", { name: "ARRI's" }));
    await h!.user.click(next());
    await screen.findByRole("heading", { level: 1, name: "Drop Sony ARW files." });
    expect((await h!.api.preferencesGet()).display_rendering).toEqual({ kind: "arri_display", id: "arri-display:logc4:rec709-g24" });
    expect(h!.api.system.trashed).toHaveLength(2);
  }, 40_000);

  it("imports a loose .cube with the answer that was given", async () => {
    await wizard({ looksImported: false });
    await toRecognise(LL4, FAILURE_ZIPS.ownCube);
    const group = screen.getByRole("radiogroup", { name: `Input of ${FAILURE_ZIPS.ownCube}` });
    await h!.user.click(within(group).getByRole("radio", { name: "LogC3" }));
    await h!.user.click(next());
    await screen.findByRole("heading", { level: 1, name: "Imported" }, { timeout: 8000 });
    const mine = (await h!.api.looksList()).find((l) => l.id === "user:warm-film");
    expect(mine).toMatchObject({ name: "Warm Film", input: "logc3_ei800", group: { kind: "user" } });
    await h!.user.click(next());
    await screen.findByRole("heading", { level: 1, name: "88 looks ready" });
  }, 40_000);

  it("says what went wrong when the import fails, and tries again", async () => {
    await wizard({ looksImported: false });
    await toRecognise(LL4);
    h!.api.failNext("looksImport", { scope: "looks", message: "The looks folder can't be written to. Check its permissions." });
    await h!.user.click(next());
    await screen.findByRole("heading", { level: 1, name: "Not imported" });
    expect(screen.getByRole("alert")).toHaveTextContent("The looks folder can't be written to. Check its permissions.");
    expect(next()).toHaveAttribute("aria-disabled", "true");
    // The window is free again.
    expect(screen.getByRole("button", { name: "Cancel" })).not.toHaveAttribute("aria-disabled");
    await h!.user.click(screen.getByRole("button", { name: "Try again" }));
    await screen.findByRole("heading", { level: 1, name: "Imported" }, { timeout: 5000 });
  }, 30_000);

  it("importing the same package twice adds nothing", async () => {
    await wizard({ looksImported: true });
    await toRecognise(LL4);
    await h!.user.click(next());
    await screen.findByRole("heading", { level: 1, name: "Imported" }, { timeout: 5000 });
    expect((await h!.api.looksList()).filter((l) => l.group.kind === "arri")).toHaveLength(87);
  }, 30_000);
});
