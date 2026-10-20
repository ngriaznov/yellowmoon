import { act, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { MockOptions } from "../api.mock";
import { PACKAGE_ZIPS } from "../api.mock";
import { openPhotos, renderApp } from "../test/harness";
import { capturingPointer } from "../test/pointer";
import { mockRect } from "../test/raf";
import type { Harness } from "../test/harness";

/** A manual contrast or saturation (`docs/19` section 5: Auto or a value). */
const manual = (value: number) => ({ kind: "manual", value }) as const;

let h: Harness | undefined;
afterEach(() => {
  h?.unmount();
  h = undefined;
  vi.unstubAllGlobals();
});

/** The mockup's state: DSC04204 current, DSC04203 to DSC04205 picked. */
const MOCKUP = { current: 3, picked: [2, 3, 4] };

async function setup(options: MockOptions = MOCKUP): Promise<Harness> {
  h = renderApp(options);
  await openPhotos(h);
  return h;
}

/** Opens the Export panel with ⌘E and waits for the core's estimate. */
async function openExport(options: MockOptions = MOCKUP): Promise<HTMLElement> {
  await setup(options);
  await h!.user.keyboard("{Meta>}e{/Meta}");
  const panel = await screen.findByRole("complementary", { name: "Export" });
  await waitFor(() => expect(within(panel).getByRole("button", { name: /^Export/ })).toHaveTextContent("≈"));
  return panel;
}

const radio = (group: string, name: string): HTMLElement => within(screen.getByRole("radiogroup", { name: group })).getByRole("radio", { name });
const checked = (group: string): string | undefined =>
  within(screen.getByRole("radiogroup", { name: group }))
    .getAllByRole("radio")
    .find((r) => r.getAttribute("aria-checked") === "true")?.textContent ?? undefined;
const exportButton = (): HTMLElement => within(screen.getByRole("complementary", { name: "Export" })).getByRole("button", { name: /^Export \d+ photos?/ });

describe("the options (design/mockups/export.html)", () => {
  it("opens with the mockup's title, blurb, cards and footer", async () => {
    const panel = await openExport();
    expect(within(panel).getByRole("heading", { level: 2, name: "Export" })).toBeInTheDocument();
    expect(within(panel).getByText("3 photos from Lisbon. Nothing is saved besides the exported files.")).toBeInTheDocument();
    for (const card of ["Photos", "Format", "Colour space", "Size", "Save to", "Same thing from the command line"]) {
      expect(within(panel).getByRole("region", { name: card })).toBeInTheDocument();
    }
    expect(checked("Photos")).toBe("Picked · 3");
    expect(within(screen.getByRole("radiogroup", { name: "Photos" })).getAllByRole("radio").map((r) => r.textContent)).toEqual(["This photo", "Picked · 3", "All · 12"]);
    expect(checked("Format")).toBe("JPEG");
    expect(checked("Colour space")).toBe("sRGB");
    expect(checked("Size")).toBe("Full · 8640 px");
    expect(checked("If the file exists")).toBe("Keep both");
    expect(within(panel).getByText("Best for sharing and the web.")).toBeInTheDocument();
    expect(exportButton()).toHaveTextContent("Export 3 photos");
    expect(exportButton()).toHaveTextContent("≈ 75 MB · ⌘↩");
  });

  it("starts from This photo when nothing is picked", async () => {
    await openExport({ current: 3, picked: [] });
    expect(checked("Photos")).toBe("This photo");
    expect(screen.getByText("1 photo from Lisbon. Nothing is saved besides the exported files.")).toBeInTheDocument();
    expect(exportButton()).toHaveTextContent("Export 1 photo");
  });

  it("changes the count, the names and the estimate with the scope", async () => {
    await openExport();
    await h!.user.click(radio("Photos", "All · 12"));
    await waitFor(() => expect(exportButton()).toHaveTextContent("Export 12 photos"));
    await waitFor(() => expect(exportButton()).toHaveTextContent("≈ 299 MB"));
    expect(screen.getByText("12 photos from Lisbon. Nothing is saved besides the exported files.")).toBeInTheDocument();
    await h!.user.click(radio("Photos", "This photo"));
    await waitFor(() => expect(screen.getByText("DSC04204_neutral.jpg")).toBeInTheDocument());
  });

  it("dims the photos that won't be exported in the filmstrip, and counts the ones that will", async () => {
    await openExport();
    const strip = screen.getByRole("navigation", { name: "Photos to export" });
    const flags = within(strip).getAllByRole("img").map((el) => el.getAttribute("data-in-export"));
    expect(flags.filter((f) => f === "true")).toHaveLength(3);
    expect(flags.filter((f) => f === "false")).toHaveLength(9);
    expect(within(strip).getByText("3 to export")).toBeInTheDocument();
    await h!.user.click(radio("Photos", "All · 12"));
    await waitFor(() => expect(within(strip).getAllByRole("img").filter((el) => el.getAttribute("data-in-export") === "true")).toHaveLength(12));
  });

  it("shows Quality for JPEG, Bit depth and Compression for TIFF, Bit depth for PNG", async () => {
    await openExport();
    const slider = screen.getByRole("slider", { name: "Quality" });
    expect(slider).toHaveAttribute("aria-valuemin", "50");
    expect(slider).toHaveAttribute("aria-valuemax", "100");
    expect(slider).toHaveAttribute("aria-valuenow", "92");
    await h!.user.click(radio("Format", "TIFF"));
    expect(screen.queryByRole("slider", { name: "Quality" })).toBeNull();
    expect(checked("Bit depth")).toBe("16-bit");
    expect(checked("Compression")).toBe("LZW");
    await h!.user.click(radio("Compression", "ZIP"));
    expect(checked("Compression")).toBe("ZIP");
    await h!.user.click(radio("Bit depth", "8-bit"));
    await waitFor(() => expect(exportButton()).toHaveTextContent("≈ 304 MB")); // 149 MB × 0.68 × 3
    await h!.user.click(radio("Format", "PNG"));
    expect(checked("Bit depth")).toBe("8-bit"); // TIFF and PNG share the preference
    expect(screen.queryByRole("radiogroup", { name: "Compression" })).toBeNull();
  });

  it("sets JPEG quality from the box, and the estimate follows", async () => {
    await openExport();
    const box = screen.getByRole("textbox", { name: "Quality value" });
    await h!.user.click(box);
    await h!.user.clear(box);
    await h!.user.type(box, "60{Enter}");
    await waitFor(() => expect(screen.getByRole("slider", { name: "Quality" })).toHaveAttribute("aria-valuenow", "60"));
    await waitFor(() => expect(exportButton()).toHaveTextContent("≈ 32 MB")); // (6 + 10 × 0.45) × 3 = 31.5
  });

  it("explains DNG, with the Adobe notice, and drops Colour space and Size", async () => {
    await openExport();
    await h!.user.click(radio("Format", "DNG"));
    expect(screen.getByText("Linear DNG with your look baked in. Opens in Lightroom close to this render, with room to adjust exposure, highlights and white balance.")).toBeInTheDocument();
    // What other raw editors show: the sentence of `ym render --help` (the Rust side checks the wording).
    expect(screen.getByText("Apps other than Adobe's show the stored linear image darker until their own exposure handling is applied.")).toBeInTheDocument();
    expect(screen.getByText(/This product includes DNG technology under license by Adobe\./)).toBeInTheDocument();
    expect(screen.getByText(/White balance changes made in another app are approximate/)).toBeInTheDocument();
    expect(screen.queryByRole("region", { name: "Colour space" })).toBeNull();
    expect(screen.queryByRole("region", { name: "Size" })).toBeNull();
    await waitFor(() => expect(exportButton()).toHaveTextContent("≈ 900 MB"));
    expect(screen.getByText("DSC04203_neutral.dng")).toBeInTheDocument();
  });

  it("says what each colour space is for", async () => {
    await openExport();
    await h!.user.click(radio("Colour space", "Display P3"));
    expect(screen.getByText("Wider colours. Use it for recent Macs, iPhones and iPads.")).toBeInTheDocument();
    expect(screen.queryByText("Best for sharing and the web.")).toBeNull();
    await h!.user.click(radio("Colour space", "Rec.2020"));
    expect(screen.getByText("Very wide colours, for video and HDR work.")).toBeInTheDocument();
  });

  it("offers Long edge as a slider from 1000 px to the full size in 20 px steps", async () => {
    await openExport();
    expect(screen.queryByRole("slider", { name: "Long edge" })).toBeNull();
    await h!.user.click(radio("Size", "Long edge"));
    const slider = screen.getByRole("slider", { name: "Long edge" });
    expect(slider).toHaveAttribute("aria-valuemin", "1000");
    expect(slider).toHaveAttribute("aria-valuemax", "8640");
    expect(slider).toHaveAttribute("aria-valuenow", "3000");
    expect(screen.getByRole("textbox", { name: "Long edge value" })).toHaveValue("3000 px");
    await waitFor(() => expect(exportButton()).toHaveTextContent("≈ 9 MB")); // 24.9 × (3000/8640)² × 3
    // One arrow key is one step of 20 px (Radix's slider has no step attribute to read).
    slider.focus();
    await h!.user.keyboard("{ArrowRight}");
    await waitFor(() => expect(screen.getByRole("textbox", { name: "Long edge value" })).toHaveValue("3020 px"));
    await h!.user.keyboard("{ArrowLeft}");
    await waitFor(() => expect(screen.getByRole("textbox", { name: "Long edge value" })).toHaveValue("3000 px"));
    // A press on the track at 87 of its 200 px lands on 4320 px (1000 + 87/200 × 7640 = 4323, rounded to the 20 px grid).
    const root = slider.closest('[data-slot="slider"]') as HTMLElement;
    mockRect(root, { left: 0, width: 200, height: 38 });
    const pointer = capturingPointer(root);
    pointer.down(86.9);
    pointer.up();
    await waitFor(() => expect(exportButton()).toHaveTextContent("≈ 19 MB"));
    expect(screen.getByRole("slider", { name: "Long edge" })).toHaveAttribute("aria-valuenow", "4320");
    // Back to Full and again: the last long edge comes back.
    await h!.user.click(radio("Size", "Full · 8640 px"));
    await h!.user.click(radio("Size", "Long edge"));
    expect(screen.getByRole("slider", { name: "Long edge" })).toHaveAttribute("aria-valuenow", "4320");
  });

  it("keeps Long edge snapped to the grid when the value is typed", async () => {
    await openExport();
    await h!.user.click(radio("Size", "Long edge"));
    const box = screen.getByRole("textbox", { name: "Long edge value" });
    await h!.user.click(box);
    await h!.user.clear(box);
    await h!.user.type(box, "2500{Enter}");
    expect(screen.getByRole("slider", { name: "Long edge" })).toHaveAttribute("aria-valuenow", "2500");
    await h!.user.click(box);
    await h!.user.clear(box);
    await h!.user.type(box, "99999{Enter}");
    expect(screen.getByRole("slider", { name: "Long edge" })).toHaveAttribute("aria-valuenow", "8640");
  });
});

describe("Save to, the file name and what to do when it exists", () => {
  it("shows the session's folder, and Choose… takes the one the system chooser returns", async () => {
    await openExport();
    const card = screen.getByRole("region", { name: "Save to" });
    expect(within(card).getByText("~/Pictures/Lisbon export")).toBeInTheDocument();
    expect(within(card).queryByRole("button", { name: "Use default" })).toBeNull();
    h!.api.queueChoice("folder", "~/Exports/Portugal");
    await h!.user.click(within(card).getByRole("button", { name: "Choose…" }));
    expect(await within(card).findByText("~/Exports/Portugal")).toBeInTheDocument();
    await waitFor(() => expect(screen.getByRole("region", { name: "Same thing from the command line" })).toHaveTextContent('--out "~/Exports/Portugal"'));
    await h!.user.click(within(card).getByRole("button", { name: "Use default" }));
    expect(await within(card).findByText("~/Pictures/Lisbon export")).toBeInTheDocument();
  });

  it("leaves things as they are when the chooser is cancelled", async () => {
    await openExport();
    h!.api.queueChoice("folder", null);
    await h!.user.click(within(screen.getByRole("region", { name: "Save to" })).getByRole("button", { name: "Choose…" }));
    expect(screen.getByText("~/Pictures/Lisbon export")).toBeInTheDocument();
  });

  it("shows the name the template gives the first photo", async () => {
    await openExport();
    expect(screen.getByRole("textbox", { name: "File name template" })).toHaveValue("{name}_{look}");
    expect(screen.getByText("DSC04203_neutral.jpg")).toBeInTheDocument();
    await h!.user.click(radio("Format", "TIFF"));
    expect(screen.getByText("DSC04203_neutral.tif")).toBeInTheDocument();
  });

  it("previews a typed template, and remembers it", async () => {
    await openExport();
    const box = screen.getByRole("textbox", { name: "File name template" });
    await h!.user.clear(box);
    await h!.user.type(box, "{{date}_{{name}_{{seq}");
    expect(await screen.findByText("2026-10-04_DSC04203_001.jpg")).toBeInTheDocument();
    await h!.user.tab();
    await waitFor(async () => expect((await h!.api.preferencesGet()).export_defaults.name_template).toBe("{date}_{name}_{seq}"));
  });

  it("says what is wrong with a template that can't be used, and holds the export back", async () => {
    await openExport();
    const box = screen.getByRole("textbox", { name: "File name template" });
    await h!.user.clear(box);
    await h!.user.type(box, "{{name}_{{colour}");
    expect(await screen.findByRole("alert")).toHaveTextContent("{colour} is not a token. Use {name}, {look}, {date}, {seq} or {camera}.");
    expect(box).toHaveAttribute("aria-invalid", "true");
    expect(exportButton()).toHaveAttribute("aria-disabled", "true");
    expect(exportButton()).toHaveAccessibleDescription("{colour} is not a token. Use {name}, {look}, {date}, {seq} or {camera}.");
    // A template that can't be used is never remembered.
    await h!.user.tab();
    expect((await h!.api.preferencesGet()).export_defaults.name_template).toBe("{name}_{look}");
    await h!.user.clear(box);
    await h!.user.type(box, "{{name}");
    await waitFor(() => expect(exportButton()).not.toHaveAttribute("aria-disabled"));
  });

  it("warns when every photo would get the same name", async () => {
    await openExport();
    const box = screen.getByRole("textbox", { name: "File name template" });
    await h!.user.clear(box);
    await h!.user.type(box, "{{look}");
    expect(await screen.findByText(/Every photo would get the same name\. Add \{name\} or \{seq\}\./)).toBeInTheDocument();
    expect(exportButton()).not.toHaveAttribute("aria-disabled");
  });

  it("offers Skip, Replace and Keep both, and passes the choice on", async () => {
    await openExport();
    await h!.user.click(radio("If the file exists", "Skip"));
    await waitFor(() => expect(screen.getByRole("region", { name: "Same thing from the command line" })).toHaveTextContent("--skip-existing"));
    await h!.user.click(radio("If the file exists", "Replace"));
    await waitFor(() => expect(screen.getByRole("region", { name: "Same thing from the command line" })).toHaveTextContent("--overwrite"));
  });
});

describe("the command line", () => {
  it("shows the equivalent command, as the mockup does", async () => {
    await openExport();
    const card = screen.getByRole("region", { name: "Same thing from the command line" });
    await waitFor(() => expect(within(card).getByLabelText("Command line")).toHaveTextContent("ym render DSC04203.ARW DSC04204.ARW DSC04205.ARW"));
    const text = within(card).getByLabelText("Command line").textContent ?? "";
    for (const flag of ["--look neutral", "--format jpeg", "--quality 92", "--colour-space srgb", '--out "~/Pictures/Lisbon export"']) expect(text).toContain(flag);
  });

  it("copies the command and says Copied", async () => {
    await openExport();
    const card = screen.getByRole("region", { name: "Same thing from the command line" });
    await waitFor(() => expect(within(card).getByRole("button", { name: "Copy" })).not.toHaveAttribute("aria-disabled"));
    await h!.user.click(within(card).getByRole("button", { name: "Copy" }));
    expect(await navigator.clipboard.readText()).toContain("ym render DSC04203.ARW DSC04204.ARW DSC04205.ARW");
    expect(await within(card).findByRole("button", { name: "Copied" })).toBeInTheDocument();
  });

  it("says nothing about adjustments when there are none, and how many photos have them when there are", async () => {
    await openExport();
    expect(screen.queryByText(/adjustments this command doesn't include/)).toBeNull();
    h!.unmount();
    await setup();
    await h!.store.update({ contrast: manual(25) });
    await h!.user.keyboard("{Meta>}e{/Meta}");
    expect(await screen.findByText("1 photo has adjustments this command doesn't include.")).toBeInTheDocument();
    await h!.user.click(radio("Photos", "All · 12"));
    await h!.api.updateSettings(1, { saturation: manual(10) });
    await h!.user.click(radio("Photos", "This photo"));
    await h!.user.click(radio("Photos", "All · 12"));
    expect(await screen.findByText("2 photos have adjustments this command doesn't include.")).toBeInTheDocument();
  });
});

describe("DNG and a look with its own display rendering", () => {
  async function withRec709Look(): Promise<HTMLElement> {
    await setup(MOCKUP);
    await act(async () => {
      const importing = h!.api.looksImport(`~/Downloads/${PACKAGE_ZIPS.look_library_logc3_to_rec709}`);
      await importing;
    });
    await h!.store.refreshLooks();
    await h!.store.chooseLook("arri:3000:film-warm-logc3-rec709");
    await h!.user.keyboard("{Meta>}e{/Meta}");
    return screen.findByRole("complementary", { name: "Export" });
  }

  const OWN_DISPLAY = "With a look that includes its own display rendering, the DNG is close to, not the same as, the picture you see.";

  it("keeps DNG available and says that the DNG is close to the picture, not the same", async () => {
    await withRec709Look();
    await h!.user.click(radio("Format", "DNG"));
    expect(checked("Format")).toBe("DNG");
    expect(await screen.findByText(OWN_DISPLAY)).toBeInTheDocument();
    expect(screen.getByText("Linear DNG with your look baked in.", { exact: false })).toBeInTheDocument();
    expect(screen.queryByRole("alert")).toBeNull();
    await waitFor(() => expect(exportButton()).not.toHaveAttribute("aria-disabled", "true"));
  });

  it("says nothing about it for a look of ours, and nothing outside the DNG note", async () => {
    await setup(MOCKUP);
    await h!.store.setExportOptions({ format: "dng" });
    await h!.user.keyboard("{Meta>}e{/Meta}");
    await screen.findByRole("complementary", { name: "Export" });
    expect(screen.queryByText(OWN_DISPLAY)).toBeNull();
    await h!.store.setExportOptions({ format: "jpeg" });
    expect(screen.queryByText(OWN_DISPLAY)).toBeNull();
  });
});

describe("remembering the options", () => {
  it("saves changes as the defaults, so the next export starts from them", async () => {
    await openExport();
    await h!.user.click(radio("Format", "PNG"));
    await h!.user.click(radio("Colour space", "Display P3"));
    await waitFor(async () => expect((await h!.api.preferencesGet()).export_defaults).toMatchObject({ format: "png", color_space: "display_p3" }));
    // Close the session and open the photos again: the panel starts from what was saved.
    await h!.store.closeSession();
    await h!.store.openDialog("files");
    h!.api.flush();
    await waitFor(() => expect(h!.store.getState().session?.photos.every((p) => p.status.kind === "ready")).toBe(true));
    expect(h!.store.getState().exportOptions).toMatchObject({ format: "png", color_space: "display_p3" });
  });

  it("does not save while a slider is dragged, only when it is released", async () => {
    await openExport();
    const save = vi.spyOn(h!.api, "preferencesSet");
    h!.store.setExportOptions({ jpeg_quality: 70 }, false);
    h!.store.setExportOptions({ jpeg_quality: 65 }, false);
    expect(save).not.toHaveBeenCalled();
    h!.store.setExportOptions({ jpeg_quality: 64 }, true);
    expect(save).toHaveBeenCalledTimes(1);
  });
});

describe("running", () => {
  async function start(options: MockOptions = MOCKUP): Promise<HTMLElement> {
    const panel = await openExport(options);
    await h!.user.click(exportButton());
    await within(panel).findByRole("heading", { level: 2, name: "Exporting" });
    return panel;
  }

  it("shows the queue with every file, the note and Cancel", async () => {
    const panel = await start();
    expect(within(panel).getByText("0 of 3 done.")).toBeInTheDocument();
    const list = within(panel).getByRole("list", { name: "Export queue" });
    expect(within(list).getAllByRole("listitem")).toHaveLength(3);
    expect(within(list).getByText("DSC04203_neutral.jpg")).toBeInTheDocument();
    expect(within(panel).getByText("Originals are never changed. You can keep working while this runs; an update waits until the export is finished.")).toBeInTheDocument();
    expect(within(panel).getByRole("button", { name: "Cancel" })).toBeInTheDocument();
    expect(within(panel).getByRole("progressbar", { name: "Exporting" })).toBeInTheDocument();
    expect(within(panel).queryByRole("region", { name: "Format" })).toBeNull();
  });

  it("goes through Waiting, a percentage and Done", async () => {
    h = renderApp({ ...MOCKUP, speed: 1 });
    await openPhotos(h);
    await h.user.keyboard("{Meta>}e{/Meta}");
    const panel = await screen.findByRole("complementary", { name: "Export" });
    await waitFor(() => expect(within(panel).getByRole("button", { name: /^Export/ })).toHaveTextContent("≈"));
    await h.user.click(exportButton());
    await within(panel).findByText("Rendering at full quality");
    await waitFor(() => expect(within(panel).getByText("1 of 3 done.")).toBeInTheDocument(), { timeout: 4000 });
    const list = within(panel).getByRole("list", { name: "Export queue" });
    expect(within(list).getByText("Done")).toBeInTheDocument();
    expect(within(list).getByText(/^\d+%$/)).toBeInTheDocument();
    expect(within(list).getByText("Waiting")).toBeInTheDocument();
    await h.user.click(within(panel).getByRole("button", { name: "Cancel" }));
  }, 20_000);

  it("finishes with Exported, the folder, Show in Finder and Done", async () => {
    const panel = await start();
    act(() => h!.api.flush());
    await within(panel).findByRole("heading", { level: 2, name: "Exported" });
    expect(within(panel).getByText("3 photos saved to Lisbon export.")).toBeInTheDocument();
    expect(within(panel).getByText("Originals are never changed. Open the folder or go back to your photos.")).toBeInTheDocument();
    expect(within(panel).getAllByText("Done")).toHaveLength(3 + 1); // three rows and the button
    await h!.user.click(within(panel).getByRole("button", { name: "Show in Finder" }));
    expect(h!.api.system.revealed).toEqual(["~/Pictures/Lisbon export"]);
    await h!.user.click(within(panel).getByRole("button", { name: "Done" }));
    expect(await screen.findByRole("complementary", { name: "Tools" })).toBeInTheDocument();
    // The next visit starts from the options again.
    await h!.user.keyboard("{Meta>}e{/Meta}");
    expect(await screen.findByRole("region", { name: "Format" })).toBeInTheDocument();
  });

  it("says in plain words which file failed and why, and which were skipped", async () => {
    h = renderApp({
      ...MOCKUP,
      exportFailures: { DSC04204: "The disk is full. Free some space, then export again." },
      existingExports: ["DSC04205_neutral.jpg"],
    });
    await openPhotos(h);
    await h.user.keyboard("{Meta>}e{/Meta}");
    await h.user.click(await screen.findByRole("radio", { name: "Skip" }));
    const panel = screen.getByRole("complementary", { name: "Export" });
    await waitFor(() => expect(within(panel).getByRole("button", { name: /^Export/ })).toHaveTextContent("≈"));
    await h.user.click(exportButton());
    await within(panel).findByRole("heading", { level: 2, name: "Exporting" });
    act(() => h!.api.flush());
    await within(panel).findByRole("heading", { level: 2, name: "Exported" });
    expect(within(panel).getByText("The disk is full. Free some space, then export again.")).toBeInTheDocument();
    expect(within(panel).getByText("Failed")).toBeInTheDocument();
    expect(within(panel).getByText("Skipped")).toBeInTheDocument();
    expect(within(panel).getByText("1 photo saved to Lisbon export. 1 photo could not be saved. 1 photo skipped because the file already exists.")).toBeInTheDocument();
  });

  it("Cancel stops it and goes back to the options, saying what was saved", async () => {
    const panel = await start();
    await h!.user.click(within(panel).getByRole("button", { name: "Cancel" }));
    expect(await screen.findByRole("region", { name: "Format" })).toBeInTheDocument();
    expect(screen.getByText("Export stopped. Nothing was saved.")).toBeInTheDocument();
    expect(h!.store.getState().exportRun).toBeNull();
  });

  it("keeps running while the panel is closed, and the header says where it is", async () => {
    h = renderApp({ ...MOCKUP, speed: 1 });
    await openPhotos(h);
    await h.user.keyboard("{Meta>}e{/Meta}");
    const panel = await screen.findByRole("complementary", { name: "Export" });
    await waitFor(() => expect(within(panel).getByRole("button", { name: /^Export/ })).toHaveTextContent("≈"));
    await h.user.click(exportButton());
    await within(panel).findByRole("heading", { level: 2, name: "Exporting" });
    // Browse: back to the photos, another photo, a tool tab.
    await h.user.click(screen.getByRole("button", { name: "← Back to photos" }));
    expect(await screen.findByRole("complementary", { name: "Tools" })).toBeInTheDocument();
    await h.user.keyboard("{ArrowRight}");
    await h.user.click(screen.getByRole("tab", { name: "Light" }));
    const cell = screen.getByRole("button", { name: /^Exporting \d of 3/ });
    expect(cell).toBeInTheDocument();
    await waitFor(() => expect(screen.getByRole("button", { name: /^Exporting [12] of 3/ })).toBeInTheDocument(), { timeout: 4000 });
    // The cell opens the queue again.
    await h.user.click(screen.getByRole("button", { name: /^Exporting/ }));
    expect(await screen.findByRole("heading", { level: 2, name: "Exporting" })).toBeInTheDocument();
    act(() => h!.api.flush());
    await screen.findByRole("heading", { level: 2, name: "Exported" });
    // Closed again, a finished export shows as Exported until Done.
    await h.user.click(screen.getByRole("button", { name: "← Back to photos" }));
    expect(screen.getByRole("button", { name: /^Exported/ })).toBeInTheDocument();
  }, 20_000);

  it("shows a progress strip in the header cell while it runs", async () => {
    h = renderApp({ ...MOCKUP, speed: 1 });
    await openPhotos(h);
    await h.user.keyboard("{Meta>}e{/Meta}");
    await waitFor(() => expect(within(screen.getByRole("complementary", { name: "Export" })).getByRole("button", { name: /^Export/ })).toHaveTextContent("≈"));
    await h.user.click(exportButton());
    await h.user.click(await screen.findByRole("button", { name: "← Back to photos" }));
    await waitFor(() => expect(document.querySelector("[data-export-strip] > span")).not.toBeNull());
    await h.store.cancelExport();
  }, 20_000);

  it("holds an update back until the queue is empty", async () => {
    const { updater } = await import("../updater");
    const panel = await start();
    act(() => updater.applyExternal({ phase: "ready", version: "0.1.1" }));
    const pill = await screen.findByRole("button", { name: /Update ready/ });
    expect(pill).toHaveAttribute("aria-disabled", "true");
    expect(pill).toHaveAccessibleDescription("Restarts after the export finishes");
    act(() => h!.api.flush());
    await within(panel).findByRole("heading", { level: 2, name: "Exported" });
    await waitFor(() => expect(screen.getByRole("button", { name: /Update ready/ })).not.toHaveAttribute("aria-disabled"));
    act(() => updater.applyExternal({ phase: "idle" }));
  });

  it("starts with ⌘↩ and ignores it while a run is going", async () => {
    await openExport();
    await h!.user.keyboard("{Meta>}{Enter}{/Meta}");
    await screen.findByRole("heading", { level: 2, name: "Exporting" });
    const calls = vi.spyOn(h!.api, "exportStart");
    await h!.user.keyboard("{Meta>}{Enter}{/Meta}");
    expect(calls).not.toHaveBeenCalled();
  });

  it("says when there is nothing to export instead of starting", async () => {
    await openExport({ current: 3, picked: [2, 3] });
    await h!.user.click(radio("Photos", "Picked · 2"));
    // Unpick both: the scope is empty.
    await h!.store.clearPicks();
    await waitFor(() => expect(screen.getByText("No photos are picked.")).toBeInTheDocument());
    expect(within(screen.getByRole("complementary", { name: "Export" })).getByRole("button", { name: /^Export/ })).toHaveAttribute("aria-disabled", "true");
  });

  it("shows an error from the core and stays on the options", async () => {
    await openExport();
    h!.api.failNext("exportStart", { scope: "export", message: "The folder can't be written to. Choose another folder." });
    await h!.user.click(exportButton());
    expect(await screen.findByRole("alert")).toHaveTextContent("The folder can't be written to. Choose another folder.");
    expect(screen.getByRole("region", { name: "Format" })).toBeInTheDocument();
    expect(h!.store.getState().exportRun).toBeNull();
  });
});

describe("closing the session during an export (⌘W)", () => {
  it("asks first, and Keep exporting changes nothing", async () => {
    await openExport();
    await h!.user.click(exportButton());
    await screen.findByRole("heading", { level: 2, name: "Exporting" });
    await h!.user.keyboard("{Meta>}w{/Meta}");
    const dialog = await screen.findByRole("dialog", { name: "An export is running" });
    expect(within(dialog).getByText("Closing the session stops it. The files that are already saved stay in the folder.")).toBeInTheDocument();
    expect(within(dialog).getByRole("button", { name: "Keep exporting" })).toHaveFocus();
    await h!.user.click(within(dialog).getByRole("button", { name: "Keep exporting" }));
    expect(screen.queryByRole("dialog")).toBeNull();
    expect(screen.getByRole("complementary", { name: "Export" })).toBeInTheDocument();
    expect(h!.store.getState().session).not.toBeNull();
  });

  it("stops the export and closes when asked to", async () => {
    await openExport();
    await h!.user.click(exportButton());
    await screen.findByRole("heading", { level: 2, name: "Exporting" });
    await h!.user.keyboard("{Meta>}w{/Meta}");
    await h!.user.click(await screen.findByRole("button", { name: "Stop export and close" }));
    await screen.findByRole("heading", { level: 1, name: "Drop Sony ARW files." });
    expect(h!.store.getState().exportRun).toBeNull();
    expect(h!.store.getState().session).toBeNull();
  });

  it("closes without asking when nothing is running", async () => {
    await openExport();
    await h!.user.keyboard("{Meta>}w{/Meta}");
    await screen.findByRole("heading", { level: 1, name: "Drop Sony ARW files." });
    expect(screen.queryByRole("dialog")).toBeNull();
  });
});
