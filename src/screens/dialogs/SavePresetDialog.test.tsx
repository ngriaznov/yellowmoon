/**
 * The Save preset dialog and its neighbours (`docs/30` section 9.3, 9.4 and 10): the Name field, the eight boxes with the groups the photo
 * changed ticked, the line about Auto, the disabled reason, the question "Replace Warm evening?", Rename, Update, and the import's question.
 */
import { screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { MockOptions } from "../../api.mock";
import { mainGradePatch, outputPatch } from "../../state/settingsGraph";
import { openPhotos, renderApp } from "../../test/harness";
import type { Harness } from "../../test/harness";
import { saveBlocked } from "./PresetDialogs";

let h: Harness | undefined;
afterEach(() => {
  h?.unmount();
  h = undefined;
});

async function onLook(options: MockOptions = {}): Promise<void> {
  h = renderApp({ current: 3, ...options });
  await openPhotos(h);
  await waitFor(() => expect(h!.store.getState().presets).not.toBeNull());
}

async function openSave(): Promise<HTMLElement> {
  await h!.user.click(screen.getByRole("button", { name: /^Save( preset)?…$/ }));
  return screen.findByRole("dialog", { name: "Save preset" });
}

const boxes = (dialog: HTMLElement): Record<string, boolean> =>
  Object.fromEntries(within(dialog).getAllByRole("checkbox").map((box) => [box.getAttribute("aria-labelledby") ? document.getElementById(box.getAttribute("aria-labelledby")!)!.textContent! : "", box.getAttribute("aria-checked") === "true"]));

describe("the Save preset dialog", () => {
  it("is named, starts in the Name field, and has the eight boxes with the one line each, in the order of the groups", async () => {
    await onLook();
    const dialog = await openSave();
    await waitFor(() => expect(within(dialog).getByRole("textbox", { name: "Name" })).toHaveFocus());
    expect(within(dialog).getByRole("textbox", { name: "Name" })).toHaveAttribute("placeholder", "Name");
    const group = within(dialog).getByRole("group", { name: "Save these" });
    const boxes = within(group).getAllByRole("checkbox");
    expect(boxes.map((b) => document.getElementById(b.getAttribute("aria-labelledby")!)!.textContent)).toEqual(["Look", "Lens", "Light", "Colour", "Detail", "Curve", "RAW", "Nodes"]);
    expect(boxes.map((b) => document.getElementById(b.getAttribute("aria-describedby")!)!.textContent)).toEqual([
      "The look and how strongly it is applied.",
      "The lens and its amounts.",
      "Exposure, contrast, brightness, highlights, shadows, white and black.",
      "White balance, colour balance, saturation and colour warp.",
      "Sharpening, noise reduction and output sharpening.",
      "The luma, red, green and blue curves.",
      "Decode options and the range.",
      "The other nodes, their links and masks.",
    ]);
    expect(within(dialog).getByText("Auto stays Auto, so each photo keeps its own exposure and white balance.")).toBeInTheDocument();
    expect(within(dialog).queryByText(/older Yellowmoon/)).toBeNull();
  });

  it("ticks nothing for a photo nobody touched, and Save says why it is off", async () => {
    await onLook();
    const dialog = await openSave();
    expect(Object.values(boxes(dialog)).every((on) => !on)).toBe(true);
    const save = within(dialog).getByRole("button", { name: "Save" });
    expect(save).toHaveAttribute("aria-disabled", "true");
    expect(save).toHaveAccessibleDescription("Give the preset a name.");
    await h!.user.type(within(dialog).getByRole("textbox", { name: "Name" }), "Warm evening");
    expect(save).toHaveAccessibleDescription("Tick at least one group");
    await h!.user.click(within(dialog).getByRole("checkbox", { name: "Light" }));
    expect(save).not.toHaveAttribute("aria-disabled");
  });

  it("ticks the groups the photo changed, which the core worked out", async () => {
    await onLook();
    await h!.store.update(outputPatch({ look: "builtin:signature" }));
    await h!.store.update(mainGradePatch({ contrast: { kind: "manual", value: 10 } }));
    const dialog = await openSave();
    expect(boxes(dialog)).toEqual({ Look: true, Lens: false, Light: true, Colour: false, Detail: false, Curve: false, RAW: false, Nodes: false });
  });

  it("asks for a name under the field and refuses 65 characters", async () => {
    await onLook();
    const dialog = await openSave();
    const field = within(dialog).getByRole("textbox", { name: "Name" });
    await h!.user.type(field, "x");
    await h!.user.clear(field);
    expect(within(dialog).getByText("Give the preset a name.", { selector: "p" })).toBeInTheDocument();
    expect(field).toHaveAttribute("aria-invalid", "true");
    await h!.user.click(field);
    await h!.user.paste("x".repeat(65));
    expect(within(dialog).getByText("Use 64 characters or fewer.", { selector: "p" })).toBeInTheDocument();
    expect(within(dialog).getByRole("button", { name: "Save" })).toHaveAttribute("aria-disabled", "true");
  });

  it("saves with Return in the Name field, closes, and says it", async () => {
    await onLook();
    await h!.store.update(mainGradePatch({ contrast: { kind: "manual", value: 10 } }));
    const dialog = await openSave();
    await h!.user.type(within(dialog).getByRole("textbox", { name: "Name" }), "Warm evening{Enter}");
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
    expect(screen.getByRole("status")).toHaveTextContent("Saved the preset Warm evening.");
    expect(h!.api.savedPresets.map((p) => [p.name, p.groups])).toEqual([["Warm evening", ["light"]]]);
    // The card follows, with its count and its first tile.
    expect(await screen.findByRole("button", { name: "Warm evening preset" })).toBeInTheDocument();
  });

  it("Esc and Cancel close it without saving, and the focus goes back to what opened it", async () => {
    await onLook();
    const opener = screen.getByRole("button", { name: "Save preset…" });
    opener.focus();
    const dialog = await openSave();
    await h!.user.keyboard("{Escape}");
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
    expect(h!.api.savedPresets).toHaveLength(0);
    await waitFor(() => expect(opener).toHaveFocus());
    void dialog;
  });

  it("is also opened by the menu item, and not twice", async () => {
    await onLook();
    h!.api.emitEvent("menu_action", { action: "save_preset" });
    const dialog = await screen.findByRole("dialog", { name: "Save preset" });
    h!.api.emitEvent("menu_action", { action: "save_preset" });
    expect(screen.getAllByRole("dialog")).toHaveLength(1);
    expect(dialog).toBeInTheDocument();
  });

  it("opens for an exported photo with the Name empty, and the older export's line", async () => {
    await onLook();
    h!.api.queueChoice("exported", "/x/older export.jpg");
    h!.api.emitEvent("menu_action", { action: "new_preset_from_export" });
    const dialog = await screen.findByRole("dialog", { name: "Save preset" });
    expect(within(dialog).getByRole("textbox", { name: "Name" })).toHaveValue("");
    expect(within(dialog).getByText("This photo was exported by an older Yellowmoon, so its automatic values are saved as fixed numbers. Untick Light, Colour and Detail to leave them out.")).toBeInTheDocument();
    expect(boxes(dialog)).toMatchObject({ Look: true, Light: false, Colour: false });
  });
});

describe("the question when the name exists", () => {
  async function toTheQuestion(): Promise<HTMLElement> {
    await onLook({ presets: 3 });
    await h!.store.update(mainGradePatch({ contrast: { kind: "manual", value: 10 } }));
    const dialog = await openSave();
    await h!.user.type(within(dialog).getByRole("textbox", { name: "Name" }), "warm EVENING{Enter}");
    return screen.findByRole("dialog", { name: "Replace Warm evening?" });
  }

  it("replaces the dialog's body, says it can't be undone, and the focus is on Cancel", async () => {
    const question = await toTheQuestion();
    expect(within(question).getByText("A preset with this name is already saved. Replacing it can't be undone.")).toBeInTheDocument();
    await waitFor(() => expect(within(question).getByRole("button", { name: "Cancel" })).toHaveFocus());
    expect(within(question).queryByRole("checkbox")).toBeNull();
    expect(h!.api.savedPresets).toHaveLength(3);
  });

  it("Cancel goes back to the form with the focus on Name, Replace writes over it and says Replaced", async () => {
    const question = await toTheQuestion();
    await h!.user.click(within(question).getByRole("button", { name: "Cancel" }));
    const form = await screen.findByRole("dialog", { name: "Save preset" });
    await waitFor(() => expect(within(form).getByRole("textbox", { name: "Name" })).toHaveFocus());
    expect(within(form).getByRole("textbox", { name: "Name" })).toHaveValue("warm EVENING");
    await h!.user.keyboard("{Enter}");
    const again = await screen.findByRole("dialog", { name: "Replace Warm evening?" });
    await h!.user.click(within(again).getByRole("button", { name: "Replace" }));
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
    expect(screen.getByRole("status")).toHaveTextContent("Replaced the preset warm EVENING.");
    expect(h!.api.savedPresets).toHaveLength(3);
  });
});

describe("Rename", () => {
  it("is the Name field alone, refuses a name that exists under the field, and renames", async () => {
    await onLook({ presets: 3 });
    h!.store.openRenamePreset("Warm evening");
    const dialog = await screen.findByRole("dialog", { name: "Rename preset" });
    expect(within(dialog).queryByRole("checkbox")).toBeNull();
    const field = within(dialog).getByRole("textbox", { name: "Name" });
    await waitFor(() => expect(field).toHaveFocus());
    expect(field).toHaveValue("Warm evening");
    await h!.user.clear(field);
    await h!.user.type(field, "cool morning{Enter}");
    expect(await within(dialog).findByText("A preset called cool morning already exists.")).toBeInTheDocument();
    expect(screen.getByRole("dialog", { name: "Rename preset" })).toBeInTheDocument();
    await h!.user.clear(field);
    await h!.user.type(field, "Late sun");
    await h!.user.click(within(dialog).getByRole("button", { name: "Rename" }));
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
    expect(screen.getByRole("status")).toHaveTextContent("Renamed the preset Warm evening to Late sun.");
  });
});

describe("the import's question", () => {
  it("is the same dialog, asked for each name that exists, with the focus on Cancel", async () => {
    await onLook({ presets: 3 });
    await h!.store.openPaths(["/x/Warm evening.ympreset"]);
    const question = await screen.findByRole("dialog", { name: "Replace Warm evening?" });
    await waitFor(() => expect(within(question).getByRole("button", { name: "Cancel" })).toHaveFocus());
    const spy = vi.spyOn(h!.api, "presetsImport");
    await h!.user.click(within(question).getByRole("button", { name: "Replace" }));
    await waitFor(() => expect(spy).toHaveBeenCalledWith(["/x/Warm evening.ympreset"], true));
    await waitFor(() => expect(screen.getByRole("status")).toHaveTextContent("Replaced the preset Warm evening."));
  });
});

describe("the reasons Save is off", () => {
  it("names the first thing missing", () => {
    expect(saveBlocked("", ["look"])).toBe("Give the preset a name.");
    expect(saveBlocked("x".repeat(70), ["look"])).toBe("Use 64 characters or fewer.");
    expect(saveBlocked("Name", [])).toBe("Tick at least one group");
    expect(saveBlocked("Name", ["look"])).toBeNull();
  });
});
