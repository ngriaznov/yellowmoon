/**
 * The mock's presets repeat the core's rules (`docs/30` sections 2 to 6): the groups, Auto stays Auto, the ring, stacking, the missing
 * look, names, the sorted list, damaged files and imports.
 */
import { afterEach, describe, expect, it } from "vitest";
import { createMockApi } from "./api.mock";
import type { MockApi, MockOptions } from "./api.mock";
import type { PresetGroup } from "./api/types";
import { inputOf, inputPatch, mainGradeOf, mainGradePatch, outputOf, outputPatch } from "./state/settingsGraph";
import { PRESET_GROUPS } from "./fixtures/presets";

let api: MockApi;
afterEach(() => api?.reset());

async function openTwelve(options: MockOptions = {}): Promise<MockApi> {
  api = createMockApi({ speed: Number.POSITIVE_INFINITY, ...options });
  await api.openDialog("files");
  api.flush();
  return api;
}

const manual = (value: number) => ({ kind: "manual", value }) as const;
const ids = async (): Promise<number[]> => (await api.sessionState()).photos.map((p) => p.id);
const settings = async (id: number) => (await api.getSettings(id)).settings;

describe("the mock's preset folder", () => {
  it("starts empty (the owner: no default presets), or with the fourteen of the mock-ups", async () => {
    await openTwelve();
    expect((await api.presetsList(null)).presets).toEqual([]);
    api.reset({ presets: 14, speed: Number.POSITIVE_INFINITY });
    const listing = await api.presetsList(null);
    expect(listing.presets).toHaveLength(14);
    expect(listing.presets.map((p) => p.name)).toEqual([...listing.presets.map((p) => p.name)].sort((a, b) => a.toLowerCase().localeCompare(b.toLowerCase())));
    expect(listing.folder).toContain("presets");
  });

  it("lists a file it cannot read as damaged, with the reason, and Delete removes it", async () => {
    await openTwelve({ damagedPresets: true });
    expect((await api.presetsList(null)).damaged).toEqual([{ file: "Old.ympreset", reason: "This preset was made by a newer version of Yellowmoon. Update to use it." }]);
    await api.presetRemoveFile("Old.ympreset");
    expect((await api.presetsList(null)).damaged).toEqual([]);
  });
});

describe("saving", () => {
  it("pre-ticks the groups in which the photo differs from the default, and nothing for an untouched photo", async () => {
    await openTwelve();
    const [first] = await ids();
    expect((await api.presetInspect({ kind: "photo", photo_id: first! })).changed).toEqual([]);
    await api.updateSettings(first!, outputPatch({ look: "arri:3000:film-warm" }));
    await api.updateSettings(first!, mainGradePatch({ contrast: manual(20) }));
    await api.updateSettings(first!, inputPatch({ exposure: { kind: "manual", ev: -0.5 } }));
    expect((await api.presetInspect({ kind: "photo", photo_id: first! })).changed).toEqual(["look", "light"]);
    // A picked white is saved as a manual white balance, so it ticks Colour.
    await api.updateSettings(first!, inputPatch({ white_balance: { kind: "manual", kelvin: 6100, tint: 3 } }));
    expect((await api.presetInspect({ kind: "photo", photo_id: first! })).changed).toEqual(["look", "light", "color"]);
  });

  it("refuses an empty name, a long name and a name that exists unless it replaces", async () => {
    await openTwelve();
    const [first] = await ids();
    const source = { kind: "photo", photo_id: first! } as const;
    await expect(api.presetSave("  ", ["light"], source, false)).rejects.toMatchObject({ scope: "presets", message: "Give the preset a name." });
    await expect(api.presetSave("x".repeat(65), ["light"], source, false)).rejects.toMatchObject({ message: "Use 64 characters or fewer." });
    await api.presetSave("Warm evening", ["light"], source, false);
    await expect(api.presetSave("warm EVENING", ["light"], source, false)).rejects.toMatchObject({ message: "A preset called Warm evening already exists." });
    const again = await api.presetSave("warm EVENING", ["light"], source, true);
    expect(again.name).toBe("warm EVENING");
    expect((await api.presetsList(null)).presets).toHaveLength(1);
  });

  it("changes the revision when a preset is written again, and tells the page with presets_changed", async () => {
    await openTwelve();
    const [first] = await ids();
    let changed = 0;
    api.on("presets_changed", () => (changed += 1));
    const one = await api.presetSave("A", ["light"], { kind: "photo", photo_id: first! }, false);
    const two = await api.presetSave("A", ["light"], { kind: "photo", photo_id: first! }, true);
    expect(two.revision).not.toBe(one.revision);
    await Promise.resolve();
    expect(changed).toBe(2);
  });
});

describe("applying (the Paste path)", () => {
  it("sets the groups' fields and leaves the rest; Auto stays Auto", async () => {
    await openTwelve();
    const [a, b] = await ids();
    // Photo a: Auto exposure, a manual contrast and a look. A Light preset holds all seven Light fields, Auto included.
    await api.updateSettings(a!, mainGradePatch({ contrast: manual(30) }));
    await api.presetSave("Contrasty", ["light"], { kind: "photo", photo_id: a! }, false);
    await api.updateSettings(b!, inputPatch({ exposure: { kind: "manual", ev: 1 } }));
    await api.updateSettings(b!, outputPatch({ look: "builtin:signature" }));
    const report = await api.presetApply("Contrasty", [b!]);
    expect(report.skipped).toEqual([]);
    expect(report.missing_look).toBeNull();
    const after = await settings(b!);
    expect(mainGradeOf(after).contrast).toEqual(manual(30));
    // Exposure is in Light, and the preset says Auto: the photo's own manual exposure becomes Auto again, never a number of photo a.
    expect(inputOf(after).exposure).toEqual({ kind: "auto" });
    // The look is not in Light: untouched.
    expect(outputOf(after).look).toBe("builtin:signature");
    // One undo step per photo: the report's before and after.
    expect(report.pasted).toHaveLength(1);
    expect(report.pasted[0]?.id).toBe(b);
  });

  it("is ringed after it is applied, and not after a change to one of its groups", async () => {
    await openTwelve();
    const [a, b] = await ids();
    await api.updateSettings(a!, mainGradePatch({ contrast: manual(30) }));
    await api.presetSave("Contrasty", ["light"], { kind: "photo", photo_id: a! }, false);
    expect((await api.presetsList(b!)).presets[0]?.matches).toBe(false);
    await api.presetApply("Contrasty", [b!]);
    expect((await api.presetsList(b!)).presets[0]?.matches).toBe(true);
    expect((await api.presetsList(null)).presets[0]?.matches).toBe(false);
    await api.updateSettings(b!, mainGradePatch({ brightness: 5 }));
    expect((await api.presetsList(b!)).presets[0]?.matches).toBe(false);
    // A change outside its groups keeps the ring.
    await api.presetApply("Contrasty", [b!]);
    await api.updateSettings(b!, outputPatch({ look: "builtin:signature" }));
    expect((await api.presetsList(b!)).presets[0]?.matches).toBe(true);
  });

  it("stacks: the later preset wins on a shared group and the earlier one's other groups stay", async () => {
    await openTwelve();
    const [a, b, c] = await ids();
    await api.updateSettings(a!, mainGradePatch({ contrast: manual(10), saturation: manual(20) }));
    await api.presetSave("One", ["light", "color"], { kind: "photo", photo_id: a! }, false);
    await api.updateSettings(b!, mainGradePatch({ contrast: manual(-10) }));
    await api.updateSettings(b!, outputPatch({ look: "arri:3000:film-warm" }));
    await api.presetSave("Two", ["light", "look"], { kind: "photo", photo_id: b! }, false);
    await api.presetApply("One", [c!]);
    await api.presetApply("Two", [c!]);
    const after = await settings(c!);
    expect(mainGradeOf(after).contrast).toEqual(manual(-10));
    expect(mainGradeOf(after).saturation).toEqual(manual(20));
    expect(outputOf(after).look).toBe("arri:3000:film-warm");
  });

  it("skips photos that are not developed and says which look is missing", async () => {
    await openTwelve({ presets: 14, missingLooks: ["film-warm"], skipped: { DSC04207: "This file can't be opened." } });
    const all = await ids();
    const list = await api.presetsList(all[0]!);
    const warm = list.presets.find((p) => p.name === "Warm evening");
    expect(warm?.look_missing).toBe(true);
    const report = await api.presetApply("Warm evening", [all[0]!, all[6]!]);
    expect(report.missing_look).toBe("Film warm");
    expect(report.skipped).toEqual([all[6]]);
    // The look field was left out, the rest applied.
    const after = await settings(all[0]!);
    expect(outputOf(after).look).not.toBe("arri:3000:film-warm");
    expect(mainGradeOf(after).contrast).toEqual(manual(12));
  });

  it("with Nodes replaces the other nodes; without it they stay", async () => {
    await openTwelve();
    const [a, b] = await ids();
    await api.updateSettings(a!, { nodes: { "2": { kind: "corrector", name: "Skin", pos: [0, 1] } }, links: [
      { from: { node: "input", port: "picture" }, to: { node: 1, port: "picture" } },
      { from: { node: 1, port: "picture" }, to: { node: 2, port: "picture" } },
      { from: { node: 2, port: "picture" }, to: { node: "output", port: "picture" } },
    ] });
    await api.updateSettings(b!, { nodes: { "3": { kind: "corrector", name: "Sky", pos: [0, -1] } }, links: [
      { from: { node: "input", port: "picture" }, to: { node: 1, port: "picture" } },
      { from: { node: 1, port: "picture" }, to: { node: 3, port: "picture" } },
      { from: { node: 3, port: "picture" }, to: { node: "output", port: "picture" } },
    ] });
    await api.presetSave("Skin", ["nodes"], { kind: "photo", photo_id: a! }, false);
    await api.presetSave("Light only", ["light"], { kind: "photo", photo_id: a! }, false);
    await api.presetApply("Light only", [b!]);
    expect((await settings(b!)).nodes.items.map((i) => i.id)).toEqual(["input", 1, 3, "output"]);
    await api.presetApply("Skin", [b!]);
    expect((await settings(b!)).nodes.items.map((i) => i.id)).toEqual(["input", 1, 2, "output"]);
  });

  it("never holds a photo's crop, turn or white pick, and never touches the target's", async () => {
    await openTwelve();
    const [a, b] = await ids();
    await api.updateSettings(a!, inputPatch({ orientation: 90, crop: { rect: { x: 0.1, y: 0.1, width: 0.5, height: 0.5 }, angle_deg: 0, aspect: "free" } as never }));
    await api.presetSave("Everything", [...PRESET_GROUPS] as PresetGroup[], { kind: "photo", photo_id: a! }, false);
    await api.presetApply("Everything", [b!]);
    const after = inputOf(await settings(b!));
    expect(after.orientation).toBe(0);
    expect(after.crop).toBeNull();
  });
});

describe("renaming, removing, exporting and importing", () => {
  it("renames, refusing a name that exists", async () => {
    await openTwelve({ presets: 3 });
    const names = (await api.presetsList(null)).presets.map((p) => p.name);
    await expect(api.presetRename(names[0]!, names[1]!.toUpperCase())).rejects.toMatchObject({ scope: "presets" });
    await api.presetRename(names[0]!, "Brand new");
    expect((await api.presetsList(null)).presets.map((p) => p.name)).toContain("Brand new");
    await expect(api.presetApply(names[0]!, [])).rejects.toMatchObject({ scope: "presets" });
  });

  it("removes to the Trash, and a tile of a preset that is gone is refused with the scope presets", async () => {
    await openTwelve({ presets: 2 });
    const [a] = await ids();
    const [first] = (await api.presetsList(a!)).presets;
    await api.presetRemove(first!.name);
    expect(api.system.trashed.at(-1)).toContain(`${first!.name}.ympreset`);
    await expect(api.presetThumbnail(a!, first!.name)).rejects.toMatchObject({ scope: "presets" });
  });

  it("exports to a path the panel gave, and nothing when the panel is cancelled", async () => {
    await openTwelve({ presets: 1 });
    const [first] = (await api.presetsList(null)).presets;
    api.queueChoice("save", "~/Desktop/Warm.ympreset");
    expect(await api.presetExport(first!.name)).toEqual({ path: "~/Desktop/Warm.ympreset" });
    api.queueChoice("save", null);
    expect(await api.presetExport(first!.name)).toBeNull();
  });

  it("imports: adds, answers the names that exist without replacing, refuses newer and unreadable files", async () => {
    await openTwelve({ presets: 1 });
    const [first] = (await api.presetsList(null)).presets;
    const result = await api.presetsImport(["/x/New one.ympreset", `/x/${first!.name}.ympreset`, "/x/Old newer.ympreset", "/x/bad.ympreset"], false);
    expect(result.added).toEqual(["New one"]);
    expect(result.exists).toEqual([{ name: first!.name, path: `/x/${first!.name}.ympreset` }]);
    expect(result.refused.map((r) => r.file)).toEqual(["Old newer.ympreset", "bad.ympreset"]);
    expect(result.refused[0]?.reason).toBe("This preset was made by a newer version of Yellowmoon. Update to use it.");
    expect(result.refused[1]?.reason).toBe("This file isn't a Yellowmoon preset.");
    const replaced = await api.presetsImport([`/x/${first!.name}.ympreset`], true);
    expect(replaced.replaced).toEqual([first!.name]);
  });
});
