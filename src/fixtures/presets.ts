/**
 * The presets' rules, for the mock API (`docs/30-presets.md` sections 2 to 4). The core owns them (`ym_core::presets`); the mock plays
 * the core in a browser and in the tests, so it saves, applies, matches and pre-ticks by the same rules.
 *
 * A preset is the settings of a photo read over the default graph, cut to the groups that were ticked: a field of a group that is not
 * ticked has its default. Pure functions on plain data.
 */
import type { DevelopSettings, LookId, NodeId, NodeItem, NodePatch, PresetGroup, ResolvedSettings, SettingsPatch } from "../api/types";
import {
  applyPatch,
  clone,
  defaultSettings,
  INPUT_ID,
  inputOf,
  MAIN_ID,
  mainOf,
  numberedOf,
  OUTPUT_ID,
  outputOf,
  same,
} from "../state/settingsGraph";
import { copiedPatch } from "./graph";

/** The eight groups in the order of `docs/30` section 2: the dialog, the file and the help list them so. */
export const PRESET_GROUPS: readonly PresetGroup[] = ["look", "lens", "light", "color", "detail", "curve", "raw", "nodes"];

/** The Input's fields of each group. */
const INPUT_FIELDS: Readonly<Partial<Record<PresetGroup, readonly string[]>>> = {
  lens: ["lens"],
  light: ["exposure"],
  color: ["light_source", "white_balance"],
  detail: ["sharpen", "noise_reduction"],
  raw: ["decode", "rolloff"],
};

/** Main's grade fields of each group. */
const GRADE_FIELDS: Readonly<Partial<Record<PresetGroup, readonly string[]>>> = {
  light: ["contrast", "brightness", "highlights", "shadows", "white", "black"],
  color: ["color_balance", "saturation", "color_warp"],
  curve: ["curve"],
};

/** The Output's fields of each group. */
const OUTPUT_FIELDS: Readonly<Partial<Record<PresetGroup, readonly string[]>>> = {
  look: ["look", "look_intensity"],
  detail: ["output_sharpening"],
};

/** Main's own fields that go with Nodes (`docs/30` section 2). */
const MAIN_NODE_FIELDS = ["name", "bypass", "trim", "masks"] as const;

type Fields = Record<string, unknown>;

const pick = (from: object, names: readonly string[] | undefined): Fields => {
  const out: Fields = {};
  for (const name of names ?? []) out[name] = clone((from as Fields)[name]);
  return out;
};

/** What a group holds of a settings document, as plain data to compare. */
export function groupValue(settings: DevelopSettings, group: PresetGroup): unknown {
  const main = mainOf(settings);
  if (group === "nodes") {
    return {
      others: settings.nodes.items.filter((item) => item.id !== INPUT_ID && item.id !== OUTPUT_ID && item.id !== MAIN_ID),
      links: settings.nodes.links,
      main: pick(main, MAIN_NODE_FIELDS),
      pos: [inputOf(settings).pos, main.pos, outputOf(settings).pos],
    };
  }
  return {
    input: pick(inputOf(settings), INPUT_FIELDS[group]),
    grade: pick(main.grade, GRADE_FIELDS[group]),
    output: pick(outputOf(settings), OUTPUT_FIELDS[group]),
  };
}

/** Empties the Brush rows of a node (a preset never holds strokes, `docs/30` section 3); Object clicks are emptied by `copiedPatch`. */
function withoutStrokes(item: NodeItem): NodeItem {
  if ((item.kind !== "corrector" && item.kind !== "film") || !item.masks.items.some((row) => row.kind === "brush")) return item;
  return {
    ...item,
    masks: {
      ...item.masks,
      items: item.masks.items.map((row) => (row.kind === "brush" ? { ...row, strokes: [] } : row)),
    },
  } as NodeItem;
}

/**
 * The settings Save makes of a photo before the cut (`Preset::from_photo`, `docs/30` section 4.1): Copy settings' patch laid over the
 * photo's own (Auto stays Auto, a picked white is its manual Kelvin and tint, Object clicks emptied) with the Brush strokes emptied. The
 * crop, the turn and the white pick stay the photo's, and no group holds them.
 */
function savedSettings(rs: ResolvedSettings): DevelopSettings {
  const settings = applyPatch(rs.settings, copiedPatch(rs));
  return {
    nodes: {
      ...settings.nodes,
      items: settings.nodes.items.map(withoutStrokes),
    },
  };
}

/** The groups in which the photo differs from `defaults` (`changed_groups`, `docs/30` section 4.1), in the order of the groups. */
export function changedGroups(rs: ResolvedSettings, defaults: DevelopSettings): PresetGroup[] {
  const saved = savedSettings(rs);
  return PRESET_GROUPS.filter((group) => !same(groupValue(saved, group), groupValue(defaults, group)));
}

/** A saved preset in the mock's folder. */
export interface StoredPreset {
  name: string;
  groups: PresetGroup[];
  look: { id: LookId; name: string } | null;
  /** The photo's settings read over the defaults, cut to `groups`. */
  settings: DevelopSettings;
  /** Counts the times it was written, so that a preset saved again has a new picture key (the core uses the file's hash). */
  version: number;
}

/** Cuts settings to the groups, over the default graph. */
function cutTo(source: DevelopSettings, groups: readonly PresetGroup[]): DevelopSettings {
  let out = defaultSettings(outputOf(source).look);
  const set: Record<string, NodePatch> = {};
  const into = (id: string, fields: NodePatch): void => {
    set[id] = { ...set[id], ...fields };
  };
  const main = mainOf(source);
  for (const group of groups) {
    into(INPUT_ID, pick(inputOf(source), INPUT_FIELDS[group]) as NodePatch);
    into(OUTPUT_ID, pick(outputOf(source), OUTPUT_FIELDS[group]) as NodePatch);
    const grade = pick(main.grade, GRADE_FIELDS[group]);
    if (Object.keys(grade).length > 0)
      into(String(MAIN_ID), {
        grade: {
          ...((set[String(MAIN_ID)]?.grade as object | undefined) ?? {}),
          ...grade,
        },
      });
  }
  out = applyPatch(out, { nodes: set });
  if (groups.includes("nodes")) {
    const nodes: Record<string, NodePatch> = {
      [String(MAIN_ID)]: {
        ...pick(main, MAIN_NODE_FIELDS),
        pos: clone(main.pos),
      } as NodePatch,
      [INPUT_ID]: { pos: clone(inputOf(source).pos) },
      [OUTPUT_ID]: { pos: clone(outputOf(source).pos) },
    };
    for (const item of source.nodes.items) {
      if (item.id === INPUT_ID || item.id === OUTPUT_ID || item.id === MAIN_ID) continue;
      const { id: _id, ...fields } = clone(item);
      nodes[String(item.id)] = fields as NodePatch;
    }
    out = applyPatch(out, { nodes, links: clone(source.nodes.links) });
  }
  return out;
}

/** `Preset::from_photo`: the preset of a photo's settings for the chosen groups; `lookName` is recorded with the Look group. */
export function presetFromPhoto(
  name: string,
  rs: ResolvedSettings,
  groups: readonly PresetGroup[],
  lookName: string | null,
  version = 1,
): StoredPreset {
  const ordered = PRESET_GROUPS.filter((g) => groups.includes(g));
  const settings = cutTo(savedSettings(rs), ordered);
  const lookId = outputOf(rs.settings).look;
  return {
    name,
    groups: ordered,
    look: ordered.includes("look") ? { id: lookId, name: lookName ?? lookId } : null,
    settings,
    version,
  };
}

/** What `patch_for` makes for one photo (`docs/30` section 4.2). */
export interface PresetPatch {
  patch: SettingsPatch;
  /** The name of the look the preset holds when it is not usable, so that the report can say it. */
  missingLook: string | null;
}

/** `Preset::patch_for`: the node patch that sets the preset's fields of its groups on `target`. */
export function patchFor(preset: StoredPreset, target: DevelopSettings, lookUsable: boolean): PresetPatch {
  const nodes: Record<string, NodePatch> = {};
  const add = (id: string, fields: NodePatch): void => {
    nodes[id] = { ...nodes[id], ...fields };
  };
  const source = preset.settings;
  const main = mainOf(source);
  let missingLook: string | null = null;
  for (const group of preset.groups) {
    if (group === "nodes") continue;
    if (group === "look" && !lookUsable) {
      missingLook = preset.look?.name ?? "";
      continue;
    }
    const input = pick(inputOf(source), INPUT_FIELDS[group]);
    if (Object.keys(input).length > 0) add(INPUT_ID, input as NodePatch);
    const output = pick(outputOf(source), OUTPUT_FIELDS[group]);
    if (Object.keys(output).length > 0) add(OUTPUT_ID, output as NodePatch);
    const grade = pick(main.grade, GRADE_FIELDS[group]);
    if (Object.keys(grade).length > 0)
      add(String(MAIN_ID), {
        grade: {
          ...((nodes[String(MAIN_ID)]?.grade as object | undefined) ?? {}),
          ...grade,
        },
      });
  }
  const patch: SettingsPatch = { nodes };
  if (preset.groups.includes("nodes")) {
    add(String(MAIN_ID), {
      ...pick(main, MAIN_NODE_FIELDS),
      pos: clone(main.pos),
    } as NodePatch);
    add(INPUT_ID, { pos: clone(inputOf(source).pos) });
    add(OUTPUT_ID, { pos: clone(outputOf(source).pos) });
    const given = new Set<NodeId>();
    for (const item of source.nodes.items) {
      if (item.id === INPUT_ID || item.id === OUTPUT_ID || item.id === MAIN_ID) continue;
      const { id: _id, ...fields } = clone(item);
      nodes[String(item.id)] = fields as NodePatch;
      given.add(item.id);
    }
    patch.links = clone(source.nodes.links);
    patch.remove = numberedOf(target)
      .map((item) => item.id)
      .filter((id) => id !== MAIN_ID && !given.has(id));
  }
  return { patch, missingLook };
}

/** `Preset::matches` (`docs/30` section 4.3): applying it to `target` would change nothing. */
export function presetMatches(preset: StoredPreset, target: DevelopSettings, lookUsable: boolean): boolean {
  const { patch } = patchFor(preset, target, lookUsable);
  const next = applyPatch(target, patch);
  return same(next, target);
}

/** A preset the mock can start with (`?mockPresets=14`): a name, its groups and the patch that makes its values over the default graph. */
export interface PresetRecipe {
  name: string;
  groups: readonly PresetGroup[];
  patch: SettingsPatch;
}

const manual = (value: number): { kind: "manual"; value: number } => ({
  kind: "manual",
  value,
});
const light = (look: string, intensity: number): NodePatch => ({
  look,
  look_intensity: intensity,
});
const wb = (kelvin: number, tint: number): NodePatch => ({
  white_balance: { kind: "manual", kelvin, tint },
});
const grade = (g: object): NodePatch => ({ grade: g });

/**
 * The fourteen presets of the mock-ups (`design/presets/01`): the card starts empty in the app, so these only exist when asked for. The list is
 * sorted by name (`docs/30` section 6), so the last eight are named to come after "Warm evening": the first six tiles are the mock-up's six.
 */
export const MOCK_PRESET_RECIPES: readonly PresetRecipe[] = [
  {
    name: "Warm evening",
    groups: ["look", "light", "color"],
    patch: {
      nodes: {
        output: light("arri:3000:film-warm", 100),
        input: { exposure: { kind: "manual", ev: -0.3 }, ...wb(6400, 4) },
        "1": grade({ contrast: manual(12), saturation: manual(8) }),
      },
    },
  },
  {
    name: "Cool morning",
    groups: ["look", "light", "color"],
    patch: {
      nodes: {
        output: light("arri:3000:film-cool", 90),
        input: { exposure: { kind: "manual", ev: 0.2 }, ...wb(4900, -3) },
        "1": grade({ contrast: manual(-6), saturation: manual(-4) }),
      },
    },
  },
  {
    name: "Soft portrait",
    groups: ["light", "color", "detail"],
    patch: {
      nodes: {
        input: {
          exposure: { kind: "manual", ev: 0.4 },
          ...wb(5800, 6),
          sharpen: { kind: "manual", amount: 20, radius_px: 1 },
        },
        "1": grade({ contrast: manual(-14), saturation: manual(-8) }),
      },
    },
  },
  {
    name: "Street grit",
    groups: ["look", "light", "detail"],
    patch: {
      nodes: {
        output: light("arri:3000:bleach-bypass", 70),
        input: { sharpen: { kind: "manual", amount: 70, radius_px: 1 } },
        "1": grade({ contrast: manual(28), shadows: manual(-20) }),
      },
    },
  },
  {
    name: "Faded film",
    groups: ["look", "light", "curve"],
    patch: {
      nodes: {
        output: light("arri:3000:soft-print", 100),
        "1": grade({
          black: manual(30),
          curve: {
            luma: [
              [0, 0.08],
              [1, 0.95],
            ],
          },
        }),
      },
    },
  },
  {
    name: "Night blue",
    groups: ["look", "color"],
    patch: {
      nodes: {
        output: light("arri:3000:night-blue", 100),
        input: wb(4300, -6),
      },
    },
  },
  {
    name: "Winter light",
    groups: ["light", "color"],
    patch: {
      nodes: {
        input: { exposure: { kind: "manual", ev: -0.2 }, ...wb(7200, 8) },
        "1": grade({ saturation: manual(14) }),
      },
    },
  },
  {
    name: "Whisper white",
    groups: ["color", "detail"],
    patch: {
      nodes: {
        input: {
          ...wb(5500, 0),
          noise_reduction: {
            kind: "manual",
            luminance: 20,
            color: 30,
            detail: 50,
          },
        },
      },
    },
  },
  {
    name: "Wide sky",
    groups: ["look", "color"],
    patch: {
      nodes: {
        output: light("arri:3000:teal-and-orange", 60),
        input: wb(5000, -12),
      },
    },
  },
  {
    name: "Willow",
    groups: ["look", "light"],
    patch: {
      nodes: {
        output: light("arri:3000:soft-print", 60),
        "1": grade({ highlights: manual(-25), black: manual(18) }),
      },
    },
  },
  {
    name: "Yellow hour",
    groups: ["light", "curve"],
    patch: {
      nodes: {
        "1": grade({
          black: manual(40),
          curve: {
            luma: [
              [0, 0.12],
              [1, 1],
            ],
          },
        }),
      },
    },
  },
  {
    name: "Zinc grey",
    groups: ["light", "color"],
    patch: {
      nodes: {
        input: { exposure: { kind: "manual", ev: 0.7 }, ...wb(5600, 2) },
        "1": grade({ shadows: manual(30), saturation: manual(-6) }),
      },
    },
  },
  {
    name: "Xerox fade",
    groups: ["color"],
    patch: {
      nodes: {
        "1": grade({
          color_balance: { shadows: { kind: "manual", point: [-0.3, 0.2] } },
        }),
      },
    },
  },
  {
    name: "Wet street",
    groups: ["look", "color"],
    patch: {
      nodes: {
        output: light("arri:3000:monochrome", 40),
        "1": grade({ saturation: manual(-30) }),
      },
    },
  },
];

/** `Preset::from_photo` for settings that are not a photo's: the recipe's values over the default graph, cut to its groups. */
export function presetFromRecipe(recipe: PresetRecipe, lookName: (id: LookId) => string | null): StoredPreset {
  const settings = applyPatch(defaultSettings(), recipe.patch);
  const groups = PRESET_GROUPS.filter((g) => recipe.groups.includes(g));
  const id = outputOf(settings).look;
  return {
    name: recipe.name,
    groups,
    look: groups.includes("look") ? { id, name: lookName(id) ?? id } : null,
    settings: cutTo(settings, groups),
    version: 1,
  };
}
