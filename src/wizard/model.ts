/**
 * The ARRI look wizard's rules (`docs/08-looks.md` §Wizard, `design/mockups/import-looks.html`), as a reducer and
 * selectors with no React in them: which packages there are, what each step needs before the next unlocks, how
 * the files that arrived are judged, and the words the screen shows. `ImportLooks.tsx` draws the state; the
 * tests in `model.test.ts` drive it directly.
 */
import type { DisplayLut, LookId, LookInput, PackageId, SourceRecognition } from "../api/types";

// ---------------------------------------------------------------------------
// The five packages
// ---------------------------------------------------------------------------

/** What the Choose step shows for one package. The copy is the mockup's. */
export interface PackageDef {
  id: PackageId;
  /** "Look Library · LogC4". */
  name: string;
  description: string;
  /** "87 looks · 65-point · 181 MB". */
  meta: string;
  recommended: boolean;
  /** Hidden until "Show 3 older LogC3 packages". */
  older: boolean;
  /** Looks run before the display transform; display LUTs replace it. */
  role: "looks" | "display";
  /** Download size in megabytes. */
  megabytes: number;
  /** Files the package holds. */
  fileCount: number;
  /** What the LUTs of a looks package expect as input; null for display LUTs, which no answer can stand in for. */
  input: LookInput | null;
  /** The title of the download on ARRI's page, for the words that say what to download (`data/looks/arri-packages.json`, `arri_title`). */
  arriTitle: string;
  /** The title of the same looks as ARRI Look Files, when ARRI has them that way (the two look libraries that run in LogC4 and LogC3 to Rec.709). */
  lookFilesTitle?: string;
}

/** ARRI's five free packages, in the order of the Choose step (`docs/08` §What users can import). */
export const PACKAGES: readonly PackageDef[] = [
  {
    id: "look_library_logc4",
    name: "Look Library · LogC4",
    description: "87 creative looks, log to log. They run before the display transform, so the automatic development still applies.",
    meta: "87 looks · 65-point · 181 MB",
    recommended: true,
    older: false,
    role: "looks",
    megabytes: 181,
    fileCount: 87,
    input: "logc4",
    arriTitle: "Look Library for LogC4 (3D-LUTs)",
    lookFilesTitle: "Look Library for LogC4 (ALF4 Look Files)",
  },
  {
    id: "display_luts_logc4",
    name: "Display LUTs · LogC4",
    description: "ARRI's own display rendering for Rec.709, P3, Rec.2020, HLG and PQ. Optional: use it instead of our display transform.",
    meta: "22 LUTs · 32 MB",
    recommended: false,
    older: false,
    role: "display",
    megabytes: 32,
    fileCount: 22,
    input: null,
    arriTitle: "LogC4 LUT Package",
  },
  {
    id: "look_library_logc3",
    name: "Look Library · LogC3",
    description: "The same 87 looks for the older LogC3 encoding. They run in LogC3.",
    meta: "87 looks · 33-point · 30 MB",
    recommended: false,
    older: true,
    role: "looks",
    megabytes: 30,
    fileCount: 87,
    input: "logc3_ei800",
    arriTitle: "ARRI Look Library LogC3 (3D-LUTs)",
  },
  {
    id: "look_library_logc3_to_rec709",
    name: "Look Library · LogC3 to Rec.709",
    description: "Looks with the display rendering baked in. They replace our display transform.",
    meta: "87 looks · 32 MB",
    recommended: false,
    older: true,
    role: "looks",
    megabytes: 32,
    fileCount: 87,
    input: "logc3_to_rec709",
    arriTitle: "ARRI Look Library LogC3 (to Rec709) 3D-LUTs",
    lookFilesTitle: "ARRI Look Library LogC3 (Look Files)",
  },
  {
    id: "display_luts_logc3",
    name: "Display LUTs · LogC3",
    description: "Classic ALEXA display rendering, including the classic Rec.709.",
    meta: "48 LUTs · 46 MB",
    recommended: false,
    older: true,
    role: "display",
    megabytes: 46,
    fileCount: 48,
    input: null,
    arriTitle: "LogC3 LUT Package",
  },
];

/** ARRI's page that lists the look files (`docs/08` §Wizard step 2). */
export const ARRI_LOOKS_URL = "https://www.arri.com/en/learn/camera-systems/image-science/look-files";

/** The package with this id. */
export function packageDef(id: PackageId): PackageDef {
  const def = PACKAGES.find((p) => p.id === id);
  if (!def) throw new Error(`Unknown package ${id}`);
  return def;
}

// ---------------------------------------------------------------------------
// Steps
// ---------------------------------------------------------------------------

/** The five steps, as the rail shows them. */
export const STEPS: ReadonlyArray<{ title: string; text: string }> = [
  { title: "Choose", text: "Pick the ARRI packages you want." },
  { title: "Get the files", text: "Download them from ARRI's site." },
  { title: "Recognise", text: "We check what each package is." },
  { title: "Import", text: "Copy the looks into the app." },
  { title: "Finish", text: "Pick a default and clean up." },
];

/** 0 Choose, 1 Get the files, 2 Recognise, 3 Import, 4 Finish. */
export type StepIndex = 0 | 1 | 2 | 3 | 4;
export const STEP_CHOOSE = 0;
export const STEP_GET = 1;
export const STEP_RECOGNISE = 2;
export const STEP_IMPORT = 3;
export const STEP_FINISH = 4;

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

/** A file, folder or `.cube` that has arrived (dropped, chosen, or found in Downloads). */
export interface Received {
  path: string;
  /** What the recogniser said; null while it is still looking. */
  found: SourceRecognition | null;
  /** The answer to "what does this LUT expect as input?" for `needs_answer` sources. */
  answer: LookInput | null;
  /** Arrived through the Downloads watcher (so it can be offered to the Trash even when it is a zip). */
  fromDownloads?: boolean;
}

/** How the Import step stands. */
export interface ImportState {
  status: "idle" | "running" | "done" | "failed";
  /** Files copied so far, over all sources. */
  done: number;
  total: number;
  /** Plain-words reason when it failed. */
  error: string | null;
}

/** What the import added, for the Finish step. */
export interface ImportOutcome {
  /** Looks the packages provide ("87 looks ready"). */
  looks: number;
  displayLuts: number;
  /** Sources that were imported, for the Trash question. */
  zips: string[];
}

/** The choices of the Finish step. */
export interface FinishState {
  /** The default look; null keeps the current one. */
  defaultLook: LookId | null;
  display: "ours" | "arri";
  zip: "keep" | "trash";
}

/** Everything the wizard holds. */
export interface WizardState {
  step: StepIndex;
  chosen: readonly PackageId[];
  showOlder: boolean;
  sources: readonly Received[];
  /** The sources are being looked at again on entering Recognise. */
  checking: boolean;
  import: ImportState;
  outcome: ImportOutcome | null;
  finish: FinishState;
}

/** A wizard at its first step with Look Library · LogC4 ticked. */
export function initialWizard(): WizardState {
  return {
    step: STEP_CHOOSE,
    chosen: ["look_library_logc4"],
    showOlder: false,
    sources: [],
    checking: false,
    import: { status: "idle", done: 0, total: 0, error: null },
    outcome: null,
    finish: { defaultLook: null, display: "ours", zip: "trash" },
  };
}

/** What can happen to the wizard. */
export type WizardAction =
  | { type: "toggle-package"; id: PackageId }
  | { type: "toggle-older" }
  /** Next, Import or Done: moves on when `canAdvance` allows it. */
  | { type: "advance" }
  /** Back. */
  | { type: "back" }
  /** A finished step of the rail was pressed. Locked steps and the current one do nothing. */
  | { type: "select"; step: number }
  | { type: "received"; path: string; fromDownloads?: boolean }
  | { type: "recognised"; found: SourceRecognition }
  | { type: "checking"; on: boolean }
  | { type: "remove"; path: string }
  | { type: "answer"; path: string; input: LookInput }
  | { type: "import-start"; total: number }
  | { type: "import-progress"; done: number }
  | { type: "import-done"; outcome: ImportOutcome }
  | { type: "import-failed"; error: string }
  | { type: "finish"; patch: Partial<FinishState> };

const clampStep = (n: number): StepIndex => Math.max(0, Math.min(4, n)) as StepIndex;

/** The file name of a path: "~/Downloads/x.zip" gives "x.zip". */
export const baseName = (path: string): string => path.replace(/[\\/]+$/, "").split(/[\\/]/).pop() ?? path;

/** Moves the wizard on. Pure. */
export function reduceWizard(state: WizardState, action: WizardAction): WizardState {
  switch (action.type) {
    case "toggle-package": {
      const on = state.chosen.includes(action.id);
      const chosen = on ? state.chosen.filter((id) => id !== action.id) : PACKAGES.map((p) => p.id).filter((id) => id === action.id || state.chosen.includes(id));
      return { ...state, chosen };
    }
    case "toggle-older": {
      const showOlder = !state.showOlder;
      // Older packages that are hidden again are not chosen any more: what you can't see is not imported.
      const chosen = showOlder ? state.chosen : state.chosen.filter((id) => !packageDef(id).older);
      return { ...state, showOlder, chosen };
    }
    case "advance":
      return canAdvance(state) && state.step < STEP_FINISH ? { ...state, step: clampStep(state.step + 1) } : state;
    case "back":
      return state.step > STEP_CHOOSE ? { ...state, step: clampStep(state.step - 1) } : state;
    case "select":
      return action.step < state.step && action.step >= STEP_CHOOSE ? { ...state, step: clampStep(action.step) } : state;
    case "received": {
      if (state.sources.some((s) => s.path === action.path)) return state;
      const source: Received = { path: action.path, found: null, answer: null, fromDownloads: action.fromDownloads };
      return { ...state, sources: [...state.sources, source] };
    }
    case "recognised": {
      const sources = state.sources.map((s) => {
        if (s.path !== action.found.path) return s;
        // A new answer is asked for when the file is now something else.
        const keep = s.found?.recognition.kind === "needs_answer" && action.found.recognition.kind === "needs_answer";
        return { ...s, found: action.found, answer: keep ? s.answer : action.found.recognition.kind === "needs_answer" ? (action.found.recognition.suggestion ?? null) : null };
      });
      return { ...state, sources };
    }
    case "checking":
      return { ...state, checking: action.on };
    case "remove":
      return { ...state, sources: state.sources.filter((s) => s.path !== action.path) };
    case "answer":
      return { ...state, sources: state.sources.map((s) => (s.path === action.path ? { ...s, answer: action.input } : s)) };
    case "import-start":
      return { ...state, import: { status: "running", done: 0, total: action.total, error: null }, outcome: null };
    case "import-progress":
      return state.import.status === "running" ? { ...state, import: { ...state.import, done: Math.min(state.import.total, action.done) } } : state;
    case "import-done":
      return { ...state, import: { ...state.import, status: "done", done: state.import.total }, outcome: action.outcome };
    case "import-failed":
      return { ...state, import: { ...state.import, status: "failed", error: action.error } };
    case "finish":
      return { ...state, finish: { ...state.finish, ...action.patch } };
  }
}

// ---------------------------------------------------------------------------
// Selectors
// ---------------------------------------------------------------------------

/** The packages shown on the Choose step: the first two, and the older three when asked. */
export const visiblePackages = (state: WizardState): PackageDef[] => PACKAGES.filter((p) => !p.older || state.showOlder);

/** The chosen packages, in the order of the Choose step. */
export const chosenPackages = (state: WizardState): PackageDef[] => PACKAGES.filter((p) => state.chosen.includes(p.id));

/** Megabytes to download for the chosen packages. */
export const downloadTotalMb = (state: WizardState): number => chosenPackages(state).reduce((sum, p) => sum + p.megabytes, 0);

/** "A", "A and B", "A, B and C". */
export function joinList(items: readonly string[]): string {
  if (items.length <= 1) return items[0] ?? "";
  return `${items.slice(0, -1).join(", ")} and ${items[items.length - 1]}`;
}

/** The chosen packages as a sentence: "Look Library · LogC4 and Display LUTs · LogC4"; "the packages you chose" when none. */
export const wantList = (state: WizardState): string => joinList(chosenPackages(state).map((p) => p.name)) || "the packages you chose";

/** How the Recognise step judges one source. */
export type Verdict =
  | { kind: "checking" }
  /** A package that was chosen. */
  | { kind: "match"; packageId: PackageId; title: string; what: string; contents: string; note: string | null }
  /** A package that is not among the chosen ones. */
  | { kind: "other-package"; packageId: PackageId; title: string; what: string; contents: string; explanation: string }
  /** `.cube` files that need the generic question; `input` is the answer so far. */
  | { kind: "question"; fileCount: number; lutSize: number | null; note: string; input: LookInput | null; standsFor: string | null }
  /** ARRI Look Files of the person's own: each says what it expects, so nothing is asked. */
  | { kind: "look-files"; what: string; contents: string; fileCount: number; input: LookInput | null; note: string | null; standsFor: string | null }
  | { kind: "unusable"; explanation: string }
  | { kind: "missing"; explanation: string };

/**
 * The chosen packages as ARRI's page names them: "Look Library for LogC4 (3D-LUTs)". For the words that say what to download. A
 * look library that ARRI also publishes as look files names both: "'Look Library for LogC4 (3D-LUTs)' (or 'Look Library for LogC4
 * (ALF4 Look Files)')".
 */
export const downloadList = (packages: readonly PackageDef[]): string =>
  joinList(packages.map((p) => (p.lookFilesTitle ? `'${p.arriTitle}' (or '${p.lookFilesTitle}')` : `'${p.arriTitle}'`)));

/**
 * The chosen package a set of the person's own files stands in for: a zip or folder (not one loose file) of unknown `.cube` files
 * whose input has been answered, or of ARRI Look Files that all expect the same input, and a chosen looks package with that input.
 * The person said what the set is, or the files did, so the package counts as present; the files are imported as their own looks
 * with that input (`docs/08` §Any other .cube, §ARRI Look Files). Null when there is none.
 */
export function standsInFor(state: WizardState, source: Received): PackageDef | null {
  const found = source.found;
  if (!found || found.source === "file") return null;
  const r = found.recognition;
  const input = r.kind === "needs_answer" ? source.answer : r.kind === "look_files" ? r.input : null;
  if (!input) return null;
  return chosenPackages(state).find((p) => p.role === "looks" && p.input === input) ?? null;
}

/** What the recogniser found, set against what was chosen. */
export function verdictFor(state: WizardState, source: Received): Verdict {
  const found = source.found;
  if (!found) return { kind: "checking" };
  const r = found.recognition;
  switch (r.kind) {
    case "package": {
      if (state.chosen.includes(r.package_id)) {
        return { kind: "match", packageId: r.package_id, title: r.title, what: r.what, contents: r.contents, note: r.note };
      }
      const chosenList = chosenPackages(state);
      const chosen = joinList(chosenList.map((p) => p.name));
      const explanation = chosen
        ? `This is ${r.title}, not ${chosen}, which you chose. Download ${downloadList(chosenList)} from ARRI's page, or go back and choose this one.`
        : `This is ${r.title}. Go back and choose it to import it.`;
      return { kind: "other-package", packageId: r.package_id, title: r.title, what: r.what, contents: r.contents, explanation };
    }
    case "needs_answer":
      return { kind: "question", fileCount: r.file_count, lutSize: r.lut_size, note: r.note, input: source.answer, standsFor: standsInFor(state, source)?.name ?? null };
    case "look_files":
      return { kind: "look-files", what: r.what, contents: r.contents, fileCount: r.file_count, input: r.input, note: r.note, standsFor: standsInFor(state, source)?.name ?? null };
    case "unusable":
      return { kind: "unusable", explanation: r.reason };
    case "missing":
      return { kind: "missing", explanation: r.reason };
  }
}

/** A step of the rail is locked while it is ahead of the current one. */
export const stepLocked = (state: WizardState, step: number): boolean => step > state.step;

/** The package ids that have a source recognised as them, or an answered set of unknown files standing in for them. */
export function matchedPackages(state: WizardState): PackageId[] {
  const ids: PackageId[] = [];
  for (const source of state.sources) {
    const v = verdictFor(state, source);
    const id = v.kind === "match" ? v.packageId : v.kind === "question" || v.kind === "look-files" ? (standsInFor(state, source)?.id ?? null) : null;
    if (id && !ids.includes(id)) ids.push(id);
  }
  return ids;
}

/** Chosen packages that no source matches yet, and no answered set stands in for. */
export const unmatchedPackages = (state: WizardState): PackageDef[] => {
  const matched = matchedPackages(state);
  return chosenPackages(state).filter((p) => !matched.includes(p.id));
};

/** True while any source is still being looked at. */
export const anyChecking = (state: WizardState): boolean => state.checking || state.sources.some((s) => s.found === null);

/**
 * The sources that will be imported, in order: packages that were chosen, `.cube` files whose input has been
 * answered, and the person's own look files. Everything else is left out and the Recognise step says why.
 */
export interface PlanItem {
  path: string;
  /** The answer to the generic question, for `.cube` files that match no package. */
  input?: LookInput;
  /** Files that will be copied. */
  files: number;
  kind: "package" | "cube" | "look-files";
  packageId?: PackageId;
}

export function importPlan(state: WizardState): PlanItem[] {
  const plan: PlanItem[] = [];
  for (const source of state.sources) {
    const v = verdictFor(state, source);
    if (v.kind === "match") {
      const r = source.found?.recognition;
      plan.push({ path: source.path, files: r?.kind === "package" ? r.file_count : packageDef(v.packageId).fileCount, kind: "package", packageId: v.packageId });
    } else if (v.kind === "question" && v.input) {
      plan.push({ path: source.path, input: v.input, files: v.fileCount, kind: "cube" });
    } else if (v.kind === "look-files") {
      // The files say what they are for: nothing to answer.
      plan.push({ path: source.path, files: v.fileCount, kind: "look-files" });
    }
  }
  return plan;
}

/** Whether the import copies ARRI Look Files (`.alf4`, `.alf4c`, `.aml`) as well as, or instead of, `.cube` files. */
export function copiesLookFiles(state: WizardState): boolean {
  const planned = new Set(importPlan(state).map((p) => p.path));
  return state.sources.some((s) => {
    const r = s.found?.recognition;
    if (!planned.has(s.path) || !r) return false;
    return r.kind === "look_files" || (r.kind === "package" && !r.contents.includes(".cube"));
  });
}

/**
 * Why Import is not available yet on the Recognise step, in plain words; null when it is. Import waits for the files to be
 * looked at, for every set of unknown `.cube` files to be answered, and for something to import. A chosen package that nothing
 * matches does not hold it back while there is something to import: [`recogniseNote`] says so instead.
 */
export function recogniseBlocker(state: WizardState): string | null {
  if (anyChecking(state)) return "Checking the files…";
  const unanswered = state.sources.filter((s) => {
    const v = verdictFor(state, s);
    return v.kind === "question" && v.input === null;
  });
  if (unanswered.length > 0) return "Say what the other LUTs expect as input.";
  if (importPlan(state).length === 0) {
    const missing = unmatchedPackages(state);
    if (missing.length === 1) return `${missing[0]!.name} is missing. Go back and add it.`;
    if (missing.length > 1) return "Some packages are missing. Go back and add them.";
    return "There is nothing to import yet. Go back and add a package.";
  }
  return null;
}

/** A chosen package that nothing matches, when there is something else to import: said in the footer, never a lock. Null otherwise. */
export function recogniseNote(state: WizardState): string | null {
  if (recogniseBlocker(state) !== null) return null;
  const missing = unmatchedPackages(state);
  if (missing.length === 0) return null;
  return missing.length === 1
    ? `${missing[0]!.name} isn't here. You can import the rest now and add it later.`
    : "Some of the packages you chose aren't here. You can import the rest now and add them later.";
}

/**
 * Whether the Next button of a step is available. Choose needs a package; Get the files needs a file for each chosen
 * package (the Recognise step judges what they are); Recognise needs something to import, with every question answered
 * (a chosen package that is missing only gets a note); Import waits for the copy to finish.
 */
export function canAdvance(state: WizardState): boolean {
  switch (state.step) {
    case STEP_CHOOSE:
      return state.chosen.length > 0;
    case STEP_GET:
      return state.sources.length >= Math.max(1, state.chosen.length);
    case STEP_RECOGNISE:
      return recogniseBlocker(state) === null && importPlan(state).length > 0;
    case STEP_IMPORT:
      return state.import.status === "done";
    default:
      return true;
  }
}

/** The label of the Next button on each step. */
export const NEXT_LABELS = ["Next", "Next", "Import", "Next", "Done"] as const;

/** The grey line left of the Next button. */
export function footNote(state: WizardState): string {
  switch (state.step) {
    case STEP_CHOOSE:
      return state.chosen.length > 0 ? `Download total: ${downloadTotalMb(state)} MB` : "Choose at least one package";
    case STEP_GET: {
      const need = Math.max(1, state.chosen.length);
      if (state.sources.length >= need) return "Everything has arrived.";
      return need === 1 ? "Next unlocks when the zip arrives." : `Next unlocks when all ${need} files have arrived.`;
    }
    case STEP_RECOGNISE: {
      const blocker = recogniseBlocker(state);
      if (blocker) return blocker;
      const note = recogniseNote(state);
      if (note) return note;
      const n = matchedPackages(state).length;
      return `${n} ${n === 1 ? "package matches" : "packages match"} what you chose`;
    }
    case STEP_IMPORT:
      if (state.import.status === "failed") return "Something went wrong. Go back and try again.";
      return state.import.status === "done" ? "All files copied and checked" : "This takes a few seconds";
    default:
      return "You can change the default look any time";
  }
}

/** The big heading of each step. */
export function heading(state: WizardState): string {
  switch (state.step) {
    case STEP_CHOOSE:
      return "Choose what to import";
    case STEP_GET:
      return "Download from ARRI";
    case STEP_RECOGNISE:
      return importPlan(state).length > 0 && recogniseBlocker(state) === null && unmatchedPackages(state).length === 0 ? "Found it" : "Check what arrived";
    case STEP_IMPORT:
      return state.import.status === "done" ? "Imported" : state.import.status === "failed" ? "Not imported" : "Importing";
    default: {
      const o = state.outcome;
      if (!o) return "Done";
      if (o.looks > 0) return `${o.looks} ${o.looks === 1 ? "look" : "looks"} ready`;
      return `${o.displayLuts} display ${o.displayLuts === 1 ? "LUT" : "LUTs"} ready`;
    }
  }
}

/** The line under the heading. */
export function lead(state: WizardState): string {
  switch (state.step) {
    case STEP_CHOOSE:
      return "ARRI publishes these looks for free. We can't include them in the app, so you download them from ARRI and the app imports them.";
    case STEP_GET:
      return "Your browser opens ARRI's Look Files page. Download the package there and the app picks it up.";
    case STEP_RECOGNISE:
      return "Each package is identified from its folder names, file names and LUT sizes.";
    case STEP_IMPORT:
      return "Nothing leaves this Mac. The looks are ready the moment this finishes.";
    default:
      return state.outcome && state.outcome.looks > 0
        ? "Here they are on your current photo. Pick a default and you are done."
        : "Pick how the display should look and you are done.";
  }
}

/** Whether the Finish step offers the display rendering choice: display LUTs were imported now. */
export const offersDisplayChoice = (state: WizardState): boolean => (state.outcome?.displayLuts ?? 0) > 0;

/** Zips (and other files) that the Trash question is about. */
export const trashCandidates = (state: WizardState): string[] => {
  const planned = new Set(importPlan(state).map((p) => p.path));
  return state.sources.filter((s) => planned.has(s.path) && s.found?.source === "zip").map((s) => s.path);
};

/** "181 MB" for a size in bytes; "" when unknown. */
export function formatSize(bytes: number | null): string {
  if (bytes === null) return "";
  const mb = bytes / 1_000_000;
  if (mb >= 1000) return `${(mb / 1000).toFixed(1)} GB`;
  return mb >= 1 ? `${Math.round(mb)} MB` : `${Math.max(1, Math.round(bytes / 1000))} KB`;
}

/** The three answers of the generic `.cube` question (`docs/08` §Any other .cube). */
export const INPUT_CHOICES: ReadonlyArray<{ value: LookInput; label: string; detail: string }> = [
  { value: "logc4", label: "LogC4", detail: "LogC4 with ARRI Wide Gamut 4. Used like the Look Library for LogC4." },
  { value: "logc3_ei800", label: "LogC3", detail: "LogC3 (EI 800) with ARRI Wide Gamut 3." },
  { value: "logc3_to_rec709", label: "LogC3 · Rec.709", detail: "LogC3 in, Rec.709 out. It replaces our display rendering." },
];

/**
 * The display LUT "ARRI's" stands for in the display rendering choice: the LogC4 Rec.709 one when it is imported, else the
 * first of the LogC4 ones, else the first of the LogC3 ones. Null when none are imported.
 */
export function preferredDisplayLut(luts: readonly DisplayLut[] | null): DisplayLut | null {
  if (!luts || luts.length === 0) return null;
  const rank = (l: DisplayLut): number => (l.input === "logc4" ? 0 : 10) + (l.target.startsWith("rec709-g24") ? 0 : l.target.startsWith("rec709") ? 1 : 2);
  return [...luts].sort((a, b) => rank(a) - rank(b))[0] ?? null;
}
