/**
 * What the Export panel works out from the session before it draws anything: which photos are in the job, the file names
 * they will get, where they go, and whether the template can be used. Pure, so it is tested without a screen
 * (`model.test.ts`).
 */
import type { ApplyScope, ExportOptions, Look, PhotoSummary, Preferences, ResolvedSettings, SessionState } from "../api/types";
import { captureDate, photoName } from "../state/photo";
import { fullLongEdge, photosInScope, isExportable } from "./options";
import { previewFileName, slugify, templateProblem, templateWarning } from "./template";
import type { TemplateParts } from "./template";

/** What `{date}` becomes for a photo with no capture time: the word the core writes (`docs/09` §Names and folders). */
export const UNDATED = "undated";

/** Where files go when nothing says otherwise and the session has no folder of its own. */
export const FALLBACK_DESTINATION = "~/Pictures/Yellowmoon export";

/** The part of the app's state the model reads. */
export interface ExportModelInput {
  session: SessionState | null;
  settings: Readonly<Record<number, ResolvedSettings>>;
  looks: readonly Look[] | null;
  preferences: Preferences | null;
}

/** The facts the Export panel draws. */
export interface ExportModel {
  /** The photos in the job, in filmstrip order. */
  photos: PhotoSummary[];
  count: number;
  /** Photos that are picked, and photos that can be exported: the counts in "Picked · 3" and "All · 12". */
  nPicked: number;
  nAll: number;
  /** The ids in the job, for the filmstrip's dimming. */
  included: ReadonlySet<number>;
  /** The longest side of the biggest photo, for "Full · 8640 px"; null when there is no photo. */
  fullEdge: number | null;
  /** The folder as the panel shows it: the chosen one, else the session's default. */
  destination: string;
  /** The file names the template gives, in order (the core's names replace them as files are reached). */
  names: string[];
  /** The first name, for the line under the template box. */
  example: string;
  /** Why the template can't be used; null when it is fine. */
  problem: string | null;
  /** A remark about the template that does not stop the export. */
  warning: string | null;
}

/** The look a photo will be exported with: its own when its settings are known, else the default. */
function lookIdFor(photo: PhotoSummary, input: ExportModelInput): string {
  const known = input.settings[photo.id]?.settings.look;
  if (known) return known;
  return input.looks?.find((l) => l.is_default)?.id ?? input.preferences?.default_look ?? "builtin:neutral";
}

const slugOfLookId = (id: string): string => id.slice(id.lastIndexOf(":") + 1);

/** The model for a scope and options. */
export function exportModel(input: ExportModelInput, scope: ApplyScope, options: ExportOptions): ExportModel {
  const { session } = input;
  const photos = photosInScope(session, scope);
  const nPicked = session?.photos.filter((p) => p.picked && isExportable(p)).length ?? 0;
  const nAll = session?.photos.filter(isExportable).length ?? 0;
  const destination = options.destination ?? session?.default_export_folder ?? FALLBACK_DESTINATION;
  const parts = (photo: PhotoSummary, index: number): TemplateParts => ({
    name: photoName(photo.file_name),
    look: slugOfLookId(lookIdFor(photo, input)),
    date: captureDate(photo) || UNDATED,
    seq: index + 1,
    camera: slugify(photo.camera.body),
  });
  const problem = templateProblem(options.name_template);
  const names = problem ? [] : photos.map((p, i) => previewFileName(options.name_template, parts(p, i), options.format));
  return {
    photos,
    count: photos.length,
    nPicked,
    nAll,
    included: new Set(photos.map((p) => p.id)),
    fullEdge: fullLongEdge(photos) ?? fullLongEdge(session?.photos.filter(isExportable) ?? []),
    destination,
    names,
    example: names[0] ?? "",
    problem,
    warning: problem ? null : templateWarning(options.name_template, photos.length),
  };
}
