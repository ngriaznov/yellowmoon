/**
 * Which photos an export covers, the options it starts from, and the job it sends
 * (`docs/09-export.md` §Options). Pure functions over the session, so the rules are tested without a screen.
 */
import type { ApplyScope, ExportJob, ExportOptions, PhotoSummary, SessionState } from "../api/types";

/** The panel's options before anything is remembered: `docs/09` §Options defaults. */
export function defaultExportOptions(): ExportOptions {
  return {
    format: "jpeg",
    jpeg_quality: 92,
    bit_depth: 16,
    tiff_compression: "lzw",
    color_space: "srgb",
    size: { kind: "full" },
    destination: null,
    name_template: "{name}_{look}",
    if_exists: "keep_both",
  };
}

/** A file that can't be developed can't be exported. */
export const isExportable = (photo: Pick<PhotoSummary, "status">): boolean => photo.status.kind !== "skipped";

/** The photos a scope covers, in filmstrip order: the current one, the picked ones, or every photo that can be exported. */
export function photosInScope(session: SessionState | null, scope: ApplyScope): PhotoSummary[] {
  if (!session) return [];
  const photos = session.photos;
  if (scope === "this") {
    const current = session.current_index === null ? undefined : photos[session.current_index];
    return current && isExportable(current) ? [current] : [];
  }
  if (scope === "picked") return photos.filter((p) => p.picked && isExportable(p));
  return photos.filter(isExportable);
}

/** The scope the panel opens with: Picked when anything is picked, else This photo (`docs/09`). */
export function defaultScope(session: SessionState | null): ApplyScope {
  return session && session.photos.some((p) => p.picked) ? "picked" : "this";
}

/** The longest side of the biggest photo in a scope, for "Full · 8640 px" and the Long edge slider. Null for an empty scope. */
export function fullLongEdge(photos: readonly PhotoSummary[]): number | null {
  if (photos.length === 0) return null;
  return Math.max(...photos.map((p) => Math.max(p.width_px, p.height_px)));
}

/** The job the core is asked to estimate, preview and run. */
export function buildJob(scope: ApplyScope, options: ExportOptions): ExportJob {
  return { scope, options };
}

/** Whether two sets of options are the same, for deciding whether to remember them. */
export function sameOptions(a: ExportOptions, b: ExportOptions): boolean {
  return JSON.stringify(a) === JSON.stringify(b);
}
