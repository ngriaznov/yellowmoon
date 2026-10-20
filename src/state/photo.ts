/** Small helpers that turn a photo's summary into the words and pictures the screens show. */
import type { PhotoId, PhotoStatus, PhotoSummary, SessionState } from "../api/types";

/** "DSC04204.ARW" gives "DSC04204". */
export const photoName = (fileName: string): string => fileName.replace(/\.[^./\\]+$/, "");

/** "1/125" for 0.008 s, "2 s" for two seconds, "1/2" for half a second. */
export function shutterLabel(seconds: number): string {
  if (!(seconds > 0)) return "";
  if (seconds >= 1) return `${Number.isInteger(seconds) ? seconds : Math.round(seconds * 10) / 10} s`;
  return `1/${Math.round(1 / seconds)}`;
}

/**
 * The camera line under the photo: "Sony A1 · DSC04204 · ISO 400 · 1/125 · f/2.8". The Export screen leaves the body out
 * (`body: false`), as its mockup does: "DSC04204 · ISO 400 · 1/125 · f/2.8".
 */
export function cameraLine(photo: PhotoSummary, { body = true }: { body?: boolean } = {}): string {
  const aperture = String(Math.round(photo.aperture_f * 10) / 10);
  return [body ? photo.camera.body : "", photoName(photo.file_name), `ISO ${photo.iso}`, shutterLabel(photo.shutter_s), `f/${aperture}`]
    .filter(Boolean)
    .join(" · ");
}

/** The capture date of a photo, "2026-10-04"; empty when the file has none (the core sends "" for a file it could not probe). */
export const captureDate = (photo: PhotoSummary): string => photo.captured_at.slice(0, 10);

/**
 * "2026-10-04 · 12 photos" for the folder cell: the date of the first photo that has one, and the count. Without any date it is
 * just "12 photos".
 */
export function folderSubline(session: Pick<SessionState, "photos">): string {
  const count = session.photos.length;
  const noun = count === 1 ? "photo" : "photos";
  if (count === 0) return "0 photos";
  const date = session.photos.map(captureDate).find((d) => d !== "");
  return date ? `${date} · ${count} ${noun}` : `${count} ${noun}`;
}

/**
 * The session after files were added to one that is already open. `snapshot` is what the shell answered `open_paths` or `open_dialog`
 * with: taken a little before the answer is handled, so about the photos that were open it is older than what the page holds, which
 * events, clicks and picks have been changing since. The core only appends, so the page's photos keep their places and their entries
 * (status, picked) and the current photo stays; the snapshot supplies the photos that are new, the folder name and the default export
 * folder. A status that changed after the snapshot comes as an event, which a snapshot taken earlier would otherwise undo for good.
 */
export function mergeSession(old: SessionState, snapshot: SessionState): SessionState {
  const known = new Set(old.photos.map((p) => p.id));
  const photos = [...old.photos, ...snapshot.photos.filter((p) => !known.has(p.id))];
  return {
    ...snapshot,
    photos,
    current_index: old.current_index ?? snapshot.current_index,
    picked: photos.filter((p) => p.picked).map((p) => p.id),
  };
}

/**
 * The session after photos were taken out of the filmstrip (`remove_photos`). The page's own photos stay as they are, entries (status,
 * picked) included, minus the removed ones: the answer's snapshot is older than what events have told the page since, and replacing
 * the page's entries with it would bring back a status that has changed. `current` is the photo on screen afterwards (the photo that
 * was, unless it left; then the shell's choice, the next photo or the previous one); with none left the session is empty.
 */
export function withoutPhotos(old: SessionState, removed: ReadonlySet<PhotoId>, current: PhotoId | null): SessionState {
  const photos = old.photos.filter((p) => !removed.has(p.id));
  const index = current === null ? -1 : photos.findIndex((p) => p.id === current);
  return {
    ...old,
    photos,
    current_index: photos.length === 0 ? null : Math.max(0, index),
    picked: photos.filter((p) => p.picked).map((p) => p.id),
  };
}

/**
 * The session after photos were put back (`restore_photos`). The order is the shell's (the photos have their old places again), the
 * entries of the photos that were put back are the shell's (it knows their status and picks) and the entries of the others are the
 * page's own, as in {@link mergeSession}. `current` is the photo on screen afterwards.
 */
export function withRestoredPhotos(old: SessionState, snapshot: SessionState, restored: ReadonlySet<PhotoId>, current: PhotoId | null): SessionState {
  const mine = new Map(old.photos.map((p) => [p.id, p]));
  const inSnapshot = new Set(snapshot.photos.map((p) => p.id));
  const photos = [...snapshot.photos.map((p) => (restored.has(p.id) ? p : (mine.get(p.id) ?? p))), ...old.photos.filter((p) => !inSnapshot.has(p.id))];
  const index = current === null ? -1 : photos.findIndex((p) => p.id === current);
  return {
    ...snapshot,
    opened: undefined,
    photos,
    current_index: photos.length === 0 ? null : Math.max(0, index),
    picked: photos.filter((p) => p.picked).map((p) => p.id),
  };
}

/** What Photos says after a removal: "Removed DSC04204. The file is still on disk." or "Removed 3 photos. The files are still on disk." */
export function removedNotice(names: readonly string[]): string {
  return names.length === 1 ? `Removed ${names[0]}. The file is still on disk.` : `Removed ${names.length} photos. The files are still on disk.`;
}

/** What Photos says after Undo: "Put back DSC04204." or "Put back 3 photos.", and how many stayed out because their file was added again. */
export function restoredNotice(names: readonly string[], alreadyOpen: number): string {
  const put = names.length === 1 ? `Put back ${names[0]}.` : `Put back ${names.length} photos.`;
  return alreadyOpen === 0 ? put : `${put} ${alreadyOpen} ${alreadyOpen === 1 ? "was" : "were"} already open.`;
}

/** True when a photo can be shown and edited. */
export const isReady = (photo: Pick<PhotoSummary, "status">): boolean => photo.status.kind === "ready";

/**
 * How far a photo is on its way, 0–1, for the Start screen's progress bar. The stages are weighted
 * so that a photo that has been decoded counts for a fifth, one that has its estimates for two
 * fifths, and a ready or skipped one for all of it.
 */
export function developFraction(status: PhotoStatus): number {
  switch (status.kind) {
    case "waiting":
      return 0;
    case "developing":
      return { probe: 0.1, decode: 0.2, estimate: 0.4, thumbnail: 0.8 }[status.stage];
    case "ready":
    case "skipped":
      return 1;
  }
}

/**
 * A PNG's bytes as an image URL: an object URL where the browser has them, else a data URL. Only the mock API makes pictures this
 * way; the app's shell keeps its pictures and answers with `ym-frame://` addresses (thumbnails, look tiles, viewport frames).
 */
export function bytesToUrl(bytes: readonly number[]): string {
  const data = Uint8Array.from(bytes);
  if (typeof URL !== "undefined" && typeof URL.createObjectURL === "function") {
    return URL.createObjectURL(new Blob([data], { type: "image/png" }));
  }
  let binary = "";
  for (let i = 0; i < data.length; i += 8192) binary += String.fromCharCode(...data.subarray(i, i + 8192));
  return `data:image/png;base64,${btoa(binary)}`;
}

/** Frees a URL made by `bytesToUrl`; any other address (the shell's `ym-frame://` ones) is left alone. */
export function releaseUrl(url: string): void {
  if (url.startsWith("blob:") && typeof URL !== "undefined" && typeof URL.revokeObjectURL === "function") URL.revokeObjectURL(url);
}
