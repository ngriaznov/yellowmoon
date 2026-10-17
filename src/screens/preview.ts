/**
 * The picture a browser build of the photo cell shows (the app shows the native viewport
 * instead). It is the mockup's flat-shape photo, adjusted and graded the way the mock API
 * grades its thumbnails (`fixtures/grade.ts`), so a change in a control is visible. Photos that
 * aren't one of the twelve fixtures have no picture.
 */
import type { PhotoSummary, ResolvedSettings } from "../api/types";
import { developScene, flatScene, previewDataUrl } from "../fixtures/grade";
import { SCENES } from "../fixtures/scenes";
import { photoName } from "../state/photo";

/** The last part of a look id: "arri:3000:film-warm" gives "film-warm". */
export const lookSlugOf = (id: string): string => id.slice(id.lastIndexOf(":") + 1);

/** A data URL for the photo cell, or undefined when there is no fixture for the photo. */
export function previewUrl(photo: PhotoSummary, rs: ResolvedSettings | null, before: boolean): string | undefined {
  const scene = SCENES.find((s) => s.name === photoName(photo.file_name));
  if (!scene) return undefined;
  const geometry = rs ? { orientation: rs.values.orientation, angle_deg: rs.values.crop?.angle_deg ?? 0, crop: rs.values.crop?.rect ?? null } : {};
  if (!rs || before) return previewDataUrl(before ? flatScene(scene) : scene, geometry);
  const { values, estimates } = rs;
  const graded = developScene(
    scene,
    { slug: lookSlugOf(values.look), intensity: values.look_intensity },
    {
      ev: values.exposure_ev,
      autoEv: estimates.exposure_ev,
      kelvin: values.kelvin,
      autoKelvin: estimates.kelvin,
      tint: values.tint,
      autoTint: estimates.tint,
      warp: values.color_warp.pins,
    },
  );
  return previewDataUrl(graded, geometry);
}
