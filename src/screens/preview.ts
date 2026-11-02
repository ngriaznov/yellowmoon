/**
 * The picture a browser build of the photo cell shows (the app shows the native viewport
 * instead). It is the mockup's flat-shape photo, adjusted and graded the way the mock API
 * grades its thumbnails (`fixtures/grade.ts`), so a change in a control is visible. Photos that
 * aren't one of the twelve fixtures have no picture.
 */
import type { MaskView, PhotoSummary, ResolvedSettings } from "../api/types";
import { developScene, flatScene, previewDataUrl } from "../fixtures/grade";
import type { PreviewOptions } from "../fixtures/grade";
import { maskViewSvg } from "../fixtures/maskView";
import { nodeOf } from "../state/settingsGraph";
import { SCENES } from "../fixtures/scenes";
import { photoName } from "../state/photo";

/** The last part of a look id: "arri:3000:film-warm" gives "film-warm". */
export const lookSlugOf = (id: string): string => id.slice(id.lastIndexOf(":") + 1);

/**
 * A data URL for the photo cell, or undefined when there is no fixture for the photo. With `mask` (the mask view the page asked the core
 * for, `docs/23` §5.8) the mock's own picture carries it, drawn over the scene as `fixtures/maskView.ts` does; never in the Before view.
 */
export function previewUrl(photo: PhotoSummary, rs: ResolvedSettings | null, before: boolean, mask: MaskView | null = null): string | undefined {
  const scene = SCENES.find((s) => s.name === photoName(photo.file_name));
  if (!scene) return undefined;
  const node = mask && rs && !before ? nodeOf(rs.settings, mask.node) : undefined;
  const over: PreviewOptions["over"] = mask && node?.kind === "corrector" ? (box) => maskViewSvg(mask.style, node.masks, box) : undefined;
  const geometry: PreviewOptions = rs
    ? { orientation: rs.values.input.orientation, angle_deg: rs.values.input.crop?.angle_deg ?? 0, crop: rs.values.input.crop?.rect ?? null, over }
    : {};
  if (!rs || before) return previewDataUrl(before ? flatScene(scene) : scene, geometry);
  const { values, estimates } = rs;
  const graded = developScene(
    scene,
    { slug: lookSlugOf(values.output.look), intensity: values.output.look_intensity },
    {
      ev: values.input.exposure_ev,
      autoEv: estimates.input.exposure_ev,
      kelvin: values.input.kelvin,
      autoKelvin: estimates.input.kelvin,
      tint: values.input.tint,
      autoTint: estimates.input.tint,
      warp: values.main.grade.color_warp.pins,
    },
  );
  return previewDataUrl(graded, geometry);
}
