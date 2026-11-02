/**
 * The picture a browser build of the photo cell shows (the app shows the native viewport
 * instead). It is the mockup's flat-shape photo, adjusted and graded the way the mock API
 * grades its thumbnails (`fixtures/grade.ts`), so a change in a control is visible. Photos that
 * aren't one of the twelve fixtures have no picture.
 */
import type { MaskKind, MaskView, PhotoSummary, ResolvedSettings } from "../api/types";
import { developScene, flatScene, previewDataUrl } from "../fixtures/grade";
import type { PreviewOptions } from "../fixtures/grade";
import { maskViewSvg, turnMatrix, visionFoundIn } from "../fixtures/maskView";
import { nodeOf } from "../state/settingsGraph";
import { SCENES, SVG_HEIGHT, SVG_WIDTH } from "../fixtures/scenes";
import { photoName } from "../state/photo";

/** The last part of a look id: "arri:3000:film-warm" gives "film-warm". */
export const lookSlugOf = (id: string): string => id.slice(id.lastIndexOf(":") + 1);

/**
 * What the page knows about Subject and People, for the browser build's picture (the core draws the real masks in the app, `docs/23` §4.5): the kinds this
 * computer can make (`mask_kinds`; null while they are not known, which draws as supported) and whether the core is making the masks now
 * (`viewport_busy.finding`: a viewer never waits, so the rows select nothing until they are made).
 */
export interface VisionContext {
  kinds: readonly MaskKind[] | null;
  finding: boolean;
}

/**
 * A data URL for the photo cell, or undefined when there is no fixture for the photo. With `mask` (the mask view the page asked the core
 * for, `docs/23` §5.8) the mock's own picture carries it, drawn over the scene as `fixtures/maskView.ts` does; never in the Before view.
 * Subject and People rows are drawn from what the flat scene "has" (the far hill, a figure in some) where `vision` says this computer makes them and
 * the core is not making them now.
 */
export function previewUrl(
  photo: PhotoSummary,
  rs: ResolvedSettings | null,
  before: boolean,
  mask: MaskView | null = null,
  vision: VisionContext = { kinds: null, finding: false },
): string | undefined {
  const scene = SCENES.find((s) => s.name === photoName(photo.file_name));
  if (!scene) return undefined;
  const node = mask && rs && !before ? nodeOf(rs.settings, mask.node) : undefined;
  // The shapes live in the photo frame: the picture draws the photo turned and cropped, so they are carried by the same turn (the crop is the box).
  const turn = rs?.values.input.orientation ?? 0;
  const inScene = visionFoundIn(scene.name);
  const found = {
    subject: inScene.subject && (vision.kinds?.includes("subject") ?? true) && !vision.finding,
    people: inScene.people && (vision.kinds?.includes("people") ?? true) && !vision.finding,
  };
  const over: PreviewOptions["over"] =
    mask && node?.kind === "corrector"
      ? (box) => maskViewSvg(mask.style, node.masks, box, { width: SVG_WIDTH, height: SVG_HEIGHT, matrix: turnMatrix(turn, SVG_WIDTH, SVG_HEIGHT) }, found)
      : undefined;
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
