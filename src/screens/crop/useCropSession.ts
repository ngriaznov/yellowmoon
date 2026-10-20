import { useCallback, useRef, useState } from "react";
import type { CropAspect, Orientation, PhotoId, PhotoSummary, ResolvedSettings } from "../../api/types";
import type { AppStore } from "../../state/store";
import { fitRatio, FULL_RECT, imageAspectOf, isFullRect, presetRatio, rotateRect90 } from "./cropMath";
import type { Rect } from "./cropMath";

/** What the crop panel is editing. Nothing reaches the settings as a crop until Done. */
export interface CropDraft {
  rect: Rect;
  angle_deg: number;
  aspect: CropAspect;
  orientation: Orientation;
}

/** The crop session: the draft and the actions of the panel and the frame over the photo. */
export interface CropSession {
  draft: CropDraft;
  /** Width ÷ height of the whole photo as it is turned now. */
  imageAspect: number;
  /** The pixel ratio the frame is locked to, or null for Free. */
  lockedRatio: number | null;
  setRect: (rect: Rect) => void;
  setAspect: (aspect: CropAspect) => void;
  setAngle: (degrees: number) => void;
  rotate: () => void;
  reset: () => void;
  done: () => void;
  cancel: () => void;
}

const normalise = (degrees: number): Orientation => ((((Math.round(degrees / 90) * 90) % 360) + 360) % 360) as Orientation;

/** Turns a draft into the `crop` setting: nothing when it is the whole photo, straight. */
export function draftToCrop(draft: CropDraft): ResolvedSettings["settings"]["crop"] {
  if (isFullRect(draft.rect) && Math.abs(draft.angle_deg) < 1e-9) return null;
  return { rect: draft.rect, angle_deg: draft.angle_deg, aspect: draft.aspect };
}

/**
 * Starts and ends a crop (`docs/11-ui.md` §Crop). While it is open the core is told only the
 * straighten angle and the orientation, with the frame set to the whole photo, so the viewport
 * keeps showing the whole (turned, straightened) photo while the frame is drawn over it by the
 * webview. The frame itself is sent with Done, together with the angle and the orientation, as
 * one change that undo takes back in one step. Cancel puts the old crop and orientation back.
 */
export function useCropSession(
  store: AppStore,
  photo: PhotoSummary | null,
  rs: ResolvedSettings | null,
): { session: CropSession | null; start: () => void; active: boolean } {
  const [draft, setDraft] = useState<CropDraft | null>(null);
  const original = useRef<{ crop: ResolvedSettings["settings"]["crop"]; orientation: Orientation } | null>(null);
  const touched = useRef(false);
  /** The photo the crop was started on: its changes go there, wherever the person is when they are sent. */
  const target = useRef<PhotoId | undefined>(undefined);
  const latest = useRef(draft);
  latest.current = draft;

  const push = useCallback(
    (next: CropDraft) => {
      touched.current = true;
      // Not final: the whole session is one gesture, ended by Done or Cancel.
      void store.update({ crop: { rect: FULL_RECT, angle_deg: next.angle_deg, aspect: next.aspect }, orientation: next.orientation }, false, target.current);
    },
    [store],
  );

  const start = useCallback(() => {
    if (!rs) return;
    const crop = rs.settings.crop;
    original.current = { crop, orientation: rs.settings.orientation };
    target.current = rs.photo_id;
    touched.current = false;
    setDraft({
      rect: crop?.rect ?? FULL_RECT,
      angle_deg: crop?.angle_deg ?? 0,
      aspect: crop?.aspect ?? "original",
      orientation: rs.settings.orientation,
    });
  }, [rs]);

  if (!draft || !photo) return { session: null, start, active: false };

  const imageAspect = imageAspectOf(photo.width_px, photo.height_px, draft.orientation);
  const lockedRatio = presetRatio(draft.aspect, imageAspect, draft.rect.width * imageAspect >= draft.rect.height);

  const withAspect = (rect: Rect, aspect: CropAspect, orientation: Orientation): Rect => {
    const aspectOfImage = imageAspectOf(photo.width_px, photo.height_px, orientation);
    const ratio = presetRatio(aspect, aspectOfImage, rect.width * aspectOfImage >= rect.height);
    return ratio === null ? rect : fitRatio(rect, ratio, aspectOfImage);
  };

  const session: CropSession = {
    draft,
    imageAspect,
    lockedRatio,
    setRect: (rect) => setDraft({ ...draft, rect }),
    setAspect: (aspect) => {
      const next = { ...draft, aspect, rect: withAspect(draft.rect, aspect, draft.orientation) };
      setDraft(next);
      push(next);
    },
    setAngle: (angle_deg) => {
      const next = { ...draft, angle_deg };
      setDraft(next);
      push(next);
    },
    rotate: () => {
      const orientation = normalise(draft.orientation + 90);
      const turned = rotateRect90(draft.rect);
      const next = { ...draft, orientation, rect: withAspect(turned, draft.aspect, orientation) };
      setDraft(next);
      push(next);
    },
    reset: () => {
      const next: CropDraft = { rect: FULL_RECT, angle_deg: 0, aspect: "original", orientation: 0 };
      setDraft(next);
      push(next);
    },
    done: () => {
      const current = latest.current;
      if (!current) return;
      setDraft(null);
      void store.update({ crop: draftToCrop(current), orientation: current.orientation }, true, target.current);
    },
    cancel: () => {
      const before = original.current;
      setDraft(null);
      if (before && touched.current) void store.update({ crop: before.crop, orientation: before.orientation }, true, target.current);
    },
  };
  return { session, start, active: true };
}
