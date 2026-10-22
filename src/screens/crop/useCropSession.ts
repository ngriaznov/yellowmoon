import { useCallback, useEffect, useRef, useState } from "react";
import type { CropAspect, CropSuggestion, Orientation, PhotoId, PhotoSummary, ResolvedSettings } from "../../api/types";
import { isReady } from "../../state/photo";
import type { AppStore } from "../../state/store";
import { canvasOf, constrainFrame, fitInside, fitRatio, FULL_RECT, imageAspectOf, isFullRect, presetRatio, rotateRect90 } from "./cropMath";
import type { Canvas, Rect } from "./cropMath";

/** What the crop panel is editing. Nothing reaches the settings as a crop until Done. */
export interface CropDraft {
  /**
   * The frame as the person set it (the base). The frame shown, drawn and sent is this one fitted inside the straightened photo
   * ({@link shownFrame}): it shrinks as the angle grows and grows back to the base towards 0°.
   */
  rect: Rect;
  angle_deg: number;
  aspect: CropAspect;
  orientation: Orientation;
  /** Auto crop is on (`docs/20-auto-crop.md` §1): the angle and the frame follow the core's proposal for the aspect and orientation. */
  auto: boolean;
}

/**
 * What the Frame card's note says about Auto crop: a proposal on its way, the proposal in effect, or that none could be made (for
 * {@link AUTO_CROP_FAILED_MS}, with Auto off). Null: nothing.
 */
export type AutoCropNote = { kind: "asking" } | { kind: "answer"; suggestion: CropSuggestion } | { kind: "failed" } | null;

/** How long the note says that no frame could be suggested. */
export const AUTO_CROP_FAILED_MS = 4000;

/** The crop session: the draft and the actions of the panel and the frame over the photo. */
export interface CropSession {
  draft: CropDraft;
  /** The frame shown over the photo and sent with Done: the draft's frame fitted inside the photo straightened by its angle. */
  frame: Rect;
  /** Width ÷ height of the whole photo as it is turned now. */
  imageAspect: number;
  /** The pixel ratio the frame is locked to, or null for Free. */
  lockedRatio: number | null;
  /** Auto crop can't be asked for: the photo is still developing (as Crop itself can't start). */
  autoLocked: boolean;
  /** What the Frame card's note says about Auto crop. */
  autoNote: AutoCropNote;
  /**
   * A drag or an arrow key asks for this frame (worked out from {@link CropSession.frame}). It is kept inside the straightened photo
   * ({@link constrainFrame}) and becomes the draft's frame.
   */
  setRect: (rect: Rect) => void;
  setAspect: (aspect: CropAspect) => void;
  setAngle: (degrees: number) => void;
  rotate: () => void;
  reset: () => void;
  /** The Auto toggle and A: on asks the core for a proposal; off puts back the draft from just before it was turned on. */
  toggleAuto: () => void;
  done: () => void;
  cancel: () => void;
}

const normalise = (degrees: number): Orientation => ((((Math.round(degrees / 90) * 90) % 360) + 360) % 360) as Orientation;

/** The canvas of `photo` (its pixels) turned by `orientation`. */
const canvasFor = (photo: PhotoSummary, orientation: Orientation): Canvas => canvasOf(photo.width_px, photo.height_px, orientation);

/**
 * The frame a draft shows and sends: its frame fitted inside the photo straightened by its angle, on the canvas of its orientation, so
 * that the picture has no empty corner (`docs/11-ui.md` §Crop, the owner's decision of 2026-10-22). At 0° it is the draft's own.
 */
export function shownFrame(draft: CropDraft, photo: PhotoSummary): Rect {
  return fitInside(draft.rect, draft.angle_deg, canvasFor(photo, draft.orientation));
}

/**
 * Turns a draft into the `crop` setting: nothing when it is the whole photo, straight. While Auto crop is on the crop says so
 * (`auto: true`); off, the field is left out, as the core leaves `false` out (`docs/20-auto-crop.md` §2).
 */
export function draftToCrop(draft: CropDraft): ResolvedSettings["settings"]["crop"] {
  if (isFullRect(draft.rect) && Math.abs(draft.angle_deg) < 1e-9) return null;
  const crop = { rect: draft.rect, angle_deg: draft.angle_deg, aspect: draft.aspect };
  return draft.auto ? { ...crop, auto: true } : crop;
}

/**
 * Starts and ends a crop (`docs/11-ui.md` §Crop). While it is open the core is told only the
 * straighten angle and the orientation, with the frame set to the whole photo, so the viewport
 * keeps showing the whole (turned, straightened) photo while the frame is drawn over it by the
 * webview. The frame itself is sent with Done, together with the angle and the orientation, as
 * one change that undo takes back in one step. Cancel puts the old crop and orientation back.
 *
 * No border in photos (the owner's decision of 2026-10-22): the draft keeps the frame the person set (the base), and the frame shown,
 * drawn and sent is the base fitted inside the straightened photo ({@link shownFrame}). Straighten changes only the angle, so the shown
 * frame shrinks as the angle grows and grows back to the base towards 0°. A drag, a handle or an arrow key acts on the shown frame, is
 * kept inside ({@link constrainFrame}) and becomes the new base; an aspect, Rotate 90° and Reset set the base as before; Auto crop's
 * frame, inside already, is fitted all the same and becomes the base.
 *
 * Auto crop (`docs/20-auto-crop.md` §1): while the toggle is on, the angle and the frame follow the core's proposal for the draft's
 * aspect and orientation, asked again when either changes; moving the frame, a handle or Straighten, and Reset, turn it off. Only the
 * answer to the latest request counts, and only while Auto is still on in the open panel.
 */
export function useCropSession(
  store: AppStore,
  photo: PhotoSummary | null,
  rs: ResolvedSettings | null,
): { session: CropSession | null; start: () => void; active: boolean } {
  const [draft, setDraft] = useState<CropDraft | null>(null);
  const [autoNote, setAutoNote] = useState<AutoCropNote>(null);
  const original = useRef<{ crop: ResolvedSettings["settings"]["crop"]; orientation: Orientation } | null>(null);
  const touched = useRef(false);
  /** The photo the crop was started on: its changes go there, wherever the person is when they are sent. */
  const target = useRef<PhotoId | undefined>(undefined);
  const latest = useRef(draft);
  latest.current = draft;
  /** The photo as it is now, for an answer that arrives later (its canvas is what the answer's frame is fitted to). */
  const latestPhoto = useRef(photo);
  latestPhoto.current = photo;
  /** The draft from just before Auto was turned on in this panel; null when the panel opened with Auto on. */
  const beforeAuto = useRef<CropDraft | null>(null);
  /** Counts the requests for a proposal. An answer is used only when no request, Auto off or close came after it. */
  const request = useRef(0);
  const failedTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  /** Sets the draft, and at once what an answer arriving before the next render reads. */
  const commit = useCallback((next: CropDraft | null) => {
    latest.current = next;
    setDraft(next);
  }, []);

  const clearFailedTimer = useCallback(() => {
    if (failedTimer.current === null) return;
    clearTimeout(failedTimer.current);
    failedTimer.current = null;
  }, []);

  /** Auto turns off, or the panel closes: a proposal on its way is not used, and the note goes. */
  const forgetAuto = useCallback(() => {
    request.current += 1;
    clearFailedTimer();
    setAutoNote(null);
  }, [clearFailedTimer]);

  // An answer that arrives after the screen has gone is not used either.
  useEffect(
    () => () => {
      request.current += 1;
      if (failedTimer.current !== null) clearTimeout(failedTimer.current);
    },
    [],
  );

  const push = useCallback(
    (next: CropDraft) => {
      touched.current = true;
      // Not final: the whole session is one gesture, ended by Done or Cancel.
      void store.update({ crop: { rect: FULL_RECT, angle_deg: next.angle_deg, aspect: next.aspect }, orientation: next.orientation }, false, target.current);
    },
    [store],
  );

  /**
   * Asks the core for its proposal at an aspect and orientation. The answer sets the draft's angle (told to the core, as Straighten
   * does) and its frame (the first, best one); a failure turns Auto off and says so for {@link AUTO_CROP_FAILED_MS}.
   */
  const ask = useCallback(
    (aspect: CropAspect, orientation: Orientation) => {
      const id = target.current;
      if (id === undefined) return;
      request.current += 1;
      const asked = request.current;
      clearFailedTimer();
      setAutoNote({ kind: "asking" });
      void store.suggestCrop(id, aspect, orientation).then((answer) => {
        const current = latest.current;
        if (asked !== request.current || !current?.auto) return;
        if (!answer) {
          request.current += 1;
          commit({ ...current, auto: false });
          setAutoNote({ kind: "failed" });
          failedTimer.current = setTimeout(() => {
            failedTimer.current = null;
            setAutoNote((note) => (note?.kind === "failed" ? null : note));
          }, AUTO_CROP_FAILED_MS);
          return;
        }
        // The core's frame is inside already; the fit is a guard (a whole-photo frame at 0° stays the whole photo).
        const proposed = answer.frames[0]?.rect ?? current.rect;
        const summary = latestPhoto.current;
        const rect = summary ? fitInside(proposed, answer.angle_deg, canvasFor(summary, current.orientation)) : proposed;
        const next: CropDraft = { ...current, rect, angle_deg: answer.angle_deg };
        commit(next);
        if (next.angle_deg !== current.angle_deg) push(next);
        setAutoNote({ kind: "answer", suggestion: answer });
      });
    },
    [store, push, commit, clearFailedTimer],
  );

  const start = useCallback(() => {
    if (!rs) return;
    const crop = rs.settings.crop;
    original.current = { crop, orientation: rs.settings.orientation };
    target.current = rs.photo_id;
    touched.current = false;
    beforeAuto.current = null;
    forgetAuto();
    const next: CropDraft = {
      rect: crop?.rect ?? FULL_RECT,
      angle_deg: crop?.angle_deg ?? 0,
      aspect: crop?.aspect ?? "original",
      orientation: rs.settings.orientation,
      auto: crop?.auto === true,
    };
    // A stored crop that pokes past the straightened photo (an older crop, or one from a settings file) is the base: the panel shows and
    // sends its fit, and Cancel leaves the stored one as it was.
    commit(next);
    // A crop made by Auto opens with Auto on, and the proposal is asked for again to fill the note (it is the same, docs/20 §9).
    if (next.auto) ask(next.aspect, next.orientation);
  }, [rs, ask, commit, forgetAuto]);

  if (!draft || !photo) return { session: null, start, active: false };

  const imageAspect = imageAspectOf(photo.width_px, photo.height_px, draft.orientation);
  const frame = shownFrame(draft, photo);
  const lockedRatio = presetRatio(draft.aspect, imageAspect, frame.width * imageAspect >= frame.height);
  const autoLocked = !isReady(photo);

  const withAspect = (rect: Rect, aspect: CropAspect, orientation: Orientation): Rect => {
    const aspectOfImage = imageAspectOf(photo.width_px, photo.height_px, orientation);
    const ratio = presetRatio(aspect, aspectOfImage, rect.width * aspectOfImage >= rect.height);
    return ratio === null ? rect : fitRatio(rect, ratio, aspectOfImage);
  };

  /** Ends the panel: nothing more of Auto arrives. */
  const close = (): void => {
    forgetAuto();
    commit(null);
  };

  const session: CropSession = {
    draft,
    frame,
    imageAspect,
    lockedRatio,
    autoLocked,
    autoNote,
    // A drag or an arrow key: what was moved stays and Auto turns off. A drag keeps the callbacks of the render it began in, so the
    // draft is read as it is now. The frame asked for is kept inside the straightened photo, going from the frame shown now (inside)
    // as far towards it as it can (constrainFrame: the closed form of a bisection between the two), and becomes the base. At 0° every
    // frame on the photo is inside, so nothing changes there.
    setRect: (rect) => {
      const current = latest.current ?? draft;
      if (current.auto) forgetAuto();
      const canvas = canvasFor(photo, current.orientation);
      const from = fitInside(current.rect, current.angle_deg, canvas);
      const aspectNow = imageAspectOf(photo.width_px, photo.height_px, current.orientation);
      const locked = presetRatio(current.aspect, aspectNow, from.width * aspectNow >= from.height) !== null;
      commit({ ...current, rect: constrainFrame(from, rect, current.angle_deg, canvas, locked), auto: false });
    },
    setAspect: (aspect) => {
      const next = { ...draft, aspect, rect: withAspect(draft.rect, aspect, draft.orientation) };
      commit(next);
      push(next);
      if (next.auto) ask(aspect, next.orientation);
    },
    setAngle: (angle_deg) => {
      const current = latest.current ?? draft;
      if (current.auto) forgetAuto();
      const next = { ...current, angle_deg, auto: false };
      commit(next);
      push(next);
    },
    rotate: () => {
      const orientation = normalise(draft.orientation + 90);
      const turned = rotateRect90(draft.rect);
      const next = { ...draft, orientation, rect: withAspect(turned, draft.aspect, orientation) };
      commit(next);
      push(next);
      if (next.auto) ask(next.aspect, orientation);
    },
    reset: () => {
      if (draft.auto) forgetAuto();
      const next: CropDraft = { rect: FULL_RECT, angle_deg: 0, aspect: "original", orientation: 0, auto: false };
      commit(next);
      push(next);
    },
    toggleAuto: () => {
      if (autoLocked) return;
      if (!draft.auto) {
        beforeAuto.current = draft;
        const next = { ...draft, auto: true };
        commit(next);
        ask(next.aspect, next.orientation);
        return;
      }
      forgetAuto();
      // Back to the draft from before Auto was turned on here; when the panel opened with Auto on, the whole photo, straight, in the
      // aspect chosen now.
      const before = beforeAuto.current;
      const next: CropDraft = before
        ? { ...before, auto: false }
        : { ...draft, rect: withAspect(FULL_RECT, draft.aspect, draft.orientation), angle_deg: 0, auto: false };
      commit(next);
      push(next);
    },
    // The frame sent is the one shown: the base fitted inside the straightened photo. The whole photo at 0° is still no crop.
    done: () => {
      const current = latest.current;
      if (!current) return;
      close();
      const sent = { ...current, rect: shownFrame(current, photo) };
      void store.update({ crop: draftToCrop(sent), orientation: current.orientation }, true, target.current);
    },
    cancel: () => {
      const before = original.current;
      close();
      if (before && touched.current) void store.update({ crop: before.crop, orientation: before.orientation }, true, target.current);
    },
  };
  return { session, start, active: true };
}
