import { useEffect } from "react";
import type { FaceState, PhotoId } from "../../api/types";
import { useAppState } from "../../state/context";
import type { AppStore } from "../../state/store";

/**
 * Where the faces of a photo are (`face_state`), for the panel of a Face Refinement node: what the store has heard (`undefined` before anything), and a question to
 * the core when the node's panel opens, so that a worker that ended before the page listened is not left on "Finding faces".
 */
export function useFaces(store: AppStore, photoId: PhotoId | null, on: boolean): FaceState | undefined {
  useEffect(() => {
    if (on && photoId !== null) void store.refreshFaces(photoId);
  }, [store, photoId, on]);
  return useAppState((s) => (photoId === null ? undefined : s.faces[photoId]));
}
