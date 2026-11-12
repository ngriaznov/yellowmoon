/**
 * The two places the Face Refinement interface talks to the core beyond settings (`docs/27` S-11c):
 *
 * - **`fetchFacePicture`**: the round picture of one face in the Faces list, from the core's `face_crop(photo_id, number)`. The address is kept per photo and face
 *   number until `face_state` changes; a failed fetch leaves the item on its neutral disc.
 * - **`setSkinAreasView`**: the eye (**Show skin areas**), the store's view state that sends `MaskView.face` with `viewport_set_mode`, as the mask eye does.
 */
import { useEffect, useState } from "react";
import type { NodeId, PhotoId } from "../../api/types";
import type { AppStore } from "../../state/store";

/** The addresses asked for, per `FaceState` the store holds: a new `face_state` is a new object, so its pictures are asked for again. */
const pictures = new WeakMap<object, Map<string, Promise<string | null>>>();

/** The address of the picture of face `number` of a photo as the Input has it (not retouched), or null when the core cannot make one. It never rejects. */
export function fetchFacePicture(store: AppStore, photoId: PhotoId, number: number): Promise<string | null> {
  const state = store.getState().faces[photoId];
  if (!state) return Promise.resolve(null);
  let kept = pictures.get(state);
  if (!kept) pictures.set(state, (kept = new Map()));
  const id = `${photoId}:${number}`;
  const had = kept.get(id);
  if (had) return had;
  const asked = store.api.faceCrop(photoId, number).then(
    (url) => url || null,
    () => null,
  );
  kept.set(id, asked);
  // A failure is not kept: the next time the item is shown it asks again.
  void asked.then((url) => {
    if (url === null && kept.get(id) === asked) kept.delete(id);
  });
  return asked;
}

/** Turns the skin-areas view of face `number` of node `node` on or off (`on` false ends it). It is the store's view state, so it ends with the node, the photo and Before. */
export function setSkinAreasView(store: AppStore, _photoId: PhotoId, _node: NodeId, number: number, on: boolean): void {
  if (on) store.setSkinAreas(number);
  else if (store.getState().skinAreas !== null) store.setSkinAreas(null);
}

/** The picture of one face for its item in the list: the address once `fetchFacePicture` has one, else undefined. It is asked again when the face or the photo changes. */
export function useFacePicture(store: AppStore, photoId: PhotoId, number: number, key: unknown): string | undefined {
  const [src, setSrc] = useState<string | undefined>();
  useEffect(() => {
    let live = true;
    setSrc(undefined);
    fetchFacePicture(store, photoId, number).then(
      (url) => {
        if (live && url) setSrc(url);
      },
      () => {},
    );
    return () => {
      live = false;
    };
  }, [store, photoId, number, key]);
  return src;
}
