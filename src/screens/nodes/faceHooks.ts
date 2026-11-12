/**
 * The two places the Face Refinement interface waits on the core (`docs/27` S-11c). Each is one function, so that wiring the core's side is a change here and
 * nowhere else:
 *
 * - **`fetchFacePicture`**: the round picture of one face in the Faces list. The core's `face_crop(photo_id, number)` will answer a picture address; until
 *   then there is none and the item shows its number on a neutral disc.
 * - **`setSkinAreasView`**: the eye (**Show skin areas**). The core's `MaskView.face` for `viewport_set_mode` will show where the tools work on the face; until
 *   then the eye only keeps its on and off state in the panel.
 */
import { useEffect, useState } from "react";
import type { NodeId, PhotoId } from "../../api/types";
import type { AppStore } from "../../state/store";

/** The address of the picture of face `number` of a photo as the Input has it (not retouched), or null while the core cannot make one. */
export function fetchFacePicture(_store: AppStore, _photoId: PhotoId, _number: number): Promise<string | null> {
  return Promise.resolve(null);
}

/** Turns the skin-areas view of face `number` of node `node` on or off (`on` false ends it). Nothing is sent until the core has the view. */
export function setSkinAreasView(_store: AppStore, _photoId: PhotoId, _node: NodeId, _number: number, _on: boolean): void {}

/** The picture of one face for its item in the list: the address once `fetchFacePicture` has one, else undefined. It is asked again when the face or the photo changes. */
export function useFacePicture(store: AppStore, photoId: PhotoId, number: number, key: string): string | undefined {
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
