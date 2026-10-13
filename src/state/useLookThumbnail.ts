import { useCallback, useEffect, useRef, useState } from "react";
import type { LookId, PhotoId } from "../api/types";
import type { AppStore } from "./store";

/**
 * How many times a tile asks again for one picture after a failure before it stays blank: once
 * for the request the core cancelled when the person left the photo and came back, once for a
 * picture the shell had already thrown away. A tile whose pictures keep failing must not loop.
 */
export const MAX_TILE_RETRIES = 2;

/** What a look tile needs to draw its picture. */
export interface LookThumbnail {
  /** The address of the picture, once it is known. */
  src: string | undefined;
  /** For the `error` event of the `<img>` that shows `src`. */
  onImageError: () => void;
}

/** True for a refusal that asking again cannot change (the look's file is gone, the look is not in the library). */
function isFinal(error: unknown): boolean {
  return typeof error === "object" && error !== null && (error as { scope?: unknown }).scope === "looks";
}

/**
 * The picture of a photo through one look, for a tile of the Look card or of the wizard's last
 * step. The store makes each picture once per photo, look and `key` and hands the same request to
 * every tile that asks, which has two consequences handled here:
 *
 * - A request can fail for a reason that is not the tile's (the core cancels the tiles of a photo
 *   when the person moves to another one, and a tile that comes back meanwhile is handed that
 *   request). The store forgets a failed request; the tile asks again, up to {@link MAX_TILE_RETRIES}
 *   times, unless it has left (it is not drawn any more, or its photo or settings changed) or the
 *   answer is final.
 * - An address can stop answering (the shell keeps a bounded amount of tiles and drops the oldest
 *   photos' pictures, so an address from a photo visited a while ago gives 404). When the `<img>`
 *   reports an error the store forgets that address and the tile asks for a new picture, within the
 *   same allowance.
 *
 * `enabled` false (a tile not yet scrolled into view, a look whose file is missing) asks for nothing.
 */
export function useLookThumbnail(store: AppStore, photoId: PhotoId | null, lookId: LookId, key: string, enabled = true): LookThumbnail {
  const [src, setSrc] = useState<string | undefined>();
  // Bumped to ask again.
  const [round, setRound] = useState(0);
  // The allowance belongs to one picture (photo, look and settings), and is renewed for the next.
  const spent = useRef({ picture: "", count: 0 });

  useEffect(() => {
    if (!enabled || photoId === null) return;
    const picture = `${photoId} ${lookId} ${key}`;
    if (spent.current.picture !== picture) spent.current = { picture, count: 0 };
    let live = true;
    store.lookThumbnail(photoId, lookId, key).then(
      (url) => {
        if (live) setSrc(url);
      },
      (error: unknown) => {
        if (!live || isFinal(error) || spent.current.count >= MAX_TILE_RETRIES) return;
        spent.current.count += 1;
        setRound((n) => n + 1);
      },
    );
    return () => {
      live = false;
    };
  }, [store, enabled, photoId, lookId, key, round]);

  const onImageError = useCallback(() => {
    if (src === undefined || photoId === null) return;
    store.forgetLookThumbnail(photoId, lookId, key, src);
    setSrc(undefined);
    if (spent.current.count >= MAX_TILE_RETRIES) return;
    spent.current.count += 1;
    setRound((n) => n + 1);
  }, [store, src, photoId, lookId, key]);

  return { src, onImageError };
}
