import { useCallback, useEffect, useRef, useState } from "react";
import type { LookId, PhotoId } from "../api/types";
import { isSuperseded } from "./requestLimiter";
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

/** True for a refusal that asking again cannot change, of the scope given (the look's file is gone, the preset is not saved any more). */
function isFinalIn(scope: string, error: unknown): boolean {
  return typeof error === "object" && error !== null && (error as { scope?: unknown }).scope === scope;
}

/**
 * The picture of a tile that is made by the core and kept by the store (the look tiles, the preset tiles of `docs/30` section 9.6). The
 * store makes each picture once per photo, tile and `key` and hands the same request to every tile that asks, which has consequences
 * handled here:
 *
 * - A request can fail for a reason that is not the tile's (the core cancels the tiles of a photo
 *   when the person moves to another one, and a tile that comes back meanwhile is handed that
 *   request). The store forgets a failed request; the tile asks again, up to {@link MAX_TILE_RETRIES}
 *   times, unless it has left (it is not drawn any more, or its photo or settings changed) or the
 *   answer is final.
 * - A request still waiting for a place (the store lets only a few run at once) is dropped when the photo's settings change or the
 *   person moves to another photo, and rejects as superseded. That is no failure: the tile does not ask again for the old picture
 *   and does not spend its allowance; if it is still on screen it has already asked for the new one.
 * - An address can stop answering (the shell keeps a bounded amount of tiles and drops the oldest
 *   photos' pictures, so an address from a photo visited a while ago gives 404). When the `<img>`
 *   reports an error the store forgets that address and the tile asks for a new picture, within the
 *   same allowance.
 *
 * `picture` names the picture (photo, tile and settings): the allowance is renewed when it changes. `enabled` false (a tile out of view,
 * a look whose file is missing) asks for nothing and keeps the picture it has.
 */
export function useTilePicture(
  picture: string,
  enabled: boolean,
  ask: () => Promise<string>,
  forget: (url: string) => void,
  scope: string,
): LookThumbnail {
  const [src, setSrc] = useState<string | undefined>();
  // Bumped to ask again.
  const [round, setRound] = useState(0);
  // The allowance belongs to one picture, and is renewed for the next.
  const spent = useRef({ picture: "", count: 0 });
  // The callbacks of the latest draw: a new function every draw must not ask again.
  const latest = useRef({ ask, forget });
  latest.current = { ask, forget };

  useEffect(() => {
    if (!enabled) return;
    if (spent.current.picture !== picture) spent.current = { picture, count: 0 };
    let live = true;
    latest.current.ask().then(
      (url) => {
        if (live) setSrc(url);
      },
      (error: unknown) => {
        if (!live || isSuperseded(error) || isFinalIn(scope, error) || spent.current.count >= MAX_TILE_RETRIES) return;
        spent.current.count += 1;
        setRound((n) => n + 1);
      },
    );
    return () => {
      live = false;
    };
  }, [enabled, picture, scope, round]);

  const onImageError = useCallback(() => {
    if (src === undefined) return;
    latest.current.forget(src);
    setSrc(undefined);
    if (spent.current.count >= MAX_TILE_RETRIES) return;
    spent.current.count += 1;
    setRound((n) => n + 1);
  }, [src]);

  return { src, onImageError };
}

/** The picture of a photo through one look, for a tile of the Look card or of the wizard's last step (see {@link useTilePicture}). */
export function useLookThumbnail(store: AppStore, photoId: PhotoId | null, lookId: LookId, key: string, enabled = true): LookThumbnail {
  return useTilePicture(
    `${photoId} ${lookId} ${key}`,
    enabled && photoId !== null,
    () => store.lookThumbnail(photoId as PhotoId, lookId, key),
    (url) => store.forgetLookThumbnail(photoId as PhotoId, lookId, key, url),
    "looks",
  );
}

/** The picture of a photo with one preset applied, for a tile of the Presets card (`docs/30` section 9.6). `revision` changes when the preset is saved again. */
export function usePresetThumbnail(store: AppStore, photoId: PhotoId | null, name: string, revision: string, key: string, enabled = true): LookThumbnail {
  return useTilePicture(
    `${photoId} ${name} ${revision} ${key}`,
    enabled && photoId !== null,
    () => store.presetThumbnail(photoId as PhotoId, name, revision, key),
    (url) => store.forgetPresetThumbnail(photoId as PhotoId, name, revision, key, url),
    "presets",
  );
}
