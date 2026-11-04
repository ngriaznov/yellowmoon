import { useCallback, useEffect, useRef, useState } from "react";
import type { FilmCatalogue, PhotoId } from "../../api/types";
import { MAX_TILE_RETRIES } from "../../state/useLookThumbnail";
import { isSuperseded } from "../../state/requestLimiter";
import type { AppStore } from "../../state/store";

/** The Film node's list of stocks and papers (`film_catalogue`), read once; `error` is the plain sentence when it cannot be read. */
export function useFilmCatalogue(store: AppStore): { catalogue: FilmCatalogue | null; error: string | null } {
  const [state, setState] = useState<{ catalogue: FilmCatalogue | null; error: string | null }>({ catalogue: null, error: null });
  useEffect(() => {
    let live = true;
    store.filmCatalogue().then(
      (catalogue) => {
        if (live) setState({ catalogue, error: null });
      },
      (error: unknown) => {
        const message = typeof error === "object" && error !== null && typeof (error as { message?: unknown }).message === "string" ? (error as { message: string }).message : "";
        if (live) setState({ catalogue: null, error: message });
      },
    );
    return () => {
      live = false;
    };
  }, [store]);
  return state;
}

/** What a stock tile needs to draw its picture. */
export interface FilmTile {
  /** The address of the picture, once it is known. */
  src: string | undefined;
  /** For the `error` event of the `<img>` that shows `src`. */
  onImageError: () => void;
}

/**
 * The photo through one stock of Film node `node` (`film_tile`), for a tile of the stock picker. The store makes each picture once and hands
 * the same request to every tile that asks; this asks again, up to `MAX_TILE_RETRIES` times, when the core cancelled the request (the person
 * left the photo and came back) or the shell had thrown the address away, and never for a request that was only replaced (`Superseded`).
 * `enabled` false asks for nothing: the picker asks while it is open. A tile that has no picture yet shows the stock's name on a plain card.
 */
export function useFilmTile(store: AppStore, photoId: PhotoId | null, node: number, stock: string, key: string, enabled: boolean): FilmTile {
  const [src, setSrc] = useState<string | undefined>();
  const [round, setRound] = useState(0);
  const spent = useRef({ picture: "", count: 0 });
  useEffect(() => {
    if (!enabled || photoId === null) return;
    const picture = `${photoId} ${node} ${stock} ${key}`;
    if (spent.current.picture !== picture) spent.current = { picture, count: 0 };
    let live = true;
    store.filmTile(photoId, node, stock, key).then(
      (url) => {
        if (live) setSrc(url);
      },
      (error: unknown) => {
        if (!live || isSuperseded(error) || spent.current.count >= MAX_TILE_RETRIES) return;
        spent.current.count += 1;
        setRound((n) => n + 1);
      },
    );
    return () => {
      live = false;
    };
  }, [store, enabled, photoId, node, stock, key, round]);
  const onImageError = useCallback(() => {
    if (src === undefined || photoId === null) return;
    store.forgetFilmTile(photoId, node, stock, key, src);
    setSrc(undefined);
    if (spent.current.count >= MAX_TILE_RETRIES) return;
    spent.current.count += 1;
    setRound((n) => n + 1);
  }, [store, src, photoId, node, stock, key]);
  return { src, onImageError };
}
