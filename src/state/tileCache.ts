import type { PhotoId } from "../api/types";
import { releaseUrl } from "./photo";
import { RequestLimiter } from "./requestLimiter";

/** One tile's request: the promise of its address, and the address once it is known (so a tile that fails to load it can be recognised). */
interface Tile {
  promise: Promise<string>;
  url: string | null;
}

/** Milliseconds a replaced tile's address is kept, so that a tile never shows a broken picture. */
const URL_GRACE_MS = 3000;

/**
 * The tiles of one kind (the preset tiles, `docs/30` section 9.6) with the look tiles' machinery: per photo, one cache keyed by what the
 * picture was made from (`key`: the photo's other settings); a request waits for one of `concurrency` places, the newest first, and is
 * dropped with `Superseded` when the photo's settings change or the person moves to another photo; a request that failed is not kept; an
 * address that stopped answering is forgotten and asked for again. In memory only.
 */
export class TileCache {
  private readonly photos = new Map<PhotoId, { key: string; tiles: Map<string, Tile> }>();
  private readonly limiter: RequestLimiter<{ photoId: PhotoId; key: string }>;

  constructor(concurrency: number) {
    this.limiter = new RequestLimiter(concurrency);
  }

  /** The picture of tile `id` of a photo; `key` stands for everything about the photo the picture was made from. */
  request(photoId: PhotoId, id: string, key: string, make: () => Promise<string>): Promise<string> {
    let entry = this.photos.get(photoId);
    if (!entry || entry.key !== key) {
      if (entry) {
        this.releaseLater([...entry.tiles.values()]);
        this.limiter.supersede((t) => t.photoId === photoId && t.key !== key);
      }
      entry = { key, tiles: new Map() };
      this.photos.set(photoId, entry);
    }
    const cache = entry.tiles;
    let tile = cache.get(id);
    if (!tile) {
      const made: Tile = {
        promise: this.limiter.run({ photoId, key }, make),
        url: null,
      };
      tile = made;
      cache.set(id, made);
      made.promise.then(
        (url) => {
          made.url = url;
        },
        () => {
          if (cache.get(id) === made) cache.delete(id);
        },
      );
    }
    return tile.promise;
  }

  /** Drops the address a tile was given because its picture did not load; the next `request` makes a new one. */
  forget(photoId: PhotoId, id: string, key: string, url: string): void {
    const entry = this.photos.get(photoId);
    if (entry?.key !== key) return;
    const tile = entry.tiles.get(id);
    if (tile?.url !== url) return;
    entry.tiles.delete(id);
    this.releaseLater([tile]);
  }

  /** Drops what waits for a place for every photo but `keep`: the person moved to another photo. */
  supersedeOthers(keep: PhotoId): void {
    this.limiter.supersede((t) => t.photoId !== keep);
  }

  /** The session ended: everything is dropped and the addresses are let go. */
  clear(): void {
    this.limiter.supersede(() => true);
    for (const entry of this.photos.values()) for (const tile of entry.tiles.values()) void tile.promise.then(releaseUrl, () => {});
    this.photos.clear();
  }

  /** How many requests are out, and how many wait (for tests). */
  get load(): { inFlight: number; queued: number } {
    return { inFlight: this.limiter.inFlight, queued: this.limiter.queued };
  }

  private releaseLater(tiles: Tile[]): void {
    const pending = tiles.map((t) => t.promise);
    setTimeout(() => {
      for (const url of pending) void url.then(releaseUrl, () => {});
    }, URL_GRACE_MS);
  }
}
