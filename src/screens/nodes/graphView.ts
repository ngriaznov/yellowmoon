import { useRef } from "react";
import type { DevelopSettings } from "../../api/types";

/**
 * What the graph draws of a photo's settings: the nodes (id, kind, name, bypass, place, the kinds of their mask rows and whether they are
 * on) and the links. A slider of the panel changes none of it, so the same object comes back for as long as these are the same and the graph
 * is not drawn again at each tick of a drag (U-02). Everything else in the settings is of no interest to a card, a port or a link.
 */
export function useGraphView(settings: DevelopSettings | null): DevelopSettings | null {
  const kept = useRef<{ key: string; settings: DevelopSettings } | null>(null);
  if (settings === null) return null;
  const key = graphKey(settings);
  if (kept.current?.key !== key) kept.current = { key, settings };
  return kept.current.settings;
}

/** A string that is the same while what the graph draws is the same. */
export function graphKey(settings: DevelopSettings): string {
  return JSON.stringify([
    settings.nodes.links,
    settings.nodes.items.map((item) => [
      item.id,
      item.kind,
      "name" in item ? item.name : null,
      "bypass" in item ? item.bypass : null,
      "pos" in item ? item.pos : null,
      "masks" in item ? item.masks.items.map((row) => [row.kind, row.enabled]) : null,
    ]),
  ]);
}
