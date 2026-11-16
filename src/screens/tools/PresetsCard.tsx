import { memo, useEffect, useMemo, useRef, useState } from "react";
import { ContextMenu, ContextMenuContent, ContextMenuItem, ContextMenuSeparator, ContextMenuTrigger } from "@/components/glass/context-menu";
import { Button, Card, Field, LinkButton, LookTile, Segmented } from "../../components";
import type { ApplyScope, PresetEntry } from "../../api/types";
import { perfCount } from "../../perf";
import { useAppState } from "../../state/context";
import { NO_PICKED_NOTICE, tileDescription, TILES_SHOWN, tileTooltip, visiblePresets } from "../../state/presets";
import { isReady } from "../../state/photo";
import type { AppStore } from "../../state/store";
import { usePresetThumbnail } from "../../state/useLookThumbnail";
import { createTileWatch, THUMBNAIL_SETTLE_MS, useSettled } from "./tiles";
import type { TileWatch } from "./tiles";
import type { TabProps } from "./shared";

interface PresetTileProps {
  store: AppStore;
  entry: PresetEntry;
  photoId: number;
  /** Everything about the photo; when it settles on other settings the pictures are made again. */
  thumbKey: string;
  picked: number;
  developed: boolean;
  watch: TileWatch;
}

/**
 * A preset tile (`docs/30` section 9.1, 9.4): the look tile's size, picture, name line, ring and focus, the photo with the preset applied
 * as its picture (asked for while the tile is in view, and again when the photo's other settings settle), and a right-click menu named
 * "<name> actions" (Control-click, the Menu key and Shift+F10 open it too). A click applies the preset to the card's "Apply to"; the
 * scope is read when the click is made.
 */
const PresetTile = memo(function PresetTile({ store, entry, photoId, thumbKey, picked, developed, watch }: PresetTileProps) {
  perfCount("PresetTile");
  const button = useRef<HTMLButtonElement>(null);
  const [inView, setInView] = useState(typeof IntersectionObserver === "undefined");
  useEffect(() => {
    const el = button.current;
    return el ? watch.observe(el, setInView) : undefined;
  }, [watch]);
  const { src, onImageError } = usePresetThumbnail(store, photoId, entry.name, entry.revision, thumbKey, inView);
  return (
    <ContextMenu>
      <ContextMenuTrigger asChild>
        <LookTile
          ref={button}
          kind="preset"
          name={entry.name}
          src={src}
          selected={entry.matches}
          title={tileTooltip(entry)}
          description={tileDescription(entry)}
          onSelect={() => void store.applyPreset(entry.name)}
          onImageError={onImageError}
        />
      </ContextMenuTrigger>
      <ContextMenuContent aria-label={`${entry.name} actions`}>
        <ContextMenuItem
          disabled={picked === 0}
          title={picked === 0 ? NO_PICKED_NOTICE : undefined}
          onSelect={() => void store.applyPreset(entry.name, "picked")}
        >
          Apply to Picked Photos
        </ContextMenuItem>
        <ContextMenuItem
          disabled={!developed}
          title={developed ? undefined : "Wait for the photo to develop"}
          onSelect={() => store.openUpdatePreset(entry.name)}
        >
          Update from This Photo
        </ContextMenuItem>
        <ContextMenuSeparator />
        <ContextMenuItem onSelect={() => store.openRenamePreset(entry.name)}>Rename…</ContextMenuItem>
        <ContextMenuItem onSelect={() => void store.exportPreset(entry.name)}>Export…</ContextMenuItem>
        <ContextMenuItem onSelect={() => void store.revealPreset(entry.name)}>Show in Finder</ContextMenuItem>
        <ContextMenuSeparator />
        <ContextMenuItem variant="destructive" onSelect={() => void store.removePreset(entry.name)}>
          Delete
        </ContextMenuItem>
      </ContextMenuContent>
    </ContextMenu>
  );
});

/**
 * The Presets card (`docs/30` section 9.1): the last card of the Look tab. A header with the count and **Save…**, the card's own
 * "Apply to" (starts on This photo, not remembered), the tiles (six, then **Show all N**), the damaged files under them with Delete; with no
 * preset the empty state (section 9.2). The list is asked for again when the photo's settings settle, because the ring (`matches`) is
 * about them.
 */
export function PresetsCard({ store, photo, rs }: TabProps) {
  perfCount("PresetsCard");
  const listing = useAppState((s) => s.presets);
  const scope = useAppState((s) => s.presetScope);
  const showAll = useAppState((s) => s.presetsShowAll);
  const nPicked = useAppState((s) => s.session?.picked.length ?? 0);
  const nAll = useAppState((s) => s.session?.photos.length ?? 0);
  const developed = isReady(photo);

  // The tiles show the photo with the preset applied, so they follow every change of the photo (the Look card's follow all but the look).
  const { settings, estimates } = rs;
  const rawKey = useMemo(
    () => JSON.stringify([settings, estimates.input.exposure_ev, estimates.input.kelvin, estimates.input.tint, estimates.main]),
    [settings, estimates.input.exposure_ev, estimates.input.kelvin, estimates.input.tint, estimates.main],
  );
  const thumbKey = useSettled(rawKey, THUMBNAIL_SETTLE_MS);
  useEffect(() => {
    void store.refreshPresets();
  }, [store, photo.id, thumbKey]);

  const [watch] = useState(createTileWatch);
  useEffect(() => () => watch.disconnect(), [watch]);

  const presets = listing?.presets ?? [];
  const damaged = listing?.damaged ?? [];
  const shown = visiblePresets(presets, showAll);
  const scopeOptions: Array<{ value: ApplyScope; label: string }> = [
    { value: "this", label: "This photo" },
    { value: "picked", label: `Picked · ${nPicked}` },
    { value: "all", label: `All · ${nAll}` },
  ];
  const bodyId = `presets-tiles-${photo.id}`;

  return (
    <Card
      title="Presets"
      className="ym-presets"
      action={
        presets.length > 0 ? (
          <span className="ym-card__actions">
            <span className="ym-presets__count">{presets.length}</span>
            <LinkButton size="sm" disabled={!developed} onClick={() => void store.openSavePreset()}>
              Save…
            </LinkButton>
          </span>
        ) : undefined
      }
    >
      {listing === null ? null : presets.length === 0 ? (
        <div className="ym-presets__empty">
          <p className="ym-presets__empty-text">Set a photo the way you like it, then save it as a preset.</p>
          <Button
            variant="secondary"
            disabled={!developed}
            title={developed ? undefined : "Wait for the photo to develop"}
            onClick={() => void store.openSavePreset()}
          >
            Save preset…
          </Button>
        </div>
      ) : (
        <>
          <Field label="Apply to">
            <Segmented label="Apply preset to" value={scope} onChange={(next) => store.setPresetScope(next)} options={scopeOptions} />
          </Field>
          <div id={bodyId} className="ym-presets__tiles ym-look-grid" role="group" aria-label="Presets">
            {shown.map((entry) => (
              <PresetTile
                key={entry.name}
                store={store}
                entry={entry}
                photoId={photo.id}
                thumbKey={thumbKey}
                picked={nPicked}
                developed={developed}
                watch={watch}
              />
            ))}
          </div>
          {presets.length > TILES_SHOWN ? (
            <div className="ym-presets__more">
              <LinkButton aria-expanded={showAll} aria-controls={bodyId} onClick={() => store.setPresetsShowAll(!showAll)}>
                {showAll ? "Show fewer" : `Show all ${presets.length}`}
              </LinkButton>
            </div>
          ) : null}
        </>
      )}
      {damaged.length > 0 ? (
        <ul className="ym-presets__damaged" aria-label="Files that are not presets">
          {damaged.map((d) => (
            <li key={d.file}>
              <span className="ym-presets__damaged-file">{d.file}</span>
              <LinkButton size="sm" aria-label={`Delete ${d.file}`} onClick={() => void store.removeDamagedPreset(d.file)}>
                Delete
              </LinkButton>
              <span className="ym-presets__damaged-reason">{d.reason}</span>
            </li>
          ))}
        </ul>
      ) : null}
    </Card>
  );
}
