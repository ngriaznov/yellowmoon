import { useEffect, useMemo, useRef, useState } from "react";
import { AutoRow, Card, Field, formats, LinkButton, LookTile, ResetLink, Rows, SearchField, Segmented, Slider } from "../../components";
import type { TabId } from "../../components";
import type { ApplyScope, Look } from "../../api/types";
import { useAppState, useStore } from "../../state/context";
import { shortBody } from "./RawTab";
import { tintLabel } from "./shared";
import type { TabProps } from "./shared";

/** Said on a tile whose look file is gone or changed (`docs/08-looks.md` §Missing looks). */
export const MISSING_LOOK_NOTE = "File missing or changed. Import again.";

/** Groups of the look grid, in order. */
const GROUPS = [
  { kind: "built_in", label: "Built in" },
  { kind: "arri", label: "ARRI looks · imported" },
  { kind: "user", label: "Your LUTs" },
] as const;

/** The looks in the order the tiles are drawn: built in, then ARRI, then the person's own. This is also the order of the keys 1–9. */
export function looksInDisplayOrder(looks: readonly Look[]): Look[] {
  return GROUPS.flatMap((g) => looks.filter((l) => l.group.kind === g.kind));
}

/** Waits `ms` after a value stops changing before passing it on; the first value passes at once. */
function useSettled<T>(value: T, ms: number): T {
  const [settled, setSettled] = useState(value);
  const first = useRef(true);
  useEffect(() => {
    if (first.current) {
      first.current = false;
      return;
    }
    const timer = setTimeout(() => setSettled(value), ms);
    return () => clearTimeout(timer);
  }, [value, ms]);
  return settled;
}

interface LazyTileProps {
  look: Look;
  index: number;
  selected: boolean;
  photoId: number;
  /** Everything about the photo except its look; when it changes the thumbnails are made again. */
  thumbKey: string;
  onSelect: () => void;
}

/** A look tile that asks for its thumbnail when it scrolls into view, and again when the photo's other settings change. */
function LazyLookTile({ look, index, selected, photoId, thumbKey, onSelect }: LazyTileProps) {
  const store = useStore();
  const button = useRef<HTMLButtonElement>(null);
  const [visible, setVisible] = useState(typeof IntersectionObserver === "undefined");
  const [src, setSrc] = useState<string | undefined>();

  useEffect(() => {
    const el = button.current;
    if (!el || typeof IntersectionObserver === "undefined") return;
    const observer = new IntersectionObserver((entries) => {
      if (entries.some((e) => e.isIntersecting)) {
        setVisible(true);
        observer.disconnect();
      }
    });
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    if (!visible || look.missing) return;
    let live = true;
    store
      .lookThumbnail(photoId, look.id, thumbKey)
      .then((url) => {
        if (live) setSrc(url);
      })
      .catch(() => {});
    return () => {
      live = false;
    };
  }, [store, visible, look.id, look.missing, photoId, thumbKey]);

  return (
    <LookTile
      ref={button}
      name={look.name}
      src={src}
      index={index <= 9 ? index : undefined}
      isDefault={look.is_default}
      selected={selected}
      disabled={look.missing}
      note={look.missing ? MISSING_LOOK_NOTE : undefined}
      onSelect={onSelect}
    />
  );
}

export interface LookTabProps extends TabProps {
  /** The Automatic card's arrows open the tab that owns the value. */
  onOpenTab: (tab: TabId) => void;
}

/**
 * Look: the Automatic card (what was worked out for this photo, and where to change it) and the
 * Look card (strength, "Apply to", search and the grid of looks with lazy thumbnails).
 */
export function LookTab({ store, photo, rs, onOpenTab }: LookTabProps) {
  const looks = useAppState((s) => s.looks);
  const session = useAppState((s) => s.session);
  const scope = useAppState((s) => s.applyScope);
  const [query, setQuery] = useState("");
  const { settings, values, auto } = rs;

  const ordered = useMemo(() => looksInDisplayOrder(looks ?? []), [looks]);
  const keyOf = useMemo(() => new Map(ordered.map((l, i) => [l.id, i + 1])), [ordered]);
  const current = ordered.find((l) => l.id === settings.look);
  const defaultLook = ordered.find((l) => l.is_default);
  const q = query.trim().toLowerCase();

  // Thumbnails show the photo as it is developed now, so they follow every change except the look itself.
  const rawKey = useMemo(() => {
    const { look: _look, look_intensity: _intensity, ...rest } = settings;
    return JSON.stringify(rest);
  }, [settings]);
  const thumbKey = useSettled(rawKey, 250);

  const groups = GROUPS.map((g) => {
    const all = ordered.filter((l) => l.group.kind === g.kind);
    const shown = all.filter((l) => !q || l.name.toLowerCase().includes(q));
    return { ...g, all, shown };
  }).filter((g) => g.kind !== "user" || g.all.length > 0);
  const anyShown = groups.some((g) => g.shown.length > 0);
  const arriCount = ordered.filter((l) => l.group.kind === "arri").length;

  const nPicked = session?.picked.length ?? 0;
  const nAll = session?.photos.length ?? 0;
  const scopeOptions: Array<{ value: ApplyScope; label: string }> = [
    { value: "this", label: "This photo" },
    { value: "picked", label: `Picked · ${nPicked}` },
    { value: "all", label: `All · ${nAll}` },
  ];

  const noiseValue = `Luminance ${Math.round(values.noise_reduction.luminance)}`;
  const whiteAuto = auto.white_balance && auto.light_source && settings.white_pick === null;
  const profileText =
    photo.camera.match_kind.kind === "exact" ? `${shortBody(photo.camera.body)} · measured profile` : `${shortBody(photo.camera.body)} · ${photo.camera.profile} profile`;

  return (
    <>
      <Card title="Automatic" action={<ResetLink label="Re-run" size="card" onClick={() => void store.rerunAuto()} />}>
        <Rows>
          <AutoRow
            label="Exposure"
            value={`${formats.ev(values.exposure_ev)} EV`}
            isAuto={auto.exposure}
            tabLabel="Light"
            onOpen={() => onOpenTab("light")}
          />
          <AutoRow
            label="White balance"
            value={`${Math.round(values.kelvin)} K · tint ${tintLabel(values.tint)}`}
            isAuto={whiteAuto}
            tabLabel="Colour"
            onOpen={() => onOpenTab("color")}
          />
          <AutoRow label="Noise" value={noiseValue} isAuto={auto.noise_reduction} tabLabel="Detail" onOpen={() => onOpenTab("detail")} />
          <AutoRow label="Camera" value={profileText} isAuto tabLabel="RAW" onOpen={() => onOpenTab("raw")} />
        </Rows>
      </Card>
      <Card
        title="Look"
        action={
          <span className="ym-card__actions">
            {current && defaultLook && current.id !== defaultLook.id ? (
              <LinkButton size="sm" onClick={() => void store.makeDefault(current.id)}>
                Make default
              </LinkButton>
            ) : null}
            <ResetLink
              size="card"
              onClick={() => (defaultLook ? void store.chooseLook(defaultLook.id, 100) : void store.setLookIntensity(100, true))}
            />
          </span>
        }
      >
        <Slider
          label="Intensity"
          value={values.look_intensity}
          min={0}
          max={100}
          defaultValue={100}
          format={formats.percent}
          onChange={(v, final) => void store.setLookIntensity(v, final)}
        />
        <Field label="Apply to">
          <Segmented label="Apply look to" value={scope} onChange={(next) => store.setApplyScope(next)} options={scopeOptions} />
        </Field>
        <div className="ym-look-block">
          <SearchField className="ym-look-search" value={query} onChange={setQuery} label="Search looks" />
          {groups.map((g) => {
            if (g.kind === "arri" && arriCount === 0) {
              // Nothing imported: say so, and offer the wizard.
              return q ? null : (
                <div className="ym-look-group" key={g.kind}>
                  <div className="ym-look-group__header">
                    <span>ARRI looks</span>
                    <span className="ym-look-group__count">0</span>
                  </div>
                  <p className="ym-look-cta">The free ARRI Look Library is not imported yet.</p>
                  <LinkButton className="ym-look-import" onClick={() => store.go("import-looks")}>
                    Import ARRI looks →
                  </LinkButton>
                </div>
              );
            }
            if (g.shown.length === 0) return null;
            return (
              <div className="ym-look-group" key={g.kind}>
                <div className="ym-look-group__header">
                  <span>{g.label}</span>
                  <span className="ym-look-group__count">{q ? `${g.shown.length} of ${g.all.length}` : g.all.length}</span>
                </div>
                <div className="ym-look-grid">
                  {g.shown.map((look) => (
                    <LazyLookTile
                      key={look.id}
                      look={look}
                      index={keyOf.get(look.id) ?? 0}
                      selected={look.id === settings.look}
                      photoId={photo.id}
                      thumbKey={thumbKey}
                      onSelect={() => void store.chooseLook(look.id)}
                    />
                  ))}
                </div>
              </div>
            );
          })}
          {!anyShown && q ? <p className="ym-look-empty">No looks match “{query.trim()}”.</p> : null}
          <div className="ym-look-links">
            {arriCount > 0 ? (
              <LinkButton className="ym-look-import" onClick={() => store.go("import-looks")}>
                Import ARRI looks →
              </LinkButton>
            ) : null}
            <LinkButton className="ym-look-import" onClick={() => void store.startCubeImport()}>
              Import .cube…
            </LinkButton>
          </div>
        </div>
      </Card>
    </>
  );
}

