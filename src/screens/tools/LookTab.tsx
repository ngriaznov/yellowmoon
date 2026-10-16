import { useEffect, useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";
import { AutoRow, AutoToggle, Card, Field, formats, LinkButton, LookTile, ResetLink, Rows, SearchField, Segmented, Slider, Tag } from "../../components";
import type { TabId } from "../../components";
import type { ApplyScope, Look, LookLibrary, ResolvedSettings } from "../../api/types";
import { useAppState, useStore } from "../../state/context";
import { isArriLook, LOOK_LIBRARIES, looksInDisplayOrder, lookTooltip } from "../../state/looks";
import type { LookLibraryInfo } from "../../state/looks";
import type { AppStore } from "../../state/store";
import { useLookThumbnail } from "../../state/useLookThumbnail";
import { shortBody } from "./RawTab";
import { ContrastSlider, KelvinSlider, SaturationSlider, whiteBalanceState, whiteBalanceToAuto } from "./controls";
import { AUTO, tintLabel } from "./shared";
import type { TabProps } from "./shared";

/** Said on a tile whose look file is gone or changed (`docs/08-looks.md` §Missing looks). */
export const MISSING_LOOK_NOTE = "File missing or changed. Import again.";

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

/**
 * A look tile that asks for its thumbnail when it scrolls into view, and again when the photo's other settings change, when the
 * request it was handed failed, and when the picture it was given does not load (`useLookThumbnail`).
 */
function LazyLookTile({ look, index, selected, photoId, thumbKey, onSelect }: LazyTileProps) {
  const store = useStore();
  const button = useRef<HTMLButtonElement>(null);
  const [visible, setVisible] = useState(typeof IntersectionObserver === "undefined");

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

  const { src, onImageError } = useLookThumbnail(store, photoId, look.id, thumbKey, visible && !look.missing);

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
      title={lookTooltip(look)}
      onSelect={onSelect}
      onImageError={onImageError}
    />
  );
}

interface LookSectionProps {
  library: LookLibraryInfo;
  /** Every look of the library. */
  total: number;
  /** How many of them match the search. */
  shown: number;
  /** The search box has text: the section is open whatever the person folded, and its header does not fold. */
  searching: boolean;
  /** The person folded the section. */
  collapsed: boolean;
  /** The current photo's look is in this library. */
  holdsCurrent: boolean;
  onToggle: () => void;
  children: ReactNode;
}

/**
 * One library of the Look tab: a header that is a disclosure button (label, count, a chevron) over the grid of its tiles. Folded, the
 * tiles are not drawn at all (so their thumbnails are not asked for); a small filled square in the header says the current look is
 * inside. While the search box has text the section is open and the header only reads "2 of 87" (`aria-disabled`, and out of the Tab
 * order: it does nothing, and Tab from the search box goes to the first tile as it always did), so a search always finds a look in a
 * section that was folded.
 */
function LookSection({ library, total, shown, searching, collapsed, holdsCurrent, onToggle, children }: LookSectionProps) {
  const open = searching || !collapsed;
  const marker = !open && holdsCurrent;
  const bodyId = `look-section-${library.id}`;
  const noun = total === 1 ? "look" : "looks";
  const amount = searching ? `${shown} of ${total} ${noun}` : `${total} ${noun}`;
  const name = [library.label, amount, ...(marker ? ["holds the current look"] : []), ...(searching ? [] : [open ? "collapse" : "expand"])].join(", ");
  return (
    <div className="ym-look-group" data-library={library.id}>
      <button
        type="button"
        className="ym-look-group__header ym-look-group__toggle"
        aria-expanded={open}
        aria-controls={bodyId}
        aria-label={name}
        aria-disabled={searching || undefined}
        tabIndex={searching ? -1 : undefined}
        title={searching ? "Clear the search to fold this section." : undefined}
        onClick={() => {
          if (!searching) onToggle();
        }}
      >
        <span className="ym-look-group__title">
          <span className="ym-look-group__chevron" data-open={open} aria-hidden="true" />
          <span>{library.label}</span>
          {marker ? <span className="ym-look-group__marker" aria-hidden="true" /> : null}
        </span>
        <span className="ym-look-group__count">{searching ? `${shown} of ${total}` : total}</span>
      </button>
      <div id={bodyId} hidden={!open}>
        {open ? <div className="ym-look-grid">{children}</div> : null}
      </div>
    </div>
  );
}

/**
 * The Tone card: Temperature, Contrast and Saturation, under the Look card (owner's decision of
 * 2026-10-13, `docs/16`; the mockup has no such card). The rows are the same settings as Colour ›
 * White balance › Kelvin, Light › Exposure › Contrast and Colour › Colour balance › Saturation,
 * built from the same controls (`controls.tsx`): moving one copy moves the other, with undo, redo
 * and Reset. Temperature keeps the Kelvin control's Auto semantics: it shows the photo's Auto Kelvin,
 * moving it makes the white balance Manual at that Kelvin with the Auto tint, and its Reset goes
 * back to Auto. As on the Exposure card, the AUTO button speaks for the white balance only. Tint
 * stays on the Colour tab. The card's Reset sends Auto for the white balance (a picked white and a
 * light preset go with it, as on the Colour tab) and puts Contrast and Saturation at 0.
 */
export function ToneCard({ store, rs }: { store: AppStore; rs: ResolvedSettings }) {
  const { picked, isAuto } = whiteBalanceState(rs);
  return (
    <Card
      title="Tone"
      tag={
        <>
          {picked ? <Tag variant="edited" title="White picked from the photo" /> : null}
          <AutoToggle isAuto={isAuto} onAuto={() => whiteBalanceToAuto(store)} aria-label="Auto temperature" />
        </>
      }
      action={<ResetLink size="card" onClick={() => void store.update({ white_balance: AUTO, light_source: "auto", white_pick: null, contrast: 0, saturation: 0 })} />}
    >
      <KelvinSlider store={store} rs={rs} label="Temperature" />
      <ContrastSlider store={store} rs={rs} />
      <SaturationSlider store={store} rs={rs} />
    </Card>
  );
}

export interface LookTabProps extends TabProps {
  /** The Automatic card's arrows open the tab that owns the value. */
  onOpenTab: (tab: TabId) => void;
}

/**
 * Look: the Automatic card (what was worked out for this photo, and where to change it), the
 * Look card (strength, "Apply to", search and the grid of looks with lazy thumbnails) and the Tone
 * card (Temperature, Contrast, Saturation).
 */
export function LookTab({ store, photo, rs, onOpenTab }: LookTabProps) {
  const looks = useAppState((s) => s.looks);
  const session = useAppState((s) => s.session);
  const scope = useAppState((s) => s.applyScope);
  const [query, setQuery] = useState("");
  const { settings, values, auto } = rs;

  const collapsed = useAppState((s) => s.collapsedLookLibraries);
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

  // One section per library that has looks, in display order; the search narrows what each shows.
  const sections = useMemo(() => {
    const byLibrary = new Map<LookLibrary, Look[]>();
    for (const look of ordered) {
      const list = byLibrary.get(look.library);
      if (list) list.push(look);
      else byLibrary.set(look.library, [look]);
    }
    return LOOK_LIBRARIES.map((library) => {
      const all = byLibrary.get(library.id) ?? [];
      return { library, all, shown: all.filter((l) => !q || l.name.toLowerCase().includes(q)) };
    });
  }, [ordered, q]);
  const anyShown = sections.some((s) => s.shown.length > 0);
  const arriCount = ordered.filter(isArriLook).length;

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
            label="White"
            value={values.white_ev > 0 ? `${formats.ev(values.white_ev)} EV` : "None"}
            isAuto={values.white_offset_ev === 0}
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
          {sections.map(({ library, all, shown }) => {
            if (library.id === "arri_logc4" && arriCount === 0) {
              // Nothing of ARRI's is imported: say so, and offer the wizard.
              return q ? null : (
                <div className="ym-look-group" key="arri-none">
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
            if (shown.length === 0) return null;
            return (
              <LookSection
                key={library.id}
                library={library}
                total={all.length}
                shown={shown.length}
                searching={q !== ""}
                collapsed={collapsed.includes(library.id)}
                holdsCurrent={all.some((l) => l.id === settings.look)}
                onToggle={() => store.toggleLookLibrary(library.id)}
              >
                {shown.map((look) => (
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
              </LookSection>
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
      <ToneCard store={store} rs={rs} />
    </>
  );
}

