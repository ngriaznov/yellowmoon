import { memo, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ChevronDownIcon } from "lucide-react";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/glass/collapsible";
import type { ReactNode } from "react";
import { AutoRow, AutoToggle, Button, Card, Field, formats, LinkButton, LookTile, ResetLink, RowNote, Rows, SearchField, Segmented, Slider, Tag } from "../../components";
import type { TabId } from "../../components";
import type { AppleAuto, AppleAutoSteps, ApplyScope, Look, LookId, LookLibrary, ResolvedSettings } from "../../api/types";
import { perfCount } from "../../perf";
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

/** The Automatic card's line under its rows while Apple Auto is on for the session (`docs/19` section 10), by the photo's state. */
export const APPLE_AUTO_LINES = {
  answered: "Set with Apple Intelligence.",
  pending: "Apple Intelligence is looking at this photo…",
  failed: "Apple Intelligence couldn't look at this photo, so Auto uses Yellowmoon's own settings.",
} as const;

/**
 * The Automatic card's line for a photo's Apple Auto state, or null while Apple Auto is off or not available: then the card has none
 * of Apple Auto's rows and no line, and is the card it was before Apple Auto.
 */
export function appleAutoLine(apple: AppleAuto): string | null {
  return apple.state === "answered" || apple.state === "pending" || apple.state === "failed" ? APPLE_AUTO_LINES[apple.state] : null;
}

/** The model's plan for the photo, shown under the card's line (`docs/19` section 10); null without an answer or with no words. */
export function appleAutoPlan(apple: AppleAuto): string | null {
  if (apple.state !== "answered") return null;
  const plan = apple.answer.plan.trim();
  return plan === "" ? null : plan;
}

const STRENGTH_WORDS: Readonly<Record<number, string>> = { 1: "subtle", 2: "medium", 3: "strong" };

/**
 * A colour Apple Auto set for the shadows or the highlights, in words: "None", or "Teal · medium"; null when the answer set nothing
 * for that wheel (no colour, or a colour without a strength, `docs/19` section 5).
 */
export function appleColorText(color: string | null, strength: number | null): string | null {
  if (color === null) return null;
  if (color === "none") return "None";
  if (strength === null) return null;
  const name = color.charAt(0).toUpperCase() + color.slice(1);
  return `${name} · ${STRENGTH_WORDS[strength] ?? strength}`;
}

/** Which of Apple Auto's rows the Automatic card lists: the values the answer in effect set (`docs/19` section 10), in the table's order. */
export function appleRows(steps: AppleAutoSteps | null): {
  highlights: boolean;
  shadows: boolean;
  black: boolean;
  contrast: boolean;
  saturation: boolean;
  shadowColor: string | null;
  highlightColor: string | null;
} {
  return {
    highlights: steps?.highlights != null,
    shadows: steps?.shadows != null,
    black: steps?.blacks != null,
    contrast: steps?.contrast != null,
    saturation: steps?.saturation != null,
    shadowColor: steps ? appleColorText(steps.shadow_color, steps.shadow_color_strength) : null,
    highlightColor: steps ? appleColorText(steps.highlight_color, steps.highlight_color_strength) : null,
  };
}

/**
 * Brings the Tone card into view in the panel that scrolls it and puts the focus on its slider called `label` (the Automatic card's
 * Contrast and Saturation rows: "the arrow opens the Look tab's Tone card", `docs/19` section 10). Only the panel scrolls:
 * `scrollIntoView` would also move the locked page.
 */
function revealToneSlider(card: HTMLElement | null, label: string): void {
  if (!card) return;
  let scroller = card.parentElement;
  while (scroller && !(scroller.scrollHeight > scroller.clientHeight && /(auto|scroll)/.test(getComputedStyle(scroller).overflowY))) {
    scroller = scroller.parentElement;
  }
  if (scroller) scroller.scrollTop += card.getBoundingClientRect().top - scroller.getBoundingClientRect().top;
  const nameOf = (el: HTMLElement): string | null => {
    const ids = el.getAttribute("aria-labelledby");
    return ids ? ids.split(/\s+/).map((id) => document.getElementById(id)?.textContent ?? "").join(" ") : el.getAttribute("aria-label");
  };
  const slider = [...card.querySelectorAll<HTMLElement>('[role="slider"]')].find((el) => nameOf(el) === label);
  slider?.focus({ preventScroll: true });
}

/**
 * How long the photo's settings must rest before the tiles are made again: a person moving a slider on this tab (the Tone card is
 * here) is not asking to see every look through each step, and each remake is a job for the core's one foreground worker.
 */
export const THUMBNAIL_SETTLE_MS = 400;

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

/**
 * Tells the tiles of the tab when they are in view, with one `IntersectionObserver` for all of them (a tab can hold hundreds of
 * tiles; one observer each was a few hundred objects the page had to keep). Without `IntersectionObserver` (a test) every tile is
 * always in view.
 */
export interface TileWatch {
  /** Calls `callback(true)` while `el` is in view and `callback(false)` while it is not, both ways. Returns the function that stops. */
  observe: (el: Element, callback: (inView: boolean) => void) => () => void;
  /** Stops watching everything; `observe` starts again if asked. */
  disconnect: () => void;
}

/** Makes a {@link TileWatch}. */
export function createTileWatch(): TileWatch {
  const callbacks = new Map<Element, (inView: boolean) => void>();
  let observer: IntersectionObserver | null = null;
  return {
    observe(el, callback) {
      if (typeof IntersectionObserver === "undefined") {
        callback(true);
        return () => undefined;
      }
      observer ??= new IntersectionObserver((entries) => {
        for (const entry of entries) callbacks.get(entry.target)?.(entry.isIntersecting);
      });
      callbacks.set(el, callback);
      observer.observe(el);
      return () => {
        callbacks.delete(el);
        observer?.unobserve(el);
      };
    },
    disconnect() {
      observer?.disconnect();
      observer = null;
      callbacks.clear();
    },
  };
}

interface LazyTileProps {
  look: Look;
  index: number;
  selected: boolean;
  photoId: number;
  /** Everything about the photo except its look; when it changes the thumbnails are made again. */
  thumbKey: string;
  /** One function for every tile of the grid (a tile of its own would be a new prop for each, and every tile would draw again). */
  onSelect: (id: LookId) => void;
  /** The tab's shared observer. */
  watch: TileWatch;
}

/**
 * A look tile that asks for its thumbnail while it is in view (the tab's shared `TileWatch`), and again when the photo's other settings
 * change, when the request it was handed failed, and when the picture it was given does not load (`useLookThumbnail`). A tile out of
 * view asks for nothing and keeps the picture it has; it asks for a new one when it comes back with other settings.
 *
 * It draws again only when one of its own props changes (`memo`): a tick of the status line, a slider moving or another tile
 * becoming the selected one leaves the others alone.
 */
const LazyLookTile = memo(function LazyLookTile({ look, index, selected, photoId, thumbKey, onSelect, watch }: LazyTileProps) {
  perfCount("LazyLookTile");
  const store = useStore();
  const button = useRef<HTMLButtonElement>(null);
  const [inView, setInView] = useState(typeof IntersectionObserver === "undefined");

  useEffect(() => {
    const el = button.current;
    return el ? watch.observe(el, setInView) : undefined;
  }, [watch]);

  const { src, onImageError } = useLookThumbnail(store, photoId, look.id, thumbKey, inView && !look.missing);

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
      onSelect={() => onSelect(look.id)}
      onImageError={onImageError}
    />
  );
});

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
    <Collapsible
      open={open}
      disabled={searching}
      onOpenChange={() => {
        if (!searching) onToggle();
      }}
      className="ym-look-group"
      data-library={library.id}
    >
      <CollapsibleTrigger asChild>
        <Button
          variant="quiet"
          size="sm"
          className="ym-look-group__header ym-look-group__toggle"
          aria-label={name}
          aria-controls={bodyId}
          // It does nothing while a search holds the sections open, so Tab skips it (it stays readable to a screen reader).
          tabIndex={searching ? -1 : undefined}
          title={searching ? "Clear the search to fold this section." : undefined}
        >
          <span className="ym-look-group__title">
            <ChevronDownIcon className="ym-look-group__chevron" data-open={open} aria-hidden="true" />
            <span>{library.label}</span>
            {marker ? <span className="ym-look-group__marker" aria-hidden="true" /> : null}
          </span>
          <span className="ym-look-group__count">{searching ? `${shown} of ${total}` : total}</span>
        </Button>
      </CollapsibleTrigger>
      <CollapsibleContent id={bodyId} className="ym-look-group__body">
        <div className="ym-look-grid">{children}</div>
      </CollapsibleContent>
    </Collapsible>
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
 * light preset go with it, as on the Colour tab) and puts Contrast and Saturation back to Auto (0, or Apple Auto's answer, `docs/19`).
 */
export function ToneCard({ store, rs }: { store: AppStore; rs: ResolvedSettings }) {
  const { picked, isAuto } = whiteBalanceState(rs);
  return (
    <Card
      className="ym-tone-card"
      title="Tone"
      tag={
        <>
          {picked ? <Tag variant="edited" title="White picked from the photo" /> : null}
          <AutoToggle isAuto={isAuto} onAuto={() => whiteBalanceToAuto(store)} aria-label="Auto temperature" />
        </>
      }
      action={<ResetLink size="card" onClick={() => void store.update({ white_balance: AUTO, light_source: "auto", white_pick: null, contrast: AUTO, saturation: AUTO }, true, rs.photo_id)} />}
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
export const LookTab = memo(function LookTab({ store, photo, rs, onOpenTab }: LookTabProps) {
  perfCount("LookTab");
  const looks = useAppState((s) => s.looks);
  const nPicked = useAppState((s) => s.session?.picked.length ?? 0);
  const nAll = useAppState((s) => s.session?.photos.length ?? 0);
  const scope = useAppState((s) => s.applyScope);
  const [query, setQuery] = useState("");
  const { settings, values, auto } = rs;

  const collapsed = useAppState((s) => s.collapsedLookLibraries);
  const ordered = useMemo(() => looksInDisplayOrder(looks ?? []), [looks]);
  const keyOf = useMemo(() => new Map(ordered.map((l, i) => [l.id, i + 1])), [ordered]);
  const current = ordered.find((l) => l.id === settings.look);
  const defaultLook = ordered.find((l) => l.is_default);
  const q = query.trim().toLowerCase();

  // Thumbnails show the photo as it is developed now, so they follow every change except the look itself, and the Auto values the
  // settings resolve to: Apple Auto's answer (`docs/19`) or a burst's median moves them without a change of the settings.
  const { estimates } = rs;
  const rawKey = useMemo(() => {
    const { look: _look, look_intensity: _intensity, ...rest } = settings;
    return JSON.stringify([
      rest,
      estimates.exposure_ev,
      estimates.kelvin,
      estimates.tint,
      estimates.contrast,
      estimates.saturation,
      estimates.highlights,
      estimates.shadows,
      estimates.black,
      estimates.balance_shadows,
      estimates.balance_highlights,
    ]);
  }, [
    settings,
    estimates.exposure_ev,
    estimates.kelvin,
    estimates.tint,
    estimates.contrast,
    estimates.saturation,
    estimates.highlights,
    estimates.shadows,
    estimates.black,
    estimates.balance_shadows,
    estimates.balance_highlights,
  ]);
  const thumbKey = useSettled(rawKey, THUMBNAIL_SETTLE_MS);

  // One observer and one select function for every tile, so a tile's props change only when the tile does.
  const [watch] = useState(createTileWatch);
  useEffect(() => () => watch.disconnect(), [watch]);
  // A click names the photo the tab is drawn for, so the look goes to the photo whose tiles were clicked.
  const photoId = photo.id;
  const selectLook = useCallback((id: LookId) => void store.chooseLook(id, undefined, photoId), [store, photoId]);

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

  const scopeOptions: Array<{ value: ApplyScope; label: string }> = [
    { value: "this", label: "This photo" },
    { value: "picked", label: `Picked · ${nPicked}` },
    { value: "all", label: `All · ${nAll}` },
  ];

  const noiseValue = `Luminance ${Math.round(values.noise_reduction.luminance)}`;
  // Apple Auto on for the session: the photo is waiting for its answer, has it, or failed (`docs/19` section 10). Its rows are the
  // values the answer in effect set, in the order of the table; a value the model left empty has no row.
  const appleLine = appleAutoLine(rs.estimates.apple);
  const applePlan = appleAutoPlan(rs.estimates.apple);
  const rows = appleRows(appleLine !== null ? rs.estimates.apple_steps : null);
  const openLight = (): void => onOpenTab("light");
  const openColor = (): void => onOpenTab("color");
  const openTone = (label: string): void => revealToneSlider(document.querySelector<HTMLElement>(".ym-tone-card"), label);
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
          {rows.highlights ? (
            <AutoRow label="Highlights" value={formats.signed(values.highlights)} isAuto={auto.highlights} tabLabel="Light" onOpen={openLight} />
          ) : null}
          {rows.shadows ? (
            <AutoRow label="Shadows" value={formats.signed(values.shadows)} isAuto={auto.shadows} tabLabel="Light" onOpen={openLight} />
          ) : null}
          {rows.black ? (
            <AutoRow label="Black" value={formats.signed(values.black)} isAuto={auto.black} tabLabel="Light" onOpen={openLight} />
          ) : null}
          {rows.contrast ? (
            <AutoRow
              label="Contrast"
              value={formats.signed(values.contrast)}
              isAuto={auto.contrast}
              tabLabel="Tone"
              onOpen={() => openTone("Contrast")}
            />
          ) : null}
          {rows.saturation ? (
            <AutoRow
              label="Saturation"
              value={formats.signed(values.saturation)}
              isAuto={auto.saturation}
              tabLabel="Tone"
              onOpen={() => openTone("Saturation")}
            />
          ) : null}
          {rows.shadowColor !== null ? (
            <AutoRow
              label="Shadows colour"
              value={auto.balance_shadows ? rows.shadowColor : "Custom"}
              isAuto={auto.balance_shadows}
              tabLabel="Colour"
              onOpen={openColor}
            />
          ) : null}
          {rows.highlightColor !== null ? (
            <AutoRow
              label="Highlights colour"
              value={auto.balance_highlights ? rows.highlightColor : "Custom"}
              isAuto={auto.balance_highlights}
              tabLabel="Colour"
              onOpen={openColor}
            />
          ) : null}
          <AutoRow label="Noise" value={noiseValue} isAuto={auto.noise_reduction} tabLabel="Detail" onOpen={() => onOpenTab("detail")} />
          <AutoRow label="Camera" value={profileText} isAuto tabLabel="RAW" onOpen={() => onOpenTab("raw")} />
        </Rows>
        {appleLine !== null ? <RowNote>{appleLine}</RowNote> : null}
        {applePlan !== null ? <RowNote>{`\u201c${applePlan}\u201d`}</RowNote> : null}
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
              onClick={() => (defaultLook ? void store.chooseLook(defaultLook.id, 100, photo.id) : void store.setLookIntensity(100, true, photo.id))}
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
          onChange={(v, final) => void store.setLookIntensity(v, final, photo.id)}
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
                    onSelect={selectLook}
                    watch={watch}
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
});

