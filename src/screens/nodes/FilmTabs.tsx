/**
 * The tabs of a Film node (`docs/24` section 4.1): Film · Print · Halation · Grain, each with its cards; the Mask tab is a corrector's.
 * Every control writes the selected node as one node patch (one undo step, a drag is one). The list of stocks and papers is the core's
 * (`film_catalogue`). A slide dims the Print tab and says there is no print; the Film tab ends with the credit line and its link.
 */
import { ChevronRightIcon } from "lucide-react";
import { useMemo, useState } from "react";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/glass/select";
import type { PhotoSummary, ResolvedSettings } from "../../api/types";
import { Card, Field, formats, MINUS, ResetLink, RowNote, Slider } from "../../components";
import { patchOf } from "../../state/settingsGraph";
import type { FilmItem, FilmPatch } from "../../state/settingsGraph";
import type { AppStore } from "../../state/store";
import { FILM_DATA_MISSING, LIGHTS_HINT, NO_PAPER, SLIDE_SENTENCE, lightFormat, paperName, stockLine } from "./filmData";
import { FilmCredit } from "./FilmCredit";
import { StockPicker } from "./StockPicker";
import { useFilmCatalogue } from "./useFilm";

export interface FilmTabProps {
  store: AppStore;
  photo: PhotoSummary;
  rs: ResolvedSettings;
  /** The Film node the panel is on. */
  node: number;
  item: FilmItem;
}

/** What the photo is up to this node, except its stock: what the stock tiles are made from. */
function pictureKeyOf(rs: ResolvedSettings, node: number): string {
  return JSON.stringify(
    rs.settings.nodes.items.map((item) => (item.id === node && item.kind === "film" ? { ...item, film: { ...item.film, stock: "" } } : item)),
  );
}

/** The matching paper's id on the Select: `null` in the settings is "Matching paper". */
const MATCHING = "__matching__";

/** Film › Film: the Stock (a tile picker) with the kind and ISO under it, Exposure and Strength, and the credit line. */
export function FilmCard({ store, photo, rs, node, item }: FilmTabProps) {
  const { catalogue, error } = useFilmCatalogue(store);
  const key = useMemo(() => pictureKeyOf(rs, node), [rs, node]);
  const stock = catalogue?.stocks.find((s) => s.id === item.film.stock);
  const send = (partial: FilmPatch, final = true): void => void store.update(patchOf(node, partial), final, rs.photo_id);
  return (
    <>
      <Card
        title="Film"
        action={<ResetLink size="card" onClick={() => send({ film: { stock: DEFAULT_STOCK_ID, exposure_ev: 0 }, strength: 100 })} />}
      >
        <Field label="Stock">
          {catalogue ? (
            <StockPicker store={store} photoId={photo.id} node={node} catalogue={catalogue} value={item.film.stock} pictureKey={key} onChange={(next) => send({ film: { stock: next } })} />
          ) : (
            <span className="ym-stock-field" data-disabled="true">
              <span className="ym-stock-field__value">{error === null ? "…" : item.film.stock}</span>
            </span>
          )}
        </Field>
        {catalogue ? (
          <RowNote className="ym-film-line">{stock ? stockLine(stock) : ""}</RowNote>
        ) : error !== null ? (
          <RowNote role="status" className="ym-film-line">
            {error === "" ? FILM_DATA_MISSING : error}
          </RowNote>
        ) : null}
        <Slider
          label="Exposure"
          value={item.film.exposure_ev}
          min={-4}
          max={4}
          step={0.05}
          defaultValue={0}
          format={formats.ev}
          onChange={(exposure_ev, final) => send({ film: { exposure_ev } }, final)}
        />
        <Slider
          label="Strength"
          value={item.strength}
          min={0}
          max={100}
          step={1}
          defaultValue={100}
          format={formats.percent}
          onChange={(strength, final) => send({ strength }, final)}
        />
      </Card>
      <FilmCredit store={store} credit={catalogue?.credit} />
    </>
  );
}

/** The stock a new Film node has: Reset in the Film card puts it back. */
const DEFAULT_STOCK_ID = "kodak_portra_400";

/**
 * Film › Print: the Paper (Matching paper and the six papers; the quiet line names the matching paper), Print exposure, the Lights
 * (Yellow, Magenta) with their hint and Cyan under More. A slide has no print: the tab is dimmed and says so.
 */
export function PrintTab({ store, rs, node, item }: Omit<FilmTabProps, "photo">) {
  const { catalogue } = useFilmCatalogue(store);
  const [more, setMore] = useState(false);
  const stock = catalogue?.stocks.find((s) => s.id === item.film.stock);
  const slide = stock?.kind === "slide";
  const send = (partial: FilmPatch, final = true): void => void store.update(patchOf(node, partial), final, rs.photo_id);
  const lights = item.print.lights;
  const matching = catalogue ? paperName(catalogue, stock?.matching_paper ?? null) : null;
  const shownPaper = item.print.paper ?? MATCHING;
  const light = (label: "Yellow" | "Magenta" | "Cyan", field: "y" | "m" | "c") => (
    <Slider
      label={label}
      value={lights[field]}
      min={-50}
      max={50}
      step={0.5}
      defaultValue={0}
      format={lightFormat}
      disabled={slide}
      onChange={(value, final) => send({ print: { lights: { [field]: value } } }, final)}
    />
  );
  return (
    <div className="ym-film-print" data-slide={slide || undefined}>
      {slide ? (
        <p className="ym-film-slide" role="note">
          {SLIDE_SENTENCE}
        </p>
      ) : null}
      <Card title="Paper" action={<ResetLink size="card" disabled={slide} onClick={() => send({ print: { paper: null, exposure_ev: 0 } })} />}>
        <Field label="Paper">
          <Select
            value={slide ? NO_PAPER : shownPaper}
            disabled={slide || !catalogue}
            onValueChange={(next) => send({ print: { paper: next === MATCHING ? null : next } })}
          >
            <SelectTrigger aria-label="Paper" className="ym-film-select">
              <SelectValue>{slide ? NO_PAPER : item.print.paper === null ? "Matching paper" : (catalogue && paperName(catalogue, item.print.paper)) || item.print.paper}</SelectValue>
            </SelectTrigger>
            <SelectContent>
              {slide ? <SelectItem value={NO_PAPER}>{NO_PAPER}</SelectItem> : null}
              <SelectItem value={MATCHING}>Matching paper</SelectItem>
              {catalogue?.papers.map((paper) => (
                <SelectItem key={paper.id} value={paper.id}>
                  {paper.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </Field>
        {!slide && matching && item.print.paper === null ? <RowNote className="ym-film-line">{matching}</RowNote> : null}
        <Slider
          label="Print exposure"
          value={item.print.exposure_ev}
          min={-3}
          max={3}
          step={0.05}
          defaultValue={0}
          format={formats.ev}
          disabled={slide}
          onChange={(exposure_ev, final) => send({ print: { exposure_ev } }, final)}
        />
      </Card>
      <Card title="Lights" last action={<ResetLink size="card" disabled={slide} onClick={() => send({ print: { lights: { c: 0, m: 0, y: 0 } } })} />}>
        {light("Yellow", "y")}
        {light("Magenta", "m")}
        <RowNote className="ym-film-line">{LIGHTS_HINT}</RowNote>
        <button type="button" className="ym-film-more" aria-expanded={more} disabled={slide} onClick={() => setMore((was) => !was)}>
          <ChevronRightIcon aria-hidden="true" data-open={more || undefined} />
          More
        </button>
        {more ? light("Cyan", "c") : null}
      </Card>
    </div>
  );
}

/** Film › Halation: Amount (0 to 300 %) and Size (50 to 200 %); 100 is the stock's own. */
export function HalationTab({ store, rs, node, item }: Omit<FilmTabProps, "photo">) {
  const send = (partial: FilmPatch, final = true): void => void store.update(patchOf(node, partial), final, rs.photo_id);
  return (
    <Card title="Halation" last action={<ResetLink size="card" onClick={() => send({ halation: { amount: 100, size: 100 } })} />}>
      <Slider label="Amount" value={item.halation.amount} min={0} max={300} step={1} defaultValue={100} format={formats.percent} onChange={(amount, final) => send({ halation: { amount } }, final)} />
      <Slider label="Size" value={item.halation.size} min={50} max={200} step={1} defaultValue={100} format={formats.percent} onChange={(size, final) => send({ halation: { size } }, final)} />
      <RowNote className="ym-film-line">100% is the film's own.</RowNote>
    </Card>
  );
}

/** Film › Grain: Amount (0 to 200 %) and Size (50 to 200 %); 100 is the stock's own. The seed is not here. */
export function GrainTab({ store, rs, node, item }: Omit<FilmTabProps, "photo">) {
  const send = (partial: FilmPatch, final = true): void => void store.update(patchOf(node, partial), final, rs.photo_id);
  return (
    <Card title="Grain" last action={<ResetLink size="card" onClick={() => send({ grain: { amount: 100, size: 100 } })} />}>
      <Slider label="Amount" value={item.grain.amount} min={0} max={200} step={1} defaultValue={100} format={formats.percent} onChange={(amount, final) => send({ grain: { amount } }, final)} />
      <Slider label="Size" value={item.grain.size} min={50} max={200} step={1} defaultValue={100} format={formats.percent} onChange={(size, final) => send({ grain: { size } }, final)} />
      <RowNote className="ym-film-line">100% is the film's own.</RowNote>
    </Card>
  );
}

/** The sign used by the lights, for tests. */
export const LIGHT_MINUS = MINUS;
