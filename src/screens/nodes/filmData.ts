/**
 * What the Film node's panel says about a stock (`docs/24` sections 4.1 and 4.6): the groups of the picker, the quiet line under the
 * Stock field and the format of the lights. The list itself is `names.json` through the core (`film_catalogue`); nothing here knows a stock.
 */
import { MINUS } from "../../components";
import type { FilmCatalogue, FilmStockInfo } from "../../api/types";

/** The groups of the stock picker, in order, with the name of the kind in `names.json`. */
export const FILM_GROUPS: ReadonlyArray<{ kind: string; label: string }> = [
  { kind: "colour_negative", label: "Colour negative" },
  { kind: "cine_negative", label: "Cine negative" },
  { kind: "slide", label: "Slide" },
];

/** The words for a kind ("Colour negative"). */
export const kindLabel = (kind: string): string => FILM_GROUPS.find((g) => g.kind === kind)?.label ?? kind;

/** "Colour negative · ISO 400": the quiet line under the Stock field, and "ISO 400" on a tile. */
export function stockLine(stock: FilmStockInfo): string {
  return stock.iso === null ? kindLabel(stock.kind) : `${kindLabel(stock.kind)} · ISO ${stock.iso}`;
}

/** "ISO 400". */
export const isoLine = (stock: FilmStockInfo): string => (stock.iso === null ? "" : `ISO ${stock.iso}`);

/** The stocks of one group, in the order of the list. */
export const stocksOf = (catalogue: FilmCatalogue, kind: string): FilmStockInfo[] => catalogue.stocks.filter((s) => s.kind === kind);

/** The name of a paper, or null when the list does not have it. */
export const paperName = (catalogue: FilmCatalogue, id: string | null): string | null => (id === null ? null : (catalogue.papers.find((p) => p.id === id)?.name ?? null));

/** The format of a light (-50 to 50 in halves): "0", "+5", "−12.5". */
export function lightFormat(value: number): string {
  const rounded = Math.round(value * 2) / 2;
  if (rounded === 0) return "0";
  return (rounded > 0 ? "+" : MINUS) + String(Math.abs(rounded));
}

/** "None" is what the Paper field shows for a slide, which is scanned as it is. */
export const NO_PAPER = "None";
/** The Print tab's sentence for a slide (`docs/24` section 4.6). */
export const SLIDE_SENTENCE = "Slide film is scanned as it is. There is no print.";
/** The hint under the lights. */
export const LIGHTS_HINT = "0 is neutral for this film and paper.";
/** What the Film tab says when the list of stocks cannot be read. */
export const FILM_DATA_MISSING = "Yellowmoon's film data is missing. Reinstall the app.";
/** The line the Output's Look tab carries while a Film node is live (`docs/24` section 4.6). */
export const OUTPUT_FILM_NOTE = "The look acts after the Film node. Choose Neutral to see the print as scanned.";
