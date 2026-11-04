/**
 * What the mock answers for the Film node (`docs/24`): the catalogue as `names.json` lists it (the 20 camera films and the 6 still papers),
 * and a flat-shape rendering of each stock for the tiles and the viewer, in the manner of the look grades of `grade.ts`. The numbers of
 * the renderings are hand-set and say nothing about how a real stock looks: the app draws the real ones with the core.
 */
import type { DevelopSettings, FilmCatalogue, FilmPaperInfo, FilmStockInfo } from "../api/types";
import { cardStates } from "../state/graphOps";
import { filmsOf } from "../state/settingsGraph";
import { FILM_CREDIT, FILM_DISCLAIMER, FILM_TRADEMARKS } from "./filmCredit";
import { gradeScene } from "./grade";
import type { LookGrade } from "./grade";
import type { Scene } from "./scenes";

/** The camera films, in the order of `names.json`. */
export const MOCK_FILM_STOCKS: readonly FilmStockInfo[] = [
  { id: "fujifilm_c200", name: "Fujifilm C200", kind: "colour_negative", group: "Fujifilm", iso: 200, matching_paper: "fujifilm_crystal_archive_typeii" },
  { id: "fujifilm_pro_400h", name: "Fujifilm Pro 400H", kind: "colour_negative", group: "Fujifilm", iso: 400, matching_paper: "fujifilm_crystal_archive_typeii" },
  { id: "fujifilm_xtra_400", name: "Fujifilm X-Tra 400", kind: "colour_negative", group: "Fujifilm", iso: 400, matching_paper: "fujifilm_crystal_archive_typeii" },
  { id: "kodak_ektar_100", name: "Kodak Ektar 100", kind: "colour_negative", group: "Kodak", iso: 100, matching_paper: "kodak_portra_endura" },
  { id: "kodak_gold_200", name: "Kodak Gold 200", kind: "colour_negative", group: "Kodak", iso: 200, matching_paper: "kodak_portra_endura" },
  { id: "kodak_portra_160", name: "Kodak Portra 160", kind: "colour_negative", group: "Kodak", iso: 160, matching_paper: "kodak_portra_endura" },
  { id: "kodak_portra_400", name: "Kodak Portra 400", kind: "colour_negative", group: "Kodak", iso: 400, matching_paper: "kodak_portra_endura" },
  { id: "kodak_portra_800", name: "Kodak Portra 800", kind: "colour_negative", group: "Kodak", iso: 800, matching_paper: "kodak_portra_endura" },
  { id: "kodak_portra_800_push1", name: "Kodak Portra 800 (Push 1)", kind: "colour_negative", group: "Kodak", iso: 800, matching_paper: "kodak_portra_endura" },
  { id: "kodak_portra_800_push2", name: "Kodak Portra 800 (Push 2)", kind: "colour_negative", group: "Kodak", iso: 800, matching_paper: "kodak_portra_endura" },
  { id: "kodak_ultramax_400", name: "Kodak Ultramax 400", kind: "colour_negative", group: "Kodak", iso: 400, matching_paper: "kodak_portra_endura" },
  { id: "kodak_verita_200d", name: "Kodak Verita 200D", kind: "cine_negative", group: "Kodak", iso: 200, matching_paper: "kodak_portra_endura" },
  { id: "kodak_vision3_200t", name: "Kodak Vision3 200T", kind: "cine_negative", group: "Kodak", iso: 200, matching_paper: "kodak_portra_endura" },
  { id: "kodak_vision3_250d", name: "Kodak Vision3 250D", kind: "cine_negative", group: "Kodak", iso: 250, matching_paper: "kodak_portra_endura" },
  { id: "kodak_vision3_500t", name: "Kodak Vision3 500T", kind: "cine_negative", group: "Kodak", iso: 500, matching_paper: "kodak_portra_endura" },
  { id: "kodak_vision3_50d", name: "Kodak Vision3 50D", kind: "cine_negative", group: "Kodak", iso: 50, matching_paper: "kodak_portra_endura" },
  { id: "fujifilm_provia_100f", name: "Fujifilm Provia 100F", kind: "slide", group: "Fujifilm", iso: 100, matching_paper: null },
  { id: "fujifilm_velvia_100", name: "Fujifilm Velvia 100", kind: "slide", group: "Fujifilm", iso: 100, matching_paper: null },
  { id: "kodak_ektachrome_100", name: "Kodak Ektachrome 100", kind: "slide", group: "Kodak", iso: 100, matching_paper: null },
  { id: "kodak_kodachrome_64", name: "Kodak Kodachrome 64", kind: "slide", group: "Kodak", iso: 64, matching_paper: null },
];

/** The still papers; the two cine print films are hidden and not listed. */
export const MOCK_FILM_PAPERS: readonly FilmPaperInfo[] = [
  { id: "fujifilm_crystal_archive_typeii", name: "Fujifilm Crystal Archive Type II" },
  { id: "kodak_ektacolor_edge", name: "Kodak Ektacolor Edge" },
  { id: "kodak_endura_premier", name: "Kodak Professional Endura Premier" },
  { id: "kodak_portra_endura", name: "Kodak Professional Portra Endura" },
  { id: "kodak_supra_endura", name: "Kodak Professional Supra Endura" },
  { id: "kodak_ultra_endura", name: "Kodak Professional Ultra Endura" },];

/** `film_catalogue` as the mock answers it. */
export function mockFilmCatalogue(): FilmCatalogue {
  return {
    stocks: MOCK_FILM_STOCKS.map((s) => ({ ...s })),
    papers: MOCK_FILM_PAPERS.map((p) => ({ ...p })),
    profile_version: "0.3.2",
    credit: FILM_CREDIT,
    disclaimer: FILM_DISCLAIMER,
    trademarks: FILM_TRADEMARKS,
  };
}

/** True for the stocks that have no print (the slides). */
export const isSlideStock = (stock: string): boolean => MOCK_FILM_STOCKS.find((s) => s.id === stock)?.kind === "slide";

const SOFT: LookGrade = { hi: "#F0D9C0", ha: 0.18, sh: "#5A4F4C", sa: 0.18, ds: 0.04 };

/** Hand-set flat-shape gradings by stock id; a stock that is not here takes the kind's. */
const STOCK_GRADES: Readonly<Record<string, LookGrade>> = {
  kodak_portra_400: SOFT,
  kodak_portra_160: { hi: "#F2DEC8", ha: 0.15, sh: "#5C5654", sa: 0.15, ds: 0.06 },
  kodak_portra_800: { hi: "#F0CFA8", ha: 0.22, sh: "#4F4440", sa: 0.22, ds: 0.04 },
  kodak_portra_800_push1: { hi: "#EFC79C", ha: 0.26, sh: "#453A36", sa: 0.26, ds: 0.03 },
  kodak_portra_800_push2: { hi: "#EDBE90", ha: 0.3, sh: "#3C312E", sa: 0.3, ds: 0.02 },
  kodak_ektar_100: { hi: "#FFD9A8", ha: 0.14, sh: "#1B2A33", sa: 0.3, ds: 0 },
  kodak_gold_200: { hi: "#F8C878", ha: 0.28, sh: "#4A3322", sa: 0.22, ds: 0 },
  kodak_ultramax_400: { hi: "#F5C98A", ha: 0.24, sh: "#43362A", sa: 0.24, ds: 0 },
  fujifilm_c200: { hi: "#D6E4DC", ha: 0.2, sh: "#2B3E3A", sa: 0.24, ds: 0.03 },
  fujifilm_pro_400h: { hi: "#D3E6E0", ha: 0.24, sh: "#34443F", sa: 0.2, ds: 0.05 },
  fujifilm_xtra_400: { hi: "#C9DDE6", ha: 0.2, sh: "#25363F", sa: 0.26, ds: 0.02 },
  fujifilm_velvia_100: { hi: "#FFE0B0", ha: 0.1, sh: "#0C1F2B", sa: 0.45, ds: 0 },
  fujifilm_provia_100f: { hi: "#F2F0E6", ha: 0.08, sh: "#10222E", sa: 0.35, ds: 0 },
  kodak_ektachrome_100: { hi: "#E6EEF4", ha: 0.1, sh: "#0F2230", sa: 0.38, ds: 0 },
  kodak_kodachrome_64: { hi: "#F6D2A0", ha: 0.16, sh: "#2A1C14", sa: 0.42, ds: 0 },
};
const CINE: LookGrade = { hi: "#E8D2B8", ha: 0.22, sh: "#3C4A52", sa: 0.22, ds: 0.1 };

/** The scene as it comes out of this stock at this Strength (percent), in the flat shapes the mock draws. */
export function filmScene(scene: Scene, stock: string, strength = 100): Scene {
  const info = MOCK_FILM_STOCKS.find((s) => s.id === stock);
  const grade = STOCK_GRADES[stock] ?? (info?.kind === "cine_negative" ? CINE : SOFT);
  return gradeScene(scene, grade, Math.max(0, Math.min(100, strength)) / 100);
}

/** The scene through every live Film node that is not bypassed, in the order of their numbers in the plan. */
export function sceneThroughFilms(scene: Scene, settings: DevelopSettings): Scene {
  const states = cardStates(settings);
  return filmsOf(settings)
    .filter((item) => !item.bypass && states.get(item.id as number)?.live !== false)
    .sort((a, b) => (states.get(a.id as number)?.number ?? 0) - (states.get(b.id as number)?.number ?? 0))
    .reduce((out, item) => filmScene(out, item.film.stock, item.strength), scene);
}
