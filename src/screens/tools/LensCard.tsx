import { Card, ChoiceList, formats, ResetLink, RowNote, Slider } from "../../components";
import type { LensInfo, LensKind, LensSettings, ResolvedSettings } from "../../api/types";
import { useAppState } from "../../state/context";
import type { AppStore } from "../../state/store";

/** The note under the Lens card's row while the lens data could not be loaded (`docs/21` §1: the list is empty, so nothing can be chosen). */
export const LENS_MISSING_NOTE = "Lens data are missing, so lenses can't be used.";

/** No lens: the Default lens with all three amounts at 0 (`docs/21` §2). The Amounts card's Reset sends it. */
export const NO_LENS: LensSettings = { kind: "default", flares: 0, edges: 0, anamorphic: 0 };

/** The three amount sliders, in the card's order (`docs/21` §1). */
const AMOUNTS = [
  { field: "flares", label: "Flares" },
  { field: "edges", label: "Edges" },
  { field: "anamorphic", label: "Anamorphic" },
] as const satisfies ReadonlyArray<{ field: keyof Omit<LensSettings, "kind">; label: string }>;

/** A lens's name before the list has arrived: its id with a capital, which is how every name of `data/lenses/` is spelt. */
const kindName = (kind: LensKind): string => kind.charAt(0).toUpperCase() + kind.slice(1);

/** What a card of the Lens tab needs: the store, and the settings of the photo on screen. */
interface LensCardProps {
  store: AppStore;
  rs: ResolvedSettings;
}

/** Sends the whole `lens` object, the photo's values with the change in them, for the photo the card was drawn for. */
const sendLens = (store: AppStore, rs: ResolvedSettings, next: LensSettings, final: boolean): void =>
  void store.update({ lens: next }, final, rs.photo_id);

/**
 * The Lens tab's Lens card (`docs/21` §1, `docs/11` *Lens card*): the seven lenses as a list, one row each with its name and its line
 * under it, in the order `lens_list` answers (glasscn's `RadioGroup` of `Item` rows: `ChoiceList`). The chosen row is ringed in the tab
 * colour.
 *
 * Choosing a lens sends `kind` with that lens's starting Flares and Edges in one patch (the core moves no amount by itself) and keeps
 * Anamorphic: one undo step. The lenses come from `lens_list`, asked for once (`store.state.lenses`). While the list has not arrived, or
 * is empty (the lens data could not be loaded), the card shows one row, off, for the photo's own lens, and says why.
 */
export function LensCard({ store, rs }: LensCardProps) {
  const lenses = useAppState((s) => s.lenses);
  const lens = rs.values.lens;
  const missing = lenses !== null && lenses.length === 0;
  const unusable = lenses === null || missing;
  const options = unusable
    ? [{ value: lens.kind, name: kindName(lens.kind) }]
    : lenses.map((info: LensInfo) => ({ value: info.kind, name: info.name, line: info.line }));

  const choose = (kind: LensKind): void => {
    const info = lenses?.find((l) => l.kind === kind);
    if (info) sendLens(store, rs, { kind: info.kind, flares: info.start_flares, edges: info.start_edges, anamorphic: lens.anamorphic }, true);
  };

  return (
    <Card title="Lens">
      <ChoiceList label="Lens" options={options} value={lens.kind} onChange={choose} disabled={unusable} />
      {missing ? <RowNote>{LENS_MISSING_NOTE}</RowNote> : null}
    </Card>
  );
}

/**
 * The Lens tab's Amounts card (`docs/21` §1): Flares, Edges and Anamorphic, each 0 to 100, step 1, the detent and Reset at 0, shown with
 * `formats.integer`. Every change sends the whole `lens` object through `store.update` for the photo the card was drawn for; a drag is one
 * undo step. The card's Reset sends the Default lens at 0, 0 and 0. While the lens data are missing the sliders are off and Reset still
 * works.
 *
 * The chosen lens's line is in its row of the Lens card, so the card has no second copy of it. The bokeh line of `docs/21` §1 ("This photo
 * has little out-of-focus background, so its bokeh changes little.") would come under the sliders once the webview reads
 * `render_stats.lens` (`docs/21` §10); it does not yet.
 */
export function AmountsCard({ store, rs }: LensCardProps) {
  const lenses = useAppState((s) => s.lenses);
  const lens = rs.values.lens;
  const missing = lenses !== null && lenses.length === 0;
  const send = (next: LensSettings, final: boolean): void => sendLens(store, rs, next, final);

  return (
    <Card title="Amounts" action={<ResetLink size="card" onClick={() => send(NO_LENS, true)} />}>
      {AMOUNTS.map(({ field, label }) => (
        <Slider
          key={field}
          label={label}
          value={lens[field]}
          min={0}
          max={100}
          step={1}
          defaultValue={0}
          format={formats.integer}
          disabled={missing}
          onChange={(value, final) => send({ ...lens, [field]: value }, final)}
          onReset={() => send({ ...lens, [field]: 0 }, true)}
        />
      ))}
    </Card>
  );
}
