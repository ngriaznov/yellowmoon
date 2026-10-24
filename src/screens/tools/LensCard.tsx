import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/glass/select";
import { Card, Field, formats, ResetLink, RowNote, Slider } from "../../components";
import type { LensKind, LensSettings, ResolvedSettings } from "../../api/types";
import { useAppState } from "../../state/context";
import type { AppStore } from "../../state/store";

/** The note under the card while the lens data could not be loaded (`docs/21` §1: the list is empty, so nothing can be chosen). */
export const LENS_MISSING_NOTE = "Lens data are missing, so lenses can't be used.";

/** No lens: the Default lens with all three amounts at 0 (`docs/21` §2). The card's Reset sends it. */
export const NO_LENS: LensSettings = { kind: "default", flares: 0, edges: 0, anamorphic: 0 };

/** The three amount sliders, in the card's order (`docs/21` §1). */
const AMOUNTS = [
  { field: "flares", label: "Flares" },
  { field: "edges", label: "Edges" },
  { field: "anamorphic", label: "Anamorphic" },
] as const satisfies ReadonlyArray<{ field: keyof Omit<LensSettings, "kind">; label: string }>;

/** A lens's name before the list has arrived: its id with a capital, which is how every name of `data/lenses/` is spelt. */
const kindName = (kind: LensKind): string => kind.charAt(0).toUpperCase() + kind.slice(1);

/**
 * The Look tab's Lens card (`docs/21` §1, `docs/11` *Lens card*): a lens choice (glasscn's `Select`, each lens with its line under its
 * name in the menu), the Flares, Edges and Anamorphic sliders (0 to 100, step 1, the detent and Reset at 0, `formats.integer`) and a
 * note with the chosen lens's line.
 *
 * Every change sends the whole `lens` object, the photo's values with the change in them, through `store.update` for the photo the
 * card was drawn for: one undo step per drag and per choice. Choosing a lens sends its `kind` with its starting Flares and Edges in
 * that one patch (the core moves no amount by itself); Anamorphic stays. The card's Reset sends the Default lens at 0, 0 and 0.
 *
 * The lenses come from `lens_list`, asked for once (`store.state.lenses`). While the list is empty (the lens data could not be loaded)
 * the menu and the sliders are off and the note says so; the card's Reset still works.
 */
export function LensCard({ store, rs }: { store: AppStore; rs: ResolvedSettings }) {
  const lenses = useAppState((s) => s.lenses);
  const lens = rs.values.lens;
  const loading = lenses === null;
  const missing = lenses !== null && lenses.length === 0;
  const chosen = lenses?.find((l) => l.kind === lens.kind);
  const send = (next: LensSettings, final: boolean): void => void store.update({ lens: next }, final, rs.photo_id);

  const choose = (kind: string): void => {
    const info = lenses?.find((l) => l.kind === kind);
    if (info) send({ kind: info.kind, flares: info.start_flares, edges: info.start_edges, anamorphic: lens.anamorphic }, true);
  };

  return (
    <Card className="ym-lens-card" title="Lens" action={<ResetLink size="card" onClick={() => send(NO_LENS, true)} />}>
      <Field label="Lens">
        <Select value={lens.kind} onValueChange={choose} disabled={loading || missing}>
          <SelectTrigger aria-label="Lens" className="w-full">
            <SelectValue>{chosen?.name ?? kindName(lens.kind)}</SelectValue>
          </SelectTrigger>
          <SelectContent>
            {(lenses ?? []).map((info) => (
              <SelectItem key={info.kind} value={info.kind} textValue={info.name}>
                <span className="ym-lens-option">
                  <span className="ym-lens-option__name">{info.name}</span>
                  {/* The space is not drawn (the two spans are flex items) but it keeps the option's name one sentence: "Vintage Glows wide open, …". */}{" "}
                  <span className="ym-lens-option__line">{info.line}</span>
                </span>
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </Field>
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
      {missing ? <RowNote>{LENS_MISSING_NOTE}</RowNote> : chosen ? <RowNote>{chosen.line}</RowNote> : null}
    </Card>
  );
}
