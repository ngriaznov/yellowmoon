import { ChevronDownIcon } from "lucide-react";
import { Item } from "@/components/glass/item";
import type { MouseEvent } from "react";
import { Button, Card, Field, KeyHint, RowNote, Segmented } from "../../components";
import type { SegmentedOption } from "../../components";
import type { MaskStyle } from "../../api/types";
import { useAppState, useStore } from "../../state/context";
import { maskRowLabel, maskRowName, maskSubtitle } from "../../state/maskRows";
import { nodeOf } from "../../state/settingsGraph";
import type { TabProps } from "../tools/shared";

/** What the Masks card says under its rows (`docs/23` §7.6), word for word. */
export const MASKS_ADD_UP = "Masks add up. Subtract cuts a mask out. Colour and brightness ranges keep only their part of the rest.";

/** Why Add mask is off until the Mask tab is built (N-12). */
export const MASKS_LATER = "Masks come in the next step";

/** The View card's two segments (`docs/23` §7.6). The mockups draw a third, "Off"; the toolbar's Show mask and M turn the view on and off. */
export const MASK_VIEW_OPTIONS: ReadonlyArray<SegmentedOption<MaskStyle>> = [
  { value: "overlay", label: "Overlay" },
  { value: "mask_only", label: "Mask only" },
];

/**
 * The View card (`docs/23` §7.6): "VIEW" with the key chip M, and one row, Show mask, with **Overlay · Mask only**. Choosing a segment sets the
 * style and turns the view on, also when it is the segment that is already chosen (the view may be off). The style is kept with the view.
 */
export function ViewCard() {
  const store = useStore();
  const { style } = useAppState((s) => s.maskView);
  // A click on the chosen segment is not a change for the control, but it is a request for the view.
  const onClick = (event: MouseEvent<HTMLDivElement>): void => {
    const segment = (event.target as HTMLElement).closest<HTMLElement>('[role="radio"]');
    const value = segment?.dataset.value;
    if (value === "overlay" || value === "mask_only") store.setMaskView({ style: value });
  };
  return (
    <Card
      title="View"
      last
      action={
        <span className="ym-card__keys" aria-hidden="true">
          <KeyHint>M</KeyHint>
        </span>
      }
    >
      <Field label="Show mask">
        <div className="ym-view-style" onClick={onClick}>
          <Segmented label="Show mask" value={style} onChange={(next) => store.setMaskView({ style: next })} options={MASK_VIEW_OPTIONS} />
        </div>
      </Field>
    </Card>
  );
}

/**
 * A corrector's Mask tab (`docs/23` §7.6), as far as N-09 and N-11 build it: the Masks card with the node's rows **listed, read only** (the kind in
 * bold, its subtitle under it, and Add or Subtract, with Off for a row that is not used), the line "Masks add up…" and **Add mask ▾ off**,
 * with the tooltip "Masks come in the next step"; under it the View card (N-11). The rows' controls, the Add mask menu, Invert, Reset and the
 * selected mask's card are N-12's.
 */
export function MaskTab({ rs, node }: Pick<TabProps, "rs"> & { node: number }) {
  const item = nodeOf(rs.settings, node);
  const rows = item?.kind === "corrector" ? item.masks.items : [];
  return (
    <>
    <Card
      title="Masks"
      action={
        <Button variant="header" size="sm" disabled aria-haspopup="menu" title={MASKS_LATER} icon={undefined}>
          Add mask
          <ChevronDownIcon aria-hidden="true" />
        </Button>
      }
    >
      {rows.length > 0 ? (
        <ul className="ym-mask-rows" aria-label="Masks of this node">
          {rows.map((row, i) => (
            // glasscn's `Item`, as the Camera card's rows are; the row's name is the kind, "mask" and the subtitle.
            <Item key={i} asChild size="sm" className="ym-mask-row" data-off={!row.enabled || undefined}>
              <li aria-label={maskRowLabel(row)}>
                <span className="ym-mask-row__text" aria-hidden="true">
                  <span className="ym-mask-row__name">{maskRowName(row)}</span>
                  <span className="ym-mask-row__sub">{maskSubtitle(row)}</span>
                </span>
                <span className="ym-mask-row__mode" aria-hidden="true">
                  {row.enabled ? (row.mode === "subtract" ? "Subtract" : "Add") : "Off"}
                </span>
              </li>
            </Item>
          ))}
        </ul>
      ) : null}
      <RowNote>{MASKS_ADD_UP}</RowNote>
    </Card>
    <ViewCard />
    </>
  );
}
