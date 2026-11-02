import { ChevronDownIcon } from "lucide-react";
import { Item } from "@/components/glass/item";
import { Button, Card, RowNote } from "../../components";
import { maskRowLabel, maskRowName, maskSubtitle } from "../../state/maskRows";
import { nodeOf } from "../../state/settingsGraph";
import type { TabProps } from "../tools/shared";

/** What the Masks card says under its rows (`docs/23` §7.6), word for word. */
export const MASKS_ADD_UP = "Masks add up. Subtract cuts a mask out. Colour and brightness ranges keep only their part of the rest.";

/** Why Add mask is off until the Mask tab is built (N-12). */
export const MASKS_LATER = "Masks come in the next step";

/**
 * A corrector's Mask tab (`docs/23` §7.6), as far as N-09 builds it: the Masks card with the node's rows **listed, read only** (the kind in
 * bold, its subtitle under it, and Add or Subtract, with Off for a row that is not used), the line "Masks add up…" and **Add mask ▾ off**,
 * with the tooltip "Masks come in the next step". The rows' controls, the Add mask menu, Invert, Reset, the selected mask's card and the View
 * card are N-12's.
 */
export function MaskTab({ rs, node }: Pick<TabProps, "rs"> & { node: number }) {
  const item = nodeOf(rs.settings, node);
  const rows = item?.kind === "corrector" ? item.masks.items : [];
  return (
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
  );
}
