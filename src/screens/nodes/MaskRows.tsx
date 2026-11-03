import { EyeIcon, EyeOffIcon } from "lucide-react";
import { ContextMenu, ContextMenuContent, ContextMenuItem, ContextMenuTrigger } from "@/components/glass/context-menu";
import { Item } from "@/components/glass/item";
import { Spinner } from "@/components/glass/spinner";
import type { MaskItem, MaskMode } from "../../api/types";
import { Button, Segmented } from "../../components";
import type { SegmentedOption } from "../../components";
import { FINDING, isFinding, isUnsupported, maskRowLabel, maskRowName, maskSubtitle, objectBusyText } from "../../state/maskRows";
import type { ObjectBusy, VisionRows } from "../../state/maskRows";
import { MASK_KIND_ICON } from "./AddMaskMenu";

/** The mockups' Invert glyph: a square split on its diagonal, the lower half filled. */
function InvertGlyph() {
  return (
    <svg viewBox="0 0 16 16" width="16" height="16" aria-hidden="true" focusable="false">
      <rect x="1.5" y="1.5" width="13" height="13" rx="3" fill="none" stroke="currentColor" strokeWidth="1.5" />
      <path d="M13 3 3 13h8a2 2 0 0 0 2-2z" fill="currentColor" />
    </svg>
  );
}
export { InvertGlyph };

const MODES: ReadonlyArray<SegmentedOption<MaskMode>> = [
  { value: "add", label: "Add" },
  { value: "subtract", label: "Subtract" },
];

export interface MaskRowsProps {
  rows: readonly MaskItem[];
  /** The selected row, or none. */
  selected: number | null;
  onSelect: (index: number) => void;
  /** A change to a row: its mode, its Invert, its eye. */
  onEdit: (index: number, edit: { mode?: MaskMode; invert?: boolean; enabled?: boolean }) => void;
  /** Takes the row out (Delete or Backspace on a selected row, or the context menu's Delete mask). */
  onRemove: (index: number) => void;
  /** Moves a row up or down the list (the context menu). */
  onMove: (index: number, by: -1 | 1) => void;
  /** The core is making the photo's first blur map or its Subject and People masks (`viewport_busy.finding`): a Background, Subject or People row with the eye on says "Finding…" with a spinner. */
  finding?: boolean;
  /** The core works for the photo's Object rows: SAM's encoder ("Getting ready…") or a click being answered ("Finding…"). */
  object?: ObjectBusy;
  /** Subject and People on this computer and on this photo: the kinds it cannot make, and the ones that found nothing. */
  vision?: VisionRows;
}

/**
 * The rows of the Masks card (`docs/23` §7.6): for each, the kind's icon, the name in bold with its subtitle, the segmented **Add | Subtract**,
 * the **Invert** icon toggle ("Invert this mask", pressed in the mask accent) and the **eye** ("Show this mask"). The selected row has a 2 px ring
 * in the mask accent. A row's name is the kind, "mask" and the subtitle ("Radial mask, Feather 45"). **Delete or Backspace** on a row of the
 * list takes the row out (`MaskTab` handles the key for the whole card), and a right click offers **Move up · Move down · Delete mask**.
 * Choosing a row, or a control in it, selects it. While a raster is being made for the first time on a photo (a Background row's blur map) its
 * subtitle is a spinner and "Finding…" (`docs/23` §7.6; so is a Subject or People row's while Vision's masks are made); an Object row says "Getting
 * ready…" while SAM's encoder runs and "Finding…" while a click is answered. A Subject or People row on a computer that cannot make it says "Subject
 * masks need a Mac that supports them" and one whose mask found nothing says "Nothing found".
 */
export function MaskRows({ rows, selected, onSelect, onEdit, onRemove, onMove, finding = false, object, vision }: MaskRowsProps) {
  if (rows.length === 0) return null;
  return (
    <ul className="ym-mask-rows" aria-label="Masks of this node" data-mask-rows="">
      {rows.map((row, i) => {
        const name = maskRowName(row);
        // A row this computer cannot make says so, whatever else the core is doing; one being made says "Finding…"; one that found nothing says that.
        const busyText = isUnsupported(row, vision) ? null : isFinding(row, finding) ? FINDING : objectBusyText(row, object);
        return (
          <ContextMenu key={i}>
            <ContextMenuTrigger asChild>
              {/* glasscn's `Item`, as the Camera card's rows are. Choosing the row, or a control in it, selects it. */}
              <Item asChild size="sm" className="ym-mask-row" data-selected={selected === i || undefined} data-off={!row.enabled || undefined}>
                <li data-index={i} onClick={() => onSelect(i)}>
                  <button type="button" className="ym-mask-row__select" aria-label={maskRowLabel(row, finding, object, vision)} aria-pressed={selected === i}>
                    <span className="ym-mask-row__icon">{MASK_KIND_ICON[row.kind]}</span>
                    <span className="ym-mask-row__text" aria-hidden="true">
                      <span className="ym-mask-row__name">{name}</span>
                      {busyText ? (
                        <span className="ym-mask-row__sub" data-finding="">
                          <Spinner aria-hidden="true" />
                          {busyText}
                        </span>
                      ) : (
                        <span className="ym-mask-row__sub" data-unsupported={isUnsupported(row, vision) || undefined} title={isUnsupported(row, vision) ? maskSubtitle(row, vision) : undefined}>
                          {maskSubtitle(row, vision)}
                        </span>
                      )}
                    </span>
                  </button>
                  <Segmented label={`${name} mask, add or subtract`} size="sm" fit value={row.mode} options={MODES} onChange={(mode) => onEdit(i, { mode })} />
                  <Button
                    variant="quiet"
                    size="sm"
                    className="ym-mask-row__tool"
                    pressed={row.invert}
                    data-tool="invert"
                    aria-label="Invert this mask"
                    title="Invert this mask"
                    onClick={() => onEdit(i, { invert: !row.invert })}
                  >
                    <InvertGlyph />
                  </Button>
                  <Button
                    variant="quiet"
                    size="sm"
                    className="ym-mask-row__tool"
                    pressed={row.enabled}
                    data-tool="eye"
                    aria-label="Show this mask"
                    title="Show this mask"
                    onClick={() => onEdit(i, { enabled: !row.enabled })}
                  >
                    {row.enabled ? <EyeIcon aria-hidden="true" /> : <EyeOffIcon aria-hidden="true" />}
                  </Button>
                </li>
              </Item>
            </ContextMenuTrigger>
            <ContextMenuContent aria-label={`${name} mask actions`}>
              <ContextMenuItem disabled={i === 0} onSelect={() => onMove(i, -1)}>
                Move up
              </ContextMenuItem>
              <ContextMenuItem disabled={i === rows.length - 1} onSelect={() => onMove(i, 1)}>
                Move down
              </ContextMenuItem>
              <ContextMenuItem variant="destructive" onSelect={() => onRemove(i)}>
                Delete mask
              </ContextMenuItem>
            </ContextMenuContent>
          </ContextMenu>
        );
      })}
    </ul>
  );
}
