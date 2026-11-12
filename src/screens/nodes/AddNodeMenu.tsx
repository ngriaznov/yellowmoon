import { ChevronDownIcon, ChevronsRightIcon, FilmIcon, LayersIcon, ScanFaceIcon, SlidersVerticalIcon, SplitIcon } from "lucide-react";
import { Fragment } from "react";
import type { ReactNode } from "react";
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuSeparator, DropdownMenuTrigger } from "@/components/glass/dropdown-menu";
import { TooltipTrigger } from "@/components/glass/tooltip";
import { OptionalTooltip } from "@/components/OptionalTooltip";
import { Button } from "../../components";

/** Most nodes a photo can have, Main included (`docs/23` §3.5). */
const MAX_NODES = 64;

/** What each item of the menu adds, by the number of nodes (`docs/25` §5.1): the item is off when the photo cannot take them all. */
const ADDS = { corrector: 1, film: 1, face_refinement: 1, parallel: 2, layer: 2, trio: 5 } as const;

export interface AddNodeMenuProps {
  /** Adds a corrector after the selected node. Without it the item is off. */
  onAddCorrector?: () => void;
  /** Adds a Film node after the selected node (`docs/24`). Without it the item is off. */
  onAddFilm?: () => void;
  /** Adds a Face Refinement node after the selected node (`docs/27`). Without it the item is off. */
  onAddFaceRefinement?: () => void;
  /** Adds a Parallel with a corrector beside the selected node (`docs/25` §5.4). */
  onAddParallel?: () => void;
  /** Adds a Layer with a corrector beside the selected node. */
  onAddLayer?: () => void;
  /** Adds a Splitter, three correctors and a Combiner after the selected node. */
  onAddTrio?: () => void;
  /** How many numbered nodes the photo has: an item that adds more than a photo has room for is off, with the sentence as its tooltip. */
  count?: number;
}

/** Why an item is off at this many nodes: "A photo can have up to 64 nodes." when none fits, else how many the item adds; null when it fits. */
export function blockedText(adds: number, count: number): string | null {
  if (count + adds <= MAX_NODES) return null;
  if (adds === 1 || count >= MAX_NODES) return "A photo can have up to 64 nodes.";
  return `This adds ${adds === 2 ? "two" : "five"} nodes, and a photo can have up to 64.`;
}

function Line({ name, line }: { name: string; line: string }) {
  return (
    <span className="ym-menu-line">
      <span className="ym-menu-line__name">{name}</span>
      <span className="ym-menu-line__text">{line}</span>
    </span>
  );
}

/**
 * "Add node ▾" (`docs/23` §7.4, `docs/25` §5.1): a glass menu that opens upward from the toolbar, over the photo. Corrector, Film and Face Refinement add a
 * node after the selected one; after a rule come Parallel, Layer and Splitter and combiner, each with its line. An item is off when the
 * photo has no room for the nodes it adds: at 64 nodes all of them say "A photo can have up to 64 nodes.", Parallel and Layer (two
 * nodes) from 63 and Splitter and combiner (five) from 60 say how many they add. The tooltip is the item's description for a screen reader.
 */
export function AddNodeMenu({ onAddCorrector, onAddFilm, onAddFaceRefinement, onAddParallel, onAddLayer, onAddTrio, count = 0 }: AddNodeMenuProps) {
  const items: ReadonlyArray<{ key: keyof typeof ADDS; name: string; line: string; icon: ReactNode; onAdd?: () => void }> = [
    { key: "corrector", name: "Corrector", line: "Grades all or part of the photo.", icon: <SlidersVerticalIcon aria-hidden="true" />, onAdd: onAddCorrector },
    { key: "film", name: "Film", line: "Models a colour negative, a print and a scan.", icon: <FilmIcon aria-hidden="true" />, onAdd: onAddFilm },
    { key: "face_refinement", name: "Face Refinement", line: "Refines each face: skin, under-eyes and teeth.", icon: <ScanFaceIcon aria-hidden="true" />, onAdd: onAddFaceRefinement },
    { key: "parallel", name: "Parallel", line: "Grade branches side by side.", icon: <ChevronsRightIcon aria-hidden="true" />, onAdd: onAddParallel },
    { key: "layer", name: "Layer", line: "Blend one branch over another.", icon: <LayersIcon aria-hidden="true" />, onAdd: onAddLayer },
    { key: "trio", name: "Splitter and combiner", line: "Split a picture, then join it again.", icon: <SplitIcon aria-hidden="true" />, onAdd: onAddTrio },
  ];
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="header">
          Add node
          <ChevronDownIcon data-icon="inline-end" aria-hidden="true" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent side="top" align="start" aria-label="Add node" className="ym-add-node-menu">
        {items.map((item, i) => {
          const blocked = blockedText(ADDS[item.key], count);
          return (
            <Fragment key={item.key}>
              {i === 3 ? <DropdownMenuSeparator /> : null}
              <OptionalTooltip text={blocked} side="right">
                <TooltipTrigger asChild>
                  <DropdownMenuItem
                    disabled={!item.onAdd || blocked !== null}
                    className="data-disabled:pointer-events-auto"
                    aria-description={blocked ?? undefined}
                    onSelect={() => item.onAdd?.()}
                  >
                    {item.icon}
                    <Line name={item.name} line={item.line} />
                  </DropdownMenuItem>
                </TooltipTrigger>
              </OptionalTooltip>
            </Fragment>
          );
        })}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
