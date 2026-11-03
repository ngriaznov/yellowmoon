import { ChevronDownIcon, ChevronsRightIcon, LayersIcon, SlidersVerticalIcon, SplitIcon } from "lucide-react";
import type { ReactNode } from "react";
import { Badge } from "@/components/glass/badge";
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuSeparator, DropdownMenuTrigger } from "@/components/glass/dropdown-menu";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/glass/tooltip";
import { Button } from "../../components";

/** The kinds of node that come later (`docs/23` §7.4): shown greyed, so the menu says where the graph is going. */
const LATER: ReadonlyArray<{ name: string; line: string; icon: ReactNode }> = [
  { name: "Parallel", line: "Grade branches side by side.", icon: <ChevronsRightIcon aria-hidden="true" /> },
  { name: "Layer", line: "Blend one branch over another.", icon: <LayersIcon aria-hidden="true" /> },
  { name: "Splitter and combiner", line: "Split a picture, then join it again.", icon: <SplitIcon aria-hidden="true" /> },
];

/** What the greyed items say when the pointer rests on them. */
export const LATER_TOOLTIP = "Not in the first version";

export interface AddNodeMenuProps {
  /** Adds a corrector after the selected node. Without it the item is off. */
  onAddCorrector?: () => void;
  /** Why a node cannot be added now ("A photo can have up to 64 nodes."): the item is off, and the sentence is its tooltip. */
  blocked?: string | null;
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
 * "Add node ▾" (`docs/23` §7.4): a glass menu that opens upward from the toolbar, over the photo. Corrector first (it adds a node after the
 * selected one, and is off at 64 nodes with the sentence as its tooltip); then, after a rule,
 * Parallel, Layer and Splitter and combiner, greyed with the tag "Later" and `aria-disabled` (Radix sets it), each with the
 * tooltip "Not in the first version".
 */
export function AddNodeMenu({ onAddCorrector, blocked = null }: AddNodeMenuProps) {
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="header">
          Add node
          <ChevronDownIcon data-icon="inline-end" aria-hidden="true" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent side="top" align="start" aria-label="Add node" className="ym-add-node-menu">
        <Tooltip>
          <TooltipTrigger asChild>
            <DropdownMenuItem
              disabled={!onAddCorrector || blocked !== null}
              className="data-disabled:pointer-events-auto"
              aria-description={blocked ?? undefined}
              onSelect={() => onAddCorrector?.()}
            >
              <SlidersVerticalIcon aria-hidden="true" />
              <Line name="Corrector" line="Grades all or part of the photo." />
            </DropdownMenuItem>
          </TooltipTrigger>
          {blocked !== null ? <TooltipContent side="right">{blocked}</TooltipContent> : null}
        </Tooltip>
        <DropdownMenuSeparator />
        {LATER.map((item) => (
          <Tooltip key={item.name}>
            <TooltipTrigger asChild>
              <DropdownMenuItem disabled className="data-disabled:pointer-events-auto" aria-description={LATER_TOOLTIP}>
                {item.icon}
                <Line name={item.name} line={item.line} />
                <Badge variant="outline" className="ym-menu-later">
                  Later
                </Badge>
              </DropdownMenuItem>
            </TooltipTrigger>
            <TooltipContent side="right">{LATER_TOOLTIP}</TooltipContent>
          </Tooltip>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
