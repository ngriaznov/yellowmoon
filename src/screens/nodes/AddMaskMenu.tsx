import {
  ChevronDownIcon,
  CircleDotIcon,
  PaintbrushIcon,
  PanelTopIcon,
  PaletteIcon,
  SparklesIcon,
  SunIcon,
  UserIcon,
  ApertureIcon,
} from "lucide-react";
import { useRef, useState } from "react";
import type { ReactNode } from "react";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuGroup,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/glass/dropdown-menu";
import type { MaskKind } from "../../api/types";
import { Button } from "../../components";

/** The icon of a kind of mask (the row's and the menu's). */
export const MASK_KIND_ICON: Readonly<Record<MaskKind, ReactNode>> = {
  object: <SparklesIcon aria-hidden="true" />,
  subject: <SparklesIcon aria-hidden="true" />,
  people: <UserIcon aria-hidden="true" />,
  background: <ApertureIcon aria-hidden="true" />,
  linear: <PanelTopIcon aria-hidden="true" />,
  radial: <CircleDotIcon aria-hidden="true" />,
  brush: <PaintbrushIcon aria-hidden="true" />,
  color_range: <PaletteIcon aria-hidden="true" />,
  brightness_range: <SunIcon aria-hidden="true" />,
};

/** The gap between the menu and the panel's left edge, in CSS pixels. */
const ALONGSIDE = 8;

/** One item of the menu: the kind, its name and the line under it (`docs/23` §7.6). */
interface MenuKind {
  kind: MaskKind;
  name: string;
  line: string;
}

/** The groups of the Add mask menu with their items and lines, word for word as `docs/23` §7.6 has them. */
export const ADD_MASK_GROUPS: ReadonlyArray<{ title: string; kinds: readonly MenuKind[] }> = [
  {
    title: "Find",
    kinds: [
      { kind: "object", name: "Object", line: "Click the photo, or draw a box" },
      { kind: "subject", name: "Subject", line: "The main thing in the photo" },
      { kind: "people", name: "People", line: "Each person. macOS 14 or later" },
      { kind: "background", name: "Background", line: "What is out of focus, from blur" },
    ],
  },
  {
    title: "Shapes",
    kinds: [
      { kind: "linear", name: "Linear", line: "A gradient across the photo" },
      { kind: "radial", name: "Radial", line: "An oval, soft at the edge" },
      { kind: "brush", name: "Brush", line: "Paint where the node acts" },
    ],
  },
  {
    title: "Keys",
    kinds: [
      { kind: "color_range", name: "Colour range", line: "Pick colours to select" },
      { kind: "brightness_range", name: "Brightness range", line: "Select light or dark tones" },
    ],
  },
];

export interface AddMaskMenuProps {
  /** The kinds this machine can make (`mask_kinds`); the menu lists exactly these, a group with none is left out. Null until they are known. */
  kinds: readonly MaskKind[] | null;
  /** Adds a row of a kind (the screen selects it and starts its tool). */
  onAdd: (kind: MaskKind) => void;
  /** Why no row can be added now ("A node can have up to 64 masks."): the button is off and the sentence is its tooltip. */
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
 * "Add mask ▾" (`docs/23` §7.6): a glass menu that opens to the left of the panel, over the photo, in three groups (Find, Shapes, Keys) of
 * the kinds `mask_kinds` lists. A kind that is not in the list is **not shown**, greyed or not (Subject and People on a Mac that cannot
 * make them, Object without its weights, the kinds whose task is not built); a group left with no item is not shown either. Choosing an
 * item adds the row and starts its tool.
 */
export function AddMaskMenu({ kinds, onAdd, blocked = null }: AddMaskMenuProps) {
  const listed = new Set<MaskKind>(kinds ?? []);
  const groups = ADD_MASK_GROUPS.map((g) => ({ ...g, kinds: g.kinds.filter((k) => listed.has(k.kind)) })).filter((g) => g.kinds.length > 0);
  const off = kinds === null || groups.length === 0 || blocked !== null;
  // The menu opens to the left of the **panel**, not of the button: its right edge is the panel's left edge (`docs/23` §7.6, screen 08), a gap
  // of the menu's offset from it. The button is in the panel, so the offset is the button's distance from the panel's edge.
  const trigger = useRef<HTMLButtonElement>(null);
  const [offset, setOffset] = useState(ALONGSIDE);
  const measure = (open: boolean): void => {
    const button = trigger.current;
    const panel = button?.closest("aside");
    if (open && button && panel) setOffset(Math.max(ALONGSIDE, button.getBoundingClientRect().left - panel.getBoundingClientRect().left + ALONGSIDE));
  };
  return (
    <DropdownMenu onOpenChange={measure}>
      <DropdownMenuTrigger asChild disabled={off}>
        <Button ref={trigger} variant="header" size="sm" disabled={off} title={blocked ?? undefined}>
          Add mask
          <ChevronDownIcon data-icon="inline-end" aria-hidden="true" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent side="left" align="start" sideOffset={offset} aria-label="Add mask" className="ym-add-mask-menu">
        {groups.map((group, i) => (
          <DropdownMenuGroup key={group.title}>
            {i > 0 ? <DropdownMenuSeparator /> : null}
            <DropdownMenuLabel className="ym-menu-group">{group.title}</DropdownMenuLabel>
            {group.kinds.map((item) => (
              <DropdownMenuItem key={item.kind} onSelect={() => onAdd(item.kind)}>
                {MASK_KIND_ICON[item.kind]}
                <Line name={item.name} line={item.line} />
              </DropdownMenuItem>
            ))}
          </DropdownMenuGroup>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
