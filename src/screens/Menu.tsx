import { useRef } from "react";
import type { ReactNode } from "react";
import { DropdownMenu, DropdownMenuContent, DropdownMenuRadioGroup, DropdownMenuRadioItem, DropdownMenuTrigger } from "@/components/glass/dropdown-menu";

/** One choice of a `Menu`. */
export interface MenuOption<T extends string | number> {
  value: T;
  label: string;
}

/** What a menu's trigger button gets from the menu: a ref for the button; Radix adds the rest (`aria-haspopup`, `aria-expanded`, the keys). */
export interface MenuTriggerProps {
  ref: (node: HTMLButtonElement | null) => void;
}

export interface MenuProps<T extends string | number> {
  /** Accessible name of the menu. */
  label: string;
  options: readonly MenuOption<T>[];
  /** The checked option. */
  value: T | null;
  onSelect: (value: T) => void;
  /** Draws the button that opens the menu. Spread the props onto a `<button>`. */
  trigger: (props: MenuTriggerProps) => ReactNode;
  /** Which edge of the trigger the list lines up with. Default start. */
  align?: "start" | "end";
}

/**
 * A small pop-up list of radio items (the zoom menu, the default look picker): glasscn's `DropdownMenu` on Radix, a liquid-glass list
 * of `menuitemradio` items. The trigger opens it (click, Return, Space or ↓) and the focus goes to the checked item (the first when
 * none is); the arrows, Home and End move through the items and wrap, Return or Space chooses, Esc and a click outside close it, and
 * the focus goes back to the trigger. The list is named `label` (Radix names it by its trigger, whose text is the current value).
 */
export function Menu<T extends string | number>({ label, options, value, onSelect, trigger, align = "start" }: MenuProps<T>) {
  const list = useRef<HTMLDivElement>(null);
  // Radix focuses the list itself when it opens. A pop-up button's list opens on the item that is chosen (the first when none is), so the
  // arrows start from it. `onOpenAutoFocus` is a prop of Radix's menu content that the DropdownMenu typings leave out, hence the spread.
  const focusCheckedItem = {
    onOpenAutoFocus: (event: Event): void => {
      const items = list.current?.querySelectorAll<HTMLElement>('[role="menuitemradio"]');
      const target = list.current?.querySelector<HTMLElement>('[role="menuitemradio"][aria-checked="true"]') ?? items?.[0];
      if (!target) return;
      event.preventDefault();
      target.focus({ preventScroll: true });
    },
  };
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>{trigger({ ref: () => undefined })}</DropdownMenuTrigger>
      <DropdownMenuContent
        ref={list}
        aria-label={label}
        aria-labelledby={undefined}
        loop
        {...focusCheckedItem}
        align={align === "end" ? "end" : "start"}
        className="min-w-44"
      >
        <DropdownMenuRadioGroup
          value={value === null ? "" : String(value)}
          onValueChange={(next) => {
            const chosen = options.find((option) => String(option.value) === next);
            if (chosen) onSelect(chosen.value);
          }}
        >
          {options.map((option) => (
            <DropdownMenuRadioItem key={option.value} value={String(option.value)}>
              {option.label}
            </DropdownMenuRadioItem>
          ))}
        </DropdownMenuRadioGroup>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
