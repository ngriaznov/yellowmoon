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
 * of `menuitemradio` items. The trigger opens it (click, Return, Space or ↓); the arrows, Home and End move through the items,
 * Return or Space chooses, Esc and a click outside close it, and the focus goes back to the trigger.
 */
export function Menu<T extends string | number>({ label, options, value, onSelect, trigger, align = "start" }: MenuProps<T>) {
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>{trigger({ ref: () => undefined })}</DropdownMenuTrigger>
      <DropdownMenuContent aria-label={label} align={align === "end" ? "end" : "start"} className="ym-menu__list min-w-44">
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
