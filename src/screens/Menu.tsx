import { useEffect, useId, useRef, useState } from "react";
import type { KeyboardEvent, ReactNode } from "react";

/** One choice of a `Menu`. */
export interface MenuOption<T extends string | number> {
  value: T;
  label: string;
}

/** What a menu's trigger button needs from the menu. */
export interface MenuTriggerProps {
  ref: (node: HTMLButtonElement | null) => void;
  "aria-haspopup": "menu";
  "aria-expanded": boolean;
  "aria-controls": string | undefined;
  onClick: () => void;
  onKeyDown: (event: KeyboardEvent<HTMLButtonElement>) => void;
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
 * A small pop-up list of radio items (the zoom menu, the default look picker). The trigger opens
 * it; the arrows, Home and End move through the items, Return or Space chooses, Esc and a click
 * outside close it and focus goes back to the trigger.
 */
export function Menu<T extends string | number>({ label, options, value, onSelect, trigger, align = "start" }: MenuProps<T>) {
  const [open, setOpen] = useState(false);
  const listId = useId();
  const root = useRef<HTMLDivElement>(null);
  const button = useRef<HTMLButtonElement | null>(null);
  const items = useRef<Array<HTMLButtonElement | null>>([]);

  const close = (returnFocus: boolean): void => {
    setOpen(false);
    if (returnFocus) button.current?.focus();
  };

  // Focus the checked item (or the first) when the list opens, and only then: the screen draws again
  // for many reasons while the menu is open, and focus must stay where the arrows put it.
  const latest = useRef({ options, value });
  latest.current = { options, value };
  useEffect(() => {
    if (!open) return;
    const { options: list, value: current } = latest.current;
    items.current[Math.max(0, list.findIndex((o) => o.value === current))]?.focus();
  }, [open]);

  // A click or focus outside closes it.
  useEffect(() => {
    if (!open) return;
    const outside = (event: Event): void => {
      if (root.current && !root.current.contains(event.target as Node)) setOpen(false);
    };
    document.addEventListener("pointerdown", outside);
    document.addEventListener("focusin", outside);
    return () => {
      document.removeEventListener("pointerdown", outside);
      document.removeEventListener("focusin", outside);
    };
  }, [open]);

  const move = (from: number, delta: number): void => {
    const next = (from + delta + options.length) % options.length;
    items.current[next]?.focus();
  };

  const onItemKeyDown = (event: KeyboardEvent<HTMLButtonElement>, index: number): void => {
    switch (event.key) {
      case "ArrowDown":
        event.preventDefault();
        move(index, 1);
        break;
      case "ArrowUp":
        event.preventDefault();
        move(index, -1);
        break;
      case "Home":
        event.preventDefault();
        items.current[0]?.focus();
        break;
      case "End":
        event.preventDefault();
        items.current[options.length - 1]?.focus();
        break;
      case "Escape":
        event.preventDefault();
        event.stopPropagation();
        close(true);
        break;
      case "Tab":
        setOpen(false);
        break;
    }
  };

  return (
    <div ref={root} className="ym-menu">
      {trigger({
        ref: (node) => {
          button.current = node;
        },
        "aria-haspopup": "menu",
        "aria-expanded": open,
        "aria-controls": open ? listId : undefined,
        onClick: () => setOpen((o) => !o),
        onKeyDown: (event) => {
          if (event.key === "ArrowDown" && !open) {
            event.preventDefault();
            setOpen(true);
          }
        },
      })}
      {open ? (
        <ul id={listId} role="menu" aria-label={label} className="ym-menu__list glass-liquid absolute top-full left-0 z-20 mt-1.5 max-h-[min(60vh,420px)] min-w-44 list-none overflow-y-auto rounded-surface-sm p-1.5 text-popover-foreground data-[align=end]:right-0 data-[align=end]:left-auto" data-align={align === "end" ? "end" : undefined}>
          {options.map((option, index) => (
            <li key={option.value} role="none">
              <button
                ref={(node) => {
                  items.current[index] = node;
                }}
                type="button"
                role="menuitemradio"
                aria-checked={option.value === value}
                className="ym-menu__item glass-focus-inset flex h-9 w-full cursor-default items-center justify-between gap-4 rounded-control-sm px-2.5 text-left text-sm whitespace-nowrap hover:bg-fill-strong focus-visible:bg-fill-strong aria-checked:font-semibold aria-checked:after:size-1.5 aria-checked:after:rounded-full aria-checked:after:bg-primary aria-checked:after:content-['']"
                onKeyDown={(event) => onItemKeyDown(event, index)}
                onClick={() => {
                  onSelect(option.value);
                  close(true);
                }}
              >
                {option.label}
              </button>
            </li>
          ))}
        </ul>
      ) : null}
    </div>
  );
}
