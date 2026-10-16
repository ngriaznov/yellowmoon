import { useId } from "react";
import type { ButtonHTMLAttributes, MouseEvent, ReactNode, Ref } from "react";
import { Button as GlassButton } from "@/components/glass/button";
import { Kbd } from "@/components/glass/kbd";
import { Toggle } from "@/components/glass/toggle";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/glass/tooltip";
import { cx } from "./cx";

/** Visual kind of a `Button`. */
export type ButtonVariant = "primary" | "secondary" | "header" | "quiet";

export interface ButtonProps extends Omit<ButtonHTMLAttributes<HTMLButtonElement>, "type" | "disabled"> {
  /**
   * `primary`: the accent (Open photos, Next, Export N photos), glasscn's prominent glass button. `secondary` and `header`: glasscn's glass
   * button (Open folder, Cancel, Add photos). `quiet`: its clear ghost button. A button that has `pressed` is glasscn's `Toggle`. Default `primary`.
   */
  variant?: ButtonVariant;
  /** `sm` is the compact control, `md` the control height, `lg` the large one (Next, Export, Cancel). */
  size?: "sm" | "md" | "lg";
  /** Fills the row (`flex: 1`), as the footer's Export and Cancel buttons do. */
  block?: boolean;
  /**
   * Disabled in the accessible sense: it reads as muted, ignores clicks and stays focusable so
   * it can explain itself (`title`). Never removed from the tab order.
   */
  disabled?: boolean;
  /** Toggle state for header buttons (Before): the active one takes the accent tint. */
  pressed?: boolean;
  /** A shortcut hint drawn as a keycap after the label ("⌘E"). */
  shortcut?: string;
  /** Small text at the right end, for a split button ("≈ 450 MB · ⌘↩"). Makes the label sit left. */
  aside?: ReactNode;
  /** Content before the label, such as an icon. */
  icon?: ReactNode;
  /** The `<button>` element, for a menu that returns focus to it. */
  ref?: Ref<HTMLButtonElement>;
  children?: ReactNode;
}

const KIND = { primary: "prominent", secondary: "glass", header: "glass", quiet: "ghost" } as const;

/**
 * The app's button, drawn by glasscn's `Button` (`src/components/glass/button.tsx`). A plain `<button type="button">`; `disabled`
 * uses `aria-disabled` so the button keeps its place in the tab order (the wizard's Next explains why it is locked). `title` is shown
 * as a glass tooltip (the span around the button lets a disabled one explain itself on hover too).
 */
export function Button({
  variant = "primary",
  size = "md",
  block = false,
  disabled = false,
  pressed,
  shortcut,
  aside,
  icon,
  className,
  onClick,
  title,
  children,
  ...rest
}: ButtonProps) {
  const handleClick = (event: MouseEvent<HTMLButtonElement>): void => {
    if (disabled) {
      event.preventDefault();
      return;
    }
    onClick?.(event);
  };
  const split = aside !== undefined;
  // The tooltip replaces the native `title`, which gave assistive technology the button's description even while closed; a hidden
  // span keeps that.
  const descriptionId = useId();
  const describedBy = [rest["aria-describedby"], title ? descriptionId : undefined].filter(Boolean).join(" ") || undefined;
  const toggle = pressed !== undefined;
  const shared = {
    ...rest,
    type: "button" as const,
    "data-variant": variant,
    "data-size": size,
    "data-block": block || undefined,
    "data-split": split || undefined,
    "aria-disabled": disabled || undefined,
    "aria-describedby": describedBy,
    onClick: handleClick,
  };
  const content = (
    <>
      {icon}
      {split ? (
        <>
          <span>{children}</span>
          {" "}
          <span className="text-xs font-semibold tabular-nums opacity-75">{aside}</span>
        </>
      ) : (
        children
      )}
      {/* The blank is not drawn (white space between flex items is dropped) but keeps the name readable: "Open files ⌘O". */}
      {shortcut ? (
        <>
          {" "}
          <KeyHint>{shortcut}</KeyHint>
        </>
      ) : null}
    </>
  );
  const sizeName = size === "sm" ? "sm" : size === "lg" ? "lg" : "default";
  const button = toggle ? (
    // A button with a pressed state is glasscn's Toggle: it draws the pressed style and reports aria-pressed. The state is the app's, so
    // Radix's own toggling is a no-op.
    // glasscn's Toggle only dims on `disabled`; a locked one here is `aria-disabled`, so it dims, and stops reacting to the pointer, like
    // the Button does. The span around it (below) still takes the hover, so the tooltip explains the lock.
    <Toggle
      {...shared}
      variant="outline"
      size={size === "lg" ? "lg" : "sm"}
      pressed={pressed}
      onPressedChange={() => undefined}
      className={cx("aria-disabled:pointer-events-none aria-disabled:opacity-45", block && "w-full", className)}
    >
      {content}
    </Toggle>
  ) : (
    <GlassButton
      {...shared}
      variant={KIND[variant]}
      size={variant === "header" && size === "md" ? "sm" : sizeName}
      className={cx(block && "w-full", split && "justify-between", className)}
    >
      {content}
    </GlassButton>
  );
  // The tree is the same whether or not there is a title: a button whose title comes and goes (the wizard's Next, the Export panel's
  // estimate) keeps its element, so the focus stays on it. Without a title the span is inert (no content, no description).
  return (
    <Tooltip>
      {/* The tooltip opens on hover only. On focus it would take the first Esc (Radix dismisses the topmost layer and calls preventDefault,
          so the dialog or the mode behind it would need a second press); the reason is still read from the description, and the screens
          that lock a button say why in their own note. Radix skips its focus handler when this one calls preventDefault. */}
      <TooltipTrigger asChild onFocus={(event) => event.preventDefault()}>
        <span className={cx("inline-flex", block && "flex-1")}>
          {button}
          {title ? (
            <span id={descriptionId} className="sr-only">
              {title}
            </span>
          ) : null}
        </span>
      </TooltipTrigger>
      {title ? <TooltipContent>{title}</TooltipContent> : null}
    </Tooltip>
  );
}

/** A keycap such as "⌘E", on glass. */
export function KeyHint({ children, className }: { children: ReactNode; className?: string }) {
  return <Kbd className={className}>{children}</Kbd>;
}

export interface LinkButtonProps extends Omit<ButtonHTMLAttributes<HTMLButtonElement>, "type" | "disabled"> {
  /** `md` is 14 px, `sm` 13 px. */
  size?: "sm" | "md";
  /** `heavy` (default) or `bold` ("Start over"). */
  weight?: "heavy" | "bold";
  disabled?: boolean;
  /** The `<button>` element, for a menu that returns focus to it. */
  ref?: Ref<HTMLButtonElement>;
  children?: ReactNode;
}

/** A text action in the accent: "Change", "Import →", "Choose…", "Start over", "← Back". glasscn's `link` button. */
export function LinkButton({ size = "md", weight = "heavy", disabled = false, className, onClick, children, ...rest }: LinkButtonProps) {
  return (
    <GlassButton
      {...rest}
      type="button"
      variant="link"
      size={size === "sm" ? "xs" : "sm"}
      className={className}
      data-size={size}
      data-weight={weight}
      aria-disabled={disabled || undefined}
      onClick={(event) => {
        if (disabled) event.preventDefault();
        else onClick?.(event);
      }}
    >
      {children}
    </GlassButton>
  );
}
