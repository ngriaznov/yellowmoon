import type { ButtonHTMLAttributes, MouseEvent, ReactNode, Ref } from "react";
import { cx } from "./cx";

/** Visual kind of a `Button`. */
export type ButtonVariant = "primary" | "secondary" | "header";

export interface ButtonProps extends Omit<ButtonHTMLAttributes<HTMLButtonElement>, "type" | "disabled"> {
  /**
   * `primary`: ink fill (Open photos, Next, Export N photos). `secondary`: outlined (Open
   * folder, Cancel, Show in Finder). `header`: borderless caps text in a header cell (Before,
   * Crop, Pick white). Default `primary`.
   */
  variant?: ButtonVariant;
  /** `md` is 48 px, `lg` is 56 px (Next, Export, Cancel). Header buttons are always 36 px. */
  size?: "md" | "lg";
  /** Fills the row (`flex: 1`), as the footer's Export and Cancel buttons do. */
  block?: boolean;
  /**
   * Disabled in the accessible sense: it reads as muted, ignores clicks and stays focusable so
   * it can explain itself (`title`). Never removed from the tab order.
   */
  disabled?: boolean;
  /** Toggle state for header buttons (Before): the active one is brand yellow. */
  pressed?: boolean;
  /** A shortcut hint drawn as a boxed chip after the label ("⌘E"). */
  shortcut?: string;
  /** Small text at the right end, for a split button ("≈ 450 MB · ⌘↩"). Makes the label sit left. */
  aside?: ReactNode;
  /** Content before the label, such as an icon. */
  icon?: ReactNode;
  /** The `<button>` element, for a menu that returns focus to it. */
  ref?: Ref<HTMLButtonElement>;
  children?: ReactNode;
}

/**
 * The buttons of the mockups: caps 800 text, square corners, no shadow. A plain
 * `<button type="button">`; `disabled` uses `aria-disabled` so the button keeps its place in
 * the tab order (the wizard's Next explains why it is locked).
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
  return (
    <button
      {...rest}
      type="button"
      className={cx("ym-btn", className)}
      data-variant={variant}
      data-size={size}
      data-block={block || undefined}
      data-split={split || undefined}
      aria-disabled={disabled || undefined}
      aria-pressed={pressed}
      onClick={handleClick}
    >
      {icon}
      {split ? (
        <>
          <span>{children}</span>
          <span className="ym-btn__aside">{aside}</span>
        </>
      ) : (
        children
      )}
      {shortcut ? (
        <>
          {" "}
          <KeyHint>{shortcut}</KeyHint>
        </>
      ) : null}
    </button>
  );
}

/** A boxed key hint such as "⌘E", drawn in the current text colour. */
export function KeyHint({ children, className }: { children: ReactNode; className?: string }) {
  return <span className={cx("ym-key-hint", className)}>{children}</span>;
}

export interface LinkButtonProps extends Omit<ButtonHTMLAttributes<HTMLButtonElement>, "type" | "disabled"> {
  /** `md` is 14 px, `sm` 13 px. */
  size?: "sm" | "md";
  /** `heavy` (800, default) or `bold` (700, "Start over"). */
  weight?: "heavy" | "bold";
  disabled?: boolean;
  children?: ReactNode;
}

/** An underlined text action: "Change", "Import →", "Choose…", "Start over", "← Back". */
export function LinkButton({
  size = "md",
  weight = "heavy",
  disabled = false,
  className,
  onClick,
  children,
  ...rest
}: LinkButtonProps) {
  return (
    <button
      {...rest}
      type="button"
      className={cx("ym-link", className)}
      data-size={size}
      data-weight={weight}
      aria-disabled={disabled || undefined}
      onClick={(event) => {
        if (disabled) event.preventDefault();
        else onClick?.(event);
      }}
    >
      {children}
    </button>
  );
}
