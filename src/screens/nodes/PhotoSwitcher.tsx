import { ChevronLeftIcon, ChevronRightIcon } from "lucide-react";
import type { MouseEvent } from "react";
import { Button as GlassButton } from "@/components/glass/button";

export interface PhotoSwitcherProps {
  /** The photo on screen, counted from 1; null when none is. */
  position: number | null;
  /** How many photos the session has. */
  count: number;
  onPrevious: () => void;
  onNext: () => void;
}

interface StepButtonProps {
  label: string;
  disabled: boolean;
  onClick: () => void;
  children: React.ReactNode;
}

/** An icon button of the switcher. Off at the ends of the folder, but kept in the tab order (`aria-disabled`), so the focus never drops. */
function StepButton({ label, disabled, onClick, children }: StepButtonProps) {
  return (
    <GlassButton
      type="button"
      variant="ghost"
      size="icon-sm"
      aria-label={label}
      aria-disabled={disabled || undefined}
      onClick={(event: MouseEvent<HTMLButtonElement>) => {
        if (disabled) event.preventDefault();
        else onClick();
      }}
    >
      {children}
    </GlassButton>
  );
}

/**
 * The photo switcher of the graph's toolbar, "‹ 2 of 7 ›" (`docs/23` §7.3): it takes the filmstrip's place in the node view. The two
 * arrows are buttons named "Previous photo" and "Next photo"; the count is a polite status, so a screen reader hears the photo it has
 * moved to. ← and → still switch photos from the keyboard (`docs/11` §Keyboard shortcuts), as they do with the filmstrip.
 */
export function PhotoSwitcher({ position, count, onPrevious, onNext }: PhotoSwitcherProps) {
  const here = position ?? 0;
  return (
    <div role="group" aria-label="Photos" className="ym-switcher">
      <StepButton label="Previous photo" disabled={here <= 1} onClick={onPrevious}>
        <ChevronLeftIcon aria-hidden="true" />
      </StepButton>
      <span className="ym-switcher__count" role="status">
        {here} of {count}
      </span>
      <StepButton label="Next photo" disabled={here === 0 || here >= count} onClick={onNext}>
        <ChevronRightIcon aria-hidden="true" />
      </StepButton>
    </div>
  );
}
