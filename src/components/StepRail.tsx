import { CheckIcon } from "lucide-react";
import { Item, ItemContent, ItemDescription, ItemMedia, ItemTitle } from "@/components/glass/item";
import { cx } from "./cx";

/** One step of the wizard rail. */
export interface StepRailStep {
  /** "Choose", "Get the files", "Recognise", "Import", "Finish". */
  title: string;
  /** One line: "Pick the ARRI packages you want." */
  text: string;
}

export interface StepRailProps {
  steps: readonly StepRailStep[];
  /** Index of the current step (0-based). Earlier steps are done, later ones are locked. */
  current: number;
  /** A finished step was pressed: go back to it. Locked and current steps do nothing. */
  onSelect?: (index: number) => void;
  /** Accessible name. Default "Steps". */
  label?: string;
  className?: string;
}

/**
 * The wizard's left rail (`docs/11-ui.md` §Import ARRI looks): one glasscn `Item` per step with its number, a title and a line of
 * text. The current step takes the accent with `aria-current="step"`, finished steps show a tick and can be revisited, future
 * steps are locked (`aria-disabled`) and slightly faded.
 */
export function StepRail({ steps, current, onSelect, label = "Steps", className }: StepRailProps) {
  return (
    <nav aria-label={label} className={cx("ym-step-rail", className)}>
      <ol className="ym-step-rail__list">
        {steps.map((step, index) => {
          const state = index < current ? "done" : index === current ? "current" : "locked";
          return (
            <li key={step.title}>
              <Item asChild size="sm">
                <button
                  type="button"
                  className="ym-step-rail__step"
                  data-state={state}
                  aria-current={state === "current" ? "step" : undefined}
                  aria-disabled={state === "locked" ? true : undefined}
                  aria-label={`${step.title}${state === "done" ? ", done" : ""}. ${step.text}`}
                  onClick={() => {
                    if (state === "done") onSelect?.(index);
                  }}
                >
                  <ItemMedia variant="icon" className="ym-step-rail__mark" aria-hidden="true">
                    {state === "done" ? <CheckIcon /> : index + 1}
                  </ItemMedia>
                  <ItemContent className="ym-step-rail__body">
                    <ItemTitle className="ym-step-rail__title">{step.title}</ItemTitle>
                    <ItemDescription className="ym-step-rail__text">{step.text}</ItemDescription>
                  </ItemContent>
                </button>
              </Item>
            </li>
          );
        })}
      </ol>
    </nav>
  );
}
