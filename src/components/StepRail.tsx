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
 * The wizard's left rail (`docs/11-ui.md` §Import ARRI looks): one block per step with its number
 * in 36 px, a title and a line of text. The current step is brand yellow with
 * `aria-current="step"`, finished steps show a tick and can be revisited, future steps are locked
 * (`aria-disabled`) and slightly faded.
 */
export function StepRail({ steps, current, onSelect, label = "Steps", className }: StepRailProps) {
  return (
    <nav aria-label={label} className={cx("ym-step-rail", className)}>
      <ol className="ym-step-rail__list">
        {steps.map((step, index) => {
          const state = index < current ? "done" : index === current ? "current" : "locked";
          return (
            <li key={step.title}>
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
                <span className="ym-step-rail__mark" aria-hidden="true">
                  {state === "done" ? (
                    <svg viewBox="0 0 36 36" fill="none" stroke="currentColor" strokeWidth="5.6">
                      <path d="M5.4 17.7L12.7 24.8L28.4 9.4" />
                    </svg>
                  ) : (
                    index + 1
                  )}
                </span>
                <span className="ym-step-rail__body">
                  <span className="ym-step-rail__title">{step.title}</span>
                  <span className="ym-step-rail__text">{step.text}</span>
                </span>
              </button>
            </li>
          );
        })}
      </ol>
    </nav>
  );
}
