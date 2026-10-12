import { Button, LinkButton, MoonMark, StepRail } from "../components";
import {
  canAdvance,
  footNote,
  heading,
  lead,
  NEXT_LABELS,
  STEP_CHOOSE,
  STEP_FINISH,
  STEP_GET,
  STEP_IMPORT,
  STEP_RECOGNISE,
  STEPS,
} from "../wizard/model";
import { useStore } from "../state/context";
import { ChooseStep, FinishStep, GetStep, ImportStep, RecogniseStep } from "./wizard/steps";
import { useWizard } from "./wizard/useWizard";

/**
 * The ARRI look wizard (`design/mockups/import-looks.html`, `docs/08-looks.md` §Wizard): a full-window detour with a rail of
 * five steps, the step's content, and a footer with Back, a line of context and Next. Later steps stay locked until their
 * turn; the rules are in `wizard/model.ts`. Nothing is copied before Import.
 */
export function ImportLooks() {
  const store = useStore();
  const wizard = useWizard(store);
  const { state } = wizard;
  const importing = state.import.status === "running";
  const last = state.step === STEP_FINISH;
  const nextOk = canAdvance(state);

  return (
    <div className="ym-screen ym-wizard" data-screen="import-looks" data-step={state.step}>
      <header className="ym-header ym-wizard__header">
        <div className="ym-brand">
          <span className="ym-brand-title">
            <MoonMark />
            ARRI looks
          </span>
          <span className="ym-brand-sub">Import · step {state.step + 1} of 5</span>
        </div>
        <div className="ym-wizard__header-text">Free from ARRI. Imported to this Mac only, never shared or uploaded.</div>
        <button
          type="button"
          className="ym-wizard__cancel"
          aria-disabled={importing || undefined}
          title={importing ? "Importing. This takes a few seconds." : undefined}
          onClick={wizard.cancel}
        >
          Cancel
        </button>
      </header>

      <StepRail className="ym-wizard__rail" steps={STEPS} current={state.step} onSelect={wizard.select} />

      <main className="ym-wizard__main">
        <h1 aria-live="polite">{heading(state)}</h1>
        <p className="ym-wizard__lead">{lead(state)}</p>
        {state.step === STEP_CHOOSE ? <ChooseStep wizard={wizard} /> : null}
        {state.step === STEP_GET ? <GetStep wizard={wizard} /> : null}
        {state.step === STEP_RECOGNISE ? <RecogniseStep wizard={wizard} /> : null}
        {state.step === STEP_IMPORT ? <ImportStep wizard={wizard} /> : null}
        {state.step === STEP_FINISH ? <FinishStep wizard={wizard} /> : null}
      </main>

      <footer className="ym-wizard__footer">
        <LinkButton className="ym-wizard__back" disabled={state.step === STEP_CHOOSE || importing} onClick={wizard.back}>
          ← Back
        </LinkButton>
        <span className="ym-wizard__foot-note" role="status">
          {footNote(state)}
        </span>
        <Button
          size="lg"
          disabled={!nextOk}
          title={nextOk ? undefined : footNote(state)}
          onClick={wizard.advance}
        >
          {last ? "Done →" : NEXT_LABELS[state.step]}
        </Button>
      </footer>
    </div>
  );
}
