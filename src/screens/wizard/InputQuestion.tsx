import { useId } from "react";
import type { LookInput } from "../../api/types";
import { Segmented } from "../../components";
import { INPUT_CHOICES } from "../../wizard/model";

export interface InputQuestionProps {
  /** The file the question is about, for the accessible name. */
  name: string;
  value: LookInput | null;
  onChange: (input: LookInput) => void;
  /** What the names suggest, when they do ("LogC4"). */
  suggestion?: LookInput | null;
}

/**
 * The generic `.cube` question (`docs/08` §Any other .cube): what does this LUT expect as input? Three answers, with a line
 * about the one picked. Used by the wizard's Recognise step and by the Import .cube dialog.
 */
export function InputQuestion({ name, value, onChange, suggestion }: InputQuestionProps) {
  const id = useId();
  const picked = INPUT_CHOICES.find((c) => c.value === value);
  const hint = INPUT_CHOICES.find((c) => c.value === suggestion);
  return (
    <div className="ym-wizard__question">
      <p id={id} className="ym-wizard__question-text">
        What does this LUT expect as input?
      </p>
      <Segmented<LookInput>
        label={`Input of ${name}`}
        size="sm"
        columns={3}
        value={(value ?? "") as LookInput}
        onChange={onChange}
        options={INPUT_CHOICES.map((c) => ({ value: c.value, label: c.label }))}
      />
      <p className="ym-wizard__question-detail" aria-live="polite">
        {picked ? picked.detail : hint ? `The names suggest ${hint.label}.` : "Pick one. If you are not sure, LogC4 is what most new LUTs expect."}
      </p>
    </div>
  );
}
