import { Item, ItemContent, ItemDescription, ItemTitle } from "@/components/glass/item";
import { RadioGroup, RadioGroupItem } from "@/components/glass/radio-group";
import { cx } from "./cx";

/** One choice of a `ChoiceList`. */
export interface ChoiceOption<T extends string> {
  value: T;
  /** The name, in bold ("Vintage"). */
  name: string;
  /** The line under the name, in the quieter label colour. Left out when there is none. */
  line?: string;
}

export interface ChoiceListProps<T extends string> {
  options: ReadonlyArray<ChoiceOption<T>>;
  value: T;
  onChange: (value: T) => void;
  /** Accessible name of the group ("Lens"). */
  label: string;
  /** Every row is off: nothing can be chosen (the lens data could not be loaded). */
  disabled?: boolean;
  className?: string;
}

/**
 * A short list of choices, one row each with its name and a line under it, for a choice that wants its words in view: glasscn's
 * `RadioGroup` of `Item` rows. It has radio semantics (`role="radiogroup"` with a `role="radio"` and `aria-checked` per row, one Tab stop,
 * the arrows move and choose, wrapping at the ends). A click anywhere on a row chooses it. The chosen row is ringed in the tab colour, as
 * the look in use is, and its circle takes the same colour; a row's accessible name is its name followed by its line.
 */
export function ChoiceList<T extends string>({ options, value, onChange, label, disabled = false, className }: ChoiceListProps<T>) {
  return (
    <RadioGroup aria-label={label} value={value} onValueChange={(next) => onChange(next as T)} disabled={disabled} className={cx("ym-choice-list", className)}>
      {options.map((option) => (
        <Item
          key={option.value}
          asChild
          size="sm"
          variant="outline"
          className="ym-choice-row cursor-pointer items-start hover:bg-fill has-data-[state=checked]:inset-ring-2 has-data-[state=checked]:inset-ring-primary has-disabled:cursor-not-allowed has-disabled:opacity-50"
        >
          <label>
            <RadioGroupItem value={option.value} className="ym-choice-row__radio" />
            <ItemContent className="ym-choice-row__text">
              <ItemTitle className="ym-choice-row__name">{option.name}</ItemTitle>
              {option.line ? <ItemDescription className="ym-choice-row__line">{option.line}</ItemDescription> : null}
            </ItemContent>
          </label>
        </Item>
      ))}
    </RadioGroup>
  );
}
