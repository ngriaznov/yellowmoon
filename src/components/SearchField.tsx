import { SearchField as GlassSearchField } from "@/components/glass/search-field";
import { cx } from "./cx";

export interface SearchFieldProps {
  value: string;
  onChange: (value: string) => void;
  /** Accessible name ("Search looks"). */
  label: string;
  /** Default: the label with an ellipsis ("Search looks…"). */
  placeholder?: string;
  className?: string;
}

/** A search field: glasscn's capsule with a magnifier and a Clear button while there is text. A `type="search"` input named by `label`. */
export function SearchField({ value, onChange, label, placeholder, className }: SearchFieldProps) {
  return (
    <GlassSearchField
      className={cx("ym-search", className)}
      value={value}
      onValueChange={onChange}
      aria-label={label}
      placeholder={placeholder ?? `${label}…`}
      showCancel="never"
    />
  );
}
