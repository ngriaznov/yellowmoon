import type { ChangeEvent } from "react";
import { cx } from "./cx";

export interface SearchFieldProps {
  value: string;
  onChange: (value: string) => void;
  /** Accessible name; also the default placeholder without the ellipsis. */
  label: string;
  placeholder?: string;
  className?: string;
}

/** The 44 px search box of the Look tab: a 3 px border and a magnifier at the right. */
export function SearchField({ value, onChange, label, placeholder, className }: SearchFieldProps) {
  return (
    <div className={cx("ym-search", className)}>
      <input
        type="search"
        className="ym-search__input"
        value={value}
        placeholder={placeholder ?? `${label}…`}
        aria-label={label}
        onChange={(event: ChangeEvent<HTMLInputElement>) => onChange(event.target.value)}
      />
      <svg
        className="ym-search__icon"
        width="20"
        height="20"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2.6"
        strokeLinecap="round"
        aria-hidden="true"
      >
        <circle cx="10.5" cy="10.5" r="6.5" />
        <path d="M15.5 15.5L21 21" />
      </svg>
    </div>
  );
}
