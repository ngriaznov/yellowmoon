import { cx } from "./cx";
import type { TabId } from "./tabs";

export interface SwitchProps {
  checked: boolean;
  onChange: (checked: boolean) => void;
  /** Accessible name: "Neural demosaic". The switch has no visible text of its own. */
  label: string;
  disabled?: boolean;
  /** Tab whose colour an "on" switch takes. Inherited from `data-tab` above when omitted. */
  tab?: TabId;
  id?: string;
  className?: string;
}

/**
 * The 44 × 24 square switch: on is the tab colour with a white 16 px knob on the right, off is
 * paper with an ink knob on the left. `role="switch"`; Space and Return toggle it.
 */
export function Switch({ checked, onChange, label, disabled = false, tab, id, className }: SwitchProps) {
  return (
    <button
      id={id}
      type="button"
      role="switch"
      aria-checked={checked}
      aria-label={label}
      aria-disabled={disabled || undefined}
      className={cx("ym-switch", className)}
      data-tab={tab}
      onClick={() => {
        if (!disabled) onChange(!checked);
      }}
    >
      <span className="ym-switch__knob" aria-hidden="true" />
    </button>
  );
}
