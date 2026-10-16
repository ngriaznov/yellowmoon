import { Switch as GlassSwitch } from "@/components/glass/switch";
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
 * glasscn's switch: a capsule that fills with the accent, a knob that springs across and stretches while held. `role="switch"`;
 * Space toggles it.
 */
export function Switch({ checked, onChange, label, disabled = false, tab, id, className }: SwitchProps) {
  return (
    <GlassSwitch
      id={id}
      aria-label={label}
      checked={checked}
      disabled={disabled}
      data-tab={tab}
      className={className}
      onCheckedChange={(next) => onChange(next)}
    />
  );
}
