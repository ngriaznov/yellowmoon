import { Switch as GlassSwitch } from "@/components/glass/switch";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/glass/tooltip";
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
  /** `sm` is a smaller capsule for a row of a list. */
  size?: "sm" | "default";
  /** The tooltip, on hover only (as a Button's). */
  title?: string;
}

/**
 * glasscn's switch: a capsule that fills with the accent, a knob that springs across and stretches while held. `role="switch"`;
 * Space toggles it.
 */
export function Switch({ checked, onChange, label, disabled = false, tab, id, className, size = "default", title }: SwitchProps) {
  const control = (
    <GlassSwitch
      id={id}
      aria-label={label}
      checked={checked}
      disabled={disabled}
      data-tab={tab}
      size={size}
      className={className}
      onCheckedChange={(next) => onChange(next)}
    />
  );
  if (!title) return control;
  return (
    <Tooltip>
      {/* On hover only, as a Button's: on focus the tooltip would take the first Esc. The trigger is a span around the switch: on the switch itself it would
          write its own `data-state` ("closed") over the switch's ("checked"), and the capsule would lose its fill. */}
      <TooltipTrigger asChild onFocus={(event) => event.preventDefault()}>
        <span className="ym-switch-tip">{control}</span>
      </TooltipTrigger>
      <TooltipContent>{title}</TooltipContent>
    </Tooltip>
  );
}
