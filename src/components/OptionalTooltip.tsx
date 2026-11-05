import { useState } from "react";
import type { ComponentProps, ReactNode } from "react";
import { Tooltip, TooltipContent } from "@/components/glass/tooltip";

/**
 * A tooltip whose text comes and goes: pass the `TooltipTrigger` as the child and the text (or `null` for none).
 * Radix closes a tooltip on pointer leave only through its mounted content, so a trigger without content keeps an "open" state that nothing
 * can close; the content then appears by itself when the text arrives, with the pointer elsewhere. So the open state is kept only while there is text.
 */
export function OptionalTooltip({
  text,
  side,
  children,
}: {
  text: ReactNode;
  side?: ComponentProps<typeof TooltipContent>["side"];
  children: ReactNode;
}) {
  const [open, setOpen] = useState(false);
  const has = text !== null && text !== undefined && text !== false && text !== "";
  return (
    <Tooltip open={has && open} onOpenChange={(next) => setOpen(has && next)}>
      {children}
      {has ? <TooltipContent side={side}>{text}</TooltipContent> : null}
    </Tooltip>
  );
}
