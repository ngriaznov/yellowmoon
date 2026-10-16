import * as React from "react"
import { cn } from "cn"
import { Label as LabelPrimitive } from "radix-ui"

/**
 * An accessible label. `required` adds a marker (hidden from assistive tech — mark the control
 * `required` too); `optional` adds a quiet "(optional)" hint.
 */
function Label({
  className,
  required,
  optional,
  children,
  ...props
}: React.ComponentProps<typeof LabelPrimitive.Root> & { required?: boolean; optional?: boolean | string }) {
  // markers only when asked: with asChild, Radix needs a single child
  const content =
    required || optional ? (
      <>
        {children}
        {required ? (
          <span aria-hidden data-slot="label-required" className="-ms-1.5 text-destructive-ink">
            *
          </span>
        ) : null}
        {optional ? (
          <span data-slot="label-optional" className="font-normal text-muted-foreground">
            {typeof optional === "string" ? optional : "(optional)"}
          </span>
        ) : null}
      </>
    ) : (
      children
    )
  return (
    <LabelPrimitive.Root
      data-slot="label"
      className={cn(
        "flex items-center gap-2 text-sm leading-snug font-medium select-none group-data-[disabled=true]:pointer-events-none group-data-[disabled=true]:text-glass-label-tertiary peer-disabled:cursor-not-allowed peer-disabled:text-glass-label-tertiary",
        className
      )}
      {...props}
    >
      {content}
    </LabelPrimitive.Root>
  )
}

export { Label }
