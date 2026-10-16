import * as React from "react"
import { cva, type VariantProps } from "class-variance-authority"
import { cn } from "cn"
import { Slot } from "radix-ui"

import { Separator } from "@/components/glass/separator"

/**
 * shadcn's ButtonGroup: buttons joined into one control — shared outer corners, a hairline
 * between. `variant="glass"` is Apple's toolbar grouping: the group floats on one liquid capsule
 * and the buttons inside go clear (use it over content, not on a pane — never glass on glass).
 */
const buttonGroupVariants = cva(
  "flex w-fit items-stretch has-[>[data-slot=button-group]]:gap-2 *:focus-visible:relative *:focus-visible:z-10 [&>[data-slot=select-trigger]:not([class*='w-'])]:w-fit [&>input]:flex-1",
  {
    variants: {
      orientation: {
        horizontal:
          "[&>*:not(:first-child)]:rounded-s-none [&>*:not(:first-child)]:border-s-0 [&>*:not(:last-child)]:rounded-e-none [&>[data-slot]:not(:first-child)]:-ms-px [&>[data-slot=button-group-separator]]:ms-0",
        vertical:
          "flex-col [&>*:not(:first-child)]:rounded-t-none [&>*:not(:first-child)]:border-t-0 [&>*:not(:last-child)]:rounded-b-none [&>[data-slot]:not(:first-child)]:-mt-px [&>[data-slot=button-group-separator]]:mt-0",
      },
      variant: {
        default: "",
        glass:
          "glass-liquid rounded-button p-1 [--glass-elevation:0_calc(var(--glass-shadow-y)*0.3)_calc(var(--glass-shadow-blur)*0.3)_color-mix(in_oklch,var(--glass-shadow-color)_var(--glass-shadow-opacity),transparent)] *:data-[slot=button]:rounded-button! *:data-[slot=button]:bg-transparent *:data-[slot=button]:shadow-none *:data-[slot=button]:before:hidden *:data-[slot=button]:hover:bg-fill",
      },
    },
    defaultVariants: { orientation: "horizontal", variant: "default" },
  }
)

function ButtonGroup({ className, orientation, variant, ...props }: React.ComponentProps<"div"> & VariantProps<typeof buttonGroupVariants>) {
  return (
    <div
      role="group"
      data-slot="button-group"
      data-orientation={orientation ?? "horizontal"}
      data-variant={variant ?? "default"}
      aria-orientation={orientation === "vertical" ? "vertical" : undefined}
      className={cn(buttonGroupVariants({ orientation, variant }), className)}
      {...props}
    />
  )
}

/** A label or value inside the group, on the same fill as a secondary button. */
function ButtonGroupText({ className, asChild = false, ...props }: React.ComponentProps<"div"> & { asChild?: boolean }) {
  const Comp = asChild ? Slot.Root : "div"
  return (
    <Comp
      data-slot="button-group-text"
      className={cn("flex items-center gap-2 rounded-control bg-fill px-pad-sm text-sm font-medium [&_svg]:pointer-events-none [&_svg:not([class*='size-'])]:size-4", className)}
      {...props}
    />
  )
}

function ButtonGroupSeparator({ className, orientation = "vertical", ...props }: React.ComponentProps<typeof Separator>) {
  return <Separator data-slot="button-group-separator" orientation={orientation} className={cn("relative m-0! self-stretch data-vertical:h-auto", className)} {...props} />
}

export { ButtonGroup, ButtonGroupSeparator, ButtonGroupText, buttonGroupVariants }
