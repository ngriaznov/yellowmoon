import * as React from "react"
import { cva, type VariantProps } from "class-variance-authority"
import { cn } from "cn"
import { Toggle as TogglePrimitive } from "radix-ui"

/**
 * shadcn's Toggle on glass: a two-state button (iOS's toggle button outside lists). Off it's
 * clear (or a hairline with `outline`); on it takes the accent's tint.
 */
const toggleVariants = cva(
  "inline-flex shrink-0 items-center justify-center gap-2 rounded-control text-sm font-medium whitespace-nowrap text-foreground transition-[background-color,color,box-shadow,scale] duration-(--glass-duration) ease-glass glass-focus glass-press hover:bg-fill disabled:pointer-events-none disabled:opacity-45 aria-invalid:inset-ring-2 aria-invalid:inset-ring-destructive/70 data-[state=on]:bg-primary/16 data-[state=on]:text-primary-ink dark:data-[state=on]:bg-primary/22 [&_svg]:pointer-events-none [&_svg]:shrink-0 [&_svg:not([class*='size-'])]:size-4",
  {
    variants: {
      variant: {
        default: "bg-transparent",
        outline: "bg-transparent inset-ring inset-ring-glass-border",
      },
      size: {
        default: "h-control min-w-control px-pad-xs",
        sm: "h-control-sm min-w-control-sm px-2 hit-area",
        lg: "h-control-lg min-w-control-lg px-pad-sm",
      },
    },
    defaultVariants: { variant: "default", size: "default" },
  }
)

function Toggle({ className, variant, size, ...props }: React.ComponentProps<typeof TogglePrimitive.Root> & VariantProps<typeof toggleVariants>) {
  return <TogglePrimitive.Root data-slot="toggle" className={cn(toggleVariants({ variant, size, className }))} {...props} />
}

export { Toggle, toggleVariants }
