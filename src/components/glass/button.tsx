import * as React from "react"
import { cva, type VariantProps } from "class-variance-authority"
import { cn } from "cn"
import { Slot } from "radix-ui"

import { Spinner as ActivitySpinner } from "@/components/glass/spinner"

/**
 * shadcn's Button with glass variants. Capsule by default (`shape="rounded"` for
 * shadcn's corners), so icon sizes are circles. Solid variants squash elastically when
 * pressed; liquid ones grow and light up under the finger.
 *
 * `variant="glass"` is liquid glass — the control layer, for buttons floating over
 * content — with a soft shadow; `variant="prominent"` is the same glass tinted with the
 * accent (one per bar). `default` stays solid for shadcn compatibility. Disabled, liquid
 * keeps its material and dims the label; the other variants fade as a whole.
 * `aria-disabled` disables too, so an `asChild` link can be disabled.
 */
const dim = "disabled:opacity-45 aria-disabled:opacity-45"
const buttonVariants = cva(
  "group/button relative inline-flex shrink-0 items-center justify-center font-medium whitespace-nowrap transition-[scale,background-color,box-shadow,opacity,color,filter] duration-[calc(var(--glass-duration)*1.5)] ease-glass glass-focus select-none aria-busy:pointer-events-none disabled:pointer-events-none aria-disabled:pointer-events-none aria-invalid:ring-(length:--glass-ring-width) aria-invalid:ring-destructive/30 [&_svg]:pointer-events-none [&_svg]:shrink-0 [&_svg:not([class*='size-'])]:size-4",
  {
    variants: {
      variant: {
        default: `glass-press bg-primary text-primary-foreground shadow-glass-accent hover:brightness-105 ${dim}`,
        // liquid: the control layer. Hover takes it to the content material's frost; the elevation is shadow-glass-1, computed here from primitives
        glass:
          "glass-liquid glass-illuminate glass-press-liquid [--glass-elevation:0_calc(var(--glass-shadow-y)*0.3)_calc(var(--glass-shadow-blur)*0.3)_color-mix(in_oklch,var(--glass-shadow-color)_var(--glass-shadow-opacity),transparent)] text-foreground hover:[--glass-liquid-opacity:var(--glass-opacity)] disabled:text-glass-label-tertiary aria-disabled:text-glass-label-tertiary",
        // tinted liquid glass, the accent's label colour on it; illuminates like glass
        prominent:
          "glass-prominent glass-press-liquid [--glass-elevation:0_calc(var(--glass-shadow-y)*0.4)_calc(var(--glass-shadow-blur)*0.4)_color-mix(in_oklch,var(--primary)_30%,transparent)] hover:brightness-105 disabled:opacity-45 aria-disabled:opacity-45",
        tinted: `glass-press bg-primary/16 text-primary-ink hover:bg-primary/24 dark:bg-primary/22 ${dim}`,
        secondary: `glass-press bg-fill text-foreground hover:bg-fill-strong ${dim}`,
        outline: `glass-press border border-glass-border bg-transparent text-foreground hover:bg-fill ${dim}`,
        ghost: `glass-press text-foreground hover:bg-fill ${dim}`,
        destructive: `glass-press bg-destructive/14 text-destructive-ink hover:bg-destructive/22 ${dim}`,
        link: `text-primary-ink underline-offset-4 hover:underline ${dim}`,
      },
      size: {
        default: "h-control gap-2 px-pad text-sm has-data-[icon=inline-end]:pe-4 has-data-[icon=inline-start]:ps-4",
        xs: "h-control-xs gap-1 px-pad-xs hit-area text-xs [&_svg:not([class*='size-'])]:size-3",
        sm: "h-control-sm gap-1.5 px-pad-sm hit-area text-[0.8rem] [&_svg:not([class*='size-'])]:size-3.5",
        lg: "h-control-lg gap-2 px-pad-lg text-base [&_svg:not([class*='size-'])]:size-5",
        icon: "size-control",
        "icon-xs": "size-control-xs hit-area [&_svg:not([class*='size-'])]:size-3",
        "icon-sm": "size-control-sm hit-area [&_svg:not([class*='size-'])]:size-3.5",
        "icon-lg": "size-control-lg [&_svg:not([class*='size-'])]:size-5",
      },
      shape: {
        pill: "rounded-button",
        rounded: "rounded-control",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
      shape: "pill",
    },
  }
)

/** The activity indicator a loading button shows; the button's aria-busy announces it. */
const Spinner = (props: React.ComponentProps<"svg">) => <ActivitySpinner data-slot="button-spinner" aria-hidden className="motion-reduce:animate-spin" {...props} />

/**
 * While `loading` the button is busy (`aria-busy`, clicks ignored) and keeps its width: a
 * leading icon is swapped for the spinner, otherwise the spinner sits over the hidden label.
 * `loadingText` replaces the label instead (the width may then change).
 */
function withLoading(children: React.ReactNode, size: string | null | undefined, loadingText: React.ReactNode) {
  if (loadingText !== undefined)
    return (
      <>
        <Spinner data-icon="inline-start" />
        {loadingText}
      </>
    )
  const items = React.Children.toArray(children)
  const [first, ...rest] = items
  if (!size?.startsWith("icon") && React.isValidElement(first) && items.length > 1)
    return (
      <>
        <Spinner data-icon="inline-start" />
        {rest}
      </>
    )
  return (
    <>
      <span className="inline-flex items-center gap-[inherit] opacity-0">{children}</span>
      <span className="absolute inset-0 flex items-center justify-center">
        <Spinner />
      </span>
    </>
  )
}

function Button({
  className,
  variant = "default",
  size = "default",
  shape = "pill",
  asChild = false,
  loading = false,
  loadingText,
  children,
  onClick,
  ...props
}: React.ComponentProps<"button"> &
  VariantProps<typeof buttonVariants> & {
    asChild?: boolean
    /** Shows a spinner and marks the button busy; clicks are ignored. */
    loading?: boolean
    /** A label to show while loading (e.g. "Saving…"). */
    loadingText?: React.ReactNode
  }) {
  const Comp = asChild ? Slot.Root : "button"
  const busy = loading && !asChild
  return (
    <Comp
      data-slot="button"
      data-variant={variant}
      data-size={size}
      aria-busy={busy || undefined}
      className={cn(buttonVariants({ variant, size, shape, className }))}
      onClick={busy ? (e: React.MouseEvent<HTMLButtonElement>) => e.preventDefault() : onClick}
      {...props}
    >
      {busy ? withLoading(children, size, loadingText) : children}
    </Comp>
  )
}

/** A button that opens a menu: the label and a trailing chevron. Wrap it in DropdownMenuTrigger asChild. */
function PullDownButton({ children, ...props }: React.ComponentProps<typeof Button>) {
  return (
    <Button {...props}>
      {children}
      <svg data-icon="inline-end" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.25" strokeLinecap="round" strokeLinejoin="round" aria-hidden className="opacity-70">
        <path d="m6 9 6 6 6-6" />
      </svg>
    </Button>
  )
}

/**
 * A primary action joined to a menu of alternatives: two buttons that read as one (shared outer
 * corners, a hairline between). Children: the action Button, then a PullDownButton (or icon Button).
 */
function SplitButton({ className, ...props }: React.ComponentProps<"div">) {
  return (
    <div
      role="group"
      data-slot="split-button"
      className={cn(
        "inline-flex w-fit items-stretch *:data-[slot=button]:focus-visible:z-10 [&>[data-slot=button]:not(:first-child)]:rounded-s-none [&>[data-slot=button]:not(:first-child)]:border-s [&>[data-slot=button]:not(:first-child)]:border-separator [&>[data-slot=button]:not(:first-child)]:ps-2.5 [&>[data-slot=button]:not(:last-child)]:rounded-e-none",
        className
      )}
      {...props}
    />
  )
}

export { Button, buttonVariants, PullDownButton, SplitButton }
