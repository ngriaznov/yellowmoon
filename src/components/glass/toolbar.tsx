import * as React from "react"
import { cn } from "cn"
import { Slot } from "radix-ui"
import { EllipsisIcon } from "lucide-react"

import { GlassGroup } from "@/components/glass/glass-group"
import { DropdownMenu, DropdownMenuContent, DropdownMenuTrigger } from "@/components/glass/dropdown-menu"
import { useScrollMinimized } from "@/components/glass/scroll-edge"
import { useGlassPointer } from "@/hooks/use-glass-pointer"

/**
 * A floating toolbar on the control layer: groups of buttons that share one liquid
 * capsule, spacers between them, a title, an overflow menu and a standalone
 * prominent action — Apple's `.toolbar` for the web.
 *
 *   <Toolbar placement="top" aria-label="Entry">
 *     <ToolbarGroup>
 *       <ToolbarItem aria-label="Back"><ChevronLeftIcon /></ToolbarItem>
 *     </ToolbarGroup>
 *     <ToolbarSpacer />
 *     <ToolbarTitle>Tonight</ToolbarTitle>
 *     <ToolbarSpacer />
 *     <ToolbarGroup>
 *       <ToolbarItem aria-label="Share"><ShareIcon /></ToolbarItem>
 *       <ToolbarOverflow>…DropdownMenuItems…</ToolbarOverflow>
 *     </ToolbarGroup>
 *     <ToolbarItem prominent aria-label="Done"><CheckIcon /></ToolbarItem>
 *   </Toolbar>
 *
 * Fixed at the top or bottom, it sets `--toolbar-inset-top` / `--toolbar-inset-bottom`
 * on <html> so content can pad itself clear. Arrow keys move between items.
 */

const focusable = 'button:not([disabled]), a[href], input:not([disabled]), [tabindex]:not([tabindex="-1"])'

/** Arrow keys / Home / End move focus between the toolbar's controls (mirrored in RTL). */
function onToolbarKeyDown(e: React.KeyboardEvent<HTMLElement>) {
  if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(e.key)) return
  if (e.target instanceof HTMLInputElement) return
  const items = Array.from(e.currentTarget.querySelectorAll<HTMLElement>(focusable))
  const i = items.indexOf(document.activeElement as HTMLElement)
  if (i < 0) return
  const rtl = getComputedStyle(e.currentTarget).direction === "rtl"
  const step = e.key === "ArrowRight" ? (rtl ? -1 : 1) : e.key === "ArrowLeft" ? (rtl ? 1 : -1) : 0
  const next = e.key === "Home" ? 0 : e.key === "End" ? items.length - 1 : (i + step + items.length) % items.length
  items[next]?.focus()
  e.preventDefault()
}

function useToolbarInset(ref: React.RefObject<HTMLElement | null>, edge: "top" | "bottom" | null) {
  React.useLayoutEffect(() => {
    const el = ref.current
    if (!el || !edge) return
    const root = document.documentElement
    const name = `--toolbar-inset-${edge}`
    let value = ""
    const update = () => {
      value = `calc(${el.offsetHeight}px + max(0.75rem, env(safe-area-inset-${edge})))`
      root.style.setProperty(name, value)
    }
    update()
    const observer = new ResizeObserver(update)
    observer.observe(el)
    return () => {
      observer.disconnect()
      if (root.style.getPropertyValue(name) === value) root.style.removeProperty(name)
    }
  }, [ref, edge])
}

type ToolbarProps = React.ComponentProps<"div"> & {
  /** `top` / `bottom` float fixed over the page; `sticky-top` / `sticky-bottom` stick inside a scroll container; `static` sits in flow. */
  placement?: "top" | "bottom" | "sticky-top" | "sticky-bottom" | "static"
  /** `onScrollDown` shrinks the toolbar while the content scrolls down; it returns on scroll up, hover or focus. */
  minimize?: "never" | "onScrollDown"
}

function Toolbar({ className, placement = "top", minimize = "never", onKeyDown, ref: forwardedRef, ...props }: ToolbarProps) {
  const ref = React.useRef<HTMLDivElement>(null)
  React.useImperativeHandle(forwardedRef, () => ref.current as HTMLDivElement)
  const edge = placement === "static" ? null : placement.endsWith("top") ? "top" : "bottom"
  useToolbarInset(ref, placement === "top" || placement === "bottom" ? edge : null)
  const minimized = useScrollMinimized(ref, minimize === "onScrollDown")
  return (
    <div
      ref={ref}
      role="toolbar"
      aria-orientation="horizontal"
      data-slot="toolbar"
      data-placement={placement}
      data-bar-edge={edge ?? undefined}
      data-minimized={minimized ? "" : undefined}
      onKeyDown={(e) => {
        onKeyDown?.(e)
        if (!e.defaultPrevented) onToolbarKeyDown(e)
      }}
      className={cn(
        "group/toolbar pointer-events-none z-40 flex items-center gap-2 px-3 *:pointer-events-auto",
        "data-[placement=bottom]:fixed data-[placement=top]:fixed data-[placement=bottom]:inset-x-0 data-[placement=top]:inset-x-0 data-[placement=top]:top-[max(0.75rem,env(safe-area-inset-top))] data-[placement=bottom]:bottom-[max(0.75rem,env(safe-area-inset-bottom))]",
        "data-[placement=sticky-bottom]:sticky data-[placement=sticky-top]:sticky data-[placement=sticky-top]:top-3 data-[placement=sticky-bottom]:bottom-3",
        "origin-top transition-transform duration-(--glass-duration) ease-glass data-[bar-edge=bottom]:origin-bottom data-minimized:scale-90 data-minimized:hover:scale-100 data-minimized:focus-within:scale-100 motion-reduce:transition-none",
        className
      )}
      {...props}
    />
  )
}

/**
 * Items that share one liquid capsule. The buttons inside are transparent — one
 * piece of glass per group, never glass on glass.
 */
function ToolbarGroup({ className, ...props }: React.ComponentProps<typeof GlassGroup>) {
  return <GlassGroup data-slot="toolbar-group" className={cn("glass-light glass-shadow-adaptive", className)} {...props} />
}

/** `flexible` (default) pushes what follows to the far edge; `fixed` is a small gap between groups. */
function ToolbarSpacer({ className, variant = "flexible", ...props }: React.ComponentProps<"div"> & { variant?: "flexible" | "fixed" }) {
  return <div aria-hidden data-slot="toolbar-spacer" data-variant={variant} className={cn("data-[variant=fixed]:w-2 data-[variant=flexible]:flex-1", className)} {...props} />
}

/** The title between the groups: centred, truncating; hidden while minimized. */
function ToolbarTitle({ className, ...props }: React.ComponentProps<"div">) {
  return (
    <div
      data-slot="toolbar-title"
      className={cn("min-w-0 truncate type-glass-heading text-base transition-opacity duration-(--glass-duration) ease-glass group-data-minimized/toolbar:opacity-0", className)}
      {...props}
    />
  )
}

/**
 * A toolbar button. Icon-only by default (give it an aria-label); text grows it into
 * a capsule. Inside a group it is transparent with a fill on hover. `prominent`
 * is the standalone tinted action (Done, Send) — put it outside any group.
 */
function ToolbarItem({ className, prominent = false, asChild = false, ref: forwardedRef, ...props }: React.ComponentProps<"button"> & { prominent?: boolean; asChild?: boolean }) {
  const Comp = asChild ? Slot.Root : "button"
  // the press glow follows the finger
  const ref = useGlassPointer<HTMLButtonElement>({}, forwardedRef)
  return (
    <Comp
      ref={ref}
      data-slot="toolbar-item"
      data-prominent={prominent ? "" : undefined}
      type={asChild ? undefined : "button"}
      className={cn(
        "glass-focus hit-area inline-flex h-control min-w-control shrink-0 items-center justify-center gap-1.5 rounded-button px-2 text-sm font-medium whitespace-nowrap text-foreground transition-[background-color,transform] duration-(--glass-duration) ease-glass hover:bg-fill active:scale-(--glass-press-scale) disabled:pointer-events-none disabled:opacity-50 [&_svg]:pointer-events-none [&_svg:not([class*='size-'])]:size-5",
        "data-prominent:glass-prominent data-prominent:glass-illuminate data-prominent:glass-press-liquid data-prominent:glass-shadow-adaptive data-prominent:h-control-lg data-prominent:min-w-control-lg data-prominent:text-primary-foreground data-prominent:hover:bg-transparent",
        className
      )}
      {...props}
    />
  )
}

/** "More": the items that don't fit, in a glass menu. Children are DropdownMenu items. */
function ToolbarOverflow({ children, label = "More", align = "end", className }: { children: React.ReactNode; label?: string; align?: "start" | "center" | "end"; className?: string }) {
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <ToolbarItem aria-label={label} data-slot="toolbar-overflow" className={className}>
          <EllipsisIcon />
        </ToolbarItem>
      </DropdownMenuTrigger>
      <DropdownMenuContent align={align}>{children}</DropdownMenuContent>
    </DropdownMenu>
  )
}

export { Toolbar, ToolbarGroup, ToolbarItem, ToolbarOverflow, ToolbarSpacer, ToolbarTitle }
