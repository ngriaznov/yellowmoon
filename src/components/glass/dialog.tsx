import * as React from "react"
import { cn } from "cn"
import { Dialog as DialogPrimitive } from "radix-ui"
import { XIcon } from "lucide-react"
import { flushSync } from "react-dom"

import { Button } from "@/components/glass/button"

type Morph = { setTrigger: (el: HTMLElement | null) => void; setContent: (el: HTMLElement | null) => void; morphing: boolean }
const MorphContext = React.createContext<Morph | null>(null)
const MORPH_NAME = "glass-dialog"

/** A same-document View Transition between the trigger and the panel, where supported and motion is allowed. */
type TransitionDocument = Document & { startViewTransition?: (cb: () => void) => { finished: Promise<void> } }
const canMorph = () =>
  typeof document !== "undefined" && typeof (document as TransitionDocument).startViewTransition === "function" && !matchMedia("(prefers-reduced-motion: reduce)").matches

function viewTransition(update: () => void, from: HTMLElement | null | undefined, to: () => HTMLElement | null | undefined): Promise<void> {
  const doc = document as TransitionDocument
  if (!doc.startViewTransition || !from?.isConnected) {
    update()
    return Promise.resolve()
  }
  from.style.viewTransitionName = MORPH_NAME
  const t = doc.startViewTransition(() => {
    from.style.viewTransitionName = ""
    flushSync(update)
    const el = to()
    if (el) el.style.viewTransitionName = MORPH_NAME
  })
  // snapshots are flat images: the live element (and its backdrop blur) takes over when it ends
  return t.finished.catch(() => {}).finally(() => {
    const el = to()
    if (el) el.style.viewTransitionName = ""
  })
}

/**
 * Opens out of its trigger: the trigger morphs into the panel (and back on close) with a
 * View Transition where the browser has them; elsewhere, and under reduced motion, the panel
 * materialises in place. Controlled or not, as Radix's Dialog.
 */
function Dialog({ open: openProp, defaultOpen = false, onOpenChange, ...props }: React.ComponentProps<typeof DialogPrimitive.Root>) {
  const [inner, setInner] = React.useState(defaultOpen)
  const [morphing, setMorphing] = React.useState(false)
  const open = openProp ?? inner
  const trigger = React.useRef<HTMLElement | null>(null)
  const content = React.useRef<HTMLElement | null>(null)
  const morph = React.useMemo<Morph>(
    () => ({
      setTrigger: (el) => {
        trigger.current = el
      },
      setContent: (el) => {
        content.current = el
      },
      morphing,
    }),
    [morphing]
  )
  const setOpen = (next: boolean) => {
    const apply = () => {
      if (openProp === undefined) setInner(next)
      onOpenChange?.(next)
    }
    const t = trigger.current
    if (!t?.isConnected || !canMorph()) return apply()
    flushSync(() => setMorphing(true))
    const done = next ? viewTransition(apply, t, () => content.current) : viewTransition(apply, content.current, () => t)
    done.finally(() => setMorphing(false))
  }
  return (
    <MorphContext.Provider value={morph}>
      <DialogPrimitive.Root data-slot="dialog" open={open} onOpenChange={setOpen} {...props} />
    </MorphContext.Provider>
  )
}

function DialogTrigger({ onClick, ...props }: React.ComponentProps<typeof DialogPrimitive.Trigger>) {
  const morph = React.useContext(MorphContext)
  return (
    <DialogPrimitive.Trigger
      data-slot="dialog-trigger"
      onClick={(e) => {
        morph?.setTrigger(e.currentTarget)
        onClick?.(e)
      }}
      {...props}
    />
  )
}

function DialogPortal({ ...props }: React.ComponentProps<typeof DialogPrimitive.Portal>) {
  return <DialogPrimitive.Portal data-slot="dialog-portal" {...props} />
}

function DialogClose({ ...props }: React.ComponentProps<typeof DialogPrimitive.Close>) {
  return <DialogPrimitive.Close data-slot="dialog-close" {...props} />
}

/** The page dims behind the glass (no blur: a blurred scrim under a frosted pane is blur on blur). */
function DialogOverlay({ className, ...props }: React.ComponentProps<typeof DialogPrimitive.Overlay>) {
  return (
    <DialogPrimitive.Overlay
      data-slot="dialog-overlay"
      data-glass-motion=""
      className={cn(
        "fixed inset-0 isolate z-50 bg-glass-dim duration-(--glass-duration) ease-glass data-open:animate-in data-open:fade-in-0 data-closed:animate-out data-closed:fade-out-0",
        className
      )}
      {...props}
    />
  )
}

/**
 * A liquid-glass panel (the control layer) over a dimmed page. Never taller than the
 * viewport: long content scrolls — wrap it in DialogBody to keep the header and footer in view.
 * Inside it use fills and `secondary` / `ghost` buttons, never more glass.
 */
function DialogContent({
  className,
  children,
  showCloseButton = true,
  ref,
  ...props
}: React.ComponentProps<typeof DialogPrimitive.Content> & { showCloseButton?: boolean }) {
  const morph = React.useContext(MorphContext)
  const setRef = (node: HTMLDivElement | null) => {
    morph?.setContent(node)
    if (typeof ref === "function") ref(node)
    else if (ref) ref.current = node
  }
  return (
    <DialogPortal>
      <DialogOverlay />
      <DialogPrimitive.Content
        ref={setRef}
        data-slot="dialog-content"
        data-glass-motion=""
        // the View Transition is the animation: don't materialise on top of it
        data-morphing={morph?.morphing || undefined}
        className={cn(
          "glass-liquid fixed top-1/2 left-1/2 z-50 grid max-h-[calc(100dvh-2rem)] w-full max-w-[calc(100%-2rem)] -translate-x-1/2 -translate-y-1/2 gap-4 overflow-y-auto overscroll-contain rounded-surface p-6 text-sm text-popover-foreground focus-visible:outline-hidden sm:max-w-md glass-materialize data-morphing:animate-none!",
          className
        )}
        {...props}
      >
        {children}
        {showCloseButton && (
          <DialogPrimitive.Close data-slot="dialog-close" asChild>
            <Button variant="secondary" className="hit-area absolute top-3.5 end-3.5" size="icon-sm">
              <XIcon />
              <span className="sr-only">Close</span>
            </Button>
          </DialogPrimitive.Close>
        )}
      </DialogPrimitive.Content>
    </DialogPortal>
  )
}

function DialogHeader({ className, ...props }: React.ComponentProps<"div">) {
  return <div data-slot="dialog-header" className={cn("flex flex-col gap-1.5 pe-8", className)} {...props} />
}

/** The scrolling middle of a long dialog: the header and footer stay put while this scrolls. */
function DialogBody({ className, ...props }: React.ComponentProps<"div">) {
  return <div data-slot="dialog-body" className={cn("-mx-6 min-h-0 overflow-y-auto overscroll-contain px-6", className)} {...props} />
}

function DialogFooter({ className, showCloseButton = false, children, ...props }: React.ComponentProps<"div"> & { showCloseButton?: boolean }) {
  return (
    <div data-slot="dialog-footer" className={cn("flex flex-col-reverse gap-2 pt-2 sm:flex-row sm:justify-end", className)} {...props}>
      {children}
      {showCloseButton && (
        <DialogPrimitive.Close asChild>
          <Button variant="secondary">Close</Button>
        </DialogPrimitive.Close>
      )}
    </div>
  )
}

function DialogTitle({ className, ...props }: React.ComponentProps<typeof DialogPrimitive.Title>) {
  return <DialogPrimitive.Title data-slot="dialog-title" className={cn("type-glass-heading text-lg", className)} {...props} />
}

function DialogDescription({ className, ...props }: React.ComponentProps<typeof DialogPrimitive.Description>) {
  return <DialogPrimitive.Description data-slot="dialog-description" className={cn("text-sm text-muted-foreground", className)} {...props} />
}

export { Dialog, DialogBody, DialogClose, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogOverlay, DialogPortal, DialogTitle, DialogTrigger }
