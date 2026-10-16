import * as React from "react"

/**
 * Feeds the light of a liquid control from the pointer, without React state:
 *
 *   --glass-press-x / --glass-press-y   where the finger is (the `glass-illuminate` glow)
 *   --glass-light-angle                 with `followLight`, the rim turns toward a hovering pointer
 *
 * Writes happen at most once a frame. `followLight` is for hover devices only and is off under
 * reduced motion. Returns a callback ref; a forwarded ref still receives the node.
 */
export function useGlassPointer<T extends HTMLElement>({ followLight = false }: { followLight?: boolean } = {}, forwardedRef?: React.Ref<T>) {
  const [node, setNode] = React.useState<T | null>(null)
  React.useImperativeHandle(forwardedRef, () => node as T, [node])

  React.useEffect(() => {
    if (!node) return
    let frame = 0
    let last: PointerEvent | null = null
    const canFollow =
      followLight &&
      typeof matchMedia === "function" &&
      matchMedia("(hover: hover) and (pointer: fine)").matches &&
      !matchMedia("(prefers-reduced-motion: reduce)").matches

    const write = () => {
      frame = 0
      if (!last) return
      const r = node.getBoundingClientRect()
      const x = last.clientX - r.left
      const y = last.clientY - r.top
      node.style.setProperty("--glass-press-x", `${x}px`)
      node.style.setProperty("--glass-press-y", `${y}px`)
      // the lit side faces the pointer: conic 0deg is "up", so angle = atan2(dx, -dy)
      if (canFollow) node.style.setProperty("--glass-light-angle", `${Math.round((Math.atan2(x - r.width / 2, r.height / 2 - y) * 180) / Math.PI)}deg`)
    }
    const onMove = (e: PointerEvent) => {
      if (!canFollow && e.type === "pointermove" && !e.buttons) return
      last = e
      if (!frame) frame = requestAnimationFrame(write)
    }
    const onLeave = () => {
      if (canFollow) node.style.removeProperty("--glass-light-angle")
    }

    node.addEventListener("pointerdown", onMove)
    node.addEventListener("pointermove", onMove)
    node.addEventListener("pointerleave", onLeave)
    return () => {
      cancelAnimationFrame(frame)
      node.removeEventListener("pointerdown", onMove)
      node.removeEventListener("pointermove", onMove)
      node.removeEventListener("pointerleave", onLeave)
    }
  }, [node, followLight])

  return setNode
}
