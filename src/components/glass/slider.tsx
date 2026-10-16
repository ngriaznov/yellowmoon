import * as React from "react"
import { cn } from "cn"
import { Slider as SliderPrimitive } from "radix-ui"

type SliderProps = React.ComponentProps<typeof SliderPrimitive.Root> & {
  /** `progress` is thumbless (a scrubber, like a media timeline); the track thickens while dragged. */
  variant?: "default" | "progress"
  /** Tick marks: `true` for one per step (up to 21), or the values to mark. */
  ticks?: boolean | number[]
  /** The range grows from this value instead of `min` (e.g. 0 on a −10…10 balance). */
  neutralValue?: number
  /** Icons at the ends of the track (e.g. a quiet and a loud speaker). */
  minIcon?: React.ReactNode
  maxIcon?: React.ReactNode
  /** A bubble above the thumb showing its value while dragging; a function formats it. */
  valueBubble?: boolean | ((value: number) => React.ReactNode)
}

const pct = (v: number, min: number, max: number) => ((v - min) / (max - min || 1)) * 100

/**
 * The iOS 27 slider: a soft track, an accent range and a capsule knob that turns into a liquid
 * lens while dragged (stretching with the drag). Ranges, ticks, a neutral origin, end icons, a
 * value bubble and a thumbless `progress` variant.
 */
function Slider({
  className,
  defaultValue,
  value,
  min = 0,
  max = 100,
  step = 1,
  orientation = "horizontal",
  variant = "default",
  ticks,
  neutralValue,
  minIcon,
  maxIcon,
  valueBubble,
  onValueChange,
  onPointerDown,
  onPointerMove,
  "aria-label": ariaLabel,
  "aria-labelledby": ariaLabelledby,
  ...props
}: SliderProps) {
  const [inner, setInner] = React.useState<number[]>(() => (Array.isArray(defaultValue) ? defaultValue : [min]))
  const values = Array.isArray(value) ? value : inner
  const [dragging, setDragging] = React.useState(false)
  const last = React.useRef(0)
  const vertical = orientation === "vertical"
  const neutral = neutralValue !== undefined && values.length === 1 ? Math.min(max, Math.max(min, neutralValue)) : undefined

  const marks = React.useMemo(() => {
    if (Array.isArray(ticks)) return ticks
    if (!ticks) return []
    const n = Math.round((max - min) / step)
    return n > 0 && n <= 20 ? Array.from({ length: n + 1 }, (_, i) => min + i * step) : []
  }, [ticks, min, max, step])

  const endDrag = (e: React.PointerEvent<HTMLElement>) => {
    setDragging(false)
    e.currentTarget.style.setProperty("--glass-knob-stretch-x", "0")
  }

  const slider = (
    <SliderPrimitive.Root
      data-slot="slider"
      data-variant={variant}
      data-dragging={dragging || undefined}
      defaultValue={defaultValue}
      value={value}
      min={min}
      max={max}
      step={step}
      orientation={orientation}
      onValueChange={(v) => {
        setInner(v)
        onValueChange?.(v)
      }}
      onPointerDown={(e) => {
        onPointerDown?.(e)
        last.current = vertical ? e.clientY : e.clientX
        setDragging(true)
      }}
      onPointerMove={(e) => {
        onPointerMove?.(e)
        if (!dragging) return
        // the lens stretches with the speed of the drag
        const at = vertical ? e.clientY : e.clientX
        const v = Math.max(-1, Math.min(1, (at - last.current) / 12))
        last.current = at
        e.currentTarget.style.setProperty("--glass-knob-stretch-x", String(Math.round(v * 100) / 100))
      }}
      onPointerUp={endDrag}
      onLostPointerCapture={endDrag}
      className={cn(
        "group/slider relative flex w-full touch-none items-center select-none data-disabled:opacity-50 data-vertical:h-full data-vertical:min-h-40 data-vertical:w-auto data-vertical:flex-col",
        variant === "default" ? "data-horizontal:min-h-knob data-vertical:min-w-knob" : "data-horizontal:min-h-3 data-vertical:min-w-3",
        !(minIcon || maxIcon) && className
      )}
      {...props}
    >
      <SliderPrimitive.Track
        data-slot="slider-track"
        className="relative grow overflow-hidden rounded-button bg-fill-strong transition-[height,width] duration-(--glass-duration) ease-glass data-horizontal:h-1.5 data-horizontal:w-full data-vertical:h-full data-vertical:w-1.5 group-data-[variant=progress]/slider:group-data-dragging/slider:data-horizontal:h-2.5 group-data-[variant=progress]/slider:group-data-dragging/slider:data-vertical:w-2.5"
      >
        {neutral === undefined ? (
          <SliderPrimitive.Range data-slot="slider-range" className="absolute bg-primary data-horizontal:h-full data-vertical:w-full" />
        ) : (
          <span
            data-slot="slider-range"
            className="absolute bg-primary data-horizontal:h-full data-vertical:w-full"
            data-orientation={orientation}
            style={
              vertical
                ? { bottom: `${Math.min(pct(neutral, min, max), pct(values[0] ?? min, min, max))}%`, height: `${Math.abs(pct(values[0] ?? min, min, max) - pct(neutral, min, max))}%` }
                : { insetInlineStart: `${Math.min(pct(neutral, min, max), pct(values[0] ?? min, min, max))}%`, width: `${Math.abs(pct(values[0] ?? min, min, max) - pct(neutral, min, max))}%` }
            }
          />
        )}
      </SliderPrimitive.Track>
      {marks.length > 0 && (
        <span data-slot="slider-ticks" aria-hidden className="pointer-events-none absolute inset-0">
          {marks.map((m) => (
            <span
              key={m}
              className="absolute size-1 -translate-1/2 rounded-full bg-glass-label-tertiary data-horizontal:top-1/2 data-vertical:left-1/2"
              data-orientation={orientation}
              style={vertical ? { bottom: `${pct(m, min, max)}%` } : { insetInlineStart: `${pct(m, min, max)}%` }}
            />
          ))}
        </span>
      )}
      {values.map((v, i) => (
        <SliderPrimitive.Thumb
          data-slot="slider-thumb"
          key={i}
          // the thumb is the focusable slider: it carries the name (a range names each end)
          aria-label={ariaLabel && values.length > 1 ? `${ariaLabel} ${i === 0 ? "minimum" : "maximum"}` : ariaLabel}
          aria-labelledby={ariaLabelledby}
          data-glass-motion
          className={cn(
            "glass-knob-lens relative block shrink-0 rounded-button border border-glass-border shadow-glass-knob glass-focus hit-area data-disabled:pointer-events-none motion-reduce:transition-none",
            // a capsule, wider along the track
            "data-horizontal:h-knob data-horizontal:w-[calc(2.25rem*var(--glass-density))] data-vertical:h-[calc(2.25rem*var(--glass-density))] data-vertical:w-knob",
            variant === "progress" && "size-0! border-0 opacity-0 focus-visible:size-knob! focus-visible:opacity-100"
          )}
        >
          {valueBubble && (
            <span
              data-slot="slider-value"
              aria-hidden
              className="pointer-events-none absolute bottom-full left-1/2 mb-2 -translate-x-1/2 scale-75 rounded-badge bg-primary px-2 py-0.5 text-xs font-semibold text-primary-foreground tabular-nums opacity-0 transition-[opacity,scale] duration-(--glass-duration) ease-glass group-data-dragging/slider:scale-100 group-data-dragging/slider:opacity-100"
            >
              {typeof valueBubble === "function" ? valueBubble(v) : v}
            </span>
          )}
        </SliderPrimitive.Thumb>
      ))}
    </SliderPrimitive.Root>
  )

  if (!(minIcon || maxIcon)) return slider
  return (
    <div data-slot="slider-container" className={cn("flex w-full items-center gap-3 text-muted-foreground [&_svg:not([class*='size-'])]:size-4", vertical && "h-full flex-col-reverse", className)}>
      {minIcon && <span aria-hidden>{minIcon}</span>}
      {slider}
      {maxIcon && <span aria-hidden>{maxIcon}</span>}
    </div>
  )
}

export { Slider }
