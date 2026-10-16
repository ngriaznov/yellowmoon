"use client"

import * as React from "react"
import { cn } from "cn"
import { SearchIcon, XIcon } from "lucide-react"

/**
 * A capsule of liquid glass for search: a magnifier, the field, Clear while there is
 * text and Cancel while it is in use. `minimized` starts it as a round button that
 * morphs open; `placement="bottom"` floats it at the bottom edge and rides the
 * on-screen keyboard (visualViewport). Pass `suggestions` to get a listbox under
 * (or, at the bottom, over) the field — a combobox, arrow keys to move, Enter to pick.
 *
 *   <SearchField placeholder="Search entries" value={q} onValueChange={setQ}
 *     suggestions={matches} onSuggestionSelect={open} />
 */

type Suggestion = string | { value: string; label: React.ReactNode }
const suggestionValue = (s: Suggestion) => (typeof s === "string" ? s : s.value)
const suggestionLabel = (s: Suggestion) => (typeof s === "string" ? s : s.label)

type SearchFieldProps = Omit<React.ComponentProps<"input">, "value" | "defaultValue" | "onChange" | "onSubmit"> & {
  value?: string
  defaultValue?: string
  onValueChange?: (value: string) => void
  onSubmit?: (value: string) => void
  /** Start as a round search button; it opens on press and folds back when left empty. */
  minimized?: boolean
  expanded?: boolean
  onExpandedChange?: (expanded: boolean) => void
  /** `bottom` floats it fixed at the bottom edge, above the keyboard when one is up. */
  placement?: "inline" | "bottom"
  /** Show Cancel while focused (`auto`), always, or never. */
  showCancel?: "auto" | "always" | "never"
  cancelLabel?: string
  onCancel?: () => void
  suggestions?: Suggestion[]
  onSuggestionSelect?: (value: string) => void
}

/** Keeps `--keyboard-inset` on the element: how much of the layout viewport the keyboard covers. */
function useKeyboardInset(ref: React.RefObject<HTMLElement | null>, enabled: boolean) {
  React.useEffect(() => {
    const el = ref.current
    const vv = typeof window === "undefined" ? undefined : window.visualViewport
    if (!el || !enabled || !vv) return
    const update = () => el.style.setProperty("--keyboard-inset", `${Math.max(0, window.innerHeight - vv.height - vv.offsetTop)}px`)
    update()
    vv.addEventListener("resize", update)
    vv.addEventListener("scroll", update)
    return () => {
      vv.removeEventListener("resize", update)
      vv.removeEventListener("scroll", update)
    }
  }, [ref, enabled])
}

function useControlled<T>(value: T | undefined, initial: T, onChange?: (v: T) => void) {
  const [inner, setInner] = React.useState(initial)
  const current = value ?? inner
  const set = React.useCallback(
    (v: T) => {
      setInner(v)
      onChange?.(v)
    },
    [onChange]
  )
  return [current, set] as const
}

function SearchField({
  className,
  value,
  defaultValue = "",
  onValueChange,
  onSubmit,
  minimized = false,
  expanded,
  onExpandedChange,
  placement = "inline",
  showCancel = "auto",
  cancelLabel = "Cancel",
  onCancel,
  suggestions = [],
  onSuggestionSelect,
  placeholder = "Search",
  onFocus,
  onBlur,
  onKeyDown,
  id,
  ...props
}: SearchFieldProps) {
  const [text, setText] = useControlled(value, defaultValue, onValueChange)
  const [open, setOpen] = useControlled(expanded, !minimized, onExpandedChange)
  const [focused, setFocused] = React.useState(false)
  const [active, setActive] = React.useState(-1)
  const root = React.useRef<HTMLDivElement>(null)
  const input = React.useRef<HTMLInputElement>(null)
  const autoId = React.useId()
  const listId = `${id ?? autoId}-suggestions`
  useKeyboardInset(root, placement === "bottom")

  const listOpen = focused && suggestions.length > 0
  const cancel = showCancel === "always" || (showCancel === "auto" && (focused || (minimized && open)))

  const expand = () => {
    setOpen(true)
    requestAnimationFrame(() => input.current?.focus())
  }
  const finish = () => {
    setText("")
    setActive(-1)
    input.current?.blur()
    if (minimized) setOpen(false)
    onCancel?.()
  }
  const pick = (s: Suggestion) => {
    const v = suggestionValue(s)
    setText(v)
    setActive(-1)
    onSuggestionSelect?.(v)
  }

  return (
    <div
      ref={root}
      data-slot="search-field"
      data-placement={placement}
      data-expanded={open ? "" : undefined}
      data-bar-edge={placement === "bottom" ? "bottom" : undefined}
      className={cn(
        "relative flex items-center gap-2",
        "data-[placement=bottom]:fixed data-[placement=bottom]:inset-x-3 data-[placement=bottom]:z-40 data-[placement=bottom]:bottom-[calc(max(0.75rem,env(safe-area-inset-bottom))+var(--keyboard-inset,0px))] data-[placement=bottom]:transition-[bottom] data-[placement=bottom]:duration-(--glass-duration) data-[placement=bottom]:ease-glass",
        className
      )}
    >
      {!open ? (
        <button
          type="button"
          data-slot="search-field-button"
          aria-label={placeholder}
          aria-expanded={false}
          onClick={expand}
          className="glass-liquid glass-illuminate glass-press-liquid glass-shadow-adaptive glass-focus hit-area inline-flex size-control-lg shrink-0 items-center justify-center rounded-button text-foreground [&_svg]:size-5"
        >
          <SearchIcon />
        </button>
      ) : (
        <div
          data-slot="search-field-control"
          className="glass-liquid glass-shadow-adaptive flex h-control-lg w-full min-w-0 flex-1 items-center gap-2 rounded-button ps-3.5 pe-1.5 transition-[width] duration-(--glass-duration) ease-glass focus-within:outline-(length:--glass-ring-width) focus-within:outline-ring/60 starting:w-control-lg motion-reduce:transition-none"
        >
          <SearchIcon aria-hidden className="size-4.5 shrink-0 text-muted-foreground" />
          <input
            ref={input}
            id={id}
            type="search"
            enterKeyHint="search"
            autoComplete="off"
            data-slot="search-field-input"
            placeholder={placeholder}
            aria-label={props["aria-label"] ?? (typeof placeholder === "string" ? placeholder : "Search")}
            role={suggestions.length ? "combobox" : undefined}
            aria-autocomplete={suggestions.length ? "list" : undefined}
            aria-expanded={suggestions.length ? listOpen : undefined}
            aria-controls={suggestions.length ? listId : undefined}
            aria-activedescendant={listOpen && active >= 0 ? `${listId}-${active}` : undefined}
            value={text}
            onChange={(e) => {
              setText(e.target.value)
              setActive(-1)
            }}
            onFocus={(e) => {
              setFocused(true)
              onFocus?.(e)
            }}
            onBlur={(e) => {
              setFocused(false)
              if (minimized && !e.target.value && !root.current?.contains(e.relatedTarget as Node)) setOpen(false)
              onBlur?.(e)
            }}
            onKeyDown={(e) => {
              onKeyDown?.(e)
              if (e.defaultPrevented) return
              if (listOpen && (e.key === "ArrowDown" || e.key === "ArrowUp")) {
                const n = suggestions.length
                setActive((i) => (e.key === "ArrowDown" ? (i + 1) % n : (i - 1 + n) % n))
                e.preventDefault()
              } else if (e.key === "Enter") {
                if (listOpen && active >= 0 && suggestions[active] !== undefined) pick(suggestions[active])
                else onSubmit?.(text)
                e.preventDefault()
              } else if (e.key === "Escape") {
                // Local change to the registry copy (Yellowmoon): an Esc that clears text is used up, but one in an empty field is let
                // through to the page, so a mode that Esc cancels (Pick white, Crop) still hears it.
                if (text) {
                  setText("")
                  e.preventDefault()
                } else finish()
              }
            }}
            className="h-full min-w-0 flex-1 bg-transparent text-base text-foreground outline-hidden placeholder:text-muted-foreground sm:text-sm [&::-webkit-search-cancel-button]:appearance-none"
            {...props}
          />
          {text ? (
            <button
              type="button"
              data-slot="search-field-clear"
              aria-label="Clear"
              // Local change to the registry copy (Yellowmoon): Clear is a click target, not a Tab stop, as in a macOS search field; Tab
              // goes from the field to what it filters.
              tabIndex={-1}
              onClick={() => {
                setText("")
                input.current?.focus()
              }}
              className="glass-focus hit-area inline-flex size-control-xs shrink-0 items-center justify-center rounded-button bg-fill text-muted-foreground hover:bg-fill-strong [&_svg]:size-3.5"
            >
              <XIcon />
            </button>
          ) : null}
        </div>
      )}
      {open && cancel ? (
        <button
          type="button"
          data-slot="search-field-cancel"
          // keep focus in the field until the click lands
          onMouseDown={(e) => e.preventDefault()}
          onClick={finish}
          className="glass-focus shrink-0 rounded-button px-1 text-sm font-medium text-primary-ink"
        >
          {cancelLabel}
        </button>
      ) : null}
      {open && listOpen ? (
        <ul
          id={listId}
          role="listbox"
          data-slot="search-field-suggestions"
          aria-label="Suggestions"
          className="glass-liquid absolute inset-x-0 z-50 grid max-h-72 gap-0.5 overflow-y-auto rounded-surface p-1.5 in-data-[placement=bottom]:bottom-full in-data-[placement=bottom]:mb-2 in-data-[placement=inline]:top-full in-data-[placement=inline]:mt-2"
        >
          {suggestions.map((s, i) => (
            <li
              key={suggestionValue(s)}
              id={`${listId}-${i}`}
              role="option"
              aria-selected={i === active}
              data-slot="search-field-suggestion"
              onMouseDown={(e) => e.preventDefault()}
              onClick={() => pick(s)}
              onPointerMove={() => setActive(i)}
              className="flex h-control cursor-default items-center gap-2 rounded-control px-pad-sm text-sm aria-selected:bg-fill"
            >
              <SearchIcon aria-hidden className="size-3.5 text-muted-foreground" />
              {suggestionLabel(s)}
            </li>
          ))}
        </ul>
      ) : null}
    </div>
  )
}

export { SearchField }
export type { SearchFieldProps, Suggestion }
