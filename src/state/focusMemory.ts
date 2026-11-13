/**
 * Where the keyboard focus was when a layer opened, so it can go back there when the layer closes (`docs/11` §Keyboard, `docs/28` A-04).
 *
 * The Export panel and the node view replace parts of the screen, and the control that had the focus (Fit, Nodes, a thumbnail) is often
 * drawn again in the same step: the browser drops the focus to `body` and a keyboard user has to Tab in from the top. A layer calls
 * {@link rememberFocus} when it opens, in the event that opens it (the key, the click, the menu item: the only moment the focused element is
 * still the right one), and {@link restoreFocus} once it has closed.
 */

/** What is remembered of the focused control: the element itself, and enough to find its replacement when the screen drew it again. */
interface Remembered {
  element: HTMLElement;
  tag: string;
  role: string | null;
  name: string;
}

const slots = new Map<string, Remembered>();

/** The accessible name that is good enough to tell a control from its neighbours: its label, else its text. */
function nameOf(element: Element): string {
  return (element.getAttribute("aria-label") ?? element.textContent ?? "").replace(/\s+/g, " ").trim();
}

/** True for an element the keyboard can be sent to: in the page, and not inside something inert or hidden. */
function usable(element: HTMLElement): boolean {
  return element.isConnected && element.closest("[inert], [hidden]") === null;
}

/**
 * The focus is on nothing: the page's body, or nothing at all, or a control that has just been hidden or made inert (a browser moves the
 * focus off such a control at its next frame, after the effect that restores it has run).
 */
function lost(): boolean {
  if (typeof document === "undefined") return true;
  const active = document.activeElement;
  return active === null || active === document.body || !(active instanceof HTMLElement) || !usable(active);
}

/**
 * Remembers the focused control under `slot`. Nothing is remembered when the focus is on nothing (a click on a button in Safari does not
 * focus it); the slot is then cleared, so an earlier visit cannot be restored by mistake.
 */
export function rememberFocus(slot: string): void {
  slots.delete(slot);
  if (typeof document === "undefined" || lost()) return;
  const element = document.activeElement;
  if (!(element instanceof HTMLElement)) return;
  slots.set(slot, { element, tag: element.tagName, role: element.getAttribute("role"), name: nameOf(element) });
}

/** The remembered control if it is still on the page, else a control of the same kind and name that has taken its place. */
function find(memory: Remembered): HTMLElement | null {
  if (usable(memory.element)) return memory.element;
  if (memory.name === "") return null;
  for (const candidate of document.querySelectorAll<HTMLElement>(memory.tag)) {
    if (candidate.getAttribute("role") === memory.role && usable(candidate) && nameOf(candidate) === memory.name) return candidate;
  }
  return null;
}

/**
 * Gives the focus back to what `slot` remembers, when the layer has closed and the focus has fallen to nothing. A focus that is somewhere
 * (the person clicked a thumbnail meanwhile) is left alone. When the remembered control is gone for good, `fallback` says where to go.
 * The slot is emptied. Returns the element that took the focus, or null.
 */
export function restoreFocus(slot: string, fallback: () => HTMLElement | null): HTMLElement | null {
  const memory = slots.get(slot);
  slots.delete(slot);
  if (!lost()) return null;
  const target = (memory ? find(memory) : null) ?? fallback();
  if (target === null || !usable(target)) return null;
  target.focus({ preventScroll: true });
  return target;
}
