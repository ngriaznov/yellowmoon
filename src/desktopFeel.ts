/**
 * What keeps the web view from behaving like a web page (`docs/11-ui.md` §Desktop feel, task M3-16): a pinch that no part of the page
 * handles never reaches the browser's own zoom or scroll, and a right click does not open WebKit's page menu (Reload, Back) over an app.
 * The scrolling and the text selection are CSS (`src/styles/global.css`); the photo's own wheel and pinch are `usePhotoGestures`.
 */

/** WebKit's pinch events. They are not standard and not in `lib.dom.d.ts`. */
export const GESTURE_EVENTS = ["gesturestart", "gesturechange", "gestureend"] as const;

/** A field, or a text the person may select (and so may copy): the page menu is allowed there. */
const EDITABLE = 'input, textarea, select, [contenteditable]:not([contenteditable="false"])';

const SELECTABLE = new Set(["text", "all", "contain"]);

/** The computed `user-select` of an element; WebKit has it prefixed. Empty when the style can't say (a test without CSS). */
function userSelectOf(el: Element): string {
  const style = getComputedStyle(el) as CSSStyleDeclaration & { webkitUserSelect?: string };
  return style.userSelect || style.webkitUserSelect || "";
}

/**
 * True when `target` is in editable text or in text the person can select. `auto` takes the parent's value, so the nearest ancestor
 * that says something decides (a paragraph inside the About box is selectable because its section is).
 */
export function isTextTarget(target: EventTarget | null): boolean {
  const node = target as Node | null;
  const el = node instanceof Element ? node : (node?.parentElement ?? null);
  if (!el) return false;
  if (el.closest(EDITABLE)) return true;
  for (let up: Element | null = el; up; up = up.parentElement) {
    const value = userSelectOf(up);
    if (value && value !== "auto") return SELECTABLE.has(value);
  }
  return false;
}

/**
 * Cancels the pinch and the page menu everywhere except where something handles them itself: the photo cell handles its own
 * gesture events before they get here, and a field or a selectable text keeps its menu (Copy, Paste). Returns the function that
 * takes the listeners off again.
 */
export function installDesktopFeel(win: Window = window): () => void {
  const cancel = (event: Event): void => event.preventDefault();
  const onContextMenu = (event: Event): void => {
    if (!isTextTarget(event.target)) event.preventDefault();
  };
  for (const name of GESTURE_EVENTS) win.addEventListener(name, cancel, { passive: false });
  win.addEventListener("contextmenu", onContextMenu);
  return () => {
    for (const name of GESTURE_EVENTS) win.removeEventListener(name, cancel);
    win.removeEventListener("contextmenu", onContextMenu);
  };
}
