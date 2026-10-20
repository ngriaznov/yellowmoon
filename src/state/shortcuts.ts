/**
 * The keyboard shortcuts of `docs/11-ui.md` §Keyboard shortcuts, as one pure function from a
 * key event to an action name, so the table is tested without a screen. ⌘ is the Meta key; the
 * app is macOS only. Keys that type (P, Z, C, W, 1–9, the arrows) are ignored while a text box
 * has focus, and so are ⇧⌘C, ⇧⌘V and ⌥⇧⌘V (copy and paste settings; plain ⌘C and ⌘V never reach this
 * table: they stay the system's Copy and Paste). The arrows are left to the control that has focus when
 * it uses them (a slider, the filmstrip, the tabs). Delete and Backspace remove the photo on screen from the filmstrip, except where a
 * control uses them itself (a text box, a slider with its knob, a menu, a dialog) or a mode (crop, pick white) is on.
 */
import type { TabId } from "../components/tabs";

/** What a shortcut asks for. */
export type ShortcutAction =
  | "open-files"
  | "open-folder"
  | "export"
  | "close-session"
  | "undo"
  | "redo"
  | "previous-photo"
  | "next-photo"
  | "toggle-pick"
  | "remove-photo"
  | "pick-all"
  | "pick-look"
  | "before"
  | "toggle-zoom"
  | "zoom-in"
  | "zoom-out"
  | "crop"
  | "pick-white"
  | "copy-settings"
  | "paste-settings"
  | "paste-settings-picked"
  | "preferences"
  | "escape"
  | "confirm";

/** A shortcut that fired. `look` is 1–9 for `pick-look`. */
export interface Shortcut {
  action: ShortcutAction;
  look?: number;
}

/** The parts of a key event the table reads. */
export interface KeyLike {
  key: string;
  code?: string;
  metaKey: boolean;
  ctrlKey: boolean;
  altKey: boolean;
  shiftKey: boolean;
  defaultPrevented?: boolean;
  /** The key is held down and repeating. */
  repeat?: boolean;
  target: EventTarget | null;
}

/** Where the shortcuts apply. */
export interface ShortcutContext {
  /** The tab on screen; 1–9 pick looks only on the Look tab. */
  tab?: TabId;
  /** A modal-like mode is on (crop, pick white, pick colour): the number keys that pick looks are ignored. */
  mode?: "crop" | "pick-white" | "pick-color" | null;
}

/** True when the target takes typed text. */
export function isTextTarget(target: EventTarget | null): boolean {
  const el = target as HTMLElement | null;
  if (!el || typeof el.tagName !== "string") return false;
  if (el.isContentEditable) return true;
  const tag = el.tagName.toLowerCase();
  if (tag === "textarea" || tag === "select") return true;
  if (tag !== "input") return false;
  const type = (el as HTMLInputElement).type;
  // A range input is a slider: it takes the arrows but not letters.
  return !["range", "checkbox", "radio", "button"].includes(type);
}

/** True when Return on the target presses it (a button, a link, a radio) rather than reaching the screen. The crop frame's own handles are not counted. */
function pressesItself(target: EventTarget | null): boolean {
  const el = target as HTMLElement | null;
  if (!el || typeof el.tagName !== "string") return false;
  if (el.closest(".ym-crop")) return false;
  const tag = el.tagName.toLowerCase();
  if (tag === "button" || tag === "summary" || (tag === "a" && el.hasAttribute("href"))) return true;
  const role = el.getAttribute("role");
  return role !== null && ["button", "radio", "tab", "switch", "menuitem", "menuitemradio", "slider"].includes(role);
}

/** True when the target uses the left and right arrows itself. */
function usesHorizontalArrows(target: EventTarget | null): boolean {
  const el = target as HTMLElement | null;
  if (!el || typeof el.tagName !== "string") return false;
  if (isTextTarget(el)) return true;
  if (el.tagName.toLowerCase() === "input" && (el as HTMLInputElement).type === "range") return true;
  const role = el.getAttribute("role");
  if (role && ["slider", "radio", "tab", "menuitemradio", "menuitem", "application"].includes(role)) return true;
  if (el.closest("[data-photo-id]")) return true; // a filmstrip thumbnail
  if (el.closest(".ym-crop")) return true;
  return false;
}

/**
 * True when the target uses Delete and Backspace itself: a text box, a slider (the colour warp's pins and the curve's points remove
 * themselves with them, a wheel centres), a menu, a list, a dialog, or the crop frame. They are not the person's way to remove a photo
 * then.
 */
function usesDeleteKey(target: EventTarget | null): boolean {
  const el = target as HTMLElement | null;
  if (!el || typeof el.tagName !== "string") return false;
  if (isTextTarget(el)) return true;
  if (el.tagName.toLowerCase() === "input" && (el as HTMLInputElement).type === "range") return true;
  const role = el.getAttribute("role");
  if (role && ["slider", "spinbutton", "textbox", "combobox", "listbox", "option", "menuitem", "menuitemradio", "menuitemcheckbox", "tree", "treeitem", "grid", "gridcell"].includes(role)) return true;
  return el.closest('.ym-crop, [role="menu"], [role="dialog"], [role="alertdialog"], [role="listbox"]') !== null;
}

/** The shortcut a key event stands for, or null. */
export function shortcutFor(event: KeyLike, context: ShortcutContext = {}): Shortcut | null {
  if (event.defaultPrevented) return null;
  const key = event.key.length === 1 ? event.key.toLowerCase() : event.key;

  // ⌥⇧⌘V pastes the copied settings onto the picked photos. With ⌥ held the key often reports the character ⌥ makes (√, ◊), so the
  // key's place on the keyboard names it.
  if (event.metaKey && event.altKey && event.shiftKey && !event.ctrlKey && (event.code === "KeyV" || key === "v")) {
    return isTextTarget(event.target) ? null : { action: "paste-settings-picked" };
  }

  if (event.metaKey && !event.ctrlKey && !event.altKey) {
    switch (key) {
      case "o":
        return { action: event.shiftKey ? "open-folder" : "open-files" };
      case "e":
        return event.shiftKey ? null : { action: "export" };
      case "w":
        return event.shiftKey ? null : { action: "close-session" };
      case "z":
        // A text box keeps its own undo.
        return isTextTarget(event.target) ? null : { action: event.shiftKey ? "redo" : "undo" };
      case "a":
        return event.shiftKey || isTextTarget(event.target) ? null : { action: "pick-all" };
      // Plain ⌘C and ⌘V are the system's Copy and Paste; the shifted keys copy and paste a photo's settings. A text box keeps them.
      case "c":
        return event.shiftKey && !isTextTarget(event.target) ? { action: "copy-settings" } : null;
      case "v":
        return event.shiftKey && !isTextTarget(event.target) ? { action: "paste-settings" } : null;
      case ",":
        return { action: "preferences" };
      case "+":
      case "=":
        return { action: "zoom-in" };
      case "-":
      case "_":
        return { action: "zoom-out" };
      default:
        return null;
    }
  }
  if (event.metaKey || event.ctrlKey || event.altKey) return null;

  if (key === "Escape") return { action: "escape" };
  // Return finishes a crop, unless the focused control is a button or link that Return presses itself.
  if (key === "Enter" && context.mode === "crop" && !isTextTarget(event.target) && !pressesItself(event.target)) return { action: "confirm" };
  if (isTextTarget(event.target)) return null;

  switch (key) {
    case "ArrowLeft":
    case "ArrowRight":
      if (usesHorizontalArrows(event.target)) return null;
      return { action: key === "ArrowLeft" ? "previous-photo" : "next-photo" };
    case "Delete":
    case "Backspace":
      // A mode (crop, Pick white) has its own use of the keys; with Shift the key is not this shortcut. A key held down removes one
      // photo, not the whole strip (only the last removal can be undone).
      if (event.shiftKey || event.repeat || context.mode || usesDeleteKey(event.target)) return null;
      return { action: "remove-photo" };
    case "p":
      return { action: "toggle-pick" };
    case "\\":
      return { action: "before" };
    case "z":
      return { action: "toggle-zoom" };
    case "c":
      return { action: "crop" };
    case "w":
      return { action: "pick-white" };
    default:
      break;
  }
  if (/^[1-9]$/.test(key) && context.tab === "look" && !context.mode) return { action: "pick-look", look: Number(key) };
  return null;
}
