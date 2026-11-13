/**
 * What the Photos screen says after a shortcut (`docs/28` A-08, A-P7; the table is in `docs/11` §Announcements). One pure function,
 * {@link announcementFor}, maps a shortcut's result to a short plain sentence, so each one is tested without a screen; {@link createAnnouncer}
 * decides when the sentence is said (at once for a key pressed, once at the end for a key held down).
 */
import type { ViewportMode } from "../api/types";
import { lookLabeller } from "./looks";
import type { ShortcutAction } from "./shortcuts";
import type { AppState } from "./store";
import { showsWhole } from "./zoom";

/** The actions that have a sentence: the shortcuts, and the menu's Clear Picks, which no key does. */
export type Announced = ShortcutAction | "clear-picks";

/** What the sentences are made of, read after the action has run. */
export interface AnnounceFacts {
  /** The look on the photo now, named as the look chip names it. */
  look: string | null;
  /** The Before view is on. */
  before: boolean;
  /** The zoom now. */
  zoom: ViewportMode;
  /** The zoom shows the whole photo (Fit, or a step no larger than Fit). */
  whole: boolean;
  /** The photo on screen is picked. */
  picked: boolean;
  /** How many photos are picked. */
  pickedCount: number;
  /** The photo on screen, 1-based, and how many there are. */
  position: number | null;
  count: number;
  /**
   * The state a toggle went to (crop, auto crop, the node view, bypass, Show mask), or for undo and redo whether there was a step to take. Only
   * the actions that toggle read it.
   */
  on: boolean;
}

/** Facts from the store's state. `fit` is `store.fitPercentExact()`; `on` is for the toggles. */
export function announceFacts(state: Pick<AppState, "session" | "settings" | "looks" | "before" | "zoom">, fit: number | null, on = true): AnnounceFacts {
  const session = state.session;
  const photos = session?.photos ?? [];
  const index = session?.current_index ?? null;
  const photo = index === null ? undefined : photos[index];
  const looks = state.looks ?? [];
  const lookId = photo ? state.settings[photo.id]?.values.output.look : undefined;
  const look = lookId ? looks.find((l) => l.id === lookId) : undefined;
  return {
    look: look ? lookLabeller(looks)(look) : null,
    before: state.before,
    zoom: state.zoom,
    whole: showsWhole(state.zoom, fit),
    picked: photo?.picked === true,
    pickedCount: session?.picked.length ?? 0,
    position: index === null ? null : index + 1,
    count: photos.length,
    on,
  };
}

/**
 * The sentence for what `action` did, or null when it has none: its result moves the focus or opens something that is announced (a
 * dialog, a menu, the Export panel, Pick white's hint, the notice bar), or nothing happened.
 */
export function announcementFor(action: Announced, facts: AnnounceFacts): string | null {
  switch (action) {
    case "pick-look":
      return facts.look ? `Look: ${facts.look}.` : null;
    case "before":
      return facts.before ? "Before." : "After.";
    case "toggle-zoom":
    case "zoom-in":
    case "zoom-out":
      if (facts.whole || facts.zoom.kind === "fit") return "Zoom to fit.";
      return `Zoom ${Math.round(facts.zoom.value)} %.`;
    case "toggle-pick":
      if (facts.picked) return `Picked. ${facts.pickedCount} picked.`;
      return facts.pickedCount === 0 ? "Not picked. None picked." : `Not picked. ${facts.pickedCount} picked.`;
    case "pick-all":
      return facts.pickedCount === 0 ? null : `Picked all. ${facts.pickedCount} picked.`;
    case "clear-picks":
      return "Picks cleared.";
    case "previous-photo":
    case "next-photo":
      return facts.position === null ? null : `Photo ${facts.position} of ${facts.count}.`;
    case "undo":
      return facts.on ? "Undone." : null;
    case "redo":
      return facts.on ? "Redone." : null;
    case "crop":
      return facts.on ? "Crop. Press Return to finish." : null;
    case "confirm":
      return "Crop finished.";
    case "auto-crop":
      return facts.on ? "Auto crop on." : "Auto crop off.";
    case "nodes":
      return facts.on ? "Node view." : "Simple view.";
    case "bypass-node":
      return facts.on ? "Node bypassed." : "Node on.";
    case "mask-view":
      return facts.on ? "Mask shown." : "Mask hidden.";
    default:
      return null;
  }
}

/** How long a key held down must be quiet before its sentence is said. */
export const HELD_QUIET_MS = 400;

/** How long a sentence stays in the region before it is cleared. */
export const CLEAR_AFTER_MS = 6000;

/** The sentences as they are said. */
export interface Announcer {
  /**
   * Says a sentence. A key pressed (`held` false) says it at once. A key held down (auto-repeat) says nothing at each step: the latest
   * sentence is said once, when the repeats have stopped for {@link HELD_QUIET_MS}. A null sentence says nothing.
   */
  say: (sentence: string | null, held?: boolean) => void;
  /** Empties the region, and drops a sentence that was waiting. */
  clear: () => void;
  /** Stops the timers (the screen is going). */
  dispose: () => void;
}

/** Decides when each sentence is said; `emit` puts it in the live region and `emit("")` empties it. */
export function createAnnouncer(emit: (text: string) => void, quietMs = HELD_QUIET_MS, clearMs = CLEAR_AFTER_MS): Announcer {
  let quiet: ReturnType<typeof setTimeout> | null = null;
  let clearing: ReturnType<typeof setTimeout> | null = null;
  const stop = (): void => {
    if (quiet) clearTimeout(quiet);
    quiet = null;
  };
  const speak = (sentence: string): void => {
    emit(sentence);
    if (clearing) clearTimeout(clearing);
    clearing = setTimeout(() => {
      clearing = null;
      emit("");
    }, clearMs);
  };
  return {
    say(sentence, held = false) {
      if (sentence === null) return;
      stop();
      if (!held) {
        speak(sentence);
        return;
      }
      quiet = setTimeout(() => {
        quiet = null;
        speak(sentence);
      }, quietMs);
    },
    clear() {
      stop();
      if (clearing) clearTimeout(clearing);
      clearing = null;
      emit("");
    },
    dispose() {
      stop();
      if (clearing) clearTimeout(clearing);
      clearing = null;
    },
  };
}
