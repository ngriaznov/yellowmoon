import { useEffect, useMemo } from "react";
import { getApi } from "./api";
import type { Api } from "./api";
import { Overlays } from "./screens/dialogs/Overlays";
import { ErrorBar } from "./screens/ErrorBar";
import { ImportLooks } from "./screens/ImportLooks";
import { Photos } from "./screens/Photos";
import { Start } from "./screens/Start";
import { StoreProvider, useAppState, useStore } from "./state/context";
import { runMenuAction } from "./state/menu";
import { useShortcuts } from "./state/useShortcuts";
import { createStore } from "./state/store";
import type { AppStore, Screen } from "./state/store";

export type { Screen };

/** Every routed screen, in navigation order. */
export const SCREENS: readonly Screen[] = ["start", "photos", "import-looks"];

interface Props {
  /** The core's API. The app uses `getApi()`; tests pass a mock of their own. */
  api?: Api;
  /** A store to use instead of making one for `api`. */
  store?: AppStore;
}

/** Shortcuts that work on every screen. The Photos screen adds its own. */
function GlobalShortcuts() {
  const store = useStore();
  const screen = useAppState((s) => s.screen);
  const overlay = useAppState((s) => s.overlay);
  // Preferences opens from every screen, also over a dialog.
  useShortcuts((shortcut) => {
    if (shortcut.action !== "preferences") return false;
    store.openOverlay("preferences");
    return true;
  });
  useShortcuts(
    (shortcut) => {
      switch (shortcut.action) {
        case "open-files":
          void store.openDialog("files");
          return true;
        case "open-folder":
          void store.openDialog("folder");
          return true;
        case "close-session":
          if (!store.getState().session) return false;
          void store.requestCloseSession();
          return true;
        case "undo":
          // After the last photo was removed the Start screen shows, and ⌘Z puts the photos back.
          if (!store.hasRemovalToUndo()) return false;
          void store.undo();
          return true;
        default:
          return false;
      }
    },
    {},
    // The Photos screen handles these too, with the rest of its shortcuts; a dialog takes the keyboard.
    screen !== "photos" && overlay === null,
  );
  return null;
}

/** Hands the menu bar's items to the store's handler for as long as the app is on screen. */
function MenuBridge() {
  const store = useStore();
  useEffect(() => {
    store.menuHandler = (action) => void runMenuAction(store, action);
    return () => {
      store.menuHandler = null;
    };
  }, [store]);
  return null;
}

function Screens() {
  const screen = useAppState((s) => s.screen);
  const hasSession = useAppState((s) => s.session !== null);
  switch (screen) {
    case "photos":
      return hasSession ? <Photos /> : <Start />;
    case "import-looks":
      return <ImportLooks />;
    default:
      return <Start />;
  }
}

/**
 * Screen routing. State lives in the store: Start until files are open, then Photos, with the ARRI
 * look wizard as a full-window detour.
 */
export default function App({ api, store: given }: Props) {
  const store = useMemo(() => given ?? createStore(api ?? getApi()), [given, api]);
  return (
    <StoreProvider store={store}>
      <GlobalShortcuts />
      <MenuBridge />
      <Screens />
      <Overlays />
      <ErrorBar />
    </StoreProvider>
  );
}
