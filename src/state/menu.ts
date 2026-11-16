/**
 * The menu bar's items that need the webview (`docs/11-ui.md` §Menus). The native menu lives in the Rust shell; it sends
 * `menu_action` with one of these names and this file does the work, with the same store calls the keyboard shortcuts and the
 * buttons use. Items the shell handles by itself (Show Looks Folder, the Help links) never arrive.
 */
import type { MenuAction, UpdateState } from "../api/types";
import { updater } from "../updater";
import type { CheckResult } from "../updater";
import type { ShortcutAction } from "./shortcuts";
import type { AppStore } from "./store";
import { zoomStep } from "./zoom";

/**
 * What "Check for Updates…" tells the person. An update that is found says what the header's update cell does next; one that is
 * already downloading or ready (the check keeps it as the answer) must not point at a button that is no longer on screen.
 */
export function checkNotice(result: CheckResult, state: UpdateState): string {
  if (result.kind === "up_to_date") return "Yellowmoon is up to date.";
  if (result.kind === "error") return "Yellowmoon couldn't check for updates. Check your internet connection and try again.";
  if (state.phase === "ready") return `Version ${result.version} is ready. Press Update ready to restart.`;
  if (state.phase === "downloading") return `Version ${result.version} is downloading.`;
  return `Version ${result.version} is available. It downloads when you press Update available.`;
}

/** The shortcut each Photo menu item stands for. */
const SINGLE_KEY_ACTIONS = {
  crop: "crop",
  auto_crop: "auto-crop",
  pick_white: "pick-white",
  remove_photo: "remove-photo",
  bypass_node: "bypass-node",
  mask_view: "mask-view",
} as const satisfies Partial<Record<MenuAction, ShortcutAction>>;

/** Runs one menu item. */
export async function runMenuAction(store: AppStore, action: MenuAction): Promise<void> {
  const state = store.getState();
  switch (action) {
    case "about":
      store.openOverlay("about");
      return;
    case "preferences":
      store.openOverlay("preferences");
      return;
    case "check_for_updates": {
      const result = await updater.checkNow();
      store.showNotice(checkNotice(result, updater.getState()));
      return;
    }
    case "install_cli":
      await store.installCli();
      return;
    case "open_files":
      await store.openDialog("files");
      return;
    case "open_folder":
      await store.openDialog("folder");
      return;
    case "export":
      if (!state.session) return;
      if (state.screen !== "photos") await store.openPhotos();
      store.openExport();
      return;
    case "close_session":
      await store.requestCloseSession();
      return;
    case "undo":
      {
        const can = state.canUndo;
        await store.undo();
        store.announce?.("undo", can);
      }
      return;
    case "redo":
      {
        const can = state.canRedo;
        await store.redo();
        store.announce?.("redo", can);
      }
      return;
    case "pick":
      await store.togglePick();
      store.announce?.("toggle-pick");
      return;
    case "pick_all":
      await store.pickAll();
      store.announce?.("pick-all");
      return;
    case "clear_picks":
      await store.clearPicks();
      store.announce?.("clear-picks");
      return;
    case "before_after":
      await store.toggleBefore();
      store.announce?.("before");
      return;
    case "nodes":
      // The Photos screen owns the node view; a dialog has the keyboard and the menu alike.
      if (state.overlay === null) store.shortcutCommand?.({ action: "nodes" });
      return;
    // The Photo menu and Apply Look 1 to 9 (`docs/28` A-P11): the actions of single keys, so a person who turned those keys off still has
    // every one. Each runs the key's own handler on the Photos screen, which has the same guards (nothing in Crop or the Export panel where the key does nothing).
    case "crop":
    case "auto_crop":
    case "pick_white":
    case "remove_photo":
    case "bypass_node":
    case "mask_view":
      if (state.overlay === null) store.shortcutCommand?.({ action: SINGLE_KEY_ACTIONS[action] });
      return;
    case "look_1":
    case "look_2":
    case "look_3":
    case "look_4":
    case "look_5":
    case "look_6":
    case "look_7":
    case "look_8":
    case "look_9":
      if (state.overlay === null) store.shortcutCommand?.({ action: "pick-look", look: Number(action.slice(5)) });
      return;
    case "fit":
      await store.setZoom({ kind: "fit" });
      store.announce?.("toggle-zoom");
      return;
    case "zoom_100":
      await store.setZoom({ kind: "percent", value: 100 });
      store.announce?.("toggle-zoom");
      return;
    case "zoom_in":
      await store.setZoom(zoomStep(state.zoom, 1, store.fitPercentExact()));
      store.announce?.("zoom-in");
      return;
    case "zoom_out":
      await store.setZoom(zoomStep(state.zoom, -1, store.fitPercentExact()));
      store.announce?.("zoom-out");
      return;
    case "next_photo":
      await store.step(1);
      store.announce?.("next-photo");
      return;
    case "previous_photo":
      await store.step(-1);
      store.announce?.("previous-photo");
      return;
    case "import_arri_looks":
      store.go("import-looks");
      return;
    case "import_cube":
      await store.startCubeImport();
      return;
    case "copy_settings":
      await store.copySettings();
      return;
    case "paste_settings":
      await store.pasteSettings("this");
      return;
    case "paste_settings_to_picked":
      await store.pasteSettings("picked");
      return;
    case "make_default_look": {
      const rs = store.currentSettings();
      if (rs) await store.makeDefault(rs.values.output.look);
      return;
    }
  }
}
