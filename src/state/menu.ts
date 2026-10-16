/**
 * The menu bar's items that need the webview (`docs/11-ui.md` §Menus). The native menu lives in the Rust shell; it sends
 * `menu_action` with one of these names and this file does the work, with the same store calls the keyboard shortcuts and the
 * buttons use. Items the shell handles by itself (Show Looks Folder, the Help links) never arrive.
 */
import type { MenuAction, UpdateState } from "../api/types";
import { updater } from "../updater";
import type { CheckResult } from "../updater";
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
      await store.undo();
      return;
    case "redo":
      await store.redo();
      return;
    case "pick":
      await store.togglePick();
      return;
    case "pick_all":
      await store.pickAll();
      return;
    case "clear_picks":
      await store.clearPicks();
      return;
    case "before_after":
      await store.toggleBefore();
      return;
    case "fit":
      await store.setZoom({ kind: "fit" });
      return;
    case "zoom_100":
      await store.setZoom({ kind: "percent", value: 100 });
      return;
    case "zoom_in":
      await store.setZoom(zoomStep(state.zoom, 1, store.fitPercentExact()));
      return;
    case "zoom_out":
      await store.setZoom(zoomStep(state.zoom, -1, store.fitPercentExact()));
      return;
    case "next_photo":
      await store.step(1);
      return;
    case "previous_photo":
      await store.step(-1);
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
      if (rs) await store.makeDefault(rs.values.look);
      return;
    }
  }
}
