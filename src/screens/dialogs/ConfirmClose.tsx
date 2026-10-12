import { Button } from "../../components";
import { useStore } from "../../state/context";
import { Dialog } from "./Dialog";

/**
 * The question ⌘W asks while an export is running (`docs/11-ui.md` §Keyboard shortcuts). Closing the session stops the
 * export; the files saved so far stay in the folder, and a half-written file is deleted.
 */
export function ConfirmClose() {
  const store = useStore();
  return (
    <Dialog
      title="An export is running"
      width={480}
      onClose={() => store.closeOverlay()}
      footer={
        <>
          <Button variant="secondary" data-autofocus="" onClick={() => store.closeOverlay()}>
            Keep exporting
          </Button>
          <Button onClick={() => void store.stopExportAndClose()}>Stop export and close</Button>
        </>
      }
    >
      <p className="ym-dialog__text">Closing the session stops it. The files that are already saved stay in the folder.</p>
    </Dialog>
  );
}
