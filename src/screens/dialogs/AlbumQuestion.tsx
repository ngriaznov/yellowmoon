import { Button } from "../../components";
import { useAppState, useStore } from "../../state/context";
import { albumQuestion } from "../../state/presets";
import { Dialog } from "./Dialog";

/**
 * "Save changes to the album Lisbon?" (`docs/31` section 6): Save (the default), Don't Save and Cancel. Asked before closing a session whose
 * album has changes, and before another album replaces it; Save writes the album and goes on, and a save that did not happen stays.
 */
export function AlbumQuestion() {
  const store = useStore();
  const name = useAppState((s) => s.session?.album?.name ?? "");
  return (
    <Dialog
      title={albumQuestion(name)}
      blurb="Your changes will be lost if you don't save them."
      width={480}
      onClose={() => void store.answerAlbumQuestion("cancel")}
      footer={
        <>
          <Button variant="secondary" onClick={() => void store.answerAlbumQuestion("cancel")}>
            Cancel
          </Button>
          <Button variant="secondary" onClick={() => void store.answerAlbumQuestion("discard")}>
            Don&apos;t Save
          </Button>
          <Button data-autofocus="" onClick={() => void store.answerAlbumQuestion("save")}>
            Save
          </Button>
        </>
      }
    >
      <span hidden />
    </Dialog>
  );
}
