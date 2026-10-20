import { AlertCircleIcon, InfoIcon } from "lucide-react";
import { Item, ItemActions, ItemContent, ItemMedia } from "@/components/glass/item";
import { Button, LinkButton } from "../components";
import { useAppState, useStore } from "../state/context";

/**
 * Shows the last error from the core in a bar at the bottom of the window until it is dismissed: a glasscn `Item` on glass, with
 * Dismiss as a button. Errors already say what went wrong and what to do (`docs/01` §Errors and logging); this only shows them.
 * A message that is not an error ("Yellowmoon is up to date.") uses the same bar as a status. The notice of a removal from the filmstrip
 * ("Removed 3 photos. The files are still on disk.") has an Undo button next to Dismiss.
 */
export function ErrorBar() {
  const store = useStore();
  const error = useAppState((s) => s.error);
  const notice = useAppState((s) => s.notice);
  const action = useAppState((s) => s.noticeAction);
  if (error) {
    return (
      <Item role="alert" size="sm" className="ym-error-bar">
        <ItemMedia variant="icon" aria-hidden="true">
          <AlertCircleIcon />
        </ItemMedia>
        <ItemContent>
          <span>{error.message}</span>
        </ItemContent>
        <ItemActions>
          <LinkButton onClick={() => store.dismissError()}>Dismiss</LinkButton>
        </ItemActions>
      </Item>
    );
  }
  if (notice) {
    return (
      <Item role="status" size="sm" className="ym-error-bar" data-kind="notice">
        <ItemMedia variant="icon" aria-hidden="true">
          <InfoIcon />
        </ItemMedia>
        <ItemContent>
          <span>{notice}</span>
        </ItemContent>
        <ItemActions>
          {action === "undo-removal" ? (
            <Button variant="secondary" size="sm" onClick={() => void store.undoRemoval()}>
              Undo
            </Button>
          ) : null}
          <LinkButton onClick={() => store.dismissNotice()}>Dismiss</LinkButton>
        </ItemActions>
      </Item>
    );
  }
  return null;
}
