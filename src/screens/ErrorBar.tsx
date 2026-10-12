import { LinkButton } from "../components";
import { useAppState, useStore } from "../state/context";

/**
 * Shows the last error from the core in a black bar at the bottom of the window until it is
 * dismissed. Errors already say what went wrong and what to do (`docs/01` §Errors and logging);
 * this only shows them. A message that is not an error ("Yellowmoon is up to date.") uses the same bar as a status.
 */
export function ErrorBar() {
  const store = useStore();
  const error = useAppState((s) => s.error);
  const notice = useAppState((s) => s.notice);
  if (error) {
    return (
      <div role="alert" className="ym-error-bar">
        <span>{error.message}</span>
        <LinkButton onClick={() => store.dismissError()}>Dismiss</LinkButton>
      </div>
    );
  }
  if (notice) {
    return (
      <div role="status" className="ym-error-bar" data-kind="notice">
        <span>{notice}</span>
        <LinkButton onClick={() => store.dismissNotice()}>Dismiss</LinkButton>
      </div>
    );
  }
  return null;
}
