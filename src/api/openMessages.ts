/**
 * How the core's answer to `open_paths` and `open_dialog` begins when a folder could not be read (`docs/26` H-02): "Yellowmoon can't
 * read the folder Desktop. Check that it is still there and that you may open it, then try again." The sentence is the whole message
 * of an error of scope `open`; Start shows it in the drop zone, Photos in the notice bar, where "No ARW files in what you dropped."
 * goes.
 */
export const UNREADABLE_PREFIX = "Yellowmoon can't read the folder ";

/** True for the core's sentence about a folder it could not read. */
export function isUnreadableMessage(message: string): boolean {
  return message.startsWith(UNREADABLE_PREFIX);
}
