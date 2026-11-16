import { useEffect, useMemo, useRef, useState } from "react";
import { createAnnouncer } from "./announce";
import type { Announcer } from "./announce";

/**
 * The text of a polite live region and the announcer that sets it (`docs/28` A-08). A region that gets the text it already has says
 * nothing, so every sentence ends in a no-break space on every other time it is said: the text always changes, and the space is not spoken.
 */
export function useAnnouncer(): { text: string; announcer: Announcer } {
  const [text, setText] = useState("");
  const flip = useRef(false);
  const announcer = useMemo(
    () =>
      createAnnouncer((next) => {
        if (next === "") {
          setText("");
          return;
        }
        flip.current = !flip.current;
        setText(flip.current ? `${next} ` : next);
      }),
    [],
  );
  useEffect(() => () => announcer.dispose(), [announcer]);
  return { text, announcer };
}
