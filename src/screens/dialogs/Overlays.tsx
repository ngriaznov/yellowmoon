import { useAppState } from "../../state/context";
import { About } from "./About";
import { AutoConsent } from "./AutoConsent";
import { ConfirmClose } from "./ConfirmClose";
import { CubeImport } from "./CubeImport";
import { Preferences } from "./Preferences";

/** The dialog laid over the screen, if any (`AppState.overlay`). */
export function Overlays() {
  const overlay = useAppState((s) => s.overlay);
  switch (overlay) {
    case "preferences":
      // The consent dialog (`docs/22` section 12.3) opens over Preferences, so Esc closes only it.
      return (
        <>
          <Preferences />
          <AutoConsent />
        </>
      );
    case "about":
      return <About />;
    case "confirm-close":
      return <ConfirmClose />;
    case "import-cube":
      return <CubeImport />;
    default:
      return null;
  }
}
