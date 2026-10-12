import { useAppState } from "../../state/context";
import { About } from "./About";
import { ConfirmClose } from "./ConfirmClose";
import { CubeImport } from "./CubeImport";
import { Preferences } from "./Preferences";

/** The dialog laid over the screen, if any (`AppState.overlay`). */
export function Overlays() {
  const overlay = useAppState((s) => s.overlay);
  switch (overlay) {
    case "preferences":
      return <Preferences />;
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
