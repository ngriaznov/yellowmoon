import { useState } from "react";
import { ScreenPlaceholder } from "./screens/ScreenPlaceholder";
import { Start } from "./screens/Start";

/** The screens of the app (`docs/11-ui.md` §Screens that have a window of their own). */
export type Screen = "start" | "photos" | "import-looks";

/** Every routed screen, in navigation order. */
export const SCREENS: readonly Screen[] = ["start", "photos", "import-looks"];

interface Props {
  /** Screen to show first; Start in the app, anything in tests. */
  initialScreen?: Screen;
}

/**
 * Screen routing. The state is a single value, as in the plan: Start until files are open,
 * then Photos, with the ARRI look wizard as a full-window detour. Only the Start placeholder
 * is built so far; nothing navigates away from it yet.
 */
export default function App({ initialScreen = "start" }: Props) {
  const [screen, setScreen] = useState<Screen>(initialScreen);
  const backToStart = () => setScreen("start");

  switch (screen) {
    case "start":
      return <Start />;
    case "photos":
      return <ScreenPlaceholder screen="photos" title="Photos" task="M3-06" onBack={backToStart} />;
    case "import-looks":
      return <ScreenPlaceholder screen="import-looks" title="Import ARRI looks" task="M3-09" onBack={backToStart} />;
  }
}
