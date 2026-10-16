import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import { getApi } from "./api";
import { mockApi } from "./api.mock";
import { installDesktopFeel } from "./desktopFeel";
import { isTauri } from "./env";
import { installPerf } from "./perf";
import { installTestHooks } from "./testHooks";
import { updater } from "./updater";
import "./styles/glass.css";
import "./styles/legacy.css";
import "./styles/glass-app.css";
import "./styles/glass-brand.css";

// An app, not a page: no pinch zoom, no page menu (docs/11 §Desktop feel). Measuring is off unless the page was opened with ?perf.
installDesktopFeel();
installPerf();

// The core mirrors its own updater messages (for example from the app menu) as events; the mock
// sends them too, so a browser can show every state of the update cell.
getApi().on("update_state", (state) => updater.applyExternal(state));
// Quiet update checks run only inside the app; in a browser there is nothing to update.
if (isTauri()) updater.start();
else installTestHooks({ mockApi, updater });

const root = document.getElementById("root");
if (!root) throw new Error("index.html has no #root element");

createRoot(root).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
