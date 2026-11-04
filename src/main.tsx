import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import { getApi } from "./api";
import { mockApi } from "./api.mock";
import { installDesktopFeel } from "./desktopFeel";
import { isTauri } from "./env";
import { installPerf, perf } from "./perf";
import { installTimeline } from "./perfTimeline";
import { installTestHooks } from "./testHooks";
import { updater } from "./updater";
import "./styles/glass.css";
import "./styles/legacy.css";
import "./styles/glass-c2.css";
import "./styles/glass-c2-pages.css";

// An app, not a page: no pinch zoom, no page menu (docs/11 §Desktop feel). Measuring is off unless the page was opened with ?perf.
installDesktopFeel();
installPerf();
// With the shell started with YM_PERF the page stamps each slider tick and click and reports them once a second (`src/perfTimeline.ts`).
installTimeline(perf.enabled && isTauri(), (json) => void getApi().perfReport(json).catch(() => undefined));
// YM_GLASS=off (the shell sets `window.__YM_GLASS__`) or ?glass=off: the glass without its blur, to compare on the same machine. No change of look otherwise.
if ((globalThis as { __YM_GLASS__?: unknown }).__YM_GLASS__ === "off" || new URLSearchParams(location.search).get("glass") === "off") {
  document.documentElement.setAttribute("data-glass", "off");
}

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
