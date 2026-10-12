import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import { getApi } from "./api";
import { isTauri } from "./env";
import { updater } from "./updater";
import "./styles/tokens.css";
import "./styles/global.css";
import "./styles/app.css";

// Quiet update checks run only inside the app; in a browser there is nothing to update.
if (isTauri()) {
  const api = getApi();
  updater.start();
  // The core mirrors its own updater messages (for example from the app menu) as events.
  api.on("update_state", (state) => updater.applyExternal(state));
}

const root = document.getElementById("root");
if (!root) throw new Error("index.html has no #root element");

createRoot(root).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
