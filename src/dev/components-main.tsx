/**
 * Entry of `components.html`, the Storybook-free dev page (docs/15 M3-01): every component in every
 * state, run against nothing but fixtures. `?replica=photos` (also `start`, `export`) shows a screen of
 * the mockups rebuilt from the components, which `e2e/components.spec.ts` compares with the mockups.
 */
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import type { CurveChannel, TabId } from "../components";
import { TAB_IDS } from "../components";
import "../styles/tokens.css";
import "../styles/global.css";
import "../styles/components.css";
import "../styles/app.css";
import "./dev.css";
import { ComponentsPage } from "./ComponentsPage";
import { ExportReplica } from "./ExportReplica";
import { PhotosReplica } from "./PhotosReplica";
import { StartReplica } from "./StartReplica";
import { WizardReplica } from "./WizardReplica";

const params = new URLSearchParams(window.location.search);
const replica = params.get("replica");
const tabParam = params.get("tab");
const tab: TabId = (TAB_IDS as readonly string[]).includes(tabParam ?? "") ? (tabParam as TabId) : "look";
const channelParam = params.get("curve");
const channel: CurveChannel = channelParam === "red" || channelParam === "green" || channelParam === "blue" ? channelParam : "luma";

const root = document.getElementById("root");
if (!root) throw new Error("components.html has no #root element");

createRoot(root).render(
  <StrictMode>
    {replica === "photos" ? (
      <PhotosReplica
        initialTab={tab}
        initialBefore={params.get("before") === "1"}
        initialQuery={params.get("q") ?? ""}
        initialLook={params.get("look") ?? "neutral"}
        initialCurveChannel={channel}
      />
    ) : replica === "export" ? (
      <ExportReplica
        initialFormat={(["jpeg", "tiff", "png", "dng"] as const).find((f) => f === params.get("fmt")) ?? "jpeg"}
        initialProgress={params.has("running") ? Number(params.get("running")) : undefined}
        initialColorSpace={(["srgb", "p3", "rec2020"] as const).find((f) => f === params.get("cs")) ?? "srgb"}
        initialSize={params.get("size") === "edge" ? "edge" : "full"}
      />
    ) : replica === "start" ? (
      <StartReplica
        initialProgress={params.has("prog") ? Number(params.get("prog")) : undefined}
        skippedIndex={params.has("skip") ? Number(params.get("skip")) : undefined}
      />
    ) : replica === "wizard" ? (
      <WizardReplica initialStep={params.get("step") === "3" ? 3 : 1} initialImported={params.has("imp") ? Number(params.get("imp")) : 40} />
    ) : (
      <ComponentsPage />
    )}
  </StrictMode>,
);
