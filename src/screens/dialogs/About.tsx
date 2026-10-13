import { useEffect, useState } from "react";
import type { AppInfo } from "../../api/types";
import { Button, LinkButton, MoonMark } from "../../components";
import { useStore } from "../../state/context";
import { ADOBE_NOTICE } from "../export/ExportOptionsForm";
import { Dialog } from "./Dialog";

/** The project's home, opened in the system browser (`docs/12`). */
export const GITHUB_URL = "https://github.com/ngriaznov/yellowmoon";

/** The acknowledgement the IJG licence of the JPEG encoder's forward DCT asks for (`docs/14`, `NOTICE`); the system's About panel says it too. */
export const IJG_NOTICE = "This software is based in part on the work of the Independent JPEG Group.";

/** Said in About (`docs/11-ui.md` §Menus: "Not affiliated with ARRI or Sony"; `NOTICE`). */
export const TRADEMARK_NOTICE =
  "Not affiliated with ARRI or Sony. ARRI and ALEXA are trademarks of ARRI AG; Sony is a trademark of Sony Group Corporation.";

/** Third-party components and their licences, as in `NOTICE`. */
export const CREDITS: ReadonlyArray<string> = [
  "ALEXA colour matrices from the Academy's aces-dev (AMPAS licence).",
  "Sony camera spectral measurements by Weta Digital, from rawtoaces-data (Apache License 2.0).",
  "ACES 2.0 output transform from aces-output (Apache License 2.0).",
  "The Figtree typeface (SIL Open Font License 1.1).",
];

/**
 * About Yellowmoon (`docs/11-ui.md` §Menus): the version, the licence, what is built on what, the Adobe DNG notice that
 * Adobe's patent licence asks for (`docs/09` §Legal), the Independent JPEG Group's acknowledgement, and the line that says the project
 * is not affiliated with ARRI or Sony.
 */
export function About() {
  const store = useStore();
  const [info, setInfo] = useState<AppInfo | null>(null);
  useEffect(() => {
    let live = true;
    store.api.appInfo().then(
      (value) => {
        if (live) setInfo(value);
      },
      (error) => {
        if (live) store.reportError(error);
      },
    );
    return () => {
      live = false;
    };
  }, [store]);

  return (
    <Dialog
      title="About Yellowmoon"
      width={600}
      onClose={() => store.closeOverlay()}
      footer={
        <>
          <LinkButton className="ym-dialog__footer-start" onClick={() => void store.api.openUrl(GITHUB_URL).catch((e) => store.reportError(e))}>
            Yellowmoon on GitHub
          </LinkButton>
          <Button data-autofocus="" onClick={() => store.closeOverlay()}>
            Done
          </Button>
        </>
      }
    >
      <div className="ym-about-dialog__brand">
        <MoonMark />
        Yellowmoon
      </div>
      <p className="ym-about-dialog__version">{info ? `Version ${info.version} · pipeline ${info.pipeline_version}` : "Version …"}</p>
      <section className="ym-about-dialog__section" aria-label="What it is">
        <p>
          A raw processor for Sony ARW files. Every photo is developed automatically, with ALEXA-style colour. Free and open
          source. Nothing is saved besides the files you export.
        </p>
      </section>
      <section className="ym-about-dialog__section" aria-label="Licence">
        <h3>Licence</h3>
        <p>Apache License 2.0. Copyright 2026 Mykyta Hriaznov and Yellowmoon contributors.</p>
      </section>
      <section className="ym-about-dialog__section" aria-label="Built with">
        <h3>Built with</h3>
        <ul>
          {CREDITS.map((credit) => (
            <li key={credit}>{credit}</li>
          ))}
        </ul>
        <p style={{ marginTop: 8 }}>The full licence texts are in the NOTICE and THIRD-PARTY.md files inside the app.</p>
      </section>
      <p className="ym-about-dialog__adobe">{ADOBE_NOTICE}</p>
      <p className="ym-about-dialog__legal">{IJG_NOTICE}</p>
      <p className="ym-about-dialog__legal">{TRADEMARK_NOTICE}</p>
    </Dialog>
  );
}
