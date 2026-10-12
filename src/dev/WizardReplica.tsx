/**
 * Two steps of the ARRI look wizard of `design/mockups/import-looks.html` (Get the files, Import), rebuilt
 * from the components: the step rail, the yellow switch, the compact drop zone and the large progress bar.
 * `/components.html?replica=wizard&step=1` and `&step=3&imp=40`. The other steps are screen content (M3-09).
 */
import { useState } from "react";
import { Button, DropZone, LinkButton, MoonMark, ProgressBar, StepRail, Switch } from "../components";

export interface WizardReplicaProps {
  /** 0-based step index: 1 is "Get the files", 3 is "Import". */
  initialStep?: 1 | 3;
  /** Files copied so far in the import step. */
  initialImported?: number;
}

const STEPS = [
  { title: "Choose", text: "Pick the ARRI packages you want." },
  { title: "Get the files", text: "Download them from ARRI's site." },
  { title: "Recognise", text: "We check what each package is." },
  { title: "Import", text: "Copy the looks into the app." },
  { title: "Finish", text: "Pick a default and clean up." },
];
const TOTAL = 87;

/** The wizard as the mockup draws it. */
export function WizardReplica({ initialStep = 1, initialImported = 40 }: WizardReplicaProps) {
  const [step, setStep] = useState<number>(initialStep);
  const [watch, setWatch] = useState(true);
  const done = Math.min(TOTAL, initialImported);
  const importing = step === 3;
  const pct = Math.round((done / TOTAL) * 100);
  const want = "Look Library · LogC4";

  return (
    <div className="rp-screen rw-screen">
      <header className="ym-header rp-header rw-header">
        <div className="ym-brand rw-brand">
          <span className="ym-brand-title">
            <MoonMark />
            ARRI looks
          </span>
          <span className="ym-brand-sub">Import · step {step + 1} of 5</span>
        </div>
        <div className="rw-header-text">Free from ARRI. Imported to this Mac only, never shared or uploaded.</div>
        <a className="rw-cancel" href="components.html?replica=photos">
          Cancel
        </a>
      </header>

      <StepRail className="rw-rail" steps={STEPS} current={step} onSelect={setStep} />

      <main className="rw-main">
        <h1>{step === 1 ? "Download from ARRI" : importing ? (done >= TOTAL ? "Imported" : "Importing") : "Step"}</h1>
        <p className="rw-lead">
          {step === 1
            ? "Your browser opens ARRI's Look Files page. Download the package there and the app picks it up."
            : "Nothing leaves this Mac. The looks are ready the moment this finishes."}
        </p>
        {step === 1 ? (
          <div className="rw-body rw-get">
            <div className="rw-open">
              <Button size="lg">Open ARRI&apos;s page ↗</Button>
              <span>Opens in your browser. Download {want} there.</span>
            </div>
            <div className="rw-watch">
              <span className="rw-watch-text">
                <span className="rw-watch-title">Watch my Downloads folder</span>
                <span>Picks up the zip as soon as it lands. Only while this window is open.</span>
              </span>
              <Switch label="Watch my Downloads folder" checked={watch} onChange={setWatch} />
            </div>
            <DropZone variant="compact" title="Or drop the zip here" hint="A zip or an unpacked folder both work.">
              <LinkButton>Choose file…</LinkButton>
            </DropZone>
            <div className="rw-waiting">
              <span aria-hidden="true" />
              <span>{watch ? `Watching Downloads for ${want}…` : `Waiting for ${want}…`}</span>
            </div>
          </div>
        ) : (
          <div className="rw-body">
            <div className="rw-count">
              <span className="rw-count-n">{done}</span>
              <span className="rw-count-of">of {TOTAL} files</span>
            </div>
            <ProgressBar className="rw-progress" size="lg" value={pct} label="Importing looks" />
            <div className="rw-facts">
              <div>
                <b>Copied</b>
                <span>Only the .cube files, into the app&apos;s looks folder</span>
              </div>
              <div>
                <b>Grouped</b>
                <span>The way ARRI groups them</span>
              </div>
              <div>
                <b>Recorded per file</b>
                <span>Source address, date and SHA-256 checksum</span>
              </div>
            </div>
          </div>
        )}
      </main>

      <footer className="rw-footer">
        <LinkButton disabled={step === 0} onClick={() => setStep((s) => Math.max(0, s - 1))} className="rw-back">
          ← Back
        </LinkButton>
        <span className="rw-foot-note">
          {step === 1 ? "Next unlocks when the zip arrives. For this mock, it is already here." : done >= TOTAL ? "All files copied and checked" : "This takes a few seconds"}
        </span>
        <Button size="lg" disabled={importing && done < TOTAL} onClick={() => setStep((s) => Math.min(4, s + 1))}>
          Next
        </Button>
      </footer>
    </div>
  );
}
