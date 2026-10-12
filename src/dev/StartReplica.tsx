/**
 * The Start screen of `design/mockups/start.html`, rebuilt from the components (see
 * `PhotosReplica`). `/components.html?replica=start` (idle), `&prog=5.4` (developing), `&prog=12` (done).
 */
import { useState } from "react";
import { Button, DevelopList, DropZone, KeyHint, LinkButton, MoonMark, ProgressBar, UpdatePill } from "../components";
import type { DevelopItem } from "../components";
import { SCENES } from "../fixtures/scenes";

export interface StartReplicaProps {
  /** Photos developed so far (fractions are the one in progress); undefined shows the empty drop zone. */
  initialProgress?: number;
  /** Photo index (0-based) to mark skipped, with a reason. */
  skippedIndex?: number;
}

const STEPS = [
  { n: "1", title: "Open", text: "Drop ARW files or a whole folder." },
  { n: "2", title: "Adjust, if you want", text: "Pick a look. The rest is automatic." },
  { n: "3", title: "Export", text: "JPEG, TIFF, PNG or DNG. Done." },
];

/** The Start screen as the mockup draws it. */
export function StartReplica({ initialProgress, skippedIndex }: StartReplicaProps) {
  const [prog, setProg] = useState<number | undefined>(initialProgress);
  const busy = prog !== undefined;
  const n = SCENES.length;
  const p = prog ?? 0;
  const doneCount = Math.min(n, Math.floor(p));
  const skipped = (i: number): boolean => i === skippedIndex;
  const items: DevelopItem[] = SCENES.map((scene, i) => {
    if (i < doneCount) {
      return skipped(i)
        ? { id: scene.name, name: scene.name, state: "skipped", note: "This file is from a camera mode we can't read yet." }
        : { id: scene.name, name: scene.name, state: "ready" };
    }
    if (i === doneCount && p < n) return { id: scene.name, name: scene.name, state: "developing" };
    return { id: scene.name, name: scene.name, state: "waiting" };
  });
  const finished = busy && p >= n;
  const skippedCount = SCENES.filter((_, i) => i < doneCount && skipped(i)).length;
  const ready = doneCount - skippedCount;
  const pct = Math.round((Math.min(p, n) / n) * 100);
  const hint = finished
    ? skippedCount
      ? `${skippedCount} file skipped. Everything else is developed and ready to export.`
      : "Everything is developed and ready to export."
    : "You can start looking while the rest finishes.";

  return (
    <div className="rp-screen rs-screen">
      <header className="ym-header rp-header" style={{ gridColumn: "1 / -1", gridRow: 1 }}>
        <div className="ym-brand">
          <span className="ym-brand-title">
            <MoonMark />
            {busy ? "Lisbon" : "Yellowmoon"}
          </span>
          <span className="ym-brand-sub">{busy ? `2026-10-04 · ${n} photos` : "Drop ARW files to start"}</span>
        </div>
        <nav aria-label="Open" className="rp-nav">
          <Button variant="header" shortcut="⌘O" onClick={() => setProg(0)}>
            Open files
          </Button>
          <Button variant="header" shortcut="⇧⌘O" onClick={() => setProg(0)}>
            Open folder
          </Button>
        </nav>
        <div className="rp-header-right">
          <UpdatePill state={{ phase: "ready", version: "0.1.1" }} />
          <span className="rp-export" aria-disabled="true">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="square" aria-hidden="true">
              <path d="M12 3v12" />
              <path d="M6.5 9.5L12 15l5.5-5.5" />
              <path d="M4 20h16" />
            </svg>
            Export
            <KeyHint>⌘E</KeyHint>
          </span>
        </div>
      </header>

      <section aria-label="About" className="rs-about">
        <h1>Drop Sony ARW files.</h1>
        <p>
          Every photo is developed automatically: exposure, white balance, noise and cinema colour. Adjust if you want, then export. Your original files are never changed.
        </p>
        <div className="rs-table">
          <div className="rs-table-row">
            <span className="rs-table-label">Camera</span>
            <span>Any Sony camera · every kind of ARW</span>
          </div>
          <div className="rs-table-row" data-action="true">
            <span className="rs-table-label">Default look</span>
            <span>Neutral</span>
            <LinkButton>Change</LinkButton>
          </div>
          <div className="rs-table-row" data-action="true">
            <span className="rs-table-label">ARRI looks</span>
            <span>Not imported yet</span>
            <LinkButton>Import →</LinkButton>
          </div>
        </div>
      </section>

      <section aria-label="Drop files" className="rs-drop">
        {!busy ? (
          <DropZone title="Drop files or a folder" hint="ARW only, as many as you like">
            <Button onClick={() => setProg(0)}>Open files…</Button>
            <Button variant="secondary" onClick={() => setProg(0)}>
              Open folder…
            </Button>
          </DropZone>
        ) : (
          <div className="rs-busy">
            <div className="rs-busy-head">
              <h2>{finished ? `${ready} photos ready` : `Developing ${n} photos`}</h2>
              <LinkButton weight="bold" onClick={() => setProg(undefined)}>
                Start over
              </LinkButton>
            </div>
            <ProgressBar className="rs-progress" value={pct} label="Developing photos" />
            <DevelopList items={items} className="rs-list" />
            <div className="rs-busy-foot">
              <span>{hint}</span>
              <Button>Open photos →</Button>
            </div>
          </div>
        )}
      </section>

      <section aria-label="How it works" className="rs-steps">
        {STEPS.map((step, i) => {
          const on = (!busy && i === 0) || (busy && i === 1);
          return (
            <div key={step.n} className="rs-step" data-on={on || undefined}>
              <span className="rs-step-n">{step.n}</span>
              <span className="rs-step-text">
                <span className="rs-step-title">{step.title}</span>
                <span>{step.text}</span>
              </span>
            </div>
          );
        })}
      </section>
    </div>
  );
}
