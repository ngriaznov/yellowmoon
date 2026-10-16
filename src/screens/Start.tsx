import { useMemo } from "react";
import { Button, DevelopList, DropZone, LinkButton, ProgressBar, useWindowDrop } from "../components";
import type { DevelopItem } from "../components";
import { useAppState, useStore } from "../state/context";
import { lookLabeller } from "../state/looks";
import { developFraction, folderSubline, isReady, photoName } from "../state/photo";
import { ExportCell, Header } from "./Header";
import { Menu } from "./Menu";

const STEPS = [
  { n: "1", title: "Open", text: "Drop ARW files or a whole folder." },
  { n: "2", title: "Adjust, if you want", text: "Pick a look. The rest is automatic." },
  { n: "3", title: "Export", text: "JPEG, TIFF, PNG or DNG. Done." },
];

const plural = (n: number, one: string, many: string): string => (n === 1 ? one : many);

/**
 * The Start screen (`design/mockups/start.html`, `docs/11-ui.md` §Start): a yellow block that
 * says what the app does, the drop zone, and three steps. Files dropped anywhere in the window
 * open. After a drop the zone turns into the list of photos being developed, and "Open photos →"
 * works as soon as the first photo is ready.
 */
export function Start() {
  const store = useStore();
  const session = useAppState((s) => s.session);
  const looks = useAppState((s) => s.looks);
  const notice = useAppState((s) => s.dropNotice);
  const opening = useAppState((s) => s.opening);
  const { dragging } = useWindowDrop({ onDrop: (paths) => void store.openPaths(paths) });

  const busy = session !== null && session.photos.length > 0;
  const photos = session?.photos ?? [];
  const total = photos.length;
  const ready = photos.filter(isReady).length;
  const skipped = photos.filter((p) => p.status.kind === "skipped").length;
  const finished = busy && ready + skipped === total;
  const progress = total === 0 ? 0 : Math.round((photos.reduce((sum, p) => sum + developFraction(p.status), 0) / total) * 100);

  const items: DevelopItem[] = useMemo(
    () =>
      photos.map((p) => ({
        id: p.id,
        name: photoName(p.file_name),
        state: p.status.kind,
        note: p.status.kind === "skipped" ? p.status.reason : undefined,
      })),
    [photos],
  );

  const title = !busy
    ? "Drop files or a folder"
    : !finished
      ? `Developing ${total} ${plural(total, "photo", "photos")}`
      : ready === 0
        ? "No photos ready"
        : `${ready} ${plural(ready, "photo", "photos")} ready`;
  const hint = !finished
    ? "You can start looking while the rest finishes."
    : ready === 0
      ? `${skipped} ${plural(skipped, "file", "files")} skipped. Start over and open other files.`
      : skipped > 0
        ? `${skipped} ${plural(skipped, "file", "files")} skipped. Everything else is developed and ready to export.`
        : "Everything is developed and ready to export.";

  const defaultLook = looks?.find((l) => l.is_default);
  const imported = looks?.filter((l) => l.group.kind === "arri").length ?? 0;
  // A title that two libraries share says which one it is ("Film · LogC3"), since the list stands without its sections.
  const label = lookLabeller(looks ?? []);
  const lookOptions = (looks ?? []).filter((l) => !l.missing).map((l) => ({ value: l.id, label: label(l) }));
  const stepOn = busy ? 1 : 0;

  return (
    <div className="ym-screen" data-screen="start">
      <Header
        title={busy ? (session?.folder_name ?? "Yellowmoon") : "Yellowmoon"}
        subline={busy && session ? folderSubline(session) : "Drop ARW files to start"}
        export={<ExportCell label="Export" disabled title="Open photos to export" />}
      >
        <Button variant="header" shortcut="⌘O" disabled={opening} onClick={() => void store.openDialog("files")}>
          Open files
        </Button>
        <Button variant="header" shortcut="⇧⌘O" disabled={opening} onClick={() => void store.openDialog("folder")}>
          Open folder
        </Button>
      </Header>

      <section aria-label="About" className="ym-about">
        <h1>Drop Sony ARW files.</h1>
        <p>
          Every photo is developed automatically: exposure, white balance, noise and cinema colour. Adjust if you want, then export. Your
          original files are never changed.
        </p>
        <div className="ym-about-table">
          <div className="ym-about-row">
            <span className="ym-about-row__label">Camera</span>
            <span>Any Sony camera · every kind of ARW</span>
          </div>
          <div className="ym-about-row" data-action="true">
            <span className="ym-about-row__label">Default look</span>
            <span>{defaultLook ? label(defaultLook) : "Neutral"}</span>
            <Menu
              label="Default look"
              options={lookOptions}
              value={defaultLook?.id ?? null}
              onSelect={(id) => void store.makeDefault(id)}
              trigger={({ ref, ...props }) => (
                <button ref={ref} type="button" className="ym-link" {...props}>
                  Change
                </button>
              )}
              align="end"
            />
          </div>
          <div className="ym-about-row" data-action={looks !== null && imported === 0 ? "true" : undefined}>
            <span className="ym-about-row__label">ARRI looks</span>
            <span>{looks === null ? "" : imported === 0 ? "Not imported yet" : `${imported} ${plural(imported, "look", "looks")} imported`}</span>
            {looks !== null && imported === 0 ? <LinkButton onClick={() => store.go("import-looks")}>Import →</LinkButton> : null}
          </div>
        </div>
      </section>

      <main aria-label="Drop files" className="ym-drop">
        {!busy ? (
          <DropZone title="Drop files or a folder" hint="ARW only, as many as you like" notice={notice ?? undefined} active={dragging}>
            <Button disabled={opening} onClick={() => void store.openDialog("files")}>
              Open files…
            </Button>
            <Button variant="secondary" disabled={opening} onClick={() => void store.openDialog("folder")}>
              Open folder…
            </Button>
          </DropZone>
        ) : (
          <div className="ym-develop" data-active={dragging || undefined}>
            <div className="ym-develop__head">
              <h2>{title}</h2>
              <LinkButton weight="bold" onClick={() => void store.closeSession()}>
                Start over
              </LinkButton>
            </div>
            <ProgressBar className="ym-develop__progress" value={progress} label="Developing photos" />
            <DevelopList items={items} className="ym-develop__list" />
            {notice ? (
              <p role="alert" className="ym-row-note">
                {notice}
              </p>
            ) : null}
            <div className="ym-develop__foot">
              <span role="status">{hint}</span>
              <Button
                disabled={ready === 0}
                title={ready === 0 ? "Available when the first photo is ready" : undefined}
                onClick={() => void store.openPhotos()}
              >
                Open photos →
              </Button>
            </div>
          </div>
        )}
      </main>

      <section aria-label="How it works" className="ym-steps">
        {STEPS.map((step, i) => (
          <div key={step.n} className="ym-step" data-on={i === stepOn || undefined}>
            <span className="ym-step__n">{step.n}</span>
            <span className="ym-step__text">
              <span className="ym-step__title">{step.title}</span>
              <span>{step.text}</span>
            </span>
          </div>
        ))}
      </section>
    </div>
  );
}
