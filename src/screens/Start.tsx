import { useMemo } from "react";
import { Aurora } from "@/components/glass/aurora";
import { Item, ItemActions, ItemContent, ItemDescription, ItemGroup, ItemMedia, ItemTitle } from "@/components/glass/item";
import { Button, DevelopList, DropZone, LinkButton, ProgressBar, useWindowDrop } from "../components";
import type { DevelopItem } from "../components";
import { gradedByNote, gradingHint, gradingTitle } from "../state/autoProviders";
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
  // The provider that grades the waiting photos: the saved choice, which the app follows (a photo only waits while it acts).
  const grader = useAppState((s) => s.preferences?.auto_provider ?? "apple");
  const notice = useAppState((s) => s.dropNotice);
  const opening = useAppState((s) => s.opening);
  const { dragging } = useWindowDrop({ onDrop: (paths) => void store.openPaths(paths) });

  const busy = session !== null && session.photos.length > 0;
  const photos = session?.photos ?? [];
  const total = photos.length;
  const ready = photos.filter(isReady).length;
  const skipped = photos.filter((p) => p.status.kind === "skipped").length;
  // With a provider acting (Apple Intelligence, Claude Code or Antigravity), it grades each photo after it is developed; its values are
  // only final then (`docs/19` section 6, `docs/22` section 12.5), so the screen waits for it too. A photo counts half when it is developed and whole when it is graded.
  const grading = photos.filter((p) => p.apple_pending && p.status.kind !== "skipped").length;
  const appleActs = grading > 0;
  const developed = ready + skipped === total;
  const finished = busy && developed && !appleActs;
  const fraction = (p: (typeof photos)[number]): number => {
    const f = developFraction(p.status);
    return appleActs && p.status.kind !== "skipped" ? f / 2 + (p.apple_pending ? 0 : 0.5) : f;
  };
  const progress = total === 0 ? 0 : Math.round((photos.reduce((sum, p) => sum + fraction(p), 0) / total) * 100);

  const items: DevelopItem[] = useMemo(
    () =>
      photos.map((p) => {
        const waitsForApple = p.status.kind === "ready" && p.apple_pending;
        return {
          id: p.id,
          name: photoName(p.file_name),
          state: waitsForApple ? "developing" : p.status.kind,
          note: p.status.kind === "skipped" ? p.status.reason : waitsForApple ? gradedByNote(grader) : undefined,
        };
      }),
    [photos, grader],
  );

  const title = !busy
    ? "Drop files or a folder"
    : !developed
      ? `Developing ${total} ${plural(total, "photo", "photos")}`
      : appleActs
        ? gradingTitle(grading, grader)
        : ready === 0
          ? "No photos ready"
          : `${ready} ${plural(ready, "photo", "photos")} ready`;
  const hint = !finished
    ? developed
      ? gradingHint(grader)
      : "You can start looking while the rest finishes."
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
      <Aurora className="ym-aurora" animate={false} />
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
        <ItemGroup className="ym-about-table">
          <Item size="sm" role="listitem" className="ym-about-row">
            <ItemTitle className="ym-about-row__label">Camera</ItemTitle>
            <ItemContent>Any Sony camera · every kind of ARW</ItemContent>
          </Item>
          <Item size="sm" role="listitem" className="ym-about-row" data-action="true">
            <ItemTitle className="ym-about-row__label">Default look</ItemTitle>
            <ItemContent>{defaultLook ? label(defaultLook) : "Neutral"}</ItemContent>
            <ItemActions>
              <Menu
                label="Default look"
                options={lookOptions}
                value={defaultLook?.id ?? null}
                onSelect={(id) => void store.makeDefault(id)}
                trigger={({ ref }) => <LinkButton ref={ref}>Change</LinkButton>}
                align="end"
              />
            </ItemActions>
          </Item>
          <Item size="sm" role="listitem" className="ym-about-row" data-action={looks !== null && imported === 0 ? "true" : undefined}>
            <ItemTitle className="ym-about-row__label">ARRI looks</ItemTitle>
            <ItemContent>{looks === null ? "" : imported === 0 ? "Not imported yet" : `${imported} ${plural(imported, "look", "looks")} imported`}</ItemContent>
            {looks !== null && imported === 0 ? (
              <ItemActions>
                <LinkButton onClick={() => store.go("import-looks")}>Import →</LinkButton>
              </ItemActions>
            ) : null}
          </Item>
        </ItemGroup>
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
              <Item variant="muted" size="sm" role="alert" className="ym-develop__notice">
                {notice}
              </Item>
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
        <ItemGroup className="ym-steps__list">
          {STEPS.map((step, i) => (
            <Item key={step.n} role="listitem" variant={i === stepOn ? "muted" : "default"} className="ym-step" data-on={i === stepOn || undefined}>
              <ItemMedia className="ym-step__n">{step.n}</ItemMedia>
              <ItemContent className="ym-step__text">
                <ItemTitle className="ym-step__title">{step.title}</ItemTitle>
                <ItemDescription>{step.text}</ItemDescription>
              </ItemContent>
            </Item>
          ))}
        </ItemGroup>
      </section>
    </div>
  );
}
