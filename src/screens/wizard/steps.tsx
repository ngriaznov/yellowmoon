import { useId } from "react";
import { CheckIcon, CircleAlertIcon } from "lucide-react";
import { Card, CardImage } from "@/components/glass/card";
import { Item, ItemActions, ItemContent, ItemDescription, ItemGroup, ItemMedia, ItemTitle } from "@/components/glass/item";
import { Spinner } from "@/components/glass/spinner";
import type { Look } from "../../api/types";
import { Button, DropZone, LinkButton, ProgressBar, Segmented, Switch, Tag } from "../../components";
import type { TagVariant } from "../../components";
import { useAppState, useStore } from "../../state/context";
import { lookLabeller, looksInDisplayOrder } from "../../state/looks";
import { isReady } from "../../state/photo";
import { useLookThumbnail } from "../../state/useLookThumbnail";
import {
  anyChecking,
  copiedLabel,
  formatSize,
  joinList,
  offersDisplayChoice,
  trashCandidates,
  unmatchedPackages,
  verdictFor,
  visiblePackages,
  wantList,
} from "../../wizard/model";
import type { Received, Verdict } from "../../wizard/model";
import { InputQuestion } from "./InputQuestion";
import type { Wizard } from "./useWizard";

// ---------------------------------------------------------------------------
// 1 · Choose
// ---------------------------------------------------------------------------

/** Step 1: which of ARRI's packages to import. */
export function ChooseStep({ wizard }: { wizard: Wizard }) {
  const { state, dispatch } = wizard;
  const uid = useId();
  return (
    <div className="ym-wizard__body ym-wizard__choose">
      {visiblePackages(state).map((p) => {
        const on = state.chosen.includes(p.id);
        return (
          <Item key={p.id} asChild variant="outline">
            <button
              type="button"
              role="checkbox"
              aria-checked={on}
              aria-labelledby={`${uid}-${p.id}-name`}
              aria-describedby={`${uid}-${p.id}-desc ${uid}-${p.id}-meta`}
              className="ym-wizard__package"
              data-on={on || undefined}
              onClick={() => dispatch({ type: "toggle-package", id: p.id })}
            >
              <ItemMedia variant="icon" className="ym-wizard__box" aria-hidden="true">
                {on ? <CheckIcon /> : null}
              </ItemMedia>
              <ItemContent className="ym-wizard__package-text">
                <ItemTitle id={`${uid}-${p.id}-name`} className="ym-wizard__package-name">
                  {p.name}
                </ItemTitle>
                <ItemDescription id={`${uid}-${p.id}-desc`} className="ym-wizard__package-desc">
                  {p.description}
                </ItemDescription>
                <span id={`${uid}-${p.id}-meta`} className="ym-wizard__package-meta">
                  {p.meta}
                </span>
              </ItemContent>
              {p.recommended ? <Tag variant="recommended" size="lg" /> : null}
            </button>
          </Item>
        );
      })}
      <LinkButton className="ym-wizard__older" onClick={() => dispatch({ type: "toggle-older" })}>
        {state.showOlder ? "Hide the older LogC3 packages" : "Show 3 older LogC3 packages"}
      </LinkButton>
    </div>
  );
}

// ---------------------------------------------------------------------------
// 2 · Get the files
// ---------------------------------------------------------------------------

/** What the received list says about a file while the recogniser is still looking, and after. */
function receivedLabel(source: Received): string {
  const r = source.found?.recognition;
  if (!source.found) return "Checking…";
  if (r?.kind === "package") return r.title;
  return formatSize(source.found.size_bytes);
}

/** Step 2: open ARRI's page, watch Downloads, take a dropped or chosen zip. */
export function GetStep({ wizard }: { wizard: Wizard }) {
  const { state, watching, watchProblem, dragging } = wizard;
  const watchOn = useAppState((s) => s.preferences?.watch_downloads_during_import ?? false);
  const unmatched = unmatchedPackages(state);
  const settled = state.sources.length > 0 && !anyChecking(state);
  const waitingFor = settled && unmatched.length > 0 ? joinList(unmatched.map((p) => p.name)) : wantList(state);
  const arrived = state.sources.length >= Math.max(1, state.chosen.length);
  return (
    <div className="ym-wizard__body ym-wizard__get">
      <div className="ym-wizard__open">
        <Button size="lg" onClick={wizard.openArriPage}>
          Open ARRI&apos;s page ↗
        </Button>
        <span>Opens in your browser. Download {wantList(state)} there.</span>
      </div>
      <Item variant="outline" className="ym-wizard__watch">
        <ItemContent className="ym-wizard__watch-text">
          <ItemTitle className="ym-wizard__watch-title">Watch my Downloads folder</ItemTitle>
          <ItemDescription>Picks up the zip as soon as it lands. Only while this window is open.</ItemDescription>
        </ItemContent>
        <ItemActions>
          <Switch label="Watch my Downloads folder" checked={watchOn} onChange={wizard.setWatch} />
        </ItemActions>
      </Item>
      {watchProblem ? (
        <Item variant="muted" size="sm" role="alert" className="ym-wizard__problem">
          <ItemMedia variant="icon" aria-hidden="true">
            <CircleAlertIcon />
          </ItemMedia>
          <ItemContent>{watchProblem}</ItemContent>
        </Item>
      ) : null}
      <DropZone variant="compact" title="Or drop the zip here" hint="A zip or an unpacked folder both work." active={dragging}>
        <LinkButton onClick={() => void wizard.chooseFiles()}>Choose file…</LinkButton>
      </DropZone>
      <Item variant="muted" size="sm" className="ym-wizard__waiting" role="status" data-arrived={arrived || undefined}>
        <ItemMedia aria-hidden="true">{arrived ? <CheckIcon /> : <Spinner />}</ItemMedia>
        <ItemContent>
          {arrived
            ? "Everything has arrived. Press Next to check what each file is."
            : watching
              ? `Watching Downloads for ${waitingFor}…`
              : `Waiting for ${waitingFor}…`}
        </ItemContent>
      </Item>
      {state.sources.length > 0 ? (
        <ul className="ym-wizard__received" aria-label="Files that arrived">
          {state.sources.map((source) => (
            <Item key={source.path} asChild size="sm">
              <li>
                <span className="ym-wizard__received-name">{source.found?.name ?? source.path}</span>
                <span className="ym-wizard__received-meta">
                  {source.found && source.found.recognition.kind === "package" ? `${source.found.recognition.title} · ${formatSize(source.found.size_bytes)}` : receivedLabel(source)}
                </span>
                <LinkButton size="sm" aria-label={`Remove ${source.found?.name ?? source.path}`} onClick={() => wizard.remove(source.path)}>
                  Remove
                </LinkButton>
              </li>
            </Item>
          ))}
        </ul>
      ) : null}
    </div>
  );
}

// ---------------------------------------------------------------------------
// 3 · Recognise
// ---------------------------------------------------------------------------

interface RowView {
  what: string;
  contents: string;
  tag: TagVariant;
  tagLabel: string;
  /** Plain words under the row: what is wrong and what to do. */
  explanation: string | null;
  note: string | null;
}

/** What a verdict looks like in the table. */
export function rowView(v: Verdict): RowView {
  switch (v.kind) {
    case "checking":
      return { what: "Checking…", contents: "", tag: "waiting", tagLabel: "Checking", explanation: null, note: null };
    case "match":
      return { what: v.what, contents: v.contents, tag: "match", tagLabel: "Match", explanation: null, note: v.note };
    case "other-package":
      return { what: v.what, contents: v.contents, tag: "failed", tagLabel: "Not chosen", explanation: v.explanation, note: null };
    case "question":
      return {
        what: "Not from a package we know",
        contents: `${v.fileCount} .cube${v.lutSize ? ` · ${v.lutSize}-point` : ""}`,
        tag: v.input ? "match" : "active",
        tagLabel: v.input ? "Ready" : "Question",
        explanation: v.note,
        // An answered set stands in for the chosen package that has the same input, so that package is not missing.
        note: v.standsFor ? `Counts as ${v.standsFor}. Imported as your own LUTs, with the input you picked.` : null,
      };
    case "look-files":
      return {
        what: v.what,
        contents: v.contents,
        tag: "match",
        tagLabel: "Ready",
        explanation: null,
        // A folder or zip of the person's own look files stands in for the chosen package that has the same input.
        note: [v.standsFor ? `Counts as ${v.standsFor}.` : null, v.note].filter(Boolean).join(" ") || null,
      };
    case "unusable":
      return { what: "Can't be imported", contents: "", tag: "failed", tagLabel: "Can't use", explanation: v.explanation, note: null };
    case "missing":
      return { what: "Not found", contents: "", tag: "failed", tagLabel: "Missing", explanation: v.explanation, note: null };
  }
}

/** Step 3: each file that arrived, what it is, and a tick or an explanation. Nothing is copied yet. */
export function RecogniseStep({ wizard }: { wizard: Wizard }) {
  const { state } = wizard;
  return (
    <div className="ym-wizard__body ym-wizard__recognise">
      <div role="table" aria-label="What arrived" className="ym-wizard__table">
        <div role="row" className="ym-wizard__head">
          <span role="columnheader">File</span>
          <span role="columnheader">Recognised as</span>
          <span role="columnheader">Contents</span>
          <span role="columnheader">
            <span className="ym-sr-only">Result</span>
          </span>
        </div>
        {state.sources.map((source) => {
          const v = verdictFor(state, source);
          const view = rowView(v);
          const name = source.found?.name ?? source.path;
          const size = formatSize(source.found?.size_bytes ?? null);
          return (
            <div role="row" key={source.path} className="ym-wizard__row" data-verdict={v.kind}>
              <span role="cell" className="ym-wizard__file">
                <span>{name}</span>
                {size ? <span className="ym-wizard__size">{size}</span> : null}
              </span>
              <span role="cell">{view.what}</span>
              <span role="cell" className="ym-wizard__contents">
                {view.contents}
              </span>
              <span role="cell" className="ym-wizard__tag">
                <Tag variant={view.tag} size="lg">{view.tagLabel}</Tag>
              </span>
              {view.explanation || view.note || v.kind === "question" ? (
                <span role="cell" className="ym-wizard__why">
                  {view.explanation ? (
                    <Item variant="muted" size="sm" className="ym-wizard__explain" data-tone={v.kind === "question" ? "plain" : "error"}>
                      <ItemContent>{view.explanation}</ItemContent>
                    </Item>
                  ) : null}
                  {view.note ? <span className="ym-wizard__explain">{view.note}</span> : null}
                  {v.kind === "question" ? (
                    <InputQuestion
                      name={name}
                      value={v.input}
                      suggestion={source.found?.recognition.kind === "needs_answer" ? source.found.recognition.suggestion : null}
                      onChange={(input) => wizard.answer(source.path, input)}
                    />
                  ) : null}
                  {v.kind !== "match" && v.kind !== "checking" ? (
                    <LinkButton size="sm" className="ym-wizard__remove" aria-label={`Remove ${name}`} onClick={() => wizard.remove(source.path)}>
                      Remove
                    </LinkButton>
                  ) : null}
                </span>
              ) : null}
            </div>
          );
        })}
      </div>
      <Item variant="muted" size="sm" className="ym-wizard__callout">
        <ItemContent>
          If a package isn&apos;t the one you chose, for example LogC3 instead of LogC4, the wizard says what it is and what to download instead.
          Nothing is copied until you continue.
        </ItemContent>
      </Item>
    </div>
  );
}

// ---------------------------------------------------------------------------
// 4 · Import
// ---------------------------------------------------------------------------

/** Step 4: the count, the bar and what the import does. */
export function ImportStep({ wizard }: { wizard: Wizard }) {
  const { state } = wizard;
  const { done, total, status, error } = state.import;
  const pct = total ? Math.round((done / total) * 100) : 0;
  return (
    <div className="ym-wizard__body">
      <div className="ym-wizard__count">
        <span className="ym-wizard__count-n">{done}</span>
        <span className="ym-wizard__count-of">of {total} files</span>
      </div>
      <ProgressBar className="ym-wizard__progress" size="lg" value={pct} label="Importing looks" />
      {status === "failed" ? (
        <Item variant="muted" size="sm" role="alert" className="ym-wizard__callout ym-wizard__callout--error">
          <ItemContent>{error}</ItemContent>
          <ItemActions>
            <Button variant="secondary" onClick={wizard.retryImport}>
              Try again
            </Button>
          </ItemActions>
        </Item>
      ) : null}
      <ItemGroup className="ym-wizard__facts">
        <Item size="sm" role="listitem">
          <ItemTitle>Copied</ItemTitle>
          <ItemContent>{copiedLabel(state)}, into the app&apos;s looks folder</ItemContent>
        </Item>
        <Item size="sm" role="listitem">
          <ItemTitle>Grouped</ItemTitle>
          <ItemContent>The way ARRI groups them</ItemContent>
        </Item>
        <Item size="sm" role="listitem">
          <ItemTitle>Recorded per file</ItemTitle>
          <ItemContent>Source address, date and SHA-256 checksum</ItemContent>
        </Item>
      </ItemGroup>
    </div>
  );
}

// ---------------------------------------------------------------------------
// 5 · Finish
// ---------------------------------------------------------------------------

/** One look of the Finish step on the current photo, with its name below. */
function WizardTile({ look, photoId }: { look: Look; photoId: number | null }) {
  const store = useStore();
  const { src, onImageError } = useLookThumbnail(store, photoId, look.id, "wizard");
  return (
    <Card size="sm" className="ym-wizard__tile">
      <CardImage className="ym-wizard__tile-picture">{src ? <img src={src} alt="" onError={onImageError} /> : null}</CardImage>
      <span className="ym-wizard__tile-name">{look.name}</span>
    </Card>
  );
}

/** How many looks the tile grid shows before "And N more". */
export const TILES_SHOWN = 7;

/** Step 5: the imported looks on the current photo, the default look, the display rendering, and the zip. */
export function FinishStep({ wizard }: { wizard: Wizard }) {
  const { state, dispatch } = wizard;
  const looks = useAppState((s) => s.looks);
  const session = useAppState((s) => s.session);
  const outcome = state.outcome;

  const ordered = looksInDisplayOrder(looks ?? []);
  // Looks the packages gave; when there are none, the person's own LUTs (a set of unknown files that was answered is imported as those).
  const arriOnly = ordered.filter((l) => l.group.kind === "arri");
  const arri = arriOnly.length > 0 ? arriOnly : ordered.filter((l) => l.group.kind === "user");
  const builtIn = ordered.filter((l) => l.group.kind === "built_in");
  const current = session && session.current_index !== null ? session.photos[session.current_index] : undefined;
  const photoId = current && isReady(current) ? current.id : null;

  const shown = arri.slice(0, TILES_SHOWN);
  const more = Math.max(0, (outcome?.looks ?? 0) - shown.length);
  const currentDefault = ordered.find((l) => l.is_default)?.id ?? builtIn[0]?.id ?? "";
  const defaultOptions = [...builtIn.slice(0, 2), ...(arri[0] ? [arri[0]] : [])];
  const defaultValue = state.finish.defaultLook ?? currentDefault;
  // A name two libraries share says which one it is (the options stand without their section).
  const label = lookLabeller(looks ?? []);
  const zips = trashCandidates(state);

  return (
    <div className="ym-wizard__body ym-wizard__finish">
      {outcome && outcome.looks > 0 ? (
        <div className="ym-wizard__tiles">
          {shown.map((look) => (
            <WizardTile key={look.id} look={look} photoId={photoId} />
          ))}
          {more > 0 ? (
            <Card size="sm" className="ym-wizard__tile">
              <CardImage className="ym-wizard__tile-picture ym-wizard__tile-more">+{more}</CardImage>
              <span className="ym-wizard__tile-name">And {more} more</span>
            </Card>
          ) : null}
        </div>
      ) : null}
      <ItemGroup className="ym-wizard__options">
        {outcome && outcome.looks > 0 ? (
          <Item size="sm" role="listitem" className="ym-wizard__option">
            <ItemContent className="ym-wizard__option-text">
              <ItemTitle className="ym-wizard__option-title">Default look</ItemTitle>
              <ItemDescription>For every photo you open</ItemDescription>
            </ItemContent>
            <Segmented
              label="Default look"
              size="lg"
              value={defaultValue}
              onChange={(id) => dispatch({ type: "finish", patch: { defaultLook: id } })}
              options={defaultOptions.map((l) => ({ value: l.id, label: label(l) }))}
            />
          </Item>
        ) : null}
        {offersDisplayChoice(state) ? (
          <Item size="sm" role="listitem" className="ym-wizard__option">
            <ItemContent className="ym-wizard__option-text">
              <ItemTitle className="ym-wizard__option-title">Display rendering</ItemTitle>
              <ItemDescription>How log becomes a picture</ItemDescription>
            </ItemContent>
            <Segmented
              label="Display rendering"
              size="lg"
              value={state.finish.display}
              onChange={(display) => dispatch({ type: "finish", patch: { display } })}
              options={[
                { value: "ours", label: "Ours · ACES 2" },
                { value: "arri", label: "ARRI's" },
              ]}
            />
          </Item>
        ) : null}
        {zips.length > 0 ? (
          <Item size="sm" role="listitem" className="ym-wizard__option">
            <ItemContent className="ym-wizard__option-text">
              <ItemTitle className="ym-wizard__option-title">Downloaded zip</ItemTitle>
              <ItemDescription>No longer needed</ItemDescription>
            </ItemContent>
            <Segmented
              label="Downloaded zip"
              size="lg"
              value={state.finish.zip}
              onChange={(zip) => dispatch({ type: "finish", patch: { zip } })}
              options={[
                { value: "keep", label: "Keep it" },
                { value: "trash", label: "Move to Trash" },
              ]}
            />
          </Item>
        ) : null}
      </ItemGroup>
    </div>
  );
}
