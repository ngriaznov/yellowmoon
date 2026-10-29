import { useEffect, useId, useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";
import { Badge } from "@/components/glass/badge";
import { Item, ItemActions, ItemContent, ItemDescription, ItemTitle } from "@/components/glass/item";
import { Label } from "@/components/glass/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/glass/select";
import { Textarea } from "@/components/glass/textarea";
import type { AutoProvider, AutoStatus, ProviderStatus } from "../../api/types";
import { Button, LinkButton, RowNote, Segmented, Switch } from "../../components";
import { exportModel } from "../../export/model";
import {
  AUTO_GRADING_TITLE,
  autoRowShown,
  dropdownValue,
  EXPERIMENTAL_LABEL,
  EXPERIMENTAL_NOTE,
  INSTRUCTIONS_COPY,
  PROVIDER_HINTS,
  PROVIDER_LIST_NAMES,
  PROVIDER_NAMES,
  providerNote,
  savedChoiceNote,
  unavailableTag,
  USE_DEFAULT_AUTO,
} from "../../state/autoProviders";
import { useAppState, useStore } from "../../state/context";
import type { AppStore } from "../../state/store";
import { lookLabeller } from "../../state/looks";
import { preferredDisplayLut } from "../../wizard/model";
import { ExportOptionsForm } from "../export/ExportOptionsForm";
import { Dialog } from "./Dialog";

/** The label of the instructions field while Apple Intelligence grades (`docs/22` section 12.4; the others are in `INSTRUCTIONS_COPY`). */
export const APPLE_INSTRUCTIONS_LABEL = INSTRUCTIONS_COPY.apple.label;

/** Its hint. */
export const APPLE_INSTRUCTIONS_HINT = INSTRUCTIONS_COPY.apple.hint;

/** The button that puts the default instructions back (`docs/19` section 2). */
export const APPLE_INSTRUCTIONS_RESET = "Reset to default";

/** How long typing must rest before the instructions are saved, in milliseconds: not one save per keystroke. */
export const APPLE_INSTRUCTIONS_SAVE_MS = 800;

/** The length of a text as the core counts it: characters (code points), not UTF-16 units. */
const charCount = (text: string): number => [...text].length;

/**
 * The instructions the grading model reads before each photo (`docs/19` section 2, `docs/22` section 12.4), under the Auto grading row
 * while a provider other than Default Auto grades: a glasscn text area holding the instructions in use (the default when none
 * are saved), a hint, a count against the limit and a Reset to default button, disabled while the text is the default. The text is one
 * for every provider; its label and hint name the one that grades (`provider`). It is saved when the field loses focus or when typing
 * has rested for {@link APPLE_INSTRUCTIONS_SAVE_MS}, never per keystroke; the default text is saved as `""` (the default), and text over
 * the limit is not saved. A save that changes them asks every photo again (the core does it); Reset to default saves at once.
 */
export function AppleInstructions({
  store,
  saved,
  status,
  provider = "apple",
}: {
  store: AppStore;
  saved: string;
  status: AutoStatus;
  provider?: Exclude<AutoProvider, "yellowmoon">;
}) {
  const copy = INSTRUCTIONS_COPY[provider];
  const defaultText = status.default_instructions;
  const max = status.instructions_max_chars;
  const inUse = saved.trim() === "" ? defaultText : saved;
  const [text, setText] = useState(inUse);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const id = useId();

  // Follows a change made elsewhere (Reset, another window) while the field is not being typed in.
  useEffect(() => {
    if (timer.current === null) setText(inUse);
  }, [inUse]);
  useEffect(
    () => () => {
      if (timer.current !== null) clearTimeout(timer.current);
    },
    [],
  );

  const save = (value: string): void => {
    if (timer.current !== null) {
      clearTimeout(timer.current);
      timer.current = null;
    }
    if (charCount(value) > max) return;
    const own = value.trim() === "" || value === defaultText ? "" : value;
    if (own !== saved) void store.setPreferences({ apple_auto_instructions: own });
  };

  const count = charCount(text);
  const isDefault = text === defaultText;
  return (
    <div className="ym-pref__instructions" role="group" aria-labelledby={`${id}-label`}>
      <Label id={`${id}-label`} htmlFor={`${id}-text`} className="ym-pref__title">
        {copy.label}
      </Label>
      <Textarea
        id={`${id}-text`}
        value={text}
        maxLength={max}
        maxRows={12}
        spellCheck
        aria-describedby={`${id}-hint ${id}-count`}
        aria-invalid={count > max || undefined}
        onChange={(event) => {
          const value = event.target.value;
          setText(value);
          if (timer.current !== null) clearTimeout(timer.current);
          timer.current = setTimeout(() => save(value), APPLE_INSTRUCTIONS_SAVE_MS);
        }}
        onBlur={() => save(text)}
      />
      <div className="ym-pref__instructions-foot">
        <p id={`${id}-hint`} className="ym-pref__hint">
          {copy.hint}{" "}
          <span id={`${id}-count`} className="ym-pref__count">
            {`${count} of ${max} characters.`}
          </span>
        </p>
        <Button
          variant="secondary"
          size="sm"
          disabled={isDefault}
          title={isDefault ? "These are the default instructions." : undefined}
          onClick={() => {
            if (timer.current !== null) {
              clearTimeout(timer.current);
              timer.current = null;
            }
            setText(defaultText);
            void store.setPreferences({ apple_auto_instructions: "" });
          }}
        >
          {APPLE_INSTRUCTIONS_RESET}
        </Button>
      </div>
    </div>
  );
}

interface RowProps {
  title: string;
  hint?: ReactNode;
  /** The hint is a paragraph: shown whole, where a hint is otherwise cut after two lines. */
  longHint?: boolean;
  /** Under the hint, in the same column: a note that belongs to the row (Antigravity's Experimental note). */
  note?: ReactNode;
  /** The control sits at the top, level with the title, instead of in the middle of a tall text column. */
  top?: boolean;
  children: ReactNode;
}

/** One row of Preferences: a title and a line of help at the left, the control at the right. */
function Row({ title, hint, longHint = false, note, top = false, children }: RowProps) {
  return (
    <Item size="sm" role="group" aria-label={title} className={top ? "ym-pref ym-pref--top" : "ym-pref"}>
      <ItemContent className="ym-pref__text">
        <ItemTitle className="ym-pref__title" aria-hidden="true">
          {title}
        </ItemTitle>
        {hint ? <ItemDescription className={longHint ? "ym-pref__hint line-clamp-none" : "ym-pref__hint"}>{hint}</ItemDescription> : null}
        {note}
      </ItemContent>
      <ItemActions className="ym-pref__control">{children}</ItemActions>
    </Item>
  );
}

/**
 * What a screen reader says for an entry of the dropdown: the name, "experimental" for Antigravity and, for a program that is listed but
 * cannot be chosen, why ("Claude Code, not signed in", `docs/22` section 12.9). The visible tag and badge say the same in short.
 */
function entryLabel(status: ProviderStatus): string {
  const tag = status.selectable ? null : unavailableTag(status.state);
  return [PROVIDER_LIST_NAMES[status.provider], status.experimental ? "experimental" : null, tag ? tag.toLowerCase() : null]
    .filter((part) => part !== null)
    .join(", ");
}

/** An entry of the dropdown: the name, Antigravity's Experimental badge, and the muted tag of a program that cannot be chosen (`docs/22` section 12.1). */
function EntryName({ status }: { status: ProviderStatus }) {
  const tag = status.selectable ? null : unavailableTag(status.state);
  return (
    <>
      {PROVIDER_LIST_NAMES[status.provider]}
      {status.experimental ? (
        <Badge variant="outline" className="ym-pref__badge">
          {EXPERIMENTAL_LABEL}
        </Badge>
      ) : null}
      {tag ? (
        <Badge variant="secondary" className="ym-pref__tag">
          {tag}
        </Badge>
      ) : null}
    </>
  );
}

/**
 * The "Auto grading" row and what belongs to it (`docs/22` section 12): a glasscn `Select` of Default Auto, Apple Intelligence
 * (while available), Claude Code and Antigravity (while listed; a program that is installed but cannot be used is listed disabled, with
 * a tag), the hint of the provider that grades, Antigravity's Experimental note, a note under the row for each program that cannot be
 * used and for a saved choice that is not available (with a "Use Default Auto" button that withdraws it), and the instructions. The
 * select shows who grades now (`effective`), or the saved program itself while that is listed but cannot be chosen, so that choosing
 * Default Auto is a real change (`dropdownValue`); choosing a program the person has not agreed to opens the consent dialog and changes
 * nothing until they say yes (`AppStore.chooseProvider`).
 */
function AutoGrading({ store, auto, saved, os }: { store: AppStore; auto: AutoStatus; saved: string; os: string | null }) {
  // When the consent dialog closes the focus goes back to the dropdown that opened it (Radix cannot: the list had already closed when
  // the dialog mounted, so there was nothing to remember).
  const trigger = useRef<HTMLButtonElement>(null);
  const consentFor = useAppState((s) => s.consentFor);
  const asked = useRef(false);
  useEffect(() => {
    if (consentFor !== null) asked.current = true;
    else if (asked.current) {
      asked.current = false;
      trigger.current?.focus({ preventScroll: true });
    }
  }, [consentFor]);
  const entries = auto.providers.filter((p) => p.provider === "yellowmoon" || p.listed);
  const notes = auto.providers.map((p) => providerNote(p, os ?? "")).filter((note): note is string => note !== null);
  const saveNote = savedChoiceNote(auto.preference, auto.effective);
  const experimental = auto.effective === "agy";
  // The dropdown's value: the saved program itself while it is listed but not usable, else who grades. In the trigger that entry says why
  // it is not usable (its tag) in place of the Experimental badge, which would not fit beside it in 220 px.
  const shown = dropdownValue(auto);
  const shownStatus = auto.providers.find((p) => p.provider === shown);
  const shownTag = shownStatus !== undefined && !shownStatus.selectable && shown !== "yellowmoon" ? unavailableTag(shownStatus.state) : null;
  const shownBadge = shown === "agy" && shownTag === null;
  return (
    <>
      <Row
        title={AUTO_GRADING_TITLE}
        hint={PROVIDER_HINTS[auto.effective]}
        longHint
        top
        note={
          experimental ? (
            <div className="ym-pref__experimental">
              <Badge variant="outline">{EXPERIMENTAL_LABEL}</Badge>
              <RowNote className="ym-pref__note">{EXPERIMENTAL_NOTE}</RowNote>
            </div>
          ) : null
        }
      >
        <Select value={shown} onValueChange={(next) => void store.chooseProvider(next as AutoProvider)}>
          <SelectTrigger ref={trigger} aria-label={AUTO_GRADING_TITLE} className="w-[220px]">
            {/* Antigravity's name is short here ("Antigravity (agy)" is in the list) so that its badge fits the 220 px of the trigger. */}
            <SelectValue>
              {PROVIDER_NAMES[shown]}
              {shownBadge ? (
                <>
                  {" "}
                  <Badge variant="outline" className="ym-pref__badge h-4 px-1.5 text-[0.625rem]">
                    {EXPERIMENTAL_LABEL}
                  </Badge>
                </>
              ) : null}
              {shownTag !== null ? (
                <>
                  {" "}
                  <Badge variant="secondary" className="ym-pref__tag ym-pref__tag--trigger text-[0.625rem]">
                    {shownTag}
                  </Badge>
                </>
              ) : null}
            </SelectValue>
          </SelectTrigger>
          <SelectContent>
            {entries.map((entry) => (
              // Radix names an option by its text; `aria-labelledby` is left out so that the label above says why an entry is disabled.
              <SelectItem
                key={entry.provider}
                value={entry.provider}
                disabled={entry.provider !== "yellowmoon" && !entry.selectable}
                aria-label={entryLabel(entry)}
                aria-labelledby={undefined}
              >
                <EntryName status={entry} />
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </Row>
      {notes.length > 0 || saveNote !== null ? (
        <div className="ym-pref__notes">
          {notes.map((note) => (
            <RowNote key={note} role="note" className="ym-pref__note">
              {note}
            </RowNote>
          ))}
          {saveNote !== null ? (
            <div className="ym-pref__saved">
              <RowNote role="note" className="ym-pref__note">
                {saveNote}
              </RowNote>
              {/* Choosing Default Auto in the dropdown is no change while it is the value shown, and a saved program that is not listed
                  cannot be shown there at all: this always withdraws the saved choice. No consent is involved. */}
              <LinkButton
                size="sm"
                onClick={() => {
                  void store.chooseProvider("yellowmoon");
                  // The button goes with the note: the focus stays in the row (the dropdown), unless the row goes too.
                  trigger.current?.focus({ preventScroll: true });
                }}
              >
                {USE_DEFAULT_AUTO}
              </LinkButton>
            </div>
          ) : null}
        </div>
      ) : null}
      {auto.effective !== "yellowmoon" ? <AppleInstructions store={store} saved={saved} status={auto} provider={auto.effective} /> : null}
    </>
  );
}

/**
 * Preferences (⌘,, `docs/11-ui.md` §Preferences): a small window in the same style, one column of rows. Everything here is
 * a preference of `docs/01` §Persistence and is saved as soon as it changes; nothing needs a Save button. "Auto grading" is there only
 * while Apple Intelligence is available or Claude Code or Antigravity is installed, which is asked each time the window opens
 * (`docs/22` sections 3 and 12): absent everywhere else, the browser build included. The consent dialog (`AutoConsent`) opens over it.
 */
export function Preferences() {
  const store = useStore();
  const prefs = useAppState((s) => s.preferences);
  const looks = useAppState((s) => s.looks);
  const luts = useAppState((s) => s.displayLuts);
  const cli = useAppState((s) => s.cliLink);
  const auto = useAppState((s) => s.auto);
  const os = useAppState((s) => s.os);
  const exportOptions = useAppState((s) => s.exportOptions);
  const session = useAppState((s) => s.session);
  const settings = useAppState((s) => s.settings);
  const [editing, setEditing] = useState(false);

  const model = useMemo(
    () => exportModel({ session, settings, looks, preferences: prefs }, "this", exportOptions),
    [session, settings, looks, prefs, exportOptions],
  );

  const defaultLook = looks?.find((l) => l.is_default);
  // A title that two libraries share says which one it is ("Film · LogC3"), since the list stands without its sections.
  const label = lookLabeller(looks ?? []);
  const lookOptions = (looks ?? []).filter((l) => !l.missing).map((l) => ({ value: l.id, label: label(l) }));
  const arriLut = preferredDisplayLut(luts);
  const usingArri = prefs?.display_rendering.kind === "arri_display";

  return (
    <Dialog
      title="Preferences"
      width={620}
      onClose={() => store.closeOverlay()}
      footer={
        <>
          <LinkButton className="ym-dialog__footer-start" onClick={() => store.openOverlay("about")}>
            About Yellowmoon
          </LinkButton>
          <Button data-autofocus="" onClick={() => store.closeOverlay()}>
            Done
          </Button>
        </>
      }
    >
      {prefs === null ? (
        <p className="ym-dialog__text">Loading your preferences…</p>
      ) : (
        <>
          <Row title="Default look" hint="The look every new photo opens with.">
            <Select value={defaultLook?.id ?? ""} onValueChange={(next) => void store.makeDefault(next)}>
              <SelectTrigger aria-label="Default look" className="w-[220px]">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {lookOptions.map((option) => (
                  <SelectItem key={option.value} value={option.value}>
                    {option.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </Row>
          <Row
            title="Display rendering"
            hint={arriLut ? "How log becomes a picture." : "Import ARRI's display LUTs to use them here."}
          >
            <Segmented
              fit
              label="Display rendering"
              size="md"
              value={usingArri ? "arri" : "ours"}
              onChange={(next) =>
                void store.setPreferences({
                  display_rendering: next === "arri" && arriLut ? { kind: "arri_display", id: arriLut.id } : { kind: "aces2" },
                })
              }
              options={[
                { value: "ours", label: "Ours · ACES 2" },
                { value: "arri", label: "ARRI's", disabled: !arriLut, ariaLabel: arriLut ? "ARRI's display rendering" : "ARRI's display rendering, not imported" },
              ]}
            />
          </Row>
          {auto !== null && autoRowShown(auto) ? <AutoGrading store={store} auto={auto} saved={prefs.apple_auto_instructions} os={os} /> : null}
          <Row title="Export defaults" hint="Format, quality, size, where files go and how they are named.">
            <LinkButton aria-expanded={editing} onClick={() => setEditing((on) => !on)}>
              {editing ? "Hide" : "Edit…"}
            </LinkButton>
          </Row>
          {editing ? (
            <div className="ym-pref__defaults">
              <ExportOptionsForm store={store} options={exportOptions} mode="defaults" model={model} estimate={null} cli={null} />
            </div>
          ) : null}
          <Row title="Remove location from exports" hint="Leaves the GPS position out of the files you export.">
            <Switch
              label="Remove location from exports"
              checked={prefs.remove_location_from_exports}
              onChange={(on) => void store.setPreferences({ remove_location_from_exports: on })}
            />
          </Row>
          <Row title="Watch Downloads during import" hint="Picks up ARRI's zip as soon as it lands. Only while the wizard is open.">
            <Switch
              label="Watch Downloads during import"
              checked={prefs.watch_downloads_during_import}
              onChange={(on) => void store.setPreferences({ watch_downloads_during_import: on })}
            />
          </Row>
          <Row title="Check for updates automatically" hint="Quietly, a few seconds after launch and every six hours.">
            <Switch
              label="Check for updates automatically"
              checked={prefs.check_for_updates_automatically}
              onChange={(on) => void store.setPreferences({ check_for_updates_automatically: on })}
            />
          </Row>
          <Row
            title="Command-line tool"
            hint={cli === null ? "Checking…" : cli.installed ? `Installed at ${cli.path}` : "Not installed. It lets you run ym from Terminal."}
          >
            <LinkButton onClick={() => void store.installCli()}>{cli?.installed ? "Reinstall…" : "Install…"}</LinkButton>
          </Row>
        </>
      )}
    </Dialog>
  );
}
