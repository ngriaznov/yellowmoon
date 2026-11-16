import { useEffect, useId, useRef, useState } from "react";
import type { KeyboardEvent } from "react";
import { Checkbox } from "@/components/glass/checkbox";
import { Input } from "@/components/glass/input";
import type { PresetGroup, PresetSource } from "../../api/types";
import { Button } from "../../components";
import { useAppState, useStore } from "../../state/context";
import {
  AUTO_STAYS_AUTO,
  GROUP_LABELS,
  GROUP_LINES,
  groupsText,
  MAX_NAME_CHARS,
  nameExists,
  nameProblem,
  OLDER_EXPORT_LINE,
  PRESET_GROUP_ORDER,
} from "../../state/presets";
import type { PresetDialog } from "../../state/store";
import { Dialog } from "./Dialog";

/** The reason Save is off, or null (`docs/30` section 9.3): the name's own sentence is under the field, the group's is Save's title. */
export function saveBlocked(name: string, groups: readonly PresetGroup[]): string | null {
  const problem = nameProblem(name);
  if (problem !== null) return problem;
  return groups.length === 0 ? "Tick at least one group" : null;
}

/** One question with Cancel and a primary answer: "Replace Warm evening?", "Update Warm evening?". The focus starts on Cancel (the primary answer is not the default). */
function Question({
  title,
  lines,
  answer,
  onAnswer,
  onCancel,
  focusKey,
}: {
  title: string;
  lines: string[];
  answer: string;
  onAnswer: () => void;
  onCancel: () => void;
  focusKey?: string;
}) {
  const cancel = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    cancel.current?.focus({ preventScroll: true });
  }, [focusKey]);
  return (
    <Dialog
      title={title}
      width={480}
      onClose={onCancel}
      footer={
        <>
          <Button variant="secondary" ref={cancel} data-autofocus="" onClick={onCancel}>
            Cancel
          </Button>
          <Button onClick={onAnswer}>{answer}</Button>
        </>
      }
    >
      <div className="ym-preset-question">
        {lines.map((line) => (
          <p key={line} className="ym-preset-question__line">
            {line}
          </p>
        ))}
      </div>
    </Dialog>
  );
}

/**
 * The Save preset dialog (`docs/30` section 9.3): the Name field, eight boxes (the groups the photo changed are ticked), the line about
 * Auto, Cancel and Save. When the name exists the body is replaced by "Replace Warm evening?". Opened for a photo, or for an exported
 * photo's file (then the Name starts empty and an older export says so in one line).
 */
function SaveDialog({ source, changed, olderExport }: { source: PresetSource; changed: PresetGroup[]; olderExport: boolean }) {
  const store = useStore();
  const presets = useAppState((s) => s.presets?.presets ?? []);
  const [name, setName] = useState("");
  const [touched, setTouched] = useState(false);
  const [groups, setGroups] = useState<PresetGroup[]>(() => PRESET_GROUP_ORDER.filter((g) => changed.includes(g)));
  const [asking, setAsking] = useState(false);
  const [busy, setBusy] = useState(false);
  const field = useRef<HTMLInputElement>(null);
  const id = useId();
  const problem = nameProblem(name);
  const blocked = saveBlocked(name, groups);
  const shownProblem = problem !== null && (touched || name.trim() !== "") ? problem : null;
  const existing = presets.find((p) => p.name.toLowerCase() === name.trim().toLowerCase());

  const save = async (replace: boolean): Promise<void> => {
    if (busy) return;
    setBusy(true);
    const saved = await store.savePreset(name, groups, source, replace);
    if (!saved) setBusy(false);
  };
  const submit = (): void => {
    setTouched(true);
    if (blocked !== null) return;
    if (existing && nameExists(presets, name)) setAsking(true);
    else void save(false);
  };
  // Cancel on the question goes back to the form with the focus on Name.
  const [refocus, setRefocus] = useState(0);
  const back = (): void => {
    setAsking(false);
    setRefocus((n) => n + 1);
  };
  useEffect(() => {
    if (refocus > 0) field.current?.focus({ preventScroll: true });
  }, [refocus]);

  const cancelAsk = useRef<HTMLButtonElement>(null);
  const asked = asking && existing !== undefined;
  useEffect(() => {
    if (asked) cancelAsk.current?.focus({ preventScroll: true });
  }, [asked]);
  // One dialog for both steps, so that the focus goes back to what opened it whichever step ends it.
  return (
    <Dialog
      title={asked ? `Replace ${existing.name}?` : "Save preset"}
      width={480}
      onClose={() => (asked ? back() : store.closeOverlay())}
      footer={
        asked ? (
          <>
            <Button variant="secondary" ref={cancelAsk} onClick={back}>
              Cancel
            </Button>
            <Button onClick={() => void save(true)}>Replace</Button>
          </>
        ) : (
          <>
            <Button variant="secondary" onClick={() => store.closeOverlay()}>
              Cancel
            </Button>
            <Button disabled={blocked !== null} title={blocked ?? undefined} onClick={submit}>
              Save
            </Button>
          </>
        )
      }
    >
      {asked ? (
        <div className="ym-preset-question">
          <p className="ym-preset-question__line">A preset with this name is already saved. Replacing it can&apos;t be undone.</p>
        </div>
      ) : (
        <div className="ym-preset-form">
          <div className="ym-preset-form__name">
            <label className="ym-preset-form__label" htmlFor={`${id}-name`}>
              Name
            </label>
            <Input
              id={`${id}-name`}
              ref={field}
              data-autofocus=""
              placeholder="Name"
              value={name}
              maxLength={MAX_NAME_CHARS * 4}
              aria-invalid={shownProblem !== null || undefined}
              aria-describedby={shownProblem !== null ? `${id}-problem` : undefined}
              autoComplete="off"
              spellCheck={false}
              onChange={(e) => {
                setName(e.target.value);
                setTouched(true);
              }}
              onBlur={() => setTouched(true)}
              onKeyDown={(e: KeyboardEvent<HTMLInputElement>) => {
                if (e.key === "Enter") {
                  e.preventDefault();
                  submit();
                }
              }}
            />
            {shownProblem !== null ? (
              <p id={`${id}-problem`} className="ym-preset-form__problem">
                {shownProblem}
              </p>
            ) : null}
          </div>
          <fieldset className="ym-preset-groups" aria-label="Save these">
            {PRESET_GROUP_ORDER.map((group) => (
              <label key={group} className="ym-preset-group" htmlFor={`${id}-${group}`}>
                <Checkbox
                  id={`${id}-${group}`}
                  checked={groups.includes(group)}
                  aria-labelledby={`${id}-${group}-label`}
                  aria-describedby={`${id}-${group}-line`}
                  onCheckedChange={(on) => setGroups((now) => PRESET_GROUP_ORDER.filter((g) => (g === group ? on === true : now.includes(g))))}
                />
                <span className="ym-preset-group__text">
                  <span id={`${id}-${group}-label`} className="ym-preset-group__label">
                    {GROUP_LABELS[group]}
                  </span>
                  <span id={`${id}-${group}-line`} className="ym-preset-group__line">
                    {GROUP_LINES[group]}
                  </span>
                </span>
              </label>
            ))}
          </fieldset>
          <p className="ym-preset-note">{AUTO_STAYS_AUTO}</p>
          {olderExport ? (
            <p className="ym-preset-note" data-kind="older">
              {OLDER_EXPORT_LINE}
            </p>
          ) : null}
        </div>
      )}
    </Dialog>
  );
}

/** The Rename dialog (`docs/30` section 9.4): the Name field alone; a name that exists is refused under it. */
function RenameDialog({ name: current }: { name: string }) {
  const store = useStore();
  const [name, setName] = useState(current);
  const [problem, setProblem] = useState<string | null>(null);
  const id = useId();
  const blocked = nameProblem(name);
  const rename = async (): Promise<void> => {
    if (blocked !== null) return;
    setProblem(await store.renamePreset(current, name));
  };
  return (
    <Dialog
      title="Rename preset"
      width={480}
      onClose={() => store.closeOverlay()}
      footer={
        <>
          <Button variant="secondary" onClick={() => store.closeOverlay()}>
            Cancel
          </Button>
          <Button disabled={blocked !== null} title={blocked ?? undefined} onClick={() => void rename()}>
            Rename
          </Button>
        </>
      }
    >
      <div className="ym-preset-form">
        <div className="ym-preset-form__name">
          <label className="ym-preset-form__label" htmlFor={`${id}-name`}>
            Name
          </label>
          <Input
            id={`${id}-name`}
            data-autofocus=""
            placeholder="Name"
            value={name}
            aria-invalid={problem !== null || (blocked !== null && name !== "") || undefined}
            aria-describedby={problem !== null || blocked !== null ? `${id}-problem` : undefined}
            autoComplete="off"
            spellCheck={false}
            onFocus={(e) => e.currentTarget.select()}
            onChange={(e) => {
              setName(e.target.value);
              setProblem(null);
            }}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                e.preventDefault();
                void rename();
              }
            }}
          />
          {problem !== null || blocked !== null ? (
            <p id={`${id}-problem`} className="ym-preset-form__problem">
              {problem ?? blocked}
            </p>
          ) : null}
        </div>
      </div>
    </Dialog>
  );
}

/** The preset dialog that is open (`AppState.presetDialog`). */
export function PresetDialogs() {
  const store = useStore();
  const dialog: PresetDialog | null = useAppState((s) => s.presetDialog);
  if (!dialog) return null;
  switch (dialog.kind) {
    case "save":
      return <SaveDialog source={dialog.source} changed={dialog.changed} olderExport={dialog.olderExport} />;
    case "rename":
      return <RenameDialog name={dialog.name} />;
    case "update":
      return (
        <Question
          title={`Update ${dialog.name}?`}
          lines={[`It keeps its groups, ${groupsText(dialog.groups, true)}, and takes their values from ${dialog.photo}.`]}
          answer="Update"
          onAnswer={() => void store.updatePreset(dialog.name, dialog.photoId, dialog.photo)}
          onCancel={() => store.closeOverlay()}
        />
      );
    case "replace-import": {
      const first = dialog.queue[0];
      if (!first) return null;
      return (
        <Question
          title={`Replace ${first.name}?`}
          lines={["A preset with this name is already saved. Replacing it can't be undone."]}
          answer="Replace"
          onAnswer={() => void store.answerImportReplace(true)}
          onCancel={() => void store.answerImportReplace(false)}
          focusKey={first.name}
        />
      );
    }
  }
}
