import { useMemo, useState } from "react";
import type { ReactNode } from "react";
import { Button, LinkButton, Segmented, Switch } from "../../components";
import { exportModel } from "../../export/model";
import { useAppState, useStore } from "../../state/context";
import { preferredDisplayLut } from "../../wizard/model";
import { ExportOptionsForm } from "../export/ExportOptionsForm";
import { Dialog } from "./Dialog";

interface RowProps {
  title: string;
  hint?: ReactNode;
  children: ReactNode;
}

/** One row of Preferences: a title and a line of help at the left, the control at the right. */
function Row({ title, hint, children }: RowProps) {
  return (
    <div className="ym-pref" role="group" aria-label={title}>
      <span className="ym-pref__text">
        <span className="ym-pref__title" aria-hidden="true">
          {title}
        </span>
        {hint ? <span className="ym-pref__hint">{hint}</span> : null}
      </span>
      <span className="ym-pref__control">{children}</span>
    </div>
  );
}

/**
 * Preferences (⌘,, `docs/11-ui.md` §Preferences): a small window in the same style, one column of rows. Everything here is
 * a preference of `docs/01` §Persistence and is saved as soon as it changes; nothing needs a Save button.
 */
export function Preferences() {
  const store = useStore();
  const prefs = useAppState((s) => s.preferences);
  const looks = useAppState((s) => s.looks);
  const luts = useAppState((s) => s.displayLuts);
  const cli = useAppState((s) => s.cliLink);
  const exportOptions = useAppState((s) => s.exportOptions);
  const session = useAppState((s) => s.session);
  const settings = useAppState((s) => s.settings);
  const [editing, setEditing] = useState(false);

  const model = useMemo(
    () => exportModel({ session, settings, looks, preferences: prefs }, "this", exportOptions),
    [session, settings, looks, prefs, exportOptions],
  );

  const defaultLook = looks?.find((l) => l.is_default);
  const lookOptions = (looks ?? []).filter((l) => !l.missing).map((l) => ({ value: l.id, label: l.name }));
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
            <select
              className="ym-pref__select"
              aria-label="Default look"
              value={defaultLook?.id ?? ""}
              onChange={(event) => void store.makeDefault(event.target.value)}
            >
              {lookOptions.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </Row>
          <Row
            title="Display rendering"
            hint={arriLut ? "How log becomes a picture." : "Import ARRI's display LUTs to use them here."}
          >
            <Segmented
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
            hint={cli === null ? "Checking…" : cli.installed ? `Installed at ${cli.path}` : "Not installed. It lets you run yellowmoon from Terminal."}
          >
            <LinkButton onClick={() => void store.installCli()}>{cli?.installed ? "Reinstall…" : "Install…"}</LinkButton>
          </Row>
        </>
      )}
    </Dialog>
  );
}
