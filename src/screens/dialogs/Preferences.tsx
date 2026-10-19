import { useMemo, useState } from "react";
import type { ReactNode } from "react";
import { Item, ItemActions, ItemContent, ItemDescription, ItemTitle } from "@/components/glass/item";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/glass/select";
import { Button, LinkButton, Segmented, Switch } from "../../components";
import { exportModel } from "../../export/model";
import { useAppState, useStore } from "../../state/context";
import { lookLabeller } from "../../state/looks";
import { preferredDisplayLut } from "../../wizard/model";
import { ExportOptionsForm } from "../export/ExportOptionsForm";
import { Dialog } from "./Dialog";

/** The title of the Apple Auto row (`docs/19` section 2). */
export const APPLE_AUTO_TITLE = "Auto with Apple Intelligence";

/** Its hint, word for word from `docs/19` section 2. */
export const APPLE_AUTO_HINT =
  "Apple's on-device model looks at each photo and sets its exposure, contrast and saturation, and makes the white warmer or cooler. It runs on this Mac and sends nothing anywhere. Turn it off to use Yellowmoon's own Auto, the same on every computer.";

interface RowProps {
  title: string;
  hint?: ReactNode;
  /** The hint is a paragraph: shown whole, where a hint is otherwise cut after two lines. */
  longHint?: boolean;
  children: ReactNode;
}

/** One row of Preferences: a title and a line of help at the left, the control at the right. */
function Row({ title, hint, longHint = false, children }: RowProps) {
  return (
    <Item size="sm" role="group" aria-label={title} className="ym-pref">
      <ItemContent className="ym-pref__text">
        <ItemTitle className="ym-pref__title" aria-hidden="true">
          {title}
        </ItemTitle>
        {hint ? <ItemDescription className={longHint ? "ym-pref__hint line-clamp-none" : "ym-pref__hint"}>{hint}</ItemDescription> : null}
      </ItemContent>
      <ItemActions className="ym-pref__control">{children}</ItemActions>
    </Item>
  );
}

/**
 * Preferences (⌘,, `docs/11-ui.md` §Preferences): a small window in the same style, one column of rows. Everything here is
 * a preference of `docs/01` §Persistence and is saved as soon as it changes; nothing needs a Save button. "Auto with Apple
 * Intelligence" is there only while Apple Auto is available on this computer, which is asked each time the window opens
 * (`docs/19` sections 1 and 2): absent everywhere else, the browser build included.
 */
export function Preferences() {
  const store = useStore();
  const prefs = useAppState((s) => s.preferences);
  const looks = useAppState((s) => s.looks);
  const luts = useAppState((s) => s.displayLuts);
  const cli = useAppState((s) => s.cliLink);
  const appleAuto = useAppState((s) => s.appleAuto);
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
          {appleAuto?.available ? (
            <Row title={APPLE_AUTO_TITLE} hint={APPLE_AUTO_HINT} longHint>
              <Switch
                label={APPLE_AUTO_TITLE}
                checked={prefs.apple_auto}
                onChange={(on) => void store.setPreferences({ apple_auto: on })}
              />
            </Row>
          ) : null}
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
