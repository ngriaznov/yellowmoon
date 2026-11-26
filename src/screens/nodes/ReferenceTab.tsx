import { useRef } from "react";
import { ChevronDownIcon, CircleAlertIcon, ImageIcon, ImagesIcon, InfoIcon } from "lucide-react";
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/glass/dropdown-menu";
import { Item, ItemContent } from "@/components/glass/item";
import type { ResolvedSettings } from "../../api/types";
import { Button, formats, ProgressBar, RowNote, Slider, Switch } from "../../components";
import { useAppState } from "../../state/context";
import { cardKey, fitButton, REFERENCE_TEXT, referenceNotes, referenceStatus } from "../../state/reference";
import { nodeOf, patchOf } from "../../state/settingsGraph";
import type { ReferenceGradeItem } from "../../state/settingsGraph";
import type { AppStore } from "../../state/store";

export interface ReferenceTabProps {
  store: AppStore;
  rs: ResolvedSettings;
  node: number;
  item: ReferenceGradeItem;
}

/**
 * A Reference Grade node's Reference tab (`docs/34` §9.2, §9.3; `design/reference-grade/`): one card with **no title** (the owner's answer at RG-H2:
 * the tab and the panel's head already say it). From the top: the reference's picture, letterboxed in a 128 px box, with its name under it
 * (session only, `docs/34` §3.7); Choose reference… (a menu: Image file… and Photo in the strip) and Fit; Strength with its line; Skin protection with its
 * line; and under a hairline the status group, a `role="status"` region that says one state at a time, then the notes, and the error as a notice.
 * While a fit runs the progress bar and Cancel take the place of Fit, Choose reference… is off, and the step is the status line.
 */
export function ReferenceTab({ store, rs, node, item }: ReferenceTabProps) {
  const card = useAppState((s) => s.references[cardKey(rs.photo_id, node)]);
  const choosing = useAppState((s) => s.choosing !== null && s.choosing.node === node && s.choosing.photo === rs.photo_id);
  const status = referenceStatus(card, item, rs.settings);
  const button = fitButton(status.state);
  const running = status.state === "running";
  const notes = referenceNotes(card, item, rs.settings, node);
  const info = card?.info ?? null;
  const send = (partial: Parameters<typeof patchOf>[1], final = true): void => void store.update(patchOf(node, partial), final, rs.photo_id);
  const wantStrip = useRef(false);
  const fraction = card?.fit?.step === "search" ? card.fit.fraction : 0.05;

  return (
    <section aria-label="Reference" className="ym-card ym-ref" data-last>
      <div className="ym-ref__thumb" data-empty={info ? undefined : ""}>
        {info ? <img src={info.thumbnail} alt="" draggable={false} /> : <ImagesIcon aria-hidden="true" />}
      </div>
      <p className="ym-ref__name" data-testid="reference-name">
        {info ? info.label : REFERENCE_TEXT.noReferenceName}
      </p>

      <div className="ym-ref__row">
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="header" className="ym-ref__choose" pressed={choosing} disabled={running} icon={<ImageIcon aria-hidden="true" />} aria-haspopup="menu">
              Choose reference…
              <ChevronDownIcon data-icon="inline-end" aria-hidden="true" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent
            align="start"
            aria-label="Choose reference"
            // The chooser starts when the menu has closed, and takes the keyboard (`Filmstrip`, mode `choose`): the menu keeps the focus in itself until
            // then and gives it back to its button, so it is told not to when the strip is what was chosen.
            onCloseAutoFocus={(event) => {
              if (!wantStrip.current) return;
              wantStrip.current = false;
              event.preventDefault();
              store.startChoosingReference(node);
            }}
          >
            <DropdownMenuItem onSelect={() => void store.chooseReferenceFile(node)}>Image file…</DropdownMenuItem>
            <DropdownMenuItem
              onSelect={() => {
                wantStrip.current = true;
              }}
            >
              Photo in the strip
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
        {running ? null : (
          <Button
            variant={button.prominent ? "primary" : "header"}
            className="ym-ref__fit"
            data-accent={button.accent ? "" : undefined}
            disabled={button.disabled}
            title={button.disabled ? REFERENCE_TEXT.chooseFirst : undefined}
            onClick={() => void store.fitReference(node)}
          >
            {button.label}
          </Button>
        )}
      </div>
      {running ? (
        <div className="ym-ref__row ym-ref__running">
          <ProgressBar value={fraction * 100} label="Fitting" className="ym-ref__bar" />
          <Button variant="header" title="Cancel the fit (Esc)" onClick={() => store.cancelReferenceFit()}>
            Cancel
          </Button>
        </div>
      ) : null}

      <div className="ym-ref__strength">
        <Slider
          label="Strength"
          value={item.strength}
          min={0}
          max={100}
          step={1}
          defaultValue={100}
          format={formats.percent}
          onChange={(strength, final) => send({ strength }, final)}
        />
        <RowNote className="ym-ref__line">{REFERENCE_TEXT.strengthLine}</RowNote>
      </div>

      <div className="ym-ref__skin">
        <div className="ym-ref__skin-text">
          <span className="ym-ref__skin-label" aria-hidden="true">
            Skin protection
          </span>
          <RowNote className="ym-ref__line">{REFERENCE_TEXT.skinLine}</RowNote>
        </div>
        <Switch label="Skin protection" checked={item.skin_protection} onChange={(skin_protection) => send({ skin_protection })} />
      </div>

      <div role="status" aria-live="polite" className="ym-ref__status" data-state={status.state}>
        {card?.error ? (
          <Item variant="muted" size="sm" className="ym-ref__error" data-testid="reference-error">
            <CircleAlertIcon aria-hidden="true" />
            <ItemContent>{card.error}</ItemContent>
          </Item>
        ) : null}
        <p className="ym-ref__status-line">{status.text}</p>
        {notes.map((text) => (
          <Item key={text} variant="muted" size="sm" className="ym-ref__note" role="note">
            <InfoIcon aria-hidden="true" />
            <ItemContent>{text}</ItemContent>
          </Item>
        ))}
      </div>
    </section>
  );
}

/** The node's item from the resolved settings, or null when it is not a Reference Grade node. */
export function referenceItemOf(rs: ResolvedSettings, node: number): ReferenceGradeItem | null {
  const item = nodeOf(rs.settings, node);
  return item?.kind === "reference_grade" ? item : null;
}
