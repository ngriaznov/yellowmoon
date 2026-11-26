import { EyeIcon } from "lucide-react";
import { useEffect } from "react";
import { Spinner } from "@/components/glass/spinner";
import type { ResolvedSettings } from "../../api/types";
import { Button, Card, formats, ResetLink, Slider } from "../../components";
import { useAppState } from "../../state/context";
import { cardKey, NOT_FOUND_SHARE, shareOf, shareText } from "../../state/reference";
import type { ElementName } from "../../state/reference";
import { patchOf } from "../../state/settingsGraph";
import type { ReferenceGradeItem } from "../../state/settingsGraph";
import type { AppStore } from "../../state/store";

/** What the Elements tab says (`docs/34` §9.4a). */
export const ELEMENTS_TEXT = {
  notFound: "Not found in this photo.",
  looking: "Looking at the photo…",
  eye: "Show mask",
  eyeOn: "Hide mask",
  eyeTip: "Show where this element is in the photo.",
  eyeOnTip: "Hide the mask.",
  eyeOffTip: "This element is not in the photo.",
} as const;

/** The seven cards, in the order the grades run (`docs/34` §2.3), with their names and the most each may turn the hue, in degrees (the fit's own caps, §4a.1). */
export const ELEMENT_CARDS: ReadonlyArray<{ id: ElementName | "subject"; title: string; cap: number }> = [
  { id: "sky", title: "Sky", cap: 40 },
  { id: "vegetation", title: "Vegetation", cap: 35 },
  { id: "water", title: "Water", cap: 35 },
  { id: "ground", title: "Ground", cap: 30 },
  { id: "built", title: "Built", cap: 25 },
  { id: "skin", title: "Skin", cap: 15 },
  { id: "subject", title: "Subject", cap: 0 },
];

/** 20 → "+20°", −6 → "−6°", 0 → "0°". */
const degrees = (v: number): string => `${formats.signed(v)}°`;

/** The numbers of the cards as the core's `MaskView.element` numbers them (the order of {@link ELEMENT_CARDS}). */
const elementNumber = (id: ElementName | "subject"): number => ELEMENT_CARDS.findIndex((c) => c.id === id);

interface ElementCardProps {
  store: AppStore;
  rs: ResolvedSettings;
  node: number;
  item: ReferenceGradeItem;
  id: ElementName | "subject";
  title: string;
  cap: number;
  last: boolean;
}

/**
 * One element's card (`docs/34` §9.4a, `design/reference-grade/06-elements-tab.jpg`): its name, the share of the photo the last fit found, Exposure (−1 to +1
 * EV), Hue (± the element's cap) and Chroma (50 to 160 % of the photo's own), each with its Reset, and the card's Reset when something is set. The subject has
 * Exposure only. Under 2 % of the photo the card says it was not found and greys, and its sliders still work (a grade there acts on nothing); while the scene is
 * made it says it is looking.
 */
function ElementCard({ store, rs, node, item, id, title, cap, last }: ElementCardProps) {
  const card = useAppState((s) => s.references[cardKey(rs.photo_id, node)]);
  const looking = card?.fit?.step === "scene";
  const share = id === "subject" ? null : shareOf(card, id);
  const absent = share !== null && share < NOT_FOUND_SHARE;
  const values = id === "subject" ? { exposure_ev: item.elements.subject.exposure_ev, hue_deg: 0, chroma: 1 } : item.elements[id];
  const number = elementNumber(id);
  const eye = useAppState((s) => s.elementEye === number);
  // An element the photo has none of, or one the scene is still being made for, has no mask to show: its eye stays off.
  const noMask = absent || looking;
  const changed = values.exposure_ev !== 0 || values.hue_deg !== 0 || values.chroma !== 1;
  const send = (partial: { exposure_ev?: number; hue_deg?: number; chroma?: number }, final = true): void =>
    void store.update(patchOf(node, { elements: { [id]: partial } }), final, rs.photo_id);
  return (
    <Card
      title={title}
      last={last}
      className="ym-element"
      tag={
        <>
          {looking ? (
            <span className="ym-element__share" data-looking="">
              <Spinner aria-hidden="true" />
              {ELEMENTS_TEXT.looking}
            </span>
          ) : share !== null && !absent ? (
            <span className="ym-element__share">{shareText(share)}</span>
          ) : null}
          <Button
            variant="header"
            className="ym-element__eye"
            data-tool="eye"
            pressed={eye}
            disabled={noMask && !eye}
            aria-label={`${eye ? ELEMENTS_TEXT.eyeOn : ELEMENTS_TEXT.eye}, ${title}`}
            title={eye ? ELEMENTS_TEXT.eyeOnTip : absent ? ELEMENTS_TEXT.eyeOffTip : ELEMENTS_TEXT.eyeTip}
            icon={<EyeIcon aria-hidden="true" />}
            onClick={() => store.setElementEye(eye ? null : number)}
          />
        </>
      }
      action={
        changed ? (
          <ResetLink
            size="card"
            aria-label={`Reset ${title}`}
            onClick={() => send(id === "subject" ? { exposure_ev: 0 } : { exposure_ev: 0, hue_deg: 0, chroma: 1 })}
          />
        ) : undefined
      }
    >
      <div className="ym-element__rows" data-absent={absent || undefined}>
        {absent ? <p className="ym-element__absent">{ELEMENTS_TEXT.notFound}</p> : null}
        <Slider
          label="Exposure (EV)"
          value={values.exposure_ev}
          min={-1}
          max={1}
          step={0.01}
          defaultValue={0}
          format={formats.ev}
          onChange={(exposure_ev, final) => send({ exposure_ev }, final)}
        />
        {id === "subject" ? null : (
          <>
            <Slider label="Hue" value={values.hue_deg} min={-cap} max={cap} step={1} defaultValue={0} format={degrees} onChange={(hue_deg, final) => send({ hue_deg }, final)} />
            <Slider
              label="Chroma"
              value={Math.round(values.chroma * 100)}
              min={50}
              max={160}
              step={1}
              defaultValue={100}
              format={formats.percent}
              onChange={(percent, final) => send({ chroma: percent / 100 }, final)}
            />
          </>
        )}
      </div>
    </Card>
  );
}

/** A Reference Grade node's Elements tab (`docs/34` §9.4a): seven cards on the corrector's slider rows, in the order the grades run. */
export function ElementsTab({ store, rs, node, item }: { store: AppStore; rs: ResolvedSettings; node: number; item: ReferenceGradeItem }) {
  // The eye is the store's view state: it ends with the node, the photo, Before and Esc there, and with this tab here.
  useEffect(
    () => () => {
      if (store.getState().elementEye !== null) store.setElementEye(null);
    },
    [store],
  );
  return (
    <>
      {ELEMENT_CARDS.map((card, i) => (
        <ElementCard key={card.id} store={store} rs={rs} node={node} item={item} {...card} last={i === ELEMENT_CARDS.length - 1} />
      ))}
    </>
  );
}
