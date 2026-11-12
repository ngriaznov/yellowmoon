/**
 * Face Refinement › Face (`docs/27` S-11c, `design/skin/`): the node's first tab. Five cards in this order: **Faces** (the faces the core found, round items, Same for
 * all faces), **Face N** (the eye, Blemish, Dark circles, Even skin and Texture), **Skin grading**, **Teeth** (with Copy to other faces) and **Protect** (the brush).
 * What the core is doing shows instead of them: finding the faces, none found, or a failure said plainly.
 *
 * Every control writes the node as one patch (a drag is one undo step): `all` while Same for all faces is on, else `faces` (replaced whole, so the entry of the face
 * the tab is on changes and the others stay), and `protect` for the brush.
 */
import { EyeIcon, ScanFaceIcon } from "lucide-react";
import { useEffect, useRef } from "react";
import type { ReactNode } from "react";
import { Progress } from "@/components/glass/progress";
import { Spinner } from "@/components/glass/spinner";
import { TooltipTrigger } from "@/components/glass/tooltip";
import { OptionalTooltip } from "@/components/OptionalTooltip";
import type { FaceSettings, FaceState, PhotoSummary, ResolvedSettings } from "../../api/types";
import { Button, Card, Field, formats, ResetLink, RowNote, Segmented, Slider, SwitchRow } from "../../components";
import type { SegmentedOption } from "../../components";
import { useAppState } from "../../state/context";
import { DEFAULT_BRUSH } from "../../state/brushStrokes";
import { defaultFaceSettings, patchOf } from "../../state/settingsGraph";
import type { FaceRefinementItem } from "../../state/settingsGraph";
import type { AppStore } from "../../state/store";
import { sizeField, sizeSlider } from "../tools/maskScales";
import {
  FACE_FIELDS,
  FACE_TEXT,
  faceCardTitle,
  facesCopied,
  facesOnSplit,
  facesWith,
  GRADING_FIELDS,
  resetPartial,
  selectedFace,
  shownSettings,
  targetName,
  TEETH_FIELDS,
} from "./faceModel";
import { setSkinAreasView, useFacePicture } from "./faceHooks";
import { useFaces } from "./useFaces";

export interface FaceTabProps {
  store: AppStore;
  photo: PhotoSummary;
  rs: ResolvedSettings;
  /** The Face Refinement node the panel is on. */
  node: number;
  item: FaceRefinementItem;
}

/** The Faces card has room for this many faces before it scrolls (two rows of five). */
const ROOM = 10;

/** A glass tooltip around a control that has none of its own (a row of sliders, a Reset): a span is the trigger, so the control need not hold a ref. */
function Tip({ text, children, className }: { text: string; children: ReactNode; className?: string }) {
  return (
    <OptionalTooltip text={text} side="top">
      <TooltipTrigger asChild>
        <span className={className ?? "ym-face-tip"}>{children}</span>
      </TooltipTrigger>
    </OptionalTooltip>
  );
}

/** One item of the Faces list: the face's picture (a neutral disc with its number until the core can make one) and, when it has a picture, its number on a badge. */
function FaceItem({ store, photoId, faces, number, selected, ringed, sameForAll, onSelect }: { store: AppStore; photoId: number; faces: FaceState; number: number; selected: boolean; ringed: boolean; sameForAll: boolean; onSelect: (n: number) => void }) {
  const src = useFacePicture(store, photoId, number, faces);
  return (
    <OptionalTooltip text={sameForAll ? FACE_TEXT.faceTipAll(number) : FACE_TEXT.faceTip(number)} side="top">
      <TooltipTrigger asChild>
        <button type="button" className="ym-face" aria-pressed={selected} aria-label={`Face ${number}`} data-selected={selected || undefined} data-ringed={ringed || undefined} onClick={() => onSelect(number)}>
          <span className="ym-face__pic" data-empty={src ? undefined : true}>
            {src ? <img src={src} alt="" draggable={false} /> : <span className="ym-face__disc">{number}</span>}
          </span>
          {src ? <span className="ym-face__num">{number}</span> : null}
        </button>
      </TooltipTrigger>
    </OptionalTooltip>
  );
}

/** The tints of Show skin areas, as the core draws them (`docs/27` S-11b2). */
const LEGEND: ReadonlyArray<{ label: string; color: string; dashed?: boolean }> = [
  { label: "Skin", color: "#ff6fd2" },
  { label: "Under the eyes", color: "#3d9bff" },
  { label: "Even skin and grading", color: "#8ee56b", dashed: true },
  { label: "Teeth", color: "#8a5cff" },
];

/** What the colours of the skin areas on the photo mean; shown under the sliders while the eye is on. */
function SkinLegend() {
  return (
    <ul className="ym-face-legend" aria-label="Skin areas">
      {LEGEND.map((item) => (
        <li key={item.label}>
          <span className="ym-face-legend__swatch" data-dashed={item.dashed || undefined} style={{ "--swatch": item.color } as React.CSSProperties} aria-hidden="true" />
          {item.label}
        </li>
      ))}
    </ul>
  );
}

/** What the tab shows while the faces are not known, or there are none, or the worker failed: a card that says so, and nothing to set. */
function FacesNote({ state }: { state: FaceState | undefined }) {
  if (!state || state.phase === "finding") {
    return (
      <Card title="Faces" tag={<span className="ym-face-count">Finding…</span>} last>
        <div className="ym-face-finding" role="status">
          <Spinner aria-hidden="true" />
          <div>
            <p className="ym-face-finding__title">{FACE_TEXT.finding}</p>
            <p className="ym-face-finding__line">{FACE_TEXT.findingLine}</p>
          </div>
        </div>
        <Progress aria-label="Finding faces" className="ym-face-bar" />
      </Card>
    );
  }
  if (state.phase === "ready") {
    return (
      <Card title="Faces" last>
        <div className="ym-face-empty">
          <span className="ym-face-empty__icon" aria-hidden="true">
            <ScanFaceIcon width={22} height={22} />
          </span>
          <p className="ym-face-empty__title">{FACE_TEXT.none}</p>
          <RowNote>{FACE_TEXT.noneLine}</RowNote>
        </div>
      </Card>
    );
  }
  return (
    <Card title="Faces" last>
      <RowNote role="status" className="ym-face-message">
        {state.message ?? (state.phase === "failed" ? "Faces can't be found in this photo." : "Face Refinement needs the face models, and this copy of Yellowmoon has none. Reinstall the app.")}
      </RowNote>
    </Card>
  );
}

/** The Protect card (`docs/27` S-11c): Paint · Erase, Size and Clear; the brush paints on the photo while the tool is on, and a stroke is sent on release. */
function ProtectCard({ store, rs, node, item }: { store: AppStore; rs: ResolvedSettings; node: number; item: FaceRefinementItem }) {
  const brush = useAppState((s) => s.brush);
  const on = useAppState((s) => s.brushTool);
  // The tool belongs to this card: leaving the tab, the node or the photo ends it.
  useEffect(() => () => store.setBrushTool(false), [store]);
  const has = item.protect.strokes.length > 0;
  const choose = (next: "paint" | "erase"): void => {
    store.setBrush({ erase: next === "erase" });
    store.setBrushTool(true);
  };
  const tools: ReadonlyArray<SegmentedOption<"paint" | "erase">> = [
    { value: "paint", label: "Paint" },
    { value: "erase", label: "Erase" },
  ];
  return (
    <Card
      title="Protect"
      last
      action={<ResetLink size="card" label="Clear" disabled={!has} onClick={() => void store.update(patchOf(node, { protect: { strokes: [] } }), true, rs.photo_id)} />}
    >
      <RowNote className="ym-face-line">{on ? FACE_TEXT.protectActive : FACE_TEXT.protectLine}</RowNote>
      <Field label="Brush">
        {/* A click on the chosen segment is not a change for the control, but it is a request for the tool. */}
        <div
          className="ym-brush-tool"
          onClick={(event) => {
            const value = (event.target as HTMLElement).closest<HTMLElement>('[role="radio"]')?.dataset.value;
            if (value === "paint" || value === "erase") choose(value);
          }}
        >
          <Segmented label="Paint or erase" value={brush.erase ? "erase" : "paint"} onChange={choose} options={tools} />
        </div>
      </Field>
      <Slider
        label="Size"
        value={sizeSlider(brush.size)}
        min={0}
        max={100}
        defaultValue={sizeSlider(DEFAULT_BRUSH.size)}
        format={formats.integer}
        onChange={(v) => store.setBrush({ size: sizeField(v) })}
        onReset={() => store.setBrush({ size: DEFAULT_BRUSH.size })}
      />
    </Card>
  );
}

/** Face Refinement › Face. */
export function FaceTab({ store, photo, rs, node, item }: FaceTabProps) {
  const state = useFaces(store, photo.id, true);
  const selected = useAppState((s) => s.faceSelected);
  // The eye is the store's view state: it ends with the node, the photo, Before and Esc there, and with this tab here.
  const eye = useAppState((s) => s.skinAreas !== null);
  const count = state?.phase === "ready" ? state.faces.length : 0;
  const number = selectedFace(selected, count);
  const shown = shownSettings(item, number);
  const latest = useRef({ photo: photo.id, node, number });
  latest.current = { photo: photo.id, node, number };
  // With the eye on, the view follows the chosen face.
  useEffect(() => {
    if (eye) setSkinAreasView(store, photo.id, node, number, true);
  }, [store, eye, photo.id, node, number]);
  useEffect(
    () => () => {
      setSkinAreasView(store, latest.current.photo, latest.current.node, latest.current.number, false);
    },
    [store],
  );

  if (!state || state.phase !== "ready" || count === 0) return <FacesNote state={state} />;

  const send = (partial: Partial<FaceSettings>, final = true): void => {
    const patch = item.same_for_all ? patchOf(node, { all: partial }) : patchOf(node, { faces: facesWith(item, count, number, partial) });
    void store.update(patch, final, rs.photo_id);
  };
  const setSame = (same: boolean): void => {
    void store.update(patchOf(node, same ? { same_for_all: true } : { same_for_all: false, faces: facesOnSplit(item, count) }), true, rs.photo_id);
  };
  const defaults = defaultFaceSettings();
  const row = (field: keyof FaceSettings, label: string, range: readonly [number, number], extra: { title?: string; indent?: boolean } = {}) => {
    const slider = (
      <Slider
        label={label}
        value={shown[field]}
        min={range[0]}
        max={range[1]}
        defaultValue={defaults[field]}
        format={range[0] < 0 ? formats.signed : formats.integer}
        className={extra.indent ? "ym-face-sub" : undefined}
        onChange={(value, final) => send({ [field]: value }, final)}
      />
    );
    return extra.title ? (
      <Tip key={field} text={extra.title} className="ym-face-row">
        {slider}
      </Tip>
    ) : (
      <div key={field} className="ym-face-row">
        {slider}
      </div>
    );
  };
  const target = targetName(item, count, number);
  const tag = <span className="ym-face-tag">{target}</span>;
  const list = state.faces;
  return (
    <div className="ym-face-tab">
      <Card title="Faces" tag={<span className="ym-face-count">{count} found</span>}>
        <div className="ym-faces" role="group" aria-label="Faces" data-scroll={count > ROOM || undefined}>
          {list.map((face) => (
            <FaceItem
              key={face.number}
              store={store}
              photoId={photo.id}
              faces={state}
              number={face.number}
              selected={face.number === number}
              ringed={item.same_for_all && count > 1}
              sameForAll={item.same_for_all && count > 1}
              onSelect={(n) => store.selectFace(n)}
            />
          ))}
        </div>
        {count > 1 ? <SwitchRow label={FACE_TEXT.sameForAll} title={FACE_TEXT.sameForAllTip} checked={item.same_for_all} onChange={setSame} className="ym-face-same" /> : null}
      </Card>
      <Card
        title={faceCardTitle(item, count, number)}
        tag={
          <Button
            variant="header"
            className="ym-face-eye"
            pressed={eye}
            aria-label={FACE_TEXT.eye}
            title={eye ? FACE_TEXT.eyeOnTip : FACE_TEXT.eyeTip}
            icon={<EyeIcon aria-hidden="true" />}
            onClick={() => setSkinAreasView(store, photo.id, node, number, !eye)}
          />
        }
        action={<ResetLink size="card" onClick={() => send(resetPartial(FACE_FIELDS, "defaults"))} />}
      >
        {row("blemish", "Blemish", [0, 100])}
        {row("dark_circles", "Dark circles", [0, 100])}
        {row("even_skin", "Even skin", [0, 100])}
        {row("texture", "Texture", [-100, 100], { indent: true })}
        {eye ? <SkinLegend /> : null}
      </Card>
      <Card
        title="Skin grading"
        tag={tag}
        action={
          <Tip text={FACE_TEXT.gradingResetTip}>
            <ResetLink size="card" onClick={() => send(resetPartial(GRADING_FIELDS, "zero"))} />
          </Tip>
        }
      >
        {row("skin_contrast", "Contrast", [-100, 100], { title: FACE_TEXT.contrastTip })}
        {row("shadow_desaturation", "Shadow desaturation", [0, 100], { title: FACE_TEXT.shadowTip })}
        {row("shine_removal", "Shine removal", [0, 100], { title: FACE_TEXT.shineTip })}
      </Card>
      <Card
        title="Teeth"
        tag={tag}
        action={
          <Tip text={FACE_TEXT.teethResetTip}>
            <ResetLink size="card" onClick={() => send(resetPartial(TEETH_FIELDS, "zero"))} />
          </Tip>
        }
      >
        {row("teeth_whitening", "Whitening", [0, 100], { title: FACE_TEXT.whiteningTip })}
        {!item.same_for_all && count > 1 ? (
          <div className="ym-face-copy">
            <Button
              variant="header"
              size="sm"
              title={FACE_TEXT.copyTip}
              onClick={() => void store.update(patchOf(node, { faces: facesCopied(item, count, number) }), true, rs.photo_id)}
            >
              {FACE_TEXT.copy}
            </Button>
            <span className="ym-face-copy__note">Copies all of Face {number}&apos;s settings.</span>
          </div>
        ) : null}
      </Card>
      <ProtectCard store={store} rs={rs} node={node} item={item} />
    </div>
  );
}
