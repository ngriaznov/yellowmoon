import { Button, Tag } from "../../components";
import { useAppState, useStore } from "../../state/context";
import { formatSize } from "../../wizard/model";
import { InputQuestion } from "../wizard/InputQuestion";
import { Dialog } from "./Dialog";

/**
 * Looks › Import .cube… (`docs/08-looks.md` §Any other .cube): the files that were chosen, what each is, and for LUTs that
 * match no ARRI package the one question: what does it expect as input? A recognised ARRI package is imported without asking.
 */
export function CubeImport() {
  const store = useStore();
  const dialog = useAppState((s) => s.cubeImport);
  if (!dialog) return null;
  const importable = dialog.items.filter((i) => i.found.recognition.kind === "package" || (i.found.recognition.kind === "needs_answer" && i.answer));
  const unanswered = dialog.items.some((i) => i.found.recognition.kind === "needs_answer" && !i.answer);
  const locked = importable.length === 0 || unanswered;
  return (
    <Dialog
      title="Import LUTs"
      blurb="Say what each LUT expects as input, then it joins Your LUTs."
      width={600}
      dismissable={!dialog.busy}
      onClose={() => store.closeOverlay()}
      footer={
        <>
          <Button variant="secondary" disabled={dialog.busy} onClick={() => store.closeOverlay()}>
            Cancel
          </Button>
          <Button
            // Focus starts on Import only when it can be pressed; while it is locked it starts on the first question, the next thing to do.
            data-autofocus={locked ? undefined : ""}
            disabled={dialog.busy || locked}
            title={unanswered ? "Say what each LUT expects as input." : importable.length === 0 ? "Nothing here can be imported." : undefined}
            onClick={() => void store.confirmCubeImport()}
          >
            {dialog.busy ? "Importing…" : "Import"}
          </Button>
        </>
      }
    >
      <ul className="ym-cube-list" aria-label="LUTs to import">
        {dialog.items.map((item) => {
          const r = item.found.recognition;
          const size = formatSize(item.found.size_bytes);
          return (
            <li key={item.found.path}>
              <span className="ym-cube-list__name">
                <span>
                  {item.found.name}
                  {size ? ` · ${size}` : ""}
                </span>
                {r.kind === "package" ? <Tag variant="match">Match</Tag> : r.kind === "needs_answer" ? <Tag variant={item.answer ? "match" : "active"}>{item.answer ? "Ready" : "Question"}</Tag> : <Tag variant="failed">Can&apos;t use</Tag>}
              </span>
              {r.kind === "package" ? <span>{r.what} · {r.contents}. It is imported as it comes.</span> : null}
              {r.kind === "needs_answer" ? (
                <>
                  <span>{r.note}</span>
                  <InputQuestion name={item.found.name} value={item.answer} suggestion={r.suggestion} onChange={(input) => store.answerCube(item.found.path, input)} />
                </>
              ) : null}
              {r.kind === "unusable" || r.kind === "missing" ? <span>{r.reason}</span> : null}
            </li>
          );
        })}
      </ul>
    </Dialog>
  );
}
