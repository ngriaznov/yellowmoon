import { Item, ItemContent, ItemMedia } from "@/components/glass/item";
import { LinkButton, NodesGlyph } from "../../components";
import type { HiddenFromSimpleView } from "../../state/settingsGraph";

/**
 * The sentence the notice of the simple view says, before its link (`docs/23` §6). With more than Main it counts the correctors ("This
 * photo has 4 nodes.") and says what the controls change; with Main alone it says what Main has that they do not show.
 */
export function nodesNoticeText(hidden: HiddenFromSimpleView): string {
  return hidden.kind === "nodes"
    ? `This photo has ${hidden.count} nodes. These controls change the input, the output and node 1, Main.`
    : "Node 1, Main, has a trim, masks or a bypass that these controls don't show.";
}

/** The tooltip of the filmstrip's glyph for a photo with this in its graph (`docs/23` §6). */
export function nodesGlyphText(hidden: HiddenFromSimpleView): string {
  return hidden.kind === "nodes" ? `${hidden.count} nodes` : "Node 1 has masks or a trim";
}

export interface NodesNoticeProps {
  hidden: HiddenFromSimpleView;
  /** "Open nodes": the node view, with Main selected. */
  onOpen: () => void;
}

/**
 * The notice of the simple view of a photo that has more than it shows, under the tab row of every tab (`docs/23` §6,
 * `design/nodes/04-simple-photo-has-nodes.jpg`): a glass pill with the nodes glyph and the sentence, and **Open nodes** at the end of the
 * text, so that it takes two lines and the pill never grows to three. A polite status. A glasscn `Item`.
 */
export function NodesNotice({ hidden, onOpen }: NodesNoticeProps) {
  return (
    <div className="ym-tools__nodes">
      <Item role="status" size="sm" className="ym-nodes-notice">
        <ItemMedia aria-hidden="true">
          <NodesGlyph />
        </ItemMedia>
        <ItemContent>
          <span className="ym-nodes-notice__text">
            {nodesNoticeText(hidden)}{" "}
            <LinkButton size="sm" weight="bold" className="ym-nodes-notice__link" onClick={onOpen}>
              Open nodes
            </LinkButton>
          </span>
        </ItemContent>
      </Item>
    </div>
  );
}
