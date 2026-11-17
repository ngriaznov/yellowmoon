import { EllipsisIcon } from "lucide-react";
import { Fragment, useRef } from "react";
import type { ReactNode } from "react";
import type { DevelopSettings, NodeId, NodeLink, NodeRef } from "../../api/types";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuSub,
  DropdownMenuSubContent,
  DropdownMenuSubTrigger,
  DropdownMenuTrigger,
} from "@/components/glass/dropdown-menu";
import { Button } from "../../components";
import { cardStates, endName, GRAPH_OP_TEXT, linksOf, linkTargets, reorderAvailability } from "../../state/graphOps";
import { CHANNEL_PORTS } from "../../state/ports";
import { isCardItem, isGlyphItem, MAIN_ID, nodeOf } from "../../state/settingsGraph";

export interface NodeMoreMenuProps {
  settings: DevelopSettings;
  /** The node the menu is for. */
  node: NodeId;
  onRename: () => void;
  onMoveEarlier: () => void;
  onMoveLater: () => void;
  /** Link this node's picture or key output to that input. */
  onLink: (from: NodeRef, to: NodeRef) => void;
  onRemoveLink: (link: NodeLink) => void;
  onDelete: () => void;
  /** Where the menu opens from its button: over the photo (`top`, the graph's toolbar) or down the panel (`bottom`, the node's head). */
  side?: "top" | "bottom";
  /** A class for the button, for the head's round icon. */
  buttonClassName?: string;
}

/**
 * More (`docs/23` §7.5, and §7.2 *The keyboard path*): what a card can do that is not a click or a key, so that every link can be made,
 * moved and removed without a pointer. Rename, Move earlier, Move later, Link picture to ▸, Link key to ▸, Remove link ▸ and Delete node;
 * Delete node is off for Main, which has no Move items either, and the Input has Link picture to only. The inputs a link can go to are the
 * ones the same rules light up while a link is dragged (`linkTargets`); a link that cannot be removed says so in the error bar when it
 * is chosen, as the drag does. It opens upward over the photo, like Add node, when it is the toolbar's, and down the panel from the head.
 *
 * It is the third icon of the node's head (N-09; until then a button of the graph's toolbar), and opens down the panel, to the left of
 * its button, from there.
 */
export function NodeMoreMenu({ settings, node, onRename, onMoveEarlier, onMoveLater, onLink, onRemoveLink, onDelete, side = "top", buttonClassName }: NodeMoreMenuProps) {
  const item = nodeOf(settings, node);
  const states = cardStates(settings);
  const glyph = isGlyphItem(item);
  // A corrector, a Film node or one of the four kinds of `docs/25`: they are renamed and deleted; only correctors have Move and a key.
  const corrector = isCardItem(item) || glyph;
  const reorder = reorderAvailability(settings, node);
  // A Splitter has three picture outputs and links each by its name; every other node has one.
  const outputs = item?.kind === "splitter" ? CHANNEL_PORTS : (["picture"] as const);
  const pictureTo = (port: NodeRef["port"]): NodeRef[] => linkTargets(settings, { node, port });
  const keyTo = isCardItem(item) ? linkTargets(settings, { node, port: "key" }) : [];
  const links = linksOf(settings, node);
  const name = (ref: NodeRef, side: "in" | "out"): string => endName(settings, ref, side, states);
  const describe = (link: NodeLink): string =>
    link.to.node === node
      ? `${link.to.port === "key" ? "Key" : "Picture"} from ${name(link.from, "out")}`
      : `${link.from.port === "key" ? "Key" : "Picture"} to ${name(link.to, "in")}`;

  // Rename puts a text box on the card. It opens once the menu has closed and is gone (the menu keeps the focus until then, and takes it
  // back to its button when it closes: not this time, the text box has it).
  const renaming = useRef(false);

  const submenu = (label: string, entries: ReactNode[]): ReactNode => (
    <DropdownMenuSub>
      <DropdownMenuSubTrigger disabled={entries.length === 0}>{label}</DropdownMenuSubTrigger>
      <DropdownMenuSubContent>{entries}</DropdownMenuSubContent>
    </DropdownMenuSub>
  );

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="header" aria-label="More" title="More" className={buttonClassName} icon={<EllipsisIcon aria-hidden="true" />} />
      </DropdownMenuTrigger>
      <DropdownMenuContent
        side={side}
        align={side === "top" ? "start" : "end"}
        aria-label="More"
        className="ym-more-menu"
        onCloseAutoFocus={(event) => {
          if (!renaming.current) return;
          renaming.current = false;
          event.preventDefault();
          onRename();
        }}
      >
        {corrector ? (
          <DropdownMenuItem
            onSelect={() => {
              renaming.current = true;
            }}
          >
            Rename
          </DropdownMenuItem>
        ) : null}
        {corrector && !glyph && node !== MAIN_ID ? (
          <>
            <DropdownMenuItem disabled={reorder.earlier !== null} onSelect={onMoveEarlier}>
              Move earlier
            </DropdownMenuItem>
            <DropdownMenuItem disabled={reorder.later !== null} onSelect={onMoveLater}>
              Move later
            </DropdownMenuItem>
          </>
        ) : null}
        {corrector ? <DropdownMenuSeparator /> : null}
        {outputs.map((port) => (
          <Fragment key={port}>
            {submenu(
              port === "picture" ? "Link picture to" : `Link ${port} to`,
              pictureTo(port).map((to) => (
                <DropdownMenuItem key={`${to.node}:${to.port}`} onSelect={() => onLink({ node, port }, to)}>
                  {name(to, "in")}
                </DropdownMenuItem>
              )),
            )}
          </Fragment>
        ))}
        {item?.kind === "corrector" || item?.kind === "film"
          ? submenu(
              "Link key to",
              keyTo.map((to) => (
                <DropdownMenuItem key={`${to.node}:${to.port}`} onSelect={() => onLink({ node, port: "key" }, to)}>
                  {name(to, "in")}
                </DropdownMenuItem>
              )),
            )
          : null}
        {submenu(
          "Remove link",
          links.map((link) => (
            <DropdownMenuItem key={`${link.from.node}.${link.from.port}>${link.to.node}.${link.to.port}`} onSelect={() => onRemoveLink(link)}>
              {describe(link)}
            </DropdownMenuItem>
          )),
        )}
        {corrector ? (
          <>
            <DropdownMenuSeparator />
            <DropdownMenuItem variant="destructive" disabled={node === MAIN_ID} aria-description={node === MAIN_ID ? GRAPH_OP_TEXT.mainDelete : undefined} onSelect={onDelete}>
              Delete node
            </DropdownMenuItem>
          </>
        ) : null}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
