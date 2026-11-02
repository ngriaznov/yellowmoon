import { EllipsisIcon } from "lucide-react";
import { useRef } from "react";
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
import { cardStates, GRAPH_OP_TEXT, linksOf, linkTargets, listName, reorderAvailability } from "../../state/graphOps";
import { MAIN_ID, nodeOf } from "../../state/settingsGraph";

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
}

/**
 * More (`docs/23` §7.5, and §7.2 *The keyboard path*): what a card can do that is not a click or a key, so that every link can be made,
 * moved and removed without a pointer. Rename, Move earlier, Move later, Link picture to ▸, Link key to ▸, Remove link ▸ and Delete node;
 * Delete node is off for Main, which has no Move items either, and the Input has Link picture to only. The inputs a link can go to are the
 * ones the same rules light up while a link is dragged (`linkTargets`); a link that cannot be removed says so in the error bar when it
 * is chosen, as the drag does. It opens upward over the photo, like Add node.
 *
 * Until the node's head comes with the panel (N-09) it is a button of the graph's toolbar; the head will hold the same menu.
 */
export function NodeMoreMenu({ settings, node, onRename, onMoveEarlier, onMoveLater, onLink, onRemoveLink, onDelete }: NodeMoreMenuProps) {
  const item = nodeOf(settings, node);
  const states = cardStates(settings);
  const corrector = item?.kind === "corrector";
  const reorder = reorderAvailability(settings, node);
  const pictureTo = linkTargets(settings, { node, port: "picture" });
  const keyTo = corrector ? linkTargets(settings, { node, port: "key" }) : [];
  const links = linksOf(settings, node);
  const name = (id: NodeId): string => listName(settings, id, states);
  const describe = (link: NodeLink): string =>
    link.to.node === node
      ? `${link.to.port === "key" ? "Key" : "Picture"} from ${name(link.from.node)}`
      : `${link.from.port === "key" ? "Key" : "Picture"} to ${name(link.to.node)}`;

  // Rename puts a text box on the card: the menu must not take the focus back to its button when it closes.
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
        <Button variant="header" aria-label="More" title="More" icon={<EllipsisIcon aria-hidden="true" />} />
      </DropdownMenuTrigger>
      <DropdownMenuContent
        side="top"
        align="start"
        aria-label="More"
        className="ym-more-menu"
        onCloseAutoFocus={(event) => {
          if (!renaming.current) return;
          renaming.current = false;
          event.preventDefault();
        }}
      >
        {corrector ? (
          <DropdownMenuItem
            onSelect={() => {
              renaming.current = true;
              onRename();
            }}
          >
            Rename
          </DropdownMenuItem>
        ) : null}
        {corrector && node !== MAIN_ID ? (
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
        {submenu(
          "Link picture to",
          pictureTo.map((to) => (
            <DropdownMenuItem key={`${to.node}:${to.port}`} onSelect={() => onLink({ node, port: "picture" }, to)}>
              {name(to.node)}
            </DropdownMenuItem>
          )),
        )}
        {corrector
          ? submenu(
              "Link key to",
              keyTo.map((to) => (
                <DropdownMenuItem key={`${to.node}:${to.port}`} onSelect={() => onLink({ node, port: "key" }, to)}>
                  {name(to.node)}
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
