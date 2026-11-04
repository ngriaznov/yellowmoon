import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { DevelopSettings, NodeId, NodeLink, NodePos, NodeRef, PhotoId } from "../../api/types";
import {
  connect,
  disconnect,
  hasPatch,
  insertAfter,
  moveCard,
  moveEarlier,
  moveLater,
  reconnect,
  remove,
  rename,
  reorderInto,
  resetNode,
  sameLink,
  toggleBypass,
} from "../../state/graphOps";
import type { GraphOutcome } from "../../state/graphOps";
import { INPUT_ID, MAIN_ID, nodeOf, OUTPUT_ID } from "../../state/settingsGraph";
import type { AppStore } from "../../state/store";

/** What the graph's toolbar, menus, keys and canvas ask for. Each is one patch and one undo step, or a sentence in the error bar. */
export interface NodeGraphActions {
  /** The selected link (the link's Delete comes before the node's). */
  selectedLink: NodeLink | null;
  /** The node whose name is being typed. */
  renaming: NodeId | null;
  /** Where its text box is: on the card, or in the panel's head (`docs/23` §7.2: double-click the name on the card or in the head, or More › Rename). */
  renamingIn: "card" | "head";
  selectNode: (id: NodeId) => void;
  selectLink: (link: NodeLink | null) => void;
  /** Add node › Corrector: after the selected node. */
  addCorrector: () => void;
  /** Add node › Film: after the selected node (`docs/24`). */
  addFilm: () => void;
  /** The key Delete and the toolbar's Delete: the selected link, else the selected node. */
  deleteSelection: () => void;
  deleteNode: (id?: NodeId) => void;
  toggleBypass: () => void;
  /** The head's Reset: the selected node's values at their defaults, its masks, name, bypass and links kept (`docs/23` §7.5). One undo step. */
  resetNode: () => void;
  startRename: (id?: NodeId, where?: "card" | "head") => void;
  finishRename: (id: NodeId, name: string | null) => void;
  moveCard: (id: NodeId, pos: NodePos) => void;
  reorderInto: (id: NodeId, link: NodeLink) => void;
  moveEarlier: () => void;
  moveLater: () => void;
  connect: (from: NodeRef, to: NodeRef) => void;
  reconnect: (link: NodeLink, to: NodeRef) => void;
  disconnect: (link: NodeLink) => void;
  /** Esc: ends a rename or lets go of the selected link; false when there was neither. */
  escape: () => boolean;
  /** Says a refusal in the error bar. */
  refuse: (reason: string) => void;
}

export interface UseNodeGraphOptions {
  store: AppStore;
  /** The graph of the photo on screen, or null while there is none. */
  settings: DevelopSettings | null;
  photoId: PhotoId | null;
  /** The node the graph and the panel are on, and how to change it. */
  selected: NodeId;
  select: (id: NodeId) => void;
}

/**
 * The node view's editing, as the screen uses it: the pure operations of `src/state/graphOps.ts` joined to the store. An operation that
 * is refused shows its sentence in the error bar and changes nothing; one that is not goes to `store.update` as one node patch, which
 * is one undo step. The new node, or the one before a deleted node, is selected once the core has answered, so a node that was never
 * made is never selected.
 */
export function useNodeGraph({ store, settings, photoId, selected, select }: UseNodeGraphOptions): NodeGraphActions {
  const [selectedLink, setSelectedLink] = useState<NodeLink | null>(null);
  const [renaming, setRenaming] = useState<NodeId | null>(null);
  const [renamingIn, setRenamingIn] = useState<"card" | "head">("card");
  // The largest id the page has seen for each photo: a node that is deleted does not give its id to the next one (`docs/23` §3.2).
  const used = useRef(new Map<PhotoId, number>());
  const latest = useRef({ settings, photoId, selected, selectedLink });
  latest.current = { settings, photoId, selected, selectedLink };

  useEffect(() => {
    if (settings === null || photoId === null) return;
    let most = used.current.get(photoId) ?? 0;
    for (const item of settings.nodes.items) if (typeof item.id === "number") most = Math.max(most, item.id);
    used.current.set(photoId, most);
  }, [settings, photoId]);

  // A photo, or an undo, that takes away the link or the node the page is on: both let go.
  useEffect(() => {
    setRenaming(null);
    setSelectedLink(null);
  }, [photoId]);
  useEffect(() => {
    if (selectedLink && settings && !settings.nodes.links.some((l) => sameLink(l, selectedLink))) setSelectedLink(null);
  }, [settings, selectedLink]);

  const refuse = useCallback((reason: string): void => store.reportError({ scope: "render", message: reason }), [store]);

  /** Sends what an operation answers, or says why it is refused. */
  const apply = useCallback(
    (outcome: GraphOutcome): void => {
      const { photoId: id } = latest.current;
      if (!outcome.ok) {
        refuse(outcome.reason);
        return;
      }
      if (!hasPatch(outcome) || id === null) return;
      const { select: next } = outcome;
      void store.update(outcome.patch, true, id).then(() => {
        const now = store.getState().settings[id]?.settings;
        if (next !== undefined && now && nodeOf(now, next)) select(next);
      });
    },
    [store, refuse, select],
  );
  /** Runs an operation on the settings as they are now. */
  const act = useCallback(
    (make: (settings: DevelopSettings, selected: NodeId) => GraphOutcome): void => {
      const { settings: current, selected: here } = latest.current;
      if (current) apply(make(current, here));
    },
    [apply],
  );

  const selectNode = useCallback(
    (id: NodeId): void => {
      setSelectedLink(null);
      select(id);
    },
    [select],
  );

  const actions: NodeGraphActions = {
    selectedLink,
    renaming,
    renamingIn,
    selectNode,
    selectLink: setSelectedLink,
    addCorrector: () => act((s, here) => insertAfter(s, here, { usedMax: used.current.get(latest.current.photoId ?? -1) ?? 0 })),
    addFilm: () => act((s, here) => insertAfter(s, here, { usedMax: used.current.get(latest.current.photoId ?? -1) ?? 0, kind: "film", name: "Film" })),
    deleteSelection: () => {
      const { selectedLink: link } = latest.current;
      if (link) {
        setSelectedLink(null);
        act((s) => disconnect(s, link));
      } else {
        act((s, here) => remove(s, here));
      }
    },
    deleteNode: (id) => act((s, here) => remove(s, id ?? here)),
    toggleBypass: () => act((s, here) => toggleBypass(s, here)),
    resetNode: () => act((s, here) => resetNode(s, here, { look: store.getState().looks?.find((look) => look.is_default)?.id })),
    startRename: (id, where = "card") => {
      const target = id ?? latest.current.selected;
      const node = latest.current.settings ? nodeOf(latest.current.settings, target) : undefined;
      if (node?.kind === "corrector" || node?.kind === "film") {
        select(target);
        setRenamingIn(where);
        setRenaming(target);
      }
    },
    finishRename: (id, name) => {
      setRenaming(null);
      if (name !== null) act((s) => rename(s, id, name));
    },
    moveCard: (id, pos) => act((s) => moveCard(s, id, pos)),
    reorderInto: (id, link) => act((s) => reorderInto(s, id, link)),
    moveEarlier: () => act((s, here) => moveEarlier(s, here)),
    moveLater: () => act((s, here) => moveLater(s, here)),
    connect: (from, to) => act((s) => connect(s, from, to)),
    reconnect: (link, to) => act((s) => reconnect(s, link, to)),
    disconnect: (link) => {
      setSelectedLink((was) => (was && sameLink(was, link) ? null : was));
      act((s) => disconnect(s, link));
    },
    escape: () => {
      if (renaming !== null) {
        setRenaming(null);
        return true;
      }
      if (latest.current.selectedLink) {
        setSelectedLink(null);
        return true;
      }
      return false;
    },
    refuse,
  };
  return useStableActions(actions);
}

/**
 * The same actions with functions that never change (each calls the newest) and an object that changes only with the three values it holds.
 * The operations are closures made at each render, and the graph, memoised, would draw again at each tick of a slider drag for a new
 * function that does the same (U-02).
 */
function useStableActions(actions: NodeGraphActions): NodeGraphActions {
  const latest = useRef(actions);
  latest.current = actions;
  const proxies = useMemo(() => {
    const out: Record<string, unknown> = {};
    for (const [key, value] of Object.entries(actions)) {
      if (typeof value === "function") out[key] = (...args: unknown[]) => (latest.current as unknown as Record<string, (...a: unknown[]) => unknown>)[key]?.(...args);
    }
    // The set of operations is fixed.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    return out;
  }, []);
  const { selectedLink, renaming, renamingIn } = actions;
  return useMemo(
    () => ({ ...proxies, selectedLink, renaming, renamingIn }) as unknown as NodeGraphActions,
    [proxies, selectedLink, renaming, renamingIn],
  );
}

/** True when the Delete key, the toolbar's Delete and More › Delete node may remove this node. */
export const isDeletable = (id: NodeId): boolean => id !== MAIN_ID && id !== INPUT_ID && id !== OUTPUT_ID;
