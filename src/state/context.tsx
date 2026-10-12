/** React access to the store: a provider that connects it, and selector hooks. */
import { createContext, useContext, useEffect, useSyncExternalStore } from "react";
import type { ReactNode } from "react";
import type { PhotoSummary, ResolvedSettings } from "../api/types";
import { AppStore } from "./store";
import type { AppState } from "./store";

const StoreContext = createContext<AppStore | null>(null);

export interface StoreProviderProps {
  store: AppStore;
  children?: ReactNode;
}

/** Connects the store to the core's events for as long as it is mounted, and offers it to the tree. */
export function StoreProvider({ store, children }: StoreProviderProps) {
  useEffect(() => store.connect(), [store]);
  return <StoreContext.Provider value={store}>{children}</StoreContext.Provider>;
}

/** The store. Throws outside a `StoreProvider`. */
export function useStore(): AppStore {
  const store = useContext(StoreContext);
  if (!store) throw new Error("useStore needs a StoreProvider above it.");
  return store;
}

/**
 * A piece of the state. `select` must return something that stays the same object while the
 * state it reads is unchanged (a field, not a freshly built array), or the component draws on
 * every change.
 */
export function useAppState<T>(select: (state: AppState) => T): T {
  const store = useStore();
  return useSyncExternalStore(store.subscribe, () => select(store.getState()));
}

/** The photo on screen, or null. */
export function useCurrentPhoto(): PhotoSummary | null {
  return useAppState((s) => (s.session && s.session.current_index !== null ? (s.session.photos[s.session.current_index] ?? null) : null));
}

/** The resolved settings of the photo on screen, once they have arrived. */
export function useCurrentSettings(): ResolvedSettings | null {
  return useAppState((s) => {
    const index = s.session?.current_index;
    const photo = index === null || index === undefined ? undefined : s.session?.photos[index];
    return photo ? (s.settings[photo.id] ?? null) : null;
  });
}
