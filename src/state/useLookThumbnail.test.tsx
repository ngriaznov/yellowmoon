import { act, renderHook, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { AppStore } from "./store";
import { MAX_TILE_RETRIES, useLookThumbnail } from "./useLookThumbnail";

/** The two store methods a tile uses, as spies; `lookThumbnail` answers from `answers` in turn (the last one repeats). */
function fakeStore(answers: Array<Promise<string> | (() => Promise<string>)>) {
  let n = 0;
  const lookThumbnail = vi.fn((_photo: number, _look: string, _key: string) => {
    const answer = answers[Math.min(n++, answers.length - 1)]!;
    return typeof answer === "function" ? answer() : answer;
  });
  const forgetLookThumbnail = vi.fn();
  return { store: { lookThumbnail, forgetLookThumbnail } as unknown as AppStore, lookThumbnail, forgetLookThumbnail };
}

const cancelled = { scope: "render", message: "The render was cancelled." };

describe("useLookThumbnail", () => {
  it("gives the address the store answers with", async () => {
    const { store } = fakeStore([Promise.resolve("blob:a")]);
    const { result } = renderHook(() => useLookThumbnail(store, 4, "builtin:neutral", "k"));
    await waitFor(() => expect(result.current.src).toBe("blob:a"));
  });

  it("asks for a new picture when the photo's other settings change, and keeps the old one until it comes", async () => {
    let arrive: (url: string) => void = () => {};
    const { store, lookThumbnail } = fakeStore([Promise.resolve("blob:a"), new Promise<string>((yes) => (arrive = yes))]);
    const { result, rerender } = renderHook(({ key }) => useLookThumbnail(store, 4, "builtin:neutral", key), { initialProps: { key: "k1" } });
    await waitFor(() => expect(result.current.src).toBe("blob:a"));
    rerender({ key: "k2" });
    await waitFor(() => expect(lookThumbnail).toHaveBeenCalledTimes(2));
    expect(result.current.src).toBe("blob:a");
    await act(async () => arrive("blob:b"));
    expect(result.current.src).toBe("blob:b");
    expect(lookThumbnail.mock.calls.map((c) => c[2])).toEqual(["k1", "k2"]);
  });

  it("asks for nothing while disabled or without a photo", () => {
    const { store, lookThumbnail } = fakeStore([Promise.resolve("blob:a")]);
    renderHook(() => useLookThumbnail(store, 4, "builtin:neutral", "k", false));
    renderHook(() => useLookThumbnail(store, null, "builtin:neutral", "k"));
    expect(lookThumbnail).not.toHaveBeenCalled();
  });

  it("asks again once when the request it was handed was cancelled", async () => {
    const { store, lookThumbnail } = fakeStore([() => Promise.reject(cancelled), () => Promise.resolve("blob:b")]);
    const { result } = renderHook(() => useLookThumbnail(store, 4, "builtin:neutral", "k"));
    await waitFor(() => expect(result.current.src).toBe("blob:b"));
    expect(lookThumbnail).toHaveBeenCalledTimes(2);
  });

  it("does not ask again for a look the library refuses", async () => {
    const refused = { scope: "looks", message: "File missing or changed. Import again." };
    const { store, lookThumbnail } = fakeStore([() => Promise.reject(refused)]);
    const { result } = renderHook(() => useLookThumbnail(store, 4, "arri:3000:film-warm", "k"));
    await act(async () => {
      await new Promise((r) => setTimeout(r, 20));
    });
    expect(lookThumbnail).toHaveBeenCalledTimes(1);
    expect(result.current.src).toBeUndefined();
  });

  it("gives up after its allowance, not in a loop", async () => {
    const { store, lookThumbnail } = fakeStore([() => Promise.reject(cancelled)]);
    const { result } = renderHook(() => useLookThumbnail(store, 4, "builtin:neutral", "k"));
    await waitFor(() => expect(lookThumbnail).toHaveBeenCalledTimes(1 + MAX_TILE_RETRIES));
    await act(async () => {
      await new Promise((r) => setTimeout(r, 20));
    });
    expect(lookThumbnail).toHaveBeenCalledTimes(1 + MAX_TILE_RETRIES);
    expect(result.current.src).toBeUndefined();
  });

  it("does not ask again when it has left meanwhile", async () => {
    let reject: (error: unknown) => void = () => {};
    const { store, lookThumbnail } = fakeStore([new Promise<string>((_, no) => (reject = no))]);
    const { unmount } = renderHook(() => useLookThumbnail(store, 4, "builtin:neutral", "k"));
    unmount();
    reject(cancelled);
    await act(async () => {
      await new Promise((r) => setTimeout(r, 20));
    });
    expect(lookThumbnail).toHaveBeenCalledTimes(1);
  });

  it("does not ask again for a photo it has left, and asks for the one it moved to", async () => {
    let reject: (error: unknown) => void = () => {};
    const { store, lookThumbnail } = fakeStore([new Promise<string>((_, no) => (reject = no)), Promise.resolve("blob:b")]);
    const { result, rerender } = renderHook(({ photo }) => useLookThumbnail(store, photo, "builtin:neutral", "k"), { initialProps: { photo: 4 } });
    rerender({ photo: 5 });
    await waitFor(() => expect(result.current.src).toBe("blob:b"));
    reject(cancelled);
    await act(async () => {
      await new Promise((r) => setTimeout(r, 20));
    });
    expect(lookThumbnail.mock.calls.map((c) => c[0])).toEqual([4, 5]);
    expect(result.current.src).toBe("blob:b");
  });

  it("forgets the address of a picture that does not load and shows the one it gets instead", async () => {
    const { store, lookThumbnail, forgetLookThumbnail } = fakeStore([Promise.resolve("ym-frame://localhost/look-4-1.png"), Promise.resolve("ym-frame://localhost/look-4-9.png")]);
    const { result } = renderHook(() => useLookThumbnail(store, 4, "builtin:neutral", "k"));
    await waitFor(() => expect(result.current.src).toBe("ym-frame://localhost/look-4-1.png"));
    act(() => result.current.onImageError());
    expect(forgetLookThumbnail).toHaveBeenCalledWith(4, "builtin:neutral", "k", "ym-frame://localhost/look-4-1.png");
    expect(result.current.src).toBeUndefined();
    await waitFor(() => expect(result.current.src).toBe("ym-frame://localhost/look-4-9.png"));
    expect(lookThumbnail).toHaveBeenCalledTimes(2);
  });

  it("stops asking when every address it is given fails to load", async () => {
    let n = 0;
    const { store, lookThumbnail } = fakeStore([() => Promise.resolve(`ym-frame://localhost/look-4-${++n}.png`)]);
    const { result } = renderHook(() => useLookThumbnail(store, 4, "builtin:neutral", "k"));
    for (let i = 0; i <= MAX_TILE_RETRIES; i++) {
      await waitFor(() => expect(result.current.src).toBeDefined());
      act(() => result.current.onImageError());
    }
    await act(async () => {
      await new Promise((r) => setTimeout(r, 20));
    });
    expect(lookThumbnail).toHaveBeenCalledTimes(1 + MAX_TILE_RETRIES);
    expect(result.current.src).toBeUndefined();
  });

  it("renews the allowance for the next photo", async () => {
    // Photo 4 spends its whole allowance on cancelled requests. Photo 5's first request is cancelled too, and it must still get its
    // retry: with an allowance that was carried over, the tile would stay blank.
    const spent = Array.from({ length: 1 + MAX_TILE_RETRIES }, () => () => Promise.reject(cancelled));
    const { store, lookThumbnail } = fakeStore([...spent, () => Promise.reject(cancelled), () => Promise.resolve("blob:ok")]);
    const { result, rerender } = renderHook(({ photo }) => useLookThumbnail(store, photo, "builtin:neutral", "k"), { initialProps: { photo: 4 } });
    await waitFor(() => expect(lookThumbnail).toHaveBeenCalledTimes(1 + MAX_TILE_RETRIES));
    rerender({ photo: 5 });
    await waitFor(() => expect(result.current.src).toBe("blob:ok"));
    expect(lookThumbnail.mock.calls.map((c) => c[0])).toEqual([...Array(1 + MAX_TILE_RETRIES).fill(4), 5, 5]);
  });
});
