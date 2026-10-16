import { afterEach, describe, expect, it } from "vitest";
import { GESTURE_EVENTS, installDesktopFeel, isTextTarget } from "./desktopFeel";

afterEach(() => {
  document.body.replaceChildren();
});

function add<K extends keyof HTMLElementTagNameMap>(tag: K, setup: (el: HTMLElementTagNameMap[K]) => void = () => undefined): HTMLElementTagNameMap[K] {
  const el = document.createElement(tag);
  setup(el);
  document.body.append(el);
  return el;
}

const menu = (target: EventTarget): Event => {
  const event = new MouseEvent("contextmenu", { bubbles: true, cancelable: true });
  target.dispatchEvent(event);
  return event;
};

describe("isTextTarget", () => {
  it("is true in a field and in editable text", () => {
    expect(isTextTarget(add("input"))).toBe(true);
    expect(isTextTarget(add("textarea"))).toBe(true);
    const editable = add("div", (el) => el.setAttribute("contenteditable", "true"));
    expect(isTextTarget(editable)).toBe(true);
    const inner = document.createElement("span");
    editable.append(inner);
    expect(isTextTarget(inner)).toBe(true);
    expect(isTextTarget(editable.appendChild(document.createTextNode("x")))).toBe(true);
  });

  it("is false for the rest of the page, for contenteditable=false and for nothing", () => {
    expect(isTextTarget(add("div"))).toBe(false);
    expect(isTextTarget(add("button"))).toBe(false);
    expect(isTextTarget(add("div", (el) => el.setAttribute("contenteditable", "false")))).toBe(false);
    expect(isTextTarget(null)).toBe(false);
    expect(isTextTarget(window)).toBe(false);
  });
});

describe("installDesktopFeel", () => {
  it("cancels the page menu over the page and keeps it in a field", () => {
    const off = installDesktopFeel();
    expect(menu(add("div")).defaultPrevented).toBe(true);
    expect(menu(add("button")).defaultPrevented).toBe(true);
    expect(menu(add("input")).defaultPrevented).toBe(false);
    expect(menu(add("textarea")).defaultPrevented).toBe(false);
    off();
  });

  it("cancels WebKit's pinch events that nothing handled, so the page cannot zoom", () => {
    const off = installDesktopFeel();
    for (const name of GESTURE_EVENTS) {
      const event = new Event(name, { bubbles: true, cancelable: true });
      add("div").dispatchEvent(event);
      expect(event.defaultPrevented, name).toBe(true);
    }
    off();
  });

  it("takes its listeners off again", () => {
    const off = installDesktopFeel();
    off();
    expect(menu(add("div")).defaultPrevented).toBe(false);
    const event = new Event("gesturechange", { bubbles: true, cancelable: true });
    add("div").dispatchEvent(event);
    expect(event.defaultPrevented).toBe(false);
  });
});
