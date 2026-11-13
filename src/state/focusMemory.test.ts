import { afterEach, describe, expect, it } from "vitest";
import { rememberFocus, restoreFocus } from "./focusMemory";

function page(html: string): void {
  document.body.innerHTML = html;
}

afterEach(() => {
  document.body.innerHTML = "";
});

describe("focus memory (docs/28 A-04)", () => {
  it("gives the focus back to the control that had it", () => {
    page('<button id="a">Fit</button><button id="b">Export</button>');
    document.getElementById("a")?.focus();
    rememberFocus("export");
    document.getElementById("b")?.focus();
    (document.activeElement as HTMLElement).blur();
    expect(restoreFocus("export", () => null)?.id).toBe("a");
    expect(document.activeElement?.id).toBe("a");
  });

  it("finds the control again when the screen drew it again", () => {
    page('<div id="header"><button>Fit</button></div>');
    document.querySelector("button")?.focus();
    rememberFocus("export");
    page('<div id="header"><button>Fit</button></div>');
    expect(document.activeElement).toBe(document.body);
    const back = restoreFocus("export", () => null);
    expect(back?.textContent).toBe("Fit");
    expect(document.activeElement).toBe(back);
  });

  it("goes to the fallback when the control is gone", () => {
    page('<button id="gone">Crop</button>');
    document.getElementById("gone")?.focus();
    rememberFocus("export");
    page('<button id="home">Export</button>');
    const back = restoreFocus("export", () => document.getElementById("home"));
    expect(back?.id).toBe("home");
  });

  it("goes to the fallback when nothing had the focus", () => {
    page('<button id="home">Export</button>');
    rememberFocus("export");
    expect(restoreFocus("export", () => document.getElementById("home"))?.id).toBe("home");
  });

  it("does not take the focus from a control that has it now", () => {
    page('<button id="a">Fit</button><button id="b">Thumb</button>');
    document.getElementById("a")?.focus();
    rememberFocus("export");
    document.getElementById("b")?.focus();
    expect(restoreFocus("export", () => document.getElementById("a"))).toBeNull();
    expect(document.activeElement?.id).toBe("b");
  });

  it("forgets an earlier visit, and each slot on its own", () => {
    page('<button id="a">A</button><button id="home">Home</button>');
    document.getElementById("a")?.focus();
    rememberFocus("nodes");
    (document.activeElement as HTMLElement).blur();
    rememberFocus("nodes");
    expect(restoreFocus("export", () => document.getElementById("home"))?.id).toBe("home");
    (document.activeElement as HTMLElement).blur();
    expect(restoreFocus("nodes", () => document.getElementById("home"))?.id).toBe("home");
  });

  it("does not use a control that is inert or hidden", () => {
    page('<div inert><button id="a">Fit</button></div><button id="home">Export</button>');
    rememberFocus("export");
    expect(restoreFocus("export", () => document.querySelector<HTMLElement>("#a"))).toBeNull();
  });
});

describe("a focus that is on a control just hidden", () => {
  it("counts as lost, so the focus can be given back", () => {
    page('<div id="area"><button id="inside">Add node</button></div><button id="a">Fit</button>');
    document.getElementById("a")?.focus();
    rememberFocus("nodes");
    document.getElementById("inside")?.focus();
    document.getElementById("area")?.setAttribute("hidden", "");
    expect(restoreFocus("nodes", () => null)?.id).toBe("a");
  });
});
