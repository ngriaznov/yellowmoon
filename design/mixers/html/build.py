#!/usr/bin/env python3
"""Builds the pages of design/mixers/html/pages/ (task X-02).

The pages are the DOM and CSS of design/nodes/html/pages/12.html (the app's header, photo cell, toolbar, panel, buttons) with the graph and the
panel rebuilt from the app's own classes (`.ym-gnode`, `.ym-port`, `.ym-marker`, `.ym-nodehead`, `.ym-add-node-menu`, copied from
src/styles/glass-c2.css into mixers.css). Everything is at 100 % zoom. Run: python3 build.py   (no dependencies but bs4).
"""
import math
import os
import re
from bs4 import BeautifulSoup

HERE = os.path.dirname(os.path.abspath(__file__))
NODES = os.path.join(HERE, "..", "..", "nodes", "html", "pages")
A = "../../../nodes/html/assets/"   # the nodes' assets, from a page of ours
M = "../assets/"                    # ours


def read(p):
    with open(p, encoding="utf-8") as f:
        return f.read()


P12 = read(os.path.join(NODES, "12.html"))
P06 = read(os.path.join(NODES, "06.html"))
BODY12 = P12[P12.find("<body"):]
BODY06 = P06[P06.find("<body"):]
HTML_TAG = re.search(r"<html[^>]*>", P12).group(0)
SCRIPT = BODY12[BODY12.find("<script"):BODY12.find("</script>") + 9]
SCRIPT = SCRIPT.replace(
    "var b = t.getBoundingClientRect(), w = m.getBoundingClientRect();",
    "var b = t.getBoundingClientRect(); if (m.hasAttribute('data-menu-match')) m.style.width = b.width + 'px'; var w = m.getBoundingClientRect();",
)

# ------------------------------------------------------------------------------------------------------------------------------ icons
LUC = 'aria-hidden="true" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="2" viewBox="0 0 24 24"'


def svg(inner, size=16, extra=""):
    return f'<svg {LUC} width="{size}" height="{size}" {extra}>{inner}</svg>'


ICON = {
    "corrector": '<line x1="4" x2="4" y1="21" y2="14"/><line x1="4" x2="4" y1="10" y2="3"/><line x1="12" x2="12" y1="21" y2="12"/><line x1="12" x2="12" y1="8" y2="3"/><line x1="20" x2="20" y1="21" y2="16"/><line x1="20" x2="20" y1="12" y2="3"/><line x1="2" x2="6" y1="14" y2="14"/><line x1="10" x2="14" y1="8" y2="8"/><line x1="18" x2="22" y1="16" y2="16"/>',
    "film": '<rect width="18" height="18" x="3" y="3" rx="2"/><path d="M7 3v18"/><path d="M3 7.5h4"/><path d="M3 12h18"/><path d="M3 16.5h4"/><path d="M17 3v18"/><path d="M17 7.5h4"/><path d="M17 16.5h4"/>',
    "parallel": '<path d="m6 17 5-5-5-5"/><path d="m13 17 5-5-5-5"/>',
    "layer": '<path d="M12.83 2.18a2 2 0 0 0-1.66 0L2.6 6.08a1 1 0 0 0 0 1.83l8.58 3.91a2 2 0 0 0 1.66 0l8.58-3.9a1 1 0 0 0 0-1.83z"/><path d="M2 12a1 1 0 0 0 .58.91l8.6 3.91a2 2 0 0 0 1.65 0l8.58-3.9A1 1 0 0 0 22 12"/><path d="M2 17a1 1 0 0 0 .58.91l8.6 3.91a2 2 0 0 0 1.65 0l8.58-3.9A1 1 0 0 0 22 17"/>',
    "split-menu": '<path d="M16 3h5v5"/><path d="M8 3H3v5"/><path d="M12 22v-8.3a4 4 0 0 0-1.172-2.872L3 3"/><path d="m15 9 6-6"/>',
    # the glyphs X-02 draws for the cards: one line in and three out, three in and one out (the ends are dots, the lines meet at a point)
    "splitter": '<path d="M3 12h5"/><path d="M8 12c3 0 3.5-7 8-7h5"/><path d="M8 12h13"/><path d="M8 12c3 0 3.5 7 8 7h5"/>',
    "combiner": '<path d="M21 12h-5"/><path d="M16 12c-3 0-3.5-7-8-7H3"/><path d="M16 12H3"/><path d="M16 12c-3 0-3.5 7-8 7H3"/>',
    "lock": '<rect width="18" height="11" x="3" y="11" rx="2" ry="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/>',
    "check": '<path d="M20 6 9 17l-5-5"/>',
    "crop": '<path d="M6 2v14a2 2 0 0 0 2 2h14"/><path d="M18 22V8a2 2 0 0 0-2-2H2"/>',
    "nodes": '<circle cx="18" cy="5" r="3"/><circle cx="6" cy="12" r="3"/><circle cx="18" cy="19" r="3"/><line x1="8.59" x2="15.42" y1="13.51" y2="17.49"/><line x1="15.41" x2="8.59" y1="6.51" y2="10.49"/>',
    "before": '<rect width="18" height="18" x="3" y="3" rx="2"/><path d="M12 3v18"/>',
    "grip": '<circle cx="9" cy="12" r="1"/><circle cx="9" cy="5" r="1"/><circle cx="9" cy="19" r="1"/><circle cx="15" cy="12" r="1"/><circle cx="15" cy="5" r="1"/><circle cx="15" cy="19" r="1"/>',
}
SPARKLE = '<svg aria-hidden="true" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9.937 15.5A2 2 0 0 0 8.5 14.063l-6.135-1.582a.5.5 0 0 1 0-.962L8.5 9.936A2 2 0 0 0 9.937 8.5l1.582-6.135a.5.5 0 0 1 .963 0L14.063 8.5A2 2 0 0 0 15.5 9.937l6.135 1.581a.5.5 0 0 1 0 .964L15.5 14.063a2 2 0 0 0-1.437 1.437l-1.582 6.135a.5.5 0 0 1-.963 0z"/><path d="M20 3v4"/><path d="M22 5h-4"/></svg>'
HALF = '<svg aria-hidden="true" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><path d="M12 3a9 9 0 0 1 0 18z" fill="currentColor"/></svg>'

CHANNEL = {"red": ("R", "#ff7a6e"), "green": ("G", "#62d68f"), "blue": ("B", "#6fadff")}
TAIL = "../" * 0

# ------------------------------------------------------------------------------------------------------------------------------ geometry
UNIT_W, UNIT_H = 214, 150
CARD_W, CARD_H = 150, 112
GLYPH_W = 56
PORT_PIC_Y = 56
PORT_KEY_Y = 97


def glyph_h(n_ports):
    """A glyph card's height (spec §5.2): 64 px with two ports, 24 px more for each further one."""
    return 64 + 24 * max(0, n_ports - 2)


def port_ys(n, h):
    """The centres of n ports stacked 24 px apart around the card's middle."""
    return [h / 2 + 24 * (i - (n - 1) / 2) for i in range(n)]


class Graph:
    """Collects cards, markers and links in canvas pixels (100 % zoom) and renders them as the app would."""

    def __init__(self):
        self.html = []
        self.nodes = {}
        self.links = []
        self.draft = None
        self.lit = None  # a function (node, port) -> "lit" | "dim" | None while a link is dragged

    # -- the three kinds of box ------------------------------------------------------------------------------------------------
    def marker(self, key, word, cx, cy, selected=False):
        x, y = cx - 16, cy - 16
        self.nodes[key] = dict(kind="marker", x=x, y=y, w=32, h=32, word=word)
        port = ""
        if key == "input":
            port = f'<span class="ym-port react-flow__handle" data-port="picture" data-side="out" data-linked style="left:32px;top:12px;width:8px;height:8px"><span class="ym-port__glyph ym-port__square"></span></span>'
        else:
            st = self._state(key, "picture")
            port = f'<span class="ym-port react-flow__handle" data-port="picture" data-side="in" data-linked{st} style="left:-10px;top:11px;width:8px;height:10px">{ARROW_SVG}</span>'
        sel = " data-selected" if selected else ""
        self.html.append(
            f'<div style="position:absolute;left:{x}px;top:{y}px;width:32px;height:32px"><div class="ym-marker"{sel} role="button" aria-label="{word}"><span class="ym-marker__dot">{svg(ICON["lock"], 13)}</span><span class="ym-marker__word">{word}</span>{port}</div></div>'
        )

    def corrector(self, key, name, x, y, num, thumb, masks=(), selected=False, bypass=False, dead=False, key_out=False):
        n = dict(kind="corrector", x=x, y=y, w=CARD_W, h=CARD_H, name=name)
        self.nodes[key] = n
        flags = (" data-selected" if selected else "") + (" data-bypass" if bypass else "") + (" data-dead" if dead else "")
        tools = "".join(
            f'<span class="ym-gnode__mask" data-glyph="{g}">{SPARKLE if g == "ai" else HALF}</span>' for g in masks
        ) + ('<span data-glyph="bypass"><svg aria-hidden="true" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="8"/><path d="m6 18 12-12"/></svg></span>' if bypass else "")
        slash = '<span class="ym-gnode__slash"></span>' if bypass else ""
        ports = [
            self._in(key, "picture", -9, PORT_PIC_Y - 5, linked=False),
            f'<span class="ym-port react-flow__handle" data-port="picture" data-side="out" data-linked style="left:146px;top:{PORT_PIC_Y - 4}px;width:8px;height:8px"><span class="ym-port__glyph ym-port__square"></span></span>',
            self._in(key, "key", -9, PORT_KEY_Y - 5, linked=False),
            f'<span class="ym-port react-flow__handle" data-port="key" data-side="out"{" data-linked" if key_out else ""} style="left:146px;top:{PORT_KEY_Y - 4}px;width:8px;height:8px"><span class="ym-port__glyph ym-port__square"></span></span>',
        ]
        self.html.append(
            f'<div class="ym-gnode" style="position:absolute;left:{x}px;top:{y}px"{flags}>'
            f'<span class="ym-gnode__name">{short(name)}</span>'
            f'<div class="ym-gnode__box" role="button" aria-label="Node {num} {name}"><div class="ym-gnode__thumb"><img alt="" src="{thumb}"/>{slash}</div>'
            f'<div class="ym-gnode__foot"><span class="ym-gnode__num">{num}</span><span class="ym-gnode__tools">{tools}</span></div></div>'
            f'<span class="ym-gnode__ring" aria-hidden="true"></span>{"".join(ports)}</div>'
        )

    def glyph(self, key, kind, name, x, mid, n_in=None, plus=None, outs=1, ins_named=None, outs_named=None, selected=False, bypass=False,
              dead=False, mark="letters", label=None):
        """A glyph card. A mixer: n_in linked inputs and a "+" (unless n_in is 8). A Splitter: 1 input, 3 named outputs. A Combiner: 3 named inputs."""
        if kind in ("parallel", "layer"):
            plus = n_in < 8 if plus is None else plus
            n_ports = n_in + (1 if plus else 0)
            h = glyph_h(max(n_ports, 1))
            in_list = [("picture.%d" % (i + 1), None) for i in range(n_in)] + ([("plus", None)] if plus else [])
            out_list = [("picture", None)]
        elif kind == "splitter":
            h = glyph_h(3)
            in_list = [("picture", None)]
            out_list = [(c, c) for c in ("red", "green", "blue")]
        else:
            h = glyph_h(3)
            in_list = [(c, c) for c in ("red", "green", "blue")]
            out_list = [("picture", None)]
        y = mid - h / 2
        n = dict(kind=kind, x=x, y=y, w=GLYPH_W, h=h, name=name, ins={}, outs={})
        in_ys = port_ys(len(in_list), h)
        out_ys = port_ys(len(out_list), h)
        flags = (" data-selected" if selected else "") + (" data-bypass" if bypass else "") + (" data-dead" if dead else "")
        ports, letters = [], []
        for (pn, ch), cy in zip(in_list, in_ys):
            n["ins"][pn] = y + cy
            if pn == "plus":
                st = self._state(key, "plus")
                ports.append(f'<span class="ym-port react-flow__handle ym-port--plus" data-port="plus" data-side="in"{st} style="left:-16px;top:{cy - 8}px;width:15px;height:16px">{PLUS_SVG}</span>')
            else:
                tint = f"--ch:{CHANNEL[ch][1]};" if ch and mark != "none" and mark != "grey" else ""
                cls = " ym-port--ch" if tint else ""
                ports.append(self._in(key, pn, -9, cy - 5, linked=True, extra_style=tint, cls=cls))
                if ch and mark == "letters":
                    letters.append(f'<span class="mx-chl" style="left:7px;top:{cy - 7}px;--ch:{CHANNEL[ch][1]}">{CHANNEL[ch][0]}</span>')
        for (pn, ch), cy in zip(out_list, out_ys):
            n["outs"][pn] = y + cy
            tint = f"--ch:{CHANNEL[ch][1]};" if ch and mark not in ("none", "grey") else ""
            cls = " ym-port--ch" if tint else ""
            ports.append(f'<span class="ym-port react-flow__handle{cls}" data-port="{pn}" data-side="out" data-linked style="left:{GLYPH_W - 4}px;top:{cy - 4}px;width:8px;height:8px;{tint}"><span class="ym-port__glyph ym-port__square"></span></span>')
            if ch and mark == "letters":
                letters.append(f'<span class="mx-chl" style="right:7px;top:{cy - 7}px;--ch:{CHANNEL[ch][1]}">{CHANNEL[ch][0]}</span>')
        self.nodes[key] = n
        slash = '<span class="ym-gnode__slash"></span>' if bypass else ""
        aria = label or name
        self.html.append(
            f'<div class="ym-gnode mx-glyphcard" style="position:absolute;left:{x}px;top:{y}px;width:{GLYPH_W}px;height:{h}px"{flags}>'
            f'<span class="ym-gnode__name">{name}</span>'
            f'<div class="ym-gnode__box mx-glyphbox" data-kind="{kind}" style="width:{GLYPH_W}px;height:{h}px" role="button" aria-label="{aria}">{svg(ICON[kind], 20, "class=\"mx-glyph\"")}{"".join(letters)}{slash}</div>'
            f'<span class="ym-gnode__ring" style="border-radius:20px" aria-hidden="true"></span>{"".join(ports)}</div>'
        )

    # -- ports and links -----------------------------------------------------------------------------------------------------------
    def _state(self, key, port):
        if not self.lit:
            return ""
        s = self.lit(key, port)
        return f' data-state="{s}"' if s else ""

    def _in(self, key, port, left, top, linked, extra_style="", cls=""):
        st = self._state(key, port)
        link = " data-linked" if linked else ""
        return f'<span class="ym-port react-flow__handle{cls}" data-port="{port}" data-side="in"{link}{st} style="left:{left}px;top:{top}px;width:8px;height:10px;{extra_style}">{ARROW_SVG}</span>'

    def out_pt(self, key, port="picture"):
        n = self.nodes[key]
        if n["kind"] == "marker":
            return (n["x"] + 32 + 8, n["y"] + 16)
        if n["kind"] == "corrector":
            return (n["x"] + n["w"] + 4, n["y"] + (PORT_KEY_Y if port == "key" else PORT_PIC_Y))
        return (n["x"] + n["w"] + 4, n["outs"][port])

    def in_pt(self, key, port="picture"):
        n = self.nodes[key]
        if n["kind"] == "marker":
            return (n["x"] - 10, n["y"] + 16)
        if n["kind"] == "corrector":
            return (n["x"] - 9, n["y"] + (PORT_KEY_Y if port == "key" else PORT_PIC_Y))
        if port == "plus":
            return (n["x"] - 16, n["ins"]["plus"])
        return (n["x"] - 9, n["ins"][port])

    def link(self, a, b, a_port="picture", b_port="picture", selected=False):
        self.links.append((self.out_pt(a, a_port), self.in_pt(b, b_port), selected))

    def render(self, w=1040, h=520):
        paths = []
        for (p, q, sel) in self.links:
            paths.append(f'<g class="ym-link mx-link" data-kind="picture"{" data-selected" if sel else ""}><path d="{bez(p, q)}" marker-end="url(#{"mx-arrow-sel" if sel else "mx-arrow"})"/></g>')
        draft = ""
        if self.draft:
            p, q = self.draft
            draft = f'<svg class="ym-draft" data-kind="picture" style="left:0;top:0" width="{w}" height="{h}"><path d="{bez(p, q)}"/></svg>'
        defs = (
            '<defs><marker id="mx-arrow" markerWidth="8" markerHeight="8" refX="7" refY="3.5" orient="auto" markerUnits="userSpaceOnUse"><path class="ym-arrow" d="M0 0 L7 3.5 L0 7 Z"/></marker>'
            '<marker id="mx-arrow-sel" markerWidth="8" markerHeight="8" refX="7" refY="3.5" orient="auto" markerUnits="userSpaceOnUse"><path class="ym-arrow ym-arrow--selected" d="M0 0 L7 3.5 L0 7 Z"/></marker></defs>'
        )
        links = f'<svg class="mx-links" style="position:absolute;left:0;top:0;overflow:visible;pointer-events:none" width="{w}" height="{h}" aria-hidden="true">{defs}{"".join(paths)}</svg>'
        return f'<div class="mx-flow" style="position:absolute;inset:0">{links}{draft}{"".join(self.html)}</div>'


ARROW_SVG = '<svg class="ym-port__glyph" aria-hidden="true" width="8" height="10" viewBox="0 0 8 10"><path d="M0 0 L8 5 L0 10 Z" fill="currentColor"/></svg>'
PLUS_SVG = '<svg class="ym-port__glyph" aria-hidden="true" width="15" height="16" viewBox="0 0 15 16"><path d="M1 1 L14 8 L1 15 Z" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linejoin="round"/><path d="M3.4 8h4.2M5.5 5.9v4.2" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" fill="none"/></svg>'


def short(name):
    letters = list(name)
    return name if len(letters) <= 10 else "".join(letters[:9]).rstrip() + "…"


def bez(p, q):
    """xyflow's horizontal Bezier (curvature 0.25), as bezierOf in graphLayout.ts."""
    dx = q[0] - p[0]
    off = 0.5 * dx if dx >= 0 else 0.25 * 25 * math.sqrt(-dx)
    return f"M{p[0]:.1f},{p[1]:.1f} C{p[0] + off:.1f},{p[1]:.1f} {q[0] - off:.1f},{q[1]:.1f} {q[0]:.1f},{q[1]:.1f}"


# ------------------------------------------------------------------------------------------------------------------------------ page parts
soup12 = BeautifulSoup(BODY12, "html.parser")
soup06 = BeautifulSoup(BODY06, "html.parser")


def bar_parts(soup):
    out = {}
    bar = soup.select_one(".nv2-bar")
    for ch in bar.find_all(recursive=False):
        t = ch.get_text(" ", strip=True)
        if ch.get("class") and "nv-switcher" in ch["class"]:
            out["switcher"] = ch
        elif t.startswith("Add node"):
            out["add"] = ch
        elif t.startswith("Bypass"):
            out["bypass"] = ch
        elif t.startswith("Delete"):
            out["delete"] = ch
        elif t.startswith("Show mask"):
            out["mask"] = ch
        elif t.startswith("Expand"):
            out["expand"] = ch
    return out


BAR12, BAR06 = bar_parts(soup12), bar_parts(soup06)
ZOOM = str(soup12.select_one(".nv2-zoom")).replace("75%", "100%")
FOOT = str(soup12.select_one(".ym-tools__foot"))
HEAD_BTNS = [str(x) for x in soup12.select(".nv-chead__btns > span")]
GRIP = ICON["grip"]


def disable(html, label_start=None):
    """Marks the first button of an HTML piece aria-disabled, as the app does for an off button."""
    b = BeautifulSoup(html, "html.parser")
    btn = b.find("button")
    btn["aria-disabled"] = "true"
    return str(b)


def pressed(html, on):
    b = BeautifulSoup(html, "html.parser")
    btn = b.find("button")
    if btn.has_attr("aria-pressed"):
        btn["aria-pressed"] = "true" if on else "false"
    return str(b)


def prefix_of(body, narrow, fit):
    """The part of a page before the node editor: aurora, header, photo cell, divider."""
    pre = body[: body.find('<section aria-label="Node editor"')].replace('src="../assets/', 'src="' + A)
    if narrow:
        pre = re.sub(r'<div class="ym-screen"[^>]*>', '<div class="ym-screen" data-narrow="true" data-nodes="2" data-panel="tools" data-screen="photos" style="--viewer-h:270px;--photo-w:405px;--photo-h:270px">', pre, count=1)
    else:
        pre = re.sub(r'<div class="ym-screen"[^>]*>', '<div class="ym-screen" data-nodes="2" data-panel="tools" data-screen="photos">', pre, count=1)
    for label, key in (("Before", "before"), ("Crop", "crop"), ("Nodes", "nodes")):
        pre = pre.replace(f'type="button">{label}</button>', f'type="button">{svg(ICON[key], 16)}{label}</button>', 1)
    pre = re.sub(r"Fit \d+%<svg", f"Fit {fit}%<svg", pre, count=1)
    return pre


def editor(graph_html, bar, expand, add_open=False, bypass_off=False, bypass_on=False, mask_off=True, canvas_style=""):
    add = str(bar["add"])
    if add_open:
        add = add.replace('aria-expanded="false"', 'aria-expanded="true"', 1)
    byp = pressed(str(bar["bypass"]), bypass_on)
    if bypass_off:
        byp = disable(byp)
    msk = str(bar["mask"])
    if mask_off:
        msk = disable(msk)
    exp = pressed(str(bar["expand"]), expand)
    return (
        f'<section aria-label="Node editor" class="ym-filmstrip nv2-editor" data-tab="look"><div class="nv2-bar">{add}{byp}{bar["delete"]}'
        f'<span class="nv2-bar__sp"></span>{bar["switcher"]}{msk}{exp}</div>'
        f'<div class="nv2-canvas" style="--cw:1040px;{canvas_style}">{graph_html}{ZOOM}</div></section>'
    )


def head(glyph_key, name, line, bypass="on", reset="on", pressed_on=False):
    btns = []
    for i, b in enumerate(HEAD_BTNS):
        s = BeautifulSoup(b, "html.parser")
        btn = s.find("button")
        btn["class"] = btn.get("class", []) + ["ym-nodehead__btn"]
        if i == 0:
            btn["aria-pressed"] = "true" if pressed_on else "false"
            if bypass == "off":
                btn["aria-disabled"] = "true"
                s.find(class_="sr-only").string = "A splitter or a combiner can't be bypassed."
        if i == 1 and reset == "off":
            btn["aria-disabled"] = "true"
            s.find(class_="sr-only").string = "This node has nothing to reset."
        btns.append(str(s))
    return (
        f'<div class="ym-nodehead" role="group" aria-label="Node"><span class="ym-nodehead__chip">{svg(ICON[glyph_key], 13)}</span>'
        f'<div class="ym-nodehead__text"><h2 class="ym-nodehead__name" title="Double-click to rename">{name}</h2><p class="ym-nodehead__line">{line}</p></div>'
        f'<div class="ym-nodehead__actions" role="group" aria-label="Node actions">{"".join(btns)}</div></div>'
    )


TABLIST = str(soup12.select_one('[data-slot="tabs"]'))


def tabs(label):
    t = TABLIST.replace('data-tab="raw"', 'data-tab="look"')
    t = re.sub(r'<button aria-controls="ym-tabpanel-mask".*?</button>', "", t, flags=re.S)
    t = t.replace(">Blend</button>", f">{label}</button>").replace("ym-tab-raw-blend", "ym-tab-look-only").replace("ym-tabpanel-raw", "ym-tabpanel-look")
    return t


def aside(headhtml, tablabel, cards):
    return (
        f'<aside aria-label="Tools" class="ym-tools" data-tab="look">{headhtml}{tabs(tablabel)}'
        f'<div aria-labelledby="ym-tab-look-only" class="ym-tools__scroll" data-tab="look" id="ym-tabpanel-look" role="tabpanel">{cards}</div>{FOOT}</aside>'
        '<div aria-live="polite" class="sr-only"></div></div></div>'
    )


def thumbs_rows(rows, handle, words):
    """rows: [(input number, thumbnail url or None, 'node number and name', word)] top to bottom."""
    out = []
    for num, th, label, word in rows:
        g = f'<svg aria-hidden="true" class="nv-order__grip" fill="none" height="16" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="2" viewBox="0 0 24 24" width="16">{GRIP}</svg>' if handle else ""
        t = f'<span class="nv-order__thumb"><img alt="" src="{th}"/></span>' if th else ""
        w = f'<span class="nv-order__sub">{word}</span>' if words and word else ""
        out.append(f'<div class="nv-order__row" tabindex="0">{g}{t}<span class="nv-num">{num}</span><span class="nv-order__name">{label}</span>{w}</div>')
    return '<div class="nv-order">' + "".join(out) + "</div>"


NOTE = 'class="ym-row-note" style="margin:0 0 6px;color:var(--glass-label-secondary)"'


def parallel_card(base_line, rows):
    return (
        '<section class="ym-card" data-last="true"><div class="ym-card__header"><h3 class="ym-card__title">Inputs</h3></div>'
        f'<p {NOTE}>Adds the change each input makes to the picture they start from.</p>'
        f'<p class="ym-row-note" style="margin:0 0 14px;color:var(--glass-label)">{base_line}</p>'
        f"{thumbs_rows(rows, False, False)}</section>"
    )


BLEND_CARD = str(soup12.select_one(".ym-tools__scroll > section.ym-card"))
BLEND_CARD = BLEND_CARD.replace("inset-inline-start: 100.00%; width: 0.00%;", "inset-inline-start: 0%; width: 100.00%;")
BLEND_CARD = BLEND_CARD.replace('aria-label="Blend mode"', 'aria-label="Blend mode" id="mx-mode"')


def layer_cards(rows, mode_open=False):
    blend = BLEND_CARD
    if mode_open:
        blend = blend.replace('aria-expanded="false"', 'aria-expanded="true"', 1)
    inputs = (
        '<section class="ym-card" data-last="true"><div class="ym-card__header"><h3 class="ym-card__title">Inputs</h3></div>'
        f'{thumbs_rows(rows, True, True)}<p class="ym-row-note">Drag a row to change which input sits on top.</p></section>'
    )
    return blend + inputs


def channel_rows(items):
    out = []
    for ch, text, linked in items:
        letter, col = CHANNEL[ch]
        sub = f'<span class="nv-order__sub"{"" if linked else " style=\"color:var(--glass-label-tertiary)\""}>{text}</span>'
        out.append(f'<div class="nv-order__row" tabindex="0"><span class="mx-chip" style="--ch:{col}">{letter}</span><span class="nv-order__name">{ch.capitalize()}</span>{sub}</div>')
    return '<div class="nv-order">' + "".join(out) + "</div>"


def channels_card(sentence, items):
    return (
        '<section class="ym-card" data-last="true"><div class="ym-card__header"><h3 class="ym-card__title">Channels</h3></div>'
        f'<p {NOTE.replace("margin:0 0 6px", "margin:0 0 14px")}>{sentence}</p>{channel_rows(items)}</section>'
    )


# ------------------------------------------------------------------------------------------------------------------------------ menus
MENU_CLASS = "glass-liquid z-50 overflow-x-hidden overflow-y-auto overscroll-contain rounded-surface-sm p-1.5 text-popover-foreground glass-materialize"
ITEM_CLASS = ("glass-focus-inset relative flex cursor-default items-center gap-2.5 rounded-control-sm px-2.5 py-1.5 text-sm outline-hidden select-none "
              "focus:bg-fill-strong [&_svg]:pointer-events-none [&_svg]:shrink-0 [&_svg:not([class*='size-'])]:size-4 [&_svg:not([class*='text-'])]:text-muted-foreground")
SEP = '<div class="-mx-1.5 my-1.5 h-px bg-separator" data-slot="dropdown-menu-separator" role="separator"></div>'


def add_node_menu(highlight="Parallel"):
    items = [
        ("Corrector", "corrector", "Grades all or part of the photo."),
        ("Film", "film", "Models a colour negative, a print and a scan."),
        None,
        ("Parallel", "parallel", "Grade branches side by side."),
        ("Layer", "layer", "Blend one branch over another."),
        ("Splitter and combiner", "split-menu", "Split a picture, then join it again."),
    ]
    out = []
    for it in items:
        if it is None:
            out.append(SEP)
            continue
        name, ic, line = it
        hl = ' style="background:color-mix(in oklch, var(--ink) 12%, transparent)"' if name == highlight else ""
        out.append(
            f'<div class="{ITEM_CLASS}" data-slot="dropdown-menu-item" role="menuitem" tabindex="-1"{hl}>{svg(ICON[ic], 24)}'
            f'<span class="ym-menu-line"><span class="ym-menu-line__name">{name}</span><span class="ym-menu-line__text">{line}</span></span></div>'
        )
    return (
        '<div data-radix-popper-content-wrapper="" style="position:fixed;left:12px;bottom:372px;z-index:50;">'
        f'<div aria-label="Add node" class="{MENU_CLASS} ym-add-node-menu" data-slot="dropdown-menu-content" data-state="open" role="menu" '
        f'style="width:356px;outline:none;--glass-liquid-opacity:90%;--glass-liquid-blur:22px">{"".join(out)}</div></div>'
    )


MODES = ["Normal", "Add", "Subtract", "Multiply", "Lighten", "Darken", "Difference"]


def mode_menu(current="Normal"):
    out = []
    for m in MODES:
        sel = m == current
        chk = svg(ICON["check"], 24, 'class="mx-check"') if sel else ""
        hl = ' style="background:color-mix(in oklch, var(--ink) 12%, transparent)"' if m == "Multiply" else ""
        out.append(f'<div class="{ITEM_CLASS}" data-slot="select-item" role="option" aria-selected="{"true" if sel else "false"}" tabindex="-1"{hl}><span style="flex:1;{"font-weight:700" if sel else ""}">{m}</span>{chk}</div>')
    return (
        '<div data-radix-popper-content-wrapper="" data-menu-anchor="#mx-mode" data-menu-match="" style="position:fixed;left:0;top:0;z-index:50;">'
        f'<div aria-label="Blend mode" class="{MENU_CLASS}" data-slot="select-content" data-state="open" role="listbox" style="outline:none;--glass-liquid-opacity:90%;--glass-liquid-blur:22px">{"".join(out)}</div></div>'
    )


def page(title, inner):
    return (
        f'<!doctype html>\n{HTML_TAG}<head>\n<meta charset="utf-8"/>\n<meta content="width=device-width, initial-scale=1.0" name="viewport"/>\n<meta content="dark" name="color-scheme"/>\n'
        f'<title>Yellowmoon · {title}</title>\n<link href="../app.css" rel="stylesheet"/>\n<link href="../../../nodes/html/nv.css" rel="stylesheet"/>\n<link href="../mixers.css" rel="stylesheet"/>\n</head>\n'
        f'{inner}\n{SCRIPT}</body></html>\n'
    )


def write(name, html):
    with open(os.path.join(HERE, "pages", name), "w", encoding="utf-8") as f:
        f.write(html)
    print("wrote", name, len(html))


# ------------------------------------------------------------------------------------------------------------------------------ the pages
def ach(n):
    return A + n


X0 = 170  # where Main's card starts; one grid unit is 214 px


def common_main(g, mid, selected=False, x=X0, thumb=None):
    g.corrector("main", "Main", x, mid - PORT_PIC_Y, "01", thumb or ach("node-main.jpg"), selected=selected)


# ---- 01 the Add node menu --------------------------------------------------------------------------------------------------
def page01():
    g = Graph()
    mid = 160
    xs = [120, 334, 548, 762]
    g.marker("input", "Input", xs[0] - 86, mid)
    g.corrector("main", "Main", xs[0], mid - 56, "01", ach("node-main.jpg"))
    g.corrector("mountain", "Mountain", xs[1], mid - 56, "02", ach("node-mountain.jpg"), masks=("ai", "mask"), selected=True)
    g.corrector("sky", "Sky", xs[2], mid - 56, "03", ach("node-sky.jpg"), masks=("mask",))
    g.corrector("warmth", "Warmth", xs[3], mid - 56, "04", ach("node-warmth.jpg"))
    g.marker("output", "Output", xs[3] + 150 + 86, mid)
    for a, b in (("input", "main"), ("main", "mountain"), ("mountain", "sky"), ("sky", "warmth"), ("warmth", "output")):
        g.link(a, b)
    pre = prefix_of(BODY06, False, 28)
    aside06 = BODY06[BODY06.find("<aside"): BODY06.find("</aside>") + 8].replace('src="../assets/', 'src="' + A)
    tail = '<div aria-live="polite" class="sr-only"></div></div></div>'
    inner = pre + editor(g.render(h=327), BAR06, False, add_open=True, mask_off=False) + aside06 + tail + add_node_menu("Parallel")
    write("01.html", page("Add node menu", inner))


def tabs_page(name, title, g, mid_unused, headhtml, tablabel, cards, extra="", **kw):
    inner = prefix_of(BODY12, True, 17) + editor(g.render(), BAR12, True, **kw) + aside(headhtml, tablabel, cards) + extra
    write(name, page(title, inner))


# ---- 02 Parallel -------------------------------------------------------------------------------------------------------------
def build_parallel(g, with_selected=True, lit=None):
    mid = 250
    g.marker("input", "Input", X0 - 86, mid)
    g.corrector("main", "Main", X0, mid - 56, "01", ach("node-main.jpg"))
    g.corrector("skin", "Skin", X0 + UNIT_W, mid - 75 - 56, "02", ach("node-skin.jpg"), masks=("mask",))
    g.corrector("sky", "Sky", X0 + UNIT_W, mid + 75 - 56, "03", ach("node-sky.jpg"), masks=("mask",))
    g.glyph("p", "parallel", "Parallel", X0 + 2 * UNIT_W, mid, n_in=2, selected=with_selected, label="Parallel, mixer with 2 inputs, from node 2 and node 3")
    g.marker("output", "Output", X0 + 2 * UNIT_W + GLYPH_W + 86, mid)


def page02():
    g = Graph()
    build_parallel(g)
    for a, b, bp in (("input", "main", "picture"), ("main", "skin", "picture"), ("main", "sky", "picture"), ("skin", "p", "picture.1"), ("sky", "p", "picture.2"), ("p", "output", "picture")):
        g.link(a, b, b_port=bp)
    rows = [(1, ach("node-skin.jpg"), "02 Skin", None), (2, ach("node-sky.jpg"), "03 Sky", None)]
    cards = parallel_card("They start from node 1, Main.", rows)
    tabs_page("02.html", "Parallel", g, 0, head("parallel", "Parallel", "Mixer · 2 inputs"), "Inputs", cards, bypass_on=False)


# ---- 03 / 04 Layer -----------------------------------------------------------------------------------------------------------
def build_layer(g):
    mid = 250
    g.marker("input", "Input", X0 - 86, mid)
    g.corrector("main", "Main", X0, mid - 56, "01", ach("node-main.jpg"))
    g.corrector("warmth", "Warmth", X0 + UNIT_W, mid - 150 - 56, "02", ach("node-warmth2.jpg"))
    g.corrector("glow", "Glow", X0 + UNIT_W, mid - 56, "03", ach("node-glow.jpg"), masks=("mask",))
    g.corrector("sky", "Sky", X0 + UNIT_W, mid + 150 - 56, "04", ach("node-sky.jpg"), masks=("mask",))
    g.glyph("l", "layer", "Layer", X0 + 2 * UNIT_W, mid, n_in=3, selected=True, label="Layer, mixer with 3 inputs, from node 2, node 3 and node 4")
    g.marker("output", "Output", X0 + 2 * UNIT_W + GLYPH_W + 86, mid)
    for a, b, bp in (("input", "main", "picture"), ("main", "warmth", "picture"), ("main", "glow", "picture"), ("main", "sky", "picture"),
                     ("warmth", "l", "picture.1"), ("glow", "l", "picture.2"), ("sky", "l", "picture.3"), ("l", "output", "picture")):
        g.link(a, b, b_port=bp)


LAYER_ROWS = [(3, ach("node-sky.jpg"), "04 Sky", "On top"), (2, ach("node-glow.jpg"), "03 Glow", ""), (1, ach("node-warmth2.jpg"), "02 Warmth", "Below")]


def page03():
    g = Graph()
    build_layer(g)
    tabs_page("03.html", "Layer blend", g, 0, head("layer", "Layer", "Mixer · 3 inputs"), "Blend", layer_cards(LAYER_ROWS))


def page04():
    g = Graph()
    build_layer(g)
    tabs_page("04.html", "Layer mode menu", g, 0, head("layer", "Layer", "Mixer · 3 inputs"), "Blend", layer_cards(LAYER_ROWS, True), extra=mode_menu("Normal"))


# ---- 05 / 06 Splitter and combiner -----------------------------------------------------------------------------------------
def build_trio(g, sel, mark="letters"):
    mid = 250
    g.marker("input", "Input", X0 - 86, mid)
    g.corrector("main", "Main", X0, mid - 56, "01", ach("node-main.jpg"))
    g.glyph("s", "splitter", "Splitter", X0 + UNIT_W, mid, selected=(sel == "s"), mark=mark, label="Splitter, picture from node 1")
    g.corrector("r", "Red", X0 + 2 * UNIT_W, mid - 150 - 56, "02", M + "node-red-channel.jpg")
    g.corrector("g", "Green", X0 + 2 * UNIT_W, mid - 56, "03", M + "node-green-channel.jpg")
    g.corrector("b", "Blue", X0 + 2 * UNIT_W, mid + 150 - 56, "04", M + "node-blue-channel.jpg")
    g.glyph("c", "combiner", "Combiner", X0 + 3 * UNIT_W, mid, selected=(sel == "c"), mark=mark, label="Combiner, red from node 2, green from node 3, blue from node 4")
    g.marker("output", "Output", X0 + 3 * UNIT_W + GLYPH_W + 86, mid)
    g.link("input", "main")
    g.link("main", "s")
    for k, c in (("r", "red"), ("g", "green"), ("b", "blue")):
        g.link("s", k, a_port=c)
        g.link(k, "c", b_port=c)
    g.link("c", "output")


def page05():
    g = Graph()
    build_trio(g, "s")
    cards = channels_card(
        "Sends the red, green and blue of the picture to three outputs. Each output is a grey picture of one channel, so the nodes after it grade that channel.",
        [("red", "To node 2, Red", True), ("green", "To node 3, Green", True), ("blue", "To node 4, Blue", True)],
    )
    tabs_page("05.html", "Splitter and combiner", g, 0, head("splitter", "Splitter", "Channels · 3 outputs", bypass="off", reset="off"), "Channels", cards, bypass_off=True)


def page06():
    g = Graph()
    build_trio(g, "c")
    cards = channels_card(
        "Makes one picture: the red of its red input, the green of its green input and the blue of its blue input.",
        [("red", "From node 2, Red", True), ("green", "From node 3, Green", True), ("blue", "From node 4, Blue", True)],
    )
    tabs_page("06.html", "Combiner", g, 0, head("combiner", "Combiner", "Channels · 3 inputs", bypass="off", reset="off"), "Channels", cards, bypass_off=True)


# ---- 07 dragging a link onto "+" ---------------------------------------------------------------------------------------------
def page07():
    g = Graph()
    # lit inputs while a picture link is dragged from Sky's output: every picture input that is not upstream of Sky; never a key port
    dim_nodes = {("main", "picture"), ("sky", "picture")}

    def lit(key, port):
        if port == "key":
            return "dim"
        if (key, port) in dim_nodes:
            return "dim"
        return "lit"

    g.lit = lit
    xs = [X0 + i * UNIT_W for i in range(4)]
    g.marker("input", "Input", X0 - 86, 250)
    g.corrector("main", "Main", xs[0], 250 - 56, "01", ach("node-main.jpg"))
    g.corrector("skin", "Skin", xs[1], 105 - 56, "02", ach("node-skin.jpg"), masks=("mask",))
    g.corrector("sky", "Sky", xs[1], 250 - 56, "03", ach("node-sky.jpg"), masks=("mask",))
    g.corrector("rest", "Rest", xs[1], 395 - 56, "04", ach("node-rest.jpg"))
    g.glyph("p", "parallel", "Parallel", xs[2], 177, n_in=2, selected=True, label="Parallel, mixer with 2 inputs, from node 2 and node 3")
    g.glyph("c", "combiner", "Combiner", xs[3], 250, mark="letters", label="Combiner, red from Parallel, green from node 4, blue from the input")
    g.marker("output", "Output", xs[3] + GLYPH_W + 86, 250)
    for a, b, bp in (("input", "main", "picture"), ("main", "skin", "picture"), ("main", "sky", "picture"), ("main", "rest", "picture"),
                     ("skin", "p", "picture.1"), ("sky", "p", "picture.2"), ("p", "c", "red"), ("rest", "c", "green"), ("c", "output", "picture")):
        g.link(a, b, b_port=bp)
    # the dragged link ends on the "+" of the Parallel
    g.draft = (g.out_pt("sky"), (g.in_pt("p", "plus")[0] + 2, g.in_pt("p", "plus")[1]))
    # the "+" is under the pointer
    g.html = [h.replace('data-port="plus" data-side="in" data-state="lit"', 'data-port="plus" data-side="in" data-state="lit" data-hover') for h in g.html]
    rows = [(1, ach("node-skin.jpg"), "02 Skin", None), (2, ach("node-sky.jpg"), "03 Sky", None)]
    cards = parallel_card("They start from node 1, Main.", rows)
    inner = prefix_of(BODY12, True, 17) + editor(g.render().replace('class="mx-flow"', 'class="mx-flow ym-flow" data-linking=""'), BAR12, True) + aside(head("parallel", "Parallel", "Mixer · 2 inputs"), "Inputs", cards)
    write("07.html", page("Link onto plus", inner))


# ---- states ------------------------------------------------------------------------------------------------------------------
def tip(text, cls=""):
    return f'<span class="mx-tip glass-liquid {cls}" style="--glass-liquid-opacity:90%;--glass-liquid-blur:22px">{text}</span>'


def sheet_cell(g_fn, caption, w=170, h=260, mid=None, cap_note=""):
    g = Graph()
    mid = mid if mid is not None else h / 2 + 8
    g_fn(g, mid)
    body = g.render(w, h)
    return f'<figure class="mx-cell" style="width:{w}px"><div class="mx-cell__stage" style="height:{h}px">{body}</div><figcaption>{caption}{("<small>" + cap_note + "</small>") if cap_note else ""}</figcaption></figure>'


def states_page():
    cells_p, cells_l = [], []
    for kind, bag in (("parallel", cells_p), ("layer", cells_l)):
        nm = kind.capitalize()
        specs = [
            ("Normal", dict(n_in=2)),
            ("Selected", dict(n_in=2, selected=True)),
            ("Bypassed", dict(n_in=2, bypass=True)),
            ("Dead", dict(n_in=2, dead=True)),
            ("1 input", dict(n_in=1)),
            ("8 inputs", dict(n_in=8)),
        ]
        for cap, kw in specs:
            def fn(g, mid, kw=kw, kind=kind, nm=nm):
                g.glyph("x", kind, nm, 47, mid, **kw)
            note = {"Normal": "2 inputs and the +", "1 input": "1 input and the +", "8 inputs": "no +", "Dead": "45 %"}.get(cap, "")
            bag.append(sheet_cell(fn, cap, 150, 260, cap_note=note))
    cells_s = []
    for cap, kw in (("Normal", {}), ("Selected", dict(selected=True)), ("Dead", dict(dead=True))):
        cells_s.append(sheet_cell(lambda g, mid, kw=kw: g.glyph("x", "splitter", "Splitter", 47, mid, mark="letters", **kw), "Splitter · " + cap, 150, 200))
    for cap, kw in (("Normal", {}), ("Selected", dict(selected=True)), ("Dead", dict(dead=True))):
        cells_s.append(sheet_cell(lambda g, mid, kw=kw: g.glyph("x", "combiner", "Combiner", 47, mid, mark="letters", **kw), "Combiner · " + cap, 150, 200))
    cells_m = []
    for cap, mk, note in (("A · plain ports", "grey", "all pictures are grey; the port's place says the channel"),
                          ("B · tinted ports", "tint", "colour only"),
                          ("C · tinted ports and letters", "letters", "proposed: the letter does not rely on colour")):
        for kind in ("splitter", "combiner"):
            cells_m.append(sheet_cell(lambda g, mid, kind=kind, mk=mk: g.glyph("x", kind, kind.capitalize(), 47, mid, mark=mk), kind.capitalize() + " · " + cap, 150, 170, cap_note=note if kind == "splitter" else ""))
    cells_m = cells_m
    tips = [
        ("Dead card", "Not connected to the output."),
        ("Reorder (Move earlier, Move later, ⌥← and ⌥→)", "This node can't be moved along a line. Change its links instead."),
        ("Bypass on a Splitter or a Combiner (the toolbar, B, the head)", "A splitter or a combiner can't be bypassed."),
        ("Reset on a Parallel, a Splitter or a Combiner", "This node has nothing to reset."),
        ("Show mask on any of the four", "This node has no mask."),
        ("Add node at 64 nodes (all items; Splitter and combiner from 60)", "A photo can have up to 64 nodes."),
    ]
    tip_html = "".join(f'<div class="mx-tiprow"><span>{a}</span>{tip(b)}</div>' for a, b in tips)
    inner = (
        '<body><div class="mx-sheet">'
        '<h1 class="mx-sheet__title">New cards · states</h1>'
        '<p class="mx-sheet__sub">At 100 % zoom. A glyph card is 56 px wide; its height is 64 px with two ports and 24 px more for each further port.</p>'
        f'<section><h2>Parallel</h2><div class="mx-row">{"".join(cells_p)}</div></section>'
        f'<section><h2>Layer</h2><div class="mx-row">{"".join(cells_l)}</div></section>'
        f'<section><h2>Splitter and combiner</h2><div class="mx-row">{"".join(cells_s)}</div></section>'
        f'<section><h2>Channel ports: three options for the owner</h2><div class="mx-row">{"".join(cells_m)}</div></section>'
        f'<section><h2>Tooltips of the refusals</h2><div class="mx-tips">{tip_html}</div></section>'
        "</div>"
    )
    write("states.html", page("Card states", inner))


if __name__ == "__main__":
    page01()
    page02()
    page03()
    page04()
    page05()
    page06()
    page07()
    states_page()
