#!/usr/bin/env python3
"""Draws the placeholder pictures of design/skin/ (task S-11, design step): neutral illustrated portraits, made here, no photograph.

The repository holds no licence-clean portrait photograph, so the mockups use drawings: a soft studio backdrop and stylised heads with a few
blemishes, one mole and faint under-eye shadows, so that the overlays have something to point at. Nothing here is a person's likeness.
Everything is in scene units (1500 x 1000); `face()` draws one head at (cx, cy) with head height h. Run via build.py (it calls `write_assets`).
"""
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 1500, 1000

TONES = {  # skin base, light, shade
    "t1": ("#f0c8a8", "#f8dcc4", "#d9a688"),
    "t2": ("#e0a878", "#eec294", "#c48a5c"),
    "t3": ("#b97b52", "#cf9468", "#965e3c"),
    "t4": ("#8a5636", "#a56d48", "#6c3f26"),
    "t5": ("#5f3a26", "#7a4d33", "#46281a"),
    "t6": ("#ecbfa0", "#f6d6bc", "#d2a083"),
}
HEAD = "M0,-0.5 C0.30,-0.5 0.38,-0.26 0.372,0 C0.362,0.27 0.2,0.5 0,0.5 C-0.2,0.5 -0.362,0.27 -0.372,0 C-0.38,-0.26 -0.30,-0.5 0,-0.5 Z"


def _defs(uid, tone, hair):
    base, light, shade = TONES[tone]
    return (
        f'<radialGradient id="sk{uid}" cx="0.42" cy="0.38" r="0.75"><stop offset="0" stop-color="{light}"/><stop offset="0.6" stop-color="{base}"/><stop offset="1" stop-color="{shade}"/></radialGradient>'
        f'<linearGradient id="nk{uid}" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="{shade}"/><stop offset="1" stop-color="{base}"/></linearGradient>'
        f'<filter id="bl{uid}" x="-20%" y="-20%" width="140%" height="140%"><feGaussianBlur stdDeviation="0.012"/></filter>'
        f'<filter id="bm{uid}" x="-30%" y="-30%" width="160%" height="160%"><feGaussianBlur stdDeviation="0.03"/></filter>'
    )


def face(uid, cx, cy, h, tone="t1", hair="#3a2a20", style="short", cloth="#4a5a6a", tilt=0, marks=True, shirt_y=0.62):
    """One head, neck and shoulders. Returns (defs, group)."""
    base, light, shade = TONES[tone]
    hairp = {
        "short": f'<path d="M-0.40,-0.05 C-0.44,-0.42 -0.2,-0.60 0.02,-0.58 C0.26,-0.58 0.44,-0.40 0.40,-0.05 C0.36,-0.2 0.28,-0.30 0.2,-0.32 C0.05,-0.26 -0.18,-0.30 -0.3,-0.20 C-0.36,-0.15 -0.38,-0.10 -0.40,-0.05 Z" fill="{hair}"/>',
        "long": f'<path d="M-0.42,0.05 C-0.52,-0.45 -0.2,-0.62 0.03,-0.60 C0.28,-0.60 0.5,-0.42 0.42,0.05 C0.40,-0.18 0.30,-0.31 0.18,-0.34 C0.02,-0.22 -0.2,-0.30 -0.30,-0.20 C-0.37,-0.14 -0.40,-0.05 -0.42,0.05 Z" fill="{hair}"/>',
        "bun": f'<ellipse cx="0" cy="-0.64" rx="0.13" ry="0.1" fill="{hair}"/><path d="M-0.39,-0.04 C-0.43,-0.40 -0.2,-0.56 0.02,-0.55 C0.26,-0.55 0.43,-0.40 0.39,-0.04 C0.34,-0.22 0.22,-0.32 0.1,-0.34 C-0.05,-0.30 -0.2,-0.32 -0.3,-0.2 C-0.35,-0.14 -0.38,-0.08 -0.39,-0.04 Z" fill="{hair}"/>',
        "curly": "".join(f'<circle cx="{0.43 * math.cos(a):.3f}" cy="{-0.14 + 0.5 * math.sin(a):.3f}" r="0.11" fill="{hair}"/>' for a in [math.pi * (1 + i / 9) for i in range(10)])
        + f'<path d="M-0.38,-0.06 C-0.4,-0.40 -0.1,-0.5 0.02,-0.5 C0.2,-0.5 0.4,-0.40 0.38,-0.06 C0.3,-0.26 0.2,-0.32 0.0,-0.32 C-0.2,-0.32 -0.32,-0.24 -0.38,-0.06 Z" fill="{hair}"/>',
    }[style]
    back = ""
    if style == "long":
        back = f'<path d="M-0.46,-0.05 C-0.5,0.5 -0.44,0.85 -0.30,0.95 L0.30,0.95 C0.44,0.85 0.5,0.5 0.46,-0.05 Z" fill="{hair}"/>'
    if style == "curly":
        back = f'<ellipse cx="0" cy="-0.1" rx="0.5" ry="0.58" fill="{hair}"/>'
    eyes = "".join(
        f'<g transform="translate({sx * 0.15},-0.06)"><ellipse rx="0.058" ry="0.021" fill="#f4efe9"/><circle r="0.021" fill="#3b2a20"/><circle cx="0.006" cy="-0.006" r="0.006" fill="#fff" opacity=".8"/>'
        f'<path d="M-0.062,-0.002 Q0,-0.03 0.062,-0.002" fill="none" stroke="#2a1d17" stroke-width="0.006" stroke-linecap="round"/></g>'
        f'<path d="M{sx * 0.15 - 0.075},-0.125 Q{sx * 0.15},-0.152 {sx * 0.15 + 0.075},-0.122" fill="none" stroke="{hair}" stroke-width="0.016" stroke-linecap="round"/>'
        f'<ellipse cx="{sx * 0.15}" cy="-0.012" rx="0.09" ry="0.032" fill="#7a4a46" opacity="0.22" filter="url(#bl{uid})"/>'
        for sx in (-1, 1)
    )
    spots = ""
    if marks:
        spots = (
            f'<g filter="url(#bl{uid})"><circle cx="-0.20" cy="0.13" r="0.011" fill="#b5483f" opacity=".55"/><circle cx="0.24" cy="0.06" r="0.009" fill="#b5483f" opacity=".5"/>'
            f'<circle cx="0.05" cy="-0.27" r="0.008" fill="#b5483f" opacity=".5"/><circle cx="-0.06" cy="0.40" r="0.01" fill="#b5483f" opacity=".5"/></g>'
            f'<circle cx="0.22" cy="0.16" r="0.0105" fill="#4a2a1c" opacity=".85"/>'
        )
    g = (
        f'<g transform="translate({cx},{cy}) rotate({tilt}) scale({h})">'
        f'{back}'
        f'<path d="M-0.15,0.3 L-0.15,0.8 L0.15,0.8 L0.15,0.3 Z" fill="url(#nk{uid})"/>'
        f'<path d="M-1.0,1.5 C-0.98,0.95 -0.7,{shirt_y + 0.2} -0.16,{shirt_y + 0.08} Q0,{shirt_y + 0.2} 0.16,{shirt_y + 0.08} C0.7,{shirt_y + 0.2} 0.98,0.95 1.0,1.5 Z" fill="{cloth}"/>'
        f'<ellipse cx="-0.375" cy="0.0" rx="0.045" ry="0.08" fill="{base}"/><ellipse cx="0.375" cy="0.0" rx="0.045" ry="0.08" fill="{base}"/>'
        f'<path d="{HEAD}" fill="url(#sk{uid})"/>'
        f'<ellipse cx="-0.2" cy="0.14" rx="0.08" ry="0.05" fill="#d86a5a" opacity=".16" filter="url(#bm{uid})"/><ellipse cx="0.2" cy="0.14" rx="0.08" ry="0.05" fill="#d86a5a" opacity=".16" filter="url(#bm{uid})"/>'
        f'<path d="M0.02,-0.04 C0.035,0.05 0.04,0.1 0.055,0.14 C0.03,0.165 -0.03,0.165 -0.055,0.14" fill="none" stroke="{shade}" stroke-width="0.012" stroke-linecap="round" opacity=".6"/>'
        f'{eyes}{spots}'
        f'<path d="M-0.075,0.27 Q0,0.31 0.075,0.27 Q0,0.285 -0.075,0.27 Z" fill="#b3524f" stroke="#a24544" stroke-width="0.008" stroke-linejoin="round"/>'
        f'{hairp}</g>'
    )
    return _defs(uid, tone, hair), g


def backdrop(uid, c1, c2):
    bokeh = "".join(
        f'<circle cx="{x}" cy="{y}" r="{r}" fill="#fff" opacity="{o}"/>'
        for x, y, r, o in [(120, 130, 70, .10), (1380, 160, 90, .08), (700, 70, 50, .07), (1250, 880, 110, .07), (200, 820, 80, .06)]
    )
    return (
        f'<defs><linearGradient id="bg{uid}" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="{c1}"/><stop offset="1" stop-color="{c2}"/></linearGradient>'
        f'<filter id="bb{uid}"><feGaussianBlur stdDeviation="14"/></filter></defs>'
        f'<rect width="{W}" height="{H}" fill="url(#bg{uid})"/><g filter="url(#bb{uid})">{bokeh}</g>'
    )


# ---- the three scenes. Each face: (cx, cy, h, tone, hair, style, cloth, tilt)
FAMILY = [  # 4 faces and a small one in the back, numbered left to right
    (250, 470, 330, "t6", "#6b4a2a", "long", "#5c6f7a", -3),
    (590, 450, 350, "t2", "#2a1d17", "short", "#7a4f3a", 2),
    (930, 520, 300, "t4", "#1f1712", "curly", "#cdb25a", -2),
    (1240, 450, 340, "t1", "#8a6a3a", "bun", "#35524a", 3),
    (1428, 250, 84, "t3", "#241a14", "short", "#5a5a66", 0),
]
PORTRAIT = [(750, 470, 640, "t3", "#2a1d17", "long", "#44525e", 0)]
GROUP12 = [
    (190 + 370 * (i % 4) + (20 if (i // 4) % 2 else 0), 190 + 310 * (i // 4), 220, ["t1", "t2", "t3", "t4", "t5", "t6"][(i * 5) % 6],
     ["#2a1d17", "#6b4a2a", "#1f1712", "#8a6a3a", "#3a2a20"][i % 5], ["short", "long", "bun", "curly"][(i * 3) % 4],
     ["#5c6f7a", "#7a4f3a", "#35524a", "#8a5a6a", "#4a4a5a"][i % 5], (i % 3) - 1)
    for i in range(12)
]
SCENES = {"family": (FAMILY, "#c9b8a6", "#7e8a96"), "portrait": (PORTRAIT, "#d5c3ad", "#8a8f98"), "group12": (GROUP12, "#cdbfae", "#7a8590")}


def scene_svg(name):
    faces, c1, c2 = SCENES[name]
    defs, body = [], []
    # the small and far faces first would be behind; draw by size: smallest first
    order = sorted(range(len(faces)), key=lambda i: faces[i][2])
    for i in order:
        cx, cy, h, tone, hair, style, cloth, tilt = faces[i]
        d, g = face(f"{name}{i}", cx, cy, h, tone, hair, style, cloth, tilt, marks=(h > 150))
        defs.append(d)
        body.append(g)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}"><defs>{"".join(defs)}</defs>'
        f'{backdrop(name, c1, c2)}{"".join(body)}</svg>'
    )


def geom(name, i):
    """(cx, cy, h) of face i in scene units (a head is 0.75 h wide and h high)."""
    f = SCENES[name][0][i]
    return f[0], f[1], f[2]


def outline(name, faces, selected, numbers=True, dim_others=True):
    """The page's overlay: a soft ring on the selected face and a small number tag on each. Scene units; draw over the photo."""
    out = []
    for i in range(len(SCENES[name][0])):
        cx, cy, h = geom(name, i)
        rx, ry = 0.47 * h, 0.62 * h
        sel = i == selected
        if sel:
            out.append(f'<ellipse cx="{cx}" cy="{cy - 0.02 * h}" rx="{rx}" ry="{ry}" fill="none" stroke="#ffc20e" stroke-width="{max(5, 0.012 * h)}" opacity=".95"/>'
                       f'<ellipse cx="{cx}" cy="{cy - 0.02 * h}" rx="{rx}" ry="{ry}" fill="none" stroke="#ffc20e" stroke-width="{max(14, 0.04 * h)}" opacity=".28" filter="url(#glow)"/>')
        else:
            out.append(f'<ellipse cx="{cx}" cy="{cy - 0.02 * h}" rx="{rx}" ry="{ry}" fill="none" stroke="#fff" stroke-width="{max(3, 0.006 * h)}" stroke-dasharray="{max(6, 0.03 * h)} {max(8, 0.04 * h)}" opacity=".5"/>')
        if numbers:
            r = max(15, 0.05 * h)
            tx, ty = cx + rx * 0.72, cy - ry * 0.72
            fill, ink = ("#ffc20e", "#1a1400") if sel else ("rgba(20,20,22,.78)", "#fff")
            out.append(f'<g transform="translate({tx:.0f},{ty:.0f})"><circle r="{r}" fill="{fill}"/><text y="{r * 0.36}" text-anchor="middle" font-family="Figtree,system-ui,sans-serif" font-weight="800" font-size="{r * 1.05}" fill="{ink}">{i + 1}</text></g>')
    return f'<svg class="sk-over" viewBox="0 0 {W} {H}" preserveAspectRatio="xMidYMid meet"><defs><filter id="glow"><feGaussianBlur stdDeviation="9"/></filter></defs>{"".join(out)}</svg>'


def mask_overlay(name, i, neck=False):
    """What the tools act on, for face i: skin (pink), under the eyes (cyan), the even-skin core (green). Scene units."""
    cx, cy, h = geom(name, i)
    skin = f'<path d="{HEAD}" fill-rule="evenodd"/>'
    eyes = "".join(f'<ellipse cx="{s * 0.15}" cy="-0.06" rx="0.07" ry="0.034"/>' for s in (-1, 1))
    under = "".join(f'<ellipse cx="{s * 0.15}" cy="-0.005" rx="0.095" ry="0.036"/>' for s in (-1, 1))
    neckp = '<path d="M-0.15,0.32 L-0.15,0.8 L0.15,0.8 L0.15,0.32 Z" fill="#ff6fd2" opacity=".2"/>' if neck else ""
    return (
        f'<svg class="sk-over" viewBox="0 0 {W} {H}" preserveAspectRatio="xMidYMid meet"><defs><filter id="mb"><feGaussianBlur stdDeviation="0.006"/></filter>'
        f'<clipPath id="skinclip"><path d="{HEAD}"/></clipPath></defs>'
        f'<g transform="translate({cx},{cy}) scale({h})" filter="url(#mb)">{neckp}'
        f'<path d="{HEAD}" fill="#ff6fd2" opacity=".34"/>'
        f'<g fill="#0b0b0c" opacity=".55">{eyes}</g>'
        f'<g fill="#58d4ff" opacity=".6">{under}</g>'
        f'<g transform="scale(0.8)" clip-path="none"><path d="{HEAD}" fill="#9be56a" opacity=".28"/></g>'
        f'<g fill="#0b0b0c" opacity=".35"><ellipse cx="0.02" cy="0.09" rx="0.05" ry="0.1"/></g>'
        f'</g></svg>'
    )


def protect_overlay(name, i):
    """A protect stroke round the mole of face i, and the brush cursor."""
    cx, cy, h = geom(name, i)
    x, y = cx + 0.22 * h, cy + 0.16 * h
    return (
        f'<svg class="sk-over" viewBox="0 0 {W} {H}" preserveAspectRatio="xMidYMid meet"><defs><filter id="pb"><feGaussianBlur stdDeviation="3"/></filter></defs>'
        f'<circle cx="{x}" cy="{y}" r="{0.045 * h}" fill="#6ee7c8" opacity=".45" filter="url(#pb)"/>'
        f'<circle cx="{x + 0.06 * h}" cy="{y + 0.02 * h}" r="{0.05 * h}" fill="none" stroke="#fff" stroke-width="3"/></svg>'
    )


def write_assets():
    for n in SCENES:
        with open(os.path.join(HERE, "assets", f"scene-{n}.svg"), "w", encoding="utf-8") as f:
            f.write(scene_svg(n))
        print("wrote scene-" + n + ".svg")


if __name__ == "__main__":
    write_assets()
