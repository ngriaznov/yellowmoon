"""The blind page of the A/B of Apple Auto: `index.html` (the same file for every plan) and `plan.js` (the trials, the
picture paths and how many quarter turns each picture is turned by).

The page is one file with inline CSS and JavaScript. It loads nothing from the network, and it never names a variant
or what made a picture: the picture paths are folder codes (`plan.py`), and the font stack does without the CSS name
that would put the word under test in the page. It runs from `file://`, so `plan.js` is a script, not a JSON request.
Answers are kept in the browser's localStorage under the plan id, and exported as `results.json`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Mapping

PAGE_FILE = "index.html"
PLAN_JS_FILE = "plan.js"
RESULTS_FILE = "results.json"
RESULTS_FORMAT = "yellowmoon-ab-results"

PAGE = r"""<!doctype html>
<html lang="en-GB">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="dark">
<title>Which picture do you prefer?</title>
<style>
:root {
  --bg: #161616;
  --panel: #0e0e0e;
  --line: #333333;
  --line-strong: #4a4a4a;
  --text: #ececec;
  --muted: #a3a3a3;
  --accent: #e6d07a;
  --button: #262626;
  --warn: #e0b37a;
}
* { box-sizing: border-box; }
html, body { margin: 0; background: var(--bg); color: var(--text); }
body {
  font: 15px/1.5 system-ui, "Helvetica Neue", Arial, sans-serif;
  min-height: 100vh;
}
header, main { max-width: 1480px; margin: 0 auto; padding-left: 20px; padding-right: 20px; }
header { padding-top: 18px; }
main { padding-top: 6px; padding-bottom: 28px; }
h1 { font-size: 20px; font-weight: 600; margin: 0 0 6px; }
p { margin: 4px 0; color: var(--muted); font-size: 14px; }
kbd {
  font: 12px/1 ui-monospace, Menlo, Consolas, monospace;
  color: var(--text);
  background: #2b2b2b;
  border: 1px solid var(--line-strong);
  border-bottom-width: 2px;
  border-radius: 4px;
  padding: 2px 5px;
}
.progress {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  flex-wrap: wrap;
  gap: 12px 16px;
  margin: 14px 0 10px;
}
#count { font-size: 17px; font-weight: 600; font-variant-numeric: tabular-nums; }
#saved { color: var(--muted); font-size: 13px; }
#notice { min-height: 20px; color: var(--warn); font-size: 13px; }
.stage { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
.side { display: flex; flex-direction: column; gap: 8px; min-width: 0; }
.frame {
  display: block;
  position: relative;
  width: 100%;
  height: clamp(260px, calc(100vh - 380px), 980px);
  padding: 0;
  margin: 0;
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 6px;
  overflow: hidden;
  cursor: zoom-in;
}
/* The script sets each picture's size and turn (fitPicture), so the picture's own box is what is measured. */
.frame img { position: absolute; left: 50%; top: 50%; display: block; }
.flags { display: flex; justify-content: center; }
.flag {
  font: inherit;
  font-size: 13px;
  color: var(--muted);
  background: transparent;
  border: 1px solid var(--line-strong);
  border-radius: 999px;
  padding: 4px 14px;
  cursor: pointer;
}
.flag[aria-pressed="true"] { color: #161616; background: var(--accent); border-color: var(--accent); }
.answers { display: flex; flex-wrap: wrap; justify-content: center; gap: 10px; margin: 12px 0 8px; }
.answer {
  font: inherit;
  color: var(--text);
  background: var(--button);
  border: 1px solid var(--line-strong);
  border-radius: 8px;
  padding: 10px 18px;
  min-width: 190px;
  cursor: pointer;
}
.tools { display: flex; flex-wrap: wrap; justify-content: center; align-items: center; gap: 12px; margin-top: 6px; }
.tool {
  font: inherit;
  font-size: 14px;
  color: var(--text);
  background: transparent;
  border: 1px solid var(--line-strong);
  border-radius: 8px;
  padding: 7px 14px;
  cursor: pointer;
}
.answer:hover, .flag:hover, .tool:hover, .frame:hover { border-color: #777; }
#export-note { color: var(--muted); font-size: 13px; }
#done { display: none; text-align: center; padding: 48px 0 12px; font-size: 16px; }
#full-hint {
  display: none;
  position: fixed;
  left: 0;
  right: 0;
  bottom: 6px;
  text-align: center;
  font-size: 13px;
  color: #9a9a9a;
  pointer-events: none;
  z-index: 6;
}
body.finished .stage, body.finished .answers { display: none; }
body.finished #done { display: block; }
body.full header, body.full .progress, body.full .answers, body.full .tools, body.full #notice { display: none; }
body.full main { max-width: none; padding: 0; }
body.full .stage {
  position: fixed;
  inset: 0;
  gap: 0;
  padding: 0 0 26px;
  background: #000;
  z-index: 5;
}
body.full .side { gap: 4px; background: #000; }
body.full .frame { flex: 1 1 auto; height: auto; min-height: 0; border: 0; border-radius: 0; cursor: zoom-out; }
body.full #full-hint { display: block; }
</style>
</head>
<body>
<header>
  <h1>Which picture do you prefer?</h1>
  <p>Each trial shows two versions of one photo, side by side, in random order. Choose the one you prefer, or say
  that they are about the same. The page does not say which version is which, and nothing is uploaded.</p>
  <p>Left is better: <kbd>&larr;</kbd> or <kbd>A</kbd>. Right is better: <kbd>&rarr;</kbd> or <kbd>L</kbd>.
  About the same: <kbd>&darr;</kbd> or <kbd>S</kbd>. Undo the last answer: <kbd>Z</kbd>.</p>
  <p>Press <kbd>F</kbd> or <kbd>J</kbd> to mark the left or the right picture as needing a fix: it is more than
  about half a stop too dark or too bright, or it has a colour cast you would correct. Click a picture to see both
  full screen. <kbd>Esc</kbd> goes back.</p>
</header>
<main>
  <div class="progress">
    <span id="count">Loading the trials</span>
    <span id="saved"></span>
  </div>
  <div id="notice" role="status"></div>
  <div class="stage" id="stage">
    <div class="side">
      <button type="button" class="frame" id="frame-left" aria-label="Left picture. Click to see both full screen.">
        <img id="img-left" alt="Left picture">
      </button>
      <div class="flags">
        <button type="button" class="flag" id="flag-left" aria-pressed="false">Needs a fix (F)</button>
      </div>
    </div>
    <div class="side">
      <button type="button" class="frame" id="frame-right" aria-label="Right picture. Click to see both full screen.">
        <img id="img-right" alt="Right picture">
      </button>
      <div class="flags">
        <button type="button" class="flag" id="flag-right" aria-pressed="false">Needs a fix (J)</button>
      </div>
    </div>
  </div>
  <div class="answers">
    <button type="button" class="answer" id="btn-left">Left is better (&larr;)</button>
    <button type="button" class="answer" id="btn-same">About the same (&darr;)</button>
    <button type="button" class="answer" id="btn-right">Right is better (&rarr;)</button>
  </div>
  <div class="tools">
    <button type="button" class="tool" id="btn-undo">Undo the last answer (Z)</button>
    <button type="button" class="tool" id="btn-export">Export results</button>
    <span id="export-note"></span>
  </div>
  <p id="done">Every trial is answered. Press Export results and keep the file results.json.</p>
</main>
<div id="full-hint">Press Esc to go back</div>
<script src="plan.js"></script>
<script>
(function () {
  "use strict";

  var plan = window.YM_AB_PLAN;
  var total = 0;
  var cur = -1;
  var state = { answers: {}, order: [], draft: null };
  var storeKey = "";

  function $(id) { return document.getElementById(id); }

  if (!plan || !plan.trials || plan.trials.length === 0) {
    $("count").textContent = "The trials did not load. Keep index.html and plan.js in the same folder.";
    $("btn-export").disabled = true;
    return;
  }
  total = plan.trials.length;
  storeKey = "ym-ab:" + plan.plan_id;
  state = load();
  cur = nextOpen(0);

  function load() {
    var fresh = { answers: {}, order: [], draft: null };
    try {
      var raw = window.localStorage.getItem(storeKey);
      if (!raw) { return fresh; }
      var saved = JSON.parse(raw);
      if (!saved || typeof saved.answers !== "object" || !Array.isArray(saved.order)) { return fresh; }
      var order = saved.order.filter(function (id) {
        return Object.prototype.hasOwnProperty.call(saved.answers, String(id));
      });
      return { answers: saved.answers, order: order, draft: saved.draft || null };
    } catch (err) {
      return fresh;
    }
  }

  function save() {
    var ok = true;
    try {
      window.localStorage.setItem(storeKey, JSON.stringify(state));
    } catch (err) {
      ok = false;
    }
    $("saved").textContent = ok ? "Saved in this browser" : "Not saved: export your results before you close the page";
  }

  function isAnswered(id) {
    return Object.prototype.hasOwnProperty.call(state.answers, String(id));
  }

  function nextOpen(from) {
    for (var i = from; i < total; i++) {
      if (!isAnswered(plan.trials[i].id)) { return i; }
    }
    for (var j = 0; j < from; j++) {
      if (!isAnswered(plan.trials[j].id)) { return j; }
    }
    return -1;
  }

  function draftFor(trial) {
    if (!state.draft || state.draft.id !== trial.id) {
      state.draft = { id: trial.id, left: false, right: false };
    }
    return state.draft;
  }

  function notice(text) {
    $("notice").textContent = text;
    window.setTimeout(function () {
      if ($("notice").textContent === text) { $("notice").textContent = ""; }
    }, 4000);
  }

  function setPicture(img, picture) {
    if (img.getAttribute("data-path") === picture.path) { return; }
    img.setAttribute("data-path", picture.path);
    img.setAttribute("data-turns", String(picture.turns));
    img.src = picture.path;
  }

  function turnsOf(img) {
    var turns = parseInt(img.getAttribute("data-turns"), 10);
    return isNaN(turns) ? 0 : ((turns % 4) + 4) % 4;
  }

  // The frame's inside (without border and padding), exact to the fraction: clientWidth and clientHeight round, and a
  // frame 840.5 px tall would be read as 841 px, one pixel too tall for the picture.
  function innerBox(frame) {
    var rect = frame.getBoundingClientRect();
    var style = window.getComputedStyle(frame);
    function sum(first, second) { return parseFloat(style[first]) + parseFloat(style[second]); }
    return {
      width: rect.width - sum("borderLeftWidth", "borderRightWidth") - sum("paddingLeft", "paddingRight"),
      height: rect.height - sum("borderTopWidth", "borderBottomWidth") - sum("paddingTop", "paddingBottom")
    };
  }

  // Sizes a picture to fit its frame, then turns it clockwise by its quarter turns. A quarter turn swaps the sides of
  // the box the picture is fitted into, so the turned picture fits the frame. The picture's box is its own size, so its
  // bounding box is the picture. Like object-fit: contain, a small picture is enlarged to fit.
  function fitPicture(img) {
    var inside = innerBox(img.parentNode);
    var turns = turnsOf(img);
    var boxW = inside.width;
    var boxH = inside.height;
    if (turns % 2 === 1) {
      var swap = boxW;
      boxW = boxH;
      boxH = swap;
    }
    if (!img.naturalWidth || !img.naturalHeight || boxW <= 0 || boxH <= 0) { return; }
    var scale = Math.min(boxW / img.naturalWidth, boxH / img.naturalHeight);
    img.style.width = Math.floor(img.naturalWidth * scale) + "px";
    img.style.height = Math.floor(img.naturalHeight * scale) + "px";
    img.style.transform = "translate(-50%, -50%) rotate(" + (turns * 90) + "deg)";
  }

  function fitAll() {
    fitPicture($("img-left"));
    fitPicture($("img-right"));
  }

  function preload(path) {
    var image = new Image();
    image.src = path;
  }

  function renderFlags() {
    var draft = cur >= 0 ? draftFor(plan.trials[cur]) : { left: false, right: false };
    $("flag-left").setAttribute("aria-pressed", draft.left ? "true" : "false");
    $("flag-right").setAttribute("aria-pressed", draft.right ? "true" : "false");
  }

  function render() {
    if (cur < 0) {
      document.body.classList.add("finished");
      $("count").textContent = "All " + total + " trials answered";
      return;
    }
    document.body.classList.remove("finished");
    var trial = plan.trials[cur];
    $("count").textContent = "Trial " + trial.id + " of " + total + " · " + state.order.length + " answered";
    setPicture($("img-left"), trial.pictures.left);
    setPicture($("img-right"), trial.pictures.right);
    renderFlags();
    var next = plan.trials[cur + 1];
    if (next) {
      preload(next.pictures.left.path);
      preload(next.pictures.right.path);
    }
  }

  function answer(choice) {
    if (cur < 0) { return; }
    var trial = plan.trials[cur];
    var draft = draftFor(trial);
    state.answers[String(trial.id)] = {
      answer: choice,
      flags: { left: !!draft.left, right: !!draft.right },
      timestamp: new Date().toISOString()
    };
    state.order.push(trial.id);
    state.draft = null;
    cur = nextOpen(0);
    save();
    render();
  }

  function undo() {
    if (state.order.length === 0) {
      notice("There is no answer to undo.");
      return;
    }
    var id = state.order.pop();
    var old = state.answers[String(id)] || { flags: {} };
    delete state.answers[String(id)];
    state.draft = { id: id, left: !!(old.flags && old.flags.left), right: !!(old.flags && old.flags.right) };
    cur = nextOpen(0);
    save();
    render();
  }

  function toggleFlag(side) {
    if (cur < 0) { return; }
    var draft = draftFor(plan.trials[cur]);
    draft[side] = !draft[side];
    save();
    renderFlags();
  }

  function setFull(on) {
    document.body.classList.toggle("full", !!on);
  }

  function exportResults() {
    var rows = Object.keys(state.answers).map(function (key) {
      var a = state.answers[key];
      var flags = a.flags || {};
      return {
        trial: Number(key),
        answer: a.answer,
        flags: { left: !!flags.left, right: !!flags.right },
        timestamp: a.timestamp
      };
    }).sort(function (x, y) { return x.trial - y.trial; });
    var doc = {
      format: "yellowmoon-ab-results",
      version: 1,
      plan_id: plan.plan_id,
      trial_count: total,
      answered: rows.length,
      exported_at: new Date().toISOString(),
      answers: rows
    };
    var blob = new Blob([JSON.stringify(doc, null, 2) + "\n"], { type: "application/json" });
    var url = URL.createObjectURL(blob);
    var link = document.createElement("a");
    link.href = url;
    link.download = "results.json";
    document.body.appendChild(link);
    link.click();
    link.remove();
    window.setTimeout(function () { URL.revokeObjectURL(url); }, 4000);
    $("export-note").textContent = "Exported " + rows.length + " of " + total + " answers.";
  }

  function onKey(event) {
    if (event.ctrlKey || event.metaKey || event.altKey) { return; }
    var key = event.key;
    var handled = true;
    if (key === "ArrowLeft" || key === "a" || key === "A") { answer("left"); }
    else if (key === "ArrowRight" || key === "l" || key === "L") { answer("right"); }
    else if (key === "ArrowDown" || key === "s" || key === "S") { answer("same"); }
    else if (key === "f" || key === "F") { toggleFlag("left"); }
    else if (key === "j" || key === "J") { toggleFlag("right"); }
    else if (key === "z" || key === "Z") { undo(); }
    else if (key === "Escape" && document.body.classList.contains("full")) { setFull(false); }
    else { handled = false; }
    if (handled) { event.preventDefault(); }
  }

  function clicked(button, action) {
    return function () {
      button.blur();
      action();
    };
  }

  $("btn-left").addEventListener("click", clicked($("btn-left"), function () { answer("left"); }));
  $("btn-same").addEventListener("click", clicked($("btn-same"), function () { answer("same"); }));
  $("btn-right").addEventListener("click", clicked($("btn-right"), function () { answer("right"); }));
  $("btn-undo").addEventListener("click", clicked($("btn-undo"), undo));
  $("btn-export").addEventListener("click", clicked($("btn-export"), exportResults));
  $("flag-left").addEventListener("click", clicked($("flag-left"), function () { toggleFlag("left"); }));
  $("flag-right").addEventListener("click", clicked($("flag-right"), function () { toggleFlag("right"); }));
  $("frame-left").addEventListener("click", function () {
    setFull(!document.body.classList.contains("full"));
  });
  $("frame-right").addEventListener("click", function () {
    setFull(!document.body.classList.contains("full"));
  });
  document.addEventListener("keydown", onKey);
  $("img-left").addEventListener("load", fitAll);
  $("img-right").addEventListener("load", fitAll);
  if (typeof window.ResizeObserver === "function") {
    var frameSizes = new window.ResizeObserver(fitAll);
    frameSizes.observe($("frame-left"));
    frameSizes.observe($("frame-right"));
  } else {
    window.addEventListener("resize", fitAll);
  }

  render();
  if (state.order.length > 0) { save(); }
})();
</script>
</body>
</html>
"""


def _trial_pictures(trial: Mapping[str, object]) -> Dict[str, Dict[str, object]]:
    """The two pictures of a trial for the page: each one's path and its quarter turns clockwise."""
    paths: Mapping[str, str] = trial["pictures"]  # type: ignore[assignment]
    turns: Mapping[str, int] = trial["turns"]  # type: ignore[assignment]
    return {side: {"path": paths[side], "turns": turns[side]} for side in ("left", "right")}


def plan_js(plan: Mapping[str, object]) -> str:
    """The script `plan.js`: the plan id, the number of trials, and each trial's two pictures (path and turns)."""
    trials = plan["trials"]  # type: ignore[index]
    payload = {
        "plan_id": plan["plan_id"],  # type: ignore[index]
        "trial_count": len(trials),  # type: ignore[arg-type]
        "trials": [{"id": t["id"], "pictures": _trial_pictures(t)} for t in trials],  # type: ignore[index]
    }
    return "window.YM_AB_PLAN = " + json.dumps(payload, separators=(",", ":"), ensure_ascii=True) + ";\n"


def write_page(out: Path, plan: Mapping[str, object]) -> None:
    """Writes `index.html` and `plan.js` into the out folder."""
    out.mkdir(parents=True, exist_ok=True)
    (out / PAGE_FILE).write_text(PAGE, encoding="utf-8")
    (out / PLAN_JS_FILE).write_text(plan_js(plan), encoding="utf-8")
