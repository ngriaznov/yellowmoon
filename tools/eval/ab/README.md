# Blind A/B of Apple Auto (docs/19 section 13)

This kit answers one question with your own photos: do you prefer the pictures Apple Auto makes? You look at two
versions of the same photo, side by side, without being told which is which, and pick one. The kit then scores your
picks against the gate that decides whether Apple Auto stays on by default (AA-1 in `docs/STATUS.md`).

The two versions of each photo:

- **Apple Auto**: Apple's on-device model looks at the photo and grades it: exposure, highlights, shadows, black
  level, contrast, warmth, tint, saturation and the colours of the shadows and highlights (`docs/19`, prompt version
  2). This is the picture under test.
- **The rule**: Apple Auto off, Yellowmoon's own Auto, the same on every computer.

Both are made at the defaults, as a new photo opens in the app: the built-in default look and every setting at Auto.
Each photo gives one trial. With `--with-camera` a photo gives a second trial, Apple Auto against **the camera's own
JPEG**; that comparison is reported, but it is not part of the gate.

Two kinds of photo give no trial, and the kit counts both:

- a photo where Apple's answer **changed nothing** (every value it set was "keep" or "none", or its picture came out
  the same as the rule's), because there would be nothing to choose between;
- a photo where Apple's answer **failed** (for example no answer within 30 seconds), because its picture is the rule's.

**The gate** passes when, for Apple Auto against the rule:

- Apple Auto wins at least 60 % of the trials (a tie counts as half a win);
- the lower end of the 95 % Wilson interval of the win share is above 50 % (ties are left out of this one);
- no stratum (a kind of photo, see below) has a win share below 45 %;
- Apple Auto's picture is not marked "needs a fix" more often than the rule's.

These are **provisional thresholds** (`docs/19` section 13): they are the numbers of the earlier `autoset` test, and
they need your agreement. If you want others, say so; the score keeps every number, so the gate can be read again.

Nothing is uploaded. The renders, the page and the answers stay on your Mac.

## What you need

- A Mac with Apple silicon and macOS 27, with Apple Intelligence on in System Settings and its model downloaded.
- This repository on the Mac, on the branch `claude/apple-auto`, for the kit itself. Rust is not needed.
- Python 3.9 or newer (`python3 --version`; the Xcode command line tools provide it).
- About 150 fresh Sony ARW files (step 3).
- About 1 GB of free disk for 150 photos (two renders each, and a copy of the camera's JPEG with `--with-camera`).

## 1. Install the test build

1. On GitHub, open **Actions › Build › Run workflow**, choose the branch `claude/apple-auto` and the architecture
   `aarch64`, and run it (or use the run the lead names). When it has finished, download its artefact
   `Yellowmoon-<commit>-aarch64`, unzip it and open the disk image.
2. Drag Yellowmoon to a folder of its own, for example `~/Applications` (in Finder: **Go › Home**, then a new folder
   named `Applications`). Do not put it over the released app in `/Applications`.
3. Allow it to open, as the main README says under "Allow Yellowmoon to open". The Terminal way also covers the
   `ym` inside it, which the test runs:

       xattr -dr com.apple.quarantine ~/Applications/Yellowmoon.app

You do not need to open the app for the test. The kit uses only the `ym` inside it:
`~/Applications/Yellowmoon.app/Contents/MacOS/ym`.

**Or build `ym` from the branch** (needs Rust and Xcode 27, whose macOS 27 SDK the bridge is compiled with):

    git checkout claude/apple-auto
    cargo build --release -p ym-cli

The kit then uses `target/release/ym` with the repository's `data/` folder by itself, so `--ym` can be left out in
step 4 (for step 2: `YELLOWMOON_DATA=data target/release/ym doctor`).

## 2. Check that Apple Auto is available

    ~/Applications/Yellowmoon.app/Contents/MacOS/ym doctor

The row **Apple Auto** should read `available · Apple Intelligence · <model> · macOS 27… · prompt 3 · table 2`, with
the instructions it reads (`default`, or `edited` when you changed them in Preferences). If it
adds "turned off in Preferences", that is fine: the test turns Apple Auto on for its own renders. If it says
`not available (…)`, the words in brackets say what to change, for example "Apple Intelligence is off in System
Settings" or "Apple Intelligence is still downloading its model". The kit checks this too, and stops with the same
reason when Apple Auto is not available.

## 3. Choose the photos (strata)

Choose about 150 **fresh** photos: photos you have not used in an earlier test or when the table of `docs/19` was set
by eye, so neither you nor the table has seen them.

Put them in folders named for the kind of picture. Each folder is a stratum, and the score shows each stratum on its
own. Photos directly in the top folder count as `other`. For example:

- `portrait`
- `landscape`
- `indoor`
- `night`
- `backlit`
- `snow-or-beach`
- `mixed-light`

So a photo goes in, for example, `~/Pictures/apple-ab/night/DSC01234.ARW`. Aim for at least 15 photos in each folder:
a stratum with a few photos can pass or fail by chance, and the score shows how many trials each one has. Choose the
photos before you look at any version, and keep the hard ones: a test that leaves out the difficult photos tells you
little. Take one frame of a burst, not several: the kit renders each photo on its own, so the frames of one burst
would be near copies of one another.

## 4. Make the test

From the repository folder:

    tools/eval/ab/ab.sh prepare --ym ~/Applications/Yellowmoon.app/Contents/MacOS/ym ~/Pictures/apple-ab

For each photo the kit renders the Apple Auto version, and then, when Apple's answer changed something, the rule's
version. It writes the page and prints a summary: Apple's model and the macOS version, the photos found and in the
test, **how many photos Apple Auto left as they were**, **how many it failed on** (each with its reason), the strata,
the number of trials, and the command that opens the page.

Rendering takes a while. Each render with Apple Auto waits for Apple's answer, about 1 to 4 seconds more per photo,
and these renders run one at a time, as the app asks one question at a time. The progress lines show how far it has
got.

If Apple Auto failed on many photos (for example "macOS asked Yellowmoon to wait"), run the same command again with
`--resume`: it keeps the pictures already made and asks again only where the answer failed. Please tell the lead what
the failures said; they are among the open questions of `docs/19` section 13.

Useful options:

- `--out DIR`: where the test goes (default `testdata/ab` in the repository; git ignores it).
- `--with-camera`: also compare Apple Auto with the camera's JPEG (reported, not part of the gate).
- `--long-edge 2000`: the size of the renders in pixels (the default is 2000).
- `--jobs N`: photos at the same time (default: half your CPUs). Each render decodes a full-size raw file, so if the
  Mac starts to swap, use a smaller number.
- `--apple-jobs N`: renders waiting for Apple's model at the same time (default 1).
- `--seed N`: the seed of the trial order and the sides (default 20261019). The same photos, answers and seed give
  the same test.
- `--resume`: finish a run that stopped part way. Run the same command again with `--resume`, and the kit keeps the
  pictures it already made (with the same photos, `--out` and `--seed`) and makes only the missing ones. Without it,
  every picture is made again and Apple is asked again.
- `--data DIR`: another data folder (not needed: the `ym` inside the app uses the app's own).
- `--instructions FILE`: the instructions Apple's model reads, from a text file, instead of the default ones
  (`docs/19` section 2; at most 2000 characters). The test never uses the instructions you saved in Preferences, so
  to test your own, copy them into a file and pass it here. The plan and the summary record which instructions were
  read (`default` or `edited`) with the first 12 hex digits of their SHA-256; `--resume` asks Apple again for every
  photo whose answer was given to other instructions.

If you run `prepare` again into a folder that holds another test, the old `plan.json` is replaced. Answers given to
the old test no longer match.

## 5. Answer the trials

Open the page the summary prints. With the default folder, that is:

    open testdata/ab/index.html

(from the repository folder; the summary gives the full path).

Each trial shows two pictures of one photo. Pick the one you prefer, or say they are about the same:

| Key | Action |
|---|---|
| `←` or `A` | left is better |
| `→` or `L` | right is better |
| `↓` or `S` | about the same |
| `F` | the left picture needs a fix |
| `J` | the right picture needs a fix |
| `Z` | undo the last answer |
| click a picture | show both full screen (`Esc` goes back) |

A picture **needs a fix** when it is more than about half a stop too dark or too bright, or when it has a colour cast,
too much or too little contrast, or colours you would correct. Mark it with `F` or `J` before you answer. Use the same
screen, at a normal brightness, for the whole test.

The page does not say which version is which, and it keeps the file names of the pictures free of the versions' names,
so you cannot tell from them either. Please do not open `plan.json` or the `records` folder until you have finished:
they are the key.

Your answers are saved in this browser and come back when you reload the page. You can stop at any time. When every
trial is answered, press **Export results**. Your browser saves `results.json`, usually in Downloads. You can also
export part of the way; the score then says not every trial is answered.

## 6. Score the answers, and send them

    tools/eval/ab/ab.sh score ~/Downloads/results.json

The kit prints a table with a row per comparison: Apple Auto's wins, losses and ties, the win share, the Wilson bound,
the "needs a fix" rates and the verdict. Under it come the strata, then what helps tune the table of `docs/19`
section 5: Apple Auto's win share by what it did (brighter or darker, the highlights, the shadows and the black level,
more or less contrast, warmer or cooler, greener or more magenta, more or less saturation), by the colour it gave the
shadows or the highlights, for photos where it changed one thing only, and step by step; how often Apple's model left
each field empty (it then set nothing); and the trials it lost, with the scene Apple's model described and its plan. The last line gives the verdict of the gate, with its provisional thresholds. It also writes
`score.json` next to the results.

**Send the lead `score.json`.** It holds the numbers, Apple's model and macOS version, and the diagnostics. Or send
`results.json` and the test folder's `plan.json`. These files hold your photos' file names and Apple's short scene
descriptions and plans, but no pictures.

The score finds `plan.json` next to the results, or in the default test folder. If it cannot find the plan, pass it:
`ab.sh score RESULTS_JSON --plan PLAN_JSON`. Results from another test are refused.

## What stays on your Mac

- The page is `index.html` with its code inside, plus `plan.js`, which lists the trials. It loads nothing from the
  network, and the kit sends nothing anywhere. Apple's model runs on the Mac (`docs/19`).
- The raw files are only read. The renders, `ym`'s records of them and the camera's JPEG copies are written to the out
  folder.
- The out folder holds your file names (`plan.json`, `records`) and your photos (the pictures). Delete it when you
  have finished, or keep it out of any shared folder. Do not commit it.

## Known limits

- **One photo at a time.** Each photo is rendered on its own, so a burst's shared exposure and shared answers
  (`docs/19` section 5) do not come in: both versions use the photo's own values.
- **The defaults.** Each render gets its own empty home folder, so your preferences and imported looks cannot change
  the test, and the renders use the built-in default look. Your Apple Auto preference does not matter either: the
  renders ask for Apple Auto on (`--apple-auto on`) or off (`--apple-auto off`).
- **Only the photos Apple changed.** A photo whose picture Apple Auto left as it was gives no trial, so the win share
  is over the photos where Apple Auto did something. The summary and the score say how many photos that leaves out.
- **The camera's JPEG** (`--with-camera`). It is shown turned upright from the raw file's orientation, as the renders
  are; its pixels are the camera's own. Some cameras set to 16:9 store a 16:9 JPEG, which is shown as it is. The page
  shows every picture scaled to fit the same box.
- **Repeat.** Running `prepare` again with the same photos, seed, app and macOS gives the same plan and the same
  pictures: Apple's model gives the same answer to the same picture on one Mac (`docs/19` section 7). Another macOS
  build may answer differently; the plan records the model and the macOS version.
- **The test advisor.** With `YELLOWMOON_TEST_APPLE_AUTO=fake` in the environment (for developers), the answers come
  from Yellowmoon's test stand-in, not from Apple's model, and the summary says so. Do not set it for the real test.

## Developers

    python3 -I -m unittest discover -s tools/eval/ab -p 'test_*.py' -v

The tests need no `ym`, no network and no raw files: a stand-in `ym` prints the JSON of `docs/19` section 8 (answers,
failures, unchanged photos, an unavailable doctor). CI runs them in the `fetch-tools` job, which also shellchecks
`ab.sh`. On Linux the whole path runs with the test advisor: link a few files of `testdata/synthetic` into stratum
folders, then, for example:

    YELLOWMOON_TEST_APPLE_AUTO=fake tools/eval/ab/ab.sh prepare --ym target/debug/ym --long-edge 600 \
        --out /tmp/ab-smoke /tmp/ab-photos
