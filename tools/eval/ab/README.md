# Blind A/B for automatic settings (gate 5, docs/18 section 8)

This kit answers one question with your own photos: does `autoset`, the new automatic settings, make
pictures you prefer? You look at two versions of the same photo, side by side, without being told
which is which, and pick one. The kit then scores your picks against gate 5.

There are three versions of each photo:

- **autoset** (the new automatic settings), the picture under test;
- **the rule** (the old automatic settings: the camera's exposure, the light fully neutralised,
  contrast 0);
- **the camera's JPEG** (the camera's own rendering), the picture embedded in the raw file, copied as
  it is.

Each photo gives two trials, one per comparison: autoset against the rule, and autoset against the
camera's JPEG.

**Gate 5 passes** when, for both comparisons (autoset against the rule, and autoset against the
camera's JPEG):

- autoset wins at least 60 % of the trials (a tie counts as half a win);
- the lower end of the 95 % Wilson interval of the win share is above 50 % (ties are left out of
  this one);
- no stratum (a kind of photo, see below) has a win share below 45 %;
- autoset is not marked "needs a fix" more often than the other picture.

Nothing is uploaded. The renders, the page and the answers stay on your Mac.

## What you need

- This repository on your Mac, and Rust (`rustup`).
- Python 3.9 or newer (`python3 --version`; the Xcode command line tools provide it).
- Your Sony ARW files, in one folder. Subfolders are strata (below).
- About 2 GB of free disk for about 150 photos. Each photo leaves two renders and a copy of the
  camera's JPEG in the out folder.
- Time: rendering takes a while, because every photo is rendered twice. Start it in the evening. The
  progress lines show how far it has got.

## 1. Build the renderer

From the repository folder:

    cargo build -p ym-cli --release

This makes `target/release/ym`, which the kit uses. It renders on the CPU, so the pictures do not
depend on the graphics card.

## 2. Choose the photos (strata)

Put the photos in folders named for the kind of picture. Each folder is a stratum, and the score
shows each stratum on its own. Photos directly in the top folder count as `other`.

About 150 photos, spread over everyday kinds of light and subject, for example:

- `portrait`
- `landscape`
- `indoor`
- `night`
- `backlit`
- `snow-or-beach`
- `mixed-light`

Aim for at least 15 photos in each folder. A stratum with a few photos can pass or fail by chance,
and the score shows how many trials each one has. Choose the photos before you look at any version,
and keep the hard ones: a test that leaves out the difficult photos tells you little.

Each photo gives two trials. So 15 photos in a folder give 30 trials for that stratum, and 150
photos give 300 trials in all.

## 3. Make the test

    tools/eval/ab/ab.sh prepare ~/Pictures/ab-test

For each photo the kit renders two versions (autoset, and the rule), copies the camera's JPEG, and
writes the page. It prints a summary: the photos, the strata, the number of trials, any photo that
was left out, and the command that opens the page. A photo that fails in any version is left out of
every trial, and the summary says why.

Useful options:

- `--out DIR`: where the test goes (default `testdata/ab` in the repository; git ignores it).
- `--long-edge 2000`: the size of the renders in pixels (the default is 2000).
- `--jobs N`: renders at the same time (default: half your CPUs). Each render decodes a full-size raw
  file, so if the Mac starts to swap, use a smaller number.
- `--seed N`: the seed of the trial order and the sides (default 20261018). The same photos and seed
  give the same test.
- `--ym PATH`, `--data DIR`: other renderer or data folders (normally not needed).
- `--resume`: finish a run that stopped part way. Run the same command again with `--resume`, and the
  kit keeps the pictures it already made (with the same photos, `--out` and `--seed`) and makes only
  the missing ones. Without it, every picture is made again.

If you run `prepare` again into a folder that holds another test, the old `plan.json` is replaced.
Answers given to the old test no longer match.

## 4. Answer the trials

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

A picture **needs a fix** when it is more than about half a stop too dark or too bright, or when it
has a colour cast you would correct. Mark it with `F` or `J` before you answer. Use the same
screen, at a normal brightness, for the whole test.

The page does not say which version is which, and it keeps the file names of the pictures free of
variant names, so you cannot tell from them either. Please do not open `plan.json` until you have
finished: it is the key.

Your answers are saved in this browser and come back when you reload the page. You can stop at any
time. When every trial is answered, press **Export results**. Your browser saves `results.json`,
usually in Downloads. You can also export part of the way; the score then says it is provisional.

## 5. Score the answers

    tools/eval/ab/ab.sh score ~/Downloads/results.json

The kit prints one table with a row per comparison: the wins, losses and ties of autoset, the win
share, the Wilson bound, the "needs a fix" rates and the verdict. Under it come the strata, and the
reasons for any failure. The last line gives the verdict of gate 5. It also writes `score.json` next
to the results.

The score finds `plan.json` next to the results, or in the default test folder. If it cannot find
the plan, pass it: `ab.sh score RESULTS_JSON --plan PLAN_JSON`. Results from another test are refused.

## What stays on your Mac

- The page is `index.html` with its code inside, plus `plan.js`, which lists the trials. It loads
  nothing from the network, and the kit sends nothing anywhere.
- The raw files are only read. The renders and the camera's JPEG copies are written to the out
  folder.
- The out folder holds your file names (`plan.json`) and your photos (the pictures). Delete it when
  you have finished, or keep it out of any shared folder. Do not commit it.

## Known limits

- **The camera's JPEG.** The camera's JPEG is shown turned upright from the raw file's orientation, as the
  renders are; its pixels are the camera's own. Some cameras set to 16:9 store a 16:9 JPEG, which is shown
  as it is.
- **Sizes.** The camera's JPEG is used as it is. The page shows every picture scaled to fit the same
  box, so a file with more pixels than the screen is scaled down and a render shown larger than its
  pixels is enlarged. If that is a concern, make the test again with `--long-edge 3000`.
- **The rule.** The rule uses a temporary copy of the data folder without the autoset model, and
  `--contrast 0 --white-preference 0`. The copy is deleted at the end.
- **Your preferences.** Each render gets its own empty home folder, so your preferences and imported
  looks cannot change the test. The renders use the built-in default look.
- **Repeat.** Running `prepare` again with the same photos, seed and renderer gives the same plan
  and the same pictures.

## Developers

    python3 -I -m unittest discover -s tools/eval/ab -p 'test_*.py' -v

The tests need no `ym`, no network and no raw files. CI runs them in the fetch-tools job.
