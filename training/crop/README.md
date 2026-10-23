# training/crop: the learned Auto crop pilot

A small cropping network of our own (the *student*, `model.py`, 1.1M parameters, trained from scratch) learns to
crop like FrameLite (the *teacher*, an Apache-2.0 cropper built on Qwen3.5-0.8B) from photos whose licences allow it.
Background and the evaluation that started it: `research/crop-models-2026-10-23.md`. Nothing here ships yet: a
student goes to `data/models/` only after it wins the owner's blind A/B, with its model card.

Why a student: FrameLite crops well (it won 50, lost 4 and tied 80 of 134 blind pairs against the uncropped photo
on 150 unseen Play Raw photos) but is 1.9 GB, takes seconds a photo, runs in Python and was trained on crop
datasets without a licence, so it may teach (owner, 2026-10-23) but does not ship (`docs/14` rule 5).

## Data and licences

- **Photos:** Open Images V7 (`openimages.py`): a seeded selection of upright images listed as CC BY 2.0, from the
  CVDF mirror. `manifest.json` keeps each photo's author, Flickr page and licence for the model card and `NOTICE`.
  Weights trained on CC BY photos keep the repository's Apache-2.0 (`docs/14` rule 5).
- **Labels:** FrameLite's free crop of each photo (`teacher.py`), FrameLite's code at a pinned commit and its
  weights and base model at pinned revisions. The photos FrameLite learned from never reach the student.

## Pipeline

| Step | Module | Where |
|---|---|---|
| select and download photos | `openimages.py` | Modal CPU (`prepare`), or locally |
| label with the teacher | `teacher.py` | Modal L4 GPUs (`Labeller`, 6 at a time), or locally on the CPU (about 8 s a photo) |
| letterbox to 256 x 256, arrays | `data.py` | Modal CPU (`build_arrays`), or locally |
| train the student | `train.py` | Modal L4 (`train_student`), or locally on the CPU for a smoke run |
| crops for a folder of photos | `predict.py` | locally (CPU) |

`train.py` reports the validation IoU with the teacher next to two baselines, the whole photo and the training
set's mean crop: a student that only learned the average crop shows as one. `predict.py` writes the free crop
and, for Square, 4:5 and 16:9, the frame of that aspect closest to it (IoU) among 6 sizes x 9 x 9 positions.

## Running it

Modal's client speaks gRPC, which the agents' cloud proxy does not carry, so the pilot runs from GitHub Actions
(`.github/workflows/crop-pilot.yml`): a push to the branch `crop-pilot-run` starts it with the workflow's
`PILOT_ARGS`, and the results (each run's `model.pt`, `config.json`, `metrics.json`, FrameLite's `labels.json`, the
photos' `manifest.json` and the log) come back on the temporary branch `crop-pilot-runs`. It needs the repository
secrets `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET`. From a computer with direct internet and the same two variables:

```bash
pip install modal==1.6.2 numpy pillow
modal run -m training.crop.modal_app::pilot --n 20000 --curve 5000,10000,0 --epochs 40 --budget 20
modal run -m training.crop.modal_app::results
modal run -m training.crop.modal_app::fetch          # into training/crop/runs/ (git-ignored)
```

**Budget.** The owner's Modal account holds $30 a month (the Starter plan's credits), and nothing may go past it
without their permission (2026-10-23). `budget.py` keeps the run's own estimate from Modal's published prices and how
long each call took, with a 25 % margin. The pilot labels a first chunk of 250 photos on one GPU, measures FrameLite's
seconds per photo, reserves the cost of the arrays and of the training runs (each stopped at `--max-minutes`), and
labels only as many more photos as the rest of `--budget` pays for (and `--max-label-hours` allows). It stops if
fewer than 3,000 photos can be labelled, if under half the first chunk's crops are valid, or if the estimate passes
the budget. `modal run` without `--detach` stops the app when the client stops, so cancelling the workflow stops the
spending. Modal's own Workspace budget (Usage & Billing) is the hard cap above all this.

The pilot trains at three training-set sizes (a learning curve) so the run says whether more photos would help.
Every stage skips what the volume `yellowmoon-crop` already holds, so a failed run is resumed by running it again;
`--budget` is then what the new run may spend.

Locally (smoke run on 16 photos, CPU; FrameLite's code checked out and its weights downloaded):

```bash
python3 -m training.crop.openimages --subset validation --n 16 --out testdata/crop/smoke
FRAMELITE_SRC=<FrameLite checkout> python3 -m training.crop.teacher --images testdata/crop/smoke/images \
    --weights <weights dir> --download --out testdata/crop/smoke/labels.json
python3 -m training.crop.data --images testdata/crop/smoke/images --labels testdata/crop/smoke/labels.json \
    --out testdata/crop/smoke/arrays
python3 -m training.crop.train --arrays testdata/crop/smoke/arrays --out testdata/crop/runs/smoke --epochs 3 \
    --batch 8 --device cpu
python3 -m training.crop.predict --model testdata/crop/runs/smoke/model.pt --images <JPEGs> --out pred.json
python3 -m pytest -q training/crop/tests
```

Training on a GPU is not bit-for-bit repeatable (cuDNN); the seeds fix the photo selection, the split and the
order of the batches.
