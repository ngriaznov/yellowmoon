# Mykyta Griaznov's photos for Yellowmoon

This branch holds a few of Mykyta Griaznov's own Sony ARW files and his reference exports of them. They are © Mykyta Griaznov, all rights reserved, and not covered by the Apache License 2.0. He permits them in the Yellowmoon repository (this branch) and their use by the project: as test and evaluation data, in CI, and in images made from them that the repository and its README may publish (screenshots, comparison sheets). They are not part of the application and are never bundled with it. This branch is never merged into any other branch.

- `*.ARW`: Sony A1 raw files (Git LFS).
- `ref-export/*.jpg`: Mykyta's reference exports of the same photos (2000 px, sRGB), used as a brightness and colour reference next to the camera's embedded JPEG.

Fetch them into a Yellowmoon checkout with `tools/fetch/owner-photos.sh` (Git LFS required); it checks every file's SHA-256 against `tools/fetch/owner-photos.sha256`.

Raw files are stored with Git LFS (`.gitattributes`). To add photos:

    git fetch origin readme-photos
    git switch readme-photos        # or: git worktree add ../yellowmoon-photos readme-photos
    git lfs install
    cp ~/Pictures/…/DSC0xxxx.ARW .
    git add *.ARW && git commit -m "Add photos" && git push

Push raw files from your own machine: Claude's cloud session can download LFS objects but not upload them.
