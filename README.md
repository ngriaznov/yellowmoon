# Source photos for the README screenshots

This branch holds a few of Mykyta Griaznov's own Sony ARW files. They are used only to render the screenshots in Yellowmoon's README (`tools/readme/screenshots.mjs`). They are not part of the product and not test data. This branch is never merged into any other branch.

The screenshots made from these photos may be published in the Yellowmoon repository and its README. The photos themselves stay © Mykyta Griaznov, all rights reserved.

Raw files are stored with Git LFS (`.gitattributes`). To add photos:

    git fetch origin readme-photos
    git switch readme-photos        # or: git worktree add ../yellowmoon-photos readme-photos
    git lfs install
    cp ~/Pictures/…/DSC0xxxx.ARW .
    git add *.ARW && git commit -m "Add photos" && git push

Push from your own machine: Claude's cloud session can download LFS objects but not upload them.
