## Task

<!-- The task ID from docs/15-milestones.md, for example M1-07. Branch names follow it (m1-07-logc4). -->

Task: M?-??

## What changed

<!-- Short and concrete. Say what the reviewer should look at first. -->

## Acceptance criteria

<!-- Copy the "Done when" criteria of the task and mark each one. Link the test that proves it,
     or say why it is blocked (for example "needs a Mac"). -->

- [ ] criterion 1 (test: `path::to::test`)
- [ ] criterion 2

## New dependencies and their licences

<!-- docs/14 rule 2. One line per new crate or npm package: name, version, licence. Write "none" if there are none.
     GPL, LGPL and AGPL are not allowed in shipped code. -->

| Dependency | Version | Licence | Shipped or dev only |
|---|---|---|---|
| none | | | |

References read: <!-- GPL/LGPL/AGPL projects you read for format or algorithm knowledge (docs/14 rule 3), or "none" -->

## Gate numbers

<!-- Numbers measured by this change (gate name, value, threshold), or "none". Never loosen a threshold here:
     a failed gate is reported in docs/STATUS.md and needs the maintainer's agreement. -->

## Checks

- [ ] `cargo fmt --check`
- [ ] `cargo clippy --workspace --all-targets -- -D warnings`
- [ ] `cargo test --workspace`
- [ ] `cargo deny check`
- [ ] `npm ci && npm test && npm run build` (when the frontend is touched)
- [ ] New public items have doc comments
- [ ] The relevant spec in `docs/` is updated if a decision was made
- [ ] `docs/STATUS.md` is updated (task done, gate numbers, waiting-on and questions)
- [ ] `CHANGELOG.md` has an entry under Unreleased (and the version is bumped only in a "Release x.y.z" pull request)
- [ ] No raw files, training data or ARRI files are committed; originals are never written to
