# GitHub Upload

Repository: https://github.com/xing-lab-pitt/livecellx
Run commands from the repository root. No commit, push or release publication
has been performed by the preparation tools.

## Prepared Scope

The selected revision panels are main Fig. 4g-i and 5d, and Extended Data
Fig. 3i and 4k. FIGURE_CODE_MAP.md maps each panel to its code and inputs.
Extended Data Table 1 evaluation materials and old/unadopted experimental
outputs are excluded. Required shared helpers and reusable preparation code
are retained; they may contain functions not used by the selected panels.

There are 36 selected files, split into two disjoint lists:

- GITHUB_PACKAGE_FILES.txt: five existing package bug-fix modules.
- GITHUB_REVISION_FILES.txt: 31 revision sources, helpers, config, tests and docs.
- GITHUB_UPLOAD_FILES.txt: their combined list; use for inventory, not to
  combine the two commits accidentally.

## 1. Package Fixes to Main via Pull Request

First inspect the worktree and ensure the index contains no unrelated staged
changes. Never use git add . in this working directory.

```bash
cd /home/zhz187/livecellx
git status --short
git diff --cached --stat
git switch -c fix/revision-package-fixes
git add --dry-run --pathspec-from-file=revision_code/GITHUB_PACKAGE_FILES.txt
git add --pathspec-from-file=revision_code/GITHUB_PACKAGE_FILES.txt
git diff --cached --check
git diff --cached
```

After reviewing the five-file diff:

```bash
git commit -m "Fix trajectory serialization, paths and editing"
git push -u origin fix/revision-package-fixes
```

On GitHub, open a Pull Request from this branch to main and merge after review.
Do not include unrelated local tutorial notebooks or training files.

The five modules are datasets.py, single_cell.py, napari_visualizer.py,
sc_seg_operator.py and sct_operator.py under livecellx/core. Their changes
predate this release organization. Packaging does not certify native GUI
stability or replace review of those fixes.

## 2. Revision Code on Its Own Branch

After the package PR is merged, update main. If Git reports conflicting local
changes, stop and preserve them; do not reset or discard them.

```bash
git switch main
git pull --ff-only origin main
git switch -c revision/final-submission
git add --dry-run --pathspec-from-file=revision_code/GITHUB_REVISION_FILES.txt
git add --pathspec-from-file=revision_code/GITHUB_REVISION_FILES.txt
git diff --cached --check
git diff --cached --stat
git diff --cached
```

After review:

```bash
git commit -m "Add selected revision analyses and release instructions"
git push -u origin revision/final-submission
```

This branch may remain separate or be merged through another PR. It must
contain the package fixes. A Release tag must point to this revision commit
(or its merged equivalent), not to a main commit lacking the revision scripts.

## 3. Data and Models as Release Attachments

The only final asset directory is revision_code/revision_release/.
Upload exactly the files listed in its UPLOAD_ASSETS.txt. Do not put these
archives in ordinary Git history.

On GitHub: Releases -> Draft a new release. Select the reviewed revision
commit/branch, choose an unused version tag, attach the listed files and use
RELEASE_README.md for the description. Check public-sharing permission and
licenses before clicking Publish release.

The code archive is a supplementary working-tree snapshot, not a substitute
for the readable source commits. If source files change during PR review,
refresh the code archive and manifests before publication. No DOI, data/model
license, authorship permission or publication status has been invented.

After downloading the attachments, run in the attachment directory:

```bash
sha256sum -c SHA256SUMS.txt
```

Follow RELEASE_README.md to restore into a new checkout. Add the actual
published Release URL to the repository README afterward.

## Selection Checks

All selected files exist; Python syntax and direct local revision imports were
checked. Notebook outputs are cleared in the release notebook. The deferred
table's evaluation files, backups, padding/candidate experiments, obsolete
figure outputs and manuscript PDFs are not selected.

Original scientific inputs and saved outputs were not overwritten. An earlier
unnecessary numerical audit was stopped; no numerical rerun is needed to
package or upload these already generated results. Archive integrity/path
checks are not independent validation of the scientific methods.
