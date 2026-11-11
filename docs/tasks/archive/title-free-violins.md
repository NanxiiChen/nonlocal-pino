# Title-free publication violins

## Goal and acceptance criteria

Remove example-name titles from the two independent violin figures. Keep example
identifiers in PDF/PNG filenames, update both example READMEs and plotting rules,
preserve all 256 scores and automatic log axes, and publish through dev to main.
User explicitly authorized commit and push in this turn.

## Changes

- Removed the violin title call and its case-title mapping. The manifest explicitly
  disables example titles and raises the axes' top margin from 0.85 to 0.95.
  Physical figure size remains 55 x 45.72 mm; PDF/PNG names retain the case ID.
- Refreshed both distribution PNGs and separate vector PDFs. Updated example
  captions, assets guide, plotting requirements and provenance/audit records.
- Includes the existing comparison configurations, input-pair audits and evaluation
  scripts required by README reproduction commands, plus linked study records.
  No training/default change in this turn. Unrelated Melting experiment code and
  its example README remain local and uncommitted.
- Memory: docs/DECISIONS.md, docs/PROGRESS.md and this archive.

## Artifacts and reproduction

External root:
`/root/autodl-tmp/tf-logs/pino-derivative-eval64-20261111-01/title-free-violins-20261112`

- electropolish2d_distribution.pdf: 17,422 bytes; PNG: 27,568 bytes.
- particle_corrosion2d_distribution.pdf: 17,182 bytes; PNG: 15,458 bytes.
- before_redraw/: exact prior renderer, manifests, captions and four PNGs.
- validate_pdf.py: independent PDF, score and density/axis audit.
- Curated PNGs remain in assets/pino-comparisons; PDF paths are in the manifest.

```bash
JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python scripts/render_pino_derivative_comparisons.py \
  --case all --figures distribution
```

## Validation

The real render passes source/config/reference audits. All 256 scores, complete
KDE records, quartiles and automatic axis limits are exactly unchanged against
the archived receipt. Both field-sheet PNGs remain byte-identical. The two new
PNGs were visually inspected; all four comparison PNGs total 1,093,320 bytes.
Actual PDFs have no example titles or raster images, embed Helvetica TrueType,
retain 6–7 pt dark editable text and pass text-extraction/edit-roundtrip checks.

Seventeen CPU unittest checks pass for the existing comparison support:
`tests.test_electropolish_operator_pairing`,
`tests.test_particle_corrosion_method_validation`, and
`tests.test_pino_local_reference_identity`. The initial pytest invocation could
not run because pytest is absent; these tests use the installed stdlib unittest,
so no dependency was added. No training, checkpoint inference or solve occurred.

Staged Python syntax and all new documentation links pass. Seventeen historical
Electropolish image links target ignored local assets and predate this publication;
they are not republished or silently added to Git. All staged files are below
1 MiB; whitespace checks pass. The repository-wide
content checker still flags the pre-existing artifacts/index.yaml (1,079,277
bytes), byte-identical to HEAD and excluded from this change. This is a baseline
size violation, not a new plot payload. Reference arrays, models and fonts remain
external. Paused process group 618216 stays untouched.

## Status, blocker and next action

Figure update and verification complete; no numerical blocker. The session's
final release action is the authorized dev commit, regular merge into main and
push of main only. Git history records the resulting commit IDs. No further
figure or training work is pending.
