# GitHub Pages release for the technical report

Goal: commit and push the report and an Actions workflow, then verify the live
GitHub Pages deployment.

Acceptance: remote `main` contains the report and workflow, a deploy job
succeeds, and the public report HTML and assets load. Publish only the built
site and its bundled assets. Use the repository's dev/main Git workflow.

Context: [technical report validation](archive/research-report-site.md).
The repository is private and Pages was initially disabled. The Pages API
accepted `build_type=workflow`, `public=true`, and assigned the project URL
`https://nanxiichen.github.io/nonlocal-pinn/`.

Status: Pages source enabled. The report source was merged and pushed as
`afcdf1d`. The Actions workflow builds from the npm lockfile and uploads
only `website/dist/`. Local YAML, nine-page build and artifact scope checks pass.
Next: push the workflow merge to main and monitor the first deployment.
