# GitHub Pages release for the technical report

Goal: commit and push the report and an Actions workflow, then verify the live
GitHub Pages deployment.

Acceptance: remote `main` contains the report and workflow, a deploy job
succeeds, and the public report HTML and assets load. Publish only the built
site and its bundled assets. Use the repository's dev/main Git workflow.

Context: [technical report validation](research-report-site.md).
The repository is private and Pages was initially disabled. The Pages API
accepted `build_type=workflow`, `public=true`, and assigned the project URL
`https://nanxiichen.github.io/nonlocal-pinn/`.

Status: deployed and independently verified. The report source was merged and
pushed as `afcdf1d`. The workflow is committed on dev at `043c098`, merged to
main at `f4ab020` and pushed after the owner authorized GitHub CLI's `workflow`
scope. The initial push rejection did not affect the site source commit.

The [first Actions run](https://github.com/NanxiiChen/nonlocal-pinn/actions/runs/42346605188)
completed build in 14 s and deploy in 13 s. The workflow builds from the npm
lockfile and uploads only `website/dist/`. Local YAML, nine-page build and
artifact scope checks passed. Unauthenticated HTTP checks verified the project
homepage and all 84 artifact files byte for byte, including every chapter,
figure and KaTeX asset. The Pages site is public and the repository is private.

The root README now links the live site. Future main pushes that modify the
workflow, site sources or curated figures trigger the same build and deploy.
