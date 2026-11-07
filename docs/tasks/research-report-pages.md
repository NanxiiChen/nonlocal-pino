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

Status: Pages source enabled, awaiting the first build and deploy.
Next: commit the report and workflow, push main, and monitor the deployment.
