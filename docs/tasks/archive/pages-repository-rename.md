# GitHub Pages repository rename

## Goal and acceptance

Confirm the current live report URL after the move to `nonlocal-pino`.
Update active README, report and dashboard links. Build the nine pages and
check generated repository URLs and relative links under the new subpath.

## URL and source validation

GitHub Pages and the repository About link already point to
`https://nanxiichen.github.io/nonlocal-pino/`. Direct GET requests return
200 for all nine chapters and three representative assets under the new
address, and 404 for the old homepage. GitHub reports the renamed
repository is public. The existing live report includes the current
Electropolish result.

The root and website READMEs, active project dashboard and generated report
repository links now use the new URLs locally. The old Pages URL is
recorded as the original launch URL in the historical release note.
The nine-page build passes, with 308 relative references, eight source
links and 60 font references checked. The existing workflow covers
changes under `website/`.

## Release checks

The user authorized commit, push and verification of the Pages deployment.
Publish the source corrections through the repository's dev-to-main merge
workflow. Verify the Actions build and deploy jobs for that merge SHA.
Compare the public report files against the local build and verify the
published README and repository source links under the renamed repository.
