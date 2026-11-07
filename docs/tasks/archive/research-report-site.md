# Nonlocal PINO technical report site

Goal: implement the selected technical-reference design as a locally readable
report with build-time LaTeX rendering, a shared methods overview and five
full case chapters.

Acceptance: the method derivation and case values are traceable to the
repository, the site builds from locked dependencies, all links and assets
resolve, equations render offline, and desktop and narrow-screen layouts pass
browser checks. GitHub Pages configuration is deferred by the user.

Context: [design review](research-site-prototypes.md). Reuse the five
curated setup and evolution figure pairs. Keep local versus native reference
claims and selected versus optional training tools distinct.

Status: local implementation complete. The [build guide](../../../website/README.md)
documents the English Markdown sources and generated entry point.

Changes: replaced the layout prototypes with a nine-page technical report.
The shared template supplies the desktop sidebar, mobile chapter menu, section
anchors and adjacent-chapter links. The new source includes two vector method
diagrams. The build copies the five existing setup/evolution pairs byte for
byte, plus KaTeX styles, fonts and license. The root README links the report.
No training code, reference values or existing figure bytes changed.

Validation: locked `npm ci` and the static build pass. Chromium checked all
nine chapters at widths 1440, 768, 390 and 320, including navigation, internal
links and 186 KaTeX HTML/MathML expressions. Offline file access loaded the
figures and fonts. Final source, artifact allowlist and whitespace checks pass.
External source links were checked against paths in the checkout.

Next phase: when requested, add the Pages build/deploy workflow and confirm
repository settings. No Git history operation or hosting change occurred.
