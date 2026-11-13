# Nonlocal PINO technical report

The site presents the five principal physics-only PINO cases in a technical
report layout. It includes an overview, architecture, spatial calculus and
training chapters, and a chapter for each case. Case chapters follow the same
sequence from setup and derivation to field results, comparison and reproduction.

Read the [published technical report](https://nanxiichen.github.io/nonlocal-pino/).

## Build and preview locally

Run from the repository root with Node.js 22 or newer.

~~~bash
npm --prefix website ci
npm --prefix website run build
python3 -m http.server 8000 --bind 127.0.0.1 --directory website/dist
~~~

Open [the local site](http://127.0.0.1:8000/) while the server is running. The
generated entry point is `website/dist/index.html`, which can also be opened
directly in a browser. The generated HTML, stylesheet, figures and KaTeX fonts
are self-contained. Repository source links require access to the repository.

## Edit the report

- Edit the English Markdown in [content/](content/). Write inline LaTeX as
  `\(...\)` and place `$$` on separate lines around display equations. Use
  `aligned` inside a display block for multiple equation rows.
- Edit chapter order or shared navigation in [build.mjs](build.mjs) and styles
  in [report.css](report.css). Second-level Markdown headings populate the
  in-page contents.
- Reuse the [publication main figures](../assets/pino-main/README.md). The builder
  reads [the shared manifest](../artifacts/pino_main_figures.json), checks the
  approved file sizes and copies the same five PNG/PDF pairs used by the example
  READMEs. Each chapter links its editable PDF. Setup SVGs and the two method
  diagrams are copied separately; historical `assets/pino/` galleries are not
  used by the report. Update the chapter's caption and image dimensions when
  changing samples, layout, model or reference. Numerical parameters and data
  preparation remain documented with the example sources.
- Regenerate the architecture/training master figure with
  `scripts/render_pino_master_figure.py` and
  [its specification](../artifacts/pino_master_figure.json). The report displays
  the selected PNG from `assets/pino-architecture/` and offers its editable PDF
  and `architecture.svg`. These share the source used in the project README.
- The ceramic chapter additionally displays the approved three-method final-error
  distribution from `assets/pino-studies/`, shared with the example README.
  [Its publication manifest](../artifacts/ceramic_fem_comparison_publication.json)
  records exact model/configuration inputs and PNG/PDF sizes; the builder checks
  and copies only this pair. Figure and manifest changes trigger Pages deployment.
- The current ceramic default and its final64 validation are recorded separately
  in [the execution receipt](../artifacts/ceramic_execution_validation.json).
  Keep current SOTA numbers and launch instructions in the case/overview chapters
  synchronized with the example README. Historical comparison figures retain
  their own checkpoint, cohort, norm and timing labels.

KaTeX renders standard LaTeX expressions at build time to HTML and MathML.
The generated site loads its math fonts locally. [KaTeX API](https://katex.org/docs/api)

## Publishing

The [Pages workflow](../.github/workflows/pages.yml) runs the locked install
and build on changes to the site sources, publication main figures, their shared
manifest, architecture master assets, setup SVGs or workflow on `main`. It can
also be dispatched manually from `main`. The deploy job publishes
`website/dist/` to the `github-pages` environment. The report's links and
assets use relative paths so the same build works under a project subpath.
