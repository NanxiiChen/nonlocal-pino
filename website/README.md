# Nonlocal PINO technical report

The site presents the five principal physics-only PINO cases in a technical
report layout. It includes an overview, architecture, spatial calculus and
training chapters, and a chapter for each case. Case chapters follow the same
sequence from setup and derivation to field results, comparison and reproduction.

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
- Reuse the [curated figure pairs](../assets/README.md). The builder copies the
  five case setup/evolution pairs and the two method diagrams. Numerical
  parameters and data preparation remain documented with the example sources.

KaTeX renders standard LaTeX expressions at build time to HTML and MathML.
The generated site loads its math fonts locally. [KaTeX API](https://katex.org/docs/api)

## Publishing

The [Pages workflow](../.github/workflows/pages.yml) runs the locked install
and build on changes to the site sources, figures or workflow on `main`. It can
also be dispatched manually from `main`. The deploy job publishes
`website/dist/` to the `github-pages` environment. The report's links and
assets use relative paths so the same build works under a project subpath.
