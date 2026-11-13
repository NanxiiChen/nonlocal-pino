import { copyFile, mkdir, readFile, readdir, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import MarkdownIt from 'markdown-it';
import { katex } from '@mdit/plugin-katex';

const root = import.meta.dirname;
const output = join(root, 'dist');
const repository = 'https://github.com/NanxiiChen/nonlocal-pino';

const pages = [
  { slug: 'index', title: 'Nonlocal physics‑informed neural operators', label: 'Overview', group: 'Report' },
  { slug: 'architecture', title: 'Conditional operator architecture', label: 'Architecture', group: 'Method' },
  { slug: 'spatial-calculus', title: 'Nonlocal spatial calculus', label: 'Spatial calculus', group: 'Method' },
  { slug: 'training', title: 'Training and validation', label: 'Training & validation', group: 'Method' },
  { slug: 'ice-melting', title: 'Ice melting', label: 'Ice melting', group: 'Case studies', number: '01', source: 'icemelting3d', badge: '3D / Random initial field' },
  { slug: 'electropolish', title: 'Electropolish', label: 'Electropolish', group: 'Case studies', number: '02', source: 'electropolish2d', badge: '2D / Random initial field' },
  { slug: 'spinodal', title: 'Confined spinodal decomposition', label: 'Spinodal decomposition', group: 'Case studies', number: '03', source: 'confined_ch2d', badge: '2D / Random geometry' },
  { slug: 'two-phase-flow', title: 'Two-phase channel flow', label: 'Two-phase flow', group: 'Case studies', number: '04', source: 'channel_nsch2d', badge: '2D / Random geometry' },
  { slug: 'particle-corrosion', title: 'Ceramic-particle corrosion', label: 'Ceramic-particle corrosion', group: 'Case studies', number: '05', source: 'particle_corrosion2d', badge: '2D / Random geometry' },
];

const markdown = new MarkdownIt({ html: true }).use(katex, {
  delimiters: 'all',
  throwOnError: true,
  output: 'htmlAndMathml',
});
const escape = markdown.utils.escapeHtml;
markdown.renderer.rules.table_open = () => '<div class="table-scroll" tabindex="0"><table>\n';
markdown.renderer.rules.table_close = () => '</table></div>\n';

markdown.core.ruler.push('chapter_headings', state => {
  const sections = [];
  for (const [index, token] of state.tokens.entries()) {
    if (token.type !== 'heading_open' || token.tag !== 'h2') continue;
    const title = state.tokens[index + 1].content;
    const id = title.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');
    if (!id || sections.some(section => section.id === id)) {
      throw new Error(`Duplicate or empty chapter anchor for ${title}`);
    }
    token.attrSet('id', id);
    sections.push({ id, title });
  }
  state.env.sections = sections;
});

function navigation(current) {
  return ['Report', 'Method', 'Case studies'].map(group => `
    <div class="nav-group"><p class="eyebrow">${group}</p>
    ${pages.filter(page => page.group === group).map(page => `
      <a href="${page.slug}.html"${page === current ? ' aria-current="page"' : ''}>
        ${page.number ? `<span class="nav-number" aria-hidden="true">${page.number}</span>` : ''}${escape(page.label)}
      </a>`).join('')}</div>`).join('');
}

function renderPage(page, index, body, sections) {
  const title = escape(page.title);
  const nav = navigation(page);
  const previous = pages[index - 1];
  const next = pages[index + 1];
  const category = page.number ? `Case ${page.number} <span class="slash">/</span> ${page.badge}` : `${page.group} <span class="slash">/</span> Technical report`;
  return `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="${title} · Nonlocal PINO technical report">
  <title>${title} · Nonlocal PINO</title>
  <link rel="stylesheet" href="assets/vendor/katex.min.css">
  <link rel="stylesheet" href="assets/report.css">
</head>
<body class="${page.slug === 'index' ? 'overview' : 'chapter'}">
  <a class="skip-link" href="#main">Skip to content</a>
  <header class="mast"><div class="mast-inner">
    <a class="brand" href="index.html"><span class="brand-symbol" aria-hidden="true">∴</span> Nonlocal PINO</a>
    <span class="mast-title">Technical report</span>
    <a class="repo-link" href="${repository}">Repository <span aria-hidden="true">↗</span></a>
  </div></header>
  <div class="layout">
    <aside class="sidebar">
      <nav class="desktop-nav" aria-label="Report chapters">${nav}</nav>
      <details class="mobile-nav"><summary>Report chapters</summary><nav aria-label="Report chapters">${nav}</nav></details>
    </aside>
    <main id="main" tabindex="-1">
      <div class="chapter-head"><p class="eyebrow">${category}</p><h1>${title}</h1></div>
      ${sections.length ? `<nav class="contents" aria-label="On this page"><span class="eyebrow">On this page</span>${sections.map(section => `<a href="#${section.id}">${escape(section.title)}</a>`).join('')}</nav>` : ''}
      <article class="report-body">${body}</article>
      <nav class="pagination" aria-label="Adjacent chapters">
        ${previous ? `<a href="${previous.slug}.html"><span class="eyebrow">← Previous</span>${escape(previous.label)}</a>` : ''}
        ${next ? `<a href="${next.slug}.html"><span class="eyebrow">Next →</span>${escape(next.label)}</a>` : ''}
      </nav>
      <footer class="footer"><span>Nonlocal PINO · Technical report</span><a href="#main">Back to top ↑</a></footer>
    </main>
  </div>
</body>
</html>
`;
}

const rendered = await Promise.all(pages.map(async (page, index) => {
  const source = await readFile(join(root, 'content', `${page.slug}.md`), 'utf8');
  const env = {};
  try {
    return [page.slug, renderPage(page, index, markdown.render(source, env), env.sections)];
  } catch (error) {
    throw new Error(`Could not render content/${page.slug}.md`, { cause: error });
  }
}));

const assets = join(output, 'assets');
const vendor = join(assets, 'vendor');
await mkdir(join(vendor, 'fonts'), { recursive: true });
await copyFile(join(root, 'report.css'), join(assets, 'report.css'));
await copyFile(join(root, 'architecture.svg'), join(assets, 'architecture.svg'));
for (const kind of ['png', 'pdf']) {
  await copyFile(join(root, '..', 'assets', 'pino-architecture', `pino_master_figure.${kind}`),
    join(assets, `architecture.${kind}`));
}
await copyFile(join(root, 'spatial-neighborhood.svg'), join(assets, 'spatial-neighborhood.svg'));

const katexRoot = join(root, 'node_modules', 'katex');
const katexDist = join(katexRoot, 'dist');
await copyFile(join(katexDist, 'katex.min.css'), join(vendor, 'katex.min.css'));
await copyFile(join(katexRoot, 'LICENSE'), join(vendor, 'KaTeX-LICENSE.txt'));
const fonts = await readdir(join(katexDist, 'fonts'));
await Promise.all(fonts.map(font => copyFile(join(katexDist, 'fonts', font), join(vendor, 'fonts', font))));

// The README and report use the same approved publication exports. Reject
// missing/stale approvals instead of silently serving a historical gallery.
const manifest = JSON.parse(await readFile(join(root, '..', 'artifacts', 'pino_main_figures.json'), 'utf8'));
const casePages = pages.filter(page => page.source);
if (manifest.schema !== 'pino_main_publication_v1' ||
    manifest.figures.length !== casePages.length ||
    new Set(manifest.figures.map(figure => figure.example)).size !== casePages.length) {
  throw new Error('Expected one approved publication figure per case chapter');
}
let figureBytes = 0;
for (const { source } of casePages) {
  const figure = manifest.figures.find(record => record.example === source);
  if (!figure) throw new Error(`Missing publication figure: ${source}`);
  for (const kind of ['png', 'pdf']) {
    if (figure[kind] !== `assets/pino-main/${source}.${kind}`) {
      throw new Error(`Unexpected publication asset path for ${source}.${kind}`);
    }
    const data = await readFile(join(root, '..', figure[kind]));
    if (data.length !== figure.bytes[kind] || data.length >= manifest.maximum_file_bytes) {
      throw new Error(`Publication asset size differs from its approval: ${figure[kind]}`);
    }
    figureBytes += data.length;
    await writeFile(join(assets, `${source}.${kind}`), data);
  }
  await copyFile(join(root, '..', 'assets', 'pino-setups', `${source}.svg`), join(assets, `${source}-setup.svg`));
}
if (figureBytes >= manifest.maximum_collection_bytes) {
  throw new Error('Publication figure collection exceeds its approved budget');
}

const setupManifest = JSON.parse(await readFile(join(root, '..', 'artifacts', 'pino_setup_figures.json'), 'utf8'));
if (setupManifest.schema !== 'pino_setup_publication_v1' ||
    setupManifest.figures.length !== casePages.length ||
    new Set(setupManifest.figures.map(figure => figure.example)).size !== casePages.length) {
  throw new Error('Expected one approved setup figure per case chapter');
}
let setupBytes = 0;
for (const { source } of casePages) {
  const figure = setupManifest.figures.find(record => record.example === source);
  if (!figure) throw new Error(`Missing setup figure: ${source}`);
  for (const kind of ['png', 'pdf']) {
    if (figure[kind] !== `assets/pino-setups/${source}.${kind}`) {
      throw new Error(`Unexpected setup asset path for ${source}.${kind}`);
    }
    const data = await readFile(join(root, '..', figure[kind]));
    if (data.length !== figure.bytes[kind] || data.length >= setupManifest.maximum_file_bytes) {
      throw new Error(`Setup asset size differs from its approval: ${figure[kind]}`);
    }
    setupBytes += data.length;
    await writeFile(join(assets, `${source}-setup.${kind}`), data);
  }
}
if (setupBytes >= setupManifest.maximum_collection_bytes) {
  throw new Error('Setup figure collection exceeds its approved budget');
}

const ceramic = JSON.parse(await readFile(join(root, '..', 'artifacts', 'ceramic_fem_comparison_publication.json'), 'utf8'));
if (ceramic.schema !== 'ceramic_fem_comparison_publication_v1' ||
    ceramic.test_samples !== 64 || ceramic.checkpoint_step !== 20000 || ceramic.arms.length !== 3) {
  throw new Error('Expected the approved three-method final64 ceramic comparison');
}
for (const kind of ['png', 'pdf']) {
  const name = `particle_corrosion2d_fem_distribution.${kind}`;
  if (ceramic.figure[kind] !== `assets/pino-studies/${name}`) {
    throw new Error(`Unexpected ceramic comparison asset: ${kind}`);
  }
  const data = await readFile(join(root, '..', ceramic.figure[kind]));
  if (data.length !== ceramic.bytes[kind] || data.length >= ceramic.maximum_file_bytes) {
    throw new Error(`Ceramic comparison asset differs from its approval: ${kind}`);
  }
  await writeFile(join(assets, name), data);
}

await Promise.all(rendered.map(([slug, html]) => writeFile(join(output, `${slug}.html`), html)));
console.log(`Built ${rendered.length} technical report pages in website/dist/`);
