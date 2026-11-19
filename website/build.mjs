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
if (manifest.schema !== 'pino_main_publication_v2' ||
    manifest.figures.length !== casePages.length ||
    new Set(manifest.figures.map(figure => figure.example)).size !== casePages.length) {
  throw new Error('Expected one approved publication figure per case chapter');
}
let figureBytes = 0;
for (const { source } of casePages) {
  const figure = manifest.figures.find(record => record.example === source);
  if (!figure) throw new Error(`Missing publication figure: ${source}`);
  if (!Number.isInteger(figure.cohort_count) || figure.cohort_count < 3 ||
      figure.samples.length !== 3 || new Set(figure.samples).size !== 3 ||
      figure.phase_errors.length !== 3 ||
      figure.phase_errors.some((value, i) => !Number.isFinite(value) || value < 0 ||
        (i > 0 && value < figure.phase_errors[i - 1]))) {
    throw new Error(`Invalid phase-error-ranked selection for ${source}`);
  }
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

const ceramicComponents = JSON.parse(await readFile(join(root, '..', 'artifacts', 'ceramic_component_comparison_figures.json'), 'utf8'));
if (ceramicComponents.schema !== 'ceramic_component_comparison_publication_v1' ||
    ceramicComponents.test_samples !== 64 || ceramicComponents.development_samples !== 4 ||
    ceramicComponents.checkpoint_step !== 20000 || ceramicComponents.figures.length !== 2 ||
    JSON.stringify(ceramicComponents.convergence_steps) !== JSON.stringify(Array.from({length: 21}, (_, i) => i * 1000))) {
  throw new Error('Expected ceramic two-component, three-method final64/development4 figures');
}
let ceramicComponentBytes = 0;
for (const figure of ceramicComponents.figures) {
  if (!['convergence', 'distribution'].includes(figure.kind)) throw new Error('Unknown ceramic component figure');
  for (const kind of ['png', 'pdf']) {
    const name = `particle_corrosion2d_component_${figure.kind}.${kind}`;
    if (figure[kind] !== `assets/pino-studies/${name}`) throw new Error('Unexpected ceramic component path');
    const data = await readFile(join(root, '..', figure[kind]));
    if (data.length !== figure.bytes[kind] || data.length >= ceramicComponents.maximum_file_bytes) {
      throw new Error(`Ceramic component figure differs from its record: ${name}`);
    }
    ceramicComponentBytes += data.length;
    await writeFile(join(assets, name), data);
  }
}
if (ceramicComponentBytes !== ceramicComponents.collection_bytes ||
    ceramicComponentBytes >= ceramicComponents.maximum_collection_bytes) {
  throw new Error('Ceramic component figures exceed their recorded size/budget');
}

const channel = JSON.parse(await readFile(join(root, '..', 'artifacts', 'channel_fem_comparison_publication.json'), 'utf8'));
if (channel.schema !== 'channel_fem_comparison_publication_v3' ||
    channel.test_samples !== 3 || channel.development_samples !== 1 ||
    channel.convergence_interval !== 'approximate_pointwise95_trend_bootstrap' ||
    JSON.stringify(channel.convergence_steps) !== JSON.stringify(Array.from({length: 41}, (_, i) => i * 500)) ||
    channel.checkpoint_step !== 20000 || channel.figures.length !== 2 ||
    new Set(channel.figures.map(figure => figure.kind)).size !== 2) {
  throw new Error('Expected the approved development trend and reserved-three violin comparison');
}
let channelBytes = 0;
for (const figure of channel.figures) {
  if (!['convergence', 'distribution'].includes(figure.kind)) {
    throw new Error(`Unexpected channel figure: ${figure.kind}`);
  }
  for (const kind of ['png', 'pdf']) {
    const name = `channel_nsch2d_fem_${figure.kind}.${kind}`;
    if (figure[kind] !== `assets/pino-studies/${name}`) {
      throw new Error(`Unexpected channel comparison path: ${figure[kind]}`);
    }
    const data = await readFile(join(root, '..', figure[kind]));
    if (data.length !== figure.bytes[kind] || data.length >= channel.maximum_file_bytes) {
      throw new Error(`Channel comparison asset differs from its approval: ${name}`);
    }
    channelBytes += data.length;
    await writeFile(join(assets, name), data);
  }
}
if (channelBytes !== channel.total_bytes || channelBytes >= channel.maximum_collection_bytes) {
  throw new Error('Channel comparison collection exceeds its approved budget');
}

const threeMethod = JSON.parse(await readFile(join(root, '..', 'artifacts', 'pino_three_method_figures.json'), 'utf8'));
const comparisonCases = {
  channel_nsch2d: { budget: 20000, monitor: 1, test: 3, interval: 500 },
  electropolish2d: { budget: 10000, monitor: 64, test: 64, interval: 250 },
  icemelting3d: { budget: 10000, monitor: 64, test: 64, interval: 250 },
  confined_ch2d: { budget: 50000, monitor: 8, test: 8, interval: 500 },
};
if (threeMethod.schema !== 'pino_three_method_publication_v1' ||
    threeMethod.maximum_file_bytes !== 1024 ** 2 ||
    threeMethod.maximum_collection_bytes !== 4 * 1024 ** 2 ||
    !threeMethod.cases.length ||
    new Set(threeMethod.cases.map(record => record.example)).size !== threeMethod.cases.length) {
  throw new Error('Expected unique completed three-method comparison figures');
}
let threeMethodBytes = 0;
for (const record of threeMethod.cases) {
  const expected = comparisonCases[record.example];
  const twoMethod = record.comparison_kind === 'spinodal_nonlocal_fem';
  const methods = twoMethod ? ['nonlocal', 'fem'] : ['nonlocal', 'local_ad', 'fem'];
  const suffix = twoMethod ? 'nonlocal_fem' : 'three_method';
  if ((twoMethod && (record.example !== 'confined_ch2d' ||
      record.omitted_method !== 'local_ad: omitted by user after resource measurement')) ||
      ('comparison_kind' in record && !twoMethod)) {
    throw new Error('Only the declared Spinodal Local AD omission is supported');
  }
  if (!expected || record.budget !== expected.budget ||
      record.monitor_ids.length !== expected.monitor || record.test_ids.length !== expected.test ||
      JSON.stringify(record.steps) !== JSON.stringify(Array.from(
        { length: expected.budget / expected.interval + 1 }, (_, i) => i * expected.interval)) ||
      JSON.stringify(record.arms.map(arm => arm.method)) !== JSON.stringify(methods) ||
      record.figures.length !== 2 || new Set(record.figures.map(figure => figure.kind)).size !== 2) {
    throw new Error(`Incomplete or mismatched comparison: ${record.example}`);
  }
  for (const figure of record.figures) {
    if (!['convergence', 'distribution'].includes(figure.kind)) throw new Error('Unknown three-method figure');
    for (const kind of ['png', 'pdf']) {
      const name = `${record.example}_${suffix}_${figure.kind}.${kind}`;
      if (figure[kind] !== `assets/pino-studies/${name}`) throw new Error(`Unexpected comparison path: ${name}`);
      const data = await readFile(join(root, '..', figure[kind]));
      if (data.length !== figure.bytes[kind] || data.length >= threeMethod.maximum_file_bytes) {
        throw new Error(`Comparison figure differs from its record: ${name}`);
      }
      threeMethodBytes += data.length;
      await writeFile(join(assets, name), data);
    }
  }
}
if (threeMethodBytes !== threeMethod.collection_bytes || threeMethodBytes >= threeMethod.maximum_collection_bytes) {
  throw new Error('Three-method figures exceed their recorded size/budget');
}

const timeFusion = JSON.parse(await readFile(join(root, '..', 'artifacts', 'pino_time_fusion_figures.json'), 'utf8'));
const fusionClocks = JSON.parse(await readFile(join(root, '..', 'artifacts', 'pino_time_fusion_wall_clock.json'), 'utf8'));
if (fusionClocks.schema !== 'pino_time_fusion_clock_v1' || fusionClocks.status !== 'passed') {
  throw new Error('Qualified cold wall-clock evidence is required for temporal figures');
}
if (timeFusion.schema !== 'pino_time_fusion_publication_v1' ||
    timeFusion.maximum_file_bytes !== 1024 ** 2 ||
    timeFusion.maximum_collection_bytes !== 2 * 1024 ** 2 ||
    new Set(timeFusion.cases.map(record => record.example)).size !== timeFusion.cases.length) {
  throw new Error('Expected unique, reviewed temporal-control figures');
}
let timeFusionBytes = 0;
for (const record of timeFusion.cases) {
  const temporalBudgets = { electropolish2d: [[10000, 15000], 250], icemelting3d: [[10000], 250], particle_corrosion2d: [[20000], 500] };
  const expected = temporalBudgets[record.example];
  if (!expected ||
      record.comparison_kind !== 'time_fusion' || !expected[0].includes(record.budget) ||
      record.monitor_ids.length !== 64 || new Set(record.test_ids).size !== 64 ||
      JSON.stringify(record.monitor_ids) !== JSON.stringify(record.test_ids) ||
      JSON.stringify(record.steps) !== JSON.stringify(Array.from({ length: record.budget / expected[1] + 1 }, (_, i) => i * expected[1])) ||
      JSON.stringify(record.arms.map(arm => arm.method)) !== JSON.stringify(['decoder_film', 'joint_query']) ||
      record.pairing.shared_initial_backbone_equal !== true ||
      record.pairing.all_actual_sampling_arrays_equal !== true ||
      record.pairing.source_bytes_verified !== true || record.figures.length !== 2 ||
      new Set(record.figures.map(figure => figure.kind)).size !== 2) {
    throw new Error(`Incomplete or mismatched temporal control: ${record.example}`);
  }
  if (record.example === 'electropolish2d' && record.budget === 15000 &&
      record.supersedes?.reason !== 'User-requested paired cold15k controls supersede10k study') {
    throw new Error('Electropolish15k requires the explicitly audited replacement of its10k study');
  }
  const clocks = fusionClocks.cases.filter(clock => clock.example === record.example);
  if (record.convergence_axis.kind !== 'cold_wall_time' || clocks.length !== 1 ||
      JSON.stringify(clocks[0].steps) !== JSON.stringify(record.steps)) {
    throw new Error(`Missing cold elapsed axis: ${record.example}`);
  }
  for (const arm of record.arms) {
    const matched = clocks[0].arms.filter(clock => clock.method === arm.method);
    if (matched.length !== 1 || matched[0].run !== arm.run ||
        JSON.stringify(matched[0].elapsed_seconds) !== JSON.stringify(arm.elapsed_seconds) ||
        arm.elapsed_seconds.length !== record.steps.length ||
        arm.elapsed_seconds.some((value, i, all) => !Number.isFinite(value) || value <= 0 || (i && value <= all[i - 1])) ||
        arm.elapsed_seconds.at(-1) !== arm.timing.cold_process_seconds) {
      throw new Error(`Cold elapsed observations differ: ${record.example}/${arm.method}`);
    }
  }
  for (const figure of record.figures) {
    if (!['convergence', 'distribution'].includes(figure.kind)) throw new Error('Unknown temporal figure');
    for (const kind of ['png', 'pdf']) {
      const name = `${record.example}_time_fusion_${figure.kind}.${kind}`;
      if (figure[kind] !== `assets/pino-studies/${name}`) throw new Error(`Unexpected temporal path: ${name}`);
      const data = await readFile(join(root, '..', figure[kind]));
      if (data.length !== figure.bytes[kind] || data.length >= timeFusion.maximum_file_bytes) {
        throw new Error(`Temporal figure differs from its record: ${name}`);
      }
      timeFusionBytes += data.length;
      await writeFile(join(assets, name), data);
    }
  }
}
if (timeFusionBytes !== timeFusion.collection_bytes || timeFusionBytes >= timeFusion.maximum_collection_bytes) {
  throw new Error('Temporal figures exceed their recorded size/budget');
}

const libraryStudy = JSON.parse(await readFile(join(root, '..', 'artifacts', 'electropolish_library15k_results.json'), 'utf8'));
const libraryCounts = [32, 64, 128, 256, 512];
if (libraryStudy.status !== 'passed' || libraryStudy.schema !== 'electropolish_library15k_v1' ||
    libraryStudy.example !== 'electropolish2d' || libraryStudy.budget !== 15000 ||
    JSON.stringify(libraryStudy.test_ids) !== JSON.stringify(Array.from({ length: 64 }, (_, i) => i)) ||
    JSON.stringify(libraryStudy.counts) !== JSON.stringify(libraryCounts) ||
    JSON.stringify(libraryStudy.fixed.map(record => record.count)) !== JSON.stringify(libraryCounts) ||
    JSON.stringify(libraryStudy.publication.selected_examples) !== JSON.stringify(['electropolish2d']) ||
    libraryStudy.publication.maximum_file_bytes !== 192 * 1024 ||
    libraryStudy.publication.maximum_collection_bytes !== 256 * 1024) {
  throw new Error('The complete reviewed Electropolish-only15k library study is required');
}
for (const record of [...libraryStudy.fixed, libraryStudy.online]) {
  if (record.budget !== 15000 || record.sample_count !== 64 ||
      record.channel_errors.length !== 64 ||
      record.channel_errors.some(row => row.length !== 2 || row.some(v => !Number.isFinite(v) || v <= 0)) ||
      !Number.isFinite(record.mean) || !Number.isFinite(record.sample_std) ||
      !Number.isFinite(record.cold_process_seconds) || record.cold_process_seconds <= 0) {
    throw new Error('Incomplete final15k library endpoint');
  }
}
let libraryBytes = 0;
for (const kind of ['png', 'pdf']) {
  const name = `electropolish2d_library_size_error.${kind}`;
  if (libraryStudy.publication[kind] !== `assets/pino-library-size/${name}`) throw new Error('Unexpected library figure path');
  const data = await readFile(join(root, '..', libraryStudy.publication[kind]));
  if (data.length !== libraryStudy.publication.bytes[kind] || data.length >= libraryStudy.publication.maximum_file_bytes) {
    throw new Error(`Library figure differs from its approval: ${name}`);
  }
  libraryBytes += data.length;
  await writeFile(join(assets, name), data);
}
if (libraryBytes >= libraryStudy.publication.maximum_collection_bytes) throw new Error('Library figure budget exceeded');

const timeAccuracy = JSON.parse(await readFile(join(root, '..', 'artifacts', 'ceramic_time_derivative_accuracy.json'), 'utf8'));
const accuracyIntervals = [null, 25, 50, 100, 150, 200];
if (timeAccuracy.schema !== 'ceramic_time_derivative_accuracy_v1' ||
    timeAccuracy.example !== 'particle_corrosion2d' || timeAccuracy.final_updates !== 20000 ||
    timeAccuracy.test_ids.length !== 64 || timeAccuracy.test_ids.some((id, index) => id !== index) ||
    timeAccuracy.plot.yscale !== 'log' || timeAccuracy.cases.length !== accuracyIntervals.length ||
    timeAccuracy.cases.some((entry, index) =>
      entry.intervals !== accuracyIntervals[index] || entry.method !== (index === 0 ? 'jvp' : 'bdf2') ||
      entry.relative_l2_by_channel.length !== 64 ||
      entry.relative_l2_by_channel.some(row => row.length !== 2 || row.some(v => !Number.isFinite(v) || v <= 0)))) {
  throw new Error('Expected six complete, paired Ceramic time-derivative accuracy distributions');
}
for (const kind of ['png', 'pdf']) {
  const name = `particle_corrosion2d_time_derivatives.${kind}`;
  const figure = timeAccuracy.assets[kind];
  if (figure.path !== `assets/pino-time-derivatives/${name}`) throw new Error(`Unexpected accuracy path: ${name}`);
  const data = await readFile(join(root, '..', figure.path));
  if (data.length !== figure.bytes || data.length >= 192 * 1024) {
    throw new Error(`Accuracy figure differs from its approval: ${name}`);
  }
  await writeFile(join(assets, name), data);
}

await Promise.all(rendered.map(([slug, html]) => writeFile(join(output, `${slug}.html`), html)));
console.log(`Built ${rendered.length} technical report pages in website/dist/`);
