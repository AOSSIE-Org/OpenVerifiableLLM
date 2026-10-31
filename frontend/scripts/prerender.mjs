// Render every route to static HTML after `vite build` + `vite build --ssr`, so pages read without JavaScript.
import { copyFileSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const { renderAll } = await import(pathToFileURL(path.resolve('dist-ssr/entry-server.js')).href);
const template = readFileSync('dist/index.html', 'utf8');
if (!template.includes('<div id="root"><!--app--></div>')) throw new Error('prerender: root marker not found');
const ui = JSON.parse(readFileSync('src/content/ui.json', 'utf8'));
const attr = (s) => String(s).replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/</g, '&lt;');
const bytes = readFileSync('dist/data/site-data.json');
const data = JSON.parse(bytes.toString('utf8'));
// The HTML names the exact data it was rendered from: a content-hashed copy (safe to cache forever)
// and its sha256, which the client checks before hydrating. Other data never hydrates this HTML.
const sha = createHash('sha256').update(bytes).digest('hex');
copyFileSync('dist/data/site-data.json', `dist/data/site-data.${sha.slice(0, 16)}.json`);
const dataMeta = `<meta name="ovl-data" content="${sha}" />`;
// GitHub Pages cannot set headers, so the policy ships as a meta tag (frame-ancestors is header-only).
// Built pages have no inline script or style element; the timeline's per-cell flex-grow is a style attribute.
const CSP = "default-src 'none'; script-src 'self'; style-src 'self'; style-src-attr 'unsafe-inline'; img-src 'self' data:; font-src 'self'; connect-src 'self'; base-uri 'none'; form-action 'none'; object-src 'none'";
const cspMeta = `<meta http-equiv="Content-Security-Policy" content="${CSP}" />`;
const titleOf = (html, key) => (new RegExp(`data-label="${key === 'notfound' ? 'notfound' : `nav.${key}`}">([^<]*)<`).exec(html)?.[1] ?? '');
for (const r of renderAll(data)) {
  const pageTitle = titleOf(r.html, r.key);
  const fullTitle = pageTitle ? `${pageTitle} · OpenVerifiableLLM` : 'OpenVerifiableLLM';
  const desc = ui[`meta.${r.key}`] ?? ui['meta.default'] ?? '';
  const html = template
    .replace('<div id="root"><!--app--></div>', () => `<div id="root">${r.html}</div>`)
    .replace('<title>OpenVerifiableLLM</title>', () => `<title>${attr(fullTitle)}</title>`)
    .replace(/<meta name="description" content="[^"]*" \/>/, () => `<meta name="description" content="${attr(desc)}" />`)
    .replace(/<meta property="og:title" content="[^"]*" \/>/, () => `<meta property="og:title" content="${attr(fullTitle)}" />`)
    .replace(/<meta property="og:description" content="[^"]*" \/>/, () => `<meta property="og:description" content="${attr(desc)}" />`)
    .replace('</head>', () => `  ${dataMeta}\n  </head>`)
    .replace(/(<meta charset="UTF-8" \/>)/, (_, m) => `${m}\n    ${cspMeta}`);
  // 404.html is static: no app script, no preloads, so it never fetches site-data.json.
  if (r.slug.endsWith('.html')) writeFileSync(path.join('dist', r.slug), html.replace(/<script type="module"[^>]*><\/script>\s*/g, '').replace(/<link rel="modulepreload"[^>]*>\s*/g, ''));
  else {
    const dir = path.join('dist', r.slug);
    mkdirSync(dir, { recursive: true });
    writeFileSync(path.join(dir, 'index.html'), html);
  }
  console.log(`prerender: /${r.slug} (${(html.length / 1024).toFixed(1)} KB)`);
}
rmSync('dist-ssr', { recursive: true, force: true });
