// Render every route to static HTML after `vite build` + `vite build --ssr`, so pages read without JavaScript.
import { mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const { renderAll } = await import(pathToFileURL(path.resolve('dist-ssr/entry-server.js')).href);
const template = readFileSync('dist/index.html', 'utf8');
if (!template.includes('<div id="root"><!--app--></div>')) throw new Error('prerender: root marker not found');
const data = JSON.parse(readFileSync('dist/data/site-data.json', 'utf8'));
const titleOf = (html, key) => (new RegExp(`data-label="nav.${key}">([^<]*)<`).exec(html)?.[1] ?? '');
for (const r of renderAll(data)) {
  const html = template
    .replace('<div id="root"><!--app--></div>', `<div id="root">${r.html}</div>`)
    .replace('<title>OpenVerifiableLLM</title>', `<title>${titleOf(r.html, r.key)} · OpenVerifiableLLM</title>`);
  const dir = path.join('dist', r.slug);
  mkdirSync(dir, { recursive: true });
  writeFileSync(path.join(dir, 'index.html'), html);
  console.log(`prerender: /${r.slug} (${(html.length / 1024).toFixed(1)} KB)`);
}
rmSync('dist-ssr', { recursive: true, force: true });
