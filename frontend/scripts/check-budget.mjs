// Phone performance budget, checked on the real build output (gzip level 9).
import { readFileSync, readdirSync, statSync } from 'node:fs';
import path from 'node:path';
import { gzipSync } from 'node:zlib';

export const BUDGET = { entryJsKB: 80, totalJsKB: 100, pageJsKB: 12, cssKB: 10, htmlKB: 30, fontsKB: 125, dataKB: 60 };
const gz = (f) => gzipSync(readFileSync(f), { level: 9 }).length / 1024;
const files = (d) => readdirSync(d).flatMap((f) => (statSync(path.join(d, f)).isDirectory() ? files(path.join(d, f)) : [path.join(d, f)]));
const all = files('dist');
const html = readFileSync('dist/index.html', 'utf8');
const entry = /<script type="module" crossorigin src="[^"]*assets\/([^"]+\.js)"/.exec(html)?.[1];
if (!entry) throw new Error('budget: entry script not found');
const preloaded = [...html.matchAll(/modulepreload" crossorigin href="[^"]*assets\/([^"]+\.js)"/g)].map((m) => m[1]);
const js = all.filter((f) => f.endsWith('.js'));
const initial = [entry, ...preloaded].reduce((n, f) => n + gz(path.join('dist/assets', f)), 0);
const pages = js.filter((f) => ![entry, ...preloaded].includes(path.basename(f)));
const rows = [
  ['initial JS (entry + preloads)', initial, BUDGET.entryJsKB],
  ['all JS (every page)', js.reduce((n, f) => n + gz(f), 0), BUDGET.totalJsKB],
  ['largest lazy page chunk', Math.max(0, ...pages.map(gz)), BUDGET.pageJsKB],
  ['CSS', all.filter((f) => f.endsWith('.css')).reduce((n, f) => n + gz(f), 0), BUDGET.cssKB],
  ['largest prerendered HTML', Math.max(...all.filter((f) => f.endsWith('.html')).map(gz)), BUDGET.htmlKB],
  ['fonts (raw woff2)', all.filter((f) => f.endsWith('.woff2')).reduce((n, f) => n + statSync(f).size / 1024, 0), BUDGET.fontsKB],
  ['site-data.json', gz('dist/data/site-data.json'), BUDGET.dataKB],
];
let fail = false;
for (const [name, kb, max] of rows) {
  const ok = kb <= max;
  fail ||= !ok;
  console.log(`${ok ? 'PASS' : 'FAIL'} ${name}: ${kb.toFixed(1)} KB (budget ${max} KB)`);
}
if (all.some((f) => /\.(png|jpe?g|gif|webp|mp4)$/i.test(f))) { console.log('FAIL images in build'); fail = true; }
if (fail) process.exit(1);
