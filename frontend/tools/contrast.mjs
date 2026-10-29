// WCAG 2.x contrast check of the tokens actually shipped in src/styles/site.css (light and dark).
// Text pairs need 4.5:1; non-text indicators 3:1. The focus indicator is a two-layer ring
// (gold outline + ink halo), so it passes when either layer reaches 3:1 against the page.
import { readFileSync } from 'node:fs';

const css = readFileSync(new URL('../src/styles/site.css', import.meta.url), 'utf8');
const block = (re) => Object.fromEntries([...(re.exec(css)?.[1] ?? '').matchAll(/--([a-z0-9-]+):(#[0-9A-Fa-f]{6})/g)].map((m) => [m[1], m[2]]));
const light = block(/:root\{([\s\S]*?)\n\}/);
const dark = { ...light, ...block(/@media \(prefers-color-scheme:dark\)\{:root\{([\s\S]*?)\}\}/) };
const lin = (c) => ((c /= 255) <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
const L = (h) => {
  const [r, g, b] = [1, 3, 5].map((i) => lin(parseInt(h.slice(i, i + 2), 16)));
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
};
export const ratio = (a, b) => {
  const [x, y] = [L(a), L(b)].sort((p, q) => q - p);
  return (x + 0.05) / (y + 0.05);
};
let fails = 0, n = 0;
const check = (mode, name, fg, bg, need) => {
  const r = ratio(fg, bg);
  n++;
  const ok = r >= need;
  if (!ok) fails++;
  console.log(`${ok ? 'PASS' : 'FAIL'} ${mode.padEnd(5)} ${name.padEnd(42)} ${r.toFixed(2)}:1 (need ${need})`);
};
for (const [mode, T] of [['light', light], ['dark', dark]]) {
  for (const t of ['ink', 'ink-2', 'muted', 'link', 'match', 'mismatch', 'project', 'independent', 'pending', 'stamp'])
    for (const bg of ['bg', 'surface', `${t}-tint`, ...(t === 'ink' || t === 'muted' ? ['sunk'] : [])]) if (T[bg]) check(mode, `${t} on ${bg}`, T[t], T[bg], 4.5);
  check(mode, 'surface text on solid independent', T.surface, T.independent, 4.5);
  check(mode, 'link on sunk (chips, panels)', T.link, T.sunk, 4.5);
  check(mode, 'project tile border on surface', T.project, T.surface, 3);
  check(mode, 'pending dashed border on bg', T.pending, T.bg, 3);
  const best = Math.max(ratio(T.focus, T.bg), ratio(T['focus-ink'], T.bg));
  n++;
  const ok = best >= 3 && ratio(T['focus-ink'], T.focus) >= 3;
  if (!ok) fails++;
  console.log(`${ok ? 'PASS' : 'FAIL'} ${mode.padEnd(5)} ${'two-layer focus ring (best layer on bg)'.padEnd(42)} ${best.toFixed(2)}:1 (need 3)`);
}
console.log(`${n - fails}/${n} pairs pass`);
if (fails) process.exit(1);
