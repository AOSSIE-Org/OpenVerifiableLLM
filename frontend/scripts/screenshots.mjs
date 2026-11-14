// Full-page screenshots of every page at 1440 and 390, light and dark, from the static build.
//   node scripts/static-server.mjs & OUT=/path/to/dir node scripts/screenshots.mjs
import { mkdirSync } from 'node:fs';
import { chromium } from '@playwright/test';

const out = process.env.OUT ?? 'screenshots';
const base = process.env.URL ?? 'http://127.0.0.1:4173/';
const pages = [['overview', ''], ['check', 'check/'], ['verification', 'verification/'], ['timeline', 'timeline/'], ['exclusion', 'exclusion/'], ['downloads', 'downloads/']];
mkdirSync(out, { recursive: true });
const browser = await chromium.launch();
for (const scheme of ['light', 'dark']) {
  for (const width of [1440, 390]) {
    const ctx = await browser.newContext({ viewport: { width, height: 900 }, reducedMotion: 'reduce', deviceScaleFactor: width === 390 ? 2 : 1 });
    const page = await ctx.newPage();
    await page.addInitScript((t) => {
      const html = document.documentElement;
      if (!html) return;
      html.setAttribute('data-theme', t);
      html.style.colorScheme = t;
      try { localStorage.setItem('ovl-theme', t); } catch { /* ignore */ }
    }, scheme);
    for (const [name, slug] of pages) {
      await page.goto(base + slug);
      await page.waitForSelector('footer [data-path="revShort"]');
      await page.waitForLoadState('networkidle');
      await page.evaluate(() => document.fonts.ready);
      const file = `${out}/${name}-${width}-${scheme}.png`;
      await page.screenshot({ path: file, fullPage: true });
      console.log(file);
    }
    await ctx.close();
  }
}
await browser.close();
