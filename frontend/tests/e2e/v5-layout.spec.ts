import { test, expect } from '@playwright/test';
import { PAGES, ready } from './util';

const TILE_WIDTHS = [375, ...Array.from({ length: 23 }, (_, i) => 641 + i * 20), 1099, 1100, 1280, 1440];

const STRIPS = [
  { name: 'overview strip', slug: '', sel: '.overview-top .tiles-strip .tile' },
  { name: 'verification map', slug: 'verification/', sel: 'main ol.tiles:not(.tiles-indep) .tile' },
] as const;

for (const s of STRIPS) test.describe(`${s.name}: every number fits its tile and clears the code-change bar by 4px`, () => {
  for (const w of TILE_WIDTHS) {
    test(`@${w}`, async ({ page }) => {
      await page.setViewportSize({ width: w, height: 900 });
      await page.goto(s.slug);
      await ready(page);
      const r = await page.evaluate((sel) => {
        const text = (el: Element) => { const g = document.createRange(); g.selectNodeContents(el); return g.getBoundingClientRect(); };
        const tiles = [...document.querySelectorAll<HTMLElement>(sel)];
        const boxes = tiles.map((t) => t.getBoundingClientRect());
        const bar = parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--bar-w'));
        const clipped: string[] = [];
        let clear = Infinity;
        for (const t of tiles) {
          const c = t.getBoundingClientRect();
          const bw = parseFloat(getComputedStyle(t).borderLeftWidth);
          const v = t.querySelector('.v')!;
          for (const el of [v, t.querySelector('.glyph')]) {
            if (!el || el.getBoundingClientRect().width <= 1) continue; // visually hidden below 641px
            const a = text(el);
            if (a.left < c.left + bw - 0.01 || a.right > c.right - bw + 0.01 || a.top < c.top || a.bottom > c.bottom) clipped.push(v.textContent!);
          }
          if (t.dataset.event?.includes('code-change') && v.getBoundingClientRect().width > 1) clear = Math.min(clear, text(v).left - (c.left + bw + bar));
        }
        const spread = (a: number[]) => Math.max(...a) - Math.min(...a);
        return { n: tiles.length, w: spread(boxes.map((b) => b.width)), h: spread(boxes.map((b) => b.height)), clipped, clear, // Content may use the right padding (no bar there) but never reaches the border.
          scroll: tiles.filter((t) => t.scrollWidth - t.clientWidth > parseFloat(getComputedStyle(t).paddingRight)).length };
      }, s.sel);
      expect(r.n).toBe(37);
      expect(r.w).toBeLessThan(0.05);
      expect(r.h).toBeLessThan(0.05);
      expect(r.clipped).toEqual([]);
      expect(r.scroll).toBe(0);
      if (r.clear !== Infinity) expect(r.clear).toBeGreaterThanOrEqual(4);
      if (s.slug || w > 640) expect(r.clear).not.toBe(Infinity); // the number is visible here, so it must be measured
    });
  }
});

test.describe('no sideways scroll on phones with every chip open', () => {
  for (const w of [320, 375, 390]) {
    for (const p of PAGES) {
      test(`${p.key} @${w}`, async ({ page }) => {
        await page.setViewportSize({ width: w, height: 800 });
        await page.goto(p.slug);
        await ready(page);
        // Each chip alone, then every disclosure on the page open at once.
        const r = await page.evaluate(async () => {
          const frame = () => new Promise(requestAnimationFrame);
          const over = () => document.documentElement.scrollWidth - innerWidth;
          const bad: string[] = [];
          for (const d of document.querySelectorAll<HTMLDetailsElement>('details.chip')) {
            d.open = true;
            await frame();
            if (over() > 0) bad.push(`${d.querySelector('summary')!.textContent!.slice(0, 60)} +${over()}px`);
            d.open = false;
          }
          document.querySelectorAll('details').forEach((d) => (d.open = true));
          await frame();
          return { bad, all: over() };
        });
        expect(r.bad).toEqual([]);
        expect(r.all).toBeLessThanOrEqual(0);
      });
    }
  }
});
