import { limitParts } from '../../src/components/ClaimBlock';
import AxeBuilder from '@axe-core/playwright';
import { expect, test } from '@playwright/test';
import { formatValue, type Fmt } from '../../src/data/format';
import { getPath } from '../../src/data/path';
import { bannedIn, claims, ui } from '../../src/content/ledger';
import { PAGES, WIDTHS, data, ready } from './util';

for (const scheme of ['light', 'dark'] as const) {
  for (const w of WIDTHS) {
    for (const p of PAGES) {
      test(`${p.key} @${w} ${scheme}: renders, no overflow, axe clean, no console errors`, async ({ page }) => {
        const errors: string[] = [];
        page.on('console', (m) => m.type() === 'error' && errors.push(m.text()));
        page.on('pageerror', (e) => errors.push(String(e)));
        await page.emulateMedia({ colorScheme: scheme, reducedMotion: 'reduce' });
        await page.setViewportSize({ width: w, height: 900 });
        await page.goto(p.slug);
        await ready(page);
        await expect(page.locator('h1')).toBeVisible();
        await expect(page.locator('h1')).toHaveText(ui[`h1.${p.key}`]!);
        const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
        expect(overflow).toBeLessThanOrEqual(0);
        const axe = await new AxeBuilder({ page }).withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa', 'wcag22aa']).analyze();
        expect(axe.violations.map((v) => `${v.id}: ${v.nodes.length} (${v.nodes[0]?.target.join(' ')})`)).toEqual([]);
        expect(errors).toEqual([]);
      });
    }
  }
}

for (const p of PAGES) {
  test(`${p.key}: every text on the page comes from the ledger, ui.json or site-data.json`, async ({ page }) => {
    await page.goto(p.slug);
    await ready(page);
    const stray = await page.evaluate(() => {
      const out: string[] = [];
      const walk = document.createTreeWalker(document.querySelector('main')!, NodeFilter.SHOW_TEXT);
      for (let n = walk.nextNode(); n; n = walk.nextNode()) {
        const t = n.textContent ?? '';
        if (!/[\p{L}\p{N}]/u.test(t)) continue; // whitespace and punctuation only
        if (!n.parentElement?.closest('[data-claim-part],[data-label],[data-path],[data-verbatim]')) out.push(t.trim());
      }
      return out;
    });
    expect(stray).toEqual([]);
  });

  test(`${p.key}: every displayed value equals site-data.json and every claim equals the ledger`, async ({ page }) => {
    const d = data();
    await page.goto(p.slug);
    await ready(page);
    const vals = await page.$$eval('main [data-path]', (els) => els.map((e) => ({ path: (e as HTMLElement).dataset.path!, fmt: (e as HTMLElement).dataset.fmt ?? 'auto', text: e.textContent })));
    for (const v of vals) expect(v.text, v.path).toBe(formatValue(getPath(d, v.path), v.fmt as Fmt));
    const parts = await page.$$eval('main [data-claim-part]', (els) => els.map((e) => ({ id: (e as HTMLElement).dataset.claimRef ?? e.closest<HTMLElement>('[data-claim]')!.dataset.claim!, part: (e as HTMLElement).dataset.claimPart!, text: e.textContent })));
    expect(parts.length).toBeGreaterThan(0);
    for (const x of parts) {
      const c = claims.get(x.id)!;
      expect(x.text?.trim(), x.id).toBe(x.part === 'plain' ? c.plain : x.part === 'developer' ? c.developer : x.part === 'limit-text' ? limitParts(c.plain).rest : c.limit);
    }
    const labels = await page.$$eval('[data-label]', (els) => els.map((e) => [(e as HTMLElement).dataset.label!, e.textContent]));
    for (const [k, t] of labels) expect(t).toBe(ui[k!]);
  });

  test(`${p.key}: no banned phrase anywhere on the page`, async ({ page }) => {
    await page.goto(p.slug);
    await ready(page);
    const text = await page.evaluate(() => document.body.innerText + ' ' + document.title);
    expect(bannedIn(text.replace(/\n+/g, '\n'))).toEqual([]);
  });

  test(`${p.key}: no banned phrase in the shipped HTML, including closed source panels`, async ({ request }) => {
    const html = await (await request.get(p.slug)).text();
    const text = html.replace(/<[^>]+>/g, ' ').replace(/&#x27;|&#39;/g, "'").replace(/&amp;/g, '&');
    // Includes verbatim upstream quotes in closed chips, so only the ledger's banned phrases apply here;
    // the site's own style rules are checked on the rendered text above.
    expect(bannedIn(text, undefined, false)).toEqual([]);
    expect(text.toLowerCase().replaceAll('unlearning-3i-atlas', '')).not.toMatch(/erase|unlearn/);
  });

  test(`${p.key}: core content is in the HTML without JavaScript`, async ({ browser }) => {
    const ctx = await browser.newContext({ javaScriptEnabled: false });
    const page = await ctx.newPage();
    await page.goto(p.slug);
    await expect(page.locator('h1')).toHaveText(ui[`h1.${p.key}`]!);
    expect(await page.locator('main [data-claim]').count()).toBeGreaterThan(0);
    await ctx.close();
  });
}

test('links: every internal link resolves on the static build; external links are https and pinned', async ({ page, request }) => {
  const d = data();
  const hrefs = new Set<string>();
  for (const p of PAGES) {
    await page.goto(p.slug);
    await ready(page);
    for (const h of await page.$$eval('a[href]', (as) => as.map((a) => (a as HTMLAnchorElement).href))) hrefs.add(h);
  }
  const internal = [...hrefs].filter((h) => h.startsWith('http://127.0.0.1:4173/'));
  expect(internal.length).toBeGreaterThan(6);
  for (const h of internal) expect((await request.get(h.split('#')[0]!)).status(), h).toBe(200);
  for (const h of [...hrefs].filter((x) => !x.startsWith('http://127.0.0.1:4173/'))) {
    expect(h, h).toMatch(/^https:\/\//);
    if (h.includes('huggingface.co/datasets/')) expect(h, h).toContain(d.revision);
    if (h.includes('github.com/AOSSIE-Org/OpenVerifiableLLM/blob/')) expect(h, h).toContain(d.upstream.commit);
  }
  for (const f of ['fonts/InterVariable-latin.woff2', 'data/site-data.json']) expect((await request.get(f)).status()).toBe(200);
});
