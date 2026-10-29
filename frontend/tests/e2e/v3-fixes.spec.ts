import AxeBuilder from '@axe-core/playwright';
import { expect, test, type Page } from '@playwright/test';
import { label } from '../../src/content/ledger';
import { PAGES, ready } from './util';

const PINNED_GH = '1049d2a7';
const PINNED_HF = '96dc282a';

test.describe('layout', () => {
  for (const w of [320, 768, 1024, 1920]) {
    for (const p of PAGES) {
      test(`${p.key} @${w}: no horizontal scroll`, async ({ page }) => {
        await page.setViewportSize({ width: w, height: 900 });
        await page.goto(p.slug);
        await ready(page);
        const { sw, cw } = await page.evaluate(() => ({ sw: document.documentElement.scrollWidth, cw: document.documentElement.clientWidth }));
        expect(sw).toBeLessThanOrEqual(cw);
      });
    }
  }

  for (const w of [390, 768, 1440]) {
    test(`minimum font size is at least 12px on every page @${w}`, async ({ page }) => {
      await page.setViewportSize({ width: w, height: 900 });
      for (const p of PAGES) {
        await page.goto(p.slug);
        await ready(page);
        await page.evaluate(() => document.querySelectorAll('details').forEach((d) => (d.open = true)));
        const small = await page.evaluate(() => {
          const out: string[] = [];
          const walk = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
          for (let n = walk.nextNode(); n; n = walk.nextNode()) {
            const el = n.parentElement!;
            if (!n.textContent!.trim() || el.closest('.sr, .skip, [hidden]')) continue;
            const r = el.getBoundingClientRect();
            if (r.width === 0 || r.height === 0 || getComputedStyle(el).visibility === 'hidden') continue;
            const fs = parseFloat(getComputedStyle(el).fontSize);
            if (fs < 12) out.push(`${el.className || el.tagName} ${fs}px "${n.textContent!.trim().slice(0, 20)}"`);
          }
          return out;
        });
        expect(small, p.key).toEqual([]);
      }
    });
  }

  for (const w of [320, 390, 1440]) {
    test(`nav and footer links are at least 24px tall @${w} (44px on phones)`, async ({ page }) => {
      await page.setViewportSize({ width: w, height: 900 });
      await page.goto('');
      await ready(page);
      const min = w <= 640 ? 44 : 24;
      for (const a of await page.locator('header.top a:not(.skip), footer.foot a').all()) {
        const b = (await a.boundingBox())!;
        expect(b.height, await a.textContent() ?? '').toBeGreaterThanOrEqual(min);
      }
    });
  }
});

test.describe('without JavaScript', () => {
  test.use({ javaScriptEnabled: false });
  for (const p of PAGES) {
    test(`${p.key}: no dead buttons; extra sources reachable`, async ({ page }) => {
      await page.goto(p.slug);
      await expect(page.locator('button')).toHaveCount(0);
      for (const det of await page.locator('details.more-src').all()) {
        const sum = det.locator('> summary');
        const n = await det.locator('.chips-more details.chip').count();
        expect(n).toBeGreaterThan(0);
        await expect(det.locator('.chips-more details.chip').first()).toBeHidden();
        await sum.click();
        await expect(det.locator('.chips-more details.chip:visible')).toHaveCount(n);
      }
    });
  }
  test('verification: each of the 37 tiles links to its segment row', async ({ page }) => {
    await page.goto('verification/');
    const tiles = page.locator('.tiles-int a.tile');
    await expect(tiles).toHaveCount(37);
    for (const t of await tiles.all()) {
      const href = (await t.getAttribute('href'))!;
      const id = href.split('#')[1]!;
      await expect(page.locator(`[id="${id}"]`)).toHaveCount(1);
    }
    await tiles.nth(21).click();
    await expect(page).toHaveURL(/#seg-21$/);
    await expect(page.locator('#seg-21')).toBeInViewport();
  });
  test('404 page works without JavaScript', async ({ page }) => {
    const res = await page.goto('no-such-page/');
    expect(res!.status()).toBe(404);
    await expect(page.locator('h1')).toHaveText(label('notfound'));
  });
});

const FAILURES: [string, (p: Page) => Promise<unknown>][] = [
  ['404', (p) => p.route('**/data/site-data.json', (r) => r.fulfill({ status: 404, body: 'not found' }))],
  ['corrupt', (p) => p.route('**/data/site-data.json', (r) => r.fulfill({ status: 200, contentType: 'application/json', body: '{"schema": "ovl-site-data.v2", "revi' }))],
  ['wrong shape', (p) => p.route('**/data/site-data.json', (r) => r.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify([{ result: 'MATCH' }]) }))],
  ['abort', (p) => p.route('**/data/site-data.json', (r) => r.abort())],
];
test.describe('data failure', () => {
  for (const [name, setup] of FAILURES) {
    for (const p of [PAGES[0], PAGES[2]]) {
      test(`${name} on ${p.key}: scoped fallback, zero checked tiles, prerendered content stays`, async ({ page }) => {
        await setup(page);
        await page.goto(p.slug);
        await expect(page.locator('#root[data-failed]')).toHaveCount(1);
        const alerts = page.getByRole('alert');
        expect(await alerts.count()).toBeGreaterThan(0);
        for (const a of await alerts.all()) await expect(a).toHaveText(label('error.section'));
        await expect(page.locator('.tile')).toHaveCount(0);
        await expect(page.locator('[data-result]')).toHaveCount(0);
        await expect(page.locator('h1')).toHaveText(label(`h1.${p.key}`));
        expect(await page.locator('main [data-claim]').count()).toBeGreaterThan(0);
        await expect(page.getByRole('navigation')).toBeVisible();
      });
    }
  }
});

test('404 page: real status, navigation and a link home', async ({ page }) => {
  const res = await page.goto('definitely/not/here');
  expect(res!.status()).toBe(404);
  await ready(page);
  await expect(page.locator('h1')).toHaveText(label('notfound'));
  await expect(page.getByRole('navigation').getByRole('link')).toHaveCount(6);
  const home = page.locator('[data-notfound] a');
  await expect(home).toHaveAttribute('href', '/');
  await home.click();
  await expect(page.locator('h1')).toHaveText(label('h1.overview'));
});

test.describe('dark mode, everything expanded', () => {
  for (const p of PAGES) {
    test(`${p.key}: axe clean`, async ({ page }) => {
      await page.emulateMedia({ colorScheme: 'dark' });
      await page.goto(p.slug);
      await ready(page);
      await page.evaluate(() => document.querySelectorAll('details').forEach((d) => (d.open = true)));
      const tile = page.locator('.tiles-int button').first();
      if (await tile.count()) await tile.click();
      const axe = await new AxeBuilder({ page }).withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa', 'wcag22aa']).analyze();
      expect(axe.violations.map((v) => `${v.id}: ${v.nodes.length} (${v.nodes[0]?.target.join(' ')})`)).toEqual([]);
    });
  }
});

test('reduced motion: opening a source shows no animation', async ({ page }) => {
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await page.goto('verification/');
  await ready(page);
  await page.locator('[data-claim="VE-04"] details.chip > summary').first().click();
  await page.locator('[data-claim="VE-04"] summary.more').click();
  const running = await page.evaluate(() => document.getAnimations().length);
  expect(running).toBe(0);
});

test('"+N more": aria-expanded follows the disclosure and N equals the revealed chips', async ({ page }) => {
  for (const p of PAGES) {
    await page.goto(p.slug);
    await ready(page);
    for (const det of await page.locator('details.more-src').all()) {
      const sum = det.locator('summary.more');
      await expect(sum).toHaveAttribute('aria-expanded', 'false');
      const n = Number(await sum.locator('.more-closed [data-path]').textContent());
      await sum.click();
      await expect(sum).toHaveAttribute('aria-expanded', 'true');
      await expect(det.locator('.chips-more details.chip:visible')).toHaveCount(n);
      await sum.click();
      await expect(sum).toHaveAttribute('aria-expanded', 'false');
    }
  }
});

test('claim deep links land on the claim; a bad hash is harmless', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(String(e)));
  await page.goto('verification/#ve-04');
  await ready(page);
  await expect(page.locator('#ve-04')).toBeInViewport();
  await page.goto('check/#ck-05');
  await ready(page);
  await expect(page.locator('#ck-05')).toBeInViewport();
  await page.goto('verification/#does-not-exist');
  await ready(page);
  await expect(page.locator('h1')).toHaveText(label('h1.verification'));
  await page.goto('verification/#%3Cimg%20src=x%3E');
  await ready(page);
  await expect(page.locator('h1')).toBeVisible();
  expect(errors).toEqual([]);
});

test('external links: rel=noopener noreferrer and pinned revisions', async ({ page }) => {
  for (const p of PAGES) {
    await page.goto(p.slug);
    await ready(page);
    const links = await page.locator('a[href^="http"]').evaluateAll((as) => as.map((a) => ({ href: a.getAttribute('href')!, rel: a.getAttribute('rel') ?? '' })));
    expect(links.length).toBeGreaterThan(0);
    for (const l of links) {
      expect(l.rel.split(' '), l.href).toEqual(expect.arrayContaining(['noopener', 'noreferrer']));
      if (/github\.com\/AOSSIE-Org\/OpenVerifiableLLM\/(blob|tree|raw)\//.test(l.href)) expect(l.href, l.href).toContain(PINNED_GH);
      if (/huggingface\.co\/.*\/(blob|resolve|tree)\//.test(l.href)) expect(l.href, l.href).toContain(PINNED_HF);
    }
  }
});

test('#ck-05 limit note stays within the reading measure', async ({ page }) => {
  await page.setViewportSize({ width: 1920, height: 900 });
  await page.goto('check/');
  await ready(page);
  const lim = page.locator('#ck-05 .limit');
  if (await lim.count()) {
    const { w, ch } = await lim.evaluate((el) => {
      const probe = document.createElement('span');
      probe.textContent = '0';
      el.appendChild(probe);
      const ch = probe.getBoundingClientRect().width;
      probe.remove();
      return { w: el.getBoundingClientRect().width, ch };
    });
    expect(w / ch).toBeLessThanOrEqual(70);
  }
});

test('no banned phrase anywhere on any page, quoted source lines included (no roadmap exception)', async ({ page }) => {
  const { ledger } = await import('../../src/content/ledger');
  for (const p of [...PAGES, { slug: 'no-such-page/', key: 'notfound' }]) {
    await page.goto(p.slug);
    await ready(page);
    await page.evaluate(() => document.querySelectorAll('details').forEach((d) => (d.open = true)));
    const text = (await page.locator('body').textContent())!.toLowerCase().replaceAll('unlearning-3i-atlas', '').replaceAll('us$1–1.50', '');
    const hits = ledger.forbidden_phrases.map((b) => b.phrase.toLowerCase()).filter((b) => text.includes(b));
    expect(hits, p.key).toEqual([]);
    expect(/\btraining steps?\b|\bsteps\b/.test(text), p.key).toBe(false);
  }
});

for (const w of [320, 390, 768, 1024, 1440]) {
  test(`overview strip @${w}: every tile shows its data-driven glyph, even rows, no horizontal scroll`, async ({ page }) => {
    const { data } = await import('./util');
    const tiles = data().verifications.tiles;
    await page.setViewportSize({ width: w, height: 900 });
    await page.goto('');
    await ready(page);
    const glyphs = page.locator('.tiles-strip .tile .glyph');
    await expect(glyphs).toHaveCount(tiles.length);
    const want = (r: string) => label(`tile.${r === 'MATCH' || r === 'MISMATCH' ? r : 'pending'}`);
    for (let k = 0; k < tiles.length; k++) {
      const g = glyphs.nth(k);
      await expect(g).toBeVisible();
      await expect(g).toHaveText(want(tiles[k]!.result));
      const fs = await g.evaluate((el) => parseFloat(getComputedStyle(el).fontSize));
      expect(fs).toBeGreaterThanOrEqual(12);
      const box = (await g.boundingBox())!;
      const tile = (await page.locator('.tiles-strip .tile').nth(k).boundingBox())!;
      expect(box.x).toBeGreaterThanOrEqual(tile.x - 0.5);
      expect(box.x + box.width).toBeLessThanOrEqual(tile.x + tile.width + 0.5);
    }
    const rows = await page.locator('.tiles-strip li').evaluateAll((li) => {
      const m = new Map<number, number>();
      for (const el of li) { const y = Math.round(el.getBoundingClientRect().top); m.set(y, (m.get(y) ?? 0) + 1); }
      return [...m.values()];
    });
    for (const r of rows.slice(0, -1)) expect(r).toBe(rows[0]);
    expect(rows[rows.length - 1]!).toBeLessThanOrEqual(rows[0]!);
    if (w >= 1100) expect(rows).toEqual([37]);
    const { sw, cw } = await page.evaluate(() => ({ sw: document.documentElement.scrollWidth, cw: document.documentElement.clientWidth }));
    expect(sw).toBeLessThanOrEqual(cw);
  });
}

test.describe('404 page weight and tap target', () => {
  for (const w of [320, 390, 1440]) {
    test(`@${w}: no JS, no site-data.json, home link >= ${w <= 640 ? 44 : 24}px`, async ({ page }) => {
      const reqs: string[] = [];
      page.on('request', (r) => reqs.push(r.url()));
      await page.setViewportSize({ width: w, height: 900 });
      const res = await page.goto('no-such-page/');
      expect(res!.status()).toBe(404);
      await page.waitForLoadState('networkidle');
      expect(reqs.filter((u) => u.endsWith('.js') || u.includes('site-data.json'))).toEqual([]);
      await expect(page.locator('script')).toHaveCount(0);
      const b = (await page.locator('[data-notfound] a').boundingBox())!;
      expect(b.height).toBeGreaterThanOrEqual(w <= 640 ? 44 : 24);
    });
  }
});

test.describe('targeted segment row, JS off', () => {
  test.use({ javaScriptEnabled: false });
  for (const scheme of ['light', 'dark'] as const) {
    test(`${scheme}: the row a tile link lands on is highlighted with the stamp tokens`, async ({ page }) => {
      await page.emulateMedia({ colorScheme: scheme });
      await page.goto('verification/');
      await page.locator('.tiles-int a.tile').nth(21).click();
      await expect(page).toHaveURL(/#seg-21$/);
      const style = (sel: string) => page.locator(sel).evaluate((el) => ({ bg: getComputedStyle(el).backgroundColor, outline: getComputedStyle(el).outlineStyle }));
      const tokens = await page.evaluate(() => {
        const probe = document.createElement('div');
        document.body.appendChild(probe);
        probe.style.background = 'var(--stamp-tint)';
        const bg = getComputedStyle(probe).backgroundColor;
        probe.remove();
        return bg;
      });
      const hit = await style('#seg-21');
      expect(hit.bg).toBe(tokens);
      expect(hit.outline).toBe('solid');
      const other = await style('#seg-20');
      expect(other.bg).not.toBe(tokens);
    });
  }
});

for (const w of [320, 390, 768, 1440]) {
  test(`"+N more" summary is >= ${w <= 640 ? 44 : 24}px tall @${w} and causes no overflow`, async ({ page }) => {
    await page.setViewportSize({ width: w, height: 900 });
    for (const slug of ['', 'verification/']) {
      await page.goto(slug);
      await ready(page);
      for (const s of await page.locator('summary.more').all()) {
        const b = (await s.boundingBox())!;
        expect(b.height).toBeGreaterThanOrEqual(w <= 640 ? 44 : 24);
        expect(b.x + b.width).toBeLessThanOrEqual(w);
      }
      const { sw, cw } = await page.evaluate(() => ({ sw: document.documentElement.scrollWidth, cw: document.documentElement.clientWidth }));
      expect(sw).toBeLessThanOrEqual(cw);
    }
  });
}
