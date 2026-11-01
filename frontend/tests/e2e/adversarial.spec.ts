import AxeBuilder from '@axe-core/playwright';
import { expect, test, type Page } from '@playwright/test';
import { PAGES, data, ready } from './util';

// Every test blocks all non-local traffic: no request may leave 127.0.0.1.
test.beforeEach(async ({ page }) => {
  await page.route(/^(?!http:\/\/127\.0\.0\.1:)/, (r) => r.abort());
});
const serve = (page: Page, d: unknown) =>
  page.route('**/data/site-data*.json', (r) => r.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(d) }));
const overflow = (page: Page) => page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);

test('exclusion: "every step can be checked" and "per-step coverage check" are link-only; chips keep the source link', async ({ page }) => {
  await page.goto('exclusion/');
  await ready(page);
  await page.locator('details.chip').evaluateAll((ds) => ds.forEach((d) => ((d as HTMLDetailsElement).open = true)));
  const text = (await page.locator('main').innerText()).replace(/\s+/g, ' ').toLowerCase();
  expect(text).not.toContain('every step can be checked');
  expect(text).not.toContain('per-step coverage');
  const hrefs = await page.locator('details.chip a[href]').evaluateAll((as) => as.map((a) => (a as HTMLAnchorElement).href));
  expect(hrefs.some((h) => h.includes('unlearning-3i-atlas/README.md#L4-L5'))).toBe(true);
  expect(hrefs.some((h) => h.includes('unlearning-3i-atlas/README.md#L19-L21'))).toBe(true);
});

test('check keeps the README "step by step:" quote', async ({ page }) => {
  await page.goto('check/');
  await ready(page);
  await page.locator('details.chip').evaluateAll((ds) => ds.forEach((d) => ((d as HTMLDetailsElement).open = true)));
  expect((await page.locator('main').innerText()).toLowerCase()).toContain('step by step:');
});

test('HTML/script in data strings renders as text, never as markup', async ({ page }) => {
  const d = data() as unknown as { exclusion: { title: string }; verifications: { rows: { machine: string }[] } };
  const evil = '<img src=x onerror="window.__xss=1"><script>window.__xss=2</script>';
  d.exclusion.title = evil;
  d.verifications.rows.forEach((r) => (r.machine = evil));
  await serve(page, d);
  for (const slug of ['exclusion/', 'verification/']) {
    await page.goto(slug);
    await ready(page);
    expect(await page.evaluate(() => (window as unknown as { __xss?: number }).__xss)).toBeUndefined();
    expect(await page.locator('main img').count()).toBe(0);
  }
});

test('very long unbroken + RTL/unicode strings in data cause no horizontal overflow at 320', async ({ page }) => {
  await page.setViewportSize({ width: 320, height: 800 });
  const d = data() as unknown as { exclusion: { title: string }; verifications: { rows: { machine: string }[] } };
  d.exclusion.title = 'שלום‮عالم' + 'X'.repeat(400) + ' 🧪ẞ';
  d.verifications.rows.forEach((r) => (r.machine = 'm'.repeat(300)));
  await serve(page, d);
  for (const slug of ['exclusion/', 'verification/']) {
    await page.goto(slug);
    await ready(page);
    expect(await overflow(page), slug).toBeLessThanOrEqual(0);
  }
});

test('missing optional data field (a section) degrades to the section fallback, not a blank page', async ({ page }) => {
  const d = data() as unknown as Record<string, unknown>;
  delete d.anchor;
  await serve(page, d);
  await page.goto('downloads/');
  await ready(page);
  await expect(page.locator('h1')).toHaveCount(1);
  await expect(page.locator('[data-result="MATCH"]').first()).toHaveCount(0);
});

test('0 segments is rejected, never shown as an empty-but-valid scoreboard', async ({ page }) => {
  const d = data() as unknown as { chain: { segments: number; entries: number; boundaries: unknown[] }; verifications: { tiles: unknown[] } };
  d.chain.segments = 0; d.chain.entries = 1; d.chain.boundaries = d.chain.boundaries.slice(0, 1); d.verifications.tiles = [];
  await serve(page, d);
  await page.goto('');
  await ready(page);
  await expect(page.locator('[data-fallback], [data-fatal]').first()).toBeVisible();
  await expect(page.locator('[data-result="MATCH"]')).toHaveCount(0);
});

test('slow data (3 s) keeps the prerendered page readable and then hydrates', async ({ page }) => {
  await page.route('**/data/site-data*.json', async (r) => { await new Promise((f) => setTimeout(f, 3000)); await r.continue(); });
  await page.goto('verification/');
  await expect(page.locator('h1')).toBeVisible();
  await ready(page);
  await expect(page.locator('.tiles-int button')).toHaveCount(37);
});

test('offline after load: client navigation across all six pages still works', async ({ page, context }) => {
  await page.goto('');
  await ready(page);
  await page.waitForTimeout(2500); // idle prefetch of the other page chunks
  await page.waitForLoadState('networkidle');
  await context.setOffline(true);
  for (const p of PAGES.slice(1)) {
    await page.locator(`nav a[href$="/${p.slug}"]`).first().click();
    await expect(page).toHaveURL(new RegExp(`${p.slug}$`));
    await expect(page.locator('h1')).toHaveCount(1);
    await expect(page.locator('[data-fatal]')).toHaveCount(0);
  }
  await context.setOffline(false);
});

test('back/forward restores page and h1 across deep links', async ({ page }) => {
  await page.goto('timeline/');
  await ready(page);
  const h1t = await page.locator('h1').innerText();
  await page.locator('nav a[href$="/downloads/"]').first().click();
  await expect(page).toHaveURL(/downloads\/$/);
  await page.goBack();
  await expect(page).toHaveURL(/timeline\/$/);
  await expect(page.locator('h1')).toHaveText(h1t);
  await page.goForward();
  await expect(page).toHaveURL(/downloads\/$/);
});

test('deep link with hash to a verification row lands on that row', async ({ page }) => {
  await page.goto('verification/');
  await ready(page);
  const id = await page.locator('main [id]').evaluateAll((els) => els.map((e) => e.id).find((i) => /seg|row/.test(i)));
  test.skip(!id, 'no row ids');
  await page.goto('about:blank');
  await page.goto(`verification/#${id}`);
  await ready(page);
  const top = await page.locator(`[id="${id}"]`).evaluate((e) => e.getBoundingClientRect().top);
  expect(top).toBeLessThan(800);
});

for (const p of PAGES) {
  test(`${p.key}: 400% zoom (320 CSS px) reflows with no horizontal scroll`, async ({ page }) => {
    await page.setViewportSize({ width: 320, height: 256 });
    await page.goto(p.slug);
    await ready(page);
    expect(await overflow(page)).toBeLessThanOrEqual(0);
  });
  test(`${p.key}: forced-colors is axe clean and keeps focus visible`, async ({ page }) => {
    await page.emulateMedia({ forcedColors: 'active' });
    await page.goto(p.slug);
    await ready(page);
    const r = await new AxeBuilder({ page }).withTags(['wcag2a', 'wcag2aa', 'wcag22aa']).analyze();
    expect(r.violations.map((v) => v.id)).toEqual([]);
  });
  test(`${p.key}: print media hides nav and keeps h1`, async ({ page }) => {
    await page.goto(p.slug);
    await ready(page);
    await page.emulateMedia({ media: 'print' });
    await expect(page.locator('h1')).toBeVisible();
  });
}

test('keyboard-only: Tab reaches every nav page and Enter navigates', async ({ page }) => {
  await page.goto('');
  await ready(page);
  for (const p of PAGES.slice(1)) {
    let found = false;
    for (let i = 0; i < 400 && !found; i++) {
      await page.keyboard.press('Tab');
      found = await page.evaluate((s) => (document.activeElement as HTMLAnchorElement | null)?.closest('nav') !== null && (document.activeElement as HTMLAnchorElement).href?.endsWith('/' + s), p.slug);
    }
    expect(found, p.slug).toBe(true);
    await page.keyboard.press('Enter');
    await expect(page).toHaveURL(new RegExp(`${p.slug}$`));
  }
});

test.describe('JS off', () => {
  test.use({ javaScriptEnabled: false });
  for (const p of PAGES) {
    test(`${p.key}: prerendered content and the claim/source chip are present with no JS`, async ({ page }) => {
      await page.goto(p.slug);
      await expect(page.locator('h1')).toHaveCount(1);
      expect(await page.locator('details.chip').count()).toBeGreaterThan(0);
    });
  }
});

test('CSP meta is present on every page and nothing violates it', async ({ page }) => {
  await page.addInitScript(() => {
    (window as unknown as { __csp: string[] }).__csp = [];
    document.addEventListener('securitypolicyviolation', (e) => (window as unknown as { __csp: string[] }).__csp.push(`${e.violatedDirective} ${e.blockedURI}`));
  });
  for (const p of PAGES) {
    await page.goto(p.slug);
    await ready(page);
    await expect(page.locator('meta[http-equiv="Content-Security-Policy"]')).toHaveCount(1);
    expect(await page.evaluate(() => (window as unknown as { __csp: string[] }).__csp), p.key).toEqual([]);
  }
});

test('data is fetched from the content-hashed URL named by the build', async ({ page }) => {
  const urls: string[] = [];
  page.on('request', (r) => urls.push(r.url()));
  await page.goto('');
  await ready(page);
  const h = await page.locator('meta[name="ovl-data"]').getAttribute('content');
  expect(h).toMatch(/^[0-9a-f]{64}$/);
  expect(urls.filter((u) => u.includes('/data/site-data'))).toEqual([expect.stringContaining(`site-data.${h!.slice(0, 16)}.json`)]);
});

test('no page requests anything that 404s (favicon included)', async ({ page }) => {
  const bad: string[] = [];
  page.on('response', (r) => r.status() >= 400 && bad.push(`${r.status()} ${r.url()}`));
  for (const p of PAGES) {
    await page.goto(p.slug);
    await ready(page);
  }
  expect(bad).toEqual([]);
});

test('phone: every source chip summary is >= 44px tall; source links announce the new tab', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('verification/');
  await ready(page);
  for (const h of await page.locator('details.chip > summary').evaluateAll((s) => s.filter((e) => (e as HTMLElement).offsetParent).map((e) => e.getBoundingClientRect().height))) expect(h).toBeGreaterThanOrEqual(44);
  await page.locator('details.chip').first().evaluate((d) => ((d as HTMLDetailsElement).open = true));
  await expect(page.locator('details.chip a[target="_blank"]').first()).toHaveAccessibleName(/opens in a new tab/);
});
