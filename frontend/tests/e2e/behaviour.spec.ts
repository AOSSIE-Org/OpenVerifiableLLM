import { expect, test } from '@playwright/test';
import { label } from '../../src/content/ledger';
import { data, ready } from './util';

test('keyboard: skip link first, then into main; tiles are focusable buttons operated with Enter and Space', async ({ page }) => {
  await page.goto('verification/');
  await ready(page);
  await page.keyboard.press('Tab');
  await expect(page.locator('.skip')).toBeFocused();
  await page.keyboard.press('Enter');
  await expect(page).toHaveURL(/#main$/);
  const tile = page.locator('.tiles-int button').nth(10);
  await tile.focus();
  const ring = await tile.evaluate((el) => {
    const s = getComputedStyle(el);
    return { outline: s.outlineStyle, width: s.outlineWidth, shadow: s.boxShadow };
  });
  expect(ring.outline).toBe('solid');
  expect(ring.shadow).not.toBe('none'); // two-layer ring: gold outline + ink halo
  await page.keyboard.press('Enter');
  await expect(tile).toHaveAttribute('aria-pressed', 'true');
  await expect(page.locator('.tile-detail .checks li')).toHaveCount(data().verifications.rows.filter((r) => r.segment === 10).length);
  await page.keyboard.press('Space');
  await expect(tile).toHaveAttribute('aria-pressed', 'false');
  await page.keyboard.press('Tab');
  await expect(page.locator('.tiles-int button').nth(11)).toBeFocused();
});

test('navigation: client-side links move focus to main and update aria-current; back works', async ({ page }) => {
  await page.goto('');
  await ready(page);
  await page.getByRole('navigation').getByRole('link', { name: label('nav.timeline') }).click();
  await expect(page.locator('h1')).toHaveText(label('h1.timeline'));
  await expect(page.locator('main')).toBeFocused();
  await expect(page.locator('nav a[aria-current="page"]')).toHaveText(label('nav.timeline'));
  await page.goBack();
  await expect(page.locator('h1')).toHaveText(label('h1.overview'));
});

test('overview: 37-tile scoreboard and "independent: 0" under the title', async ({ page }) => {
  const d = data();
  await page.goto('');
  await ready(page);
  await expect(page.locator('.tile')).toHaveCount(d.verifications.tiles.length);
  await expect(page.locator('.tile[data-checker="project"]')).toHaveCount(37);
  await expect(page.locator('[data-independent] [data-path="verifications.independentChecks"]')).toHaveText('0');
  const order = await page.evaluate(() => {
    const h1 = document.querySelector('h1')!, ind = document.querySelector('[data-independent]')!;
    return h1.compareDocumentPosition(ind) & Node.DOCUMENT_POSITION_FOLLOWING;
  });
  expect(order).toBeTruthy();
  await expect(page.locator('[data-claim]')).toHaveCount(6);
});

test('overview at 1440: title, independent count and the whole scoreboard are in the first screen', async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto('');
  await ready(page);
  for (const sel of ['h1', '[data-independent]', '.tiles-strip li:last-child .tile']) {
    const box = (await page.locator(sel).boundingBox())!;
    expect(box.y + box.height, sel).toBeLessThanOrEqual(900);
  }
});

test('extra sources sit behind a keyboard-operable "+N more sources" toggle', async ({ page }) => {
  await page.goto('verification/');
  await ready(page);
  const ev = page.locator('[data-claim="VE-04"]');
  const btn = ev.locator('button.more');
  await expect(ev.locator('details.chip:visible')).toHaveCount(1);
  await expect(btn).toHaveAttribute('aria-expanded', 'false');
  await btn.focus();
  await page.keyboard.press('Enter');
  await expect(btn).toHaveAttribute('aria-expanded', 'true');
  await expect(ev.locator('details.chip:visible')).toHaveCount(2);
  await page.keyboard.press('Space');
  await expect(btn).toHaveAttribute('aria-expanded', 'false');
});

test('verification: odds 95.09% at 110 and 99.02% at 169; fresh machine only segment 21; 46 rows', async ({ page }) => {
  await page.goto('verification/');
  await ready(page);
  await expect(page.locator('[data-path="odds.m95"]').first()).toHaveText('110');
  await expect(page.locator('[data-path="odds.p95"]')).toHaveText('95.09%');
  await expect(page.locator('[data-path="odds.m99"]').first()).toHaveText('169');
  await expect(page.locator('[data-path="odds.p99"]')).toHaveText('99.02%');
  await expect(page.locator('.groups').first().locator('tbody tr')).toHaveCount(46);
  const freshRows = page.locator('section.group').nth(2).locator('tbody [data-path$=".segment"]');
  await expect(freshRows).toHaveText(['21']);
});

test('verification at 390: hardware map is one card per machine, table hidden', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 900 });
  await page.goto('verification/');
  await ready(page);
  await expect(page.locator('.hw-cards > li')).toHaveCount(data().hardware.length);
  await expect(page.locator('.hw-table')).toBeHidden();
  await expect(page.locator('.hw-cards > li').first()).toBeVisible();
});

test('timeline: restart and code-change markings come from the data', async ({ page }) => {
  const d = data();
  await page.goto('timeline/');
  await ready(page);
  for (const e of d.events) await expect(page.locator(`.strip li`).nth(e.segment)).toHaveAttribute('data-event', e.id);
  await expect(page.locator('.strip li[data-event]')).toHaveCount(d.events.length);
});

test('check: the consistency commands are verbatim from README main@1049d2a7 and copy exactly', async ({ page, context }) => {
  await context.grantPermissions(['clipboard-read', 'clipboard-write']);
  const d = data();
  await page.goto('check/');
  await ready(page);
  const block = page.locator('[data-claim="CK-02"] .cmd pre');
  expect(await block.evaluate((e) => e.textContent)).toBe(d.check.commands);
  await page.locator('[data-claim="CK-02"] .cmd button').click();
  await expect(page.locator('[data-claim="CK-02"] [role="status"]')).toHaveText(label('copied'));
  expect(await page.evaluate(() => navigator.clipboard.readText())).toBe(d.check.commands);
});

test('source chip opens the exact cited row', async ({ page }) => {
  const d = data();
  await page.goto('verification/');
  await ready(page);
  const chip = page.locator('[data-claim="VE-04"] details.chip').first();
  await chip.locator('summary').click();
  await expect(chip.locator('pre')).toHaveText(d.claimSources['VE-04']![0]!.text!);
  await expect(chip.locator('a')).toHaveAttribute('href', d.claimSources['VE-04']![0]!.url!);
});

test('data that fails to load shows the error page and never a MATCH', async ({ page }) => {
  await page.route('**/data/site-data.json', (r) => r.fulfill({ status: 500, body: 'x' }));
  await page.goto('verification/');
  await expect(page.locator('[data-fatal] h1')).toHaveText(label('error.title'));
  await expect(page.locator('.result-MATCH, [data-result="MATCH"]')).toHaveCount(0);
});

test('data with an unpinned revision is rejected', async ({ page }) => {
  const d = data();
  d.revision = 'main';
  await page.route('**/data/site-data.json', (r) => r.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(d) }));
  await page.goto('');
  await expect(page.locator('[data-fatal]')).toBeVisible();
  await expect(page.locator('[data-result="MATCH"]')).toHaveCount(0);
});

test('one malformed section shows "couldn\'t load this evidence" and the rest of the page stays', async ({ page }) => {
  const d = data() as unknown as Record<string, unknown>;
  d.hardware = [{ result: 'ok' }];
  await page.route('**/data/site-data.json', (r) => r.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(d) }));
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(String(e)));
  await page.goto('verification/');
  await ready(page);
  await expect(page.locator('[data-fallback]')).toHaveCount(1);
  await expect(page.locator('[data-fallback]')).toHaveText(label('error.section'));
  await expect(page.locator('.tiles-int button')).toHaveCount(37);
  await expect(page.locator('.hw-table')).toHaveCount(0);
  expect(errors).toEqual([]);
});
