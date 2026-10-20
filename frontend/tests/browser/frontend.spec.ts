import { test, expect } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';
import { mkdir, readFile } from 'node:fs/promises';
const fixture = JSON.parse(await readFile('public/data/snapshot.json', 'utf8'));
const views = [
  ['overview', '/', 'A model with a traceable beginning.'],
  ['evidence', '/evidence', 'Inspect the evidence.'],
  ['detail', '/evidence/synthetic-pilot-replay', 'Sampled replay report'],
  ['releases', '/releases', 'A release is more than a download.'],
  ['verification', '/verification', 'Choose the check. Know its limits.'],
  ['inference', '/inference', 'Generation, when it can be reproduced.'],
];
test('evidence search, filters, direct detail, missing parent, supersession and back navigation', async ({ page }) => {
  await page.goto('#/evidence');
  await page.getByLabel('Scope', { exact: true }).selectOption('pilot');
  await page.getByLabel('Result', { exact: true }).selectOption('PASS');
  await page.getByRole('searchbox').fill('replay');
  await expect(page).toHaveURL(/scope=pilot.*result=PASS.*q=replay/);
  await page.reload();
  await expect(page.getByRole('searchbox')).toHaveValue('replay');
  await page.getByRole('link', { name: 'Sampled replay report', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Sampled replay report', level: 1 })).toBeVisible();
  await page.getByRole('link', { name: /synthetic-parent-missing/ }).click();
  await expect(page.getByRole('heading', { name: 'Missing evidence reference' })).toBeVisible();
  await page.goBack();
  await page.getByRole('link', { name: /Earlier sampled replay report/ }).click();
  await expect(page.getByText('FAIL', { exact: true })).toBeVisible();
  await page.getByRole('link', { name: 'Sampled replay report', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Sampled replay report', level: 1 })).toBeVisible();
});
test('full digest copy with feedback', async ({ page, context }) => {
  await context.grantPermissions(['clipboard-read', 'clipboard-write']);
  await page.goto('#/evidence/synthetic-pilot-replay');
  await page.getByRole('button', { name: 'Copy full digest' }).click();
  await expect(page.getByText('Copied to clipboard.')).toBeVisible();
  expect(await page.evaluate(() => navigator.clipboard.readText())).toBe('0123456789abcdef'.repeat(4));
});
test('unreleased models and public generation remain unavailable despite query flags', async ({ page }) => {
  await page.goto('#/releases');
  await expect(page.getByText('Not released yet.', { exact: true })).toHaveCount(2);
  const downloads = page.getByRole('button', { name: 'Download unavailable' });
  await expect(downloads).toHaveCount(2);
  for (const b of await downloads.all()) await expect(b).toBeDisabled();
  await page.goto('?mock=true&fixture=future-release#/inference?mock=true');
  await expect(page.getByRole('heading', { name: 'Generation is unavailable.', exact: true })).toBeVisible();
  await expect(page.getByRole('textbox')).toHaveCount(0);
  await expect(page.getByText('Exercise the interface.')).toHaveCount(0);
});
test('snapshot errors never fall back to mock success, and retry recovers supplied data', async ({ page }) => {
  let bad = true;
  await page.route('**/data/snapshot.json', route => route.fulfill({ contentType: 'application/json', body: JSON.stringify(bad ? { ...fixture, checks: null } : fixture) }));
  await page.goto('#/evidence');
  await expect(page.getByRole('heading', { name: 'Snapshot unavailable or invalid' })).toBeVisible();
  await expect(page.locator('.result-PASS')).toHaveCount(0);
  bad = false; await page.getByRole('button', { name: 'Retry loading snapshot' }).click();
  await expect(page.getByRole('heading', { name: 'Inspect the evidence.' })).toBeVisible();
});
test('loading and empty checks states do not announce verification success', async ({ page }) => {
  let release: (() => void) | undefined;
  const gate = new Promise<void>(r => { release = r; });
  await page.route('**/data/snapshot.json', async route => { await gate; await route.fulfill({ contentType: 'application/json', body: JSON.stringify({ ...fixture, checks: [] }) }); });
  await page.goto('#/', { waitUntil: 'domcontentloaded' });
  await expect(page.getByText('Loading display snapshot…')).toBeVisible();
  release!();
  await expect(page.getByRole('heading', { name: 'No checks reported' })).toBeVisible();
  await expect(page.locator('.result-PASS')).toHaveCount(0);
});
test('HTTP failure, unsupported schema and unapproved public data remain errors', async ({ page }) => {
  for (const payload of [{ status: 503, body: '{}' }, { status: 200, body: JSON.stringify({ ...fixture, schemaVersion: 900 }) }, { status: 200, body: JSON.stringify({ ...fixture, mode: 'public-snapshot' }) }]) {
    await page.route('**/data/snapshot.json', route => route.fulfill({ ...payload, contentType: 'application/json' }));
    await page.goto('#/'); await page.reload();
    await expect(page.getByRole('heading', { name: 'Snapshot unavailable or invalid' })).toBeVisible();
    await expect(page.locator('.result-PASS')).toHaveCount(0);
    await page.unroute('**/data/snapshot.json');
  }
});
for (const viewport of [{ width: 1440, height: 1000 }, { width: 390, height: 844 }, { width: 720, height: 900 }]) {
  for (const [name, route, title] of views) {
    test(name + ' at ' + viewport.width + 'px: no overflow, accessible structure and local fonts', async ({ page }) => {
      await page.setViewportSize(viewport); const errors: string[] = [];
      page.on('pageerror', e => errors.push(e.message));
      await page.goto('#' + route);
      await expect(page.getByRole('heading', { name: title, level: 1 })).toBeVisible();
      await page.evaluate(() => document.fonts.ready);
      expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
      const a11y = await new AxeBuilder({ page }).withTags(['wcag2a', 'wcag2aa', 'wcag21aa', 'wcag22aa']).disableRules(['color-contrast']).analyze();
      expect(a11y.violations.map(v => ({ id: v.id, impact: v.impact, nodes: v.nodes.map(n => n.target) }))).toEqual([]);
      expect(errors).toEqual([]);
      if (viewport.width !== 720) {
        await mkdir('screenshots', { recursive: true });
        await page.screenshot({ path: 'screenshots/' + name + '-' + viewport.width + '.png', fullPage: true });
      }
    });
  }
}
test('mobile menu, skip link, focus and evidence navigation work from keyboard', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 }); await page.goto('#/');
  await expect(page.getByRole('heading', { level: 1 })).toBeVisible();
  await page.keyboard.press('Tab'); await expect(page.getByRole('link', { name: 'Skip to content' })).toBeFocused();
  await page.keyboard.press('Enter'); await expect(page.locator('main')).toBeFocused();
  await page.getByRole('button', { name: 'Menu' }).focus(); await page.keyboard.press('Enter');
  await expect(page.getByRole('button', { name: 'Menu' })).toHaveAttribute('aria-expanded', 'true');
  await page.getByRole('navigation', { name: 'Primary' }).getByRole('link', { name: 'Evidence', exact: true }).focus();
  await page.keyboard.press('Enter');
  await expect(page.getByRole('heading', { name: 'Inspect the evidence.' })).toBeVisible();
  await page.getByRole('link', { name: 'Sampled replay report', exact: true }).focus(); await page.keyboard.press('Enter');
  await expect(page.getByRole('heading', { level: 1 })).toHaveText('Sampled replay report');
});
test('200 percent browser zoom equivalent (720 CSS px) remains usable with reduced motion', async ({ page }) => {
  // Native browser zoom halves the CSS viewport at a fixed physical width.
  // CSS zoom alone does not update media queries, so use the equivalent layout viewport.
  await page.setViewportSize({ width: 720, height: 500 }); await page.emulateMedia({ reducedMotion: 'reduce' });
  for (const [, route, title] of views) {
    await page.goto('#' + route); await expect(page.getByRole('heading', { level: 1, name: title })).toBeVisible();
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
    expect(await page.evaluate(() => matchMedia('(prefers-reduced-motion: reduce)').matches)).toBe(true);
  }
});
test('isolated development mock supports errors, identity mismatch, success and cancellation', async ({ page }) => {
  await page.goto('http://127.0.0.1:5173/#/inference');
  await expect(page.getByText('Exercise the interface.')).toBeVisible();
  await page.getByLabel('Completion prompt').fill('Public example');
  await page.getByLabel('Development scenario').selectOption('identity-mismatch');
  await page.getByRole('button', { name: 'Generate scripted example' }).click();
  await expect(page.getByText('Returned release identity mismatch.')).toBeVisible();
  await page.getByLabel('Development scenario').selectOption('error');
  await page.getByRole('button', { name: 'Retry example' }).click();
  await expect(page.getByText('Synthetic backend error. No model was called.')).toBeVisible();
  await page.getByLabel('Development scenario').selectOption('success');
  await page.getByRole('button', { name: 'Generate scripted example' }).click();
  await expect(page.getByRole('button', { name: 'Copy response' })).toBeVisible();
  await expect(page.getByText('Example UI response — not generated by a released OpenVerifiableLLM model.')).toHaveCount(2);
  await page.getByRole('button', { name: 'Generate scripted example' }).click();
  await page.getByRole('button', { name: 'Cancel request' }).click();
  await expect(page.getByText(/Request cancelled. A future server/)).toBeVisible();
});
