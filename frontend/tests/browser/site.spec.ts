import { test, expect } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';

test.describe('OpenVerifiableLLM evidence page', () => {
  test('renders every section with numbers from evidence.json', async ({ page }) => {
    await page.goto('/');
    await expect(page.getByRole('heading', { level: 1 })).toHaveText('OpenVerifiableLLM');

    for (const name of [
      'Overview',
      'Check it yourself',
      'Verification',
      'Training timeline',
      'Verifiable exclusion',
      'Downloads',
      'References',
    ]) {
      await expect(page.getByRole('heading', { level: 2, name: new RegExp(name) })).toBeVisible();
    }

    const main = page.locator('main');
    await expect(main.getByText('7,232,582', { exact: true })).toBeVisible();
    await expect(main.getByText('7,118,121,139', { exact: true })).toBeVisible();
    await expect(main.getByText('3,481', { exact: true })).toBeVisible();

    await expect(page.getByText('Checked by the project (37/37)')).toBeVisible();
    await expect(page.getByText('Checked independently (0/37)')).toBeVisible();
    await expect(page.locator('#verification').getByText(/self-reported and unsigned/)).toBeVisible();

    for (const v of ['3.3604', '3.3860', '2.8710', '2.8712']) {
      await expect(page.locator('#exclusion').getByText(v, { exact: true })).toBeVisible();
    }

    await expect(page.getByRole('link', { name: 'ovl-23m-chat.gguf' })).toBeVisible();
    await expect(
      page.getByText('541d49c97b87cd4ed0acb90b4327535ee20b8c11c452e2583d460154e9d980de'),
    ).toBeVisible();
  });

  test('citations jump to the reference list', async ({ page }) => {
    await page.goto('/');
    await page.locator('#verification sup.ref a').first().click();
    await expect(page).toHaveURL(/#ref-verifications$/);
    await expect(page.locator('#ref-verifications')).toContainText('release/simple-run/verifications.jsonl');
  });

  test('copy button copies the command', async ({ page, context }) => {
    await context.grantPermissions(['clipboard-read', 'clipboard-write']);
    await page.goto('/');
    await page.getByRole('button', { name: 'Copy', exact: true }).first().click();
    await expect(page.getByRole('button', { name: 'Copied' })).toBeVisible();
    const text = await page.evaluate(() => navigator.clipboard.readText());
    expect(text).toBe('ollama run ryoari/openverifiablellm');
  });

  test('skip link moves focus to the article', async ({ page }) => {
    await page.goto('/');
    await page.waitForSelector('h1');
    await page.keyboard.press('Tab');
    await expect(page.getByRole('link', { name: 'Skip to content' })).toBeFocused();
  });

  test('shows an error and no numbers when the snapshot fails', async ({ page }) => {
    await page.route('**/data/evidence.json', route =>
      route.fulfill({ status: 500, contentType: 'application/json', body: '{}' }),
    );
    await page.goto('/');
    await expect(page.getByRole('alert')).toBeVisible();
    await expect(page.getByRole('heading', { name: 'Snapshot unavailable or invalid' })).toBeVisible();
    await expect(page.getByText('7,232,582')).toHaveCount(0);
  });

  test('nothing moves on scroll', async ({ page }) => {
    await page.goto('/');
    await page.waitForSelector('#references');
    const animations = await page.evaluate(async () => {
      window.scrollTo(0, document.body.scrollHeight / 2);
      await new Promise(r => setTimeout(r, 200));
      return document.getAnimations().length;
    });
    expect(animations).toBe(0);
  });

  const viewports = [
    { name: 'desktop', width: 1440, height: 900 },
    { name: 'tablet-200pct-zoom', width: 720, height: 900 },
    { name: 'mobile', width: 390, height: 844 },
  ];

  for (const vp of viewports) {
    for (const scheme of ['light', 'dark'] as const) {
      test(`a11y and layout: ${vp.name} (${vp.width}px), ${scheme}`, async ({ page }) => {
        await page.emulateMedia({ colorScheme: scheme });
        await page.setViewportSize({ width: vp.width, height: vp.height });
        await page.goto('/');
        await page.waitForSelector('#references');
        // open the collapsed tables so they are checked too
        await page.evaluate(() => document.querySelectorAll('details').forEach(d => (d.open = true)));

        const results = await new AxeBuilder({ page })
          .withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa', 'wcag22aa'])
          .analyze();
        expect(results.violations).toEqual([]);

        const noHorizontalScroll = await page.evaluate(
          () => document.documentElement.scrollWidth <= window.innerWidth,
        );
        expect(noHorizontalScroll).toBe(true);
      });
    }
  }
});
