import { test, expect } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';

test.describe('OpenVerifiableLLM Site', () => {
  test('renders all sections and numbers from evidence.json', async ({ page }) => {
    await page.goto('/');

    // Article title, byline with separated dates, and lead
    await expect(page.getByText(/Release v1\.0\.1 · 28 October 2026 · Record timestamped 20 October 2026/)).toBeVisible();
    await expect(page.getByRole('heading', { level: 1 })).toHaveText('OpenVerifiableLLM');
    await expect(page.getByText(/is a language model whose training anyone can recompute/)).toBeVisible();
    await expect(page.getByRole('complementary', { name: 'Key facts' }).first()).toBeVisible();

    // Check Overview text clarifying checker attribution
    await expect(
      page.locator('#overview').getByText('Recomputed bit for bit on a second RTX 5090 by the project; 0 independent checks so far.')
    ).toBeVisible();

    // Check Overview numbers
    await expect(page.getByText('7,232,582', { exact: true })).toBeVisible();
    await expect(page.getByText('7,118,121,139', { exact: true })).toBeVisible();
    await expect(page.getByText('3,481', { exact: true })).toBeVisible();

    // Check Verification: 37 of 37 matched
    await expect(page.getByText('37 / 37 matched')).toBeVisible();
    await expect(page.getByText('■ Project: 37')).toBeVisible();
    await expect(page.getByText('▨ Independent: 0')).toBeVisible();
    await expect(page.getByText(/46 checks across 37 segments/)).toBeVisible();

    // Check Verifiable Exclusion losses
    await expect(page.getByText('3.3604')).toBeVisible();
    await expect(page.getByText('3.3860')).toBeVisible();
    await expect(page.getByText('2.8710')).toBeVisible();
    await expect(page.getByText('2.8712')).toBeVisible();

    // Check Downloads
    await expect(page.getByText('ovl-23m-chat.gguf')).toBeVisible();
    await page.locator('#downloads summary').first().click();
    await expect(page.getByText('541d49c97b87cd4ed0acb90b4327535ee20b8c11c452e2583d460154e9d980de')).toBeVisible();
  });

  test('copy button provides visual feedback and clipboard content', async ({ page, context }) => {
    await context.grantPermissions(['clipboard-read', 'clipboard-write']);
    await page.goto('/');

    const copyBtn = page.getByRole('button', { name: 'Copy', exact: true }).first();
    await expect(copyBtn).toBeVisible();
    await copyBtn.click();
    await expect(page.getByRole('button', { name: 'Copied ✓' })).toBeVisible();
    const clipboardText = await page.evaluate(() => navigator.clipboard.readText());
    expect(clipboardText.length).toBeGreaterThan(10);
  });

  test('theme toggle switches between light and dark themes', async ({ page }) => {
    await page.goto('/');
    const html = page.locator('html');
    await expect(html).not.toHaveAttribute('data-theme', 'dark');

    const themeBtn = page.getByRole('button', { name: /switch between dark and light theme/i });
    await themeBtn.click();
    await expect(html).toHaveAttribute('data-theme', 'dark');

    await themeBtn.click();
    await expect(html).not.toHaveAttribute('data-theme', 'dark');
  });

  test('contents links jump to every section', async ({ page }) => {
    await page.setViewportSize({ width: 1440, height: 900 });
    await page.goto('/');
    const toc = page.getByRole('navigation', { name: 'Contents' });
    const links = toc.getByRole('link');
    await expect(links).toHaveCount(6);
    for (const href of await links.evaluateAll(els => els.map(e => e.getAttribute('href')))) {
      await expect(page.locator(href as string)).toHaveCount(1);
    }
  });

  test('keyboard-only navigation: skip link and tab navigation', async ({ page }) => {
    await page.setViewportSize({ width: 1440, height: 900 });
    await page.goto('/');
    await page.waitForSelector('.skip');
    await page.locator('body').focus();

    await page.keyboard.press('Tab');
    const skipLink = page.getByRole('link', { name: 'Skip to content' });
    await expect(skipLink).toBeFocused();

    await page.keyboard.press('Enter');
    await expect(page.locator('#main')).toBeVisible();
  });

  test('handles snapshot load errors gracefully', async ({ page }) => {
    await page.route('**/data/evidence.json', (route) => {
      route.fulfill({
        status: 500,
        contentType: 'application/json',
        body: JSON.stringify({ error: 'Internal Server Error' }),
      });
    });

    await page.goto('/');
    await expect(page.getByRole('alert')).toBeVisible();
    await expect(page.getByRole('heading', { name: 'Snapshot unavailable or invalid' })).toBeVisible();
    await expect(page.getByRole('button', { name: 'Retry loading snapshot' })).toBeVisible();
  });

  const viewports = [
    { name: 'desktop', width: 1440, height: 900 },
    { name: 'tablet-200pct-zoom', width: 720, height: 900 },
    { name: 'mobile', width: 390, height: 844 },
  ];

  for (const vp of viewports) {
    for (const theme of ['dark', 'light'] as const) {
      test(`a11y and layout: ${vp.name} (${vp.width}px) in ${theme} theme passes WCAG 2.2 AA`, async ({ page }) => {
        await page.setViewportSize({ width: vp.width, height: vp.height });
        await page.goto('/');

        if (theme === 'dark') {
          const themeBtn = page.getByRole('button', { name: /switch between dark and light theme/i });
          await themeBtn.click();
          await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark');
        }

        const axeResults = await new AxeBuilder({ page })
          .withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa', 'wcag22aa'])
          .analyze();

        expect(axeResults.violations).toEqual([]);

        // Ensure no horizontal overflow
        const hasNoHorizontalScroll = await page.evaluate(
          () => document.documentElement.scrollWidth <= window.innerWidth
        );
        expect(hasNoHorizontalScroll, `Horizontal scroll detected on ${vp.name} in ${theme} theme`).toBe(true);
      });
    }
  }

  for (const width of [1440, 768, 390, 320]) {
    test(`evidence figures stay aligned at ${width}px`, async ({ page }) => {
      await page.setViewportSize({ width, height: 900 });
      await page.goto('/');
      await page.getByRole('heading', { level: 1 }).waitFor();

      const figures = await page.evaluate(() =>
        [...document.querySelectorAll('.figures dd')].map(dd => {
          const box = dd.getBoundingClientRect();
          const text = document.createRange();
          text.selectNodeContents(dd);
          return {
            block: [...document.querySelectorAll('.figures')].indexOf(dd.closest('.figures')!),
            value: dd.textContent,
            top: Math.round(box.top),
            lines: text.getClientRects().length,
            spill: text.getBoundingClientRect().right - dd.parentElement!.getBoundingClientRect().right,
            font: getComputedStyle(dd).fontFamily,
          };
        })
      );

      expect(figures.length).toBeGreaterThan(0);
      for (const f of figures) {
        expect(f.font, `${f.value} must not use the old-style serif numerals`).not.toContain('Georgia');
        expect(f.spill, `${f.value} must fit its column`).toBeLessThanOrEqual(0);
      }
      // Values on one visual row share one top edge, however long their labels are.
      const rows = new Map<string, Set<number>>();
      const tops = await page.evaluate(() =>
        [...document.querySelectorAll('.figures > div')].map(cell => ({
          block: [...document.querySelectorAll('.figures')].indexOf(cell.parentElement!),
          row: Math.round(cell.getBoundingClientRect().top),
          top: Math.round(cell.querySelector('dd')!.getBoundingClientRect().top),
        }))
      );
      for (const t of tops) {
        const key = `${t.block}:${t.row}`;
        rows.set(key, (rows.get(key) ?? new Set<number>()).add(t.top));
      }
      for (const [row, seen] of rows) {
        expect(seen.size, `values in row ${row} must share a top edge`).toBe(1);
      }

      const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
      expect(overflow).toBeLessThanOrEqual(0);
    });
  }

  test('respects prefers-reduced-motion', async ({ page }) => {
    await page.emulateMedia({ reducedMotion: 'reduce' });
    await page.goto('/');

    const title = page.getByRole('heading', { level: 1 });
    await expect(title).toBeVisible();
  });
});
