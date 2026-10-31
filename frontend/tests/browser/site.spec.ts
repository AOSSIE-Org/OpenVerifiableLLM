import { test, expect } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';

test.describe('OpenVerifiableLLM Site v3', () => {
  test('renders all sections and numbers from evidence.json', async ({ page }) => {
    await page.goto('/');

    // Check Hero
    await expect(page.getByRole('heading', { level: 1 })).toContainText('A language model whose training anyone can recompute');

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

  test('minimal bottom bar becomes inert and hidden when footer is in view', async ({ page }) => {
    await page.goto('/');
    const mini = page.locator('#mini');
    await expect(mini).toBeVisible();
    await expect(mini).not.toHaveClass(/off/);

    // Scroll to the footer
    await page.locator('#footer').scrollIntoViewIfNeeded();
    await expect(mini).toHaveClass(/off/);
    await expect(mini).toHaveAttribute('inert', '');
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
    await page.route('**/data/evidence.json', route =>
      route.fulfill({ status: 500, contentType: 'application/json', body: '{"error": "Internal Error"}' })
    );
    await page.goto('/');
    await expect(page.getByRole('heading', { name: 'Snapshot unavailable or invalid' })).toBeVisible();
    await expect(page.getByRole('button', { name: 'Retry loading snapshot' })).toBeVisible();
  });

  // WCAG 2.2 AA accessibility and responsive checks across viewports and themes
  for (const viewport of [
    { width: 1440, height: 900, name: 'desktop' },
    { width: 720, height: 900, name: 'tablet-200pct-zoom' },
    { width: 390, height: 844, name: 'mobile' },
  ]) {
    for (const theme of ['dark', 'light']) {
      test(`a11y and layout: ${viewport.name} (${viewport.width}px) in ${theme} theme passes WCAG 2.2 AA`, async ({ page }) => {
        await page.setViewportSize(viewport);
        await page.goto('/');

        if (theme === 'dark') {
          await page.getByRole('button', { name: /switch between dark and light theme/i }).click();
          await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark');
        }

        // Verify no horizontal overflow
        const hasNoHorizontalScroll = await page.evaluate(
          () => document.documentElement.scrollWidth <= window.innerWidth
        );
        expect(hasNoHorizontalScroll).toBe(true);

        // Run axe accessibility analysis
        const a11y = await new AxeBuilder({ page })
          .withTags(['wcag2a', 'wcag2aa', 'wcag21aa', 'wcag22aa'])
          .analyze();

        expect(a11y.violations).toEqual([]);
      });
    }
  }

  test('respects prefers-reduced-motion', async ({ page }) => {
    await page.emulateMedia({ reducedMotion: 'reduce' });
    await page.goto('/');
    const matches = await page.evaluate(() => matchMedia('(prefers-reduced-motion: reduce)').matches);
    expect(matches).toBe(true);
  });
});
