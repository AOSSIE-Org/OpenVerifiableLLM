import { test, expect } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';
import { loadEvidence, withMismatch, withoutChecks } from '../evidence-cases';

const PINNED_EVIDENCE = 'https://huggingface.co/datasets/AOSSIE/openverifiable-enwiki-20260901-20260918-r1-evidence/tree/96dc282a';
const THEME_BUTTON = /switch to (dark|light) theme/i;
const WIDTHS = [1440, 1024, 768, 390, 320];

test.describe('OpenVerifiableLLM Site', () => {
  test('renders all sections and numbers from evidence.json', async ({ page }) => {
    await page.goto('/');

    // Article title, byline with separated dates, and lead
    await expect(page.getByText(/Release v1\.1\.0 · 31 October 2026 · Record timestamped 20 October 2026/)).toBeVisible();
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
    await expect(page.getByText('Project: 37 / 37 matched')).toBeVisible();
    await expect(page.getByText('Independent: 0 / 37 checked')).toBeVisible();
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

    const themeBtn = page.getByRole('button', { name: THEME_BUTTON });
    await expect(themeBtn).toHaveAccessibleName('Switch to dark theme');
    await themeBtn.click();
    await expect(html).toHaveAttribute('data-theme', 'dark');
    await expect(themeBtn).toHaveAccessibleName('Switch to light theme');

    await themeBtn.click();
    await expect(html).not.toHaveAttribute('data-theme', 'dark');
  });

  test('a stored dark theme applies before the evidence loads, and on the error page', async ({ page }) => {
    await page.addInitScript(() => localStorage.setItem('ovl-theme', 'dark'));
    await page.route('**/data/evidence.json', route => route.fulfill({ status: 500, body: '{}' }));
    await page.goto('/');
    await expect(page.getByRole('alert')).toBeVisible();
    await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark');
  });

  test('every evidence link points at the pinned revision', async ({ page }) => {
    await page.goto('/');
    await page.getByRole('heading', { level: 1 }).waitFor();
    await expect(page.getByRole('link', { name: 'Evidence', exact: true })).toHaveAttribute('href', PINNED_EVIDENCE);
    await expect(page.getByRole('link', { name: 'Evidence on Hugging Face' })).toHaveAttribute('href', PINNED_EVIDENCE);
    const revisionLinks = page.getByRole('link', { name: '96dc282a', exact: true });
    await expect(revisionLinks).toHaveCount(2);
    for (const link of await revisionLinks.all()) {
      await expect(link).toHaveAttribute('href', PINNED_EVIDENCE);
    }
    const unpinned = await page.evaluate(() =>
      [...document.querySelectorAll('a[href*="huggingface.co/datasets"]')]
        .map(a => a.getAttribute('href') ?? '')
        .filter(href => !href.includes('96dc282a')),
    );
    expect(unpinned).toEqual([]);
    await expect(page.locator('#check pre', { hasText: 'hf download' })).toContainText('--revision 96dc282a9803616c6b30bb31bfc3a0b91a7191e6');
  });

  test('a mismatch shows on the page and removes the all-matched claims', async ({ page }) => {
    await page.route('**/data/evidence.json', route => route.fulfill({ json: withMismatch(5) }));
    await page.setViewportSize({ width: 1440, height: 900 });
    await page.goto('/');
    await expect(page.locator('.figures').first()).toContainText('36 / 37');
    await expect(page.getByText(/Project: 36 \/ 37 matched \(1 mismatched\)/)).toBeVisible();
    await expect(page.locator('.covgrid .cells').first().locator('.cell.mismatch')).toHaveCount(1);
    await expect(page.locator('.tl-chart rect[fill="var(--miss)"]')).toHaveCount(1);
    await expect(page.locator('#overview .notice')).not.toContainText('all matching');
    await expect(page.locator('#overview .notice')).not.toContainText('Every segment matched');
    await expect(page.locator('#timeline figcaption')).not.toContainText('all matched');
    const violations = (await new AxeBuilder({ page }).withTags(['wcag2a', 'wcag2aa', 'wcag21aa', 'wcag22aa']).analyze()).violations;
    expect(violations).toEqual([]);
  });

  test('a missing check shows as not checked, never as a pass', async ({ page }) => {
    await page.route('**/data/evidence.json', route => route.fulfill({ json: withoutChecks(25) }));
    await page.setViewportSize({ width: 1440, height: 900 });
    await page.goto('/');
    await expect(page.getByText(/Project: 36 \/ 37 matched \(1 not checked\)/)).toBeVisible();
    await expect(page.locator('.covgrid .cells').first().locator('.cell.none')).toHaveCount(1);
    await expect(page.locator('.covgrid .cells').first().locator('.cell.mismatch')).toHaveCount(0);
    await expect(page.locator('.milestones')).toContainText('NOT CHECKED');
    await expect(page.locator('#timeline .sub')).not.toContainText('still matched');
  });

  test('independent checks are shown as unavailable, in their own row', async ({ page }) => {
    await page.goto('/');
    const rows = page.locator('.covgrid .cells');
    await expect(rows.nth(1).locator('.cell.none')).toHaveCount(37);
    await expect(rows.nth(1).locator('.cell.match')).toHaveCount(0);
    await expect(rows.nth(0).locator('.cell.match')).toHaveCount(37);
    await expect(page.locator('#overview .sub')).toContainText('by the project; 0 independent checks so far');
  });

  const corruptions: { name: string; change: (e: ReturnType<typeof loadEvidence>) => void; message: RegExp }[] = [
    { name: 'a broken chain link', change: e => { e.chain[10].previous = '0'.repeat(64); }, message: /Broken previous link at boundary 10/ },
    { name: 'a hidden mismatch', change: e => { e.verifications.rawChecks[5].result = 'MISMATCH'; }, message: /Coverage disagrees with raw checks/ },
    { name: 'a tampered anchor', change: e => { e.anchor.chain_head_sha256 = 'a'.repeat(64); }, message: /Anchor chain head/ },
    { name: 'a changed time with an unchanged hash', change: e => { e.chain[3].time += 1; }, message: /time at boundary 3 does not match its record/ },
    { name: 'shell characters in the dataset', change: e => { e.dataset = 'owner/name; rm -rf /'; }, message: /Invalid dataset identifier/ },
    { name: 'shell characters in the release tag', change: e => { e.release.tag = 'v1.1.0`whoami`'; }, message: /Invalid release tag/ },
    { name: 'a fork point that does not fit', change: e => { e.exclusion.fork.fromEntry = 29; }, message: /Fork entry 31 does not follow from entry 29/ },
    { name: 'a missing boundary time', change: e => { delete (e.chain[3] as Partial<(typeof e.chain)[number]>).time; }, message: /Invalid time at boundary 3/ },
  ];
  for (const { name, change, message } of corruptions) {
    test(`refuses to show evidence with ${name}`, async ({ page }) => {
      const bad = loadEvidence();
      change(bad);
      await page.route('**/data/evidence.json', route => route.fulfill({ json: bad }));
      await page.goto('/');
      await expect(page.getByRole('alert')).toContainText(message);
      await expect(page.getByRole('heading', { level: 1 })).toHaveText('Snapshot unavailable or invalid');
      await expect(page.locator('.figures')).toHaveCount(0);
    });
  }

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
    await expect(page.locator('#main')).toBeFocused();
  });

  test('every control can be reached by keyboard and shows a focus ring', async ({ page }) => {
    await page.setViewportSize({ width: 1440, height: 900 });
    await page.goto('/');
    await page.getByRole('heading', { level: 1 }).waitFor();
    const seen: string[] = [];
    for (let i = 0; i < 70; i++) {
      await page.keyboard.press('Tab');
      const focused = await page.evaluate(() => {
        const el = document.activeElement;
        if (!el || el === document.body) return null;
        const style = getComputedStyle(el);
        const box = el.getBoundingClientRect();
        const toc = document.querySelector('.toc')?.getBoundingClientRect();
        return {
          name: el.getAttribute('aria-label') || el.textContent?.trim().slice(0, 30) || el.tagName,
          ring: style.outlineStyle !== 'none' && parseFloat(style.outlineWidth) >= 2,
          hidden: !el.closest('.toc') && !!toc && box.top >= toc.top && box.bottom <= toc.bottom,
        };
      });
      if (!focused) break;
      expect(focused.ring, `${focused.name} has no focus ring`).toBe(true);
      expect(focused.hidden, `${focused.name} is covered by the sticky bar`).toBe(false);
      seen.push(focused.name);
    }
    expect(seen).toContain('Skip to content');
    expect(seen).toContain('Switch to dark theme');
    expect(seen).toContain('Explore the results');
  });

  test('the phone contents bar keeps the current section in view', async ({ page }) => {
    await page.setViewportSize({ width: 320, height: 700 });
    await page.goto('/');
    await page.locator('#faq').evaluate(el => el.scrollIntoView({ block: 'start' }));
    const link = page.locator('.toc a[aria-current]');
    await expect(link).toHaveText('FAQ');
    const inView = await link.evaluate(a => {
      const box = a.getBoundingClientRect();
      return box.left >= 0 && box.right <= window.innerWidth + 1;
    });
    expect(inView).toBe(true);
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
    await expect(page.getByRole('heading', { level: 1, name: 'Snapshot unavailable or invalid' })).toBeVisible();
    await expect(page.getByRole('button', { name: 'Retry loading snapshot' })).toBeVisible();
  });

  const viewports = [
    { name: 'desktop', width: 1440, height: 900 },
    { name: 'laptop', width: 1024, height: 768 },
    { name: 'tablet', width: 768, height: 1024 },
    { name: 'tablet-200pct-zoom', width: 720, height: 900 },
    { name: 'mobile', width: 390, height: 844 },
    { name: 'narrow-400pct-zoom', width: 320, height: 640 },
  ];

  for (const vp of viewports) {
    for (const theme of ['dark', 'light'] as const) {
      test(`a11y and layout: ${vp.name} (${vp.width}px) in ${theme} theme passes WCAG 2.2 AA`, async ({ page }) => {
        await page.setViewportSize({ width: vp.width, height: vp.height });
        await page.goto('/');

        if (theme === 'dark') {
          const themeBtn = page.getByRole('button', { name: THEME_BUTTON });
          await themeBtn.click();
          await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark');
        }

        const axeResults = await new AxeBuilder({ page })
          .withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa', 'wcag22aa'])
          .analyze();

        expect(axeResults.violations).toEqual([]);

        // Open every folded section: what's inside has to pass too
        await page.evaluate(() => document.querySelectorAll('details').forEach(d => (d.open = true)));
        const openResults = await new AxeBuilder({ page })
          .withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa', 'wcag22aa'])
          .analyze();
        expect(openResults.violations).toEqual([]);

        // Ensure no horizontal overflow
        const hasNoHorizontalScroll = await page.evaluate(
          () => document.documentElement.scrollWidth <= window.innerWidth
        );
        expect(hasNoHorizontalScroll, `Horizontal scroll detected on ${vp.name} in ${theme} theme`).toBe(true);
      });
    }
  }

  for (const width of WIDTHS) {
    test(`evidence figures stay aligned at ${width}px`, async ({ page }) => {
      await page.setViewportSize({ width, height: 900 });
      await page.goto('/');
      await page.getByRole('heading', { level: 1 }).waitFor();

      const figures = await page.evaluate(() =>
        [...document.querySelectorAll('.figures dd, .loss dd')].map(dd => {
          const box = dd.getBoundingClientRect();
          const text = document.createRange();
          text.selectNodeContents(dd);
          return {
            value: dd.textContent,
            top: Math.round(box.top),
            lines: new Set([...text.getClientRects()].map(rect => Math.round(rect.top))).size,
            numerals: getComputedStyle(dd).fontVariantNumeric,
            spill: text.getBoundingClientRect().right - dd.parentElement!.getBoundingClientRect().right,
            font: getComputedStyle(dd).fontFamily,
          };
        })
      );

      expect(figures.length).toBeGreaterThan(0);
      for (const f of figures) {
        expect(f.font, `${f.value} must not use the old-style serif numerals`).not.toContain('Georgia');
        expect(f.spill, `${f.value} must fit its column`).toBeLessThanOrEqual(0);
        expect(f.lines, `${f.value} must stay on one line`).toBe(1);
        expect(f.numerals).toContain('tabular-nums');
        expect(f.numerals).toContain('lining-nums');
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
    const moving = await page.evaluate(() =>
      [...document.querySelectorAll('body *')].filter(el => {
        const style = getComputedStyle(el);
        return style.animationName !== 'none' || parseFloat(style.transitionDuration) > 0;
      }).length,
    );
    expect(moving).toBe(0);
  });

  test('headings step down one level at a time', async ({ page }) => {
    await page.goto('/');
    await page.getByRole('heading', { level: 1 }).waitFor();
    const levels = await page.evaluate(() =>
      [...document.querySelectorAll('h1, h2, h3, h4, h5, h6')].map(h => Number(h.tagName[1])),
    );
    expect(levels[0]).toBe(1);
    expect(levels.filter(level => level === 1)).toHaveLength(1);
    levels.forEach((level, i) => {
      if (i > 0) expect(level - levels[i - 1]).toBeLessThanOrEqual(1);
    });
  });

  for (const width of WIDTHS) {
    test(`nothing is clipped or overlapping at ${width}px`, async ({ page }) => {
      await page.setViewportSize({ width, height: 900 });
      await page.goto('/');
      await page.getByRole('heading', { level: 1 }).waitFor();
      await page.evaluate(() => document.querySelectorAll('details').forEach(d => (d.open = true)));

      const problems = await page.evaluate(() => {
        const found: string[] = [];
        const viewport = document.documentElement.clientWidth;
        for (const el of document.querySelectorAll<HTMLElement>('main *, header *, footer *')) {
          if (el.closest('.sr-only, svg, pre, .toc ol') || el.classList.contains('sr-only')) continue;
          const box = el.getBoundingClientRect();
          if (box.width === 0 || box.height === 0) continue;
          if (box.right > viewport + 1 || box.left < -1) {
            found.push(`${el.tagName}.${el.className} leaves the page: ${Math.round(box.left)}-${Math.round(box.right)}`);
          }
        }
        return found;
      });
      expect(problems).toEqual([]);

      // The project name is one long word and must not break
      const title = page.getByRole('heading', { level: 1 });
      const titleLines = await title.evaluate(h => {
        const range = document.createRange();
        range.selectNodeContents(h);
        return new Set([...range.getClientRects()].map(rect => Math.round(rect.top))).size;
      });
      expect(titleLines).toBe(1);

      // The one-line command breaks at its space, not inside the name
      const command = page.locator('#check .step > .code pre');
      const commandLines = await command.evaluate(pre => {
        const range = document.createRange();
        range.selectNodeContents(pre);
        const rows = new Set([...range.getClientRects()].map(rect => Math.round(rect.top))).size;
        const width = pre.clientWidth - parseFloat(getComputedStyle(pre).paddingLeft) - parseFloat(getComputedStyle(pre).paddingRight);
        return { rows, fits: range.getBoundingClientRect().width <= width + 1 };
      });
      expect(commandLines.rows).toBeLessThanOrEqual(2);
      expect(commandLines.fits).toBe(true);
    });
  }
});
