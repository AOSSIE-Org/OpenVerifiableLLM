import { readFileSync } from 'node:fs';
import path from 'node:path';
import type { Page } from '@playwright/test';
import type { SiteData } from '../../src/data/types';

export const PAGES = [
  { slug: '', key: 'overview' },
  { slug: 'check/', key: 'check' },
  { slug: 'verification/', key: 'verification' },
  { slug: 'timeline/', key: 'timeline' },
  { slug: 'exclusion/', key: 'exclusion' },
  { slug: 'downloads/', key: 'downloads' },
] as const;
export const WIDTHS = [1440, 390] as const;
export const data = (): SiteData => JSON.parse(readFileSync(path.resolve('public/data/site-data.json'), 'utf8')) as SiteData;

/** Wait for the client to load and validate the data (hydration done). */
export async function ready(page: Page) {
  await page.waitForFunction(() => document.querySelector('[data-fatal]') || document.querySelector('footer code [data-path="revShort"]'));
  await page.waitForLoadState('networkidle');
}
