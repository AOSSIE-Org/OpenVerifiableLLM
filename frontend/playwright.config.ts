import { defineConfig } from '@playwright/test';

// Runs against the static production build (npm run build), served with no SPA fallback.
export default defineConfig({
  testDir: './tests/e2e',
  fullyParallel: true,
  workers: process.env.CI ? 2 : 4,
  timeout: 60_000,
  reporter: [['list']],
  use: { baseURL: 'http://127.0.0.1:4173/', browserName: 'chromium', trace: 'retain-on-failure' },
  webServer: { command: 'node scripts/static-server.mjs', url: 'http://127.0.0.1:4173/', reuseExistingServer: !process.env.CI },
});
