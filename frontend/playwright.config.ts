import { defineConfig } from '@playwright/test';

// Own port per checkout (E2E_PORT) and never reuse a running server: on a shared machine a
// reused server can be another checkout's dist, which makes the gate test the wrong build.
const port = Number(process.env.E2E_PORT ?? 4173);
const origin = `http://127.0.0.1:${port}/`;

// Runs against the static production build (npm run build), served with no SPA fallback.
export default defineConfig({
  testDir: './tests/e2e',
  fullyParallel: true,
  workers: process.env.CI ? 2 : 4,
  timeout: 60_000,
  reporter: [['list']],
  use: { baseURL: origin, browserName: 'chromium', trace: 'retain-on-failure' },
  webServer: { command: 'node scripts/static-server.mjs', url: origin, env: { PORT: String(port) }, reuseExistingServer: false },
});
