import { defineConfig } from '@playwright/test';
export default defineConfig({
  testDir: './tests/browser', fullyParallel: false, workers: 1,
  timeout: 60000,
  use: { baseURL: 'http://127.0.0.1:4173/OpenVerifiableLLM/', browserName: 'chromium', trace: 'retain-on-failure' },
  reporter: [['list'], ['html', { open: 'never' }]],
  webServer: [
    { command: 'node scripts/static-server.mjs', url: 'http://127.0.0.1:4173/OpenVerifiableLLM/', reuseExistingServer: !process.env.CI },
    { command: 'npm run dev -- --port 5173 --strictPort', env: { VITE_MOCK_INFERENCE: 'true' }, url: 'http://127.0.0.1:5173', reuseExistingServer: !process.env.CI },
  ],
});
