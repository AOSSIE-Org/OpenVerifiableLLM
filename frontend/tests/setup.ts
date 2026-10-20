import '@testing-library/jest-dom/vitest';
import { cleanup } from '@testing-library/react';
import { afterEach } from 'vitest';
afterEach(cleanup);

// jsdom lacks these browser APIs, which framer-motion's scroll tracking uses.
class NoopObserver { observe() {} unobserve() {} disconnect() {} takeRecords() { return []; } }
for (const name of ['ResizeObserver', 'IntersectionObserver']) {
  if (!(name in globalThis)) Object.defineProperty(globalThis, name, { value: NoopObserver, configurable: true });
}
