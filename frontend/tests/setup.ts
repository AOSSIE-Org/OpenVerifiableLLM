import '@testing-library/jest-dom/vitest';
import { cleanup } from '@testing-library/react';
import { afterEach } from 'vitest';

afterEach(cleanup);

// Mock browser observers if missing in jsdom
class NoopObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
  takeRecords() {
    return [];
  }
}

for (const name of ['ResizeObserver', 'IntersectionObserver']) {
  if (!(name in globalThis)) {
    Object.defineProperty(globalThis, name, {
      value: NoopObserver,
      configurable: true,
    });
  }
}

// Mock localStorage in jsdom environment
if (typeof window !== 'undefined') {
  let store: Record<string, string> = {};
  const mockLocalStorage = {
    getItem: (key: string) => store[key] ?? null,
    setItem: (key: string, value: string) => {
      store[key] = String(value);
    },
    removeItem: (key: string) => {
      delete store[key];
    },
    clear: () => {
      store = {};
    },
  };
  Object.defineProperty(window, 'localStorage', {
    value: mockLocalStorage,
    writable: true,
  });
}
