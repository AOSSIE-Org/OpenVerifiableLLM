import { useSyncExternalStore } from 'react';

const noop = () => () => {};
/** False in the prerendered HTML and during hydration, true once React runs in the browser.
 *  Controls that need JavaScript render only when this is true, so the page never shows dead buttons. */
export function useHydrated(): boolean {
  return useSyncExternalStore(noop, () => true, () => false);
}
