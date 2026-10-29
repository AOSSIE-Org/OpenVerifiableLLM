import { useState } from 'react';
import { L } from './text';
import { useHydrated } from './hydrated';

/** Full hash is always in the DOM (wraps on phones); copy puts the exact digest on the clipboard. */
export function Hash({ value, path }: { value: string; path?: string }) {
  const hydrated = useHydrated();
  const [state, setState] = useState<'idle' | 'copied' | 'failed'>('idle');
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(value);
      setState('copied');
    } catch {
      setState('failed');
    }
  };
  return (
    <span className="hash">
      <code {...(path ? { 'data-path': path } : { 'data-verbatim': 'hash' })}>{value}</code>
      {hydrated ? (
        <button type="button" className="btn btn-small" onClick={() => void copy()}>
          <L k="copy" />
        </button>
      ) : null}
      <span role="status" aria-live="polite" className="cmd-status">
        {state === 'copied' ? <L k="copied" /> : state === 'failed' ? <L k="copy.failed" /> : null}
      </span>
    </span>
  );
}
