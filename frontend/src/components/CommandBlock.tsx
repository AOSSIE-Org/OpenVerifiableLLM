import { useState } from 'react';
import { L } from './text';

export function CommandBlock({ text, name }: { text: string; name: string }) {
  const [state, setState] = useState<'idle' | 'copied' | 'failed'>('idle');
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(text);
      setState('copied');
    } catch {
      setState('failed');
    }
  };
  return (
    <div className="cmd">
      <pre tabIndex={0} aria-label={name} data-verbatim="command"><code>{text}</code></pre>
      <button type="button" className="btn" onClick={() => void copy()}>
        <L k="copy" />
      </button>
      <span role="status" aria-live="polite" className="cmd-status">
        {state === 'copied' ? <L k="copied" /> : state === 'failed' ? <L k="copy.failed" /> : null}
      </span>
    </div>
  );
}
