import React, { useEffect, useRef, useState } from 'react';

interface CopyButtonProps {
  textToCopy: string;
  label?: string;
  className?: string;
}

type Status = 'idle' | 'copied' | 'failed';

export const CopyButton: React.FC<CopyButtonProps> = ({
  textToCopy,
  label = 'Copy',
  className = 'copy-btn',
}) => {
  const [status, setStatus] = useState<Status>('idle');
  const timer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);

  useEffect(() => () => clearTimeout(timer.current), []);

  const handleCopy = async () => {
    let next: Status = 'failed';
    try {
      await navigator.clipboard.writeText(textToCopy);
      next = 'copied';
    } catch {
      // no clipboard here, or the write was refused
    }
    setStatus(next);
    clearTimeout(timer.current);
    timer.current = setTimeout(() => setStatus('idle'), 2000);
  };

  return (
    <>
      <button type="button" className={className} onClick={handleCopy}>
        {status === 'copied' ? 'Copied ✓' : status === 'failed' ? 'Copy failed' : label}
      </button>
      <span className="sr-only" aria-live="polite">
        {status === 'copied' ? 'Copied to clipboard.' : status === 'failed' ? 'Could not copy. Select the text instead.' : ''}
      </span>
    </>
  );
};
