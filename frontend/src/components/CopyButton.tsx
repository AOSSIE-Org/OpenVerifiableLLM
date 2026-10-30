import React, { useState } from 'react';

interface CopyButtonProps {
  textToCopy: string;
  label?: string;
  className?: string;
}

export const CopyButton: React.FC<CopyButtonProps> = ({
  textToCopy,
  label = 'Copy',
  className = 'copy-btn',
}) => {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      if (navigator.clipboard) {
        await navigator.clipboard.writeText(textToCopy);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
      }
    } catch {
      /* clipboard write rejected */
    }
  };

  return (
    <>
      <button
        type="button"
        className={className}
        onClick={handleCopy}
      >
        {copied ? 'Copied ✓' : label}
      </button>
      <span className="sr-only" aria-live="polite">
        {copied ? 'Copied to clipboard.' : ''}
      </span>
    </>
  );
};
