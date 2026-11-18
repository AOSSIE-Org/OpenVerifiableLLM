import React from 'react';

interface ExplainProps {
  label?: string;
  children: React.ReactNode;
}

// A longer explanation folded away under a section, for readers who want it.
export const Explain: React.FC<ExplainProps> = ({ label = 'Explain this', children }) => (
  <details className="explain">
    <summary>{label}</summary>
    <div>{children}</div>
  </details>
);
