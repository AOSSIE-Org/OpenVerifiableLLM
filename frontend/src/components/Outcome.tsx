import React from 'react';
import type { CheckOutcome } from '../data/contracts';

// One way to show a result everywhere: a word and a mark, so it never rests on colour.
// No result reads as not checked, never as a pass.
export const Outcome: React.FC<{ result: CheckOutcome }> = ({ result }) => {
  if (result === 'MATCH') return <span className="res match">✓ MATCH</span>;
  if (result === 'MISMATCH') return <span className="res mismatch">✕ MISMATCH</span>;
  return <span className="res none">– NOT CHECKED</span>;
};
