import React from 'react';
import { copy } from '../content/copy';

// A short summary at the top of a section, for readers who skim.
export const InShort: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <div className="inshort">
    <b>{copy.article.inShort}</b>
    <p>{children}</p>
  </div>
);
