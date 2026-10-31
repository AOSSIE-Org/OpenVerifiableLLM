import React from 'react';
import { copy } from '../content/copy';

// A plain table of contents, numbered like the section headings.
export const Contents: React.FC = () => (
  <nav className="toc" aria-labelledby="toc-title">
    <h2 id="toc-title">Contents</h2>
    <ol>
      <li>
        <a href="#top">(Top)</a>
      </li>
      {copy.contents.map((s, i) => (
        <li key={s.id}>
          <a href={`#${s.id}`}>
            <span className="tocnumber">{i + 1}</span> {s.label}
          </a>
        </li>
      ))}
    </ol>
  </nav>
);

/** Section heading with the same number as its contents entry. */
export const SectionHeading: React.FC<{ id: string }> = ({ id }) => {
  const i = copy.contents.findIndex(s => s.id === id);
  return (
    <h2 id={`${id}-h`}>
      <span className="secnum">{i + 1}</span> {copy.contents[i].label}
    </h2>
  );
};
