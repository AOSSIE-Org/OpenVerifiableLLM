import React, { useId, useState } from 'react';
import { copy } from '../content/copy';

// A short numbered list of questions; one answer open at a time.
export const Faq: React.FC = () => {
  const [open, setOpen] = useState<number | null>(0);
  const id = useId();

  return (
    <section id="faq" className="faq">
      <div className="wrap faq-wrap">
        <div className="sh">
          <h2>{copy.faq.h2}</h2>
          <p>{copy.faq.sub}</p>
        </div>

        <ol className="faq-list">
          {copy.faq.items.map((item, i) => {
            const isOpen = open === i;
            const panel = `${id}-a${i}`;
            return (
              <li key={item.q} className={isOpen ? 'open' : undefined}>
                <h3>
                  <button
                    type="button"
                    aria-expanded={isOpen}
                    aria-controls={panel}
                    onClick={() => setOpen(isOpen ? null : i)}
                  >
                    <span className="faq-n">{String(i + 1).padStart(2, '0')}</span>
                    <span className="faq-q">{item.q}</span>
                    <span className="faq-icon" aria-hidden="true">
                      {isOpen ? '−' : '+'}
                    </span>
                  </button>
                </h3>
                <div id={panel} hidden={!isOpen}>
                  <p>{item.a}</p>
                </div>
              </li>
            );
          })}
        </ol>

        <p className="faq-still">
          {copy.faq.stillTitle} <a href="https://discord.gg/hjUhu33uAn">{copy.faq.discord}</a> ·{' '}
          <a href={copy.faq.issueHref}>{copy.faq.issue}</a>
        </p>
      </div>
    </section>
  );
};
