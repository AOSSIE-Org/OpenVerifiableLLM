import React from 'react';
import { copy } from '../content/copy';
import type { ChainBoundary, ReleaseAnchor } from '../data/contracts';

interface ChainQuoteProps {
  chain: ChainBoundary[];
  anchor: ReleaseAnchor;
}

const HEAD = 6;

// The claim, shown with its evidence: the last three chain entries, where the
// start of each entry's "previous" field visibly matches the hash above it.
export const ChainQuote: React.FC<ChainQuoteProps> = ({ chain, anchor }) => {
  const last = chain.slice(-3);

  return (
    <section className="quote-sec">
      <div className="wrap quote">
        <div className="quote-text">
          <div className="kicker">{copy.quote.kicker}</div>
          <p className="quote-line">{copy.quote.text}</p>
          <p className="quote-meta">
            {anchor.chain_entries} {copy.quote.entries} · {copy.quote.head}{' '}
            <code>{anchor.chain_head_sha256.slice(0, 12)}…</code>
          </p>
        </div>

        <ol className="qchain" aria-label={copy.quote.aria}>
          {last.map((b, i) => (
            <li key={b.index}>
              <div className="qc-top">
                <span className="qc-i">#{String(b.index).padStart(2, '0')}</span>
                <span className="qc-kind">{b.kind}</span>
              </div>
              <div className="qc-row">
                <small>{copy.quote.hash}</small>
                <code>
                  <mark className={i < last.length - 1 ? 'link' : undefined}>{b.entryHash.slice(0, HEAD)}</mark>
                  {b.entryHash.slice(HEAD, 22)}…
                </code>
              </div>
              <div className="qc-row">
                <small>{copy.quote.previous}</small>
                <code>
                  <mark className={i > 0 ? 'link' : undefined}>{(b.previous ?? '').slice(0, HEAD)}</mark>
                  {(b.previous ?? '').slice(HEAD, 22)}…
                </code>
              </div>
            </li>
          ))}
        </ol>
      </div>
    </section>
  );
};
