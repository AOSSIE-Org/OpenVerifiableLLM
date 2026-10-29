import { useState, type ReactNode } from 'react';
import { claim } from '../content/ledger';
import { KNOWN_ISSUES } from '../content/audit';
import { useData } from '../data/context';
import { CommandBlock } from './CommandBlock';
import { SourceChip } from './SourceChip';
import { L, V, Who } from './text';

/** Evidence block, in the brief's order: claim → evidence → who checked → limit → source chip.
 *  Plain sentence first, developer line underneath. */
export function ClaimBlock({ id, children, heading, hideLimit = false, asLimit = false }: { id: string; children?: ReactNode; heading?: string; hideLimit?: boolean; asLimit?: boolean }) {
  const c = claim(id);
  const [all, setAll] = useState(false);
  const { data } = useData();
  const sources = data.claimSources[id] ?? [];
  return (
    <article className="ev" data-claim={id} id={id.toLowerCase()}>
      {heading ? (
        <h3>
          <L k={heading} />
        </h3>
      ) : null}
      {asLimit ? <LimitSentence plain={c.plain} /> : <p className="plain" data-claim-part="plain">{c.plain}</p>}
      {c.who ? <Who who={c.who} /> : null}
      {/* A ledger line with a number the pinned data does not support is withheld (see ledger-issues.json). */}
      {KNOWN_ISSUES.some((i) => i.id === id && i.part === 'developer') ? null : <p className="dev" data-claim-part="developer">{c.developer}</p>}
      {c.verbatim ? <CommandBlock text={c.verbatim} name={c.id} /> : null}
      {children}
      {c.limit && !hideLimit && !KNOWN_ISSUES.some((i) => i.id === id && i.part === 'limit') ? (
        <p className="limit">
          <strong>
            <L k="limit" />
          </strong>{' '}
          <span data-claim-part="limit">{c.limit}</span>
        </p>
      ) : null}
      {sources.length ? (
        <div className="chips">
          <SourceChip s={sources[0]!} />
          {sources.length > 1 ? (
            <>
              <button type="button" className="more" aria-expanded={all} aria-controls={`${id}-src`} onClick={() => setAll(!all)}>
                {all ? (
                  <L k="sources.fewer" />
                ) : (
                  <>
                    <span aria-hidden="true" className="plus" /><V path={`claimExtraSources.${id}`} /> <L k={sources.length === 2 ? 'sources.one' : 'sources.more'} />
                  </>
                )}
              </button>
              <div id={`${id}-src`} className="chips-more" hidden={!all}>
                {sources.slice(1).map((s, i) => (
                  <SourceChip key={i} s={s} />
                ))}
              </div>
            </>
          ) : null}
        </div>
      ) : null}
    </article>
  );
}

/** In a Limits list the heading already says "Limits", so the ledger's "Limit:" prefix is dropped.
 *  "Limit noted by the project:" becomes the swappable label "The project notes:" so the sentence
 *  still says who noted it. The rest of the sentence is the ledger text, unchanged. */
export const LIMIT_PREFIX = /^Limit( noted by the project)?:\s*/;
export function limitParts(plain: string): { noted: boolean; rest: string } {
  const m = LIMIT_PREFIX.exec(plain);
  if (!m) return { noted: false, rest: plain };
  const rest = plain.slice(m[0].length);
  return { noted: Boolean(m[1]), rest: m[1] ? rest : rest.charAt(0).toUpperCase() + rest.slice(1) };
}
function LimitSentence({ plain }: { plain: string }) {
  const { noted, rest } = limitParts(plain);
  return (
    <p className="plain">
      {noted ? <><L k="limit.noted" />{' '}</> : null}
      <span data-claim-part="limit-text">{rest}</span>
    </p>
  );
}
