import React from 'react';
import { claimsFor, controlLossNote, removedLossNote, timelineClaims } from '../content/claims';
import { copy } from '../content/copy';
import type { ChainBoundary, CheckOutcome, ExclusionData, SegmentCoverage } from '../data/contracts';
import { combine, type Facts } from '../data/facts';
import { Explain } from './Explain';
import { Outcome } from './Outcome';

interface TrainingHistoryProps {
  facts: Facts;
  chain: ChainBoundary[];
  coverage: SegmentCoverage[];
  exclusion: ExclusionData;
}

const signed = (n: number) => (n > 0 ? `+${n.toFixed(4)}` : n.toFixed(4));

// What happened during training, and the fork that left one article out.
export const TrainingHistory: React.FC<TrainingHistoryProps> = ({ facts, chain, coverage, exclusion }) => {
  const t = copy.timeline;
  const baseIndex = chain.findIndex(b => b.kind === 'base');
  const last = chain.length - 1;
  const convSeconds = baseIndex > 0 ? Math.round(chain[last].time - chain[baseIndex].time) : 0;
  const { hostFailureSegment, codeChangeBoundary } = facts;

  // Each event's mark is the result of the segments it touched, never assumed.
  const failure = coverage[hostFailureSegment]?.projectResult ?? null;
  const codeChange = combine(coverage.slice(codeChangeBoundary).map(c => c.projectResult));
  const claims = timelineClaims(facts, { failure, codeChange });

  return (
    <section id="timeline" className="sec">
      <div className="sec-head">
        <h2>{t.kicker}</h2>
        <p className="sub">{claims.intro}</p>
      </div>

      <div className="sec-body">
        <figure className="timeline">
          <TimelineChart facts={facts} chain={chain} coverage={coverage} baseIndex={baseIndex} description={`${claims.intro} ${claims.caption}`} />
          <figcaption>{claims.caption}</figcaption>
          <ol className="milestones">
            <li>
              <span className="when">Segment {hostFailureSegment}</span>
              <b>{t.restartTitle}</b>
              <p>{t.restartDesc}</p>
              <Outcome result={failure} />
            </li>
            <li>
              <span className="when">Boundary {codeChangeBoundary} onward</span>
              <b>{t.codeChangeTitle}</b>
              <p>{t.codeChangeDesc}</p>
              <Outcome result={codeChange} />
            </li>
            {baseIndex > 0 && (
              <li>
                <span className="when">Boundary {baseIndex}</span>
                <b>{t.baseTitle}</b>
                <p>Wikipedia pass done.</p>
              </li>
            )}
            <li>
              <span className="when">Boundary {last}</span>
              <b>{t.finalTitle}</b>
              <p>Conversation phase took {convSeconds} s.</p>
            </li>
          </ol>
        </figure>
        <p className="note">
          {claimsFor(facts).timestampNote} {t.sourceLabel}{' '}
          <a href={facts.eventsSourceUrl}>
            <code>docs/how-it-works.md</code>
          </a>{' '}
          at {facts.releaseTag}.
        </p>

        <table className="sr-only">
          <caption>Training boundaries sequence and timestamps</caption>
          <thead>
            <tr>
              <th scope="col">Boundary</th>
              <th scope="col">Phase</th>
              <th scope="col">Step</th>
              <th scope="col">Timestamp</th>
            </tr>
          </thead>
          <tbody>
            {chain.map(entry => (
              <tr key={entry.index}>
                <td>{entry.index}</td>
                <td>
                  {entry.kind} ({entry.control.phase})
                </td>
                <td>{entry.control.global_step}</td>
                <td>{new Date(entry.time * 1000).toISOString()}</td>
              </tr>
            ))}
          </tbody>
        </table>

        <Exclusion facts={facts} exclusion={exclusion} />
      </div>
    </section>
  );
};

const segmentFill = (result: CheckOutcome) =>
  result === 'MATCH' ? 'var(--match-fill)' : result === 'MISMATCH' ? 'var(--miss)' : 'none';

const TimelineChart: React.FC<{
  facts: Facts;
  chain: ChainBoundary[];
  coverage: SegmentCoverage[];
  baseIndex: number;
  description: string;
}> = ({ facts, chain, coverage, baseIndex, description }) => {
  // Drawn near its usual width, so the labels stay near their stated size.
  const W = 760;
  const L = 8;
  const R = 8;
  const yy = 58;
  const segments = chain.length - 1;
  const { hostFailureSegment, codeChangeBoundary } = facts;
  const x = (i: number): number => L + (i * (W - L - R)) / segments;
  const events = [
    { at: (x(hostFailureSegment) + x(hostFailureSegment + 1)) / 2, label: copy.timeline.restartTitle, event: true },
    { at: x(codeChangeBoundary), label: copy.timeline.codeChangeTitle, event: true },
    ...(baseIndex > 0 ? [{ at: x(baseIndex), label: 'Base model', event: false }] : []),
  ];
  const ticks = [
    ...Array.from({ length: Math.floor((segments - 1) / 5) + 1 }, (_, i) => i * 5).filter(i => segments - i > 2),
    segments,
  ];

  return (
    <svg viewBox={`0 0 ${W} 110`} width="100%" className="tl-chart" role="img" aria-labelledby="tl-title tl-desc">
      <title id="tl-title">Training timeline: {chain.length} boundaries, evenly spaced</title>
      <desc id="tl-desc">
        {description} {copy.timeline.restartTitle} in segment {hostFailureSegment}. {copy.timeline.codeChangeTitle} from boundary {codeChangeBoundary} onward.
      </desc>

      {coverage.map(({ segment, projectResult }) => {
        // A mismatch is taller as well as red, so it doesn't rest on colour.
        const half = projectResult === 'MISMATCH' ? 11 : 6;
        return (
          <rect
            key={segment}
            x={x(segment) + 2}
            y={yy - half}
            width={x(segment + 1) - x(segment) - 4}
            height={half * 2}
            fill={segmentFill(projectResult)}
            stroke={projectResult === null ? 'var(--rule-strong)' : 'none'}
          >
            <title>{`Segment ${segment}: ${projectResult ?? 'not checked'}, by the project`}</title>
          </rect>
        );
      })}

      {events.map(({ at, label, event }) => (
        <g key={label}>
          <line x1={at} x2={at} y1={yy - 34} y2={yy + 12} stroke={event ? 'var(--event)' : 'var(--ink-3)'} strokeWidth={1.5} />
          <text x={at} y={yy - 40} fontSize="13" textAnchor="middle" className={event ? 'event' : undefined}>
            {label}
          </text>
        </g>
      ))}

      {ticks.map(i => (
        <text key={i} x={x(i)} y={yy + 32} fontSize="12" textAnchor={i === 0 ? 'start' : i === segments ? 'end' : 'middle'}>
          {i}
        </text>
      ))}
    </svg>
  );
};

const Exclusion: React.FC<{ facts: Facts; exclusion: ExclusionData }> = ({ facts, exclusion }) => {
  const c = copy.exclusion;
  // Both bars share one scale, so the two changes can be compared directly.
  const scale = Math.max(Math.abs(exclusion.removedLossDelta), Math.abs(exclusion.controlLossDelta));
  const rows = [
    {
      label: c.removedArticleLabel,
      original: exclusion.originalRemovedLoss,
      fork: exclusion.forkRemovedLoss,
      delta: exclusion.removedLossDelta,
      note: removedLossNote(exclusion.removedLossDelta),
    },
    {
      label: `Loss on ${exclusion.controlArticlesCount} control articles`,
      original: exclusion.originalControlLossMean,
      fork: exclusion.forkControlLossMean,
      delta: exclusion.controlLossDelta,
      note: controlLossNote(exclusion.controlLossDelta, exclusion.removedLossDelta),
    },
  ];

  return (
    <div id="exclusion">
      <h3>{c.kicker}</h3>
      <p>
        {c.h2} Removed article: <b>{exclusion.removedArticle}</b>.
      </p>
      <div className="losses">
        {rows.map(row => (
          <div className="loss" key={row.label}>
            <h4>{row.label}</h4>
            <dl>
              <div>
                <dt>Original</dt>
                <dd>{row.original.toFixed(4)}</dd>
              </div>
              <div>
                <dt>Fork</dt>
                <dd>{row.fork.toFixed(4)}</dd>
              </div>
              <div>
                <dt>Change</dt>
                <dd>{signed(row.delta)}</dd>
              </div>
            </dl>
            <div className="meter" aria-hidden="true">
              <span style={{ width: `${scale > 0 ? (Math.abs(row.delta) / scale) * 100 : 0}%` }} />
            </div>
            <p className="note">{row.note}</p>
          </div>
        ))}
      </div>

      <div className="cols">
        <div>
          <h4>{c.whyHoldsTitle}</h4>
          <ul className="ticks">
            {claimsFor(facts).forkWhy.map(item => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </div>
        <div>
          <h4>{c.limitsTitle}</h4>
          <ul className="circles">
            {[...c.limitsItems, ...claimsFor(facts).forkLimits].map(item => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </div>
      </div>
      <Explain label="Why this matters">{c.explain}</Explain>
      <p className="note">
        {c.evidenceLabel} <code>{exclusion.evidencePath}</code>
      </p>
    </div>
  );
};
