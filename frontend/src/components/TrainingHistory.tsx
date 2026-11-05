import React from 'react';
import { copy } from '../content/copy';
import { Explain } from './Explain';
import type { ChainBoundary, ExclusionData } from '../data/contracts';

interface TrainingHistoryProps {
  chain: ChainBoundary[];
  exclusion: ExclusionData;
}

const signed = (n: number) => (n > 0 ? `+${n.toFixed(4)}` : n.toFixed(4));

// What happened during training, and the fork that left one article out.
export const TrainingHistory: React.FC<TrainingHistoryProps> = ({ chain, exclusion }) => {
  const t = copy.timeline;
  const baseIndex = chain.findIndex(b => b.kind === 'base');
  const last = chain.length - 1;
  const convSeconds = baseIndex > 0 ? Math.round(chain[last].time - chain[baseIndex].time) : 0;

  return (
    <section id="timeline" className="sec">
      <div className="sec-head">
        <h2>{t.kicker}</h2>
        <p className="sub">{t.intro}</p>
      </div>

      <div className="sec-body">
        <figure className="timeline">
          <TimelineChart chain={chain} baseIndex={baseIndex} />
          <figcaption>{chain.length - 1} segments, all matched. Evenly spaced, not to time.</figcaption>
          <ol className="milestones">
            <li>
              <span className="when">{t.restartTag}</span>
              <b>{t.restartTitle}</b>
              <p>
                {t.restartDesc} <span className="res match">✓ MATCH</span>
              </p>
            </li>
            <li>
              <span className="when">{t.codeChangeTag}</span>
              <b>{t.codeChangeTitle}</b>
              <p>
                {t.codeChangeDesc} <span className="res match">✓ MATCH</span>
              </p>
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
        <p className="note">{t.timestampNote}</p>

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

        <Exclusion exclusion={exclusion} />
      </div>
    </section>
  );
};

const TimelineChart: React.FC<{ chain: ChainBoundary[]; baseIndex: number }> = ({ chain, baseIndex }) => {
  const W = 900;
  const L = 8;
  const R = 8;
  const yy = 58;
  const segments = chain.length - 1;
  const x = (i: number): number => L + (i * (W - L - R)) / segments;
  const rx = (x(10) + x(11)) / 2;

  return (
    <svg viewBox={`0 0 ${W} 110`} width="100%" className="tl-chart" role="img" aria-labelledby="tl-title tl-desc">
      <title id="tl-title">Training timeline: {chain.length} boundaries, evenly spaced</title>
      <desc id="tl-desc">{copy.timeline.h2} Host restart at segment 10. Code change from segment 19 onward.</desc>

      {Array.from({ length: segments }).map((_, k) => (
        <rect key={k} x={x(k) + 2} y={yy - 6} width={x(k + 1) - x(k) - 4} height={12} fill="var(--match-fill)">
          <title>{`Segment ${k}: MATCH`}</title>
        </rect>
      ))}

      <line x1={rx} x2={rx} y1={yy - 34} y2={yy + 12} stroke="var(--event)" strokeWidth={1.5} />
      <text x={rx} y={yy - 40} fontSize="13" textAnchor="middle" className="event">
        Host restart
      </text>
      <line x1={x(19)} x2={x(19)} y1={yy - 34} y2={yy + 12} stroke="var(--event)" strokeWidth={1.5} />
      <text x={x(19)} y={yy - 40} fontSize="13" textAnchor="middle" className="event">
        Code change
      </text>
      {baseIndex > 0 && (
        <>
          <line x1={x(baseIndex)} x2={x(baseIndex)} y1={yy - 34} y2={yy + 12} stroke="var(--ink-3)" strokeWidth={1.5} />
          <text x={x(baseIndex)} y={yy - 40} fontSize="13" textAnchor="middle">
            Base model
          </text>
        </>
      )}

      {[0, 5, 10, 15, 20, 25, 30, segments].map(i => (
        <text key={i} x={x(i)} y={yy + 32} fontSize="12" textAnchor={i === 0 ? 'start' : i === segments ? 'end' : 'middle'}>
          {i}
        </text>
      ))}
    </svg>
  );
};

const Exclusion: React.FC<{ exclusion: ExclusionData }> = ({ exclusion }) => {
  const c = copy.exclusion;
  // Both bars share one scale, so the two changes can be compared directly.
  const scale = Math.max(Math.abs(exclusion.removedLossDelta), Math.abs(exclusion.controlLossDelta));
  const rows = [
    {
      label: c.removedArticleLabel,
      original: exclusion.originalRemovedLoss,
      fork: exclusion.forkRemovedLoss,
      delta: exclusion.removedLossDelta,
      note: c.removedArticleNote,
    },
    {
      label: c.controlArticlesLabel,
      original: exclusion.originalControlLossMean,
      fork: exclusion.forkControlLossMean,
      delta: exclusion.controlLossDelta,
      note: c.controlArticlesNote,
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
              <span style={{ width: `${(Math.abs(row.delta) / scale) * 100}%` }} />
            </div>
            <p className="note">{row.note}</p>
          </div>
        ))}
      </div>

      <div className="cols">
        <div>
          <h4>{c.whyHoldsTitle}</h4>
          <ul className="ticks">
            {c.whyHoldsItems.map(item => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </div>
        <div>
          <h4>{c.limitsTitle}</h4>
          <ul className="circles">
            {c.limitsItems.map(item => (
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
