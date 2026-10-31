import React from 'react';
import { copy } from '../content/copy';
import type { ChainBoundary } from '../data/contracts';
import { SectionHeading } from './Contents';
import { Ref } from './Ref';

const RESTART_SEGMENT = 10;
const CODE_CHANGE_SEGMENT = 19;

const utc = (t: number) => new Date(t * 1000).toISOString().replace('T', ' ').slice(0, 19) + ' UTC';

// A static diagram: boundaries evenly spaced, segments drawn between them.
const Diagram: React.FC<{ chain: ChainBoundary[] }> = ({ chain }) => {
  const segs = chain.length - 1;
  const W = 900;
  const L = 20;
  const R = 20;
  const y = 70;
  const x = (i: number) => L + (i * (W - L - R)) / segs;
  const restartX = (x(RESTART_SEGMENT) + x(RESTART_SEGMENT + 1)) / 2;
  const baseIndex = chain.findIndex(b => b.kind === 'base');

  return (
    <svg viewBox={`0 0 ${W} 130`} className="diagram" role="img" aria-labelledby="tl-title tl-desc">
      <title id="tl-title">Training timeline: {chain.length} boundaries</title>
      <desc id="tl-desc">
        {segs} segments, all MATCH. Host restart during segment {RESTART_SEGMENT}. Code change from segment{' '}
        {CODE_CHANGE_SEGMENT} onward.
      </desc>
      <rect x={x(CODE_CHANGE_SEGMENT)} y={y - 14} width={x(segs) - x(CODE_CHANGE_SEGMENT)} height={28} className="d-band" />
      <text x={x(CODE_CHANGE_SEGMENT) + 4} y={y - 20} className="d-label">
        Code change, segment {CODE_CHANGE_SEGMENT} onward
      </text>
      <line x1={L} x2={W - R} y1={y} y2={y} className="d-axis" />
      {Array.from({ length: segs }).map((_, k) => (
        <rect key={k} x={x(k) + 2} y={y - 4} width={x(k + 1) - x(k) - 4} height={8} className="d-seg">
          <title>{`Segment ${k}: MATCH`}</title>
        </rect>
      ))}
      {chain.map((b, i) => (
        <g key={b.index}>
          <line x1={x(i)} x2={x(i)} y1={y - 9} y2={y + 9} className="d-tick" />
          {(i % 5 === 0 || i === segs) && i !== 35 && (
            <text x={x(i)} y={y + 26} textAnchor="middle" className="d-num">
              {i}
            </text>
          )}
        </g>
      ))}
      <line x1={restartX} x2={restartX} y1={y - 40} y2={y + 12} className="d-event" />
      <text x={restartX} y={y - 44} textAnchor="middle" className="d-label">
        Host restart, segment {RESTART_SEGMENT}
      </text>
      {baseIndex > 0 && (
        <>
          <path d={`M${x(baseIndex)} ${y + 36}v6H${x(segs)}v-6`} className="d-bracket" />
          <text x={x(segs)} y={y + 56} textAnchor="end" className="d-num">
            Conversation phase (boundaries {baseIndex}–{segs})
          </text>
        </>
      )}
    </svg>
  );
};

export const TrainingTimeline: React.FC<{ chain: ChainBoundary[] }> = ({ chain }) => {
  const baseIndex = chain.findIndex(b => b.kind === 'base');
  const last = chain.length - 1;
  const wikiHours = baseIndex > 0 ? (chain[baseIndex].time - chain[0].time) / 3600 : 0;
  const convSeconds = baseIndex > 0 ? Math.round(chain[last].time - chain[baseIndex].time) : 0;

  return (
    <section id="timeline" aria-labelledby="timeline-h">
      <SectionHeading id="timeline" />
      <p>
        {copy.timeline.intro}
        <Ref k="chain" />
      </p>

      <figure className="thumb">
        <div className="table-scroll" tabIndex={0} role="region" aria-label="Training timeline diagram">
          <Diagram chain={chain} />
        </div>
        <figcaption>
          Boundaries are evenly spaced, not placed by time. The Wikipedia pass took about {wikiHours.toFixed(1)}{' '}
          hours; the conversation phase took {convSeconds} seconds.
        </figcaption>
      </figure>

      <table className="wikitable">
        <caption>Events during training</caption>
        <thead>
          <tr>
            <th scope="col">Event</th>
            <th scope="col">Where</th>
            <th scope="col">What happened</th>
            <th scope="col">Result</th>
          </tr>
        </thead>
        <tbody>
          {copy.timeline.events.map(e => (
            <tr key={e.title}>
              <td>{e.title}</td>
              <td>{e.where}</td>
              <td>{e.desc}</td>
              <td>
                <span className="res match">MATCH</span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="note">
        {copy.timeline.timestampNote}
        <Ref k="anchor" />
      </p>

      <details>
        <summary>
          {copy.timeline.tableTitle}: all {chain.length} entries from <code>chain.jsonl</code>
        </summary>
        <div className="table-scroll" tabIndex={0} role="region" aria-label="Chain boundaries">
          <table className="wikitable">
            <thead>
              <tr>
                <th scope="col">Index</th>
                <th scope="col">Kind</th>
                <th scope="col">Phase</th>
                <th scope="col">Epoch</th>
                <th scope="col">Global step</th>
                <th scope="col">Recorded time</th>
                <th scope="col">Entry hash</th>
              </tr>
            </thead>
            <tbody>
              {chain.map(b => (
                <tr key={b.index}>
                  <td className="num">{b.index}</td>
                  <td>{b.kind}</td>
                  <td>{b.control.phase}</td>
                  <td className="num">{b.control.epoch}</td>
                  <td className="num">{b.control.global_step.toLocaleString('en-US')}</td>
                  <td>{utc(b.time)}</td>
                  <td>
                    <code className="hash" title={b.entryHash}>
                      {b.entryHash.slice(0, 12)}…
                    </code>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
    </section>
  );
};
