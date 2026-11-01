import React from 'react';
import { copy } from '../content/copy';
import type { ChainBoundary } from '../data/contracts';

interface TrainingTimelineProps {
  chain: ChainBoundary[];
}

export const TrainingTimeline: React.FC<TrainingTimelineProps> = ({ chain }) => {
  const W = 900;
  const L = 16;
  const R = 16;
  const yy = 78;
  const x = (i: number): number => L + (i * (W - L - R)) / 37;
  const rx = (x(10) + x(11)) / 2;

  // Ticks are evenly spaced, so the very short conversation phase gets a
  // bracket that says how long it really took.
  const baseIndex = chain.findIndex(b => b.kind === 'base');
  const last = chain.length - 1;
  const convSeconds = baseIndex > 0 ? Math.round(chain[last].time - chain[baseIndex].time) : 0;

  return (
    <section id="timeline">
      <h2>{copy.timeline.kicker}</h2>
      <p className="sub">{copy.timeline.intro}</p>

      <figure className="thumb wide">
        <div className="tlwrap" tabIndex={0} role="region" aria-label="Training timeline chart">
          <svg
            viewBox={`0 0 ${W} 160`}
            width="100%"
            style={{ minWidth: '640px' }}
            role="img"
            aria-labelledby="tl-title tl-desc"
          >
            <title id="tl-title">Training timeline: 38 boundaries placed by time</title>
            <desc id="tl-desc">
              {copy.timeline.h2} Host restart at segment 10. Code change from segment 19 onward.
              Both MATCH.
            </desc>

            <line
              x1={x(19)}
              x2={x(19)}
              y1={yy - 44}
              y2={yy + 16}
              stroke="var(--event)"
              strokeWidth={1.5}
            />
            <text x={x(19) + 8} y={yy - 34} fontSize="12" style={{ fill: 'var(--event)' }}>
              Code change from segment 19 →
            </text>

            {/* Baseline axis */}
            <line x1={L} x2={W - R} y1={yy} y2={yy} stroke="var(--rule-strong)" strokeWidth={1.5} />

            {/* Green segment bars (MATCH) */}
            {Array.from({ length: 37 }).map((_, k) => (
              <rect
                key={k}
                x={x(k) + 3}
                y={yy - 4}
                width={x(k + 1) - x(k) - 6}
                height={8}
                rx={2}
                fill="var(--match-fill)"
              >
                <title>{`Segment ${k}: MATCH`}</title>
              </rect>
            ))}

            {/* Host restart at segment 10 */}
            <line
              x1={rx}
              x2={rx}
              y1={yy - 44}
              y2={yy + 16}
              stroke="var(--event)"
              strokeWidth={1.5}
            />
            <text
              x={rx}
              y={yy - 50}
              fontSize="12"
              textAnchor="middle"
              style={{ fill: 'var(--event)' }}
            >
              Host restart at segment 10
            </text>

            {/* Boundary tick marks */}
            {Array.from({ length: 38 }).map((_, i) => {
              const isMajor = i === 0 || i === 37;
              // 35 is skipped to avoid label collision with 37
              const showNumber = (i % 5 === 0 && i !== 35) || i === 37;
              return (
                <g key={i}>
                  <line
                    x1={x(i)}
                    x2={x(i)}
                    y1={yy - 9}
                    y2={yy + 9}
                    stroke="var(--ink-3)"
                    strokeWidth={isMajor ? 2 : 1}
                  >
                    <title>{`Boundary ${i}${chain[i] ? ` (${chain[i].kind}, step ${chain[i].control.global_step})` : ''}`}</title>
                  </line>
                  {showNumber && (
                    <text
                      x={x(i)}
                      y={yy + 34}
                      fontSize="11"
                      textAnchor="middle"
                      style={{ fill: 'var(--ink-3)' }}
                    >
                      {i}
                    </text>
                  )}
                </g>
              );
            })}

            {baseIndex > 0 && (
              <g>
                <path
                  d={`M${x(baseIndex)} ${yy + 44}v6H${x(last)}v-6`}
                  fill="none"
                  stroke="var(--ink-3)"
                  strokeWidth={1}
                />
                <text
                  x={x(last)}
                  y={yy + 64}
                  fontSize="12"
                  textAnchor="end"
                  style={{ fill: 'var(--ink-2)' }}
                >
                  Conversation phase: {convSeconds} s · not to scale
                </text>
              </g>
            )}
          </svg>
        </div>
        <figcaption>37 segments, all matched. Evenly spaced, not to time.</figcaption>
      </figure>

      <div className="split">
        <dl className="events">
          <dt>
            {copy.timeline.restartTitle} ({copy.timeline.restartTag.toLowerCase()})
          </dt>
          <dd>
            {copy.timeline.restartDesc} <span className="res match">✓ MATCH</span>
          </dd>
          <dt>
            {copy.timeline.codeChangeTitle} ({copy.timeline.codeChangeTag.toLowerCase()})
          </dt>
          <dd>
            {copy.timeline.codeChangeDesc} <span className="res match">✓ MATCH</span>
          </dd>
        </dl>

        <p className="note">{copy.timeline.timestampNote}</p>
      </div>

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
    </section>
  );
};
