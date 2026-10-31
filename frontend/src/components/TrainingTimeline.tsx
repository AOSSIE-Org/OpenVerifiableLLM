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

  return (
    <section id="timeline">
      <div className="wrap">
        <div className="sh">
          <div className="kicker">{copy.timeline.kicker}</div>
          <h2>{copy.timeline.h2}</h2>
        </div>

        <div className="panel">
          <div
            className="tlwrap"
            tabIndex={0}
            role="region"
            aria-label="Training timeline chart"
          >
            <svg
              viewBox={`0 0 ${W} 160`}
              width="100%"
              style={{ minWidth: '640px' }}
              role="img"
              aria-labelledby="tl-title tl-desc"
            >
              <title id="tl-title">
                Training timeline: 38 boundaries placed by time
              </title>
              <desc id="tl-desc">
                {copy.timeline.h2} Host restart at segment 10. Code change from segment 19 onward. Both MATCH.
              </desc>

              {/* Code change highlight band from segment 19 to end */}
              <rect
                x={x(19)}
                y={yy - 22}
                width={x(37) - x(19)}
                height={38}
                rx={6}
                fill="color-mix(in srgb, var(--gold) 7%, transparent)"
                stroke="color-mix(in srgb, var(--gold) 35%, transparent)"
              />
              <line
                x1={x(19)}
                x2={x(19)}
                y1={yy - 44}
                y2={yy + 16}
                stroke="var(--gold)"
                strokeWidth={1.5}
              />
              <text
                x={x(19) + 8}
                y={yy - 34}
                fontSize="12"
                style={{ fill: 'var(--gold-text)' }}
              >
                Code change from segment 19 →
              </text>

              {/* Baseline axis */}
              <line
                x1={L}
                x2={W - R}
                y1={yy}
                y2={yy}
                stroke="var(--rule-strong)"
                strokeWidth={1.5}
              />

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
                stroke="var(--gold)"
                strokeWidth={1.5}
              />
              <text
                x={rx}
                y={yy - 50}
                fontSize="12"
                textAnchor="middle"
                style={{ fill: 'var(--gold-text)' }}
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

              <text x={L} y={yy + 60} fontSize="12" style={{ fill: 'var(--ink-3)' }}>
                {copy.timeline.caption}
              </text>
            </svg>
          </div>

          <div className="events">
            <div className="ev">
              <span className="tag">{copy.timeline.restartTag}</span>
              <h3>{copy.timeline.restartTitle}</h3>
              <p>
                {copy.timeline.restartDesc}{' '}
                <span className="chip m">✓ MATCH</span>
              </p>
            </div>

            <div className="ev">
              <span className="tag">{copy.timeline.codeChangeTag}</span>
              <h3>{copy.timeline.codeChangeTitle}</h3>
              <p>
                {copy.timeline.codeChangeDesc}{' '}
                <span className="chip m">✓ MATCH</span>
              </p>
            </div>
          </div>

          <p style={{ marginTop: '20px', fontSize: '13px', color: 'var(--ink-3)' }}>
            {copy.timeline.timestampNote}
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
              {chain.map((entry) => (
                <tr key={entry.index}>
                  <td>{entry.index}</td>
                  <td>{entry.kind} ({entry.control.phase})</td>
                  <td>{entry.control.global_step}</td>
                  <td>{new Date(entry.time * 1000).toISOString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  );
};
