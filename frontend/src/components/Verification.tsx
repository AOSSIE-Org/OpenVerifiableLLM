import React from 'react';
import { copy } from '../content/copy';
import type { HardwareProbe, VerificationsData } from '../data/contracts';

interface VerificationProps {
  verifications: VerificationsData;
  hardwareMap: HardwareProbe[];
}

export const Verification: React.FC<VerificationProps> = ({
  verifications,
  hardwareMap,
}) => {
  // SVG odds calculations
  const W = 520;
  const H = 250;
  const L = 44;
  const R = 16;
  const T = 34;
  const B = 34;
  const max = 200;

  const getX = (m: number) => L + (m / max) * (W - L - R);
  const getY = (p: number) => T + (1 - p) * (H - T - B);
  const getOdds = (m: number) => 1 - Math.pow(36 / 37, m);

  let linePath = '';
  let areaPath = `M${getX(0)},${getY(0)}`;
  for (let m = 0; m <= max; m += 2) {
    const pt = `${getX(m).toFixed(1)},${getY(getOdds(m)).toFixed(1)}`;
    linePath += (m === 0 ? 'M' : 'L') + pt;
    areaPath += 'L' + pt;
  }
  areaPath += `L${getX(max)},${getY(0)}Z`;

  return (
    <section id="verification">
      <div className="wrap">
        <div className="sh">
          <div className="kicker">{copy.verification.kicker}</div>
          <h2>{copy.verification.h2}</h2>
          <p>{copy.verification.subtitle}</p>
        </div>

        <div className="panel">
          <div className="cov-head">
            <div>
              <div className="sub" style={{ marginBottom: '8px' }}>
                Coverage map
              </div>
              <div className="cov-big">
                {verifications.uniqueSegments}
                <small> / {verifications.uniqueSegments} matched</small>
              </div>
            </div>
            <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
              <span className="chip m">■ Project: {verifications.uniqueSegments}</span>
              <span className="chip h">▨ Independent: {verifications.independentChecksCount}</span>
            </div>
          </div>

          <div
            className="covgrid"
            id="cov"
            role="img"
            aria-label={`${verifications.uniqueSegments} segments. Project checks: ${verifications.uniqueSegments} of ${verifications.uniqueSegments} MATCH. Independent checks: ${verifications.independentChecksCount} of ${verifications.uniqueSegments}.`}
          >
            <span className="rl">Project</span>
            {verifications.coverage.map((seg) => (
              <span
                key={`p-${seg.segment}`}
                className="cell p"
                title={`Segment ${seg.segment}: MATCH, checked by the project`}
              />
            ))}

            <span className="rl">Independent</span>
            {verifications.coverage.map((seg) => (
              <span
                key={`i-${seg.segment}`}
                className="cell i"
                title={`Segment ${seg.segment}: not checked independently yet`}
              />
            ))}
          </div>

          <div className="axis" aria-hidden="true">
            <span></span>
            {verifications.coverage.map((seg) => {
              const k = seg.segment;
              const isKey = k % 9 === 0 || k === 36;
              const showText = k % 3 === 0 || k === 36;
              return (
                <span key={k} className={isKey ? 'k' : ''}>
                  {showText ? k : ''}
                </span>
              );
            })}
          </div>

          <div className="warn">
            <span className="dot" aria-hidden="true" />
            <div>
              <b>{copy.verification.warningNotice}</b>
              <div style={{ marginTop: '4px', fontSize: '13px', color: 'var(--ink-3)' }}>
                {copy.verification.captionNotice}
              </div>
            </div>
          </div>
        </div>

        <div className="two">
          <div className="panel">
            <h3>{copy.verification.oddsTitle}</h3>
            <div className="sub mono">{copy.verification.oddsFormula}</div>

            <svg
              viewBox={`0 0 ${W} ${H}`}
              width="100%"
              role="img"
              aria-labelledby="odds-title odds-desc"
            >
              <title id="odds-title">Detection odds curve for 37 segments</title>
              <desc id="odds-desc">{copy.verification.oddsDescription}</desc>

              {/* Grid lines */}
              {[0, 0.5, 1].map((p) => (
                <g key={p}>
                  <line
                    x1={L}
                    x2={W - R}
                    y1={getY(p)}
                    y2={getY(p)}
                    stroke="var(--rule)"
                  />
                  <text
                    x={L - 8}
                    y={getY(p) + 4}
                    fontSize="11"
                    textAnchor="end"
                  >
                    {p * 100}%
                  </text>
                </g>
              ))}

              {/* X Axis */}
              {[0, 50, 100, 150, 200].map((m) => (
                <text
                  key={m}
                  x={getX(m)}
                  y={H - 12}
                  fontSize="11"
                  textAnchor="middle"
                >
                  {m}
                </text>
              ))}

              {/* Area & curve */}
              <path d={areaPath} fill="var(--ink)" opacity={0.05} />
              <path d={linePath} fill="none" stroke="var(--ink)" strokeWidth={2} />

              {/* Threshold points (110 and 169) */}
              {[
                { m: 110, p: 0.95, label: '110 checks: 95%', align: 'left' },
                { m: 169, p: 0.99, label: '169 checks: 99%', align: 'center' },
              ].map(({ m, label, align }) => {
                const cx = getX(m);
                const cy = getY(getOdds(m));
                return (
                  <g key={m}>
                    <line
                      x1={cx}
                      x2={cx}
                      y1={cy}
                      y2={getY(0)}
                      stroke="var(--ink-3)"
                      strokeDasharray="2 3"
                    />
                    <circle
                      cx={cx}
                      cy={cy}
                      r={4.5}
                      fill="var(--sheet)"
                      stroke="var(--ink)"
                      strokeWidth={2}
                    />
                    <text
                      x={align === 'left' ? cx + 8 : cx}
                      y={align === 'left' ? cy + 20 : cy - 12}
                      fontSize="12"
                      textAnchor={align === 'center' ? 'middle' : 'start'}
                      style={{ fill: 'var(--ink)', fontWeight: 600 }}
                    >
                      {label}
                    </text>
                  </g>
                );
              })}

              {/* Current position (M = 0) */}
              <circle
                cx={getX(0)}
                cy={getY(0)}
                r={7}
                fill="var(--gold)"
                stroke="var(--paper)"
                strokeWidth={2}
              />
              <text
                x={getX(0) + 14}
                y={getY(0) - 10}
                fontSize="12"
                style={{ fill: 'var(--ink)', fontWeight: 600 }}
              >
                You are here: 0 independent checks
              </text>
            </svg>

            <table className="sr-only">
              <caption>Detection odds by number of independent checks</caption>
              <thead>
                <tr>
                  <th scope="col">Checks (M)</th>
                  <th scope="col">Detection Probability</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>0</td>
                  <td>0% (Current: 0 independent checks)</td>
                </tr>
                <tr>
                  <td>110</td>
                  <td>95.0%</td>
                </tr>
                <tr>
                  <td>169</td>
                  <td>99.0%</td>
                </tr>
              </tbody>
            </table>
          </div>

          <div className="panel hw-shining-card">
            <div className="hw-header">
              <h3>{copy.verification.hardwareTitle}</h3>
              <div className="sub">{copy.verification.hardwareSubtitle}</div>
            </div>
            <table className="hw-table">
              <thead>
                <tr>
                  <th scope="col">Hardware &amp; Architecture</th>
                  <th scope="col">Result</th>
                </tr>
              </thead>
              <tbody>
                {hardwareMap.map((probe, i) => {
                  const isFullRun = probe.segment === 'all 37';
                  const scopeLabel = isFullRun
                    ? 'Full run · all 37 segments verified'
                    : `Segment ${probe.segment} probe${probe.note ? ` · ${probe.note.split(';')[0]}` : ''}`;
                  return (
                    <tr key={i}>
                      <td>
                        <div className="hw-info">
                          <span className="hw-title">{probe.hardware}</span>
                          <span className={`hw-scope${isFullRun ? ' full-run' : ''}`}>
                            {scopeLabel}
                          </span>
                        </div>
                      </td>
                      <td style={{ textAlign: 'right' }}>
                        <span className={`chip res-chip ${probe.result === 'MATCH' ? 'match' : 'mismatch'}`}>
                          {probe.result === 'MATCH' ? '✓ MATCH' : '✕ MISMATCH'}
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
            <p className="hw-note">
              <span aria-hidden="true">ℹ</span>
              {copy.verification.hardwareNote}
            </p>
          </div>
        </div>
      </div>
    </section>
  );
};
