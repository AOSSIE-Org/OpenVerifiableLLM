import React from 'react';
import { copy } from '../content/copy';
import { Explain } from './Explain';
import type { HardwareProbe, PreparationData, VerificationsData } from '../data/contracts';

interface ResultsProps {
  verifications: VerificationsData;
  hardwareMap: HardwareProbe[];
  preparation: PreparationData;
}

const count = (n: number) => n.toLocaleString('en-US');

// What was reproduced, who checked it, and what that does and doesn't show.
export const Results: React.FC<ResultsProps> = ({ verifications, hardwareMap, preparation }) => {
  const o = copy.overview;
  const v = copy.verification;
  return (
    <section id="overview" className="sec">
      <div className="sec-head">
        <h2>{o.kicker}</h2>
        <p className="sub">{o.subtitle}</p>
        <Explain>{o.explain}</Explain>
      </div>

      <div className="sec-body">
        <div id="verification">
          <h3>{v.coverageTitle}</h3>
          <p className="cov-sum">
            <b>
              {verifications.uniqueSegments} / {verifications.uniqueSegments} matched
            </b>
            <span className="legend">
              <span className="lg lg-p">■ Project: {verifications.uniqueSegments}</span>
              <span className="lg lg-i">▨ Independent: {verifications.independentChecksCount}</span>
            </span>
          </p>
          <CoverageMap verifications={verifications} />
          <p className="notice" role="note">
            <b>{v.warningNotice}</b> {v.captionNotice} {v.intro}
          </p>
        </div>

        <div className="cols">
          <div>
            <h3>{o.shownTitle}</h3>
            <ul className="ticks">
              {o.shownItems.map(item => (
                <li key={item}>{item}</li>
              ))}
            </ul>
          </div>
          <div>
            <h3>{o.notShownTitle}</h3>
            <ul className="circles">
              {o.notShownItems.map(item => (
                <li key={item}>{item}</li>
              ))}
            </ul>
          </div>
        </div>

        <div className="cols">
          <div>
            <h3>{v.hardwareTitle}</h3>
            <p className="note">{v.hardwareNote}</p>
            <table className="table">
              <thead>
                <tr>
                  <th scope="col">Hardware</th>
                  <th scope="col">Probe</th>
                  <th scope="col">Result</th>
                </tr>
              </thead>
              <tbody>
                {hardwareMap.map((probe, i) => (
                  <tr key={i}>
                    <th scope="row">{probe.hardware}</th>
                    <td>{probe.segment === 'all 37' ? 'All 37 segments' : `Segment ${probe.segment}`}</td>
                    <td>
                      <span className={`res ${probe.result === 'MATCH' ? 'match' : 'mismatch'}`}>
                        {probe.result === 'MATCH' ? '✓ MATCH' : '✕ MISMATCH'}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div>
            <h3>{v.oddsTitle}</h3>
            <dl className="figures small">
              <div className="caveat">
                <dd>{verifications.independentChecksCount}</dd>
                <dt>checks today</dt>
              </div>
              <div>
                <dd>110</dd>
                <dt>checks for 95% odds</dt>
              </div>
              <div>
                <dd>169</dd>
                <dt>checks for 99% odds</dt>
              </div>
            </dl>
            <Explain label="Show the curve">
              <p>{v.explain}</p>
              <p className="note">
                <code>{v.oddsFormula}</code>
              </p>
              <OddsChart />
            </Explain>
          </div>
        </div>

        <div>
          <h3>{o.dataTitle}</h3>
          <dl className="figures small data">
            <div>
              <dd>{count(preparation.articles)}</dd>
              <dt>Articles</dt>
            </div>
            <div>
              <dd>{count(preparation.wikipediaTargets)}</dd>
              <dt>Wikipedia targets</dt>
            </div>
            <div>
              <dd>{count(preparation.conversations)}</dd>
              <dt>Conversations</dt>
            </div>
            <div>
              <dd>{preparation.trainingLoopLines}</dd>
              <dt>Training loop</dt>
            </div>
          </dl>
          <p className="note">
            {o.dataRebuilt} <code>{preparation.preparedDataId.slice(0, 8)}…</code>
          </p>
        </div>
      </div>
    </section>
  );
};

const CoverageMap: React.FC<{ verifications: VerificationsData }> = ({ verifications }) => {
  const n = verifications.uniqueSegments;
  return (
    <div
      className="covgrid"
      role="img"
      aria-label={`${n} segments. Project checks: ${n} of ${n} MATCH. Independent checks: ${verifications.independentChecksCount} of ${n}.`}
    >
      <span className="rl">Project</span>
      <div className="cells">
        {verifications.coverage.map(seg => (
          <span
            key={`p-${seg.segment}`}
            className="cell p"
            title={`Segment ${seg.segment}: MATCH, checked by the project`}
          />
        ))}
      </div>
      <span className="rl">Independent</span>
      <div className="cells">
        {verifications.coverage.map(seg => (
          <span
            key={`i-${seg.segment}`}
            className="cell i"
            title={`Segment ${seg.segment}: not checked independently yet`}
          />
        ))}
      </div>
      <span className="rl" aria-hidden="true" />
      <div className="cells axis" aria-hidden="true">
        {verifications.coverage.map(seg => (
          <span key={seg.segment}>{seg.segment % 6 === 0 ? seg.segment : ''}</span>
        ))}
      </div>
    </div>
  );
};

const OddsChart: React.FC = () => {
  const W = 520;
  const H = 230;
  const L = 44;
  const R = 16;
  const T = 24;
  const B = 34;
  const max = 200;
  const getX = (m: number) => L + (m / max) * (W - L - R);
  const getY = (p: number) => T + (1 - p) * (H - T - B);
  const getOdds = (m: number) => 1 - Math.pow(36 / 37, m);

  let linePath = '';
  for (let m = 0; m <= max; m += 2) {
    linePath += `${m === 0 ? 'M' : 'L'}${getX(m).toFixed(1)},${getY(getOdds(m)).toFixed(1)}`;
  }

  return (
    <figure className="chart">
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-labelledby="odds-title odds-desc">
        <title id="odds-title">Detection odds curve for 37 segments</title>
        <desc id="odds-desc">{copy.verification.oddsDescription}</desc>
        {[0, 0.5, 1].map(p => (
          <g key={p}>
            <line x1={L} x2={W - R} y1={getY(p)} y2={getY(p)} stroke="var(--rule)" />
            <text x={L - 8} y={getY(p) + 4} fontSize="11" textAnchor="end">
              {p * 100}%
            </text>
          </g>
        ))}
        {[0, 50, 100, 150, 200].map(m => (
          <text key={m} x={getX(m)} y={H - 12} fontSize="11" textAnchor="middle">
            {m}
          </text>
        ))}
        <path d={linePath} fill="none" stroke="var(--ink)" strokeWidth={2} />
        {[
          { m: 110, label: '110: 95%', dy: 22 },
          { m: 169, label: '169: 99%', dy: 22 },
        ].map(({ m, label, dy }) => (
          <g key={m}>
            <circle cx={getX(m)} cy={getY(getOdds(m))} r={4} fill="var(--paper)" stroke="var(--ink)" strokeWidth={2} />
            <text x={getX(m)} y={getY(getOdds(m)) + dy} fontSize="12" textAnchor="middle" className="strong">
              {label}
            </text>
          </g>
        ))}
        <circle cx={getX(0)} cy={getY(0)} r={6} fill="var(--event)" />
        <text x={getX(0) + 12} y={getY(0) - 8} fontSize="12" className="strong">
          Today: 0 independent checks
        </text>
      </svg>
      <table className="sr-only">
        <caption>Detection odds by number of independent checks</caption>
        <thead>
          <tr>
            <th scope="col">Checks (M)</th>
            <th scope="col">Detection probability</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>0</td>
            <td>0% (current)</td>
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
    </figure>
  );
};
