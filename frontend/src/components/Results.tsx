import React from 'react';
import { claimsFor } from '../content/claims';
import { copy } from '../content/copy';
import type { CheckOutcome, HardwareProbe, PreparationData, SegmentCoverage } from '../data/contracts';
import { detectionOdds, type Facts } from '../data/facts';
import { Explain } from './Explain';
import { Outcome } from './Outcome';

interface ResultsProps {
  facts: Facts;
  coverage: SegmentCoverage[];
  hardwareMap: HardwareProbe[];
  preparation: PreparationData;
}

const count = (n: number) => n.toLocaleString('en-US');

const probeLabel = (segment: number | string) =>
  typeof segment === 'number' ? `Segment ${segment}` : `${segment.charAt(0).toUpperCase()}${segment.slice(1)} segments`;

const outcomeWords = (result: CheckOutcome) =>
  result === 'MATCH' ? 'MATCH' : result === 'MISMATCH' ? 'MISMATCH' : 'not checked';

const cellClass = (result: CheckOutcome) =>
  result === 'MATCH' ? 'cell match' : result === 'MISMATCH' ? 'cell mismatch' : 'cell none';

// What was reproduced, who checked it, and what that does and doesn't show.
export const Results: React.FC<ResultsProps> = ({ facts, coverage, hardwareMap, preparation }) => {
  const o = copy.overview;
  const v = copy.verification;
  const claims = claimsFor(facts);
  const r = claims.results;
  const showMismatchKey = facts.projectMismatched + facts.independentMismatched > 0;

  return (
    <section id="overview" className="sec">
      <div className="sec-head">
        <h2>{o.kicker}</h2>
        <p className="sub">{r.subtitle}</p>
        <Explain>{r.explain}</Explain>
      </div>

      <div className="sec-body">
        <div id="verification">
          <h3>{v.coverageTitle}</h3>
          <p className="cov-sum">
            <span className="cov-counts">
              <b>
                {r.projectSummary}
                {r.projectExtra}
              </b>
              <span className="lg-i">
                {r.independentSummary}
                {r.independentExtra}
              </span>
            </span>
            <span className="legend" aria-hidden="true">
              <span className="lg">
                <span className="swatch cell match" /> Match
              </span>
              {showMismatchKey && (
                <span className="lg">
                  <span className="swatch cell mismatch" /> Mismatch
                </span>
              )}
              <span className="lg">
                <span className="swatch cell none" /> Not checked
              </span>
            </span>
          </p>
          <CoverageMap coverage={coverage} label={r.mapLabel} />
          <p className="notice" role="note">
            <b>{v.warningNotice}</b> {r.notice} {r.noticeClose}
          </p>
        </div>

        <div className="cols">
          <div>
            <h3>{o.shownTitle}</h3>
            <ul className="ticks">
              {r.shown.map(item => (
                <li key={item}>{item}</li>
              ))}
            </ul>
          </div>
          <div>
            <h3>{o.notShownTitle}</h3>
            <ul className="circles">
              {r.notShown.map(item => (
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
                {hardwareMap.map(probe => (
                  <tr key={`${probe.hardware}-${probe.segment}`}>
                    <th scope="row">{probe.hardware}</th>
                    <td>{probeLabel(probe.segment)}</td>
                    <td>
                      <Outcome result={probe.result} />
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
                <dt>{claims.odds.checksToday}</dt>
                <dd>{facts.independentChecks}</dd>
              </div>
              <div>
                <dt>checks for 95% odds</dt>
                <dd>{facts.checksFor95}</dd>
              </div>
              <div>
                <dt>checks for 99% odds</dt>
                <dd>{facts.checksFor99}</dd>
              </div>
            </dl>
            <Explain label="Show the curve">
              <p>{claims.odds.explain}</p>
              <p className="note">
                <code>{claims.odds.formula}</code>
              </p>
              <OddsChart facts={facts} description={claims.odds.description} today={claims.odds.today} />
            </Explain>
          </div>
        </div>

        <div>
          <h3>{o.dataTitle}</h3>
          <dl className="figures small data">
            <div>
              <dt>Articles</dt>
              <dd>{count(preparation.articles)}</dd>
            </div>
            <div>
              <dt>Wikipedia targets</dt>
              <dd>{count(preparation.wikipediaTargets)}</dd>
            </div>
            <div>
              <dt>Conversations</dt>
              <dd>{count(preparation.conversations)}</dd>
            </div>
            <div>
              <dt>Training code, ovl.py</dt>
              <dd>{count(preparation.trainingCodeLines)} lines</dd>
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

// The two rows are drawn separately, so a project check can never read as an independent one.
const CoverageMap: React.FC<{ coverage: SegmentCoverage[]; label: string }> = ({ coverage, label }) => {
  const rows = [
    { name: 'Project', key: 'projectResult', who: 'the project' },
    { name: 'Independent', key: 'independentResult', who: 'anyone independent' },
  ] as const;

  return (
    <>
      <div className="covgrid" role="img" aria-label={label}>
        {rows.map(row => (
          <React.Fragment key={row.key}>
            <span className="rl">{row.name}</span>
            <div className="cells" style={{ gridTemplateColumns: `repeat(${coverage.length}, minmax(0, 1fr))` }}>
              {coverage.map(seg => (
                <span
                  key={seg.segment}
                  className={cellClass(seg[row.key])}
                  title={`Segment ${seg.segment}: ${outcomeWords(seg[row.key])}, by ${row.who}`}
                />
              ))}
            </div>
          </React.Fragment>
        ))}
        <span className="rl" aria-hidden="true" />
        <div className="cells axis" aria-hidden="true" style={{ gridTemplateColumns: `repeat(${coverage.length}, minmax(0, 1fr))` }}>
          {coverage.map(seg => (
            <span key={seg.segment}>{seg.segment % 6 === 0 ? seg.segment : ''}</span>
          ))}
        </div>
      </div>
      <table className="sr-only">
        <caption>Result of each segment, by who checked it</caption>
        <thead>
          <tr>
            <th scope="col">Segment</th>
            <th scope="col">Project</th>
            <th scope="col">Independent</th>
          </tr>
        </thead>
        <tbody>
          {coverage.map(seg => (
            <tr key={seg.segment}>
              <td>{seg.segment}</td>
              <td>{outcomeWords(seg.projectResult)}</td>
              <td>{outcomeWords(seg.independentResult)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </>
  );
};

const OddsChart: React.FC<{ facts: Facts; description: string; today: string }> = ({ facts, description, today }) => {
  const W = 520;
  const H = 230;
  const L = 44;
  const R = 16;
  const T = 24;
  const B = 34;
  const { segments, independentChecks, checksFor95, checksFor99 } = facts;
  const max = Math.max(50, Math.ceil(Math.max(checksFor99, independentChecks) / 50) * 50 + 50);
  const getX = (m: number) => L + (m / max) * (W - L - R);
  const getY = (p: number) => T + (1 - p) * (H - T - B);
  const getOdds = (m: number) => detectionOdds(m, segments);
  const percent = (m: number) => `${(getOdds(m) * 100).toFixed(1)}%`;

  let linePath = '';
  for (let m = 0; m <= max; m += 2) {
    linePath += `${m === 0 ? 'M' : 'L'}${getX(m).toFixed(1)},${getY(getOdds(m)).toFixed(1)}`;
  }
  const ticks = Array.from({ length: max / 50 + 1 }, (_, i) => i * 50);

  return (
    <figure className="chart">
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-labelledby="odds-title odds-desc">
        <title id="odds-title">Detection odds curve for {segments} segments</title>
        <desc id="odds-desc">{description}</desc>
        {[0, 0.5, 1].map(p => (
          <g key={p}>
            <line x1={L} x2={W - R} y1={getY(p)} y2={getY(p)} stroke="var(--rule)" />
            <text x={L - 8} y={getY(p) + 4} fontSize="11" textAnchor="end">
              {p * 100}%
            </text>
          </g>
        ))}
        {ticks.map(m => (
          <text key={m} x={getX(m)} y={H - 12} fontSize="11" textAnchor="middle">
            {m}
          </text>
        ))}
        <path d={linePath} fill="none" stroke="var(--ink)" strokeWidth={2} />
        {[
          { m: checksFor95, label: `${checksFor95}: 95%` },
          { m: checksFor99, label: `${checksFor99}: 99%` },
        ].map(({ m, label }) => (
          <g key={m}>
            <circle cx={getX(m)} cy={getY(getOdds(m))} r={4} fill="var(--paper)" stroke="var(--ink)" strokeWidth={2} />
            <text x={getX(m)} y={getY(getOdds(m)) + 22} fontSize="12" textAnchor="middle" className="strong">
              {label}
            </text>
          </g>
        ))}
        <circle cx={getX(independentChecks)} cy={getY(getOdds(independentChecks))} r={6} fill="var(--event)" />
        <text x={getX(independentChecks) + 12} y={getY(getOdds(independentChecks)) - 8} fontSize="12" className="strong">
          {today}
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
            <td>{independentChecks}</td>
            <td>{percent(independentChecks)} (current)</td>
          </tr>
          <tr>
            <td>{checksFor95}</td>
            <td>{percent(checksFor95)}</td>
          </tr>
          <tr>
            <td>{checksFor99}</td>
            <td>{percent(checksFor99)}</td>
          </tr>
        </tbody>
      </table>
    </figure>
  );
};
