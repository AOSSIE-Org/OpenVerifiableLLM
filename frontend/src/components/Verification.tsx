import React from 'react';
import { copy } from '../content/copy';
import { KNOWN_CHECKER_MAP, type HardwareProbe, type VerificationsData } from '../data/contracts';
import { SectionHeading } from './Contents';
import { Ref } from './Ref';

interface VerificationProps {
  verifications: VerificationsData;
  hardwareMap: HardwareProbe[];
}

const ODDS_ROWS = [0, 10, 37, 50, 110, 169, 200];

export const detectionOdds = (n: number, m: number): number => 1 - Math.pow((n - 1) / n, m);

const Result: React.FC<{ r: 'MATCH' | 'MISMATCH' }> = ({ r }) => (
  <span className={r === 'MATCH' ? 'res match' : 'res mismatch'}>{r}</span>
);

export const Verification: React.FC<VerificationProps> = ({ verifications: v, hardwareMap }) => {
  const n = v.uniqueSegments;
  const matched = v.coverage.filter(c => c.projectResult === 'MATCH').length;
  const independent = v.independentChecksCount;

  return (
    <section id="verification" aria-labelledby="verification-h">
      <SectionHeading id="verification" />
      <p>
        {matched} of {n} segments have been recomputed and matched. All {v.totalChecks} recorded checks were run by
        the project; {independent} were independent.
        <Ref k="verifications" />
      </p>
      <p className="hatnote" role="note">
        {copy.verification.selfReported}
      </p>

      <h3>
        <span className="secnum">3.1</span> {copy.verification.coverageTitle}
      </h3>
      <p>{copy.verification.coverageIntro}</p>
      <div className="coverage">
        <div className="cov-row">
          <div className="cov-label" id="cov-project">
            Checked by the project ({matched}/{n})
          </div>
          <ol className="tiles" aria-labelledby="cov-project">
            {v.coverage.map(c => (
              <li
                key={c.segment}
                className={c.projectResult === 'MATCH' ? 'tile match' : 'tile mismatch'}
                title={`Segment ${c.segment}: ${c.projectResult}, checked by the project`}
              >
                {c.segment}
                <span className="sr-only">: {c.projectResult}</span>
              </li>
            ))}
          </ol>
        </div>
        <div className="cov-row">
          <div className="cov-label" id="cov-independent">
            Checked independently ({independent}/{n})
          </div>
          <ol className="tiles" aria-labelledby="cov-independent">
            {v.coverage.map(c => (
              <li
                key={c.segment}
                className={c.independentResult ? `tile ${c.independentResult.toLowerCase()}` : 'tile empty'}
                title={`Segment ${c.segment}: ${c.independentResult ?? 'not checked independently yet'}`}
              >
                {c.segment}
                <span className="sr-only">: {c.independentResult ?? 'not checked'}</span>
              </li>
            ))}
          </ol>
        </div>
        <p className="legend">
          <span className="tile match sample" aria-hidden="true" /> MATCH{' '}
          <span className="tile empty sample" aria-hidden="true" /> not checked yet
        </p>
      </div>

      <h3>
        <span className="secnum">3.2</span> {copy.verification.oddsTitle}
      </h3>
      <p>{copy.verification.oddsIntro}</p>
      <p className="formula">
        <code>{copy.verification.oddsFormula}</code>, N = {n}
      </p>
      <table className="wikitable">
        <caption>Chance that M random independent checks catch one bad segment</caption>
        <thead>
          <tr>
            <th scope="col">Independent checks (M)</th>
            <th scope="col">Detection probability</th>
          </tr>
        </thead>
        <tbody>
          {ODDS_ROWS.map(m => (
            <tr key={m} className={m === independent ? 'current' : undefined}>
              <td className="num">
                {m}
                {m === independent && <span className="note"> (current count)</span>}
              </td>
              <td className="num">{(detectionOdds(n, m) * 100).toFixed(1)}%</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p>
        Current independent checks: <b>{independent}</b>. 110 random checks would give 95%; 169 would give 99%.
      </p>

      <h3>
        <span className="secnum">3.3</span> {copy.verification.hardwareTitle}
      </h3>
      <p>
        {copy.verification.hardwareIntro}
        <Ref k="hardware" />
      </p>
      <div className="table-scroll" tabIndex={0} role="region" aria-label="Hardware map">
        <table className="wikitable">
          <thead>
            <tr>
              <th scope="col">Hardware</th>
              <th scope="col">Segment</th>
              <th scope="col">Driver</th>
              <th scope="col">Software</th>
              <th scope="col">Result</th>
              <th scope="col">Note</th>
            </tr>
          </thead>
          <tbody>
            {hardwareMap.map((h, i) => (
              <tr key={i}>
                <td>{h.hardware}</td>
                <td>{h.segment}</td>
                <td>
                  <code>{h.driver}</code>
                </td>
                <td>
                  <code>{h.software}</code>
                </td>
                <td>
                  <Result r={h.result} />
                </td>
                <td>{h.note}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <h3>
        <span className="secnum">3.4</span> {copy.verification.checksTitle}
      </h3>
      <details>
        <summary>
          Show all {v.rawChecks.length} checks from <code>verifications.jsonl</code>
        </summary>
        <div className="table-scroll" tabIndex={0} role="region" aria-label="All recorded checks">
          <table className="wikitable">
            <thead>
              <tr>
                <th scope="col">Segment</th>
                <th scope="col">Result</th>
                <th scope="col">Checked by</th>
                <th scope="col">Checker</th>
                <th scope="col">GPU</th>
                <th scope="col">Driver</th>
                <th scope="col">Code</th>
              </tr>
            </thead>
            <tbody>
              {v.rawChecks.map((c, i) => (
                <tr key={i}>
                  <td className="num">{c.segment}</td>
                  <td>
                    <Result r={c.result} />
                  </td>
                  <td>{KNOWN_CHECKER_MAP[c.checker] ?? 'unknown'}</td>
                  <td>{c.checker}</td>
                  <td>{c.gpu}</td>
                  <td>
                    <code>{c.driver}</code>
                  </td>
                  <td>
                    <code>{c.code}</code>
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
