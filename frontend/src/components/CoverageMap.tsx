import { useState } from 'react';
import { useData } from '../data/context';
import { hasLabel } from '../content/ui';
import { SourceChip } from './SourceChip';
import { L, ResultChip, V, Who } from './text';
import { useHydrated } from './hydrated';
import { BASE } from '../routes';

const p = 'verifications';
/** Glyph from the tile's data status: ✓ only for MATCH, ✗ for MISMATCH, … for anything else. */
export const glyphKey = (r: string) => (r === 'MATCH' || r === 'MISMATCH' ? r : 'pending');

function CheckerName({ checker }: { checker: string }) {
  return hasLabel(`checker.${checker}`) ? <L k={`checker.${checker}`} /> : <span data-verbatim="checker">{checker}</span>;
}
export { CheckerName };

function Legend() {
  return (
    <ul className="legend">
      <li className="lg lg-project"><L k="legend.project" /></li>
      <li className="lg lg-independent"><L k="legend.independent" /></li>
      <li className="lg lg-restart"><L k="legend.restart" /></li>
      <li className="lg lg-code"><L k="legend.code" /></li>
    </ul>
  );
}

function TileInner({ k, result, who }: { k: number; result: string; who: string }) {
  return (
    <>
      <V path={`${p}.tiles.${k}.segment`} />
      <span aria-hidden="true" className="glyph" data-glyph={glyphKey(result)}>
        <L k={`tile.${glyphKey(result)}`} />
      </span>
      <span className="sr">
        {' '}
        <V path={`${p}.tiles.${k}.result`} /> <L k={`who.${who}`} />
      </span>
    </>
  );
}

/** 37 tiles. Interactive: links to each segment's rows before hydration, then real <button>s that reveal every check for a segment. Compact: a static list. */
export function CoverageMap({ interactive, strip = false, lanes = false }: { interactive: boolean; strip?: boolean; lanes?: boolean }) {
  const { data } = useData();
  const tiles = data.verifications.tiles;
  const [sel, setSel] = useState<number | null>(null);
  const hydrated = useHydrated();
  const selected = sel === null ? null : tiles[sel];
  const projectRow = (
    <ol className={`tiles ${interactive ? 'tiles-int' : ''} ${strip ? 'tiles-strip' : ''}`}>
      {tiles.map((t, k) => {
        const attrs = {
          className: 'tile',
          'data-result': t.result,
          'data-checker': t.checkedBy,
          'data-event': t.events.join(' ') || undefined,
        };
        const inner = <TileInner k={k} result={t.result} who={t.checkedBy} />;
        return (
          <li key={t.segment}>
            {interactive && hydrated ? (
              <button type="button" {...attrs} aria-pressed={sel === k} onClick={() => setSel(sel === k ? null : k)}>
                {inner}
              </button>
            ) : interactive ? (
              // Before JavaScript runs, each tile links to its segment's rows in the table below.
              <a {...attrs} href={`${BASE}verification/#seg-${t.segment}`}>{inner}</a>
            ) : (
              <span {...attrs}>{inner}</span>
            )}
          </li>
        );
      })}
    </ol>
  );
  const independentRow = lanes ? (
    <ol className="tiles tiles-indep">
      {tiles.map((t, k) => {
        const indep = t.checks.map((i) => data.verifications.rows[i]).find((r) => r?.checkedBy === 'independent');
        const result = indep?.result ?? 'NOT_RUN';
        const who = indep ? 'independent' : 'pending';
        return (
          <li key={`i-${t.segment}`}>
            <span className="tile" data-result={result} data-checker={who}>
              <V path={`${p}.tiles.${k}.segment`} />
              <span aria-hidden="true" className="glyph" data-glyph={glyphKey(result)}>
                <L k={`tile.${glyphKey(result)}`} />
              </span>
              <span className="sr">
                {' '}
                <L k={`who.${who}`} />
              </span>
            </span>
          </li>
        );
      })}
    </ol>
  ) : null;
  return (
    <div className="map">
      <Legend />
      {lanes ? (
        <div className="cov-lanes">
          <div className="cov-lane">
            <span className="cov-lab"><L k="map.row.project" /></span>
            {projectRow}
          </div>
          <div className="cov-lane">
            <span className="cov-lab"><L k="map.row.independent" /></span>
            {independentRow}
          </div>
        </div>
      ) : projectRow}
      {interactive ? (
        <div className="tile-detail" aria-live="polite">
          {selected ? (
            <>
              <h3>
                <L k="tile.checks" /> <V path={`${p}.tiles.${sel}.segment`} />
              </h3>
              <ul className="checks">
                {selected.checks.map((i) => (
                  <li key={i}>
                    <ResultChip path={`${p}.rows.${i}.result`} /> <CheckerName checker={data.verifications.rows[i]!.checker} />{' '}
                    <Who who={data.verifications.rows[i]!.checkedBy} /> <SourceChip s={data.verifications.rows[i]!.src} />
                  </li>
                ))}
              </ul>
            </>
          ) : (
            <p className="muted"><L k="tile.hint" /></p>
          )}
        </div>
      ) : null}
      {interactive ? (
        <details className="table-fallback">
          <summary><L k="table.show" /></summary>
          <div className="scroll">
            <table>
              <thead>
                <tr>
                  <th scope="col"><L k="th.segment" /></th>
                  <th scope="col"><L k="th.result" /></th>
                  <th scope="col"><L k="th.who" /></th>
                  <th scope="col"><L k="th.machine" /></th>
                </tr>
              </thead>
              <tbody>
                {tiles.map((t, k) => (
                  <tr key={k}>
                    <th scope="row"><V path={`${p}.tiles.${k}.segment`} /></th>
                    <td><ResultChip path={`${p}.tiles.${k}.result`} /></td>
                    <td>{t.checkedBy === 'pending' ? <L k="who.pending" /> : <Who who={t.checkedBy} />}</td>
                    <td>
                      {t.checks.map((i, j) => (
                        <span key={i} className="cell-item">
                          {j ? ' ' : ''}
                          <CheckerName checker={data.verifications.rows[i]!.checker} />
                        </span>
                      ))}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </details>
      ) : null}
    </div>
  );
}
