import { useData } from '../data/context';
import { CheckerName } from './CoverageMap';
import { SourceChip } from './SourceChip';
import { L, ResultChip, V, Who } from './text';

/** Every row of a verifications.jsonl, grouped by the computer that ran it. */
export function RowGroups({ base }: { base: 'verifications' | 'exclusion' }) {
  const { data } = useData();
  const block = base === 'verifications' ? data.verifications : data.exclusion;
  // The first row of each segment carries the seg-N anchor the coverage tiles link to.
  const firstRow = new Map<number, number>();
  if (base === 'verifications') block.rows.forEach((r, i) => { if (!firstRow.has(r.segment)) firstRow.set(r.segment, i); });
  return (
    <div className="groups">
      {block.groups.map((g, gi) => (
        <section key={g.checker} className="group" aria-labelledby={`${base}-g${gi}`}>
          <h3 id={`${base}-g${gi}`}>
            <CheckerName checker={g.checker} /> <span className="count"><V path={`${base}.groups.${gi}.rows`} /></span>
          </h3>
          <Who who={g.checkedBy} />
          <div className="scroll">
            <table>
              <thead>
                <tr>
                  <th scope="col"><L k="th.segment" /></th>
                  <th scope="col"><L k="th.result" /></th>
                  {base === 'verifications' ? <th scope="col"><L k="th.code" /></th> : null}
                  <th scope="col"><L k="source" /></th>
                </tr>
              </thead>
              <tbody>
                {block.rows.map((r, i) =>
                  r.checker !== g.checker ? null : (
                    <tr key={i} {...(firstRow.get(r.segment) === i ? { id: `seg-${r.segment}` } : {})}>
                      <th scope="row"><V path={`${base}.rows.${i}.segment`} /></th>
                      <td><ResultChip path={`${base}.rows.${i}.result`} /></td>
                      {base === 'verifications' ? <td><V path={`${base}.rows.${i}.code`} /></td> : null}
                      <td><SourceChip s={r.src} /></td>
                    </tr>
                  ),
                )}
              </tbody>
            </table>
          </div>
        </section>
      ))}
    </div>
  );
}
