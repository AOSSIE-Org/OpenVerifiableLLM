import { SourceChip } from './SourceChip';
import { useData } from '../data/context';
import { L, V } from './text';

export function LossTable() {
  const { data } = useData();
  const rows = [
    { k: 'loss.removed', p: 'exclusion.removedLoss' },
    { k: 'loss.control', p: 'exclusion.controlLoss' },
  ];
  return (
    <div className="scroll">
      <table className="loss">
        <thead>
          <tr>
            <th scope="col"><L k="th.measure" /></th>
            <th scope="col"><L k="th.before" /></th>
            <th scope="col"><L k="th.after" /></th>
            <th scope="col"><L k="th.delta" /></th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.k}>
              <th scope="row"><L k={r.k} /></th>
              <td className="num"><V path={`${r.p}.before`} /></td>
              <td className="num"><V path={`${r.p}.after`} /></td>
              <td className="num"><V path={`${r.p}.delta`} /></td>
            </tr>
          ))}
        </tbody>
      </table>
      <SourceChip s={data.exclusion.src} />
    </div>
  );
}
