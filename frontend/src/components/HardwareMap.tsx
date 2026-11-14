import { useData } from '../data/context';
import { SourceChip } from './SourceChip';
import { L, ResultChip, V } from './text';

/** Table on wide screens, one card per GPU/CPU on phones (CSS picks one; both come from the same rows). */
export function HardwareMap() {
  const { data } = useData();
  const h = data.hardware;
  return (
    <div className="hw">
      <div className="scroll hw-table">
        <table>
          <thead>
            <tr>
              <th scope="col"><L k="th.hardware" /></th>
              <th scope="col"><L k="th.segment" /></th>
              <th scope="col"><L k="th.result" /></th>
              <th scope="col"><L k="th.software" /></th>
              <th scope="col"><L k="th.driver" /></th>
              <th scope="col"><L k="th.note" /></th>
              <th scope="col"><L k="source" /></th>
            </tr>
          </thead>
          <tbody>
            {h.map((r, i) => (
              <tr key={i}>
                <th scope="row"><V path={`hardware.${i}.hardware`} /></th>
                <td><V path={`hardware.${i}.segment`} /></td>
                <td><ResultChip path={`hardware.${i}.result`} /></td>
                <td><V path={`hardware.${i}.software`} /></td>
                <td><V path={`hardware.${i}.driver`} /></td>
                <td><V path={`hardware.${i}.note`} /></td>
                <td><SourceChip s={r.src} /></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <ul className="hw-cards">
        {h.map((r, i) => (
          <li key={i} className="card" data-result={r.result}>
            <p className="card-title"><V path={`hardware.${i}.hardware`} /></p>
            <ResultChip path={`hardware.${i}.result`} />
            <dl>
              <dt><L k="th.segment" /></dt><dd><V path={`hardware.${i}.segment`} /></dd>
              <dt><L k="th.software" /></dt><dd><V path={`hardware.${i}.software`} /></dd>
              <dt><L k="th.driver" /></dt><dd><V path={`hardware.${i}.driver`} /></dd>
              <dt><L k="th.note" /></dt><dd><V path={`hardware.${i}.note`} /></dd>
            </dl>
            <SourceChip s={r.src} />
          </li>
        ))}
      </ul>
    </div>
  );
}
