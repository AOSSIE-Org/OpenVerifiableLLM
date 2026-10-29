import { useData } from '../data/context';
import { L, V } from './text';

/** One cell per segment, width proportional to its recorded wall time; markings come from data.events. */
export function Timeline() {
  const { data } = useData();
  const B = data.chain.boundaries;
  const tiles = data.verifications.tiles;
  return (
    <figure className="timeline">
      <ol className="strip">
        {tiles.map((t, k) => (
          <li
            key={k}
            className="seg"
            data-phase={t.phase}
            data-event={t.events.join(' ') || undefined}
            style={{ flexGrow: Math.max(1, Math.round(B[k + 1]!.time - B[k]!.time)) }}
          >
            <span className="sr">
              <L k="th.segment" /> <V path={`verifications.tiles.${k}.segment`} /> <L k={`phase.${t.phase}`} />
              {t.events.map((e) => (
                <span key={e}> <L k={`event.${e}`} /></span>
              ))}
            </span>
          </li>
        ))}
      </ol>
      <figcaption>
        <ul className="legend">
          {data.chain.phases.map((ph, i) => (
            <li key={ph.phase} className={`lg lg-phase-${ph.phase}`}>
              <L k={`phase.${ph.phase}`} />: <V path={`chain.phases.${i}.first`} />–<V path={`chain.phases.${i}.last`} />
            </li>
          ))}
          {data.events.map((e, i) => (
            <li key={e.id} className={`lg lg-${e.id === 'restart' ? 'restart' : 'code'}`}>
              <L k={`event.${e.id}`} />: <V path={`events.${i}.segment`} />
            </li>
          ))}
        </ul>
        <p className="muted">
          <V path="chain.start" fmt="utc" /> – <V path="chain.end" fmt="utc" />
        </p>
      </figcaption>
    </figure>
  );
}
