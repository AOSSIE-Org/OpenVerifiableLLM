import { useData } from '../data/context';
import { detect } from '../data/validate';
import { L, V } from './text';

/** Detection probability 1-((N-1)/N)^M as an inline SVG. Every tick comes from site-data.json.
 *  `atN` shows the M = N row (pAtN), which is computed and not a ledger number; set it false to drop that row. */
export function OddsBar({ atN = true }: { atN?: boolean }) {
  const { data } = useData();
  const o = data.odds;
  const maxM = Math.ceil(o.m99 * 1.15);
  const W = 600, H = 160, pad = 28;
  const x = (m: number) => pad + ((W - 2 * pad) * m) / maxM;
  const y = (p: number) => H - pad - (H - 2 * pad) * p;
  const pts = Array.from({ length: maxM + 1 }, (_, m) => `${x(m).toFixed(1)},${y(detect(o.N, m)).toFixed(1)}`).join(' ');
  const ticks = [
    { m: o.currentM, path: 'odds.currentM', now: true },
    ...(atN ? [{ m: o.N, path: 'odds.N', now: false }] : []),
    { m: o.m95, path: 'odds.m95', now: false },
    { m: o.m99, path: 'odds.m99', now: false },
  ];
  const cap = (
    <figcaption id="odds-cap">
      <ul className="odds-list">
        <li className="now">
          <L k="odds.now" />: <V path="odds.currentM" /> <L k="odds.checks" /> <L k="odds.gives" /> <V path="odds.currentP" fmt="pct2" />
        </li>
        {atN ? (
          <li>
            <V path="odds.N" /> <L k="odds.checks" /> <L k="odds.wouldGive" /> <V path="odds.pAtN" fmt="pct2" />
          </li>
        ) : null}
        <li>
          <V path="odds.m95" /> <L k="odds.checks" /> <L k="odds.wouldGive" /> <V path="odds.p95" fmt="pct2" />
        </li>
        <li>
          <V path="odds.m99" /> <L k="odds.checks" /> <L k="odds.wouldGive" /> <V path="odds.p99" fmt="pct2" />
        </li>
      </ul>
      <code><V path="odds.formula" /></code>
    </figcaption>
  );
  return (
    <figure className="odds">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-labelledby="odds-cap" preserveAspectRatio="xMidYMid meet">
        <line x1={pad} x2={W - pad} y1={H - pad} y2={H - pad} className="axis" />
        <polyline points={pts} className="curve" fill="none" />
        {ticks.map((t) => (
          <g key={t.path} className={t.now ? 'tick now' : 'tick'}>
            <line x1={x(t.m)} x2={x(t.m)} y1={H - pad} y2={y(detect(o.N, t.m))} />
            <circle cx={x(t.m)} cy={y(detect(o.N, t.m))} r={t.now ? 5 : 3.5} />
          </g>
        ))}
      </svg>
      {cap}
    </figure>
  );
}
