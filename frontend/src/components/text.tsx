import { formatValue, type Fmt } from '../data/format';
import { getPath } from '../data/path';
import { useData } from '../data/context';
import { label } from '../content/ui';

/** A value read from site-data.json. The only way a number reaches the page. */
export function V({ path, fmt = 'auto' }: { path: string; fmt?: Fmt }) {
  const { data } = useData();
  return <span className="v" data-path={path} data-fmt={fmt}>{formatValue(getPath(data, path), fmt)}</span>;
}
/** A UI label from ui.json. */
export function L({ k }: { k: string }) {
  return <span data-label={k}>{label(k)}</span>;
}
export function Who({ who }: { who: 'project' | 'independent' | 'pending' }) {
  return (
    <span className={`pill pill-${who}`} data-who={who}>
      <L k={`who.${who}`} />
    </span>
  );
}
export function ResultChip({ path }: { path: string }) {
  const { data } = useData();
  const r = String(getPath(data, path));
  return (
    <span className={`result result-${r}`} data-result={r}>
      <V path={path} />
    </span>
  );
}
