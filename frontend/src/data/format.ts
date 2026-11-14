/** The only place values become text. Tests import this to recompute every displayed number. */
export type Fmt = 'auto' | 'int' | 'pct2' | 'utc' | 'short' | 'raw' | 'list' | 'yesno';

const intFmt = new Intl.NumberFormat('en-US');
const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

export function formatValue(value: unknown, fmt: Fmt = 'auto'): string {
  if (value === undefined || value === null) throw new Error('missing value');
  switch (fmt) {
    case 'int': return intFmt.format(Number(value));
    case 'pct2': return `${(Number(value) * 100).toFixed(2)}%`;
    case 'short': return `${String(value).slice(0, 8)}…`;
    case 'list': return (value as unknown[]).map(String).join(', ');
    case 'yesno': return value === true ? 'yes' : value === false ? 'no' : String(value);
    case 'utc': {
      const d = new Date(String(value));
      if (Number.isNaN(d.getTime())) throw new Error(`bad date ${String(value)}`);
      const p = (n: number) => String(n).padStart(2, '0');
      return `${d.getUTCDate()} ${MONTHS[d.getUTCMonth()]} ${d.getUTCFullYear()}, ${p(d.getUTCHours())}:${p(d.getUTCMinutes())}:${p(d.getUTCSeconds())} UTC`;
    }
    case 'raw':
    case 'auto':
    default:
      return String(value);
  }
}
