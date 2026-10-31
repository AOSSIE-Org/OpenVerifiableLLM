/** Read a dotted path ("odds.p95", "events.0.segment", "hardware.length") from a JSON value. */
export function getPath(root: unknown, path: string): unknown {
  let cur: unknown = root;
  for (const key of path.split('.')) {
    if (cur === null || cur === undefined) return undefined;
    if (Array.isArray(cur) && key === 'length') { cur = cur.length; continue; }
    if (typeof cur !== 'object') return undefined;
    cur = (cur as Record<string, unknown>)[key];
  }
  return cur;
}
