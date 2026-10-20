import { validateSnapshot, type Snapshot } from './contracts';
export const MAX_SNAPSHOT_BYTES = 1_000_000;

// Only a configured same-origin display snapshot is fetched. Never evidence payloads.
export function snapshotUrl(base: string, path: string, pageUrl: string): URL {
  const origin = new URL(pageUrl);
  const root = new URL(base.endsWith('/') ? base : base + '/', origin);
  const target = new URL(path, root);
  if (target.origin !== origin.origin || !target.pathname.startsWith(root.pathname) || !/^https?:$/.test(target.protocol))
    throw new Error('Snapshot path must stay inside the deployment directory on the same origin.');
  return target;
}
export async function loadSnapshot(url: URL, mode: Snapshot['mode'], signal: AbortSignal): Promise<Snapshot> {
  const response = await fetch(url, { signal, credentials: 'omit' });
  if (!response.ok) throw new Error('Snapshot could not be loaded (HTTP ' + response.status + ').');
  if (!response.body) throw new Error('Snapshot response has no body.');
  const reader = response.body.getReader();
  const parts: Uint8Array[] = []; let size = 0;
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      size += value.byteLength;
      if (size > MAX_SNAPSHOT_BYTES) throw new Error('Display snapshot exceeds the 1 MB limit.');
      parts.push(value);
    }
  } finally { await reader.cancel(); }
  const bytes = new Uint8Array(size); let offset = 0;
  for (const p of parts) { bytes.set(p, offset); offset += p.length; }
  let data: unknown;
  try { data = JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(bytes)); }
  catch { throw new Error('Snapshot is not valid UTF-8 JSON.'); }
  return validateSnapshot(data, mode);
}
export function isStale(s: Snapshot, now = Date.now()) { return now - Date.parse(s.generatedAt) > 7 * 86400000; }
