import { validateSite, type Validated } from './validate';

export const DATA_URL = `${import.meta.env.BASE_URL}data/site-data.json`;

/** sha256 of the data the prerendered HTML was built from (set by scripts/prerender.mjs), or null. */
export function builtDataHash(doc: Document | undefined = typeof document === 'undefined' ? undefined : document): string | null {
  const v = doc?.querySelector('meta[name="ovl-data"]')?.getAttribute('content') ?? '';
  return /^[0-9a-f]{64}$/.test(v) ? v : null;
}
/** The content-hashed copy for prerendered pages; the plain file otherwise (dev, client-only). */
export const dataUrlFor = (hash: string | null) => (hash ? `${import.meta.env.BASE_URL}data/site-data.${hash.slice(0, 16)}.json` : DATA_URL);

async function sha256(text: string): Promise<string | null> {
  const subtle = globalThis.crypto?.subtle;
  if (!subtle) return null; // not a secure context: the hashed URL still pins the file
  const buf = await subtle.digest('SHA-256', new TextEncoder().encode(text));
  return [...new Uint8Array(buf)].map((b) => b.toString(16).padStart(2, '0')).join('');
}

/** Fetch and validate. Any failure returns an error, never partial "success" data. When the page was
 *  prerendered, data whose sha256 differs from the build's is an error too, so it never hydrates. */
export async function loadData(fetcher: typeof fetch = fetch, hash: string | null = builtDataHash()): Promise<Validated | { error: string }> {
  try {
    const res = await fetcher(dataUrlFor(hash), hash ? {} : { cache: 'no-cache' });
    if (!res.ok) return { error: `HTTP ${res.status}` };
    const text = await res.text();
    if (hash) {
      const got = await sha256(text);
      if (got !== null && got !== hash) return { error: 'data does not match the build' };
    }
    return validateSite(JSON.parse(text));
  } catch (e) {
    return { error: e instanceof Error ? e.message : String(e) };
  }
}
