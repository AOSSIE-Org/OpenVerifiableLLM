import { validateSite, type Validated } from './validate';

export const DATA_URL = `${import.meta.env.BASE_URL}data/site-data.json`;

/** Fetch and validate. Any failure returns an error, never partial "success" data. */
export async function loadData(fetcher: typeof fetch = fetch): Promise<Validated | { error: string }> {
  try {
    const res = await fetcher(DATA_URL, { cache: 'no-cache' });
    if (!res.ok) return { error: `HTTP ${res.status}` };
    return validateSite(await res.json());
  } catch (e) {
    return { error: e instanceof Error ? e.message : String(e) };
  }
}
