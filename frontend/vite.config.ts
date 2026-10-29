import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import type { Plugin } from 'vite';

// The browser only needs the fields it renders. claims.json stays the single source; this trims the
// bundled copy (rules, notes, values other than hashes) at build time without editing the file.
const KEEP = ['id', 'page', 'status', 'plain', 'developer', 'limit', 'verbatim', 'sources', 'value', 'rule'];
function slimLedger(): Plugin {
  return {
    name: 'slim-ledger',
    enforce: 'pre',
    transform(code, id) {
      if (!id.endsWith('/src/content/claims.json')) return null;
      const l = JSON.parse(code) as { claims: Record<string, unknown>[] } & Record<string, unknown>;
      const claims = l.claims.map((c) => {
        const o = Object.fromEntries(KEEP.filter((k) => k in c).map((k) => [k, c[k]]));
        if (!(typeof o.value === 'string' && /^[0-9a-f]{64}$/.test(o.value))) o.value = null;
        o.rule = /checked by the project/i.test(String(c.rule ?? '')) ? 'checked by the project' : null;
        o.sources = (c.sources as { rev: string; file: string; lines: string }[]).map(({ rev, file, lines }) => ({ rev, file, lines }));
        return o;
      });
      return { code: JSON.stringify({ schema: l.schema, pins: l.pins, claims, forbidden_phrases: l.forbidden_phrases }), map: null };
    },
  };
}

// BASE_PATH lets the same build run at / or under a sub-path such as /OpenVerifiableLLM/.
const base = process.env.BASE_PATH ?? '/';

export default defineConfig({
  base,
  plugins: [slimLedger(), react()],
  build: { target: 'es2022', cssCodeSplit: false, reportCompressedSize: true, sourcemap: false },
});
