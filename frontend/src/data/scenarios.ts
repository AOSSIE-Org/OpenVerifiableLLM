import defaultSnapshot from '../../public/data/snapshot.json';

// Imported only through the development branch. None is approved public metadata.
export function scenario(name: string): unknown {
  const s = structuredClone(defaultSnapshot);
  switch (name) {
    case 'empty': s.checks = []; break;
    case 'no-evidence': s.evidence = []; s.checks = []; break;
    case 'malformed': return { ...s, checks: null };
    case 'unsupported-schema': return { ...s, schemaVersion: 999 };
    case 'missing-evidence': s.evidence = s.evidence.filter(e => e.id !== 'synthetic-pilot-replay'); break;
    case 'stale': s.generatedAt = '2020-01-01T00:00:00Z'; break;
    case 'fetch-error': throw new Error('Simulated snapshot fetch failure.');
    case 'public-unapproved': return { ...s, mode: 'public-snapshot' };
    case 'future-release': {
      const identity = {
        repository: 'synthetic-example/not-a-project-release', revision: 'a'.repeat(40), root: 'b'.repeat(64),
        files: [{ path: 'model.safetensors', sizeBytes: 1024, sha256: 'c'.repeat(64) }],
        inventoryUrl: 'https://example.org/synthetic/inventory.json', licenseUrl: 'https://example.org/synthetic/license',
        runtime: 'Synthetic runtime — compatibility not established', modelCardUrl: 'https://example.org/synthetic/model-card',
      };
      return { ...s, releases: [
        { ...s.releases[0], id: 'synthetic-base', availability: 'available', identity, checkIds: [s.checks[0].id] },
        { ...s.releases[1], id: 'synthetic-chat', availability: 'available', identity: { ...identity, root: 'd'.repeat(64) }, parentReleaseId: 'synthetic-base', checkIds: [] },
      ] };
    }
  }
  return s;
}
