import { SnapshotError, type Snapshot } from '../contracts';
import { validateSnapshot } from '../validate';
import emptyChecks from '../fixtures/empty-checks.json';
import missingParent from '../fixtures/missing-parent.json';
import publicSnapshot from '../fixtures/public-snapshot.json';
import superseded from '../fixtures/superseded.json';

export const SNAPSHOT_IDS = ['public', 'missing-parent', 'superseded', 'empty'] as const;
export type SnapshotId = (typeof SNAPSHOT_IDS)[number];

const RAW: Record<SnapshotId, unknown> = {
  public: publicSnapshot,
  'missing-parent': missingParent,
  superseded,
  empty: emptyChecks,
};

export function loadSnapshot(id: SnapshotId = 'public'): Snapshot {
  const raw = RAW[id];
  if (raw === undefined) {
    throw new SnapshotError(`Unknown snapshot ${id}`);
  }
  return validateSnapshot(raw);
}

export function parseSnapshotId(value: string | null): SnapshotId {
  if (value && (SNAPSHOT_IDS as readonly string[]).includes(value)) {
    return value as SnapshotId;
  }
  return 'public';
}
