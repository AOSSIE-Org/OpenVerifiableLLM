import {
  createContext,
  useContext,
  useMemo,
  type ReactNode,
} from 'react';
import { useSearchParams } from 'react-router-dom';
import {
  loadSnapshot,
  parseSnapshotId,
  type SnapshotId,
} from './data/adapters/snapshotAdapter';
import { SnapshotError, type Snapshot } from './data/contracts';

interface SnapshotState {
  snapshot: Snapshot | null;
  error: SnapshotError | null;
  id: SnapshotId;
}

const Ctx = createContext<SnapshotState | null>(null);

export function SnapshotProvider({ children }: { children: ReactNode }) {
  const [params] = useSearchParams();
  const id = parseSnapshotId(params.get('snapshot'));
  const value = useMemo(() => {
    try {
      return { snapshot: loadSnapshot(id), error: null, id };
    } catch (err) {
      const error =
        err instanceof SnapshotError
          ? err
          : new SnapshotError(err instanceof Error ? err.message : 'Invalid snapshot');
      return { snapshot: null, error, id };
    }
  }, [id]);

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useSnapshotState(): SnapshotState {
  const ctx = useContext(Ctx);
  if (!ctx) throw new Error('SnapshotProvider missing');
  return ctx;
}

export function useSnapshot(): Snapshot {
  const { snapshot, error } = useSnapshotState();
  if (error || !snapshot) {
    throw error ?? new SnapshotError('Snapshot unavailable');
  }
  return snapshot;
}
