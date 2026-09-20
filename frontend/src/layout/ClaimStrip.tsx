import type { Snapshot } from '../data/contracts';

export function ClaimStrip({ snapshot }: { snapshot: Snapshot }) {
  const revision = snapshot.source.revision ?? 'unspecified';
  return (
    <div className="claim">
      <div className="claim-inner">
        <div>
          <p className="k">What we can say today</p>
          <p>
            <strong>
              {snapshot.summary.mayClaim.join(' · ') || 'nothing beyond the snapshot'}
            </strong>
          </p>
        </div>
        <div>
          <p className="k">What we cannot say</p>
          <p>{snapshot.summary.mayNotClaim.join(' · ')}</p>
        </div>
        <p className="meta">
          Saved {snapshot.generatedAt} · {revision} · this website did not re-run the checks
        </p>
      </div>
    </div>
  );
}
