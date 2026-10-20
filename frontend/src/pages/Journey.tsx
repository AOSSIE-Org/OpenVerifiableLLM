import { useState } from 'react';
import { JOURNEY, SCOPE_LABEL, worstResult } from '../content/copy';
import type { CheckResult, PipelineStep, Snapshot } from '../data/contracts';
import { StatusMark } from '../layout/StatusMark';

function stageResult(snapshot: Snapshot, pipelineIds: string[], checkProfile: string | null): CheckResult {
  const fromSteps = pipelineIds.map(
    (id) => snapshot.pipeline.find((step) => step.id === id)?.result ?? 'NOT_RUN',
  );
  const fromCheck = checkProfile
    ? snapshot.checks.filter((c) => c.profile === checkProfile).map((c) => c.result)
    : [];
  return worstResult([...fromSteps, ...fromCheck]);
}

function stepsFor(snapshot: Snapshot, ids: string[]): PipelineStep[] {
  return ids
    .map((id) => snapshot.pipeline.find((step) => step.id === id))
    .filter((step): step is PipelineStep => Boolean(step));
}

const KIND_LINE = {
  evidence: 'Project evidence',
  verification: 'Verification',
  inference: 'Inference',
} as const;

export function Journey({ snapshot }: { snapshot: Snapshot }) {
  const [openId, setOpenId] = useState<string | null>(null);

  return (
    <ol className="journey">
      {JOURNEY.map((stage) => {
        const result = stageResult(snapshot, stage.pipelineIds, stage.checkProfile);
        const steps = stepsFor(snapshot, stage.pipelineIds);
        const isOpen = openId === stage.id;
        return (
          <li key={stage.id} className={`journey-card kind-${stage.kind}`}>
            <p className="journey-label">{stage.label}</p>
            <StatusMark result={result} />
            <p className="journey-what">{stage.what}</p>
            <details open={isOpen}>
              <summary
                onClick={(event) => {
                  event.preventDefault();
                  setOpenId((current) => (current === stage.id ? null : stage.id));
                }}
              >
                {isOpen ? 'Hide steps' : 'Steps'}
              </summary>
              {isOpen ? (
                <>
                  <p className="step-meta">{stage.why}</p>
                  <p className="step-meta">{KIND_LINE[stage.kind]}</p>
                  {steps.length === 0 && stage.checkProfile ? (
                    <p className="step-meta">Check: {stage.checkProfile}</p>
                  ) : null}
                  <ul className="journey-steps">
                    {steps.map((step) => (
                      <li key={step.id}>
                        {step.title} — <StatusMark result={step.result} /> · {SCOPE_LABEL[step.scope]}
                      </li>
                    ))}
                  </ul>
                </>
              ) : null}
            </details>
          </li>
        );
      })}
    </ol>
  );
}
