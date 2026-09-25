import type { CheckResult } from '../data/contracts';

const LABELS: Record<CheckResult, string> = {
  PASS: 'PASS',
  FAIL: 'FAIL',
  NOT_RUN: 'NOT RUN',
  UNAVAILABLE: 'UNAVAILABLE',
  UNSUPPORTED: 'UNSUPPORTED',
};

export function StatusMark({ result }: { result: CheckResult }) {
  const tone =
    result === 'PASS' ? 'pass' : result === 'FAIL' ? 'fail' : 'hold';
  return (
    <span className={`mark ${tone}`}>
      <i aria-hidden="true" />
      {LABELS[result]}
    </span>
  );
}

export function pipelineClass(result: CheckResult): string {
  if (result === 'PASS') return 'pass';
  if (result === 'FAIL') return 'fail';
  return 'hold';
}
