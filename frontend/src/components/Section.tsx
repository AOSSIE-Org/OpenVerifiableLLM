import { Component, type ErrorInfo, type ReactNode } from 'react';
import { useData } from '../data/context';
import type { SectionId } from '../data/validate';
import { L } from './text';

function Fallback() {
  return (
    <div className="fallback" role="alert" data-fallback="">
      <L k="error.section" />
    </div>
  );
}

class Boundary extends Component<{ children: ReactNode }, { failed: boolean }> {
  override state = { failed: false };
  static getDerivedStateFromError() {
    return { failed: true };
  }
  override componentDidCatch(error: Error, info: ErrorInfo) {
    console.warn('section failed to render', error.message, info.componentStack?.split('\n')[1]);
  }
  override render() {
    return this.state.failed ? <Fallback /> : this.props.children;
  }
}

/** One piece of evidence. If its data failed validation, or it throws while rendering,
 *  only this piece shows "We couldn't load this evidence." and the rest of the page stays. */
export function Section({ needs, children }: { needs: SectionId[]; children: ReactNode }) {
  const { errors } = useData();
  if (needs.some((n) => errors[n])) return <Fallback />;
  // data-section marks evidence that depends on site-data.json, so a failed load can swap just these parts.
  return (
    <div className="section" data-section={needs.join(' ')}>
      <Boundary>{children}</Boundary>
    </div>
  );
}
export { Boundary as ErrorBoundary };
