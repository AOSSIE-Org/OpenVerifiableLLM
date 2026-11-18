import React from 'react';

// The message shown whenever the evidence can't be shown, so the page never goes blank.
export const SnapshotProblem: React.FC<{ message: string }> = ({ message }) => (
  <main className="page-msg" role="alert">
    <h1>Snapshot unavailable or invalid</h1>
    <p className="err">{message}</p>
    <button type="button" className="btn" onClick={() => window.location.reload()}>
      Retry loading snapshot
    </button>
  </main>
);

interface BoundaryState {
  message: string | null;
}

// The validator should stop bad evidence first. This catches anything it missed while rendering.
export class ErrorBoundary extends React.Component<{ children: React.ReactNode }, BoundaryState> {
  state: BoundaryState = { message: null };

  static getDerivedStateFromError(error: unknown): BoundaryState {
    return { message: error instanceof Error ? error.message : String(error) };
  }

  render() {
    return this.state.message === null ? this.props.children : <SnapshotProblem message={this.state.message} />;
  }
}
