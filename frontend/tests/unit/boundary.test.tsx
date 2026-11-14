import { afterEach, describe, expect, it, vi } from 'vitest';
import { cleanup, render, screen } from '@testing-library/react';
import { ErrorBoundary } from '../../src/components/Section';

afterEach(cleanup);

function Boom({ n }: { n: number }) {
  if (n === 1) throw new Error('boom');
  return <p>ok {n}</p>;
}

describe('page ErrorBoundary', () => {
  it('shows a new page after the route key changes', () => {
    vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    vi.spyOn(console, 'error').mockImplementation(() => undefined);
    const { rerender } = render(
      <ErrorBoundary key="check">
        <Boom n={1} />
      </ErrorBoundary>,
    );
    expect(screen.getByRole('alert')).toBeTruthy();
    rerender(
      <ErrorBoundary key="overview">
        <Boom n={0} />
      </ErrorBoundary>,
    );
    expect(screen.getByText('ok 0')).toBeTruthy();
  });
});
