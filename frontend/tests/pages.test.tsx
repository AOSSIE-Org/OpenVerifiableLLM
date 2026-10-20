import { render, screen, fireEvent } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import fixture from '../public/data/snapshot.json';
import { validateSnapshot } from '../src/data/contracts';
import { scenario } from '../src/data/scenarios';
import { EvidenceDetail, EvidenceExplorer, InferenceUnavailable, Overview, Releases } from '../src/pages';
import { Copy } from '../src/components';
const s = validateSnapshot(fixture);
describe('claim semantics and navigation', () => {
  it('empty checks never render a PASS on overview or explorer', () => {
    const empty = validateSnapshot(scenario('empty'));
    const { unmount } = render(<Overview snapshot={empty} />);
    expect(screen.getByRole('heading', { name: 'No checks reported', level: 2 })).toBeInTheDocument();
    expect(document.querySelector('.result-PASS')).toBeNull(); unmount();
    render(<EvidenceExplorer snapshot={empty} params={new URLSearchParams()} />);
    expect(screen.getByRole('heading', { name: 'No checks reported', level: 2 })).toBeInTheDocument();
    expect(document.querySelector('.result-PASS')).toBeNull();
  });
  it('filters phase, scope, kind and result together', () => {
    render(<EvidenceExplorer snapshot={s} params={new URLSearchParams('scope=pilot&result=PASS&kind=report&phase=Training+%2F+replay')} />);
    expect(screen.getByRole('link', { name: 'Sampled replay report' })).toHaveAttribute('href', '#/evidence/synthetic-pilot-replay');
    expect(screen.queryByRole('link', { name: /Artifact integrity check/ })).not.toBeInTheDocument();
  });
  it('writes filter state to static-safe URL', () => {
    render(<EvidenceExplorer snapshot={s} params={new URLSearchParams('q=replay')} />);
    fireEvent.change(screen.getByLabelText('Scope'), { target: { value: 'pilot' } });
    expect(window.location.hash).toBe('#/evidence?q=replay&scope=pilot');
  });
  it('renders a useful empty filtered result', () => {
    render(<EvidenceExplorer snapshot={s} params={new URLSearchParams('q=nonexistent')} />);
    expect(screen.getByText('No matching evidence')).toBeInTheDocument();
  });
  it('has functional parent and superseded-report links', () => {
    render(<EvidenceDetail snapshot={s} id="synthetic-pilot-replay" />);
    expect(screen.getByRole('link', { name: /Earlier sampled replay report/ })).toHaveAttribute('href', '#/evidence/synthetic-replay-old');
    expect(screen.getByRole('link', { name: /synthetic-parent-missing/ })).toHaveAttribute('href', '#/evidence/synthetic-parent-missing');
    expect(screen.getByText('No — this frontend displays reports.')).toBeInTheDocument();
  });
  it('missing evidence cannot become successful verification', () => {
    render(<EvidenceDetail snapshot={s} id="missing" />);
    expect(screen.getByText('Missing evidence reference')).toBeInTheDocument();
    expect(screen.queryByText('PASS')).not.toBeInTheDocument();
  });
  it('keeps original failed reports visible with replacement navigation', () => {
    render(<EvidenceDetail snapshot={s} id="synthetic-replay-old" />);
    expect(screen.getByText('FAIL')).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /Sampled replay report/ })).toHaveAttribute('href', '#/evidence/synthetic-pilot-replay');
  });
  it('unreleased models have no usable download action', () => {
    render(<Releases snapshot={s} />);
    expect(screen.getAllByText('Not released yet.')).toHaveLength(2);
    expect(screen.getAllByRole('button', { name: 'Download unavailable' }).every(b => b.hasAttribute('disabled'))).toBe(true);
    expect(screen.queryByRole('link', { name: /Download/ })).not.toBeInTheDocument();
  });
  it('even future-release fixtures do not enable downloads', () => {
    render(<Releases snapshot={validateSnapshot(scenario('future-release'))} />);
    expect(screen.getAllByRole('button', { name: /Download unavailable/ }).every(b => b.hasAttribute('disabled'))).toBe(true);
  });
  it('public inference never promises a released model', () => {
    render(<InferenceUnavailable />);
    expect(screen.getByText('Generation is unavailable until a verified model release and supported inference backend are connected.')).toBeInTheDocument();
    expect(screen.queryByRole('textbox')).not.toBeInTheDocument();
  });
  it('copies full digest and announces success', async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, 'clipboard', { value: { writeText }, configurable: true });
    render(<Copy value={'0123456789abcdef'.repeat(4)} />);
    fireEvent.click(screen.getByRole('button'));
    expect(await screen.findByText('Copied to clipboard.')).toBeInTheDocument();
    expect(writeText).toHaveBeenCalledWith('0123456789abcdef'.repeat(4));
  });
  it('announces clipboard failure without claiming copy succeeded', async () => {
    Object.defineProperty(navigator, 'clipboard', { value: { writeText: vi.fn().mockRejectedValue(new Error()) }, configurable: true });
    render(<Copy value="digest" />);
    fireEvent.click(screen.getByRole('button'));
    expect(await screen.findByText(/Copy unavailable/)).toBeInTheDocument();
  });
  it('escapes technical metadata and titles', () => {
    const data = structuredClone(fixture); data.evidence[0].title = '<img src=x onerror=alert(1)>';
    render(<EvidenceDetail snapshot={validateSnapshot(data)} id="synthetic-pilot-replay" />);
    expect(screen.getAllByText('<img src=x onerror=alert(1)>').length).toBeGreaterThan(0);
    expect(document.querySelector('img')).toBeNull();
  });
});
