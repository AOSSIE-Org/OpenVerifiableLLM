import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { render, screen, fireEvent, act, within } from '@testing-library/react';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { Header } from '../src/components/Header';
import { Contents } from '../src/components/Contents';
import { CopyButton } from '../src/components/CopyButton';
import { Verification, detectionOdds } from '../src/components/Verification';
import { References } from '../src/components/References';
import { validateEvidence } from '../src/data/validate';

const evidence = validateEvidence(
  JSON.parse(readFileSync(resolve(process.cwd(), 'public/data/evidence.json'), 'utf8')),
);

describe('Header and contents', () => {
  it('renders the site name and external links', () => {
    render(<Header />);
    expect(screen.getByText('OpenVerifiableLLM')).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'GitHub' })).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'Evidence' })).toBeInTheDocument();
  });

  it('lists every section in order', () => {
    render(<Contents />);
    const nav = screen.getByRole('navigation', { name: 'Contents' });
    const labels = within(nav).getAllByRole('link').map(a => a.textContent);
    expect(labels).toEqual([
      '(Top)',
      '1 Overview',
      '2 Check it yourself',
      '3 Verification',
      '4 Training timeline',
      '5 Verifiable exclusion',
      '6 Downloads',
      '7 Terms used on this page',
      '8 References',
    ]);
  });
});

describe('Verification', () => {
  it('computes the detection odds from the brief', () => {
    expect(detectionOdds(37, 0)).toBe(0);
    expect(detectionOdds(37, 110)).toBeGreaterThanOrEqual(0.95);
    expect(detectionOdds(37, 109)).toBeLessThan(0.95);
    expect(detectionOdds(37, 169)).toBeGreaterThanOrEqual(0.99);
    expect(detectionOdds(37, 168)).toBeLessThan(0.99);
  });

  it('shows 37 project tiles and no independent results', () => {
    render(<Verification verifications={evidence.verifications} hardwareMap={evidence.hardwareMap} />);
    const project = screen.getByRole('list', { name: /checked by the project/i });
    const independent = screen.getByRole('list', { name: /checked independently/i });
    expect(within(project).getAllByRole('listitem')).toHaveLength(37);
    expect(within(project).getAllByText(/: MATCH/)).toHaveLength(37);
    expect(within(independent).getAllByText(/: not checked/)).toHaveLength(37);
    expect(screen.getByText(/self-reported and unsigned/)).toBeInTheDocument();
  });
});

describe('References', () => {
  it('lists each evidence file with its SHA-256 at the pinned revision', () => {
    render(<References evidence={evidence} />);
    for (const [path, sha] of Object.entries(evidence.sourceHashes)) {
      expect(screen.getByText(path)).toBeInTheDocument();
      expect(screen.getByText(sha)).toBeInTheDocument();
    }
    const link = screen.getByText('release/simple-run/chain.jsonl').closest('a');
    expect(link?.getAttribute('href')).toContain(`/blob/${evidence.revision}/`);
  });
});

describe('CopyButton', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    Object.assign(navigator, { clipboard: { writeText: vi.fn().mockResolvedValue(undefined) } });
  });
  afterEach(() => vi.useRealTimers());

  it('copies text and provides feedback', async () => {
    render(<CopyButton textToCopy="test command" />);
    const btn = screen.getByRole('button', { name: 'Copy' });
    await act(async () => {
      fireEvent.click(btn);
    });
    expect(navigator.clipboard.writeText).toHaveBeenCalledWith('test command');
    expect(btn).toHaveTextContent('Copied');
    act(() => {
      vi.advanceTimersByTime(2500);
    });
    expect(btn).toHaveTextContent('Copy');
  });
});
