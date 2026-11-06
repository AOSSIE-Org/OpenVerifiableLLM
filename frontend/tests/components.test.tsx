import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { render, screen, fireEvent, act, within } from '@testing-library/react';
import { Header } from '../src/components/Header';
import { Hero } from '../src/components/Hero';
import { CopyButton } from '../src/components/CopyButton';
import { Outcome } from '../src/components/Outcome';
import { Results } from '../src/components/Results';
import { TrainingHistory } from '../src/components/TrainingHistory';
import { deriveCoverage, summarize } from '../src/data/facts';
import type { EvidenceData } from '../src/data/contracts';
import { loadEvidence, withMismatch, withoutChecks } from './evidence-cases';

const PINNED = 'https://huggingface.co/datasets/AOSSIE/openverifiable-enwiki-20260901-20260918-r1-evidence/tree/96dc282a';

const renderResults = (evidence: EvidenceData) =>
  render(
    <Results
      facts={summarize(evidence)}
      coverage={evidence.verifications.coverage}
      hardwareMap={evidence.hardwareMap}
      preparation={evidence.preparation}
    />,
  );

const renderHistory = (evidence: EvidenceData) =>
  render(
    <TrainingHistory
      facts={summarize(evidence)}
      chain={evidence.chain}
      coverage={evidence.verifications.coverage}
      exclusion={evidence.exclusion}
    />,
  );

const projectCells = (container: HTMLElement) => [...container.querySelectorAll('.covgrid .cells')][0].children;
const independentCells = (container: HTMLElement) => [...container.querySelectorAll('.covgrid .cells')][1].children;

describe('Header component', () => {
  beforeEach(() => {
    localStorage.clear();
    document.documentElement.removeAttribute('data-theme');
  });

  afterEach(() => {
    document.documentElement.removeAttribute('data-theme');
  });

  it('renders branding and section links', () => {
    render(<Header evidenceHref={PINNED} />);
    expect(screen.getByText('OpenVerifiable')).toBeInTheDocument();
    expect(screen.getByText('LLM')).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'Source' })).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'Evidence' })).toHaveAttribute('href', PINNED);
  });

  it('starts light and toggles to dark', () => {
    render(<Header evidenceHref={PINNED} />);
    const toDark = screen.getByRole('button', { name: 'Switch to dark theme' });
    expect(document.documentElement.getAttribute('data-theme')).toBeNull();

    fireEvent.click(toDark);
    expect(document.documentElement.getAttribute('data-theme')).toBe('dark');
    expect(localStorage.getItem('ovl-theme')).toBe('dark');

    fireEvent.click(screen.getByRole('button', { name: 'Switch to light theme' }));
    expect(document.documentElement.getAttribute('data-theme')).toBeNull();
  });
});

describe('CopyButton component', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    Object.assign(navigator, {
      clipboard: {
        writeText: vi.fn().mockResolvedValue(undefined),
      },
    });
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('copies text and provides feedback', async () => {
    render(<CopyButton textToCopy="test command" label="Copy" />);
    const btn = screen.getByRole('button', { name: /copy/i });
    expect(btn).toHaveTextContent('Copy');

    await act(async () => {
      fireEvent.click(btn);
    });

    expect(navigator.clipboard.writeText).toHaveBeenCalledWith('test command');
    expect(btn).toHaveTextContent('Copied ✓');

    act(() => {
      vi.advanceTimersByTime(2500);
    });

    expect(btn).toHaveTextContent('Copy');
  });

  it('says so when the copy fails, rather than doing nothing', async () => {
    Object.assign(navigator, { clipboard: { writeText: vi.fn().mockRejectedValue(new Error('denied')) } });
    render(<CopyButton textToCopy="x" />);
    await act(async () => {
      fireEvent.click(screen.getByRole('button'));
    });
    expect(screen.getByRole('button')).toHaveTextContent('Copy failed');
  });
});

describe('Outcome', () => {
  it('shows no result as not checked, never as a pass', () => {
    const { container } = render(<Outcome result={null} />);
    expect(container).toHaveTextContent('NOT CHECKED');
    expect(container).not.toHaveTextContent('✓');
    expect(container).not.toHaveTextContent('MATCH');
  });
});

describe('Hero', () => {
  it('links the displayed revision to that exact revision', () => {
    render(<Hero facts={summarize(loadEvidence())} />);
    expect(screen.getByRole('link', { name: '96dc282a' })).toHaveAttribute('href', PINNED);
  });

  it('counts only matched segments, and only says all matching when they did', () => {
    render(<Hero facts={summarize(withMismatch(5))} />);
    expect(screen.getByText('36 / 37')).toBeInTheDocument();
    expect(screen.queryByText('37 / 37')).toBeNull();
    expect(screen.getByText('project checks, 1 mismatching')).toBeInTheDocument();
  });

  it('puts each label before its value, for screen readers', () => {
    const { container } = render(<Hero facts={summarize(loadEvidence())} />);
    for (const group of container.querySelectorAll('.figures > div')) {
      expect(group.firstElementChild?.tagName).toBe('DT');
    }
  });
});

describe('Hero honesty', () => {
  it('says the page checks consistency, and points to the real checks', () => {
    render(<Hero facts={summarize(loadEvidence())} />);
    expect(screen.getByText(/consistent with itself/)).toBeInTheDocument();
  });
});

describe('Results coverage map', () => {
  it('shows the real snapshot: every project cell matched, every independent cell unchecked', () => {
    const { container } = renderResults(loadEvidence());
    expect(screen.getByText('Project: 37 / 37 matched')).toBeInTheDocument();
    expect(screen.getByText('Independent: 0 / 37 checked')).toHaveClass('caveat');
    expect([...projectCells(container)].every(c => c.classList.contains('match'))).toBe(true);
    expect([...independentCells(container)].every(c => c.classList.contains('none'))).toBe(true);
    expect(independentCells(container)[0]).toHaveAttribute('title', 'Segment 0: not checked, by anyone independent');
  });

  it('marks a mismatched segment, and only that one', () => {
    const { container } = renderResults(withMismatch(5));
    expect(projectCells(container)[5]).toHaveClass('mismatch');
    expect(projectCells(container)[5]).toHaveAttribute('title', 'Segment 5: MISMATCH, by the project');
    expect(projectCells(container)[4]).toHaveClass('match');
    expect(container.querySelector('.covgrid')).toHaveAttribute('aria-label', expect.stringContaining('36 matched, 1 mismatched, 0 not checked'));
    expect(container.textContent).not.toContain('Every segment matched');
    expect(container.textContent).not.toContain('all matching');
  });

  it('shows a segment with no checks as not checked, not as a pass or a mismatch', () => {
    const { container } = renderResults(withoutChecks(7));
    expect(projectCells(container)[7]).toHaveClass('none');
    expect(projectCells(container)[7]).toHaveAttribute('title', 'Segment 7: not checked, by the project');
    expect(screen.getByText(/Project: 36 \/ 37 matched \(1 not checked\)/)).toBeInTheDocument();
  });

  it('keeps an independent result in the independent row', () => {
    const evidence = loadEvidence();
    const rows = evidence.verifications.rawChecks.filter(c => c.segment === 3);
    const categories = Object.fromEntries(rows.map(c => [c.checker, 'project' as const]));
    const outside = { ...rows[0], checker: 'outside checker', result: 'MISMATCH' as const };
    evidence.verifications.coverage = deriveCoverage(
      [...evidence.verifications.rawChecks, outside],
      37,
      { ...categories, 'project second pod': 'project', 'project training pod': 'project', 'project fresh machine public files only': 'project', 'outside checker': 'independent' },
    );
    const { container } = renderResults(evidence);
    expect(independentCells(container)[3]).toHaveClass('mismatch');
    expect(independentCells(container)[3]).toHaveAttribute('title', 'Segment 3: MISMATCH, by anyone independent');
    expect(projectCells(container)[3]).toHaveClass('match');
    expect(independentCells(container)[2]).toHaveClass('none');
  });

  it('offers a per-segment table to screen readers', () => {
    renderResults(withMismatch(5));
    const table = screen.getByRole('table', { name: /Result of each segment/ });
    const row = within(table).getByRole('row', { name: /^5 MISMATCH not checked$/ });
    expect(row).toBeInTheDocument();
  });

  it('labels hardware probes from the data', () => {
    renderResults(loadEvidence());
    expect(screen.getByText('All 37 segments')).toBeInTheDocument();
    expect(screen.getAllByText('Segment 32').length).toBeGreaterThan(0);
  });

  it('works out the odds from the segment count', () => {
    const { container } = renderResults(loadEvidence());
    const odds = container.querySelector<HTMLElement>('.figures.small:not(.data)')!;
    expect(within(odds).getByText('110')).toBeInTheDocument();
    expect(within(odds).getByText('169')).toBeInTheDocument();
    expect(screen.getByText('P = 1 − (36/37)^M, with N = 37')).toBeInTheDocument();
  });

  it('counts the training code from the pinned file', () => {
    renderResults(loadEvidence());
    expect(screen.getByText('529 lines')).toBeInTheDocument();
  });
});

describe('Training history', () => {
  it('marks both events with the results of their segments', () => {
    const { container } = renderHistory(loadEvidence());
    expect(container.querySelectorAll('.milestones .res.match')).toHaveLength(2);
    expect(container.querySelector('.sub')).toHaveTextContent('still matched');
    expect(container.querySelector('figcaption')).toHaveTextContent('37 segments, all matched by the project');
  });

  it('shows a mismatch on the host failure segment', () => {
    const { container } = renderHistory(withMismatch(10));
    expect(container.querySelector('.milestones li .res')).toHaveTextContent('MISMATCH');
    expect(container.textContent).not.toContain('still matched');
    expect(container.textContent).not.toContain('all matched');
  });

  it('fails the code change result if any later segment mismatched', () => {
    const { container } = renderHistory(withMismatch(30));
    expect(container.querySelectorAll('.milestones li .res')[1]).toHaveTextContent('MISMATCH');
  });

  it('shows a missing check on the code change as not checked', () => {
    const { container } = renderHistory(withoutChecks(25));
    expect(container.querySelectorAll('.milestones li .res')[1]).toHaveTextContent('NOT CHECKED');
    const rect = container.querySelectorAll('.tl-chart rect')[25];
    expect(rect).toHaveAttribute('fill', 'none');
    expect(rect.querySelector('title')).toHaveTextContent('Segment 25: not checked');
  });

  it('colours each timeline segment from its result', () => {
    const { container } = renderHistory(withMismatch(5));
    const rects = container.querySelectorAll('.tl-chart rect');
    expect(rects).toHaveLength(37);
    expect(rects[5]).toHaveAttribute('fill', 'var(--miss)');
    expect(rects[6]).toHaveAttribute('fill', 'var(--match-fill)');
    expect(rects[5].querySelector('title')).toHaveTextContent('Segment 5: MISMATCH');
  });

  it('reports the exclusion from its own checks and counts', () => {
    renderHistory(loadEvidence());
    expect(screen.getByText('Loss on 50 control articles')).toBeInTheDocument();
    expect(screen.getByText(/All 8 segments after the split matched, recomputed by the project/)).toBeInTheDocument();
  });

  it('does not claim the fork matched when one of its checks did not', () => {
    const evidence = loadEvidence();
    evidence.exclusion.forkChecks[2].result = 'MISMATCH';
    renderHistory(evidence);
    expect(screen.getByText(/7 of the 8 segments after the split matched/)).toBeInTheDocument();
  });
});
