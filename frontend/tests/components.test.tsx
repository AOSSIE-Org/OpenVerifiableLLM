import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { render, screen, fireEvent, act } from '@testing-library/react';
import { Header } from '../src/components/Header';
import { CopyButton } from '../src/components/CopyButton';

describe('Header component', () => {
  beforeEach(() => {
    localStorage.clear();
    document.documentElement.removeAttribute('data-theme');
  });

  afterEach(() => {
    document.documentElement.removeAttribute('data-theme');
  });

  it('renders branding and section links', () => {
    render(<Header />);
    expect(screen.getByText('OpenVerifiable')).toBeInTheDocument();
    expect(screen.getByText('LLM')).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'Source' })).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'Evidence' })).toBeInTheDocument();
  });

  it('starts light and toggles to dark', () => {
    render(<Header />);
    const button = screen.getByRole('button', { name: /switch between dark and light theme/i });
    expect(document.documentElement.getAttribute('data-theme')).toBeNull();

    fireEvent.click(button);
    expect(document.documentElement.getAttribute('data-theme')).toBe('dark');

    fireEvent.click(button);
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
});
