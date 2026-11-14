import { afterEach, describe, expect, it } from 'vitest';
import { applyStoredTheme, isTheme, readTheme, THEME_COLOR, THEME_KEY, writeTheme } from '../../src/theme';

afterEach(() => {
  localStorage.removeItem(THEME_KEY);
  document.documentElement.removeAttribute('data-theme');
  document.documentElement.style.colorScheme = '';
});

describe('theme', () => {
  it('accepts only light and dark', () => {
    expect(isTheme('light')).toBe(true);
    expect(isTheme('dark')).toBe(true);
    expect(isTheme('system')).toBe(false);
    expect(isTheme('<script>')).toBe(false);
    expect(isTheme('javascript:alert(1)')).toBe(false);
    expect(isTheme(null)).toBe(false);
  });

  it('defaults to light when storage is empty or hostile', () => {
    expect(readTheme()).toBe('light');
    localStorage.setItem(THEME_KEY, '<img src=x onerror=alert(1)>');
    expect(readTheme()).toBe('light');
    applyStoredTheme();
    expect(document.documentElement.dataset.theme).toBe('light');
  });

  it('persists a valid choice and ignores anything else', () => {
    writeTheme('dark');
    expect(localStorage.getItem(THEME_KEY)).toBe('dark');
    expect(document.documentElement.dataset.theme).toBe('dark');
    expect(document.documentElement.style.colorScheme).toBe('dark');
    localStorage.setItem(THEME_KEY, 'purple');
    applyStoredTheme();
    expect(document.documentElement.dataset.theme).toBe('light');
  });

  it('updates theme-color when the meta tag is present', () => {
    const meta = document.createElement('meta');
    meta.name = 'theme-color';
    meta.content = THEME_COLOR.light;
    document.head.appendChild(meta);
    writeTheme('dark');
    expect(meta.getAttribute('content')).toBe(THEME_COLOR.dark);
    writeTheme('light');
    expect(meta.getAttribute('content')).toBe(THEME_COLOR.light);
    meta.remove();
  });
});
