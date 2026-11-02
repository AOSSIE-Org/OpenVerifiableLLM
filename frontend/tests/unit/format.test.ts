import { describe, expect, it } from 'vitest';
import { formatValue } from '../../src/data/format';
import { getPath } from '../../src/data/path';
import { fresh } from './helpers';

describe('formatValue', () => {
  it('formats the odds the way the brief states them', () => {
    const d = fresh();
    expect(formatValue(d.odds.p95, 'pct2')).toBe('95.09%');
    expect(formatValue(d.odds.p99, 'pct2')).toBe('99.02%');
  });
  it('formats integers, dates, hashes and lists', () => {
    expect(formatValue(7232582, 'int')).toBe('7,232,582');
    expect(formatValue('2026-10-20T14:38:43Z', 'utc')).toBe('20 Oct 2026, 14:38:43 UTC');
    expect(formatValue('8de1d60065b1', 'short')).toBe('8de1d600…');
    expect(formatValue([21], 'list')).toBe('21');
    expect(formatValue(true, 'yesno')).toBe('yes');
  });
  it('refuses to format a missing value instead of printing "undefined"', () => {
    expect(() => formatValue(undefined)).toThrow();
    expect(() => formatValue('not a date', 'utc')).toThrow();
  });
  it('reads dotted paths including array indexes and length', () => {
    const d = fresh();
    expect(getPath(d, 'events.0.segment')).toBe(10);
    expect(getPath(d, 'hardware.length')).toBe(5);
    expect(getPath(d, 'nope.deeper')).toBeUndefined();
  });
});
