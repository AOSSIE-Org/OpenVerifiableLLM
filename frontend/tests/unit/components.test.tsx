import { afterEach, describe, expect, it, vi } from 'vitest';
import { act, cleanup, fireEvent, render, screen, within } from '@testing-library/react';
import type { ComponentType } from 'react';
import { App } from '../../src/App';
import { DataContext } from '../../src/data/context';
import { validateSite, type SectionErrors } from '../../src/data/validate';
import { formatValue, type Fmt } from '../../src/data/format';
import { getPath } from '../../src/data/path';
import { claim, claims, label } from '../../src/content/ledger';
import { limitParts } from '../../src/components/ClaimBlock';
import { loadData } from '../../src/data/load';
import { ROUTES } from '../../src/routes';
import Overview from '../../src/pages/Overview';
import Check from '../../src/pages/Check';
import Verification from '../../src/pages/Verification';
import Timeline from '../../src/pages/Timeline';
import Exclusion from '../../src/pages/Exclusion';
import Downloads from '../../src/pages/Downloads';
import { Section } from '../../src/components/Section';
import { distinctLimits, duplicateOf } from '../../src/pages/Verification';
import { OVERVIEW_FACTS, OVERVIEW_LIMITS } from '../../src/pages/Overview';
import type { SiteData } from '../../src/data/types';
import { fresh } from './helpers';
import { fillCounts } from '../../src/content/fill';

afterEach(cleanup);
const PAGES: [string, ComponentType][] = [['overview', Overview], ['check', Check], ['verification', Verification], ['timeline', Timeline], ['exclusion', Exclusion], ['downloads', Downloads]];

function mount(Page: ComponentType, data: SiteData = fresh(), errors: SectionErrors = {}) {
  return render(
    <DataContext.Provider value={{ data, errors }}>
      <Page />
    </DataContext.Provider>,
  );
}

describe.each(PAGES)('%s page', (key, Page) => {
  it('renders every value from site-data.json exactly as formatted', () => {
    const d = fresh();
    const { container } = mount(Page, d);
    const vals = container.querySelectorAll<HTMLElement>('[data-path]');
    // "Check it yourself" is commands and ledger text only; every other page shows data values.
    if (key !== 'check') expect(vals.length).toBeGreaterThan(0);
    for (const el of vals) expect(el.textContent).toBe(formatValue(getPath(d, el.dataset.path!), (el.dataset.fmt ?? 'auto') as Fmt));
  });

  it('renders claims only from the ledger, each with a source chip and in brief order', () => {
    const { container } = mount(Page);
    for (const ev of container.querySelectorAll<HTMLElement>('[data-claim]')) {
      const c = claims.get(ev.dataset.claim!)!;
      expect(c).toBeTruthy();
      // Inside a Limits list only the "Limit…:" prefix is dropped; the rest is the ledger text.
      const lt = ev.querySelector('[data-claim-part="limit-text"]');
      const plain = fillCounts(c.id, c.plain, fresh())!;
      if (lt) expect(lt.textContent).toBe(limitParts(plain).rest);
      else expect(ev.querySelector('[data-claim-part="plain"]')!.textContent).toBe(plain);
      expect(ev.querySelector('details.chip')).not.toBeNull();
      const order = [...ev.querySelectorAll('.plain, .limit, .chips')].map((n) => n.className);
      expect(order[0]).toBe('plain');
      if (c.limit) expect(order.indexOf('limit')).toBeLessThan(order.indexOf('chips'));
    }
  });

  it('shows the fallback for one malformed section without crashing the page', () => {
    const errors: SectionErrors = { verifications: 'x', hardware: 'x', downloads: 'x', exclusion: 'x', check: 'x', chain: 'x', odds: 'x', events: 'x', preparation: 'x', anchor: 'x' };
    const { container } = mount(Page, fresh(), errors);
    expect(container.querySelector('h1')).not.toBeNull();
    expect(container.querySelectorAll('[data-fallback]').length).toBeGreaterThan(0);
    for (const f of container.querySelectorAll('[data-fallback]')) expect(f.textContent).toBe(label('error.section'));
  });
});

it('renders every allowed claim somewhere, except de-duplicated limits', () => {
  const seen = new Set<string>();
  for (const [, Page] of PAGES) {
    const { container } = mount(Page);
    container.querySelectorAll<HTMLElement>('[data-claim]').forEach((e) => seen.add(e.dataset.claim!));
    cleanup();
  }
  // A claim may move pages; a limit claim that only repeats another claim ("see X") may be dropped
  // where X is shown. Nothing else may go missing.
  const missing = [...claims.keys()].filter((id) => !seen.has(id));
  for (const id of missing) expect(duplicateOf(id), id).not.toBeNull();
  for (const id of missing) expect(seen.has(duplicateOf(id)!), id).toBe(true);
});

describe('coverage map', () => {
  it('has 37 real buttons and a table fallback with 37 rows', () => {
    mount(Verification);
    const map = document.querySelector('.tiles-int')!;
    expect(within(map as HTMLElement).getAllByRole('button')).toHaveLength(37);
    expect(document.querySelectorAll('.table-fallback tbody tr')).toHaveLength(37);
  });

  it('opens every check for a segment when a tile is pressed', () => {
    const d = fresh();
    mount(Verification, d);
    const tile = document.querySelectorAll<HTMLButtonElement>('.tiles-int button')[22]!;
    fireEvent.click(tile);
    expect(tile.getAttribute('aria-pressed')).toBe('true');
    const items = document.querySelectorAll('.tile-detail .checks li');
    expect(items).toHaveLength(d.verifications.rows.filter((r) => r.segment === 22).length);
  });

  it('places event markings wherever the data puts them', () => {
    const d = fresh();
    d.verifications.tiles[10]!.events = [];
    d.verifications.tiles[3]!.events = ['restart'];
    mount(Overview, d);
    const tiles = document.querySelectorAll<HTMLElement>('.tile');
    expect(tiles[3]!.dataset.event).toBe('restart');
    expect(tiles[10]!.dataset.event).toBeUndefined();
    expect(tiles[19]!.dataset.event).toBe('code-change');
  });

  it('draws a solid independent tile only when the data has an independent check', () => {
    mount(Overview);
    expect(document.querySelectorAll('.tile[data-checker="independent"]')).toHaveLength(0);
    expect(document.querySelectorAll('.tile[data-checker="project"]')).toHaveLength(37);
  });
});

describe('verification rows', () => {
  it('shows all 46 rows, grouped 37 / 8 / 1, all by the project', () => {
    mount(Verification);
    const groups = [...document.querySelectorAll('.groups')][0]!.querySelectorAll('section.group');
    expect([...groups].map((g) => g.querySelectorAll('tbody tr').length)).toEqual([37, 8, 1]);
    for (const g of groups) expect(g.querySelector('[data-who]')!.getAttribute('data-who')).toBe('project');
  });
  it('shows the hardware map as one card per machine for phones', () => {
    mount(Verification);
    expect(document.querySelectorAll('.hw-cards > li')).toHaveLength(fresh().hardware.length);
  });
});

describe('Section error boundary', () => {
  it('contains a render crash to its own section', () => {
    const Boom = () => {
      throw new Error('boom');
    };
    const spy = vi.spyOn(console, 'error').mockImplementation(() => {});
    vi.spyOn(console, 'warn').mockImplementation(() => {});
    render(
      <DataContext.Provider value={{ data: fresh(), errors: {} }}>
        <p>before</p>
        <Section needs={[]}>
          <Boom />
        </Section>
        <p>after</p>
      </DataContext.Provider>,
    );
    expect(screen.getByText('before')).toBeTruthy();
    expect(screen.getByText('after')).toBeTruthy();
    expect(screen.getByRole('alert').textContent).toBe(label('error.section'));
    spy.mockRestore();
  });
});

describe('App states', () => {
  it('shows the error page and never a MATCH when data fails', () => {
    const { container } = render(<App state={{ error: 'HTTP 500' }} route={ROUTES[0]!} Page={Overview} />);
    expect(screen.getByRole('alert').textContent).toContain(label('error.title'));
    expect(container.querySelectorAll('.result-MATCH, [data-result="MATCH"]')).toHaveLength(0);
  });
  it('shows a loading state that claims nothing', () => {
    const { container } = render(<App state="loading" route={ROUTES[0]!} Page={null} />);
    expect(screen.getByRole('status').textContent).toBe(label('loading'));
    expect(container.querySelectorAll('[data-result]')).toHaveLength(0);
  });
  it('marks the current page in the navigation', () => {
    render(<App state={validateSite(fresh())} route={ROUTES[2]!} Page={Verification} />);
    const current = document.querySelector('nav a[aria-current="page"]')!;
    expect(current.textContent).toBe(label('nav.verification'));
  });
});

describe('loadData', () => {
  it('returns an error on HTTP failure', async () => {
    const r = await loadData((async () => new Response('nope', { status: 500 })) as typeof fetch);
    expect(r).toEqual({ error: 'HTTP 500' });
  });
  it('returns an error on malformed JSON or an unpinned revision', async () => {
    expect('error' in (await loadData((async () => new Response('{not json')) as typeof fetch))).toBe(true);
    const d = fresh();
    d.revision = 'main';
    expect(await loadData((async () => new Response(JSON.stringify(d))) as typeof fetch)).toEqual({ error: 'revision is not the pinned 96dc282a' });
  });
});

describe('copy buttons', () => {
  it('puts the exact full digest on the clipboard and announces it', async () => {
    const writeText = vi.fn(async () => {});
    Object.defineProperty(navigator, 'clipboard', { value: { writeText }, configurable: true });
    const d = fresh();
    mount(Downloads, d);
    const btn = document.querySelector<HTMLButtonElement>('.files .hash button')!;
    await act(async () => fireEvent.click(btn));
    expect(writeText).toHaveBeenCalledWith(d.downloads[0]!.files[0]!.sha256);
    expect(btn.parentElement!.querySelector('[role="status"]')!.textContent).toBe(label('copied'));
  });
});

describe('overview and limits layout', () => {
  it('shows only the scoreboard, three headline facts and three limits', () => {
    mount(Overview);
    const ids = [...document.querySelectorAll<HTMLElement>('[data-claim]')].map((e) => e.dataset.claim);
    expect(ids).toEqual([...OVERVIEW_FACTS, ...OVERVIEW_LIMITS]);
    expect(document.querySelectorAll('.tiles-strip .tile')).toHaveLength(37);
    for (const id of OVERVIEW_LIMITS) expect(document.querySelector(`[data-claim="${id}"] .limit`)).toBeNull();
  });
  it('shows each distinct limit once across a page', () => {
    for (const Page of [Overview, Verification]) {
      mount(Page);
      const texts = [...document.querySelectorAll('.limits-list .plain')].map((e) => e.textContent);
      expect(new Set(texts).size).toBe(texts.length);
      cleanup();
    }
    expect(distinctLimits()).toEqual(['LI-02', 'LI-05', 'LI-06']);
    // The same point must not appear as a claim and again as a limit: a "see X" limit is never
    // listed on a page that already shows claim X.
    for (const Page of [Overview, Verification]) {
      mount(Page);
      for (const el of document.querySelectorAll<HTMLElement>('.limits-list [data-claim]')) {
        const d = duplicateOf(el.dataset.claim!);
        if (d) expect(document.querySelector(`[data-claim="${d}"]`), `${el.dataset.claim} repeats ${d}`).toBeNull();
      }
      cleanup();
    }
    expect(distinctLimits()).not.toContain('LI-01');
  });
});

describe('source toggle', () => {
  it('shows one chip and keeps the rest in a native <details> whose summary reports aria-expanded', () => {
    const d = fresh();
    mount(Overview, d);
    const ev = document.querySelector('[data-claim="OV-04"]')!;
    const id = (ev as HTMLElement).dataset.claim!;
    const n = d.claimSources[id]!.length;
    expect(n).toBeGreaterThan(1);
    const det = ev.querySelector<HTMLDetailsElement>('details.more-src')!;
    const sum = det.querySelector('summary.more')!;
    expect(sum.getAttribute('aria-expanded')).toBe('false');
    expect(sum.textContent).toContain(String(n - 1));
    const more = document.getElementById(sum.getAttribute('aria-controls')!)!;
    expect(det.contains(more)).toBe(true);
    expect(det.open).toBe(false);
    act(() => {
      det.open = true;
      det.dispatchEvent(new Event('toggle'));
    });
    expect(sum.getAttribute('aria-expanded')).toBe('true');
    expect(more.querySelectorAll('details.chip')).toHaveLength(n - 1);
    expect(ev.querySelectorAll('details.chip')).toHaveLength(n);
  });
});

describe('without JavaScript (prerendered HTML)', () => {
  it.each(PAGES)('%s: no <button> at all, extra sources in <details>, tiles link to seg-N rows', async (key, Page) => {
    const { renderToString } = await import('react-dom/server');
    const d = fresh();
    const html = renderToString(
      <DataContext.Provider value={{ data: d, errors: {} }}>
        <Page />
      </DataContext.Provider>,
    );
    expect(html).not.toContain('<button');
    expect(html).not.toContain('hidden=""');
    if (key === 'verification') {
      const links = [...html.matchAll(/<a class="tile"[^>]*href="\/verification\/#seg-(\d+)"/g)].map((m) => Number(m[1]));
      expect(links).toHaveLength(37);
      for (const k of links) expect(html).toContain(`id="seg-${k}"`);
    }
  });
});

describe('v3 follow-up', () => {
  it('Overview limits are the brief three, LI-01 included and not repeated on Verification', () => {
    expect(OVERVIEW_LIMITS).toEqual(['OV-13', 'OV-12', 'LI-01']);
    mount(Verification);
    expect(document.querySelector('.limits-list [data-claim="LI-01"]')).toBeNull();
    expect(document.querySelector('.limits-list [data-claim="LI-04"]')).toBeNull();
    expect(document.querySelector('[data-claim="VE-10"]')).not.toBeNull();
  });
  it('the trained sentence is the ledger plain lines of OV-01/02/03, verbatim', () => {
    mount(Overview);
    const parts = [...document.querySelectorAll<HTMLElement>('[data-trained] [data-claim-ref]')];
    expect(parts.map((e) => e.dataset.claimRef)).toEqual(['OV-01', 'OV-02', 'OV-03']);
    for (const e of parts) expect(e.textContent!.trim()).toBe(claim(e.dataset.claimRef!).plain);
  });
  it('limit lines drop the "Limit" prefixes and still name the project', () => {
    for (const Page of [Overview, Verification]) {
      mount(Page);
      for (const el of document.querySelectorAll('.limits-list .plain')) {
        expect(el.textContent).not.toMatch(/^Limit/);
        expect(el.textContent).toMatch(/the project/i);
      }
      cleanup();
    }
  });
  it('a ✓ glyph comes from MATCH status only; other statuses get distinct glyphs; names unchanged', () => {
    const d = fresh();
    d.verifications.tiles[0]!.result = 'MISMATCH';
    d.verifications.tiles[1]!.result = 'NOT_RUN';
    mount(Verification, d);
    const tiles = [...document.querySelectorAll<HTMLElement>('.tiles-int .tile')];
    tiles.forEach((t, k) => {
      const g = t.querySelector<HTMLElement>('.glyph')!;
      expect(g.getAttribute('aria-hidden')).toBe('true');
      const want = k === 0 ? '✗' : k === 1 ? '…' : '✓';
      expect(g.textContent).toBe(want);
      expect(t.querySelector('.sr')!.textContent).toContain(d.verifications.tiles[k]!.result);
    });
  });
  it('one extra source reads "more source", several read "more sources"', () => {
    expect(label('sources.one')).toBe('more source');
    expect(label('sources.more')).toBe('more sources');
  });
});
