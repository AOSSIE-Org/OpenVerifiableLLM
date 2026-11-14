import { describe, expect, it } from 'vitest';

const marker = '<div id="root"><!--app--></div>';

describe('prerender replacements', () => {
  it('keeps $& and $\' in inserted HTML when the replacer is a function', () => {
    const template = `head${marker}tail`;
    const app = "cost $& and $'";
    const want = `head<div id="root">${app}</div>tail`;
    expect(template.replace(marker, `<div id="root">${app}</div>`)).not.toBe(want);
    expect(template.replace(marker, () => `<div id="root">${app}</div>`)).toBe(want);
  });
});
