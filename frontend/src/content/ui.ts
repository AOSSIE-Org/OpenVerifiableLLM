import uiJson from './ui.json' with { type: 'json' };

/** Navigation, headings and table labels. No facts and no numbers live here. */
export const ui = uiJson as Record<string, string>;
export function label(key: string): string {
  const v = ui[key];
  if (v === undefined) throw new Error(`missing UI label ${key}`);
  return v;
}
export const hasLabel = (key: string) => key in ui;
