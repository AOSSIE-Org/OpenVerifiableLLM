import { createContext, useContext } from 'react';
import type { SectionErrors } from './validate';
import type { SiteData } from './types';

export interface DataState { data: SiteData; errors: SectionErrors }
export const DataContext = createContext<DataState | null>(null);
export function useData(): DataState {
  const v = useContext(DataContext);
  if (!v) throw new Error('DataContext missing');
  return v;
}
