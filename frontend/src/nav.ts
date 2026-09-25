import { useSearchParams } from 'react-router-dom';

export function useSnapSearch(): string {
  const [params] = useSearchParams();
  const search = params.toString();
  return search ? `?${search}` : '';
}
