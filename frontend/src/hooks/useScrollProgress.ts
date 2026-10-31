import { useEffect, useRef, useState } from 'react';

// Tracks how far the page has scrolled through a tall element: 0 when its top
// reaches the top of the viewport, 1 when its bottom reaches the bottom. The
// value is written to the element's --p custom property (so CSS can animate
// without React re-rendering) and passed to onProgress when given.
export function useScrollProgress<T extends HTMLElement>(
  enabled: boolean,
  onProgress?: (progress: number) => void
) {
  const ref = useRef<T>(null);
  const callback = useRef(onProgress);
  callback.current = onProgress;

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    if (!enabled) {
      el.style.removeProperty('--p');
      return;
    }

    let frame = 0;
    const update = () => {
      frame = 0;
      const rect = el.getBoundingClientRect();
      const range = rect.height - window.innerHeight;
      const progress = range > 0 ? Math.min(Math.max(-rect.top / range, 0), 1) : 0;
      el.style.setProperty('--p', progress.toFixed(4));
      callback.current?.(progress);
    };
    const onScroll = () => {
      if (!frame) frame = window.requestAnimationFrame(update);
    };

    update();
    window.addEventListener('scroll', onScroll, { passive: true });
    window.addEventListener('resize', onScroll);
    return () => {
      window.removeEventListener('scroll', onScroll);
      window.removeEventListener('resize', onScroll);
      if (frame) window.cancelAnimationFrame(frame);
    };
  }, [enabled]);

  return ref;
}

// True while the viewport is at most maxWidth wide.
export function useNarrow(maxWidth: number): boolean {
  const query = `(max-width: ${maxWidth}px)`;
  const [narrow, setNarrow] = useState(
    () => typeof window !== 'undefined' && window.matchMedia?.(query).matches === true
  );

  useEffect(() => {
    const media = window.matchMedia?.(query);
    if (!media) return;
    const onChange = () => setNarrow(media.matches);
    onChange();
    media.addEventListener('change', onChange);
    return () => media.removeEventListener('change', onChange);
  }, [query]);

  return narrow;
}
