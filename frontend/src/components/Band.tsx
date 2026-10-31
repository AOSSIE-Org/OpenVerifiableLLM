import React, { useEffect, useRef } from 'react';
import { usePrefersReducedMotion } from '../hooks/usePrefersReducedMotion';

// Wraps a run of sections that fill with a warm yellow as you scroll into
// them. --bp goes from 0 (Verification just entering at the bottom of the
// screen) to 1 (it has reached the upper part), and every colour in the band
// is mixed by that amount in base.css, so the change is steady and reversible.
// With reduced motion the band is simply yellow.
export const Band: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const ref = useRef<HTMLDivElement>(null);
  const reduced = usePrefersReducedMotion();

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    if (reduced) {
      el.style.setProperty('--bp', '1');
      return;
    }

    let frame = 0;
    const update = () => {
      frame = 0;
      const vh = window.innerHeight;
      const top = el.getBoundingClientRect().top;
      const p = Math.min(Math.max((vh * 0.95 - top) / (vh * 0.6), 0), 1);
      el.style.setProperty('--bp', p.toFixed(3));
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
  }, [reduced]);

  return (
    <div ref={ref} className="band">
      {children}
    </div>
  );
};
