import { useEffect, useState } from 'react';
import { motion, useScroll, useSpring, useTransform, useReducedMotion } from 'framer-motion';
import { Globe } from '@/components/ui/globe';

/** Hand-drawn style lines hugging both screen edges: one enters from the top, one from the bottom, and they draw as you scroll. */
export function ScrollLines() {
  const { scrollYProgress } = useScroll();
  const reduce = useReducedMotion();
  const drawn = useSpring(scrollYProgress, { stiffness: 90, damping: 24, mass: 0.6 });
  const length = useTransform(drawn, v => 0.22 + v * 0.78);
  const [size, setSize] = useState({ w: 1440, h: 900 });
  useEffect(() => {
    const read = () => setSize({ w: window.innerWidth, h: window.innerHeight });
    read(); window.addEventListener('resize', read);
    return () => window.removeEventListener('resize', read);
  }, []);
  // Coordinates are percentages of the viewport, converted to pixels so the stroke keeps a uniform width.
  const px = (d: string) => { let i = 0; return d.replace(/-?\d+(\.\d+)?/g, n => ((i++ % 2 === 0 ? +n * size.w : +n * size.h) / 100).toFixed(1)); };
  const left = px('M 4 0 C 15 11, 2 21, 9 33 S 17 50, 4 59 S -5 75, 7 85 S 13 95, 5 100');
  const right = px('M 96 100 C 85 89, 98 79, 91 67 S 83 49, 97 41 S 105 25, 94 15 S 87 5, 95 0');
  const common = { fill: 'none', stroke: 'var(--sage)', strokeWidth: 1.6, strokeLinecap: 'round' as const };
  return <svg className="scroll-lines" viewBox={`0 0 ${size.w} ${size.h}`} aria-hidden="true" focusable="false">
    <motion.path d={left} {...common} style={reduce ? undefined : { pathLength: length }} initial={false} />
    <motion.path d={right} {...common} style={reduce ? undefined : { pathLength: length }} initial={false} />
  </svg>;
}

/** Fades page blocks in as they enter the viewport. */
export function useScrollReveal(deps: unknown[]) {
  useEffect(() => {
    if (typeof IntersectionObserver === 'undefined' || window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
    const targets = document.querySelectorAll<HTMLElement>('main > *, main section > *, .reveal');
    const io = new IntersectionObserver(entries => entries.forEach(e => {
      if (e.isIntersecting) { e.target.classList.add('reveal-in'); io.unobserve(e.target); }
    }), { threshold: 0.08, rootMargin: '0px 0px -6% 0px' });
    targets.forEach(el => { if (!el.classList.contains('reveal-in')) { el.classList.add('reveal-on'); io.observe(el); } });
    return () => io.disconnect();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
}

export function SiteFooter({ nav, sample }: { nav: string[][]; sample: boolean }) {
  return <footer className="site-footer">
    <div className="container footer-inner">
      <div className="footer-content">
        <div className="footer-lead">
          <p className="footer-eyebrow">Provenance &amp; Verification</p>
          <p className="footer-title">Open to inspect from anywhere.</p>
          <p className="footer-text">Every report here is a display of what was supplied. Read the record, check the scope.</p>
        </div>
        <nav aria-label="Footer" className="footer-nav">
          {nav.map(([p, label]) => <a key={p} href={'#' + p} className="footer-nav-link">{label}</a>)}
        </nav>
        <div className="footer-bottom">
          <div className="footer-credit">
            <a href="https://github.com/AOSSIE-Org/OpenVerifiableLLM" target="_blank" rel="noopener noreferrer">Made by AOSSIE<span className="sr-only"> (external, opens in a new tab)</span></a>
          </div>
        </div>
      </div>
    </div>
    <div className="footer-globe"><Globe /></div>
  </footer>;
}
