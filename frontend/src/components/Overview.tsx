import React, { useEffect, useRef, useState } from 'react';
import { copy } from '../content/copy';
import type { PreparationData, VerificationsData } from '../data/contracts';
import { usePrefersReducedMotion } from '../hooks/usePrefersReducedMotion';

interface OverviewProps {
  preparation: PreparationData;
  verifications: VerificationsData;
  authorities: string[];
}

// The timestamp seal travels with the scroll: while its card moves up the
// screen, the seal flies from the top-left corner of the window, tumbling in
// 3D, and settles onto the card. Scrolling back up sends it back out.
const Stamp: React.FC<{ authorities: string[] }> = ({ authorities }) => {
  const ref = useRef<HTMLSpanElement>(null);
  const reduced = usePrefersReducedMotion();
  const [landed, setLanded] = useState(false);

  useEffect(() => {
    const el = ref.current;
    const slot = el?.parentElement;
    if (!el || !slot || reduced) return;

    let frame = 0;
    const update = () => {
      frame = 0;
      // measure the untransformed slot, not the moving seal
      const r = slot.getBoundingClientRect();
      const vh = window.innerHeight;
      const p = Math.min(Math.max((vh - r.top) / (vh * 0.55), 0), 1);
      const away = 1 - (1 - (1 - p) * (1 - p)); // eased: 1 far away, 0 landed
      el.style.transform = [
        `translate(${away * (-r.left - 40)}px, ${away * (-r.top - 40)}px)`,
        `rotateX(${away * 60}deg)`,
        `rotateY(${away * -40}deg)`,
        `rotateZ(${-12 - away * 150}deg)`,
        `scale(${1 + away * 1.4})`,
      ].join(' ');
      el.style.opacity = String(Math.min(1, p * 3));
      setLanded(p >= 1);
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
    <span ref={ref} className={`art-stamp${landed ? ' landed' : ''}`}>
      <svg viewBox="0 0 120 120">
        <circle cx="60" cy="60" r="52" />
        <circle cx="60" cy="60" r="42" className="inner" />
      </svg>
      <b>STAMPED</b>
      <small>{authorities.length} TSAs</small>
    </span>
  );
};

// A small drawing above each headline fact, made from the evidence.
const FactArt: React.FC<{ index: number } & OverviewProps> = ({ index, preparation, verifications, authorities }) => {
  if (index === 0) {
    return (
      <span className="art-ticks">
        {verifications.coverage.map(c => (
          <i key={c.segment} className={c.projectResult === 'MATCH' ? 'ok' : ''} />
        ))}
      </span>
    );
  }
  if (index === 1) {
    const id = `${preparation.preparedDataId.slice(0, 8)}…`;
    return (
      <span className="art-same">
        <span>
          raw dump <i /> <code>{id}</code>
        </span>
        <span>
          rebuilt <i /> <code>{id}</code>
        </span>
      </span>
    );
  }
  return <Stamp authorities={authorities} />;
};

export const Overview: React.FC<OverviewProps> = props => {
  const { preparation } = props;
  return (
    <section id="overview">
      <div className="wrap">
        <div className="sh">
          <div className="kicker">{copy.overview.kicker}</div>
          <h2>{copy.overview.h2}</h2>
          <p>{copy.overview.subtitle}</p>
        </div>

        <div className="facts" tabIndex={0} role="region" aria-label="Headline facts">
          {copy.overview.facts.map((fact, i) => (
            <div key={i} className="panel fact">
              <div className="fact-art" aria-hidden="true">
                <FactArt index={i} {...props} />
              </div>
              <div className="n">
                {fact.value}
                {fact.total && <small>{fact.total}</small>}
              </div>
              <h3>{fact.title}</h3>
              <span className="chip h">{fact.chip}</span>
            </div>
          ))}
        </div>

        <div className="split" tabIndex={0} role="region" aria-label="What is shown and what is not">
          <div className="panel yes">
            <h3>
              <span className="chip m">✓</span> {copy.overview.shownTitle}
            </h3>
            <ul>
              {copy.overview.shownItems.map((item, i) => (
                <li key={i}>{item}</li>
              ))}
            </ul>
          </div>

          <div className="panel no">
            <h3>
              <span className="chip h">○</span> {copy.overview.notShownTitle}
            </h3>
            <ul>
              {copy.overview.notShownItems.map((item, i) => (
                <li key={i}>{item}</li>
              ))}
            </ul>
          </div>
        </div>

        <div className="counts">
          <div>
            <b>{preparation.articles.toLocaleString()}</b>
            <span>Articles</span>
          </div>
          <div>
            <b>{preparation.wikipediaTargets.toLocaleString()}</b>
            <span>Wikipedia targets</span>
          </div>
          <div>
            <b>{preparation.conversations.toLocaleString()}</b>
            <span>Conversations</span>
          </div>
          <div>
            <b>{preparation.trainingLoopLines}</b>
            <span>Training loop</span>
          </div>
        </div>
      </div>
    </section>
  );
};
