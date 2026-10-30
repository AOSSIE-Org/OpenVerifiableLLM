import React, { useEffect, useRef, useState } from 'react';
import { copy } from '../content/copy';

export const MiniBar: React.FC = () => {
  const [hidden, setHidden] = useState(false);
  const barRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const footer = document.getElementById('footer');
    if (!footer) return;

    const observer = new IntersectionObserver(
      ([entry]) => {
        setHidden(entry.isIntersecting);
      },
      { threshold: 0.05 }
    );

    observer.observe(footer);
    return () => observer.disconnect();
  }, []);

  return (
    <div
      ref={barRef}
      className={`mini ${hidden ? 'off' : ''}`}
      id="mini"
      inert={hidden ? true : undefined}
    >
      <div className="wrap">
        <a className="org" href="https://aossie.org">
          <img src="brand/aossie-logo.svg" alt="" height="22" width="20" />
          <span>
            <b>An AOSSIE project</b> · GPL-3.0
          </span>
        </a>
        <nav aria-label="Quick links">
          {copy.miniBar.links.map((link) => (
            <a key={link.label} href={link.href}>
              {link.label}
            </a>
          ))}
        </nav>
      </div>
    </div>
  );
};

