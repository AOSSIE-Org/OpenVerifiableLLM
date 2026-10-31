import React from 'react';
import { copy } from '../content/copy';

interface FooterProps {
  revision: string;
}

// A plain page footer: who publishes it, the licence, the evidence revision
// and how to get in touch.
export const Footer: React.FC<FooterProps> = ({ revision }) => (
  <footer id="footer">
    <div className="foot">
      <a href="https://aossie.org" className="foot-org">
        <img src="brand/aossie-logo.svg" alt="" height="40" width="38" />
        <span>{copy.footer.org}</span>
      </a>
      <ul className="foot-lines">
        <li>{copy.footer.license}</li>
        <li>
          {copy.footer.evidenceNote} <code>{revision}</code>.
        </li>
        <li>
          {copy.footer.contactHeading}:{' '}
          <a href={`mailto:${copy.footer.contactEmail}`}>{copy.footer.contactEmail}</a>
        </li>
      </ul>
      <div className="socials">
        <a href="https://github.com/AOSSIE-Org" aria-label="AOSSIE on GitHub">
          <svg viewBox="0 0 24 24" aria-hidden="true">
            <path d="M15 22v-4a4.8 4.8 0 0 0-1-3.5c3 0 6-2 6-5.5.08-1.25-.27-2.48-1-3.5.28-1.15.28-2.35 0-3.5 0 0-1 0-3 1.5-2.64-.5-5.36-.5-8 0C6 2 5 2 5 2c-.3 1.15-.3 2.35 0 3.5A5.4 5.4 0 0 0 4 9c0 3.5 3 5.5 6 5.5-.39.49-.68 1.05-.85 1.65-.17.6-.22 1.23-.15 1.85v4" />
            <path d="M9 18c-4.51 2-5-2-7-2" />
          </svg>
        </a>

        <a href="https://discord.gg/hjUhu33uAn" aria-label="AOSSIE on Discord">
          <svg viewBox="0 0 24 24" aria-hidden="true">
            <circle cx="9" cy="12" r="1" />
            <circle cx="15" cy="12" r="1" />
            <path d="M7.5 7.5c3.5-1 5.5-1 9 0" />
            <path d="M7 16.5c3.5 1 6.5 1 10 0" />
            <path d="M15.5 17c0 1 1.5 3 2 3 1.5 0 2.83-1.67 3.5-3 .67-1.67.5-5.83-1.5-11.5-1.46-1.02-3-1.34-4.5-1.5l-.97 1.92" />
            <path d="M8.5 17c0 1-1.36 3-1.83 3-1.43 0-2.7-1.67-3.34-3-.63-1.67-.47-5.83 1.43-11.5 1.39-1.02 2.78-1.34 4.24-1.5l1 2.5" />
          </svg>
        </a>

        <a href="https://x.com/aossie_org" aria-label="AOSSIE on X">
          <svg viewBox="0 0 24 24" aria-hidden="true">
            <path d="M4 4l11.73 16H20L8.27 4z" />
            <path d="M4 20l6.77-6.77M13.23 10.77 20 4" />
          </svg>
        </a>

        <a href="https://www.linkedin.com/company/aossie/" aria-label="AOSSIE on LinkedIn">
          <svg viewBox="0 0 24 24" aria-hidden="true">
            <path d="M16 8a6 6 0 0 1 6 6v7h-4v-7a2 2 0 0 0-4 0v7h-4v-7a6 6 0 0 1 6-6z" />
            <rect x="2" y="9" width="4" height="12" />
            <circle cx="4" cy="4" r="2" />
          </svg>
        </a>

        <a href="https://www.youtube.com/@AOSSIE-Org" aria-label="AOSSIE on YouTube">
          <svg viewBox="0 0 24 24" aria-hidden="true">
            <path d="M2.5 17a24 24 0 0 1 0-10 2 2 0 0 1 1.4-1.4 49.6 49.6 0 0 1 16.2 0A2 2 0 0 1 21.5 7a24 24 0 0 1 0 10 2 2 0 0 1-1.4 1.4 49.6 49.6 0 0 1-16.2 0A2 2 0 0 1 2.5 17" />
            <path d="m10 15 5-3-5-3z" />
          </svg>
        </a>

        <a href={`mailto:${copy.footer.contactEmail}`} aria-label="Email AOSSIE">
          <svg viewBox="0 0 24 24" aria-hidden="true">
            <rect x="2" y="4" width="20" height="16" rx="2" />
            <path d="m22 7-8.97 5.7a1.94 1.94 0 0 1-2.06 0L2 7" />
          </svg>
        </a>
      </div>
    </div>
  </footer>
);
