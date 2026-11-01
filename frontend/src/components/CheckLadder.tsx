import React from 'react';
import { copy } from '../content/copy';
import { CopyButton } from './CopyButton';

// Three ways to check, side by side, each marked by a thin rule on its left.
// A one-line command shows in place; a longer script folds away.
export const CheckLadder: React.FC = () => (
  <section id="check">
    <h2>{copy.check.kicker}</h2>
    <p className="sub">{copy.check.intro}</p>

    <ol className="steps">
      {copy.check.ladder.map(step => (
        <li key={step.time} className={step.isKey ? 'step key' : 'step'}>
          <p className="step-meta">
            {step.time} · {step.meta}
          </p>
          <h3>{step.title}</h3>
          <p className="step-desc">{step.desc}</p>
          <ul className="step-proves">
            {step.chips.map(chip => (
              <li key={chip.label} className={chip.type}>
                {chip.label}
              </li>
            ))}
          </ul>

          {step.command && !step.command.includes('\n') && (
            <div className="code">
              <pre tabIndex={0} role="region" aria-label={`${step.title} command`}>
                {step.command}
              </pre>
              <CopyButton textToCopy={step.command} />
            </div>
          )}

          {step.command && step.command.includes('\n') && (
            <details className="step-code">
              <summary>Show the commands</summary>
              <div className="code">
                <pre tabIndex={0} role="region" aria-label={`${step.title} commands`}>
                  {step.command}
                </pre>
                <CopyButton textToCopy={step.command} />
              </div>
            </details>
          )}

          {step.action && (
            <p className="step-action">
              <a href={step.action.href}>{step.action.label} →</a>
            </p>
          )}
        </li>
      ))}
    </ol>
  </section>
);
