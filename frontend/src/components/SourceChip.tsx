import type { ClaimSource, Src } from '../data/types';
import { L } from './text';

/** Native <details>: works without JavaScript and with a keyboard. Opening it shows the exact cited lines. */
export function SourceChip({ s }: { s: ClaimSource | Src }) {
  const isClaim = 'lines' in s;
  const lines = isClaim ? s.lines : s.line ? `${s.line}${s.lineEnd ? `-${s.lineEnd}` : ''}` : '';
  const text = isClaim ? s.text : s.raw;
  const brief = isClaim && s.origin === 'brief';
  return (
    <details className="chip">
      <summary>
        {brief ? <L k="brief" /> : <span data-verbatim="source">{`${s.file}${lines ? `:${lines}` : ''} @${s.rev}`}</span>}
      </summary>
      <div className="panel">
        {text ? <pre data-verbatim="source">{text}</pre> : null}
        {s.url ? (
          <a href={s.url} rel="noopener noreferrer" target="_blank">
            <L k="source" />
          </a>
        ) : null}
      </div>
    </details>
  );
}
