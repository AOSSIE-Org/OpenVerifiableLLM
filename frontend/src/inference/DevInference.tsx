import { useEffect, useRef, useState } from 'react';
import { Button, Copy, Label } from '../components';
import { capabilities, createMockAdapter, MOCK_LABEL, type MockScenario } from './mock';
import { checkedGenerate, type GenerationResponse, type Message } from './adapter';

export default function DevInference() {
  const [mode, setMode] = useState<'base' | 'conversation'>('base');
  const [prompt, setPrompt] = useState(''); const [history, setHistory] = useState<Message[]>([]);
  const [scenario, setScenario] = useState<MockScenario>('success');
  const [status, setStatus] = useState(''); const [busy, setBusy] = useState(false);
  const [output, setOutput] = useState<GenerationResponse | null>(null);
  const controller = useRef<AbortController | null>(null);
  useEffect(() => () => controller.current?.abort(), []);
  const count = prompt.length + (mode === 'conversation' ? history.reduce((sum, m) => sum + m.content.length, 0) : 0);
  async function run() {
    controller.current?.abort(); const current = new AbortController(); controller.current = current;
    setOutput(null); setBusy(true); setStatus('Loading scripted response. No model is running.');
    const messages: Message[] = [...history, { role: 'user', content: prompt }];
    try {
      const response = await checkedGenerate(createMockAdapter(scenario), {
        release: capabilities.releases[mode], mode, prompt, messages,
        settings: { strategy: 'greedy', maxOutputTokens: 128 },
      }, current.signal);
      if (controller.current !== current) return;
      if (!response.text) { setStatus('Empty output. Retry or change the development scenario.'); return; }
      setOutput(response); setStatus('Scripted example received. No verification was performed.');
      if (mode === 'conversation') { setHistory([...messages, { role: 'assistant', content: response.text }]); setPrompt(''); }
    } catch (e) {
      if (controller.current !== current) return;
      setStatus(e instanceof DOMException && e.name === 'AbortError' ? 'Request cancelled. A future server must define whether cancellation stops compute.' : e instanceof Error ? e.message : 'Unknown mock error.');
    } finally { if (controller.current === current) setBusy(false); }
  }
  function clear() { controller.current?.abort(); controller.current = null; setBusy(false); setPrompt(''); setHistory([]); setOutput(null); setStatus('History cleared from memory.'); }
  return <section className="dev-preview"><Label>Development-only mock / no released model</Label><h2>Exercise the interface.</h2><p>{MOCK_LABEL}</p>
    <form className="dev-form" onSubmit={e => { e.preventDefault(); void run(); }}>
      <label htmlFor="mock-mode">Mode</label><select id="mock-mode" value={mode} disabled={busy} onChange={e => { clear(); setMode(e.target.value as typeof mode); }}><option value="base">Base completion</option><option value="conversation">Conversation</option></select>
      <label htmlFor="mock-scenario">Development scenario</label><select id="mock-scenario" value={scenario} disabled={busy} onChange={e => setScenario(e.target.value as MockScenario)}>{['success', 'error', 'empty', 'timeout', 'identity-mismatch'].map(s => <option key={s}>{s}</option>)}</select>
      {history.length > 0 && <ol className="relation-list" aria-label="Conversation history">{history.map((m, i) => <li key={i}><strong>{m.role}</strong><p>{m.role === 'assistant' && <span>{MOCK_LABEL}<br /></span>}{m.content}</p></li>)}</ol>}
      <label htmlFor="mock-prompt">{mode === 'base' ? 'Completion prompt' : 'User message'}</label>
      <textarea id="mock-prompt" value={prompt} maxLength={capabilities.inputLimit} disabled={busy} aria-describedby="input-limit" onChange={e => setPrompt(e.target.value)} />
      <p className="small" id="input-limit">{count} / {capabilities.inputLimit} UTF-16 code units, including history. Fixed greedy decoding; at most 128 output tokens.</p>
      <div className="actions"><button className="button" type="submit" disabled={busy || !prompt.trim() || count > capabilities.inputLimit}>Generate scripted example</button>{busy && <Button onClick={() => controller.current?.abort()}>Cancel request</Button>}<button type="button" onClick={clear}>Clear history</button></div>
    </form>
    <p role="status" aria-live="polite">{status}</p>
    {!busy && status && !output && <Button disabled={!prompt.trim() || count > capabilities.inputLimit} onClick={() => void run()}>Retry example</Button>}
    {output && <div className="stack"><p className="eyebrow">{MOCK_LABEL}</p><p className="mock-response">{output.text}</p><Copy value={output.text} label="Copy response" /><details><summary>Response identity and settings</summary><pre>{JSON.stringify({ release: output.release, settings: output.settings, locallyReproduced: false, mock: true }, null, 2)}</pre></details></div>}
  </section>;
}
