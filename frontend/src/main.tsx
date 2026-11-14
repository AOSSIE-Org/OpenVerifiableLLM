import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';
import { applyTheme, initialTheme } from './theme';
import './styles/base.css';

const root = document.getElementById('root');
if (!root) throw new Error('root element missing');

// Before the first paint, so the loading and error states follow the reader's theme too.
applyTheme(initialTheme());

createRoot(root).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
