// Applies the stored theme before first paint. External file so a strict CSP (script-src 'self') allows it.
// Only "light" or "dark" is ever used; any other stored value is ignored. Mirrors src/theme.ts.
(function () {
  var t = 'light';
  try { var v = localStorage.getItem('ovl-theme'); if (v === 'light' || v === 'dark') t = v; } catch { /* storage blocked */ }
  var h = document.documentElement;
  h.setAttribute('data-theme', t);
  h.style.colorScheme = t;
  var m = document.querySelector('meta[name="theme-color"]');
  if (m) m.setAttribute('content', t === 'dark' ? '#1A1816' : '#F0EEE9');
})();
