// Plain static file server for tests and previews: no SPA fallback, no runtime logic.
// Serves dist/ under BASE_PATH (default "/"); a directory serves its index.html.
import http from 'node:http';
import { readFile, stat } from 'node:fs/promises';
import path from 'node:path';

const root = path.resolve(process.env.DIST ?? 'dist');
const base = process.env.BASE_PATH ?? '/';
const port = Number(process.env.PORT ?? 4173);
const types = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript', '.css': 'text/css', '.json': 'application/json', '.woff2': 'font/woff2', '.txt': 'text/plain', '.svg': 'image/svg+xml' };
http
  .createServer(async (req, res) => {
    try {
      const url = new URL(req.url ?? '/', 'http://x');
      let p = decodeURIComponent(url.pathname);
      if (!p.startsWith(base)) return void res.writeHead(404).end();
      let file = path.resolve(root, p.slice(base.length));
      if (!file.startsWith(root)) return void res.writeHead(403).end();
      if ((await stat(file).catch(() => null))?.isDirectory()) file = path.join(file, 'index.html');
      const body = await readFile(file);
      res.writeHead(200, { 'Content-Type': types[path.extname(file)] ?? 'application/octet-stream' }).end(body);
    } catch {
      // Like GitHub Pages: unknown paths get dist/404.html with status 404.
      const page = await readFile(path.join(root, '404.html')).catch(() => null);
      if (page) res.writeHead(404, { 'Content-Type': 'text/html; charset=utf-8' }).end(page);
      else res.writeHead(404, { 'Content-Type': 'text/plain' }).end('not found');
    }
  })
  .listen(port, '127.0.0.1', () => console.log(`static server http://127.0.0.1:${port}${base}`));
