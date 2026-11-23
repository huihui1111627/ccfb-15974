"""Web 展示: 机房平面热力图、空调设定/出力、热点预测与告警。

零依赖(标准库 http.server), 启动: python3 -m simem.app [port]
数据来源: results/run_*.json (由 acceptance.py 生成) + 可选的实时仿真接口。
"""
from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(__file__))
RESULTS = os.path.join(ROOT, 'results')
STATIC = os.path.join(ROOT, 'static')


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype='application/json'):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split('?')[0]
        if path == '/' or path == '/index.html':
            return self._file(os.path.join(STATIC, 'index.html'), 'text/html')
        if path.startswith('/static/'):
            return self._file(os.path.join(STATIC, path[8:]))
        if path == '/api/summary':
            return self._json(os.path.join(RESULTS, 'summary.json'))
        if path == '/api/placement':
            return self._json(os.path.join(RESULTS, 'placement.json'))
        if path.startswith('/api/run/'):
            tag = path.split('/')[-1]
            return self._json(os.path.join(RESULTS, f'run_{tag}.json'))
        if path == '/api/runs':
            runs = []
            if os.path.isdir(RESULTS):
                for f in os.listdir(RESULTS):
                    if f.startswith('run_') and f.endswith('.json'):
                        runs.append(f[4:-5])
            return self._send(200, sorted(runs))
        return self._send(404, {'error': 'not found'})

    def _file(self, p, ctype=None):
        if not os.path.isfile(p):
            return self._send(404, 'not found', 'text/plain')
        ext = os.path.splitext(p)[1].lower()
        m = {'.html': 'text/html; charset=utf-8', '.js': 'application/javascript',
             '.css': 'text/css', '.json': 'application/json', '.png': 'image/png'}
        with open(p, 'rb') as f:
            self._send(200, f.read(), ctype or m.get(ext, 'application/octet-stream'))

    def _json(self, p):
        if not os.path.isfile(p):
            return self._send(404, {'error': 'no data', 'path': p})
        with open(p, 'rb') as f:
            self._send(200, f.read())


def serve(port=8000):
    srv = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    print(f'serving on http://127.0.0.1:{port}')
    srv.serve_forever()


if __name__ == '__main__':
    import sys
    serve(int(sys.argv[1]) if len(sys.argv) > 1 else 8000)
