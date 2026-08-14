#!/usr/bin/env python3
"""Small HTTP server to receive test webhooks and write JSON bodies to a file.
Usage: python3 test_webhook_server.py
Writes received payloads appended to ./webhook_received.jsonl
"""
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import os

OUT = os.path.join(os.path.dirname(__file__), 'webhook_received.jsonl')

class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get('content-length', 0))
        body = self.rfile.read(length) if length else b''
        try:
            parsed = json.loads(body.decode('utf-8')) if body else None
        except Exception:
            parsed = body.decode('utf-8', errors='replace')
        with open(OUT, 'a') as fh:
            fh.write(json.dumps({'path': self.path, 'payload': parsed}) + "\n")
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'OK')

if __name__ == '__main__':
    server = HTTPServer(('127.0.0.1', 9000), Handler)
    print('Listening on http://127.0.0.1:9000')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()
