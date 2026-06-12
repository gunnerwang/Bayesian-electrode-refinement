#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal OpenAI->Ollama translating proxy that disables hybrid-model "thinking".

Why: Qwen3.5 is a hybrid reasoning model. Ollama's OpenAI-compatible endpoint
ignores the `think` flag, so by default every advisor call burns ~6k hidden
reasoning tokens (~100 s/call). Ollama's NATIVE /api/chat honors `think:false`
(0 reasoning tokens, <1 s/call, identical JSON answer format). This proxy lets
the paper pipeline keep speaking the standard OpenAI protocol (selected purely
via OPENAI_BASE_URL) while the model is served in non-thinking instruct mode —
a serving configuration, not a pipeline change; disclosed in run provenance.

Translation (non-streaming chat completions only — all the advisor needs):
  POST /v1/chat/completions  ->  POST {ollama}/api/chat  (think:false injected,
      temperature / max_tokens|max_completion_tokens mapped to options)
  GET  /v1/*                 ->  forwarded verbatim (preflight uses /v1/models)

Usage:  python scripts/nothink_proxy.py [--port 11435] [--ollama http://127.0.0.1:11434]
"""
import json, time, argparse, urllib.request, urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

OLLAMA = 'http://127.0.0.1:11434'


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # quiet
        pass

    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        try:
            with urllib.request.urlopen(OLLAMA + self.path, timeout=30) as r:
                data = r.read()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except Exception as e:
            self._send(502, {'error': {'message': f'proxy: {e}', 'type': 'proxy_error'}})

    def do_POST(self):
        if not self.path.rstrip('/').endswith('/chat/completions'):
            self._send(404, {'error': {'message': f'proxy: unsupported path {self.path}',
                                       'type': 'proxy_error'}})
            return
        try:
            n = int(self.headers.get('Content-Length', 0))
            req = json.loads(self.rfile.read(n))
            native = {
                'model': req['model'],
                'messages': req['messages'],
                'think': False,
                'stream': False,
                'options': {},
            }
            if 'temperature' in req:
                native['options']['temperature'] = req['temperature']
            cap = req.get('max_completion_tokens', req.get('max_tokens'))
            if cap:
                native['options']['num_predict'] = cap
            body = json.dumps(native).encode()
            r = urllib.request.Request(OLLAMA + '/api/chat', data=body,
                                       headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(r, timeout=300) as resp:
                out = json.load(resp)
            if out.get('error'):
                self._send(502, {'error': {'message': str(out['error']), 'type': 'upstream_error'}})
                return
            msg = out.get('message', {})
            self._send(200, {
                'id': f'chatcmpl-proxy-{int(time.time()*1000)}',
                'object': 'chat.completion',
                'created': int(time.time()),
                'model': out.get('model', req['model']),
                'choices': [{
                    'index': 0,
                    'message': {'role': msg.get('role', 'assistant'),
                                'content': msg.get('content', '')},
                    'finish_reason': 'length' if out.get('done_reason') == 'length' else 'stop',
                }],
                'usage': {
                    'prompt_tokens': out.get('prompt_eval_count', 0),
                    'completion_tokens': out.get('eval_count', 0),
                    'total_tokens': out.get('prompt_eval_count', 0) + out.get('eval_count', 0),
                },
            })
        except urllib.error.URLError as e:
            self._send(502, {'error': {'message': f'proxy upstream: {e}', 'type': 'proxy_error'}})
        except Exception as e:
            self._send(500, {'error': {'message': f'proxy: {type(e).__name__}: {e}', 'type': 'proxy_error'}})


def main():
    global OLLAMA
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=11435)
    ap.add_argument('--ollama', type=str, default=OLLAMA)
    args = ap.parse_args()
    OLLAMA = args.ollama.rstrip('/')
    srv = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    print(f'nothink proxy on 127.0.0.1:{args.port} -> {OLLAMA} (think:false injected)', flush=True)
    srv.serve_forever()


if __name__ == '__main__':
    main()
