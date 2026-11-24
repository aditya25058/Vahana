"""OpenAI-compatible HTTP server for a loaded Vahana model (standard library only).

Endpoints: GET /health, GET /v1/models, POST /v1/completions, POST /v1/chat/completions.
Request fields: prompt | messages, max_tokens (16), temperature (0 = greedy), top_p (1), seed (0), stop (str or
list), stream (false).
v0.4-b: requests share decode steps through engine.Engine (continuous batching); every response is bit-identical to
running that request alone (v0.4-c: also for models with recurrent / linear-attention layers).

Determinism: greedy decoding is bit-exact. With temperature > 0, tokens are drawn on the host from the step's fp32
logits with a torch.Generator seeded by `seed`, so the same request always returns the same text.
"""
from __future__ import annotations

import json
import queue
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch

from . import decode


def make_sampler(temperature: float, top_p: float, seed: int):
    if not temperature or temperature <= 0:
        return None
    g = torch.Generator().manual_seed(int(seed))

    def sample(logits: torch.Tensor) -> torch.Tensor:
        p = torch.softmax(logits.detach().to("cpu", torch.float64) / float(temperature), -1)
        if top_p < 1.0:
            sp, si = torch.sort(p, -1, descending=True)
            keep = sp.cumsum(-1) - sp < top_p
            sp = torch.where(keep, sp, torch.zeros_like(sp))
            p = torch.zeros_like(p).scatter_(-1, si, sp)
        return torch.multinomial(p / p.sum(-1, keepdim=True), 1, generator=g)
    return sample


class _State:
    def __init__(self, model, tok, name, slots=16, max_len=4096):
        self.model, self.tok, self.name = model, tok, name
        self.lock = threading.Lock()
        self.engine = None
        from . import engine as _engine
        if slots and slots > 1:
            self.engine = _engine.make(model, slots=slots, max_len=max_len)     # None: one request at a time
        if self.engine is not None:
            self.cv = threading.Condition()
            threading.Thread(target=self._loop, daemon=True, name="vahana-engine").start()
        eos = tok.eos_token_id
        gen_eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
        ids = set()
        for e in (eos, gen_eos):
            if isinstance(e, int):
                ids.add(e)
            elif isinstance(e, (list, tuple)):
                ids.update(e)
        self.eos = sorted(ids)

    def _loop(self):
        eng = self.engine
        while True:
            with self.cv:
                while eng.idle():
                    self.cv.wait()
            try:
                eng.step()
            except Exception as e:  # noqa: BLE001 - end the affected requests, keep serving
                eng.fail_all(f"{type(e).__name__}: {str(e)[:300]}")

    def submit(self, ids, max_new, sampler, on_token):
        r = self.engine.submit(ids, max_new, eos_token_id=self.eos, sampler=sampler, on_token=on_token)
        with self.cv:
            self.cv.notify()
        return r


def _run(st: _State, text: str, req: dict, emit=None):
    """Generate for one prompt; emit(delta_text) is called as text arrives. Returns (text, finish, n_in, n_out)."""
    ids = st.tok(text, return_tensors="pt").input_ids.to(getattr(st.model, "_vahana_device", "cuda"))
    max_tokens = int(req.get("max_tokens") or 16)
    stops = req.get("stop") or []
    stops = [stops] if isinstance(stops, str) else list(stops)
    sampler = make_sampler(float(req.get("temperature") or 0), float(req.get("top_p") or 1.0), int(req.get("seed") or 0))
    out_ids: list[int] = []
    state = {"text": "", "sent": 0, "finish": "length"}

    def on_token(t):
        tid = int(t[0, 0])
        if tid in st.eos:
            state["finish"] = "stop"
            return True
        out_ids.append(tid)
        state["text"] = st.tok.decode(out_ids, skip_special_tokens=True)
        for s in stops:
            k = state["text"].find(s)
            if k >= 0:
                state["text"] = state["text"][:k]
                state["finish"] = "stop"
                break
        if emit is not None and len(state["text"]) > state["sent"] and not state["text"].endswith("�"):
            emit(state["text"][state["sent"]:])
            state["sent"] = len(state["text"])
        return state["finish"] == "stop"

    if st.engine is not None:
        # The engine thread calls on_token; text goes back through a queue so a slow or closed client can never stall
        # or break the shared decode loop.
        q: queue.Queue = queue.Queue()
        user_emit, emit = emit, (lambda d: q.put(d))
        req = st.submit(ids[0].tolist(), max_tokens, sampler, lambda t: on_token(torch.tensor([[t]])))
        while True:
            try:
                d = q.get(timeout=0.05)
            except queue.Empty:
                if req.done.is_set() and q.empty():
                    break
                continue
            if user_emit is not None:
                user_emit(d)
        if req.error:
            raise RuntimeError(req.error)
        emit = user_emit
    else:
        with st.lock:
            decode.generate(st.model, ids, max_new_tokens=max_tokens, sampler=sampler, on_token=on_token)
    if emit is not None and len(state["text"]) > state["sent"]:
        emit(state["text"][state["sent"]:])
    return state["text"], state["finish"], ids.shape[1], len(out_ids)


def serve(model, tok, host: str = "127.0.0.1", port: int = 8000, name: str = "vahana-model", slots: int = 16,
          max_len: int = 4096):
    st = _State(model, tok, name, slots=slots, max_len=max_len)

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *a):          # quiet
            pass

        def _json(self, code, obj):
            b = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            if self.path == "/health":
                return self._json(200, {"status": "ok"})
            if self.path == "/v1/models":
                return self._json(200, {"object": "list", "data": [{"id": st.name, "object": "model",
                                                                     "owned_by": "vahana"}]})
            self._json(404, {"error": {"message": "not found"}})

        def do_POST(self):
            try:
                self._post()
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as e:  # noqa: BLE001 - report, keep serving
                try:
                    self._json(500, {"error": {"message": f"{type(e).__name__}: {str(e)[:300]}"}})
                except Exception:  # noqa: BLE001
                    pass

        def _post(self):
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            except json.JSONDecodeError:
                return self._json(400, {"error": {"message": "invalid JSON"}})
            chat = self.path == "/v1/chat/completions"
            if self.path not in ("/v1/completions", "/v1/chat/completions"):
                return self._json(404, {"error": {"message": "not found"}})
            if chat:
                msgs = req.get("messages")
                if not isinstance(msgs, list):
                    return self._json(400, {"error": {"message": "messages must be a list"}})
                try:
                    text = st.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                except ValueError as e:              # base models without a chat template
                    return self._json(400, {"error": {"message": f"chat template unavailable: {str(e)[:200]}"}})
            else:
                text = req.get("prompt")
                if not isinstance(text, str):
                    return self._json(400, {"error": {"message": "prompt must be a string (one prompt per request)"}})
            rid = ("chatcmpl-" if chat else "cmpl-") + uuid.uuid4().hex[:24]
            created = int(time.time())
            obj = "chat.completion" if chat else "text_completion"
            if req.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()

                def send(payload):
                    data = f"data: {payload}\n\n".encode()
                    self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
                    self.wfile.flush()

                def emit(delta):
                    ch = {"index": 0, "delta": {"content": delta}, "finish_reason": None} if chat else \
                        {"index": 0, "text": delta, "finish_reason": None}
                    send(json.dumps({"id": rid, "object": obj + ".chunk" if chat else obj, "created": created,
                                     "model": st.name, "choices": [ch]}))
                _, finish, _, _ = _run(st, text, req, emit)
                end = {"index": 0, "delta": {}, "finish_reason": finish} if chat else \
                    {"index": 0, "text": "", "finish_reason": finish}
                send(json.dumps({"id": rid, "object": obj + ".chunk" if chat else obj, "created": created,
                                 "model": st.name, "choices": [end]}))
                send("[DONE]")
                self.wfile.write(b"0\r\n\r\n")
                return
            out, finish, n_in, n_out = _run(st, text, req)
            choice = {"index": 0, "message": {"role": "assistant", "content": out}, "finish_reason": finish} \
                if chat else {"index": 0, "text": out, "finish_reason": finish}
            self._json(200, {"id": rid, "object": obj, "created": created, "model": st.name, "choices": [choice],
                             "usage": {"prompt_tokens": n_in, "completion_tokens": n_out,
                                       "total_tokens": n_in + n_out}})

    srv = ThreadingHTTPServer((host, port), H)
    mode = f"continuous batching, {slots} slots" if st.engine is not None else "one request at a time"
    print(f"vahana: serving {name} on http://{host}:{port}/v1 ({mode})", flush=True)
    srv.serve_forever()
