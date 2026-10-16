"""One streamed OpenAI `/v1/chat/completions` request, reduced to timings.

Speeds are taken from the server's own `timings` block (llama.cpp's names, which
Strata and llama.cpp both emit) when present, and otherwise computed on the
client. Every number records its source so the two are never confused.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request


class Client:
    def __init__(self, base: str, api_key: str = "", model: str = "llm", timeout: float = 7200.0):
        self.base = base.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def _post(self, url: str, body: dict, timeout: float):
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)
        return urllib.request.urlopen(req, timeout=timeout)

    def chat(self, messages: list, max_tokens: int, temperature: float = 0.0,
             thinking: bool = False, extra: dict | None = None) -> dict:
        """Send one request and return a timing record.

        Some servers reject the Qwen-style thinking fields; on a 400/422 the
        request is retried once with a minimal body and the retry is noted.
        """
        body = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        if not thinking:
            body["reasoning_effort"] = "none"
            body["chat_template_kwargs"] = {"enable_thinking": False}
        if extra:
            body.update(extra)

        try:
            rec = self._stream(body)
        except urllib.error.HTTPError as e:
            if e.code in (400, 404, 422) and ("reasoning_effort" in body):
                minimal = {"model": self.model, "messages": messages, "max_tokens": max_tokens,
                           "temperature": temperature, "stream": True}
                rec = self._stream(minimal)
                rec["notes"] = ["retried without thinking fields after HTTP %d" % e.code]
            else:
                rec = _error("HTTP %s: %r" % (e.code, e.read()[:200]))
        except Exception as e:  # noqa: BLE001
            rec = _error(repr(e)[:300])
        return rec

    def _stream(self, body: dict) -> dict:
        t0 = time.perf_counter()
        first = None
        gen_pieces = 0
        usage = timings = None
        err = None
        blank_first = "none"
        with self._post(self.base + "/v1/chat/completions", body, self.timeout) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    j = json.loads(payload)
                except ValueError:
                    continue
                if j.get("error"):
                    err = str(j["error"])[:300]
                for ch in j.get("choices") or []:
                    dl = ch.get("delta") or {}
                    piece = (dl.get("content") or "") + (dl.get("reasoning_content") or "")
                    if piece:
                        if first is None:
                            first = time.perf_counter() - t0
                            blank_first = "reasoning" if dl.get("reasoning_content") else "content"
                        gen_pieces += 1
                if j.get("usage"):
                    usage = j["usage"]
                if j.get("timings"):
                    timings = j["timings"]
        total = time.perf_counter() - t0
        return self._digest(usage, timings, first, blank_first, gen_pieces, total, err)

    @staticmethod
    def _digest(usage, timings, first, blank_first, pieces, total, err) -> dict:
        u = usage or {}
        t = timings or {}
        gen = u.get("completion_tokens") or pieces or 0
        prompt_tokens = u.get("prompt_tokens")
        reused = t.get("cache_n")
        cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
        if reused is None:
            reused = cached

        prefill_tps, prefill_src = t.get("prompt_per_second"), "engine"
        if prefill_tps is None and first and prompt_tokens:
            prefill_tps, prefill_src = round(prompt_tokens / first, 2), "client"
        if prefill_tps is None:
            prefill_src = None

        decode_tps, decode_src = t.get("predicted_per_second"), "engine"
        if decode_tps is None and first is not None and gen > 1 and total > first:
            decode_tps, decode_src = round((gen - 1) / (total - first), 2), "client"
        if decode_tps is None:
            decode_src = None

        return {
            "ok": err is None and first is not None,
            "error": err,
            "ttft_s": None if first is None else round(first, 3),
            "total_s": round(total, 3),
            "first_token_kind": blank_first if first is not None else None,
            "prompt_tokens": prompt_tokens,
            "reused_tokens": reused,
            "read_tokens": t.get("prompt_n"),
            "generated_tokens": gen,
            "prefill_tps": prefill_tps,
            "prefill_tps_source": prefill_src,
            "decode_tps": decode_tps,
            "decode_tps_source": decode_src,
            "prompt_ms": t.get("prompt_ms"),
            "decode_ms": t.get("predicted_ms"),
            "drafts_offered": t.get("draft_n"),
            "drafts_accepted": t.get("draft_n_accepted"),
        }


def _error(msg: str) -> dict:
    return {
        "ok": False, "error": msg, "ttft_s": None, "total_s": None, "first_token_kind": None,
        "prompt_tokens": None, "reused_tokens": None, "read_tokens": None, "generated_tokens": 0,
        "prefill_tps": None, "prefill_tps_source": None, "decode_tps": None,
        "decode_tps_source": None, "prompt_ms": None, "decode_ms": None,
        "drafts_offered": None, "drafts_accepted": None,
    }
