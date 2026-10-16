"""The prefill/decode sweep: fresh cache-proof prompts at several context lengths.

Each run carries a unique nonce at the very front of the prompt, so no server can
match a cached prefix and the prefill number is real work. Prompt size is reached
by calibration (two tiny requests give tokens-per-word), then a deterministic
word list fills the body. A failed run is retried once before being recorded.
"""
from __future__ import annotations

import random
import time

from llmbench.client import Client
from llmbench.util import stat

WORDS = ("river stone market lantern copper winter garden signal harbor needle orbit "
         "velvet canyon ember ledger quartz meadow anchor violet thunder").split()
TAIL = "\n\nSummarize what kinds of words appear in the list, in five sentences."


def word_prompt(words: int) -> str:
    """A deterministic prompt of roughly `words` words (same body every call)."""
    rng = random.Random(7)
    body = " ".join(rng.choice(WORDS) for _ in range(words))
    return "Here is a list of words:\n" + body + TAIL


def _token_count(rec: dict, text: str) -> int:
    if rec.get("prompt_tokens"):
        return int(rec["prompt_tokens"])
    return max(1, len(text) // 4)  # rough fallback when the server omits usage


def calibrate(client: Client, nonce: str) -> tuple[float, int]:
    """(tokens per word, fixed overhead) from two tiny tagged requests."""
    pts = []
    for words in (200, 800):
        text = "[cal %s %d]\n" % (nonce, words) + word_prompt(words)
        rec = client.chat([{"role": "user", "content": text}], 1)
        pts.append(_token_count(rec, text))
    slope = (pts[1] - pts[0]) / 600 if pts[1] > pts[0] else 1.1
    overhead = max(0, pts[0] - 200 * slope)
    return slope, int(overhead)


def summarize(records: list[dict], lengths: list[int]) -> list[dict]:
    summary = []
    for target in lengths:
        ok = [r for r in records if r["target"] == target and r.get("ok")]
        summary.append({
            "target_tokens": target,
            "runs_attempted": sum(1 for r in records if r["target"] == target),
            "runs_ok": len(ok),
            "prompt_tokens": stat(r.get("prompt_tokens") for r in ok),
            "reused_tokens": stat(r.get("reused_tokens") for r in ok),
            "prefill_tps": stat(r.get("prefill_tps") for r in ok),
            "decode_tps": stat(r.get("decode_tps") for r in ok),
            "ttft_s": stat(r.get("ttft_s") for r in ok),
            "total_s": stat(r.get("total_s") for r in ok),
            "generated_tokens": stat(r.get("generated_tokens") for r in ok),
        })
    return summary


def run(client: Client, lengths: list[int], runs: int, max_tokens: int,
        temperature: float = 0.0, thinking: bool = False, retries: int = 1,
        log=print, skip=None, on_record=None) -> dict:
    """Send the sweep. Returns {calibration, records}.

    `skip`: a set of (target, run) pairs already measured (for --resume).
    `on_record(rec)`: called after every attempt so the caller can save progress
    incrementally - useful when the server under test is flaky.
    """
    skip = skip or set()
    nonce = str(int(time.time()))
    log("  warming up ...")
    client.chat([{"role": "user", "content": "[warm %s] Say hello." % nonce}], 8)
    tpw, overhead = calibrate(client, nonce)
    log("  calibrated: %.3f tokens/word, overhead %d tokens" % (tpw, overhead))
    client.chat([{"role": "user", "content": "[warm2 %s]\n" % nonce + word_prompt(1000)}], 8)

    records = []
    for target in lengths:
        words = max(50, int((target - overhead) / tpw))
        for run_i in range(runs):
            if (target, run_i) in skip:
                log("  L=%-7d r%d: skipped (already measured)" % (target, run_i))
                continue
            body = word_prompt(words)
            text = "[bench %s L%d r%d]\n%s" % (nonce, target, run_i, body)
            rec = client.chat([{"role": "user", "content": text}], max_tokens, temperature, thinking)
            attempts = 1
            while not rec.get("ok") and attempts <= retries:
                log("    retry L=%d r%d after: %s" % (target, run_i, rec.get("error")))
                time.sleep(2)
                rec = client.chat([{"role": "user", "content": text}], max_tokens, temperature, thinking)
                attempts += 1
            rec.update(target=target, run=run_i, attempts=attempts)
            records.append(rec)
            if on_record:
                on_record(rec)
            if rec["ok"]:
                acc = "" if not rec.get("drafts_offered") else \
                    "  draft %s/%s" % (rec.get("drafts_accepted"), rec.get("drafts_offered"))
                log("  L=%-7d r%d: prompt %6s tok @ %8s tok/s | decode %s tok/s | ttft %ss%s"
                    % (target, run_i, rec.get("prompt_tokens"), rec.get("prefill_tps"),
                       rec.get("decode_tps"), rec.get("ttft_s"), acc))
            else:
                log("  L=%-7d r%d: FAILED %s" % (target, run_i, rec.get("error")))

    return {"calibration": {"tokens_per_word": tpw, "overhead_tokens": overhead}, "records": records}

