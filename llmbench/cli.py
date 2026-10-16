"""Command line: detect device -> framework -> model, sweep, write JSON."""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

from llmbench import __version__, device, framework as framework_mod
from llmbench.client import Client
from llmbench.sweep import run as run_sweep, summarize
from llmbench.util import normalize_base, slug

REPO = Path(__file__).resolve().parents[1]
DEFAULT_LENGTHS = "4096,16384,32768,65536,100000"


def _device_overrides(a) -> dict:
    return {"name": a.device_name, "vendor": a.device_vendor,
            "memory_total_gib": a.device_memory_gib, "architecture": a.device_arch}


def _framework_overrides(a) -> dict:
    return {"framework": a.framework, "framework_version": a.framework_version,
            "model_name": a.model_name, "quantization": a.quantization,
            "context_length": a.context_length}


def print_target(dev: dict, fw: dict, model: dict, out=sys.stdout) -> None:
    """Device first, then framework, then model - the order the report uses."""
    acc = dev["accelerator"]
    mem = acc.get("memory_total_gib")
    bits = [b for b in (acc.get("vendor"), acc.get("architecture"),
                        ("%s GiB" % mem) if mem else None, acc.get("memory_note")) if b]
    print("DEVICE", file=out)
    print("  %s%s" % (acc.get("name") or acc.get("kind", "unknown"),
                      ("  [%s]" % ", ".join(bits)) if bits else ""), file=out)
    if acc.get("count", 0) and acc["count"] > 1:
        print("  x%d accelerators" % acc["count"], file=out)
    if acc.get("note"):
        print("  note: %s" % acc["note"], file=out)
    host = dev["host"]
    cpu = host["cpu"]
    print("  host: %s, %s cores, %s GiB RAM, %s" % (
        cpu.get("name") or "unknown CPU", cpu.get("logical_cores") or "?",
        host.get("ram_total_gib") or "?", host["os"].get("distro") or host["os"].get("system")), file=out)
    print("FRAMEWORK", file=out)
    print("  %s %s  %s  (%s)" % (fw.get("name") or "unknown", fw.get("version") or "",
                                 fw.get("endpoint"), "openai/chat-completions"), file=out)
    print("MODEL", file=out)
    print("  %s  quant=%s  context=%s" % (model.get("id") or "unknown",
                                          model.get("quantization") or "?",
                                          model.get("context_length") or "?"), file=out)


def build_result(args, dev, fw, model, lengths, calibration, records) -> dict:
    now = datetime.datetime.now().astimezone()
    records = sorted(records, key=lambda r: (r.get("target", 0), r.get("run", 0)))
    return {
        "schema": 1,
        "tool": {"name": "llm-bench", "version": __version__},
        "generated_at": now.isoformat(timespec="seconds"),
        "label": args.label,
        # order matters: device, then framework, then model
        "device": dev,
        "framework": fw,
        "model": model,
        "benchmark": {
            "type": "prefill_decode_sweep",
            "transport": "openai/chat-completions",
            "lengths": lengths,
            "runs": args.runs,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "thinking": bool(args.think),
            "prompt_source": "deterministic word list",
            "cache": "fresh per run (unique nonce prefix)",
            "timing_source": "server timings when present, else client",
            "calibration": calibration,
        },
        "results": summarize(records, lengths),
        "requests": records,
        "notes": [],
        "limitations": [
            "Speeds are the server's own timings when it reports them (the *_source field says which).",
            "A warmed expert/weight cache and a fresh prompt are different conditions; reused_tokens "
            "shows whether any prefix was reused (0 means none).",
        ],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="llmbench",
        description="Benchmark any LLM framework, model, and device over an OpenAI-compatible API.")
    ap.add_argument("--url", default="http://127.0.0.1:8080", help="base URL of the running server")
    ap.add_argument("--api-key", default="", help="bearer token, if the server needs one")
    ap.add_argument("--model", default=None, help="model id to put in the request body")
    ap.add_argument("--lengths", default=DEFAULT_LENGTHS, help="comma-separated prompt-token targets")
    ap.add_argument("--runs", type=int, default=3, help="fresh runs per length (default 3)")
    ap.add_argument("--max-tokens", type=int, default=128, help="output cap per request")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--think", action="store_true", help="leave thinking on (default off)")
    ap.add_argument("--timeout", type=float, default=7200)
    ap.add_argument("--retries", type=int, default=1, help="retries for a failed run")
    ap.add_argument("--out", default=str(REPO / "results"), help="directory for the JSON result")
    ap.add_argument("--json", default=None, help="exact path for the JSON result")
    ap.add_argument("--resume", action="store_true",
                    help="reuse an existing result file, re-running only missing runs")
    ap.add_argument("--stdout", action="store_true", help="also print the JSON to stdout")
    ap.add_argument("--detect", action="store_true", help="only print device/framework/model and exit")
    ap.add_argument("--label", default=None, help="freeform label for this run")
    ap.add_argument("--framework", default=None)
    ap.add_argument("--framework-version", default=None)
    ap.add_argument("--model-name", default=None)
    ap.add_argument("--quantization", default=None)
    ap.add_argument("--context-length", type=int, default=None)
    ap.add_argument("--device-name", default=None)
    ap.add_argument("--device-vendor", default=None)
    ap.add_argument("--device-arch", default=None)
    ap.add_argument("--device-memory-gib", type=float, default=None)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)

    base = normalize_base(a.url)
    log = (lambda *x: None) if a.quiet else (lambda *x: print(*x, flush=True))

    log("probing %s ..." % base)
    dev = device.collect(_device_overrides(a))
    fm = framework_mod.probe(base, a.api_key, _framework_overrides(a))
    fw, model = fm["framework"], fm["model"]
    print_target(dev, fw, model)

    if a.detect:
        return 0

    lengths = [int(x) for x in a.lengths.split(",") if x.strip()]
    model_id = a.model or model.get("id") or "llm"
    client = Client(base, a.api_key, model_id, a.timeout)

    now = datetime.datetime.now().astimezone()
    out_path = Path(a.json) if a.json else Path(a.out) / ("%s-%s-%s-%s.json" % (
        now.strftime("%Y-%m-%d"), slug(dev["accelerator"].get("name") or "device"),
        slug(fw.get("name")), slug(model.get("id"))))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # resume: keep every (target, run) already measured in a previous file
    by_key: dict[tuple, dict] = {}
    calibration = None
    if a.resume and out_path.exists():
        try:
            prev = json.loads(out_path.read_text(encoding="utf-8"))
            calibration = (prev.get("benchmark") or {}).get("calibration")
            for r in prev.get("requests", []):
                if r.get("ok"):
                    by_key[(r.get("target"), r.get("run"))] = r
            log("resuming from %s: %d runs already measured" % (out_path, len(by_key)))
        except Exception as e:  # noqa: BLE001
            log("could not read %s (%s); starting fresh" % (out_path, e))

    def persist():
        result = build_result(a, dev, fw, model, lengths, calibration, list(by_key.values()))
        out_path.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")

    def on_record(rec):
        by_key[(rec.get("target"), rec.get("run"))] = rec
        persist()

    log("\nprefill/decode sweep: lengths=%s, runs=%d, max_tokens=%d, fresh prompts (no cache reuse)"
        % (lengths, a.runs, a.max_tokens))
    sweep = run_sweep(client, lengths, a.runs, a.max_tokens, a.temperature, a.think, a.retries,
                      log=log, skip=set(by_key), on_record=on_record)
    calibration = calibration or sweep["calibration"]

    result = build_result(a, dev, fw, model, lengths, calibration, list(by_key.values()))
    out_path.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")

    def f(s, d=1):
        return "-" if not s else "%.*f (%.*f-%.*f)" % (d, s["median"], d, s["min"], d, s["max"])

    table = ["| target ctx | actual prompt tok | reused | prefill tok/s | decode tok/s | TTFT s | total s | runs |",
             "| ---: | ---: | ---: | --- | --- | --- | --- | ---: |"]
    for row in result["results"]:
        table.append("| %s | %s | %s | %s | %s | %s | %s | %d/%d |" % (
            row["target_tokens"], f(row["prompt_tokens"], 0), f(row["reused_tokens"], 0),
            f(row["prefill_tps"]), f(row["decode_tps"], 2), f(row["ttft_s"], 2),
            f(row["total_s"], 2), row["runs_ok"], row["runs_attempted"]))
    print("\n" + "\n".join(table), flush=True)

    log("\nwrote %s" % out_path)
    if a.stdout:
        print(json.dumps(result, indent=1))
    return 0
