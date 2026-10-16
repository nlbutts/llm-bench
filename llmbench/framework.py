"""Detect the serving framework and model from an OpenAI-compatible endpoint.

Frameworks differ in what they expose, so this probes a handful of well-known
endpoints and takes the first match. Every guess can be overridden from the
command line, so a server that looks like nothing still gets a full record.
"""
from __future__ import annotations

import re

from llmbench.util import http_get_json, http_get_text


def _metrics(base: str, key: str) -> str | None:
    return http_get_text(base + "/metrics", key, timeout=5)


_QUANT_RE = re.compile(
    r"(ud-)?(iq\d+_[a-z0-9]+|q\d+_k(_[a-z]+)?|q\d+_\d+|q\d+|iq\d+|fp8|f16|bf16|fp16|f32|int8|mxfp4|nvfp4)",
    re.I)


def _quant_from_id(model_id: str | None) -> str | None:
    """Guess a quantization from a model id such as `...-UD-IQ4_XS`."""
    if not model_id:
        return None
    m = _QUANT_RE.search(model_id)
    return (("UD-" if m.group(1) else "") + m.group(2)).upper() if m else None


def probe(base: str, key: str = "", overrides: dict | None = None) -> dict:
    ov = overrides or {}
    fw = {"endpoint": base, "name": None, "version": None, "detected_by": None,
          "dialects": [], "probes": {}, "notes": []}
    model = {"id": None, "quantization": None, "context_length": None, "context_native": None,
             "detected_by": None}

    # --- Strata: /v1/status is the richest, /health also answers
    status = http_get_json(base + "/v1/status", key)
    health = http_get_json(base + "/health", key)
    if status:
        fw["probes"]["v1_status"] = status
    if health:
        fw["probes"]["health"] = health

    is_strata = bool(status and status.get("service") == "strata") or \
                bool(health and health.get("service") == "strata")
    if is_strata:
        fw["name"] = "Strata"
        fw["version"] = (status or {}).get("engine") or (health or {}).get("engine")
        fw["detected_by"] = "/v1/status" if status else "/health"
        fw["dialects"] = list((status or {}).get("dialects") or ["/v1/chat/completions"])
        model["id"] = (status or {}).get("model") or (health or {}).get("model")
        model["detected_by"] = fw["detected_by"]
        ctx = (status or {}).get("context") or {}
        model["context_length"] = (status or {}).get("cache_max_tokens") or ctx.get("native") \
            or (health or {}).get("max_context")
        model["context_native"] = ctx.get("native")

    # --- llama.cpp server: /props
    if not fw["name"]:
        props = http_get_json(base + "/props", key)
        if props and ("default_generation_settings" in props or "n_ctx" in props
                      or "model_path" in props or "total_slots" in props):
            fw["probes"]["props"] = props
            fw["name"] = "llama.cpp (llama-server)"
            fw["detected_by"] = "/props"
            dgs = props.get("default_generation_settings") or {}
            model["context_length"] = dgs.get("n_ctx") or props.get("n_ctx")
            model["detected_by"] = "/props"
            path = props.get("model_path") or ""
            if path:
                model["id"] = path.replace("\\", "/").split("/")[-1]

    # --- Ollama: /api/version + /api/tags
    if not fw["name"]:
        ver = http_get_json(base + "/api/version", key)
        tags = http_get_json(base + "/api/tags", key)
        if ver is not None or tags is not None:
            fw["name"] = "Ollama"
            fw["version"] = (ver or {}).get("version")
            fw["detected_by"] = "/api/version" if ver is not None else "/api/tags"
            fw["dialects"] = ["/v1/chat/completions", "/api/chat"]
            if tags:
                fw["probes"]["api_tags"] = tags
                models = tags.get("models") or []
                if models:
                    model["id"] = models[0].get("name") or models[0].get("model")
                    details = models[0].get("details") or {}
                    model["quantization"] = details.get("quantization_level")
                    model["context_length"] = (models[0].get("model_info") or {}).get(
                        "general.context_length")

    # --- OpenAI /v1/models (vLLM, SGLang, LM Studio, TGI, everything else)
    if not fw["name"] or model["id"] is None:
        models = http_get_json(base + "/v1/models", key)
        if models and isinstance(models.get("data"), list) and models["data"]:
            fw["probes"].setdefault("v1_models", {"ids": [m.get("id") for m in models["data"][:10]]})
            first = models["data"][0]
            if model["id"] is None:
                model["id"] = first.get("id")
                model["detected_by"] = "/v1/models"
            if model["context_length"] is None:
                model["context_length"] = (first.get("meta") or {}).get("n_ctx") \
                    or (first.get("meta") or {}).get("context_length")
            if not fw["name"]:
                fw["name"] = "openai-compatible"
                fw["detected_by"] = "/v1/models"

    # --- /metrics identifies vLLM and TGI without their own endpoints
    if not fw["name"] or fw["name"] == "openai-compatible":
        m = _metrics(base, key)
        if m:
            low = m.lower()
            if "vllm:" in low:
                fw["name"] = "vLLM"
                fw["detected_by"] = "/metrics"
            elif "tgi_" in low or "text-generation-inference" in low:
                fw["name"] = "TGI"
                fw["detected_by"] = "/metrics"

    if not fw["name"]:
        fw["name"] = "openai-compatible"
        fw["detected_by"] = "fallback"
        fw["notes"].append("no known probe answered; assumed an OpenAI-compatible chat endpoint")

    # --- overrides win
    fw["name"] = ov.get("framework") or fw["name"]
    fw["version"] = ov.get("framework_version") or fw["version"]
    model["id"] = ov.get("model_name") or model["id"]
    model["quantization"] = ov.get("quantization") or model["quantization"]
    if not model["quantization"]:
        model["quantization"] = _quant_from_id(model["id"])
    model["context_length"] = ov.get("context_length") or model["context_length"]
    return {"framework": fw, "model": model}
