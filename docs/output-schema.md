# Output schema

Every run writes one JSON file. It is self-describing: the top-level keys are
ordered **device, framework, model** so the thing being measured reads naturally.
The schema version is `schema` (currently `1`).

```jsonc
{
  "schema": 1,
  "tool": { "name": "llm-bench", "version": "0.1.0" },
  "generated_at": "2026-10-15T20:30:00-04:00",
  "label": null,

  "device": {
    "accelerator": {
      "kind": "gpu",                 // "gpu" or "cpu"
      "vendor": "AMD",               // NVIDIA | AMD | Intel | Apple | null
      "name": "AMD Radeon 8060S",
      "architecture": "gfx1151",     // gfx* (ROCm), sm_* (CUDA), else null
      "memory_total_gib": 4.0,       // device memory, if known
      "memory_note": "unified",      // present for shared-memory parts
      "count": 1,                    // number of accelerators
      "driver": null,                // driver / runtime version, if known
      "compute_runtime": null        // "CUDA 12.4", "ROCm 7.14", ...
    },
    "host": {
      "cpu": { "name": "...", "logical_cores": 32, "physical_cores": 32 },
      "ram_total_gib": 119.4,
      "os": { "system": "Linux", "release": "7.0.0-38-generic",
              "distro": "Ubuntu 26.04.1 LTS", "machine": "x86_64", "version": "..." }
    }
  },

  "framework": {
    "endpoint": "http://127.0.0.1:8080",
    "name": "Strata",                // Strata | llama.cpp | Ollama | vLLM | TGI | openai-compatible
    "version": "0.1.43.1",
    "detected_by": "/v1/status",     // which probe identified it
    "dialects": ["/v1/chat/completions"],
    "probes": { },                   // raw probe payloads kept for provenance
    "notes": []
  },

  "model": {
    "id": "qwen3.8-flash-next-unsloth-ud-iq4_xs",
    "quantization": "UD-IQ4_XS",
    "context_length": 131072,        // configured context
    "context_native": 131072,        // trained window, when the server reports it
    "detected_by": "/v1/status"
  },

  "benchmark": {
    "type": "prefill_decode_sweep",
    "transport": "openai/chat-completions",
    "lengths": [4096, 16384, 32768, 65536, 100000],
    "runs": 3,
    "max_tokens": 128,
    "temperature": 0.0,
    "thinking": false,
    "prompt_source": "deterministic word list",
    "cache": "fresh per run (unique nonce prefix)",
    "timing_source": "server timings when present, else client",
    "calibration": { "tokens_per_word": 1.077, "overhead_tokens": 44 }
  },

  "results": [
    {
      "target_tokens": 100000,
      "runs_attempted": 3,
      "runs_ok": 3,
      "prompt_tokens": { "median": 100113, "min": 100113, "max": 100113, "n": 3 },
      "reused_tokens": { "median": 0, "min": 0, "max": 0, "n": 3 },
      "prefill_tps": { "median": 606.6, "min": 602.5, "max": 610.8, "n": 3 },
      "decode_tps": { "median": 41.2, "min": 41.0, "max": 41.4, "n": 3 },
      "ttft_s": { "median": 165.14, "min": 164.02, "max": 166.27, "n": 3 },
      "total_s": { "...": "..." },
      "generated_tokens": { "...": "..." }
    }
  ],

  "requests": [ /* one object per attempt, see below */ ],

  "notes": [],
  "limitations": [ "...", "..." ]
}
```

## Per-request fields (`requests[]`)

| field | meaning |
| --- | --- |
| `target`, `run` | which length and repetition this attempt belongs to |
| `attempts` | 1 normally; more if a retry was needed |
| `ok` | false with `error` set when the request failed |
| `prompt_tokens` | prompt size as the server counted it |
| `reused_tokens` | prefix tokens served from cache (0 = fully fresh) |
| `read_tokens` | tokens actually read (`prompt_n`), when the server reports it |
| `generated_tokens` | output length |
| `prefill_tps`, `prefill_tps_source` | prompt tokens/second and where it came from (`engine`/`client`) |
| `decode_tps`, `decode_tps_source` | generated tokens/second and its source |
| `ttft_s` | seconds from send to first non-empty token (client clock) |
| `total_s` | whole-request seconds (client clock) |
| `first_token_kind` | `content` or `reasoning` |
| `prompt_ms`, `decode_ms` | the server's own split times |
| `drafts_offered`, `drafts_accepted` | speculative decode counters, when present |

## Conventions

- **Fresh prompts.** A unique nonce leads every prompt, so `reused_tokens` is 0
  and prefill is real. If a server ignores the nonce, `reused_tokens` will show it.
- **Timing source.** `engine` means the server measured it; `client` means this
  tool did. Never mix the two in one comparison without saying so.
- **Medians.** `results[]` holds median/min/max over the successful runs of a
  length; `n` is how many succeeded.
- **Null metrics.** A `results[]` field is null when nothing ever measured it -
  a server that reports no prompt throughput leaves `prefill_tps` null, and one
  that omits `usage` leaves `prompt_tokens` null. `llmbench-plot` skips such
  rows and says which field and how many rows, rather than dropping the series
  from the figure in silence.
