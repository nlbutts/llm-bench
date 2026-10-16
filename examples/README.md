# Benchmarking different frameworks

`llmbench` talks to a server you already run over the OpenAI
`/v1/chat/completions` dialect. Start the framework however it wants, then point
the tool at its port. Below are starting points for the common ones; use
`--detect` first to confirm what the tool sees.

Every command writes `results/<date>-<device>-<framework>-<model>.json`.

## Strata

```bash
cd /path/to/Strata && ./setup.sh          # serves on 127.0.0.1:8080
python -m llmbench --url http://127.0.0.1:8080 --lengths 4096,16384,32768,65536,100000
```

Detected as `Strata`; the engine version and model come from `/v1/status`.

## llama.cpp (`llama-server`)

```bash
llama-server -m model.gguf -c 32768 --port 8080 --host 127.0.0.1
python -m llmbench --url http://127.0.0.1:8080 --lengths 4096,16384,32768
```

Detected as `llama.cpp (llama-server)` from `/props`; the context comes from
`n_ctx`. llama.cpp reports the same `timings` block the tool prefers.

## vLLM

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000
python -m llmbench --url http://127.0.0.1:8000 --model meta-llama/Llama-3.1-8B-Instruct \
    --lengths 4096,16384,32768,65536
```

vLLM is detected from `/metrics`. It does not emit `timings`, so prefill and
decode come from the client clock and are labelled `source: client` in the JSON.

## Ollama

```bash
ollama serve            # 127.0.0.1:11434
ollama pull llama3.1:8b
python -m llmbench --url http://127.0.0.1:11434 --model llama3.1:8b --lengths 4096,16384
```

Detected as `Ollama` from `/api/version`; the model list comes from `/api/tags`.

## SGLang

```bash
python -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct --port 30000
python -m llmbench --url http://127.0.0.1:30000 --lengths 4096,16384,32768
```

## LM Studio

Start the local server (Developer tab), then:

```bash
python -m llmbench --url http://127.0.0.1:1234 --lengths 4096,16384
```

## Anything else

If a server speaks OpenAI chat completions but none of the probes match it, it is
reported as `openai-compatible` and the client-side timings are used. Fill in the
gaps explicitly:

```bash
python -m llmbench --url http://127.0.0.1:9000 \
    --framework "my-framework" --framework-version 1.2.3 \
    --model-name "my-model" --quantization Q4_K_M --context-length 32768 \
    --device-name "NVIDIA RTX 5090" --device-vendor NVIDIA --device-memory-gib 32
```

## Comparing runs

Because every result is one JSON file with `prefill_tps`, `decode_tps`, and
`ttft_s`, comparing two frameworks, models, or devices is just loading two files:

```bash
python - <<'PY'
import json, glob
for f in sorted(glob.glob("results/*.json")):
    r = json.load(open(f))
    dev = r["device"]["accelerator"]["name"]
    fw = r["framework"]["name"]
    m = r["model"]["id"]
    print(f"{dev} | {fw} | {m}")
    for row in r["results"]:
        print("  %6d tok  prefill %6.1f  decode %5.1f  ttft %6.2fs"
              % (row["prompt_tokens"]["median"], row["prefill_tps"]["median"],
                 row["decode_tps"]["median"], row["ttft_s"]["median"]))
PY
```
