# llm-bench

Benchmark **any** LLM framework, model, and device through one small,
standard-library-only Python tool.

It measures the two numbers that decide whether a local model is usable:

- **prefill** — how fast a fresh prompt is read (prompt tokens/second)
- **decode** — how fast tokens are generated (tokens/second), plus time to first token

It works against any server that speaks the OpenAI
`/v1/chat/completions` dialect, so it covers Strata, llama.cpp (`llama-server`),
vLLM, SGLang, Ollama, LM Studio, TGI, and anything else with an OpenAI-compatible
API. The device, framework, and model are detected automatically and recorded
**in that order**; everything is written to a single JSON file you can diff and
commit.

## Quick start

```bash
# against a server already running on localhost:8080
python -m llmbench --url http://127.0.0.1:8080 \
    --lengths 4096,8192,16384,32768,65536,100000 --runs 3

# only detect and print the device, framework, and model
python -m llmbench --url http://127.0.0.1:8080 --detect

# overrides when auto-detection cannot know something
python -m llmbench --url http://127.0.0.1:8000 \
    --framework vllm --framework-version 0.6.3 \
    --device-name "NVIDIA RTX 5090" --device-vendor NVIDIA \
    --model-name "meta-llama/Llama-3.1-8B" --quantization FP16
```

Results are written to `results/<date>-<device>-<framework>-<model>.json`
(override with `--out DIR`). The console prints the same data, device first:

```
DEVICE
  AMD Radeon 8060S (Strix Halo)  [AMD, gfx1151, 4.0 GiB VRAM]
  host: AMD RYZEN AI MAX+ PRO 395, 32 cores, 119.4 GiB RAM, Ubuntu 26.04.1 LTS
FRAMEWORK
  Strata 0.1.43.1  http://127.0.0.1:8080  (openai/chat-completions)
MODEL
  qwen3.8-flash-next-unsloth-ud-iq4_xs  quant=UD-IQ4_XS  context=131072
```

## What it measures

For every target length the tool sends a **fresh, cache-proof prompt** (a unique
nonce sits at the front, so no server can reuse a prefix), caps the output, and
records per run:

| field | source |
| --- | --- |
| `prefill_tps` | the server's own timing (`prompt_per_second`), else `prompt_tokens / ttft` |
| `decode_tps` | the server's own timing (`predicted_per_second`), else client-side |
| `ttft_s` | this client, over the streamed response |
| `prompt_tokens` / `reused_tokens` | the server's `usage` / `timings` |

Per-length summaries carry the median, min, and max over the runs.

## Supported endpoints and detection

The tool probes, in order: `/v1/status` and `/health` (Strata), `/props`
(llama.cpp), `/api/version` + `/api/tags` (Ollama), `/v1/models` (OpenAI), and
`/metrics` (vLLM, TGI). First match wins; anything else is reported as
`openai-compatible`. All fields can be overridden on the command line, so an
unusual server still produces a complete record.

## Output

- One JSON file per run (`results/*.json`), schema documented in
  [`docs/output-schema.md`](docs/output-schema.md).
- The same JSON is returned on stdout with `--stdout` for piping.

## Compare runs with plots

`llmbench-plot` turns any number of result JSON files into comparison plots -
one series per file, labelled with that file's own `label` and nothing else:

```bash
python -m llmbench.plot results/*.json                 # plots/comparison-*.png
python -m llmbench.plot a.json b.json c.json --metric decode
python -m llmbench.plot results/ --formats png,svg --out plots
```

It writes one figure per metric (prefill, decode, TTFT, total) plus a combined
grid, with median points and min/max error bars. Plotting uses matplotlib when it
is installed and falls back to a dependency-free SVG otherwise.

Nothing goes missing quietly. A file that cannot be read, a file that is not a
result document, and a file with no rows for the metric being drawn are each
reported on stderr with the field responsible, and the run ends with a per-metric
count. A server that reports no prompt throughput drops out of the prefill
figure only:

```
warning: Strata UD-IQ4_XS (results/b.json): not plotted for prefill - prefill_tps is null on 5 of 5 rows
         Strata UD-IQ4_XS: plotted for decode, ttft, total only
plotted 4 of 4 files: prefill 3/4, decode 4/4, ttft 4/4, total 4/4
```

It also warns when a file has no `label` (the legend falls back to the whole
filename) and when two files share one `label`, which would otherwise put two
lines in the legend under the same name.

`--strict` exits non-zero when anything was left out, which is what you want in
CI; `--quiet` keeps stderr clean for scripting.

## Design notes

- Standard library only. No install step; `python -m llmbench` runs anywhere
  Python 3.9+ is available.
- Nothing is started or stopped: point it at a server you already run. This keeps
  it neutral about frameworks (they have wildly different launch stories).
- Speeds come from the server when it reports them, and are labelled with their
  source (`engine` or `client`) so a number is never mistaken for the other.

See [`docs/output-schema.md`](docs/output-schema.md) for the JSON contract.
