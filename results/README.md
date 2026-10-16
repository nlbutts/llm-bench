# Results

Committed benchmark runs, one JSON per run. The schema is in
[`../docs/output-schema.md`](../docs/output-schema.md); each file is
self-describing and ordered device -> framework -> model.

Compare any set of these files as plots:

```bash
python -m llmbench.plot results/*.json --out plots
```

## 2026-10-15 - Strata UD-IQ4_XS on Radeon 8060S (Strix Halo)

File: [`2026-10-15-radeon-8060s-strata-qwen3.8-flash-next-unsloth-ud-iq4_xs.json`](2026-10-15-radeon-8060s-strata-qwen3.8-flash-next-unsloth-ud-iq4_xs.json)

- **Device:** AMD Radeon 8060S (Strix Halo, gfx1151), unified memory; host AMD
  RYZEN AI MAX+ PRO 395, 32 cores, 119.4 GiB RAM, Ubuntu 26.04.1 LTS, ROCm 7.14.
- **Framework:** Strata 0.1.44.1 (`/v1/status`).
- **Model:** `qwen3.8-flash-next-unsloth-ud-iq4_xs`, UD-IQ4_XS, context 131,072.
- **Benchmark:** fresh cache-proof prompts (unique nonce), 128-token output cap,
  3 runs per length, thinking off. Engine timings for prefill/decode.

| Prompt tokens | Prefill tok/s (median, range) | Decode tok/s (median, range) | TTFT s | Total s |
| ---: | --- | --- | --- | --- |
| 4,019 | 641.3 (632.4-641.7) | 35.2 (32.3-36.3) | 6.31 | 9.41 |
| 8,019 | 690.4 (690.2-692.2) | 36.2 (34.9-37.8) | 11.66 | 15.13 |
| 16,058 | 689.4 (688.5-689.7) | 39.7 (34.5-40.0) | 23.35 | 26.52 |
| 32,048 | 675.4 (674.8-676.4) | 37.5 (37.3-41.4) | 47.51 | 50.85 |
| 64,040 | 646.9 (646.1-647.3) | 38.5 (36.1-40.5) | 99.08 | 102.31 |
| 100,018 | 607.0 (603.0-614.6) | 40.6 (39.6-40.7) | 164.89 | 168.00 |

All 18 requests succeeded and `reused_tokens` was 0 throughout, so the prefill
figures are entirely fresh work. Prefill rises to a plateau around 8-16K and then
falls slowly with context; decode is flat within run-to-run noise (speculative
decode is on, `--spec 4`); TTFT is close to linear in prompt length.

Plots: [`../plots/comparison-all.png`](../plots/comparison-all.png).

Reproduce (server running on :8080):

```bash
python -m llmbench --url http://127.0.0.1:8080 \
    --lengths 4096,8192,16384,32768,65536,102400 --runs 3 --max-tokens 128 \
    --label "Strata UD-IQ4_XS prefill/decode sweep"
```

Note: run-to-run medians are reported; the largest length needs about three
minutes per run on this hardware, so the whole sweep takes roughly 20 minutes.
