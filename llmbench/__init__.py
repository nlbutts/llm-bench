"""llm-bench: benchmark any LLM framework, model, and device over an OpenAI-compatible API.

The public surface is the command line:

    python -m llmbench --help

The modules are kept small and independent so a new framework adapter only has
to know how to probe its endpoint:

    device     - detect the accelerator, CPU, RAM, and OS
    framework  - detect the serving framework and model from the endpoint
    client     - one streamed OpenAI chat request, with timings
    sweep      - the prefill/decode sweep and its per-length summary
    cli        - argument parsing and orchestration
"""

__version__ = "0.1.0"
