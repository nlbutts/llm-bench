"""Plot and compare any number of `llmbench` result JSON files.

    python -m llmbench.plot results/*.json                 # -> plots/comparison-*.png
    python -m llmbench.plot a.json b.json c.json --out plots --metric decode
    python -m llmbench.plot results/ --formats png,svg

Each file is one series, labelled from its own `label` field or from its
device / framework / model. Points are the median over runs with min/max error
bars. Requires matplotlib for PNG/SVG; without it, a dependency-free SVG is
written instead so the repo's no-dependency promise still holds.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

# metric key -> (axis title, field in results[], better direction)
METRICS = {
    "prefill": ("Prefill (prompt tokens/s)", "prefill_tps", "up"),
    "decode": ("Decode (tokens/s)", "decode_tps", "up"),
    "ttft": ("Time to first token (s)", "ttft_s", "down"),
    "total": ("Total request latency (s)", "total_s", "down"),
}
PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd",
           "#8c564b", "#17becf", "#e377c2", "#7f7f7f", "#bcbd22"]


def load_results(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def label_for(r: dict, fallback: str) -> str:
    if r.get("label"):
        return str(r["label"])
    acc = (r.get("device") or {}).get("accelerator") or {}
    fw = r.get("framework") or {}
    model = r.get("model") or {}
    parts = []
    if fw.get("name"):
        parts.append(fw["name"] + ((" " + fw["version"]) if fw.get("version") else ""))
    if acc.get("name"):
        parts.append("on " + acc["name"])
    if model.get("id"):
        parts.append(model["id"] + ((" (" + model["quantization"] + ")") if model.get("quantization") else ""))
    return " | ".join(parts) if parts else fallback


def series(r: dict, metric: str):
    """Points for one metric: (xs, ys, lower_err, upper_err) over the file's lengths."""
    field = METRICS[metric][1]
    xs, ys, lo, hi = [], [], [], []
    for row in r.get("results", []):
        pt = row.get("prompt_tokens") or {}
        mv = row.get(field) or {}
        if pt.get("median") is None or mv.get("median") is None:
            continue
        xs.append(pt["median"])
        ys.append(mv["median"])
        lo.append(max(0.0, mv["median"] - mv.get("min", mv["median"])))
        hi.append(max(0.0, mv.get("max", mv["median"]) - mv["median"]))
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    return ([xs[i] for i in order], [ys[i] for i in order],
            [lo[i] for i in order], [hi[i] for i in order])


def _fmt_x(v: float, _pos=None) -> str:
    return ("%gK" % (v / 1000)) if v >= 1000 else "%g" % v


def plot(files: list[Path], outdir: Path, metrics: list[str], title: str | None,
         error_bars: bool = True, formats=("png",)) -> list[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    loaded = []
    for fp in files:
        try:
            r = load_results(fp)
        except Exception as e:  # noqa: BLE001
            print("skip %s: %s" % (fp, e))
            continue
        loaded.append((label_for(r, fp.stem), r))
    if not loaded:
        raise SystemExit("no readable result files given")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        return _svg_fallback(loaded, outdir, metrics, title)

    written = []
    matplotlib.rcParams.update({"figure.dpi": 120, "font.size": 10, "axes.grid": True,
                                "grid.alpha": 0.3})
    # one figure per metric
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        _draw(ax, loaded, metric, error_bars)
        if title:
            ax.set_title(title)
        fig.tight_layout()
        for ext in formats:
            p = outdir / ("comparison-%s.%s" % (metric, ext))
            fig.savefig(p, bbox_inches="tight")
            written.append(p)
        plt.close(fig)

    # one combined figure
    fig, axes = plt.subplots(len(metrics), 1, figsize=(7.5, 3.2 * len(metrics)), squeeze=False)
    for ax, metric in zip(axes[:, 0], metrics):
        _draw(ax, loaded, metric, error_bars)
    if title:
        axes[0, 0].set_title(title)
    fig.tight_layout()
    for ext in formats:
        p = outdir / ("comparison-all.%s" % ext)
        fig.savefig(p, bbox_inches="tight")
        written.append(p)
    plt.close(fig)
    return written


def _draw(ax, loaded, metric, error_bars):
    ylabel, _field, _dir = METRICS[metric]
    any_series = False
    for i, (label, r) in enumerate(loaded):
        xs, ys, lo, hi = series(r, metric)
        if not xs:
            continue
        any_series = True
        color = PALETTE[i % len(PALETTE)]
        if error_bars:
            ax.errorbar(xs, ys, yerr=[lo, hi], marker="o", markersize=5, linewidth=1.6,
                        capsize=3, color=color, label=label)
        else:
            ax.plot(xs, ys, marker="o", markersize=5, linewidth=1.6, color=color, label=label)
    ax.set_xlabel("prompt tokens")
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_formatter(_fmt_x)
    if any_series:
        ax.legend(fontsize=8, loc="best")


def _svg_fallback(loaded, outdir: Path, metrics: list[str], title: str | None) -> list[Path]:
    """Very small SVG writer so plots exist even without matplotlib."""
    written = []
    for metric in metrics:
        ylabel, _f, _d = METRICS[metric]
        pts, xmax, ymax = [], 0.0, 0.0
        for i, (label, r) in enumerate(loaded):
            xs, ys, _lo, _hi = series(r, metric)
            if not xs:
                continue
            pts.append((label, xs, ys, PALETTE[i % len(PALETTE)]))
            xmax = max(xmax, max(xs))
            ymax = max(ymax, max(ys))
        w, h, pad = 720, 420, 50
        xmax = xmax or 1
        ymax = ymax or 1
        sx = lambda x: pad + (w - 2 * pad) * x / xmax
        sy = lambda y: h - pad - (h - 2 * pad) * y / ymax
        lines = ['<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d">' % (w, h),
                 '<rect width="100%%" height="100%%" fill="white"/>',
                 '<text x="%d" y="22" font-size="14" font-family="sans-serif">%s</text>'
                 % (pad, title or ylabel),
                 '<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="#333"/>' % (pad, h - pad, w - pad, h - pad),
                 '<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="#333"/>' % (pad, pad, pad, h - pad),
                 '<text x="%d" y="%d" font-size="11" font-family="sans-serif" fill="#333">%s</text>'
                 % (pad, h - 14, "prompt tokens (max %s)" % _fmt_x(xmax))]
        for j, (label, xs, ys, color) in enumerate(pts):
            d = " ".join("%s%.1f,%s%.1f" % ("M" if k == 0 else "L", sx(x), sy(y)) for k, (x, y) in enumerate(zip(xs, ys)))
            lines.append('<path d="%s" fill="none" stroke="%s" stroke-width="2"/>' % (d, color))
            for x, y in zip(xs, ys):
                lines.append('<circle cx="%.1f" cy="%.1f" r="3" fill="%s"/>' % (sx(x), sy(y), color))
            lines.append('<text x="%d" y="%d" font-size="10" font-family="sans-serif" fill="%s">%s</text>'
                         % (w - pad + 4, pad + 14 * j, color, label[:24]))
        lines.append("</svg>")
        p = outdir / ("comparison-%s.svg" % metric)
        p.write_text("\n".join(lines), encoding="utf-8")
        written.append(p)
    return written


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="llmbench-plot",
                                 description="Compare llmbench result JSON files as plots.")
    ap.add_argument("files", nargs="*", help="result JSON files or directories (default: results/)")
    ap.add_argument("--out", default="plots", help="output directory (default plots/)")
    ap.add_argument("--metric", action="append", choices=list(METRICS),
                    help="restrict to one metric (repeatable); default all")
    ap.add_argument("--title", default=None)
    ap.add_argument("--formats", default="png", help="comma-separated: png,svg,pdf")
    ap.add_argument("--no-error-bars", action="store_true")
    a = ap.parse_args(argv)

    files: list[Path] = []
    for f in (a.files or ["results"]):
        p = Path(f)
        if p.is_dir():
            files.extend(sorted(p.glob("*.json")))
        else:
            files.append(p)
    if not files:
        print("no result files found")
        return 1
    metrics = a.metric or list(METRICS)
    written = plot(files, Path(a.out), metrics, a.title,
                   error_bars=not a.no_error_bars,
                   formats=tuple(x.strip() for x in a.formats.split(",") if x.strip()))
    for p in written:
        print("wrote %s" % p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
