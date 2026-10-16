"""Plot and compare any number of `llmbench` result JSON files.

    python -m llmbench.plot results/*.json                 # -> plots/comparison-*.png
    python -m llmbench.plot a.json b.json c.json --out plots --metric decode
    python -m llmbench.plot results/ --formats png,svg

Each file is one series, labelled with its own `label` field and nothing else
(the filename when a file has no label), because a composed device / framework
/ model string makes legends unreadable. Points are the median over runs with
min/max error bars. The x position is the measured `prompt_tokens`, falling
back to the sweep's `target_tokens` when a server omits token usage. Requires
matplotlib for PNG/SVG; without it, a dependency-free SVG is written instead so
the repo's no-dependency promise still holds.

Nothing is dropped silently. A file that cannot be read, a file that is not a
result document, and a file that has no rows for the metric being drawn are each
reported on stderr with the reason and the field that caused it - a server that
reports no prefill speed is the usual case, and it makes one series disappear
from one plot while the others still look fine. The run ends with a
"plotted N of M files" line saying how many series each metric got, and
`--strict` turns the reports into a non-zero exit.
"""
from __future__ import annotations

import argparse
import json
import sys
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


def _warn(msg: str) -> None:
    """Diagnostics go to stderr, leaving stdout as the list of written files."""
    print(msg, file=sys.stderr)


def load_results(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _json_kind(value) -> str:
    """What a decoded JSON value is called, for use in an error message."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _load_problem(fp: Path, err: Exception) -> str:
    """One actionable line explaining why `fp` did not load."""
    if isinstance(err, FileNotFoundError):
        return "no such file"
    if isinstance(err, IsADirectoryError):
        return "is a directory; pass the .json files inside it"
    if isinstance(err, PermissionError):
        return "cannot be read (permissions)"
    if isinstance(err, UnicodeDecodeError):
        return "is not UTF-8 text; is it really a JSON file?"
    if isinstance(err, json.JSONDecodeError):
        try:
            raw = Path(fp).read_text(encoding="utf-8", errors="replace")
        except OSError:
            raw = ""
        if not raw.strip():
            return "is empty (0 bytes); the run never wrote a result"
        return "is not valid JSON (line %d, column %d: %s)" % (err.lineno, err.colno, err.msg)
    return "%s: %s" % (type(err).__name__, err)


def _shape_problem(r) -> str | None:
    """Why this decoded document is not a plottable result, or None if it is.

    Each message is a verb phrase that reads after "it" at the call site.
    """
    if not isinstance(r, dict):
        return "has a JSON %s at the top level, expected an object" % _json_kind(r)
    if "results" not in r:
        keys = ", ".join(sorted(str(k) for k in r)[:6]) or "none"
        return "has no results[] key, so there is nothing to plot (top-level keys: %s)" % keys
    rows = r["results"]
    if not isinstance(rows, list):
        return "has results as a JSON %s, expected an array" % _json_kind(rows)
    if not rows:
        return "has an empty results[]; every request in the run failed"
    return None


def label_for(r: dict, fallback: str) -> str:
    """Legend text: the run's own `label`, else the filename stem.

    Nothing else. Composing framework / device / model into one string made
    legends unreadably long, and all of that is already in the filename and in
    the JSON itself.
    """
    if r.get("label"):
        return str(r["label"])
    return fallback


def _stat(row: dict, field: str):
    """The min/median/max block for one field, or None when absent or null."""
    v = row.get(field)
    if isinstance(v, dict) and v.get("median") is not None:
        return v
    return None


def _x_value(row):
    """X for one row: measured prompt tokens, else the sweep's target length.

    Some servers omit `usage.prompt_tokens`, which leaves `prompt_tokens` null;
    falling back to `target_tokens` keeps such a file on the x-axis instead of
    dropping every one of its points.
    """
    pt = row.get("prompt_tokens")
    if isinstance(pt, dict) and pt.get("median") is not None:
        return pt["median"]
    return row.get("target_tokens")


def series(r: dict, metric: str):
    """Points for one metric: (xs, ys, lower_err, upper_err) over the file's lengths."""
    field = METRICS[metric][1]
    xs, ys, lo, hi = [], [], [], []
    for row in r.get("results", []):
        if not isinstance(row, dict):
            continue
        x = _x_value(row)
        mv = _stat(row, field)
        if x is None or mv is None:
            continue
        xs.append(x)
        ys.append(mv["median"])
        lo.append(max(0.0, mv["median"] - mv.get("min", mv["median"])))
        hi.append(max(0.0, mv.get("max", mv["median"]) - mv["median"]))
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    return ([xs[i] for i in order], [ys[i] for i in order],
            [lo[i] for i in order], [hi[i] for i in order])


def audit(r: dict, metric: str) -> dict:
    """How much of one file is plottable for one metric, and why the rest is not.

    The counts are what makes a missing series explainable: "3 of 6 lengths,
    because `prefill_tps` is null on the other three" is actionable, whereas a
    legend with one entry missing is not.
    """
    field = METRICS[metric][1]
    rows = r.get("results")
    a = {"rows": 0, "points": 0, "no_value": 0, "no_x": 0, "bad_rows": 0, "x_fallback": 0}
    if not isinstance(rows, list):
        return a
    a["rows"] = len(rows)
    for row in rows:
        if not isinstance(row, dict):
            a["bad_rows"] += 1
            continue
        x = _x_value(row)
        if x is None:
            a["no_x"] += 1
            continue
        if _stat(row, field) is None:
            a["no_value"] += 1
            continue
        a["points"] += 1
        pt = row.get("prompt_tokens")
        if not (isinstance(pt, dict) and pt.get("median") is not None):
            a["x_fallback"] += 1
    a["dropped"] = a["rows"] - a["points"]
    return a


def _drop_reason(a: dict, metrics: list[str]) -> str:
    """The field-level explanation behind an audit's dropped count.

    Takes every metric that shares this audit so one line can cover them all.
    Phrased as noun phrases so a count of one still reads correctly.
    """
    fields = ", ".join(METRICS[m][1] for m in metrics)
    verb = "is" if len(metrics) == 1 else "are"
    reasons = []
    if a["no_value"]:
        reasons.append("%s %s null on %d of %d rows" % (fields, verb, a["no_value"], a["rows"]))
    if a["no_x"]:
        reasons.append("no prompt_tokens and no target_tokens on %d of %d rows"
                       % (a["no_x"], a["rows"]))
    if a["bad_rows"]:
        reasons.append("non-object results[] entries: %d of %d" % (a["bad_rows"], a["rows"]))
    if not reasons:
        reasons.append("no rows carry %s" % fields)
    return "; ".join(reasons)


def _unlabeled(label: str, fp: Path) -> bool:
    """True when the legend had to fall back to the filename."""
    return label == fp.stem


def _tag(label: str, fp: Path) -> str:
    """How a diagnostic names a file. The path alone when there is no label."""
    return str(fp) if _unlabeled(label, fp) else "%s (%s)" % (label, fp)


def _xml(text) -> str:
    """Escape what would break an SVG text node; labels are user-supplied."""
    s = str(text)
    for a, b in (("&", "&amp;"), ("<", "&lt;"), (">", "&gt;")):
        s = s.replace(a, b)
    return s


def _clip(text: str, n: int) -> str:
    """Shorten for the SVG legend, visibly - a silent cut reads as a real name."""
    s = str(text)
    return s if len(s) <= n else s[:n - 3] + "..."


def _report(loaded, metrics: list[str], warn) -> tuple[dict, int, int]:
    """Print what will not be plotted; return (points per metric, dead files, gaps).

    A gap is a file that contributes nothing, a file missing a whole metric, or
    a metric that no file can supply. A file losing only some of its lengths is
    a warning, not a gap, since failed runs are normal. `--strict` exits on gaps.
    """
    audits = {m: [audit(r, m) for (_label, _fp, r) in loaded] for m in metrics}
    per_metric = {m: sum(1 for a in audits[m] if a["points"]) for m in metrics}
    dead = sum(1 for i in range(len(loaded))
               if not any(audits[m][i]["points"] for m in metrics))
    gaps = 0

    # A long filename in the legend is the symptom; the missing label is the cause.
    nameless = [str(fp) for label, fp, _r in loaded if _unlabeled(label, fp)]
    if nameless:
        warn("warning: %d file%s no label field, so the legend shows the filename "
             "instead: %s"
             % (len(nameless), " has" if len(nameless) == 1 else "s have",
                ", ".join(nameless)))

    # A legend now shows the label and nothing else, so two runs of one config
    # would draw two lines under the same name. Plot them anyway, but say so.
    by_label = {}
    for label, _fp, _r in loaded:
        by_label.setdefault(label, []).append(_fp)
    for label, fps in by_label.items():
        if len(fps) > 1:
            warn("warning: %d files share the label %r, so their legend entries are "
                 "indistinguishable: %s"
                 % (len(fps), label, ", ".join(str(p) for p in fps)))

    for i, (label, fp, _r) in enumerate(loaded):
        missing = [m for m in metrics if not audits[m][i]["points"]]
        # One line per distinct outcome: a file that misses four metrics the same
        # way is one problem, not four, and the summary below is the useful part.
        groups = {}
        for metric in metrics:
            a = audits[metric][i]
            key = (a["points"], a["rows"], a["no_value"], a["no_x"], a["bad_rows"])
            groups.setdefault(key, []).append(metric)
        for (n_points, n_rows, _nv, _nx, _br), group in groups.items():
            names = ", ".join(group)
            reason = _drop_reason(audits[group[0]][i], group)
            if not n_points:
                warn("warning: %s: not plotted for %s - %s"
                     % (_tag(label, fp), names, reason))
            elif n_points < n_rows:
                warn("warning: %s: %s %s %d of %d lengths - %s"
                     % (_tag(label, fp), names, "shows" if len(group) == 1 else "show",
                        n_points, n_rows, reason))
        if missing and len(missing) == len(metrics):
            warn("         %s: no data for any requested metric, so it is not in these plots"
                 % label)
            gaps += 1
        elif missing:
            warn("         %s: plotted for %s only"
                 % (label, ", ".join(m for m in metrics if m not in missing)))
            gaps += len(missing)
        best_points = max(audits[m][i]["points"] for m in metrics)
        best_fallbacks = max(audits[m][i]["x_fallback"] for m in metrics)
        if best_points and best_fallbacks == best_points:
            warn("note: %s: x uses target_tokens for all %d points - the server omitted "
                 "prompt_tokens, so its positions are the requested lengths, not the measured ones"
                 % (_tag(label, fp), best_points))

    for metric in metrics:
        if not per_metric[metric]:
            warn("warning: no file has %s data - comparison-%s.* will be empty"
                 % (metric, metric))
            gaps += 1
    return per_metric, dead, gaps


def _fmt_x(v: float, _pos=None) -> str:
    return ("%gK" % (v / 1000)) if v >= 1000 else "%g" % v


def plot(files: list[Path], outdir: Path, metrics: list[str], title: str | None,
         error_bars: bool = True, formats=("png",), warn=_warn,
         strict: bool = False) -> list[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    loaded = []
    problems = 0
    for fp in files:
        try:
            r = load_results(fp)
        except Exception as e:  # noqa: BLE001
            warn("warning: %s skipped - %s" % (fp, _load_problem(fp, e)))
            problems += 1
            continue
        bad = _shape_problem(r)
        if bad:
            warn("warning: %s skipped - it %s" % (fp, bad))
            problems += 1
            continue
        loaded.append((label_for(r, fp.stem), fp, r))
    if not loaded:
        raise SystemExit("no plottable result files: all %d given file%s rejected (see above)"
                         % (problems, "" if problems == 1 else "s"))

    per_metric, dead, gaps = _report(loaded, metrics, warn)
    problems += gaps
    live = len(loaded) - dead
    print("plotted %d of %d file%s: %s"
          % (live, len(loaded), "" if len(loaded) == 1 else "s",
             ", ".join("%s %d/%d" % (m, per_metric[m], len(loaded)) for m in metrics)))
    if strict and problems:
        raise SystemExit("%d file%s or metric%s left out of the plots (--strict)"
                         % (problems, "" if problems == 1 else "s",
                            "" if problems == 1 else "s"))

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
    for i, (label, _fp, r) in enumerate(loaded):
        xs, ys, lo, hi = series(r, metric)
        if not xs:
            continue  # already explained by _report()
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
    else:
        ax.text(0.5, 0.5, "no %s data in any file" % metric, ha="center", va="center",
                fontsize=11, color="#a33")


def _svg_fallback(loaded, outdir: Path, metrics: list[str], title: str | None) -> list[Path]:
    """Very small SVG writer so plots exist even without matplotlib."""
    written = []
    for metric in metrics:
        ylabel, _f, _d = METRICS[metric]
        pts, xmax, ymax = [], 0.0, 0.0
        for i, (label, _fp, r) in enumerate(loaded):
            xs, ys, _lo, _hi = series(r, metric)
            if not xs:
                continue
            pts.append((label, xs, ys, PALETTE[i % len(PALETTE)]))
            xmax = max(xmax, max(xs))
            ymax = max(ymax, max(ys))
        w, h, pad = 720, 420, 50
        # room on the right for the labels, otherwise they run off the canvas
        legend = 170 if pts else 0
        xmax = xmax or 1
        ymax = ymax or 1
        sx = lambda x: pad + (w - 2 * pad) * x / xmax
        sy = lambda y: h - pad - (h - 2 * pad) * y / ymax
        lines = ['<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d">' % (w + legend, h),
                 '<rect width="100%" height="100%" fill="white"/>',
                 '<text x="%d" y="22" font-size="14" font-family="sans-serif">%s</text>'
                 % (pad, _xml(title or ylabel)),
                 '<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="#333"/>' % (pad, h - pad, w - pad, h - pad),
                 '<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="#333"/>' % (pad, pad, pad, h - pad),
                 '<text x="%d" y="%d" font-size="11" font-family="sans-serif" fill="#333">%s</text>'
                 % (pad, h - 14, "prompt tokens (max %s)" % _fmt_x(xmax))]
        if not pts:
            lines.append('<text x="%d" y="%d" font-size="12" font-family="sans-serif" fill="#a33">'
                         'no %s data in any file</text>' % (pad, h // 2, metric))
        for j, (label, xs, ys, color) in enumerate(pts):
            d = " ".join("%s%.1f,%.1f" % ("M" if k == 0 else "L", sx(x), sy(y))
                         for k, (x, y) in enumerate(zip(xs, ys)))
            lines.append('<path d="%s" fill="none" stroke="%s" stroke-width="2"/>' % (d, color))
            for x, y in zip(xs, ys):
                lines.append('<circle cx="%.1f" cy="%.1f" r="3" fill="%s"/>' % (sx(x), sy(y), color))
            lines.append('<text x="%d" y="%d" font-size="10" font-family="sans-serif" fill="%s">%s</text>'
                         % (w - pad + 4, pad + 14 * j, color, _xml(_clip(label, 28))))
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
    ap.add_argument("--quiet", action="store_true",
                    help="suppress the warnings and notes on stderr; stdout still gets the summary")
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero if any file or metric was left out of the plots")
    a = ap.parse_args(argv)

    warn = (lambda msg: None) if a.quiet else _warn
    metrics = a.metric or list(METRICS)

    files: list[Path] = []
    for f in (a.files or ["results"]):
        p = Path(f)
        if p.is_dir():
            found = sorted(p.glob("*.json"))
            if not found:
                warn("warning: no .json files in %s" % p)
            files.extend(found)
        elif not p.exists():
            warn("warning: %s skipped - no such file" % p)
        else:
            files.append(p)
    if not files:
        warn("no result files found")
        return 1
    try:
        written = plot(files, Path(a.out), metrics, a.title,
                       error_bars=not a.no_error_bars,
                       formats=tuple(x.strip() for x in a.formats.split(",") if x.strip()),
                       warn=warn, strict=a.strict)
    except SystemExit as e:
        _warn(str(e.code))
        return 2
    for p in written:
        print("wrote %s" % p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
