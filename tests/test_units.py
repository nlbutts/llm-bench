"""Unit tests for the pure logic. No server, no GPU, no network.

    python -m unittest discover -s tests
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llmbench.device import _pretty_gpu_name  # noqa: E402
from llmbench.framework import _quant_from_id  # noqa: E402
from llmbench.plot import (METRICS, _drop_reason, _load_problem, _report,  # noqa: E402
                           _shape_problem, audit, label_for, series)
from llmbench.sweep import word_prompt  # noqa: E402
from llmbench.util import normalize_base, slug, stat  # noqa: E402


class TestQuant(unittest.TestCase):
    def test_unsloth_ud(self):
        self.assertEqual(_quant_from_id("qwen3.8-flash-next-unsloth-UD-IQ4_XS"), "UD-IQ4_XS")

    def test_llama_q(self):
        self.assertEqual(_quant_from_id("Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"), "Q4_K_M")

    def test_plain_q(self):
        self.assertEqual(_quant_from_id("Llama-3.2-3B-Q8_0"), "Q8_0")

    def test_none(self):
        self.assertIsNone(_quant_from_id("meta-llama/Llama-3.1-8B-Instruct"))


class TestGpuName(unittest.TestCase):
    def test_apu_name_from_cpu(self):
        raw = ("Advanced Micro Devices, Inc. [AMD/ATI] Strix Halo "
               "[Radeon Graphics / Radeon 8050S Graphics / Radeon 8060S Graphics] (rev d1)")
        name, note = _pretty_gpu_name(raw, "AMD", "AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S")
        self.assertEqual(name, "Radeon 8060S")
        self.assertIn("unified", note)

    def test_nvidia_untouched(self):
        name, note = _pretty_gpu_name("NVIDIA GeForce RTX 5090", "NVIDIA", "Intel Core i9")
        self.assertEqual(name, "NVIDIA GeForce RTX 5090")
        self.assertIsNone(note)


class TestUtil(unittest.TestCase):
    def test_slug(self):
        self.assertEqual(slug("AMD Radeon 8060S"), "amd-radeon-8060s")

    def test_normalize_base(self):
        self.assertEqual(normalize_base("127.0.0.1:8080/"), "http://127.0.0.1:8080")

    def test_stat(self):
        s = stat([1, 2, 3])
        self.assertEqual((s["median"], s["min"], s["max"], s["n"]), (2, 1, 3, 3))

    def test_stat_empty(self):
        self.assertIsNone(stat([]))


class TestPrompt(unittest.TestCase):
    def test_deterministic(self):
        self.assertEqual(word_prompt(50), word_prompt(50))

    def test_scales(self):
        self.assertLess(len(word_prompt(10)), len(word_prompt(1000)))


class TestPlotSeries(unittest.TestCase):
    def test_sorted_with_error_bars(self):
        r = {"results": [
            {"prompt_tokens": {"median": 16384}, "decode_tps": {"median": 40, "min": 38, "max": 42}},
            {"prompt_tokens": {"median": 4096}, "decode_tps": {"median": 30, "min": 29, "max": 31}},
        ]}
        xs, ys, lo, hi = series(r, "decode")
        self.assertEqual(xs, [4096, 16384])
        self.assertEqual(ys, [30, 40])
        self.assertEqual(lo, [1, 2])
        self.assertEqual(hi, [1, 2])

    def test_skips_missing(self):
        r = {"results": [{"prompt_tokens": {"median": 100}, "decode_tps": None}]}
        self.assertEqual(series(r, "decode"), ([], [], [], []))

    def test_falls_back_to_target_tokens(self):
        # Servers that omit usage leave prompt_tokens null; target_tokens still places the point.
        r = {"results": [
            {"target_tokens": 8192, "prompt_tokens": None,
             "decode_tps": {"median": 33, "min": 31, "max": 34}},
            {"target_tokens": 4096, "prompt_tokens": None,
             "decode_tps": {"median": 30, "min": 29, "max": 30}},
        ]}
        xs, ys, lo, hi = series(r, "decode")
        self.assertEqual(xs, [4096, 8192])
        self.assertEqual(ys, [30, 33])
        self.assertEqual(lo, [1, 2])
        self.assertEqual(hi, [0, 1])

    def test_prefers_measured_prompt_tokens(self):
        r = {"results": [{"target_tokens": 4096,
                          "prompt_tokens": {"median": 4019},
                          "decode_tps": {"median": 35, "min": 35, "max": 35}}]}
        xs, _ys, _lo, _hi = series(r, "decode")
        self.assertEqual(xs, [4019])


class TestPlotLabel(unittest.TestCase):
    """The legend is the label, the whole label, and nothing but the label."""

    def test_uses_only_the_label(self):
        r = {"label": "Strata UD-IQ4_XS",
             "framework": {"name": "Strata", "version": "0.1.44"},
             "device": {"accelerator": {"name": "AMD Radeon 8060S"}},
             "model": {"id": "qwen3.8-flash-next", "quantization": "UD-IQ4_XS"}}
        self.assertEqual(label_for(r, "file-stem"), "Strata UD-IQ4_XS")

    def test_no_label_falls_back_to_the_stem_not_the_device(self):
        r = {"framework": {"name": "Strata", "version": "0.1.44"},
             "device": {"accelerator": {"name": "AMD Radeon 8060S"}},
             "model": {"id": "qwen3.8-flash-next"}}
        self.assertEqual(label_for(r, "file-stem"), "file-stem")

    def test_report_warns_when_a_file_has_no_label(self):
        row = {"target_tokens": 1, "decode_tps": {"median": 1, "min": 1, "max": 1}}
        msgs = []
        _per, _dead, _gaps = _report([("a", Path("a.json"), {"results": [row]})],
                                     ["decode"], msgs.append)
        self.assertTrue(any("no label field" in m for m in msgs), msgs)

    def test_report_warns_when_labels_collide(self):
        row = {"target_tokens": 1, "decode_tps": {"median": 1, "min": 1, "max": 1}}
        r = {"label": "same", "results": [row]}
        msgs = []
        _per, _dead, _gaps = _report([("same", Path("a.json"), r), ("same", Path("b.json"), r)],
                                     ["decode"], msgs.append)
        self.assertTrue(any("share the label" in m for m in msgs), msgs)


class TestPlotDiagnostics(unittest.TestCase):
    """A series that is not drawn has to say why, or the plot just looks thin."""

    def test_audit_counts_rows_with_a_null_metric(self):
        r = {"results": [
            {"target_tokens": 4096, "decode_tps": {"median": 30, "min": 29, "max": 31}},
            {"target_tokens": 8192, "decode_tps": None},
        ]}
        a = audit(r, "decode")
        self.assertEqual((a["rows"], a["points"], a["no_value"], a["dropped"]), (2, 1, 1, 1))

    def test_audit_counts_rows_without_an_x(self):
        a = audit({"results": [{"decode_tps": {"median": 1, "min": 1, "max": 1}}]}, "decode")
        self.assertEqual((a["points"], a["no_x"]), (0, 1))

    def test_audit_notices_the_target_tokens_fallback(self):
        r = {"results": [{"target_tokens": 4096, "prompt_tokens": None,
                          "decode_tps": {"median": 30, "min": 30, "max": 30}}]}
        self.assertEqual(audit(r, "decode")["x_fallback"], 1)

    def test_audit_survives_junk_rows(self):
        row = {"target_tokens": 1, "decode_tps": {"median": 2, "min": 2, "max": 2}}
        a = audit({"results": ["nope", row]}, "decode")
        self.assertEqual((a["points"], a["bad_rows"]), (1, 1))

    def test_drop_reason_names_the_field(self):
        a = audit({"results": [{"target_tokens": 1, "prefill_tps": None}]}, "prefill")
        self.assertIn("prefill_tps is null on 1 of 1 rows", _drop_reason(a, ["prefill"]))

    def test_drop_reason_shares_one_line_across_metrics(self):
        rows = [{"target_tokens": 1, "prefill_tps": None, "ttft_s": None}]
        a = audit({"results": rows}, "prefill")
        self.assertIn("prefill_tps, ttft_s are null on 1 of 1 rows",
                      _drop_reason(a, ["prefill", "ttft"]))

    def test_shape_problem(self):
        self.assertIn("array", _shape_problem([1, 2]))
        self.assertIn("results[]", _shape_problem({"label": "x"}))
        self.assertIn("empty", _shape_problem({"results": []}))
        self.assertIsNone(_shape_problem({"results": [{"target_tokens": 1}]}))

    def test_load_problem_for_a_missing_file(self):
        self.assertEqual(_load_problem(Path("gone.json"), FileNotFoundError()), "no such file")

    def test_load_problem_distinguishes_empty_from_malformed(self):
        with tempfile.TemporaryDirectory() as d:
            empty = Path(d) / "empty.json"
            empty.write_text("")
            broken = Path(d) / "broken.json"
            broken.write_text('{"results": [')
            for fp, want in ((empty, "empty"), (broken, "not valid JSON")):
                try:
                    json.loads(fp.read_text())
                    self.fail("expected %s to be rejected" % fp)
                except json.JSONDecodeError as e:
                    self.assertIn(want, _load_problem(fp, e))

    def test_report_collapses_identical_outcomes_into_one_line(self):
        row = {"target_tokens": 8192,
               "prefill_tps": {"median": 700, "min": 690, "max": 710},
               "decode_tps": {"median": 40, "min": 39, "max": 41},
               "ttft_s": {"median": 9, "min": 8, "max": 10},
               "total_s": {"median": 11, "min": 10, "max": 12}}
        r = {"results": [{"target_tokens": 4096}, row]}
        msgs = []
        _per, _dead, _gaps = _report([("one", Path("one.json"), r)], list(METRICS), msgs.append)
        drops = [m for m in msgs if " of 2 lengths" in m]
        self.assertEqual(len(drops), 1, drops)  # four metrics, one line
        self.assertIn("prefill, decode, ttft, total show 1 of 2 lengths", drops[0])

    def test_report_explains_a_series_missing_from_one_metric(self):
        good = {"results": [{"prompt_tokens": {"median": 4019},
                             "prefill_tps": {"median": 640, "min": 630, "max": 650},
                             "decode_tps": {"median": 35, "min": 34, "max": 36}}]}
        no_prefill = {"results": [{"target_tokens": 4096, "prefill_tps": None,
                                   "decode_tps": {"median": 20, "min": 20, "max": 21}}]}
        msgs = []
        loaded = [("strata", Path("strata.json"), good),
                  ("vllm", Path("vllm.json"), no_prefill)]
        per_metric, dead, _problems = _report(loaded, ["prefill", "decode"], msgs.append)
        self.assertEqual(per_metric, {"prefill": 1, "decode": 2})
        self.assertEqual(dead, 0)  # it is still a decode series, so not a dead file
        self.assertTrue(any("prefill_tps is null" in m for m in msgs), msgs)
        self.assertTrue(any("plotted for decode only" in m for m in msgs), msgs)

    def test_report_flags_a_file_with_nothing_to_plot(self):
        r = {"results": [{"target_tokens": 4096, "decode_tps": None}]}
        msgs = []
        per_metric, dead, problems = _report([("dead", Path("dead.json"), r)],
                                             ["prefill", "decode"], msgs.append)
        self.assertEqual(per_metric, {"prefill": 0, "decode": 0})
        self.assertEqual(dead, 1)
        self.assertGreater(problems, 0)  # so --strict fails on it
        self.assertTrue(any("not in these plots" in m for m in msgs), msgs)
        self.assertTrue(any("no file has prefill data" in m for m in msgs), msgs)


if __name__ == "__main__":
    unittest.main()
