"""Unit tests for the pure logic. No server, no GPU, no network.

    python -m unittest discover -s tests
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llmbench.device import _pretty_gpu_name  # noqa: E402
from llmbench.framework import _quant_from_id  # noqa: E402
from llmbench.plot import series  # noqa: E402
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


if __name__ == "__main__":
    unittest.main()
