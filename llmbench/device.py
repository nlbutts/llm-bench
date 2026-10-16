"""Detect the device (accelerator first, then host) with best-effort fallbacks.

Nothing here is required: every field can be overridden on the command line, so
an unusual machine still produces a complete record. Detection order is GPU
first because the accelerator is what a benchmark is really about; the host
(CPU, RAM, OS) rides along under `host`.
"""
from __future__ import annotations

import glob
import os
import platform
import re
from pathlib import Path

from llmbench.util import run_cmd


def _read(path: str) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace").strip()
    except Exception:  # noqa: BLE001
        return None


def _num(path: str):
    """Read an integer that sysfs often writes as hex ('0x1002') or decimal."""
    v = _read(path)
    try:
        return int(v, 0) if v is not None else None
    except (TypeError, ValueError):
        return None


def _cpu() -> dict:
    name, cores, phys = None, os.cpu_count(), None
    txt = _read("/proc/cpuinfo")
    if txt:
        m = re.search(r"^model name\s*:\s*(.+)$", txt, re.M)
        if m:
            name = m.group(1).strip()
        m = re.search(r"^cpu cores\s*:\s*(\d+)$", txt, re.M)
        if m:
            phys = int(m.group(1))
        ids = re.findall(r"^physical id\s*:\s*(\d+)$", txt, re.M)
        if phys and ids:
            phys *= len(set(ids))
    elif platform.system() == "Darwin":
        name = run_cmd(["sysctl", "-n", "machdep.cpu.brand_string"])
    return {"name": name, "logical_cores": cores, "physical_cores": phys}


def _ram_gib() -> float | None:
    txt = _read("/proc/meminfo")
    if txt:
        m = re.search(r"^MemTotal:\s*(\d+)\s*kB", txt, re.M)
        if m:
            return round(int(m.group(1)) / 1024 / 1024, 2)
    if platform.system() == "Darwin":
        out = run_cmd(["sysctl", "-n", "hw.memsize"])
        if out and out.isdigit():
            return round(int(out) / 2**30, 2)
    return None


def _os() -> dict:
    pretty = None
    txt = _read("/etc/os-release")
    if txt:
        m = re.search(r'^PRETTY_NAME="?([^"\n]+)"?', txt, re.M)
        if m:
            pretty = m.group(1)
    return {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "distro": pretty,
        "machine": platform.machine(),
    }


# ------------------------------------------------------------------ accelerators
def _nvidia() -> list[dict]:
    out = run_cmd(["nvidia-smi", "--query-gpu=name,memory.total,driver_version,compute_cap",
                   "--format=csv,noheader,nounits"])
    if not out:
        return []
    cuda = None
    header = run_cmd(["nvidia-smi"])
    if header:
        m = re.search(r"CUDA Version:\s*([0-9.]+)", header)
        if m:
            cuda = m.group(1)
    cards = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        cards.append({
            "kind": "gpu", "vendor": "NVIDIA", "name": parts[0],
            "memory_total_gib": round(float(parts[1]) / 1024, 2) if parts[1].replace(".", "").isdigit() else None,
            "driver": parts[2], "architecture": "sm_" + parts[3].replace(".", "") if parts[3] else None,
            "compute_runtime": ("CUDA " + cuda) if cuda else None,
        })
    return cards


def _rocm_arch() -> str | None:
    out = run_cmd(["rocminfo"], timeout=15)
    if out:
        m = re.search(r"\bgfx[0-9a-f]{3,4}\b", out)
        if m:
            return m.group(0)
    return None


def _rocm_smi() -> list[dict]:
    out = run_cmd(["rocm-smi", "--showproductname", "--showmeminfo", "vram", "--csv"], timeout=15)
    if not out:
        return []
    cards = []
    for line in out.splitlines():
        if "card" not in line.lower():
            continue
        parts = re.split(r"[,\t]+", line.strip())
        name = next((p for p in parts if re.search(r"[A-Za-z]", p) and "card" not in p.lower()), None)
        vram = None
        m = re.search(r"(\d+)\s*(?:MiB|MB)", line)
        if m:
            vram = round(int(m.group(1)) / 1024, 2)
        cards.append({"kind": "gpu", "vendor": "AMD", "name": name, "memory_total_gib": vram,
                      "architecture": _rocm_arch()})
    return cards


def _sysfs_gpu() -> list[dict]:
    """AMD/Intel/other via sysfs + lspci, when rocm-smi/nvidia-smi are absent."""
    vendor = {0x1002: "AMD", 0x10DE: "NVIDIA", 0x8086: "Intel"}
    cards = []
    lspci = run_cmd(["lspci"]) or ""
    # `card*` also matches the connectors (card1-DP-1); keep only real DRM cards.
    for dev in sorted(glob.glob("/sys/class/drm/card*/device")):
        card = os.path.basename(os.path.dirname(dev))
        if not re.match(r"^card\d+$", card):
            continue
        vid = _num(dev + "/vendor")
        if vid is None or vid not in vendor:
            continue
        # resolve the symlink to the real PCI device, e.g. .../0000:c3:00.0
        pci = os.path.basename(os.path.realpath(dev))
        short = pci.split(":", 1)[1] if ":" in pci else pci
        name = None
        for line in lspci.splitlines():
            if line.startswith(short) and re.search(r"(VGA|3D|Display)", line):
                name = line.split(": ", 1)[-1].strip()
                break
        if name is None:
            for line in lspci.splitlines():
                if re.search(r"(VGA|3D|Display)", line):
                    name = line.split(": ", 1)[-1].strip()
                    break
        vram = _num(dev + "/mem_info_vram_total")
        cards.append({
            "kind": "gpu", "vendor": vendor[vid], "name": name or (vendor[vid] + " GPU"),
            "memory_total_gib": round(vram / 2**30, 2) if vram else None,
            "architecture": _rocm_arch() if vendor[vid] == "AMD" else None,
            "pci_address": short,
        })
    return cards


def _apple() -> list[dict]:
    out = run_cmd(["system_profiler", "SPDisplaysDataType"], timeout=20)
    if not out:
        return []
    name = None
    m = re.search(r"Chipset Model:\s*(.+)", out)
    if m:
        name = m.group(1).strip()
    mem = run_cmd(["sysctl", "-n", "hw.memsize"])
    return [{"kind": "gpu", "vendor": "Apple", "name": name or "Apple GPU",
             "memory_total_gib": round(int(mem) / 2**30, 2) if mem and mem.isdigit() else None,
             "memory_note": "unified", "architecture": None}]


def _pretty_gpu_name(raw: str | None, vendor: str | None, cpu_name: str | None) -> tuple[str | None, str | None]:
    """Tidy an lspci-style name and, for AMD APUs, use the iGPU name from the CPU string.

    Returns (name, memory_note). '... w/ Radeon 8060S' in the CPU name is a
    reliable signal of a unified-memory APU, whose reported VRAM is a carve-out.
    """
    if not raw:
        return raw, None
    raw = re.sub(r"\s*\(rev [^)]+\)\s*$", "", raw).strip()
    raw = re.sub(r"^Advanced Micro Devices, Inc\.?\s*(\[AMD/ATI\])?\s*", "", raw)
    raw = re.sub(r"^\[AMD/ATI\]\s*", "", raw)
    if vendor == "AMD" and cpu_name:
        m = re.search(r"Radeon\s+([A-Za-z0-9]+(?:\s+[A-Za-z0-9]+)?)", cpu_name)
        if m and ("AMD/ATI" in (raw or "") or "Strix" in raw or "Graphics" in raw or "Radeon" in raw):
            return "Radeon " + m.group(1), "unified (shared with system RAM)"
    return raw.strip(), None


def detect_accelerator(overrides: dict | None = None, cpu_name: str | None = None) -> dict:
    ov = overrides or {}
    cards = _nvidia() or _rocm_smi() or _sysfs_gpu()
    if not cards and platform.system() == "Darwin":
        cards = _apple()
    if cards:
        dev = dict(cards[0])
        dev["count"] = len(cards)
        if len(cards) > 1:
            dev["devices"] = [c.get("name") for c in cards]
        dev["name"], note = _pretty_gpu_name(dev.get("name"), dev.get("vendor"), cpu_name)
        if note:
            dev.setdefault("memory_note", note)
    else:
        dev = {"kind": "cpu", "vendor": None, "name": None, "memory_total_gib": None,
               "architecture": None, "count": 0, "note": "no accelerator detected; benchmarking on CPU"}
    # explicit overrides win
    for k, v in ov.items():
        if v is not None:
            dev[k] = v
    return dev


def collect(overrides: dict | None = None) -> dict:
    """The full `device` block: accelerator, then the host it runs on."""
    cpu = _cpu()
    return {
        "accelerator": detect_accelerator(overrides, cpu.get("name")),
        "host": {"cpu": cpu, "ram_total_gib": _ram_gib(), "os": _os()},
    }
