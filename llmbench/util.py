"""Small helpers shared by the package: subprocess, HTTP, slugs, statistics."""
from __future__ import annotations

import json
import re
import statistics
import subprocess
import urllib.error
import urllib.request


def run_cmd(cmd, timeout: float = 10.0) -> str | None:
    """Run a command and return stdout, or None if it is missing or fails."""
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except Exception:  # noqa: BLE001 - any absence or failure is just "unknown"
        return None
    if p.returncode != 0:
        return None
    return p.stdout.strip()


def http_get_json(url: str, key: str = "", timeout: float = 10.0):
    """GET a URL and parse JSON, or return None on any failure."""
    headers = {"Authorization": "Bearer " + key} if key else {}
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception:  # noqa: BLE001
        return None


def http_get_text(url: str, key: str = "", timeout: float = 10.0) -> str | None:
    headers = {"Authorization": "Bearer " + key} if key else {}
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read().decode("utf-8", "replace")
    except Exception:  # noqa: BLE001
        return None


def normalize_base(url: str) -> str:
    url = url.strip().rstrip("/")
    if url and not re.match(r"^https?://", url):
        url = "http://" + url
    return url


def slug(text: str | None, limit: int = 40) -> str:
    """A filesystem-safe token for result filenames."""
    if not text:
        return "unknown"
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", str(text)).strip("-._")
    return (s[:limit] or "unknown").lower()


def stat(values) -> dict | None:
    v = [x for x in values if isinstance(x, (int, float))]
    if not v:
        return None
    return {
        "median": round(statistics.median(v), 3),
        "min": round(min(v), 3),
        "max": round(max(v), 3),
        "n": len(v),
    }
