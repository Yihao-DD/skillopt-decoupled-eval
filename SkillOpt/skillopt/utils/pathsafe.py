"""Make benchmark item ids safe to use as a single filesystem path component.

Some benchmarks (e.g. LiveMathematicianBench) use ids like ``202602:32`` that contain
characters Windows forbids in file/dir names (``< > : " / \\ | ? *``). Per-item
prediction dirs named directly by id therefore fail on Windows with WinError 267.
This helper replaces those characters with ``_`` so the same code runs on Windows,
Linux, and macOS. It is idempotent and a no-op on ids that are already clean, so
the write side (rollout) and read side (slow_update / trainer) stay consistent.
"""
from __future__ import annotations

import re

# Characters illegal in a Windows path component (also covers the POSIX separators).
_ILLEGAL = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def safe_id(value: object) -> str:
    """Return ``value`` as a string safe to use as ONE path component on any OS.

    Replaces Windows-illegal characters with ``_`` and trims trailing dots/spaces
    (also rejected by Windows). Idempotent.
    """
    s = _ILLEGAL.sub("_", str(value))
    return s.rstrip(" .") or "_"
