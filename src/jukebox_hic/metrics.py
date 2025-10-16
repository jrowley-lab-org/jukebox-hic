#!/usr/bin/env python
import os
import sys
from typing import Tuple

try:
    import psutil  # type: ignore
except ImportError:
    psutil = None  # type: ignore

try:
    import resource  # type: ignore
except ImportError:
    resource = None  # type: ignore


def _process_rss_mb() -> float:
    if psutil is None:
        return 0.0
    try:
        proc = psutil.Process(os.getpid())
        rss = proc.memory_info().rss
        for child in proc.children(recursive=True):
            try:
                rss += child.memory_info().rss
            except psutil.Error:
                continue
        return rss / (1024 * 1024)
    except psutil.Error:
        return 0.0


def _ru_maxrss_mb() -> float:
    if resource is None:
        return 0.0
    usage = resource.getrusage(resource.RUSAGE_SELF)
    if sys.platform.startswith("darwin") or sys.platform.startswith("win"):
        return usage.ru_maxrss / (1024 * 1024)
    return usage.ru_maxrss / 1024


def collect_memory_usage_mb() -> Tuple[float, float]:
    """
    Returns (current_rss_mb, ru_maxrss_mb).
    """
    return _process_rss_mb(), _ru_maxrss_mb()
