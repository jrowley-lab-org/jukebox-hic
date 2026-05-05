#!/usr/bin/env python
"""
Memory profiling utilities for tracking jukebox-hic resource usage.

This module provides two approaches to measuring memory consumption:

1. **psutil-based RSS** (``_process_rss_mb``): Uses the ``psutil`` library to read
   the current Resident Set Size (RSS) of the process and its children.
   - RSS is the amount of physical RAM currently occupied by the process.
   - This is a *current* snapshot — it can go down if memory is freed.
   - Includes child process memory when running in parallel mode.

2. **getrusage peak RSS** (``_ru_maxrss_mb``): Uses POSIX ``resource.getrusage()``
   to read the maximum RSS ever reached by the process.
   - This is a *high-water mark* — it only ever increases.
   - Useful for finding peak memory usage across a long run.
   - Not available on all platforms (falls back to 0.0 on Windows where
     ``resource`` is not importable).

Both are returned together by ``collect_memory_usage_mb()`` so callers can
track both current and peak memory in a single call.
"""
import os
import sys
from typing import Tuple

try:
    import psutil  # type: ignore
except ImportError:
    # psutil is an optional dependency; memory profiling silently degrades to 0.0
    # if it is not installed. It is not required for correctness, only for profiling.
    psutil = None  # type: ignore

try:
    import resource  # type: ignore
except ImportError:
    # ``resource`` is a POSIX module; it is not available on Windows.
    resource = None  # type: ignore


def _process_rss_mb() -> float:
    """
    Return the current Resident Set Size (RSS) of this process in megabytes.

    RSS is the portion of a process's memory that is currently held in physical RAM
    (as opposed to swapped out or memory-mapped but not yet loaded). It is the most
    direct measure of how much RAM the process is actively using right now.

    This function sums the RSS of the main process and all of its child processes
    (recursively), because jukebox-hic may spawn worker subprocesses in parallel
    full-noise mode. The sum gives the total RAM footprint of the entire analysis.

    Implementation uses ``psutil.Process.memory_info().rss``, which reads from
    ``/proc/<pid>/status`` on Linux or the equivalent on macOS.

    Returns 0.0 if:
    - ``psutil`` is not installed (optional dependency).
    - Any psutil error occurs (e.g. the process list changes during iteration,
      a child process exits between being listed and being queried).
    """
    if psutil is None:
        return 0.0
    try:
        proc = psutil.Process(os.getpid())
        rss = proc.memory_info().rss  # bytes
        for child in proc.children(recursive=True):
            try:
                rss += child.memory_info().rss
            except psutil.Error:
                # Child may have exited between children() and memory_info() — skip it
                continue
        # Convert bytes → megabytes
        return rss / (1024 * 1024)
    except psutil.Error:
        return 0.0


def _ru_maxrss_mb() -> float:
    """
    Return the peak Resident Set Size (RSS) ever reached by this process in megabytes.

    Unlike ``_process_rss_mb()`` which returns the *current* RSS, this function
    returns the *maximum* RSS across the entire lifetime of the process — a memory
    high-water mark. This is useful for understanding the worst-case RAM usage during
    a long run, even if memory has been freed since the peak.

    Data source: ``resource.getrusage(resource.RUSAGE_SELF).ru_maxrss``

    Platform-specific units
    -----------------------
    The ``ru_maxrss`` field has different units on different operating systems,
    which is a well-known POSIX inconsistency:

    - **Linux**: ``ru_maxrss`` is in **kilobytes** (kB). Divide by 1024 to get MB.
    - **macOS (Darwin)**: ``ru_maxrss`` is in **bytes**. Divide by 1024*1024 to get MB.
    - **Windows**: ``resource`` module is unavailable; returns 0.0.

    This inconsistency is a quirk of BSD-derived (macOS) vs. Linux implementations
    of ``getrusage``. The Linux man page explicitly states kilobytes; macOS man page
    explicitly states bytes.

    Returns 0.0 if the ``resource`` module is not available (Windows).
    """
    if resource is None:
        return 0.0
    usage = resource.getrusage(resource.RUSAGE_SELF)
    if sys.platform.startswith("darwin") or sys.platform.startswith("win"):
        # macOS and Windows report ru_maxrss in bytes
        return usage.ru_maxrss / (1024 * 1024)
    # Linux reports ru_maxrss in kilobytes
    return usage.ru_maxrss / 1024


def collect_memory_usage_mb() -> Tuple[float, float]:
    """
    Collect both current and peak memory usage in a single call.

    This is the main function used by ``noise_sampling.compute_sampled_noise()``
    and ``cli._profile_command()`` to track memory across processing steps.

    Returns
    -------
    Tuple[float, float]
        - First element: current RSS in MB from ``_process_rss_mb()``.
          Tracks how much RAM is in use *right now* (including child processes).
        - Second element: peak RSS in MB from ``_ru_maxrss_mb()``.
          Tracks the maximum RAM ever used since the process started.

    Both values are 0.0 if the required optional dependencies (psutil, resource)
    are not available.
    """
    return _process_rss_mb(), _ru_maxrss_mb()


def set_address_space_limit(limit_bytes: int) -> bool:
    """
    Set a hard virtual-address-space cap for this process (Linux/macOS only).

    Uses ``resource.setrlimit(RLIMIT_AS, ...)``. When the limit is exceeded, the
    next allocation raises ``MemoryError`` rather than crashing the process silently.
    This allows the caller to catch and handle the condition gracefully.

    Returns True if the limit was applied, False if unsupported (Windows) or if
    setting failed (e.g. the requested limit exceeds the system hard limit).
    """
    if resource is None:
        return False
    try:
        resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
        return True
    except (AttributeError, ValueError, OSError):
        return False


def estimate_worker_memory_mb(
    dump_factor: float,
    noise_bins: int,
    density_estimate: float = 0.05,
) -> float:
    """
    Estimate peak per-worker memory for one ``noise_fullmap`` band fetch (MB).

    The padded fetch window covers ``dump_factor * noise_bins + 2 * noise_bins``
    bins on each side, giving a square submatrix of that size.  For a sparse COO
    matrix at the given density, each non-zero element occupies 24 bytes (int32
    row, int32 col, float64 data).  The 0.05 default (5 % density) is appropriate
    for typical Hi-C; use 0.3 or higher for deeply-sequenced samples.

    Parameters
    ----------
    dump_factor : float
        The dump_factor passed to ``compute_full_noise()`` (default 10.0).
    noise_bins : int
        Number of bins in the noise window (= ``window_bp // resolution``).
    density_estimate : float
        Fraction of the submatrix that is non-zero (default 0.05).

    Returns
    -------
    float
        Estimated peak memory in megabytes for one worker process.
    """
    fetch_bins = int(dump_factor * noise_bins) + 2 * noise_bins
    return (fetch_bins ** 2) * density_estimate * 24 / 1e6
