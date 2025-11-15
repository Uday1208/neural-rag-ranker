# src/utils/timing.py
"""
Timing utilities for measuring and logging execution duration.

This module provides small helpers to measure elapsed time for
arbitrary code blocks so that long-running operations such as data
downloads and training loops can report how long they took.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator


@contextmanager
def time_block(label: str) -> Iterator[None]:
    """
    Context manager that measures and prints elapsed time for a code block.

    Args:
        label:
            Human-readable description of the code section being timed.

    Yields:
        None; used strictly for scoping around a timed block.
    """
    start = time.perf_counter()
    print(f"[timing] {label} started...")
    try:
        yield
    finally:
        end = time.perf_counter()
        elapsed = end - start
        print(f"[timing] {label} finished in {elapsed:.2f} seconds.")
