"""
Timing utilities for AI Engine.
"""

from __future__ import annotations

import time


class Timer:
    """
    Simple context timer.
    """

    def __init__(self):

        self.start_time = 0.0
        self.elapsed_ms = 0.0

    def __enter__(self):

        self.start_time = time.perf_counter()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):

        self.elapsed_ms = (
            time.perf_counter() - self.start_time
        ) * 1000