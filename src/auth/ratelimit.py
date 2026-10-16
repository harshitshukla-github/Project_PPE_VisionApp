"""Limit failed sign-ins per IP address, across all accounts.

Account lockout stops guessing one worker's PIN. This stops someone trying a
few PINs on every worker ID from one place. It is per server process and in
memory, which is enough for one server; several servers would need a shared
store.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable

FAILURES_PER_WINDOW = 20
WINDOW_S = 60.0


class FailureLimiter:
    """Sliding-window count of failures per key."""

    def __init__(
        self,
        limit: int = FAILURES_PER_WINDOW,
        window_s: float = WINDOW_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.limit = limit
        self.window_s = window_s
        self.clock = clock
        self._failures: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def retry_after(self, key: str) -> float | None:
        """Seconds until ``key`` may try again, or ``None`` if it may now."""
        with self._lock:
            times = self._prune(key)
            if times is None or len(times) < self.limit:
                return None
            return times[0] + self.window_s - self.clock()

    def record_failure(self, key: str) -> None:
        with self._lock:
            self._failures.setdefault(key, deque()).append(self.clock())

    def reset(self) -> None:
        with self._lock:
            self._failures.clear()

    def _prune(self, key: str) -> deque[float] | None:
        times = self._failures.get(key)
        if times is None:
            return None
        cutoff = self.clock() - self.window_s
        while times and times[0] <= cutoff:
            times.popleft()
        if not times:
            del self._failures[key]
            return None
        return times


ip_limiter = FailureLimiter()
