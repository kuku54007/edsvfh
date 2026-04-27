from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass, field
from typing import TextIO


def format_seconds(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "unknown"
    total = max(0, int(round(seconds)))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


@dataclass
class ETATracker:
    label: str
    total: int | None = None
    unit: str = "items"
    print_every: int = 1
    min_interval_sec: float = 2.0
    stream: TextIO = field(default_factory=lambda: sys.stderr)
    start_time: float = field(default_factory=time.perf_counter)
    initial_current: int = 0
    _last_print_time: float = field(default=0.0, init=False)

    def _should_print(self, current: int, force: bool) -> bool:
        if force:
            return True
        if self.print_every > 0 and current % self.print_every == 0:
            return True
        now = time.perf_counter()
        return (now - self._last_print_time) >= self.min_interval_sec

    def update(self, current: int, *, extra: str = "", force: bool = False) -> None:
        if not self._should_print(current, force):
            return
        elapsed = max(1e-9, time.perf_counter() - self.start_time)
        completed_this_run = max(0, current - self.initial_current)
        rate = completed_this_run / elapsed if completed_this_run > 0 else 0.0
        remaining_display = math.nan
        if self.total is not None:
            remaining_count = max(0, self.total - current)
            progress = f"{current}/{self.total}"
            if rate > 0:
                remaining_display = remaining_count / rate
        else:
            progress = f"{current}/?"
        unit_singular = self.unit.rstrip('s') if self.unit.endswith('s') else self.unit
        message = (
            f"[{self.label}] {progress} {self.unit} | "
            f"elapsed {format_seconds(elapsed)} | "
            f"rate {rate:.2f}/{unit_singular}/s | "
            f"remaining {format_seconds(remaining_display)}"
        )
        if extra:
            message += f" | {extra}"
        print(message, file=self.stream, flush=True)
        self._last_print_time = time.perf_counter()

    def done(self, *, current: int | None = None, extra: str = "") -> None:
        if current is None:
            current = self.total or 0
        self.update(current, extra=extra, force=True)
