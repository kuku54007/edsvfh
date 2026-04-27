"""EDSV-FH public-dataset reference implementation.

This package provides a CPU-friendly reference implementation of the
Event-Driven Subgoal Verifier with Failure Horizon (EDSV-FH) architecture.
It is designed to:

1. train on public robot-learning datasets when they are available locally;
2. keep the event-watcher / memory / verifier / calibration structure from
   Chapter 2;
3. remain runnable in constrained environments via lightweight fallback
   encoders and small fixture datasets.
"""

from .config import AppConfig
from .models import VerifierBundle
from .pipeline import EventDrivenVerifierPipeline

__all__ = ["AppConfig", "VerifierBundle", "EventDrivenVerifierPipeline"]
