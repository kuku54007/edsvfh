from __future__ import annotations

import io

from edsvfh.progress import ETATracker, format_seconds


def test_format_seconds_basic() -> None:
    assert format_seconds(5) == '00:05'
    assert format_seconds(65) == '01:05'
    assert format_seconds(3661) == '01:01:01'


def test_eta_tracker_emits_progress() -> None:
    stream = io.StringIO()
    tracker = ETATracker(label='unit-test', total=4, unit='episodes', print_every=1, min_interval_sec=0.0, stream=stream)
    tracker.update(1, extra='warmup')
    tracker.done(current=4, extra='finished')
    output = stream.getvalue()
    assert 'unit-test' in output
    assert 'remaining' in output
    assert 'finished' in output


def test_eta_tracker_resume_uses_remaining_not_elapsed_for_rate() -> None:
    stream = io.StringIO()
    tracker = ETATracker(
        label='resume-test',
        total=100,
        unit='episodes',
        print_every=1,
        min_interval_sec=0.0,
        stream=stream,
        initial_current=40,
    )
    tracker.update(40, extra='resume', force=True)
    output = stream.getvalue()
    assert '40/100' in output
    assert 'remaining unknown' in output
